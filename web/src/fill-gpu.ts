/// <reference types="@webgpu/types" />
import { type FillNet, fillOffsets } from "./fill";

/**
 * The fill-in network as a WebGPU compute pass, one thread per pixel (fill.ts's fillCpu, the same math). In: the
 * chunk's k bands, band-major int8 packed in u32 words. Out: the 64 rebuilt bands per pixel, pixel-major int8 packed
 * (16 words a pixel, so no two threads write one word), turned band-major on the CPU for the head.
 */
export type Filler = (e: Int8Array, n: number) => Promise<Int8Array>;

function wgsl(f: FillNet): string {
  const O = fillOffsets(f);
  const k = f.k, m = f.miss.length;
  const arr = (a: number[]) => `array<u32, ${a.length}>(${a.map((v) => `${v}u`).join(", ")})`;
  const sizes = [k, ...f.dims.map(([o]) => o)];
  let body = "";
  O.layers.forEach((L, l) => {
    const last = l === O.layers.length - 1;
    body += `
  var a${l + 1}: array<f32, ${L.out}>;
  for (var j = 0u; j < ${L.out}u; j++) {
    var s = wts[${L.b}u + j];
    let r = ${L.w}u + j * ${L.inp}u;
    for (var i = 0u; i < ${L.inp}u; i++) { s += wts[r + i] * a${l}[i]; }
    a${l + 1}[j] = ${last ? "s" : "max(s, 0.0)"};
  }`;
  });
  const lastA = `a${O.layers.length}`;
  return /* wgsl */ `
struct Params { n: u32 }
@group(0) @binding(0) var<storage, read> inp: array<u32>;
@group(0) @binding(1) var<storage, read> wts: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<u32>;
@group(0) @binding(3) var<uniform> prm: Params;

var<private> BANDS: array<u32, ${k}> = ${arr(f.bands)};
var<private> MISS: array<u32, ${m}> = ${arr(f.miss)};

fn q8(k: u32) -> i32 {
  return extractBits(bitcast<i32>(inp[k >> 2u]), (k & 3u) * 8u, 8u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let p = gid.x;
  let n = prm.n;
  if (p >= n) { return; }
  if (q8(p) == -128) {
    for (var w = 0u; w < 16u; w++) { outp[p * 16u + w] = 0x80808080u; }
    return;
  }
  var full: array<f32, 64>;
  var a0: array<f32, ${sizes[0]}>;
  for (var i = 0u; i < ${k}u; i++) {
    let q = f32(q8(i * n + p)) / 127.5;
    let d = q * abs(q);
    full[BANDS[i]] = d;
    a0[i] = (d - wts[${O.mu}u + i]) * wts[${O.isd}u + i];
  }${body}
  for (var j = 0u; j < ${m}u; j++) { full[MISS[j]] = ${lastA}[j] * wts[${O.ys}u + j] + wts[${O.ym}u + j]; }
  var nn = 0.0;
  for (var b = 0u; b < 64u; b++) { nn += full[b] * full[b]; }
  let inv = 1.0 / max(sqrt(nn), 1e-6);
  for (var w = 0u; w < 16u; w++) {
    var word = 0u;
    for (var t = 0u; t < 4u; t++) {
      let v = full[w * 4u + t] * inv;
      let q = clamp(round(sign(v) * sqrt(abs(v)) * 127.5), -127.0, 127.0);
      word |= (bitcast<u32>(i32(q)) & 0xffu) << (8u * t);
    }
    outp[p * 16u + w] = word;
  }
}
`;
}

export async function makeFiller(f: FillNet, w: Float32Array, maxN: number): Promise<Filler | null> {
  const gpu = (globalThis.navigator as Navigator | undefined)?.gpu;
  if (!gpu) return null;
  const adapter = await gpu.requestAdapter();
  if (!adapter) return null;
  const device = await adapter.requestDevice();
  const module = device.createShaderModule({ code: wgsl(f) });
  const info = await module.getCompilationInfo();
  const errs = info.messages.filter((m) => m.type === "error");
  if (errs.length) throw new Error(`fill shader: ${errs.map((m) => m.message).join("; ")}`);
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
  const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  // the k bands of a chunk, padded to a whole word
  const inBuf = device.createBuffer({ size: Math.ceil((f.k * maxN) / 4) * 4, usage: S });
  const wBuf = device.createBuffer({ size: w.byteLength, usage: S });
  device.queue.writeBuffer(wBuf, 0, w as Float32Array<ArrayBuffer>);
  const outBuf = device.createBuffer({ size: 64 * maxN, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readBuf = device.createBuffer({ size: 64 * maxN, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const prmBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: inBuf } },
      { binding: 1, resource: { buffer: wBuf } },
      { binding: 2, resource: { buffer: outBuf } },
      { binding: 3, resource: { buffer: prmBuf } },
    ],
  });
  let queue: Promise<unknown> = Promise.resolve();
  return (e, n) => {
    const run = async () => {
      if (n > maxN) throw new Error(`chunk of ${n} px is over the filler's ${maxN}`);
      const padded = new Int8Array(Math.ceil(e.length / 4) * 4);
      padded.set(e);
      device.queue.writeBuffer(inBuf, 0, padded.buffer, 0, padded.byteLength);
      device.queue.writeBuffer(prmBuf, 0, new Uint32Array([n, 0, 0, 0]));
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(Math.ceil(n / 64));
      pass.end();
      enc.copyBufferToBuffer(outBuf, 0, readBuf, 0, 64 * n);
      device.queue.submit([enc.finish()]);
      await readBuf.mapAsync(GPUMapMode.READ, 0, 64 * n);
      const px = new Int8Array(readBuf.getMappedRange(0, 64 * n).slice(0));
      readBuf.unmap();
      // pixel-major to band-major
      const out = new Int8Array(64 * n);
      for (let p = 0; p < n; p++) {
        const r = p * 64;
        for (let b = 0; b < 64; b++) out[b * n + p] = px[r + b]!;
      }
      return out;
    };
    const r = queue.then(run, run);
    queue = r.catch(() => {});
    return r;
  };
}
