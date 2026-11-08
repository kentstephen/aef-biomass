/// <reference types="@webgpu/types" />
import { type BiomassHead, CODE, LOG_MAX } from "./head";

/**
 * The biomass head as a WebGPU compute pass (after fire-knn's fuel/volume-gpu.ts), one thread per pixel, run once
 * per chunk-year. The year goes up as raw int8 packed in u32 words; each thread dequantizes, standardizes, runs
 * the network and writes the two codes of head.ts (biomass in the low 16 bits, height in the high 16), then the
 * class (the largest logit's index) in a second word.
 */
export type Scorer = (e: Int8Array, n: number) => Promise<Uint16Array>;

function wgsl(h: BiomassHead): string {
  const NI = h.n_in, NH = h.hidden, NO = h.n_out;
  const o1 = 2 * NI, ob1 = o1 + NH * NI, o2 = ob1 + NH, ob2 = o2 + NO * NH;
  return /* wgsl */ `
struct Params { n: u32 }
@group(0) @binding(0) var<storage, read> aef: array<u32>;
@group(0) @binding(1) var<storage, read> wts: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<u32>;
@group(0) @binding(3) var<uniform> prm: Params;

fn q8(k: u32) -> i32 {
  return extractBits(bitcast<i32>(aef[k >> 2u]), (k & 3u) * 8u, 8u);
}

fn code(v: f32) -> u32 {
  return 1u + u32(round(clamp(v, 0.0, ${LOG_MAX.toFixed(1)}) * ${CODE.toFixed(1)}));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let p = gid.x;
  let n = prm.n;
  if (p >= n) { return; }
  if (q8(p) == -128) { outp[2u * p] = 0u; outp[2u * p + 1u] = 0u; return; }
  var x: array<f32, ${NI}>;
  for (var b = 0u; b < ${NI}u; b++) {
    let f = f32(q8(b * n + p)) / 127.5;
    x[b] = (f * abs(f) - wts[b]) * wts[${NI}u + b];
  }
  var hid: array<f32, ${NH}>;
  for (var j = 0u; j < ${NH}u; j++) {
    var s = wts[${ob1}u + j];
    let r = ${o1}u + j * ${NI}u;
    for (var i = 0u; i < ${NI}u; i++) { s += wts[r + i] * x[i]; }
    hid[j] = max(s, 0.0);
  }
  var z: array<f32, ${NO}>;
  for (var c = 0u; c < ${NO}u; c++) {
    var s = wts[${ob2}u + c];
    let r = ${o2}u + c * ${NH}u;
    for (var j = 0u; j < ${NH}u; j++) { s += wts[r + j] * hid[j]; }
    z[c] = s;
  }
  outp[2u * p] = code(z[0]) | (code(z[1]) << 16u);
  var best = 0u;
  for (var c = 3u; c < ${NO}u; c++) { if (z[c] > z[2u + best]) { best = c - 2u; } }
  outp[2u * p + 1u] = best;
}
`;
}

export async function makeScorer(h: BiomassHead, w: Float32Array, maxN: number): Promise<Scorer | null> {
  const gpu = (globalThis.navigator as Navigator | undefined)?.gpu;
  if (!gpu) return null;
  const adapter = await gpu.requestAdapter();
  if (!adapter) return null;
  const device = await adapter.requestDevice();
  const module = device.createShaderModule({ code: wgsl(h) });
  const info = await module.getCompilationInfo();
  const errs = info.messages.filter((m) => m.type === "error");
  if (errs.length) throw new Error(`biomass shader: ${errs.map((m) => m.message).join("; ")}`);
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
  const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const aefBuf = device.createBuffer({ size: 64 * maxN, usage: S });
  const wBuf = device.createBuffer({ size: w.byteLength, usage: S });
  device.queue.writeBuffer(wBuf, 0, w as Float32Array<ArrayBuffer>);
  const outBuf = device.createBuffer({ size: 8 * maxN, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readBuf = device.createBuffer({ size: 8 * maxN, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const prmBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: aefBuf } },
      { binding: 1, resource: { buffer: wBuf } },
      { binding: 2, resource: { buffer: outBuf } },
      { binding: 3, resource: { buffer: prmBuf } },
    ],
  });
  // one chunk at a time per worker: the buffers are shared
  let queue: Promise<unknown> = Promise.resolve();
  return (e, n) => {
    const run = async () => {
      if (n > maxN) throw new Error(`chunk of ${n} px is over the scorer's ${maxN}`);
      // writeBuffer wants a multiple of 4 bytes; a chunk is 64 x n bytes, always one
      device.queue.writeBuffer(aefBuf, 0, e.buffer as ArrayBuffer, e.byteOffset, e.byteLength);
      device.queue.writeBuffer(prmBuf, 0, new Uint32Array([n, 0, 0, 0]));
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(Math.ceil(n / 64));
      pass.end();
      enc.copyBufferToBuffer(outBuf, 0, readBuf, 0, 8 * n);
      device.queue.submit([enc.finish()]);
      await readBuf.mapAsync(GPUMapMode.READ, 0, 8 * n);
      const packed = new Uint32Array(readBuf.getMappedRange(0, 8 * n).slice(0));
      readBuf.unmap();
      const out = new Uint16Array(3 * n);
      for (let p = 0; p < n; p++) {
        const u = packed[2 * p]!;
        out[p] = u & 0xffff;
        out[n + p] = u >>> 16;
        out[2 * n + p] = packed[2 * p + 1]!;
      }
      return out;
    };
    const r = queue.then(run, run);
    queue = r.catch(() => {});
    return r;
  };
}
