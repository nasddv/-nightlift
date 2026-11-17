/*!
 * freqssm.js - FreqSSM-Lite low-light / exposure enhancement in the browser.
 * No dependencies, works offline (also from file://). Runs the model in a Web Worker.
 *
 * Usage:
 *   <script src="models/lol.js"></script>       (and/or models/sice.js)
 *   <script src="freqssm.js"></script>
 *   const r = await FreqSSM.enhance(fileOrImage, { model: "lol", maxSide: 512 });
 *   document.body.appendChild(r.canvas);         // or: const blob = await r.toBlob();
 *
 * API
 *   FreqSSM.enhance(source, options) -> Promise<Result>
 *     source : File | Blob | HTMLImageElement | HTMLCanvasElement | ImageBitmap | ImageData | URL string
 *     options: model      "lol" (dark photos, default) | "sice" (mixed exposure) | any registered name
 *              maxSide    longest side processed, default 512; 0 = original size (slower)
 *              onProgress function(fraction 0..1)
 *     Result : { canvas, imageData, width, height, ms, toBlob(type="image/png"), toDataURL(type="image/png") }
 *   FreqSSM.registerModel(name, weights)   add weights from a bundler import (JSON of models/*.js)
 *   FreqSSM.models()                       names of the models available
 */
(function (root) {
  "use strict";

  // ---------- model code (runs inside the worker) ----------
  function ENGINE() {
  // FreqSSM-Lite inference in plain JavaScript (mirror of lfs/model.py, no-grad forward pass).
  // Tensors: {c, h, w, d: Float32Array} (one image, channels-first).
  const T = (c, h, w, d) => ({ c, h, w, d: d || new Float32Array(c * h * w) });

  function conv(x, W, b, k, stride) {
    // Row-blocked direct convolution, 4 output channels at a time: each input value loaded once
    // feeds 4 accumulators, and the output rows stay in cache.
    const C = x.c, H = x.h, Wd = x.w, O = b.length, p = (k - 1) >> 1;
    const Ho = Math.ceil(H / stride), Wo = Math.ceil(Wd / stride);
    const y = T(O, Ho, Wo), xd = x.d, yd = y.d, HW = H * Wd, oHW = Ho * Wo, kk = k * k, CK = C * kk;
    const oxa = new Int32Array(k), oxb = new Int32Array(k);
    for (let kj = 0; kj < k; kj++) {
      oxa[kj] = Math.max(0, Math.ceil((p - kj) / stride));
      oxb[kj] = Math.min(Wo - 1, Math.floor((Wd - 1 + p - kj) / stride));
    }
    const r0 = new Float32Array(Wo), r1 = new Float32Array(Wo), r2 = new Float32Array(Wo), r3 = new Float32Array(Wo);
    for (let oy = 0; oy < Ho; oy++) {
      for (let o = 0; o < O; o += 4) {
        const n = Math.min(4, O - o);
        r0.fill(b[o]); r1.fill(n > 1 ? b[o + 1] : 0); r2.fill(n > 2 ? b[o + 2] : 0); r3.fill(n > 3 ? b[o + 3] : 0);
        for (let c = 0; c < C; c++) {
          const xc = c * HW;
          for (let ki = 0; ki < k; ki++) {
            const iy = oy * stride + ki - p;
            if (iy < 0 || iy >= H) continue;
            const xr = xc + iy * Wd - p;
            for (let kj = 0; kj < k; kj++) {
              const t = c * kk + ki * k + kj;
              const w0 = W[o * CK + t], w1 = n > 1 ? W[(o + 1) * CK + t] : 0;
              const w2 = n > 2 ? W[(o + 2) * CK + t] : 0, w3 = n > 3 ? W[(o + 3) * CK + t] : 0;
              const a0 = oxa[kj], b0 = oxb[kj], base = xr + kj;
              if (stride === 1) {
                for (let ox = a0; ox <= b0; ox++) {
                  const v = xd[base + ox];
                  r0[ox] += w0 * v; r1[ox] += w1 * v; r2[ox] += w2 * v; r3[ox] += w3 * v;
                }
              } else {
                for (let ox = a0; ox <= b0; ox++) {
                  const v = xd[base + 2 * ox];
                  r0[ox] += w0 * v; r1[ox] += w1 * v; r2[ox] += w2 * v; r3[ox] += w3 * v;
                }
              }
            }
          }
        }
        const ob = oy * Wo;
        yd.set(r0, o * oHW + ob);
        if (n > 1) yd.set(r1, (o + 1) * oHW + ob);
        if (n > 2) yd.set(r2, (o + 2) * oHW + ob);
        if (n > 3) yd.set(r3, (o + 3) * oHW + ob);
      }
    }
    return y;
  }

  const leaky = (x) => { const d = x.d; for (let i = 0; i < d.length; i++) if (d[i] < 0) d[i] *= 0.1; return x; };
  const add = (a, b) => { const y = T(a.c, a.h, a.w); for (let i = 0; i < y.d.length; i++) y.d[i] = a.d[i] + b.d[i]; return y; };
  function concat(a, b) { const y = T(a.c + b.c, a.h, a.w); y.d.set(a.d, 0); y.d.set(b.d, a.d.length); return y; }
  function up2(x) {
    const y = T(x.c, x.h * 2, x.w * 2);
    for (let c = 0; c < x.c; c++) for (let i = 0; i < y.h; i++) for (let j = 0; j < y.w; j++)
      y.d[(c * y.h + i) * y.w + j] = x.d[(c * x.h + (i >> 1)) * x.w + (j >> 1)];
    return y;
  }

  // ---- FFT (radix-2 + Bluestein for any length) ----
  const plans = new Map();
  function fftPow2(re, im, inv) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = (inv ? 2 : -2) * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
      for (let i = 0; i < n; i += len) {
        let cr = 1, ci = 0;
        for (let j = 0; j < len / 2; j++) {
          const a = i + j, b = a + len / 2;
          const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
          re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
          const ncr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = ncr;
        }
      }
    }
  }
  function plan(n) {
    if (plans.has(n)) return plans.get(n);
    let m = 1; while (m < 2 * n - 1) m <<= 1;
    const cr = new Float64Array(n), ci = new Float64Array(n);
    for (let k = 0; k < n; k++) { const a = Math.PI * ((k * k) % (2 * n)) / n; cr[k] = Math.cos(a); ci[k] = Math.sin(a); }
    const br = new Float64Array(m), bi = new Float64Array(m);
    br[0] = cr[0]; bi[0] = ci[0];
    for (let k = 1; k < n; k++) { br[k] = br[m - k] = cr[k]; bi[k] = bi[m - k] = ci[k]; }
    fftPow2(br, bi, false);
    const pl = { n, m, cr, ci, br, bi, pow2: (n & (n - 1)) === 0 };
    plans.set(n, pl);
    return pl;
  }
  // in-place unnormalised DFT of (re, im), sign -1 (forward) or +1 (inverse)
  function dft(re, im, inv) {
    const n = re.length, pl = plan(n);
    if (pl.pow2) { fftPow2(re, im, inv); return; }
    const { m, cr, ci, br, bi } = pl, s = inv ? -1 : 1;
    const ar = new Float64Array(m), ai = new Float64Array(m);
    for (let k = 0; k < n; k++) { // x_k * conj(chirp) (forward) / * chirp (inverse)
      const c = cr[k], d = s * ci[k];
      ar[k] = re[k] * c + im[k] * d; ai[k] = im[k] * c - re[k] * d;
    }
    fftPow2(ar, ai, false);
    for (let k = 0; k < m; k++) {
      const bI = s * bi[k], r = ar[k] * br[k] - ai[k] * bI; ai[k] = ar[k] * bI + ai[k] * br[k]; ar[k] = r;
    }
    fftPow2(ar, ai, true);
    for (let k = 0; k < n; k++) {
      const r = ar[k] / m, i = ai[k] / m, c = cr[k], d = s * ci[k];
      re[k] = r * c + i * d; im[k] = i * c - r * d;
    }
  }
  function fft2(re, im, H, W, inv) {
    const rr = new Float64Array(W), ri = new Float64Array(W);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) { rr[x] = re[y * W + x]; ri[x] = im[y * W + x]; }
      dft(rr, ri, inv);
      for (let x = 0; x < W; x++) { re[y * W + x] = rr[x]; im[y * W + x] = ri[x]; }
    }
    const cr = new Float64Array(H), ci = new Float64Array(H);
    for (let x = 0; x < W; x++) {
      for (let y = 0; y < H; y++) { cr[y] = re[y * W + x]; ci[y] = im[y * W + x]; }
      dft(cr, ci, inv);
      for (let y = 0; y < H; y++) { re[y * W + x] = cr[y]; im[y * W + x] = ci[y]; }
    }
  }

  function freqBlock(x, P, pre) {
    const { c: C, h: H, w: W } = x, n = H * W, sc = 1 / Math.sqrt(n);
    const Xr = [], Xi = [], la = T(C, H, W);
    for (let c = 0; c < C; c++) {
      const re = new Float64Array(n), im = new Float64Array(n);
      for (let i = 0; i < n; i++) re[i] = x.d[c * n + i];
      fft2(re, im, H, W, false);
      for (let i = 0; i < n; i++) { re[i] *= sc; im[i] *= sc; la.d[c * n + i] = Math.log1p(Math.hypot(re[i], im[i])); }
      Xr.push(re); Xi.push(im);
    }
    const g = conv(la, P[pre + '.gain.w'], P[pre + '.gain.b'], 1, 1);
    const f = T(C, H, W);
    for (let c = 0; c < C; c++) {
      const re = Xr[c], im = Xi[c];
      for (let i = 0; i < n; i++) { const m = 1 + 0.5 * Math.tanh(g.d[c * n + i]); re[i] *= m; im[i] *= m; }
      fft2(re, im, H, W, true);
      for (let i = 0; i < n; i++) f.d[c * n + i] = re[i] * sc;
    }
    return add(x, conv(f, P[pre + '.proj.w'], P[pre + '.proj.b'], 1, 1));
  }

  function ssmBlock(x, P, pre) {
    const u = conv(x, P[pre + '.inp.w'], P[pre + '.inp.b'], 1, 1);
    const a = conv(x, P[pre + '.dec.w'], P[pre + '.dec.b'], 1, 1);
    for (let i = 0; i < a.d.length; i++) a.d[i] = 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, a.d[i]))));
    const z = conv(x, P[pre + '.gate.w'], P[pre + '.gate.b'], 1, 1);
    const { c: C, h: H, w: W } = x, y = T(C, H, W);
    for (let c = 0; c < C; c++) {
      const o = c * H * W;
      for (let i = 0; i < H; i++) { // along W, both ways
        let h = 0;
        for (let j = 0; j < W; j++) { const k = o + i * W + j; h = a.d[k] * h + (1 - a.d[k]) * u.d[k]; y.d[k] += h; }
        h = 0;
        for (let j = W - 1; j >= 0; j--) { const k = o + i * W + j; h = a.d[k] * h + (1 - a.d[k]) * u.d[k]; y.d[k] += h; }
      }
      for (let j = 0; j < W; j++) { // along H, both ways
        let h = 0;
        for (let i = 0; i < H; i++) { const k = o + i * W + j; h = a.d[k] * h + (1 - a.d[k]) * u.d[k]; y.d[k] += h; }
        h = 0;
        for (let i = H - 1; i >= 0; i--) { const k = o + i * W + j; h = a.d[k] * h + (1 - a.d[k]) * u.d[k]; y.d[k] += h; }
      }
    }
    for (let i = 0; i < y.d.length; i++) {
      const s = 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, z.d[i]))));
      y.d[i] = 0.25 * y.d[i] * z.d[i] * s;
    }
    return add(x, conv(y, P[pre + '.out.w'], P[pre + '.out.b'], 1, 1));
  }

  const cv = (x, P, n, k, s) => conv(x, P[n + '.w'], P[n + '.b'], k, s || 1);
  const res = (x, P, n) => add(x, cv(leaky(cv(x, P, n + '.c1', 3)), P, n + '.c2', 3));

  function forward(x, P, progress) {
    const step = progress || (() => {});
    const e1 = res(leaky(cv(x, P, 'inc', 3)), P, 'e1'); step(0.15);
    const e2 = freqBlock(res(leaky(cv(e1, P, 'down1', 3, 2)), P, 'e2'), P, 'f2'); step(0.35);
    let b = leaky(cv(e2, P, 'down2', 3, 2));
    b = res(freqBlock(ssmBlock(res(b, P, 'b1'), P, 'ssm'), P, 'fb'), P, 'b2'); step(0.55);
    const d2 = res(leaky(cv(concat(up2(b), e2), P, 'up2', 3)), P, 'd2'); step(0.75);
    const d1 = res(leaky(cv(concat(up2(d2), e1), P, 'up1', 3)), P, 'd1'); step(0.95);
    return add(x, cv(d1, P, 'outc', 3));
  }

  function reflectIdx(i, n) { return i < n ? i : 2 * (n - 1) - i; } // numpy 'reflect' for small pads

  // rgb: Float32Array (3*H*W, channels-first, 0..1). Returns enhanced array of the same size.
  function enhance(rgb, H, W, P, progress) {
    const Hp = H + ((4 - H % 4) % 4), Wp = W + ((4 - W % 4) % 4);
    const x = T(3, Hp, Wp);
    for (let c = 0; c < 3; c++) for (let i = 0; i < Hp; i++) for (let j = 0; j < Wp; j++)
      x.d[(c * Hp + i) * Wp + j] = rgb[(c * H + reflectIdx(i, H)) * W + reflectIdx(j, W)];
    const y = forward(x, P, progress);
    const out = new Float32Array(3 * H * W);
    for (let c = 0; c < 3; c++) for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
      const v = y.d[(c * Hp + i) * Wp + j];
      out[(c * H + i) * W + j] = v < 0 ? 0 : v > 1 ? 1 : v;
    }
    return out;
  }

  function loadWeights(pack) {
    const bin = typeof atob === 'function' ? Uint8Array.from(atob(pack.data), (ch) => ch.charCodeAt(0))
      : new Uint8Array(Buffer.from(pack.data, 'base64'));
    const all = new Float32Array(bin.buffer, bin.byteOffset, bin.byteLength / 4);
    const P = {};
    for (const [k, [shape, off]] of Object.entries(pack.manifest)) {
      const n = shape.reduce((a, b) => a * b, 1);
      P[k] = all.subarray(off, off + n);
    }
    return P;
  }

    const cache = {};
    onmessage = (e) => {
      const m = e.data;
      try {
        if (m.type === "weights") { cache[m.key] = loadWeights(m.pack); postMessage({ type: "ready", id: m.id }); return; }
        const t0 = performance.now();
        const out = enhance(m.rgb, m.h, m.w, cache[m.key], (p) => postMessage({ type: "progress", id: m.id, p }));
        postMessage({ type: "done", id: m.id, out, ms: performance.now() - t0 }, [out.buffer]);
      } catch (err) { postMessage({ type: "error", id: m.id, message: String((err && err.message) || err) }); }
    };
  }

  // ---------- public API (main thread) ----------
  const registry = Object.assign({}, root.FreqSSMWeights || {});
  const sent = {};
  let worker = null, nextId = 1, queue = Promise.resolve();
  const pending = {};

  function getWorker() {
    if (worker) return worker;
    const src = "(" + ENGINE.toString() + ")();";
    worker = new Worker(URL.createObjectURL(new Blob([src], { type: "text/javascript" })));
    worker.onmessage = (e) => {
      const d = e.data, p = pending[d.id];
      if (!p) return;
      if (d.type === "progress") { if (p.onProgress) p.onProgress(d.p); return; }
      delete pending[d.id];
      if (d.type === "error") p.reject(new Error(d.message)); else p.resolve(d);
    };
    return worker;
  }

  function call(msg, transfer, onProgress) {
    const id = nextId++;
    return new Promise((resolve, reject) => {
      pending[id] = { resolve, reject, onProgress };
      getWorker().postMessage(Object.assign({ id }, msg), transfer || []);
    });
  }

  function weightsFor(name) {
    const w = registry[name] || (root.FreqSSMWeights || {})[name];
    if (!w) throw new Error('FreqSSM: model "' + name + '" is not loaded. Include models/' + name + '.js or call FreqSSM.registerModel().');
    return w;
  }

  async function toDrawable(source) {
    if (typeof source === "string") {
      const img = new Image(); img.src = source; await img.decode(); return img;
    }
    if (source instanceof Blob) {
      if (typeof createImageBitmap === "function") return createImageBitmap(source);
      const img = new Image(); img.src = URL.createObjectURL(source); await img.decode(); return img;
    }
    return source;
  }

  async function enhance(source, options) {
    const opt = Object.assign({ model: "lol", maxSide: 512, onProgress: null }, options || {});
    const run = async () => {
      const key = opt.model, pack = weightsFor(key);
      if (!sent[key]) sent[key] = call({ type: "weights", key, pack });
      await sent[key];

      let W0, H0, draw = null, data = null;
      if (typeof ImageData !== "undefined" && source instanceof ImageData) { W0 = source.width; H0 = source.height; data = source; }
      else { draw = await toDrawable(source); W0 = draw.naturalWidth || draw.width; H0 = draw.naturalHeight || draw.height; }
      const s = opt.maxSide && Math.max(W0, H0) > opt.maxSide ? opt.maxSide / Math.max(W0, H0) : 1;
      const w = Math.max(4, Math.round(W0 * s)), h = Math.max(4, Math.round(H0 * s));
      const canvas = document.createElement("canvas"); canvas.width = w; canvas.height = h;
      const ctx = canvas.getContext("2d");
      if (data && s === 1) ctx.putImageData(data, 0, 0);
      else if (data) { const t = document.createElement("canvas"); t.width = W0; t.height = H0; t.getContext("2d").putImageData(data, 0, 0); ctx.drawImage(t, 0, 0, w, h); }
      else ctx.drawImage(draw, 0, 0, w, h);

      const px = ctx.getImageData(0, 0, w, h).data, n = w * h, rgb = new Float32Array(3 * n);
      for (let i = 0; i < n; i++) { rgb[i] = px[4 * i] / 255; rgb[n + i] = px[4 * i + 1] / 255; rgb[2 * n + i] = px[4 * i + 2] / 255; }
      const res = await call({ type: "run", key, rgb, h, w }, [rgb.buffer], opt.onProgress);

      const out = ctx.createImageData(w, h), o = res.out;
      for (let i = 0; i < n; i++) {
        out.data[4 * i] = Math.round(o[i] * 255); out.data[4 * i + 1] = Math.round(o[n + i] * 255);
        out.data[4 * i + 2] = Math.round(o[2 * n + i] * 255); out.data[4 * i + 3] = 255;
      }
      ctx.putImageData(out, 0, 0);
      if (opt.onProgress) opt.onProgress(1);
      return {
        canvas, imageData: out, width: w, height: h, ms: res.ms,
        toBlob: (type) => new Promise((r) => canvas.toBlob(r, type || "image/png")),
        toDataURL: (type) => canvas.toDataURL(type || "image/png"),
      };
    };
    const job = queue.then(run, run);   // one image at a time
    queue = job.catch(() => {});
    return job;
  }

  const FreqSSM = {
    enhance,
    registerModel(name, weights) { registry[name] = weights; delete sent[name]; },
    models() { return Object.keys(Object.assign({}, root.FreqSSMWeights || {}, registry)); },
  };
  root.FreqSSM = FreqSSM;
  if (typeof module !== "undefined" && module.exports) module.exports = FreqSSM;
})(typeof window !== "undefined" ? window : globalThis);
