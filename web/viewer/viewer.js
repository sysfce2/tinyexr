/*
 * TinyEXR v3 WASM viewer — application logic.
 *
 * Flow:
 *   file/drop -> read bytes (with progress) -> exrv_open + header JSON
 *             -> exrv_select(part,level) -> per-block decode loop (progress)
 *             -> copy RGBA floats out of the WASM heap -> WebGL2 float texture.
 *
 * Tone-mapping (exposure / gamma / sRGB), channel isolation, and zoom/pan all
 * happen in the fragment shader, so they are instant and never re-decode. The
 * pixel picker reads raw floats straight from the JS-side copy of the buffer.
 */

import createModule from "./exr_viewer.mjs";

/* ------------------------------------------------------------------ globals */
let M = null;             // emscripten module
let handle = 0;           // current exrv session handle
let header = null;        // parsed header JSON
let sel = { part: 0, lx: 0, ly: 0, view: 0 };
let views = [];           // channel-view options for the current part
let img = null;           // { w, h, data: Float32Array(rgba) }
let fileName = null;      // name of the currently loaded file (for info panel)
let dwMin = { x: 0, y: 0 };
let dispWin = null;       // {minX,minY,maxX,maxY} of selected part

// view transform, in device pixels: screen = pan + imagePx * zoom
let view = { zoom: 1, panX: 0, panY: 0 };

// display controls
let ctl = { exposure: 0, gamma: 2.2, srgb: true, channel: 0,
            falseColor: false, showDisp: true, cropDisp: false };

// region of interest, in image pixels (level space), or null
let roi = null;

const dpr = () => window.devicePixelRatio || 1;
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const raf = () => new Promise((r) => requestAnimationFrame(r));

/* ------------------------------------------------------------------- WebGL */
let gl, prog, vbo, tex, uni;

const VS = `#version 300 es
in vec2 aPos; in vec2 aUV; out vec2 vUV;
void main(){ vUV = aUV; gl_Position = vec4(aPos, 0.0, 1.0); }`;

const FS = `#version 300 es
precision highp float;
in vec2 vUV; out vec4 frag;
uniform sampler2D uTex;
uniform float uExposure, uGamma;
uniform int uSRGB, uChannel, uFalse;
vec3 toSRGB(vec3 c){
  c = clamp(c, 0.0, 1.0);
  return mix(12.92*c, 1.055*pow(c, vec3(1.0/2.4)) - 0.055, step(0.0031308, c));
}
// Perceptually-uniform viridis colormap (polynomial approximation).
vec3 viridis(float t){
  t = clamp(t, 0.0, 1.0);
  const vec3 c0 = vec3(0.2777273272234177, 0.005407344544966578, 0.3340998053353061);
  const vec3 c1 = vec3(0.1050930431085774, 1.404613529898575, 1.384590162594685);
  const vec3 c2 = vec3(-0.3308618287255563, 0.214847559468213, 0.09509516302823659);
  const vec3 c3 = vec3(-4.634230498983486, -5.799100973351585, -19.33244095627987);
  const vec3 c4 = vec3(6.228269936347081, 14.17993336680509, 56.69055260068105);
  const vec3 c5 = vec3(4.776384997670288, -13.74514537774601, -65.35303263337234);
  const vec3 c6 = vec3(-5.435455855934631, 4.645852612178535, 26.3124352495832);
  return c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6)))));
}
// Tone-map a scalar to display [0,1] using the same curve as the RGB path.
float tone(float s){
  s = max(s, 0.0);
  return (uSRGB == 1) ? toSRGB(vec3(s)).r : pow(clamp(s, 0.0, 1.0), 1.0 / uGamma);
}
void main(){
  vec4 t = texture(uTex, vUV);
  float e = uExposure;
  if (uChannel == 0) {                               // RGB
    vec3 c = max(t.rgb * e, 0.0);
    c = (uSRGB == 1) ? toSRGB(c) : pow(clamp(c, 0.0, 1.0), vec3(1.0 / uGamma));
    frag = vec4(c, 1.0);
    return;
  }
  float s;                                           // single scalar channel
  if      (uChannel == 1) s = t.r * e;
  else if (uChannel == 2) s = t.g * e;
  else if (uChannel == 3) s = t.b * e;
  else if (uChannel == 4) s = t.a;                   // alpha: no exposure
  else                    s = dot(t.rgb, vec3(0.2126,0.7152,0.0722)) * e;
  float d = tone(s);
  frag = (uFalse == 1) ? vec4(viridis(d), 1.0) : vec4(vec3(d), 1.0);
}`;

function compile(type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS))
    throw new Error("shader: " + gl.getShaderInfoLog(s));
  return s;
}

function initGL() {
  const canvas = document.getElementById("gl");
  gl = canvas.getContext("webgl2", { antialias: false, premultipliedAlpha: false });
  if (!gl) { showError("WebGL2 is not available in this browser."); return; }

  prog = gl.createProgram();
  gl.attachShader(prog, compile(gl.VERTEX_SHADER, VS));
  gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FS));
  gl.bindAttribLocation(prog, 0, "aPos");
  gl.bindAttribLocation(prog, 1, "aUV");
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS))
    throw new Error("link: " + gl.getProgramInfoLog(prog));

  uni = {
    tex: gl.getUniformLocation(prog, "uTex"),
    exposure: gl.getUniformLocation(prog, "uExposure"),
    gamma: gl.getUniformLocation(prog, "uGamma"),
    srgb: gl.getUniformLocation(prog, "uSRGB"),
    channel: gl.getUniformLocation(prog, "uChannel"),
    falseColor: gl.getUniformLocation(prog, "uFalse"),
  };

  vbo = gl.createBuffer();
  tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
}

function uploadTexture() {
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, img.w, img.h, 0,
                gl.RGBA, gl.FLOAT, img.data);
}

function syncCanvasSize() {
  const c = document.getElementById("gl");
  const o = document.getElementById("overlay");
  const w = Math.round(c.clientWidth * dpr());
  const h = Math.round(c.clientHeight * dpr());
  if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
  if (o.width !== w || o.height !== h) { o.width = w; o.height = h; }
}

function render() {
  if (!gl) return;
  syncCanvasSize();
  const W = gl.canvas.width, H = gl.canvas.height;
  gl.viewport(0, 0, W, H);
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);
  if (!img) { drawOverlay(); return; }

  // image rect in device pixels
  const x0 = view.panX, y0 = view.panY;
  const x1 = x0 + img.w * view.zoom, y1 = y0 + img.h * view.zoom;
  const cx = (x) => (x / W) * 2 - 1;
  const cy = (y) => 1 - (y / H) * 2;
  // strip: TL, TR, BL, BR  (pos.xy, uv.xy)
  const v = new Float32Array([
    cx(x0), cy(y0), 0, 0,
    cx(x1), cy(y0), 1, 0,
    cx(x0), cy(y1), 0, 1,
    cx(x1), cy(y1), 1, 1,
  ]);
  gl.useProgram(prog);
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, v, gl.DYNAMIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 16, 0);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 16, 8);

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.uniform1i(uni.tex, 0);
  gl.uniform1f(uni.exposure, Math.pow(2, ctl.exposure));
  gl.uniform1f(uni.gamma, ctl.gamma);
  gl.uniform1i(uni.srgb, ctl.srgb ? 1 : 0);
  gl.uniform1i(uni.channel, ctl.channel);
  gl.uniform1i(uni.falseColor, (ctl.falseColor && ctl.channel !== 0) ? 1 : 0);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

  drawOverlay();
  updateZoomLabel();
  updateLegend();
}

// Channel labels for the false-color legend (index matches the channel buttons).
const CHANNEL_NAMES = ["RGB", "R", "G", "B", "A", "Lum"];

// Viridis colormap in JS — must match the shader's viridis() so the legend is
// an accurate key for the rendered false-color image.
function viridisRGB(t) {
  t = Math.min(Math.max(t, 0), 1);
  const c0 = [0.2777273272234177, 0.005407344544966578, 0.3340998053353061];
  const c1 = [0.1050930431085774, 1.404613529898575, 1.384590162594685];
  const c2 = [-0.3308618287255563, 0.214847559468213, 0.09509516302823659];
  const c3 = [-4.634230498983486, -5.799100973351585, -19.33244095627987];
  const c4 = [6.228269936347081, 14.17993336680509, 56.69055260068105];
  const c5 = [4.776384997670288, -13.74514537774601, -65.35303263337234];
  const c6 = [-5.435455855934631, 4.645852612178535, 26.3124352495832];
  const out = [];
  for (let i = 0; i < 3; i++) {
    const v = c0[i] + t * (c1[i] + t * (c2[i] + t * (c3[i] + t * (c4[i] + t * (c5[i] + t * c6[i])))));
    out.push(Math.round(Math.min(Math.max(v, 0), 1) * 255));
  }
  return out;
}

function initLegendGradient() {
  const stops = [];
  for (let i = 0; i <= 16; i++) {
    const t = i / 16;
    const [r, g, b] = viridisRGB(t);
    stops.push(`rgb(${r},${g},${b}) ${(t * 100).toFixed(1)}%`);
  }
  document.getElementById("legendBar").style.background =
    `linear-gradient(90deg, ${stops.join(", ")})`;
}

function updateLegend() {
  const legend = document.getElementById("legend");
  const show = !!img && ctl.falseColor && ctl.channel !== 0;
  legend.classList.toggle("hidden", !show);
  if (show)
    document.getElementById("legendChan").textContent = CHANNEL_NAMES[ctl.channel] || "";
}

/* image-pixel (level space) -> device-pixel screen coords */
function imgToScreen(ix, iy) {
  return [view.panX + ix * view.zoom, view.panY + iy * view.zoom];
}
function screenToImg(sx, sy) {
  return [(sx - view.panX) / view.zoom, (sy - view.panY) / view.zoom];
}

function drawOverlay() {
  const o = document.getElementById("overlay");
  const ctx = o.getContext("2d");
  ctx.clearRect(0, 0, o.width, o.height);
  if (!img) return;

  // crop-to-display mask: darken everything outside the display window
  if (ctl.cropDisp && dispWin) {
    const [mx0, my0] = imgToScreen(dispWin.minX - dwMin.x, dispWin.minY - dwMin.y);
    const [mx1, my1] = imgToScreen(dispWin.maxX - dwMin.x + 1, dispWin.maxY - dwMin.y + 1);
    ctx.save();
    ctx.fillStyle = "rgba(0,0,0,0.6)";
    ctx.beginPath();
    ctx.rect(0, 0, o.width, o.height);
    ctx.rect(mx0, my0, mx1 - mx0, my1 - my0);
    ctx.fill("evenodd");
    ctx.restore();
  }

  // display-window outline
  if (ctl.showDisp && dispWin) {
    const [dx0, dy0] = imgToScreen(dispWin.minX - dwMin.x, dispWin.minY - dwMin.y);
    const [dx1, dy1] = imgToScreen(dispWin.maxX - dwMin.x + 1, dispWin.maxY - dwMin.y + 1);
    ctx.strokeStyle = "#ffd24c";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(dx0 + .5, dy0 + .5, dx1 - dx0 - 1, dy1 - dy0 - 1);
    ctx.setLineDash([]);
  }

  // region of interest
  if (roi) {
    const [rx0, ry0] = imgToScreen(roi.x0, roi.y0);
    const [rx1, ry1] = imgToScreen(roi.x1, roi.y1);
    ctx.strokeStyle = "#4c9aff";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(rx0 + .5, ry0 + .5, rx1 - rx0 - 1, ry1 - ry0 - 1);
    ctx.fillStyle = "rgba(76,154,255,0.12)";
    ctx.fillRect(rx0, ry0, rx1 - rx0, ry1 - ry0);
  }
}

/* --------------------------------------------------------------- view ops */
function fitView() {
  if (!img) return;
  const c = gl.canvas;
  const z = Math.min(c.width / img.w, c.height / img.h) * 0.96;
  view.zoom = z > 0 ? z : 1;
  view.panX = (c.width - img.w * view.zoom) / 2;
  view.panY = (c.height - img.h * view.zoom) / 2;
  render();
}
function oneToOne() {
  if (!img) return;
  const c = gl.canvas;
  view.zoom = dpr(); // 1 image px = 1 css px
  view.panX = (c.width - img.w * view.zoom) / 2;
  view.panY = (c.height - img.h * view.zoom) / 2;
  render();
}
function zoomToRegion() {
  if (!roi || !img) return;
  const c = gl.canvas;
  const rw = roi.x1 - roi.x0, rh = roi.y1 - roi.y0;
  if (rw < 1 || rh < 1) return;
  view.zoom = Math.min(c.width / rw, c.height / rh) * 0.95;
  view.panX = (c.width - (roi.x0 + roi.x1) * view.zoom) / 2;
  view.panY = (c.height - (roi.y0 + roi.y1) * view.zoom) / 2;
  render();
}
function updateZoomLabel() {
  const pct = Math.round((view.zoom / dpr()) * 100);
  document.getElementById("zoomLabel").textContent = pct + "%";
}

/* ----------------------------------------------------------------- decode */
function setProgress(pct, label) {
  const wrap = document.getElementById("progressWrap");
  wrap.classList.toggle("hidden", pct >= 100 || pct < 0);
  document.getElementById("progressBar").style.width = clamp(pct, 0, 100) + "%";
  document.getElementById("progressLabel").textContent = label || "";
}

function readFileWithProgress(file) {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onprogress = (e) => {
      if (e.lengthComputable)
        setProgress((e.loaded / e.total) * 100, "Reading… " +
          Math.round((e.loaded / e.total) * 100) + "%");
    };
    fr.onload = () => resolve(new Uint8Array(fr.result));
    fr.onerror = () => reject(fr.error || new Error("read failed"));
    fr.readAsArrayBuffer(file);
  });
}

// Fetch a URL into a Uint8Array, streaming download progress when the server
// reports a Content-Length (e.g. the remote asakusa.exr).
async function fetchUrlWithProgress(url, what) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(resp.status + " " + resp.statusText);
  const total = +resp.headers.get("content-length");
  if (!resp.body || !total) {
    setProgress(0, "Fetching " + what + "…");
    return new Uint8Array(await resp.arrayBuffer());
  }
  const reader = resp.body.getReader();
  const buf = new Uint8Array(total);
  let off = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf.set(value, off);
    off += value.length;
    setProgress((off / total) * 100,
      "Fetching " + what + "… " + Math.round((off / total) * 100) + "%");
  }
  return off === total ? buf : buf.subarray(0, off);
}

async function loadBytes(bytes, name) {
  hideError();
  fileName = name || null;
  if (handle) { M._exrv_close(handle); handle = 0; img = null; }

  const p = M._malloc(bytes.length);
  M.HEAPU8.set(bytes, p);
  handle = M._exrv_open(p, bytes.length);
  M._free(p);
  if (!handle) { setProgress(-1); showError("Not a valid EXR file."); return; }

  const jp = M._exrv_header_json(handle);
  header = JSON.parse(M.UTF8ToString(jp));
  M._exrv_free(jp);

  document.getElementById("dropzone").classList.add("hidden");
  document.getElementById("ctrls").classList.remove("hidden");
  document.getElementById("info").classList.remove("hidden");
  document.getElementById("viewbar").classList.remove("hidden");

  buildPartSelector();
  views = [];                 // force a fresh channel-view list for the new file
  await selectAndDecode(0, 0, 0, true);
}

async function selectAndDecode(part, lx, ly, doFit, viewIdx) {
  // Clear the canvas immediately so a part/view switch doesn't leave the old
  // image (or a partially-decoded frame) on screen while the new one decodes.
  img = null; render();

  roi = null;
  updateRoiUI();
  const ph = header.parts[part];
  dwMin = { x: ph.dataWindow.minX, y: ph.dataWindow.minY };
  dispWin = ph.displayWindow;

  // Rebuild the channel-view list when the part changes; otherwise keep it.
  const partChanged = sel.part !== part || views.length === 0;
  if (partChanged) { views = buildViews(ph); buildChannelSelector(); }
  if (viewIdx === undefined) viewIdx = partChanged ? 0 : (sel.view || 0);
  if (viewIdx < 0 || viewIdx >= views.length) viewIdx = 0;
  sel = { part, lx, ly, view: viewIdx };
  document.getElementById("channelSel").value = String(viewIdx);

  const v = views[viewIdx];
  const total = M._exrv_select_channels(handle, part, lx, ly,
    v.c[0], v.c[1], v.c[2], v.c[3]);
  if (total < 0) {
    setProgress(-1);
    img = null; render();
    showError(ph.deep ? "Deep EXR parts are not supported by this viewer."
                      : "Unable to decode this part/level.");
    renderInfo();
    return;
  }
  const w = M._exrv_level_width(handle), h = M._exrv_level_height(handle);

  setProgress(0, "Decoding 0%");
  const chunk = Math.max(1, Math.ceil(total / 100));
  for (let i = 0; i < total; i++) {
    if (M._exrv_decode_block(handle, i) < 0) {
      setProgress(-1); showError("Decode error in block " + i + "."); break;
    }
    if (i % chunk === 0 || i === total - 1) {
      const pct = ((i + 1) / total) * 100;
      setProgress(pct, "Decoding " + Math.round(pct) + "%");
      await raf();
    }
  }
  setProgress(100);

  const ptr = M._exrv_rgba(handle);
  // copy out of the heap: HEAPF32 can detach on memory growth
  const data = M.HEAPF32.slice(ptr >> 2, (ptr >> 2) + w * h * 4);
  img = { w, h, data };
  uploadTexture();
  buildLevelSelector();
  renderInfo();
  if (doFit) fitView(); else render();
}

/* ----------------------------------------------------------------- picker */
function updatePixel(sx, sy) {
  const el = document.getElementById("pixel");
  if (!img) { el.classList.add("hidden"); return; }
  const [fx, fy] = screenToImg(sx, sy);
  const ix = Math.floor(fx), iy = Math.floor(fy);
  if (ix < 0 || iy < 0 || ix >= img.w || iy >= img.h) {
    el.classList.add("hidden");
    return;
  }
  const o = (iy * img.w + ix) * 4;
  const r = img.data[o], g = img.data[o + 1], b = img.data[o + 2], a = img.data[o + 3];
  const ax = ix + dwMin.x, ay = iy + dwMin.y;
  const f = (x) => (Math.abs(x) >= 1e4 || (Math.abs(x) < 1e-4 && x !== 0))
    ? x.toExponential(3) : x.toFixed(4);
  el.textContent =
    `x ${ax}  y ${ay}\nR ${f(r)}\nG ${f(g)}\nB ${f(b)}\nA ${f(a)}`;
  el.classList.remove("hidden");
}

/* --------------------------------------------------------------- info ui */
function buildPartSelector() {
  const partSel = document.getElementById("partSel");
  partSel.innerHTML = "";
  header.parts.forEach((p, i) => {
    const opt = document.createElement("option");
    opt.value = i;
    opt.textContent = `${i}: ${p.name || "(unnamed)"} — ${p.type}`;
    partSel.appendChild(opt);
  });
  partSel.value = "0";
  partSel.parentElement.style.display = header.numParts > 1 ? "" : "none";
}

// Derive the selectable channel "views" for a part: conventional RGB(A),
// luminance-chroma (Y/RY/BY → color), named layers (prefix.R/G/B), a generic
// leading-channels-to-RGB for data parts, and every channel as grayscale.
// View.c holds header channel indices (matching p.channels order); -1 = unused.
function buildViews(p) {
  const ch = p.channels;
  const idx = (name) => ch.findIndex((c) => c.name === name);
  const primary = [];
  const R = idx("R"), G = idx("G"), B = idx("B"), A = idx("A");
  if (R >= 0 && G >= 0 && B >= 0)
    primary.push({ label: A >= 0 ? "RGBA" : "RGB", c: [R, G, B, A] });
  const Y = idx("Y");
  // Luminance-chroma (Y + RY/BY) is shown as grayscale Y; full-color YCbCr
  // reconstruction is pending a v3-core fix for subsampled-channel decode.

  // Named layers: group channels by the prefix before the last '.'.
  const layers = new Map();
  ch.forEach((c, i) => {
    const dot = c.name.lastIndexOf(".");
    if (dot > 0) {
      const pre = c.name.slice(0, dot), suf = c.name.slice(dot + 1);
      if (!layers.has(pre)) layers.set(pre, {});
      layers.get(pre)[suf] = i;
    }
  });
  for (const [pre, m] of layers) {
    if (m.R !== undefined && m.G !== undefined && m.B !== undefined)
      primary.push({ label: `${pre} (RGB)`,
        c: [m.R, m.G, m.B, m.A !== undefined ? m.A : -1] });
  }

  if (!primary.length) {
    if (Y >= 0) primary.push({ label: "Y (luminance)", c: [Y, Y, Y, -1] });
    else if (ch.length >= 2) // motion vectors / disparity / generic data
      primary.push({ label: "Channels → RGB", c: [0, 1, ch.length >= 3 ? 2 : -1, -1] });
  }

  const individual = ch.map((c, i) => ({ label: `${c.name} (gray)`, c: [i, i, i, -1] }));
  return [...primary, ...individual];
}

function buildChannelSelector() {
  const channelSel = document.getElementById("channelSel");
  channelSel.innerHTML = "";
  views.forEach((v, i) => {
    const opt = document.createElement("option");
    opt.value = i;
    opt.textContent = v.label;
    channelSel.appendChild(opt);
  });
  channelSel.value = "0";
  // Hide the combo when there's only one trivial view (e.g. a plain RGB image).
  document.getElementById("channelView").style.display = views.length > 1 ? "" : "none";
}

function buildLevelSelector() {
  const levelSel = document.getElementById("levelSel");
  const p = header.parts[sel.part];
  levelSel.innerHTML = "";
  const levels = p.levels && p.levels.length ? p.levels : [[0, 0]];
  levels.forEach(([lx, ly]) => {
    const opt = document.createElement("option");
    opt.value = lx + "," + ly;
    opt.textContent = (lx === 0 && ly === 0) ? "0 (full)" : `${lx},${ly}`;
    levelSel.appendChild(opt);
  });
  levelSel.value = sel.lx + "," + sel.ly;
  levelSel.parentElement.style.display = levels.length > 1 ? "" : "none";
}

function esc(s) {
  return String(s).replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
function kv(k, v) { return `<div class="kv"><span class="k">${k}</span><span class="v">${v}</span></div>`; }
function box(b) { return `[${b.minX}, ${b.minY}] → [${b.maxX}, ${b.maxY}]`; }

function renderInfo() {
  const p = header.parts[sel.part];
  const body = document.getElementById("infoBody");
  let h = "";
  if (fileName) h += kv("File", esc(fileName));
  h += kv("Parts", header.numParts);
  h += kv("Type", p.type + (p.tiled ? ` (${p.tileXSize}×${p.tileYSize} tiles, ${p.levelMode})` : ""));
  h += kv("Compression", p.compression);
  h += kv("Line order", p.lineOrder);
  h += kv("Resolution", `${p.width} × ${p.height}`);
  if (img && (img.w !== p.width || img.h !== p.height))
    h += kv("Level size", `${img.w} × ${img.h}`);
  h += kv("Data window", box(p.dataWindow));
  h += kv("Display window", box(p.displayWindow));
  h += kv("Pixel aspect", p.pixelAspect);
  h += `<div class="subhead">Channels (${p.channels.length})</div>`;
  h += `<table class="chan-table">`;
  for (const c of p.channels)
    h += `<tr><td>${c.name}</td><td>${c.type}${(c.xSampling !== 1 || c.ySampling !== 1) ? ` ${c.xSampling}×${c.ySampling}` : ""}</td></tr>`;
  h += `</table>`;
  body.innerHTML = h;
}

function updateRoiUI() {
  const wrap = document.getElementById("roiInfo");
  const btn = document.getElementById("btnZoomRegion");
  if (roi && (roi.x1 - roi.x0) >= 1 && (roi.y1 - roi.y0) >= 1) {
    wrap.classList.remove("hidden");
    btn.disabled = false;
    const ax0 = Math.round(roi.x0) + dwMin.x, ay0 = Math.round(roi.y0) + dwMin.y;
    const w = Math.round(roi.x1 - roi.x0), hh = Math.round(roi.y1 - roi.y0);
    document.getElementById("roiText").textContent =
      `origin ${ax0}, ${ay0}   size ${w} × ${hh}`;
  } else {
    wrap.classList.add("hidden");
    btn.disabled = true;
  }
}

/* ----------------------------------------------------------------- errors */
let errTimer = 0;
function showError(msg) {
  const t = document.getElementById("errorToast");
  t.textContent = msg;
  t.classList.remove("hidden");
  clearTimeout(errTimer);
  errTimer = setTimeout(() => t.classList.add("hidden"), 5000);
}
function hideError() { document.getElementById("errorToast").classList.add("hidden"); }

/* ------------------------------------------------------------------ input */
function setupInput() {
  const stage = document.getElementById("stage");
  const glc = document.getElementById("gl");

  // file picker
  document.getElementById("file").addEventListener("change", async (e) => {
    const f = e.target.files[0];
    if (f) loadBytes(await readFileWithProgress(f), f.name);
  });

  // sample
  document.getElementById("btnSample").addEventListener("click", async () => {
    try {
      setProgress(0, "Fetching sample.exr…");
      const resp = await fetch("sample.exr");
      if (!resp.ok) throw new Error("sample.exr not found");
      loadBytes(new Uint8Array(await resp.arrayBuffer()), "sample.exr");
    } catch (err) {
      setProgress(-1);
      showError("Place an EXR named sample.exr next to index.html (see README).");
    }
  });

  // asakusa — fetched over HTTP from the tinyexr repo (release branch)
  const ASAKUSA_URL =
    "https://raw.githubusercontent.com/syoyo/tinyexr/release/asakusa.exr";
  document.getElementById("btnAsakusa").addEventListener("click", async () => {
    try {
      loadBytes(await fetchUrlWithProgress(ASAKUSA_URL, "asakusa.exr"), "asakusa.exr");
    } catch (err) {
      setProgress(-1);
      showError("Could not fetch asakusa.exr (" + err.message + ").");
    }
  });

  // drag & drop
  ["dragenter", "dragover"].forEach((ev) =>
    stage.addEventListener(ev, (e) => {
      e.preventDefault();
      document.getElementById("dropzone").classList.remove("hidden");
      document.getElementById("dropzone").classList.add("drag");
    }));
  ["dragleave", "drop"].forEach((ev) =>
    stage.addEventListener(ev, (e) => {
      e.preventDefault();
      document.getElementById("dropzone").classList.remove("drag");
      if (ev === "dragleave" && img)
        document.getElementById("dropzone").classList.add("hidden");
    }));
  stage.addEventListener("drop", async (e) => {
    const f = e.dataTransfer.files[0];
    if (f) loadBytes(await readFileWithProgress(f), f.name);
  });

  // wheel zoom (to cursor)
  glc.addEventListener("wheel", (e) => {
    if (!img) return;
    e.preventDefault();
    const r = glc.getBoundingClientRect();
    const mx = (e.clientX - r.left) * dpr(), my = (e.clientY - r.top) * dpr();
    const [ix, iy] = screenToImg(mx, my);
    view.zoom = clamp(view.zoom * Math.exp(-e.deltaY * 0.0015), 0.01, 5000);
    view.panX = mx - ix * view.zoom;
    view.panY = my - iy * view.zoom;
    render();
  }, { passive: false });

  // drag: pan, or shift-drag: ROI
  let drag = null;
  glc.addEventListener("mousedown", (e) => {
    if (!img) return;
    const r = glc.getBoundingClientRect();
    const mx = (e.clientX - r.left) * dpr(), my = (e.clientY - r.top) * dpr();
    if (e.shiftKey) {
      const [ix, iy] = screenToImg(mx, my);
      drag = { mode: "roi", ix, iy };
      glc.classList.add("roi");
    } else {
      drag = { mode: "pan", mx, my, panX: view.panX, panY: view.panY };
      glc.classList.add("panning");
    }
  });
  window.addEventListener("mousemove", (e) => {
    const r = glc.getBoundingClientRect();
    const mx = (e.clientX - r.left) * dpr(), my = (e.clientY - r.top) * dpr();
    if (drag && drag.mode === "pan") {
      view.panX = drag.panX + (mx - drag.mx);
      view.panY = drag.panY + (my - drag.my);
      render();
    } else if (drag && drag.mode === "roi") {
      const [ix, iy] = screenToImg(mx, my);
      roi = {
        x0: clamp(Math.min(drag.ix, ix), 0, img.w),
        y0: clamp(Math.min(drag.iy, iy), 0, img.h),
        x1: clamp(Math.max(drag.ix, ix), 0, img.w),
        y1: clamp(Math.max(drag.iy, iy), 0, img.h),
      };
      updateRoiUI();
      render();
    } else {
      updatePixel(mx, my);
    }
  });
  window.addEventListener("mouseup", () => {
    drag = null;
    glc.classList.remove("panning", "roi");
  });
  glc.addEventListener("mouseleave", () =>
    document.getElementById("pixel").classList.add("hidden"));

  // view buttons
  document.getElementById("btnFit").addEventListener("click", fitView);
  document.getElementById("btn11").addEventListener("click", oneToOne);
  document.getElementById("btnZoomRegion").addEventListener("click", zoomToRegion);
  document.getElementById("clearRoi").addEventListener("click", () => {
    roi = null; updateRoiUI(); render();
  });

  // controls
  const exp = document.getElementById("exposure");
  exp.addEventListener("input", () => {
    ctl.exposure = parseFloat(exp.value);
    document.getElementById("expVal").textContent =
      (ctl.exposure >= 0 ? "+" : "") + ctl.exposure.toFixed(1) + " EV";
    render();
  });
  const gam = document.getElementById("gamma");
  gam.addEventListener("input", () => {
    ctl.gamma = parseFloat(gam.value);
    document.getElementById("gammaVal").textContent = ctl.gamma.toFixed(2);
    render();
  });
  document.getElementById("srgb").addEventListener("change", (e) => {
    ctl.srgb = e.target.checked;
    document.getElementById("gamma").disabled = ctl.srgb;
    render();
  });
  document.getElementById("channels").addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    ctl.channel = parseInt(b.dataset.ch, 10);
    for (const x of e.currentTarget.children) x.classList.toggle("active", x === b);
    // False color only applies to a single scalar channel (not RGB).
    document.getElementById("falseColor").disabled = ctl.channel === 0;
    render();
  });
  document.getElementById("falseColor").addEventListener("change", (e) => {
    ctl.falseColor = e.target.checked;
    render();
  });
  document.getElementById("showDisp").addEventListener("change", (e) => {
    ctl.showDisp = e.target.checked; render();
  });
  document.getElementById("cropDisp").addEventListener("change", (e) => {
    ctl.cropDisp = e.target.checked; render();
  });
  document.getElementById("partSel").addEventListener("change", (e) => {
    selectAndDecode(parseInt(e.target.value, 10), 0, 0, true);
  });
  document.getElementById("levelSel").addEventListener("change", (e) => {
    const [lx, ly] = e.target.value.split(",").map(Number);
    selectAndDecode(sel.part, lx, ly, true);
  });
  document.getElementById("channelSel").addEventListener("change", (e) => {
    selectAndDecode(sel.part, sel.lx, sel.ly, false, parseInt(e.target.value, 10));
  });

  window.addEventListener("resize", () => { syncCanvasSize(); render(); });
}

/* -------------------------------------------------------------------- boot */
/* ----------------------------------------------- openexr-images browser ---- */
// Browse the official OpenEXR sample-image library and load any image over HTTP.
// One GitHub tree API call lists every .exr; images are fetched from raw.*.
const OEXR_REPO = "AcademySoftwareFoundation/openexr-images";
const OEXR_TREE_API =
  "https://api.github.com/repos/" + OEXR_REPO + "/git/trees/main?recursive=1";
const OEXR_RAW_BASE = "https://raw.githubusercontent.com/" + OEXR_REPO + "/main/";

function humanSize(b) {
  if (b < 1024) return b + " B";
  if (b < 1048576) return Math.round(b / 1024) + " KB";
  return (b / 1048576).toFixed(1) + " MB";
}

function setupBrowser() {
  const modal = document.getElementById("browser");
  const treeEl = document.getElementById("browserTree");
  const filterEl = document.getElementById("browserFilter");
  let allFiles = null; // cached [{path, name, size}] after first load

  const msg = (text, err) => {
    const d = document.createElement("div");
    d.className = "msg" + (err ? " err" : "");
    d.textContent = text;
    treeEl.replaceChildren(d);
  };

  const close = () => modal.classList.add("hidden");
  const open = () => {
    modal.classList.remove("hidden");
    if (allFiles === null) load();
    filterEl.focus();
  };

  // Build a nested {dirs:Map, files:[]} tree from flat paths.
  function buildTree(files) {
    const root = { dirs: new Map(), files: [] };
    for (const f of files) {
      const parts = f.path.split("/");
      let node = root;
      for (let i = 0; i < parts.length - 1; i++) {
        if (!node.dirs.has(parts[i]))
          node.dirs.set(parts[i], { dirs: new Map(), files: [] });
        node = node.dirs.get(parts[i]);
      }
      node.files.push(f);
    }
    return root;
  }
  const countFiles = (n) =>
    n.files.length + [...n.dirs.values()].reduce((s, c) => s + countFiles(c), 0);

  async function loadRepoFile(f) {
    close();
    try {
      const url = OEXR_RAW_BASE + f.path.split("/").map(encodeURIComponent).join("/");
      loadBytes(await fetchUrlWithProgress(url, f.name), f.path);
    } catch (err) {
      setProgress(-1);
      showError("Could not fetch " + f.name + " (" + err.message + ").");
    }
  }

  function renderNode(node, openAll) {
    const frag = document.createDocumentFragment();
    for (const [name, child] of [...node.dirs.entries()].sort((a, b) =>
      a[0].localeCompare(b[0]))) {
      const det = document.createElement("details");
      det.open = openAll;
      const sum = document.createElement("summary");
      sum.textContent = name;
      const cnt = document.createElement("span");
      cnt.className = "count";
      cnt.textContent = countFiles(child) + (countFiles(child) === 1 ? " file" : " files");
      sum.appendChild(cnt);
      const kids = document.createElement("div");
      kids.className = "children";
      kids.appendChild(renderNode(child, openAll));
      det.append(sum, kids);
      frag.appendChild(det);
    }
    for (const f of [...node.files].sort((a, b) => a.name.localeCompare(b.name))) {
      const btn = document.createElement("button");
      btn.className = "tree-file";
      btn.title = "Load " + f.path;
      let thumb;
      if (f.thumb) {
        thumb = document.createElement("img");
        thumb.className = "thumb";
        thumb.loading = "lazy";
        thumb.decoding = "async";
        thumb.alt = "";
        thumb.width = 46; thumb.height = 30;
        thumb.src = f.thumb;
        // Fall back to the checker placeholder if the preview fails to load.
        thumb.addEventListener("error", () => {
          thumb.removeAttribute("src");
          thumb.classList.add("noimg");
        });
      } else {
        thumb = document.createElement("span");
        thumb.className = "thumb noimg";
      }
      const nm = document.createElement("span");
      nm.className = "nm";
      nm.textContent = f.name;
      const sz = document.createElement("span");
      sz.className = "sz";
      sz.textContent = humanSize(f.size);
      btn.append(thumb, nm, sz);
      btn.addEventListener("click", () => loadRepoFile(f));
      frag.appendChild(btn);
    }
    return frag;
  }

  function rerender() {
    const q = filterEl.value.trim().toLowerCase();
    const files = q ? allFiles.filter((f) => f.path.toLowerCase().includes(q)) : allFiles;
    if (!files.length) { msg("No matching .exr files."); return; }
    treeEl.replaceChildren(renderNode(buildTree(files), !!q));
  }

  async function load() {
    msg("Loading image list…");
    try {
      const resp = await fetch(OEXR_TREE_API);
      if (!resp.ok) throw new Error(resp.status + " " + resp.statusText);
      const data = await resp.json();
      // Each .exr in this repo has a sibling full-res .jpg preview; use it as a
      // browser-native thumbnail (lazy-loaded). Note which siblings exist.
      const blobs = new Set(data.tree.filter((b) => b.type === "blob").map((b) => b.path));
      const rawUrl = (p) => OEXR_RAW_BASE + p.split("/").map(encodeURIComponent).join("/");
      allFiles = data.tree
        .filter((b) => b.type === "blob" && b.path.toLowerCase().endsWith(".exr"))
        .map((b) => {
          const jpg = b.path.replace(/\.exr$/i, ".jpg");
          return {
            path: b.path,
            name: b.path.split("/").pop(),
            size: b.size || 0,
            thumb: blobs.has(jpg) ? rawUrl(jpg) : null,
          };
        });
      rerender();
    } catch (err) {
      allFiles = null;
      msg("Failed to load list (" + err.message + "). The GitHub API may be rate-limited; try again later.", true);
    }
  }

  document.getElementById("btnBrowse").addEventListener("click", open);
  document.getElementById("browserClose").addEventListener("click", close);
  modal.addEventListener("click", (e) => { if (e.target === modal) close(); });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !modal.classList.contains("hidden")) close();
  });
  filterEl.addEventListener("input", () => { if (allFiles) rerender(); });
}

(async function main() {
  setupInput();
  setupBrowser();
  initLegendGradient();
  document.getElementById("gamma").disabled = ctl.srgb;
  try {
    M = await createModule();
  } catch (e) {
    showError("Failed to load WASM module. Did you run ./build.sh and serve over HTTP?");
    return;
  }
  initGL();
  render();
})();
