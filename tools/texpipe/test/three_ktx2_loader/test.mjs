/*
 * TinyEXR texpipe - standalone Three.js KTX2Loader interoperability test.
 *
 * Copyright (c) 2014-2026 Syoyo Fujita and TinyEXR authors
 * SPDX-License-Identifier: BSD-3-Clause
 */

import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import process from 'node:process';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const TEST_DIR = path.dirname(fileURLToPath(import.meta.url));
const fixture = path.resolve(process.argv[2] || 'build/texpipe/three-ktx2.ktx2');
const dependencyRoot = path.resolve(
  process.env.THREE_KTX2_NODE_ROOT || TEST_DIR);
const nodeModules = path.join(dependencyRoot, 'node_modules');
const require = createRequire(path.join(dependencyRoot, 'package.json'));

function fail(message) {
  throw new Error(`three-ktx2: ${message}`);
}

if (!fs.existsSync(fixture)) fail(`fixture not found: ${fixture}`);
if (!fs.existsSync(path.join(nodeModules, 'three/build/three.module.js'))) {
  fail(`Three.js not found under ${nodeModules}; run npm install in ${TEST_DIR} ` +
    'or set THREE_KTX2_NODE_ROOT');
}

let puppeteer;
try {
  puppeteer = require('puppeteer');
} catch (error) {
  fail(`Puppeteer not found under ${nodeModules}: ${error.message}`);
}

const html = `<!doctype html>
<meta charset="utf-8">
<canvas id="canvas" width="8" height="8"></canvas>
<script type="importmap">
{"imports":{"three":"/node_modules/three/build/three.module.js",
"three/addons/":"/node_modules/three/examples/jsm/"}}
</script>
<script type="module">
import * as THREE from 'three';
import { KTX2Loader } from 'three/addons/loaders/KTX2Loader.js';

try {
  const renderer = new THREE.WebGLRenderer({canvas: document.querySelector('#canvas')});
  const loader = new KTX2Loader()
    .setTranscoderPath('/node_modules/three/examples/jsm/libs/basis/')
    .detectSupport(renderer);
  const texture = await loader.loadAsync('/fixture.ktx2');
  const mip = texture.mipmaps[0];
  const scene = new THREE.Scene();
  const material = new THREE.MeshBasicMaterial({map: texture});
  const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material);
  scene.add(quad);
  const target = new THREE.WebGLRenderTarget(8, 8);
  const pixels = new Uint8Array(8 * 8 * 4);
  renderer.setRenderTarget(target);
  renderer.render(scene, new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1));
  renderer.readRenderTargetPixels(target, 0, 0, 8, 8, pixels);
  let redMin = 255, redMax = 0, greenMin = 255, greenMax = 0;
  for (let offset = 0; offset < pixels.length; offset += 4) {
    redMin = Math.min(redMin, pixels[offset]);
    redMax = Math.max(redMax, pixels[offset]);
    greenMin = Math.min(greenMin, pixels[offset + 1]);
    greenMax = Math.max(greenMax, pixels[offset + 1]);
  }
  window.__threeKTX2Result = {
    compressed: texture.isCompressedTexture === true,
    width: texture.image.width,
    height: texture.image.height,
    mipLevels: texture.mipmaps.length,
    byteLength: mip.data.byteLength,
    colorSpace: texture.colorSpace,
    format: texture.format,
    redRange: redMax - redMin,
    greenRange: greenMax - greenMin
  };
  target.dispose();
  quad.geometry.dispose();
  material.dispose();
  loader.dispose();
  renderer.dispose();
} catch (error) {
  window.__threeKTX2Error = error?.stack || error?.message || String(error);
}
</script>`;

const mime = new Map([
  ['.html', 'text/html; charset=utf-8'],
  ['.js', 'text/javascript; charset=utf-8'],
  ['.mjs', 'text/javascript; charset=utf-8'],
  ['.wasm', 'application/wasm'],
  ['.ktx2', 'image/ktx2']
]);

function sendFile(response, filename) {
  response.writeHead(200, {
    'Content-Type': mime.get(path.extname(filename)) || 'application/octet-stream'
  });
  fs.createReadStream(filename).pipe(response);
}

const server = http.createServer((request, response) => {
  const pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
  if (pathname === '/') {
    response.writeHead(200, {'Content-Type': mime.get('.html')});
    response.end(html);
    return;
  }
  if (pathname === '/fixture.ktx2') {
    sendFile(response, fixture);
    return;
  }
  if (pathname.startsWith('/node_modules/')) {
    const relative = pathname.slice('/node_modules/'.length);
    const filename = path.resolve(nodeModules, relative);
    const root = `${path.resolve(nodeModules)}${path.sep}`;
    if (filename.startsWith(root) && fs.statSync(filename, {throwIfNoEntry: false})?.isFile()) {
      sendFile(response, filename);
      return;
    }
  }
  response.writeHead(404);
  response.end('not found');
});

let browser;
try {
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  const address = server.address();
  const launch = {
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage',
      '--use-angle=swiftshader', '--enable-unsafe-swiftshader']
  };
  if (process.env.PUPPETEER_EXECUTABLE_PATH) {
    launch.executablePath = process.env.PUPPETEER_EXECUTABLE_PATH;
  }
  browser = await puppeteer.launch(launch);
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${address.port}/`, {waitUntil: 'load'});
  await page.waitForFunction(
    () => window.__threeKTX2Result || window.__threeKTX2Error,
    {timeout: 60000});
  const state = await page.evaluate(() => ({
    result: window.__threeKTX2Result,
    error: window.__threeKTX2Error
  }));
  if (state.error) fail(`KTX2Loader failed: ${state.error}`);
  const result = state.result;
  if (!result?.compressed || result.width !== 8 || result.height !== 8 ||
      result.mipLevels !== 1 || result.byteLength !== 64 ||
      result.colorSpace !== 'srgb' || result.redRange < 100 ||
      result.greenRange < 100) {
    fail(`unexpected texture: ${JSON.stringify(result)}`);
  }
  console.log(`three-ktx2: PASS ${JSON.stringify(result)}`);
} finally {
  await browser?.close().catch(() => {});
  await new Promise((resolve) => server.close(resolve));
}
