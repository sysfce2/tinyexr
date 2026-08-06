# Three.js KTX2Loader interoperability test

This optional standalone test validates texpipe with an independent consumer:

1. `generate_fixture.c` encodes an 8×8 RGBA image to texcomp `uni` blocks and
   writes them using the standard ASTC 4×4 KTX2 carrier selected by
   `TP_UNI_ASTC_KTX2`, with Zstd supercompression (scheme 2).
2. `test.mjs` starts a loopback-only static server, launches Chrome through
   Puppeteer, and loads that file with Three.js `KTX2Loader`.
3. The test checks the compressed texture dimensions, mip count, byte length,
   and sRGB DFD result, then renders it and verifies that the decoded red/green
   gradients survive the independent decode and GPU upload.

The explicit ASTC carrier matters: texcomp's private `uni` representation is
made of valid ASTC blocks, but is not Basis UASTC wire data and must not be
labelled or consumed as such.

Install the browser dependencies once, then run the Make target from the
repository root:

```sh
cd tools/texpipe/test/three_ktx2_loader
npm install
cd ../../../..
make texpipe-three-ktx2-test
```

To reuse an existing dependency installation, point the test at a directory
containing `package.json` and `node_modules/{three,puppeteer}`:

```sh
THREE_KTX2_NODE_ROOT=/path/to/web-project \
PUPPETEER_EXECUTABLE_PATH=/usr/bin/google-chrome \
  make texpipe-three-ktx2-test
```

The test is deliberately not part of `make tools-test`: the latter remains a
self-contained pure-C gate without Node, npm, or browser requirements.
