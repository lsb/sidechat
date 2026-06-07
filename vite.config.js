import { defineConfig } from 'vite';
import { readFile, readdir, rm } from 'node:fs/promises';

// Inline plugin: serve ORT helper files at /ort/* from node_modules in dev,
// and emit them to dist/ort/ at build. This bypasses Vite's module pipeline
// (unlike putting them in public/, which breaks for dynamic import()).
//
// In addition, vite's default asset pipeline picks up `new URL('…', import.meta.url)`
// references inside onnxruntime-web and copies the referenced .wasm into
// dist/assets/ with a content-hashed filename. That duplicates the asyncify
// wasm (~22 MB) that we've already emitted to dist/ort/. ORT uses `wasmPaths`
// at runtime (not the hashed URL), so the dist/assets/ copy is never fetched
// — we purge it in a `closeBundle` hook below.
function ortHelpers() {
  const SRC = new URL('./node_modules/onnxruntime-web/dist/', import.meta.url);
  // The webgpu bundle of onnxruntime-web only references the asyncify variant
  // (`ort-wasm-simd-threaded.asyncify.{mjs,wasm}`), so we only ship that pair
  // instead of all four (base / jsep / jspi / asyncify). The others are
  // ~51 MB of pure dead weight for this build.
  const MATCH = /^ort-wasm-simd-threaded\.asyncify\.(mjs|wasm)$/;
  const mime = (name) => name.endsWith('.wasm') ? 'application/wasm' : 'text/javascript';

  return {
    name: 'ort-helpers',
    configureServer(server) {
      server.middlewares.use('/ort/', async (req, res, next) => {
        const name = (req.url || '').split('?')[0].replace(/^\//, '');
        if (!MATCH.test(name)) return next();
        try {
          const data = await readFile(new URL(name, SRC));
          res.setHeader('Content-Type', mime(name));
          res.setHeader('Cache-Control', 'no-cache');
          res.end(data);
        } catch (e) {
          next(e);
        }
      });
    },
    async generateBundle() {
      const names = (await readdir(SRC)).filter((n) => MATCH.test(n));
      for (const name of names) {
        this.emitFile({
          type: 'asset',
          fileName: `ort/${name}`,
          source: await readFile(new URL(name, SRC))
        });
      }
    },
    async closeBundle() {
      // Delete dist/assets/ort-wasm-*.wasm copies emitted by vite's default
      // asset handling — we only use the dist/ort/ copies via wasmPaths.
      const assetsDir = new URL('./dist/assets/', import.meta.url);
      try {
        const files = await readdir(assetsDir);
        const dupes = files.filter((f) => /^ort-wasm-simd-threaded.*\.wasm$/.test(f));
        for (const f of dupes) {
          await rm(new URL(f, assetsDir));
          console.log(`  [ort-helpers] removed duplicate dist/assets/${f}`);
        }
      } catch {
        // assets dir may not exist during incremental builds; ignore
      }
    },
  };
}

export default defineConfig({
  // Relative base so the built app works when served from any subpath
  // (e.g. http://host/dist/ or http://host/). Assets, the main JS bundle,
  // ORT helpers, and the local-models tree are all resolved by `main.js`
  // relative to `window.location`.
  base: './',
  plugins: [ortHelpers()],
  optimizeDeps: {
    exclude: ['@huggingface/transformers']
  }
});
