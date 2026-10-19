// Fails the build when the shipped JavaScript grows past its budget.
//
// The site is meant to stay tiny (vanilla DOM, no runtime dependencies), so the budget is
// checked on every build rather than eyeballed. Runs after `vite build`; Node strips the types.

import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gzipSync } from 'node:zlib';

const BUDGET_BYTES = 30 * 1024;
// fileURLToPath, not URL.pathname, so paths with spaces or on Windows come out right.
const assets = fileURLToPath(new URL('../dist/assets/', import.meta.url));

function gzippedSize(path: string): number {
  return gzipSync(readFileSync(path), { level: 9 }).length;
}

function kilobytes(bytes: number): string {
  return `${(bytes / 1024).toFixed(1)} KB`;
}

const files = readdirSync(assets).filter((name) => /\.(js|css)$/.test(name));
let scriptBytes = 0;
for (const name of files) {
  const size = gzippedSize(join(assets, name));
  if (name.endsWith('.js')) {
    scriptBytes += size;
  }
  console.log(`  ${name}: ${kilobytes(size)} gzipped`);
}
console.log(
  `JavaScript total: ${kilobytes(scriptBytes)} gzipped (budget ${kilobytes(BUDGET_BYTES)})`,
);
if (scriptBytes > BUDGET_BYTES) {
  console.error('The JavaScript bundle is over budget.');
  process.exit(1);
}
