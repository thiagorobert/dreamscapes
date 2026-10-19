import { defineConfig } from 'vite';

// GitHub Pages serves a project site from /<repo>/, so a Pages build (GITHUB_PAGES=1) needs
// that prefix on every asset URL. Everywhere else (dev server, preview, other hosts) the site
// lives at the root. playwright.config.ts mirrors this so e2e tests work against either build.
const base = process.env.GITHUB_PAGES === '1' ? '/dreamscapes/' : '/';

export default defineConfig({
  base,
  build: {
    target: 'es2022',
  },
});
