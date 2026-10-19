import { defineConfig, devices } from '@playwright/test';

// Runs against the built site (`pnpm build` first) served by `vite preview`.
// The base path must match the build: see vite.config.ts.
const port = 4173;
const base = process.env.GITHUB_PAGES === '1' ? '/dreamscapes/' : '/';
const origin = `http://127.0.0.1:${String(port)}`;
const isCI = Boolean(process.env.CI);

export default defineConfig({
  testDir: 'e2e',
  forbidOnly: isCI,
  retries: isCI ? 1 : 0,
  reporter: isCI ? [['github'], ['list']] : 'list',
  use: {
    baseURL: `${origin}${base}`,
    trace: 'retain-on-failure',
  },
  projects: [
    {
      name: 'desktop',
      use: { ...devices['Desktop Chrome'], viewport: { width: 1440, height: 900 } },
    },
    {
      // The iPhone 13 profile (viewport, pixel ratio, touch, mobile user agent) run in Chromium,
      // the one browser CI installs. It exercises the portrait crop, panning and touch taps;
      // WebKit itself would need its own system packages on the CI runner.
      name: 'iphone-13',
      use: { ...devices['iPhone 13'], browserName: 'chromium' },
    },
  ],
  webServer: {
    command: `pnpm exec vite preview --host 127.0.0.1 --port ${String(port)} --strictPort`,
    url: `${origin}${base}`,
    reuseExistingServer: !isCI,
  },
});
