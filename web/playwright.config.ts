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
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: {
    command: `pnpm exec vite preview --host 127.0.0.1 --port ${String(port)} --strictPort`,
    url: `${origin}${base}`,
    reuseExistingServer: !isCI,
  },
});
