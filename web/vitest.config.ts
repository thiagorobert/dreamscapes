import { defineConfig, mergeConfig } from 'vitest/config';

import viteConfig from './vite.config.ts';

export default mergeConfig(
  viteConfig,
  defineConfig({
    test: {
      // Playwright owns e2e/; keep Vitest to the unit tests.
      include: ['tests/**/*.test.ts'],
      environment: 'node',
    },
  }),
);
