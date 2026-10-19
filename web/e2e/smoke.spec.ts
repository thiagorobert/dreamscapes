import { readFileSync } from 'node:fs';

import { expect, test } from '@playwright/test';

import type { Manifest } from '../src/types';

const manifest = JSON.parse(
  readFileSync(new URL('../public/scenes.json', import.meta.url), 'utf8'),
) as Manifest;
const start = manifest.scenes.find((scene) => scene.id === manifest.start);

test('loads and shows the wordmark and the start scene title', async ({ page }) => {
  expect(start).toBeDefined();
  await page.goto('./');
  await expect(page.getByRole('heading', { name: 'Dreamscapes' })).toBeVisible();
  await expect(page.getByText(start?.title ?? '', { exact: true })).toBeVisible();
});
