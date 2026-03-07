import { test, expect } from '@playwright/test';
import path from 'path';

const PHOTO = path.resolve(__dirname, '../../../tests/fixtures/photo.jpg');
const BLANK = path.resolve(__dirname, '../../../tests/fixtures/blank.jpg');

test.describe('FrameIQ upload and analysis', () => {

  test('backend health check shows ready', async ({ page }) => {
    await page.goto('/');
    await expect(page.getByText('Backend ready')).toBeVisible({ timeout: 10_000 });
  });

  test('upload valid photo → all three tabs visible', async ({ page }) => {
    await page.goto('/');
    await page.setInputFiles('input[type="file"]', PHOTO);
    await page.getByRole('button', { name: /analyse/i }).click();

    // Wait for metrics event (first fast response)
    await expect(page.getByText(/excellent|good|average|poor|terrible/i)).toBeVisible({ timeout: 60_000 });

    // Three tabs: Report, Composition, Technical
    await expect(page.getByRole('tab', { name: /report/i })).toBeVisible();
    await expect(page.getByRole('tab', { name: /composition/i })).toBeVisible();
    await expect(page.getByRole('tab', { name: /technical/i })).toBeVisible();
  });

  test('composition tab shows scene type and alignment metrics', async ({ page }) => {
    await page.goto('/');
    await page.setInputFiles('input[type="file"]', PHOTO);
    await page.getByRole('button', { name: /analyse/i }).click();
    await expect(page.getByText(/excellent|good|average|poor|terrible/i)).toBeVisible({ timeout: 60_000 });

    await page.getByRole('tab', { name: /composition/i }).click();
    await expect(page.getByText(/scene/i)).toBeVisible();
    await expect(page.getByText(/subject placement|rule of thirds|golden ratio/i)).toBeVisible();
    await expect(page.getByText(/colour harmony|color harmony/i)).toBeVisible();
  });

  test('technical tab shows BRISQUE, MUSIQ, and sharpness', async ({ page }) => {
    await page.goto('/');
    await page.setInputFiles('input[type="file"]', PHOTO);
    await page.getByRole('button', { name: /analyse/i }).click();
    await expect(page.getByText(/excellent|good|average|poor|terrible/i)).toBeVisible({ timeout: 60_000 });

    await page.getByRole('tab', { name: /technical/i }).click();
    await expect(page.getByText(/brisque/i)).toBeVisible();
    await expect(page.getByText(/musiq/i)).toBeVisible();
    await expect(page.getByText(/sharpness|edge sharpness/i)).toBeVisible();
  });

  test('report tab shows LLM sections after streaming completes', async ({ page }) => {
    await page.goto('/');
    await page.setInputFiles('input[type="file"]', PHOTO);
    await page.getByRole('button', { name: /analyse/i }).click();

    // Wait for streaming to finish (report tab becomes populated)
    await expect(page.getByText(/composition|aesthetics|technical/i)).toBeVisible({ timeout: 90_000 });
  });

  test('blank image upload shows error', async ({ page }) => {
    await page.goto('/');
    await page.setInputFiles('input[type="file"]', BLANK);
    await page.getByRole('button', { name: /analyse/i }).click();
    await expect(page.getByText(/blank|solid|not.*photograph|error/i)).toBeVisible({ timeout: 30_000 });
  });

});
