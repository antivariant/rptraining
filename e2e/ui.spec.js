const { test, expect } = require("@playwright/test");

async function findTextarea(page, label) {
  for (let attempt = 0; attempt < 20; attempt += 1) {
    const labelLocator = page.getByText(label, { exact: true }).first();
    if (await labelLocator.count()) {
      const iframeHandle = await labelLocator.locator("xpath=following::iframe[1]").elementHandle();
      if (iframeHandle) {
        const frame = await iframeHandle.contentFrame();
        if (frame) {
          const textarea = frame.getByRole("textbox").first();
          if ((await textarea.count()) > 0) {
            return textarea;
          }
        }
      }
    }
    await page.waitForTimeout(250);
  }
  throw new Error(`Textarea with label "${label}" was not found.`);
}

test("scene textarea keeps text after blur and uses lighter font", async ({ page }) => {
  await page.goto("/");

  const sceneTextarea = await findTextarea(page, "Scene description / initial state");
  const sceneText = "сабака лаяла на карову и не исчезала после blur";

  await sceneTextarea.click();
  await sceneTextarea.fill(sceneText);
  await page.keyboard.press("Tab");

  await expect(sceneTextarea).toHaveValue(sceneText);
  const fontWeight = await sceneTextarea.evaluate((element) => window.getComputedStyle(element).fontWeight);
  expect(Number(fontWeight)).toBeLessThanOrEqual(500);
});

test("session inputs keep user text stable across focus changes", async ({ page }) => {
  await page.goto("/");

  await page.getByRole("combobox", { name: /User character/i }).click();
  await page.getByText("Nyanix", { exact: true }).last().click();

  const sceneTextarea = await findTextarea(page, "Scene description / initial state");
  await sceneTextarea.fill("DATA пытается пристыдить Nyanix за имитацию бурной деятельности.");
  await page.keyboard.press("Tab");
  await expect(sceneTextarea).toHaveValue("DATA пытается пристыдить Nyanix за имитацию бурной деятельности.");
  await page.waitForTimeout(250);

  await page.getByRole("button", { name: "Start new session" }).click();
  await expect(page.getByText("Session created.")).toBeVisible({ timeout: 20000 });

  const actionTextarea = await findTextarea(page, "Your action");
  const dialogueTextarea = await findTextarea(page, "Your dialogue");

  const actionText = "Подозрительно медленно кивает.";
  const dialogueText = "Я просто берегу энергию для драматического финала.";

  await actionTextarea.fill(actionText);
  await dialogueTextarea.click();
  await expect(actionTextarea).toHaveValue(actionText);

  await dialogueTextarea.fill(dialogueText);
  await page.keyboard.press("Tab");
  await expect(dialogueTextarea).toHaveValue(dialogueText);
});
