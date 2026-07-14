const fs = require("fs");
const os = require("os");
const path = require("path");
const { defineConfig } = require("@playwright/test");

const artifactsDir = path.join(os.tmpdir(), "infernoschool-playwright");
const dbPath = path.join(artifactsDir, "infernoschool-e2e.db");

fs.mkdirSync(artifactsDir, { recursive: true });
if (fs.existsSync(dbPath)) {
  fs.unlinkSync(dbPath);
}

module.exports = defineConfig({
  testDir: "./e2e",
  timeout: 45000,
  expect: {
    timeout: 10000,
  },
  fullyParallel: false,
  retries: 0,
  reporter: "list",
  use: {
    baseURL: "http://127.0.0.1:8501",
    headless: true,
    trace: "retain-on-failure",
  },
  webServer: {
    command: "conda run -n base streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true",
    url: "http://127.0.0.1:8501",
    reuseExistingServer: true,
    timeout: 120000,
    env: {
      ...process.env,
      INFERNOSCHOOL_DB_PATH: dbPath,
      INFERNOSCHOOL_E2E_FAKE_LLM: "1",
    },
  },
});
