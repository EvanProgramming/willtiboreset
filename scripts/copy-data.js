import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const root = join(__dirname, "..");
const targetDir = join(root, "public", "data");

const files = [
  { src: join(root, "output", "prediction.json"), dest: join(targetDir, "prediction.json") },
  { src: join(root, "output", "model_performance.json"), dest: join(targetDir, "model_performance.json") },
  { src: join(root, "data", "prediction_history.json"), dest: join(targetDir, "prediction_history.json") },
  { src: join(root, "data", "tweets.json"), dest: join(targetDir, "tweets.json") },
];

function copyData() {
  mkdirSync(targetDir, { recursive: true });

  for (const { src, dest } of files) {
    if (!existsSync(src)) {
      console.warn(`Skipping missing file: ${src}`);
      continue;
    }
    copyFileSync(src, dest);
    console.log(`Copied ${src} -> ${dest}`);
  }
}

copyData();
