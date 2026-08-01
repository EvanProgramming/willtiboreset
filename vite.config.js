import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The site is served at the root of the custom domain (willtiboreset.xyz),
// so assets use a root-relative base. Override with VITE_BASE_PATH if needed.
const base = process.env.VITE_BASE_PATH || "/";

export default defineConfig({
  plugins: [react()],
  base,
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});
