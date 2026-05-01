import { execSync } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const frontendRoot = resolve(__dirname, "..");
const workspaceRoot = resolve(frontendRoot, "..");
const publicDir = resolve(frontendRoot, "public");
const wasmSrc = resolve(workspaceRoot, "go", "nn.wasm");
const wasmDest = resolve(publicDir, "nn.wasm");

if (!existsSync(wasmSrc)) {
  throw new Error(
    `WASM-Datei nicht gefunden: ${wasmSrc}. Baue zuerst mit: cd go && GOOS=js GOARCH=wasm go build -o nn.wasm .`,
  );
}

const goroot = execSync("go env GOROOT", { encoding: "utf8" }).trim();
const wasmExecCandidates = [
  resolve(goroot, "lib", "wasm", "wasm_exec.js"),
  resolve(goroot, "misc", "wasm", "wasm_exec.js"),
];

const wasmExecSrc = wasmExecCandidates.find((candidate) =>
  existsSync(candidate),
);
if (!wasmExecSrc) {
  throw new Error(`wasm_exec.js nicht gefunden unter ${goroot}`);
}

mkdirSync(publicDir, { recursive: true });
copyFileSync(wasmSrc, wasmDest);
copyFileSync(wasmExecSrc, resolve(publicDir, "wasm_exec.js"));

console.log(
  "WASM-Artefakte synchronisiert: public/nn.wasm, public/wasm_exec.js",
);
