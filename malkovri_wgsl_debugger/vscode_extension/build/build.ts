import { parseArgs } from "jsr:@std/cli/parse-args";
import { copy } from "jsr:@std/fs/copy";
import { ensureDir } from "jsr:@std/fs/ensure-dir";

const args = parseArgs(Deno.args, { boolean: ["watch"] });

const wasmSrc =
  "../malkovri_wgsl_debugger_wasm/pkg/malkovri_wgsl_debugger_wasm_bg.wasm";

async function copyWasm() {
  await ensureDir("out");
  await copy(wasmSrc, "out/malkovri_wgsl_debugger_wasm_bg.wasm", {
    overwrite: true,
  });
}

function bundle(watch: boolean) {
  const bundleArgs = [
    "bundle",
    "--external",
    "vscode",
    "--format",
    "cjs",
    "-o",
    "out/extension.js",
    "src/extension.ts",
  ];
  if (watch) bundleArgs.push("--watch");

  const cmd = new Deno.Command("deno", {
    args: bundleArgs,
    stdout: "inherit",
    stderr: "inherit",
  });
  const proc = cmd.spawn();
  if (!watch) {
    return proc.status.then((status) => {
      if (!status.success) Deno.exit(status.code);
    });
  }
}

await copyWasm();
await bundle(args.watch);
