# WGSL Debugger

[![Build VSIX](https://github.com/MatusT/malkovri/actions/workflows/build-vsix.yml/badge.svg)](https://github.com/MatusT/malkovri/actions/workflows/build-vsix.yml)
[![Download VSIX](https://img.shields.io/badge/download-VSIX-blue)](https://nightly.link/MatusT/malkovri/workflows/build-vsix/main/malkovri-wgsl-debugger-vsix.zip)

A DAP (Debug Adapter Protocol) debugger for WGSL shaders, with a VS Code extension.

Simulates shader execution on the CPU using [naga](https://github.com/gfx-rs/naga) and exposes (so far) step-through, and variable inspection via the standard debug adapter protocol.

Supported:
- [x] Basic compute shaders
- [x] Basic buffer global inputs
- [x] All expressions

TODO:
- [ ] Multiple threads
- [ ] Image and sampler inputs
- [ ] Support for graphics pipeline with vertex + fragment shaders
- [ ] ... and like a million things :-)

## Requirements

- [Rust](https://rustup.rs/) (edition 2024, stable toolchain)
- [wasm-pack](https://rustwasm.github.io/wasm-pack/) (for the WASM component)
- [Deno](https://deno.com/) (for the VS Code extension)
- VS Code

## Build

```sh
# Build the DAP server
cargo build --release -p malkovri_wgsl_debugger_dap

# Build the WASM component
wasm-pack build malkovri_wgsl_debugger_wasm --target bundler

# Build the VS Code extension
cd vscode_extension
deno task build
```

## Run / Install

1. Build both components above.
2. Open the **root** `malkovri_wgsl_debugger/` folder in VS Code.
3. Press **F5** to build and launch the Extension Development Host.
4. Open a `.wgsl` file and create a launch configuration in `.vscode/launch.json`:

```json
{
  "type": "wgsl",
  "request": "launch",
  "name": "Debug shader",
  "program": "${workspaceFolder}/shader.wgsl",
  "shaderInputs": {
    "global_invocation_id": [0, 0, 0]
  },
  "bindings": {
    "0:0": {
      "type": "f32",
      "inline": [1.0, 2.0, 3.0, 4.0]
    }
  }
}
```

5. Press **F5** to start debugging.

## Launch config options

| Field                  | Type                           | Description                                                                        |
|------------------------|--------------------------------|------------------------------------------------------------------------------------|
| `program`              | string                         | Absolute path to the WGSL shader file.                                             |
| `shaderInputs`         | object                         | Entry-point builtin values (e.g. `global_invocation_id`, `workgroup_id`, …).       |
| `bindings`             | object                         | Resource bindings keyed by `"group:binding"` (e.g. `"0:0"`).                       |
| `bindings[].type`      | `"f32"` \| `"i32"` \| `"u32"` | Element type of the buffer.                                                        |
| `bindings[].inline`    | array                          | Inline array of values. Cannot be combined with `file`.                            |
| `bindings[].file`      | string                         | Path to a data file relative to the shader. Cannot be combined with `inline`.      |
| `bindings[].format`    | `"ron"` \| `"binary"`          | File format: `"ron"` (RON array, default) or `"binary"` (little-endian 4-byte values). |
