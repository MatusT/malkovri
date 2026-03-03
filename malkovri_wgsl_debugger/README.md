# WGSL Debugger

A DAP (Debug Adapter Protocol) debugger for WGSL shaders, with a VS Code extension.

Simulates shader execution on the CPU using [naga](https://github.com/gfx-rs/naga) and exposes (so far) step-through, and variable inspection via the standard debug adapter protocol.

Supported:
- [x] Basic compute shaders
- [x] Basic buffer global inputs
- [x] All expressions

TODO:
- [ ] Image and sampler inputs
- [ ] Support for graphics pipeline with vertex + fragment shaders
- [ ] ... and like a million things :-)

## Requirements

- [Rust](https://rustup.rs/) (edition 2024, stable toolchain)
- [Node.js](https://nodejs.org/) and npm (for the VS Code extension)
- VS Code

## Build

```sh
# Build the DAP server
cargo build --release -p malkovri_wgsl_debugger_dap

# Build the VS Code extension
cd vscode_extension
npm install
npm run compile
```

## Run / Install

1. Build both components above.
2. Open the `vscode_extension/` folder in VS Code.
3. Press **F5** to launch the Extension Development Host.
4. Open a `.wgsl` file and create a launch configuration in `.vscode/launch.json`:

```json
{
  "type": "wgsl",
  "request": "launch",
  "name": "Debug shader",
  "program": "${workspaceFolder}/shader.wgsl",
  "global_invocation_id": [0, 0, 0],
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

| Field                  | Type                          | Description                                                                        |
|------------------------|-------------------------------|------------------------------------------------------------------------------------|
| `program`              | string                        | Absolute path to the WGSL shader file.                                             |
| `global_invocation_id` | `[u, u, u]`                   | `@builtin(global_invocation_id)` passed to the entry point. Defaults to `[0,0,0]`. |
| `bindings`             | object                        | Resource bindings keyed by `"group:binding"` (e.g. `"0:0"`).                       |
| `bindings[].type`      | `"f32"` \| `"i32"` \| `"u32"` | Element type of the buffer.                                                        |
| `bindings[].inline`    | array                         | Inline data values.                                                                |
