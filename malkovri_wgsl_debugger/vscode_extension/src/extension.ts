import * as vscode from "vscode";
import { initSync, WgslDebugAdapter } from "malkovri_wgsl_debugger_wasm";

let wasmInitialized = false;

export function activate(context: vscode.ExtensionContext) {
  const outputChannel = vscode.window.createOutputChannel("WGSL Debugger");
  context.subscriptions.push(outputChannel);

  context.subscriptions.push(
    vscode.debug.registerDebugConfigurationProvider(
      "wgsl",
      new ConfigurationProvider(),
    ),
  );

  context.subscriptions.push(
    vscode.debug.registerDebugAdapterDescriptorFactory(
      "wgsl",
      new WasmDebugAdapterFactory(context.extensionUri, outputChannel),
    ),
  );
}

export function deactivate() {}

interface BindingConfig {
  type: "f32" | "i32" | "u32";
  file?: string;
  inline?: number[];
  format?: "ron" | "binary";
  fileContent?: string;
}

class ConfigurationProvider implements vscode.DebugConfigurationProvider {
  resolveDebugConfiguration(
    _folder: vscode.WorkspaceFolder | undefined,
    config: vscode.DebugConfiguration,
    _token?: vscode.CancellationToken,
  ): vscode.ProviderResult<vscode.DebugConfiguration> {
    if (!config.type && !config.request && !config.name) {
      const editor = vscode.window.activeTextEditor;
      if (editor && editor.document.languageId === "wgsl") {
        config.type = "wgsl";
        config.name = "Launch";
        config.request = "launch";
        config.program = "${file}";
        config.stopOnEntry = true;
      }
    }

    if (!config.program) {
      return vscode.window.showInformationMessage(
        "Cannot find a program to debug",
      ).then(() => undefined);
    }

    return config;
  }

  async resolveDebugConfigurationWithSubstitutedVariables(
    _folder: vscode.WorkspaceFolder | undefined,
    config: vscode.DebugConfiguration,
    _token?: vscode.CancellationToken,
  ): Promise<vscode.DebugConfiguration | undefined> {
    const programUri = vscode.Uri.file(config.program as string);

    // Read program source and inject into config for the WASM adapter
    const sourceBytes = await vscode.workspace.fs.readFile(programUri);
    config.source = new TextDecoder().decode(sourceBytes);

    // Pre-read any file-based bindings so the WASM adapter doesn't need filesystem access
    const bindings = config.bindings as
      | Record<string, BindingConfig>
      | undefined;
    if (bindings) {
      const programDir = vscode.Uri.joinPath(programUri, "..");
      for (const [_key, binding] of Object.entries(bindings)) {
        if (binding.file && !binding.inline) {
          const fileUri = vscode.Uri.joinPath(programDir, binding.file);
          const format = binding.format ?? "ron";

          if (format === "binary") {
            const bytes = await vscode.workspace.fs.readFile(fileUri);
            const view = new DataView(
              bytes.buffer,
              bytes.byteOffset,
              bytes.byteLength,
            );
            const scalarType = binding.type ?? "f32";
            const values: number[] = [];
            for (let i = 0; i + 3 < bytes.byteLength; i += 4) {
              if (scalarType === "f32") values.push(view.getFloat32(i, true));
              else if (scalarType === "i32") {
                values.push(view.getInt32(i, true));
              } else values.push(view.getUint32(i, true));
            }
            binding.inline = values;
            delete binding.file;
          } else {
            const contentBytes = await vscode.workspace.fs.readFile(fileUri);
            binding.fileContent = new TextDecoder().decode(contentBytes);
            delete binding.file;
          }
        }
      }
    }

    return config;
  }
}

class WasmDebugAdapterFactory implements vscode.DebugAdapterDescriptorFactory {
  constructor(
    private readonly extensionUri: vscode.Uri,
    private readonly outputChannel: vscode.OutputChannel,
  ) {}

  async createDebugAdapterDescriptor(
    _session: vscode.DebugSession,
    _executable: vscode.DebugAdapterExecutable | undefined,
  ): Promise<vscode.DebugAdapterDescriptor> {
    if (!wasmInitialized) {
      const wasmUri = vscode.Uri.joinPath(
        this.extensionUri,
        "out",
        "malkovri_wgsl_debugger_wasm_bg.wasm",
      );
      const wasmBytes = await vscode.workspace.fs.readFile(wasmUri);
      initSync({ module: wasmBytes });
      wasmInitialized = true;
    }

    return new vscode.DebugAdapterInlineImplementation(
      new WasmDebugAdapterImpl(this.outputChannel),
    );
  }
}

interface DapMessage {
  type: string;
  seq: number;
  command: string;
}

class WasmDebugAdapterImpl implements vscode.DebugAdapter {
  private readonly _onDidSendMessage = new vscode.EventEmitter<
    vscode.DebugProtocolMessage
  >();
  readonly onDidSendMessage: vscode.Event<vscode.DebugProtocolMessage> =
    this._onDidSendMessage.event;
  private adapter: WgslDebugAdapter;

  constructor(private readonly outputChannel: vscode.OutputChannel) {
    this.adapter = new WgslDebugAdapter();
  }

  handleMessage(message: vscode.DebugProtocolMessage): void {
    try {
      const responses = this.adapter.handleMessage(JSON.stringify(message));
      for (const json of responses) {
        this._onDidSendMessage.fire(
          JSON.parse(json) as vscode.DebugProtocolMessage,
        );
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      this.outputChannel.appendLine(`Debug adapter error: ${msg}`);

      const m = message as DapMessage & vscode.DebugProtocolMessage;
      if (m.type === "request") {
        this._onDidSendMessage.fire(
          {
            type: "response",
            request_seq: m.seq,
            seq: 0,
            success: false,
            command: m.command,
            message: msg,
          } as DapMessage & vscode.DebugProtocolMessage,
        );
      }
    }
  }

  dispose(): void {
    this.adapter.free();
  }
}
