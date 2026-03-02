import * as vscode from 'vscode';
import * as cp from 'child_process';

declare const process: { platform: string };

export function activate(context: vscode.ExtensionContext) {
	const outputChannel = vscode.window.createOutputChannel('WGSL Debugger');
	context.subscriptions.push(outputChannel);

	const provider = new ConfigurationProvider();
	context.subscriptions.push(vscode.debug.registerDebugConfigurationProvider('wgsl', provider));

	const disposable = vscode.commands.registerCommand('malkovri-wgsl-debugger.helloWorld', () => {
		vscode.window.showInformationMessage('Hello World from Malkovri WGSL Debugger!');
	});
	context.subscriptions.push(disposable);

	context.subscriptions.push(
		vscode.debug.registerDebugAdapterDescriptorFactory('wgsl', new DebugAdapterExecutableFactory(context.extensionUri, outputChannel))
	);
}

export function deactivate() {}

class ConfigurationProvider implements vscode.DebugConfigurationProvider {
	resolveDebugConfiguration(
		_folder: vscode.WorkspaceFolder | undefined,
		config: vscode.DebugConfiguration,
		_token?: vscode.CancellationToken
	): vscode.ProviderResult<vscode.DebugConfiguration> {
		if (!config.type && !config.request && !config.name) {
			const editor = vscode.window.activeTextEditor;
			if (editor && editor.document.languageId === 'wgsl') {
				config.type = 'wgsl';
				config.name = 'Launch';
				config.request = 'launch';
				config.program = '${file}';
				config.stopOnEntry = true;
			}
		}

		if (!config.program) {
			return vscode.window.showInformationMessage('Cannot find a program to debug').then(() => undefined);
		}

		return config;
	}
}

class DebugAdapterExecutableFactory implements vscode.DebugAdapterDescriptorFactory {
	constructor(
		private readonly extensionUri: vscode.Uri,
		private readonly outputChannel: vscode.OutputChannel
	) {}

	createDebugAdapterDescriptor(
		_session: vscode.DebugSession,
		_executable: vscode.DebugAdapterExecutable | undefined
	): vscode.ProviderResult<vscode.DebugAdapterDescriptor> {
		const suffix = process.platform === 'win32' ? '.exe' : '';
		const adapterPath = vscode.Uri.joinPath(
			this.extensionUri, '..', 'target', 'debug', `malkovri_wgsl_debugger_dap${suffix}`
		).fsPath;
		const executable = new vscode.DebugAdapterExecutable(adapterPath);

		const child = cp.spawn(executable.command, executable.args ?? [], {
			...executable.options,
			stdio: ['pipe', 'pipe', 'pipe'],
		});

		child.stderr?.on('data', (data: Buffer) => {
			this.outputChannel.append(data.toString());
			this.outputChannel.show(true);
		});

		child.on('error', (err) => {
			this.outputChannel.appendLine(`Failed to start debug adapter: ${err.message}`);
			this.outputChannel.show(true);
		});

		return new vscode.DebugAdapterInlineImplementation(new ChildProcessDebugAdapter(child));
	}
}

class ChildProcessDebugAdapter implements vscode.DebugAdapter {
	private readonly _onDidSendMessage = new vscode.EventEmitter<any>();
	readonly onDidSendMessage: vscode.Event<any> = this._onDidSendMessage.event;

	private rawData = Buffer.alloc(0);
	private contentLength = -1;

	constructor(private readonly child: cp.ChildProcess) {
		child.stdout?.on('data', (data: Buffer) => {
			this.rawData = Buffer.concat([this.rawData, data]);
			this.processBuffer();
		});
	}

	handleMessage(message: any): void {
		const json = JSON.stringify(message);
		const header = `Content-Length: ${Buffer.byteLength(json, 'utf8')}\r\n\r\n`;
		this.child.stdin?.write(header + json);
	}

	private processBuffer(): void {
		while (true) {
			if (this.contentLength < 0) {
				const separator = this.rawData.indexOf('\r\n\r\n');
				if (separator < 0) break;

				const header = this.rawData.toString('utf8', 0, separator);
				const match = header.match(/Content-Length: (\d+)/);
				if (!match) break;

				this.contentLength = parseInt(match[1]);
				this.rawData = this.rawData.subarray(separator + 4);
			}

			if (this.rawData.length < this.contentLength) break;

			const content = this.rawData.toString('utf8', 0, this.contentLength);
			this.rawData = this.rawData.subarray(this.contentLength);
			this.contentLength = -1;

			this._onDidSendMessage.fire(JSON.parse(content));
		}
	}

	dispose(): void {
		this.child.kill();
	}
}
