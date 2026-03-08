use malkovri_wgsl_debugger_dap::DebugAdapter;

fn main() {
    let mut debug_adapter = DebugAdapter::new();

    if let Err(e) = debug_adapter.from_stdio() {
        eprintln!("Debug adapter error: {e}");
        std::process::exit(1);
    }
}
