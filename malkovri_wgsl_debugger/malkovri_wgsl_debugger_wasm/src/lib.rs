use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct WgslDebugAdapter {
    inner: malkovri_wgsl_debugger_dap::DebugAdapter,
}

impl Default for WgslDebugAdapter {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl WgslDebugAdapter {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: malkovri_wgsl_debugger_dap::DebugAdapter::new(),
        }
    }

    /// Process a single DAP request as a JSON string.
    /// Returns an array of JSON response/event strings.
    #[wasm_bindgen(js_name = "handleMessage")]
    pub fn handle_message(&mut self, json: &str) -> Result<Vec<String>, JsError> {
        self.inner
            .handle_message(json)
            .map_err(|e| JsError::new(&e.to_string()))
    }
}
