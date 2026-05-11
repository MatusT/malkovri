use std::fmt::Debug;

#[cfg(not(target_arch = "wasm32"))]
use std::io::{BufRead, BufReader, BufWriter, Write};

use serde::Serialize;

use malkovri_wgsl_debugger::DebugThreadId;

use crate::{debug_adapter::DebugAdapter, error::DebugAdapterError};

pub(crate) type BreakpointId = u64;
pub type StackFrameId = u64;
pub(crate) type ScopeReference = u32;

pub(crate) const LOCALS_SCOPE_REF: ScopeReference = 1;
pub(crate) const ARGUMENTS_SCOPE_REF: ScopeReference = 2;
pub(crate) const GLOBALS_SCOPE_REF: ScopeReference = 3;

#[derive(Clone, Debug)]
pub enum OutgoingMessage {
    Response {
        seq: i64,
        request_seq: i64,
        body: serde_json::Value,
    },
    Event {
        seq: i64,
        event: String,
        body: serde_json::Value,
    },
}

impl OutgoingMessage {
    pub fn request_seq(&self) -> Option<i64> {
        match self {
            OutgoingMessage::Response { request_seq, .. } => Some(*request_seq),
            OutgoingMessage::Event { .. } => None,
        }
    }

    pub fn event_name(&self) -> Option<&str> {
        match self {
            OutgoingMessage::Event { event, .. } => Some(event),
            OutgoingMessage::Response { .. } => None,
        }
    }

    pub fn body(&self) -> &serde_json::Value {
        match self {
            OutgoingMessage::Response { body, .. } | OutgoingMessage::Event { body, .. } => body,
        }
    }

    pub fn to_json(&self) -> serde_json::Value {
        match self {
            OutgoingMessage::Response {
                seq,
                request_seq,
                body,
            } => serde_json::json!({
                "seq": seq,
                "type": "response",
                "request_seq": request_seq,
                "success": true,
                "message": null,
                "body": body,
            }),
            OutgoingMessage::Event { seq, event, body } => serde_json::json!({
                "seq": seq,
                "type": "event",
                "event": event,
                "body": body,
            }),
        }
    }
}

pub(crate) fn make_scope_ref(thread_id: DebugThreadId, scope: ScopeReference) -> ScopeReference {
    (thread_id as u32) * 10 + scope
}

pub(crate) fn parse_scope_ref(reference: ScopeReference) -> (DebugThreadId, ScopeReference) {
    ((reference / 10) as DebugThreadId, reference % 10)
}

pub(crate) fn make_variable(name: Option<String>, value: &str) -> dapts::Variable {
    dapts::Variable {
        name: name.clone().unwrap_or("unnamed".to_string()),
        evaluate_name: name,
        value: value.to_string(),
        variables_reference: 0,
        declaration_location_reference: None,
        indexed_variables: None,
        memory_reference: None,
        named_variables: None,
        presentation_hint: None,
        ty: None,
        value_location_reference: None,
    }
}

impl DebugAdapter {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_stdio(&mut self) -> Result<(), DebugAdapterError> {
        let stdin = std::io::stdin();
        let stdout = std::io::stdout();
        let mut reader = BufReader::new(stdin.lock());
        let mut writer = BufWriter::new(stdout.lock());
        self.from_streams(&mut reader, &mut writer)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_streams<R: BufRead, W: Write>(
        &mut self,
        reader: &mut R,
        writer: &mut W,
    ) -> Result<(), DebugAdapterError> {
        while let Some(req) = Self::poll_request(reader)? {
            for message in self.handle_request(&req)? {
                Self::write_message(writer, &message)?;
            }
        }

        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn poll_request<R: BufRead>(
        reader: &mut R,
    ) -> Result<Option<dapts::Request>, DebugAdapterError> {
        let mut buffer = String::new();

        if reader.read_line(&mut buffer)? == 0 {
            return Ok(None);
        }
        let (name, value) = buffer
            .trim_end()
            .split_once(':')
            .ok_or_else(|| DebugAdapterError::Parse("Header is incorrect".to_string()))?;
        let content_length: usize = match name {
            "Content-Length" => value.trim().parse().map_err(|_| {
                DebugAdapterError::Parse("Content-Length is not a valid number".to_string())
            })?,
            other => {
                return Err(DebugAdapterError::Parse(format!("Unknown header: {other}")));
            }
        };

        buffer.clear();
        reader.read_line(&mut buffer)?;

        let mut content = vec![0; content_length];
        reader.read_exact(&mut content)?;
        let content = std::str::from_utf8(&content)
            .map_err(|e| DebugAdapterError::Parse(format!("Invalid UTF-8: {e}")))?;
        let request: dapts::Request = serde_json::from_str(content)?;

        Ok(Some(request))
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn write_message<W: Write>(
        writer: &mut W,
        message: &OutgoingMessage,
    ) -> Result<(), DebugAdapterError> {
        let payload = serde_json::to_string(&message.to_json())?;
        let framed = format!("Content-Length: {}\r\n\r\n{}", payload.len(), payload);
        writer.write_all(framed.as_bytes())?;
        writer.flush()?;
        Ok(())
    }

    pub(crate) fn make_response<T: Serialize + Debug>(
        &mut self,
        request_seq: i64,
        body: &T,
    ) -> Result<OutgoingMessage, DebugAdapterError> {
        Ok(OutgoingMessage::Response {
            seq: self.next_sequence_number(),
            request_seq,
            body: serde_json::to_value(body)?,
        })
    }

    pub(crate) fn make_event<T: Serialize + Debug>(
        &mut self,
        event: &str,
        body: &T,
    ) -> Result<OutgoingMessage, DebugAdapterError> {
        Ok(OutgoingMessage::Event {
            seq: self.next_sequence_number(),
            event: event.to_string(),
            body: serde_json::to_value(body)?,
        })
    }
}
