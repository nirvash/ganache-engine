//! Ganache Engine's model-independent structured prediction contract.
//!
//! The core deliberately knows nothing about IMEs, languages, candidates, or
//! a particular model. A client supplies the task prompt, input value, output
//! schema, and optional session/generation state.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::BTreeMap, time::Instant};
use thiserror::Error;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GenerationParameters {
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictRequest {
    pub request_id: u64,
    #[serde(default)]
    pub session_id: Option<String>,
    pub prompt: String,
    pub input: Value,
    pub output_schema: Value,
    #[serde(default)]
    pub generation: GenerationParameters,
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictResponse {
    pub request_id: u64,
    #[serde(default)]
    pub session_id: Option<String>,
    pub output: Value,
    pub latency_ms: u32,
    pub backend: String,
}

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("backend is not available")]
    BackendUnavailable,
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("inference failed: {0}")]
    Inference(String),
}

pub trait InferenceBackend: Send + Sync {
    fn name(&self) -> &'static str;
    fn predict(&self, request: &PredictRequest) -> Result<Value, EngineError>;
}

pub fn predict(
    backend: &dyn InferenceBackend,
    request: PredictRequest,
) -> Result<PredictResponse, EngineError> {
    validate_request(&request)?;
    let started = Instant::now();
    let output = backend.predict(&request)?;
    Ok(PredictResponse {
        request_id: request.request_id,
        session_id: request.session_id,
        output,
        latency_ms: started.elapsed().as_millis().min(u32::MAX as u128) as u32,
        backend: backend.name().to_owned(),
    })
}

fn validate_request(request: &PredictRequest) -> Result<(), EngineError> {
    if request.prompt.trim().is_empty() {
        return Err(EngineError::InvalidRequest("prompt is empty".into()));
    }
    if request.prompt.chars().count() > 65_536 {
        return Err(EngineError::InvalidRequest("prompt is too long".into()));
    }
    if request.output_schema.is_null() {
        return Err(EngineError::InvalidRequest("output_schema is null".into()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestBackend;

    impl InferenceBackend for TestBackend {
        fn name(&self) -> &'static str {
            "test"
        }

        fn predict(&self, request: &PredictRequest) -> Result<Value, EngineError> {
            Ok(serde_json::json!({
                "echo": request.input,
                "ok": true
            }))
        }
    }

    #[test]
    fn predict_echoes_identity_and_returns_structured_output() {
        let request = PredictRequest {
            request_id: 7,
            session_id: Some("session".into()),
            prompt: "classify this".into(),
            input: serde_json::json!({"text": "test"}),
            output_schema: serde_json::json!({"type": "object"}),
            generation: GenerationParameters::default(),
            metadata: BTreeMap::new(),
        };
        let response = predict(&TestBackend, request).unwrap();
        assert_eq!(response.request_id, 7);
        assert_eq!(response.session_id.as_deref(), Some("session"));
        assert_eq!(response.output["ok"], true);
        assert_eq!(response.backend, "test");
    }
}
