//! Ganache Engine core contract.
//!
//! Model weights and backend implementations are intentionally kept outside
//! this initial contract crate. The public request/response types are shared
//! by the standalone library and the HTTP service.

use serde::{Deserialize, Serialize};
use std::time::Instant;
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateOneRequest {
    pub request_id: u64,
    pub document_id: String,
    pub composition_id: u64,
    pub raw_revision: u64,
    pub document_context: String,
    pub raw_input: String,
    #[serde(default)]
    pub evidence: Vec<Evidence>,
    pub current_hypothesis: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    pub hypothesis: String,
    pub strength: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candidate {
    pub text: String,
    pub probability: Option<f32>,
    pub score: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateDistribution {
    pub candidates: Vec<Candidate>,
    pub other_probability: Option<f32>,
    pub top1_margin: Option<f32>,
    pub latency_ms: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateOneResponse {
    pub request_id: u64,
    pub document_id: String,
    pub composition_id: u64,
    pub raw_revision: u64,
    pub distribution: CandidateDistribution,
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
    fn create_one(&self, request: &CreateOneRequest) -> Result<CandidateDistribution, EngineError>;
}

pub fn create_one(
    backend: &dyn InferenceBackend,
    request: CreateOneRequest,
) -> Result<CreateOneResponse, EngineError> {
    validate_request(&request)?;
    let started = Instant::now();
    let mut distribution = backend.create_one(&request)?;
    distribution.latency_ms = started.elapsed().as_millis().min(u32::MAX as u128) as u32;
    if distribution.candidates.is_empty() {
        return Err(EngineError::Inference(
            "backend returned no candidates".into(),
        ));
    }
    Ok(CreateOneResponse {
        request_id: request.request_id,
        document_id: request.document_id,
        composition_id: request.composition_id,
        raw_revision: request.raw_revision,
        distribution,
    })
}

fn validate_request(request: &CreateOneRequest) -> Result<(), EngineError> {
    if request.document_id.trim().is_empty() {
        return Err(EngineError::InvalidRequest("document_id is empty".into()));
    }
    if request.raw_input.is_empty() {
        return Err(EngineError::InvalidRequest("raw_input is empty".into()));
    }
    if request.raw_input.chars().count() > 256 {
        return Err(EngineError::InvalidRequest("raw_input is too long".into()));
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

        fn create_one(
            &self,
            _request: &CreateOneRequest,
        ) -> Result<CandidateDistribution, EngineError> {
            Ok(CandidateDistribution {
                candidates: vec![Candidate {
                    text: "テスト".into(),
                    probability: Some(1.0),
                    score: None,
                }],
                other_probability: Some(0.0),
                top1_margin: None,
                latency_ms: 0,
            })
        }
    }

    #[test]
    fn create_one_echoes_request_identity_and_measures_latency() {
        let request = CreateOneRequest {
            request_id: 7,
            document_id: "doc".into(),
            composition_id: 2,
            raw_revision: 3,
            document_context: String::new(),
            raw_input: "nihongo".into(),
            evidence: vec![],
            current_hypothesis: None,
        };
        let response = create_one(&TestBackend, request).unwrap();
        assert_eq!(response.request_id, 7);
        assert_eq!(response.raw_revision, 3);
        assert_eq!(response.distribution.candidates[0].text, "テスト");
    }
}
