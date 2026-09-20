//! Ganache Engine core contract.
//!
//! Model weights and backend implementations are intentionally kept outside
//! this initial contract crate. The public request/response types are shared
//! by the standalone library and the HTTP service.

use serde::{Deserialize, Serialize};

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
