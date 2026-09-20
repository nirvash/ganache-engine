//! Jinen v1 GGUF model backend for Ganache.
//!
//! This adapter knows the model runtime, tokenizer, and generation mechanics.
//! It does not know IME input, romaji, readings, candidates, or any other
//! client-domain semantics.

use ganache_core::{EngineError, InferenceBackend, PredictRequest};
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{LlamaModel, params::LlamaModelParams},
    sampling::LlamaSampler,
    token::LlamaToken,
};
use serde_json::Value;
use std::{num::NonZeroU32, path::Path, sync::OnceLock};
use tokenizers::Tokenizer;

static LLAMA_BACKEND: OnceLock<Result<LlamaBackend, String>> = OnceLock::new();

fn llama_backend() -> Result<&'static LlamaBackend, EngineError> {
    match LLAMA_BACKEND.get_or_init(|| LlamaBackend::init().map_err(|e| e.to_string())) {
        Ok(backend) => Ok(backend),
        Err(error) => Err(EngineError::Inference(format!(
            "llama backend init: {error}"
        ))),
    }
}

pub struct JinenBackend {
    model: LlamaModel,
    tokenizer: Tokenizer,
    n_ctx: u32,
    max_new_tokens: usize,
}

impl JinenBackend {
    pub fn load<P: AsRef<Path>, T: AsRef<Path>>(
        model_path: P,
        tokenizer_path: T,
        n_gpu_layers: u32,
        main_gpu: i32,
    ) -> Result<Self, EngineError> {
        let backend = llama_backend()?;
        let params = LlamaModelParams::default()
            .with_n_gpu_layers(n_gpu_layers)
            .with_main_gpu(main_gpu);
        let model = LlamaModel::load_from_file(backend, model_path.as_ref(), &params)
            .map_err(|error| EngineError::Inference(format!("load GGUF: {error}")))?;
        let tokenizer = Tokenizer::from_file(tokenizer_path.as_ref())
            .map_err(|error| EngineError::Inference(format!("load tokenizer: {error}")))?;
        Ok(Self {
            model,
            tokenizer,
            n_ctx: 128,
            max_new_tokens: 64,
        })
    }

    fn generate(&self, prompt: &str) -> Result<String, EngineError> {
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|error| EngineError::Inference(format!("tokenize: {error}")))?;
        let input: Vec<LlamaToken> = encoding
            .get_ids()
            .iter()
            .map(|id| LlamaToken(*id as i32))
            .collect();
        if input.is_empty() || input.len() >= self.n_ctx as usize {
            return Err(EngineError::InvalidRequest(
                "prompt exceeds context window".into(),
            ));
        }
        let backend = llama_backend()?;
        let params =
            LlamaContextParams::default().with_n_ctx(Some(NonZeroU32::new(self.n_ctx).unwrap()));
        let mut context = self
            .model
            .new_context(backend, params)
            .map_err(|error| EngineError::Inference(format!("create context: {error}")))?;
        let mut batch = LlamaBatch::new(input.len().max(64), 1);
        for (index, token) in input.iter().enumerate() {
            batch
                .add(*token, index as i32, &[0], index + 1 == input.len())
                .map_err(|error| EngineError::Inference(format!("add prompt: {error}")))?;
        }
        context
            .decode(&mut batch)
            .map_err(|error| EngineError::Inference(format!("decode prompt: {error}")))?;
        let mut sampler = LlamaSampler::greedy();
        let mut generated = Vec::new();
        for position in 0..self.max_new_tokens {
            let token = sampler.sample(&context, -1);
            if token == self.model.token_eos() || self.model.is_eog_token(token) {
                break;
            }
            generated.push(token);
            batch.clear();
            batch
                .add(token, (input.len() + position) as i32, &[0], true)
                .map_err(|error| EngineError::Inference(format!("add token: {error}")))?;
            context
                .decode(&mut batch)
                .map_err(|error| EngineError::Inference(format!("decode token: {error}")))?;
        }
        let ids: Vec<u32> = generated.iter().map(|token| token.0 as u32).collect();
        self.tokenizer
            .decode(&ids, true)
            .map(|text| text.trim().to_owned())
            .map_err(|error| EngineError::Inference(format!("decode text: {error}")))
    }
}

impl InferenceBackend for JinenBackend {
    fn name(&self) -> &'static str {
        "jinen-llama"
    }

    fn predict(&self, request: &PredictRequest) -> Result<Value, EngineError> {
        let text = self.generate(&request.prompt)?;
        if text.is_empty() {
            return Err(EngineError::Inference("model returned empty output".into()));
        }
        // Until constrained decoding is implemented, preserve arbitrary model
        // text as a valid JSON value. JSON emitted by the model remains an
        // object/array so client schemas can consume it directly.
        Ok(serde_json::from_str(&text).unwrap_or(Value::String(text)))
    }
}
