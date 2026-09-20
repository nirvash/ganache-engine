//! Jinen v1 GGUF adapter for Ganache.
//!
//! This first adapter intentionally exposes greedy generation only. Candidate
//! beam search and model-specific scoring will be added after the single
//! candidate path is benchmarked against the Rakukan baseline.

use ganache_core::{
    Candidate, CandidateDistribution, CreateOneRequest, EngineError, InferenceBackend,
};
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{LlamaModel, params::LlamaModelParams},
    sampling::LlamaSampler,
    token::LlamaToken,
};
use std::{num::NonZeroU32, path::Path, sync::OnceLock, time::Instant};
use tokenizers::Tokenizer;

mod romaji;

#[allow(dead_code)]
mod kana {
    pub fn hiragana_to_katakana(text: &str) -> String {
        text.chars()
            .map(|ch| match ch {
                'ぁ'..='ゖ' => char::from_u32(ch as u32 + 0x60).unwrap_or(ch),
                _ => ch,
            })
            .collect()
    }
}

const CONTEXT_TOKEN: char = '\u{ee02}';
const INPUT_START_TOKEN: char = '\u{ee00}';
const OUTPUT_START_TOKEN: char = '\u{ee01}';

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

    pub fn build_prompt(context: &str, reading: &str) -> String {
        let katakana: String = reading
            .chars()
            .map(|c| match c {
                'ぁ'..='ゖ' => char::from_u32(c as u32 + 0x60).unwrap_or(c),
                _ => c,
            })
            .collect();
        format!("{CONTEXT_TOKEN}{context}{INPUT_START_TOKEN}{katakana}{OUTPUT_START_TOKEN}")
    }

    fn normalize_reading(raw_input: &str) -> String {
        if raw_input
            .chars()
            .all(|ch| ch.is_ascii() && !ch.is_ascii_digit())
        {
            let mut converter = romaji::RomajiConverter::new();
            for ch in raw_input.chars() {
                converter.push(ch);
            }
            converter.flush();
            converter.output().to_owned()
        } else {
            raw_input.to_owned()
        }
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

    fn create_one(&self, request: &CreateOneRequest) -> Result<CandidateDistribution, EngineError> {
        let started = Instant::now();
        let reading = Self::normalize_reading(&request.raw_input);
        let text = self.generate(&Self::build_prompt(&request.document_context, &reading))?;
        if text.is_empty() {
            return Err(EngineError::Inference(
                "model returned empty candidate".into(),
            ));
        }
        Ok(CandidateDistribution {
            candidates: vec![Candidate {
                text,
                probability: None,
                score: None,
            }],
            other_probability: None,
            top1_margin: None,
            latency_ms: started.elapsed().as_millis().min(u32::MAX as u128) as u32,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_jinen_prompt_with_katakana_reading() {
        assert_eq!(
            JinenBackend::build_prompt("前", "にほんご"),
            "\u{ee02}前\u{ee00}ニホンゴ\u{ee01}"
        );
    }

    #[test]
    fn normalizes_romaji_before_prompt() {
        assert_eq!(JinenBackend::normalize_reading("nihongo"), "にほんご");
    }
}
