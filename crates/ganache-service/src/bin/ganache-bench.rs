//! Small, model-profile driven benchmark for comparing Ganache backends.
//!
//! The benchmark intentionally uses the same generic PredictRequest as the
//! service. It does not contain IME-specific candidate logic.

use ganache_core::{GenerationParameters, InferenceBackend, PredictRequest, predict};
use ganache_jinen::JinenBackend;
use ganache_models::Registry;
use serde_json::json;
use std::{collections::BTreeMap, env, time::Instant};

fn main() {
    let model_id = env::var("GANACHE_MODEL_ID").unwrap_or_else(|_| "jinen_v1_xsmall_q5".into());
    let samples = env_u32("GANACHE_BENCH_SAMPLES", 10) as usize;
    let warmups = env_u32("GANACHE_BENCH_WARMUPS", 1);
    let registry = Registry::embedded().expect("embedded model registry");
    let resolved = registry.resolve(&model_id).expect("model id");
    assert!(
        resolved.model_path.is_file() && resolved.tokenizer_path.is_file(),
        "model files are missing: {} and {}",
        resolved.model_path.display(),
        resolved.tokenizer_path.display()
    );

    let gpu_layers = env::var("GANACHE_GPU_LAYERS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(if cfg!(feature = "cuda") {
            resolved.spec.runtime.recommended_gpu_layers
        } else {
            0
        });
    let main_gpu = env_i32("GANACHE_MAIN_GPU", 0);
    let load_started = Instant::now();
    let backend = JinenBackend::load(
        &resolved.model_path,
        &resolved.tokenizer_path,
        gpu_layers,
        main_gpu,
    )
    .expect("load model backend");
    let load_ms = load_started.elapsed().as_millis();

    let request = || PredictRequest {
        request_id: 0,
        session_id: Some("benchmark".into()),
        prompt: "ニホンゴ".into(),
        input: json!({"raw": "nihongo"}),
        output_schema: json!({"type": "string"}),
        generation: GenerationParameters {
            max_tokens: Some(8),
            ..GenerationParameters::default()
        },
        metadata: BTreeMap::new(),
    };

    for index in 0..warmups {
        let response = predict(&backend, request()).expect("warmup prediction");
        println!(
            "warmup={} latency_ms={} output={}",
            index + 1,
            response.latency_ms,
            response.output
        );
    }

    let mut latencies = Vec::with_capacity(samples);
    let mut outputs = Vec::with_capacity(samples);
    for index in 0..samples {
        let started = Instant::now();
        let mut current = request();
        current.request_id = index as u64 + 1;
        let response = predict(&backend, current).expect("prediction");
        latencies.push(started.elapsed().as_millis() as u32);
        outputs.push(response.output);
    }
    latencies.sort_unstable();
    let output = outputs
        .first()
        .map(ToString::to_string)
        .unwrap_or_else(|| "null".into());
    println!("model={model_id}");
    println!("backend={}", backend.name());
    println!("gpu_layers={gpu_layers} main_gpu={main_gpu}");
    println!("load_ms={load_ms}");
    println!("samples={samples} warmups={warmups}");
    println!(
        "p50_ms={} p95_ms={} max_ms={}",
        percentile(&latencies, 50),
        percentile(&latencies, 95),
        latencies.last().copied().unwrap_or(0)
    );
    println!("output={output}");
}

fn percentile(values: &[u32], percentile: usize) -> u32 {
    if values.is_empty() {
        return 0;
    }
    let index = (values.len() * percentile).div_ceil(100).saturating_sub(1);
    values[index.min(values.len() - 1)]
}

fn env_u32(name: &str, default: u32) -> u32 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_i32(name: &str, default: i32) -> i32 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}
