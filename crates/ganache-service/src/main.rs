use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use ganache_core::{EngineError, InferenceBackend, PredictRequest, predict};
use ganache_jinen::JinenBackend;
use ganache_models::Registry;
use std::sync::Arc;

struct UnavailableBackend {}

impl InferenceBackend for UnavailableBackend {
    fn name(&self) -> &'static str {
        "unavailable"
    }

    fn predict(&self, _request: &PredictRequest) -> Result<serde_json::Value, EngineError> {
        Err(EngineError::BackendUnavailable)
    }
}

#[derive(Clone)]
struct AppState {
    backend: Arc<dyn InferenceBackend>,
    model_id: String,
    ready: bool,
    reason: Option<String>,
}

#[tokio::main]
async fn main() {
    let bind = std::env::var("GANACHE_BIND").unwrap_or_else(|_| "127.0.0.1:39201".into());
    let state = load_state();
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/predict", post(predict_handler))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(&bind)
        .await
        .expect("bind GANACHE_BIND");
    eprintln!("ganache-service listening on http://{bind}");
    axum::serve(listener, app).await.expect("serve");
}

async fn health(State(state): State<AppState>) -> impl IntoResponse {
    Json(serde_json::json!({
        "status": if state.ready { "ok" } else { "degraded" },
        "backend": state.backend.name(),
        "model": state.model_id,
        "reason": state.reason,
    }))
}

fn load_state() -> AppState {
    let model_id =
        std::env::var("GANACHE_MODEL_ID").unwrap_or_else(|_| "jinen_v1_xsmall_q5".into());
    let unavailable = |reason: String| AppState {
        backend: Arc::new(UnavailableBackend {}),
        model_id: model_id.clone(),
        ready: false,
        reason: Some(reason),
    };
    let registry = match Registry::embedded() {
        Ok(registry) => registry,
        Err(error) => return unavailable(error.to_string()),
    };
    let resolved = match registry.resolve(&model_id) {
        Ok(resolved) => resolved,
        Err(error) => return unavailable(error.to_string()),
    };
    if resolved.spec.adapter != "jinen-llama" {
        return unavailable(format!(
            "model adapter is not available: {}",
            resolved.spec.adapter
        ));
    }
    if !resolved.model_path.is_file() || !resolved.tokenizer_path.is_file() {
        return unavailable(format!(
            "model files missing: {} and {}",
            resolved.model_path.display(),
            resolved.tokenizer_path.display()
        ));
    }
    let gpu_layers = std::env::var("GANACHE_GPU_LAYERS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(if cfg!(feature = "cuda") {
            resolved.spec.runtime.recommended_gpu_layers
        } else {
            0
        });
    let main_gpu = std::env::var("GANACHE_MAIN_GPU")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    match JinenBackend::load(
        &resolved.model_path,
        &resolved.tokenizer_path,
        gpu_layers,
        main_gpu,
    ) {
        Ok(backend) => AppState {
            backend: Arc::new(backend),
            model_id,
            ready: true,
            reason: None,
        },
        Err(error) => unavailable(error.to_string()),
    }
}

async fn predict_handler(
    State(state): State<AppState>,
    Json(request): Json<PredictRequest>,
) -> impl IntoResponse {
    match predict(state.backend.as_ref(), request) {
        Ok(response) => (
            StatusCode::OK,
            Json(serde_json::to_value(response).unwrap()),
        )
            .into_response(),
        Err(error) => {
            let status = match &error {
                EngineError::BackendUnavailable => StatusCode::SERVICE_UNAVAILABLE,
                EngineError::InvalidRequest(_) => StatusCode::BAD_REQUEST,
                EngineError::Inference(_) => StatusCode::INTERNAL_SERVER_ERROR,
            };
            (
                status,
                Json(serde_json::json!({ "error": error.to_string() })),
            )
                .into_response()
        }
    }
}
