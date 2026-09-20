use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use ganache_core::{CreateOneRequest, EngineError, InferenceBackend, create_one};
use std::sync::Arc;

struct UnavailableBackend;

impl InferenceBackend for UnavailableBackend {
    fn name(&self) -> &'static str {
        "unavailable"
    }

    fn create_one(
        &self,
        _request: &CreateOneRequest,
    ) -> Result<ganache_core::CandidateDistribution, EngineError> {
        Err(EngineError::BackendUnavailable)
    }
}

#[derive(Clone)]
struct AppState {
    backend: Arc<dyn InferenceBackend>,
}

#[tokio::main]
async fn main() {
    let bind = std::env::var("GANACHE_BIND").unwrap_or_else(|_| "127.0.0.1:39201".into());
    let state = AppState {
        backend: Arc::new(UnavailableBackend),
    };
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/createone", post(create_one_handler))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(&bind)
        .await
        .expect("bind GANACHE_BIND");
    eprintln!("ganache-service listening on http://{bind}");
    axum::serve(listener, app).await.expect("serve");
}

async fn health(State(state): State<AppState>) -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok", "backend": state.backend.name() }))
}

async fn create_one_handler(
    State(state): State<AppState>,
    Json(request): Json<CreateOneRequest>,
) -> impl IntoResponse {
    match create_one(state.backend.as_ref(), request) {
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
