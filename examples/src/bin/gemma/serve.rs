use axum::{
    Router,
    extract::{Json, State},
    routing::post,
};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::Mutex;

// Reusing our toy model structures for the demo
// In a real app, we'd import GemmaForCausalLM from core

#[derive(Clone)]
struct AppState {
    // Mutex for thread safety since our toy model isn't thread-safe (RefCells)
    // In production, we'd use a read-only model or specific inference runtime.
    model: Arc<Mutex<ToyModel>>,
}

// --- Toy Model for Serving ---
struct ToyModel {
    // Just a placeholder for the real model
}

impl ToyModel {
    fn new() -> Self {
        Self {}
    }

    fn generate(&self, prompt: &str) -> String {
        // Mock generation logic
        // In a real scenario, we would:
        // 1. Tokenize prompt -> Tensor
        // 2. model.generate(input_ids)
        // 3. Decode output_ids -> String

        println!("Generating for prompt: {}", prompt);
        // Simulate latency
        std::thread::sleep(std::time::Duration::from_millis(100));

        format!("{} [Generated Response]", prompt)
    }
}

#[derive(Deserialize)]
struct GenerateRequest {
    prompt: String,
}

#[derive(Serialize)]
struct GenerateResponse {
    response: String,
}

async fn generate_handler(
    State(state): State<AppState>,
    Json(payload): Json<GenerateRequest>,
) -> Json<GenerateResponse> {
    let model = state.model.lock().await;
    let response = model.generate(&payload.prompt);
    Json(GenerateResponse { response })
}

#[tokio::main]
async fn main() {
    println!("Initializing Gemma Serving...");

    let model = Arc::new(Mutex::new(ToyModel::new()));
    let state = AppState { model };

    let app = Router::new()
        .route("/generate", post(generate_handler))
        .with_state(state);

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    println!("Listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
