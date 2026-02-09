//! Axum streaming server example.
//!
//! Starts an HTTP server with a `/generate` endpoint that streams tokens
//! as Server-Sent Events (SSE).
//!
//! Run with: cargo run --example axum_server --features llama3,cuda

use async_stream::stream;
use axum::{
    extract::State,
    response::sse::{Event, KeepAlive, Sse},
    routing::post,
    Json, Router,
};
use burn::tensor::f16;
use llama_burn::inference::{
    continuous_batched_streaming_handler, ContinuousBatchConfig, ContinuousBatchScheduler,
    GenerateRequest, TokenOutput,
};
use llama_burn::llama::build_chat_prompt;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, info};
use tracing_subscriber::EnvFilter;

const SYSTEM_PROMPT: &str = "You are a helpful assistant.";

#[cfg(feature = "cuda")]
use burn::backend::{cuda::CudaDevice, Cuda};
#[cfg(feature = "cuda")]
use burn::prelude::Backend;

#[cfg(feature = "pretrained")]
use burn::record::{HalfPrecisionSettings, NamedMpkFileRecorder};
#[cfg(feature = "pretrained")]
use llama_burn::pretrained::{self, ModelMeta};
#[cfg(feature = "tiny")]
use llama_burn::tokenizer::SentiencePieceTokenizer;
#[cfg(feature = "llama3")]
use llama_burn::tokenizer::Tiktoken;
use llama_burn::{llama::LlamaConfig, tokenizer::Tokenizer};

#[cfg(feature = "cuda")]
type Back = Cuda<f16, i32>;

#[cfg(all(feature = "llama3", feature = "tiny"))]
compile_error!("axum_server example supports only one of 'llama3' or 'tiny' at a time.");

#[cfg(feature = "cuda")]
struct AppState<T: Tokenizer + Send + Sync + 'static> {
    inference: Arc<
        burn_central::runtime::inference::Inference<
            Back,
            llama_burn::llama::Llama<Back, T>,
            burn_central_runtime::inference::In<GenerateRequest>,
            TokenOutput,
        >,
    >,
    device: <Back as Backend>::Device,
}

#[tokio::main]
async fn main() {
    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("This example requires the 'cuda' feature to be enabled");
        eprintln!("Run with: cargo run --example axum_server --features llama3,cuda");
        return;
    }

    #[cfg(feature = "cuda")]
    run().await;
}

#[cfg(feature = "cuda")]
async fn run() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .try_init();

    let device = CudaDevice::default();
    info!("Using CUDA device: {:?}", device);

    let max_seq_len = 256;
    let max_batch_size = 8;

    #[cfg(all(feature = "llama3", feature = "pretrained"))]
    {
        let llama =
            load_llama3_2_1b_with_batch::<Back>(max_seq_len, max_batch_size, &device).unwrap();
        run_with_model::<Tiktoken>(llama, device).await;
        return;
    }

    #[cfg(all(feature = "tiny", feature = "pretrained"))]
    {
        let llama =
            load_tiny_llama_with_batch::<Back>(max_seq_len, max_batch_size, &device).unwrap();
        run_with_model::<SentiencePieceTokenizer>(llama, device).await;
        return;
    }

    #[cfg(not(any(feature = "llama3", feature = "tiny")))]
    {
        eprintln!("Please enable either 'tiny' or 'llama3' feature");
    }
}

#[cfg(feature = "cuda")]
async fn run_with_model<T: Tokenizer + Send + Sync + 'static>(
    llama: llama_burn::llama::Llama<Back, T>,
    device: <Back as Backend>::Device,
) {
    let scheduler = Arc::new(ContinuousBatchScheduler::<Back, T>::new(
        ContinuousBatchConfig::default(),
    ));
    let inference = burn_central::runtime::inference::InferenceBuilder::<Back>::new()
        .with_model(llama)
        .with_extension(scheduler)
        .build(continuous_batched_streaming_handler);

    let state = Arc::new(AppState::<T> {
        inference: Arc::new(inference),
        device,
    });

    let app = Router::new()
        .route("/generate", post(generate::<T>))
        .with_state(state);

    let addr: SocketAddr = "127.0.0.1:3000".parse().unwrap();
    info!("Listening on http://{addr}");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

#[cfg(feature = "cuda")]
async fn generate<T: Tokenizer + Send + Sync + 'static>(
    State(state): State<Arc<AppState<T>>>,
    Json(mut request): Json<GenerateRequest>,
) -> Sse<impl futures_util::Stream<Item = Result<Event, Infallible>>> {
    request.prompt = build_chat_prompt(SYSTEM_PROMPT, &request.prompt);
    debug!(
        "request received: prompt_len={}, max_tokens={}, temperature={}, top_p={}, seed={}",
        request.prompt.len(),
        request.max_tokens,
        request.temperature,
        request.top_p,
        request.seed
    );
    let job = state
        .inference
        .infer(request)
        .with_devices([state.device.clone()])
        .spawn();

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<TokenOutput>();

    std::thread::spawn(move || {
        for token in job.stream.iter() {
            if tx.send(token).is_err() {
                break;
            }
        }
        let _ = job.join();
        debug!("request stream completed");
    });

    let stream = stream! {
        while let Some(token) = rx.recv().await {
            let payload = serde_json::to_string(&token).unwrap_or_else(|_| "{}".to_string());
            yield Ok(Event::default().data(payload));
        }
        yield Ok(Event::default().event("done").data("done"));
    };

    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(5))
            .text("keep-alive"),
    )
}

#[cfg(all(feature = "llama3", feature = "pretrained"))]
fn load_llama3_2_1b_with_batch<B: Backend>(
    max_seq_len: usize,
    max_batch_size: usize,
    device: &B::Device,
) -> Result<llama_burn::llama::Llama<B, Tiktoken>, String> {
    let model = pretrained::Llama::Llama321bInstruct.pretrained();
    let checkpoint = model
        .download_weights()
        .map_err(|err| format!("Could not download weights.\nError: {err}"))?;
    let tokenizer = model
        .download_tokenizer()
        .map_err(|err| format!("Could not download tokenizer.\nError: {err}"))?;

    let tokenizer_path = tokenizer
        .to_str()
        .ok_or("Tokenizer path is not valid UTF-8")?;
    let checkpoint_path = checkpoint
        .to_str()
        .ok_or("Checkpoint path is not valid UTF-8")?;

    let llama = LlamaConfig::llama3_2_1b(tokenizer_path)
        .with_max_seq_len(max_seq_len)
        .with_max_batch_size(max_batch_size)
        .init::<B, Tiktoken>(device)?;

    let recorder = NamedMpkFileRecorder::<HalfPrecisionSettings>::new();
    let llama = llama
        .load(checkpoint_path, &recorder)
        .map_err(|err| format!("Failed to load model.\nError: {err}"))?;

    Ok(llama)
}

#[cfg(all(feature = "tiny", feature = "pretrained"))]
fn load_tiny_llama_with_batch<B: Backend>(
    max_seq_len: usize,
    max_batch_size: usize,
    device: &B::Device,
) -> Result<llama_burn::llama::Llama<B, SentiencePieceTokenizer>, String> {
    let model = pretrained::Llama::TinyLlama.pretrained();
    let checkpoint = model
        .download_weights()
        .map_err(|err| format!("Could not download weights.\nError: {err}"))?;
    let tokenizer = model
        .download_tokenizer()
        .map_err(|err| format!("Could not download tokenizer.\nError: {err}"))?;

    let tokenizer_path = tokenizer
        .to_str()
        .ok_or("Tokenizer path is not valid UTF-8")?;
    let checkpoint_path = checkpoint
        .to_str()
        .ok_or("Checkpoint path is not valid UTF-8")?;

    let llama = LlamaConfig::tiny_llama(tokenizer_path)
        .with_max_seq_len(max_seq_len)
        .with_max_batch_size(max_batch_size)
        .init::<B, SentiencePieceTokenizer>(device)?;

    let recorder = NamedMpkFileRecorder::<HalfPrecisionSettings>::new();
    let llama = llama
        .load(checkpoint_path, &recorder)
        .map_err(|err| format!("Failed to load model.\nError: {err}"))?;

    Ok(llama)
}
