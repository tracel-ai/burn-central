use async_stream::stream;
use axum::{
    Json, Router,
    extract::State,
    response::sse::{Event, KeepAlive, Sse},
    routing::post,
};
use burn::backend::NdArray;
use burn::prelude::Backend;
use crossbeam::channel::TryRecvError;
use inference_actor_demo::runtime::{SessionHandle, spawn_session};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use super::user_lib::{GenerateRequest, RnnApp, StepOut};

#[derive(Clone)]
struct AppState {
    session: SessionHandle<GenerateRequest, StepOut, String>,
}

pub async fn run() {
    type Back = NdArray;
    let device = <Back as Backend>::Device::default();
    let session = spawn_session(RnnApp::<Back>::new(device));
    let state = Arc::new(AppState { session });

    let app = Router::new()
        .route("/generate", post(generate))
        .with_state(state);

    let addr: SocketAddr = "127.0.0.1:3001".parse().unwrap();
    println!("Listening on http://{addr}");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn generate(
    State(state): State<Arc<AppState>>,
    Json(request): Json<GenerateRequest>,
) -> Sse<impl futures_util::Stream<Item = Result<Event, Infallible>>> {
    let job = state.session.submit(request);

    let stream = stream! {
        loop {
            match job.stream.try_recv() {
                Ok(item) => {
                    let payload = serde_json::to_string(&item).unwrap_or_else(|_| "{}".to_string());
                    yield Ok(Event::default().data(payload));
                }
                Err(TryRecvError::Empty) => {
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
                Err(TryRecvError::Disconnected) => {
                    break;
                }
            }
        }
        yield Ok(Event::default().event("done").data("done"));
    };

    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(5))
            .text("keep-alive"),
    )
}
