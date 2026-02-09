use async_stream::stream;
use axum::{
    Json, Router,
    extract::State,
    response::sse::{Event, KeepAlive, Sse},
    routing::post,
};
use burn::backend::NdArray;
use burn::prelude::Backend;
use inference_actor_demo::erased::ErasedSession;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use super::user_lib::{GenerateRequest, build_session};

#[derive(Clone)]
struct AppState {
    session: Arc<dyn ErasedSession>,
}

pub async fn run() {
    type Back = NdArray;
    let device = <Back as Backend>::Device::default();
    let session = build_session::<Back>(device);
    let state = Arc::new(AppState { session: Arc::new(session) });

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
    let payload = serde_json::to_vec(&request).unwrap_or_else(|_| b"{}".to_vec());
    let mut job = state.session.submit_bytes(&payload).ok();

    let stream = stream! {
        if job.is_none() {
            yield Ok(Event::default().event("error").data("submit failed"));
            yield Ok(Event::default().event("done").data("done"));
            return;
        }

        let mut job = job.take().unwrap();
        loop {
            match job.try_recv_bytes() {
                Ok(Some(bytes)) => {
                    let payload = String::from_utf8_lossy(&bytes);
                    yield Ok(Event::default().data(payload));
                }
                Ok(None) => {
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
                Err(_) => {
                    break;
                }
            }
        }
        let _ = job.join();
        yield Ok(Event::default().event("done").data("done"));
    };

    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(5))
            .text("keep-alive"),
    )
}
