mod error;

pub use error::ServerError;

use crate::{job::Job, job_register::JobRegister, mapper::JsonMapper};
use axum::{
    Router,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::post,
};
use serde::{Serialize, de::DeserializeOwned};
use std::sync::Arc;

pub struct Server {
    register: JobRegister,
    host: String,
    port: u16,
}

impl Default for Server {
    fn default() -> Self {
        Self::new()
    }
}

impl Server {
    pub fn new() -> Self {
        Self {
            register: JobRegister::new(),
            host: "0.0.0.0".to_string(),
            port: 3000,
        }
    }

    pub fn host(mut self, host: &str) -> Self {
        self.host = host.to_string();
        self
    }

    pub fn port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    pub fn register<J, I, O>(mut self, job: J) -> Self
    where
        J: Job<I, O> + Send + Sync + 'static,
        I: DeserializeOwned + Send + Sync + 'static,
        O: 'static,
    {
        self.register = self.register.register(job, JsonMapper::new());
        self
    }

    pub fn register_with_default<J, I, O>(mut self, job: J, default: I) -> Self
    where
        J: Job<I, O> + Send + Sync + 'static,
        I: DeserializeOwned + Serialize + Send + Sync + 'static,
        O: 'static,
    {
        self.register = self
            .register
            .register(job, JsonMapper::with_default(default));
        self
    }

    pub async fn run_async(self) -> Result<(), ServerError> {
        let addr = format!("{}:{}", self.host, self.port);
        let state = Arc::new(self.register);

        let app = Router::new()
            .route("/{job_name}", post(run_job))
            .with_state(state);

        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .try_init();
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        println!();
        tracing::info!(
            "Server listening on http://localhost:{}",
            listener.local_addr()?.port()
        );
        axum::serve(listener, app).await?;
        Ok(())
    }

    pub fn run(self) -> Result<(), ServerError> {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?
            .block_on(self.run_async())
    }
}

async fn run_job(
    State(register): State<Arc<JobRegister>>,
    Path(job_name): Path<String>,
    body: String,
) -> impl IntoResponse {
    let input = match register.validate(&job_name, &body) {
        Ok(input) => input,
        Err(e) => {
            let e = ServerError::from(e);
            let status = match &e {
                ServerError::UnknownJob { .. } => StatusCode::NOT_FOUND,
                ServerError::ValidationFailed(_) => StatusCode::BAD_REQUEST,
                ServerError::ExecutionFailed(_) | ServerError::IoError(_) => {
                    StatusCode::INTERNAL_SERVER_ERROR
                }
            };
            return (status, e.to_string());
        }
    };

    let response = format!("Job '{job_name}' has started running");

    let handle = tokio::task::spawn_blocking(move || register.run(&job_name, input));

    tokio::spawn(async move {
        match handle.await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => tracing::error!("Job failed: {e}"),
            Err(e) => tracing::error!("Job panicked: {e}"),
        }
    });

    (StatusCode::OK, response)
}
