use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use burn_central_core::experiment::{ExperimentRun, ExperimentRunHandle};
use tracing::Level;
use tracing::level_filters::LevelFilter;
use tracing_subscriber::filter::filter_fn;
use tracing_subscriber::fmt::MakeWriter;
use tracing_subscriber::registry;
use tracing_subscriber::{EnvFilter, prelude::*};

#[derive(Debug, thiserror::Error)]
pub enum TelemetryError {
    #[error("Failed to initialize telemetry subscriber: {0}")]
    Init(#[from] tracing_subscriber::util::TryInitError),
}

struct LogBuffer {
    buffer: String,
    last_flush: Instant,
    flush_interval: Duration,
}

impl LogBuffer {
    fn new() -> Self {
        Self {
            buffer: String::new(),
            last_flush: Instant::now(),
            flush_interval: Duration::from_secs(1),
        }
    }

    fn push_and_take_ready(&mut self, message: &str) -> Vec<String> {
        self.buffer.push_str(message);
        let mut ready = Vec::new();

        while let Some(pos) = self.buffer.find('\n') {
            let line = self.buffer[..pos + 1].to_string();
            self.buffer.drain(..=pos);
            if !line.is_empty() {
                ready.push(line);
            }
        }

        if self.last_flush.elapsed() >= self.flush_interval && !self.buffer.is_empty() {
            ready.push(std::mem::take(&mut self.buffer));
        }

        if !ready.is_empty() {
            self.last_flush = Instant::now();
        }

        ready
    }

    fn flush_all(&mut self) -> Option<String> {
        if self.buffer.is_empty() {
            return None;
        }
        self.last_flush = Instant::now();
        Some(std::mem::take(&mut self.buffer))
    }
}

struct RemoteWriter {
    sender: Arc<ExperimentRunHandle>,
    buffer: Arc<Mutex<LogBuffer>>,
}

struct RemoteWriterMaker {
    experiment_handle: Arc<ExperimentRunHandle>,
    buffer: Arc<Mutex<LogBuffer>>,
}

impl RemoteWriterMaker {
    fn new(experiment_handle: Arc<ExperimentRunHandle>, buffer: Arc<Mutex<LogBuffer>>) -> Self {
        Self {
            experiment_handle,
            buffer,
        }
    }
}

impl MakeWriter<'_> for RemoteWriterMaker {
    type Writer = RemoteWriter;

    fn make_writer(&self) -> Self::Writer {
        let sender = self.experiment_handle.clone();
        let buffer = self.buffer.clone();
        RemoteWriter { sender, buffer }
    }
}

impl std::io::Write for RemoteWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let message = String::from_utf8_lossy(buf).to_string();

        let mut log_buffer = self.buffer.lock().unwrap();
        for line in log_buffer.push_and_take_ready(&message) {
            let _ = self.sender.try_log_info(line);
        }

        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        let mut log_buffer = self.buffer.lock().unwrap();
        if let Some(content) = log_buffer.flush_all() {
            let _ = self.sender.try_log_info(content);
        }
        Ok(())
    }
}

pub fn install_for_experiment(experiment: &ExperimentRun) -> Result<(), TelemetryError> {
    let experiment_handle = Arc::new(experiment.handle());
    let buffer = Arc::new(Mutex::new(LogBuffer::new()));

    let layer = tracing_subscriber::fmt::layer()
        .with_ansi(false)
        .with_writer(RemoteWriterMaker::new(experiment_handle.clone(), buffer))
        .with_filter(LevelFilter::INFO)
        .with_filter(filter_fn(|m| {
            if let Some(path) = m.module_path() {
                // The wgpu crate is logging too much, so we skip `info` level.
                if path.starts_with("wgpu") && *m.level() >= Level::INFO {
                    return false;
                }
            }
            true
        }));

    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    let subscriber = registry().with(env_filter).with(layer);

    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        tracing::error!("PANIC => {info}");
        hook(info);
    }));

    match subscriber.try_init() {
        Ok(()) => Ok(()),
        Err(err) => Err(TelemetryError::Init(err)),
    }
}
