use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::experiment::{ExperimentRun, ExperimentRunHandle};
use burn::train::ApplicationLoggerInstaller;
use tracing_subscriber::fmt::MakeWriter;

use tracing_core::{Level, LevelFilter};
use tracing_subscriber::filter::filter_fn;
use tracing_subscriber::prelude::*;
use tracing_subscriber::{Layer, registry};

/// The installer for the remote experiment logger.
pub struct RemoteExperimentLoggerInstaller {
    experiment_handle: Arc<ExperimentRunHandle>,
}

impl RemoteExperimentLoggerInstaller {
    /// Creates a new instance of the remote experiment logger installer with the given [BurnCentralClientState].
    pub fn new(experiment_handle: &ExperimentRun) -> Self {
        Self {
            experiment_handle: Arc::new(experiment_handle.handle()),
        }
    }
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

    fn should_flush(&self) -> bool {
        self.last_flush.elapsed() >= self.flush_interval
    }

    fn append(&mut self, message: &str) {
        self.buffer.push_str(message);
    }

    fn flush(&mut self) -> Option<String> {
        if self.buffer.is_empty() {
            return None;
        }

        let content = std::mem::take(&mut self.buffer);
        self.last_flush = Instant::now();
        Some(content)
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
        log_buffer.append(&message);

        // Flush if enough time has elapsed
        if log_buffer.should_flush() {
            if let Some(content) = log_buffer.flush() {
                self.sender.log_info(content);
            }
        }

        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        let mut log_buffer = self.buffer.lock().unwrap();
        if let Some(content) = log_buffer.flush() {
            self.sender.log_info(content);
        }
        Ok(())
    }
}

impl ApplicationLoggerInstaller for RemoteExperimentLoggerInstaller {
    fn install(&self) -> Result<(), String> {
        let make_writer = RemoteWriterMaker {
            experiment_handle: self.experiment_handle.clone(),
            buffer: Arc::new(Mutex::new(LogBuffer::new())),
        };

        let layer = tracing_subscriber::fmt::layer()
            .with_ansi(false)
            .with_writer(make_writer)
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

        if registry().with(layer).try_init().is_err() {
            return Err("Failed to install the file logger.".to_string());
        }

        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            log::error!("PANIC => {info}");
            eprintln!(
                "=== PANIC ===\nA fatal error happened, you can check the experiment logs on Burn Central.\n============="
            );
            hook(info);
        }));

        Ok(())
    }
}
