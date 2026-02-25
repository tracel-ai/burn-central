use std::{
    sync::{
        Arc,
        mpsc::{RecvTimeoutError, Sender, channel},
    },
    time::Duration,
};

use super::super::{envelope::TelemetryEnvelope, logs::LogBatch, metrics::RecorderHandle};

use super::{LogIngress, Outbox};

pub trait Batcher: Send + Sync {
    fn tick(&self) -> Result<Vec<TelemetryEnvelope>, String>;
}

pub struct BatcherHandle {
    join_handle: Option<std::thread::JoinHandle<()>>,
    shutdown_tx: Option<Sender<()>>,
}

impl BatcherHandle {
    pub fn shutdown(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }

        if let Some(join_handle) = self.join_handle.take() {
            if join_handle.join().is_err() {
                tracing::warn!("telemetry batcher thread panicked during shutdown");
            }
        }
    }
}

impl Drop for BatcherHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

pub struct MetricsEnvelopeBatcher {
    recorder: RecorderHandle,
}

impl MetricsEnvelopeBatcher {
    pub fn new(recorder: RecorderHandle) -> Self {
        Self { recorder }
    }
}

impl Batcher for MetricsEnvelopeBatcher {
    fn tick(&self) -> Result<Vec<TelemetryEnvelope>, String> {
        Ok(vec![TelemetryEnvelope::metrics(self.recorder.snapshot())])
    }
}

pub struct LogsBatcher {
    ingress: Arc<LogIngress>,
    max_batch_entries: usize,
}

impl LogsBatcher {
    pub fn new(ingress: Arc<LogIngress>, max_batch_entries: usize) -> Self {
        Self {
            ingress,
            max_batch_entries,
        }
    }
}

impl Batcher for LogsBatcher {
    fn tick(&self) -> Result<Vec<TelemetryEnvelope>, String> {
        let entries = self.ingress.pop_batch(self.max_batch_entries);

        if entries.is_empty() {
            Ok(vec![])
        } else {
            Ok(vec![TelemetryEnvelope::logs(LogBatch { entries })])
        }
    }
}

pub fn start(
    batcher: Arc<dyn Batcher>,
    outbox: Arc<dyn Outbox>,
    interval: Duration,
    thread_name: &str,
) -> BatcherHandle {
    let (shutdown_tx, shutdown_rx) = channel::<()>();
    let thread_name = thread_name.to_string();
    let join_handle = std::thread::Builder::new()
        .name(thread_name)
        .spawn(move || {
            loop {
                match batcher.tick() {
                    Ok(batches) => {
                        for batch in batches {
                            if let Err(e) = outbox.enqueue(batch) {
                                tracing::error!("failed to enqueue telemetry envelope: {e}");
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!("batcher tick failed: {e}");
                    }
                }

                match shutdown_rx.recv_timeout(interval) {
                    Ok(_) | Err(RecvTimeoutError::Disconnected) => break,
                    Err(RecvTimeoutError::Timeout) => {}
                }
            }
        })
        .expect("failed to spawn batcher thread");

    BatcherHandle {
        join_handle: Some(join_handle),
        shutdown_tx: Some(shutdown_tx),
    }
}
