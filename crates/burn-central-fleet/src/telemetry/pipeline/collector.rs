use std::{
    sync::{
        Arc,
        mpsc::{RecvTimeoutError, Sender, channel},
    },
    time::Duration,
};

use super::super::{event::TelemetryEvent, logs::LogBatch, metrics::RecorderHandle};

use super::{LogIngress, Outbox};

pub trait Collector: Send + Sync {
    fn collect(&self) -> Result<Vec<TelemetryEvent>, String>;
}

pub struct CollectorHandle {
    join_handle: Option<std::thread::JoinHandle<()>>,
    shutdown_tx: Option<Sender<()>>,
}

impl CollectorHandle {
    pub fn shutdown(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }

        if let Some(join_handle) = self.join_handle.take() {
            if join_handle.join().is_err() {
                tracing::warn!("telemetry collector thread panicked during shutdown");
            }
        }
    }
}

impl Drop for CollectorHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

pub struct MetricsEventCollector {
    recorder: RecorderHandle,
}

impl MetricsEventCollector {
    pub fn new(recorder: RecorderHandle) -> Self {
        Self { recorder }
    }
}

impl Collector for MetricsEventCollector {
    fn collect(&self) -> Result<Vec<TelemetryEvent>, String> {
        let mut events = Vec::new();

        let descriptor_delta = self.recorder.take_descriptor_delta();
        if !descriptor_delta.descriptors.is_empty() {
            events.push(TelemetryEvent::metric_descriptors(descriptor_delta));
        }

        events.push(TelemetryEvent::metrics(self.recorder.snapshot()));
        Ok(events)
    }
}

pub struct LogsCollector {
    ingress: Arc<LogIngress>,
    max_batch_entries: usize,
}

impl LogsCollector {
    pub fn new(ingress: Arc<LogIngress>, max_batch_entries: usize) -> Self {
        Self {
            ingress,
            max_batch_entries,
        }
    }
}

impl Collector for LogsCollector {
    fn collect(&self) -> Result<Vec<TelemetryEvent>, String> {
        let entries = self.ingress.pop_batch(self.max_batch_entries);

        if entries.is_empty() {
            Ok(vec![])
        } else {
            Ok(vec![TelemetryEvent::logs(LogBatch { entries })])
        }
    }
}

pub fn start(
    name: &str,
    collector: Arc<dyn Collector>,
    outbox: Arc<dyn Outbox>,
    interval: Duration,
) -> CollectorHandle {
    let (shutdown_tx, shutdown_rx) = channel::<()>();
    let thread_name = name.to_string();
    let join_handle = std::thread::Builder::new()
        .name(thread_name)
        .spawn(move || {
            loop {
                match collector.collect() {
                    Ok(events) => {
                        for event in events {
                            if let Err(e) = outbox.enqueue(event) {
                                tracing::error!("failed to enqueue telemetry event: {e}");
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!("collector collect failed: {e}");
                    }
                }

                match shutdown_rx.recv_timeout(interval) {
                    Ok(_) | Err(RecvTimeoutError::Disconnected) => break,
                    Err(RecvTimeoutError::Timeout) => {}
                }
            }
        })
        .expect("failed to spawn collector thread");

    CollectorHandle {
        join_handle: Some(join_handle),
        shutdown_tx: Some(shutdown_tx),
    }
}
