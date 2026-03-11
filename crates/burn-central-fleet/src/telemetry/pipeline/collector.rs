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
    fleet_key: String,
    recorder: RecorderHandle,
}

impl MetricsEventCollector {
    pub fn new(fleet_key: impl Into<String>, recorder: RecorderHandle) -> Self {
        Self {
            fleet_key: fleet_key.into(),
            recorder,
        }
    }
}

impl Collector for MetricsEventCollector {
    fn collect(&self) -> Result<Vec<TelemetryEvent>, String> {
        let mut events = Vec::new();

        let descriptor_delta = self.recorder.take_descriptor_delta(&self.fleet_key);
        if !descriptor_delta.descriptors.is_empty() {
            events.push(TelemetryEvent::metric_descriptors(descriptor_delta));
        }

        let snapshot = self.recorder.snapshot(&self.fleet_key);
        if !snapshot.is_empty() {
            events.push(TelemetryEvent::metrics(snapshot));
        }

        Ok(events)
    }
}

impl Drop for MetricsEventCollector {
    fn drop(&mut self) {
        self.recorder.remove_descriptor_consumer(&self.fleet_key);
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

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use crate::telemetry::pipeline::OutboxId;

    use super::*;

    #[derive(Debug, Default)]
    struct OutboxMock {
        enqueued_events: Mutex<Vec<TelemetryEvent>>,
    }

    impl OutboxMock {
        fn empty() -> Self {
            Self::default()
        }

        fn len(&self) -> usize {
            let guard = self.enqueued_events.lock().unwrap();
            guard.len()
        }

        fn is_empty(&self) -> bool {
            let guard = self.enqueued_events.lock().unwrap();
            guard.is_empty()
        }
    }

    impl Outbox for OutboxMock {
        fn enqueue(&self, data: TelemetryEvent) -> Result<(), String> {
            let mut guard = self.enqueued_events.lock().unwrap();
            guard.push(data);
            Ok(())
        }

        fn claim(&self, _count: usize) -> Result<Option<Vec<(OutboxId, TelemetryEvent)>>, String> {
            Ok(None)
        }

        fn complete(&self, _id: OutboxId) -> Result<(), String> {
            Ok(())
        }

        fn fail(&self, _id: OutboxId, _error: &str) -> Result<(), String> {
            Ok(())
        }
    }

    #[derive(Debug, Default)]
    struct TestCollector {
        events_to_return: Mutex<Vec<TelemetryEvent>>,
    }

    impl Collector for TestCollector {
        fn collect(&self) -> Result<Vec<TelemetryEvent>, String> {
            let mut guard = self.events_to_return.lock().unwrap();
            Ok(guard.drain(..).collect())
        }
    }

    #[test]
    fn test_collector_enqueues_events() {
        let outbox = Arc::new(OutboxMock::empty());
        let collector = Arc::new(TestCollector {
            events_to_return: Mutex::new(vec![
                TelemetryEvent::logs(LogBatch { entries: vec![] }),
                TelemetryEvent::logs(LogBatch { entries: vec![] }),
            ]),
        });

        let _handle = start(
            "test_collector_enqueues_events",
            collector,
            outbox.clone(),
            Duration::from_millis(0),
        );

        std::thread::sleep(Duration::from_millis(100));

        assert_eq!(outbox.len(), 2, "should have enqueued two events");
    }

    #[test]
    fn test_collector_does_not_enqueue_empty_batch() {
        let outbox = Arc::new(OutboxMock::empty());
        let collector = Arc::new(TestCollector {
            events_to_return: Mutex::new(vec![]),
        });

        let _handle = start(
            "test_collector_handles_empty_batch",
            collector,
            outbox.clone(),
            Duration::from_millis(0),
        );

        std::thread::sleep(Duration::from_millis(100));

        assert!(outbox.is_empty(), "should not have enqueued any events");
    }
}
