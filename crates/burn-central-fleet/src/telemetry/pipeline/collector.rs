use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use crossbeam::channel::{Receiver, Sender, TrySendError, bounded, select, tick};

use super::super::{
    event::TelemetryEvent,
    logs::{LogBatch, LogRecord},
    metrics::RecorderHandle,
};

use super::outbox::Outbox;

pub(super) struct LogIngress {
    sender: Sender<LogRecord>,
}

impl LogIngress {
    pub(super) fn bounded(capacity: usize) -> (Self, Receiver<LogRecord>) {
        let (sender, receiver) = bounded(capacity);
        (Self { sender }, receiver)
    }

    pub(super) fn push(&self, record: LogRecord) {
        match self.sender.try_send(record) {
            Ok(()) | Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {}
        }
    }
}

pub struct CollectorHandle {
    join_handle: Option<std::thread::JoinHandle<()>>,
    shutdown_tx: Option<Sender<()>>,
}

impl CollectorHandle {
    fn spawn(name: &str, run: impl FnOnce(Receiver<()>) + Send + 'static) -> CollectorHandle {
        let (shutdown_tx, shutdown_rx) = bounded::<()>(1);
        let thread_name = name.to_string();
        let join_handle = std::thread::Builder::new()
            .name(thread_name)
            .spawn(move || run(shutdown_rx))
            .expect("failed to spawn collector thread");

        CollectorHandle {
            join_handle: Some(join_handle),
            shutdown_tx: Some(shutdown_tx),
        }
    }

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
    interval: Duration,
}

impl MetricsEventCollector {
    pub fn new(fleet_key: impl Into<String>, recorder: RecorderHandle, interval: Duration) -> Self {
        Self {
            fleet_key: fleet_key.into(),
            recorder,
            interval,
        }
    }

    pub fn start(self, name: &str, outbox: Arc<dyn Outbox>) -> CollectorHandle {
        let span = tracing::info_span!("metrics_collector", name);
        CollectorHandle::spawn(name, move |shutdown_rx| {
            let _guard = span.enter();
            self.run(outbox, shutdown_rx)
        })
    }

    fn emit(&self, outbox: &dyn Outbox) {
        let mut events = Vec::new();

        let descriptor_delta = self.recorder.take_descriptor_delta(&self.fleet_key);
        let descriptor_count = descriptor_delta.descriptors.len();
        if !descriptor_delta.descriptors.is_empty() {
            events.push(TelemetryEvent::metric_descriptors(descriptor_delta));
        }

        let snapshot = self.recorder.snapshot(&self.fleet_key);
        let counter_count = snapshot.counters.len();
        let gauge_count = snapshot.gauges.len();
        let histogram_count = snapshot.histograms.len();
        if !snapshot.is_empty() {
            events.push(TelemetryEvent::metrics(snapshot));
        }

        if !events.is_empty() {
            tracing::debug!(
                descriptor_count,
                counter_count,
                gauge_count,
                histogram_count,
                "emitting metrics telemetry batch"
            );
        }

        enqueue_events(outbox, events);
    }

    fn run(self, outbox: Arc<dyn Outbox>, shutdown_rx: Receiver<()>) {
        tracing::trace!(interval = ?self.interval, "starting metrics collector");
        let ticker = tick(self.interval);
        let outbox = outbox.as_ref();
        loop {
            select! {
                recv(shutdown_rx) -> _ => {
                    tracing::debug!("received metrics collector shutdown, emitting final snapshot");
                    self.emit(outbox);
                    break;
                }
                recv(ticker) -> _ => {
                    tracing::trace!("received metrics collector tick");
                    self.emit(outbox);
                }
            }
        }
    }
}

impl Drop for MetricsEventCollector {
    fn drop(&mut self) {
        self.recorder.remove_descriptor_consumer(&self.fleet_key);
    }
}

const LOG_INGRESS_CAPACITY: usize = 4_096;

pub struct LogsCollector {
    ingress_rx: Receiver<LogRecord>,
    max_batch_entries: usize,
    flush_interval: Duration,
}

impl LogsCollector {
    pub fn spawn(
        name: &str,
        outbox: Arc<dyn Outbox>,
        max_batch_entries: usize,
        flush_interval: Duration,
    ) -> (LogIngress, CollectorHandle) {
        let (ingress, ingress_rx) = LogIngress::bounded(LOG_INGRESS_CAPACITY);
        let handle = Self::new(ingress_rx, max_batch_entries, flush_interval).start(name, outbox);
        (ingress, handle)
    }

    pub fn new(
        ingress_rx: Receiver<LogRecord>,
        max_batch_entries: usize,
        flush_interval: Duration,
    ) -> Self {
        Self {
            ingress_rx,
            max_batch_entries,
            flush_interval,
        }
    }

    pub fn start(self, name: &str, outbox: Arc<dyn Outbox>) -> CollectorHandle {
        let span = tracing::info_span!("logs_collector", name);
        CollectorHandle::spawn(name, move |shutdown_rx| {
            let _guard = span.enter();
            self.run(outbox, shutdown_rx)
        })
    }

    fn flush(&self, outbox: &dyn Outbox, entries: &mut Vec<LogRecord>) {
        if entries.is_empty() {
            return;
        }

        let batch = LogBatch {
            entries: std::mem::take(entries),
        };
        enqueue_events(outbox, [TelemetryEvent::logs(batch)]);
    }

    fn drain_ready(&self, entries: &mut Vec<LogRecord>) -> bool {
        while entries.len() < self.max_batch_entries {
            match self.ingress_rx.try_recv() {
                Ok(record) => entries.push(record),
                Err(crossbeam::channel::TryRecvError::Empty) => return false,
                Err(crossbeam::channel::TryRecvError::Disconnected) => {
                    tracing::debug!("log ingress disconnected while draining buffered records");
                    return true;
                }
            }
        }

        false
    }

    fn run(self, outbox: Arc<dyn Outbox>, shutdown_rx: Receiver<()>) {
        let mut entries = Vec::with_capacity(self.max_batch_entries);
        let mut flush_deadline = None;
        let outbox = outbox.as_ref();

        tracing::trace!(
            max_batch_entries = self.max_batch_entries,
            flush_interval = ?self.flush_interval,
            "starting logs collector"
        );

        loop {
            if entries.len() >= self.max_batch_entries {
                tracing::debug!(
                    "log batch reached max capacity of {}, flushing",
                    self.max_batch_entries
                );
                self.flush(outbox, &mut entries);
                flush_deadline = None;
                continue;
            }

            if let Some(deadline) = flush_deadline {
                if Instant::now() >= deadline {
                    tracing::debug!("log batch flush deadline reached, flushing");
                    self.flush(outbox, &mut entries);
                    flush_deadline = None;
                    continue;
                }
            }

            if entries.is_empty() {
                select! {
                    recv(shutdown_rx) -> _ => {
                        tracing::debug!("received logs collector shutdown");
                        self.flush(outbox, &mut entries);
                        break;
                    }
                    recv(self.ingress_rx) -> result => match result {
                        Ok(record) => {
                            entries.push(record);
                            flush_deadline = Some(Instant::now() + self.flush_interval);
                            tracing::trace!(
                                buffered_entries = entries.len(),
                                flush_interval = ?self.flush_interval,
                                "received first log entry for batch, scheduled flush deadline"
                            );
                            if self.drain_ready(&mut entries) {
                                tracing::debug!(
                                    buffered_entries = entries.len(),
                                    "flushing final log batch after ingress disconnect"
                                );
                                self.flush(outbox, &mut entries);
                                break;
                            }
                        }
                        Err(_) => {
                            tracing::debug!("log ingress disconnected, stopping logs collector");
                            self.flush(outbox, &mut entries);
                            break;
                        }
                    }
                }
                continue;
            }

            let timeout = flush_deadline
                .expect("flush deadline should be set while log buffer is non-empty")
                .saturating_duration_since(Instant::now());
            select! {
                recv(shutdown_rx) -> _ => {
                    tracing::debug!(
                        buffered_entries = entries.len(),
                        "received logs collector shutdown, flushing buffered entries"
                    );
                    self.flush(outbox, &mut entries);
                    break;
                }
                recv(self.ingress_rx) -> result => match result {
                    Ok(record) => {
                        entries.push(record);
                        if self.drain_ready(&mut entries) {
                            tracing::debug!(
                                buffered_entries = entries.len(),
                                "flushing final log batch after ingress disconnect"
                            );
                            self.flush(outbox, &mut entries);
                            break;
                        }
                    }
                    Err(_) => {
                        tracing::debug!(
                            buffered_entries = entries.len(),
                            "log ingress disconnected, flushing buffered entries and stopping"
                        );
                        self.flush(outbox, &mut entries);
                        break;
                    }
                },
                default(timeout) => {
                    self.flush(outbox, &mut entries);
                    flush_deadline = None;
                }
            }
        }
    }
}

fn enqueue_events(outbox: &dyn Outbox, events: impl IntoIterator<Item = TelemetryEvent>) {
    for event in events {
        if let Err(e) = outbox.enqueue(event) {
            tracing::error!("failed to enqueue telemetry event: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use crate::telemetry::{
        event::TelemetryData,
        logs::LogRecord,
        metrics::{InMemoryMetricsRecorder, MetricDescriptorKind},
        pipeline::outbox::OutboxId,
    };
    use tracing_subscriber::layer::SubscriberExt;

    use super::*;

    fn with_fleet_span(fleet_key: &str, test_fn: impl FnOnce()) {
        let subscriber = tracing_subscriber::registry()
            .with(crate::telemetry::logs::TelemetryLogLayer::default());
        tracing::subscriber::with_default(subscriber, || {
            let span = tracing::info_span!("test.metric_descriptor", fleet_key = fleet_key);
            let _guard = span.enter();
            test_fn();
        });
    }

    #[derive(Debug, Default)]
    struct OutboxMock {
        enqueued_events: Mutex<Vec<TelemetryEvent>>,
    }

    impl OutboxMock {
        fn empty() -> Self {
            Self::default()
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

    #[test]
    fn logs_collector_flushes_when_buffer_is_full() {
        let outbox = Arc::new(OutboxMock::empty());
        let (tx, rx) = bounded(8);

        let _handle = LogsCollector::new(rx, 2, Duration::from_secs(60))
            .start("logs_collector_flushes_when_buffer_is_full", outbox.clone());

        tx.send(sample_log("first")).unwrap();
        tx.send(sample_log("second")).unwrap();

        std::thread::sleep(Duration::from_millis(100));

        let event = {
            let guard = outbox.enqueued_events.lock().unwrap();
            guard.first().cloned()
        }
        .expect("collector should flush a log batch");

        match event.data {
            TelemetryData::Logs(batch) => {
                assert_eq!(batch.entries.len(), 2);
            }
            other => panic!("expected log batch, got {other:?}"),
        }
    }

    #[test]
    fn logs_collector_flushes_on_timeout() {
        let outbox = Arc::new(OutboxMock::empty());
        let (tx, rx) = bounded(8);

        let _handle = LogsCollector::new(rx, 8, Duration::from_millis(20))
            .start("logs_collector_flushes_on_timeout", outbox.clone());

        tx.send(sample_log("timeout")).unwrap();

        std::thread::sleep(Duration::from_millis(100));

        let event = {
            let guard = outbox.enqueued_events.lock().unwrap();
            guard.first().cloned()
        }
        .expect("collector should flush pending logs after timeout");

        match event.data {
            TelemetryData::Logs(batch) => {
                assert_eq!(batch.entries.len(), 1);
                assert_eq!(batch.entries[0].message, "timeout");
            }
            other => panic!("expected log batch, got {other:?}"),
        }
    }

    #[test]
    fn metrics_collector_emits_descriptor_delta_and_snapshot_on_tick() {
        let recorder = InMemoryMetricsRecorder::new();
        let handle = recorder.handle();
        let outbox = Arc::new(OutboxMock::empty());

        with_fleet_span("fleet-a", || {
            metrics::Recorder::describe_counter(
                &recorder,
                "fleet.requests.total".into(),
                Some(metrics::Unit::Count),
                "Total requests".into(),
            );
        });
        metrics::with_local_recorder(&recorder, || {
            metrics::counter!("fleet.requests.total", "fleet_key" => "fleet-a").increment(3);
        });

        let _collector = MetricsEventCollector::new("fleet-a", handle, Duration::from_millis(20))
            .start(
                "metrics_collector_emits_descriptor_delta_and_snapshot_on_tick",
                outbox.clone(),
            );

        std::thread::sleep(Duration::from_millis(100));

        let events = {
            let guard = outbox.enqueued_events.lock().unwrap();
            guard.clone()
        };
        assert_eq!(
            events.len(),
            2,
            "collector should emit descriptors and metrics"
        );

        assert!(events.iter().any(|event| {
            matches!(
                &event.data,
                TelemetryData::MetricDescriptors(batch)
                    if batch.descriptors.iter().any(|descriptor| {
                        descriptor.name == "fleet.requests.total"
                            && descriptor.kind == MetricDescriptorKind::Counter
                    })
            )
        }));
        assert!(events.iter().any(|event| {
            matches!(
                &event.data,
                TelemetryData::Metrics(batch)
                    if batch
                        .counters
                        .iter()
                        .any(|counter| counter.key.name == "fleet.requests.total" && counter.value == 3)
            )
        }));
    }

    fn sample_log(message: &str) -> LogRecord {
        LogRecord::new("fleet-a".to_string(), "info".to_string(), message, vec![])
    }
}
