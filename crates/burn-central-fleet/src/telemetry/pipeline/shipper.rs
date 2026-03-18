use std::sync::Arc;
use std::time::Duration;

use burn_central_client::request::{
    MetricData, MetricDescriptorIngestionEvent, MetricIngestionEvent, MetricKind,
    TelemetryIngestionEvents,
};

use crossbeam::channel::{Receiver, Sender, after, bounded, never, tick};
use crossbeam::select;
use rand::RngExt;

use crate::auth_client::AuthenticatedFleetClient;
use crate::telemetry::event::TelemetryData;
use crate::telemetry::metrics::MetricDescriptorKind;

use super::super::event::TelemetryEvent;

use super::outbox::Outbox;

pub enum TransportReadiness {
    Ready,
    NotReady(&'static str),
}

pub trait ShipperTransport: Send + Sync {
    fn readiness(&self) -> TransportReadiness {
        TransportReadiness::Ready
    }

    fn ship(&self, data: Vec<TelemetryEvent>) -> Result<(), String>;
}

pub struct BurnCentralFleetShipperTransport {
    client: AuthenticatedFleetClient,
}

impl BurnCentralFleetShipperTransport {
    pub fn new(client: AuthenticatedFleetClient) -> Self {
        Self { client }
    }
}

impl ShipperTransport for BurnCentralFleetShipperTransport {
    fn readiness(&self) -> TransportReadiness {
        match self.client.is_ready() {
            true => TransportReadiness::Ready,
            false => TransportReadiness::NotReady("fleet client is not ready"),
        }
    }

    fn ship(&self, data: Vec<TelemetryEvent>) -> Result<(), String> {
        let mut metric_descriptors = Vec::new();
        let mut metrics = Vec::new();
        let mut logs = Vec::new();

        for batch in data {
            match batch.data {
                TelemetryData::Metrics(m) => {
                    for c in m.counters {
                        metrics.push(MetricIngestionEvent {
                            name: c.key.name,
                            timestamp_ms: batch.created_at_unix_ms as _,
                            attributes: c
                                .key
                                .labels
                                .into_iter()
                                .map(|ml| (ml.key, ml.value))
                                .collect(),
                            data: MetricData::Counter { value: c.value },
                        });
                    }
                    for g in m.gauges {
                        metrics.push(MetricIngestionEvent {
                            name: g.key.name,
                            timestamp_ms: batch.created_at_unix_ms as _,
                            attributes: g
                                .key
                                .labels
                                .into_iter()
                                .map(|ml| (ml.key, ml.value))
                                .collect(),
                            data: MetricData::Gauge { value: g.value },
                        });
                    }
                    for h in m.histograms {
                        metrics.push(MetricIngestionEvent {
                            name: h.key.name,
                            timestamp_ms: batch.created_at_unix_ms as _,
                            attributes: h
                                .key
                                .labels
                                .into_iter()
                                .map(|ml| (ml.key, ml.value))
                                .collect(),
                            data: MetricData::Histogram {
                                count: h.count,
                                sum: h.sum,
                                buckets: h.buckets,
                            },
                        });
                    }
                }
                TelemetryData::MetricDescriptors(d) => {
                    for md in d.descriptors {
                        metric_descriptors.push(MetricDescriptorIngestionEvent {
                            name: md.name,
                            kind: match md.kind {
                                MetricDescriptorKind::Counter => MetricKind::Counter,
                                MetricDescriptorKind::Gauge => MetricKind::Gauge,
                                MetricDescriptorKind::Histogram => MetricKind::Histogram,
                            },
                            unit: md.unit,
                            description: Some(md.description),
                        });
                    }
                }
                TelemetryData::Logs(l) => {
                    for log in l.entries {
                        logs.push(burn_central_client::request::LogIngestionEvent {
                            timestamp_ms: log.timestamp_unix_ms as _,
                            level: log.level,
                            message: log.message,
                            attributes: log.fields.into_iter().map(|f| (f.key, f.value)).collect(),
                        });
                    }
                }
            }
        }

        let telemetry = TelemetryIngestionEvents {
            metric_descriptors,
            metrics,
            logs,
        };
        self.client
            .ingest_telemetry(telemetry)
            .map_err(|e| format!("failed to send telemetry events to Burn Central Fleet: {e}"))
    }
}

pub struct ShipperHandle {
    join_handle: Option<std::thread::JoinHandle<()>>,
    shutdown_tx: Option<Sender<()>>,
}

impl ShipperHandle {
    pub fn shutdown(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }

        if let Some(join_handle) = self.join_handle.take() {
            if join_handle.join().is_err() {
                tracing::warn!("telemetry shipper thread panicked during shutdown");
            }
        }
    }
}

impl Drop for ShipperHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[derive(Clone)]
pub struct ShipperWaker {
    tx: Sender<()>,
}

impl ShipperWaker {
    pub fn new() -> (Self, Receiver<()>) {
        let (tx, rx) = bounded(1);
        (Self { tx }, rx)
    }

    pub fn wake(&self) {
        _ = self.tx.try_send(())
    }
}

pub struct ShipperConfig {
    /// Interval for periodic sweeps of the outbox even when no new data is enqueued.
    /// This serves as a safety mechanism to ensure telemetry data eventually gets shipped even if the wake signal is missed or not triggered for some reason.
    pub idle_sweep_interval: Duration,
    /// Minimum backoff delay after a shipping failure before retrying.
    pub min_retry_interval: Duration,
    /// Maximum backoff delay after repeated shipping failures.
    pub max_retry_interval: Duration,
    /// Maximum number of events to ship in a single batch.
    pub max_batch_size: usize,
}

impl Default for ShipperConfig {
    fn default() -> Self {
        Self {
            idle_sweep_interval: Duration::from_secs(60),
            min_retry_interval: Duration::from_secs(5),
            max_retry_interval: Duration::from_secs(30 * 60),
            max_batch_size: 50,
        }
    }
}

pub fn start(
    name: &str,
    outbox: Arc<dyn Outbox>,
    wake_rx: Receiver<()>,
    transport: Arc<dyn ShipperTransport>,
    config: ShipperConfig,
) -> ShipperHandle {
    let (shutdown_tx, shutdown_rx) = bounded::<()>(1);

    let span = tracing::span!(tracing::Level::INFO, "telemetry_shipper", name);

    let join_handle = std::thread::Builder::new()
        .name(name.to_string())
        .spawn(move || {
            let _enter = span.enter();

            let idle_tick_rx = tick(config.idle_sweep_interval);
            let mut consecutive_failures = 0;
            let mut backoff_until = None::<std::time::Instant>;
            let mut should_drain = true;

            loop {
                if !should_drain {
                    let retry_rx = match backoff_until {
                        Some(deadline) => {
                            tracing::trace!(
                                "backing off telemetry shipper for {:?} until {:?} after {} consecutive failures",
                                deadline.saturating_duration_since(std::time::Instant::now()),
                                deadline,
                                consecutive_failures
                            );
                            after(deadline.saturating_duration_since(std::time::Instant::now()))
                        }
                        None => {
                            tracing::trace!("telemetry shipper is idle, waiting for wake signal or idle tick");
                            never()
                        },
                    };

                    select! {
                        recv(shutdown_rx) -> _ => break,
                        recv(wake_rx) -> _ => {
                            tracing::trace!("received shipper wake signal");
                            should_drain = true;
                        }
                        recv(idle_tick_rx) -> _ => {
                            tracing::trace!("received shipper idle tick");
                            should_drain = true;
                        }
                        recv(retry_rx) -> _ => {
                            tracing::trace!("backoff timer expired, retrying telemetry ship");
                            backoff_until = None;
                            should_drain = true;
                        }
                    }
                }

                if !should_drain {
                    continue;
                }
                should_drain = false;

                if let Some(deadline) = backoff_until {
                    let now = std::time::Instant::now();
                    if now < deadline {
                        continue;
                    }
                    backoff_until = None;
                }

                match transport.readiness() {
                    TransportReadiness::Ready => {}
                    TransportReadiness::NotReady(reason) => {
                        tracing::debug!("telemetry shipper transport not ready: {reason}");
                        continue;
                    }
                }

                loop {
                    let items = match outbox.claim(config.max_batch_size) {
                        Ok(None) => {
                            tracing::trace!("no telemetry outbox items to ship");
                            consecutive_failures = 0;
                            break;
                        }
                        Ok(Some(items)) => items,
                        Err(e) => {
                            tracing::error!("failed to claim telemetry outbox items: {e}");
                            backoff_until = Some(
                                std::time::Instant::now()
                                    + calculate_backoff_delay(
                                        consecutive_failures,
                                        config.min_retry_interval,
                                        config.max_retry_interval,
                                    ),
                            );
                            consecutive_failures = consecutive_failures.saturating_add(1);
                            break;
                        }
                    };

                    tracing::debug!("claimed {} telemetry outbox items for shipping", items.len());
                    let (ids, events): (Vec<_>, Vec<_>) = items.into_iter().unzip();
                    match transport.ship(events) {
                        Ok(_) => {
                            for id in ids {
                                if let Err(e) = outbox.complete(id) {
                                    tracing::error!(
                                        "failed to complete telemetry outbox item {id}: {e}"
                                    );
                                }
                            }
                            consecutive_failures = 0;
                            continue;
                        }
                        Err(e) => {
                            tracing::warn!("failed to ship telemetry batch: {e}");
                            for id in ids {
                                if let Err(err) = outbox.fail(id, &e) {
                                    tracing::error!(
                                        "failed to mark telemetry outbox item {id} as failed: {err}"
                                    );
                                }
                            }
                            backoff_until = Some(
                                std::time::Instant::now()
                                    + calculate_backoff_delay(
                                        consecutive_failures,
                                        config.min_retry_interval,
                                        config.max_retry_interval,
                                    ),
                            );
                            consecutive_failures = consecutive_failures.saturating_add(1);
                            break;
                        }
                    }
                }
            }
        })
        .expect("failed to spawn shipper thread");

    ShipperHandle {
        join_handle: Some(join_handle),
        shutdown_tx: Some(shutdown_tx),
    }
}

fn calculate_backoff_delay(
    consecutive_failures: usize,
    min_delay: Duration,
    max_delay: Duration,
) -> Duration {
    let exp = 2u32.saturating_pow(consecutive_failures as u32);
    let raw = min_delay.saturating_mul(exp);
    let capped = raw.min(max_delay);
    let jitter = rand::rng().random_range(0.0..1.0);
    min_delay + capped.saturating_sub(min_delay).mul_f64(jitter)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    };

    use crate::telemetry::{
        event::TelemetryEvent,
        pipeline::{outbox::Outbox, shipper::ShipperTransport},
    };

    #[derive(Debug, Default)]
    struct ShipperTransportMock {
        should_fail: bool,
        ready: bool,
        ship_called: (AtomicBool, AtomicUsize),
    }

    impl ShipperTransportMock {
        fn succeeding() -> Self {
            Self {
                should_fail: false,
                ready: true,
                ship_called: (AtomicBool::new(false), AtomicUsize::new(0)),
            }
        }

        fn failing() -> Self {
            Self {
                should_fail: true,
                ready: true,
                ship_called: (AtomicBool::new(false), AtomicUsize::new(0)),
            }
        }

        fn not_ready() -> Self {
            Self {
                should_fail: false,
                ready: false,
                ship_called: (AtomicBool::new(false), AtomicUsize::new(0)),
            }
        }
    }

    impl ShipperTransportMock {
        fn ship_called(&self) -> Option<usize> {
            if self.ship_called.0.load(Ordering::Relaxed) {
                Some(self.ship_called.1.load(Ordering::Relaxed))
            } else {
                None
            }
        }
    }

    impl ShipperTransport for ShipperTransportMock {
        fn readiness(&self) -> TransportReadiness {
            if self.ready {
                TransportReadiness::Ready
            } else {
                TransportReadiness::NotReady("transport not ready")
            }
        }

        fn ship(&self, data: Vec<TelemetryEvent>) -> Result<(), String> {
            self.ship_called.0.store(true, Ordering::Relaxed);
            self.ship_called.1.fetch_add(data.len(), Ordering::Relaxed);
            if self.should_fail {
                Err("simulated transport failure".to_string())
            } else {
                Ok(())
            }
        }
    }

    #[derive(Debug, Default)]
    struct OutboxMock {
        ids: Mutex<Option<Vec<i64>>>,
        complete_called: Mutex<Option<Vec<i64>>>,
        fail_called: Mutex<Option<Vec<i64>>>,
    }

    impl OutboxMock {
        fn empty() -> Self {
            Self {
                ids: Mutex::new(None),
                complete_called: Mutex::new(None),
                fail_called: Mutex::new(None),
            }
        }

        fn with_ids(ids: Vec<i64>) -> Self {
            Self {
                ids: Mutex::new(Some(ids)),
                complete_called: Mutex::new(None),
                fail_called: Mutex::new(None),
            }
        }

        fn failed_ids(&self) -> Option<Vec<i64>> {
            let mut fail_called = self.fail_called.lock().unwrap();
            fail_called.take()
        }

        fn completed_ids(&self) -> Option<Vec<i64>> {
            let mut complete_called = self.complete_called.lock().unwrap();
            complete_called.take()
        }
    }

    impl Outbox for OutboxMock {
        fn enqueue(&self, _data: TelemetryEvent) -> Result<(), String> {
            Ok(())
        }

        fn claim(&self, _count: usize) -> Result<Option<Vec<(i64, TelemetryEvent)>>, String> {
            Ok(self.ids.lock().unwrap().take().map(|ids| {
                ids.into_iter()
                    .map(|id| {
                        (
                            id,
                            TelemetryEvent::logs(crate::telemetry::logs::LogBatch {
                                entries: vec![],
                            }),
                        )
                    })
                    .collect()
            }))
        }

        fn complete(&self, id: i64) -> Result<(), String> {
            let mut complete_called = self.complete_called.lock().unwrap();
            if let Some(ids) = complete_called.as_mut() {
                ids.push(id);
            } else {
                *complete_called = Some(vec![id]);
            }
            Ok(())
        }

        fn fail(&self, id: i64, _error: &str) -> Result<(), String> {
            let mut fail_called = self.fail_called.lock().unwrap();
            if let Some(ids) = fail_called.as_mut() {
                ids.push(id);
            } else {
                *fail_called = Some(vec![id]);
            }
            Ok(())
        }
    }

    fn start_test_shipper(
        outbox: Arc<dyn Outbox>,
        transport: Arc<dyn ShipperTransport>,
    ) -> ShipperHandle {
        let (_, wake_rx) = ShipperWaker::new();
        start(
            "test-telemetry-shipper",
            outbox,
            wake_rx,
            transport,
            ShipperConfig::default(),
        )
    }

    #[test]
    fn test_fail_is_called_on_ship_failure() {
        let ids = vec![3, 2];
        let outbox = Arc::new(OutboxMock::with_ids(ids.clone()));
        let transport = Arc::new(ShipperTransportMock::failing());

        let _handle = start_test_shipper(outbox.clone(), transport);

        std::thread::sleep(Duration::from_millis(100));
        let failed_ids = outbox.failed_ids();

        assert_eq!(failed_ids, Some(ids));
    }

    #[test]
    fn test_complete_is_called_on_ship_success() {
        let ids = vec![3, 2];
        let outbox = Arc::new(OutboxMock::with_ids(ids.clone()));
        let transport = Arc::new(ShipperTransportMock::succeeding());

        let _handle = start_test_shipper(outbox.clone(), transport);

        std::thread::sleep(Duration::from_millis(100));
        let completed_ids = outbox.completed_ids();

        assert_eq!(completed_ids, Some(ids));
    }

    #[test]
    fn test_no_ship_on_empty_claim() {
        let outbox = Arc::new(OutboxMock::empty());
        let transport = Arc::new(ShipperTransportMock::succeeding());

        let _handle = start_test_shipper(outbox.clone(), transport.clone());

        std::thread::sleep(Duration::from_millis(100));
        let completed_ids = outbox.completed_ids();
        let failed_ids = outbox.failed_ids();

        assert_eq!(completed_ids, None);
        assert_eq!(failed_ids, None);
        assert_eq!(transport.ship_called(), None);
    }

    #[test]
    fn test_no_ship_when_transport_is_not_ready() {
        let ids = vec![3, 2];
        let outbox = Arc::new(OutboxMock::with_ids(ids));
        let transport = Arc::new(ShipperTransportMock::not_ready());

        let _handle = start_test_shipper(outbox.clone(), transport.clone());

        std::thread::sleep(Duration::from_millis(100));
        let completed_ids = outbox.completed_ids();
        let failed_ids = outbox.failed_ids();

        assert_eq!(completed_ids, None);
        assert_eq!(failed_ids, None);
        assert_eq!(transport.ship_called(), None);
    }

    #[test]
    fn test_ship_called_with_claimed_events() {
        let ids = vec![3, 2];
        let outbox = Arc::new(OutboxMock::with_ids(ids.clone()));
        let transport = Arc::new(ShipperTransportMock::succeeding());

        let _handle = start_test_shipper(outbox.clone(), transport.clone());

        std::thread::sleep(Duration::from_millis(100));
        let ship_called_count = transport.ship_called();

        assert_eq!(ship_called_count, Some(ids.len()));
    }

    #[test]
    fn test_backoff_delay_respects_configured_bounds() {
        let min = Duration::from_secs(5);
        let max = Duration::from_secs(30);

        for failures in 0..10 {
            let delay = calculate_backoff_delay(failures, min, max);
            assert!(
                delay >= min,
                "delay should not be below the minimum retry interval"
            );
            assert!(
                delay <= max,
                "delay should not exceed the maximum retry interval"
            );
        }
    }
}
