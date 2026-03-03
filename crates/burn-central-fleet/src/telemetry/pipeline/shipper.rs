use std::sync::mpsc::{RecvTimeoutError, Sender, channel};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use burn_central_client::FleetClient;
use burn_central_client::request::{
    MetricData, MetricDescriptorIngestionEvent, MetricIngestionEvent, MetricKind,
    TelemetryIngestionEvents,
};

use crate::telemetry::event::TelemetryData;
use crate::telemetry::metrics::MetricDescriptorKind;

use super::super::event::TelemetryEvent;

use super::Outbox;

pub trait ShipperTransport: Send + Sync {
    fn ship(&self, data: Vec<TelemetryEvent>) -> Result<(), String>;
}

pub struct BurnCentralFleetShipperTransport {
    auth_token: Arc<RwLock<Option<String>>>,
    client: FleetClient,
}

impl BurnCentralFleetShipperTransport {
    pub fn new(auth_token: Arc<RwLock<Option<String>>>, client: FleetClient) -> Self {
        Self { auth_token, client }
    }
}

impl ShipperTransport for BurnCentralFleetShipperTransport {
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
        let auth_token = self
            .auth_token
            .read()
            .map_err(|_| "telemetry auth token lock poisoned".to_string())?
            .clone()
            .ok_or_else(|| "missing auth token for telemetry ingestion".to_string())?;

        self.client
            .ingest_telemetry(auth_token, telemetry)
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

pub fn start(
    outbox: Arc<dyn Outbox>,
    transport: Arc<dyn ShipperTransport>,
    interval: Duration,
) -> ShipperHandle {
    let (shutdown_tx, shutdown_rx) = channel::<()>();
    let join_handle = std::thread::Builder::new()
        .name("telemetry-shipper".to_string())
        .spawn(move || {
            loop {
                match shutdown_rx.recv_timeout(interval) {
                    Ok(_) | Err(RecvTimeoutError::Disconnected) => break,
                    Err(RecvTimeoutError::Timeout) => {}
                }
                match outbox.claim(10) {
                    Ok(None) => {}
                    Ok(Some(items)) => {
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
                            }
                            Err(e) => {
                                tracing::error!("failed to ship telemetry batch: {e}");
                                for id in ids {
                                    if let Err(err) = outbox.fail(id, &e) {
                                        tracing::error!(
                                            "failed to mark telemetry outbox item {id} as failed: {err}"
                                        );
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!("failed to claim telemetry outbox items: {e}");
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    };

    use crate::telemetry::{
        event::TelemetryEvent,
        pipeline::{Outbox, shipper::ShipperTransport},
    };

    #[derive(Debug, Default)]
    struct ShipperTransportMock {
        should_fail: bool,
        ship_called: (AtomicBool, AtomicUsize),
    }

    impl ShipperTransportMock {
        fn succeeding() -> Self {
            Self {
                should_fail: false,
                ship_called: (AtomicBool::new(false), AtomicUsize::new(0)),
            }
        }

        fn failing() -> Self {
            Self {
                should_fail: true,
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

    #[test]
    fn test_fail_is_called_on_ship_failure() {
        let ids = vec![3, 2];
        let outbox = Arc::new(OutboxMock::with_ids(ids.clone()));
        let transport = Arc::new(ShipperTransportMock::failing());

        let _handle = start(outbox.clone(), transport, Duration::from_millis(0));

        std::thread::sleep(Duration::from_millis(100));
        let failed_ids = outbox.failed_ids();

        assert_eq!(failed_ids, Some(ids));
    }

    #[test]
    fn test_complete_is_called_on_ship_success() {
        let ids = vec![3, 2];
        let outbox = Arc::new(OutboxMock::with_ids(ids.clone()));
        let transport = Arc::new(ShipperTransportMock::succeeding());

        let _handle = start(outbox.clone(), transport, Duration::from_millis(0));

        std::thread::sleep(Duration::from_millis(100));
        let completed_ids = outbox.completed_ids();

        assert_eq!(completed_ids, Some(ids));
    }

    #[test]
    fn test_no_ship_on_empty_claim() {
        let outbox = Arc::new(OutboxMock::empty());
        let transport = Arc::new(ShipperTransportMock::succeeding());

        let _handle = start(outbox.clone(), transport.clone(), Duration::from_millis(0));

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

        let _handle = start(outbox.clone(), transport.clone(), Duration::from_millis(0));

        std::thread::sleep(Duration::from_millis(100));
        let ship_called_count = transport.ship_called();

        assert_eq!(ship_called_count, Some(ids.len()));
    }
}
