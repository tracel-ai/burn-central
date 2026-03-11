use serde::{Deserialize, Serialize};

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use super::logs::current_fleet_key;

const FLEET_KEY_LABEL: &str = "fleet_key";

#[derive(Debug)]
struct InnerRegistry {
    registry: metrics_util::registry::Registry<metrics::Key, metrics_util::registry::AtomicStorage>,
    descriptor_store: Mutex<DescriptorStore>,
    metric_delta_store: Mutex<MetricDeltaStore>,
}

impl InnerRegistry {
    fn new() -> Self {
        Self {
            registry: metrics_util::registry::Registry::new(metrics_util::registry::AtomicStorage),
            descriptor_store: Mutex::new(DescriptorStore::default()),
            metric_delta_store: Mutex::new(MetricDeltaStore::default()),
        }
    }

    pub fn register_counter(&self, key: &metrics::Key) -> metrics::Counter {
        self.registry
            .get_or_create_counter(key, |c| c.clone())
            .into()
    }

    pub fn register_gauge(&self, key: &metrics::Key) -> metrics::Gauge {
        self.registry.get_or_create_gauge(key, |g| g.clone()).into()
    }

    pub fn register_histogram(&self, key: &metrics::Key) -> metrics::Histogram {
        self.registry
            .get_or_create_histogram(key, |h| h.clone())
            .into()
    }

    fn snapshot(&self, fleet_key: &str) -> MetricBatch {
        let mut counters = Vec::new();
        let mut gauges = Vec::new();
        {
            let mut delta_store = self.metric_delta_store.lock().unwrap();

            self.registry.visit_counters(|key, counter| {
                let key = MetricKey::from_key(key);
                if !key.has_label(FLEET_KEY_LABEL, fleet_key) {
                    return;
                }

                let value = counter.load(Ordering::Acquire);
                let previous = delta_store.counter_values.insert(key.clone(), value);
                let delta = previous.map_or(value, |last_value| value.saturating_sub(last_value));
                if delta > 0 {
                    counters.push(MetricCounter { key, value: delta });
                }
            });

            self.registry.visit_gauges(|key, gauge| {
                let value = f64::from_bits(gauge.load(Ordering::Acquire));
                if !value.is_finite() {
                    return;
                }

                let key = MetricKey::from_key(key);
                if !key.has_label(FLEET_KEY_LABEL, fleet_key) {
                    return;
                }

                let previous = delta_store.gauge_values.get(&key).copied();
                if previous != Some(value) {
                    gauges.push(MetricGauge {
                        key: key.clone(),
                        value,
                    });
                    delta_store.gauge_values.insert(key, value);
                }
            });
        }
        counters.sort_by(|a, b| a.key.cmp(&b.key));
        gauges.sort_by(|a, b| a.key.cmp(&b.key));

        let mut histograms = Vec::new();
        self.registry.visit_histograms(|key, histogram| {
            let key = MetricKey::from_key(key);
            if !key.has_label(FLEET_KEY_LABEL, fleet_key) {
                return;
            }

            let mut samples = Vec::new();
            histogram.clear_with(|chunk| samples.extend_from_slice(chunk));

            if let Some(summary) = MetricHistogram::from_samples(key, samples) {
                histograms.push(summary);
            }
        });
        histograms.sort_by(|a, b| a.key.cmp(&b.key));

        MetricBatch {
            counters,
            gauges,
            histograms,
        }
    }

    fn describe(
        &self,
        kind: MetricDescriptorKind,
        key: metrics::KeyName,
        unit: Option<metrics::Unit>,
        description: metrics::SharedString,
    ) {
        let Some(fleet_key) = current_fleet_key() else {
            return;
        };

        let descriptor = MetricDescriptor {
            name: key.as_str().to_string(),
            kind,
            unit: unit.map(|value| value.as_str().to_string()),
            description: description.to_string(),
        };
        let descriptor_key = MetricDescriptorKey::new(descriptor.name.clone(), descriptor.kind);

        let mut store_guard = self.descriptor_store.lock().unwrap();
        let fleet_store = store_guard.by_fleet.entry(fleet_key).or_default();
        let changed = fleet_store.descriptors.get(&descriptor_key) != Some(&descriptor);
        if changed {
            fleet_store
                .descriptors
                .insert(descriptor_key.clone(), descriptor);
            fleet_store.dirty.insert(descriptor_key);
        }
    }

    fn take_descriptor_delta(&self, fleet_key: &str) -> MetricDescriptorBatch {
        let mut store_guard = self.descriptor_store.lock().unwrap();
        let Some(fleet_store) = store_guard.by_fleet.get_mut(fleet_key) else {
            return MetricDescriptorBatch {
                descriptors: Vec::new(),
            };
        };

        let dirty_keys = fleet_store.dirty.iter().cloned().collect::<Vec<_>>();
        let mut descriptors = Vec::with_capacity(dirty_keys.len());
        for key in dirty_keys {
            if let Some(descriptor) = fleet_store.descriptors.get(&key) {
                descriptors.push(descriptor.clone());
            }
        }
        fleet_store.dirty.clear();
        MetricDescriptorBatch { descriptors }
    }

    fn remove_descriptor_consumer(&self, fleet_key: &str) {
        let mut store_guard = self.descriptor_store.lock().unwrap();
        let Some(fleet_store) = store_guard.by_fleet.get_mut(fleet_key) else {
            return;
        };

        fleet_store
            .dirty
            .extend(fleet_store.descriptors.keys().cloned());
    }
}

#[derive(Debug, Clone)]
pub struct RecorderHandle {
    registry: Arc<InnerRegistry>,
}

impl RecorderHandle {
    /// Produces a snapshot of recorded metrics.
    /// Counters are emitted as positive deltas since the prior snapshot.
    /// Gauges are emitted only when their value changes.
    /// Histograms are emitted as full snapshots of all recorded samples, and are cleared after each snapshot.
    pub fn snapshot(&self, fleet_key: &str) -> MetricBatch {
        self.registry.snapshot(fleet_key)
    }

    /// Drains newly registered or updated metric descriptors for a single fleet.
    pub fn take_descriptor_delta(&self, fleet_key: &str) -> MetricDescriptorBatch {
        self.registry.take_descriptor_delta(fleet_key)
    }

    pub fn remove_descriptor_consumer(&self, fleet_key: &str) {
        self.registry.remove_descriptor_consumer(fleet_key);
    }
}

#[derive(Debug, Clone)]
pub struct InMemoryMetricsRecorder {
    registry: Arc<InnerRegistry>,
}

impl InMemoryMetricsRecorder {
    pub fn new() -> Self {
        Self {
            registry: Arc::new(InnerRegistry::new()),
        }
    }

    pub fn handle(&self) -> RecorderHandle {
        RecorderHandle {
            registry: self.registry.clone(),
        }
    }
}

impl metrics::Recorder for InMemoryMetricsRecorder {
    fn describe_counter(
        &self,
        key: metrics::KeyName,
        unit: Option<metrics::Unit>,
        description: metrics::SharedString,
    ) {
        self.registry
            .describe(MetricDescriptorKind::Counter, key, unit, description);
    }
    fn describe_gauge(
        &self,
        key: metrics::KeyName,
        unit: Option<metrics::Unit>,
        description: metrics::SharedString,
    ) {
        self.registry
            .describe(MetricDescriptorKind::Gauge, key, unit, description);
    }
    fn describe_histogram(
        &self,
        key: metrics::KeyName,
        unit: Option<metrics::Unit>,
        description: metrics::SharedString,
    ) {
        self.registry
            .describe(MetricDescriptorKind::Histogram, key, unit, description);
    }

    fn register_counter(
        &self,
        key: &metrics::Key,
        _meta: &metrics::Metadata<'_>,
    ) -> metrics::Counter {
        self.registry.register_counter(key)
    }

    fn register_gauge(&self, key: &metrics::Key, _meta: &metrics::Metadata<'_>) -> metrics::Gauge {
        self.registry.register_gauge(key)
    }

    fn register_histogram(
        &self,
        key: &metrics::Key,
        _meta: &metrics::Metadata<'_>,
    ) -> metrics::Histogram {
        self.registry.register_histogram(key)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricBatch {
    pub counters: Vec<MetricCounter>,
    pub gauges: Vec<MetricGauge>,
    pub histograms: Vec<MetricHistogram>,
}

impl MetricBatch {
    pub fn is_empty(&self) -> bool {
        self.counters.is_empty() && self.gauges.is_empty() && self.histograms.is_empty()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricDescriptorBatch {
    pub descriptors: Vec<MetricDescriptor>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "lowercase")]
pub enum MetricDescriptorKind {
    Counter,
    Gauge,
    Histogram,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MetricDescriptor {
    pub name: String,
    pub kind: MetricDescriptorKind,
    pub unit: Option<String>,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct MetricDescriptorKey {
    name: String,
    kind: MetricDescriptorKind,
}

impl MetricDescriptorKey {
    fn new(name: String, kind: MetricDescriptorKind) -> Self {
        Self { name, kind }
    }
}

#[derive(Debug, Default)]
struct DescriptorStore {
    by_fleet: BTreeMap<String, FleetDescriptorStore>,
}

#[derive(Debug, Default)]
struct FleetDescriptorStore {
    descriptors: BTreeMap<MetricDescriptorKey, MetricDescriptor>,
    dirty: BTreeSet<MetricDescriptorKey>,
}

#[derive(Debug, Default)]
struct MetricDeltaStore {
    counter_values: BTreeMap<MetricKey, u64>,
    gauge_values: BTreeMap<MetricKey, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct MetricLabel {
    pub key: String,
    pub value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct MetricKey {
    pub name: String,
    pub labels: Vec<MetricLabel>,
}

impl MetricKey {
    fn from_key(key: &metrics::Key) -> Self {
        let mut labels = key
            .labels()
            .map(|label| MetricLabel {
                key: label.key().to_string(),
                value: label.value().to_string(),
            })
            .collect::<Vec<_>>();
        labels.sort();

        Self {
            name: key.name().to_string(),
            labels,
        }
    }

    fn has_label(&self, key: &str, value: &str) -> bool {
        self.labels
            .iter()
            .any(|label| label.key == key && label.value == value)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricCounter {
    pub key: MetricKey,
    pub value: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricGauge {
    pub key: MetricKey,
    pub value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricHistogram {
    pub key: MetricKey,
    pub count: u64,
    pub sum: f64,
    pub buckets: Vec<(f64, u64)>,
}

impl MetricHistogram {
    fn from_samples(key: MetricKey, samples: Vec<f64>) -> Option<Self> {
        let mut count = 0u64;
        let mut sum = 0.0;
        let mut finite_samples = Vec::new();

        for sample in samples {
            if !sample.is_finite() {
                continue;
            }

            count += 1;
            sum += sample;
            finite_samples.push(sample);
        }

        if count == 0 {
            return None;
        }

        finite_samples.sort_by(|a, b| a.total_cmp(b));
        let mut buckets = Vec::new();
        let mut cumulative_count = 0u64;

        // Build cumulative buckets keyed by each observed value.
        for sample in finite_samples {
            cumulative_count += 1;
            if let Some((upper_bound, bucket_count)) = buckets.last_mut() {
                if *upper_bound == sample {
                    *bucket_count = cumulative_count;
                    continue;
                }
            }
            buckets.push((sample, cumulative_count));
        }

        Some(Self {
            key,
            count,
            sum,
            buckets,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracing_subscriber::layer::SubscriberExt;

    const TEST_FLEET_KEY: &str = "fleet-a";

    fn with_fleet_span(fleet_key: &str, test_fn: impl FnOnce()) {
        let subscriber = tracing_subscriber::registry()
            .with(crate::telemetry::logs::TelemetryLogLayer::default());
        tracing::subscriber::with_default(subscriber, || {
            let span = tracing::info_span!("test.metric_descriptor", fleet_key = fleet_key);
            let _guard = span.enter();
            test_fn();
        });
    }

    fn describe_counter_for_fleet(
        recorder: &InMemoryMetricsRecorder,
        fleet_key: &str,
        name: &str,
        description: &str,
    ) {
        let name = name.to_string();
        let description = description.to_string();
        with_fleet_span(fleet_key, || {
            metrics::Recorder::describe_counter(
                recorder,
                name.into(),
                Some(metrics::Unit::Count),
                description.into(),
            );
        });
    }

    #[test]
    fn snapshot_collects_counter_gauge_and_histogram() {
        let recorder = InMemoryMetricsRecorder::new();
        let handle = recorder.handle();

        metrics::with_local_recorder(&recorder, || {
            metrics::counter!("fleet.counter", "kind" => "request", "fleet_key" => TEST_FLEET_KEY)
                .increment(3);
            metrics::gauge!("fleet.gauge", "kind" => "memory", "fleet_key" => TEST_FLEET_KEY)
                .set(12.5);
            metrics::histogram!("fleet.hist", "kind" => "latency", "fleet_key" => TEST_FLEET_KEY)
                .record(2.0);
            metrics::histogram!("fleet.hist", "kind" => "latency", "fleet_key" => TEST_FLEET_KEY)
                .record(4.0);
        });

        let batch = handle.snapshot(TEST_FLEET_KEY);

        let counter = batch
            .counters
            .iter()
            .find(|metric| metric.key.name == "fleet.counter")
            .expect("counter metric should exist");
        assert_eq!(counter.value, 3);

        let gauge = batch
            .gauges
            .iter()
            .find(|metric| metric.key.name == "fleet.gauge")
            .expect("gauge metric should exist");
        assert!((gauge.value - 12.5).abs() < f64::EPSILON);

        let histogram = batch
            .histograms
            .iter()
            .find(|metric| metric.key.name == "fleet.hist")
            .expect("histogram metric should exist");
        assert_eq!(histogram.count, 2);
        assert!((histogram.sum - 6.0).abs() < f64::EPSILON);
        assert_eq!(histogram.buckets, vec![(2.0, 1), (4.0, 2)]);
    }

    #[test]
    fn snapshot_drains_histogram_samples_between_batches() {
        let recorder = InMemoryMetricsRecorder::new();
        let handle = recorder.handle();

        metrics::with_local_recorder(&recorder, || {
            metrics::histogram!("fleet.hist.drain", "fleet_key" => TEST_FLEET_KEY).record(1.0);
            metrics::histogram!("fleet.hist.drain", "fleet_key" => TEST_FLEET_KEY).record(3.0);
        });

        let first = handle.snapshot(TEST_FLEET_KEY);
        let first_hist = first
            .histograms
            .iter()
            .find(|metric| metric.key.name == "fleet.hist.drain")
            .expect("first snapshot should contain histogram");
        assert_eq!(first_hist.count, 2);

        let second = handle.snapshot(TEST_FLEET_KEY);
        let second_hist = second
            .histograms
            .iter()
            .find(|metric| metric.key.name == "fleet.hist.drain");
        assert!(second_hist.is_none());
    }

    #[test]
    fn snapshot_emits_counter_deltas_and_gauge_changes_only() {
        let recorder = InMemoryMetricsRecorder::new();
        let handle = recorder.handle();

        metrics::with_local_recorder(&recorder, || {
            metrics::counter!(
                "fleet.counter.persist",
                "kind" => "request",
                "fleet_key" => TEST_FLEET_KEY
            )
            .increment(5);
            metrics::gauge!(
                "fleet.gauge.persist",
                "kind" => "memory",
                "fleet_key" => TEST_FLEET_KEY
            )
            .set(64.0);
        });

        let first = handle.snapshot(TEST_FLEET_KEY);
        let second = handle.snapshot(TEST_FLEET_KEY);

        let first_counter = first
            .counters
            .iter()
            .find(|metric| metric.key.name == "fleet.counter.persist")
            .expect("first snapshot should contain counter");
        assert_eq!(first_counter.value, 5);
        assert!(second.counters.is_empty());

        let first_gauge = first
            .gauges
            .iter()
            .find(|metric| metric.key.name == "fleet.gauge.persist")
            .expect("first snapshot should contain gauge");
        assert!((first_gauge.value - 64.0).abs() < f64::EPSILON);
        assert!(second.gauges.is_empty());

        metrics::with_local_recorder(&recorder, || {
            metrics::counter!(
                "fleet.counter.persist",
                "kind" => "request",
                "fleet_key" => TEST_FLEET_KEY
            )
            .increment(2);
            metrics::gauge!(
                "fleet.gauge.persist",
                "kind" => "memory",
                "fleet_key" => TEST_FLEET_KEY
            )
            .set(64.0);
        });

        let third = handle.snapshot(TEST_FLEET_KEY);
        let third_counter = third
            .counters
            .iter()
            .find(|metric| metric.key.name == "fleet.counter.persist")
            .expect("third snapshot should contain only counter delta");
        assert_eq!(third_counter.value, 2);
        assert!(third.gauges.is_empty());

        metrics::with_local_recorder(&recorder, || {
            metrics::gauge!(
                "fleet.gauge.persist",
                "kind" => "memory",
                "fleet_key" => TEST_FLEET_KEY
            )
            .set(96.0);
        });

        let fourth = handle.snapshot(TEST_FLEET_KEY);
        assert!(fourth.counters.is_empty());
        let fourth_gauge = fourth
            .gauges
            .iter()
            .find(|metric| metric.key.name == "fleet.gauge.persist")
            .expect("fourth snapshot should contain changed gauge");
        assert!((fourth_gauge.value - 96.0).abs() < f64::EPSILON);
    }

    #[test]
    fn descriptor_delta_collects_and_drains_descriptions() {
        let recorder = InMemoryMetricsRecorder::new();
        let handle = recorder.handle();

        with_fleet_span(TEST_FLEET_KEY, || {
            metrics::Recorder::describe_counter(
                &recorder,
                "fleet.requests.total".into(),
                Some(metrics::Unit::Count),
                "Total request count".into(),
            );
            metrics::Recorder::describe_histogram(
                &recorder,
                "fleet.request.duration".into(),
                Some(metrics::Unit::Milliseconds),
                "Request duration".into(),
            );
        });

        let first = handle.take_descriptor_delta(TEST_FLEET_KEY);
        assert_eq!(first.descriptors.len(), 2);
        assert!(first.descriptors.iter().any(|descriptor| {
            descriptor.name == "fleet.requests.total"
                && descriptor.kind == MetricDescriptorKind::Counter
                && descriptor.unit.as_deref() == Some("count")
                && descriptor.description == "Total request count"
        }));
        assert!(first.descriptors.iter().any(|descriptor| {
            descriptor.name == "fleet.request.duration"
                && descriptor.kind == MetricDescriptorKind::Histogram
                && descriptor.unit.as_deref() == Some("milliseconds")
                && descriptor.description == "Request duration"
        }));

        let second = handle.take_descriptor_delta(TEST_FLEET_KEY);
        assert!(second.descriptors.is_empty());
    }

    #[test]
    fn descriptor_delta_only_emits_changes() {
        let recorder = InMemoryMetricsRecorder::new();
        let handle = recorder.handle();

        with_fleet_span(TEST_FLEET_KEY, || {
            metrics::Recorder::describe_counter(
                &recorder,
                "fleet.requests.total".into(),
                Some(metrics::Unit::Count),
                "Total requests".into(),
            );
        });
        let _ = handle.take_descriptor_delta(TEST_FLEET_KEY);

        with_fleet_span(TEST_FLEET_KEY, || {
            metrics::Recorder::describe_counter(
                &recorder,
                "fleet.requests.total".into(),
                Some(metrics::Unit::Count),
                "Total requests".into(),
            );
        });
        let unchanged = handle.take_descriptor_delta(TEST_FLEET_KEY);
        assert!(unchanged.descriptors.is_empty());

        with_fleet_span(TEST_FLEET_KEY, || {
            metrics::Recorder::describe_counter(
                &recorder,
                "fleet.requests.total".into(),
                Some(metrics::Unit::Count),
                "Total requests seen".into(),
            );
        });
        let changed = handle.take_descriptor_delta(TEST_FLEET_KEY);
        assert_eq!(changed.descriptors.len(), 1);
        assert_eq!(changed.descriptors[0].description, "Total requests seen");
    }

    #[test]
    fn snapshot_isolated_by_fleet_key() {
        let recorder = InMemoryMetricsRecorder::new();
        let handle = recorder.handle();

        metrics::with_local_recorder(&recorder, || {
            metrics::counter!("fleet.counter.multi", "fleet_key" => "fleet-a").increment(3);
            metrics::counter!("fleet.counter.multi", "fleet_key" => "fleet-b").increment(7);
        });

        let fleet_a = handle.snapshot("fleet-a");
        let fleet_b = handle.snapshot("fleet-b");

        assert_eq!(fleet_a.counters.len(), 1);
        assert_eq!(fleet_a.counters[0].value, 3);
        assert_eq!(fleet_b.counters.len(), 1);
        assert_eq!(fleet_b.counters[0].value, 7);
    }

    #[test]
    fn descriptor_delta_isolated_by_fleet_key() {
        let recorder = InMemoryMetricsRecorder::new();
        let handle = recorder.handle();

        describe_counter_for_fleet(&recorder, "fleet-a", "fleet.requests.a", "Fleet A requests");
        describe_counter_for_fleet(&recorder, "fleet-b", "fleet.requests.b", "Fleet B requests");

        let fleet_a = handle.take_descriptor_delta("fleet-a");
        let fleet_b = handle.take_descriptor_delta("fleet-b");

        assert_eq!(fleet_a.descriptors.len(), 1);
        assert_eq!(fleet_a.descriptors[0].name, "fleet.requests.a");
        assert_eq!(fleet_b.descriptors.len(), 1);
        assert_eq!(fleet_b.descriptors[0].name, "fleet.requests.b");
    }

    #[test]
    fn descriptor_delta_ignores_descriptions_without_fleet_context() {
        let recorder = InMemoryMetricsRecorder::new();
        let handle = recorder.handle();

        metrics::Recorder::describe_counter(
            &recorder,
            "fleet.requests.total".into(),
            Some(metrics::Unit::Count),
            "Total requests".into(),
        );

        assert!(
            handle
                .take_descriptor_delta("fleet-a")
                .descriptors
                .is_empty()
        );
        assert!(
            handle
                .take_descriptor_delta("fleet-b")
                .descriptors
                .is_empty()
        );
    }
}
