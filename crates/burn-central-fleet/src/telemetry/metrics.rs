use serde::{Deserialize, Serialize};

use std::sync::Arc;
use std::sync::atomic::Ordering;

#[derive(Debug)]
struct InnerRegistry {
    registry: metrics_util::registry::Registry<metrics::Key, metrics_util::registry::AtomicStorage>,
}

impl InnerRegistry {
    fn new() -> Self {
        Self {
            registry: metrics_util::registry::Registry::new(metrics_util::registry::AtomicStorage),
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

    fn snapshot(&self) -> MetricBatch {
        let mut counters = Vec::new();
        self.registry.visit_counters(|key, counter| {
            counters.push(MetricCounter {
                key: MetricKey::from_key(key),
                value: counter.load(Ordering::Acquire),
            });
        });
        counters.sort_by(|a, b| a.key.cmp(&b.key));

        let mut gauges = Vec::new();
        self.registry.visit_gauges(|key, gauge| {
            let value = f64::from_bits(gauge.load(Ordering::Acquire));
            if value.is_finite() {
                gauges.push(MetricGauge {
                    key: MetricKey::from_key(key),
                    value,
                });
            }
        });
        gauges.sort_by(|a, b| a.key.cmp(&b.key));

        let mut histograms = Vec::new();
        self.registry.visit_histograms(|key, histogram| {
            let mut samples = Vec::new();
            histogram.clear_with(|chunk| samples.extend_from_slice(chunk));

            if let Some(summary) = MetricHistogram::from_samples(MetricKey::from_key(key), samples)
            {
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
}

#[derive(Debug, Clone)]
pub struct RecorderHandle {
    registry: Arc<InnerRegistry>,
}

impl RecorderHandle {
    /// Produces a snapshot of recorded metrics at the current point in time.
    pub fn snapshot(&self) -> MetricBatch {
        self.registry.snapshot()
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
        _key: metrics::KeyName,
        _unit: Option<metrics::Unit>,
        _description: metrics::SharedString,
    ) {
    }
    fn describe_gauge(
        &self,
        _key: metrics::KeyName,
        _unit: Option<metrics::Unit>,
        _description: metrics::SharedString,
    ) {
    }
    fn describe_histogram(
        &self,
        _key: metrics::KeyName,
        _unit: Option<metrics::Unit>,
        _description: metrics::SharedString,
    ) {
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
    pub min: f64,
    pub max: f64,
    pub mean: f64,
}

impl MetricHistogram {
    fn from_samples(key: MetricKey, samples: Vec<f64>) -> Option<Self> {
        let mut count = 0u64;
        let mut sum = 0.0;
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;

        for sample in samples {
            if !sample.is_finite() {
                continue;
            }

            count += 1;
            sum += sample;
            min = min.min(sample);
            max = max.max(sample);
        }

        if count == 0 {
            return None;
        }

        Some(Self {
            key,
            count,
            sum,
            min,
            max,
            mean: sum / count as f64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_collects_counter_gauge_and_histogram() {
        let recorder = InMemoryMetricsRecorder::new();
        let handle = recorder.handle();

        metrics::with_local_recorder(&recorder, || {
            metrics::counter!("fleet.counter", "kind" => "request").increment(3);
            metrics::gauge!("fleet.gauge", "kind" => "memory").set(12.5);
            metrics::histogram!("fleet.hist", "kind" => "latency").record(2.0);
            metrics::histogram!("fleet.hist", "kind" => "latency").record(4.0);
        });

        let batch = handle.snapshot();

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
        assert!((histogram.min - 2.0).abs() < f64::EPSILON);
        assert!((histogram.max - 4.0).abs() < f64::EPSILON);
        assert!((histogram.mean - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn snapshot_drains_histogram_samples_between_batches() {
        let recorder = InMemoryMetricsRecorder::new();
        let handle = recorder.handle();

        metrics::with_local_recorder(&recorder, || {
            metrics::histogram!("fleet.hist.drain").record(1.0);
            metrics::histogram!("fleet.hist.drain").record(3.0);
        });

        let first = handle.snapshot();
        let first_hist = first
            .histograms
            .iter()
            .find(|metric| metric.key.name == "fleet.hist.drain")
            .expect("first snapshot should contain histogram");
        assert_eq!(first_hist.count, 2);

        let second = handle.snapshot();
        let second_hist = second
            .histograms
            .iter()
            .find(|metric| metric.key.name == "fleet.hist.drain");
        assert!(second_hist.is_none());
    }

    #[test]
    fn snapshot_keeps_counter_and_gauge_values_between_batches() {
        let recorder = InMemoryMetricsRecorder::new();
        let handle = recorder.handle();

        metrics::with_local_recorder(&recorder, || {
            metrics::counter!("fleet.counter.persist", "kind" => "request").increment(5);
            metrics::gauge!("fleet.gauge.persist", "kind" => "memory").set(64.0);
        });

        let first = handle.snapshot();
        let second = handle.snapshot();

        let first_counter = first
            .counters
            .iter()
            .find(|metric| metric.key.name == "fleet.counter.persist")
            .expect("first snapshot should contain counter");
        let second_counter = second
            .counters
            .iter()
            .find(|metric| metric.key.name == "fleet.counter.persist")
            .expect("second snapshot should still contain counter");
        assert_eq!(first_counter.value, 5);
        assert_eq!(second_counter.value, 5);

        let first_gauge = first
            .gauges
            .iter()
            .find(|metric| metric.key.name == "fleet.gauge.persist")
            .expect("first snapshot should contain gauge");
        let second_gauge = second
            .gauges
            .iter()
            .find(|metric| metric.key.name == "fleet.gauge.persist")
            .expect("second snapshot should still contain gauge");
        assert!((first_gauge.value - 64.0).abs() < f64::EPSILON);
        assert!((second_gauge.value - 64.0).abs() < f64::EPSILON);
    }
}
