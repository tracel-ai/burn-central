use std::collections::HashMap;

use burn_0_19::train::{
    logger::MetricLogger,
    metric::{MetricEntry, NumericEntry},
};
use burn_central_client::websocket::MetricLog;

use crate::experiment::{ExperimentRun, ExperimentRunHandle};

pub struct RemoteMetricLogger {
    experiment_handle: ExperimentRunHandle,

    // Metric definition tracking
    metric_definitions: HashMap<String, bool>, // name -> has_been_logged

    metric_buffer: Vec<burn_0_19::train::metric::MetricEntry>,
    current_epoch: usize,
    iteration_count: usize,
}

impl RemoteMetricLogger {
    pub fn new(experiment: &ExperimentRun) -> Self {
        Self {
            experiment_handle: experiment.handle(),
            metric_definitions: HashMap::new(),
            metric_buffer: Vec::new(),
            current_epoch: 0,
            iteration_count: 0,
        }
    }

    /// Log metric definition if this is the first time seeing this metric
    fn log_metric_definition_if_needed(&mut self, name: &str) {
        if self.metric_definitions.contains_key(name) {
            return;
        }

        match self
            .experiment_handle
            .log_metric_definition(name.to_string(), None, None, true)
        {
            Ok(_) => {
                self.metric_definitions.insert(name.to_string(), true);
            }
            Err(e) => panic!("{e}"),
        }
    }

    /// Extract numeric value from serialized metric entry
    fn extract_numeric_value(serialize: &str) -> Option<f64> {
        match NumericEntry::deserialize(serialize) {
            Ok(entry) => match entry {
                NumericEntry::Value(v) => Some(v),
                NumericEntry::Aggregated { sum, .. } => Some(sum),
            },
            Err(_) => None,
        }
    }

    /// Convert metric entries to MetricLog format
    fn get_logs_from_entries(
        &self,
        entries: &[burn_0_19::train::metric::MetricEntry],
    ) -> Vec<MetricLog> {
        entries
            .iter()
            .filter_map(|entry| {
                let name = entry.name.as_ref().clone();
                let value = Self::extract_numeric_value(&entry.serialize)?;
                Some(MetricLog { name, value })
            })
            .collect()
    }

    /// Flush the current buffer of metrics to the server
    fn flush_buffer(&mut self) {
        if self.metric_buffer.is_empty() {
            return;
        }

        let item_logs = self.get_logs_from_entries(&self.metric_buffer);

        if item_logs.is_empty() {
            self.metric_buffer.clear();
            return;
        }

        self.iteration_count += 1;

        // Send to server with default split "train" (0.19 doesn't have Split concept)
        self.experiment_handle.log_metric(
            self.current_epoch,
            "train".to_string(),
            self.iteration_count,
            item_logs,
        );

        self.metric_buffer.clear();
    }

    /// Check if a metric with the given name already exists in the buffer
    fn metric_exists_in_buffer(&self, name: &str) -> bool {
        self.metric_buffer
            .iter()
            .any(|entry| entry.name.as_ref() == name)
    }
}

impl MetricLogger for RemoteMetricLogger {
    fn log(&mut self, entry: &MetricEntry) {
        let name = entry.name.as_ref();

        self.log_metric_definition_if_needed(name);

        // Check if this metric already exists in the current buffer
        // If yes, we've completed an iteration - flush before adding new metric
        if self.metric_exists_in_buffer(name) {
            self.flush_buffer();
        }

        self.metric_buffer.push(entry.clone());
    }

    fn end_epoch(&mut self, epoch: usize) {
        self.flush_buffer();

        self.current_epoch = epoch;
    }

    fn read_numeric(
        &mut self,
        _name: &str,
        _epoch: usize,
    ) -> Result<Vec<burn_0_19::train::metric::NumericEntry>, std::string::String> {
        Ok(vec![]) // Not implemented
    }
}
