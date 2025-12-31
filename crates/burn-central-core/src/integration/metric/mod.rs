use std::collections::HashMap;
use std::sync::Arc;

use burn::train::logger::MetricLogger;
use burn::train::metric::store::{EpochSummary, MetricsUpdate, Split};
use burn::train::metric::{
    MetricAttributes, MetricDefinition, MetricEntry, MetricId, NumericEntry,
};
use burn_central_client::websocket::MetricLog;

use crate::experiment::{ExperimentRun, ExperimentRunHandle};

/// Remote implementation for burn `MetricLogger` trait.
pub struct RemoteMetricLogger {
    experiment_handle: ExperimentRunHandle,
    metric_definitions: HashMap<MetricId, MetricDefinition>,
    iteration_count: usize,
}

impl RemoteMetricLogger {
    pub fn new(experiment: &ExperimentRun) -> Self {
        Self {
            experiment_handle: experiment.handle(),
            metric_definitions: HashMap::default(),
            iteration_count: 0,
        }
    }

    fn get_logs_from_entries(&self, entries: &[MetricEntry]) -> Vec<MetricLog> {
        entries
            .iter()
            .filter_map(|entry| {
                let name = self
                    .metric_definitions
                    .get(&entry.metric_id)
                    .unwrap()
                    .name
                    .clone();
                let numeric_entry: NumericEntry =
                    match NumericEntry::deserialize(&entry.serialized_entry.serialized) {
                        Ok(e) => e,
                        Err(_) => return None,
                    };
                let value = match numeric_entry {
                    NumericEntry::Value(v) => v,
                    NumericEntry::Aggregated {
                        aggregated_value, ..
                    } => aggregated_value,
                };
                Some(MetricLog { name, value })
            })
            .collect()
    }
}

impl MetricLogger for RemoteMetricLogger {
    fn log(
        &mut self,
        update: MetricsUpdate,
        epoch: usize,
        split: Split,
        _tag: Option<Arc<String>>,
    ) {
        self.iteration_count += 1;

        let entries: Vec<_> = update
            .entries
            .iter()
            .chain(
                update
                    .entries_numeric
                    .iter()
                    .map(|numeric_update| &numeric_update.entry),
            )
            .cloned()
            .collect();

        let item_logs: Vec<MetricLog> = self.get_logs_from_entries(&entries);
        if item_logs.is_empty() {
            return;
        };

        // send to server
        self.experiment_handle.log_metric(
            epoch,
            split.to_string(),
            self.iteration_count,
            item_logs,
        );
    }

    /// Read the logs for an epoch.
    fn read_numeric(
        &mut self,
        _name: &str,
        _epoch: usize,
        _split: Split,
    ) -> Result<Vec<NumericEntry>, String> {
        Ok(vec![]) // Not implemented
    }

    fn log_metric_definition(&mut self, definition: burn::train::metric::MetricDefinition) {
        self.metric_definitions
            .insert(definition.metric_id.clone(), definition.clone());

        let (unit, higher_is_better) = match &definition.attributes {
            MetricAttributes::Numeric(attr) => (attr.unit.clone(), attr.higher_is_better),
            MetricAttributes::None => return,
        };

        match self.experiment_handle.log_metric_definition(
            definition.name,
            definition.description,
            unit,
            higher_is_better,
        ) {
            Ok(_) => (),
            Err(e) => panic!("{e}"),
        }
    }

    fn log_epoch_summary(&mut self, _summary: EpochSummary) {}
}
