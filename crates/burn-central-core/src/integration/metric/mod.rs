use std::collections::HashMap;
use std::sync::Arc;

use burn::train::logger::MetricLogger;
use burn::train::metric::store::{EpochSummary, MetricsUpdate, NumericMetricUpdate, Split};
use burn::train::metric::{MetricAttributes, MetricDefinition, MetricId, NumericEntry};
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

    fn get_definitions_from_entries(
        &self,
        entries: &[NumericMetricUpdate],
    ) -> Vec<MetricDefinition> {
        entries
            .iter()
            .filter_map(|entry| self.metric_definitions.get(&entry.entry.metric_id).cloned())
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

        let mut logs = vec![];
        let mut summaries = vec![];
        let definitions = self.get_definitions_from_entries(&update.entries_numeric);
        for (i, definition) in definitions.iter().enumerate() {
            let NumericMetricUpdate {
                entry: _,
                numeric_entry,
                running_entry,
            } = update
                .entries_numeric
                .get(i)
                .expect("Definition without numeric entry");

            let get_value_from_entry = |v: &NumericEntry| match *v {
                NumericEntry::Value(v) => v,
                NumericEntry::Aggregated {
                    aggregated_value, ..
                } => aggregated_value,
            };
            let value = get_value_from_entry(numeric_entry);
            let running_value = get_value_from_entry(running_entry);

            logs.push(MetricLog {
                name: definition.name.clone(),
                value,
            });

            summaries.push(MetricLog {
                name: definition.name.clone(),
                value: running_value,
            });
        }
        self.experiment_handle
            .log_metric(epoch, split.to_string(), self.iteration_count, logs);
        _ = self
            .experiment_handle
            .log_epoch_summary(epoch, split.to_string(), summaries);
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
