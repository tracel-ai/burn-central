use crate::burn::train::metric::{MetricEntry, NumericEntry};

use crate::experiment::{ExperimentRun, ExperimentRunHandle};

/// Stub implementation of RemoteMetricLogger for Burn 0.19
///
/// This is a no-op implementation that satisfies the trait bounds but doesn't
/// perform any actual metric logging to Burn Central.
///
/// Burn 0.19 support is provided for compatibility purposes only. For full
/// functionality, please use Burn 0.20 or later.
pub struct RemoteMetricLogger {
    _experiment_handle: ExperimentRunHandle,
}

impl RemoteMetricLogger {
    /// Create a new instance of the remote metric logger (stub for Burn 0.19)
    pub fn new(experiment: &ExperimentRun) -> Self {
        Self {
            _experiment_handle: experiment.handle(),
        }
    }
}

impl burn_0_19::train::logger::MetricLogger for RemoteMetricLogger {
    fn log(&mut self, _entry: &MetricEntry) {
        // No-op: Burn 0.19 stub implementation
    }

    fn end_epoch(&mut self, _epoch: usize) {
        // No-op: Burn 0.19 stub implementation
    }

    fn read_numeric(&mut self, _name: &str, _epoch: usize) -> Result<Vec<NumericEntry>, String> {
        Ok(vec![])
    }
}
