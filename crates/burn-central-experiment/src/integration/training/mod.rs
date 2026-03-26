//! Burn `train` adapters backed by an [`crate::ExperimentRun`].
//!
//! These adapters let learners emit metrics, write checkpoints, and respond to experiment
//! cancellation without each training loop needing to know about the underlying experiment
//! backend.
//!
//! Import [`ExperimentTrainingExt`] for the ergonomic constructors, or use the concrete adapter
//! types directly.
//!
//! # Example
//!
//! ```no_run
//! use burn_central_experiment::ExperimentRun;
//! use burn_central_experiment::integration::training::ExperimentTrainingExt;
//!
//! let experiment = ExperimentRun::local("./runs").unwrap();
//!
//! let _metrics = experiment.metric_logger();
//! let _checkpoints = experiment.checkpoint_recorder();
//! let _interrupter = experiment.interrupter();
//! ```

mod checkpoint;
mod interrupter;
mod metric;

pub use checkpoint::ExperimentCheckpointRecorder;
pub use interrupter::experiment_interrupter;
pub use metric::ExperimentMetricLogger;

use crate::ExperimentRun;

/// Extension trait adding Burn `train` adapter constructors to [`ExperimentRun`].
pub trait ExperimentTrainingExt {
    /// Create a new [`ExperimentMetricLogger`] for this run.
    fn metric_logger(&self) -> ExperimentMetricLogger;

    /// Create a new [`ExperimentCheckpointRecorder`] for this run.
    fn checkpoint_recorder(&self) -> ExperimentCheckpointRecorder;

    /// Create a new [`burn::train::Interrupter`] linked to this run's cancellation token.
    fn interrupter(&self) -> burn::train::Interrupter;
}

impl ExperimentTrainingExt for ExperimentRun {
    fn metric_logger(&self) -> ExperimentMetricLogger {
        ExperimentMetricLogger::new(self)
    }

    fn checkpoint_recorder(&self) -> ExperimentCheckpointRecorder {
        ExperimentCheckpointRecorder::new(self)
    }

    fn interrupter(&self) -> burn::train::Interrupter {
        experiment_interrupter(self)
    }
}

impl ExperimentTrainingExt for crate::ExperimentRunHandle {
    fn metric_logger(&self) -> ExperimentMetricLogger {
        ExperimentMetricLogger::new(self.clone())
    }

    fn checkpoint_recorder(&self) -> ExperimentCheckpointRecorder {
        ExperimentCheckpointRecorder::new(self.clone())
    }

    fn interrupter(&self) -> burn::train::Interrupter {
        experiment_interrupter(self.clone())
    }
}
