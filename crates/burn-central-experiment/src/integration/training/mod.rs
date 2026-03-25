//! Burn `train` integrations backed by [`crate::ExperimentRun`].
//!
//! These adapters let Burn learners write metrics and checkpoints into an experiment run and
//! react to experiment cancellation through Burn's [`burn::train::Interrupter`].
//!
//! Tracing log forwarding lives in the sibling [`super::tracing`] module.
//!
//! ```ignore
//! use burn::train::LearnerBuilder;
//! use burn_central::experiment::ExperimentRun;
//! use burn_central::experiment::integration::training::{
//!     ExperimentCheckpointRecorder,
//!     ExperimentMetricLogger,
//!     experiment_interrupter,
//! };
//!
//! let experiment: ExperimentRun = /* create a remote or local run */;
//!
//! LearnerBuilder::new("a_directory_of_your_choice")
//!     .with_metric_logger(ExperimentMetricLogger::new(&experiment))
//!     .with_file_checkpointer(ExperimentCheckpointRecorder::new(&experiment))
//!     .with_interrupter(experiment_interrupter(&experiment));
//! ```

mod checkpoint;
mod interrupter;
mod metric;

pub use checkpoint::ExperimentCheckpointRecorder;
pub use interrupter::experiment_interrupter;
pub use metric::ExperimentMetricLogger;
