//! Burn learner integrations backed by [`crate::ExperimentRun`].
//!
//! These integrations plug an experiment run into Burn's learner APIs for metrics, checkpoints,
//! and cancellation.
//!
//! ```ignore
//! use burn::train::LearnerBuilder;
//! use burn_central::experiment::ExperimentRun;
//! use burn_central::experiment::integration::{
//!     RemoteCheckpointRecorder,
//!     RemoteMetricLogger,
//!     remote_interrupter,
//! };
//!
//! let experiment: ExperimentRun = /* create a remote or local run */;
//!
//! LearnerBuilder::new("a_directory_of_your_choice")
//!     .with_metric_logger(RemoteMetricLogger::new(&experiment))
//!     .with_file_checkpointer(RemoteCheckpointRecorder::new(&experiment))
//!     .with_interrupter(remote_interrupter(&experiment));
//! ```
//!
//! The type names still use the historical `Remote*` naming, but they operate on the generic
//! experiment run abstraction.

mod checkpoint;
mod interrupter;
mod metric;

pub use checkpoint::RemoteCheckpointRecorder;
pub use interrupter::remote_interrupter;
pub use metric::RemoteMetricLogger;
