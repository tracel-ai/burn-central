//! # Burn implementations for integrating  with Burn Central
//!
//! Burn Central toolings are meant to be injected in learner instead of the basic burn define
//! toolings. They must be manually override in the learner builder like so:
//!
//! ```ignore
//! use burn::train::LearnerBuilder;
//! let client: BurnCentral;
//!
//! LearnerBuilder::new("a_directory_of_your_choice")
//!   .with_metric_logger(RemoteMetricLogger::new(client))
//!   .with_file_checkpointer(RemoteCheckpointRecorder::new(client))
//!```
//! While the default burn implementation write file to disk. Our remote toolings will send the data
//! directly to Burn Central server through API call.
//!

mod checkpoint;
mod metric;

pub use checkpoint::RemoteCheckpointRecorder;
pub use metric::RemoteMetricLogger;
