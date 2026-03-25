//! Integrations backed by [`crate::ExperimentRun`].
//!
//! Use [`training`] for Burn `train` integrations such as metric logging, checkpoint recording,
//! and cancellation-aware learner interruption.
//!
//! Use [`tracing`] to forward `tracing` events to the current experiment.

pub mod tracing;
pub mod training;
