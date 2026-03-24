use std::sync::Arc;

use burn_central_artifact::bundle::FsBundle;

use crate::{
    ArtifactKind, ExperimentId, MetricSpec, MetricValue, error::ExperimentError,
    reader::ArtifactRef,
};

#[derive(Debug, Clone)]
pub enum Event {
    Args(serde_json::Value),
    Config {
        name: String,
        value: serde_json::Value,
    },
    Log {
        message: String,
    },
    Metrics {
        epoch: usize,
        split: String,
        iteration: usize,
        items: Vec<MetricValue>,
    },
    MetricDefinition(MetricSpec),
    EpochSummary {
        epoch: usize,
        split: String,
        items: Vec<MetricValue>,
    },
    ArtifactUsed {
        experiment_id: ExperimentId,
        reference: ArtifactRef,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExperimentCompletion {
    Success,
    Failed(String),
    Cancelled,
}

pub type BundleFn<'a> = dyn FnOnce(&mut FsBundle) -> Result<(), ExperimentError> + 'a;

/// Backend-specific implementation for the active experiment run.
pub trait ExperimentSession: Send + Sync {
    fn record_event(&self, event: Event) -> Result<(), ExperimentError>;
    fn save_artifact(
        &self,
        name: &str,
        kind: ArtifactKind,
        artifact: Box<BundleFn>,
    ) -> Result<(), ExperimentError>;
    fn finish(&self, completion: ExperimentCompletion) -> Result<(), ExperimentError>;
}

impl<T> ExperimentSession for Arc<T>
where
    T: ExperimentSession,
{
    fn record_event(&self, event: Event) -> Result<(), ExperimentError> {
        self.as_ref().record_event(event)
    }

    fn save_artifact(
        &self,
        name: &str,
        kind: ArtifactKind,
        artifact: Box<BundleFn>,
    ) -> Result<(), ExperimentError> {
        self.as_ref().save_artifact(name, kind, artifact)
    }

    fn finish(&self, completion: ExperimentCompletion) -> Result<(), ExperimentError> {
        self.as_ref().finish(completion)
    }
}
