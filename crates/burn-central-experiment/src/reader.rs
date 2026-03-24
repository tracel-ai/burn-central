use burn_central_artifact::bundle::FsBundle;

use crate::ExperimentId;

#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct ExperimentReaderError {
    pub message: String,
    #[source]
    pub source: Option<Box<dyn std::error::Error + Send + Sync>>,
}

impl ExperimentReaderError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            source: None,
        }
    }

    pub fn with_source<E>(message: impl Into<String>, source: E) -> Self
    where
        E: Into<Box<dyn std::error::Error + Send + Sync + 'static>>,
    {
        Self {
            message: message.into(),
            source: Some(source.into()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ArtifactRef {
    pub id: String,
    pub name: String,
}

pub struct LoadedArtifact {
    pub reference: ArtifactRef,
    pub bundle: FsBundle,
}

impl LoadedArtifact {
    pub fn new(reference: ArtifactRef, bundle: FsBundle) -> Self {
        Self { reference, bundle }
    }
}

pub trait ExperimentArtifactReader: Send + Sync {
    fn load_artifact_raw(
        &self,
        experiment_id: ExperimentId,
        name: &str,
    ) -> Result<LoadedArtifact, ExperimentReaderError>;
}
