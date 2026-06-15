use tracel_artifact::bundle::FsBundle;
use tracel_client::Client;

use tracel_experiment::{
    ArtifactKind, ExperimentId,
    reader::{ArtifactRef, ExperimentArtifactReader, ExperimentReaderError, LoadedArtifact},
};

use crate::experiment::remote::session::{ArtifactUploadError, ArtifactUploader};

use super::{ExperimentArtifactClient, ExperimentPath};

pub struct ConsoleArtifactReader {
    client: Client,
    exp_path: ExperimentPath,
}

impl ConsoleArtifactReader {
    pub fn new(client: Client, exp_path: ExperimentPath) -> Self {
        Self { client, exp_path }
    }
}

impl ExperimentArtifactReader for ConsoleArtifactReader {
    fn load_artifact_raw(
        &self,
        experiment_id: ExperimentId,
        name: &str,
    ) -> Result<LoadedArtifact, ExperimentReaderError> {
        let num = experiment_id
            .parse::<i32>()
            .ok_or_else(|| ExperimentReaderError::new("Invalid experiment ID format"))?;

        let experiment_path = ExperimentPath::new(
            self.exp_path.owner_name().to_string(),
            self.exp_path.project_name().to_string(),
            num,
        );
        let scope = ExperimentArtifactClient::new(self.client.clone(), experiment_path);
        let artifact = scope.fetch(name).map_err(|err| {
            ExperimentReaderError::with_source("Failed to resolve experiment artifact", err)
        })?;

        scope
            .download(name)
            .map_err(|err| {
                ExperimentReaderError::with_source("Failed to download experiment artifact", err)
            })
            .map(|bundle| {
                LoadedArtifact::new(
                    ArtifactRef {
                        id: artifact.id.to_string(),
                        name: name.to_string(),
                    },
                    bundle,
                )
            })
    }
}

pub struct ConsoleArtifactUploader {
    client: ExperimentArtifactClient,
}

impl ConsoleArtifactUploader {
    pub fn new(client: Client, exp_path: ExperimentPath) -> Self {
        Self {
            client: ExperimentArtifactClient::new(client, exp_path),
        }
    }
}

impl ArtifactUploader for ConsoleArtifactUploader {
    fn upload(
        &self,
        name: &str,
        kind: ArtifactKind,
        bundle: &FsBundle,
    ) -> Result<(), ArtifactUploadError> {
        self.client
            .upload(name, kind, bundle)
            .map(|_| ())
            .map_err(|e| ArtifactUploadError {
                message: format!("Failed to upload artifact '{}'", name),
                source: Some(Box::new(e)),
            })
    }
}
