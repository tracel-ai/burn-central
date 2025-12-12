use std::collections::BTreeMap;

use crate::bundle::{BundleDecode, InMemoryBundleReader};
use crate::schemas::ModelPath;
use burn_central_client::response::{ModelResponse, ModelVersionResponse};
use burn_central_client::{Client, ClientError};

/// An interface for downloading models from Burn Central.
#[derive(Clone)]
pub struct ModelRegistry {
    client: Client,
}

impl ModelRegistry {
    pub fn new(client: Client) -> Self {
        Self { client }
    }

    /// Create a scope for a specific model within a project.
    pub fn model(&self, model_path: ModelPath) -> Result<ModelClient, ModelError> {
        let response = self
            .client
            .get_model(
                model_path.namespace(),
                model_path.project_name(),
                model_path.model_name(),
            )
            .map_err(|e| {
                if matches!(e, ClientError::NotFound) {
                    ModelError::NotFound(format!("Model not found: {}", model_path))
                } else {
                    ModelError::Client(e)
                }
            })?;

        let model = response.into();

        Ok(ModelClient::new(self.client.clone(), model_path, model))
    }

    /// Download a specific model version and decode it using the BundleDecode trait.
    pub fn download<T: BundleDecode>(
        &self,
        model_path: ModelPath,
        version: u32,
        settings: &T::Settings,
    ) -> Result<T, ModelError> {
        let scope = self.model(model_path)?;
        scope.download(version, settings)
    }

    /// Download a specific model version as a memory reader for dynamic access.
    pub fn download_raw(
        &self,
        model_path: ModelPath,
        version: u32,
    ) -> Result<InMemoryBundleReader, ModelError> {
        let scope = self.model(model_path)?;
        scope.download_raw(version)
    }
}

/// A scope for operations on a specific model within a project.
#[derive(Clone)]
pub struct ModelClient {
    client: Client,
    model_path: ModelPath,
    model: ModelResponse,
}

impl ModelClient {
    pub(crate) fn new(client: Client, model_path: ModelPath, model: ModelResponse) -> Self {
        Self {
            client,
            model_path,
            model,
        }
    }

    /// Download a specific version of this model and decode it using the BundleDecode trait.
    /// This allows reusing existing bundle decoders for models.
    pub fn download<T: BundleDecode>(
        &self,
        version: u32,
        settings: &T::Settings,
    ) -> Result<T, ModelError> {
        let reader = self.download_raw(version)?;
        T::decode(&reader, settings).map_err(|e| {
            ModelError::Decode(format!(
                "Failed to decode model {}: {}",
                self.model_path,
                e.into()
            ))
        })
    }

    /// Download a specific version of this model as a memory reader for dynamic access.
    pub fn download_raw(&self, version: u32) -> Result<InMemoryBundleReader, ModelError> {
        let resp = self
            .client
            .presign_model_download(
                self.model_path.namespace(),
                self.model_path.project_name(),
                self.model_path.model_name(),
                version,
            )
            .map_err(|e| {
                if matches!(e, ClientError::NotFound) {
                    ModelError::VersionNotFound(format!("{} v{}", self.model_path, version))
                } else {
                    ModelError::Client(e)
                }
            })?;

        let mut data = BTreeMap::new();

        for file in resp.files {
            let bytes = self.client.download_bytes_from_url(&file.url)?;
            data.insert(file.rel_path, bytes);
        }

        Ok(InMemoryBundleReader::new(data))
    }

    /// Get information about a specific model version.
    pub fn fetch(&self, version: u32) -> Result<ModelVersionResponse, ModelError> {
        let resp = self
            .client
            .get_model_version(
                self.model_path.namespace(),
                self.model_path.project_name(),
                self.model_path.model_name(),
                version,
            )
            .map_err(|e| {
                if matches!(e, ClientError::NotFound) {
                    ModelError::VersionNotFound(format!("{} v{}", self.model_path, version))
                } else {
                    ModelError::Client(e)
                }
            })?;

        Ok(resp.into())
    }

    /// Get the total number of versions available for this model.
    pub fn total_versions(&self) -> u64 {
        self.model.version_count
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("Client error: {0}")]
    Client(#[from] ClientError),
    #[error("Decode error: {0}")]
    Decode(String),
    #[error("Model not found: {0}")]
    NotFound(String),
    #[error("Model version not found: {0}")]
    VersionNotFound(String),
}
