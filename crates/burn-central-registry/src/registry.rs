use std::path::{Path, PathBuf};

use burn_central_client::Client;
use reqwest::blocking::Client as HttpClient;

use crate::error::RegistryError;
use crate::model::{ModelHandle, ModelRef};

/// Registry client for downloading and caching model artifacts.
#[derive(Clone)]
pub struct Registry {
    client: Client,
    http: HttpClient,
    cache_dir: PathBuf,
}

impl Registry {
    /// Create a registry client from a Burn Central HTTP client and config.
    pub fn new(client: Client, cache_dir: PathBuf) -> Self {
        Self {
            client,
            http: HttpClient::new(),
            cache_dir,
        }
    }

    /// Create a model handle scoped to a project.
    pub fn model(
        &self,
        namespace: impl Into<String>,
        project: impl Into<String>,
        model: impl Into<String>,
    ) -> Result<ModelHandle, RegistryError> {
        let model = ModelRef::new(namespace, project, model)?;
        Ok(ModelHandle::new(self.clone(), model))
    }

    pub fn client(&self) -> &Client {
        &self.client
    }

    pub fn http(&self) -> &HttpClient {
        &self.http
    }

    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }
}
