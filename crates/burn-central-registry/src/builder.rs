use std::path::PathBuf;

use burn_central_client::{BurnCentralCredentials, Client, Env};
use directories::{BaseDirs, ProjectDirs};

use crate::error::RegistryError;
use crate::registry::Registry;

/// Builder for a registry client.
#[derive(Debug, Clone)]
pub struct RegistryBuilder {
    credentials: BurnCentralCredentials,
    env: Option<Env>,
    cache_dir: Option<PathBuf>,
}

impl RegistryBuilder {
    /// Create a new registry builder with the given credentials.
    pub fn new(credentials: impl Into<BurnCentralCredentials>) -> Self {
        Self {
            credentials: credentials.into(),
            env: None,
            cache_dir: None,
        }
    }

    /// Use a specific environment (production by default).
    pub fn with_env(mut self, env: Env) -> Self {
        self.env = Some(env);
        self
    }

    /// Build the registry client.
    pub fn build(self) -> Result<Registry, RegistryError> {
        let client = {
            let env = self.env.unwrap_or(Env::Production);
            Client::new(env, &self.credentials)?
        };

        let cache_dir = match self.cache_dir {
            Some(dir) => dir,
            None => default_cache_dir(client.get_env())?,
        };

        Ok(Registry::new(client, cache_dir))
    }
}

/// Get the default cache directory for a given environment.
fn default_cache_dir(env: &Env) -> Result<PathBuf, RegistryError> {
    let registry_subdir = match env {
        Env::Production => "registry".to_string(),
        Env::Staging(version) => format!("registry-staging-{}", version),
        Env::Development => "registry-dev".to_string(),
    };
    if let Some(project) = ProjectDirs::from("ai", "tracel", "burn-central") {
        return Ok(project.cache_dir().join(registry_subdir));
    }
    if let Some(base) = BaseDirs::new() {
        return Ok(base.cache_dir().join("burn-central").join(registry_subdir));
    }
    Err(RegistryError::CacheDirUnavailable)
}
