use std::path::PathBuf;

use burn_central_client::fleet::FleetClient;
use burn_central_core::Env;
use directories::{BaseDirs, ProjectDirs};

use crate::inference::ModelSource;

mod model;
mod state;

pub type FleetRegistrationToken = String;

pub type DeviceMetadata = serde_json::Value;

#[derive(Debug, thiserror::Error)]
pub enum FleetError {
    #[error("fleet sync failed: {0}")]
    SyncFailed(String),
    #[error("fleet model download failed: {0}")]
    DownloadFailed(String),
    #[error("failed to determine cache directory")]
    CacheDirUnavailable,
    #[error(transparent)]
    State(#[from] state::FleetStateStoreError),
    #[error(transparent)]
    Model(#[from] model::ModelCacheError),
}

pub struct FleetDeviceSession {
    registration_token: FleetRegistrationToken,
    identity_key: String,
    state: state::FleetState,
    client: FleetClient,
    fleet_key: String,
    store: state::FleetLocalStateStore,
}

pub fn register(
    token: impl Into<FleetRegistrationToken>,
    metadata: DeviceMetadata,
    env: &Env,
) -> anyhow::Result<FleetDeviceSession> {
    let root_dir = default_data_dir(env)?;
    let registration_token = token.into();
    let fleet_key = state::fleet_key_from_registration_token(&registration_token);

    let client = FleetClient::new(env_to_client_env(env));
    let store = state::FleetLocalStateStore::new(root_dir);

    let identity_key = store.load_or_create_machine_identity_key()?;
    let state = store.load_fleet_state(&fleet_key)?.unwrap_or_default();
    let mut fleet_device = FleetDeviceSession::new(
        registration_token,
        identity_key,
        state,
        client,
        fleet_key,
        store,
    );
    if let Err(err) = fleet_device.sync(Some(metadata)) {
        tracing::warn!(%err, "initial fleet sync failed, continuing with local cache");
    }

    Ok(fleet_device)
}

impl FleetDeviceSession {
    fn new(
        registration_token: FleetRegistrationToken,
        identity_key: String,
        state: state::FleetState,
        client: FleetClient,
        fleet_key: String,
        store: state::FleetLocalStateStore,
    ) -> Self {
        Self {
            registration_token,
            identity_key,
            state,
            client,
            fleet_key,
            store,
        }
    }

    pub fn active_model_version_id(&self) -> &str {
        &self.state.active_model_version_id()
    }

    pub fn model_source(&self) -> Result<ModelSource, FleetError> {
        model::load_cached_model_source(
            &self.store.models_dir(&self.fleet_key),
            &self.state.active_model_version_id(),
        )
        .map_err(FleetError::from)
    }

    pub fn runtime_config(&self) -> &serde_json::Value {
        self.state.runtime_config()
    }

    pub fn sync(&mut self, metadata: Option<DeviceMetadata>) -> Result<(), FleetError> {
        tracing::info!(
            ?metadata,
            "syncing fleet device with fleet management service"
        );
        let snapshot = self
            .client
            .sync(
                self.registration_token.clone(),
                self.identity_key.clone(),
                metadata,
            )
            .map_err(|e| FleetError::SyncFailed(e.to_string()))?;

        let download = self
            .client
            .model_download(self.registration_token.clone(), self.identity_key.clone())
            .map_err(|e| FleetError::DownloadFailed(e.to_string()))?;

        model::ensure_cached_model(
            &self.store.models_dir(&self.fleet_key),
            &snapshot.model_version_id,
            &download,
        )?;

        self.state
            .update(snapshot.model_version_id, snapshot.runtime_config);

        self.persist_state()
    }

    fn persist_state(&self) -> Result<(), FleetError> {
        self.store
            .save_fleet_state(&self.fleet_key, &self.state)
            .map_err(FleetError::from)
    }
}

/// Get the default cache directory for a given environment.
fn default_data_dir(env: &Env) -> Result<PathBuf, FleetError> {
    let fleets_subdir = match env {
        Env::Production => "fleets".to_string(),
        Env::Staging(version) => format!("fleets-staging-{version}"),
        Env::Development => "fleets-dev".to_string(),
    };
    if let Some(project) = ProjectDirs::from("ai", "tracel", "burn-central") {
        return Ok(project.data_dir().join(fleets_subdir));
    }
    if let Some(base) = BaseDirs::new() {
        return Ok(base.data_dir().join("burn-central").join(fleets_subdir));
    }
    Err(FleetError::CacheDirUnavailable)
}

fn env_to_client_env(env: &Env) -> burn_central_client::Env {
    match env {
        Env::Production => burn_central_client::Env::Production,
        Env::Staging(v) => burn_central_client::Env::Staging(*v),
        Env::Development => burn_central_client::Env::Development,
    }
}
