use std::{path::PathBuf, sync::Arc};

use burn_central_client::{Env, FleetClient};
use directories::{BaseDirs, ProjectDirs};

use crate::{
    DeviceMetadata, FleetRegistrationToken, auth_client::AuthenticatedFleetClient,
    error::FleetError, model, state, telemetry::TelemetryPipeline,
};

pub struct FleetDeviceSession {
    registration_token: FleetRegistrationToken,
    identity_key: String,
    state: state::FleetState,
    client: AuthenticatedFleetClient,
    fleet_key: String,
    store: state::FleetLocalStateStore,
    pending_bootstrap_metadata: Option<DeviceMetadata>,
    _telemetry: Arc<TelemetryPipeline>,
}

impl FleetDeviceSession {
    pub fn init(
        token: impl Into<FleetRegistrationToken>,
        metadata: DeviceMetadata,
        env: &Env,
    ) -> Result<Self, FleetError> {
        let root_dir = default_data_dir(env)?;
        let registration_token = token.into();
        let fleet_key = state::fleet_key_from_registration_token(&registration_token);

        tracing::info!(
            fleet_key = %fleet_key,
            "registering fleet device session with fleet management service"
        );

        let store = state::FleetLocalStateStore::new(root_dir.clone());
        let identity_key = store.load_or_create_machine_identity_key()?;
        let mut state = store.load_fleet_state(&fleet_key)?.unwrap_or_default();
        if state.auth_token().is_some_and(|auth| !auth.is_valid()) {
            tracing::info!(
                fleet_key = %fleet_key,
                "found expired fleet auth token in local state, clearing it"
            );
            state.clear_auth_token();
            store.save_fleet_state(&fleet_key, &state)?;
        }
        let (client, telemetry) = if let Some(telemetry) = TelemetryPipeline::get(&fleet_key) {
            tracing::debug!(
                fleet_key = %fleet_key,
                "reusing existing telemetry pipeline and shared fleet auth client"
            );
            (telemetry.client(), telemetry)
        } else {
            let client = AuthenticatedFleetClient::new(
                FleetClient::new(env.clone()),
                state.auth_token().cloned(),
            );
            let telemetry = TelemetryPipeline::create(fleet_key.clone(), client.clone(), root_dir)?;
            (client, telemetry)
        };

        let fleet_device = FleetDeviceSession {
            registration_token,
            identity_key,
            state,
            client,
            fleet_key,
            store,
            pending_bootstrap_metadata: Some(metadata),
            _telemetry: telemetry,
        };

        Ok(fleet_device)
    }

    pub fn active_model_version_id(&self) -> &str {
        self.state.active_model_version_id()
    }

    pub fn fleet_key(&self) -> &str {
        &self.fleet_key
    }

    pub fn model_source(&self) -> Result<burn_central_artifact::bundle::FsBundle, FleetError> {
        model::load_cached_model_source(
            &self.store.models_dir(&self.fleet_key),
            self.state.active_model_version_id(),
        )
        .map_err(FleetError::from)
    }

    pub fn runtime_config(&self) -> &serde_json::Value {
        self.state.runtime_config()
    }

    pub fn sync_for_reconcile(&mut self) -> Result<(), FleetError> {
        let metadata = self.pending_bootstrap_metadata.clone();
        match self.sync(metadata) {
            Ok(()) => {
                self.pending_bootstrap_metadata = None;
                Ok(())
            }
            Err(err) => Err(err),
        }
    }

    fn sync(&mut self, metadata: Option<DeviceMetadata>) -> Result<(), FleetError> {
        tracing::info!(
            ?metadata,
            "syncing fleet device with fleet management service"
        );

        let should_refresh_token = match self.state.auth_token() {
            Some(auth) if auth.is_valid() => {
                tracing::debug!(
                    "using existing auth token with ttl {} seconds",
                    auth.expires_in_seconds().unwrap()
                );
                false
            }
            Some(_) => {
                tracing::info!(
                    "existing auth token expired, requesting a new one from fleet management service"
                );
                self.clear_expired_auth_token()?;
                true
            }
            None => {
                tracing::info!(
                    "no existing auth token, requesting new one from fleet management service"
                );
                true
            }
        };

        if should_refresh_token {
            self.refresh_auth_token(metadata.clone())?;
            tracing::info!("successfully refreshed auth token");
        }
        let snapshot = self
            .client
            .sync(metadata)
            .map_err(|e| FleetError::SyncFailed(e.to_string()))?;

        let download = self
            .client
            .model_download()
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

    fn refresh_auth_token(&mut self, metadata: Option<DeviceMetadata>) -> Result<(), FleetError> {
        let auth_response = self
            .client
            .register(
                self.registration_token.clone(),
                self.identity_key.clone(),
                metadata,
            )
            .map_err(|e| FleetError::RegistrationFailed(e.to_string()))?;

        self.state
            .set_auth_token(auth_response.access_token, auth_response.expires_in_seconds);
        let auth_token = self
            .state
            .auth_token()
            .expect("auth token should be present after successful refresh")
            .clone();
        self.client
            .set_auth_token(auth_token)
            .map_err(|e| FleetError::SyncFailed(e.to_string()))?;

        Ok(())
    }

    fn clear_expired_auth_token(&mut self) -> Result<(), FleetError> {
        self.state.clear_auth_token();
        self.client
            .clear_auth_token()
            .map_err(|e| FleetError::SyncFailed(e.to_string()))?;

        self.persist_state()
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
