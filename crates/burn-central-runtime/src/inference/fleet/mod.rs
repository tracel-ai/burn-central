use std::path::PathBuf;

use burn_central_core::Env;
use directories::{BaseDirs, ProjectDirs};
use serde::Deserialize;

mod model;
mod state;

pub type FleetRegistrationToken = String;
pub type FleetDeviceToken = String;
pub type FleetDeviceIdentityKey = String;

pub type DeviceMetadata = serde_json::Value;

#[derive(Debug, thiserror::Error)]
pub enum FleetError {
    #[error("fleet registration failed: {0}")]
    RegistrationFailed(String),
    #[error("failed to determine cache directory")]
    CacheDirUnavailable,
}

pub struct FleetDeviceSession {
    pub device_token: FleetDeviceToken,
    pub state: state::FleetState,
    pub client: Box<dyn FleetApi + Send + Sync>,
}

pub fn register(
    token: impl Into<FleetRegistrationToken>,
    metadata: DeviceMetadata,
) -> anyhow::Result<FleetDeviceSession> {
    let _root_dir = default_data_dir(&Env::Development)?;

    let identity = FleetDeviceIdentityKey::default();

    let client = NoopFleetApi;
    let device = client.register_device(token.into(), identity, metadata)?;

    let state = state::FleetState::default();

    let fleet_device = FleetDeviceSession {
        device_token: device.token,
        state,
        client: Box::new(client),
    };
    Ok(fleet_device)
}

/// Get the default cache directory for a given environment.
fn default_data_dir(env: &Env) -> Result<PathBuf, FleetError> {
    let fleets_subdir = match env {
        Env::Production => "fleets".to_string(),
        Env::Staging(version) => format!("fleets-staging-{}", version),
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

pub struct RegisteredFleetDevice {
    pub token: FleetRegistrationToken,
}

pub struct FleetSyncSnapshot {
    pub model_id: String,
    pub model_version_id: String,
    pub runtime_config: serde_json::Value,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PresignedModelFileUrlResponse {
    pub rel_path: String,
    pub url: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FleetModelDownloadResponse {
    pub model_version_id: String,
    pub files: Vec<PresignedModelFileUrlResponse>,
}

/// API client for interacting with the fleet management backend.
pub trait FleetApi {
    /// Register a new device with the fleet, returning a device token that can be used for subsequent API calls.
    fn register_device(
        &self,
        reg_token: FleetRegistrationToken,
        identity_key: FleetDeviceIdentityKey,
        metadata: DeviceMetadata,
    ) -> Result<RegisteredFleetDevice, FleetError>;

    /// Sync the device state with the fleet, returning any updates to the model version or runtime config.
    fn sync(
        &self,
        device_token: FleetDeviceToken,
        metadata: Option<DeviceMetadata>,
    ) -> Result<FleetSyncSnapshot, FleetError>;

    /// Download the model files for the specified model version, returning presigned URLs for each file.
    fn download_model(
        &self,
        device_token: FleetDeviceToken,
    ) -> Result<FleetModelDownloadResponse, FleetError>;
}

pub struct NoopFleetApi;

impl FleetApi for NoopFleetApi {
    fn register_device(
        &self,
        _reg_token: FleetRegistrationToken,
        _identity_key: FleetDeviceIdentityKey,
        _metadata: DeviceMetadata,
    ) -> Result<RegisteredFleetDevice, FleetError> {
        Ok(RegisteredFleetDevice {
            token: "noop-token".to_string(),
        })
    }

    fn sync(
        &self,
        _device_token: FleetDeviceToken,
        _metadata: Option<DeviceMetadata>,
    ) -> Result<FleetSyncSnapshot, FleetError> {
        Ok(FleetSyncSnapshot {
            model_id: "noop-model-id".to_string(),
            model_version_id: "noop-model-version-id".to_string(),
            runtime_config: serde_json::Value::Null,
        })
    }

    fn download_model(
        &self,
        _device_token: FleetDeviceToken,
    ) -> Result<FleetModelDownloadResponse, FleetError> {
        Ok(FleetModelDownloadResponse {
            model_version_id: "noop-model-version".to_string(),
            files: vec![PresignedModelFileUrlResponse {
                rel_path: "model.bin".to_string(),
                url: "https://example.com/model.bin".to_string(),
            }],
        })
    }
}
