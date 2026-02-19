use std::path::PathBuf;

use burn_central_core::Env;
use directories::{BaseDirs, ProjectDirs};

mod model;
mod state;

pub type FleetRegistrationToken = String;
pub type FleetDeviceToken = String;

pub type DeviceMetadata = serde_json::Value;

#[derive(Debug, thiserror::Error)]
pub enum FleetError {
    #[error("fleet registration failed: {0}")]
    RegistrationFailed(String),
    #[error("failed to determine cache directory")]
    CacheDirUnavailable,
}

pub fn register(
    token: impl Into<FleetRegistrationToken>,
    metadata: DeviceMetadata,
) -> anyhow::Result {
    let root_dir = directories::BaseDirs::new()
        .map(|dirs| dirs.data_local_dir().join("burn-fleet"))
        .ok_or_else(|| {
            FleetError::RegistrationFailed("failed to determine local data directory".into())
        })?;
    let fleet_device = ().into();
    // let fleet_device = fleet::register_device(token.into()); --- IGNORE ---
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

pub struct SyncSnapshot {}

pub trait FleetApi {
    fn register_device(
        &self,
        token: FleetRegistrationToken,
        metadata: DeviceMetadata,
    ) -> Result<RegisteredFleetDevice, FleetError>;

    fn sync(
        &self,
        token: FleetDeviceToken,
        metadata: Option<DeviceMetadata>,
    ) -> Result<SyncSnapshot, FleetError>;
}
