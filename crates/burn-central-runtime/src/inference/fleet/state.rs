use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// The state of a device in the fleet management system, as stored on the device and synced with the fleet management service.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FleetState {
    /// The timestamp of the last update received by the device.
    updated_at: String,
    /// The id of the model version currently active on the device. Should be updated by the device when a new model version is activated.
    active_model_version_id: String,
    /// The json-encoded runtime configuration for the device, as last synced with the fleet management service.
    runtime_config: serde_json::Value,
}

impl Default for FleetState {
    fn default() -> Self {
        Self {
            updated_at: chrono::Utc::now().to_rfc3339(),
            active_model_version_id: String::new(),
            runtime_config: serde_json::json!({}),
        }
    }
}

impl FleetState {
    pub fn update(&mut self, model_version_id: String, runtime_config: serde_json::Value) {
        self.updated_at = chrono::Utc::now().to_rfc3339();
        self.active_model_version_id = model_version_id;
        self.runtime_config = runtime_config;
    }

    pub fn active_model_version_id(&self) -> &str {
        &self.active_model_version_id
    }

    pub fn runtime_config(&self) -> &serde_json::Value {
        &self.runtime_config
    }
}
#[derive(Serialize, Deserialize, Clone, Debug)]
struct IdentityState {
    identity_key: String,
}

#[derive(Debug, thiserror::Error)]
pub enum FleetStateStoreError {
    #[error("failed to access filesystem: {0}")]
    Io(#[from] io::Error),
    #[error("failed to serialize/deserialize json: {0}")]
    Json(#[from] serde_json::Error),
}

/// Minimal file-backed local state store for runtime fleet bootstrap.
pub struct FleetLocalStateStore {
    root_dir: PathBuf,
}

impl FleetLocalStateStore {
    pub fn new(root_dir: PathBuf) -> Self {
        Self { root_dir }
    }

    pub fn models_dir(&self, fleet_key: &str) -> PathBuf {
        self.fleet_dir(fleet_key).join("models")
    }

    pub fn load_or_create_machine_identity_key(&self) -> Result<String, FleetStateStoreError> {
        let path = self.machine_state_path();
        if path.exists() {
            let machine: IdentityState = read_json(&path)?;
            return Ok(machine.identity_key);
        }

        let machine = IdentityState {
            identity_key: generate_identity_key(),
        };
        write_json_atomic(&path, &machine)?;
        Ok(machine.identity_key)
    }

    pub fn load_fleet_state(
        &self,
        fleet_key: &str,
    ) -> Result<Option<FleetState>, FleetStateStoreError> {
        let state_path = self.fleet_state_path(fleet_key);
        if !state_path.exists() {
            return Ok(None);
        }

        let value = read_json(&state_path)?;
        Ok(Some(value))
    }

    pub fn save_fleet_state(
        &self,
        fleet_key: &str,
        state: &FleetState,
    ) -> Result<(), FleetStateStoreError> {
        let state_path = self.fleet_state_path(fleet_key);
        write_json_atomic(&state_path, state)
    }

    fn machine_state_path(&self) -> PathBuf {
        self.root_dir.join("machine.json")
    }

    fn fleet_dir(&self, fleet_key: &str) -> PathBuf {
        self.root_dir.join("fleets").join(fleet_key)
    }

    fn fleet_state_path(&self, fleet_key: &str) -> PathBuf {
        self.fleet_dir(fleet_key).join("state.json")
    }
}

pub fn fleet_key_from_registration_token(token: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(token.as_bytes());
    let hash = format!("{:x}", hasher.finalize());
    format!("fleet-{}", &hash[..24])
}

fn generate_identity_key() -> String {
    let mut hasher = Sha256::new();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();

    hasher.update(std::process::id().to_string().as_bytes());
    hasher.update(nanos.to_string().as_bytes());

    let hash = format!("{:x}", hasher.finalize());
    format!("dev-{}", &hash[..24])
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T, FleetStateStoreError> {
    let bytes = fs::read(path)?;
    Ok(serde_json::from_slice(&bytes)?)
}

fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<(), FleetStateStoreError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let tmp_path = path.with_extension("tmp");
    let bytes = serde_json::to_vec_pretty(value)?;
    fs::write(&tmp_path, bytes)?;
    fs::rename(tmp_path, path)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "burn-central-runtime-{name}-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ))
    }

    #[test]
    fn persists_identity_and_fleet_state() {
        let root = temp_path("state");
        let store = FleetLocalStateStore::new(root.clone());

        let identity_a = store
            .load_or_create_machine_identity_key()
            .expect("identity should be created");
        let identity_b = store
            .load_or_create_machine_identity_key()
            .expect("identity should be loaded");
        assert_eq!(identity_a, identity_b);

        let fleet_key = fleet_key_from_registration_token("reg-token");
        let state = FleetState {
            updated_at: "2026-01-01T00:00:00Z".to_string(),
            active_model_version_id: "model-v1".to_string(),
            runtime_config: serde_json::json!({ "sample_rate": 0.2 }),
        };

        store
            .save_fleet_state(&fleet_key, &state)
            .expect("state should be persisted");

        let loaded = store
            .load_fleet_state(&fleet_key)
            .expect("state should be loaded")
            .expect("state should exist");

        assert_eq!(loaded.active_model_version_id, "model-v1");

        let _ = fs::remove_dir_all(root);
    }
}
