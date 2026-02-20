use serde::{Deserialize, Serialize};

/// The state of a device in the fleet management system, as stored on the device and synced with the fleet management service.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FleetState {
    /// Stable identity for the device, used for authentication and authorization with the fleet management service. Should not change across reinstalls or updates.
    pub identity_key: String,
    /// The id of the on-device fleet this device is assigned to.
    pub fleet_id: String, // from sync
    /// The timestamp of the last update received by the device.
    pub updated_at: String, // from sync
    /// The id of the model version currently active on the device. Should be updated by the device when a new model version is activated.
    pub active_model_version_id: String, // what is currently serving
    /// The json-encoded runtime configuration for the device, as last synced with the fleet management service.
    pub runtime_config: serde_json::Value,
}

impl Default for FleetState {
    fn default() -> Self {
        Self {
            identity_key: String::new(),
            fleet_id: String::new(),
            updated_at: chrono::Utc::now().to_rfc3339(),
            active_model_version_id: String::new(),
            runtime_config: serde_json::json!({}),
        }
    }
}
