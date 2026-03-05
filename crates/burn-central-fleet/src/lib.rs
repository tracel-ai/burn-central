mod error;
mod inference;
mod model;
mod session;
mod state;
mod telemetry;

pub use error::FleetError;
pub use inference::{FleetManagedFactory, FleetManagedInference, FleetManagedInferenceError};
pub use session::FleetDeviceSession;
pub use telemetry::{metrics_recorder, tracing_log_layer, tracing_metrics_layer};

pub type FleetRegistrationToken = String;

pub type DeviceMetadata = serde_json::Value;
