mod error;
mod inference;
mod model;
mod session;
mod state;
mod telemetry;

pub use error::FleetError;
pub use inference::{FleetManagedFactory, FleetManagedInference, FleetManagedInferenceError};
pub use model::ModelSource;
pub use session::FleetDeviceSession;

pub type FleetRegistrationToken = String;

pub type DeviceMetadata = serde_json::Value;
