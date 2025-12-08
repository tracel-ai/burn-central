mod client;

pub mod schemas;

pub mod log;
pub mod metrics;
pub mod record;

pub mod experiment;

pub use crate::client::*;
pub use burn_central_client::BurnCentralCredentials;

pub mod artifacts;
pub mod bundle;
pub mod models;

// Conditional re-export based on feature flags
#[cfg(feature = "burn_0_20")]
pub use burn_0_20 as burn;

#[cfg(feature = "burn_0_19")]
pub use burn_0_19 as burn;

// Ensure at least one version is enabled
#[cfg(not(any(feature = "burn_0_20", feature = "burn_0_19")))]
compile_error!("At least one burn version feature must be enabled: burn_0_20 or burn_0_19");
