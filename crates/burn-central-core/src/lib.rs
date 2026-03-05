//! This crate provides core functionalities for tracking experiments on the Burn Central platform.

mod client;

pub mod artifacts;
pub mod experiment;
pub mod integration;
pub mod models;
mod schemas;

pub use crate::client::*;

pub type BurnCentralCredentials = burn_central_client::BurnCentralCredentials;
pub type Env = burn_central_client::Env;
pub use schemas::*;

/// This is a temporary re-export of the bundle traits for users to implement them for their artifacts. Later, these traits will be available in a separate crate `burn-central-artifact`.
pub mod bundle {
    pub use burn_central_artifact::bundle::*;
}
