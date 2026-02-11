//! Local registry/cache for Burn Central model artifacts.
//!
//! This crate provides a client for downloading and caching model artifacts from Burn Central.
//! It handles parallel downloads, checksum verification, and local caching of model files.
//!
//! # Example
//!
//! ```no_run
//! use burn_central_registry::{RegistryBuilder, CachedModel};
//! use burn_central_client::BurnCentralCredentials;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a registry client
//! let credentials = BurnCentralCredentials::from_env()?;
//! let registry = RegistryBuilder::new(credentials).build()?;
//!
//! // Get a model handle
//! let model = registry.model("namespace", "project", "model")?;
//!
//! // Ensure the model is cached locally
//! let cached = model.ensure(1)?;
//!
//! // Access the cached model files
//! let path = cached.path();
//! # Ok(())
//! # }
//! ```

mod builder;
mod cache;
mod diagnostics;
mod download;
mod error;
mod manifest;
mod model;
mod registry;

// Public API exports
pub use builder::RegistryBuilder;
pub use diagnostics::CacheDiagnostics;
pub use error::RegistryError;
pub use manifest::{ManifestFile, ModelManifest};
pub use model::{CachedModel, ModelHandle, ModelRef, ModelVersion, ModelVersionSelector};
pub use registry::Registry;
