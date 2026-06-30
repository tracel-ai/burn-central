// #![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs)]

//! # Tracel SDK
//!
//! High-level Tracel SDK.
//!
//! This crate re-exports the main crates used to build training, experiment tracking, inference,
//! and fleet workflows on top of Tracel.
//!
//! Features:
//! - Artifact creation and management
//! - Experiment tracking
//! - Logging metrics
//! - Model versioning
//!
//! ## Crate Layout
//!
//! The most commonly used re-exports are:
//! - [`experiment`]: experiment runs, logging, artifacts, and Burn learner integrations.
//! - [`macros`]: attribute macros used to register training and inference routines.
//! - [`runtime`]: runtime support for executing registered routines.
//! - [`artifact`]: bundle and artifact utilities.
//!
//! ## Registering Routines
//!
//! Use the `register` macro to mark routines so the Tracel CLI can discover them:
//!
//! ```ignore
//! use ::macros::register;
//!
//! #[register(training, name = "my_training_procedure")]
//! async fn my_training_function() {
//!     // Your training code here
//! }
//! ```
//!
//! The `name` attribute is optional. If omitted, the function name is used.

pub use tracel_client::Env;
pub use tracel_client::TracelCredentials;

/// Experiment tracking and management.
pub mod experiment {
    pub use tracel_core::experiment::*;
    #[doc(inline)]
    pub use tracel_experiment::*;
}

/// Inference contracts and adapters.
#[doc(hidden)]
#[doc(inline)]
pub use tracel_inference as inference;

/// On-device fleet synchronization helpers.
#[doc(hidden)]
#[doc(inline)]
pub use tracel_fleet as fleet;

/// Artifact bundle utilities and adapters.
#[doc(inline)]
pub use tracel_artifact as artifact;

/// App module for job registration, CLI, and config mappers
#[doc(inline)]
pub use tracel_app as app;

pub use tracel_core::Connection;
pub use tracel_core::Context;
pub use tracel_core::ContextError;
