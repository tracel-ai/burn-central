// #![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs)]

//! # Burn Central SDK
//!
//! High-level Burn Central SDK.
//!
//! This crate re-exports the main crates used to build training, experiment tracking, inference,
//! and fleet workflows on top of Burn Central.
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
//! - [`inference`]: inference contracts and adapters.
//! - [`fleet`]: on-device fleet synchronization helpers.
//!
//! ## Registering Routines
//!
//! Use the `register` macro to mark routines so the Burn Central CLI can discover them:
//!
//! ```ignore
//! use burn_central::macros::register;
//!
//! #[register(training, name = "my_training_procedure")]
//! async fn my_training_function() {
//!     // Your training code here
//! }
//! ```
//!
//! The `name` attribute is optional. If omitted, the function name is used.

pub use burn_central_client::BurnCentralCredentials;
pub use burn_central_client::Env;

#[doc(inline)]
pub use burn_central_experiment as experiment;

/// Attribute macros for registering routines discoverable by the Burn Central CLI.
#[doc(inline)]
pub use burn_central_macros as macros;

/// Runtime support for executing training and inference routines registered with Burn Central.
#[doc(inline)]
pub use burn_central_runtime as runtime;

/// Inference contracts and adapters.
#[doc(hidden)]
#[doc(inline)]
pub use burn_central_inference as inference;

/// On-device fleet synchronization helpers.
#[doc(hidden)]
#[doc(inline)]
pub use burn_central_fleet as fleet;

/// Artifact bundle utilities and adapters.
#[doc(inline)]
pub use burn_central_artifact as artifact;
