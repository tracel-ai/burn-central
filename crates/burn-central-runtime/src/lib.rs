//! # Burn Central Runtime
//!
//! This crate is the middle layer of the Burn Central SDK. It communicate directly with the
//! generate crate created by Burn Central CLI and provide the necessary building block to execute
//! training and inference routines.
//!
//! # Features
//! - Extractor for training and inference arguments.
//! - Wrapper for return type of training and inference routines.
//! - Executor to run training and inference routines.
//! - Error handling for runtime execution.

mod error;
mod executor;
mod inference;
mod input;
mod output;
mod param;
mod routine;
mod type_name;
mod types;

#[doc(hidden)]
pub mod cli;

#[doc(inline)]
pub use executor::{ArtifactLoader, Executor, ExecutorBuilder};
#[doc(inline)]
pub use types::{Args, Model, MultiDevice};
