//! Inference registration and runtime integration.

mod registry;

pub use burn_central_inference::*;
pub use registry::{
    InferenceArgs, InferenceError, InferenceInit, InferenceRegistry, ModelSource,
    build_fleet_managed,
};
