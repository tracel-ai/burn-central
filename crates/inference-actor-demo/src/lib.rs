//! Minimal actor-style inference runtime demo.

pub mod runtime;

pub use runtime::{
    Effect, InferenceApp, Job, RequestId, SessionHandle, spawn_session,
};
