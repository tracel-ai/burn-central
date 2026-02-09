//! Minimal actor-style inference runtime demo.

pub mod erased;
pub mod runtime;

pub use erased::{ErasedJob, ErasedSession, InferenceSpec, JsonSession};
pub use runtime::{
    Effect, InferenceApp, Job, ModelExecutor, RequestId, SessionHandle, spawn_session,
};
