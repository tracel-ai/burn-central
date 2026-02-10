//! Minimal actor-style inference runtime demo.

pub mod erased;
mod helper;
pub mod runtime;

pub use erased::{ErasedJob, ErasedSession, InferenceSpec, JsonSession};
pub use helper::{app, app_with_error};
pub use runtime::{
    Action, Actions, InferenceApp, Job, ModelExecutor, RequestId, SessionHandle, spawn_session,
};
