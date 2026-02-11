//! Inference runtime module (actor-style session runtime).

mod erased;
mod helper;
mod registry;
mod session;

pub use erased::{ErasedJob, ErasedSession, JsonSession};
pub use helper::{app, app_with_error, model};
pub use registry::{InferenceError, InferenceInit, InferenceRegistry};
pub use session::{
    Action, Actions, InferenceApp, Job, ModelExecutor, RequestId, SessionHandle, spawn_session,
};
