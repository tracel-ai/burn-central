//! Inference contracts and type-erased adapters.

pub mod erased;
mod inference;
pub mod observer;
pub mod stream;
mod writer;

pub use erased::{ErasedInference, ErasedInferenceWriter, JsonInference};
pub use inference::{Inference, InferenceWrapper};
pub use writer::{InferenceWriter, InferenceWriterChannel, InferenceWriterError};
