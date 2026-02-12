//! Inference contracts and type-erased adapters.

mod erased;
mod registry;

pub use erased::{ErasedInference, ErasedInferenceWriter, JsonInference};
pub use registry::{InferenceError, InferenceInit, InferenceRegistry, build};

/// Communication channel for an inference task, allowing the app to send outputs and errors back to the session.
pub struct InferenceWriter<O> {
    channel: Box<dyn InferenceWriterChannel<O>>,
    instant: std::time::Instant,
}

/// Errors that can occur when writing to an inference channel.
#[derive(Debug, thiserror::Error)]
pub enum InferenceWriterError {
    #[error("inference was cancelled")]
    Cancelled,
    #[error("unknown error: {0}")]
    Unknown(Box<dyn std::error::Error + Send + Sync>),
}

impl<O> InferenceWriter<O> {
    fn new(channel: Box<dyn InferenceWriterChannel<O>>) -> Self {
        Self {
            channel,
            instant: std::time::Instant::now(),
        }
    }

    /// Respond with an output item. This can be called multiple times to emit multiple items.
    pub fn write(&self, output: O) -> Result<(), InferenceWriterError> {
        self.channel.write(output)
    }

    /// Signal an error on the inference.
    pub fn error<E>(&self, error: E) -> Result<(), InferenceWriterError>
    where
        E: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        self.channel.error(error.into())
    }
}

/// When the `InferenceWriter` is dropped, it signals that the inference has finished, allowing the channel to perform any necessary cleanup or finalization.
impl<O> Drop for InferenceWriter<O> {
    fn drop(&mut self) {
        self.channel.finish(self.instant.elapsed());
    }
}

/// Trait representing an inference task that can be executed with a given input and a writer for outputs.
/// The inference implementation is responsible for writing outputs and errors to the provided writer, which will be sent back to the session.
pub trait InferenceWriterChannel<O> {
    /// Write an output item to the channel. This can be called multiple times to emit multiple items.
    fn write(&self, output: O) -> Result<(), InferenceWriterError>;
    /// Signal an error on the inference, which will be sent back to the session.
    fn error(
        &self,
        error: Box<dyn std::error::Error + Send + Sync>,
    ) -> Result<(), InferenceWriterError>;
    /// Called when the `InferenceWriter` is dropped, allowing the channel to perform any necessary cleanup or finalization.
    fn finish(&self, duration: std::time::Duration);
}

pub trait Inference {
    type Input;
    type Output;

    fn infer(&self, input: Self::Input, writer: InferenceWriter<Self::Output>);
}
