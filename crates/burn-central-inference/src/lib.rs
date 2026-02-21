//! Inference contracts and type-erased adapters.

mod erased;
mod stream;

pub use erased::{ErasedInference, ErasedInferenceWriter, JsonInference};
pub use stream::*;

use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::time::Duration;

/// Runtime-owned writer statistics for a completed inference request.
#[derive(Debug, Clone, Copy)]
pub struct InferenceWriterStats {
    pub duration: Duration,
    pub outputs: usize,
    pub errors: usize,
    pub cancelled: bool,
}

/// Observer interface for writer lifecycle events.
pub trait InferenceWriterObserver: Send + Sync + 'static {
    fn on_write(&self) {}

    fn on_error(&self) {}

    fn on_cancelled(&self) {}

    fn on_finish(&self, _stats: &InferenceWriterStats) {}
}

#[derive(Default)]
struct NoopInferenceWriterObserver;

impl InferenceWriterObserver for NoopInferenceWriterObserver {}

/// Communication channel for an inference task, allowing the app to send outputs and errors back to the session.
pub struct InferenceWriter<O> {
    channel: Box<dyn InferenceWriterChannel<O>>,
    instant: std::time::Instant,
    observer: Arc<dyn InferenceWriterObserver>,
    outputs: AtomicUsize,
    errors: AtomicUsize,
    cancelled: AtomicBool,
    finished: AtomicBool,
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
    pub(crate) fn new(channel: Box<dyn InferenceWriterChannel<O>>) -> Self {
        Self {
            channel,
            instant: std::time::Instant::now(),
            observer: Arc::new(NoopInferenceWriterObserver),
            outputs: AtomicUsize::new(0),
            errors: AtomicUsize::new(0),
            cancelled: AtomicBool::new(false),
            finished: AtomicBool::new(false),
        }
    }

    pub(crate) fn from_channel<C>(channel: C) -> Self
    where
        C: InferenceWriterChannel<O> + 'static,
    {
        Self::new(Box::new(channel))
    }

    pub fn with_observer(mut self, observer: Arc<dyn InferenceWriterObserver>) -> Self {
        self.observer = observer;
        self
    }

    /// Respond with an output item. This can be called multiple times to emit multiple items.
    pub fn write(&self, output: O) -> Result<(), InferenceWriterError> {
        match self.channel.write(output) {
            Ok(()) => {
                self.outputs.fetch_add(1, Ordering::Relaxed);
                self.observer.on_write();
                Ok(())
            }
            Err(err) => {
                if matches!(&err, InferenceWriterError::Cancelled) {
                    self.cancelled.store(true, Ordering::Release);
                    self.observer.on_cancelled();
                }
                Err(err)
            }
        }
    }

    /// Signal an error on the inference.
    pub fn error<E>(&self, error: E) -> Result<(), InferenceWriterError>
    where
        E: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        match self.channel.error(error.into()) {
            Ok(()) => {
                self.errors.fetch_add(1, Ordering::Relaxed);
                self.observer.on_error();
                Ok(())
            }
            Err(err) => {
                if matches!(&err, InferenceWriterError::Cancelled) {
                    self.cancelled.store(true, Ordering::Release);
                    self.observer.on_cancelled();
                }
                Err(err)
            }
        }
    }

    fn finish(&self) {
        let duration = self.instant.elapsed();
        self.channel.finish(duration);

        if self.finished.swap(true, Ordering::AcqRel) {
            return;
        }

        self.observer.on_finish(&InferenceWriterStats {
            duration,
            outputs: self.outputs.load(Ordering::Acquire),
            errors: self.errors.load(Ordering::Acquire),
            cancelled: self.cancelled.load(Ordering::Acquire),
        });
    }
}

/// When the `InferenceWriter` is dropped, it signals that the inference has finished, allowing the channel to perform any necessary cleanup or finalization.
impl<O> Drop for InferenceWriter<O> {
    fn drop(&mut self) {
        self.finish();
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

// TODO: maybe this should require send + sync
pub trait Inference {
    type Input;
    type Output;

    fn infer(&self, input: Self::Input, writer: InferenceWriter<Self::Output>);
}

pub struct InferenceWrapper<I, O> {
    inner: Arc<dyn Inference<Input = I, Output = O> + Send + Sync>,
}

impl<I, O> Clone for InferenceWrapper<I, O> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<I, O> InferenceWrapper<I, O> {
    fn new<T>(inference: T) -> Self
    where
        T: Inference<Input = I, Output = O> + Send + Sync + 'static,
    {
        Self {
            inner: Arc::new(inference),
        }
    }
}

impl<T, I, O> From<T> for InferenceWrapper<I, O>
where
    T: Inference<Input = I, Output = O> + Send + Sync + 'static,
{
    fn from(inference: T) -> Self {
        Self::new(inference)
    }
}

impl<I, O> InferenceWrapper<I, O> {
    pub fn infer<T: InferenceWriterChannel<O> + 'static>(&self, input: I, writer: T) {
        self.inner
            .infer(input, InferenceWriter::from_channel(writer));
    }
}
