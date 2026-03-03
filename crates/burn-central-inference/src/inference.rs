use std::sync::Arc;

use crate::{InferenceWriter, InferenceWriterChannel};

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
