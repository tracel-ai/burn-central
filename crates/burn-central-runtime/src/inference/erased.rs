use crate::inference::{Inference, InferenceWriter, InferenceWriterChannel, InferenceWriterError};
use serde::{Serialize, de::DeserializeOwned};
use std::error::Error;
use std::fmt;
use std::marker::PhantomData;
use std::time::Duration;

pub trait ErasedInference: Send + Sync {
    fn infer_bytes(
        &self,
        input: &[u8],
        writer: Box<dyn ErasedInferenceWriter>,
    ) -> Result<(), String>;
}

pub trait ErasedInferenceWriter: Send + Sync {
    fn write_bytes(&self, output: Vec<u8>) -> Result<(), String>;
    fn error(&self, error: String) -> Result<(), String>;
    fn finish(&self, duration: Duration);
}

pub struct JsonInference<T, I, O> {
    inner: T,
    _types: PhantomData<fn(I, O)>,
}

impl<T, I, O> JsonInference<T, I, O> {
    pub fn new(inner: T) -> Self {
        Self {
            inner,
            _types: PhantomData,
        }
    }
}

impl<T, I, O> ErasedInference for JsonInference<T, I, O>
where
    T: Inference<Input = I, Output = O> + Send + Sync + 'static,
    I: DeserializeOwned + Send + 'static,
    O: Serialize + Send + 'static,
{
    fn infer_bytes(
        &self,
        input: &[u8],
        writer: Box<dyn ErasedInferenceWriter>,
    ) -> Result<(), String> {
        let input: I = serde_json::from_slice(input).map_err(|e| e.to_string())?;
        let channel = JsonInferenceWriterChannel::<O> {
            writer,
            _types: PhantomData,
        };
        let writer = InferenceWriter::new(Box::new(channel));
        self.inner.infer(input, writer);
        Ok(())
    }
}

struct JsonInferenceWriterChannel<O> {
    writer: Box<dyn ErasedInferenceWriter>,
    _types: PhantomData<fn(O)>,
}

#[derive(Debug)]
struct StringError(String);

impl fmt::Display for StringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl Error for StringError {}

impl<O> InferenceWriterChannel<O> for JsonInferenceWriterChannel<O>
where
    O: Serialize,
{
    fn write(&self, output: O) -> Result<(), InferenceWriterError> {
        let bytes =
            serde_json::to_vec(&output).map_err(|e| InferenceWriterError::Unknown(Box::new(e)))?;
        self.writer
            .write_bytes(bytes)
            .map_err(|err| InferenceWriterError::Unknown(Box::new(StringError(err))))
    }

    fn error(&self, error: Box<dyn Error + Send + Sync>) -> Result<(), InferenceWriterError> {
        self.writer
            .error(error.to_string())
            .map_err(|err| InferenceWriterError::Unknown(Box::new(StringError(err))))
    }

    fn finish(&self, duration: Duration) {
        self.writer.finish(duration);
    }
}
