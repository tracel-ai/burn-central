use crate::inference::{Job, RequestId, SessionHandle};
use serde::{Serialize, de::DeserializeOwned};
use std::fmt::Display;

pub trait ErasedSession: Send + Sync {
    fn submit_bytes(&self, input: &[u8]) -> Result<Box<dyn ErasedJob>, String>;
}

pub trait ErasedJob: Send {
    fn try_recv_bytes(&mut self) -> Result<Option<Vec<u8>>, String>;
    fn join(self: Box<Self>) -> Result<(), String>;
    fn id(&self) -> RequestId;
}

pub struct JsonSession<I, O, E> {
    inner: SessionHandle<I, O, E>,
}

impl<I, O, E> JsonSession<I, O, E> {
    pub fn new(inner: SessionHandle<I, O, E>) -> Self {
        Self { inner }
    }
}

impl<I, O, E> ErasedSession for JsonSession<I, O, E>
where
    I: DeserializeOwned + Send + 'static,
    O: Serialize + Send + 'static,
    E: Display + Send + 'static,
{
    fn submit_bytes(&self, input: &[u8]) -> Result<Box<dyn ErasedJob>, String> {
        let req: I = serde_json::from_slice(input).map_err(|e| e.to_string())?;
        let job = self.inner.submit(req);
        Ok(Box::new(JsonJob { job }))
    }
}

struct JsonJob<I, O, E> {
    job: Job<I, O, E>,
}

impl<I, O, E> ErasedJob for JsonJob<I, O, E>
where
    I: Send + 'static,
    O: Serialize + Send + 'static,
    E: Display + Send + 'static,
{
    fn try_recv_bytes(&mut self) -> Result<Option<Vec<u8>>, String> {
        match self.job.stream.try_recv() {
            Ok(item) => {
                let bytes = serde_json::to_vec(&item).map_err(|e| e.to_string())?;
                Ok(Some(bytes))
            }
            Err(crossbeam::channel::TryRecvError::Empty) => Ok(None),
            Err(crossbeam::channel::TryRecvError::Disconnected) => Err("stream closed".to_string()),
        }
    }

    fn join(self: Box<Self>) -> Result<(), String> {
        self.job.join().map_err(|e| e.to_string())
    }

    fn id(&self) -> RequestId {
        self.job.id
    }
}
