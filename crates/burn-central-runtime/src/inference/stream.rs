use crate::inference::{InferenceWrapper, InferenceWriterChannel, InferenceWriterError};

use crossbeam::channel as cb;

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Duration,
};

pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

pub enum StreamEvent<O> {
    Output(O),
    Error(BoxError),
    Done(Duration),
}

pub struct InferenceStream<O> {
    rx: cb::Receiver<StreamEvent<O>>,
    cancel: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}

impl<O> InferenceStream<O> {
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Release);
    }

    fn join(&mut self) {
        if let Some(worker) = self.worker.take() {
            worker.join().unwrap();
        }
    }
}

impl<O> Drop for InferenceStream<O> {
    fn drop(&mut self) {
        self.cancel();
        self.join();
    }
}

impl<O> Iterator for InferenceStream<O> {
    type Item = Result<O, BoxError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.rx.recv().ok()? {
            StreamEvent::Output(o) => Some(Ok(o)),
            StreamEvent::Error(e) => Some(Err(e)),
            StreamEvent::Done(_) => None,
        }
    }
}

struct StreamingChannel<O> {
    tx: cb::Sender<StreamEvent<O>>,
    cancel: Arc<AtomicBool>,
}

impl<O> InferenceWriterChannel<O> for StreamingChannel<O>
where
    O: Send + Sync + 'static,
{
    fn write(&self, output: O) -> Result<(), InferenceWriterError> {
        if self.cancel.load(Ordering::Acquire) {
            return Err(InferenceWriterError::Cancelled);
        }

        self.tx
            .send(StreamEvent::Output(output))
            .map_err(|e| InferenceWriterError::Unknown(Box::new(e)))
    }

    fn error(&self, error: BoxError) -> Result<(), InferenceWriterError> {
        self.tx
            .send(StreamEvent::Error(error))
            .map_err(|e| InferenceWriterError::Unknown(Box::new(e)))
    }

    fn finish(&self, duration: Duration) {
        let _ = self.tx.send(StreamEvent::Done(duration));
    }
}

impl<O> Drop for StreamingChannel<O> {
    fn drop(&mut self) {}
}

pub struct DirectInference<I, O> {
    inner: InferenceWrapper<I, O>,
}

impl<I, O> DirectInference<I, O>
where
    I: Send + 'static,
    O: Send + Sync + 'static,
{
    pub fn new(inference: InferenceWrapper<I, O>) -> Self {
        Self { inner: inference }
    }

    pub fn stream(&self, input: I) -> InferenceStream<O> {
        let (tx, rx) = cb::unbounded();
        let cancel = Arc::new(AtomicBool::new(false));

        let channel = StreamingChannel {
            tx: tx.clone(),
            cancel: cancel.clone(),
        };

        let inference = self.inner.clone();

        let worker = thread::spawn(move || {
            inference.infer(input, channel);
        });

        InferenceStream {
            rx,
            cancel,
            worker: Some(worker),
        }
    }
}
