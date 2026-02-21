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
pub(crate) struct NoopInferenceWriterObserver;

impl InferenceWriterObserver for NoopInferenceWriterObserver {}
