use crate::telemetry::event::TelemetryEvent;

pub mod wal;

pub type OutboxId = i64;

pub trait Outbox: Send + Sync {
    fn enqueue(&self, data: TelemetryEvent) -> Result<(), String>;
    fn claim(&self, count: usize) -> Result<Option<Vec<(OutboxId, TelemetryEvent)>>, String>;
    fn complete(&self, id: OutboxId) -> Result<(), String>;
    fn fail(&self, id: OutboxId, error: &str) -> Result<(), String>;
}

pub struct NotifyingOutbox<O: Outbox> {
    inner: O,
    notify_new_data: Box<dyn Fn() + Send + Sync>,
}

impl<O: Outbox> NotifyingOutbox<O> {
    pub fn new(inner: O, notify_new_data: Box<dyn Fn() + Send + Sync>) -> Self {
        Self {
            inner,
            notify_new_data,
        }
    }
}

impl<O: Outbox> Outbox for NotifyingOutbox<O> {
    fn enqueue(&self, data: TelemetryEvent) -> Result<(), String> {
        let result = self.inner.enqueue(data);
        if result.is_ok() {
            (self.notify_new_data)();
        }
        result
    }

    fn claim(&self, count: usize) -> Result<Option<Vec<(OutboxId, TelemetryEvent)>>, String> {
        self.inner.claim(count)
    }

    fn complete(&self, id: OutboxId) -> Result<(), String> {
        self.inner.complete(id)
    }

    fn fail(&self, id: OutboxId, error: &str) -> Result<(), String> {
        self.inner.fail(id, error)
    }
}
