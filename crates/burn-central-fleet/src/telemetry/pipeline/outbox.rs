use std::sync::Mutex;

use crate::telemetry::{
    event::TelemetryEvent,
    pipeline::{Outbox, OutboxId},
};

pub struct InMemoryOutbox {
    queue: Mutex<Vec<(OutboxId, TelemetryEvent)>>,
    next_id: Mutex<OutboxId>,
}

impl Default for InMemoryOutbox {
    fn default() -> Self {
        Self {
            queue: Mutex::new(Vec::new()),
            next_id: Mutex::new(0),
        }
    }
}

impl Outbox for InMemoryOutbox {
    fn enqueue(&self, data: TelemetryEvent) -> Result<(), String> {
        let mut queue_guard = self
            .queue
            .lock()
            .map_err(|_| "outbox queue lock poisoned".to_string())?;
        let mut id_guard = self
            .next_id
            .lock()
            .map_err(|_| "outbox id lock poisoned".to_string())?;
        let id = *id_guard;
        *id_guard += 1;
        queue_guard.push((id, data));
        Ok(())
    }

    fn claim(&self, count: usize) -> Result<Vec<(OutboxId, TelemetryEvent)>, String> {
        let mut queue_guard = self
            .queue
            .lock()
            .map_err(|_| "outbox queue lock poisoned".to_string())?;
        let len = queue_guard.len();
        let items = queue_guard.drain(0..std::cmp::min(count, len)).collect();
        Ok(items)
    }

    fn complete(&self, _id: OutboxId) -> Result<(), String> {
        Ok(())
    }

    fn fail(&self, _id: OutboxId, _error: &str) -> Result<(), String> {
        Ok(())
    }
}
