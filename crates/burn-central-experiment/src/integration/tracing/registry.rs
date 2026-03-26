use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::{ExperimentId, ExperimentRunHandle};

static TRACING_REGISTRY: OnceLock<TracingRegistry> = OnceLock::new();

#[derive(Default)]
pub(crate) struct TracingRegistry {
    handles: Mutex<HashMap<ExperimentId, RegisteredHandle>>,
}

struct RegisteredHandle {
    handle: ExperimentRunHandle,
    ref_count: usize,
}

pub(crate) struct TracingRegistration {
    experiment_id: ExperimentId,
}

impl TracingRegistry {
    pub(crate) fn global() -> &'static Self {
        TRACING_REGISTRY.get_or_init(Default::default)
    }

    pub(crate) fn register_handle(&self, handle: ExperimentRunHandle) -> TracingRegistration {
        let experiment_id = handle.id().clone();
        let mut handles = self.handles.lock().unwrap();

        match handles.get_mut(&experiment_id) {
            Some(existing) => {
                existing.ref_count += 1;
            }
            None => {
                handles.insert(
                    experiment_id.clone(),
                    RegisteredHandle {
                        handle,
                        ref_count: 1,
                    },
                );
            }
        }

        TracingRegistration { experiment_id }
    }

    pub(crate) fn get_handle(&self, experiment_id: &ExperimentId) -> Option<ExperimentRunHandle> {
        let handles = self.handles.lock().unwrap();
        handles.get(experiment_id).map(|entry| entry.handle.clone())
    }

    fn unregister(&self, experiment_id: &ExperimentId) {
        let mut handles = self.handles.lock().unwrap();
        let remove = match handles.get_mut(experiment_id) {
            Some(entry) if entry.ref_count > 1 => {
                entry.ref_count -= 1;
                false
            }
            Some(_) => true,
            None => false,
        };

        if remove {
            handles.remove(experiment_id);
        }
    }
}

impl Drop for TracingRegistration {
    fn drop(&mut self) {
        TracingRegistry::global().unregister(&self.experiment_id);
    }
}
