use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use arc_swap::ArcSwapOption;
use burn::prelude::Backend;
use burn_central_inference::{Inference, InferenceWriter};

use crate::FleetDeviceSession;
use crate::model::ModelSource;
use crate::telemetry::{InferenceMetadata, InferenceWriterTelemetryObserver};

#[derive(Debug, thiserror::Error)]
pub enum FleetManagedInferenceError {
    #[error("inference '{name}' failed to initialize: {message}")]
    FactoryFailed { name: String, message: String },
}

pub trait FleetManagedFactory<B: Backend, I>: Send + Sync {
    fn build(
        &self,
        model_source: ModelSource,
        runtime_config: serde_json::Value,
        device: B::Device,
    ) -> Result<I, String>;
}

impl<F, B, I> FleetManagedFactory<B, I> for F
where
    F: Fn(ModelSource, serde_json::Value, B::Device) -> Result<I, String> + Send + Sync,
    B: Backend,
{
    fn build(
        &self,
        model_source: ModelSource,
        runtime_config: serde_json::Value,
        device: B::Device,
    ) -> Result<I, String> {
        self(model_source, runtime_config, device)
    }
}

struct ActiveInference<I> {
    inference: I,
    model_version: String,
}

/// Inference wrapper that bootstraps burn-central features like fleet registration and telemetry on top of a typed inference implementation.
pub struct FleetManagedInference<B: Backend, I> {
    inference_name: String,
    fleet_session: RwLock<FleetDeviceSession>,
    factory: Box<dyn FleetManagedFactory<B, I>>,
    device: B::Device,
    active: ArcSwapOption<ActiveInference<I>>,
    reconcile_gate: Mutex<()>,
    last_sync_at: Mutex<Option<Instant>>,
    sync_interval: Duration,
}

impl<B, I> FleetManagedInference<B, I>
where
    B: Backend,
    I: Inference,
{
    pub fn init(
        inference_name: impl Into<String>,
        fleet_session: FleetDeviceSession,
        factory: Box<dyn FleetManagedFactory<B, I>>,
        device: B::Device,
    ) -> Result<Self, FleetManagedInferenceError> {
        let inference = Self {
            inference_name: inference_name.into(),
            fleet_session: RwLock::new(fleet_session),
            factory,
            device,
            active: ArcSwapOption::empty(),
            reconcile_gate: Mutex::new(()),
            last_sync_at: Mutex::new(None),
            sync_interval: Duration::from_secs(10),
        };
        inference.ensure_ready()?;
        Ok(inference)
    }

    fn maybe_sync_and_rollout(&self) -> Result<(), FleetManagedInferenceError> {
        let fleet_key = self.current_fleet_key();
        let reconcile_span = tracing::info_span!(
            "fleet.inference.reconcile",
            fleet_key = fleet_key.as_str(),
            inference_name = self.inference_name.as_str(),
        );
        let _reconcile_guard = reconcile_span.enter();

        if self.active().is_some() && !self.should_sync_now() {
            return Ok(());
        }

        let _guard = self.reconcile_gate.lock().unwrap();
        if self.active().is_some() && !self.should_sync_now() {
            return Ok(());
        }

        let (fleet_version, model_source, config) = {
            let mut session = self.fleet_session.write().unwrap();

            match session.sync_for_reconcile() {
                Ok(()) => {
                    let fleet_version = normalized_model_version(session.active_model_version_id());
                    let model_source = session.model_source().map_err(|err| {
                        FleetManagedInferenceError::FactoryFailed {
                            name: self.inference_name.clone(),
                            message: format!("fleet model source failed: {err}"),
                        }
                    })?;

                    let config = session.runtime_config();

                    (fleet_version, model_source, config.clone())
                }
                Err(sync_err) => {
                    self.mark_sync_now();

                    if self.active().is_some() {
                        tracing::warn!(
                            err = %sync_err,
                            "fleet sync failed, keeping current active model"
                        );
                        return Ok(());
                    }

                    tracing::warn!(
                        err = %sync_err,
                         "fleet sync failed and no active model, trying local cache"
                    );

                    let fleet_version = normalized_model_version(session.active_model_version_id());
                    let model_source = session.model_source().map_err(|cache_err| {
                        FleetManagedInferenceError::FactoryFailed {
                            name: self.inference_name.clone(),
                            message: format!(
                                "fleet sync failed and no usable local cache: sync={sync_err}; cache={cache_err}"
                            ),
                        }
                    })?;
                    let config = session.runtime_config();

                    (fleet_version, model_source, config.clone())
                }
            }
        };

        self.mark_sync_now();

        let active = self.active();
        if active.as_ref().map(|a| &a.model_version) == Some(&fleet_version) {
            tracing::info!(
                version = &fleet_version,
                "fleet model version is same as active, skipping rollout"
            );
            return Ok(());
        }

        let built = self
            .factory
            .build(model_source, config, self.device.clone())
            .map_err(|message| FleetManagedInferenceError::FactoryFailed {
                name: self.inference_name.clone(),
                message,
            })?;

        self.active.store(Some(Arc::new(ActiveInference {
            inference: built,
            model_version: fleet_version,
        })));

        Ok(())
    }

    fn ensure_ready(&self) -> Result<(), FleetManagedInferenceError> {
        self.maybe_sync_and_rollout()?;
        if self.active().is_none() {
            return Err(FleetManagedInferenceError::FactoryFailed {
                name: self.inference_name.clone(),
                message: "no active model after bootstrap".to_string(),
            });
        }
        Ok(())
    }

    fn should_sync_now(&self) -> bool {
        let last_sync_at = self.last_sync_at.lock().unwrap();
        match *last_sync_at {
            Some(instant) => instant.elapsed() >= self.sync_interval,
            None => true,
        }
    }

    fn mark_sync_now(&self) {
        let mut last_sync_at = self.last_sync_at.lock().unwrap();
        *last_sync_at = Some(Instant::now());
    }

    fn active(&self) -> Option<Arc<ActiveInference<I>>> {
        self.active.load_full()
    }

    fn current_fleet_key(&self) -> String {
        self.fleet_session.read().unwrap().fleet_key().to_string()
    }
}

impl<B, I> Inference for FleetManagedInference<B, I>
where
    B: Backend,
    I: Inference,
{
    type Input = <I as Inference>::Input;
    type Output = <I as Inference>::Output;

    fn infer(&self, input: Self::Input, writer: InferenceWriter<Self::Output>) {
        let fleet_key = self.current_fleet_key();
        let request_span = tracing::info_span!(
            "fleet.inference.request",
            fleet_key = fleet_key.as_str(),
            inference_name = self.inference_name.as_str(),
        );
        let _request_guard = request_span.enter();

        if let Err(err) = self.maybe_sync_and_rollout() {
            writer.error(Box::new(err)).ok();
            return;
        }

        let Some(active) = self.active() else {
            writer
                .error(Box::new(FleetManagedInferenceError::FactoryFailed {
                    name: self.inference_name.clone(),
                    message: "no active model".to_string(),
                }))
                .ok();
            return;
        };

        let metadata = InferenceMetadata::new(
            fleet_key,
            self.inference_name.clone(),
            "unknown".to_string(),
            active.model_version.clone(),
        );

        let writer =
            writer.with_observer(Arc::new(InferenceWriterTelemetryObserver::new(metadata)));
        active.inference.infer(input, writer)
    }
}

fn normalized_model_version(version: &str) -> String {
    if version.is_empty() {
        "unknown".to_string()
    } else {
        version.to_string()
    }
}
