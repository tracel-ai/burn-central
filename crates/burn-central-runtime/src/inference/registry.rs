use crate::inference::fleet::{self, FleetRegistrationToken};
use crate::inference::telemetry::{InferenceMetadata, InferenceWriterTelemetryObserver};
use crate::inference::{ErasedInference, Inference, JsonInference};
use crate::params::RoutineParam;
use crate::params::args::{LaunchArgs, deserialize_and_merge_with_default};
use crate::routine::{BoxedRoutine, IntoRoutine, Routine};
use crate::{Args, MultiDevice};
use arc_swap::ArcSwapOption;
use burn::prelude::Backend;
use burn_central_core::Env;
use burn_central_core::bundle::{BundleDecode, FsBundleReader};
use serde::{Serialize, de::DeserializeOwned};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

#[derive(Debug, thiserror::Error)]
pub enum InferenceError {
    #[error("inference handler not found: {0}")]
    NotFound(String),
    #[error("inference handler '{name}' failed to initialize: {message}")]
    FactoryFailed { name: String, message: String },
}

/// Source information for loading the model that was assigned to an inference
/// This is passed to inference factories to load the model artifacts needed for inference.
/// The actual loading logic is implemented in `ModelSource::load` which uses the [`BundleDecode`] trait to support flexible model formats.
#[derive(Debug, Clone)]
pub struct ModelSource {
    root: PathBuf,
    files: Vec<String>,
}

impl ModelSource {
    pub fn new(root: PathBuf, files: Vec<String>) -> Self {
        Self { root, files }
    }
}

impl ModelSource {
    pub fn load<D: BundleDecode>(&self, settings: &D::Settings) -> Result<D, D::Error> {
        let reader = FsBundleReader::new(self.root.clone(), self.files.clone());
        D::decode(&reader, settings)
    }
}

pub struct InferenceInit<B: Backend> {
    pub model: ModelSource,
    pub device: B::Device,
}

/// Optional inference arguments passed at model-build time.
#[derive(Clone, Debug, Default)]
pub struct InferenceArgs(Option<serde_json::Value>);

impl InferenceArgs {
    pub fn new(args_override: Option<serde_json::Value>) -> Self {
        Self(args_override)
    }

    pub fn raw(&self) -> Option<&serde_json::Value> {
        self.0.as_ref()
    }

    pub fn into_raw(self) -> Option<serde_json::Value> {
        self.0
    }

    pub fn merged_args<A: LaunchArgs>(&self) -> Result<A, serde_json::Error> {
        match self.raw() {
            Some(args) => deserialize_and_merge_with_default(args),
            None => Ok(A::default()),
        }
    }

    pub fn merged_args_or_default<A: LaunchArgs>(&self) -> A {
        self.merged_args().unwrap_or_default()
    }
}

impl<T> From<T> for InferenceArgs
where
    T: Serialize,
{
    fn from(value: T) -> Self {
        let json = serde_json::to_value(value).unwrap_or_else(|_| serde_json::Value::Null);
        Self::new(Some(json))
    }
}

/// The execution context for inference initialization routines.
pub struct InferenceContext<B: Backend> {
    init: InferenceInit<B>,
    args: InferenceArgs,
}

impl<B: Backend> InferenceContext<B> {
    pub fn new(init: InferenceInit<B>, args: impl Into<InferenceArgs>) -> Self {
        Self {
            init,
            args: args.into(),
        }
    }

    /// Retrieve args merged on top of `A::default()`.
    ///
    /// This is used by the `Args<A>` routine extractor for inference factories.
    pub fn use_merged_args<A: LaunchArgs>(&self) -> A {
        self.args.merged_args_or_default()
    }

    pub fn model(&self) -> &ModelSource {
        &self.init.model
    }

    pub fn device(&self) -> &B::Device {
        &self.init.device
    }
}

impl<B: Backend> RoutineParam<InferenceContext<B>> for ModelSource {
    type Item<'new> = ModelSource;

    fn try_retrieve(ctx: &InferenceContext<B>) -> anyhow::Result<Self::Item<'_>> {
        Ok(ctx.model().clone())
    }
}

impl<B: Backend> RoutineParam<InferenceContext<B>> for MultiDevice<B> {
    type Item<'new> = MultiDevice<B>;

    fn try_retrieve(ctx: &InferenceContext<B>) -> anyhow::Result<Self::Item<'_>> {
        Ok(MultiDevice(vec![ctx.device().clone()]))
    }
}

impl<B: Backend, C: LaunchArgs> RoutineParam<InferenceContext<B>> for Args<C> {
    type Item<'new> = Args<C>;

    fn try_retrieve(ctx: &InferenceContext<B>) -> anyhow::Result<Self::Item<'_>> {
        let cfg = ctx.use_merged_args();
        Ok(Args(cfg))
    }
}

type InferenceRoutine<B, I> = BoxedRoutine<InferenceContext<B>, (), I>;

trait ErasedFactory<B: Backend>: Send + Sync {
    fn build(&self, ctx: InferenceContext<B>) -> Result<Box<dyn ErasedInference>, InferenceError>;
}

pub trait InferenceFactoryReturn<M>: Send + 'static {
    type Inference: Inference + Send + Sync + 'static;

    fn into_inference(self) -> Result<Self::Inference, String>;
}

impl<T> InferenceFactoryReturn<()> for T
where
    T: Inference + Send + Sync + 'static,
{
    type Inference = T;

    fn into_inference(self) -> Result<Self::Inference, String> {
        Ok(self)
    }
}

pub struct IsResultInferenceFactoryReturn;

impl<T, E> InferenceFactoryReturn<(IsResultInferenceFactoryReturn,)> for Result<T, E>
where
    T: Inference + Send + Sync + 'static,
    E: ToString + Send + Sync + 'static,
{
    type Inference = T;

    fn into_inference(self) -> Result<Self::Inference, String> {
        self.map_err(|err| err.to_string())
    }
}

struct RoutineFactory<B: Backend, I, R> {
    name: String,
    routine: InferenceRoutine<B, I>,
    _types: PhantomData<fn(I) -> R>,
}

impl<B, I, R> ErasedFactory<B> for RoutineFactory<B, I, R>
where
    B: Backend,
    I: InferenceFactoryReturn<R>,
    I::Inference: Inference + Send + Sync + 'static,
    <I::Inference as Inference>::Input: DeserializeOwned + Send + Sync + 'static,
    <I::Inference as Inference>::Output: Serialize + Send + Sync + 'static,
{
    fn build(
        &self,
        mut ctx: InferenceContext<B>,
    ) -> Result<Box<dyn ErasedInference>, InferenceError> {
        let factory_output =
            self.routine
                .run((), &mut ctx)
                .map_err(|err| InferenceError::FactoryFailed {
                    name: self.name.clone(),
                    message: err.to_string(),
                })?;
        let inference =
            factory_output
                .into_inference()
                .map_err(|message| InferenceError::FactoryFailed {
                    name: self.name.clone(),
                    message,
                })?;
        // let metadata = InferenceMetadata::new(self.name.clone(), "unknown", "unknown");
        // let inference = InstrumentedInference::new(inference, metadata);

        Ok(Box::new(JsonInference::new(inference)))
    }
}

/// Registry of inference factories keyed by name.
pub struct InferenceRegistry<B: Backend> {
    factories: HashMap<String, Box<dyn ErasedFactory<B>>>,
}

impl<B: Backend> InferenceRegistry<B> {
    pub fn new() -> Self {
        Self {
            factories: HashMap::new(),
        }
    }

    pub fn infer<I, S, M, R>(&mut self, name: impl Into<String>, factory: S) -> &mut Self
    where
        I: InferenceFactoryReturn<R>,
        I::Inference: Inference + Send + Sync + 'static,
        <I::Inference as Inference>::Input: DeserializeOwned + Send + Sync + 'static,
        <I::Inference as Inference>::Output: Serialize + Send + Sync + 'static,
        S: IntoRoutine<InferenceContext<B>, (), I, M> + 'static,
        M: 'static,
        R: 'static,
    {
        let name = name.into();
        let routine = Box::new(IntoRoutine::into_routine(factory));
        let factory = RoutineFactory {
            name: name.clone(),
            routine,
            _types: PhantomData,
        };

        self.factories.insert(name, Box::new(factory));
        self
    }

    pub fn build_inference(
        &self,
        name: impl AsRef<str>,
        init: InferenceInit<B>,
        args: Option<impl Into<InferenceArgs>>,
    ) -> Result<Box<dyn ErasedInference>, InferenceError> {
        let factory = self
            .factories
            .get(name.as_ref())
            .ok_or_else(|| InferenceError::NotFound(name.as_ref().to_string()))?;
        let args = args.map(|a| a.into()).unwrap_or_default();
        let ctx = InferenceContext::new(init, args);
        factory.build(ctx)
    }
}

/// Build a typed inference instance directly from a factory routine.
pub fn build_fleet_managed<B, I, M, R>(
    factory: impl IntoRoutine<InferenceContext<B>, (), I, M>,
    token: impl Into<FleetRegistrationToken>,
    metadata: impl Serialize,
    device: B::Device,
) -> Result<FleetManagedInference<B, I::Inference>, InferenceError>
where
    B: Backend,
    I: InferenceFactoryReturn<R>,
    I::Inference: Inference + Send + Sync + 'static,
    M: 'static,
    R: 'static,
{
    let metadata = serde_json::to_value(metadata).map_err(|e| InferenceError::FactoryFailed {
        name: "metadata serialization".to_string(),
        message: e.to_string(),
    })?;
    let fleet_session =
        fleet::register(token.into(), metadata, &Env::Development).map_err(|e| {
            InferenceError::FactoryFailed {
                name: "fleet registration".to_string(),
                message: e.to_string(),
            }
        })?;

    let routine = IntoRoutine::into_routine(factory);
    let inference_name = routine.name().to_string();
    let error_name = inference_name.clone();

    let inference_factory: Box<dyn InferenceFactory<B, I::Inference>> =
        Box::new(move |ctx: &mut InferenceContext<B>| {
            let factory_output =
                routine
                    .run((), ctx)
                    .map_err(|err| InferenceError::FactoryFailed {
                        name: error_name.clone(),
                        message: err.to_string(),
                    })?;
            let inference = factory_output.into_inference().map_err(|message| {
                InferenceError::FactoryFailed {
                    name: error_name.clone(),
                    message,
                }
            })?;
            Ok(inference)
        });

    let inference = FleetManagedInference::initialize(
        inference_name,
        fleet_session,
        inference_factory,
        device,
    )?;

    Ok(inference)
}

struct ActiveInference<I> {
    inference: I,
    model_version: String,
}

/// Inference wrapper that bootstraps burn-central features like fleet registration and telemetry on top of a typed inference implementation.
pub struct FleetManagedInference<B: Backend, I> {
    inference_name: String,
    fleet_session: RwLock<fleet::FleetDeviceSession>,
    factory: Box<dyn InferenceFactory<B, I>>,
    device: B::Device,
    active: ArcSwapOption<ActiveInference<I>>,
    reconcile_gate: Mutex<()>,
    last_sync_at: Mutex<Option<Instant>>,
    sync_interval: Duration,
}

impl<B, I> FleetManagedInference<B, I>
where
    B: Backend,
    I: Inference + Send + Sync + 'static,
{
    pub fn initialize(
        inference_name: impl Into<String>,
        fleet_session: fleet::FleetDeviceSession,
        factory: Box<dyn InferenceFactory<B, I>>,
        device: B::Device,
    ) -> Result<Self, InferenceError> {
        let inference = Self {
            inference_name: inference_name.into(),
            fleet_session: RwLock::new(fleet_session),
            factory,
            device,
            active: ArcSwapOption::empty(),
            reconcile_gate: Mutex::new(()),
            last_sync_at: Mutex::new(None),
            sync_interval: Duration::from_secs(30),
        };
        inference.ensure_ready()?;
        Ok(inference)
    }

    fn maybe_sync_and_rollout(&self) -> Result<(), InferenceError> {
        if self.active().is_some() && !self.should_sync_now() {
            return Ok(());
        }

        let _guard = self.reconcile_gate.lock().unwrap();
        if self.active().is_some() && !self.should_sync_now() {
            return Ok(());
        }

        let (fleet_version, model_source, config) = {
            let mut session = self.fleet_session.write().unwrap();

            match session.sync(None) {
                Ok(()) => {
                    let fleet_version = normalized_model_version(session.active_model_version_id());
                    let model_source =
                        session
                            .model_source()
                            .map_err(|err| InferenceError::FactoryFailed {
                                name: self.inference_name.clone(),
                                message: format!("fleet model source failed: {err}"),
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
                    let model_source =
                        session
                            .model_source()
                            .map_err(|cache_err| InferenceError::FactoryFailed {
                                name: self.inference_name.clone(),
                                message: format!(
                                    "fleet sync failed and no usable local cache: sync={sync_err}; cache={cache_err}"
                                ),
                            })?;
                    let config = session.runtime_config();

                    (fleet_version, model_source, config.clone())
                }
            }
        };

        self.mark_sync_now();

        let active = self.active();
        if active.as_ref().map(|a| &a.model_version) == Some(&fleet_version) {
            return Ok(());
        }

        let init = InferenceInit {
            model: model_source,
            device: self.device.clone(),
        };
        let mut ctx = InferenceContext::new(init, InferenceArgs::new(Some(config)));
        let built = self.factory.build(&mut ctx)?;

        self.active.store(Some(Arc::new(ActiveInference {
            inference: built,
            model_version: fleet_version.clone(),
        })));

        Ok(())
    }

    fn ensure_ready(&self) -> Result<(), InferenceError> {
        self.maybe_sync_and_rollout()?;
        if self.active().is_none() {
            return Err(InferenceError::FactoryFailed {
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
}

impl<B, I> Inference for FleetManagedInference<B, I>
where
    B: Backend,
    I: Inference + Send + Sync + 'static,
{
    type Input = <I as Inference>::Input;
    type Output = <I as Inference>::Output;

    fn infer(&self, input: Self::Input, writer: super::InferenceWriter<Self::Output>) {
        if let Err(err) = self.maybe_sync_and_rollout() {
            writer.error(Box::new(err)).ok();
            return;
        }

        let Some(active) = self.active() else {
            writer
                .error(Box::new(InferenceError::FactoryFailed {
                    name: self.inference_name.clone(),
                    message: "no active model".to_string(),
                }))
                .ok();
            return;
        };

        let metadata = InferenceMetadata::new(
            self.inference_name.clone(),
            "unknown".to_string(),
            active.model_version.clone(),
        );

        let writer =
            writer.with_observer(Arc::new(InferenceWriterTelemetryObserver::new(metadata)));
        active.inference.infer(input, writer)
    }
}

pub trait InferenceFactory<B: Backend, I>: Send + Sync {
    fn build(&self, ctx: &mut InferenceContext<B>) -> Result<I, InferenceError>;
}

impl<F, B, I> InferenceFactory<B, I> for F
where
    F: Fn(&mut InferenceContext<B>) -> Result<I, InferenceError> + Send + Sync,
    B: Backend,
{
    fn build(&self, ctx: &mut InferenceContext<B>) -> Result<I, InferenceError> {
        self(ctx)
    }
}

fn normalized_model_version(version: &str) -> String {
    if version.is_empty() {
        "unknown".to_string()
    } else {
        version.to_string()
    }
}
