use crate::inference::fleet::{self, FleetRegistrationToken};
use crate::inference::telemetry::{InferenceMetadata, InferenceWriterTelemetryObserver};
use crate::inference::{ErasedInference, Inference, JsonInference};
use crate::params::RoutineParam;
use crate::params::args::{LaunchArgs, deserialize_and_merge_with_default};
use crate::routine::{BoxedRoutine, IntoRoutine, Routine};
use crate::{Args, MultiDevice};
use arc_swap::{ArcSwap, ArcSwapOption};
use burn::prelude::Backend;
use burn_central_core::bundle::{BundleDecode, FsBundleReader};
use serde::{Serialize, de::DeserializeOwned};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};

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
    pub fn dummy() -> Self {
        Self {
            root: PathBuf::new(),
            files: vec![],
        }
    }
}

impl ModelSource {
    pub fn load<D: BundleDecode>(&self, settings: D::Settings) -> Result<D, D::Error> {
        let reader = FsBundleReader::new(self.root.clone(), self.files.clone());
        D::decode(&reader, &settings)
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
        fleet::register(token.into(), metadata).map_err(|e| InferenceError::FactoryFailed {
            name: "fleet registration".to_string(),
            message: e.to_string(),
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

    let inference =
        FleetManagedInference::new(inference_name, fleet_session, inference_factory, device);

    Ok(inference)
}

/// Inference wrapper that bootstraps burn-central features like fleet registration and telemetry on top of a typed inference implementation.
pub struct FleetManagedInference<B: Backend, I> {
    inference_name: String,
    fleet_session: RwLock<fleet::FleetDeviceSession>,
    factory: Box<dyn InferenceFactory<B, I>>,
    active: ArcSwapOption<I>,
    active_model_version: ArcSwap<String>,
    staging: RwLock<StagingSlot<I>>,
    active_build_gate: Mutex<()>,
    staging_build_gate: Mutex<()>,
}

struct StagingSlot<I> {
    inference: Option<Arc<I>>,
    model_version: Option<String>,
}

impl<I> Default for StagingSlot<I> {
    fn default() -> Self {
        Self {
            inference: None,
            model_version: None,
        }
    }
}

impl<B, I> FleetManagedInference<B, I>
where
    B: Backend,
    I: Inference + Send + Sync + 'static,
{
    pub fn new(
        inference_name: impl Into<String>,
        fleet_session: fleet::FleetDeviceSession,
        factory: Box<dyn InferenceFactory<B, I>>,
        _device: B::Device,
    ) -> Self {
        let active_model_version =
            normalized_model_version(&fleet_session.state.active_model_version_id);

        Self {
            inference_name: inference_name.into(),
            fleet_session: RwLock::new(fleet_session),
            factory,
            active: ArcSwapOption::empty(),
            active_model_version: ArcSwap::from_pointee(active_model_version),
            staging: RwLock::new(StagingSlot::default()),
            active_build_gate: Mutex::new(()),
            staging_build_gate: Mutex::new(()),
        }
    }

    /// Build and cache the active model if needed, then return an `Arc` snapshot.
    ///
    /// The expensive build work is serialized and done without holding the slots lock.
    pub fn get_or_build(&self, ctx: &mut InferenceContext<B>) -> Result<Arc<I>, InferenceError> {
        if let Some(active) = self.active() {
            return Ok(active);
        }

        let _guard = self.active_build_gate.lock().unwrap();

        if let Some(active) = self.active() {
            return Ok(active);
        }

        let built = Arc::new(self.factory.build(ctx)?);
        let version = self.current_active_model_version_from_fleet();
        self.active.store(Some(built.clone()));
        self.active_model_version.store(Arc::new(version));

        Ok(built)
    }

    /// Build a new staging model version without touching currently active traffic.
    pub fn prepare_staging(
        &self,
        ctx: &mut InferenceContext<B>,
        model_version: impl Into<String>,
    ) -> Result<(), InferenceError> {
        let _guard = self.staging_build_gate.lock().unwrap();
        let built = Arc::new(self.factory.build(ctx)?);
        let model_version = normalized_model_version(&model_version.into());

        let mut staging = self.staging.write().unwrap();
        staging.inference = Some(built);
        staging.model_version = Some(model_version);
        Ok(())
    }

    /// Atomically promote staging to active.
    ///
    /// Returns `true` when promotion happened and `false` when no staging model exists.
    pub fn promote_staging(&self) -> bool {
        let (staging, staged_version) = {
            let mut staging = self.staging.write().unwrap();
            (staging.inference.take(), staging.model_version.take())
        };

        let Some(staging) = staging else {
            return false;
        };

        let version =
            staged_version.unwrap_or_else(|| self.current_active_model_version_from_fleet());
        self.active.store(Some(staging));
        self.active_model_version.store(Arc::new(version.clone()));

        let mut session = self.fleet_session.write().unwrap();
        session.state.active_model_version_id = version;

        true
    }

    /// Drop staging if present.
    pub fn rollback_staging(&self) -> bool {
        let mut staging = self.staging.write().unwrap();
        let had_staging = staging.inference.take().is_some();
        staging.model_version = None;
        had_staging
    }

    /// Get an `Arc` snapshot of the currently active model.
    pub fn active(&self) -> Option<Arc<I>> {
        self.active.load_full()
    }

    /// Current active model version id or `"unknown"`.
    pub fn active_model_version(&self) -> String {
        self.active_model_version.load_full().as_ref().to_string()
    }

    fn current_active_model_version_from_fleet(&self) -> String {
        let session = self.fleet_session.read().unwrap();
        normalized_model_version(&session.state.active_model_version_id)
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
        let active = self.active();
        let model_version = self.active_model_version();

        let metadata = InferenceMetadata::new(
            self.inference_name.clone(),
            "unknown".to_string(),
            model_version,
        );

        match active {
            Some(inference) => {
                let writer =
                    writer.with_observer(Arc::new(InferenceWriterTelemetryObserver::new(metadata)));
                inference.infer(input, writer)
            }
            None => {
                // This should never happen since `get_or_build` should be called before inference.
                writer
                    .error(Box::new(InferenceError::FactoryFailed {
                        name: self.inference_name.clone(),
                        message: "inference not built".to_string(),
                    }))
                    .ok();
            }
        }
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
