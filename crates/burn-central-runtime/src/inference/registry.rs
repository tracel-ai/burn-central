use crate::inference::{ErasedInference, Inference, JsonInference};
use crate::params::RoutineParam;
use crate::params::args::{LaunchArgs, deserialize_and_merge_with_default};
use crate::routine::{BoxedRoutine, IntoRoutine};
use crate::{Args, MultiDevice};
use burn::prelude::Backend;
use burn_central_core::{BurnCentralCredentials, Env};
use burn_central_registry::{Registry, RegistryBuilder};
use serde::{Serialize, de::DeserializeOwned};
use std::collections::HashMap;
use std::marker::PhantomData;

#[derive(Debug, thiserror::Error)]
pub enum InferenceError {
    #[error("inference handler not found: {0}")]
    NotFound(String),
    #[error("inference handler '{name}' failed to initialize: {message}")]
    FactoryFailed { name: String, message: String },
}

#[derive(Clone)]
pub struct InferenceInit<B: Backend> {
    pub registry: Registry,
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

    pub fn registry(&self) -> &Registry {
        &self.init.registry
    }

    pub fn device(&self) -> &B::Device {
        &self.init.device
    }
}

impl<B: Backend> RoutineParam<InferenceContext<B>> for &Registry {
    type Item<'new> = &'new Registry;

    fn try_retrieve(ctx: &InferenceContext<B>) -> anyhow::Result<Self::Item<'_>> {
        Ok(ctx.registry())
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

pub fn build<B, I, S, M, R>(
    factory: S,
    creds: impl Into<BurnCentralCredentials>,
    args: Option<impl Into<InferenceArgs>>,
) -> Result<Box<dyn ErasedInference>, InferenceError>
where
    B: Backend,
    I: InferenceFactoryReturn<R>,
    I::Inference: Inference + Send + Sync + 'static,
    <I::Inference as Inference>::Input: DeserializeOwned + Send + Sync + 'static,
    <I::Inference as Inference>::Output: Serialize + Send + Sync + 'static,
    S: IntoRoutine<InferenceContext<B>, (), I, M> + 'static,
    M: 'static,
    R: 'static,
{
    let model_registry = RegistryBuilder::new(creds.into())
        .with_env(Env::Development)
        .build()
        .map_err(|e| InferenceError::FactoryFailed {
            name: "registry build".to_string(),
            message: e.to_string(),
        })?;
    let mut registry = InferenceRegistry::new();
    let n = tynm::type_name::<S>();
    registry.infer(&n, factory);

    let init = InferenceInit {
        registry: model_registry,
        device: B::Device::default(),
    };
    registry.build_inference(&n, init, args)
}
