use super::streaming::{CancelToken, Emitter, OutStream};
use crate::MultiDevice;
use crate::inference::model::ModelAccessor;
use crate::inference::{Extension, Out, State};
use crate::output::RoutineOutput;
use crate::params::RoutineParam;
use burn::prelude::Backend;
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt::Display;
use std::sync::{Arc, Mutex};

/// Type-indexed shared extension map used to inject runtime-wide values into handlers.
#[derive(Default)]
pub struct InferenceExtensions {
    values: HashMap<TypeId, Arc<dyn Any + Send + Sync>>,
}

impl InferenceExtensions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert<T: Send + Sync + 'static>(&mut self, value: Arc<T>) {
        self.values.insert(TypeId::of::<T>(), value);
    }

    pub fn get<T: Send + Sync + 'static>(&self) -> Option<Arc<T>> {
        self.values.get(&TypeId::of::<T>()).and_then(|value| {
            let value = value.clone();
            Arc::downcast::<T>(value).ok()
        })
    }
}

/// Runtime context passed to the user handler providing access to the model, devices,
/// streaming emitter, cancellation token and (optional) user state.
pub struct InferenceContext<B: Backend, M, O, S> {
    pub id: String,
    pub devices: Vec<B::Device>,
    pub model: ModelAccessor<M>,
    pub emitter: Arc<dyn Emitter<O>>,
    pub cancel_token: CancelToken,
    pub state: Mutex<Option<S>>,
    pub extensions: Arc<InferenceExtensions>,
}

// --- Params

// Implementations for extracting parameters from InferenceContext
impl<B: Backend, M, O, S> RoutineParam<InferenceContext<B, M, O, S>> for CancelToken {
    type Item<'new>
        = CancelToken
    where
        B: 'new,
        M: 'new,
        O: 'new,
        S: 'new;

    fn try_retrieve(ctx: &InferenceContext<B, M, O, S>) -> anyhow::Result<Self::Item<'_>> {
        Ok(ctx.cancel_token.clone())
    }
}

impl<B: Backend, M, O, S> RoutineParam<InferenceContext<B, M, O, S>> for OutStream<O> {
    type Item<'new>
        = OutStream<O>
    where
        B: 'new,
        M: 'new,
        O: 'new,
        S: 'new;

    fn try_retrieve(ctx: &InferenceContext<B, M, O, S>) -> anyhow::Result<Self::Item<'_>> {
        Ok(OutStream::new(ctx.emitter.clone()))
    }
}

impl<B: Backend, M, O, S> RoutineParam<InferenceContext<B, M, O, S>> for ModelAccessor<M> {
    type Item<'new>
        = ModelAccessor<M>
    where
        B: 'new,
        M: 'new,
        O: 'new,
        S: 'new;

    fn try_retrieve(ctx: &InferenceContext<B, M, O, S>) -> anyhow::Result<Self::Item<'_>> {
        Ok(ctx.model.clone())
    }
}

impl<B: Backend, M, O, S> RoutineParam<InferenceContext<B, M, O, S>> for MultiDevice<B> {
    type Item<'new>
        = MultiDevice<B>
    where
        B: 'new,
        M: 'new,
        O: 'new,
        S: 'new;

    fn try_retrieve(ctx: &InferenceContext<B, M, O, S>) -> anyhow::Result<Self::Item<'_>> {
        Ok(MultiDevice(ctx.devices.clone()))
    }
}

impl<B: Backend, M, O, S> RoutineParam<InferenceContext<B, M, O, S>> for State<S> {
    type Item<'new>
        = State<S>
    where
        B: 'new,
        M: 'new,
        O: 'new,
        S: 'new;

    fn try_retrieve(ctx: &InferenceContext<B, M, O, S>) -> anyhow::Result<Self::Item<'_>> {
        Ok(State(ctx.state.lock().unwrap().take().ok_or_else(
            || anyhow::anyhow!("State has already been taken or was not provided"),
        )?))
    }
}

impl<B: Backend, M, O, S, T> RoutineParam<InferenceContext<B, M, O, S>> for Extension<T>
where
    T: Send + Sync + 'static,
{
    type Item<'new>
        = Extension<T>
    where
        B: 'new,
        M: 'new,
        O: 'new,
        S: 'new;

    fn try_retrieve(ctx: &InferenceContext<B, M, O, S>) -> anyhow::Result<Self::Item<'_>> {
        ctx.extensions
            .get::<T>()
            .map(Extension)
            .ok_or_else(|| anyhow::anyhow!("Extension not found: {}", std::any::type_name::<T>()))
    }
}

// --- Outputs
/// This trait is used for outputs that are specifically related to inference routines.
pub trait InferenceOutput<B: Backend, M, O, S>:
    RoutineOutput<InferenceContext<B, M, O, S>>
{
}

impl<B: Backend, M, O, S> RoutineOutput<InferenceContext<B, M, O, S>> for () {
    fn apply_output(self, _ctx: &mut InferenceContext<B, M, O, S>) -> anyhow::Result<Self> {
        Ok(())
    }
}

impl<B: Backend, M, O, S> InferenceOutput<B, M, O, S> for () {}

impl<B: Backend, M, T, S> RoutineOutput<InferenceContext<B, M, T, S>> for Out<T>
where
    T: Send + 'static,
{
    fn apply_output(self, ctx: &mut InferenceContext<B, M, T, S>) -> anyhow::Result<()> {
        match ctx.emitter.emit(self.0) {
            Ok(()) => Ok(()),
            Err(e) => Err(anyhow::anyhow!("Failed to emit output: {}", e.source)),
        }
    }
}

impl<B: Backend, M, T, S> InferenceOutput<B, M, T, S> for Out<T> where T: Send + 'static {}

impl<B: Backend, M, O, T, E, S> InferenceOutput<B, M, O, S> for Result<T, E>
where
    T: InferenceOutput<B, M, O, S>,
    E: Display + Send + Sync + 'static,
{
}
