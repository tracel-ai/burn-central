use crate::inference::input::RoutineInput;

use super::core::Inference;
use super::job::JobHandle;
use burn::prelude::Backend;
use std::marker::PhantomData;

/// Builder returned by [`Inference::infer`] allowing configuration (devices, state) before
/// executing the job via [`StrappedInferenceJobBuilder::run`] or spawning via [`StrappedInferenceJobBuilder::spawn`].
pub struct StrappedInferenceJobBuilder<'a, B: Backend, M, I: RoutineInput, O, S, Flag> {
    pub(crate) inference: &'a Inference<B, M, I, O, S>,
    pub(crate) input: InferenceJobBuilder<B, I, S, Flag>,
}

impl<'a, B, M, I, O, S, Flag> StrappedInferenceJobBuilder<'a, B, M, I, O, S, Flag>
where
    B: Backend,
    M: Send + 'static,
    I: RoutineInput + 'static,
    O: Send + 'static,
    S: Send + Sync + 'static,
{
    /// Specify the devices to be exposed to the handler (order preserved; first often primary).
    pub fn with_devices(mut self, devices: impl IntoIterator<Item = B::Device>) -> Self {
        self.input = self.input.with_devices(devices);
        self
    }
}

impl<'a, B, M, I, O, S> StrappedInferenceJobBuilder<'a, B, M, I, O, S, StateMissing>
where
    B: Backend,
    M: Send + 'static,
    I: RoutineInput + 'static,
    O: Send + 'static,
    S: Send + Sync + 'static,
{
    /// Provide state to the handler. Consumed exactly once during handler execution.
    pub fn with_state(
        self,
        state: S,
    ) -> StrappedInferenceJobBuilder<'a, B, M, I, O, S, StateProvided> {
        StrappedInferenceJobBuilder {
            inference: self.inference,
            input: self.input.with_state(state),
        }
    }
}

/// Internal job builder accumulating input + devices + optional state before conversion to an executable job.
pub struct InferenceJobBuilder<B: Backend, I: RoutineInput, S, Flag> {
    pub(crate) input: <I as RoutineInput>::Inner<'static>,
    pub(crate) devices: Vec<B::Device>,
    pub(crate) state: Option<S>,
    _flag: PhantomData<Flag>,
}

impl<B, I, S, Flag> InferenceJobBuilder<B, I, S, Flag>
where
    B: Backend,
    I: RoutineInput + 'static,
    S: Send + Sync + 'static,
{
    /// Create a new job builder with the provided routine input payload.
    pub fn new(input: <I as RoutineInput>::Inner<'static>) -> Self {
        Self {
            input,
            devices: Vec::new(),
            state: None,
            _flag: PhantomData,
        }
    }

    /// Set the devices collection for this job.
    pub fn with_devices(mut self, devices: impl IntoIterator<Item = B::Device>) -> Self {
        self.devices = devices.into_iter().collect();
        self
    }
}

/// Marker type indicating the job state has not been supplied.
pub struct StateMissing;
/// Marker type indicating the job state has been supplied.
pub struct StateProvided;

impl<B, I, S> InferenceJobBuilder<B, I, S, StateMissing>
where
    B: Backend,
    I: RoutineInput + 'static,
    S: Send + Sync + 'static,
{
    /// Attach state to the job; transitions the builder into the `StateProvided` phase.
    pub fn with_state(self, state: S) -> InferenceJobBuilder<B, I, S, StateProvided> {
        InferenceJobBuilder {
            input: self.input,
            devices: self.devices,
            state: Some(state),
            _flag: PhantomData,
        }
    }
}

impl<B, I, S> InferenceJobBuilder<B, I, S, StateProvided>
where
    B: Backend,
    I: RoutineInput + 'static,
    S: Send + Sync + 'static,
{
    /// Finalize the builder into an [`InferenceJob`]. Panics if state missing (by design of type-state pattern).
    pub fn build(self) -> InferenceJob<B, I, S> {
        InferenceJob {
            input: self.input,
            devices: self.devices,
            state: self.state.expect("state must be set"),
        }
    }
}

impl<'a, B, M, I, O> StrappedInferenceJobBuilder<'a, B, M, I, O, (), StateMissing>
where
    B: Backend,
    M: Send + 'static,
    I: RoutineInput + 'static,
    O: Send + 'static,
{
    /// Spawn the inference job on a background thread returning a [`JobHandle`].
    pub fn spawn(self) -> JobHandle<O>
    where
        <I as RoutineInput>::Inner<'static>: Send,
    {
        let job = InferenceJob {
            input: self.input.input,
            devices: self.input.devices,
            state: (),
        };
        self.inference.spawn(job)
    }

    /// Run the inference job to completion collecting all outputs eagerly.
    pub fn run(self) -> Result<Vec<O>, super::error::InferenceError> {
        let job = InferenceJob {
            input: self.input.input,
            devices: self.input.devices,
            state: (),
        };
        self.inference.run(job)
    }
}

impl<'a, B, M, I, O, S> StrappedInferenceJobBuilder<'a, B, M, I, O, S, StateProvided>
where
    B: Backend,
    M: Send + 'static,
    I: RoutineInput + 'static,
    O: Send + 'static,
    S: Send + Sync + 'static,
{
    /// Spawn the inference job with provided user state.
    pub fn spawn(self) -> JobHandle<O>
    where
        <I as RoutineInput>::Inner<'static>: Send,
    {
        let job = InferenceJob {
            input: self.input.input,
            devices: self.input.devices,
            state: self.input.state.expect("state must be set"),
        };
        self.inference.spawn(job)
    }

    /// Run the inference job to completion (stateful variant) collecting all outputs.
    pub fn run(self) -> Result<Vec<O>, super::error::InferenceError> {
        let job = InferenceJob {
            input: self.input.input,
            devices: self.input.devices,
            state: self.input.state.expect("state must be set"),
        };
        self.inference.run(job)
    }
}

/// Concrete job containing fully specified execution parameters passed to the runtime.
pub struct InferenceJob<B: Backend, I: RoutineInput, S> {
    pub(crate) input: <I as RoutineInput>::Inner<'static>,
    pub(crate) devices: Vec<B::Device>,
    pub(crate) state: S,
}

impl<B, I, S> InferenceJob<B, I, S>
where
    B: Backend,
    I: RoutineInput + 'static,
    S: Send + Sync + 'static,
{
    /// Create a new builder for an inference job for the given input payload.
    pub fn builder(
        input: <I as RoutineInput>::Inner<'static>,
    ) -> InferenceJobBuilder<B, I, S, StateMissing> {
        InferenceJobBuilder::new(input)
    }
}
