use super::context::InferenceContext;
use super::context::InferenceExtensions;
use super::context::InferenceOutput;
use super::error::InferenceError;
use super::init::Init;
use super::job::JobHandle;
use super::streaming::{CancelToken, CollectEmitter, SyncChannelEmitter};
use crate::inference::InferenceJob;
use crate::inference::InferenceJobBuilder;
use crate::inference::StrappedInferenceJobBuilder;
use crate::inference::input::RoutineInput;
use crate::inference::model::ModelHost;
use crate::routine::ExecutorRoutineWrapper;
use crate::routine::IntoRoutine;
use crate::routine::Routine;
use burn::prelude::Backend;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

/// Internal type alias for a routine trait object representing the user supplied inference handler.
type ArcInferenceHandler<B, M, I, O, S> =
    Arc<dyn Routine<InferenceContext<B, M, O, S>, In = I, Out = ()>>;

/// Inference instance wrapping a single model and a handler routine.
///
/// An `Inference` can create multiple jobs (sequentially or concurrently) without re-loading the model.
pub struct Inference<B: Backend, M, I, O, S = ()> {
    pub id: String,
    model: ModelHost<M>,
    handler: ArcInferenceHandler<B, M, I, O, S>,
    extensions: Arc<InferenceExtensions>,
}

impl<B, M, I, O, S> Inference<B, M, I, O, S>
where
    B: Backend,
    M: Send + 'static,
    I: RoutineInput + 'static,
    O: Send + 'static,
    S: Send + Sync + 'static,
{
    pub(crate) fn new(
        id: String,
        handler: ArcInferenceHandler<B, M, I, O, S>,
        model: M,
        extensions: InferenceExtensions,
    ) -> Self {
        Self {
            id,
            model: ModelHost::spawn(model),
            handler,
            extensions: Arc::new(extensions),
        }
    }

    /// Start building an inference job for the given input payload.
    pub fn infer(
        &'_ self,
        input: I::Inner<'static>,
    ) -> StrappedInferenceJobBuilder<'_, B, M, I, O, S, super::builder::StateMissing> {
        StrappedInferenceJobBuilder {
            inference: self,
            input: InferenceJobBuilder::new(input),
        }
    }

    /// Execute the provided job synchronously and collect all emitted outputs.
    pub fn run(&self, job: InferenceJob<B, I, S>) -> Result<Vec<O>, InferenceError> {
        let collector = Arc::new(CollectEmitter::new());
        let input = job.input;
        let devices = job.devices;
        let state = job.state;
        {
            let mut ctx = InferenceContext {
                id: self.id.clone(),
                devices: devices.into_iter().collect(),
                model: self.model.accessor(),
                emitter: collector.clone(),
                cancel_token: CancelToken::new(),
                state: Mutex::new(Some(state)),
                extensions: self.extensions.clone(),
            };
            self.handler
                .run(input, &mut ctx)
                .map_err(|e| InferenceError::HandlerExecutionFailed(e.into()))?;
        }
        let stream = Arc::try_unwrap(collector)
            .map_err(|_| InferenceError::Unexpected("Failed to unwrap collector".to_string()))?
            .into_inner();
        Ok(stream)
    }

    /// Spawn the job on a background thread returning a [`JobHandle`]. Outputs can be read from the handle's stream.
    pub fn spawn(&self, job: super::builder::InferenceJob<B, I, S>) -> JobHandle<O>
    where
        <I as RoutineInput>::Inner<'static>: Send,
    {
        let id = self.id.clone();
        let input = job.input;
        let devices = job.devices;
        let state = job.state;
        let (stream_tx, stream_rx) = crossbeam::channel::unbounded();
        let cancel_token = CancelToken::new();
        let mut ctx = InferenceContext {
            id: id.clone(),
            devices: devices.into_iter().collect(),
            model: self.model.accessor(),
            emitter: Arc::new(SyncChannelEmitter::new(stream_tx)),
            cancel_token: cancel_token.clone(),
            state: Mutex::new(Some(state)),
            extensions: self.extensions.clone(),
        };
        let handler = self.handler.clone();
        let join = std::thread::spawn(move || {
            handler
                .run(input, &mut ctx)
                .map_err(|e| InferenceError::HandlerExecutionFailed(e.into()))
        });
        JobHandle::new(id, stream_rx, cancel_token, join)
    }

    /// Consume the inference instance and retrieve ownership of the underlying model.
    pub fn into_model(self) -> M {
        self.model.into_model()
    }
}

/// Entry point builder for an [`Inference`] instance.
pub struct InferenceBuilder<B> {
    phantom_data: PhantomData<B>,
    extensions: InferenceExtensions,
}

impl<B: Backend> Default for InferenceBuilder<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> InferenceBuilder<B> {
    /// Create a new inference builder.
    pub fn new() -> Self {
        Self {
            phantom_data: Default::default(),
            extensions: InferenceExtensions::new(),
        }
    }

    /// Attach a shared extension available to handlers via `Extension<T>`.
    pub fn with_extension<T: Send + Sync + 'static>(mut self, extension: Arc<T>) -> Self {
        self.extensions.insert(extension);
        self
    }

    /// Initialize a model implementing [`Init`] from user artifacts / arguments + target device.
    pub fn init<M, InitArgs>(
        self,
        args: &InitArgs,
        device: &B::Device,
    ) -> Result<LoadedInferenceBuilder<B, M>, M::Error>
    where
        M: Init<B, InitArgs>,
        InitArgs: Send + 'static,
    {
        let model = M::init(args, device)?;
        Ok(LoadedInferenceBuilder {
            model,
            phantom_data: Default::default(),
            extensions: self.extensions,
        })
    }

    /// Provide an already constructed model instance (skips the [`Init`] flow).
    pub fn with_model<M>(self, model: M) -> LoadedInferenceBuilder<B, M> {
        LoadedInferenceBuilder {
            model,
            phantom_data: Default::default(),
            extensions: self.extensions,
        }
    }
}

/// Builder returned after a model has been loaded or supplied ready for registering a handler.
pub struct LoadedInferenceBuilder<B: Backend, M> {
    model: M,
    phantom_data: PhantomData<B>,
    extensions: InferenceExtensions,
}

impl<B, M> LoadedInferenceBuilder<B, M>
where
    B: Backend,
    M: Send + 'static,
{
    /// Attach a shared extension available to handlers via `Extension<T>`.
    pub fn with_extension<T: Send + Sync + 'static>(mut self, extension: Arc<T>) -> Self {
        self.extensions.insert(extension);
        self
    }

    /// Finalize the construction of an [`Inference`] by supplying a handler routine implementation.
    pub fn build<F, I, O, RO, Marker, S>(self, handler: F) -> Inference<B, M, I, O, S>
    where
        F: IntoRoutine<InferenceContext<B, M, O, S>, I, RO, Marker>,
        I: RoutineInput + 'static,
        O: Send + 'static,
        S: Send + Sync + 'static,
        RO: InferenceOutput<B, M, O, S> + Sync + 'static,
    {
        Inference::new(
            crate::type_name::fn_type_name::<F>(),
            Arc::new(ExecutorRoutineWrapper::new(IntoRoutine::into_routine(
                handler,
            ))),
            self.model,
            self.extensions,
        )
    }
}
