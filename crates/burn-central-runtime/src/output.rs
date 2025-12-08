use crate::ExecutionContext;
use crate::types::Model;
use burn_central_core::artifacts::ArtifactKind;
use burn_central_core::bundle::BundleEncode;
use burn_central_core::burn::prelude::Backend;
use std::fmt::Display;

/// This trait defines how a specific return type (Output) from a handler apply its effects to the execution context.
pub trait RoutineOutput<Ctx>: Sized + Send + 'static {
    /// This method takes the owned output and the mutable ExecutionContext,
    /// allowing the output to modify the context.
    fn apply_output(self, ctx: &mut Ctx) -> anyhow::Result<()>;
}

pub trait ExperimentOutput<B: Backend>: RoutineOutput<ExecutionContext<B>> {}

/// This trait is a marker for outputs that are specifically related to training routines.
pub trait TrainOutput<B: Backend>: ExperimentOutput<B> {}

/// This implementation is for the case where the output is simply `()`, meaning no output to apply.
impl<B: Backend> RoutineOutput<ExecutionContext<B>> for () {
    fn apply_output(self, _ctx: &mut ExecutionContext<B>) -> anyhow::Result<()> {
        Ok(())
    }
}

impl<T, E, Ctx> RoutineOutput<Ctx> for Result<T, E>
where
    T: RoutineOutput<Ctx>,
    E: Display + Send + Sync + 'static,
{
    fn apply_output(self, ctx: &mut Ctx) -> anyhow::Result<()> {
        match self {
            Ok(output) => Ok(output.apply_output(ctx)?),
            Err(e) => Err(anyhow::anyhow!(e.to_string())),
        }
    }
}

impl<B: Backend, M: BundleEncode + Send + 'static> RoutineOutput<ExecutionContext<B>> for Model<M> {
    fn apply_output(self, ctx: &mut ExecutionContext<B>) -> anyhow::Result<()> {
        if let Some(experiment) = ctx.experiment() {
            experiment.log_artifact("model", ArtifactKind::Model, self.0, &Default::default())?;
        }
        Ok(())
    }
}

impl<B: Backend, M: BundleEncode + Send + 'static> ExperimentOutput<B> for Model<M> {}

impl<B: Backend> ExperimentOutput<B> for () {}

/// --- TrainOutput ---
impl<B: Backend, M: BundleEncode + Send + 'static> TrainOutput<B> for Model<M> {}

impl<T, E, B: Backend> ExperimentOutput<B> for Result<T, E>
where
    E: 'static + Display + Send + Sync,
    T: TrainOutput<B>,
{
}

impl<T, E, B: Backend> TrainOutput<B> for Result<T, E>
where
    T: TrainOutput<B>,
    E: Display + Send + Sync + 'static,
{
}

impl<B: Backend> TrainOutput<B> for () {}
