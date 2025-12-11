use crate::{
    executor::ExecutionContext,
    types::{Args, Model, MultiDevice},
};
use anyhow::Result;
use burn::module::Module;
use burn::prelude::Backend;
use burn_central_core::experiment::{ExperimentArgs, ExperimentRun};
use variadics_please::all_tuples;

/// This trait defines how parameters for a routine are retrieved from the execution context.
pub trait RoutineParam<Ctx>: Sized {
    type Item<'new>
    where
        Ctx: 'new;

    /// This method retrieves the parameter from the context.
    fn retrieve(ctx: &Ctx) -> Self::Item<'_> {
        Self::try_retrieve(ctx).unwrap()
    }

    /// This method attempts to retrieve the parameter from the context, returning an error if it fails.
    fn try_retrieve(ctx: &Ctx) -> Result<Self::Item<'_>>;
}

impl<B: Backend> RoutineParam<ExecutionContext<B>> for &ExecutionContext<B> {
    type Item<'new> = &'new ExecutionContext<B>;

    fn try_retrieve(ctx: &ExecutionContext<B>) -> Result<Self::Item<'_>> {
        Ok(ctx)
    }
}

impl<B: Backend, C: ExperimentArgs> RoutineParam<ExecutionContext<B>> for Args<C> {
    type Item<'new> = Args<C>;

    fn try_retrieve(ctx: &ExecutionContext<B>) -> Result<Self::Item<'_>> {
        let cfg = ctx.use_merged_args();
        Ok(Args(cfg))
    }
}

impl<B: Backend, M: Module<B> + Default> RoutineParam<ExecutionContext<B>> for Model<M> {
    type Item<'new> = Model<M>;

    fn try_retrieve(_ctx: &ExecutionContext<B>) -> Result<Self::Item<'_>> {
        // Assuming we have a way to get the model from the context
        // For simplicity, let's just return a default model here
        let model = M::default();
        Ok(Model(model))
    }
}

impl<B: Backend> RoutineParam<ExecutionContext<B>> for MultiDevice<B> {
    type Item<'new> = MultiDevice<B>;

    fn try_retrieve(ctx: &ExecutionContext<B>) -> Result<Self::Item<'_>> {
        Ok(MultiDevice(ctx.devices().into()))
    }
}

impl<B: Backend> RoutineParam<ExecutionContext<B>> for &ExperimentRun {
    type Item<'new> = &'new ExperimentRun;

    fn try_retrieve(ctx: &ExecutionContext<B>) -> Result<Self::Item<'_>> {
        ctx.experiment()
            .ok_or_else(|| anyhow::anyhow!("Experiment run not found"))
    }
}

impl<Ctx, P: RoutineParam<Ctx>> RoutineParam<Ctx> for Option<P> {
    type Item<'new>
        = Option<P::Item<'new>>
    where
        Ctx: 'new;

    fn try_retrieve(ctx: &Ctx) -> Result<Self::Item<'_>> {
        match P::try_retrieve(ctx) {
            Ok(item) => Ok(Some(item)),
            Err(_) => Ok(None),
        }
    }
}

macro_rules! impl_routine_param_tuple {
    ($($P:ident),*) => {
        #[expect(
            clippy::allow_attributes,
            reason = "This is in a macro, and as such, the below lints may not always apply."
        )]
        #[allow(
            non_snake_case,
            reason = "Certain variable names are provided by the caller, not by us."
        )]
        #[allow(
            unused_variables,
            reason = "Zero-length tuples won't use some of the parameters."
        )]
        impl<Ctx, $($P: RoutineParam<Ctx>),*> RoutineParam<Ctx> for ($($P,)*) {
            type Item<'new> = ($($P::Item<'new>,)*) where Ctx: 'new;

            fn try_retrieve<'r>(ctx: &'r Ctx) -> Result<Self::Item<'r>> {
                Ok((
                    $(<$P as RoutineParam<Ctx>>::try_retrieve(ctx)?,)*
                ))
            }
        }
    };
}

all_tuples!(impl_routine_param_tuple, 0, 16, P);
