use derive_more::{Deref, From};
use std::sync::Arc;

#[allow(dead_code)]
#[derive(Debug, Deref, From)]
pub struct In<T>(pub T);
#[derive(Debug, Deref, From)]
pub struct Out<T>(pub T);
#[allow(dead_code)]
#[derive(Debug, Deref, From)]
pub struct State<T>(pub T);
#[allow(dead_code)]
#[derive(Clone, Deref, From)]
pub struct Extension<T>(pub Arc<T>);

use crate::{executor::ExecutionContext, params::RoutineParam};
use anyhow::Result;
use burn::prelude::Backend;
use burn_central_core::experiment::ExperimentRun;

impl<B: Backend> RoutineParam<ExecutionContext<B>> for &ExecutionContext<B> {
    type Item<'new> = &'new ExecutionContext<B>;

    fn try_retrieve(ctx: &ExecutionContext<B>) -> Result<Self::Item<'_>> {
        Ok(ctx)
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
