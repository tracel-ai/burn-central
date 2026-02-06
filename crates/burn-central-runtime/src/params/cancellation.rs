use burn::prelude::Backend;
use burn_central_core::experiment::CancelToken;

use crate::{executor::ExecutionContext, params::RoutineParam};

impl<B: Backend> RoutineParam<ExecutionContext<B>> for CancelToken {
    type Item<'new> = CancelToken;

    fn try_retrieve(ctx: &ExecutionContext<B>) -> anyhow::Result<Self::Item<'_>> {
        Ok(ctx.cancel_token().clone())
    }
}
