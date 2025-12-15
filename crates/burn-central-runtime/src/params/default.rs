use crate::{executor::ExecutionContext, params::RoutineParam};
use burn::prelude::Backend;
use derive_more::{Deref, From};

/// Wrapper around multiple devices.
///
/// Since Burn Central CLI support selecting different backend on the fly. We handle the device
/// selection in the generated crate. This structure is simply a marker for us to know where to
/// inject the devices selected by the CLI.
///
/// We are planning to support multi device training in the future, however we currenly only
/// support one so this vector will always contains one device for now.
#[derive(Clone, Debug, Deref, From)]
pub struct MultiDevice<B: Backend>(pub Vec<B::Device>);

/// Wrapper around the model returned by a routine.
///
/// This is used to differentiate the model from other return types.
/// Right now the macro force you to return a Model as we expect to be able to log it as a model
/// artifact.
#[derive(Clone, From, Deref)]
pub struct Model<M>(pub M);

impl<B: Backend> RoutineParam<ExecutionContext<B>> for MultiDevice<B> {
    type Item<'new> = MultiDevice<B>;

    fn try_retrieve(ctx: &ExecutionContext<B>) -> anyhow::Result<Self::Item<'_>> {
        Ok(MultiDevice(ctx.devices().into()))
    }
}
