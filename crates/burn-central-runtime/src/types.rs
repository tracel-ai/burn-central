use burn::prelude::Backend;
use burn_central_core::experiment::ExperimentArgs;
use derive_more::{Deref, From};

/// Args are wrapper around the config you want to inject.
///
/// The type T must implement [ExperimentArgs] trait. This trait allow us to override the
/// configuration from the CLI arguments you can specify while given us a fallback for arguments
/// you don't provide.
#[derive(From, Deref)]
pub struct Args<T: ExperimentArgs>(pub T);

/// Wrapper around multiple devices.
///
/// Since Burn Central CLI support selecting different backend on the fly. We handle the device
/// selection in the generated crate. This structure is simply a marker for us to know where to
/// inject the devices selected by the CLI.
#[derive(Clone, Debug, Deref, From)]
pub struct MultiDevice<B: Backend>(pub Vec<B::Device>);

/// Wrapper around the model returned by a routine.
///
/// This is used to differentiate the model from other return types.
/// Right now the macro force you to return a Model as we expect to be able to log it as a model
/// artifact.
#[derive(Clone, From, Deref)]
pub struct Model<M>(pub M);

#[allow(dead_code)]
#[derive(Debug, Deref, From)]
pub struct In<T>(pub T);
#[derive(Debug, Deref, From)]
pub struct Out<T>(pub T);
#[allow(dead_code)]
#[derive(Debug, Deref, From)]
pub struct State<T>(pub T);
