use burn_central_core::burn::prelude::Backend;
use burn_central_core::experiment::ExperimentArgs;
use derive_more::{Deref, From};

#[derive(From, Deref)]
pub struct Args<T: ExperimentArgs>(pub T);
#[derive(Clone, Debug, Deref, From)]
pub struct MultiDevice<B: Backend>(pub Vec<B::Device>);
#[derive(Clone, From, Deref)]
pub struct Model<M>(pub M);
#[derive(Debug, Deref, From)]
pub struct In<T>(pub T);
#[derive(Debug, Deref, From)]
pub struct Out<T>(pub T);
#[derive(Debug, Deref, From)]
pub struct State<T>(pub T);
