use burn_central_core::burn::prelude::Backend;

/// Trait for models that can be initialized from user-defined arguments.
pub trait Init<B, InitArgs = ()>: Sized
where
    B: Backend,
    InitArgs: Send + 'static,
{
    type Error;
    /// Initialize the model from the given arguments and device.
    fn init(args: &InitArgs, device: &B::Device) -> Result<Self, Self::Error>;
}
