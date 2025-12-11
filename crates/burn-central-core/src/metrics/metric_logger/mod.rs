#[cfg(feature = "burn_0_20")]
mod v0_20;

#[cfg(feature = "burn_0_19")]
mod v0_19;

#[cfg(feature = "burn_0_20")]
pub use v0_20::*;

#[cfg(feature = "burn_0_19")]
pub use v0_19::*;
