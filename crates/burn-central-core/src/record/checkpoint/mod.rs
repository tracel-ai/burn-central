#[cfg(any(feature = "burn_0_19", feature = "burn_0_20"))]
mod v1;

#[cfg(any(feature = "burn_0_19", feature = "burn_0_20"))]
pub use v1::*;
