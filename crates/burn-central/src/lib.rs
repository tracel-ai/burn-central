// #![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]

//! # Tracel

/// Burn Central Client
pub use burn_central_core::*;

/// Burn Central macros
pub mod macros {
    pub use burn_central_macros::register;
}

/// Burn Central Runtime
pub mod runtime {
    pub use burn_central_runtime::*;
}
