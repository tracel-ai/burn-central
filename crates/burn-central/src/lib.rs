// #![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs)]

//! # Burn Central SDK
//!
//! This crate allow you to bridge your burn model training and inference with Burn Central platform.
//!
//! Features:
//! - Artifact createion and management
//! - Experiment tracking
//! - Logging metrics
//! - Model versioning
//!
//! ## Components overview
//!
//! This crate is composed of three main modules:
//! - `core`: The core module that handle all interaction with Burn Central platform API.
//! - `macros`: The macros module that provide the necessary macros to register training and inference functions.
//! - `runtime`: The runtime module that execute the registered training and inference functions.
//!
//!  If you are a user you might want to checkout the documentation around the `core` crates as
//!  this is where you will find example on how to use feature of this SDK.
//!
//!  ### Runtime
//!  For the `runtime`, the only thing you need to know is that we use it to wrape the function
//!  you want to use in another crates that we control allowing us to inject more easily what you
//!  need into your function and to interact with Burn Central platform before and after the
//!  execution of your training function.
//!
//!
//!  ### Macros
//!  For the `macros`, you only need to know about the register macro that you will use to mark
//!  your training or inference functions so they can be found by Burn Central CLI. Here is
//!
//!  #### Example
//!  ```ignore
//!  use burn_central::macros::register;
//!
//!  #[register(training, name = "my_training_procedure")]
//!  async fn my_training_function() {
//!   // Your training code here
//!  }
//!  ```
//!  Note that the name attribute is optional. If not provided the function name will be used.
//!
//!

pub use burn_central_core::*;

/// This crate provide the register macros. It allow user to mark there training functions so they
/// can be found by Burn Central CLI.
#[doc(inline)]
pub use burn_central_macros as macros;

/// The runtime crate execute training and inference procedure registered with
/// Burn Central Macros. It basically form a wrapper crate that use your declare functions.
#[doc(inline)]
pub use burn_central_runtime as runtime;

/// Local registry/cache helpers for downloading models.
#[doc(inline)]
pub use burn_central_registry as registry;
