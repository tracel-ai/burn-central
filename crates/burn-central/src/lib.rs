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
//!  this is where you will find exemple on how to use feature of this SDK.
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
//!  your training functions so they can be found by Burn Central CLI. Right now we only support
//!  the training registration but inference will come soon. Here is
//!
//!  #### Example
//!  ```rust
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

/// This modules is the heart of the SDK as it interface with the Burn Central platform API.
pub mod core {
    pub use burn_central_core::*;
}

/// This crate provide the register macros. It allow user to mark there training functions so they
/// can be found by Burn Central CLI.
pub mod macros {
    //! # Burn Central Macros
    //! As define in the burn central crate documentation, this crate provide the macros to
    //! register a functions. You probably don't need more information to it then that, but if you
    //! do we got you covered.
    //!
    //! ## Role and responsablity
    //! The role of the macros is really light weight. We don't want to make the register crate the
    //! hearth of our runtime. So it simply wrap your function into another functions define in the
    //! runtime and the runtime does the rest of the magic.
    //!
    //! ## Usage
    //! To use the macros you simply need to import it from this crate and use the `register` macro
    //! to mark your training and inference functions. Here is an exemple:
    //! ```rust
    //! use burn_central::macros::register;
    //!
    //! #[register(training, name = "my_training_procedure")]
    //! async fn my_training_function() {
    //!  // Your training code here
    //! }
    //! ```

    /// Macro to register your training and inference functions.
    pub use burn_central_macros::register;
}

/// The runtime crate execute training and inference procedure registered with
/// Burn Central Macros. It basicly form a wrapper crate that use your declare fucntions.
pub mod runtime {
    pub use burn_central_runtime::*;
}
