//! This is util for generated crate to be able to test parsing at runtime.

use serde::Deserialize;

/// Burn Central configuration arguments. Those are declare here as the CLI is not a library that
/// can be used in the generated crate.
#[derive(Deserialize)]
pub struct BurnCentralArgs {
    pub namespace: String,
    pub project: String,
    pub api_key: String,
    pub endpoint: String,
}

#[derive(Deserialize)]
/// Arguments provided via CLI by the Burn Central CLI
pub struct RuntimeArgs {
    /// The device ids to use for the routine execution. If not provided, the default device will be
    /// used.
    pub devices: Option<Vec<(u16, u32)>>,
    /// The kind of routine to execute. It can be `training` or `inference`.
    pub kind: String,
    /// The name of the routine to execute. We pass the routine name here as the name might not be
    /// the name of the function if the user decide to rename it using the `name` attribute in the
    /// register macro.
    pub routine: String,
    /// JSON string representing the arguments to pass to the routine. The arguments pass here are
    /// self define by the user. Value found in this field will be merge with the Config the user
    /// is requesting using [Args] extractor in his training function.
    pub args: String,
    /// Burn Central configuration arguments.
    pub burn_central: BurnCentralArgs,
}

/// This function is an utility to parse the runtime arguments from the command line.
pub fn parse_runtime_args() -> RuntimeArgs {
    serde_json::from_str(&std::env::args().nth(1).expect("No runtime args provided"))
        .expect("Failed to parse runtime args")
}
