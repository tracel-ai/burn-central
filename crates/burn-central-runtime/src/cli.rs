//! This is util for generated crate to be able to test parsing at runtime.

use clap::{Args, Parser};

#[derive(Args, Debug)]
/// Burn Central configuration arguments. Those are declare here as the CLI is not a library that
/// can be used in the generated crate.
pub struct BurnCentralArgs {
    #[arg(long, default_value = "default")]
    pub namespace: String,
    #[arg(long, default_value = "default")]
    pub project: String,
    #[arg(long)]
    pub api_key: String,
    #[arg(long, default_value = "http://localhost:9001")]
    pub endpoint: String,
}

#[derive(Parser, Debug)]
#[command(
    name = "burn-central-runtime",
    version,
    about = "Burn Central Runtime CLI"
)]
/// Arguments provided via CLI by the Burn Central CLI
pub struct RuntimeArgs {
    /// The kind of routine to execute. It can be `training` or `inference`.
    pub kind: String,
    /// The name of the routine to execute. We pass the routine name here as the name might not be
    /// the name of the function if the user decide to rename it using the `name` attribute in the
    /// register macro.
    pub routine: String,
    /// JSON string representing the arguments to pass to the routine. The arguments pass here are
    /// self define by the user. Value found in this field will be merge with the Config the user
    /// is requesting using [Args] extractor in his training function.
    #[arg(long, default_value = "{}")]
    pub args: String,
    /// Burn Central configuration arguments.
    #[command(flatten)]
    pub burn_central: BurnCentralArgs,
}

/// This function is an utility to parse the runtime arguments from the command line.
/// It used `clap` under the hood.
pub fn parse_runtime_args() -> RuntimeArgs {
    RuntimeArgs::parse()
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_parse_runtime_args() {
        let args = vec![
            "burn-central-runtime",
            "train",
            "my_routine",
            "--namespace",
            "my_namespace",
            "--project",
            "my_project",
            "--api-key",
            "my_api_key",
        ];
        let runtime_args = RuntimeArgs::try_parse_from(args).unwrap();
        assert_eq!(runtime_args.kind, "train");
        assert_eq!(runtime_args.routine, "my_routine");
        assert_eq!(runtime_args.args, "{}");
        assert_eq!(runtime_args.burn_central.namespace, "my_namespace");
        assert_eq!(runtime_args.burn_central.project, "my_project");
        assert_eq!(runtime_args.burn_central.api_key, "my_api_key");
    }
}
