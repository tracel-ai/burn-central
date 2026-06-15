//! This is util for generated crate to be able to test parsing at runtime.

use clap::{Args, Parser};

#[derive(Args, Debug)]
/// Tracel configuration arguments. Those are declare here as the CLI is not a library that
/// can be used in the generated crate.
pub struct TracelArgs {
    #[arg(long, default_value = "default")]
    pub namespace: String,
    #[arg(long, default_value = "default")]
    pub project: String,
    #[arg(long)]
    pub api_key: String,
    #[arg(long, default_value = "Development", value_parser = serde_env_parser)]
    pub env: tracel_client::Env,
}

fn serde_env_parser(s: &str) -> Result<tracel_client::Env, String> {
    serde_json::from_str(s).map_err(|e| format!("Failed to parse env: {e}"))
}

#[derive(Parser, Debug)]
#[command(name = "tracel-runtime", version, about = "Tracel Runtime CLI")]
/// Arguments provided via CLI by the Tracel CLI
pub struct RuntimeArgs {
    /// The kind of routine to execute. It can be `training` or `inference`.
    pub kind: String,
    /// The name of the routine to execute. We pass the routine name here as the name might not be
    /// the name of the function if the user decide to rename it using the `name` attribute in the
    /// register macro.
    pub routine: String,
    /// JSON string representing the arguments to pass to the routine. The arguments pass here are
    /// self-defined by the user. Values in this field are merged with the target config when a
    /// routine requests `Args<T>`.
    #[arg(long, default_value = "{}")]
    pub args: String,
    /// tracel configuration arguments.
    #[command(flatten)]
    pub tracel: TracelArgs,
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
        let env = serde_json::to_string(&tracel_client::Env::Production).unwrap();
        let args = vec![
            "tracel-runtime",
            "train",
            "my_routine",
            "--namespace",
            "my_namespace",
            "--project",
            "my_project",
            "--api-key",
            "my_api_key",
            "--env",
            &env,
        ];
        let runtime_args = RuntimeArgs::try_parse_from(args).unwrap();
        assert_eq!(runtime_args.kind, "train");
        assert_eq!(runtime_args.routine, "my_routine");
        assert_eq!(runtime_args.args, "{}");
        assert_eq!(runtime_args.tracel.namespace, "my_namespace");
        assert_eq!(runtime_args.tracel.project, "my_project");
        assert_eq!(runtime_args.tracel.api_key, "my_api_key");
    }
}
