use crate::error::RuntimeError;
use crate::output::{ExperimentOutput, TrainOutput};
use crate::params::args::{LaunchArgs, deserialize_and_merge_with_default};
use crate::routine::{BoxedRoutine, ExecutorRoutineWrapper, IntoRoutine, Routine};
use anyhow::Result;
use std::collections::HashMap;
use tracel_experiment::integration::tracing::try_init_tracing_subscriber;
use tracel_experiment::{CancelToken, ExperimentRun, ExperimentRunHandleExt};

type ExecutorRoutine = BoxedRoutine<ExecutionContext, (), ()>;

/// The execution context for a routine, containing the necessary information to run it.
pub struct ExecutionContext {
    client: Option<tracel_client::Client>,
    namespace: String,
    project: String,
    args_override: Option<serde_json::Value>,
    experiment: Option<ExperimentRun>,
    cancel_token: CancelToken,
}

impl ExecutionContext {
    /// Retrieve args merged on top of `A::default()`.
    ///
    /// This powers the `Args<A>` routine extractor for training routines.
    /// If deserialization fails, defaults are returned.
    pub fn use_merged_args<A: LaunchArgs>(&self) -> A {
        let args = match &self.args_override {
            Some(json) => deserialize_and_merge_with_default(json).unwrap_or_default(),
            None => A::default(),
        };

        if let Some(experiment) = &self.experiment {
            experiment.log_args(&args).unwrap_or_else(|e| {
                log::error!("Failed to log experiment arguments: {}", e);
            });
        }

        args
    }

    pub fn experiment(&self) -> Option<&ExperimentRun> {
        self.experiment.as_ref()
    }

    pub fn cancel_token(&self) -> &CancelToken {
        &self.cancel_token
    }

    pub fn client(&self) -> Option<&tracel_client::Client> {
        self.client.as_ref()
    }

    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    pub fn project(&self) -> &str {
        &self.project
    }
}

/// The kind of action that can be executed by the executor.
#[derive(Clone, Debug, PartialEq, Eq, Hash, strum::Display, strum::EnumString)]
#[strum(serialize_all = "snake_case")]
pub enum ActionKind {
    Train,
    // Infer,
    // Eval,
    // Test,
    // #[strum(serialize = "custom({0})")]
    // Custom(String),
}

/// The identifier for a target, which consists of an action kind and a name.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TargetId {
    kind: ActionKind,
    name: String,
}

impl std::fmt::Display for TargetId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.kind, self.name)
    }
}

// Hide element that are only used internally by the gen crate.
#[doc(hidden)]
/// A builder for creating an `Executor` instance with registered routines.
pub struct ExecutorBuilder {
    executor: Executor,
}

impl ExecutorBuilder {
    fn new() -> Self {
        Self {
            executor: Executor {
                credentials: None,
                env: None,
                namespace: None,
                project: None,
                handlers: HashMap::new(),
            },
        }
    }

    fn register<M, O: ExperimentOutput>(
        &mut self,
        kind: ActionKind,
        name: impl Into<String>,
        handler: impl IntoRoutine<ExecutionContext, (), O, M>,
    ) -> &mut Self {
        let wrapper = ExecutorRoutineWrapper::new(IntoRoutine::into_routine(handler));
        let routine = Box::new(wrapper);
        let routine_name = routine.name();

        let target_id = TargetId {
            kind,
            name: name.into(),
        };

        log::debug!("Registering handler '{routine_name}' for target: {target_id}");

        self.executor.handlers.insert(target_id, routine);
        self
    }

    pub fn train<M, O: TrainOutput>(
        &mut self,
        name: impl Into<String>,
        handler: impl IntoRoutine<ExecutionContext, (), O, M>,
    ) -> &mut Self {
        self.register(ActionKind::Train, name, handler);
        self
    }

    pub fn build(
        self,
        credentials: impl Into<tracel_client::TracelCredentials>,
        env: tracel_client::Env,
        namespace: impl Into<String>,
        project: impl Into<String>,
    ) -> Executor {
        let mut executor = self.executor;
        executor.credentials = Some(credentials.into());
        executor.env = Some(env);
        executor.namespace = Some(namespace.into());
        executor.project = Some(project.into());
        // Possibly do some validation or final setup here
        executor
    }
}

// Hide element that are only used internally by the gen crate.
#[doc(hidden)]
/// An executor that manages the execution of routines for different targets.
pub struct Executor {
    credentials: Option<tracel_client::TracelCredentials>,
    env: Option<tracel_client::Env>,
    namespace: Option<String>,
    project: Option<String>,
    handlers: HashMap<TargetId, ExecutorRoutine>,
}

impl Executor {
    /// Creates a new `ExecutorBuilder` to configure and build an `Executor`.
    pub fn builder() -> ExecutorBuilder {
        ExecutorBuilder::new()
    }

    /// Lists all registered targets in the executor.
    pub fn targets(&self) -> Vec<TargetId> {
        self.handlers.keys().cloned().collect()
    }

    /// Runs a routine for the specified target with the given arguments override.
    pub fn run(
        &self,
        kind: ActionKind,
        name: impl AsRef<str>,
        args_override: Option<String>,
    ) -> Result<(), RuntimeError> {
        let routine = name.as_ref();

        let target_id = TargetId {
            kind,
            name: routine.to_string(),
        };

        let handler = self.handlers.get(&target_id).ok_or_else(|| {
            log::error!("Handler not found for target: {routine}");
            RuntimeError::HandlerNotFound(routine.to_string())
        })?;

        log::debug!("Starting Execution for Target: {routine}");

        let args_override = args_override
            .as_ref()
            .map(|cfg_str| serde_json::from_str::<serde_json::Value>(cfg_str))
            .transpose()
            .map_err(|e| {
                log::error!("Failed to parse experiment argument overrides: {}", e);
                RuntimeError::InvalidArgs(e.to_string())
            })?;

        let client = match (&self.credentials, &self.env) {
            (Some(creds), Some(env)) => Some(
                tracel_client::Client::new(env.clone(), creds)
                    .map_err(|e| RuntimeError::ClientInitializationFailed(e.to_string()))?,
            ),
            _ => None,
        };

        let mut ctx = ExecutionContext {
            client,
            namespace: self.namespace.clone().unwrap_or_default(),
            project: self.project.clone().unwrap_or_default(),
            args_override,
            experiment: None,
            cancel_token: CancelToken::new(),
        };

        if let Some(client) = &ctx.client {
            let code_version = option_env!("TRACEL_CODE_VERSION")
                .unwrap_or("unknown")
                .to_string();
            log::debug!("Using Burn Central client with code version: {code_version}");

            log::info!(
                "Starting experiment for target: {} in namespace: {}, project: {}",
                routine,
                ctx.namespace,
                ctx.project
            );

            let experiment = tracel_core::experiment::create_cloud_experiment_run(
                client.clone(),
                &ctx.namespace,
                &ctx.project,
                code_version,
                HashMap::new(),
            )
            .map_err(|e| {
                tracel_experiment::error::ExperimentError::with_source(
                    tracel_experiment::error::ExperimentErrorKind::Internal,
                    "Failed to create cloud experiment run",
                    e,
                )
            })?;

            let experiment_num = experiment
                .id()
                .parse::<i32>()
                .expect("Tracel experiment ids should end with an experiment number");

            println!(
                "{}",
                serde_json::to_string(&serde_json::json!({
                    "namespace": ctx.namespace(),
                    "project": ctx.project(),
                    "experiment_num": experiment_num,
                }))
                .unwrap()
            );
            ctx.cancel_token = experiment.cancel_token();
            ctx.experiment = Some(experiment);
            let _ = try_init_tracing_subscriber();
        }

        let result = match ctx
            .experiment
            .as_ref()
            .map(|experiment| experiment.handle())
        {
            Some(handle) => handle.in_scope(|| handler.run((), &mut ctx)),
            None => handler.run((), &mut ctx),
        };

        match result {
            Ok(_) => {
                if let Some(experiment) = ctx.experiment {
                    experiment.finish()?;
                    log::info!("Experiment run completed successfully.");
                }
                log::debug!("Handler {routine} executed successfully.");

                Ok(())
            }
            Err(e) => {
                log::error!("Error executing handler '{routine}': {e}");
                if let Some(experiment) = ctx.experiment {
                    experiment.fail(e.to_string())?;
                    log::error!("Experiment run failed: {e}");
                }
                Err(e)
            }
        }
    }
}

#[cfg(test)]
mod test {
    use std::convert::Infallible;

    use crate::Model;
    use crate::params::args::Args;

    use super::*;
    use serde::{Deserialize, Serialize};
    use tracel_artifact::bundle::{BundleEncode, BundleSink};

    impl ExecutorBuilder {
        pub fn build_offline(self) -> Executor {
            self.executor
        }
    }

    #[derive(Debug)]
    struct TestModel;

    impl BundleEncode for TestModel {
        type Settings = ();
        type Error = Infallible;
        fn encode<E: BundleSink>(
            self,
            _sink: &mut E,
            _settings: &Self::Settings,
        ) -> Result<(), Self::Error> {
            Ok(())
        }
    }

    #[derive(Serialize, Deserialize, Debug, Default, Clone)]
    struct TestArgs {
        lr: f32,
        epochs: usize,
    }

    // --- Test Routines ---

    fn simple_train_step() -> Result<Model<TestModel>, String> {
        let model = TestModel;
        Ok(model.into())
    }

    fn train_with_params(args: Args<TestArgs>, cancel: CancelToken) -> Model<TestModel> {
        let model = TestModel;
        assert_eq!(args.lr, 0.01);
        assert_eq!(args.epochs, 10);
        println!("Cancel token available: {}", cancel.is_cancelled());
        println!("Train step with config and model executed.");
        model.into()
    }

    fn failing_routine() -> Result<Model<TestModel>> {
        anyhow::bail!("Failing routine");
    }

    // --- Tests ---

    #[test]
    fn should_run_simple_routine_successfully() {
        let mut builder = Executor::builder();
        builder.train("simple_task", simple_train_step);
        let executor = builder.build_offline();

        let result = executor.run("train".parse().unwrap(), "simple_task", None);
        assert!(result.is_ok());
    }

    #[test]
    fn should_inject_parameters_and_handle_output() {
        let mut builder = Executor::builder();
        builder.train("complex_task", train_with_params);
        let executor = builder.build_offline();

        let args_json = r#"{"lr": 0.01, "epochs": 10}"#.to_string();

        let result = executor.run("train".parse().unwrap(), "complex_task", Some(args_json));
        assert!(result.is_ok());
    }

    #[test]
    fn should_return_handler_not_found_error() {
        let builder = Executor::builder();
        let executor = builder.build_offline();

        let result = executor.run("train".parse().unwrap(), "non_existent_task", None);

        assert!(matches!(result, Err(RuntimeError::HandlerNotFound(_))));
    }

    #[test]
    fn should_handle_failing_routine() {
        let mut builder = Executor::builder();
        builder.train("failing_task", failing_routine);
        let executor = builder.build_offline();

        let result = executor.run("train".parse().unwrap(), "failing_task", None);

        assert!(matches!(result, Err(RuntimeError::HandlerFailed(_))));
    }

    #[test]
    fn should_support_named_routines() {
        let mut builder = Executor::builder();
        builder.train("task1", simple_train_step.with_name("custom_name_1"));
        builder.train("task2", ("custom_name_2", simple_train_step));
        let executor = builder.build_offline();

        let res1 = executor.run("train".parse().unwrap(), "task1", None);
        let res2 = executor.run("train".parse().unwrap(), "task2", None);

        assert!(res1.is_ok());
        assert!(res2.is_ok());
    }
}
