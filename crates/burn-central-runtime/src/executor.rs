use anyhow::Result;
use burn::prelude::Backend;
use burn::tensor::backend::AutodiffBackend;

use crate::error::RuntimeError;
use crate::output::{ExperimentOutput, TrainOutput};
use crate::params::args::{ExperimentArgs, deserialize_and_merge_with_default};
use crate::routine::{BoxedRoutine, ExecutorRoutineWrapper, IntoRoutine, Routine};
use crate::telemetry;
use burn_central_core::BurnCentral;
use burn_central_core::experiment::ExperimentRun;
use std::collections::HashMap;

type ExecutorRoutine<B> = BoxedRoutine<ExecutionContext<B>, (), ()>;

/// The execution context for a routine, containing the necessary information to run it.
pub struct ExecutionContext<B: Backend> {
    client: Option<BurnCentral>,
    namespace: String,
    project: String,
    args_override: Option<serde_json::Value>,
    devices: Vec<B::Device>,
    experiment: Option<ExperimentRun>,
}

impl<B: Backend> ExecutionContext<B> {
    pub fn use_merged_args<A: ExperimentArgs>(&self) -> A {
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

    pub fn devices(&self) -> &[B::Device] {
        &self.devices
    }

    pub fn client(&self) -> Option<&BurnCentral> {
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
pub struct ExecutorBuilder<B: AutodiffBackend> {
    executor: Executor<B>,
}

impl<B: AutodiffBackend> ExecutorBuilder<B> {
    fn new() -> Self {
        Self {
            executor: Executor {
                client: None,
                namespace: None,
                project: None,
                handlers: HashMap::new(),
            },
        }
    }

    fn register<M, O: ExperimentOutput<B>>(
        &mut self,
        kind: ActionKind,
        name: impl Into<String>,
        handler: impl IntoRoutine<ExecutionContext<B>, (), O, M>,
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

    pub fn train<M, O: TrainOutput<B>>(
        &mut self,
        name: impl Into<String>,
        handler: impl IntoRoutine<ExecutionContext<B>, (), O, M>,
    ) -> &mut Self {
        self.register(ActionKind::Train, name, handler);
        self
    }

    pub fn build(
        self,
        client: BurnCentral,
        namespace: impl Into<String>,
        project: impl Into<String>,
    ) -> Executor<B> {
        let mut executor = self.executor;
        executor.client = Some(client);
        executor.namespace = Some(namespace.into());
        executor.project = Some(project.into());
        // Possibly do some validation or final setup here
        executor
    }
}

// Hide element that are only used internally by the gen crate.
#[doc(hidden)]
/// An executor that manages the execution of routines for different targets.
pub struct Executor<B: Backend> {
    client: Option<BurnCentral>,
    namespace: Option<String>,
    project: Option<String>,
    handlers: HashMap<TargetId, ExecutorRoutine<B>>,
}

impl<B: AutodiffBackend> Executor<B> {
    /// Creates a new `ExecutorBuilder` to configure and build an `Executor`.
    pub fn builder() -> ExecutorBuilder<B> {
        ExecutorBuilder::new()
    }

    /// Lists all registered targets in the executor.
    pub fn targets(&self) -> Vec<TargetId> {
        self.handlers.keys().cloned().collect()
    }

    /// Runs a routine for the specified target with the given devices and arguments override.
    pub fn run(
        &self,
        kind: ActionKind,
        name: impl AsRef<str>,
        devices: impl IntoIterator<Item = B::Device>,
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

        let mut ctx = ExecutionContext {
            client: self.client.clone(),
            namespace: self.namespace.clone().unwrap_or_default(),
            project: self.project.clone().unwrap_or_default(),
            args_override,
            devices: devices.into_iter().collect(),
            experiment: None,
        };

        if let Some(client) = &mut ctx.client {
            let code_version = option_env!("BURN_CENTRAL_CODE_VERSION")
                .unwrap_or("unknown")
                .to_string();
            log::debug!("Using Burn Central client with code version: {code_version}");

            log::info!(
                "Starting experiment for target: {} in namespace: {}, project: {}",
                routine,
                ctx.namespace,
                ctx.project
            );
            let experiment = client.start_experiment(
                &ctx.namespace,
                &ctx.project,
                code_version,
                routine.to_string(),
            )?;
            ctx.experiment = Some(experiment);
            if let Some(experiment) = ctx.experiment.as_ref() {
                if let Err(err) = telemetry::install_for_experiment(experiment) {
                    log::warn!("Telemetry disabled: {err}");
                }
            }
        }

        let result = handler.run((), &mut ctx);

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

    use crate::params::args::Args;
    use crate::{Model, MultiDevice};

    use super::*;
    use burn::backend::{Autodiff, NdArray};
    use burn::nn::{Linear, LinearConfig};
    use burn::prelude::*;
    use burn_central_core::bundle::{BundleEncode, BundleSink};
    use serde::{Deserialize, Serialize};

    impl<B: AutodiffBackend> ExecutorBuilder<B> {
        pub fn build_offline(self) -> Executor<B> {
            self.executor
        }
    }

    // A backend stub for testing purposes.
    type TestBackend = Autodiff<NdArray<f32>>;
    type TestDevice = <NdArray<f32> as Backend>::Device;

    #[derive(Module, Debug)]
    struct TestModel<B: Backend> {
        linear: Linear<B>,
    }

    impl<B: Backend> BundleEncode for TestModel<B> {
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

    impl<B: AutodiffBackend> TestModel<B> {
        fn new(device: &B::Device) -> Self {
            let linear = LinearConfig::new(10, 5).init(device);
            TestModel { linear }
        }
    }

    #[derive(Serialize, Deserialize, Debug, Default, Clone)]
    struct TestArgs {
        lr: f32,
        epochs: usize,
    }

    // --- Test Routines ---

    fn simple_train_step<B: AutodiffBackend>() -> Result<Model<TestModel<B>>, String> {
        let device = B::Device::default();
        let model = TestModel::new(&device);
        Ok(model.into())
    }

    fn train_with_params<B: AutodiffBackend>(
        args: Args<TestArgs>,
        devices: MultiDevice<B>,
    ) -> Model<TestModel<B>> {
        let model = TestModel::new(&devices[0]);
        assert_eq!(args.lr, 0.01);
        assert_eq!(args.epochs, 10);
        println!("Train step with config and model executed.");
        model.into()
    }

    fn failing_routine<B: AutodiffBackend>() -> Result<Model<TestModel<B>>> {
        anyhow::bail!("Failing routine");
    }

    // --- Tests ---

    #[test]
    fn should_run_simple_routine_successfully() {
        let mut builder = Executor::<TestBackend>::builder();
        builder.train("simple_task", simple_train_step::<TestBackend>);
        let executor = builder.build_offline();

        let result = executor.run(
            "train".parse().unwrap(),
            "simple_task",
            [TestDevice::default()],
            None,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn should_inject_parameters_and_handle_output() {
        let mut builder = Executor::<TestBackend>::builder();
        builder.train("complex_task", train_with_params);
        let executor = builder.build_offline();

        let args_json = r#"{"lr": 0.01, "epochs": 10}"#.to_string();

        let result = executor.run(
            "train".parse().unwrap(),
            "complex_task",
            [TestDevice::default()],
            Some(args_json),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn should_return_handler_not_found_error() {
        let builder = Executor::<TestBackend>::builder();
        let executor = builder.build_offline();

        let result = executor.run(
            "train".parse().unwrap(),
            "non_existent_task",
            [TestDevice::default()],
            None,
        );

        assert!(matches!(result, Err(RuntimeError::HandlerNotFound(_))));
    }

    #[test]
    fn should_handle_failing_routine() {
        let mut builder = Executor::<TestBackend>::builder();
        builder.train("failing_task", failing_routine::<TestBackend>);
        let executor = builder.build_offline();

        let result = executor.run(
            "train".parse().unwrap(),
            "failing_task",
            [TestDevice::default()],
            None,
        );

        assert!(matches!(result, Err(RuntimeError::HandlerFailed(_))));
    }

    #[test]
    fn should_support_named_routines() {
        let mut builder = Executor::<TestBackend>::builder();
        builder.train(
            "task1",
            simple_train_step::<TestBackend>.with_name("custom_name_1"),
        );
        builder.train("task2", ("custom_name_2", simple_train_step::<TestBackend>));
        let executor = builder.build_offline();

        let res1 = executor.run("train".parse().unwrap(), "task1", [], None);
        let res2 = executor.run("train".parse().unwrap(), "task2", [], None);

        assert!(res1.is_ok());
        assert!(res2.is_ok());
    }
}
