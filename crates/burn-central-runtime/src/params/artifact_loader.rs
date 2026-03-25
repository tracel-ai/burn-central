use crate::executor::ExecutionContext;
use crate::params::RoutineParam;
use burn::prelude::Backend;
use burn_central_artifact::bundle::BundleDecode;
use burn_central_experiment::{ExperimentId, ExperimentRun, error::ExperimentError};

/// Artifact loader for loading artifacts from Burn Central. It allow to fecth for instance other
/// experiment endpoint to be able to restart from a certain point your experiment.
///
/// You can build it yourself by using the [ArtifactLoader::new] function with your namespace (in
/// slug format (e.g. "my-team")), project name and a [burn_central_core::BurnCentral]. However, it
/// is also possible to request it directly in your routine by using declaring the param like so:
///
/// ```ignore
/// # use burn_central_runtime::ArtifactLoader;
/// # use burn_central_artifact::bundle::BundleDecode;
/// # use burn_central::register;
/// # use burn_central_runtime::Model;
/// # use burn_central_runtime::MultiDevice;
/// # use serde::*;
/// #[derive(Deserialize, Serialize, Default)]
/// pub struct LaunchArgs {
///     pub experiment_num: Option<i32>,
/// }
///
/// #[register(training, name = "mnist")]
/// pub fn training<B: AutodiffBackend>(
///     config: Args<LaunchArgs>,
///     MultiDevice(devices): MultiDevice<B>,
///     loader: ArtifactLoader<ModelArtifact<B>>,
/// ) -> Result<Model<ModelArtifact<B::InnerBackend>>, String> {
///     // Load a pretrained model if an experiment number is provided.
///     if let Some(experiment_num) = config.experiment_num {
///         let pretrained_model = loader
///             .load(experiment_num, "train_artifacts")
///             .expect("To be able to fetch artifacts");
///     }
/// }
/// ```
///
/// As you can see in the example above, you can use the loader to dynamically request experiment
/// artifacts when requested through your routine configuration.
pub struct ArtifactLoader<'a, T: BundleDecode> {
    experiment: &'a ExperimentRun,
    _artifact: std::marker::PhantomData<T>,
}

impl<'a, T: BundleDecode> ArtifactLoader<'a, T> {
    pub fn new(client: &'a ExperimentRun) -> Self {
        Self {
            experiment: client,
            _artifact: std::marker::PhantomData,
        }
    }

    /// Load an artifact by name with specific settings.
    pub fn load_with(
        &self,
        experiment_id: impl Into<ExperimentId>,
        name: impl AsRef<str>,
        settings: &T::Settings,
    ) -> Result<T, ExperimentError> {
        self.experiment
            .use_artifact(experiment_id.into(), name.as_ref(), settings)
    }

    /// Load an artifact by name with default settings.
    pub fn load(
        &self,
        experiment_id: impl Into<ExperimentId>,
        name: impl AsRef<str>,
    ) -> Result<T, ExperimentError> {
        self.experiment
            .use_artifact(experiment_id.into(), name.as_ref(), &Default::default())
    }
}

impl<B: Backend, T: BundleDecode> RoutineParam<ExecutionContext<B>> for ArtifactLoader<'_, T> {
    type Item<'new>
        = ArtifactLoader<'new, T>
    where
        ExecutionContext<B>: 'new;

    fn try_retrieve(ctx: &ExecutionContext<B>) -> anyhow::Result<Self::Item<'_>> {
        let experiment = ctx.experiment().ok_or_else(|| {
            anyhow::anyhow!("Burn Central client is not configured in the execution context")
        })?;

        Ok(ArtifactLoader::new(experiment))
    }
}
