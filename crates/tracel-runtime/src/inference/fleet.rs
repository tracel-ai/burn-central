use serde::Serialize;
use tracel_client::Env;
use tracel_fleet::{
    FleetDeviceSession, FleetManagedFactory, FleetManagedInference, FleetRegistrationToken,
};
use tracel_inference::Inference;

use crate::{
    inference::{
        InferenceArgs, InferenceError, InferenceInit, ModelSource,
        registry::{InferenceContext, InferenceFactoryReturn},
    },
    routine::{IntoRoutine, Routine},
};

/// Build a typed inference instance directly from a factory routine.
pub fn build_fleet_managed_inference<I, M, R>(
    factory: impl IntoRoutine<InferenceContext, (), I, M>,
    token: impl Into<FleetRegistrationToken>,
    metadata: impl Serialize,
) -> Result<FleetManagedInference<I::Inference>, InferenceError>
where
    I: InferenceFactoryReturn<R>,
    I::Inference: Inference + Send + Sync + 'static,
    M: 'static,
    R: 'static,
{
    let metadata = serde_json::to_value(metadata).map_err(|e| InferenceError::FactoryFailed {
        name: "metadata serialization".to_string(),
        message: e.to_string(),
    })?;

    let routine = IntoRoutine::into_routine(factory);
    let inference_name = routine.name().to_string();
    let error_name = inference_name.clone();

    let inference_factory: Box<dyn FleetManagedFactory<I::Inference>> =
        Box::new(move |model_source, runtime_config: serde_json::Value| {
            let init = InferenceInit {
                model: Some(ModelSource::from(model_source)).into(),
            };
            let mut ctx = InferenceContext::new(init, InferenceArgs::new(Some(runtime_config)));

            let factory_output = routine.run((), &mut ctx).map_err(|err| {
                format!("inference handler '{error_name}' failed to initialize: {err}")
            })?;

            factory_output.into_inference().map_err(|message| {
                format!("inference handler '{error_name}' failed to initialize: {message}")
            })
        });

    let metadata = serde_json::json!({
        "name": inference_name,
        "metadata": metadata,
    });

    // TODO: Remove hardcoding dev env
    let fleet_session = FleetDeviceSession::init(token.into(), metadata, &Env::Development)
        .map_err(|e| InferenceError::FactoryFailed {
            name: "fleet registration".to_string(),
            message: e.to_string(),
        })?;

    let inference =
        FleetManagedInference::init(inference_name.clone(), fleet_session, inference_factory)
            .map_err(|e| InferenceError::FactoryFailed {
                name: inference_name,
                message: e.to_string(),
            })?;

    Ok(inference)
}
