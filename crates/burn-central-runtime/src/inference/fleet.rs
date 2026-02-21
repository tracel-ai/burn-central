use burn::prelude::Backend;
use burn_central_core::Env;
use burn_central_fleet::{
    FleetDeviceSession, FleetManagedFactory, FleetManagedInference, FleetRegistrationToken,
};
use burn_central_inference::Inference;
use serde::Serialize;

use crate::{
    inference::{
        InferenceArgs, InferenceError, InferenceInit, ModelSource,
        registry::{InferenceContext, InferenceFactoryReturn},
    },
    routine::{IntoRoutine, Routine},
};

/// Build a typed inference instance directly from a factory routine.
pub fn build_fleet_managed<B, I, M, R>(
    factory: impl IntoRoutine<InferenceContext<B>, (), I, M>,
    token: impl Into<FleetRegistrationToken>,
    metadata: impl Serialize,
    device: B::Device,
) -> Result<FleetManagedInference<B, I::Inference>, InferenceError>
where
    B: Backend,
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

    let inference_factory: Box<dyn FleetManagedFactory<B, I::Inference>> = Box::new(
        move |model_source: burn_central_fleet::ModelSource,
              runtime_config: serde_json::Value,
              device: B::Device| {
            let init = InferenceInit {
                model: ModelSource::from(model_source),
                device,
            };
            let mut ctx = InferenceContext::new(init, InferenceArgs::new(Some(runtime_config)));

            let factory_output = routine.run((), &mut ctx).map_err(|err| {
                format!("inference handler '{error_name}' failed to initialize: {err}")
            })?;

            factory_output.into_inference().map_err(|message| {
                format!("inference handler '{error_name}' failed to initialize: {message}")
            })
        },
    );

    let metadata = serde_json::json!({
        "name": inference_name,
        "metadata": metadata,
    });

    let fleet_session = FleetDeviceSession::init(token.into(), metadata, &Env::Development)
        .map_err(|e| InferenceError::FactoryFailed {
            name: "fleet registration".to_string(),
            message: e.to_string(),
        })?;

    let inference = FleetManagedInference::init(
        inference_name.clone(),
        fleet_session,
        inference_factory,
        device,
    )
    .map_err(|e| InferenceError::FactoryFailed {
        name: inference_name,
        message: e.to_string(),
    })?;

    Ok(inference)
}
