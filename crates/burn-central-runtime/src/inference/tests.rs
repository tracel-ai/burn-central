use crate::inference::ModelAccessor;
use crate::inference::core::InferenceBuilder;
use crate::inference::init::Init;
use crate::inference::streaming::OutStream;
use crate::{CancelToken, In, MultiDevice, Out, State};
use burn::backend::NdArray;
use burn::config::Config;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::{Backend, Module};
use burn::record::{FullPrecisionSettings, NamedMpkBytesRecorder, Recorder};
use burn::tensor::Tensor;

type TestBackend = NdArray;
type Device = <TestBackend as Backend>::Device;

#[derive(Config, Debug)]
pub struct TestModelConfig {
    input_size: usize,
    output_size: usize,
}

impl TestModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TestModel<B> {
        let linear = LinearConfig::new(self.input_size, self.output_size).init(device);
        TestModel { linear }
    }
}

#[derive(Module, Debug)]
pub struct TestModel<B: Backend> {
    linear: Linear<B>,
}

impl<B: Backend> TestModel<B> {
    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(input)
    }
}

pub struct TestModelArtifacts {
    pub config: TestModelConfig,
    pub weights: Vec<u8>,
}

impl<B: Backend> Init<B, TestModelArtifacts> for TestModel<B> {
    type Error = anyhow::Error;
    fn init(args: &TestModelArtifacts, device: &B::Device) -> Result<Self, Self::Error> {
        let config = &args.config;
        println!("Loading test model with config: {config:?}");
        let model = config.init(device);

        let recorder = NamedMpkBytesRecorder::<FullPrecisionSettings>::default();
        let model_record = recorder.load(args.weights.clone(), device)?;
        let model = model.load_record(model_record);
        Ok(model)
    }
}

fn streaming_inference_handler<B: Backend>(
    In(input): In<Tensor<B, 2>>,
    model: ModelAccessor<TestModel<B>>,
    devices: MultiDevice<B>,
    cancel: CancelToken,
    output: OutStream<Tensor<B, 2>>,
) -> Result<(), String> {
    if cancel.is_cancelled() {
        return Err("Inference cancelled".into());
    }
    println!("Using device: {:?}", devices[0]);
    let mut result = input;
    for i in 0..3 {
        result = model.submit(move |m| m.forward(result.clone() + i));
        if output.emit(result.clone()).is_err() {
            return Err("Failed to emit output".into());
        }
    }
    Ok(())
}

fn direct_inference_handler<B: Backend>(
    In(input): In<Tensor<B, 2>>,
    model: ModelAccessor<TestModel<B>>,
    _devices: MultiDevice<B>,
) -> Out<Tensor<B, 2>> {
    let result = model.submit(|m| m.forward(input));
    result.into()
}

fn stateful_inference_handler<B: Backend>(
    In(input): In<Tensor<B, 2>>,
    model: ModelAccessor<TestModel<B>>,
    _devices: MultiDevice<B>,
    output: OutStream<String>,
    State(counter): State<i32>,
) -> Result<(), String> {
    let result = model.submit(|m| m.forward(input));
    for i in 0..counter {
        if output
            .emit(format!("Step {}: {:?}", i, result.to_data()))
            .is_err()
        {
            return Err("Failed to emit output".into());
        }
    }
    Ok(())
}

fn create_test_model_artifacts() -> TestModelArtifacts {
    let config = TestModelConfig::new(10, 10);
    let model = config.init::<TestBackend>(&Device::default());

    let recorder = NamedMpkBytesRecorder::<FullPrecisionSettings>::default();
    let record = model.into_record();
    let weights = recorder.record(record, ()).unwrap();
    TestModelArtifacts { config, weights }
}

#[test]
fn test_streaming_inference() {
    let artifacts = create_test_model_artifacts();
    let device = Device::default();

    let inference = InferenceBuilder::<TestBackend>::new()
        .init(&artifacts, &device)
        .unwrap()
        .build(streaming_inference_handler);

    let input = Tensor::<TestBackend, 2>::ones([1, 10], &device);
    let output = inference.infer(input).with_devices([device]).run().unwrap();

    assert_eq!(output.len(), 3);
    println!(
        "Streaming inference completed with {} outputs",
        output.len()
    );
}

#[test]
fn test_direct_inference() {
    let artifacts = create_test_model_artifacts();
    let device = Device::default();

    let inference = InferenceBuilder::<TestBackend>::new()
        .init(&artifacts, &device)
        .unwrap()
        .build(direct_inference_handler);

    let input = Tensor::<TestBackend, 2>::ones([1, 10], &device);
    let output = inference.infer(input).with_devices([device]).run().unwrap();

    assert_eq!(output.len(), 1);
    println!("Direct inference completed");
}

#[test]
fn test_stateful_inference() {
    let artifacts = create_test_model_artifacts();
    let device = Device::default();

    let inference = InferenceBuilder::<TestBackend>::new()
        .init(&artifacts, &device)
        .unwrap()
        .build(stateful_inference_handler);

    let input = Tensor::<TestBackend, 2>::ones([1, 10], &device);
    let output = inference
        .infer(input)
        .with_devices([device])
        .with_state(2)
        .run()
        .unwrap();

    assert_eq!(output.len(), 2);
    println!("Stateful inference completed with state");
}

#[test]
fn test_async_inference_job() {
    let artifacts = create_test_model_artifacts();
    let device = Device::default();

    let inference = InferenceBuilder::<TestBackend>::new()
        .init(&artifacts, &device)
        .unwrap()
        .build(streaming_inference_handler);

    let input = Tensor::<TestBackend, 2>::ones([1, 10], &device);
    let job = inference.infer(input).with_devices([device]).spawn();

    let mut output_count = 0;
    for output in job.stream.iter() {
        println!("Received output: {:?}", output.to_data());
        output_count += 1;
        if output_count >= 2 {
            job.cancel();
            break;
        }
    }

    let result = job.join();

    assert!(result.is_ok());

    assert_eq!(output_count, 2);
}

// Cancellable variant that checks the cancel token each step.
fn streaming_inference_cancellable_handler<B: Backend>(
    In(input): In<Tensor<B, 2>>,
    model: ModelAccessor<TestModel<B>>,
    devices: MultiDevice<B>,
    cancel: CancelToken,
    output: OutStream<Tensor<B, 2>>,
) -> Result<(), String> {
    println!("Using device (cancellable): {:?}", devices[0]);
    let mut result = input;
    for i in 0..10 {
        if cancel.is_cancelled() {
            return Err("Cancelled mid-stream".into());
        }
        result = model.submit(move |m| m.forward(result.clone() + i));
        output
            .emit(result.clone())
            .map_err(|_| "emit failed".to_string())?;
    }
    Ok(())
}

// Handler that immediately fails to test error propagation.
fn failing_streaming_handler<B: Backend>(
    _input: In<Tensor<B, 2>>,
    _model: ModelAccessor<TestModel<B>>,
    _devices: MultiDevice<B>,
    _cancel: CancelToken,
    _output: OutStream<Tensor<B, 2>>,
) -> Result<(), String> {
    Err("Intentional failure".into())
}

#[test]
fn test_streaming_cancellation_mid_stream() {
    let artifacts = create_test_model_artifacts();
    let device = Device::default();

    let inference = InferenceBuilder::<TestBackend>::new()
        .init(&artifacts, &device)
        .unwrap()
        .build(streaming_inference_cancellable_handler);

    let input = Tensor::<TestBackend, 2>::ones([1, 10], &device);
    let job = inference.infer(input).with_devices([device]).spawn();

    let mut received = 0usize;
    for _ in job.stream.iter() {
        println!("Got (before cancel) {}", received);
        received += 1;
        if received == 3 {
            job.cancel();
            break;
        }
    }
    let result = job.join();
    assert!(result.is_err(), "Expected cancellation error");
    assert_eq!(received, 3, "Expected exactly 3 outputs before cancel");
}

#[test]
fn test_failing_handler_propagates_error() {
    let artifacts = create_test_model_artifacts();
    let device = Device::default();

    let inference = InferenceBuilder::<TestBackend>::new()
        .init(&artifacts, &device)
        .unwrap()
        .build(failing_streaming_handler);

    let input = Tensor::<TestBackend, 2>::ones([1, 10], &device);
    let job = inference.infer(input).with_devices([device]).spawn();

    // No outputs expected
    let mut any = false;
    for _ in job.stream.iter() {
        any = true;
    }
    assert!(!any, "No outputs should be produced");
    let result = job.join();
    assert!(result.is_err(), "Failure should propagate as Err");
}

#[test]
fn test_stateful_zero_iterations() {
    let artifacts = create_test_model_artifacts();
    let device = Device::default();

    let inference = InferenceBuilder::<TestBackend>::new()
        .init(&artifacts, &device)
        .unwrap()
        .build(stateful_inference_handler);

    let input = Tensor::<TestBackend, 2>::ones([1, 10], &device);
    let output = inference
        .infer(input)
        .with_devices([device])
        .with_state(0)
        .run()
        .unwrap();

    assert_eq!(output.len(), 0, "Expected no outputs when counter=0");
}

#[test]
fn test_multiple_streaming_jobs_in_parallel() {
    use std::thread;
    let artifacts = create_test_model_artifacts();
    let device = Device::default();
    let inference = InferenceBuilder::<TestBackend>::new()
        .init(&artifacts, &device)
        .unwrap()
        .build(streaming_inference_handler);

    let input_a = Tensor::<TestBackend, 2>::ones([1, 10], &device);
    let input_b = Tensor::<TestBackend, 2>::ones([1, 10], &device);

    let job_a = inference.infer(input_a).with_devices([device]).spawn();
    let job_b = inference.infer(input_b).with_devices([device]).spawn();

    let h_a = thread::spawn(move || job_a.stream.iter().count());
    let h_b = thread::spawn(move || job_b.stream.iter().count());

    let ca = h_a.join().unwrap();
    let cb = h_b.join().unwrap();
    assert_eq!(ca, 3);
    assert_eq!(cb, 3);
}
