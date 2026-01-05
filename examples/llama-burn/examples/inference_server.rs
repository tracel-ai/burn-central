//! Llama inference server example using burn-central-runtime.
//!
//! This example demonstrates how to use llama-burn's built-in inference API
//! for streaming token generation. The inference logic lives in the library,
//! and applications just consume it.
//!
//! Run with: cargo run --example inference_server --features llama3,cuda
#[cfg(feature = "cuda")]
use burn::backend::{cuda::CudaDevice, Cuda};
#[cfg(feature = "cuda")]
use burn::prelude::Backend;
use burn::tensor::f16;
use llama_burn::inference::{streaming_handler, GenerateRequest};
use llama_burn::llama::LlamaConfig;
use std::time::Instant;

#[cfg(feature = "cuda")]
type Back = Cuda<f16, i32>;

fn main() {
    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("This example requires the 'cuda' feature to be enabled");
        eprintln!("Run with: cargo run --example inference_server --features llama3,cuda");
        return;
    }

    #[cfg(feature = "cuda")]
    run();
}

#[cfg(feature = "cuda")]
fn run() {
    let device = CudaDevice::default();
    println!("Using CUDA device: {:?}", device);

    // Load the model
    #[cfg(feature = "llama3")]
    {
        println!("Loading Llama 3.2 1B model...");
        let llama = LlamaConfig::llama3_2_1b_pretrained::<Back>(256, &device)
            .expect("Failed to load model");
        run_inference_demo(llama, device);
    }

    #[cfg(feature = "tiny")]
    {
        println!("Loading TinyLlama model...");
        let llama =
            LlamaConfig::tiny_llama_pretrained::<Back>(256, &device).expect("Failed to load model");
        run_inference_demo(llama, device);
    }

    #[cfg(not(any(feature = "tiny", feature = "llama3")))]
    {
        eprintln!("Please enable either 'tiny' or 'llama3' feature");
    }
}

fn run_inference_demo<B: Backend, T: llama_burn::tokenizer::Tokenizer + Send + 'static>(
    llama: llama_burn::llama::Llama<B, T>,
    device: B::Device,
) {
    // Build the inference instance using the convenient builder API
    let inference = burn_central::runtime::inference::InferenceBuilder::<B>::new()
        .with_model(llama)
        .build(streaming_handler);

    println!("Inference server ready!\n");

    // ============================================================
    // Example 1: Streaming generation with real-time token output
    // ============================================================

    let request = GenerateRequest::new("The future of artificial intelligence is")
        .with_max_tokens(50)
        .with_temperature(0.7)
        .with_top_p(0.9);

    println!("Prompt: {}\n", request.prompt);
    println!("Generated tokens:");

    let start = Instant::now();
    let job = inference
        .infer(request)
        .with_devices([device.clone()])
        .spawn();

    let mut token_count = 0;
    let mut full_text = String::new();

    for token in job.stream.iter() {
        print!("{}", token.token);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        full_text.push_str(&token.token);
        token_count += 1;
    }

    let elapsed = start.elapsed().as_secs_f64();

    println!("\n\n--- Generation Complete ---");
    println!("Tokens: {}", token_count);
    println!("Time: {:.2}s", elapsed);
    println!("Tokens/s: {:.2}", token_count as f64 / elapsed);

    match job.join() {
        Ok(_) => println!("Job completed successfully"),
        Err(e) => eprintln!("Job failed: {:?}", e),
    }

    // ============================================================
    // Example 2: Cancellation
    // ============================================================

    println!("\n--- Example: Cancellation ---");

    let request = GenerateRequest::new("Once upon a time")
        .with_max_tokens(100)
        .with_temperature(0.7);

    let job = inference
        .infer(request)
        .with_devices([device.clone()])
        .spawn();

    let mut count = 0;
    for token in job.stream.iter() {
        print!("{}", token.token);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        count += 1;
        if count >= 10 {
            println!("\n\n[Cancelling after 10 tokens...]");
            job.cancel();
            break;
        }
    }

    match job.join() {
        Ok(_) => println!("Job completed"),
        Err(e) => println!("Job cancelled as expected: {:?}", e),
    }

    // ============================================================
    // Example 3: Synchronous (non-streaming) generation
    // ============================================================

    println!("\n--- Example: Synchronous Generation ---");

    let request = GenerateRequest::new("In a galaxy far, far away")
        .with_max_tokens(30)
        .with_temperature(0.8);

    println!("Prompt: {}", request.prompt);

    let start = Instant::now();
    let outputs = inference
        .infer(request)
        .with_devices([device.clone()])
        .run()
        .expect("Generation failed");

    let elapsed = start.elapsed().as_secs_f64();

    println!("\nAll tokens collected:");
    for token in &outputs {
        print!("{}", token.token);
    }

    println!("\n\nCollected {} tokens in {:.2}s", outputs.len(), elapsed);
}
