//! Continuous batching inference server example.
//!
//! This example demonstrates dynamic batching (continuous batching) where
//! new requests can join mid-stream and completed requests are removed
//! without waiting for a fixed batch boundary.
//!
//! Run with: cargo run --example continuous_server --features llama3,cuda

#[cfg(feature = "cuda")]
use burn::backend::{cuda::CudaDevice, Cuda};
#[cfg(feature = "cuda")]
use burn::prelude::Backend;
use burn::tensor::f16;
use llama_burn::inference::{continuous_batched_streaming_handler, GenerateRequest};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(feature = "pretrained")]
use burn::record::{HalfPrecisionSettings, NamedMpkFileRecorder};
#[cfg(any(feature = "llama3", feature = "tiny"))]
use llama_burn::llama::LlamaConfig;
#[cfg(feature = "pretrained")]
use llama_burn::pretrained::{self, ModelMeta};
#[cfg(feature = "tiny")]
use llama_burn::tokenizer::SentiencePieceTokenizer;
#[cfg(feature = "llama3")]
use llama_burn::tokenizer::Tiktoken;

#[cfg(feature = "cuda")]
type Back = Cuda<f16, i32>;

fn main() {
    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("This example requires the 'cuda' feature to be enabled");
        eprintln!("Run with: cargo run --example continuous_server --features llama3,cuda");
        return;
    }

    #[cfg(feature = "cuda")]
    run();
}

#[cfg(feature = "cuda")]
fn run() {
    let device = CudaDevice::default();
    println!("Using CUDA device: {:?}", device);

    let max_seq_len = 256;
    let max_batch_size = 8;

    #[cfg(all(feature = "llama3", feature = "pretrained"))]
    {
        println!("Loading Llama 3.2 1B model (continuous batching)...");
        let llama =
            load_llama3_2_1b_with_batch::<Back>(max_seq_len, max_batch_size, &device).unwrap();
        run_continuous_demo(llama, device);
    }

    #[cfg(all(feature = "tiny", feature = "pretrained"))]
    {
        println!("Loading TinyLlama model (continuous batching)...");
        let llama =
            load_tiny_llama_with_batch::<Back>(max_seq_len, max_batch_size, &device).unwrap();
        run_continuous_demo(llama, device);
    }

    #[cfg(not(any(feature = "tiny", feature = "llama3")))]
    {
        eprintln!("Please enable either 'tiny' or 'llama3' feature");
    }
}

#[cfg(all(feature = "llama3", feature = "pretrained"))]
fn load_llama3_2_1b_with_batch<B: Backend>(
    max_seq_len: usize,
    max_batch_size: usize,
    device: &B::Device,
) -> Result<llama_burn::llama::Llama<B, Tiktoken>, String> {
    let model = pretrained::Llama::Llama321bInstruct.pretrained();
    let checkpoint = model
        .download_weights()
        .map_err(|err| format!("Could not download weights.\nError: {err}"))?;
    let tokenizer = model
        .download_tokenizer()
        .map_err(|err| format!("Could not download tokenizer.\nError: {err}"))?;

    let tokenizer_path = tokenizer
        .to_str()
        .ok_or("Tokenizer path is not valid UTF-8")?;
    let checkpoint_path = checkpoint
        .to_str()
        .ok_or("Checkpoint path is not valid UTF-8")?;

    let llama = LlamaConfig::llama3_2_1b(tokenizer_path)
        .with_max_seq_len(max_seq_len)
        .with_max_batch_size(max_batch_size)
        .init::<B, Tiktoken>(device)?;

    let recorder = NamedMpkFileRecorder::<HalfPrecisionSettings>::new();
    let llama = llama
        .load(checkpoint_path, &recorder)
        .map_err(|err| format!("Failed to load model.\nError: {err}"))?;

    Ok(llama)
}

#[cfg(all(feature = "tiny", feature = "pretrained"))]
fn load_tiny_llama_with_batch<B: Backend>(
    max_seq_len: usize,
    max_batch_size: usize,
    device: &B::Device,
) -> Result<llama_burn::llama::Llama<B, SentiencePieceTokenizer>, String> {
    let model = pretrained::Llama::TinyLlama.pretrained();
    let checkpoint = model
        .download_weights()
        .map_err(|err| format!("Could not download weights.\nError: {err}"))?;
    let tokenizer = model
        .download_tokenizer()
        .map_err(|err| format!("Could not download tokenizer.\nError: {err}"))?;

    let tokenizer_path = tokenizer
        .to_str()
        .ok_or("Tokenizer path is not valid UTF-8")?;
    let checkpoint_path = checkpoint
        .to_str()
        .ok_or("Checkpoint path is not valid UTF-8")?;

    let llama = LlamaConfig::tiny_llama(tokenizer_path)
        .with_max_seq_len(max_seq_len)
        .with_max_batch_size(max_batch_size)
        .init::<B, SentiencePieceTokenizer>(device)?;

    let recorder = NamedMpkFileRecorder::<HalfPrecisionSettings>::new();
    let llama = llama
        .load(checkpoint_path, &recorder)
        .map_err(|err| format!("Failed to load model.\nError: {err}"))?;

    Ok(llama)
}

fn run_continuous_demo<B: Backend, T: llama_burn::tokenizer::Tokenizer + Send + Sync + 'static>(
    llama: llama_burn::llama::Llama<B, T>,
    device: B::Device,
) {
    let inference = burn_central::runtime::inference::InferenceBuilder::<B>::new()
        .with_model(llama)
        .build(continuous_batched_streaming_handler);

    println!("Continuous batching server ready!\n");
    println!("{}", "=".repeat(80));

    // ============================================================
    // Example 1: Staggered requests (mid-stream joins)
    // ============================================================
    println!("\n[Example 1: Staggered Requests]");
    println!("Launching 4 requests with small delays...\n");

    let inference = Arc::new(inference);
    let prompts = vec![
        "Once upon a time in a distant galaxy",
        "The quick brown fox jumps over",
        "In the beginning, there was",
        "Explain the benefits of continuous batching",
    ];
    let delays = vec![0_u64, 80, 160, 240];

    let start = Instant::now();
    let handles: Vec<_> = prompts
        .into_iter()
        .zip(delays.into_iter())
        .enumerate()
        .map(|(i, (prompt, delay_ms))| {
            let inf = Arc::clone(&inference);
            let dev = device.clone();
            let request = GenerateRequest::new(prompt)
                .with_max_tokens(25)
                .with_temperature(0.7)
                .with_seed(42 + i as u64);

            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(delay_ms));
                println!("[Request {}] Started after {}ms", i + 1, delay_ms);

                let job = inf.infer(request).with_devices([dev]).spawn();
                let mut full_text = String::new();
                let mut token_count = 0;

                for token in job.stream.iter() {
                    full_text.push_str(&token.token);
                    token_count += 1;
                }

                job.join().ok();
                (i + 1, full_text, token_count)
            })
        })
        .collect();

    let mut results = Vec::new();
    for handle in handles {
        results.push(handle.join().unwrap());
    }
    results.sort_by_key(|(id, _, _)| *id);

    let elapsed = start.elapsed().as_secs_f64();
    let total_tokens: usize = results.iter().map(|(_, _, count)| count).sum();

    println!("\n--- Results ---");
    for (id, text, count) in results {
        println!("\n[Request {}]", id);
        println!("Generated: {}", text);
        println!("Tokens: {}", count);
    }

    println!("\n--- Summary ---");
    println!("Total tokens: {}", total_tokens);
    println!("Total time: {:.2}s", elapsed);
    println!(
        "Average throughput: {:.2} tok/s",
        total_tokens as f64 / elapsed
    );

    // ============================================================
    // Example 2: Cancellation
    // ============================================================
    println!("\n{}", "=".repeat(80));
    println!("\n[Example 2: Cancellation]");
    println!("Starting a request and cancelling after 5 tokens...\n");

    let request = GenerateRequest::new("The meaning of life is")
        .with_max_tokens(50)
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
        if count >= 5 {
            println!("\n\n[Cancelling after {} tokens...]", count);
            job.cancel();
            break;
        }
    }

    match job.join() {
        Ok(_) => println!("Job completed"),
        Err(e) => println!("Job cancelled as expected: {:?}", e),
    }

    println!("\n{}", "=".repeat(80));
    println!("\nAll examples completed.");
}
