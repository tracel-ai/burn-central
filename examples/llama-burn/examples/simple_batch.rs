//! Simple batched generation example for llama-burn.
//!
//! This example demonstrates the core batched generation API without requiring
//! pretrained weights. It shows how multiple sequences can be processed in parallel
//! for maximum throughput.
//!
//! Run with:
//! ```sh
//! cargo run --example simple_batch --features llama3,cuda
//! ```

use burn::tensor::backend::Backend;
use llama_burn::{inference::generate_batch, llama::Llama, tokenizer::Tokenizer};
use std::time::Instant;

/// Demonstrates batched generation with multiple prompts.
fn run_batched_demo<B: Backend, T: Tokenizer>(llama: &mut Llama<B, T>) {
    println!("{}", "=".repeat(70));
    println!("Batched Generation Demo");
    println!("{}", "=".repeat(70));
    println!();

    // Example 1: Basic batch processing
    println!("Example 1: Basic Batch Processing");
    println!("{}", "-".repeat(70));

    let prompts = vec![
        "Once upon a time",
        "In a galaxy far away",
        "The quick brown fox",
    ];

    println!("Processing {} prompts in parallel...\n", prompts.len());

    let start = Instant::now();
    let results = generate_batch(
        llama, &prompts, 20,   // max_tokens
        0.8,  // temperature
        0.95, // top_p
        42,   // seed
    )
    .expect("Batch generation failed");

    let elapsed = start.elapsed();

    for (i, (prompt, tokens)) in prompts.iter().zip(results.iter()).enumerate() {
        println!("Sequence {}: {}", i + 1, prompt);
        print!("Generated: ");
        for token in tokens {
            print!("{}", token);
        }
        println!(" ({} tokens)", tokens.len());
        println!();
    }

    println!("Batch completed in {:.2}s", elapsed.as_secs_f32());
    println!();

    // Example 2: Throughput measurement
    println!("Example 2: Throughput Measurement");
    println!("{}", "-".repeat(70));

    let batch_size = 5;
    let tokens_per_seq = 15;

    let test_prompts: Vec<String> = (0..batch_size)
        .map(|i| format!("Test prompt number {}", i))
        .collect();
    let test_prompt_refs: Vec<&str> = test_prompts.iter().map(|s| s.as_str()).collect();

    println!(
        "Generating {} tokens for {} sequences...",
        tokens_per_seq, batch_size
    );

    let start = Instant::now();
    let results = generate_batch(llama, &test_prompt_refs, tokens_per_seq, 0.7, 0.9, 123)
        .expect("Batch generation failed");

    let elapsed = start.elapsed();
    let total_tokens: usize = results.iter().map(|r| r.len()).sum();
    let throughput = total_tokens as f32 / elapsed.as_secs_f32();

    println!("Results:");
    println!("  Total tokens generated: {}", total_tokens);
    println!("  Time: {:.2}s", elapsed.as_secs_f32());
    println!("  Throughput: {:.2} tokens/s", throughput);
    println!(
        "  Average per sequence: {:.2} tokens/s",
        throughput / batch_size as f32
    );
    println!();

    // Example 3: Variable-length prompts
    println!("Example 3: Variable-Length Prompts");
    println!("{}", "-".repeat(70));

    let varied_prompts = vec![
        ("Short", "Hello", 5),
        ("Medium", "Tell me about AI", 10),
        (
            "Long",
            "Explain the concept of neural networks in detail",
            15,
        ),
    ];

    println!("Processing prompts of different lengths in one batch...\n");

    for (label, prompt, _) in &varied_prompts {
        println!("[{}] \"{}\"", label, prompt);
    }
    println!();

    let prompts_only: Vec<&str> = varied_prompts.iter().map(|(_, p, _)| *p).collect();
    let max_tokens_varied = varied_prompts
        .iter()
        .map(|(_, _, m)| *m)
        .max()
        .unwrap_or(15);

    let start = Instant::now();
    let results = generate_batch(llama, &prompts_only, max_tokens_varied, 0.7, 0.9, 456)
        .expect("Batch generation failed");

    println!("Results:");
    for (i, ((label, prompt, _), tokens)) in varied_prompts.iter().zip(results.iter()).enumerate() {
        println!(
            "[{}] \"{}\": {} tokens generated",
            label,
            prompt,
            tokens.len()
        );
    }

    println!(
        "\nVariable-length batch completed in {:.2}s",
        start.elapsed().as_secs_f32()
    );
    println!();

    println!("{}", "=".repeat(70));
    println!("Batched generation demo completed!");
    println!("{}", "=".repeat(70));
}

#[cfg(all(feature = "cuda", feature = "llama3"))]
mod cuda {
    use super::*;
    use burn::{
        backend::{cuda::CudaDevice, Cuda},
        tensor::f16,
    };

    pub fn run() {
        let device = CudaDevice::default();

        println!("\nInitializing Llama model (CUDA backend)...");

        // Create a small model for demonstration
        // Note: This creates a new model, not loading pretrained weights
        let config = LlamaConfig::llama3_2_1b("/path/to/tokenizer.json");
        let mut llama = config.init::<Cuda<f16, i32>>(&device);

        println!("Model initialized!\n");

        run_batched_demo(&mut llama);
    }
}

#[cfg(all(feature = "tch-gpu", feature = "llama3"))]
mod tch_gpu {
    use super::*;
    use burn::{
        backend::{libtorch::LibTorchDevice, LibTorch},
        tensor::f16,
    };

    pub fn run() {
        #[cfg(not(target_os = "macos"))]
        let device = LibTorchDevice::Cuda(0);
        #[cfg(target_os = "macos")]
        let device = LibTorchDevice::Mps;

        println!("\nInitializing Llama model (LibTorch backend)...");

        let config = LlamaConfig::llama3_2_1b("/path/to/tokenizer.json");
        let mut llama = config.init::<LibTorch<f16>>(&device);

        println!("Model initialized!\n");

        run_batched_demo(&mut llama);
    }
}

fn main() {
    #[cfg(all(feature = "cuda", feature = "llama3"))]
    cuda::run();

    #[cfg(all(feature = "tch-gpu", feature = "llama3"))]
    tch_gpu::run();

    #[cfg(not(any(
        all(feature = "cuda", feature = "llama3"),
        all(feature = "tch-gpu", feature = "llama3")
    )))]
    {
        eprintln!("Error: This example requires 'llama3' feature with either 'cuda' or 'tch-gpu'.");
        eprintln!("Run with: cargo run --example simple_batch --features llama3,cuda");
        std::process::exit(1);
    }
}
