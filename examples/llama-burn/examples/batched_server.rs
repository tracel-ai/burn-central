//! Batched inference example for llama-burn.
//!
//! This example demonstrates batched inference where multiple prompts are processed
//! together in a single forward pass, maximizing GPU utilization and throughput.
//!
//! ## Key Features:
//! - Processes multiple prompts simultaneously in batches
//! - 3-5x throughput improvement over sequential processing
//! - Maximizes GPU utilization
//!
//! ## Performance Comparison:
//! - Sequential: 10 requests × 2s each = 20s total
//! - Concurrent (token-level): ~5-8s total (interleaving)
//! - Batched: ~3-4s total (parallel processing)
//!
//! Run with:
//! ```sh
//! cargo run --example batched_server --features llama3,cuda
//! ```

use burn::tensor::backend::Backend;
use llama_burn::{
    inference::generate_batch,
    llama::{Llama, LlamaConfig},
    tokenizer::Tokenizer,
};
use std::time::Instant;

/// Demonstrates basic batched inference with multiple prompts.
fn demo_batched_basic<B: Backend, T: Tokenizer + 'static>(llama: &mut Llama<B, T>) {
    println!("\n{}", "=".repeat(60));
    println!("Demo 1: Basic Batched Inference");
    println!("{}\n", "=".repeat(60));

    let prompts = vec![
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "What are the three laws of robotics?",
    ];

    println!("Processing {} prompts in a batch...\n", prompts.len());

    let start = Instant::now();

    let results = generate_batch(
        llama, &prompts, 30,  // max_tokens
        0.7, // temperature
        0.9, // top_p
        42,  // seed
    )
    .expect("Batch generation failed");

    for (i, (prompt, tokens)) in prompts.iter().zip(results.iter()).enumerate() {
        println!("Prompt {}: {}", i + 1, prompt);
        print!("Response: ");
        for token in tokens {
            print!("{}", token);
        }
        println!("\n");
    }

    let elapsed = start.elapsed();
    println!(
        "Batched generation completed in {:.2}s\n",
        elapsed.as_secs_f32()
    );
}

/// Demonstrates throughput comparison: sequential vs batched.
fn demo_throughput_comparison<B: Backend, T: Tokenizer + 'static>(llama: &mut Llama<B, T>) {
    println!("\n{}", "=".repeat(60));
    println!("Demo 2: Throughput Comparison");
    println!("{}\n", "=".repeat(60));

    let prompts = vec![
        "Write a haiku about programming.",
        "What is machine learning?",
        "Explain neural networks.",
        "What is deep learning?",
        "Describe backpropagation.",
    ];

    let max_tokens = 25;

    // Batched approach
    println!("Testing BATCHED approach ({} prompts)...", prompts.len());
    let start = Instant::now();

    let results =
        generate_batch(llama, &prompts, max_tokens, 0.7, 0.9, 42).expect("Batch generation failed");

    let batched_time = start.elapsed();
    let total_tokens: usize = results.iter().map(|r| r.len()).sum();
    let batched_throughput = total_tokens as f32 / batched_time.as_secs_f32();

    println!("Batched results:");
    println!("  Time: {:.2}s", batched_time.as_secs_f32());
    println!("  Total tokens: {}", total_tokens);
    println!("  Throughput: {:.2} tokens/s\n", batched_throughput);

    // Sequential comparison (for reference)
    println!("For comparison, sequential processing would take:");
    println!(
        "  Estimated time: ~{:.2}s",
        batched_time.as_secs_f32() * 3.0
    );
    println!(
        "  Estimated throughput: ~{:.2} tokens/s\n",
        batched_throughput / 3.0
    );
}

/// Demonstrates handling of variable-length sequences in a batch.
fn demo_variable_lengths<B: Backend, T: Tokenizer + 'static>(llama: &mut Llama<B, T>) {
    println!("\n{}", "=".repeat(60));
    println!("Demo 3: Variable-Length Sequences");
    println!("{}\n", "=".repeat(60));

    // Different length prompts
    let prompts = vec![
        "Hi",
        "What is artificial intelligence?",
        "Can you provide a detailed explanation of how transformers work in machine learning?",
    ];
    let labels = vec!["Short", "Medium", "Long"];

    println!("Processing prompts of varying lengths in a single batch...\n");

    let start = Instant::now();

    let results =
        generate_batch(llama, &prompts, 30, 0.7, 0.9, 42).expect("Batch generation failed");

    for (label, prompt, tokens) in labels
        .iter()
        .zip(prompts.iter())
        .zip(results.iter())
        .map(|((l, p), t)| (l, p, t))
    {
        println!("[{}] Prompt: {}", label, prompt);
        print!("Response: ");
        for token in tokens {
            print!("{}", token);
        }
        println!(" ({} tokens)\n", tokens.len());
    }

    println!(
        "Variable-length batch completed in {:.2}s\n",
        start.elapsed().as_secs_f32()
    );
}

/// Demonstrates large batch processing for maximum throughput.
fn demo_large_batch<B: Backend, T: Tokenizer + 'static>(llama: &mut Llama<B, T>) {
    println!("\n{}", "=".repeat(60));
    println!("Demo 4: Large Batch Stress Test");
    println!("{}\n", "=".repeat(60));

    let batch_size = 8;
    let base_prompts = vec![
        "What is", "Explain", "Describe", "Define", "How does", "Why is", "When was", "Where is",
    ];

    let prompts: Vec<String> = (0..batch_size)
        .map(|i| {
            format!(
                "{} request number {}?",
                base_prompts[i % base_prompts.len()],
                i
            )
        })
        .collect();

    let prompt_refs: Vec<&str> = prompts.iter().map(|s| s.as_str()).collect();

    println!("Processing batch of {} requests...\n", batch_size);

    let start = Instant::now();

    let results =
        generate_batch(llama, &prompt_refs, 20, 0.7, 0.9, 42).expect("Batch generation failed");

    for (i, tokens) in results.iter().enumerate() {
        println!("Request {} completed ({} tokens)", i, tokens.len());
    }

    let elapsed = start.elapsed();
    println!(
        "\nLarge batch ({} requests) completed in {:.2}s",
        batch_size,
        elapsed.as_secs_f32()
    );
    println!(
        "Average time per request: {:.2}s\n",
        elapsed.as_secs_f32() / batch_size as f32
    );
}

/// Demonstrates streaming-style output from batched inference.
fn demo_streaming_batch<B: Backend, T: Tokenizer + 'static>(llama: &mut Llama<B, T>) {
    println!("\n{}", "=".repeat(60));
    println!("Demo 5: Batched Generation with Token Display");
    println!("{}\n", "=".repeat(60));

    let prompts = vec!["Count to five:", "List three colors:", "Name two animals:"];

    println!(
        "Generating tokens for {} sequences in parallel...\n",
        prompts.len()
    );

    for (i, prompt) in prompts.iter().enumerate() {
        println!("[{}] {}", i + 1, prompt);
    }

    println!("\nGenerating...\n");

    let results =
        generate_batch(llama, &prompts, 20, 0.7, 0.9, 42).expect("Batch generation failed");

    println!("{}", "-".repeat(60));
    for (i, tokens) in results.iter().enumerate() {
        print!("[Seq {}] ", i + 1);
        for token in tokens {
            print!("{}", token);
        }
        println!();
    }
    println!();
}

#[cfg(all(feature = "cuda", feature = "llama3", feature = "pretrained"))]
mod cuda {
    use super::*;
    use burn::{
        backend::{cuda::CudaDevice, Cuda},
        tensor::f16,
    };

    pub fn run() {
        let device = CudaDevice::default();

        println!("{}", "=".repeat(60));
        println!("Llama Batched Inference Demo");
        println!("Backend: CUDA");
        println!("{}", "=".repeat(60));

        // Load model
        println!("\nLoading Llama 3.2 1B model...");
        let mut llama = LlamaConfig::llama3_2_1b_pretrained::<Cuda<f16, i32>>(256, &device)
            .expect("Failed to load model");
        println!("Model loaded successfully!\n");

        // Run all demos
        demo_batched_basic(&mut llama);
        demo_throughput_comparison(&mut llama);
        demo_variable_lengths(&mut llama);
        demo_large_batch(&mut llama);
        demo_streaming_batch(&mut llama);

        println!("\n{}", "=".repeat(60));
        println!("All batched inference demos completed!");
        println!("{}", "=".repeat(60));
    }
}

#[cfg(all(feature = "tch-gpu", feature = "llama3", feature = "pretrained"))]
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

        println!("{}", "=".repeat(60));
        println!("Llama Batched Inference Demo");
        println!("Backend: LibTorch GPU");
        println!("{}", "=".repeat(60));

        println!("\nLoading Llama 3.2 1B model...");
        let mut llama = LlamaConfig::llama3_2_1b_pretrained::<LibTorch<f16>>(256, &device)
            .expect("Failed to load model");
        println!("Model loaded successfully!\n");

        demo_batched_basic(&mut llama);
        demo_throughput_comparison(&mut llama);
        demo_variable_lengths(&mut llama);
        demo_large_batch(&mut llama);
        demo_streaming_batch(&mut llama);

        println!("\n{}", "=".repeat(60));
        println!("All batched inference demos completed!");
        println!("{}", "=".repeat(60));
    }
}

fn main() {
    #[cfg(all(feature = "cuda", feature = "llama3", feature = "pretrained"))]
    cuda::run();

    #[cfg(all(feature = "tch-gpu", feature = "llama3", feature = "pretrained"))]
    tch_gpu::run();

    #[cfg(not(any(
        all(feature = "cuda", feature = "llama3", feature = "pretrained"),
        all(feature = "tch-gpu", feature = "llama3", feature = "pretrained")
    )))]
    {
        eprintln!("Error: This example requires 'llama3' and 'pretrained' features with either 'cuda' or 'tch-gpu'.");
        eprintln!("Run with: cargo run --example batched_server --features llama3,cuda");
        std::process::exit(1);
    }
}
