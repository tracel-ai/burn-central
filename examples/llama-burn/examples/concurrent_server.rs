//! Concurrent inference server example demonstrating improved concurrency.
//!
//! This example shows the difference between blocking and concurrent inference handlers.
//! The concurrent handler generates one token at a time, releasing the model between
//! tokens to allow multiple requests to interleave.
//!
//! Run with: cargo run --example concurrent_server --features llama3,cuda

#[cfg(feature = "cuda")]
use burn::backend::{cuda::CudaDevice, Cuda};
#[cfg(feature = "cuda")]
use burn::prelude::Backend;
use burn::tensor::f16;
use llama_burn::inference::{concurrent_streaming_handler, GenerateRequest};
use std::sync::Arc;
use std::time::Instant;

#[cfg(feature = "llama3")]
use llama_burn::llama::LlamaConfig;

#[cfg(feature = "cuda")]
type Back = Cuda<f16, i32>;

fn main() {
    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("This example requires the 'cuda' feature to be enabled");
        eprintln!("Run with: cargo run --example concurrent_server --features llama3,cuda");
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
        run_concurrent_demo(llama, device);
    }

    #[cfg(feature = "tiny")]
    {
        println!("Loading TinyLlama model...");
        let llama =
            LlamaConfig::tiny_llama_pretrained::<Back>(256, &device).expect("Failed to load model");
        run_concurrent_demo(llama, device);
    }

    #[cfg(not(any(feature = "tiny", feature = "llama3")))]
    {
        eprintln!("Please enable either 'tiny' or 'llama3' feature");
    }
}

fn run_concurrent_demo<B: Backend, T: llama_burn::tokenizer::Tokenizer + Send + 'static>(
    llama: llama_burn::llama::Llama<B, T>,
    device: B::Device,
) {
    // Build the concurrent inference instance
    let inference = burn_central::runtime::inference::InferenceBuilder::<B>::new()
        .with_model(llama)
        .build(concurrent_streaming_handler);

    println!("Concurrent inference server ready!\n");
    println!("{}", "=".repeat(80));

    // ============================================================
    // Example 1: Single request (baseline)
    // ============================================================

    println!("\n[Example 1: Single Request Baseline]");
    println!("Generating 30 tokens for a single request...\n");

    let request = GenerateRequest::new("The future of artificial intelligence is")
        .with_max_tokens(30)
        .with_temperature(0.7)
        .with_top_p(0.9);

    let start = Instant::now();
    let job = inference
        .infer(request)
        .with_devices([device.clone()])
        .spawn();

    let mut token_count = 0;
    for token in job.stream.iter() {
        print!("{}", token.token);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        token_count += 1;
    }

    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "\n\n✓ Generated {} tokens in {:.2}s ({:.2} tok/s)",
        token_count,
        elapsed,
        token_count as f64 / elapsed
    );

    match job.join() {
        Ok(_) => println!("✓ Job completed successfully"),
        Err(e) => eprintln!("✗ Job failed: {:?}", e),
    }

    // ============================================================
    // Example 2: Concurrent requests
    // ============================================================

    println!("\n{}", "=".repeat(80));
    println!("\n[Example 2: Concurrent Requests]");
    println!("Spawning 3 concurrent requests that will interleave...\n");

    let prompts = vec![
        "Once upon a time in a distant galaxy",
        "The quick brown fox jumps over",
        "In the beginning, there was",
    ];

    let start = Instant::now();
    let inference = Arc::new(inference);

    let handles: Vec<_> = prompts
        .into_iter()
        .enumerate()
        .map(|(i, prompt)| {
            let inf = Arc::clone(&inference);
            let dev = device.clone();
            let request = GenerateRequest::new(prompt)
                .with_max_tokens(20)
                .with_temperature(0.7)
                .with_seed(42 + i as u64);

            std::thread::spawn(move || {
                println!("[Request {}] Started: {}", i + 1, prompt);

                let job = inf.infer(request).with_devices([dev]).spawn();
                let mut tokens = Vec::new();
                let mut full_text = String::new();

                for token in job.stream.iter() {
                    tokens.push(token.token.clone());
                    full_text.push_str(&token.token);
                }

                job.join().ok();

                (i + 1, prompt, full_text, tokens.len())
            })
        })
        .collect();

    // Collect results
    let mut results = Vec::new();
    for handle in handles {
        results.push(handle.join().unwrap());
    }

    let total_elapsed = start.elapsed().as_secs_f64();

    // Sort by request ID for consistent output
    results.sort_by_key(|(id, _, _, _)| *id);

    println!("\n--- Results ---");
    let total_tokens: usize = results.iter().map(|(_, _, _, count)| count).sum();

    for (id, prompt, text, token_count) in &results {
        println!("\n[Request {}]", id);
        println!("Prompt: {}", prompt);
        println!("Generated: {}", text);
        println!("Tokens: {}", token_count);
    }

    println!("\n--- Summary ---");
    println!("Total requests: 3");
    println!("Total tokens: {}", total_tokens);
    println!("Total time: {:.2}s", total_elapsed);
    println!(
        "Average throughput: {:.2} tok/s",
        total_tokens as f64 / total_elapsed
    );
    println!("\n✓ All requests processed concurrently!");
    println!("  (Requests interleaved at token boundaries,");
    println!("   allowing better utilization of the model)");

    // ============================================================
    // Example 3: Cancellation with concurrent requests
    // ============================================================

    println!("\n{}", "=".repeat(80));
    println!("\n[Example 3: Cancellation]");
    println!("Starting a request and cancelling it after 5 tokens...\n");

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
        Ok(_) => println!("✓ Job completed"),
        Err(e) => println!("✓ Job cancelled as expected: {:?}", e),
    }

    // ============================================================
    // Example 4: Stress test - many concurrent requests
    // ============================================================

    println!("\n{}", "=".repeat(80));
    println!("\n[Example 4: Stress Test]");
    println!("Spawning 10 concurrent short requests...\n");

    let start = Instant::now();
    let num_requests = 10;

    let handles: Vec<_> = (0..num_requests)
        .map(|i| {
            let inf = Arc::clone(&inference);
            let dev = device.clone();
            let request = GenerateRequest::new(format!("Request {}: Hello", i + 1))
                .with_max_tokens(10)
                .with_temperature(0.7)
                .with_seed(100 + i);

            std::thread::spawn(move || {
                let job = inf.infer(request).with_devices([dev]).spawn();
                let mut token_count = 0;
                for _token in job.stream.iter() {
                    token_count += 1;
                }
                job.join().ok();
                token_count
            })
        })
        .collect();

    let mut total_tokens = 0;
    for (i, handle) in handles.into_iter().enumerate() {
        let count = handle.join().unwrap();
        total_tokens += count;
        println!("✓ Request {} completed: {} tokens", i + 1, count);
    }

    let stress_elapsed = start.elapsed().as_secs_f64();
    println!("\n--- Stress Test Summary ---");
    println!("Requests: {}", num_requests);
    println!("Total tokens: {}", total_tokens);
    println!("Time: {:.2}s", stress_elapsed);
    println!(
        "Throughput: {:.2} tok/s",
        total_tokens as f64 / stress_elapsed
    );
    println!(
        "\n✓ Successfully handled {} concurrent requests!",
        num_requests
    );

    println!("\n{}", "=".repeat(80));
    println!("\n🎉 All examples completed successfully!");
    println!("\nKey takeaways:");
    println!("• Concurrent handler allows multiple requests to interleave");
    println!("• Model is released between tokens, enabling better concurrency");
    println!("• Cancellation works correctly even with concurrent requests");
    println!("• Throughput scales with number of concurrent requests");
    println!("\nFor even better performance, consider implementing batch inference!");
}
