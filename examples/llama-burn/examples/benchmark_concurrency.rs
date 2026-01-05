//! Benchmark comparing blocking vs concurrent inference handlers.
//!
//! This benchmark measures the performance difference between:
//! 1. Blocking handler: Entire generation in one model.submit() call
//! 2. Concurrent handler: One token at a time, releases model between tokens
//!
//! Run with: cargo run --example benchmark_concurrency --features llama3,cuda --release

#[cfg(feature = "cuda")]
use burn::backend::{cuda::CudaDevice, Cuda};
#[cfg(feature = "cuda")]
use burn::prelude::Backend;
use burn::tensor::f16;
use llama_burn::inference::{concurrent_streaming_handler, streaming_handler, GenerateRequest};
use llama_burn::llama::LlamaConfig;
use std::sync::Arc;
use std::time::Instant;

#[cfg(feature = "cuda")]
type Back = Cuda<f16, i32>;

fn main() {
    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("This benchmark requires the 'cuda' feature to be enabled");
        eprintln!(
            "Run with: cargo run --example benchmark_concurrency --features llama3,cuda --release"
        );
        return;
    }

    #[cfg(feature = "cuda")]
    run();
}

#[cfg(feature = "cuda")]
fn run() {
    let device = CudaDevice::default();
    println!("🔧 Concurrency Benchmark");
    println!("{}", "━".repeat(80));
    println!("Device: {:?}", device);

    #[cfg(feature = "llama3")]
    {
        println!("Model: Llama 3.2 1B");
        println!("Loading model...");
        let llama = LlamaConfig::llama3_2_1b_pretrained::<Back>(256, &device)
            .expect("Failed to load model");
        run_benchmarks(llama, device);
    }

    #[cfg(feature = "tiny")]
    {
        println!("Model: TinyLlama");
        println!("Loading model...");
        let llama =
            LlamaConfig::tiny_llama_pretrained::<Back>(256, &device).expect("Failed to load model");
        run_benchmarks(llama, device);
    }

    #[cfg(not(any(feature = "tiny", feature = "llama3")))]
    {
        eprintln!("Please enable either 'tiny' or 'llama3' feature");
    }
}

#[cfg(feature = "cuda")]
fn run_benchmarks<B: Backend, T: llama_burn::tokenizer::Tokenizer + Send + 'static>(
    llama: llama_burn::llama::Llama<B, T>,
    device: B::Device,
) {
    println!("\n✓ Model loaded successfully\n");
    println!("{}", "━".repeat(80));

    // Test parameters
    let num_requests = 5;
    let tokens_per_request = 30;

    let prompts = vec![
        "The future of artificial intelligence",
        "Once upon a time in a galaxy",
        "The quick brown fox jumps",
        "In the beginning there was",
        "The meaning of life is",
    ];

    // ============================================================
    // Benchmark 1: Blocking Handler (Sequential Processing)
    // ============================================================

    println!("\n📊 Benchmark 1: BLOCKING Handler (streaming_handler)");
    println!("{}", "─".repeat(80));
    println!("Configuration:");
    println!("  • Requests: {}", num_requests);
    println!("  • Tokens/request: {}", tokens_per_request);
    println!("  • Total tokens: {}", num_requests * tokens_per_request);
    println!("\nProcessing (entire generation blocks model)...\n");

    let blocking_inference = Arc::new(
        burn_central::runtime::inference::InferenceBuilder::<B>::new()
            .with_model(llama)
            .build(streaming_handler),
    );

    let start = Instant::now();
    let mut blocking_results = Vec::new();

    for (i, prompt) in prompts.iter().enumerate() {
        let request = GenerateRequest::new(*prompt)
            .with_max_tokens(tokens_per_request)
            .with_temperature(0.7)
            .with_seed(42 + i as u64);

        let job = blocking_inference
            .infer(request)
            .with_devices([device.clone()])
            .spawn();

        let mut token_count = 0;
        for _token in job.stream.iter() {
            token_count += 1;
        }

        let _ = job.join();
        blocking_results.push(token_count);
        println!("  ✓ Request {} completed: {} tokens", i + 1, token_count);
    }

    let blocking_time = start.elapsed().as_secs_f64();
    let blocking_total_tokens: usize = blocking_results.iter().sum();

    println!("\nResults:");
    println!("  • Total time: {:.2}s", blocking_time);
    println!("  • Total tokens: {}", blocking_total_tokens);
    println!(
        "  • Throughput: {:.2} tok/s",
        blocking_total_tokens as f64 / blocking_time
    );
    println!(
        "  • Avg latency/request: {:.2}s",
        blocking_time / num_requests as f64
    );

    // Get the model back from the blocking inference
    let llama = Arc::try_unwrap(blocking_inference)
        .ok()
        .expect("Failed to unwrap Arc")
        .into_model();

    // ============================================================
    // Benchmark 2: Concurrent Handler (Interleaved Processing)
    // ============================================================

    println!("{}", "\n━".repeat(80));
    println!("\n📊 Benchmark 2: CONCURRENT Handler (concurrent_streaming_handler)");
    println!("{}", "─".repeat(80));
    println!("Configuration:");
    println!("  • Requests: {}", num_requests);
    println!("  • Tokens/request: {}", tokens_per_request);
    println!("  • Total tokens: {}", num_requests * tokens_per_request);
    println!("\nProcessing (requests interleave at token boundaries)...\n");

    let concurrent_inference = Arc::new(
        burn_central::runtime::inference::InferenceBuilder::<B>::new()
            .with_model(llama)
            .build(concurrent_streaming_handler),
    );

    let start = Instant::now();

    let handles: Vec<_> = prompts
        .iter()
        .enumerate()
        .map(|(i, prompt)| {
            let inf = Arc::clone(&concurrent_inference);
            let dev = device.clone();
            let request = GenerateRequest::new(*prompt)
                .with_max_tokens(tokens_per_request)
                .with_temperature(0.7)
                .with_seed(42 + i as u64);

            std::thread::spawn(move || {
                let job = inf.infer(request).with_devices([dev]).spawn();
                let mut token_count = 0;
                for _token in job.stream.iter() {
                    token_count += 1;
                }
                let _ = job.join();
                (i + 1, token_count)
            })
        })
        .collect();

    let mut concurrent_results = Vec::new();
    for handle in handles {
        let (id, count) = handle.join().unwrap();
        concurrent_results.push(count);
        println!("  ✓ Request {} completed: {} tokens", id, count);
    }

    let concurrent_time = start.elapsed().as_secs_f64();
    let concurrent_total_tokens: usize = concurrent_results.iter().sum();

    println!("\nResults:");
    println!("  • Total time: {:.2}s", concurrent_time);
    println!("  • Total tokens: {}", concurrent_total_tokens);
    println!(
        "  • Throughput: {:.2} tok/s",
        concurrent_total_tokens as f64 / concurrent_time
    );
    println!(
        "  • Avg latency/request: {:.2}s",
        concurrent_time / num_requests as f64
    );

    // ============================================================
    // Comparison
    // ============================================================

    println!("{}", "\n━".repeat(80));
    println!("\n📈 COMPARISON");
    println!("{}", "━".repeat(80));

    let speedup = blocking_time / concurrent_time;
    let throughput_improvement = (concurrent_total_tokens as f64 / concurrent_time)
        / (blocking_total_tokens as f64 / blocking_time);

    println!("\n┌─────────────────────────┬─────────────┬─────────────┬──────────┐");
    println!("│ Metric                  │   Blocking  │ Concurrent  │   Ratio  │");
    println!("├─────────────────────────┼─────────────┼─────────────┼──────────┤");
    println!(
        "│ Total time              │   {:>7.2}s  │   {:>7.2}s  │  {:>5.2}x  │",
        blocking_time, concurrent_time, speedup
    );
    println!(
        "│ Throughput (tok/s)      │   {:>7.2}   │   {:>7.2}   │  {:>5.2}x  │",
        blocking_total_tokens as f64 / blocking_time,
        concurrent_total_tokens as f64 / concurrent_time,
        throughput_improvement
    );
    println!(
        "│ Avg latency/request (s) │   {:>7.2}   │   {:>7.2}   │  {:>5.2}x  │",
        blocking_time / num_requests as f64,
        concurrent_time / num_requests as f64,
        (blocking_time / num_requests as f64) / (concurrent_time / num_requests as f64)
    );
    println!("└─────────────────────────┴─────────────┴─────────────┴──────────┘");

    println!("\n🎯 Key Insights:");
    if speedup > 1.5 {
        println!("  ✓ Concurrent handler is {:.1}x FASTER", speedup);
        println!("  ✓ Requests interleaved successfully at token boundaries");
        println!("  ✓ Better GPU utilization achieved");
    } else if speedup > 1.1 {
        println!("  ✓ Concurrent handler is {:.1}x faster", speedup);
        println!("  ✓ Modest improvement due to interleaving");
    } else {
        println!("  ⚠ Similar performance (speedup: {:.2}x)", speedup);
        println!("  ℹ Try more requests or longer sequences for better gains");
    }

    println!("\n💡 Recommendations:");
    println!("  • For single requests: Both handlers perform similarly");
    println!("  • For 2+ concurrent requests: Use concurrent_streaming_handler");
    println!("  • For high load (10+ requests): Consider batch inference");
    println!("  • See CONCURRENT_INFERENCE.md for advanced strategies");

    println!("{}", "━".repeat(80));
    println!("✅ Benchmark complete!\n");
}
