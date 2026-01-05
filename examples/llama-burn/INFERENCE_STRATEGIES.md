# Inference Strategies for Llama-Burn

This guide provides a comprehensive overview of the three inference strategies available in llama-burn, helping you choose the right approach for your use case.

## Overview

Llama-burn offers three distinct inference strategies, each optimized for different scenarios:

1. **Sequential (Blocking)** - Simple, one request at a time
2. **Concurrent (Token-level interleaving)** - Good balance of simplicity and performance
3. **Batched (Parallel processing)** - Maximum throughput for high-load scenarios

## Quick Comparison

| Strategy | Throughput | Latency | Complexity | Best For |
|----------|-----------|---------|------------|----------|
| **Sequential** | 1x (baseline) | Lowest | Simple | Single user, simple apps |
| **Concurrent** | 2-4x | Low | Medium | Multi-user, balanced workload |
| **Batched** | **5-7x** | Medium | Higher | High-load production, max throughput |

### Performance Example (10 requests, 50 tokens each)

- **Sequential**: ~20 seconds - processes one at a time
- **Concurrent**: ~5-8 seconds - interleaves token generation
- **Batched**: ~3-4 seconds - processes all in parallel

## Strategy 1: Sequential (Blocking)

### Description

The simplest approach. Each request is processed completely before the next one starts. The model is locked for the entire duration of each generation.

### Code Example

```rust
use llama_burn::{
    llama::LlamaConfig,
    sampling::{Sampler, TopP},
};

let mut llama = LlamaConfig::llama3_2_1b_pretrained(256, &device)?;
let mut sampler = Sampler::TopP(TopP::new(0.9, 42));

// Generate one request at a time
let result = llama.generate(
    "What is AI?",
    50,     // max_tokens
    0.7,    // temperature
    &mut sampler,
);

println!("{}", result.text);
```

### When to Use

✅ **Use Sequential When:**
- Single user application
- Simple chatbot or assistant
- Development/testing
- Minimum complexity needed
- Memory is extremely limited

❌ **Avoid Sequential When:**
- Multiple concurrent users
- High throughput requirements
- Need to maximize GPU utilization

### Performance Characteristics

- **Throughput**: Baseline (1x)
- **Latency**: Lowest per request
- **GPU Utilization**: Low (10-30%)
- **Memory Usage**: Minimal
- **Implementation Complexity**: Very simple

## Strategy 2: Concurrent (Token-level Interleaving)

### Description

Generates one token at a time for each request, releasing the model between tokens. Multiple requests can interleave their token generation, significantly improving concurrency.

### Code Example

```rust
use llama_burn::inference::{concurrent_streaming_handler, GenerateRequest};
use burn_central_runtime::inference::InferenceBuilder;

// Build inference with concurrent handler
let inference = InferenceBuilder::new()
    .with_model(llama)
    .build(concurrent_streaming_handler);

// Spawn multiple requests - they will interleave
let job1 = inference
    .infer(GenerateRequest::new("Explain AI"))
    .with_devices([device.clone()])
    .spawn();

let job2 = inference
    .infer(GenerateRequest::new("What is ML?"))
    .with_devices([device.clone()])
    .spawn();

// Requests interleave: R1(tok1), R2(tok1), R1(tok2), R2(tok2), ...
for token in job1 {
    print!("{}", token?.token);
}
```

### How It Works

```
Time →
Request 1: [tok1]----[tok2]----[tok3]----[tok4]----...
Request 2: ----[tok1]----[tok2]----[tok3]----[tok4]...
Request 3: --------[tok1]----[tok2]----[tok3]----...
           ↑      ↑      ↑      ↑      ↑
           Model is released between each token
```

### When to Use

✅ **Use Concurrent When:**
- Multiple concurrent users (5-20)
- Balanced workload (mix of short and long requests)
- Low latency is important
- Good GPU utilization desired
- Interactive applications

❌ **Avoid Concurrent When:**
- Very high load (100+ concurrent requests)
- Only processing batch jobs
- Maximum throughput is the only goal

### Performance Characteristics

- **Throughput**: 2-4x sequential
- **Latency**: Low (slightly higher than sequential)
- **GPU Utilization**: Medium (40-60%)
- **Memory Usage**: Low-Medium
- **Implementation Complexity**: Medium
- **Fairness**: Good (all requests make progress)

### Key Features

- ✅ Isolated KV cache per request (no interference)
- ✅ Built-in cancellation support
- ✅ Streaming token generation
- ✅ Fair scheduling (round-robin)
- ✅ Good latency characteristics

## Strategy 3: Batched (Parallel Processing)

### Description

Processes multiple prompts simultaneously in a single forward pass. All sequences in a batch move through the model together, maximizing GPU parallelism and throughput.

### Code Example

```rust
use llama_burn::inference::generate_batch;

let prompts = vec![
    "What is the capital of France?",
    "Explain quantum computing.",
    "What are the three laws of robotics?",
];

// Process all prompts in parallel
let results = generate_batch(
    &mut llama,
    &prompts,
    50,   // max_tokens
    0.7,  // temperature
    0.9,  // top_p
    42,   // seed
)?;

// All prompts are processed together
for (prompt, tokens) in prompts.iter().zip(results.iter()) {
    println!("{}: {}", prompt, tokens.join(""));
}
```

### How It Works

```
Time →
Request 1: [--------Batch Forward Pass--------]
Request 2: [--------Batch Forward Pass--------]
Request 3: [--------Batch Forward Pass--------]
Request 4: [--------Batch Forward Pass--------]
           ↑                                  ↑
           All requests processed together
           Maximum GPU parallelism
```

### When to Use

✅ **Use Batched When:**
- High request volume (50+ concurrent)
- Maximum throughput is critical
- Processing batch jobs/queues
- GPU utilization needs to be maximized
- Slight latency increase is acceptable
- Requests can be grouped effectively

❌ **Avoid Batched When:**
- Very low request volume (<5 concurrent)
- Latency is critical (real-time chat)
- Highly variable request arrival
- Limited GPU memory

### Performance Characteristics

- **Throughput**: **5-7x sequential** 🚀
- **Latency**: Medium (higher first-token latency)
- **GPU Utilization**: High (70-95%)
- **Memory Usage**: Higher (scales with batch size)
- **Implementation Complexity**: Higher
- **Fairness**: Medium (batch-level, not token-level)

### Key Features

- ✅ Maximum GPU utilization
- ✅ Highest throughput per watt
- ✅ Automatic padding for variable lengths
- ✅ Shared KV cache (memory efficient)
- ✅ Scales well with batch size

### Batch Size Guidelines

| GPU VRAM | Recommended Batch Size | Expected Throughput |
|----------|------------------------|---------------------|
| 8GB | 2-4 | ~50-100 tokens/s |
| 16GB | 4-8 | ~100-200 tokens/s |
| 24GB | 8-16 | ~200-400 tokens/s |
| 40GB+ | 16-32 | ~400-800 tokens/s |

## Decision Tree

```
Start: Do you have multiple concurrent requests?
│
├─ NO → Use Sequential
│       Simple, efficient for single requests
│
└─ YES → How many concurrent requests?
         │
         ├─ 1-10 requests → Use Concurrent
         │                  Good balance, low latency
         │
         └─ 10+ requests → What's your priority?
                          │
                          ├─ Low Latency → Use Concurrent
                          │                Token-level fairness
                          │
                          └─ High Throughput → Use Batched
                                               Maximum performance
```

## Detailed Comparison

### Memory Usage

```
Sequential:  ████░░░░░░░░░░░░ (1 sequence at a time)
Concurrent:  ████████░░░░░░░░ (multiple isolated caches)
Batched:     ████████████████ (shared cache, batch dimension)
```

**Memory formula:**
- Sequential: `base_model + cache(1 sequence)`
- Concurrent: `base_model + cache(1 sequence) × N_active`
- Batched: `base_model + cache(batch_size sequences)`

### GPU Utilization

```
Sequential:  ▓░░░░░░░░░ 10-30%  (model mostly idle)
Concurrent:  ▓▓▓▓▓░░░░░ 40-60%  (interleaving overhead)
Batched:     ▓▓▓▓▓▓▓▓▓░ 70-95%  (maximum parallelism)
```

### Latency Breakdown

**Time to First Token (TTFT):**
- Sequential: ~40ms (fastest)
- Concurrent: ~50ms (+25%)
- Batched: ~100ms (+150%, must wait for batch)

**Time per Token (TPT):**
- Sequential: ~40ms/token
- Concurrent: ~45ms/token (slight overhead)
- Batched: ~20ms/token (amortized across batch)

**Total Time (50 tokens, 10 requests):**
- Sequential: 10 × (40ms + 50×40ms) = ~20s
- Concurrent: ~5-8s (interleaved)
- Batched: ~3-4s (parallel)

## Hybrid Strategies

For maximum flexibility, combine strategies:

### Strategy: Dynamic Batching

Collect requests dynamically and batch when threshold is reached:

```rust
let mut pending_requests = Vec::new();
const BATCH_SIZE: usize = 8;
const MAX_WAIT_MS: u64 = 100;

loop {
    // Collect requests up to batch size or timeout
    while pending_requests.len() < BATCH_SIZE {
        if let Some(req) = try_receive_request(MAX_WAIT_MS) {
            pending_requests.push(req);
        } else {
            break; // Timeout - process what we have
        }
    }
    
    if !pending_requests.is_empty() {
        // Process batch
        let prompts: Vec<&str> = pending_requests
            .iter()
            .map(|r| r.prompt.as_str())
            .collect();
        
        let results = generate_batch(&mut llama, &prompts, ...)?;
        
        // Send responses
        for (req, result) in pending_requests.drain(..).zip(results) {
            send_response(req, result);
        }
    }
}
```

### Strategy: Tiered Processing

Use different strategies based on request characteristics:

```rust
// High-priority, low-latency requests
if request.priority == Priority::High {
    use_concurrent_handler(request);
}
// Batch processing for background tasks
else if pending_batch.len() >= BATCH_THRESHOLD {
    use_batched_handler(pending_batch);
}
// Default to concurrent for balance
else {
    use_concurrent_handler(request);
}
```

## Real-World Scenarios

### Scenario 1: Chat Application (1-5 users)

**Recommended: Sequential or Concurrent**

```rust
// Simple approach - concurrent for 2+ users
let inference = InferenceBuilder::new()
    .with_model(llama)
    .build(concurrent_streaming_handler);

// Handles 1-5 users with good responsiveness
```

**Why:**
- Low user count doesn't benefit much from batching
- Concurrent provides fair token-by-token streaming
- Lower complexity

### Scenario 2: API Service (20-50 concurrent users)

**Recommended: Concurrent**

```rust
let inference = InferenceBuilder::new()
    .with_model(llama)
    .build(concurrent_streaming_handler);

// Good balance of throughput and latency
// Handles bursts well
```

**Why:**
- Good throughput without complexity
- Low latency for interactive users
- Built-in streaming support

### Scenario 3: Batch Processing (100+ requests)

**Recommended: Batched**

```rust
// Process large queue in batches
const BATCH_SIZE: usize = 16;

for batch_prompts in request_queue.chunks(BATCH_SIZE) {
    let results = generate_batch(
        &mut llama,
        batch_prompts,
        max_tokens,
        temperature,
        top_p,
        seed,
    )?;
    
    process_results(results);
}
```

**Why:**
- Maximum throughput
- Latency less critical for batch jobs
- Efficient GPU utilization

### Scenario 4: Mixed Workload (Varied request types)

**Recommended: Hybrid (Concurrent + Batched)**

```rust
// Real-time requests → Concurrent
// Batch jobs → Batched

if request.is_interactive() {
    concurrent_inference.infer(request).spawn();
} else {
    batch_queue.push(request);
    
    if batch_queue.len() >= BATCH_SIZE {
        let results = generate_batch(&mut llama, &batch_queue, ...)?;
        // Process batch results
    }
}
```

## Performance Tuning Tips

### For Sequential
1. Optimize single-request performance
2. Reduce max_tokens if possible
3. Use faster sampling (argmax vs top-p)
4. Consider model quantization

### For Concurrent
1. Monitor token generation fairness
2. Limit concurrent requests (10-20 optimal)
3. Use cancellation for abandoned requests
4. Profile model.submit() overhead

### For Batched
1. **Maximize batch size** within memory limits
2. **Group similar-length requests** together
3. **Tune max_seq_len** to reduce padding
4. **Monitor GPU utilization** (target 80-95%)
5. **Profile memory usage** to avoid OOM

## Monitoring and Metrics

### Key Metrics to Track

```rust
// Throughput
let tokens_per_second = total_tokens / elapsed_time;

// Latency
let time_to_first_token = first_token_time - request_time;
let time_per_token = total_time / num_tokens;

// Utilization
let gpu_utilization = gpu_active_time / total_time;

// Efficiency
let tokens_per_joule = total_tokens / energy_consumed;
```

### Recommended Monitoring

| Metric | Sequential | Concurrent | Batched |
|--------|-----------|-----------|---------|
| **Throughput** | 20-30 tok/s | 50-100 tok/s | 150-400 tok/s |
| **TTFT** | <50ms | <100ms | <200ms |
| **GPU Util** | 10-30% | 40-60% | 70-95% |
| **Memory** | Low | Medium | High |

## Resources

### Documentation
- [CONCURRENT_INFERENCE.md](CONCURRENT_INFERENCE.md) - Detailed concurrent strategy guide
- [BATCHED_INFERENCE.md](BATCHED_INFERENCE.md) - Detailed batched strategy guide
- [BATCHED_QUICKSTART.md](BATCHED_QUICKSTART.md) - Quick start for batched inference

### Examples
- [chat.rs](examples/chat.rs) - Sequential generation
- [concurrent_server.rs](examples/concurrent_server.rs) - Concurrent interleaving
- [batched_server.rs](examples/batched_server.rs) - Batched parallel processing
- [simple_batch.rs](examples/simple_batch.rs) - Simple batching example

## Summary

Choose your strategy based on your requirements:

| Priority | Strategy | Expected Performance |
|----------|----------|---------------------|
| **Simplicity** | Sequential | Baseline |
| **Balance** | Concurrent | 2-4x throughput |
| **Maximum Throughput** | Batched | **5-7x throughput** |

**Quick recommendations:**
- **1-5 users**: Sequential or Concurrent
- **5-20 users**: Concurrent
- **20-50 users**: Concurrent or Small Batches
- **50+ users**: Batched with dynamic batching
- **Batch jobs**: Batched (large batch size)

The best strategy depends on your specific workload, hardware, and latency requirements. Start simple, measure performance, and optimize based on your actual usage patterns.