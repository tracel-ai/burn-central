# Batched Inference Quickstart Guide

This guide will help you get started with batched inference in llama-burn, which processes multiple prompts simultaneously for maximum throughput.

## What is Batched Inference?

Batched inference processes multiple prompts together in a single forward pass through the model. Instead of processing requests one at a time or interleaving them token-by-token, batching runs them all in parallel, maximizing GPU utilization.

### Performance at a Glance

Processing 10 requests with 50 tokens each:

- **Sequential**: ~20 seconds (1x baseline)
- **Concurrent (token-level)**: ~5-8 seconds (2.5-4x faster)
- **Batched**: ~3-4 seconds (**5-7x faster**)

## Quick Start

### Basic Usage

```rust
use llama_burn::{
    inference::generate_batch,
    llama::LlamaConfig,
};

// Load your model
let mut llama = LlamaConfig::llama3_2_1b_pretrained(256, &device)?;

// Define your prompts
let prompts = vec![
    "What is the capital of France?",
    "Explain quantum computing.",
    "What are the three laws of robotics?",
];

// Generate in batch
let results = generate_batch(
    &mut llama,
    &prompts,
    50,   // max_tokens
    0.7,  // temperature
    0.9,  // top_p
    42,   // seed
)?;

// Process results
for (prompt, tokens) in prompts.iter().zip(results.iter()) {
    println!("Prompt: {}", prompt);
    println!("Response: {}", tokens.join(""));
}
```

### Running the Examples

#### Full-Featured Demo (requires pretrained weights)

```bash
cargo run --example batched_server --features llama3,cuda
```

This comprehensive example demonstrates:
- Basic batched inference
- Throughput comparison
- Variable-length sequences
- Large batch processing
- Streaming output display

#### Simple Demo (no pretrained weights needed)

```bash
cargo run --example simple_batch --features llama3,cuda
```

A simpler example showing the core API without requiring model downloads.

## How It Works

### 1. Create Batched State

The batched generation state manages multiple sequences at once:

```rust
let mut state = llama.create_batched_generation_state(&prompts, max_tokens);
```

This creates:
- Token buffers for all sequences: `[batch_size, max_total_len]`
- Position tracking for each sequence
- Active mask to track which sequences are still generating
- Shared KV cache with batch dimension

### 2. Generate Tokens for All Sequences

A single forward pass processes the entire batch:

```rust
let batch_results = llama.generate_batched_single_token(
    &mut state,
    temperature,
    &mut samplers,
)?;
```

This returns `Vec<(token_id, token_text, is_complete)>` with one entry per sequence.

### 3. Repeat Until Complete

Continue until all sequences finish (reach max_tokens or stop token):

```rust
while state.active.iter().any(|&a| a) {
    let batch_results = llama.generate_batched_single_token(...)?;
    // Process tokens for each sequence
}
```

## API Reference

### `generate_batch`

Helper function for simple batched generation:

```rust
pub fn generate_batch<B: Backend, T: Tokenizer>(
    llama: &mut Llama<B, T>,
    prompts: &[&str],
    max_tokens: usize,
    temperature: f64,
    top_p: f64,
    seed: u64,
) -> Result<Vec<Vec<String>>, String>
```

**Parameters:**
- `llama`: Mutable reference to the Llama model
- `prompts`: Slice of prompts to process in batch
- `max_tokens`: Maximum tokens to generate per sequence
- `temperature`: Sampling temperature (0.0 = greedy)
- `top_p`: Top-p nucleus sampling parameter
- `seed`: Random seed (incremented per sequence for variety)

**Returns:**
- `Vec<Vec<String>>`: Generated tokens for each sequence

### `create_batched_generation_state`

Create state for manual token-by-token generation:

```rust
let mut state = llama.create_batched_generation_state(&prompts, max_tokens);
```

### `generate_batched_single_token`

Generate one token for all active sequences:

```rust
let results = llama.generate_batched_single_token(
    &mut state,
    temperature,
    &mut samplers,
)?;
```

## Best Practices

### 1. Choose Appropriate Batch Size

**Memory considerations:**
```
memory ≈ batch_size × max_seq_len × model_size
```

**Recommendations by GPU:**
- 8GB VRAM: batch_size = 2-4
- 16GB VRAM: batch_size = 4-8
- 24GB+ VRAM: batch_size = 8-16

### 2. Group Similar-Length Requests

For best efficiency, batch requests with similar lengths together:

```rust
// Group short and long requests separately
let short_prompts = vec!["Hi", "Hello", "Hey"];
let long_prompts = vec![
    "Explain the theory of relativity...",
    "Describe the process of photosynthesis...",
];

// Process each group separately
let short_results = generate_batch(&mut llama, &short_prompts, 10, ...)?;
let long_results = generate_batch(&mut llama, &long_prompts, 100, ...)?;
```

### 3. Monitor GPU Utilization

Use `nvidia-smi` to check GPU usage:

```bash
watch -n 1 nvidia-smi
```

Target 80-95% GPU utilization for optimal throughput.

### 4. Use Larger Batches for Throughput, Smaller for Latency

**High throughput scenario:**
```rust
// Process 16 requests at once - maximum throughput
let batch_size = 16;
```

**Low latency scenario:**
```rust
// Process 2-4 requests at once - faster first token
let batch_size = 4;
```

## Common Patterns

### Pattern 1: Batch Processing Queue

```rust
let mut request_queue: Vec<String> = get_requests();
let batch_size = 8;

while !request_queue.is_empty() {
    // Take up to batch_size requests
    let batch: Vec<&str> = request_queue
        .iter()
        .take(batch_size)
        .map(|s| s.as_str())
        .collect();
    
    // Process batch
    let results = generate_batch(&mut llama, &batch, 50, 0.7, 0.9, 42)?;
    
    // Handle results
    for (prompt, tokens) in batch.iter().zip(results.iter()) {
        send_response(prompt, tokens);
    }
    
    // Remove processed requests
    request_queue.drain(..batch.len());
}
```

### Pattern 2: Per-Sequence Parameters

For different parameters per sequence, use the lower-level API:

```rust
let mut state = llama.create_batched_generation_state(&prompts, max_tokens);

// Create different samplers for each sequence
let mut samplers = vec![
    Sampler::TopP(TopP::new(0.9, 42)),    // Creative
    Sampler::TopP(TopP::new(0.5, 43)),    // Balanced
    Sampler::Argmax,                       // Deterministic
];

while state.active.iter().any(|&a| a) {
    let results = llama.generate_batched_single_token(
        &mut state,
        0.7,  // temperature (can vary this per iteration if needed)
        &mut samplers,
    )?;
    
    // Process results...
}
```

### Pattern 3: Early Stopping

Stop specific sequences early based on content:

```rust
while state.active.iter().any(|&a| a) {
    let results = llama.generate_batched_single_token(&mut state, temp, &mut samplers)?;
    
    for (i, (_, token_text, is_complete)) in results.iter().enumerate() {
        // Check custom stopping condition
        if token_text.contains("[END]") {
            state.active[i] = false;  // Mark this sequence as complete
        }
    }
}
```

## Performance Tuning

### Measure Your Throughput

```rust
use std::time::Instant;

let start = Instant::now();
let results = generate_batch(&mut llama, &prompts, max_tokens, ...)?;
let elapsed = start.elapsed();

let total_tokens: usize = results.iter().map(|r| r.len()).sum();
let throughput = total_tokens as f32 / elapsed.as_secs_f32();

println!("Throughput: {:.2} tokens/s", throughput);
println!("Per-sequence: {:.2} tokens/s", throughput / prompts.len() as f32);
```

### Optimize for Your Workload

| Workload Type | Recommended Settings |
|---------------|---------------------|
| Many short requests | Larger batch (8-16), short max_tokens |
| Few long requests | Smaller batch (2-4), long max_tokens |
| Mixed lengths | Group by length, medium batch (4-8) |
| Maximum throughput | Largest batch that fits in memory |
| Minimum latency | Smallest effective batch (2-4) |

## Troubleshooting

### Out of Memory (OOM)

**Symptoms:** CUDA out of memory errors

**Solutions:**
1. Reduce batch size
2. Reduce max_seq_len
3. Use smaller model
4. Enable gradient checkpointing (if available)

### Low GPU Utilization

**Symptoms:** GPU usage < 70%

**Solutions:**
1. Increase batch size
2. Ensure prompts have sufficient length
3. Check for CPU bottlenecks (tokenization)
4. Profile to find bottlenecks

### Slower Than Expected

**Possible causes:**
1. Batch size too small - increase it
2. Many short sequences - consider padding/grouping
3. CPU preprocessing overhead - batch tokenization
4. Memory fragmentation - restart process

## Next Steps

- Read [BATCHED_INFERENCE.md](BATCHED_INFERENCE.md) for detailed implementation info
- Check [CONCURRENT_INFERENCE.md](CONCURRENT_INFERENCE.md) for comparison with concurrent approach
- Explore [examples/batched_server.rs](examples/batched_server.rs) for comprehensive demos
- See [examples/simple_batch.rs](examples/simple_batch.rs) for minimal working example

## Summary

Batched inference in llama-burn provides:

✅ **5-7x throughput improvement** over sequential processing  
✅ **Simple API** with `generate_batch` helper function  
✅ **Flexible control** with lower-level batch generation API  
✅ **Automatic padding** for variable-length sequences  
✅ **Memory efficient** with shared KV cache  

Perfect for high-throughput production scenarios where multiple requests need to be processed quickly.