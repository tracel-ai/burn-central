# Llama Burn

<img src="./assets/llama-burn.jpeg" alt="An image of a llama surrounded by fiery colors and a gust of fire" width="500px"/>

The popular Llama LLM is here!

This repository contains the
[Llama 3.2, Llama 3.1, Llama 3](https://github.com/meta-llama/llama-models/), and
[TinyLlama](https://github.com/jzhang38/TinyLlama) implementations with their corresponding
tokenizers. You can find the [Burn](https://github.com/tracel-ai/burn) implementation for the Llama
variants in [src/llama.rs](src/llama.rs).

## Usage

### `Cargo.toml`

Add this to your `Cargo.toml`:

```toml
[dependencies]
llama-burn = { git = "https://github.com/tracel-ai/models", package = "llama-burn", default-features = false }
```

If you want to use Llama 3 or TinyLlama (including pre-trained weights if default features are
active), enable the corresponding feature flag.

> **Important:** these features require `std`.

#### Llama 3

```toml
[dependencies]
llama-burn = { git = "https://github.com/tracel-ai/models", package = "llama-burn", features = ["llama3"] }
```

#### TinyLlama

```toml
[dependencies]
llama-burn = { git = "https://github.com/tracel-ai/models", package = "llama-burn", features = ["tiny"] }
```

### Example Usage

The [chat completion example](examples/chat.rs) initializes a Llama model from the provided weights
file and generates a sequence of text based on the input prompt. The instruction-tuned model is
loaded for dialogue applications, so the prompt is automatically formatted for chat completion.

The example can be executed on the `tch` backend (CUDA or CPU), `cuda` or `vulkan` (wgpu).

| Argument        | Description                                                                                                    |
| :-------------- | :------------------------------------------------------------------------------------------------------------- |
| `-p`            | The prompt or question to pass to the LLM (default: `"How many helicopters can a human eat in one sitting?"`). |
| `-n`            | The number of new tokens to generate (default: `50`).                                                          |
| `--top-p`       | Top-p probability threshold (default: `0.9`).                                                                  |
| `--temperature` | Temperature value for controlling randomness in sampling. (default: `0.6`).                                    |
| `--max-seq-len` | Maximum sequence length for input text. (default: `128`).                                                      |
| `--seed`        | The seed to use when generating random samples.. (default: `42`).                                              |

Any of the commands below can be used by appending any of the listed arguments by appending
`[-- <arguments>]`. For example, you can provided your own prompt/question
`-- -p "How many llamas does it take to change a lightbulb?"`.

#### Llama 3

Using the `tch` backend with CUDA:

```sh
export TORCH_CUDA_VERSION=cu128
cargo run --release --features llama3,tch-gpu --example chat
```

Using the `tch` backend with CPU:

```sh
cargo run --release --features llama3,tch-cpu --example chat
```

Using the `vulkan` backend:

```sh
cargo run --release --features llama3,vulkan --example chat
```

Using the `cuda` backend:

```sh
cargo run --release --features llama3,cuda --example chat
```

**Built with Meta Llama 3.** This example uses the
[Meta-Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) (default),
[Meta-Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct),
[Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) and
[Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
instruction-tuned models. Note that the [base pre-trained Llama-3 model](./src/pretrained.rs#L77) is
also available if you wish to use it in your application.

#### TinyLlama

Using the `tch` backend with CUDA:

```sh
export TORCH_CUDA_VERSION=cu128
cargo run --release --features tiny,tch-gpu --example chat
```

Using the `tch` backend with CPU:

```sh
cargo run --release --features tiny,tch-cpu --example chat
```

Using the `vulkan` backend:

```sh
cargo run --release --features tiny,vulkan --example chat
```

Using the `cuda` backend:

```sh
cargo run --release --features tiny,cuda --example chat
```

This example uses the
[TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
instruction-tuned model based on the Llama2 architecture and tokenizer.

## Inference Strategies

Llama-burn provides three inference strategies optimized for different use cases:

### 1. Sequential (Blocking) - Simplest Approach

The basic approach for single requests. Process one request completely before starting the next.

```rust
use llama_burn::{llama::LlamaConfig, sampling::{Sampler, TopP}};

let mut llama = LlamaConfig::llama3_2_1b_pretrained(256, &device)?;
let mut sampler = Sampler::TopP(TopP::new(0.9, 42));

let result = llama.generate("What is AI?", 50, 0.7, &mut sampler);
println!("{}", result.text);
```

**Example:** [chat.rs](examples/chat.rs)

### 2. Concurrent (Token-level Interleaving) - Balanced Performance

Generates one token at a time per request, releasing the model between tokens. Multiple requests interleave their token generation for better concurrency.

```sh
cargo run --example concurrent_server --features llama3,cuda
```

**Key improvement:** The concurrent handler generates one token at a time and releases the model between tokens, allowing other requests to be processed. This enables much better concurrency compared to the blocking approach.

**Examples demonstrated:**
- Single request baseline
- Multiple concurrent requests interleaving
- Cancellation with concurrent requests
- Stress test with 10+ concurrent requests

**Example:** [concurrent_server.rs](examples/concurrent_server.rs)  
**Documentation:** [CONCURRENT_INFERENCE.md](CONCURRENT_INFERENCE.md)

### 3. Batched (Parallel Processing) - Maximum Throughput

Processes multiple prompts simultaneously in a single forward pass, maximizing GPU utilization.

```rust
use llama_burn::inference::generate_batch;

let prompts = vec![
    "What is the capital of France?",
    "Explain quantum computing.",
    "What are the three laws of robotics?",
];

let results = generate_batch(
    &mut llama,
    &prompts,
    50,   // max_tokens
    0.7,  // temperature
    0.9,  // top_p
    42,   // seed
)?;

for (prompt, tokens) in prompts.iter().zip(results.iter()) {
    println!("{}: {}", prompt, tokens.join(""));
}
```

**Run the examples:**
```sh
# Comprehensive demo with pretrained weights
cargo run --example batched_server --features llama3,cuda

# Simple demo without pretrained weights
cargo run --example simple_batch --features llama3,cuda
```

**Examples demonstrated:**
- Basic batched inference with multiple prompts
- Throughput comparison vs. sequential processing
- Variable-length sequence handling
- Large batch stress testing (8+ concurrent requests)
- Streaming batched output

**Examples:** [batched_server.rs](examples/batched_server.rs), [simple_batch.rs](examples/simple_batch.rs)  
**Documentation:** [BATCHED_INFERENCE.md](BATCHED_INFERENCE.md), [BATCHED_QUICKSTART.md](BATCHED_QUICKSTART.md)

## Concurrent Inference Architecture

### The Problem

The original inference implementation blocks the model for the entire generation:

```rust
// Blocks model for ALL tokens (e.g., 2+ seconds for 50 tokens)
model.submit(|m| m.generate_streaming(prompt, 50, ...))
```

This means concurrent requests queue up and wait sequentially.

### The Solution: Iterative Token Generation

The new `concurrent_streaming_handler` uses iterative token generation:

```rust
// Create state outside model lock
let mut state = model.submit(|m| m.create_generation_state(prompt, max_tokens));

loop {
    // Generate ONE token (model locked for ~20-50ms)
    let (token, text, done) = model.submit(|m| {
        m.generate_single_token(&mut state, temp, &mut sampler)
    })?;
    
    output.emit(token)?;
    if done { break; }
    
    // Model is FREE here - other requests can run!
}
```

### Performance Comparison

| Scenario | Blocking Handler | Concurrent Handler | Improvement |
|----------|-----------------|-------------------|-------------|
| 1 request, 50 tokens | 2.0s | 2.0s | ~same |
| 3 requests, 50 tokens each | 6.0s (sequential) | ~2.5s (interleaved) | **2.4x faster** |
| 10 requests, 20 tokens each | 20s (sequential) | ~5s (interleaved) | **4x faster** |

**Throughput improvement:** With concurrent requests, the GPU is better utilized because requests interleave instead of waiting idle.

### Usage

```rust
use llama_burn::inference::{concurrent_streaming_handler, GenerateRequest};
use burn_central::runtime::inference::InferenceBuilder;

// Build inference with concurrent handler
let inference = InferenceBuilder::<Backend>::new()
    .with_model(llama)
    .build(concurrent_streaming_handler);

// Spawn concurrent requests
let job1 = inference.infer(request1).with_devices([device.clone()]).spawn();
let job2 = inference.infer(request2).with_devices([device.clone()]).spawn();

// Requests will interleave: R1(tok1), R2(tok1), R1(tok2), R2(tok2), ...
```

## Performance Comparison

Processing 10 requests with 50 tokens each:

| Strategy | Time | Throughput | GPU Utilization | Best For |
|----------|------|------------|-----------------|----------|
| **Sequential** | ~20s | 1x (baseline) | 10-30% | Single user, simple apps |
| **Concurrent** | ~5-8s | **2-4x** | 40-60% | Multi-user, balanced workload |
| **Batched** | ~3-4s | **5-7x** 🚀 | 70-95% | High-load production, max throughput |

### When to Use Each Strategy

**Sequential (Blocking):**
- ✅ Single user application
- ✅ Development/testing
- ✅ Minimum complexity needed
- ❌ Multiple concurrent users

**Concurrent (Token-level Interleaving):**
- ✅ 5-20 concurrent users
- ✅ Interactive applications needing low latency
- ✅ Mixed request lengths
- ✅ Fair token-by-token streaming
- ❌ Very high load (100+ requests)

**Batched (Parallel Processing):**
- ✅ 50+ concurrent requests
- ✅ Maximum throughput critical
- ✅ Batch processing jobs
- ✅ GPU utilization needs to be maximized
- ❌ Very low latency required
- ❌ Only 1-5 concurrent requests

## Documentation

### Inference Strategies
- [INFERENCE_STRATEGIES.md](INFERENCE_STRATEGIES.md) - **Comprehensive guide to all strategies**
- [CONCURRENT_INFERENCE.md](CONCURRENT_INFERENCE.md) - Concurrent token-level interleaving
- [BATCHED_INFERENCE.md](BATCHED_INFERENCE.md) - Batched parallel processing details
- [BATCHED_QUICKSTART.md](BATCHED_QUICKSTART.md) - Quick start guide for batched inference

### Examples

| Example | Strategy | Description | Run Command |
|---------|----------|-------------|-------------|
| [chat.rs](examples/chat.rs) | Sequential | Basic chat completion | `cargo run --example chat --features llama3,cuda` |
| [inference_server.rs](examples/inference_server.rs) | Sequential | Basic inference API | `cargo run --example inference_server --features llama3,cuda` |
| [concurrent_server.rs](examples/concurrent_server.rs) | Concurrent | Token-level interleaving | `cargo run --example concurrent_server --features llama3,cuda` |
| [batched_server.rs](examples/batched_server.rs) | Batched | Parallel batch processing | `cargo run --example batched_server --features llama3,cuda` |
| [simple_batch.rs](examples/simple_batch.rs) | Batched | Simple batching demo | `cargo run --example simple_batch --features llama3,cuda` |

## Quick Start: Batched Inference

For maximum throughput with multiple requests:

```rust
use llama_burn::inference::generate_batch;

// Load model
let mut llama = LlamaConfig::llama3_2_1b_pretrained(256, &device)?;

// Your prompts
let prompts = vec![
    "What is machine learning?",
    "Explain neural networks.",
    "What is deep learning?",
];

// Generate in batch (5-7x faster than sequential!)
let results = generate_batch(&mut llama, &prompts, 50, 0.7, 0.9, 42)?;

// Process results
for (prompt, tokens) in prompts.iter().zip(results.iter()) {
    println!("{}: {}", prompt, tokens.join(""));
}
```

See [BATCHED_QUICKSTART.md](BATCHED_QUICKSTART.md) for more details.

