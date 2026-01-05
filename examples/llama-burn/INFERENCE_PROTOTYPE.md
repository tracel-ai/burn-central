# Llama Inference API

This document describes the built-in inference API for llama-burn, powered by **burn-central-runtime**.

## Architecture Philosophy

**The inference logic lives in the library, not the application.**

Applications simply import `llama_burn::inference` and use the provided handlers and types. This makes it easy to:
- Build inference servers
- Create CLI tools
- Integrate with web services
- Add custom processing layers

All without reimplementing the core inference logic.

## Quick Start

```rust
use burn::backend::{Cuda, cuda::CudaDevice};
use burn::tensor::f16;
use burn_central_runtime::inference::InferenceBuilder;
use llama_burn::inference::{streaming_handler, GenerateRequest};
use llama_burn::llama::LlamaConfig;

type Backend = Cuda<f16, i32>;

fn main() {
    let device = CudaDevice::default();
    
    // Load model
    let llama = LlamaConfig::llama3_2_1b_pretrained::<Backend>(256, &device)
        .expect("Failed to load model");
    
    // Build inference instance using library's handler
    let inference = InferenceBuilder::<Backend>::new()
        .with_model(llama)
        .build(streaming_handler);
    
    // Create request
    let request = GenerateRequest::new("The future of AI is")
        .with_max_tokens(50)
        .with_temperature(0.7);
    
    // Spawn streaming job
    let job = inference.infer(request).with_devices([device]).spawn();
    
    // Stream tokens as they're generated
    for token in job.stream.iter() {
        print!("{}", token.token);
    }
}
```

## Library API

### `llama_burn::inference` Module

The library provides everything needed for inference:

#### Types

**`GenerateRequest`** - Input for text generation
```rust
pub struct GenerateRequest {
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub seed: u64,
}

// Builder pattern
let request = GenerateRequest::new("Once upon a time")
    .with_max_tokens(100)
    .with_temperature(0.8)
    .with_top_p(0.9)
    .with_seed(42);
```

**`TokenOutput`** - Output for each generated token
```rust
pub struct TokenOutput {
    pub token: String,      // Decoded token text
    pub token_id: u32,      // Token ID
    pub index: usize,       // Position in sequence
}
```

#### Handler

**`streaming_handler`** - Ready-to-use inference handler
```rust
pub fn streaming_handler<B: Backend, T: Tokenizer + 'static>(
    In(request): In<GenerateRequest>,
    model: ModelAccessor<Llama<B, T>>,
    cancel: CancelToken,
    output: OutStream<TokenOutput>,
) -> Result<(), String>
```

This handler is designed to work with `InferenceBuilder::build()` and provides:
- Token-by-token streaming via `OutStream`
- Cancellation support via `CancelToken`
- Thread-safe model access via `ModelAccessor`
- Automatic sampler configuration from request

## Usage Patterns

### 1. Streaming Generation

```rust
let job = inference.infer(request).with_devices([device]).spawn();

for token in job.stream.iter() {
    print!("{}", token.token);
    std::io::Write::flush(&mut std::io::stdout()).unwrap();
}

match job.join() {
    Ok(_) => println!("Generation complete"),
    Err(e) => eprintln!("Error: {:?}", e),
}
```

### 2. Synchronous Generation (Collect All)

```rust
let outputs = inference
    .infer(request)
    .with_devices([device])
    .run()
    .expect("Generation failed");

for token in outputs {
    print!("{}", token.token);
}
```

### 3. Cancellation

```rust
let job = inference.infer(request).with_devices([device]).spawn();

for (i, token) in job.stream.iter().enumerate() {
    print!("{}", token.token);
    if i >= 10 {
        job.cancel();
        break;
    }
}
```

### 4. Multiple Concurrent Requests

```rust
let job1 = inference.infer(request1).with_devices([device.clone()]).spawn();
let job2 = inference.infer(request2).with_devices([device.clone()]).spawn();

// Process streams concurrently
let h1 = std::thread::spawn(move || {
    job1.stream.iter().for_each(|t| print!("{}", t.token));
});

let h2 = std::thread::spawn(move || {
    job2.stream.iter().for_each(|t| print!("{}", t.token));
});

h1.join().unwrap();
h2.join().unwrap();
```

## How It Works

### 1. Model Thread

The inference API uses a dedicated thread for the model:

```
Application Thread              Model Thread
     |                              |
     | -- submit(closure) ------>   |
     |                              | execute on model
     |                              | (mutable access serialized)
     | <---- return result -------  |
     |                              |
```

This ensures:
- Thread-safe mutable access to model state (cache)
- No data races
- Efficient serialized execution

### 2. Streaming Pipeline

```
Request --> Handler --> generate_streaming() --> Tokens
                            |
                            v
                        OutStream --> Channel --> Application
                            |
                            v
                        CancelToken (check each iteration)
```

Each token is:
1. Generated by the model
2. Decoded immediately
3. Emitted through `OutStream`
4. Sent via channel to application
5. Cancellation checked before next iteration

### 3. Parameter Extraction

The handler uses burn-central-runtime's parameter extraction:

```rust
fn streaming_handler(
    In(request): In<GenerateRequest>,           // ← Extracted from job input
    model: ModelAccessor<LlamaInference<B, T>>, // ← Extracted from InferenceContext
    cancel: CancelToken,                         // ← Extracted from InferenceContext
    output: OutStream<TokenOutput>,              // ← Extracted from InferenceContext
) -> Result<(), String>
```

These are automatically extracted from the `InferenceContext` by the inference runtime.

## Core Methods

The `Llama<B, T>` type provides the core generation methods:

**`generate_streaming`** - Stream tokens one at a time
```rust
impl<B: Backend, T: Tokenizer> Llama<B, T> {
    pub fn generate_streaming<F>(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
        sampler: &mut Sampler,
        on_token: F,
    ) -> Result<usize, String>
    where
        F: FnMut(u32, &str, usize) -> Result<bool, String>
}
```

This method is used internally by the `streaming_handler` but can also be used directly for custom inference patterns.

## Example: Building an HTTP Server

Since inference logic lives in the library, building a server is straightforward:

```rust
use axum::{Json, Router, routing::post};
use axum::response::sse::{Event, Sse};
use futures::stream::Stream;

async fn generate_endpoint(
    Json(request): Json<GenerateRequest>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let job = INFERENCE.infer(request).with_devices([DEVICE.clone()]).spawn();
    
    let stream = async_stream::stream! {
        for token in job.stream.iter() {
            yield Ok(Event::default().data(token.token));
        }
    };
    
    Sse::new(stream)
}

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/generate", post(generate_endpoint));
    
    // ... serve
}
```

## Next Steps

Potential enhancements to the library API:

1. **Batch inference handler** - Process multiple prompts in parallel
2. **State support** - Add `State<S>` for request-scoped state
3. **Progress callbacks** - Emit progress updates (tokens/sec, ETA)
4. **Model variants** - Support different model sizes/configs
5. **Custom samplers** - Make sampler configurable per request
6. **Warmup API** - Pre-warm model/cache before first request

## Running the Example

```bash
# Basic streaming
cargo run --example inference_server --features llama3,cuda

# With TinyLlama
cargo run --example inference_server --features tiny,cuda

# The example demonstrates:
# - Streaming generation with real-time output
# - Cancellation after N tokens
# - Synchronous collection of all outputs
```

## Design Benefits

✅ **Library owns the logic** - Applications don't reimplement inference  
✅ **Clean separation** - Model code vs. application code  
✅ **Easy testing** - Library can unit test handlers  
✅ **Composable** - Build servers, CLIs, notebooks using same API  
✅ **Maintainable** - Bug fixes in one place  
✅ **Extensible** - Add new handlers without changing applications  

This architecture follows the principle: **libraries provide capabilities, applications provide interfaces**.