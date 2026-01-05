# Llama Iroh Gossip - Distributed LLM Inference

A peer-to-peer distributed inference system for LLMs using [iroh-gossip](https://github.com/n0-computer/iroh) and burn-central-runtime.

## Overview

This example demonstrates how to build a **decentralized inference network** where:

- 🌐 Nodes communicate via gossip protocol (no central server!)
- 🔄 Inference requests are broadcast to the network
- ⚡ Available workers claim and process requests
- 📡 Generated tokens stream back in real-time via gossip
- 🔐 All messages are cryptographically signed and verified

## Architecture

```
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│   Client    │         │   Worker    │         │   Worker    │
│    Node     │◄───────►│    Node     │◄───────►│    Node     │
└─────────────┘         └─────────────┘         └─────────────┘
       │                       │                       │
       └───────────────────────┴───────────────────────┘
                        Gossip Network
                    (iroh-gossip protocol)

Flow:
1. Client broadcasts Request message
2. Workers with capacity broadcast Claim message
3. Chosen worker processes via burn-central-runtime
4. Worker streams Token messages as generation proceeds
5. Worker broadcasts Complete/Error message
```

## Key Components

### Protocol Layer (`protocol.rs`)
- **Message Types**: Request, Claim, Token, Complete, Error, Cancel
- **Cryptographic Signing**: All messages are signed with ed25519
- **Wire Format**: Postcard serialization for efficient gossip

### Node Layer (`node.rs`)
- **InferenceNode**: Manages gossip subscription and inference execution
- **LlamaModel**: Wraps llama-burn for streaming generation
- **Request Handling**: Claims requests, processes them, streams results

### Integration with burn-central-runtime
- Uses `InferenceBuilder` to create inference instances
- Streaming via `OutStream<T>` for token-by-token generation
- Cancellation support via `CancelToken`
- Thread-safe model access via `ModelAccessor`

## Building

```bash
# With CUDA backend and Llama 3.2
cargo build --release -p llama-iroh-gossip --features cuda,llama3

# With CUDA backend and TinyLlama
cargo build --release -p llama-iroh-gossip --features cuda,tiny
```

## Running

### Worker Node (processes requests)

```bash
# Start a worker node
cargo run --release -p llama-iroh-gossip --features cuda,llama3 -- \
  --name worker1 \
  worker
```

### Multiple Workers

```bash
# Terminal 1: First worker
cargo run --release -p llama-iroh-gossip --features cuda,llama3 -- \
  --name worker1 worker

# Terminal 2: Second worker (will auto-discover first worker)
cargo run --release -p llama-iroh-gossip --features cuda,llama3 -- \
  --name worker2 worker
```

### Hybrid Node (worker + can submit requests)

```bash
cargo run --release -p llama-iroh-gossip --features cuda,llama3 -- \
  --name hybrid1 \
  hybrid
```

## How It Works

### 1. Node Startup
```rust
// Spawn inference node with llama model
let node = InferenceNode::spawn(
    "my-node".to_string(),
    llama,
    "CUDA".to_string(),
    "Llama-3.2-1B".to_string(),
    None, // auto-generate keypair
).await?;
```

### 2. Join Gossip Network
```rust
// Subscribe to inference topic
let bootstrap = BTreeSet::new(); // empty for local discovery
node.join(bootstrap).await?;

// Node broadcasts Announce message
// Listens for Request messages
```

### 3. Request Processing
```rust
// When Request message received:
// 1. Check if we have capacity
// 2. Broadcast Claim message
// 3. Create inference job via burn-central-runtime
let job = inference.infer(request).with_devices([device]).spawn();

// 4. Stream tokens via gossip
for token in job.stream.iter() {
    broadcast(InferenceMessage::Token {
        request_id,
        token: token.token,
        token_id: token.token_id,
        index: token.index,
    });
}

// 5. Broadcast completion
broadcast(InferenceMessage::Complete {
    request_id,
    total_tokens,
    duration_secs,
});
```

### 4. Message Security
All messages are cryptographically signed:

```rust
// Signing (when sending)
let signed = MessageCodec::encode(&secret_key, message)?;
gossip_sender.broadcast(signed).await?;

// Verification (when receiving)
let verified = MessageCodec::decode(&message_bytes)?;
// verified.from: PublicKey
// verified.message: InferenceMessage
```

## Protocol Messages

```rust
enum InferenceMessage {
    // Node announces availability
    Announce {
        node_name: String,
        capabilities: NodeCapabilities,
    },
    
    // Request inference
    Request {
        request_id: RequestId,
        prompt: String,
        config: GenerationConfig,
        requester: PublicKey,
    },
    
    // Claim request for processing
    Claim {
        request_id: RequestId,
        worker: PublicKey,
    },
    
    // Stream generated token
    Token {
        request_id: RequestId,
        token: String,
        token_id: u32,
        index: usize,
    },
    
    // Generation complete
    Complete {
        request_id: RequestId,
        total_tokens: usize,
        duration_secs: f64,
    },
    
    // Generation failed
    Error {
        request_id: RequestId,
        error: String,
    },
    
    // Cancel request
    Cancel {
        request_id: RequestId,
        requester: PublicKey,
    },
}
```

## Current Status

✅ **Working**:
- Gossip network setup with iroh
- Message signing and verification
- Worker node request processing
- Token streaming via burn-central-runtime
- Concurrent request handling

🚧 **Prototype Limitations**:
- Client-only mode simplified (full gossip subscription for responses not implemented)
- Bootstrap node parsing not fully implemented (relies on local discovery)
- No request routing/load balancing strategy
- No timeout/retry logic
- Interactive REPL not implemented

🔮 **Future Enhancements**:
1. Full client mode with response handling
2. Smart load balancing (consider GPU memory, queue depth)
3. Request prioritization
4. Multi-hop routing for better network coverage
5. Persistent request/response history
6. Dashboard for network monitoring
7. Support for other model architectures

## Why Iroh Gossip?

**Traditional HTTP Server**:
- Centralized (single point of failure)
- Requires public IP/domain
- NAT/firewall complexity
- Load balancing requires extra infrastructure

**Iroh Gossip**:
- 🌍 Fully decentralized P2P
- 🔌 NAT traversal built-in
- 📡 Multicast-style message distribution
- 🔐 Built-in encryption and authentication
- 🚀 Works over QUIC for performance
- 🌐 Auto-discovery via mDNS/relay servers

## Comparison to Other Approaches

| Feature | HTTP Server | iroh-gossip |
|---------|------------|-------------|
| Deployment | Complex | Simple |
| Scaling | Manual | Automatic |
| NAT Traversal | Hard | Built-in |
| Discovery | DNS/Service Registry | Automatic |
| Auth | Custom | Built-in (ed25519) |
| Real-time | WebSocket/SSE | Native |
| Fault Tolerance | Needs orchestration | Inherent |

## Development

```bash
# Check compilation
cargo check -p llama-iroh-gossip --features cuda,llama3

# Run with debug logging
RUST_LOG=debug cargo run -p llama-iroh-gossip --features cuda,llama3 -- worker

# Run tests (when added)
cargo test -p llama-iroh-gossip
```

## Dependencies

- **iroh**: QUIC-based P2P networking
- **iroh-gossip**: Epidemic broadcast protocol
- **burn**: ML framework
- **burn-central-runtime**: Inference API
- **llama-burn**: Llama model implementation

## License

MIT OR Apache-2.0