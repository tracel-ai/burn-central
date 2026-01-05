//! Llama Gossip Node - Distributed LLM inference using iroh-gossip
//!
//! This binary creates a peer-to-peer inference node that can:
//! - Join a gossip network
//! - Process inference requests from other nodes
//! - Submit inference requests to the network
//! - Stream tokens in real-time via gossip

mod node;
mod protocol;

use anyhow::Result;
use clap::{Parser, Subcommand};
use iroh::EndpointId;
use node::InferenceNode;
use protocol::GenerationConfig;
use std::collections::HashSet;
use tracing::{error, info};

#[derive(Parser, Debug)]
#[command(name = "llama-gossip-node")]
#[command(about = "Distributed LLM inference using iroh-gossip", long_about = None)]
struct Args {
    /// Node name (for identification in the network)
    #[arg(short, long, default_value = "gossip-node")]
    name: String,

    /// Maximum sequence length
    #[arg(long, default_value = "256")]
    max_seq_len: usize,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run as a worker node (processes requests)
    Worker {
        /// Bootstrap node addresses (comma-separated node IDs)
        #[arg(short, long)]
        bootstrap: Option<String>,
    },

    /// Run as a client node (submits requests)
    Client {
        /// Bootstrap node addresses (comma-separated node IDs)
        #[arg(short, long, required = true)]
        bootstrap: String,

        /// Prompt to generate from
        #[arg(short, long)]
        prompt: String,

        /// Maximum tokens to generate
        #[arg(long, default_value = "50")]
        max_tokens: usize,

        /// Temperature
        #[arg(long, default_value = "0.7")]
        temperature: f64,

        /// Top-p sampling threshold
        #[arg(long, default_value = "0.9")]
        top_p: f64,
    },

    /// Run in hybrid mode (both worker and client)
    Hybrid {
        /// Bootstrap node addresses (comma-separated node IDs)
        #[arg(short, long)]
        bootstrap: Option<String>,
    },
}

#[cfg(feature = "cuda")]
async fn run_node(args: Args) -> Result<()> {
    use burn::backend::{cuda::CudaDevice, Cuda};
    use burn::tensor::f16;
    use llama_burn::llama::LlamaConfig;

    type Backend = Cuda<f16, i32>;

    let device = CudaDevice::default();
    info!("Using CUDA device: {:?}", device);

    // Load model based on feature flags
    #[cfg(feature = "llama3")]
    {
        info!("Loading Llama 3.2 1B model...");
        let llama = LlamaConfig::llama3_2_1b_pretrained::<Backend>(args.max_seq_len, &device)
            .map_err(|e| anyhow::anyhow!("Failed to load model: {}", e))?;
        run_with_model(args, llama, "CUDA".to_string(), "Llama-3.2-1B".to_string()).await
    }

    #[cfg(all(feature = "tiny", not(feature = "llama3")))]
    {
        info!("Loading TinyLlama model...");
        let llama = LlamaConfig::tiny_llama_pretrained::<Backend>(args.max_seq_len, &device)
            .map_err(|e| anyhow::anyhow!("Failed to load model: {}", e))?;
        run_with_model(
            args,
            llama,
            "CUDA".to_string(),
            "TinyLlama-1.1B".to_string(),
        )
        .await
    }

    #[cfg(not(any(feature = "llama3", feature = "tiny")))]
    {
        anyhow::bail!("Please enable either 'llama3' or 'tiny' feature");
    }
}

#[cfg(not(feature = "cuda"))]
async fn run_node(_args: Args) -> Result<()> {
    anyhow::bail!("This example requires the 'cuda' feature to be enabled");
}

async fn run_with_model<B, T>(
    args: Args,
    llama: llama_burn::llama::Llama<B, T>,
    backend_name: String,
    model_name: String,
) -> Result<()>
where
    B: burn::tensor::backend::Backend,
    B::Device: Send,
    T: llama_burn::tokenizer::Tokenizer + Send + 'static,
{
    info!("Spawning gossip inference node: {}", args.name);

    let node =
        InferenceNode::spawn(args.name.clone(), llama, backend_name, model_name, None).await?;

    info!("Node public key: {}", node.public_key());
    info!("Ready to join gossip network");

    match args.command {
        Commands::Worker { bootstrap } => {
            run_worker(node, bootstrap).await?;
        }
        Commands::Client {
            bootstrap,
            prompt,
            max_tokens,
            temperature,
            top_p,
        } => {
            run_client(node, bootstrap, prompt, max_tokens, temperature, top_p).await?;
        }
        Commands::Hybrid { bootstrap } => {
            run_hybrid(node, bootstrap).await?;
        }
    }

    Ok(())
}

async fn run_worker<B, T>(node: InferenceNode<B, T>, bootstrap: Option<String>) -> Result<()>
where
    B: burn::tensor::backend::Backend,
    B::Device: Send,
    T: llama_burn::tokenizer::Tokenizer + Send + 'static,
{
    info!("Running in WORKER mode");

    let bootstrap_addrs = parse_bootstrap(bootstrap);
    let _sender = node.join(bootstrap_addrs).await?;

    info!("Worker node active. Processing inference requests from the network...");
    info!("Press Ctrl+C to shutdown");

    // Wait for shutdown signal
    tokio::signal::ctrl_c().await?;
    info!("Shutting down...");
    node.shutdown().await?;

    Ok(())
}

async fn run_client<B, T>(
    node: InferenceNode<B, T>,
    bootstrap: String,
    prompt: String,
    max_tokens: usize,
    temperature: f64,
    top_p: f64,
) -> Result<()>
where
    B: burn::tensor::backend::Backend,
    B::Device: Send,
    T: llama_burn::tokenizer::Tokenizer + Send + 'static,
{
    info!("Running in CLIENT mode");

    let bootstrap_addrs = parse_bootstrap(Some(bootstrap));

    // Join the gossip network
    let sender = node.join(bootstrap_addrs).await?;

    info!("Joined gossip network");
    info!("Submitting inference request...");
    info!("Prompt: {}", prompt);

    // Submit the request
    let config = GenerationConfig {
        max_tokens,
        temperature,
        top_p,
    };

    let request_id = node.submit_request(&sender, prompt, config).await?;

    println!("\n=== Generated Output ===");
    println!("Request ID: {}", request_id);
    println!("Listening for tokens from workers...\n");

    // Wait for a bit to receive responses
    tokio::time::sleep(tokio::time::Duration::from_secs(30)).await;

    info!("Shutting down client...");
    node.shutdown().await?;

    Ok(())
}

async fn run_hybrid<B, T>(node: InferenceNode<B, T>, bootstrap: Option<String>) -> Result<()>
where
    B: burn::tensor::backend::Backend,
    B::Device: Send,
    T: llama_burn::tokenizer::Tokenizer + Send + 'static,
{
    info!("Running in HYBRID mode (worker + interactive client)");

    let bootstrap_addrs = parse_bootstrap(bootstrap);
    let _sender = node.join(bootstrap_addrs).await?;

    info!("Node joined gossip network");
    info!("This node will process requests AND allow you to submit requests");
    println!("\nCommands:");
    println!("  /request <prompt> - Submit an inference request");
    println!("  /quit - Shutdown the node");
    println!();

    // In a full implementation, we'd have an interactive REPL here
    // For the prototype, just run as worker
    info!("Interactive mode not yet implemented - running as worker");
    info!("Press Ctrl+C to shutdown");

    tokio::signal::ctrl_c().await?;
    info!("Shutting down...");
    node.shutdown().await?;

    Ok(())
}

fn parse_bootstrap(bootstrap: Option<String>) -> HashSet<EndpointId> {
    // In a real implementation, this would parse node addresses from strings
    // For the prototype, return empty set (nodes will discover each other via mDNS/relay)
    let _ = bootstrap;
    HashSet::new()
}

#[tokio::main]
async fn main() -> Result<()> {
    // Setup logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,iroh=info,llama_gossip=debug".into()),
        )
        .init();

    let args = Args::parse();

    if let Err(e) = run_node(args).await {
        error!("Fatal error: {}", e);
        std::process::exit(1);
    }

    Ok(())
}
