//! Simple CLI client for the Axum streaming server.
//!
//! Run with:
//! cargo run --example axum_client --features llama3,cuda -- \
//!   --url http://127.0.0.1:3000/generate \
//!   --prompt "Hello from client"

use clap::Parser;
use futures_util::StreamExt;
use llama_burn::inference::GenerateRequest;
use serde::Deserialize;

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Server URL (SSE endpoint)
    #[arg(long, default_value = "http://127.0.0.1:3000/generate")]
    url: String,

    /// Prompt to generate from
    #[arg(long)]
    prompt: String,

    /// Maximum tokens to generate
    #[arg(long, default_value_t = 50)]
    max_tokens: usize,

    /// Temperature for sampling
    #[arg(long, default_value_t = 0.7)]
    temperature: f64,

    /// Top-p for sampling
    #[arg(long, default_value_t = 0.9)]
    top_p: f64,

    /// RNG seed
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

#[derive(Debug, Deserialize)]
struct TokenOutput {
    token: String,
    token_id: u32,
    index: usize,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let request = GenerateRequest::new(args.prompt)
        .with_max_tokens(args.max_tokens)
        .with_temperature(args.temperature)
        .with_top_p(args.top_p)
        .with_seed(args.seed);

    let client = reqwest::Client::new();
    let response = client.post(&args.url).json(&request).send().await?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!("Request failed: {status}\n{body}").into());
    }

    let mut stream = response.bytes_stream();
    let mut buffer = String::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        buffer.push_str(&String::from_utf8_lossy(&chunk));

        while let Some(pos) = buffer.find("\n\n") {
            let event = buffer[..pos].to_string();
            buffer = buffer[pos + 2..].to_string();

            if let Some((event_name, data)) = parse_event(&event) {
                if event_name.as_deref() == Some("done") {
                    println!();
                    return Ok(());
                }

                if data == "keep-alive" {
                    continue;
                }

                if let Ok(token) = serde_json::from_str::<TokenOutput>(&data) {
                    print!("{}", token.token);
                    std::io::Write::flush(&mut std::io::stdout()).ok();
                }
            }
        }
    }

    Ok(())
}

fn parse_event(raw: &str) -> Option<(Option<String>, String)> {
    let mut event_name: Option<String> = None;
    let mut data_lines: Vec<String> = Vec::new();

    for line in raw.lines() {
        if let Some(rest) = line.strip_prefix("event: ") {
            event_name = Some(rest.to_string());
        } else if let Some(rest) = line.strip_prefix("data: ") {
            data_lines.push(rest.to_string());
        }
    }

    if data_lines.is_empty() {
        return None;
    }

    Some((event_name, data_lines.join("\n")))
}
