use clap::Parser;
use futures_util::StreamExt;
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(name = "inference-actor-client")]
#[command(about = "CLI client for the axum_server demo", long_about = None)]
struct Args {
    /// Server base URL (default: http://127.0.0.1:3001)
    #[arg(long, default_value = "http://127.0.0.1:3001")]
    url: String,

    /// Steps to request
    #[arg(long, default_value_t = 5)]
    steps: usize,

    /// Input value to fill the model input vector
    #[arg(long, default_value_t = 1.0)]
    value: f32,

    /// Request timeout in seconds
    #[arg(long, default_value_t = 30)]
    timeout_secs: u64,
}

#[derive(Debug, Serialize)]
struct GenerateRequest {
    steps: usize,
    value: f32,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let endpoint = format!("{}/generate", args.url.trim_end_matches('/'));

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(args.timeout_secs))
        .build()?;

    let resp = client
        .post(endpoint)
        .json(&GenerateRequest {
            steps: args.steps,
            value: args.value,
        })
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("request failed: {}\n{}", status, body);
    }

    let mut stream = resp.bytes_stream();
    let mut buffer = String::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        let text = String::from_utf8_lossy(&chunk);
        buffer.push_str(&text);

        while let Some(pos) = buffer.find('\n') {
            let line = buffer[..pos].trim_end_matches(['\r'].as_slice()).to_string();
            buffer = buffer[pos + 1..].to_string();

            if line.starts_with("data:") {
                let data = line.trim_start_matches("data:").trim();
                if data == "done" {
                    return Ok(());
                }
                println!("{}", data);
            }
        }
    }

    Ok(())
}
