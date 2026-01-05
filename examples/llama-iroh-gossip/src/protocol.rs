//! Protocol definitions for gossip-based distributed inference.
//!
//! This module defines the message types that are gossiped between nodes
//! to coordinate inference requests and stream token responses.

use anyhow::{Context, Result};
use iroh::{PublicKey, SecretKey, Signature};
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

/// Unique identifier for an inference request
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RequestId(pub Uuid);

impl RequestId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for RequestId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for RequestId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Configuration for text generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 50,
            temperature: 0.7,
            top_p: 0.9,
        }
    }
}

/// Messages gossiped between inference nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InferenceMessage {
    /// A node announces it's available to handle inference requests
    Announce {
        node_name: String,
        capabilities: NodeCapabilities,
    },

    /// A request for inference
    Request {
        request_id: RequestId,
        prompt: String,
        config: GenerationConfig,
        requester: PublicKey,
    },

    /// A node claims a request to process it
    Claim {
        request_id: RequestId,
        worker: PublicKey,
    },

    /// A generated token (streamed as generation proceeds)
    Token {
        request_id: RequestId,
        token: String,
        token_id: u32,
        index: usize,
    },

    /// Generation completed successfully
    Complete {
        request_id: RequestId,
        total_tokens: usize,
        duration_secs: f64,
    },

    /// Generation failed
    Error {
        request_id: RequestId,
        error: String,
    },

    /// Cancel an ongoing request
    Cancel {
        request_id: RequestId,
        requester: PublicKey,
    },
}

/// Capabilities advertised by a node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCapabilities {
    pub backend: String,
    pub model: String,
    pub max_concurrent: usize,
    pub current_load: usize,
}

/// Wire format for messages (includes timestamp)
#[derive(Debug, Serialize, Deserialize)]
enum WireMessage {
    V0 {
        timestamp: u64,
        message: InferenceMessage,
    },
}

/// A signed message that has been verified
#[derive(Debug)]
pub struct VerifiedMessage {
    pub from: PublicKey,
    pub timestamp: u64,
    pub message: InferenceMessage,
}

/// A signed message (not yet verified)
#[derive(Debug, Serialize, Deserialize)]
struct SignedMessage {
    from: PublicKey,
    data: Vec<u8>,
    signature: Signature,
}

impl SignedMessage {
    /// Sign and encode a message for gossip
    pub fn sign_and_encode(secret_key: &SecretKey, message: InferenceMessage) -> Result<Vec<u8>> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("system time before UNIX epoch")?
            .as_micros() as u64;

        let wire_message = WireMessage::V0 { timestamp, message };
        let data =
            postcard::to_stdvec(&wire_message).context("failed to serialize wire message")?;

        let signature = secret_key.sign(&data);
        let from = secret_key.public();

        let signed_message = Self {
            from,
            data,
            signature,
        };

        postcard::to_stdvec(&signed_message).context("failed to serialize signed message")
    }

    /// Verify and decode a message from gossip
    pub fn verify_and_decode(bytes: &[u8]) -> Result<VerifiedMessage> {
        let signed_message: Self =
            postcard::from_bytes(bytes).context("failed to deserialize signed message")?;

        // Verify signature
        signed_message
            .from
            .verify(&signed_message.data, &signed_message.signature)
            .context("signature verification failed")?;

        // Decode inner message
        let wire_message: WireMessage = postcard::from_bytes(&signed_message.data)
            .context("failed to deserialize wire message")?;

        let WireMessage::V0 { timestamp, message } = wire_message;

        Ok(VerifiedMessage {
            from: signed_message.from,
            timestamp,
            message,
        })
    }
}

/// Public API for encoding/decoding gossip messages
pub struct MessageCodec;

impl MessageCodec {
    pub fn encode(secret_key: &SecretKey, message: InferenceMessage) -> Result<Vec<u8>> {
        SignedMessage::sign_and_encode(secret_key, message)
    }

    pub fn decode(bytes: &[u8]) -> Result<VerifiedMessage> {
        SignedMessage::verify_and_decode(bytes)
    }
}
