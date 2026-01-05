//! Gossip-based inference node implementation.
//!
//! This module implements a node that can:
//! - Join an iroh-gossip network
//! - Receive inference requests via gossip
//! - Process requests using burn-central-runtime
//! - Stream tokens back via gossip

use crate::protocol::{
    GenerationConfig, InferenceMessage, MessageCodec, NodeCapabilities, RequestId, VerifiedMessage,
};
use anyhow::{Context, Result};
use burn::tensor::backend::Backend;
use burn_central_runtime::inference::{
    CancelToken, In, Inference, InferenceBuilder, ModelAccessor, OutStream,
};
use futures::TryStreamExt;
use iroh::{protocol::Router, PublicKey, SecretKey};
use iroh_gossip::{
    api::{Event as GossipEvent, GossipSender},
    net::{Gossip, GOSSIP_ALPN},
    proto::TopicId,
};
use llama_burn::{
    llama::Llama,
    sampling::{Sampler, TopP},
    tokenizer::Tokenizer,
};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::Instant,
};
use tokio::sync::{Mutex as TokioMutex, RwLock};
use tracing::{debug, error, info, warn};

const GOSSIP_TOPIC: &str = "llama-inference-gossip";

/// Token output from generation
#[derive(Debug, Clone)]
pub struct TokenOutput {
    pub token: String,
    pub token_id: u32,
    pub index: usize,
}

/// Generation request
#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
}

/// Model wrapper for burn-central-runtime
pub struct LlamaModel<B: Backend, T: Tokenizer> {
    llama: Llama<B, T>,
}

impl<B: Backend, T: Tokenizer> LlamaModel<B, T> {
    pub fn new(llama: Llama<B, T>) -> Self {
        Self { llama }
    }

    pub fn generate_streaming<F>(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
        sampler: &mut Sampler,
        mut on_token: F,
    ) -> Result<usize, String>
    where
        F: FnMut(u32, &str, usize) -> Result<bool, String>,
    {
        use burn::tensor::{activation::softmax, ElementConversion, Int, Tensor};

        // Tokenize input
        let bos = !cfg!(feature = "tiny");
        let tokens = self.llama.tokenizer.encode(prompt, bos, false);
        let input_tokens = Tensor::<B, 1, Int>::from_ints(tokens.as_slice(), &self.llama.device);
        let prompt_len = input_tokens.dims()[0];
        let mut tokens = Tensor::<B, 1, Int>::empty([prompt_len + max_tokens], &self.llama.device);
        tokens = tokens.slice_assign([0..prompt_len], input_tokens);

        let stop_tokens = Tensor::from_ints(
            self.llama.tokenizer.stop_ids().as_slice(),
            &self.llama.device,
        );

        let mut num_tokens = 0;
        let mut input_pos = Tensor::<B, 1, Int>::arange(0..prompt_len as i64, &self.llama.device);

        for i in 0..max_tokens {
            let x = tokens.clone().select(0, input_pos.clone()).reshape([1, -1]);
            let logits = self
                .llama
                .model
                .forward(x, &mut self.llama.cache, &self.llama.rope);

            let [batch_size, seq_len, _vocab_size] = logits.dims();
            let mut next_token_logits = logits
                .slice([0..batch_size, seq_len - 1..seq_len])
                .squeeze_dim(1);

            if temperature > 0.0 {
                next_token_logits = softmax(next_token_logits / temperature, 1);
            }

            let next_token = sampler.sample(next_token_logits).squeeze_dim(0);

            if stop_tokens
                .clone()
                .equal(next_token.clone())
                .any()
                .into_scalar()
                .elem::<bool>()
            {
                break;
            }

            let token_id = next_token
                .clone()
                .into_data()
                .as_slice::<B::IntElem>()
                .unwrap()[0]
                .elem::<u32>();

            let token_text = self.llama.tokenizer.decode(vec![token_id]);

            let should_continue = on_token(token_id, &token_text, i)
                .map_err(|e| format!("Token callback failed: {}", e))?;

            if !should_continue {
                break;
            }

            tokens = tokens.slice_assign([prompt_len + i..prompt_len + i + 1], next_token);
            num_tokens += 1;

            let t = input_pos.dims()[0];
            input_pos = input_pos.slice([t - 1..t]) + 1;
        }

        Ok(num_tokens)
    }
}

/// Inference handler for burn-central-runtime
fn streaming_generate_handler<B: Backend, T: Tokenizer + 'static>(
    In(request): In<GenerateRequest>,
    model: ModelAccessor<LlamaModel<B, T>>,
    cancel: CancelToken,
    output: OutStream<TokenOutput>,
) -> Result<(), String> {
    let mut sampler = if request.temperature > 0.0 {
        Sampler::TopP(TopP::new(request.top_p, 42))
    } else {
        Sampler::Argmax
    };

    model.submit(move |llama_model| {
        llama_model
            .generate_streaming(
                &request.prompt,
                request.max_tokens,
                request.temperature,
                &mut sampler,
                |token_id, token_text, index| {
                    if cancel.is_cancelled() {
                        return Err("Generation cancelled".to_string());
                    }

                    let token_output = TokenOutput {
                        token: token_text.to_string(),
                        token_id,
                        index,
                    };

                    output
                        .emit(token_output)
                        .map_err(|e| format!("Failed to emit token: {}", e.source))?;

                    Ok(true)
                },
            )
            .map(|_| ())
            .map_err(|e| format!("Generation failed: {}", e))
    })
}

/// State tracking for ongoing inference jobs
#[derive(Default)]
struct InferenceState {
    claimed_by_us: HashMap<RequestId, Instant>,
}

/// A node in the gossip-based inference network
pub struct InferenceNode<B: Backend, T: Tokenizer + Send + 'static> {
    secret_key: SecretKey,
    node_name: String,
    router: Router,
    gossip: Gossip,
    inference: Arc<Inference<B, LlamaModel<B, T>, In<GenerateRequest>, TokenOutput>>,
    state: Arc<RwLock<InferenceState>>,
    capabilities: NodeCapabilities,
}

impl<B: Backend, T: Tokenizer + Send + 'static> InferenceNode<B, T>
where
    B::Device: Send,
{
    /// Spawn a new gossip inference node
    pub async fn spawn(
        node_name: String,
        llama: Llama<B, T>,
        backend_name: String,
        model_name: String,
        secret_key: Option<SecretKey>,
    ) -> Result<Self> {
        let secret_key = secret_key.unwrap_or_else(|| SecretKey::generate(&mut rand::rng()));

        // Setup iroh endpoint with gossip ALPN
        let endpoint = iroh::Endpoint::builder()
            .secret_key(secret_key.clone())
            .alpns(vec![GOSSIP_ALPN.to_vec()])
            .bind()
            .await
            .context("failed to bind endpoint")?;

        let endpoint_id = endpoint.id();
        info!("Gossip endpoint bound");
        info!("Endpoint ID: {endpoint_id}");

        // Setup gossip protocol
        let gossip = Gossip::builder().spawn(endpoint.clone());
        info!("Gossip protocol spawned");

        // Setup router to accept gossip connections
        let router = Router::builder(endpoint)
            .accept(GOSSIP_ALPN, gossip.clone())
            .spawn();
        info!("Router spawned");

        // Build inference instance with burn-central-runtime
        let model = LlamaModel::new(llama);
        let inference = InferenceBuilder::<B>::new()
            .with_model(model)
            .build(streaming_generate_handler);
        let inference = Arc::new(inference);

        let capabilities = NodeCapabilities {
            backend: backend_name,
            model: model_name,
            max_concurrent: 4,
            current_load: 0,
        };

        Ok(Self {
            secret_key,
            node_name,
            router,
            gossip,
            inference,
            state: Arc::new(RwLock::new(InferenceState::default())),
            capabilities,
        })
    }

    /// Get this node's public key
    pub fn public_key(&self) -> PublicKey {
        self.secret_key.public()
    }

    /// Get this node's endpoint ID
    pub fn endpoint_id(&self) -> iroh::EndpointId {
        self.router.endpoint().id()
    }

    /// Join the inference gossip network
    pub async fn join(
        &self,
        bootstrap: HashSet<iroh::EndpointId>,
    ) -> Result<Arc<TokioMutex<GossipSender>>> {
        let topic_id = TopicId::from_bytes(*blake3::hash(GOSSIP_TOPIC.as_bytes()).as_bytes());
        info!("Joining gossip topic: {topic_id}");

        // Subscribe to gossip topic
        let gossip_topic = self
            .gossip
            .subscribe(topic_id, bootstrap.into_iter().collect())
            .await
            .context("failed to subscribe to gossip topic")?;

        let (sender, mut receiver) = gossip_topic.split();
        let sender = Arc::new(TokioMutex::new(sender));

        // Broadcast announcement
        self.broadcast_announcement(&sender).await?;

        // Spawn background task to handle gossip events
        let secret_key = self.secret_key.clone();
        let inference = self.inference.clone();
        let state = self.state.clone();
        let public_key = self.public_key();
        let sender_clone = sender.clone();

        tokio::spawn(async move {
            info!("Gossip event loop started");

            loop {
                let event = match receiver.try_next().await {
                    Ok(Some(event)) => event,
                    Ok(None) => {
                        info!("Gossip receiver closed");
                        break;
                    }
                    Err(e) => {
                        warn!("Error receiving gossip event: {e}");
                        continue;
                    }
                };

                match event {
                    GossipEvent::Received(msg) => match MessageCodec::decode(&msg.content) {
                        Ok(verified) => {
                            Self::handle_message(
                                verified,
                                &sender_clone,
                                &secret_key,
                                &inference,
                                &state,
                                public_key,
                            )
                            .await;
                        }
                        Err(e) => {
                            warn!("Failed to decode message: {e}");
                        }
                    },
                    GossipEvent::NeighborUp(peer) => {
                        debug!("Neighbor up: {peer}");
                    }
                    GossipEvent::NeighborDown(peer) => {
                        debug!("Neighbor down: {peer}");
                    }
                    GossipEvent::Lagged => {
                        warn!("Gossip stream lagged, some messages may have been dropped");
                    }
                }
            }

            info!("Gossip event loop ended");
        });

        Ok(sender)
    }

    /// Broadcast our node announcement
    async fn broadcast_announcement(&self, sender: &Arc<TokioMutex<GossipSender>>) -> Result<()> {
        let message = InferenceMessage::Announce {
            node_name: self.node_name.clone(),
            capabilities: self.capabilities.clone(),
        };

        let encoded = MessageCodec::encode(&self.secret_key, message)
            .context("failed to encode announcement")?;

        sender
            .lock()
            .await
            .broadcast(encoded.into())
            .await
            .context("failed to broadcast announcement")?;

        info!("Broadcasted node announcement");
        Ok(())
    }

    /// Submit an inference request to the network
    pub async fn submit_request(
        &self,
        sender: &Arc<TokioMutex<GossipSender>>,
        prompt: String,
        config: GenerationConfig,
    ) -> Result<RequestId> {
        let request_id = RequestId::new();

        let message = InferenceMessage::Request {
            request_id,
            prompt,
            config,
            requester: self.public_key(),
        };

        let encoded =
            MessageCodec::encode(&self.secret_key, message).context("failed to encode request")?;

        sender
            .lock()
            .await
            .broadcast(encoded.into())
            .await
            .context("failed to broadcast request")?;

        info!("Submitted request: {request_id}");
        Ok(request_id)
    }

    /// Handle incoming gossip messages
    async fn handle_message(
        verified: VerifiedMessage,
        sender: &Arc<TokioMutex<GossipSender>>,
        secret_key: &SecretKey,
        inference: &Arc<Inference<B, LlamaModel<B, T>, In<GenerateRequest>, TokenOutput>>,
        state: &Arc<RwLock<InferenceState>>,
        our_public_key: PublicKey,
    ) {
        match verified.message {
            InferenceMessage::Announce {
                node_name,
                capabilities,
            } => {
                info!(
                    "Node announced: {} (from: {}, backend: {}, model: {})",
                    node_name, verified.from, capabilities.backend, capabilities.model
                );
            }

            InferenceMessage::Request {
                request_id,
                prompt,
                config,
                requester,
            } => {
                info!(
                    "Received request {request_id} from {requester}: {}...",
                    &prompt.chars().take(50).collect::<String>()
                );

                // Check if we should claim this request
                let should_claim = {
                    let state = state.read().await;
                    state.claimed_by_us.len() < 2 // Simple load balancing
                };

                if should_claim {
                    // Broadcast claim
                    let claim_msg = InferenceMessage::Claim {
                        request_id,
                        worker: our_public_key,
                    };

                    if let Ok(encoded) = MessageCodec::encode(secret_key, claim_msg) {
                        let _ = sender.lock().await.broadcast(encoded.into()).await;
                    }

                    // Process the request
                    Self::process_request(
                        request_id, prompt, config, inference, sender, secret_key, state,
                    )
                    .await;
                }
            }

            InferenceMessage::Claim { request_id, worker } => {
                debug!("Request {request_id} claimed by {worker}");
            }

            InferenceMessage::Token {
                request_id,
                token,
                index,
                ..
            } => {
                debug!("Token {index} for request {request_id}: {token}");
            }

            InferenceMessage::Complete {
                request_id,
                total_tokens,
                duration_secs,
            } => {
                info!(
                    "Request {request_id} completed: {total_tokens} tokens in {duration_secs:.2}s"
                );
            }

            InferenceMessage::Error { request_id, error } => {
                error!("Request {request_id} failed: {error}");
            }

            InferenceMessage::Cancel {
                request_id,
                requester: _,
            } => {
                info!("Cancellation requested for {request_id}");
                // Cancellation would be handled by the job holder
                // For now, we just log it
            }
        }
    }

    /// Process an inference request
    async fn process_request(
        request_id: RequestId,
        prompt: String,
        config: GenerationConfig,
        inference: &Arc<Inference<B, LlamaModel<B, T>, In<GenerateRequest>, TokenOutput>>,
        sender: &Arc<TokioMutex<GossipSender>>,
        secret_key: &SecretKey,
        state: &Arc<RwLock<InferenceState>>,
    ) {
        info!("Processing request {request_id}");

        let request = GenerateRequest {
            prompt,
            max_tokens: config.max_tokens,
            temperature: config.temperature,
            top_p: config.top_p,
        };

        // Use default device for the backend
        let device = B::Device::default();

        let job = inference.infer(request).with_devices([device]).spawn();

        // Track that we claimed this request
        {
            let mut state = state.write().await;
            state.claimed_by_us.insert(request_id, Instant::now());
        }

        // Stream tokens
        let sender = sender.clone();
        let secret_key = secret_key.clone();
        let state = state.clone();

        tokio::spawn(async move {
            let start = Instant::now();
            let mut total_tokens = 0;

            for token in job.stream.iter() {
                total_tokens += 1;

                let msg = InferenceMessage::Token {
                    request_id,
                    token: token.token,
                    token_id: token.token_id,
                    index: token.index,
                };

                if let Ok(encoded) = MessageCodec::encode(&secret_key, msg) {
                    let _ = sender.lock().await.broadcast(encoded.into()).await;
                }
            }

            let duration = start.elapsed().as_secs_f64();

            // Send completion or error
            let msg = match job.join() {
                Ok(_) => InferenceMessage::Complete {
                    request_id,
                    total_tokens,
                    duration_secs: duration,
                },
                Err(e) => InferenceMessage::Error {
                    request_id,
                    error: format!("{:?}", e),
                },
            };

            if let Ok(encoded) = MessageCodec::encode(&secret_key, msg) {
                let _ = sender.lock().await.broadcast(encoded.into()).await;
            }

            // Cleanup
            let mut state = state.write().await;
            state.claimed_by_us.remove(&request_id);
        });
    }

    /// Shutdown the node
    pub async fn shutdown(&self) -> Result<()> {
        if let Err(e) = self.router.shutdown().await {
            warn!("Failed to shutdown router cleanly: {e}");
        }
        self.router.endpoint().close().await;
        Ok(())
    }
}
