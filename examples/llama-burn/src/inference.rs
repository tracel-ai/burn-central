//! Inference API integration for Llama models.
//!
//! This module provides integration between llama-burn and burn-central-runtime's inference API,
//! enabling streaming token generation with built-in cancellation and thread-safe model access.
//!
//! The core inference logic lives in `llama.rs` (see `Llama::generate_streaming`).
//! This module provides the request/response types and handler for the inference API.
//!
//! Two handlers are provided:
//! - `streaming_handler`: Original blocking handler (entire generation in one model.submit call)
//! - `concurrent_streaming_handler`: Improved handler that generates one token at a time,
//!   releasing the model between tokens to enable concurrent request processing.

use burn::tensor::backend::Backend;
use burn::tensor::{ElementConversion, Int, Shape, Tensor, TensorData};
use burn_central_runtime::inference::{CancelToken, Extension, In, ModelAccessor, OutStream};
use crossbeam_channel as channel;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, OnceLock,
};
use std::time::Duration;
use tracing::{debug, info, warn};

use crate::{
    llama::{temperature_scaled_softmax, Llama},
    sampling::{Sampler, TopP},
    tokenizer::Tokenizer,
};

/// Request for text generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub seed: u64,
}

impl Default for GenerateRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            max_tokens: 50,
            temperature: 0.7,
            top_p: 0.9,
            seed: 42,
        }
    }
}

impl GenerateRequest {
    /// Create a new request with the given prompt and default parameters.
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            ..Default::default()
        }
    }

    /// Set the maximum number of tokens to generate.
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set the temperature for sampling.
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set the top-p value for nucleus sampling.
    pub fn with_top_p(mut self, top_p: f64) -> Self {
        self.top_p = top_p;
        self
    }

    /// Set the random seed for sampling.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

/// Output token with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenOutput {
    /// The decoded token text.
    pub token: String,
    /// The token ID.
    pub token_id: u32,
    /// The position in the generated sequence (0-indexed).
    pub index: usize,
}

/// Configuration for continuous batching.
#[derive(Clone, Debug)]
pub struct ContinuousBatchConfig {
    /// Maximum number of concurrent sequences to batch.
    pub max_batch_size: Option<usize>,
    /// Max time to wait for new requests before running a decode step.
    pub max_wait_ms: u64,
}

impl Default for ContinuousBatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: None,
            max_wait_ms: 2,
        }
    }
}

/// Continuous batching engine shared across inference requests.
pub struct ContinuousBatcher<B: Backend, T: Tokenizer + Send + Sync + 'static> {
    tx: channel::Sender<BatcherCommand>,
    _phantom: PhantomData<(B, T)>,
}

impl<B: Backend, T: Tokenizer + Send + Sync + 'static> ContinuousBatcher<B, T> {
    pub fn new(model: ModelAccessor<Llama<B, T>>, config: ContinuousBatchConfig) -> Arc<Self> {
        let (tx, rx) = channel::unbounded();
        std::thread::spawn(move || run_continuous_batcher(model, rx, config));
        Arc::new(Self {
            tx,
            _phantom: PhantomData,
        })
    }

    fn submit(
        &self,
        request: GenerateRequest,
        reply: channel::Sender<BatcherEvent>,
        cancel_flag: Arc<AtomicBool>,
    ) -> Result<(), String> {
        self.tx
            .send(BatcherCommand::Submit {
                request,
                reply,
                cancel_flag,
            })
            .map_err(|_| "Continuous batcher unavailable".to_string())
    }
}

/// Per-inference scheduler wrapper for continuous batching.
///
/// This object is intended to be attached to an `Inference` via
/// `InferenceBuilder::with_extension` and then extracted in the handler using
/// `Extension<ContinuousBatchScheduler<...>>`.
pub struct ContinuousBatchScheduler<B: Backend + 'static, T: Tokenizer + Send + Sync + 'static> {
    config: ContinuousBatchConfig,
    batcher: OnceLock<Arc<ContinuousBatcher<B, T>>>,
}

impl<B: Backend + 'static, T: Tokenizer + Send + Sync + 'static> ContinuousBatchScheduler<B, T> {
    pub fn new(config: ContinuousBatchConfig) -> Self {
        Self {
            config,
            batcher: OnceLock::new(),
        }
    }

    fn get_or_init_batcher(
        &self,
        model: ModelAccessor<Llama<B, T>>,
    ) -> Arc<ContinuousBatcher<B, T>> {
        self.batcher
            .get_or_init(|| ContinuousBatcher::new(model, self.config.clone()))
            .clone()
    }

    fn submit(
        &self,
        model: ModelAccessor<Llama<B, T>>,
        request: GenerateRequest,
        reply: channel::Sender<BatcherEvent>,
        cancel_flag: Arc<AtomicBool>,
    ) -> Result<(), String> {
        let batcher = self.get_or_init_batcher(model);
        batcher.submit(request, reply, cancel_flag)
    }
}

enum BatcherCommand {
    Submit {
        request: GenerateRequest,
        reply: channel::Sender<BatcherEvent>,
        cancel_flag: Arc<AtomicBool>,
    },
}

enum BatcherEvent {
    Token(TokenOutput),
    Done,
    Error(String),
}

struct PendingRequest {
    id: u64,
    request: GenerateRequest,
    reply: channel::Sender<BatcherEvent>,
    cancel_flag: Arc<AtomicBool>,
}

struct ActiveRequest {
    id: u64,
    slot: usize,
    max_tokens: usize,
    generated: usize,
    temperature: f64,
    sampler: Sampler,
    pending_input: u32,
    reply: channel::Sender<BatcherEvent>,
    cancel_flag: Arc<AtomicBool>,
}

fn run_continuous_batcher<B: Backend + 'static, T: Tokenizer + Send + Sync + 'static>(
    model: ModelAccessor<Llama<B, T>>,
    rx: channel::Receiver<BatcherCommand>,
    config: ContinuousBatchConfig,
) {
    let stop_ids = model.submit(|m| m.tokenizer.stop_ids());
    let capacity = model.submit(|m| m.cache.get(0).map(|c| c.lens().len()).unwrap_or(0));
    let max_batch_size = config.max_batch_size.unwrap_or(capacity).min(capacity);

    let mut free_slots: Vec<usize> = (0..max_batch_size).collect();
    let mut active: Vec<ActiveRequest> = Vec::new();
    let mut pending: Vec<PendingRequest> = Vec::new();
    let mut next_request_id: u64 = 1;

    info!(
        "continuous batcher started: capacity={}, max_batch_size={}, max_wait_ms={}",
        capacity, max_batch_size, config.max_wait_ms
    );

    loop {
        if active.is_empty() && pending.is_empty() {
            match rx.recv() {
                Ok(cmd) => {
                    let BatcherCommand::Submit {
                        request,
                        reply,
                        cancel_flag,
                    } = cmd;
                    let id = next_request_id;
                    next_request_id += 1;
                    debug!("request queued: id={}", id);
                    pending.push(PendingRequest {
                        id,
                        request,
                        reply,
                        cancel_flag,
                    });
                }
                Err(_) => break,
            }
        } else if pending.is_empty() {
            if let Ok(cmd) = rx.recv_timeout(Duration::from_millis(config.max_wait_ms)) {
                let BatcherCommand::Submit {
                    request,
                    reply,
                    cancel_flag,
                } = cmd;
                let id = next_request_id;
                next_request_id += 1;
                debug!("request queued: id={}", id);
                pending.push(PendingRequest {
                    id,
                    request,
                    reply,
                    cancel_flag,
                });
            }
        }

        while let Ok(cmd) = rx.try_recv() {
            let BatcherCommand::Submit {
                request,
                reply,
                cancel_flag,
            } = cmd;
            let id = next_request_id;
            next_request_id += 1;
            debug!("request queued: id={}", id);
            pending.push(PendingRequest {
                id,
                request,
                reply,
                cancel_flag,
            });
        }

        if max_batch_size == 0 {
            for pending_req in pending.drain(..) {
                let _ = pending_req.reply.send(BatcherEvent::Error(
                    "Continuous batching not available (max_batch_size=0)".to_string(),
                ));
            }
            continue;
        }

        // Prefill pending requests while slots are available.
        let idx = 0;
        while idx < pending.len() && !free_slots.is_empty() {
            let slot = free_slots.pop().unwrap();
            let pending_req = pending.swap_remove(idx);

            debug!(
                "assign slot: id={}, slot={}, pending={}, active={}, free={}",
                pending_req.id,
                slot,
                pending.len(),
                active.len(),
                free_slots.len()
            );

            if pending_req.cancel_flag.load(Ordering::Relaxed) {
                let _ = pending_req
                    .reply
                    .send(BatcherEvent::Error("Generation cancelled".to_string()));
                free_slots.push(slot);
                continue;
            }

            if pending_req.request.max_tokens == 0 {
                let _ = pending_req.reply.send(BatcherEvent::Done);
                free_slots.push(slot);
                continue;
            }

            model.submit(move |m| m.reset_cache_slot(slot));

            let mut sampler = if pending_req.request.temperature > 0.0 {
                Sampler::TopP(TopP::new(
                    pending_req.request.top_p,
                    pending_req.request.seed,
                ))
            } else {
                Sampler::Argmax
            };

            let request_id = pending_req.id;
            let prompt = pending_req.request.prompt.clone();
            let temperature = pending_req.request.temperature;
            let (prefill_result, returned_sampler) = model.submit(move |llama_model| {
                let prompt_tokens = llama_model.tokenize(&prompt);
                let prompt_len = prompt_tokens.dims()[0];
                debug!("prefill tokenize: id={}, prompt_len={}", request_id, prompt_len);
                if prompt_len == 0 {
                    return (Err("Prompt produced no tokens".to_string()), sampler);
                }
                let input = prompt_tokens.reshape([1, prompt_len]);
                let logits = llama_model.prefill_slot(slot, input);
                let [_batch, seq_len, _vocab_size] = logits.dims();
                let mut next_token_logits =
                    logits.slice([0..1, seq_len - 1..seq_len]).squeeze_dim(1);

                if temperature > 0.0 {
                    next_token_logits = temperature_scaled_softmax(next_token_logits, temperature);
                }

                let next_token = sampler
                    .sample(next_token_logits)
                    .squeeze_dim::<1>(0);
                let token_id = next_token
                    .clone()
                    .into_data()
                    .as_slice::<B::IntElem>()
                    .unwrap()[0]
                    .elem::<u32>();
                let token_text = llama_model.tokenizer.decode(vec![token_id]);
                (Ok((token_id, token_text)), sampler)
            });

            sampler = returned_sampler;

            let (token_id, token_text) = match prefill_result {
                Ok(result) => result,
                Err(err) => {
                    warn!("prefill failed: id={}, err={}", pending_req.id, err);
                    let _ = pending_req.reply.send(BatcherEvent::Error(err));
                    free_slots.push(slot);
                    continue;
                }
            };

            let is_stop = stop_ids.contains(&token_id);
            if is_stop {
                debug!("prefill stop token: id={}", pending_req.id);
                let _ = pending_req.reply.send(BatcherEvent::Done);
                free_slots.push(slot);
                continue;
            }

            let token_output = TokenOutput {
                token: token_text,
                token_id,
                index: 0,
            };

            if pending_req
                .reply
                .send(BatcherEvent::Token(token_output))
                .is_err()
            {
                warn!("prefill emit failed: id={}", pending_req.id);
                free_slots.push(slot);
                continue;
            }

            if 1 >= pending_req.request.max_tokens {
                debug!("request completed at prefill: id={}", pending_req.id);
                let _ = pending_req.reply.send(BatcherEvent::Done);
                free_slots.push(slot);
                continue;
            }

            active.push(ActiveRequest {
                id: pending_req.id,
                slot,
                max_tokens: pending_req.request.max_tokens,
                generated: 1,
                temperature,
                sampler,
                pending_input: token_id,
                reply: pending_req.reply,
                cancel_flag: pending_req.cancel_flag,
            });
        }

        // Remove cancelled requests before decoding.
        let mut i = 0;
        while i < active.len() {
            if active[i].cancel_flag.load(Ordering::Relaxed) {
                let req = active.swap_remove(i);
                debug!("request cancelled: id={}", req.id);
                let _ = req
                    .reply
                    .send(BatcherEvent::Error("Generation cancelled".to_string()));
                model.submit(move |m| m.reset_cache_slot(req.slot));
                free_slots.push(req.slot);
                continue;
            }
            i += 1;
        }

        if active.is_empty() {
            continue;
        }

        let batch_size = active.len();
        debug!(
            "decode step: batch_size={}, pending={}, free={}",
            batch_size,
            pending.len(),
            free_slots.len()
        );
        let mut slots: Vec<usize> = Vec::with_capacity(batch_size);
        let mut input_ids: Vec<u32> = Vec::with_capacity(batch_size);
        let mut temperatures: Vec<f64> = Vec::with_capacity(batch_size);
        let mut samplers: Vec<Sampler> = Vec::with_capacity(batch_size);

        for req in active.iter_mut() {
            slots.push(req.slot);
            input_ids.push(req.pending_input);
            temperatures.push(req.temperature);
            samplers.push(std::mem::replace(&mut req.sampler, Sampler::Argmax));
        }

        let (results, returned_samplers) = model.submit(move |llama_model| {
            let input = Tensor::<B, 2, Int>::from_data(
                TensorData::new(input_ids, Shape::new([batch_size, 1])),
                &llama_model.device,
            );
            let logits = llama_model.decode_batch_with_slots(input, &slots);
            let [_batch, seq_len, _vocab_size] = logits.dims();

            let mut outputs: Vec<(u32, String)> = Vec::with_capacity(batch_size);
            for i in 0..batch_size {
                let mut next_token_logits = logits
                    .clone()
                    .slice([i..i + 1, seq_len - 1..seq_len])
                    .squeeze_dim(1);

                if temperatures[i] > 0.0 {
                    next_token_logits =
                        temperature_scaled_softmax(next_token_logits, temperatures[i]);
                }

                let next_token = samplers[i]
                    .sample(next_token_logits)
                    .squeeze_dim::<1>(0);
                let token_id = next_token
                    .clone()
                    .into_data()
                    .as_slice::<B::IntElem>()
                    .unwrap()[0]
                    .elem::<u32>();
                let token_text = llama_model.tokenizer.decode(vec![token_id]);
                outputs.push((token_id, token_text));
            }
            (outputs, samplers)
        });

        for (req, sampler) in active.iter_mut().zip(returned_samplers.into_iter()) {
            req.sampler = sampler;
        }

        let mut to_remove: Vec<usize> = Vec::new();
        for (i, (token_id, token_text)) in results.into_iter().enumerate() {
            let req = &mut active[i];

            if req.cancel_flag.load(Ordering::Relaxed) {
                debug!("request cancelled during decode: id={}", req.id);
                let _ = req
                    .reply
                    .send(BatcherEvent::Error("Generation cancelled".to_string()));
                to_remove.push(i);
                continue;
            }

            let is_stop = stop_ids.contains(&token_id);
            if is_stop {
                debug!("stop token reached: id={}", req.id);
                let _ = req.reply.send(BatcherEvent::Done);
                to_remove.push(i);
                continue;
            }

            let index = req.generated;
            let token_output = TokenOutput {
                token: token_text,
                token_id,
                index,
            };

            if req.reply.send(BatcherEvent::Token(token_output)).is_err() {
                warn!("emit failed: id={}", req.id);
                to_remove.push(i);
                continue;
            }

            req.generated += 1;
            if req.generated >= req.max_tokens {
                debug!("request completed: id={}", req.id);
                let _ = req.reply.send(BatcherEvent::Done);
                to_remove.push(i);
            } else {
                req.pending_input = token_id;
            }
        }

        for idx in to_remove.into_iter().rev() {
            let req = active.swap_remove(idx);
            model.submit(move |m| m.reset_cache_slot(req.slot));
            free_slots.push(req.slot);
        }
    }
}

/// Original streaming handler - blocks the model for entire generation.
///
/// This handler processes the entire generation in a single `model.submit()` call,
/// which means the model is locked for the full duration. This prevents concurrent
/// request processing.
///
/// For better concurrency, use `concurrent_streaming_handler` instead.
pub fn streaming_handler<B: Backend, T: Tokenizer + 'static>(
    In(request): In<GenerateRequest>,
    model: ModelAccessor<Llama<B, T>>,
    cancel: CancelToken,
    output: OutStream<TokenOutput>,
) -> Result<(), String> {
    let mut sampler = if request.temperature > 0.0 {
        Sampler::TopP(TopP::new(request.top_p, request.seed))
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
                    // Check for cancellation
                    if cancel.is_cancelled() {
                        return Err("Generation cancelled".to_string());
                    }

                    // Emit the token
                    let token_output = TokenOutput {
                        token: token_text.to_string(),
                        token_id,
                        index,
                    };

                    output
                        .emit(token_output)
                        .map_err(|e| format!("Failed to emit token: {}", e.source))?;

                    Ok(true) // Continue generation
                },
            )
            .map(|_| ()) // Convert usize to ()
            .map_err(|e| format!("Generation failed: {}", e))
    })
}

/// Concurrent streaming handler - generates one token at a time for better concurrency.
///
/// This handler uses iterative token generation, releasing the model between each token.
/// This allows multiple requests to interleave their generation, significantly improving
/// concurrency and responsiveness under load.
///
/// **Recommended for production use** when handling multiple concurrent requests.
///
/// # Example
/// ```ignore
/// let inference = InferenceBuilder::<Backend>::new()
///     .with_model(llama)
///     .build(concurrent_streaming_handler);
///
/// // Multiple requests can now interleave at token boundaries
/// let job1 = inference.infer(req1).spawn();
/// let job2 = inference.infer(req2).spawn();
/// // job1 and job2 will interleave: tok1, tok2, tok1, tok2, ...
/// ```
pub fn concurrent_streaming_handler<B: Backend, T: Tokenizer + 'static>(
    In(request): In<GenerateRequest>,
    model: ModelAccessor<Llama<B, T>>,
    cancel: CancelToken,
    output: OutStream<TokenOutput>,
) -> Result<(), String> {
    // Create the generation state outside of model.submit
    // This only needs the model briefly to tokenize and setup
    let prompt = request.prompt.clone();
    let max_tokens = request.max_tokens;
    let temperature = request.temperature;
    let top_p = request.top_p;
    let seed = request.seed;

    let mut state =
        model.submit(move |llama_model| llama_model.create_generation_state(&prompt, max_tokens));

    let mut sampler = if temperature > 0.0 {
        Sampler::TopP(TopP::new(top_p, seed))
    } else {
        Sampler::Argmax
    };

    // Generate tokens one at a time, releasing the model between iterations
    loop {
        // Check for cancellation before each token
        if cancel.is_cancelled() {
            return Err("Generation cancelled".to_string());
        }

        // Generate a single token - model is only locked for this one operation
        // We need to move state and sampler into the closure and get them back
        let (result, returned_state, returned_sampler) = model.submit(move |llama_model| {
            let result = llama_model.generate_single_token(&mut state, temperature, &mut sampler);
            (result, state, sampler)
        });

        let (token_id, token_text, is_complete) = result?;
        state = returned_state;
        sampler = returned_sampler;

        // Emit the token
        let token_output = TokenOutput {
            token: token_text,
            token_id,
            index: state.num_tokens_generated - 1,
        };

        output
            .emit(token_output)
            .map_err(|e| format!("Failed to emit token: {}", e.source))?;

        // Check if generation is complete
        if is_complete {
            break;
        }

        // Model is now free for other requests to use
    }

    Ok(())
}

/// Continuous batched streaming handler.
///
/// This handler uses a shared background batcher to combine requests into
/// dynamic batches. It supports mid-stream joins, cancellation, and token
/// streaming while keeping the GPU busy.
///
/// Requires a `ContinuousBatchScheduler<B, T>` extension to be attached to the
/// `InferenceBuilder`.
pub fn continuous_batched_streaming_handler<
    B: Backend + 'static,
    T: Tokenizer + Send + Sync + 'static,
>(
    In(request): In<GenerateRequest>,
    model: ModelAccessor<Llama<B, T>>,
    Extension(scheduler): Extension<ContinuousBatchScheduler<B, T>>,
    cancel: CancelToken,
    output: OutStream<TokenOutput>,
) -> Result<(), String> {
    let (reply_tx, reply_rx) = channel::unbounded();
    let cancel_flag = Arc::new(AtomicBool::new(false));

    scheduler.submit(model, request, reply_tx, cancel_flag.clone())?;

    loop {
        if cancel.is_cancelled() {
            cancel_flag.store(true, Ordering::Relaxed);
        }

        match reply_rx.recv_timeout(Duration::from_millis(50)) {
            Ok(BatcherEvent::Token(token)) => {
                output
                    .emit(token)
                    .map_err(|e| format!("Failed to emit token: {}", e.source))?;
            }
            Ok(BatcherEvent::Done) => break,
            Ok(BatcherEvent::Error(err)) => return Err(err),
            Err(channel::RecvTimeoutError::Timeout) => continue,
            Err(_) => break,
        }
    }

    Ok(())
}

/// Helper function for batched generation - processes multiple prompts in parallel.
///
/// This is a utility function that demonstrates batched inference where multiple sequences
/// are processed together in a single forward pass, maximizing GPU utilization.
///
/// **Note**: This is a standalone helper, not an inference handler. For production use,
/// integrate batched generation into your application logic or build a custom batching layer.
///
/// # Performance Benefits
/// - **3-5x throughput improvement** over sequential processing
/// - Maximizes GPU utilization by processing multiple sequences simultaneously
/// - Reduces per-token overhead through batching
///
/// # Example
/// ```ignore
/// let prompts = vec!["prompt1", "prompt2", "prompt3"];
/// let results = generate_batch(
///     &mut llama,
///     &prompts,
///     50,      // max_tokens
///     0.7,     // temperature
///     0.9,     // top_p
///     42,      // seed
/// )?;
///
/// for (i, tokens) in results.iter().enumerate() {
///     println!("Sequence {}: {}", i, tokens.join(""));
/// }
/// ```
pub fn generate_batch<B: Backend, T: Tokenizer>(
    llama: &mut Llama<B, T>,
    prompts: &[&str],
    max_tokens: usize,
    temperature: f64,
    top_p: f64,
    seed: u64,
) -> Result<Vec<Vec<String>>, String> {
    let batch_size = prompts.len();
    if batch_size == 0 {
        return Ok(vec![]);
    }

    // Create batched generation state
    let mut state = llama.create_batched_generation_state(prompts, max_tokens);

    // Create samplers for each sequence
    let mut samplers: Vec<_> = (0..batch_size)
        .map(|i| {
            if temperature > 0.0 {
                Sampler::TopP(TopP::new(top_p, seed + i as u64))
            } else {
                Sampler::Argmax
            }
        })
        .collect();

    // Collect generated tokens for each sequence
    let mut results: Vec<Vec<String>> = vec![vec![]; batch_size];

    // Generate tokens for the entire batch
    while state.active.iter().any(|&a| a) {
        let batch_results =
            llama.generate_batched_single_token(&mut state, temperature, &mut samplers)?;

        for (i, (_token_id, token_text, _is_complete)) in batch_results.iter().enumerate() {
            if state.active[i] || state.current_steps[i] > 0 {
                results[i].push(token_text.clone());
            }
        }
    }

    Ok(results)
}
