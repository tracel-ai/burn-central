use burn::nn::{Linear, LinearConfig};
use burn::prelude::Backend;
use burn::tensor::{ElementConversion, Shape, Tensor, TensorData, activation};
use inference_actor_demo::runtime::{Effect, InferenceApp, RequestId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Deserialize)]
pub struct GenerateRequest {
    pub steps: usize,
    pub value: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct StepOut {
    pub id: RequestId,
    pub index: usize,
    pub sample: f32,
}

pub const INPUT_DIM: usize = 256;
pub const HIDDEN_DIM: usize = 512;
pub const OUTPUT_DIM: usize = 128;

#[derive(Debug)]
struct RnnModel<B: Backend> {
    in_linear: Linear<B>,
    hid_linear: Linear<B>,
    out_linear: Linear<B>,
}

impl<B: Backend> RnnModel<B> {
    fn new(device: &B::Device) -> Self {
        let in_linear = LinearConfig::new(INPUT_DIM, HIDDEN_DIM).init(device);
        let hid_linear = LinearConfig::new(HIDDEN_DIM, HIDDEN_DIM).init(device);
        let out_linear = LinearConfig::new(HIDDEN_DIM, OUTPUT_DIM).init(device);
        Self {
            in_linear,
            hid_linear,
            out_linear,
        }
    }

    fn step(&self, input: Tensor<B, 2>, hidden: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let input_proj = self.in_linear.forward(input);
        let hidden_proj = self.hid_linear.forward(hidden);
        let next_hidden = activation::tanh(input_proj + hidden_proj);
        let output = self.out_linear.forward(next_hidden.clone());
        (next_hidden, output)
    }
}

#[derive(Debug)]
struct RequestState<B: Backend> {
    remaining: usize,
    generated: usize,
    input: Vec<B::FloatElem>,
    hidden: Vec<B::FloatElem>,
}

#[derive(Debug)]
pub struct RnnApp<B: Backend> {
    model: RnnModel<B>,
    device: B::Device,
    requests: HashMap<RequestId, RequestState<B>>,
    active: Vec<RequestId>,
}

impl<B: Backend> RnnApp<B> {
    pub fn new(device: B::Device) -> Self {
        Self {
            model: RnnModel::new(&device),
            device,
            requests: HashMap::new(),
            active: Vec::new(),
        }
    }
}

#[derive(Debug)]
pub enum ModelOp {
    Step,
}

#[derive(Debug)]
pub enum ModelEvent {
    Batch(Vec<(RequestId, StepOut, bool)>),
}

impl<B> InferenceApp for RnnApp<B>
where
    B: Backend,
{
    type Input = GenerateRequest;
    type Output = StepOut;
    type ModelOp = ModelOp;
    type ModelEvent = ModelEvent;
    type Error = String;

    fn on_submit(
        &mut self,
        id: RequestId,
        input: Self::Input,
    ) -> Vec<Effect<Self::Output, Self::ModelOp, Self::Error>> {
        let input_value = input.value.elem::<B::FloatElem>();
        let input_vec = vec![input_value; INPUT_DIM];
        let hidden = vec![0.0f32.elem::<B::FloatElem>(); HIDDEN_DIM];
        self.requests.insert(
            id,
            RequestState {
                remaining: input.steps,
                generated: 0,
                input: input_vec,
                hidden,
            },
        );
        self.active.push(id);
        vec![Effect::RunModel(ModelOp::Step)]
    }

    fn on_cancel(
        &mut self,
        id: RequestId,
    ) -> Vec<Effect<Self::Output, Self::ModelOp, Self::Error>> {
        self.requests.remove(&id);
        self.active.retain(|rid| *rid != id);
        vec![Effect::Finish { id, result: Ok(()) }]
    }

    fn on_model_event(
        &mut self,
        event: Self::ModelEvent,
    ) -> Vec<Effect<Self::Output, Self::ModelOp, Self::Error>> {
        match event {
            ModelEvent::Batch(items) => {
                let mut effects = Vec::new();
                for (id, out, done) in items {
                    effects.push(Effect::Emit { id, item: out });
                    if done {
                        effects.push(Effect::Finish { id, result: Ok(()) });
                    }
                }
                if !self.active.is_empty() {
                    effects.push(Effect::RunModel(ModelOp::Step));
                }
                effects
            }
        }
    }

    fn on_model_error(
        &mut self,
        error: Self::Error,
    ) -> Vec<Effect<Self::Output, Self::ModelOp, Self::Error>> {
        let mut effects = Vec::new();
        for id in self.active.drain(..) {
            effects.push(Effect::Finish {
                id,
                result: Err(error.clone()),
            });
        }
        effects
    }

    fn execute(&mut self, op: Self::ModelOp) -> Result<Self::ModelEvent, Self::Error> {
        match op {
            ModelOp::Step => {
                if self.active.is_empty() {
                    return Ok(ModelEvent::Batch(Vec::new()));
                }
                let mut batch = Vec::new();
                let mut to_remove = Vec::new();

                let batch_size = self.active.len();
                let mut input_data: Vec<B::FloatElem> = Vec::with_capacity(batch_size * INPUT_DIM);
                let mut hidden_data: Vec<B::FloatElem> =
                    Vec::with_capacity(batch_size * HIDDEN_DIM);

                for id in &self.active {
                    let state = self
                        .requests
                        .get(id)
                        .ok_or_else(|| "missing request state".to_string())?;
                    input_data.extend_from_slice(&state.input);
                    hidden_data.extend_from_slice(&state.hidden);
                }

                let input = Tensor::<B, 2>::from_data(
                    TensorData::new(input_data, Shape::new([batch_size, INPUT_DIM])),
                    &self.device,
                );
                let hidden = Tensor::<B, 2>::from_data(
                    TensorData::new(hidden_data, Shape::new([batch_size, HIDDEN_DIM])),
                    &self.device,
                );
                let (next_hidden, output) = self.model.step(input, hidden);
                let output_data = output.into_data();
                let output_slice = output_data
                    .as_slice::<B::FloatElem>()
                    .map_err(|_| "output slice missing".to_string())?;
                let hidden_data = next_hidden.into_data();
                let hidden_slice = hidden_data
                    .as_slice::<B::FloatElem>()
                    .map_err(|_| "hidden slice missing".to_string())?;

                for (pos, id) in self.active.iter().enumerate() {
                    let state = self
                        .requests
                        .get_mut(id)
                        .ok_or_else(|| "missing request state".to_string())?;

                    if state.remaining > 0 {
                        state.remaining -= 1;
                    }
                    state.generated += 1;

                    let done = state.remaining == 0;
                    if done {
                        to_remove.push(*id);
                    }

                    let sample = output_slice[pos * OUTPUT_DIM].elem::<f32>();
                    let index = state.generated - 1;
                    let hidden_start = pos * HIDDEN_DIM;
                    let hidden_end = hidden_start + HIDDEN_DIM;
                    state
                        .hidden
                        .copy_from_slice(&hidden_slice[hidden_start..hidden_end]);

                    batch.push((
                        *id,
                        StepOut {
                            id: *id,
                            index,
                            sample,
                        },
                        done,
                    ));
                }

                for id in to_remove {
                    self.requests.remove(&id);
                    self.active.retain(|rid| *rid != id);
                }

                Ok(ModelEvent::Batch(batch))
            }
        }
    }
}
