use burn::nn::{Linear, LinearConfig};
use burn::prelude::Backend;
use burn::tensor::{ElementConversion, Shape, Tensor, TensorData, activation};
use inference_actor_demo::JsonSession;
use inference_actor_demo::erased::InferenceSpec;
use inference_actor_demo::runtime::{
    Action, Actions, InferenceApp, ModelExecutor, RequestId, spawn_session,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GenerateRequest {
    pub steps: usize,
    pub value: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct StepOut {
    pub index: usize,
    pub sample: f32,
}

pub const INPUT_DIM: usize = 256;
pub const HIDDEN_DIM: usize = 512;
pub const OUTPUT_DIM: usize = 128;
pub const SPEC: InferenceSpec = InferenceSpec {
    name: "rnn_demo",
    input_schema: r#"{\"type\":\"object\",\"fields\":[{\"name\":\"steps\",\"type\":\"u64\"},{\"name\":\"value\",\"type\":\"f32\"}]}"#,
    output_schema: r#"{\"type\":\"object\",\"fields\":[{\"name\":\"id\",\"type\":\"u64\"},{\"name\":\"index\",\"type\":\"u64\"},{\"name\":\"sample\",\"type\":\"f32\"}]}"#,
    streaming: true,
};

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
    requests: HashMap<RequestId, RequestState<B>>,
    active: Vec<RequestId>,
    step_in_flight: bool,
}

impl<B: Backend> RnnApp<B> {
    pub fn new() -> Self {
        Self {
            requests: HashMap::new(),
            active: Vec::new(),
            step_in_flight: false,
        }
    }

    fn schedule_step(&mut self) -> Actions<StepOut, ModelStepOp<B>, String> {
        if self.step_in_flight || self.active.is_empty() {
            return Actions::new();
        }

        let batch_size = self.active.len();
        let mut input: Vec<B::FloatElem> = Vec::with_capacity(batch_size * INPUT_DIM);
        let mut hidden: Vec<B::FloatElem> = Vec::with_capacity(batch_size * HIDDEN_DIM);
        let mut ids: Vec<RequestId> = Vec::with_capacity(batch_size);

        for id in &self.active {
            let state = match self.requests.get(id) {
                Some(state) => state,
                None => continue,
            };
            ids.push(*id);
            input.extend_from_slice(&state.input);
            hidden.extend_from_slice(&state.hidden);
        }

        self.step_in_flight = true;
        Actions::single(Action::run_model(ModelStepOp {
            ids,
            input,
            hidden,
            batch_size,
        }))
    }
}

#[derive(Debug)]
pub struct ModelStepOp<B: Backend> {
    pub ids: Vec<RequestId>,
    pub input: Vec<B::FloatElem>,
    pub hidden: Vec<B::FloatElem>,
    pub batch_size: usize,
}

#[derive(Debug)]
pub struct ModelStepResult<B: Backend> {
    pub ids: Vec<RequestId>,
    pub output: Vec<B::FloatElem>,
    pub hidden: Vec<B::FloatElem>,
}

impl<B> InferenceApp for RnnApp<B>
where
    B: Backend,
{
    type Input = GenerateRequest;
    type Output = StepOut;
    type ModelOp = ModelStepOp<B>;
    type ModelEvent = ModelStepResult<B>;
    type Error = String;

    fn on_submit(
        &mut self,
        id: RequestId,
        input: Self::Input,
    ) -> Actions<Self::Output, Self::ModelOp, Self::Error> {
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
        self.schedule_step()
    }

    fn on_cancel(&mut self, id: RequestId) -> Actions<Self::Output, Self::ModelOp, Self::Error> {
        self.requests.remove(&id);
        self.active.retain(|rid| *rid != id);
        Actions::single(Action::finish_ok(id))
    }

    fn on_model_event(
        &mut self,
        event: Self::ModelEvent,
    ) -> Actions<Self::Output, Self::ModelOp, Self::Error> {
        let mut actions = Actions::new();
        self.step_in_flight = false;

        let ModelStepResult {
            ids,
            output,
            hidden,
        } = event;
        for (pos, id) in ids.iter().enumerate() {
            let state = match self.requests.get_mut(id) {
                Some(state) => state,
                None => continue,
            };

            if state.remaining > 0 {
                state.remaining -= 1;
            }
            state.generated += 1;

            let done = state.remaining == 0;
            let sample = output[pos * OUTPUT_DIM].elem::<f32>();
            let index = state.generated - 1;

            let hidden_start = pos * HIDDEN_DIM;
            let hidden_end = hidden_start + HIDDEN_DIM;
            state
                .hidden
                .copy_from_slice(&hidden[hidden_start..hidden_end]);

            actions.emit(*id, StepOut { index, sample });
            if done {
                actions.finish_ok(*id);
            }
        }

        self.active.retain(|id| {
            self.requests
                .get(id)
                .map(|state| state.remaining > 0)
                .unwrap_or(false)
        });

        actions.extend(self.schedule_step());
        actions
    }

    fn on_model_error(
        &mut self,
        error: Self::Error,
    ) -> Actions<Self::Output, Self::ModelOp, Self::Error> {
        let mut actions = Actions::new();
        self.step_in_flight = false;
        for id in self.active.drain(..) {
            actions.finish(id, Err(error.clone()));
        }
        actions
    }
}

pub struct RnnExecutor<B: Backend> {
    model: RnnModel<B>,
    device: B::Device,
}

impl<B: Backend> RnnExecutor<B> {
    pub fn new(device: B::Device) -> Self {
        Self {
            model: RnnModel::new(&device),
            device,
        }
    }
}

impl<B> ModelExecutor<ModelStepOp<B>, ModelStepResult<B>, String> for RnnExecutor<B>
where
    B: Backend,
{
    fn execute(&mut self, op: ModelStepOp<B>) -> Result<ModelStepResult<B>, String> {
        let ModelStepOp {
            ids,
            input,
            hidden,
            batch_size,
        } = op;

        if batch_size == 0 {
            return Ok(ModelStepResult {
                ids,
                output: Vec::new(),
                hidden: Vec::new(),
            });
        }

        let input = Tensor::<B, 2>::from_data(
            TensorData::new(input, Shape::new([batch_size, INPUT_DIM])),
            &self.device,
        );
        let hidden = Tensor::<B, 2>::from_data(
            TensorData::new(hidden, Shape::new([batch_size, HIDDEN_DIM])),
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

        Ok(ModelStepResult {
            ids,
            output: output_slice.to_vec(),
            hidden: hidden_slice.to_vec(),
        })
    }
}

pub fn build_session<B: Backend>(
    device: B::Device,
) -> JsonSession<GenerateRequest, StepOut, String> {
    let session = spawn_session(RnnApp::<B>::new(), RnnExecutor::<B>::new(device));
    JsonSession::new(session, &SPEC)
}
