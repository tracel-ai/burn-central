use inference_actor_demo::runtime::{
    Effect, InferenceApp, ModelExecutor, RequestId, spawn_session,
};

#[derive(Debug, Clone)]
struct Input {
    value: f32,
}

#[derive(Debug, Clone)]
struct Output {
    value: f32,
}

#[derive(Debug, Clone)]
struct ModelOp {
    id: RequestId,
    input: Input,
}

#[derive(Debug, Clone)]
struct ModelEvent {
    id: RequestId,
    output: Output,
}

#[derive(Debug)]
struct SimpleModel;

impl SimpleModel {
    fn process(&self, input: Input) -> Output {
        // Synchronous, single-step compute (no batching).
        Output {
            value: (input.value * input.value).sin(),
        }
    }
}

#[derive(Debug)]
struct DirectApp;

impl DirectApp {
    fn new() -> Self {
        Self
    }
}

struct DirectExecutor {
    model: SimpleModel,
}

impl DirectExecutor {
    fn new() -> Self {
        Self { model: SimpleModel }
    }
}

impl ModelExecutor<ModelOp, ModelEvent, String> for DirectExecutor {
    fn execute(&mut self, op: ModelOp) -> Result<ModelEvent, String> {
        let output = self.model.process(op.input);
        Ok(ModelEvent { id: op.id, output })
    }
}

impl InferenceApp for DirectApp {
    type Input = Input;
    type Output = Output;
    type ModelOp = ModelOp;
    type ModelEvent = ModelEvent;
    type Error = String;

    fn on_submit(
        &mut self,
        id: RequestId,
        input: Self::Input,
    ) -> Vec<Effect<Self::Output, Self::ModelOp, Self::Error>> {
        vec![Effect::RunModel(ModelOp { id, input })]
    }

    fn on_cancel(
        &mut self,
        id: RequestId,
    ) -> Vec<Effect<Self::Output, Self::ModelOp, Self::Error>> {
        vec![Effect::Finish { id, result: Ok(()) }]
    }

    fn on_model_event(
        &mut self,
        event: Self::ModelEvent,
    ) -> Vec<Effect<Self::Output, Self::ModelOp, Self::Error>> {
        vec![
            Effect::Emit {
                id: event.id,
                item: event.output,
            },
            Effect::Finish {
                id: event.id,
                result: Ok(()),
            },
        ]
    }
}

fn main() {
    let session = spawn_session(DirectApp::new(), DirectExecutor::new());

    let job = session.submit(Input { value: 3.0 });
    for out in job.stream.iter() {
        println!("output: {:?}", out);
    }
    job.join().unwrap();

    session.shutdown();
}
