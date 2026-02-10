use inference_actor_demo::{Action, Actions, ModelExecutor, RequestId, app, spawn_session};

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

fn main() {
    let app = app(
        (),
        |_state: &mut (), id: RequestId, input: Input| {
            Actions::single(Action::run_model(ModelOp { id, input }))
        },
        |_state: &mut (), id: RequestId| Actions::single(Action::finish_ok(id)),
        |_state: &mut (), event: ModelEvent| {
            let mut actions = Actions::new();
            actions.emit(event.id, event.output);
            actions.finish_ok(event.id);
            actions
        },
    );

    let session = spawn_session(app, DirectExecutor::new());

    let job = session.submit(Input { value: 3.0 });
    for out in job.stream.iter() {
        println!("output: {:?}", out);
    }
    job.join().unwrap();

    session.shutdown();
}
