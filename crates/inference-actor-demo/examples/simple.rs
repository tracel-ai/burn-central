use inference_actor_demo::runtime::{
    Effect, InferenceApp, ModelExecutor, RequestId, spawn_session,
};
use std::collections::HashMap;
use std::thread;

#[derive(Debug, Clone)]
struct TickOut {
    id: RequestId,
    index: usize,
}

#[derive(Debug)]
struct CounterApp {
    remaining: HashMap<RequestId, usize>,
    active: Vec<RequestId>,
    step_in_flight: bool,
}

impl CounterApp {
    fn new() -> Self {
        Self {
            remaining: HashMap::new(),
            active: Vec::new(),
            step_in_flight: false,
        }
    }

    fn schedule_step(&mut self) -> Vec<Effect<TickOut, ModelStepOp, String>> {
        if self.step_in_flight || self.active.is_empty() {
            return Vec::new();
        }

        let mut batch = Vec::with_capacity(self.active.len());
        for id in &self.active {
            if let Some(remaining) = self.remaining.get(id) {
                batch.push(StepInput {
                    id: *id,
                    remaining: *remaining,
                });
            }
        }

        if batch.is_empty() {
            return Vec::new();
        }

        self.step_in_flight = true;
        vec![Effect::RunModel(ModelStepOp { batch })]
    }
}

#[derive(Debug)]
struct StepInput {
    id: RequestId,
    remaining: usize,
}

#[derive(Debug)]
struct ModelStepOp {
    batch: Vec<StepInput>,
}

#[derive(Debug)]
struct StepResult {
    id: RequestId,
    remaining: usize,
    out: TickOut,
    done: bool,
}

#[derive(Debug)]
struct ModelStepResult {
    items: Vec<StepResult>,
}

struct CounterExecutor;

impl ModelExecutor<ModelStepOp, ModelStepResult, String> for CounterExecutor {
    fn execute(&mut self, op: ModelStepOp) -> Result<ModelStepResult, String> {
        let mut items = Vec::with_capacity(op.batch.len());
        for input in op.batch {
            let mut remaining = input.remaining;
            let index = remaining;
            if remaining > 0 {
                remaining -= 1;
            }
            let done = remaining == 0;
            let out = TickOut {
                id: input.id,
                index,
            };
            items.push(StepResult {
                id: input.id,
                remaining,
                out,
                done,
            });
        }
        Ok(ModelStepResult { items })
    }
}

impl InferenceApp for CounterApp {
    type Input = usize;
    type Output = TickOut;
    type ModelOp = ModelStepOp;
    type ModelEvent = ModelStepResult;
    type Error = String;

    fn on_submit(
        &mut self,
        id: RequestId,
        input: Self::Input,
    ) -> Vec<Effect<Self::Output, Self::ModelOp, Self::Error>> {
        self.remaining.insert(id, input);
        self.active.push(id);
        self.schedule_step()
    }

    fn on_cancel(
        &mut self,
        id: RequestId,
    ) -> Vec<Effect<Self::Output, Self::ModelOp, Self::Error>> {
        self.remaining.remove(&id);
        self.active.retain(|rid| *rid != id);
        vec![Effect::Finish { id, result: Ok(()) }]
    }

    fn on_model_event(
        &mut self,
        event: Self::ModelEvent,
    ) -> Vec<Effect<Self::Output, Self::ModelOp, Self::Error>> {
        let mut effects = Vec::new();
        self.step_in_flight = false;

        for item in event.items {
            if item.done {
                self.remaining.remove(&item.id);
            } else {
                self.remaining.insert(item.id, item.remaining);
            }
            effects.push(Effect::Emit {
                id: item.id,
                item: item.out,
            });
            if item.done {
                effects.push(Effect::Finish {
                    id: item.id,
                    result: Ok(()),
                });
            }
        }

        self.active.retain(|id| {
            self.remaining
                .get(id)
                .map(|remaining| *remaining > 0)
                .unwrap_or(false)
        });

        effects.extend(self.schedule_step());
        effects
    }

    fn on_model_error(
        &mut self,
        err: Self::Error,
    ) -> Vec<Effect<Self::Output, Self::ModelOp, Self::Error>> {
        let mut effects = Vec::new();
        self.step_in_flight = false;
        for id in self.active.drain(..) {
            effects.push(Effect::Finish {
                id,
                result: Err(err.clone()),
            });
        }
        effects
    }
}

fn main() {
    let session = spawn_session(CounterApp::new(), CounterExecutor);

    let s1 = session.clone();
    let t1 = thread::spawn(move || {
        let job = s1.submit(3);
        for out in job.stream.iter() {
            println!("job1: {:?}", out);
        }
        job.join().unwrap();
    });

    let s2 = session.clone();
    let t2 = thread::spawn(move || {
        let job = s2.submit(2);
        for out in job.stream.iter() {
            println!("job2: {:?}", out);
        }
        job.join().unwrap();
    });

    t1.join().unwrap();
    t2.join().unwrap();

    session.shutdown();
}
