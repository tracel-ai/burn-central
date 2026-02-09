use inference_actor_demo::runtime::{Effect, InferenceApp, RequestId, spawn_session};
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
}

impl CounterApp {
    fn new() -> Self {
        Self {
            remaining: HashMap::new(),
            active: Vec::new(),
        }
    }
}

#[derive(Debug)]
enum ModelOp {
    Step,
}

#[derive(Debug)]
enum ModelEvent {
    Batch(Vec<(RequestId, TickOut, bool)>),
}

impl InferenceApp for CounterApp {
    type Input = usize;
    type Output = TickOut;
    type ModelOp = ModelOp;
    type ModelEvent = ModelEvent;
    type Error = String;

    fn on_submit(
        &mut self,
        id: RequestId,
        input: Self::Input,
    ) -> Vec<Effect<Self::Output, Self::ModelOp, Self::Error>> {
        self.remaining.insert(id, input);
        self.active.push(id);
        vec![Effect::RunModel(ModelOp::Step)]
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
        err: Self::Error,
    ) -> Vec<Effect<Self::Output, Self::ModelOp, Self::Error>> {
        let mut effects = Vec::new();
        for id in self.active.drain(..) {
            effects.push(Effect::Finish {
                id,
                result: Err(err.clone()),
            });
        }
        effects
    }

    fn execute(&mut self, op: Self::ModelOp) -> Result<Self::ModelEvent, Self::Error> {
        match op {
            ModelOp::Step => {
                let mut batch = Vec::new();
                let mut to_remove = Vec::new();

                for id in &self.active {
                    let remaining = self
                        .remaining
                        .get_mut(id)
                        .ok_or_else(|| "missing request state".to_string())?;

                    let index = *remaining;
                    if *remaining > 0 {
                        *remaining -= 1;
                    }

                    let done = *remaining == 0;
                    if done {
                        to_remove.push(*id);
                    }

                    batch.push((*id, TickOut { id: *id, index }, done));
                }

                for id in to_remove {
                    self.remaining.remove(&id);
                    self.active.retain(|rid| *rid != id);
                }

                Ok(ModelEvent::Batch(batch))
            }
        }
    }
}

fn main() {
    let session = spawn_session(CounterApp::new());

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
