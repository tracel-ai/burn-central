use crossbeam::channel::{self, Receiver, Sender};
use std::collections::{HashMap, VecDeque};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

pub type RequestId = u64;

#[derive(Debug)]
pub enum Effect<O, Op, E> {
    Emit {
        id: RequestId,
        item: O,
    },
    Finish {
        id: RequestId,
        result: Result<(), E>,
    },
    RunModel(Op),
}

pub trait InferenceApp: Send + 'static {
    type Input: Send + 'static;
    type Output: Send + 'static;
    type ModelOp: Send + 'static;
    type ModelEvent: Send + 'static;
    type Error: Send + 'static;

    fn on_submit(
        &mut self,
        id: RequestId,
        input: Self::Input,
    ) -> Vec<Effect<Self::Output, Self::ModelOp, Self::Error>>;

    fn on_cancel(&mut self, id: RequestId)
    -> Vec<Effect<Self::Output, Self::ModelOp, Self::Error>>;

    fn on_model_event(
        &mut self,
        event: Self::ModelEvent,
    ) -> Vec<Effect<Self::Output, Self::ModelOp, Self::Error>>;

    fn on_model_error(
        &mut self,
        error: Self::Error,
    ) -> Vec<Effect<Self::Output, Self::ModelOp, Self::Error>> {
        let _ = error;
        Vec::new()
    }

    fn execute(&mut self, op: Self::ModelOp) -> Result<Self::ModelEvent, Self::Error>;
}

enum Command<I, M, E> {
    Submit {
        id: RequestId,
        input: I,
        out: Sender<M>,
        done: Sender<Result<(), E>>,
    },
    Cancel {
        id: RequestId,
    },
    Shutdown,
}

pub struct Job<I, O, E> {
    pub id: RequestId,
    pub stream: Receiver<O>,
    done: Receiver<Result<(), E>>,
    cancel: Sender<Command<I, O, E>>,
}

impl<I, O, E> Job<I, O, E> {
    pub fn cancel(&self) {
        let _ = self.cancel.send(Command::Cancel { id: self.id });
    }

    /// Blocking receive for sync contexts.
    pub fn recv_blocking(&mut self) -> Option<O> {
        self.stream.recv().ok()
    }

    pub fn join(self) -> Result<(), E> {
        self.done.recv().unwrap_or_else(|_| Ok(()))
    }
}

#[derive(Clone)]
pub struct SessionHandle<I, O, E> {
    tx: Sender<Command<I, O, E>>,
    next_id: Arc<AtomicU64>,
}

impl<I: Send + 'static, O: Send + 'static, E: Send + 'static> SessionHandle<I, O, E> {
    pub fn submit(&self, input: I) -> Job<I, O, E> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (out_tx, out_rx) = channel::unbounded();
        let (done_tx, done_rx) = channel::bounded(1);

        let _ = self.tx.send(Command::Submit {
            id,
            input,
            out: out_tx,
            done: done_tx,
        });

        Job {
            id,
            stream: out_rx,
            done: done_rx,
            cancel: self.tx.clone(),
        }
    }

    pub fn shutdown(&self) {
        let _ = self.tx.send(Command::Shutdown);
    }
}

/// Spawn a single-writer session actor.
/// All request state lives in `app` and is mutated only on this thread,
/// while callers can submit/cancel concurrently via the returned handle.
pub fn spawn_session<A>(mut app: A) -> SessionHandle<A::Input, A::Output, A::Error>
where
    A: InferenceApp,
{
    let (tx, rx) = channel::unbounded::<Command<A::Input, A::Output, A::Error>>();
    let handle = SessionHandle {
        tx: tx.clone(),
        next_id: Arc::new(AtomicU64::new(1)),
    };

    std::thread::spawn(move || {
        let mut outputs: HashMap<RequestId, Sender<A::Output>> = HashMap::new();
        let mut dones: HashMap<RequestId, Sender<Result<(), A::Error>>> = HashMap::new();

        while let Ok(cmd) = rx.recv() {
            let effects = match cmd {
                Command::Submit {
                    id,
                    input,
                    out,
                    done,
                } => {
                    outputs.insert(id, out);
                    dones.insert(id, done);
                    app.on_submit(id, input)
                }
                Command::Cancel { id } => app.on_cancel(id),
                Command::Shutdown => break,
            };

            process_effects(&mut app, effects, &mut outputs, &mut dones);
        }
    });

    handle
}

fn process_effects<A: InferenceApp>(
    app: &mut A,
    effects: Vec<Effect<A::Output, A::ModelOp, A::Error>>,
    outputs: &mut HashMap<RequestId, Sender<A::Output>>,
    dones: &mut HashMap<RequestId, Sender<Result<(), A::Error>>>,
) {
    let mut queue: VecDeque<Effect<A::Output, A::ModelOp, A::Error>> = effects.into();

    while let Some(effect) = queue.pop_front() {
        match effect {
            Effect::Emit { id, item } => {
                if let Some(out) = outputs.get(&id) {
                    let _ = out.send(item);
                }
            }
            Effect::Finish { id, result } => {
                outputs.remove(&id);
                if let Some(done) = dones.remove(&id) {
                    let _ = done.send(result);
                }
            }
            Effect::RunModel(op) => match app.execute(op) {
                Ok(ev) => queue.extend(app.on_model_event(ev)),
                Err(err) => queue.extend(app.on_model_error(err)),
            },
        }
    }
}
