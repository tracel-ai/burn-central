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

pub trait ModelExecutor<Op, Event, Error>: Send + 'static {
    fn execute(&mut self, op: Op) -> Result<Event, Error>;
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

    pub fn recv(&mut self) -> Option<O> {
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

/// Spawn a single-writer session actor with a dedicated model executor worker.
/// All request state lives in `app` and is mutated only on this thread,
/// while callers can submit/cancel concurrently via the returned handle.
pub fn spawn_session<A, X>(
    mut app: A,
    mut executor: X,
) -> SessionHandle<A::Input, A::Output, A::Error>
where
    A: InferenceApp,
    X: ModelExecutor<A::ModelOp, A::ModelEvent, A::Error>,
{
    let (tx, rx) = channel::unbounded::<Command<A::Input, A::Output, A::Error>>();
    let (op_tx, op_rx) = channel::unbounded::<A::ModelOp>();
    let (ev_tx, ev_rx) = channel::unbounded::<Result<A::ModelEvent, A::Error>>();

    let handle = SessionHandle {
        tx: tx.clone(),
        next_id: Arc::new(AtomicU64::new(1)),
    };

    std::thread::spawn(move || {
        while let Ok(op) = op_rx.recv() {
            let res = executor.execute(op);
            let _ = ev_tx.send(res);
        }
    });

    std::thread::spawn(move || {
        let mut outputs: HashMap<RequestId, Sender<A::Output>> = HashMap::new();
        let mut dones: HashMap<RequestId, Sender<Result<(), A::Error>>> = HashMap::new();

        loop {
            crossbeam::select! {
                recv(rx) -> msg => match msg {
                    Ok(Command::Submit { id, input, out, done }) => {
                        outputs.insert(id, out);
                        dones.insert(id, done);
                        let effects = app.on_submit(id, input);
                        process_effects_with_sender(effects, &op_tx, &mut outputs, &mut dones);
                    }
                    Ok(Command::Cancel { id }) => {
                        let effects = app.on_cancel(id);
                        process_effects_with_sender(effects, &op_tx, &mut outputs, &mut dones);
                    }
                    Ok(Command::Shutdown) | Err(_) => break,
                },
                recv(ev_rx) -> msg => match msg {
                    Ok(Ok(ev)) => {
                        let effects = app.on_model_event(ev);
                        process_effects_with_sender(effects, &op_tx, &mut outputs, &mut dones);
                    }
                    Ok(Err(err)) => {
                        let effects = app.on_model_error(err);
                        process_effects_with_sender(effects, &op_tx, &mut outputs, &mut dones);
                    }
                    Err(_) => break,
                }
            }
        }
    });

    handle
}

fn process_effects_with_sender<O, Op, E>(
    effects: Vec<Effect<O, Op, E>>,
    op_tx: &Sender<Op>,
    outputs: &mut HashMap<RequestId, Sender<O>>,
    dones: &mut HashMap<RequestId, Sender<Result<(), E>>>,
) {
    let mut queue: VecDeque<Effect<O, Op, E>> = effects.into();

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
            Effect::RunModel(op) => {
                let _ = op_tx.send(op);
            }
        }
    }
}
