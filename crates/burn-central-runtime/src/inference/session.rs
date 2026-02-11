use crossbeam::channel::{self, Receiver, Sender};
use std::collections::{HashMap, VecDeque};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

pub type RequestId = u64;

#[derive(Debug)]
pub enum Action<O, Op, E, K = RequestId> {
    Emit {
        id: RequestId,
        item: O,
    },
    Finish {
        id: RequestId,
        result: Result<(), E>,
    },
    RunModel {
        key: K,
        op: Op,
    },
}

#[derive(Debug, Default)]
pub struct Actions<O, Op, E, K = RequestId>(Vec<Action<O, Op, E, K>>);

impl<O, Op, E, K> Actions<O, Op, E, K> {
    pub fn new() -> Self {
        Self(Vec::new())
    }

    pub fn single(action: Action<O, Op, E, K>) -> Self {
        Self(vec![action])
    }

    pub fn push(&mut self, action: Action<O, Op, E, K>) {
        self.0.push(action);
    }

    pub fn respond(&mut self, id: RequestId, item: O) -> &mut Self {
        self.0.push(Action::emit(id, item));
        self.0.push(Action::finish_ok(id));
        self
    }

    pub fn emit(&mut self, id: RequestId, item: O) {
        self.0.push(Action::emit(id, item));
    }

    pub fn finish_ok(&mut self, id: RequestId) {
        self.0.push(Action::finish_ok(id));
    }

    pub fn finish(&mut self, id: RequestId, result: Result<(), E>) {
        self.0.push(Action::finish(id, result));
    }

    pub fn run_model(&mut self, key: K, op: Op) {
        self.0.push(Action::run_model(key, op));
    }

    pub fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = Action<O, Op, E, K>>,
    {
        self.0.extend(iter);
    }
}

impl<O, Op, E, K> IntoIterator for Actions<O, Op, E, K> {
    type Item = Action<O, Op, E, K>;
    type IntoIter = std::vec::IntoIter<Action<O, Op, E, K>>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<O, Op, E, K> From<Action<O, Op, E, K>> for Actions<O, Op, E, K> {
    fn from(action: Action<O, Op, E, K>) -> Self {
        Self::single(action)
    }
}

impl<O, Op, E, K> From<Vec<Action<O, Op, E, K>>> for Actions<O, Op, E, K> {
    fn from(actions: Vec<Action<O, Op, E, K>>) -> Self {
        Self(actions)
    }
}

impl<O, Op, E, K> FromIterator<Action<O, Op, E, K>> for Actions<O, Op, E, K> {
    fn from_iter<I: IntoIterator<Item = Action<O, Op, E, K>>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl<O, Op, E, K> Action<O, Op, E, K> {
    pub fn emit(id: RequestId, item: O) -> Self {
        Self::Emit { id, item }
    }

    pub fn finish(id: RequestId, result: Result<(), E>) -> Self {
        Self::Finish { id, result }
    }

    pub fn finish_ok(id: RequestId) -> Self {
        Self::Finish { id, result: Ok(()) }
    }

    pub fn run_model(key: K, op: Op) -> Self {
        Self::RunModel { key, op }
    }
}

pub trait ModelExecutor<Op, Event, Error>: Send + 'static {
    fn execute(&mut self, op: Op) -> Result<Event, Error>;
}

impl<Op, Event, Error, F> ModelExecutor<Op, Event, Error> for F
where
    F: FnMut(Op) -> Result<Event, Error> + Send + 'static,
{
    fn execute(&mut self, op: Op) -> Result<Event, Error> {
        self(op)
    }
}

pub trait InferenceApp: Send + 'static {
    type Input: Send + 'static;
    type Output: Send + 'static;
    type Error: Send + 'static;
    type ModelOp: Send + 'static;
    type ModelEvent: Send + 'static;
    type ModelError: Send + 'static;
    type Key: Send + 'static;

    fn on_submit(
        &mut self,
        id: RequestId,
        input: Self::Input,
    ) -> Actions<Self::Output, Self::ModelOp, Self::Error, Self::Key>;

    fn on_cancel(
        &mut self,
        id: RequestId,
    ) -> Actions<Self::Output, Self::ModelOp, Self::Error, Self::Key>;

    fn on_model_event(
        &mut self,
        key: Self::Key,
        event: Self::ModelEvent,
    ) -> Actions<Self::Output, Self::ModelOp, Self::Error, Self::Key>;

    fn on_model_error(
        &mut self,
        key: Self::Key,
        error: Self::ModelError,
    ) -> Actions<Self::Output, Self::ModelOp, Self::Error, Self::Key>;
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
pub fn spawn_session<A, X>(mut app: A, mut model: X) -> SessionHandle<A::Input, A::Output, A::Error>
where
    A: InferenceApp,
    X: ModelExecutor<A::ModelOp, A::ModelEvent, A::ModelError>,
{
    let (tx, rx) = channel::unbounded::<Command<A::Input, A::Output, A::Error>>();
    let (op_tx, op_rx) = channel::unbounded::<(A::Key, A::ModelOp)>();
    let (ev_tx, ev_rx) = channel::unbounded::<(A::Key, Result<A::ModelEvent, A::ModelError>)>();

    let handle = SessionHandle {
        tx: tx.clone(),
        next_id: Arc::new(AtomicU64::new(1)),
    };

    std::thread::spawn(move || {
        while let Ok((key, op)) = op_rx.recv() {
            let res = model.execute(op);
            let _ = ev_tx.send((key, res));
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
                    Ok((key, Ok(ev))) => {
                        let effects = app.on_model_event(key, ev);
                        process_effects_with_sender(effects, &op_tx, &mut outputs, &mut dones);
                    }
                    Ok((key, Err(err))) => {
                        let effects = app.on_model_error(key, err);
                        process_effects_with_sender(effects, &op_tx, &mut outputs, &mut dones);
                    }
                    Err(_) => break,
                }
            }
        }
    });

    handle
}

fn process_effects_with_sender<O, Op, E, K>(
    actions: Actions<O, Op, E, K>,
    op_tx: &Sender<(K, Op)>,
    outputs: &mut HashMap<RequestId, Sender<O>>,
    dones: &mut HashMap<RequestId, Sender<Result<(), E>>>,
) {
    let mut queue: VecDeque<Action<O, Op, E, K>> = actions.into_iter().collect();

    while let Some(action) = queue.pop_front() {
        match action {
            Action::Emit { id, item } => {
                if let Some(out) = outputs.get(&id) {
                    let _ = out.send(item);
                }
            }
            Action::Finish { id, result } => {
                outputs.remove(&id);
                if let Some(done) = dones.remove(&id) {
                    let _ = done.send(result);
                }
            }
            Action::RunModel { key, op } => {
                let _ = op_tx.send((key, op));
            }
        }
    }
}
