use std::marker::PhantomData;

use crate::inference::{Actions, InferenceApp, ModelExecutor, RequestId};

fn default_on_model_error<State, Output, ModelOp, Error, Key>(
    _state: &mut State,
    _key: Key,
    _error: Error,
) -> Actions<Output, ModelOp, Error, Key> {
    Actions::new()
}

pub fn app<State, Input, Output, ModelOp, ModelEvent, Error, Key, Submit, Cancel, ModelEventFn>(
    state: State,
    on_submit: Submit,
    on_cancel: Cancel,
    on_model_event: ModelEventFn,
) -> impl InferenceApp<
    Input = Input,
    Output = Output,
    ModelOp = ModelOp,
    ModelEvent = ModelEvent,
    Error = Error,
    Key = Key,
>
where
    State: Send + 'static,
    Input: Send + 'static,
    Output: Send + 'static,
    ModelOp: Send + 'static,
    ModelEvent: Send + 'static,
    Error: Send + 'static,
    Key: Send + 'static,
    Submit: FnMut(&mut State, RequestId, Input) -> Actions<Output, ModelOp, Error, Key>
        + Send
        + 'static,
    Cancel: FnMut(&mut State, RequestId) -> Actions<Output, ModelOp, Error, Key> + Send + 'static,
    ModelEventFn:
        FnMut(&mut State, Key, ModelEvent) -> Actions<Output, ModelOp, Error, Key> + Send + 'static,
{
    app_with_error(
        state,
        on_submit,
        on_cancel,
        on_model_event,
        default_on_model_error::<State, Output, ModelOp, Error, Key>,
    )
}

pub fn app_with_error<
    State,
    Input,
    Output,
    ModelOp,
    ModelEvent,
    Error,
    Key,
    Submit,
    Cancel,
    ModelEventFn,
    ModelErrorFn,
>(
    state: State,
    on_submit: Submit,
    on_cancel: Cancel,
    on_model_event: ModelEventFn,
    on_model_error: ModelErrorFn,
) -> impl InferenceApp<
    Input = Input,
    Output = Output,
    ModelOp = ModelOp,
    ModelEvent = ModelEvent,
    Error = Error,
    Key = Key,
>
where
    State: Send + 'static,
    Input: Send + 'static,
    Output: Send + 'static,
    ModelOp: Send + 'static,
    ModelEvent: Send + 'static,
    Error: Send + 'static,
    Key: Send + 'static,
    Submit: FnMut(&mut State, RequestId, Input) -> Actions<Output, ModelOp, Error, Key>
        + Send
        + 'static,
    Cancel: FnMut(&mut State, RequestId) -> Actions<Output, ModelOp, Error, Key> + Send + 'static,
    ModelEventFn:
        FnMut(&mut State, Key, ModelEvent) -> Actions<Output, ModelOp, Error, Key> + Send + 'static,
    ModelErrorFn:
        FnMut(&mut State, Key, Error) -> Actions<Output, ModelOp, Error, Key> + Send + 'static,
{
    use std::marker::PhantomData;

    struct Instance<
        State,
        Input,
        Output,
        ModelOp,
        ModelEvent,
        Error,
        Key,
        Submit,
        Cancel,
        ModelEventFn,
        ModelErrorFn,
    > {
        state: State,
        on_submit: Submit,
        on_cancel: Cancel,
        on_model_event: ModelEventFn,
        on_model_error: ModelErrorFn,
        _types: PhantomData<(Input, Output, ModelOp, ModelEvent, Error, Key)>,
    }

    impl<
        State,
        Input,
        Output,
        ModelOp,
        ModelEvent,
        Error,
        Key,
        Submit,
        Cancel,
        ModelEventFn,
        ModelErrorFn,
    > InferenceApp
        for Instance<
            State,
            Input,
            Output,
            ModelOp,
            ModelEvent,
            Error,
            Key,
            Submit,
            Cancel,
            ModelEventFn,
            ModelErrorFn,
        >
    where
        State: Send + 'static,
        Input: Send + 'static,
        Output: Send + 'static,
        ModelOp: Send + 'static,
        ModelEvent: Send + 'static,
        Error: Send + 'static,
        Key: Send + 'static,
        Submit: FnMut(&mut State, RequestId, Input) -> Actions<Output, ModelOp, Error, Key>
            + Send
            + 'static,
        Cancel:
            FnMut(&mut State, RequestId) -> Actions<Output, ModelOp, Error, Key> + Send + 'static,
        ModelEventFn: FnMut(&mut State, Key, ModelEvent) -> Actions<Output, ModelOp, Error, Key>
            + Send
            + 'static,
        ModelErrorFn:
            FnMut(&mut State, Key, Error) -> Actions<Output, ModelOp, Error, Key> + Send + 'static,
    {
        type Input = Input;
        type Output = Output;
        type ModelOp = ModelOp;
        type ModelEvent = ModelEvent;
        type Error = Error;
        type Key = Key;
        fn on_submit(
            &mut self,
            id: RequestId,
            input: Self::Input,
        ) -> Actions<Self::Output, Self::ModelOp, Self::Error, Self::Key> {
            (self.on_submit)(&mut self.state, id, input)
        }

        fn on_cancel(
            &mut self,
            id: RequestId,
        ) -> Actions<Self::Output, Self::ModelOp, Self::Error, Self::Key> {
            (self.on_cancel)(&mut self.state, id)
        }

        fn on_model_event(
            &mut self,
            key: Self::Key,
            event: Self::ModelEvent,
        ) -> Actions<Self::Output, Self::ModelOp, Self::Error, Self::Key> {
            (self.on_model_event)(&mut self.state, key, event)
        }

        fn on_model_error(
            &mut self,
            key: Self::Key,
            error: Self::Error,
        ) -> Actions<Self::Output, Self::ModelOp, Self::Error, Self::Key> {
            (self.on_model_error)(&mut self.state, key, error)
        }
    }

    Instance {
        state,
        on_submit,
        on_cancel,
        on_model_event,
        on_model_error,
        _types: PhantomData,
    }
}

pub fn model<State: Send + 'static, Op, Event, Error, F>(
    state: State,
    f: F,
) -> impl ModelExecutor<Op, Event, Error>
where
    Op: Send + 'static,
    Event: Send + 'static,
    Error: Send + 'static,
    F: FnMut(&mut State, Op) -> Result<Event, Error> + Send + 'static,
{
    struct Instance<State, Op, Event, Error, F> {
        state: State,
        f: F,
        _types: PhantomData<(Op, Event, Error)>,
    }

    impl<State, Op, Event, Error, F> ModelExecutor<Op, Event, Error>
        for Instance<State, Op, Event, Error, F>
    where
        State: Send + 'static,
        Op: Send + 'static,
        Event: Send + 'static,
        Error: Send + 'static,
        F: FnMut(&mut State, Op) -> Result<Event, Error> + Send + 'static,
    {
        fn execute(&mut self, op: Op) -> Result<Event, Error> {
            (self.f)(&mut self.state, op)
        }
    }

    Instance {
        state,
        f,
        _types: PhantomData,
    }
}
