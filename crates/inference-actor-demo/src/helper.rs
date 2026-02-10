use crate::{Actions, InferenceApp, RequestId};

fn default_on_model_error<State, Output, ModelOp, Error>(
    _state: &mut State,
    _error: Error,
) -> Actions<Output, ModelOp, Error> {
    Actions::new()
}

pub fn app<State, Input, Output, ModelOp, ModelEvent, Error, Submit, Cancel, ModelEventFn>(
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
>
where
    State: Send + 'static,
    Input: Send + 'static,
    Output: Send + 'static,
    ModelOp: Send + 'static,
    ModelEvent: Send + 'static,
    Error: Send + 'static,
    Submit: FnMut(&mut State, RequestId, Input) -> Actions<Output, ModelOp, Error> + Send + 'static,
    Cancel: FnMut(&mut State, RequestId) -> Actions<Output, ModelOp, Error> + Send + 'static,
    ModelEventFn: FnMut(&mut State, ModelEvent) -> Actions<Output, ModelOp, Error> + Send + 'static,
{
    app_with_error(
        state,
        on_submit,
        on_cancel,
        on_model_event,
        default_on_model_error::<State, Output, ModelOp, Error>,
    )
}

pub fn app_with_error<
    State,
    Input,
    Output,
    ModelOp,
    ModelEvent,
    Error,
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
>
where
    State: Send + 'static,
    Input: Send + 'static,
    Output: Send + 'static,
    ModelOp: Send + 'static,
    ModelEvent: Send + 'static,
    Error: Send + 'static,
    Submit: FnMut(&mut State, RequestId, Input) -> Actions<Output, ModelOp, Error> + Send + 'static,
    Cancel: FnMut(&mut State, RequestId) -> Actions<Output, ModelOp, Error> + Send + 'static,
    ModelEventFn: FnMut(&mut State, ModelEvent) -> Actions<Output, ModelOp, Error> + Send + 'static,
    ModelErrorFn: FnMut(&mut State, Error) -> Actions<Output, ModelOp, Error> + Send + 'static,
{
    use std::marker::PhantomData;

    struct Instance<
        State,
        Input,
        Output,
        ModelOp,
        ModelEvent,
        Error,
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
        _types: PhantomData<(Input, Output, ModelOp, ModelEvent, Error)>,
    }

    impl<
        State,
        Input,
        Output,
        ModelOp,
        ModelEvent,
        Error,
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
        Submit:
            FnMut(&mut State, RequestId, Input) -> Actions<Output, ModelOp, Error> + Send + 'static,
        Cancel: FnMut(&mut State, RequestId) -> Actions<Output, ModelOp, Error> + Send + 'static,
        ModelEventFn:
            FnMut(&mut State, ModelEvent) -> Actions<Output, ModelOp, Error> + Send + 'static,
        ModelErrorFn: FnMut(&mut State, Error) -> Actions<Output, ModelOp, Error> + Send + 'static,
    {
        type Input = Input;
        type Output = Output;
        type ModelOp = ModelOp;
        type ModelEvent = ModelEvent;
        type Error = Error;

        fn on_submit(
            &mut self,
            id: RequestId,
            input: Self::Input,
        ) -> Actions<Self::Output, Self::ModelOp, Self::Error> {
            (self.on_submit)(&mut self.state, id, input)
        }

        fn on_cancel(
            &mut self,
            id: RequestId,
        ) -> Actions<Self::Output, Self::ModelOp, Self::Error> {
            (self.on_cancel)(&mut self.state, id)
        }

        fn on_model_event(
            &mut self,
            event: Self::ModelEvent,
        ) -> Actions<Self::Output, Self::ModelOp, Self::Error> {
            (self.on_model_event)(&mut self.state, event)
        }

        fn on_model_error(
            &mut self,
            error: Self::Error,
        ) -> Actions<Self::Output, Self::ModelOp, Self::Error> {
            (self.on_model_error)(&mut self.state, error)
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
