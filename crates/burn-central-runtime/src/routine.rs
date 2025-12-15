use crate::error::RuntimeError;
use crate::inference::{RoutineIn, RoutineInput};
use crate::output::RoutineOutput;
use crate::params::RoutineParam;
use crate::type_name::fn_type_name;
use std::marker::PhantomData;
use variadics_please::all_tuples;

#[diagnostic::on_unimplemented(message = "`{Self}` is not a routine", label = "invalid routine")]
pub trait Routine<Ctx>: Send + Sync + 'static {
    type In: RoutineInput;
    type Out;

    fn name(&self) -> &str;
    fn run(
        &self,
        input: RoutineIn<'_, Ctx, Self>,
        ctx: &mut Ctx,
    ) -> anyhow::Result<Self::Out, RuntimeError>;
}

pub type RoutineParamItem<'ctx, Ctx, P> = <P as RoutineParam<Ctx>>::Item<'ctx>;

#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a valid routine",
    label = "invalid routine"
)]
pub trait RoutineParamFunction<Ctx, Marker>: Send + Sync + 'static {
    type In: RoutineInput;
    type Out;
    type Param: RoutineParam<Ctx>;

    fn run(
        &self,
        input: <Self::In as RoutineInput>::Inner<'_>,
        param_value: RoutineParamItem<Ctx, Self::Param>,
    ) -> anyhow::Result<Self::Out, RuntimeError>;
}

#[doc(hidden)]
pub struct HasRoutineInput;

macro_rules! impl_routine_function {
    ($($param: ident),*) => {
        #[expect(
            clippy::allow_attributes,
            reason = "This is within a macro, and as such, the below lints may not always apply."
        )]
        #[allow(
            non_snake_case,
            reason = "Certain variable names are provided by the caller, not by us."
        )]
        #[allow(clippy::too_many_arguments)]
        impl<Ctx, Out, Func, $($param: RoutineParam<Ctx>),*> RoutineParamFunction<Ctx, fn($($param,)*) -> Out> for Func
        where
            Func: Send + Sync + 'static,
            for <'a> &'a Func:
                Fn($($param),*) -> Out +
                Fn($(RoutineParamItem<Ctx, $param>),*) -> Out,
            Out: 'static,
            Ctx: 'static,
        {
            type In = ();
            type Out = Out;
            type Param = ($($param,)*);
            #[inline]
            fn run(&self, _input: (), param_value: RoutineParamItem<Ctx, ($($param,)*)>) -> Result<Self::Out, RuntimeError> {
                #[expect(
                    clippy::allow_attributes,
                    reason = "This is within a macro, and as such, the below lints may not always apply."
                )]
                #[allow(clippy::too_many_arguments)]
                fn call_inner<Out, $($param,)*>(
                    f: impl Fn($($param,)*)->Out,
                    $($param: $param,)*
                )->Out{
                    f($($param,)*)
                }
                let ($($param,)*) = param_value;
                Ok(call_inner(self, $($param),*))
            }
        }

        #[expect(
            clippy::allow_attributes,
            reason = "This is within a macro, and as such, the below lints may not always apply."
        )]
        #[allow(
            non_snake_case,
            reason = "Certain variable names are provided by the caller, not by us."
        )]
        #[allow(clippy::too_many_arguments)]
        impl<Ctx, In, Out, Func, $($param: RoutineParam<Ctx>),*> RoutineParamFunction<Ctx, (HasRoutineInput, fn(In, $($param,)*) -> Out)> for Func
        where
            Func: Send + Sync + 'static,
            for <'a> &'a Func:
                Fn(In, $($param),*) -> Out +
                Fn(In::Param<'_>, $(RoutineParamItem<Ctx, $param>),*) -> Out,
            In: RoutineInput + 'static,
            Out: 'static,
            Ctx: 'static,
        {
            type In = In;
            type Out = Out;
            type Param = ($($param,)*);
            #[inline]
            fn run(&self, input: In::Inner<'_>, param_value: RoutineParamItem<Ctx, ($($param,)*)>) -> Result<Self::Out, RuntimeError> {
                fn call_inner<In: RoutineInput, Out, $($param,)*>(
                    _: PhantomData<In>,
                    f: impl Fn(In::Param<'_>, $($param,)*)->Out,
                    input: In::Inner<'_>,
                    $($param: $param,)*
                )->Out{
                    f(In::wrap(input), $($param,)*)
                }
                let ($($param,)*) = param_value;
                Ok(call_inner(PhantomData::<In>, self, input, $($param),*))
            }
        }
    };
}

all_tuples!(impl_routine_function, 0, 16, F);

#[doc(hidden)]
pub struct IsFunctionRoutine;

pub struct FunctionRoutine<Marker, F> {
    func: F,
    name: String,
    _marker: PhantomData<fn() -> Marker>,
}

impl<Marker, F> FunctionRoutine<Marker, F> {
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
}

impl<Marker, F: Clone> Clone for FunctionRoutine<Marker, F> {
    fn clone(&self) -> Self {
        FunctionRoutine {
            func: self.func.clone(),
            name: self.name.clone(),
            _marker: PhantomData,
        }
    }
}

impl<Ctx, Marker, F> IntoRoutine<Ctx, F::In, F::Out, (IsFunctionRoutine, Marker)> for F
where
    Marker: 'static,
    F: RoutineParamFunction<Ctx, Marker>,
{
    type Routine = FunctionRoutine<Marker, F>;

    fn into_routine(func: Self) -> Self::Routine {
        FunctionRoutine {
            func,
            name: fn_type_name::<F>(),
            _marker: PhantomData,
        }
    }
}

impl<Ctx, Marker, F> Routine<Ctx> for FunctionRoutine<Marker, F>
where
    Marker: 'static,
    F: RoutineParamFunction<Ctx, Marker>,
{
    type In = F::In;
    type Out = F::Out;

    fn name(&self) -> &str {
        self.name.as_str()
    }

    fn run(
        &self,
        input: RoutineIn<'_, Ctx, Self>,
        ctx: &mut Ctx,
    ) -> anyhow::Result<Self::Out, RuntimeError> {
        let params = <F::Param as RoutineParam<Ctx>>::try_retrieve(ctx).map_err(|e| {
            RuntimeError::HandlerFailed(anyhow::anyhow!("Failed to retrieve parameters: {}", e))
        })?;
        let output = self.func.run(input, params)?;
        Ok(output)
    }
}

impl<Ctx, T: Routine<Ctx>> IntoRoutine<Ctx, T::In, T::Out, ()> for T {
    type Routine = T;
    fn into_routine(this: Self) -> Self::Routine {
        this
    }
}

#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a valid routine with output `{Output}`",
    label = "invalid routine"
)]
pub trait IntoRoutine<Ctx, Input, Output, Marker>: Sized {
    type Routine: Routine<Ctx, In = Input, Out = Output>;

    #[allow(clippy::wrong_self_convention)]
    fn into_routine(this: Self) -> Self::Routine;

    fn with_name(self, name: impl Into<String>) -> IntoNamedRoutine<Ctx, Self> {
        IntoNamedRoutine {
            routine: self,
            name: name.into(),
            marker: Default::default(),
        }
    }
}

#[derive(Clone)]
pub struct IntoNamedRoutine<Ctx, S> {
    routine: S,
    name: String,
    marker: PhantomData<fn(Ctx)>,
}

pub struct NamedRoutine<S> {
    inner: S,
    name: String,
}

impl<Ctx, S> Routine<Ctx> for NamedRoutine<S>
where
    S: Routine<Ctx>,
{
    type In = S::In;
    type Out = S::Out;

    fn name(&self) -> &str {
        &self.name
    }

    fn run(
        &self,
        input: RoutineIn<'_, Ctx, Self>,
        ctx: &mut Ctx,
    ) -> anyhow::Result<Self::Out, RuntimeError> {
        self.inner.run(input, ctx)
    }
}

#[doc(hidden)]
pub struct IsNamedRoutine;
impl<Ctx, I, O, M, S> IntoRoutine<Ctx, I, O, (IsNamedRoutine, M)> for IntoNamedRoutine<Ctx, S>
where
    S: IntoRoutine<Ctx, I, O, M>,
{
    type Routine = NamedRoutine<S::Routine>;

    fn into_routine(this: Self) -> Self::Routine {
        NamedRoutine {
            inner: IntoRoutine::into_routine(this.routine),
            name: this.name,
        }
    }
}

impl<Ctx, I, O, M, S, N> IntoRoutine<Ctx, I, O, (IsNamedRoutine, N, M)> for (N, S)
where
    S: IntoRoutine<Ctx, I, O, M>,
    N: Into<String>,
{
    type Routine = NamedRoutine<S::Routine>;

    fn into_routine(this: Self) -> Self::Routine {
        let (name, routines) = this;
        NamedRoutine {
            inner: IntoRoutine::into_routine(routines),
            name: name.into(),
        }
    }
}

pub struct ExecutorRoutineWrapper<S, Ctx>(S, PhantomData<Ctx>);

impl<S, Ctx, Input, Output> ExecutorRoutineWrapper<S, Ctx>
where
    S: Routine<Ctx, In = Input, Out = Output>,
{
    pub fn new(routine: S) -> Self {
        ExecutorRoutineWrapper(routine, PhantomData)
    }
}

impl<Ctx, S, Input, Output> Routine<Ctx> for ExecutorRoutineWrapper<S, Ctx>
where
    S: Routine<Ctx, In = Input, Out = Output>,
    Input: RoutineInput,
    Output: RoutineOutput<Ctx>,
    Ctx: Send + Sync + 'static,
{
    type In = Input;
    type Out = ();

    fn name(&self) -> &str {
        self.0.name()
    }

    fn run(
        &self,
        input: RoutineIn<'_, Ctx, Self>,
        ctx: &mut Ctx,
    ) -> anyhow::Result<Self::Out, RuntimeError> {
        match self.0.run(input, ctx) {
            Ok(output) => {
                output.apply_output(ctx).map_err(|e| {
                    RuntimeError::HandlerFailed(anyhow::anyhow!("Failed to apply output: {}", e))
                })?;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }
}

pub type BoxedRoutine<Ctx, In, Out> = Box<dyn Routine<Ctx, In = In, Out = Out>>;
