use variadics_please::all_tuples;

pub mod args;
pub mod artifact_loader;
pub mod cancellation;
pub mod default;

/// This trait defines how parameters for a routine are retrieved from the execution context.
pub trait RoutineParam<Ctx>: Sized {
    type Item<'new>
    where
        Ctx: 'new;

    /// This method retrieves the parameter from the context.
    fn retrieve(ctx: &Ctx) -> Self::Item<'_> {
        Self::try_retrieve(ctx).unwrap()
    }

    /// This method attempts to retrieve the parameter from the context, returning an error if it fails.
    fn try_retrieve(ctx: &Ctx) -> anyhow::Result<Self::Item<'_>>;
}

macro_rules! impl_routine_param_tuple {
    ($($P:ident),*) => {
        #[expect(
            clippy::allow_attributes,
            reason = "This is in a macro, and as such, the below lints may not always apply."
        )]
        #[allow(
            non_snake_case,
            reason = "Certain variable names are provided by the caller, not by us."
        )]
        #[allow(
            unused_variables,
            reason = "Zero-length tuples won't use some of the parameters."
        )]
        impl<Ctx, $($P: RoutineParam<Ctx>),*> RoutineParam<Ctx> for ($($P,)*) {
            type Item<'new> = ($($P::Item<'new>,)*) where Ctx: 'new;

            fn try_retrieve<'r>(ctx: &'r Ctx) -> anyhow::Result<Self::Item<'r>> {
                Ok((
                    $(<$P as RoutineParam<Ctx>>::try_retrieve(ctx)?,)*
                ))
            }
        }
    };
}

all_tuples!(impl_routine_param_tuple, 0, 16, P);
