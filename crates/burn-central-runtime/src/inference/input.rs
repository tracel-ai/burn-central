use crate::{inference::In, routine::Routine};
use variadics_please::all_tuples;

pub type RoutineIn<'a, Ctx, S> = <<S as Routine<Ctx>>::In as RoutineInput>::Inner<'a>;

pub trait RoutineInput: Sized {
    type Param<'i>: RoutineInput;
    type Inner<'i>;
    fn wrap(this: Self::Inner<'_>) -> Self::Param<'_>;
}

macro_rules! impl_routine_input_tuple {
    ($(#[$meta:meta])* $($name:ident),*) => {
        $(#[$meta])*
        impl<$($name: RoutineInput),*> RoutineInput for ($($name,)*) {
            type Param<'i> = ($($name::Param<'i>,)*);
            type Inner<'i> = ($($name::Inner<'i>,)*);

            #[expect(
                clippy::allow_attributes,
                reason = "This is in a macro; as such, the below lints may not always apply."
            )]
            #[allow(
                non_snake_case,
                reason = "Certain variable names are provided by the caller, not by us."
            )]
            #[allow(
                clippy::unused_unit,
                reason = "Zero-length tuples won't have anything to wrap."
            )]
            fn wrap(this: Self::Inner<'_>) -> Self::Param<'_> {
                let ($($name,)*) = this;
                ($($name::wrap($name),)*)
            }
        }
    };
}

all_tuples!(impl_routine_input_tuple, 0, 8, I);

impl<T: 'static> RoutineInput for In<T> {
    type Param<'i> = In<T>;
    type Inner<'i> = T;

    fn wrap(this: Self::Inner<'_>) -> Self::Param<'_> {
        In(this)
    }
}
