use burn::prelude::Backend;
use derive_more::{Deref, From};
use json_patch::merge;
use serde::{Deserialize, Serialize};

use crate::{executor::ExecutionContext, params::RoutineParam};

/// Marker trait for routine arguments.
///
/// Argument types must implement [Default] so runtime overrides can be merged on top of defaults.
pub trait LaunchArgs: Serialize + for<'de> Deserialize<'de> + Default {}
impl<T> LaunchArgs for T where T: Serialize + for<'de> Deserialize<'de> + Default {}

/// Deserialize `args` into `T` after merging it with `T::default()`.
///
/// Merge behavior follows JSON Merge Patch semantics:
/// - object fields are merged recursively
/// - arrays are replaced as a whole
/// - `null` clears optional fields
pub fn deserialize_and_merge_with_default<T: LaunchArgs>(
    args: &serde_json::Value,
) -> Result<T, serde_json::Error> {
    let mut merged = serde_json::to_value(T::default())?;

    merge(&mut merged, args);

    serde_json::from_value(merged)
}

/// Args are wrapper around the config you want to inject.
///
/// Use this in routine signatures (`Args<MyArgs>`) to retrieve merged arguments.
/// The runtime override JSON is merged with `MyArgs::default()`, so omitted fields keep their
/// default values.
#[derive(From, Deref)]
pub struct Args<T: LaunchArgs>(pub T);

impl<B: Backend, C: LaunchArgs> RoutineParam<ExecutionContext<B>> for Args<C> {
    type Item<'new> = Args<C>;

    fn try_retrieve(ctx: &ExecutionContext<B>) -> anyhow::Result<Self::Item<'_>> {
        let cfg = ctx.use_merged_args();
        Ok(Args(cfg))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use serde_json::json;

    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Nested {
        x: bool,
        y: u64,
    }

    impl Default for Nested {
        fn default() -> Self {
            Nested { x: true, y: 10 }
        }
    }

    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct MyArgs {
        a: i32,
        b: Option<String>,
        nested: Nested,
        list: Vec<i32>,
    }

    impl Default for MyArgs {
        fn default() -> Self {
            MyArgs {
                a: 5,
                b: Some("hello".to_owned()),
                nested: Nested::default(),
                list: vec![1, 2, 3],
            }
        }
    }

    #[test]
    fn empty_override_returns_default() {
        let cfg: MyArgs = deserialize_and_merge_with_default(&json!({})).unwrap();
        assert_eq!(cfg, MyArgs::default());
    }

    #[test]
    fn override_top_level_field() {
        let cfg: MyArgs = deserialize_and_merge_with_default(&json!({ "a": 42 })).unwrap();
        let expected = MyArgs {
            a: 42,
            ..Default::default()
        };
        assert_eq!(cfg, expected);
    }

    #[test]
    fn deep_override_nested_field() {
        let cfg: MyArgs =
            deserialize_and_merge_with_default(&json!({ "nested": { "y": 99 } })).unwrap();
        let mut expected = MyArgs::default();
        expected.nested.y = 99;
        assert_eq!(cfg, expected);
    }

    #[test]
    fn null_becomes_json_null_for_optional() {
        let cfg: MyArgs = deserialize_and_merge_with_default(&json!({ "b": null })).unwrap();
        assert_eq!(cfg.b, None);
    }

    #[test]
    fn null_becomes_json_null_for_required() {
        let err = deserialize_and_merge_with_default::<MyArgs>(&json!({ "a": null })).unwrap_err();
        assert!(err.is_data());
    }

    #[test]
    fn override_list_replaces_array() {
        let cfg: MyArgs = deserialize_and_merge_with_default(&json!({ "list": [9,8,7] })).unwrap();
        assert_eq!(cfg.list, vec![9, 8, 7]);
    }

    #[test]
    fn type_mismatch_in_nested_errors_data() {
        let err = deserialize_and_merge_with_default::<MyArgs>(
            &json!({ "nested": { "x": "not_a_bool" } }),
        )
        .unwrap_err();
        assert!(err.is_data());
    }

    #[test]
    fn patch_application_error_propagates() {
        let err =
            deserialize_and_merge_with_default::<MyArgs>(&json!({ "nested": { "y": [1, 2, 3] } }))
                .unwrap_err();
        assert!(err.is_data());
    }
}
