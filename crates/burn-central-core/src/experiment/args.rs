use json_patch::merge;
use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    PatchError(json_patch::PatchError),
    #[error(transparent)]
    Serialization(serde_json::Error),
}

impl Error {
    pub fn is_syntax(&self) -> bool {
        matches!(self, Error::Serialization(e) if e.is_syntax())
    }

    pub fn is_data(&self) -> bool {
        matches!(self, Error::Serialization(e) if e.is_data())
    }
}

/// Trait for experiments arguments. It specify that the type must be serializable, deserializable
/// and implement default. The reason it must implement default is that when you override a value
/// it will only override the value you provide, the rest will be filled with the default value.
pub trait ExperimentArgs: Serialize + for<'de> Deserialize<'de> + Default {}
impl<T> ExperimentArgs for T where T: Serialize + for<'de> Deserialize<'de> + Default {}

pub fn deserialize_and_merge_with_default<T: ExperimentArgs>(
    args: &serde_json::Value,
) -> Result<T, Error> {
    let mut merged = serde_json::to_value(T::default()).map_err(Error::Serialization)?;

    merge(&mut merged, args);

    serde_json::from_value(merged).map_err(Error::Serialization)
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
