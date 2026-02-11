use crate::inference::{ErasedSession, JsonSession, SessionHandle};
use burn::prelude::Backend;
use burn_central_registry::Registry;
use serde::{Serialize, de::DeserializeOwned};
use std::collections::HashMap;
use std::fmt::Display;
use std::marker::PhantomData;

#[derive(Debug, thiserror::Error)]
pub enum InferenceError {
    #[error("inference handler not found: {0}")]
    NotFound(String),
    #[error("inference handler '{name}' failed to initialize: {message}")]
    FactoryFailed { name: String, message: String },
}

pub struct InferenceInit<B: Backend> {
    pub registry: Registry,
    pub namespace: String,
    pub project: String,
    pub device: B::Device,
}

trait ErasedFactory<B: Backend>: Send + Sync {
    fn build(&self, init: InferenceInit<B>) -> Result<Box<dyn ErasedSession>, InferenceError>;
}

struct JsonFactory<F, I, O, E, M> {
    name: String,
    factory: F,
    _types: PhantomData<fn(I, O, E) -> M>,
}

impl<B, F, I, O, E, M> ErasedFactory<B> for JsonFactory<F, I, O, E, M>
where
    B: Backend,
    F: SessionFactoryFn<B, I, O, E, M>,
    I: DeserializeOwned + Send + Sync + 'static,
    O: Serialize + Send + Sync + 'static,
    E: Display + Send + Sync + 'static,
{
    fn build(&self, init: InferenceInit<B>) -> Result<Box<dyn ErasedSession>, InferenceError> {
        let session = self
            .factory
            .build(init)
            .map_err(|err| InferenceError::FactoryFailed {
                name: self.name.clone(),
                message: err.to_string(),
            })?;
        Ok(Box::new(JsonSession::new(session)))
    }
}

pub trait SessionFactoryFn<B: Backend, I, O, E, M>: Send + Sync {
    fn build(&self, init: InferenceInit<B>) -> Result<SessionHandle<I, O, E>, String>;
}

impl<B, F, I, O, E, E1: ToString> SessionFactoryFn<B, I, O, E, ()> for F
where
    B: Backend,
    F: Fn(InferenceInit<B>) -> Result<SessionHandle<I, O, E>, E1> + Send + Sync,
    I: DeserializeOwned + Send + 'static,
    O: Serialize + Send + 'static,
    E: Display + Send + 'static,
{
    fn build(&self, init: InferenceInit<B>) -> Result<SessionHandle<I, O, E>, String> {
        (self)(init).map_err(|e| e.to_string())
    }
}

struct IsOkSessionFactoryFn;

impl<B, F, I, O, E> SessionFactoryFn<B, I, O, E, (IsOkSessionFactoryFn,)> for F
where
    B: Backend,
    F: Fn(InferenceInit<B>) -> SessionHandle<I, O, E> + Send + Sync,
    I: DeserializeOwned + Send + 'static,
    O: Serialize + Send + 'static,
    E: Display + Send + 'static,
{
    fn build(&self, init: InferenceInit<B>) -> Result<SessionHandle<I, O, E>, String> {
        Ok((self)(init))
    }
}

/// Registry of inference session factories keyed by name.
pub struct InferenceRegistry<B: Backend> {
    factories: HashMap<String, Box<dyn ErasedFactory<B>>>,
}

impl<B: Backend> InferenceRegistry<B> {
    pub fn new() -> Self {
        Self {
            factories: HashMap::new(),
        }
    }

    pub fn infer<I, O, E, F, M>(&mut self, name: impl Into<String>, factory: F) -> &mut Self
    where
        I: DeserializeOwned + Send + Sync + 'static,
        O: Serialize + Send + Sync + 'static,
        E: Display + Send + Sync + 'static,
        F: SessionFactoryFn<B, I, O, E, M> + 'static,
        M: 'static,
    {
        let name = name.into();
        let factory = JsonFactory {
            name: name.clone(),
            factory,
            _types: PhantomData,
        };
        self.factories.insert(name, Box::new(factory));
        self
    }

    pub fn build_session(
        &self,
        name: &str,
        init: InferenceInit<B>,
    ) -> Result<Box<dyn ErasedSession>, InferenceError> {
        let factory = self
            .factories
            .get(name)
            .ok_or_else(|| InferenceError::NotFound(name.to_string()))?;
        factory.build(init)
    }
}
