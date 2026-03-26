//! Ambient experiment context propagation.
//!
//! This module provides two related pieces:
//! - [`ExperimentGlobalExt`] for entering a thread-local ambient experiment scope.
//! - [`ExperimentInstrument`] for re-entering that scope whenever a future is polled.

use std::cell::RefCell;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use crate::{ExperimentId, ExperimentRun, ExperimentRunHandle};

thread_local! {
    static CURRENT_EXPERIMENTS: RefCell<Vec<ExperimentRunHandle>> = const { RefCell::new(Vec::new()) };
}

/// Guard returned when entering an ambient experiment scope.
///
/// Ambient scopes are thread-local and stack-based. When the guard is dropped, the previous
/// ambient experiment on that thread is restored.
///
/// This is returned by [`ExperimentRunHandle::enter`] and [`ExperimentGlobalExt::enter`].
#[derive(Debug)]
#[must_use = "ambient experiment guards must be kept alive to maintain the experiment scope"]
pub struct CurrentExperimentGuard {
    experiment_id: ExperimentId,
}

/// Future wrapper that re-enters the ambient experiment scope on every poll.
///
/// Instances are created through [`ExperimentInstrument`].
pub struct WithCurrentExperiment<F> {
    future: F,
    handle: ExperimentRunHandle,
}

/// Extension trait for propagating an experiment context with a future.
///
/// Use this when a future may be polled on a different thread or after the current ambient scope
/// has ended.
///
/// # Example
///
/// ```no_run
/// use burn_central_experiment::{ExperimentInstrument, ExperimentRun, ExperimentRunHandleExt};
///
/// let run = ExperimentRun::local("./runs").unwrap();
///
/// let _future = async move {
///     tracing::info!("this future will poll inside the experiment context");
/// }
/// .in_experiment(&run);
/// ```
pub trait ExperimentInstrument: Future + Sized {
    /// Instrument this future to automatically enter the given experiment context.
    fn in_experiment<H>(self, experiment: H) -> WithCurrentExperiment<Self>
    where
        H: Into<ExperimentRunHandle>,
    {
        WithCurrentExperiment::new(self, experiment.into())
    }

    /// Instrument this future to automatically enter the current ambient experiment context, if one exists.
    ///
    /// # Panics
    ///
    /// Panics if there is no ambient experiment to propagate.
    fn in_current_experiment(self) -> WithCurrentExperiment<Self> {
        let handle = current_experiment().expect("no ambient experiment to propagate");
        WithCurrentExperiment::new(self, handle)
    }
}

impl<F: Future> ExperimentInstrument for F {}

impl<F> WithCurrentExperiment<F> {
    fn new(future: F, handle: ExperimentRunHandle) -> Self {
        Self { future, handle }
    }
}

/// Extension trait providing methods for entering and propagating an ambient experiment context.
///
/// Ambient context is thread-local and stack-based. Nested scopes temporarily override the current
/// experiment and restore the previous one when their guard is dropped.
///
/// This trait is implemented for [`ExperimentRun`]. [`ExperimentRunHandle`] provides matching
/// [`ExperimentRunHandle::enter`] and [`ExperimentRunHandle::in_scope`] methods when you only have
/// a cloned handle.
///
/// # Example
///
/// ```no_run
/// use burn_central_experiment::{ExperimentGlobalExt, ExperimentRun};
///
/// let run = ExperimentRun::local("./runs").unwrap();
///
/// run.in_scope(|| {
///     let current = ExperimentRun::current().unwrap();
///     assert_eq!(current.id(), run.id());
/// });
/// ```
pub trait ExperimentGlobalExt {
    /// Enter an ambient scope for this run until the returned guard is dropped.
    fn enter(&self) -> CurrentExperimentGuard;

    /// Run a closure with this run installed as the ambient experiment.
    fn in_scope<T>(&self, f: impl FnOnce() -> T) -> T;

    /// Return the current ambient experiment handle for this thread, if any.
    fn current() -> Option<ExperimentRunHandle>;
}

impl ExperimentRunHandle {
    /// See [`ExperimentGlobalExt::enter`].
    pub fn enter(&self) -> CurrentExperimentGuard {
        enter_experiment_handle(self.clone())
    }

    /// See [`ExperimentGlobalExt::in_scope`].
    pub fn in_scope<T>(&self, f: impl FnOnce() -> T) -> T {
        with_experiment_handle(self.clone(), f)
    }
}

impl ExperimentGlobalExt for ExperimentRun {
    fn enter(&self) -> CurrentExperimentGuard {
        enter_experiment_handle(self.handle.clone())
    }

    fn in_scope<T>(&self, f: impl FnOnce() -> T) -> T {
        with_experiment_handle(self.handle.clone(), f)
    }

    fn current() -> Option<ExperimentRunHandle> {
        current_experiment()
    }
}

impl<F: Future> Future for WithCurrentExperiment<F> {
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: We never move `future` after `self` has been pinned. We only project a mutable
        // reference to it for polling.
        let this = unsafe { self.get_unchecked_mut() };
        let _guard = enter_experiment_handle(this.handle.clone());
        // SAFETY: `future` is pinned together with `self` and is never moved after pinning.
        unsafe { Pin::new_unchecked(&mut this.future) }.poll(cx)
    }
}

/// Return the current ambient experiment handle, if one is in scope on this thread.
fn current_experiment() -> Option<ExperimentRunHandle> {
    CURRENT_EXPERIMENTS.with(|experiments| experiments.borrow().last().cloned())
}

fn enter_experiment_handle(handle: ExperimentRunHandle) -> CurrentExperimentGuard {
    CURRENT_EXPERIMENTS.with(|experiments| {
        experiments.borrow_mut().push(handle.clone());
    });

    CurrentExperimentGuard {
        experiment_id: handle.id().clone(),
    }
}

fn with_experiment_handle<T>(handle: ExperimentRunHandle, f: impl FnOnce() -> T) -> T {
    let _guard = enter_experiment_handle(handle);
    f()
}

impl Drop for CurrentExperimentGuard {
    fn drop(&mut self) {
        CURRENT_EXPERIMENTS.with(|experiments| {
            let popped = experiments.borrow_mut().pop();
            debug_assert_eq!(
                popped.as_ref().map(|handle| handle.id()),
                Some(&self.experiment_id),
                "ambient experiment scopes must unwind in stack order",
            );
        });
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::context::ExperimentGlobalExt as _;
    use crate::error::ExperimentError;
    use crate::reader::{ExperimentArtifactReader, ExperimentReaderError, LoadedArtifact};
    use crate::session::{BundleFn, ExperimentCompletion, ExperimentSession};
    use crate::{ArtifactKind, CancelToken, ExperimentId, ExperimentRun, ExperimentRunHandleExt};

    use super::ExperimentInstrument;

    #[derive(Default)]
    struct MockSession;

    impl ExperimentSession for MockSession {
        fn record_event(&self, _event: crate::session::Event) -> Result<(), ExperimentError> {
            Ok(())
        }

        fn save_artifact(
            &self,
            _name: &str,
            _kind: ArtifactKind,
            _artifact: Box<BundleFn>,
        ) -> Result<(), ExperimentError> {
            Ok(())
        }

        fn finish(&self, _completion: ExperimentCompletion) -> Result<(), ExperimentError> {
            Ok(())
        }
    }

    #[derive(Default)]
    struct NoopExperimentDataReader;

    impl ExperimentArtifactReader for NoopExperimentDataReader {
        fn load_artifact_raw(
            &self,
            _experiment_id: ExperimentId,
            _name: &str,
        ) -> Result<LoadedArtifact, ExperimentReaderError> {
            Err(ExperimentReaderError::new("Artifact not found"))
        }
    }

    fn create_run(id: &str) -> ExperimentRun {
        ExperimentRun::new(
            id,
            Arc::new(MockSession),
            NoopExperimentDataReader,
            CancelToken::default(),
        )
    }

    #[test]
    fn nested_experiment_scopes_restore_the_previous_experiment() {
        let run_a = create_run("ambient-test-a");
        let run_b = create_run("ambient-test-b");

        assert!(ExperimentRun::current().is_none());
        {
            let _outer = run_a.enter();
            let current = ExperimentRun::current().expect("current experiment should be set");
            assert_eq!(current.id(), run_a.id());

            {
                let _inner = run_b.enter();
                let current =
                    ExperimentRun::current().expect("inner experiment should override outer scope");
                assert_eq!(current.id(), run_b.id());
            }

            let current = ExperimentRun::current().expect("outer experiment should be restored");
            assert_eq!(current.id(), run_a.id());
        }
        assert!(ExperimentRun::current().is_none());
    }

    #[test]
    fn handle_with_current_can_be_used_in_spawned_threads() {
        let run = create_run("ambient-test-thread");
        let handle = run.handle();

        std::thread::spawn(move || {
            handle.in_scope(|| {
                let current = ExperimentRun::current().expect("current experiment should be set");
                assert_eq!(current.id().as_str(), "ambient-test-thread");
            })
        })
        .join()
        .expect("thread should complete successfully");

        assert!(ExperimentRun::current().is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn instrumented_tokio_tasks_keep_multiple_experiments_isolated() {
        let run_a = create_run("ambient-test-tokio-a");
        let run_b = create_run("ambient-test-tokio-b");

        let task_a = tokio::spawn(
            async {
                let current = ExperimentRun::current().expect("current experiment should be set");
                assert_eq!(current.id().as_str(), "ambient-test-tokio-a");

                tokio::task::yield_now().await;

                let current =
                    ExperimentRun::current().expect("current experiment should still be set");
                assert_eq!(current.id().as_str(), "ambient-test-tokio-a");
            }
            .in_experiment(&run_a),
        );

        let task_b = tokio::spawn(
            async {
                let current = ExperimentRun::current().expect("current experiment should be set");
                assert_eq!(current.id().as_str(), "ambient-test-tokio-b");

                tokio::task::yield_now().await;

                let current =
                    ExperimentRun::current().expect("current experiment should still be set");
                assert_eq!(current.id().as_str(), "ambient-test-tokio-b");
            }
            .in_experiment(&run_b),
        );

        task_a
            .await
            .expect("first tokio task should complete successfully");
        task_b
            .await
            .expect("second tokio task should complete successfully");
        assert!(ExperimentRun::current().is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn nested_tokio_spawn_can_use_in_current_experiment() {
        let run = create_run("ambient-test-tokio-nested");

        let outer = tokio::spawn(
            async {
                let current = ExperimentRun::current().expect("current experiment should be set");
                assert_eq!(current.id().as_str(), "ambient-test-tokio-nested");

                let inner = tokio::spawn(
                    async {
                        let current =
                            ExperimentRun::current().expect("current experiment should be set");
                        assert_eq!(current.id().as_str(), "ambient-test-tokio-nested");

                        tokio::task::yield_now().await;

                        let current = ExperimentRun::current()
                            .expect("current experiment should still be set");
                        assert_eq!(current.id().as_str(), "ambient-test-tokio-nested");
                    }
                    .in_current_experiment(),
                );

                tokio::task::yield_now().await;

                let current =
                    ExperimentRun::current().expect("outer current experiment should still be set");
                assert_eq!(current.id().as_str(), "ambient-test-tokio-nested");

                inner
                    .await
                    .expect("nested tokio task should complete successfully");
            }
            .in_experiment(&run),
        );

        outer
            .await
            .expect("outer tokio task should complete successfully");
        assert!(ExperimentRun::current().is_none());
    }
}
