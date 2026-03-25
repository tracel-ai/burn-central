use std::cell::RefCell;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use crate::{ExperimentHandle, ExperimentId, ExperimentRun};

thread_local! {
    static CURRENT_EXPERIMENTS: RefCell<Vec<ExperimentHandle>> = const { RefCell::new(Vec::new()) };
}

/// Guard returned when entering an ambient experiment scope.
///
/// The previous ambient experiment is restored when the guard is dropped.
pub struct CurrentExperimentGuard {
    experiment_id: ExperimentId,
}

/// Future wrapper that re-enters the ambient experiment scope on every poll.
pub struct WithCurrentExperiment<F> {
    future: F,
    handle: ExperimentHandle,
}

/// Extension trait for propagating an experiment context with a future.
pub trait ExperimentInstrument: Future + Sized {
    /// Poll this future inside the provided ambient experiment context.
    fn in_experiment<H>(self, experiment: H) -> WithCurrentExperiment<Self>
    where
        H: Into<ExperimentHandle>,
    {
        WithCurrentExperiment::new(self, experiment.into())
    }

    /// Poll this future inside the current ambient experiment context, if one exists.
    fn in_current_experiment(self) -> WithCurrentExperiment<Self> {
        let handle = current_experiment().expect("no ambient experiment to propagate");
        WithCurrentExperiment::new(self, handle)
    }
}

impl<F: Future> ExperimentInstrument for F {}

impl<F> WithCurrentExperiment<F> {
    fn new(future: F, handle: ExperimentHandle) -> Self {
        Self { future, handle }
    }
}

impl ExperimentHandle {
    /// Enter an ambient experiment scope for the duration of the returned guard.
    pub fn enter(&self) -> CurrentExperimentGuard {
        enter_experiment_handle(self.clone())
    }

    /// Run a closure inside this handle's ambient experiment context.
    pub fn in_scope<T>(&self, f: impl FnOnce() -> T) -> T {
        with_experiment_handle(self.clone(), f)
    }

    /// Return a callable that enters this handle's ambient experiment context when invoked.
    pub fn bind_current<T, F>(&self, f: F) -> impl FnOnce() -> T + use<T, F>
    where
        F: FnOnce() -> T,
    {
        let handle = self.clone();
        move || with_experiment_handle(handle, f)
    }
}

impl ExperimentRun {
    /// Enter an ambient experiment scope for the duration of the returned guard.
    pub fn enter(&self) -> CurrentExperimentGuard {
        self.handle().enter()
    }

    /// Run a closure inside this run's ambient experiment context.
    pub fn in_scope<T>(&self, f: impl FnOnce() -> T) -> T {
        self.handle().in_scope(f)
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

/// Return the current ambient experiment handle, if one is in scope.
pub fn current_experiment() -> Option<ExperimentHandle> {
    CURRENT_EXPERIMENTS.with(|experiments| experiments.borrow().last().cloned())
}

pub(crate) fn enter_experiment_handle(handle: ExperimentHandle) -> CurrentExperimentGuard {
    CURRENT_EXPERIMENTS.with(|experiments| {
        experiments.borrow_mut().push(handle.clone());
    });

    CurrentExperimentGuard {
        experiment_id: handle.id().clone(),
    }
}

pub(crate) fn with_experiment_handle<T>(handle: ExperimentHandle, f: impl FnOnce() -> T) -> T {
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

    use crate::error::ExperimentError;
    use crate::reader::{ExperimentArtifactReader, ExperimentReaderError, LoadedArtifact};
    use crate::session::{BundleFn, ExperimentCompletion, ExperimentSession};
    use crate::{ArtifactKind, CancelToken, ExperimentId, ExperimentRun};

    use super::{ExperimentInstrument, current_experiment};

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

        assert!(current_experiment().is_none());
        {
            let _outer = run_a.enter();
            let current = current_experiment().expect("current experiment should be set");
            assert_eq!(current.id(), run_a.id());

            {
                let _inner = run_b.enter();
                let current =
                    current_experiment().expect("inner experiment should override outer scope");
                assert_eq!(current.id(), run_b.id());
            }

            let current = current_experiment().expect("outer experiment should be restored");
            assert_eq!(current.id(), run_a.id());
        }
        assert!(current_experiment().is_none());
    }

    #[test]
    fn handle_with_current_can_be_used_in_spawned_threads() {
        let run = create_run("ambient-test-thread");
        let handle = run.handle();

        std::thread::spawn(handle.bind_current(move || {
            let current = current_experiment().expect("current experiment should be set");
            assert_eq!(current.id().as_str(), "ambient-test-thread");
        }))
        .join()
        .expect("thread should complete successfully");

        assert!(current_experiment().is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn instrumented_tokio_tasks_keep_multiple_experiments_isolated() {
        let run_a = create_run("ambient-test-tokio-a");
        let run_b = create_run("ambient-test-tokio-b");

        let task_a = tokio::spawn(
            async {
                let current = current_experiment().expect("current experiment should be set");
                assert_eq!(current.id().as_str(), "ambient-test-tokio-a");

                tokio::task::yield_now().await;

                let current = current_experiment().expect("current experiment should still be set");
                assert_eq!(current.id().as_str(), "ambient-test-tokio-a");
            }
            .in_experiment(run_a.handle()),
        );

        let task_b = tokio::spawn(
            async {
                let current = current_experiment().expect("current experiment should be set");
                assert_eq!(current.id().as_str(), "ambient-test-tokio-b");

                tokio::task::yield_now().await;

                let current = current_experiment().expect("current experiment should still be set");
                assert_eq!(current.id().as_str(), "ambient-test-tokio-b");
            }
            .in_experiment(run_b.handle()),
        );

        task_a
            .await
            .expect("first tokio task should complete successfully");
        task_b
            .await
            .expect("second tokio task should complete successfully");
        assert!(current_experiment().is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn nested_tokio_spawn_can_use_in_current_experiment() {
        let run = create_run("ambient-test-tokio-nested");

        let outer = tokio::spawn(
            async {
                let current = current_experiment().expect("current experiment should be set");
                assert_eq!(current.id().as_str(), "ambient-test-tokio-nested");

                let inner = tokio::spawn(
                    async {
                        let current =
                            current_experiment().expect("current experiment should be set");
                        assert_eq!(current.id().as_str(), "ambient-test-tokio-nested");

                        tokio::task::yield_now().await;

                        let current =
                            current_experiment().expect("current experiment should still be set");
                        assert_eq!(current.id().as_str(), "ambient-test-tokio-nested");
                    }
                    .in_current_experiment(),
                );

                tokio::task::yield_now().await;

                let current =
                    current_experiment().expect("outer current experiment should still be set");
                assert_eq!(current.id().as_str(), "ambient-test-tokio-nested");

                inner
                    .await
                    .expect("nested tokio task should complete successfully");
            }
            .in_experiment(run.handle()),
        );

        outer
            .await
            .expect("outer tokio task should complete successfully");
        assert!(current_experiment().is_none());
    }
}
