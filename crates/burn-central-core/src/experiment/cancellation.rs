use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// A trait representing a cancellable task or operation. Implementors should ensure that `cancel` is idempotent and thread-safe, and that `is_cancelled` accurately reflects the cancellation state.
pub trait Cancellable: Send + Sync {
    /// Cancel the task. Should be idempotent and thread-safe.
    fn cancel(&self);
    /// Check if the task has been cancelled. Should be thread-safe.
    fn is_cancelled(&self) -> bool;
}

type CancellableRef = Arc<dyn Cancellable>;

/// A cancellation token that can be shared across tasks. When cancelled, it will cancel all linked child tasks/tokens. Thread-safe and can be cloned.
#[derive(Clone, Default)]
pub struct CancelToken {
    inner: Arc<Inner>,
}

#[derive(Default)]
struct Inner {
    cancelled: AtomicBool,
    children: Mutex<Vec<CancellableRef>>,
}

impl CancelToken {
    /// Create a new cancellation token that is not cancelled and has no children.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Inner {
                cancelled: AtomicBool::new(false),
                children: Mutex::new(Vec::new()),
            }),
        }
    }

    /// Check if this token has been cancelled. Thread-safe.
    pub fn is_cancelled(&self) -> bool {
        self.inner.cancelled.load(Ordering::Acquire)
    }

    /// Attach a child task/token to be cancelled when this token is cancelled.
    /// If already cancelled, child is cancelled immediately.
    pub fn link<T: Cancellable + 'static>(&self, child: T) {
        let child = Arc::new(child);
        if self.is_cancelled() {
            child.cancel();
            return;
        }

        let mut children = self.inner.children.lock().unwrap();

        if self.is_cancelled() {
            drop(children);
            child.cancel();
            return;
        }

        children.push(child);
    }

    /// Cancel this token and all currently-linked children.
    pub fn cancel(&self) {
        if self.inner.cancelled.swap(true, Ordering::AcqRel) {
            return;
        }

        let children = {
            let mut children = self.inner.children.lock().unwrap();
            std::mem::take(&mut *children)
        };

        for c in children {
            c.cancel();
        }
    }

    /// Create a new cancellable task/token of type T, link it to this token, and return it.
    /// See [Self::link] for more details.
    pub fn into_linked<T: Cancellable + Default + Clone + 'static>(&self) -> T {
        let merged = T::default();
        self.link(merged.clone());
        merged
    }

    /// Link an existing cancellable task/token of type T to this token and return it.
    /// See [Self::link] for more details.
    pub fn linked<T: Cancellable + Clone + 'static>(&self, child: T) -> T {
        self.link(child.clone());
        child
    }
}

impl Cancellable for CancelToken {
    fn cancel(&self) {
        CancelToken::cancel(self)
    }
    fn is_cancelled(&self) -> bool {
        self.is_cancelled()
    }
}

impl<T: Cancellable> Cancellable for Arc<T> {
    fn cancel(&self) {
        self.as_ref().cancel()
    }
    fn is_cancelled(&self) -> bool {
        self.as_ref().is_cancelled()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::thread;
    use std::time::Duration;

    struct TestCancellable {
        cancelled: AtomicBool,
        cancel_count: AtomicUsize,
    }

    impl TestCancellable {
        fn new() -> Self {
            Self {
                cancelled: AtomicBool::new(false),
                cancel_count: AtomicUsize::new(0),
            }
        }

        fn cancel_count(&self) -> usize {
            self.cancel_count.load(Ordering::Relaxed)
        }
    }

    impl Cancellable for TestCancellable {
        fn cancel(&self) {
            if !self.cancelled.swap(true, Ordering::AcqRel) {
                self.cancel_count.fetch_add(1, Ordering::AcqRel);
            }
        }

        fn is_cancelled(&self) -> bool {
            self.cancelled.load(Ordering::Acquire)
        }
    }

    #[test]
    fn test_cancel_token() {
        let token = CancelToken::new();

        token.cancel();

        assert!(token.is_cancelled());
    }

    #[test]
    fn test_cancel_children() {
        let token = CancelToken::new();
        let child1 = Arc::new(TestCancellable::new());
        let child2 = Arc::new(TestCancellable::new());
        token.link(child1.clone());
        token.link(child2.clone());

        token.cancel();

        assert!(child1.is_cancelled());
        assert!(child2.is_cancelled());
    }

    #[test]
    fn test_idempotent_cancel() {
        let token = CancelToken::new();
        let child = Arc::new(TestCancellable::new());
        token.link(child.clone());
        token.cancel();

        token.cancel();

        assert!(token.is_cancelled());
        assert!(child.is_cancelled());
        assert_eq!(child.cancel_count(), 1);
    }

    #[test]
    fn test_concurrent_cancel() {
        let token = CancelToken::new();
        let child = Arc::new(TestCancellable::new());
        token.link(child.clone());
        let token_clone = token.clone();
        let handle = thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            token_clone.cancel();
        });

        token.cancel();

        handle.join().unwrap();
        assert!(token.is_cancelled());
        assert!(child.is_cancelled());
        assert_eq!(child.cancel_count(), 1); // Should only be cancelled once
    }

    #[test]
    fn test_link_after_cancel() {
        let token = CancelToken::new();
        token.cancel();

        let child = Arc::new(TestCancellable::new());
        token.link(child.clone());

        assert!(child.is_cancelled());
    }
}
