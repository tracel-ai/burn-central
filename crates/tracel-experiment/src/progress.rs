//! Progress tracking primitives for experiment runs.
//!
//! Progress is modeled as a tree of named nodes. Starting a node emits a
//! [`ProgressEvent::Started`] event, updates emit [`ProgressEvent::Updated`], and
//! explicit or drop-based completion emits [`ProgressEvent::Finished`].

use std::{
    num::NonZeroU64,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

/// Opaque non-zero identifier for a progress node.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash)]
pub struct ProgressId(NonZeroU64);

impl ProgressId {
    /// Create an identifier from a non-zero numeric value.
    pub fn new(id: NonZeroU64) -> Self {
        Self(id)
    }

    /// Return the underlying numeric identifier.
    pub fn as_u64(self) -> u64 {
        self.0.get()
    }
}

/// Metadata describing a progress node when it starts.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProgressNode {
    /// Unique identifier for this node.
    pub id: ProgressId,
    /// Parent node identifier, when this node is nested under another node.
    pub parent: Option<ProgressId>,
    /// Human-readable node name.
    pub name: String,
    /// Optional unit for progress values, such as `steps` or `bytes`.
    pub unit: Option<String>,
    /// Optional expected total for the node.
    pub total: Option<u64>,
    /// Extra structured metadata attached by the caller.
    pub attributes: serde_json::Map<String, serde_json::Value>,
}

/// Terminal state for a progress node.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum ProgressStatus {
    /// The node completed successfully.
    Success,
    /// The node stopped before successful completion.
    Abandoned,
}

/// Event emitted by a progress node.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type")]
pub enum ProgressEvent {
    /// A node was started.
    Started {
        /// The started node metadata.
        node: ProgressNode,
    },
    /// A node's numeric progress changed.
    Updated {
        /// The updated node identifier.
        id: ProgressId,
        /// The current progress value.
        current: u64,
        /// The expected total, if known.
        total: Option<u64>,
    },
    /// A node emitted a human-readable message.
    Message {
        /// The node identifier.
        id: ProgressId,
        /// The message text.
        message: String,
    },
    /// A node reached a terminal state.
    Finished {
        /// The finished node identifier.
        id: ProgressId,
        /// The terminal status.
        status: ProgressStatus,
        /// Optional completion message.
        message: Option<String>,
    },
}

/// Sink for progress events.
pub trait ProgressEventReporter: Send + Sync {
    /// Report one progress event.
    fn report(&self, event: ProgressEvent);
}

impl<F> ProgressEventReporter for F
where
    F: Fn(ProgressEvent) + Send + Sync,
{
    fn report(&self, event: ProgressEvent) {
        self(event);
    }
}

/// Allocates unique progress node identifiers.
pub trait ProgressIdAllocator: Send + Sync {
    /// Return the next identifier.
    fn next_id(&self) -> ProgressId;
}

/// Lock-free progress identifier allocator.
#[derive(Debug)]
pub struct AtomicProgressIdAllocator {
    next: AtomicU64,
}

impl AtomicProgressIdAllocator {
    /// Create an allocator that starts at identifier `1`.
    pub fn new() -> Self {
        Self {
            next: AtomicU64::new(1),
        }
    }
}

impl Default for AtomicProgressIdAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl ProgressIdAllocator for AtomicProgressIdAllocator {
    fn next_id(&self) -> ProgressId {
        let id = self.next.fetch_add(1, Ordering::Relaxed);

        // Starts at 1, so this should only fail after overflow or wraparound.
        let id = NonZeroU64::new(id).expect("progress id allocator overflowed or produced zero");

        ProgressId(id)
    }
}

/// Builder used to configure and start a progress node.
pub struct ProgressBuilder {
    reporter: Arc<dyn ProgressEventReporter>,
    id_allocator: Arc<dyn ProgressIdAllocator>,
    parent: Option<ProgressId>,
    name: String,
    unit: Option<String>,
    total: Option<u64>,
    attributes: serde_json::Map<String, serde_json::Value>,
}

impl ProgressBuilder {
    /// Create a root progress builder.
    pub(crate) fn new(
        reporter: Arc<dyn ProgressEventReporter>,
        id_allocator: Arc<dyn ProgressIdAllocator>,
        name: impl Into<String>,
    ) -> Self {
        Self {
            reporter,
            id_allocator,
            parent: None,
            name: name.into(),
            unit: None,
            total: None,
            attributes: serde_json::Map::new(),
        }
    }

    /// Set the expected total for this node.
    pub fn total(mut self, total: u64) -> Self {
        self.total = Some(total);
        self
    }

    /// Set the unit used by progress values.
    pub fn unit(mut self, unit: impl Into<String>) -> Self {
        self.unit = Some(unit.into());
        self
    }

    /// Add one serializable attribute.
    pub fn attr<T>(mut self, key: impl Into<String>, value: T) -> Result<Self, serde_json::Error>
    where
        T: serde::Serialize,
    {
        self.attributes
            .insert(key.into(), serde_json::to_value(value)?);
        Ok(self)
    }

    /// Add pre-serialized attributes.
    pub fn attrs<T>(mut self, attributes: T) -> Result<Self, serde_json::Error>
    where
        T: IntoIterator<Item = (String, serde_json::Value)>,
    {
        for (key, value) in attributes {
            self.attributes.insert(key, value);
        }
        Ok(self)
    }

    /// Start this node and return a guard for sending updates.
    pub fn start(self) -> ProgressGuard {
        let id = self.id_allocator.next_id();

        let node = ProgressNode {
            id,
            parent: self.parent,
            name: self.name,
            unit: self.unit,
            total: self.total,
            attributes: self.attributes,
        };

        self.reporter
            .report(ProgressEvent::Started { node: node.clone() });

        ProgressGuard::new(self.reporter, self.id_allocator, node)
    }
}

/// Active progress node that reports updates and completion.
pub struct ProgressGuard {
    reporter: Arc<dyn ProgressEventReporter>,
    id_allocator: Arc<dyn ProgressIdAllocator>,
    node: ProgressNode,
    current: u64,
    finished: bool,
}

impl ProgressGuard {
    /// Create a guard for an already-started node.
    fn new(
        reporter: Arc<dyn ProgressEventReporter>,
        id_allocator: Arc<dyn ProgressIdAllocator>,
        node: ProgressNode,
    ) -> Self {
        Self {
            reporter,
            id_allocator,
            node,
            current: 0,
            finished: false,
        }
    }

    /// Create a builder for a child node.
    pub fn child(&self, name: impl Into<String>) -> ProgressBuilder {
        ProgressBuilder {
            reporter: self.reporter.clone(),
            id_allocator: self.id_allocator.clone(),
            parent: Some(self.node.id),
            name: name.into(),
            unit: None,
            total: None,
            attributes: serde_json::Map::new(),
        }
    }

    /// Increase the current progress value by `delta`.
    pub fn inc(&mut self, delta: u64) {
        self.current = self.current.saturating_add(delta);

        self.reporter.report(ProgressEvent::Updated {
            id: self.node.id,
            current: self.current,
            total: self.node.total,
        });
    }

    /// Set the current progress value.
    pub fn set(&mut self, current: u64) {
        self.current = current;

        self.reporter.report(ProgressEvent::Updated {
            id: self.node.id,
            current,
            total: self.node.total,
        });
    }

    /// Emit a message for this node.
    pub fn message(&self, message: impl Into<String>) {
        self.reporter.report(ProgressEvent::Message {
            id: self.node.id,
            message: message.into(),
        });
    }

    /// Mark the node as successful.
    pub fn finish(mut self) {
        self.finish_inner(ProgressStatus::Success, None);
    }

    /// Mark the node as successful with a message.
    pub fn finish_with_message(mut self, message: impl Into<String>) {
        self.finish_inner(ProgressStatus::Success, Some(message.into()));
    }

    /// Mark the node as abandoned.
    pub fn abandon(mut self) {
        self.finish_inner(ProgressStatus::Abandoned, None);
    }

    /// Mark the node as abandoned with a message.
    pub fn abandon_with_message(mut self, message: impl Into<String>) {
        self.finish_inner(ProgressStatus::Abandoned, Some(message.into()));
    }

    fn finish_inner(&mut self, status: ProgressStatus, message: Option<String>) {
        if self.finished {
            return;
        }

        self.finished = true;
        self.reporter.report(ProgressEvent::Finished {
            id: self.node.id,
            status,
            message,
        });
    }
}

impl Drop for ProgressGuard {
    fn drop(&mut self) {
        self.finish_inner(ProgressStatus::Abandoned, None);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use serde_json::json;

    use super::*;

    #[derive(Default)]
    struct MockReporter {
        events: Mutex<Vec<ProgressEvent>>,
    }

    impl MockReporter {
        fn events(&self) -> Vec<ProgressEvent> {
            self.events.lock().unwrap().clone()
        }
    }

    impl ProgressEventReporter for MockReporter {
        fn report(&self, event: ProgressEvent) {
            self.events.lock().unwrap().push(event);
        }
    }

    fn builder(reporter: Arc<MockReporter>, name: &str) -> ProgressBuilder {
        ProgressBuilder::new(
            reporter,
            Arc::new(AtomicProgressIdAllocator::new()),
            name.to_string(),
        )
    }

    #[test]
    fn start_reports_configured_node() {
        let reporter = Arc::new(MockReporter::default());

        let _guard = builder(reporter.clone(), "load")
            .total(12)
            .unit("items")
            .attr("split", "train")
            .unwrap()
            .start();

        let events = reporter.events();
        let ProgressEvent::Started { node } = &events[0] else {
            panic!("unexpected event: {:?}", events[0]);
        };
        assert_eq!(node.name, "load");
        assert_eq!(node.total, Some(12));
        assert_eq!(node.unit.as_deref(), Some("items"));
        assert_eq!(node.attributes.get("split"), Some(&json!("train")));
    }

    #[test]
    fn child_start_reports_parent_id() {
        let reporter = Arc::new(MockReporter::default());
        let parent = builder(reporter.clone(), "parent").start();

        let _child = parent.child("child").start();

        let events = reporter.events();
        let ProgressEvent::Started { node: parent } = events[0].clone() else {
            panic!("unexpected event: {:?}", events[0]);
        };
        let ProgressEvent::Started { node: child } = events[1].clone() else {
            panic!("unexpected event: {:?}", events[1]);
        };
        assert_eq!(child.parent, Some(parent.id));
    }

    #[test]
    fn inc_reports_updated_progress() {
        let reporter = Arc::new(MockReporter::default());
        let mut guard = builder(reporter.clone(), "items").total(8).start();

        guard.inc(3);

        let events = reporter.events();
        let ProgressEvent::Updated { current, total, .. } = &events[1] else {
            panic!("unexpected event: {:?}", events[1]);
        };
        assert_eq!((*current, *total), (3, Some(8)));
    }

    #[test]
    fn finish_reports_one_success_completion() {
        let reporter = Arc::new(MockReporter::default());

        builder(reporter.clone(), "node").start().finish();

        let events = reporter.events();
        let finished: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                ProgressEvent::Finished { status, .. } => Some(status),
                _ => None,
            })
            .collect();
        assert!(matches!(finished.as_slice(), [ProgressStatus::Success]));
    }

    #[test]
    fn drop_reports_abandoned_completion() {
        let reporter = Arc::new(MockReporter::default());

        drop(builder(reporter.clone(), "node").start());

        let events = reporter.events();
        assert!(matches!(
            events.last(),
            Some(ProgressEvent::Finished {
                status: ProgressStatus::Abandoned,
                ..
            })
        ));
    }
}
