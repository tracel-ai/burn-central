//! Activity tracking primitives for experiment runs.
//!
//! Progress is modeled as a tree of named activities. Starting an activity emits a
//! [`ActivityEvent::Started`] event, metered updates emit [`ActivityEvent::Updated`],
//! and explicit or drop-based completion emits [`ActivityEvent::Finished`].

use std::{
    num::NonZeroU64,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

/// Opaque non-zero identifier for an activity.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash)]
pub struct ActivityId(NonZeroU64);

impl ActivityId {
    /// Create an identifier from a non-zero numeric value.
    pub fn new(id: NonZeroU64) -> Self {
        Self(id)
    }

    /// Return the underlying numeric identifier.
    pub fn as_u64(self) -> u64 {
        self.0.get()
    }
}

/// Numeric meter definition for an activity.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ActivityMeter {
    /// Optional unit for progress values, such as `steps` or `bytes`.
    pub unit: Option<String>,
    /// Optional expected total for the activity.
    pub total: Option<u64>,
}

/// Metadata describing an activity when it starts.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Activity {
    /// Unique identifier for this activity.
    pub id: ActivityId,
    /// Parent activity identifier, when this activity is nested under another activity.
    pub parent: Option<ActivityId>,
    /// Human-readable activity name.
    pub name: String,
    /// Numeric meter definition, when this activity has its own meter.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub meter: Option<ActivityMeter>,
    /// Extra structured metadata attached by the caller.
    #[serde(default)]
    pub attributes: serde_json::Map<String, serde_json::Value>,
}

/// Terminal state for an activity.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum ActivityStatus {
    /// The activity completed successfully.
    Success,
    /// The activity stopped before successful completion.
    Abandoned,
}

/// Event emitted by an activity.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type")]
pub enum ActivityEvent {
    /// An activity was started.
    Started {
        /// The started activity metadata.
        activity: Activity,
    },
    /// An activity's numeric progress changed.
    Updated {
        /// The updated activity identifier.
        id: ActivityId,
        /// The current progress value.
        current: u64,
    },
    /// An activity emitted a human-readable message.
    Message {
        /// The activity identifier.
        id: ActivityId,
        /// The message text.
        message: String,
    },
    /// An activity reached a terminal state.
    Finished {
        /// The finished activity identifier.
        id: ActivityId,
        /// The terminal status.
        status: ActivityStatus,
        /// Optional completion message.
        message: Option<String>,
    },
}

/// Sink for activity events.
pub trait ActivityEventReporter: Send + Sync {
    /// Report one progress event.
    fn report(&self, event: ActivityEvent);
}

impl<F> ActivityEventReporter for F
where
    F: Fn(ActivityEvent) + Send + Sync,
{
    fn report(&self, event: ActivityEvent) {
        self(event);
    }
}

/// Allocates unique activity identifiers.
pub trait ActivityIdAllocator: Send + Sync {
    /// Return the next identifier.
    fn next_id(&self) -> ActivityId;
}

/// Lock-free activity identifier allocator.
#[derive(Debug)]
pub struct AtomicActivityIdAllocator {
    next: AtomicU64,
}

impl AtomicActivityIdAllocator {
    /// Create an allocator that starts at identifier `1`.
    pub fn new() -> Self {
        Self {
            next: AtomicU64::new(1),
        }
    }
}

impl Default for AtomicActivityIdAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl ActivityIdAllocator for AtomicActivityIdAllocator {
    fn next_id(&self) -> ActivityId {
        let id = self.next.fetch_add(1, Ordering::Relaxed);

        // Starts at 1, so this should only fail after overflow or wraparound.
        let id = NonZeroU64::new(id).expect("activity id allocator overflowed or produced zero");

        ActivityId(id)
    }
}

/// Typestate marker for an activity without a numeric meter.
#[derive(Debug, Clone, Copy, Default)]
pub struct Unmetered;

/// Typestate marker for an activity with a numeric meter.
#[derive(Debug, Clone)]
pub struct Metered {
    meter: ActivityMeter,
    current: u64,
}

/// Builder used to configure and start an activity.
pub struct ActivityBuilder<State = Unmetered> {
    reporter: Arc<dyn ActivityEventReporter>,
    id_allocator: Arc<dyn ActivityIdAllocator>,
    parent: Option<ActivityId>,
    name: String,
    attributes: serde_json::Map<String, serde_json::Value>,
    state: State,
}

impl ActivityBuilder<Unmetered> {
    /// Create a root activity builder.
    pub(crate) fn new(
        reporter: Arc<dyn ActivityEventReporter>,
        id_allocator: Arc<dyn ActivityIdAllocator>,
        name: impl Into<String>,
    ) -> Self {
        Self {
            reporter,
            id_allocator,
            parent: None,
            name: name.into(),
            attributes: serde_json::Map::new(),
            state: Unmetered,
        }
    }

    /// Configure this activity with a numeric meter.
    pub fn progress(self) -> ActivityBuilder<Metered> {
        let ActivityBuilder {
            reporter,
            id_allocator,
            parent,
            name,
            attributes,
            state: _,
        } = self;

        ActivityBuilder {
            reporter,
            id_allocator,
            parent,
            name,
            attributes,
            state: Metered {
                meter: ActivityMeter {
                    unit: None,
                    total: None,
                },
                current: 0,
            },
        }
    }

    /// Start this activity and return a guard for child activities and completion.
    pub fn start(self) -> ActivityGuard<Unmetered> {
        ActivityGuard {
            inner: self.start_inner(None),
            state: Unmetered,
        }
    }
}

impl ActivityBuilder<Metered> {
    /// Set the expected total for this activity's meter.
    pub fn total(mut self, total: u64) -> Self {
        self.state.meter.total = Some(total);
        self
    }

    /// Set the unit used by this activity's meter.
    pub fn unit(mut self, unit: impl Into<String>) -> Self {
        self.state.meter.unit = Some(unit.into());
        self
    }

    /// Start this activity and return a guard for sending metered updates.
    pub fn start(self) -> ActivityGuard<Metered> {
        let state = self.state.clone();

        ActivityGuard {
            inner: self.start_inner(Some(state.meter.clone())),
            state,
        }
    }
}

impl<State> ActivityBuilder<State> {
    /// Add one serializable attribute.
    pub fn attr<T>(mut self, key: impl Into<String>, value: T) -> Result<Self, serde_json::Error>
    where
        T: serde::Serialize,
    {
        self.insert_attr(key, value)?;
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

    fn insert_attr<T>(&mut self, key: impl Into<String>, value: T) -> Result<(), serde_json::Error>
    where
        T: serde::Serialize,
    {
        self.attributes
            .insert(key.into(), serde_json::to_value(value)?);
        Ok(())
    }

    fn start_inner(self, meter: Option<ActivityMeter>) -> ActiveActivity {
        let id = self.id_allocator.next_id();

        let activity = Activity {
            id,
            parent: self.parent,
            name: self.name,
            meter,
            attributes: self.attributes,
        };

        self.reporter.report(ActivityEvent::Started {
            activity: activity.clone(),
        });

        ActiveActivity::new(self.reporter, self.id_allocator, activity)
    }
}

/// Active activity state shared by builders and guards.
struct ActiveActivity {
    reporter: Arc<dyn ActivityEventReporter>,
    id_allocator: Arc<dyn ActivityIdAllocator>,
    activity: Activity,
    finished: bool,
}

impl ActiveActivity {
    /// Create a guard for an already-started activity.
    fn new(
        reporter: Arc<dyn ActivityEventReporter>,
        id_allocator: Arc<dyn ActivityIdAllocator>,
        activity: Activity,
    ) -> Self {
        Self {
            reporter,
            id_allocator,
            activity,
            finished: false,
        }
    }

    /// Create a builder for a child activity.
    fn activity(&self, name: impl Into<String>) -> ActivityBuilder {
        ActivityBuilder {
            reporter: self.reporter.clone(),
            id_allocator: self.id_allocator.clone(),
            parent: Some(self.activity.id),
            name: name.into(),
            attributes: serde_json::Map::new(),
            state: Unmetered,
        }
    }

    /// Emit a message for this activity.
    fn message(&self, message: impl Into<String>) {
        self.reporter.report(ActivityEvent::Message {
            id: self.activity.id,
            message: message.into(),
        });
    }

    fn finish_inner(&mut self, status: ActivityStatus, message: Option<String>) {
        if self.finished {
            return;
        }

        self.finished = true;
        self.reporter.report(ActivityEvent::Finished {
            id: self.activity.id,
            status,
            message,
        });
    }
}

impl Drop for ActiveActivity {
    fn drop(&mut self) {
        self.finish_inner(ActivityStatus::Abandoned, None);
    }
}

/// Active activity that can own child activities and completion.
pub struct ActivityGuard<State = Unmetered> {
    inner: ActiveActivity,
    state: State,
}

impl<State> ActivityGuard<State> {
    /// Return the activity identifier.
    pub fn id(&self) -> ActivityId {
        self.inner.activity.id
    }

    /// Create a builder for a child activity without a numeric meter.
    pub fn activity(&self, name: impl Into<String>) -> ActivityBuilder {
        self.inner.activity(name)
    }

    /// Emit a message for this activity.
    pub fn message(&self, message: impl Into<String>) {
        self.inner.message(message);
    }

    /// Mark the activity as successful.
    pub fn finish(mut self) {
        self.inner.finish_inner(ActivityStatus::Success, None);
    }

    /// Mark the activity as successful with a message.
    pub fn finish_with_message(mut self, message: impl Into<String>) {
        self.inner
            .finish_inner(ActivityStatus::Success, Some(message.into()));
    }

    /// Mark the activity as abandoned.
    pub fn abandon(mut self) {
        self.inner.finish_inner(ActivityStatus::Abandoned, None);
    }

    /// Mark the activity as abandoned with a message.
    pub fn abandon_with_message(mut self, message: impl Into<String>) {
        self.inner
            .finish_inner(ActivityStatus::Abandoned, Some(message.into()));
    }
}

impl ActivityGuard<Metered> {
    /// Increase the current progress value by `delta`.
    pub fn inc(&mut self, delta: u64) {
        self.state.current = self.state.current.saturating_add(delta);
        self.report_update();
    }

    /// Set the current progress value.
    pub fn set(&mut self, current: u64) {
        self.state.current = current;
        self.report_update();
    }

    fn report_update(&self) {
        self.inner.reporter.report(ActivityEvent::Updated {
            id: self.inner.activity.id,
            current: self.state.current,
        });
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use serde_json::json;

    use super::*;

    #[derive(Default)]
    struct MockReporter {
        events: Mutex<Vec<ActivityEvent>>,
    }

    impl MockReporter {
        fn events(&self) -> Vec<ActivityEvent> {
            self.events.lock().unwrap().clone()
        }
    }

    impl ActivityEventReporter for MockReporter {
        fn report(&self, event: ActivityEvent) {
            self.events.lock().unwrap().push(event);
        }
    }

    fn builder(reporter: Arc<MockReporter>, name: &str) -> ActivityBuilder<Metered> {
        ActivityBuilder::new(
            reporter,
            Arc::new(AtomicActivityIdAllocator::new()),
            name.to_string(),
        )
        .progress()
    }

    #[test]
    fn start_reports_configured_activity() {
        let reporter = Arc::new(MockReporter::default());

        let _guard = builder(reporter.clone(), "load")
            .total(12)
            .unit("items")
            .attr("split", "train")
            .unwrap()
            .start();

        let events = reporter.events();
        let ActivityEvent::Started { activity } = &events[0] else {
            panic!("unexpected event: {:?}", events[0]);
        };
        assert_eq!(activity.name, "load");
        let meter = activity.meter.as_ref().expect("expected activity meter");
        assert_eq!(meter.total, Some(12));
        assert_eq!(meter.unit.as_deref(), Some("items"));
        assert_eq!(activity.attributes.get("split"), Some(&json!("train")));
    }

    #[test]
    fn activity_start_reports_no_meter() {
        let reporter = Arc::new(MockReporter::default());

        let _guard = ActivityBuilder::new(
            reporter.clone(),
            Arc::new(AtomicActivityIdAllocator::new()),
            "epoch",
        )
        .start();

        let events = reporter.events();
        let ActivityEvent::Started { activity } = &events[0] else {
            panic!("unexpected event: {:?}", events[0]);
        };
        assert_eq!(activity.name, "epoch");
        assert!(activity.meter.is_none());
    }

    #[test]
    fn child_activity_start_reports_parent_id() {
        let reporter = Arc::new(MockReporter::default());
        let parent = builder(reporter.clone(), "parent").start();

        let _child = parent.activity("child").start();

        let events = reporter.events();
        let ActivityEvent::Started { activity: parent } = events[0].clone() else {
            panic!("unexpected event: {:?}", events[0]);
        };
        let ActivityEvent::Started { activity: child } = events[1].clone() else {
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
        let ActivityEvent::Updated { current, .. } = &events[1] else {
            panic!("unexpected event: {:?}", events[1]);
        };
        assert_eq!(*current, 3);
    }

    #[test]
    fn finish_reports_one_success_completion() {
        let reporter = Arc::new(MockReporter::default());

        builder(reporter.clone(), "node").start().finish();

        let events = reporter.events();
        let finished: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                ActivityEvent::Finished { status, .. } => Some(status),
                _ => None,
            })
            .collect();
        assert!(matches!(finished.as_slice(), [ActivityStatus::Success]));
    }

    #[test]
    fn drop_reports_abandoned_completion() {
        let reporter = Arc::new(MockReporter::default());

        drop(builder(reporter.clone(), "node").start());

        let events = reporter.events();
        assert!(matches!(
            events.last(),
            Some(ActivityEvent::Finished {
                status: ActivityStatus::Abandoned,
                ..
            })
        ));
    }
}
