use std::{
    collections::{BTreeMap, HashMap},
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    sync::Mutex,
};

use crate::telemetry::{
    event::TelemetryEvent,
    pipeline::{Outbox, OutboxId},
};

#[derive(Debug)]
pub struct WalOutbox {
    inner: Mutex<WalOutboxInner>,
}

#[derive(Debug)]
struct WalOutboxInner {
    path: PathBuf,
    writer: Option<BufWriter<File>>,
    state: WalState,
    completed_since_compaction: usize,
    compaction_min_completes: usize,
    compaction_min_size_bytes: u64,
}

#[derive(Debug, Default)]
struct WalState {
    next_id: OutboxId,
    pending: BTreeMap<OutboxId, TelemetryEvent>,
    inflight: HashMap<OutboxId, TelemetryEvent>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum WalEntry {
    Enqueue { id: OutboxId, event: TelemetryEvent },
    Complete { id: OutboxId },
}

const DEFAULT_COMPACTION_MIN_COMPLETES: usize = 512;
const DEFAULT_COMPACTION_MIN_SIZE_BYTES: u64 = 8 * 1024 * 1024;

#[derive(Debug, Clone, Copy)]
struct WalOutboxOptions {
    compaction_min_completes: usize,
    compaction_min_size_bytes: u64,
}

impl Default for WalOutboxOptions {
    fn default() -> Self {
        Self {
            compaction_min_completes: DEFAULT_COMPACTION_MIN_COMPLETES,
            compaction_min_size_bytes: DEFAULT_COMPACTION_MIN_SIZE_BYTES,
        }
    }
}

impl WalOutbox {
    pub fn new(path: PathBuf) -> Result<Self, String> {
        Self::new_with_options(path, WalOutboxOptions::default())
    }

    fn new_with_options(path: PathBuf, options: WalOutboxOptions) -> Result<Self, String> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                format!(
                    "failed to create telemetry outbox directory '{}': {e}",
                    parent.display()
                )
            })?;
        }

        let state = load_state_from_wal(&path)?;
        let writer_file = open_wal_append_file(&path)?;

        Ok(Self {
            inner: Mutex::new(WalOutboxInner {
                path,
                writer: Some(BufWriter::new(writer_file)),
                state,
                completed_since_compaction: 0,
                compaction_min_completes: options.compaction_min_completes,
                compaction_min_size_bytes: options.compaction_min_size_bytes,
            }),
        })
    }
}

impl Outbox for WalOutbox {
    fn enqueue(&self, data: TelemetryEvent) -> Result<(), String> {
        let mut inner_guard = self
            .inner
            .lock()
            .map_err(|_| "wal outbox lock poisoned".to_string())?;

        let id = inner_guard.state.next_id;
        let entry = WalEntry::Enqueue { id, event: data };
        let writer = inner_guard
            .writer
            .as_mut()
            .ok_or_else(|| "wal writer unavailable".to_string())?;
        append_entry(writer, &entry)?;
        inner_guard.state.next_id += 1;
        if let WalEntry::Enqueue { id, event } = entry {
            inner_guard.state.pending.insert(id, event);
        }

        Ok(())
    }

    fn claim(&self, count: usize) -> Result<Option<Vec<(OutboxId, TelemetryEvent)>>, String> {
        if count == 0 {
            return Ok(None);
        }

        let mut inner_guard = self
            .inner
            .lock()
            .map_err(|_| "wal outbox lock poisoned".to_string())?;

        let ids = inner_guard
            .state
            .pending
            .keys()
            .copied()
            .take(count)
            .collect::<Vec<_>>();

        if ids.is_empty() {
            return Ok(None);
        }

        let mut claimed = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(event) = inner_guard.state.pending.remove(&id) {
                inner_guard.state.inflight.insert(id, event.clone());
                claimed.push((id, event));
            }
        }

        Ok(Some(claimed))
    }

    fn complete(&self, id: OutboxId) -> Result<(), String> {
        let mut inner_guard = self
            .inner
            .lock()
            .map_err(|_| "wal outbox lock poisoned".to_string())?;

        let removed = inner_guard.state.inflight.remove(&id).is_some()
            || inner_guard.state.pending.remove(&id).is_some();
        if !removed {
            return Ok(());
        }

        let writer = inner_guard
            .writer
            .as_mut()
            .ok_or_else(|| "wal writer unavailable".to_string())?;
        append_entry(writer, &WalEntry::Complete { id })?;
        inner_guard.completed_since_compaction =
            inner_guard.completed_since_compaction.saturating_add(1);
        if let Err(err) = maybe_compact(&mut inner_guard) {
            tracing::warn!(error = %err, "failed to compact telemetry wal outbox");
        }
        Ok(())
    }

    fn fail(&self, id: OutboxId, _error: &str) -> Result<(), String> {
        let mut inner_guard = self
            .inner
            .lock()
            .map_err(|_| "wal outbox lock poisoned".to_string())?;

        let Some(event) = inner_guard.state.inflight.remove(&id) else {
            return Ok(());
        };

        inner_guard.state.pending.insert(id, event);
        Ok(())
    }
}

fn append_entry(writer: &mut BufWriter<File>, entry: &WalEntry) -> Result<(), String> {
    serde_json::to_writer(&mut *writer, entry)
        .map_err(|e| format!("failed to write wal entry payload: {e}"))?;
    writer
        .write_all(b"\n")
        .map_err(|e| format!("failed to write wal entry delimiter: {e}"))?;
    writer
        .flush()
        .map_err(|e| format!("failed to flush wal entry: {e}"))?;
    Ok(())
}

fn maybe_compact(inner: &mut WalOutboxInner) -> Result<(), String> {
    if inner.completed_since_compaction < inner.compaction_min_completes {
        return Ok(());
    }

    let wal_size = fs::metadata(&inner.path)
        .map_err(|e| {
            format!(
                "failed to stat telemetry wal '{}': {e}",
                inner.path.display()
            )
        })?
        .len();
    if wal_size < inner.compaction_min_size_bytes {
        return Ok(());
    }

    compact(inner)
}

fn compact(inner: &mut WalOutboxInner) -> Result<(), String> {
    let live_entries = live_entries(&inner.state);

    let mut current_writer = inner
        .writer
        .take()
        .ok_or_else(|| "wal writer unavailable".to_string())?;
    current_writer
        .flush()
        .map_err(|e| format!("failed to flush telemetry wal before compaction: {e}"))?;
    drop(current_writer);

    let compact_result = rewrite_compacted_wal(&inner.path, &live_entries);
    let reopen_result = open_wal_append_file(&inner.path);

    match reopen_result {
        Ok(file) => inner.writer = Some(BufWriter::new(file)),
        Err(e) => {
            let message = match compact_result {
                Ok(()) => e,
                Err(compaction_err) => format!("{compaction_err}; also failed to reopen wal: {e}"),
            };
            return Err(message);
        }
    }

    compact_result?;
    inner.completed_since_compaction = 0;
    Ok(())
}

fn live_entries(state: &WalState) -> BTreeMap<OutboxId, TelemetryEvent> {
    let mut entries = BTreeMap::new();
    for (id, event) in &state.pending {
        entries.insert(*id, event.clone());
    }
    for (id, event) in &state.inflight {
        entries.insert(*id, event.clone());
    }
    entries
}

fn rewrite_compacted_wal(
    path: &Path,
    live_entries: &BTreeMap<OutboxId, TelemetryEvent>,
) -> Result<(), String> {
    let tmp_path = wal_compaction_path(path);
    {
        let compact_file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&tmp_path)
            .map_err(|e| {
                format!(
                    "failed to open compacted telemetry wal '{}': {e}",
                    tmp_path.display()
                )
            })?;
        let mut compact_writer = BufWriter::new(compact_file);

        for (id, event) in live_entries {
            append_entry(
                &mut compact_writer,
                &WalEntry::Enqueue {
                    id: *id,
                    event: event.clone(),
                },
            )?;
        }
        compact_writer
            .flush()
            .map_err(|e| format!("failed to flush compacted telemetry wal: {e}"))?;
    }

    fs::rename(&tmp_path, path).map_err(|e| {
        format!(
            "failed to replace telemetry wal '{}' with compacted file '{}': {e}",
            path.display(),
            tmp_path.display()
        )
    })?;

    Ok(())
}

fn wal_compaction_path(path: &Path) -> PathBuf {
    path.with_extension("wal.compact")
}

fn open_wal_append_file(path: &Path) -> Result<File, String> {
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| {
            format!(
                "failed to open telemetry outbox wal file '{}': {e}",
                path.display()
            )
        })
}

fn load_state_from_wal(path: &Path) -> Result<WalState, String> {
    if !path.exists() {
        return Ok(WalState::default());
    }

    let file = File::open(path).map_err(|e| {
        format!(
            "failed to open telemetry outbox wal '{}': {e}",
            path.display()
        )
    })?;
    let reader = BufReader::new(file);

    let mut state = WalState::default();
    for line_result in reader.lines() {
        let line = line_result.map_err(|e| {
            format!(
                "failed to read telemetry outbox wal line from '{}': {e}",
                path.display()
            )
        })?;
        if line.trim().is_empty() {
            continue;
        }

        let entry: WalEntry = match serde_json::from_str(&line) {
            Ok(entry) => entry,
            Err(e) => {
                tracing::warn!(
                    path = %path.display(),
                    error = %e,
                    "skipping malformed telemetry wal entry"
                );
                continue;
            }
        };

        match entry {
            WalEntry::Enqueue { id, event } => {
                state.next_id = state.next_id.max(id.saturating_add(1));
                state.pending.insert(id, event);
            }
            WalEntry::Complete { id } => {
                state.pending.remove(&id);
                state.inflight.remove(&id);
            }
        }
    }

    Ok(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry::metrics::MetricBatch;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "burn-central-fleet-telemetry-{name}-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ))
    }

    fn sample_event() -> TelemetryEvent {
        TelemetryEvent::metrics(MetricBatch {
            counters: Vec::new(),
            gauges: Vec::new(),
            histograms: Vec::new(),
        })
    }

    fn remove_dir(path: &Path) {
        let _ = fs::remove_dir_all(path);
    }

    #[test]
    fn wal_outbox_claim_fail_and_complete_cycle() {
        let dir = temp_dir("cycle");
        let wal_path = dir.join("outbox.wal");
        let outbox = WalOutbox::new(wal_path).expect("wal outbox should initialize");

        outbox
            .enqueue(sample_event())
            .expect("enqueue should write wal entry");

        let first_claim = outbox
            .claim(10)
            .expect("claim should return queued row")
            .expect("claim should return Some for available rows");
        assert_eq!(first_claim.len(), 1);
        let id = first_claim[0].0;

        outbox
            .fail(id, "upstream unavailable")
            .expect("fail should requeue row");

        let second_claim = outbox
            .claim(10)
            .expect("failed row should be claimable again")
            .expect("claim should return Some for available rows");
        assert_eq!(second_claim.len(), 1);
        assert_eq!(second_claim[0].0, id);

        outbox
            .complete(id)
            .expect("complete should remove claimed row");

        let final_claim = outbox.claim(10).expect("claim should still succeed");
        assert!(final_claim.is_none());

        drop(outbox);
        remove_dir(&dir);
    }

    #[test]
    fn wal_outbox_recovers_claimed_rows_on_restart() {
        let dir = temp_dir("restart");
        let wal_path = dir.join("outbox.wal");

        {
            let outbox = WalOutbox::new(wal_path.clone()).expect("wal outbox should initialize");
            outbox
                .enqueue(sample_event())
                .expect("enqueue should write wal entry");

            let claimed = outbox
                .claim(1)
                .expect("claim should return queued row")
                .expect("claim should return Some for available rows");
            assert_eq!(claimed.len(), 1);
        }

        let outbox = WalOutbox::new(wal_path).expect("wal outbox should reopen");
        let recovered = outbox
            .claim(1)
            .expect("claimed row should be available again after restart")
            .expect("claim should return Some for available rows");
        assert_eq!(recovered.len(), 1);

        drop(outbox);
        remove_dir(&dir);
    }

    #[test]
    fn wal_outbox_persists_completions_across_restart() {
        let dir = temp_dir("complete");
        let wal_path = dir.join("outbox.wal");

        {
            let outbox = WalOutbox::new(wal_path.clone()).expect("wal outbox should initialize");
            outbox
                .enqueue(sample_event())
                .expect("enqueue should write wal entry");

            let claimed = outbox
                .claim(1)
                .expect("claim should return queued row")
                .expect("claim should return Some for available rows");
            assert_eq!(claimed.len(), 1);
            outbox
                .complete(claimed[0].0)
                .expect("complete should write wal completion");
        }

        let outbox = WalOutbox::new(wal_path).expect("wal outbox should reopen");
        let claimed = outbox.claim(1).expect("claim should succeed after restart");
        assert!(claimed.is_none());

        drop(outbox);
        remove_dir(&dir);
    }

    #[test]
    fn wal_outbox_returns_none_on_claim_when_no_rows_available() {
        let dir = temp_dir("empty");
        let wal_path = dir.join("outbox.wal");
        let outbox = WalOutbox::new(wal_path).expect("wal outbox should initialize");

        let claim = outbox.claim(10).expect("claim should succeed");
        assert!(
            claim.is_none(),
            "claim should return None when no rows available"
        );

        drop(outbox);
        remove_dir(&dir);
    }

    #[test]
    fn wal_outbox_compaction_bounds_file_size_after_completions() {
        let dir = temp_dir("compact-size");
        let wal_path = dir.join("outbox.wal");
        let outbox = WalOutbox::new_with_options(
            wal_path.clone(),
            WalOutboxOptions {
                compaction_min_completes: 1,
                compaction_min_size_bytes: 0,
            },
        )
        .expect("wal outbox should initialize");

        for _ in 0..32 {
            outbox
                .enqueue(sample_event())
                .expect("enqueue should write wal entry");
            let claimed = outbox
                .claim(1)
                .expect("claim should succeed")
                .expect("claim should return row");
            outbox
                .complete(claimed[0].0)
                .expect("complete should succeed");
        }

        drop(outbox);

        let size = fs::metadata(&wal_path).expect("wal should exist").len();
        assert_eq!(size, 0, "compacted wal should be empty when no live rows");

        remove_dir(&dir);
    }

    #[test]
    fn wal_outbox_compaction_preserves_inflight_rows() {
        let dir = temp_dir("compact-inflight");
        let wal_path = dir.join("outbox.wal");
        let outbox = WalOutbox::new_with_options(
            wal_path.clone(),
            WalOutboxOptions {
                compaction_min_completes: 1,
                compaction_min_size_bytes: 0,
            },
        )
        .expect("wal outbox should initialize");

        outbox
            .enqueue(sample_event())
            .expect("first enqueue should succeed");
        outbox
            .enqueue(sample_event())
            .expect("second enqueue should succeed");

        let inflight = outbox
            .claim(1)
            .expect("claim should succeed")
            .expect("claim should return one row");
        let inflight_id = inflight[0].0;

        // Complete the remaining pending row to trigger compaction.
        outbox
            .complete(inflight_id + 1)
            .expect("complete should succeed");

        drop(outbox);

        let reopened = WalOutbox::new(wal_path).expect("wal outbox should reopen");
        let recovered = reopened
            .claim(10)
            .expect("claim should succeed")
            .expect("inflight row should survive compaction");
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].0, inflight_id);

        drop(reopened);
        remove_dir(&dir);
    }
}
