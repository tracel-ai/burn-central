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
    writer: BufWriter<File>,
    state: WalState,
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

impl WalOutbox {
    pub fn new(path: PathBuf) -> Result<Self, String> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                format!(
                    "failed to create telemetry outbox directory '{}': {e}",
                    parent.display()
                )
            })?;
        }

        let state = load_state_from_wal(&path)?;
        let writer_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| {
                format!(
                    "failed to open telemetry outbox wal file '{}': {e}",
                    path.display()
                )
            })?;

        Ok(Self {
            inner: Mutex::new(WalOutboxInner {
                writer: BufWriter::new(writer_file),
                state,
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
        append_entry(&mut inner_guard.writer, &entry)?;
        inner_guard.state.next_id += 1;
        if let WalEntry::Enqueue { id, event } = entry {
            inner_guard.state.pending.insert(id, event);
        }

        Ok(())
    }

    fn claim(&self, count: usize) -> Result<Vec<(OutboxId, TelemetryEvent)>, String> {
        if count == 0 {
            return Ok(Vec::new());
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

        let mut claimed = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(event) = inner_guard.state.pending.remove(&id) {
                inner_guard.state.inflight.insert(id, event.clone());
                claimed.push((id, event));
            }
        }

        Ok(claimed)
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

        append_entry(&mut inner_guard.writer, &WalEntry::Complete { id })?;
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

        let first_claim = outbox.claim(10).expect("claim should return queued row");
        assert_eq!(first_claim.len(), 1);
        let id = first_claim[0].0;

        outbox
            .fail(id, "upstream unavailable")
            .expect("fail should requeue row");

        let second_claim = outbox
            .claim(10)
            .expect("failed row should be claimable again");
        assert_eq!(second_claim.len(), 1);
        assert_eq!(second_claim[0].0, id);

        outbox
            .complete(id)
            .expect("complete should remove claimed row");

        let final_claim = outbox.claim(10).expect("claim should still succeed");
        assert!(final_claim.is_empty());

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

            let claimed = outbox.claim(1).expect("claim should return queued row");
            assert_eq!(claimed.len(), 1);
        }

        let outbox = WalOutbox::new(wal_path).expect("wal outbox should reopen");
        let recovered = outbox
            .claim(1)
            .expect("claimed row should be available again after restart");
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

            let claimed = outbox.claim(1).expect("claim should return queued row");
            assert_eq!(claimed.len(), 1);
            outbox
                .complete(claimed[0].0)
                .expect("complete should write wal completion");
        }

        let outbox = WalOutbox::new(wal_path).expect("wal outbox should reopen");
        let claimed = outbox.claim(1).expect("claim should succeed after restart");
        assert!(claimed.is_empty());

        drop(outbox);
        remove_dir(&dir);
    }
}
