use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::thread::{self, JoinHandle};

use burn_central_artifact::bundle::FsBundle;
use crossbeam::channel::{Sender, unbounded};

use crate::error::{ExperimentError, ExperimentErrorKind};
use crate::reader::{ArtifactRef, ExperimentArtifactReader, ExperimentReaderError, LoadedArtifact};
use crate::session::{BundleFn, Event, ExperimentCompletion, ExperimentSession};
use crate::{ArtifactKind, ExperimentId, ExperimentRun};

pub fn create_experiment_run(root: PathBuf) -> Result<ExperimentRun, ExperimentError> {
    let root = root
        .canonicalize()
        .or_else(|_| {
            fs::create_dir_all(&root)?;
            root.canonicalize()
        })
        .map_err(|err| {
            ExperimentError::with_source(
                ExperimentErrorKind::Internal,
                "Failed to create local experiment directory",
                err,
            )
        })?;
    let (id, run_root) = create_local_run_dir(&root).map_err(|err| {
        ExperimentError::with_source(
            ExperimentErrorKind::Internal,
            "Failed to create local experiment run directory",
            err,
        )
    })?;

    let session = LocalExperimentSession::new(run_root).map_err(|err| {
        ExperimentError::with_source(
            ExperimentErrorKind::Internal,
            "Failed to initialize local experiment session",
            err,
        )
    })?;
    let reader = LocalExperimentReader { root };

    Ok(ExperimentRun::new(id, session, reader, Default::default()))
}

struct LocalExperimentSession {
    root: PathBuf,
    active: Mutex<Option<LocalWorker>>,
}

impl LocalExperimentSession {
    fn new(root: PathBuf) -> Result<Self, std::io::Error> {
        fs::create_dir_all(root.join("artifacts"))?;
        let (sender, receiver) = unbounded();
        let events_path = root.join("events.log");
        let status_path = root.join("status.txt");
        let join = thread::spawn(move || local_worker(receiver, events_path, status_path));

        Ok(Self {
            root,
            active: Mutex::new(Some(LocalWorker { sender, join })),
        })
    }

    fn sender(&self) -> Result<Sender<LocalWrite>, ExperimentError> {
        let guard = self.active.lock().unwrap();
        guard
            .as_ref()
            .map(|worker| worker.sender.clone())
            .ok_or_else(|| {
                ExperimentError::new(
                    ExperimentErrorKind::AlreadyFinished,
                    "Local experiment session has already finished",
                )
            })
    }

    fn finish_worker(&self, completion: ExperimentCompletion) -> Result<(), ExperimentError> {
        let worker = self.active.lock().unwrap().take().ok_or_else(|| {
            ExperimentError::new(
                ExperimentErrorKind::AlreadyFinished,
                "Local experiment session has already finished",
            )
        })?;

        let send_result = worker
            .sender
            .send(LocalWrite::Finish(format!("{completion:?}")));
        let join_result = worker.join.join();

        match join_result {
            Ok(Ok(())) => {
                if send_result.is_err() {
                    return Err(ExperimentError::new(
                        ExperimentErrorKind::Internal,
                        "Failed to send local experiment completion",
                    ));
                }
                Ok(())
            }
            Ok(Err(err)) => Err(ExperimentError::with_source(
                ExperimentErrorKind::Internal,
                "Local experiment writer failed",
                err,
            )),
            Err(_) => Err(ExperimentError::new(
                ExperimentErrorKind::Internal,
                "Local experiment writer thread panicked",
            )),
        }
    }
}

impl ExperimentSession for LocalExperimentSession {
    fn record_event(&self, event: Event) -> Result<(), ExperimentError> {
        self.sender()?
            .send(LocalWrite::Event(format!("{event:?}")))
            .map_err(|_| {
                ExperimentError::new(
                    ExperimentErrorKind::Internal,
                    "Failed to queue local experiment event",
                )
            })
    }

    fn save_artifact(
        &self,
        name: &str,
        _kind: ArtifactKind,
        artifact: Box<BundleFn>,
    ) -> Result<(), ExperimentError> {
        let artifact_root = self.root.join("artifacts").join(name);
        if artifact_root.exists() {
            fs::remove_dir_all(&artifact_root).map_err(|err| {
                ExperimentError::with_source(
                    ExperimentErrorKind::Artifact,
                    "Failed to replace existing local artifact",
                    err,
                )
            })?;
        }

        let mut bundle = FsBundle::create(artifact_root.clone()).map_err(|err| {
            ExperimentError::with_source(
                ExperimentErrorKind::Artifact,
                "Failed to create local artifact bundle",
                err,
            )
        })?;

        let res = artifact(&mut bundle);

        if res.is_err() {
            _ = bundle.delete();
        }

        res
    }

    fn finish(&self, completion: ExperimentCompletion) -> Result<(), ExperimentError> {
        self.finish_worker(completion)
    }
}

struct LocalWorker {
    sender: Sender<LocalWrite>,
    join: JoinHandle<Result<(), std::io::Error>>,
}

enum LocalWrite {
    Event(String),
    Finish(String),
}

struct LocalExperimentReader {
    root: PathBuf,
}

impl ExperimentArtifactReader for LocalExperimentReader {
    fn load_artifact_raw(
        &self,
        experiment_id: ExperimentId,
        name: &str,
    ) -> Result<LoadedArtifact, ExperimentReaderError> {
        let experiment_root = parse_local_experiment_root(&self.root, &experiment_id)?;
        let artifact_root = experiment_root.join("artifacts").join(name);

        if !artifact_root.is_dir() {
            return Err(ExperimentReaderError::new(format!(
                "Local artifact not found: {}",
                artifact_root.display()
            )));
        }

        let files = collect_bundle_files(&artifact_root, &artifact_root).map_err(|err| {
            ExperimentReaderError::with_source("Failed to inspect local artifact files", err)
        })?;
        let bundle = FsBundle::with_files(artifact_root.clone(), files).map_err(|err| {
            ExperimentReaderError::with_source("Failed to create local artifact bundle", err)
        })?;

        Ok(LoadedArtifact::new(
            ArtifactRef {
                id: artifact_root.to_string_lossy().to_string(),
                name: name.to_string(),
            },
            bundle,
        ))
    }
}

fn parse_local_experiment_root(
    root: &Path,
    experiment_id: &ExperimentId,
) -> Result<PathBuf, ExperimentReaderError> {
    let experiment_dir = experiment_id.as_str();
    if experiment_dir.is_empty() || experiment_dir.contains('/') || experiment_dir.contains('\\') {
        return Err(ExperimentReaderError::new(
            "Invalid local experiment ID format",
        ));
    }

    Ok(root.join(experiment_dir))
}

fn append_line(path: PathBuf, line: &str) -> Result<(), std::io::Error> {
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    file.write_all(line.as_bytes())?;
    file.write_all(b"\n")?;
    Ok(())
}

fn local_worker(
    receiver: crossbeam::channel::Receiver<LocalWrite>,
    events_path: PathBuf,
    status_path: PathBuf,
) -> Result<(), std::io::Error> {
    while let Ok(message) = receiver.recv() {
        match message {
            LocalWrite::Event(line) => append_line(events_path.clone(), &line)?,
            LocalWrite::Finish(line) => {
                append_line(status_path, &line)?;
                return Ok(());
            }
        }
    }

    Ok(())
}

fn collect_bundle_files(root: &Path, current: &Path) -> Result<Vec<String>, std::io::Error> {
    let mut files = Vec::new();
    for entry in fs::read_dir(current)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            files.extend(collect_bundle_files(root, &path)?);
            continue;
        }

        let rel = path
            .strip_prefix(root)
            .map_err(std::io::Error::other)?
            .to_string_lossy()
            .to_string();
        files.push(rel);
    }
    Ok(files)
}

fn create_local_run_dir(root: &Path) -> Result<(ExperimentId, PathBuf), std::io::Error> {
    let mut next_id = 1u64;

    for entry in fs::read_dir(root)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }

        if let Some(name) = entry.file_name().to_str() {
            if let Ok(id) = name.parse::<u64>() {
                next_id = next_id.max(id + 1);
            }
        }
    }

    loop {
        let id = next_id.to_string();
        let run_root = root.join(&id);
        match fs::create_dir(&run_root) {
            Ok(()) => return Ok((ExperimentId::from(id), run_root)),
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
                next_id += 1;
            }
            Err(err) => return Err(err),
        }
    }
}
