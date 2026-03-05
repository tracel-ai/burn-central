use std::path::{Path, PathBuf};

/// Normalize a path within a bundle (use forward slashes, remove leading slash)
pub fn normalize_bundle_path<S: AsRef<str>>(s: S) -> String {
    s.as_ref()
        .replace('\\', "/")
        .trim_start_matches('/')
        .to_string()
}

/// Sanitize a relative path to prevent directory traversal attacks.
pub fn sanitize_rel_path(path: &str) -> Result<PathBuf, String> {
    let normalized = normalize_bundle_path(path);
    if normalized.is_empty() {
        return Err("invalid path component: empty path".to_string());
    }

    let rel = Path::new(&normalized);
    for component in rel.components() {
        use std::path::Component;
        match component {
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(format!("invalid path component: {path}"));
            }
            Component::CurDir => {
                return Err(format!("invalid path component: {path}"));
            }
            Component::Normal(_) => {}
        }
    }
    Ok(PathBuf::from(normalized))
}

/// Safely join a root path with a relative path.
pub fn safe_join(root: &Path, rel: &str) -> Result<PathBuf, String> {
    let rel = sanitize_rel_path(rel)?;
    Ok(root.join(rel))
}
