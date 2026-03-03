/// Normalize a checksum string (strip prefixes, lowercase).
pub fn normalize_checksum(value: &str) -> Result<String, String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err("checksum is empty".to_string());
    }
    let lower = trimmed.to_ascii_lowercase();
    if let Some(rest) = lower.strip_prefix("sha256:") {
        return Ok(rest.to_string());
    }
    if lower.contains(':') {
        return Err(format!("unsupported checksum format: {trimmed}"));
    }
    Ok(lower)
}
