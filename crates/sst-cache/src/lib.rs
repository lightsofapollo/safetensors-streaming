use bytes::Bytes;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CacheError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("cache is disabled")]
    Disabled,
}

pub type Result<T> = std::result::Result<T, CacheError>;

/// Configuration for the tensor disk cache.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Root directory for cached files.
    pub cache_dir: PathBuf,
    /// Maximum total cache size in bytes (0 = unlimited). Not enforced yet.
    pub max_size_bytes: u64,
    /// Whether the cache is enabled.
    pub enabled: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("/tmp"))
            .join("safetensors-streaming");

        Self {
            cache_dir,
            max_size_bytes: 0,
            enabled: true,
        }
    }
}

/// Disk cache for streamed safetensors data.
///
/// Cache key is `sha256(url + etag)`. Files are stored as
/// `<cache_dir>/<hex_hash>.safetensors`.
#[derive(Debug, Clone)]
pub struct TensorCache {
    config: CacheConfig,
}

impl TensorCache {
    /// Create a new `TensorCache` with the given config.
    pub fn new(config: CacheConfig) -> Self {
        Self { config }
    }

    /// Create a `TensorCache` with default config.
    pub fn with_defaults() -> Self {
        Self::new(CacheConfig::default())
    }

    /// Compute the cache file path for a given URL and optional ETag.
    pub fn cache_path(&self, url: &str, etag: Option<&str>) -> PathBuf {
        let hash = Self::compute_hash(url, etag);
        self.config.cache_dir.join(format!("{hash}.safetensors"))
    }

    /// Check if a cached file exists for the given URL/ETag. Returns the path
    /// if present.
    pub fn lookup(&self, url: &str, etag: Option<&str>) -> Option<PathBuf> {
        if !self.config.enabled {
            return None;
        }
        let path = self.cache_path(url, etag);
        if path.is_file() {
            tracing::debug!(url, ?etag, ?path, "cache hit");
            Some(path)
        } else {
            tracing::debug!(url, ?etag, "cache miss");
            None
        }
    }

    /// Write tensor data to the cache.
    pub fn store(&self, url: &str, etag: Option<&str>, data: &[u8]) -> Result<()> {
        if !self.config.enabled {
            return Err(CacheError::Disabled);
        }

        let path = self.cache_path(url, etag);

        // Ensure the cache directory exists.
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        // Atomic write: write to a temp file then rename to avoid partial reads.
        let tmp_path = path.with_extension("tmp");
        fs::write(&tmp_path, data)?;
        fs::rename(&tmp_path, &path)?;

        tracing::debug!(url, ?etag, ?path, bytes = data.len(), "cached to disk");
        Ok(())
    }

    /// Load cached data from disk.
    pub fn load(&self, path: &Path) -> Result<Bytes> {
        let data = fs::read(path)?;
        Ok(Bytes::from(data))
    }

    /// Compute SHA-256 hex digest of `url + etag`.
    fn compute_hash(url: &str, etag: Option<&str>) -> String {
        let mut hasher = Sha256::new();
        hasher.update(url.as_bytes());
        if let Some(tag) = etag {
            hasher.update(tag.as_bytes());
        }
        let result = hasher.finalize();
        hex::encode(result)
    }
}

/// Minimal hex encoding (avoids pulling in the `hex` crate).
mod hex {
    pub fn encode(bytes: impl AsRef<[u8]>) -> String {
        bytes
            .as_ref()
            .iter()
            .fold(String::new(), |mut s, b| {
                use std::fmt::Write;
                let _ = write!(s, "{b:02x}");
                s
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn test_cache(dir: &Path) -> TensorCache {
        TensorCache::new(CacheConfig {
            cache_dir: dir.to_path_buf(),
            max_size_bytes: 0,
            enabled: true,
        })
    }

    #[test]
    fn cache_miss_store_hit_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let cache = test_cache(tmp.path());

        let url = "https://example.com/model.safetensors";
        let etag = Some("abc123");
        let data = b"fake tensor payload";

        // Initially a miss.
        assert!(cache.lookup(url, etag).is_none());

        // Store data.
        cache.store(url, etag, data).unwrap();

        // Now a hit.
        let path = cache.lookup(url, etag).expect("should be a hit");
        assert!(path.is_file());

        // Load and verify contents.
        let loaded = cache.load(&path).unwrap();
        assert_eq!(&loaded[..], data);
    }

    #[test]
    fn different_urls_produce_different_paths() {
        let tmp = TempDir::new().unwrap();
        let cache = test_cache(tmp.path());

        let path_a = cache.cache_path("https://a.com/model.safetensors", None);
        let path_b = cache.cache_path("https://b.com/model.safetensors", None);
        assert_ne!(path_a, path_b);
    }

    #[test]
    fn etag_change_invalidates_cache() {
        let tmp = TempDir::new().unwrap();
        let cache = test_cache(tmp.path());

        let url = "https://example.com/model.safetensors";
        let data_v1 = b"version 1";
        let data_v2 = b"version 2";

        // Store with etag v1.
        cache.store(url, Some("v1"), data_v1).unwrap();
        assert!(cache.lookup(url, Some("v1")).is_some());

        // Different etag => miss (different hash => different file).
        assert!(cache.lookup(url, Some("v2")).is_none());

        // Store v2.
        cache.store(url, Some("v2"), data_v2).unwrap();
        assert!(cache.lookup(url, Some("v2")).is_some());

        // v1 is still there (we don't evict yet).
        assert!(cache.lookup(url, Some("v1")).is_some());
    }

    #[test]
    fn disabled_cache_always_misses() {
        let tmp = TempDir::new().unwrap();
        let cache = TensorCache::new(CacheConfig {
            cache_dir: tmp.path().to_path_buf(),
            max_size_bytes: 0,
            enabled: false,
        });

        let url = "https://example.com/model.safetensors";
        assert!(cache.lookup(url, None).is_none());
        assert!(cache.store(url, None, b"data").is_err());
    }

    #[test]
    fn no_etag_uses_url_only_hash() {
        let tmp = TempDir::new().unwrap();
        let cache = test_cache(tmp.path());

        let url = "https://example.com/model.safetensors";
        cache.store(url, None, b"data").unwrap();

        // Lookup with None etag should hit.
        assert!(cache.lookup(url, None).is_some());
        // Lookup with Some etag should miss (different hash).
        assert!(cache.lookup(url, Some("etag")).is_none());
    }
}
