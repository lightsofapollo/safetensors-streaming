pub mod error;
#[cfg(feature = "xet")]
mod xet;

pub use error::FetchError;

use bytes::Bytes;
use reqwest::Client;
use std::time::Duration;

/// Maximum number of retry attempts for transient HTTP errors.
const MAX_RETRIES: u32 = 3;

/// Base backoff duration for the first retry (doubles each attempt).
const BASE_BACKOFF: Duration = Duration::from_millis(100);

/// Maximum jitter added to each backoff delay.
const MAX_JITTER: Duration = Duration::from_millis(100);

/// HTTP status codes that are eligible for retry.
const RETRYABLE_STATUS_CODES: &[u16] = &[429, 500, 502, 503, 504];

/// Compute the backoff duration for a given attempt (0-indexed).
/// Formula: base * 2^attempt + deterministic jitter.
/// Jitter is derived from (attempt + 1) * 37ms mod MAX_JITTER to avoid
/// synchronized retry storms without requiring a random number generator.
fn backoff_duration(attempt: u32) -> Duration {
    let base = BASE_BACKOFF.saturating_mul(1 << attempt);
    let jitter_ms = ((attempt as u64 + 1) * 37) % MAX_JITTER.as_millis() as u64;
    base + Duration::from_millis(jitter_ms)
}

/// Returns true if a reqwest error is transient and should be retried.
fn is_retryable_request_error(err: &reqwest::Error) -> bool {
    err.is_connect() || err.is_timeout()
}

/// Returns true if an HTTP status code should be retried.
fn is_retryable_status(status: u16) -> bool {
    RETRYABLE_STATUS_CODES.contains(&status)
}

/// Parsed S3 URL components.
#[derive(Debug, Clone)]
pub struct S3Url {
    pub bucket: String,
    pub key: String,
}

/// Parse an `s3://bucket/key` URL into bucket and key.
/// Returns `None` if the URL doesn't start with `s3://` or is malformed.
pub fn parse_s3_url(url: &str) -> Result<S3Url, FetchError> {
    let rest = url
        .strip_prefix("s3://")
        .ok_or_else(|| FetchError::InvalidS3Url("URL must start with s3://".into()))?;

    let (bucket, key) = rest
        .split_once('/')
        .ok_or_else(|| FetchError::InvalidS3Url("missing key after bucket name".into()))?;

    if bucket.is_empty() {
        return Err(FetchError::InvalidS3Url("empty bucket name".into()));
    }
    if key.is_empty() {
        return Err(FetchError::InvalidS3Url("empty key".into()));
    }

    Ok(S3Url {
        bucket: bucket.to_string(),
        key: key.to_string(),
    })
}

/// Returns true if the URL uses the `s3://` scheme.
fn is_s3_url(url: &str) -> bool {
    url.starts_with("s3://")
}

/// Internal fetcher variant — HTTP, S3, or Xet.
enum FetcherInner {
    Http {
        client: Client,
        url: String,
        total_size: u64,
    },
    #[cfg(feature = "s3")]
    S3 {
        client: aws_sdk_s3::Client,
        bucket: String,
        key: String,
        total_size: u64,
    },
    #[cfg(feature = "xet")]
    Xet {
        cas_client: std::sync::Arc<dyn cas_client::Client>,
        file_hash: merklehash::MerkleHash,
        total_size: u64,
    },
}

/// Async client for fetching byte ranges from safetensors files.
///
/// Supports HTTP(S) URLs (including HuggingFace redirects) and
/// S3 native URLs (`s3://bucket/key`) when the `s3` feature is enabled.
pub struct RangeFetcher {
    inner: FetcherInner,
}

/// Check if a URL is a HuggingFace resolve URL that needs redirect handling.
fn is_hf_resolve_url(url: &str) -> bool {
    url.contains("huggingface.co") && url.contains("/resolve/")
}

impl RangeFetcher {
    /// Create a new fetcher.
    ///
    /// - For `s3://` URLs: uses AWS SDK with default credential chain
    /// - For HuggingFace URLs: resolves the redirect to the CDN URL
    /// - For other HTTP(S) URLs: uses the URL directly
    ///
    /// Fetches total file size via HEAD (HTTP) or HeadObject (S3).
    pub async fn new(url: &str) -> Result<Self, FetchError> {
        if is_s3_url(url) {
            #[cfg(feature = "s3")]
            {
                return Self::new_s3(url).await;
            }
            #[cfg(not(feature = "s3"))]
            {
                return Err(FetchError::InvalidS3Url(
                    "S3 support not enabled — build with feature 's3'".into(),
                ));
            }
        }

        Self::new_http(url).await
    }

    /// Build an HTTP-backed fetcher (or Xet-backed if X-Xet-Hash is detected).
    async fn new_http(url: &str) -> Result<Self, FetchError> {
        let no_redirect_client = Client::builder()
            .use_rustls_tls()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(FetchError::HttpClient)?;

        let client = Client::builder()
            .use_rustls_tls()
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(FetchError::HttpClient)?;

        let (resolved_url, total_size) = if is_hf_resolve_url(url) {
            tracing::debug!(url, "resolving HuggingFace redirect");

            let resp = no_redirect_client
                .get(url)
                .send()
                .await
                .map_err(FetchError::Request)?;

            let status = resp.status();

            // Check for Xet-backed file before following redirect
            #[cfg(feature = "xet")]
            if let Some(xet_hash) = resp.headers().get("X-Xet-Hash") {
                let xet_hash_str = xet_hash
                    .to_str()
                    .map_err(FetchError::InvalidHeader)?
                    .to_string();

                tracing::info!(xet_hash = %xet_hash_str, "detected Xet-backed file");

                // Get total file size — try X-Linked-Size first (HF specific),
                // then follow the redirect and HEAD the CDN URL for Content-Length
                let total_size = if let Some(xls) = resp.headers().get("X-Linked-Size") {
                    let s = xls.to_str().map_err(FetchError::InvalidHeader)?;
                    s.parse::<u64>().map_err(|_| FetchError::MissingContentLength)?
                } else if status.is_redirection() {
                    let location = resp
                        .headers()
                        .get(reqwest::header::LOCATION)
                        .ok_or(FetchError::MissingLocation)?
                        .to_str()
                        .map_err(FetchError::InvalidHeader)?;
                    let head_resp = client.head(location).send().await
                        .map_err(FetchError::Request)?;
                    content_length_from_response(&head_resp)?
                } else {
                    return Err(FetchError::MissingContentLength);
                };

                return xet::new_xet_fetcher(url, &xet_hash_str, total_size, &client).await;
            }

            if status.is_redirection() {
                let location = resp
                    .headers()
                    .get(reqwest::header::LOCATION)
                    .ok_or(FetchError::MissingLocation)?
                    .to_str()
                    .map_err(FetchError::InvalidHeader)?
                    .to_string();

                tracing::debug!(location = %location, "resolved HF redirect");

                // Get file size via HEAD on the CDN URL
                let head_resp = client
                    .head(&location)
                    .send()
                    .await
                    .map_err(FetchError::Request)?;

                let size = content_length_from_response(&head_resp)?;
                (location, size)
            } else {
                // Not a redirect — maybe direct access is allowed
                let size = content_length_from_response(&resp)?;
                (url.to_string(), size)
            }
        } else {
            tracing::debug!(url, "using direct URL (non-HF)");

            let head_resp = client
                .head(url)
                .send()
                .await
                .map_err(FetchError::Request)?;

            let size = content_length_from_response(&head_resp)?;
            (url.to_string(), size)
        };

        tracing::info!(url = %resolved_url, total_size, "RangeFetcher ready (HTTP)");

        Ok(Self {
            inner: FetcherInner::Http {
                client,
                url: resolved_url,
                total_size,
            },
        })
    }

    /// Build an S3-backed fetcher.
    #[cfg(feature = "s3")]
    async fn new_s3(url: &str) -> Result<Self, FetchError> {
        let parsed = parse_s3_url(url)?;

        tracing::debug!(
            bucket = %parsed.bucket,
            key = %parsed.key,
            "initializing S3 fetcher"
        );

        let config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
        let s3_client = aws_sdk_s3::Client::new(&config);

        // HeadObject to get total file size
        let head = s3_client
            .head_object()
            .bucket(&parsed.bucket)
            .key(&parsed.key)
            .send()
            .await
            .map_err(|e| FetchError::S3HeadObject(e.to_string()))?;

        let total_size = head
            .content_length()
            .ok_or(FetchError::S3MissingContentLength)?;

        if total_size < 0 {
            return Err(FetchError::S3MissingContentLength);
        }
        let total_size = total_size as u64;

        tracing::info!(
            bucket = %parsed.bucket,
            key = %parsed.key,
            total_size,
            "RangeFetcher ready (S3)"
        );

        Ok(Self {
            inner: FetcherInner::S3 {
                client: s3_client,
                bucket: parsed.bucket,
                key: parsed.key,
                total_size,
            },
        })
    }

    /// Fetch a contiguous range covering multiple tensors in one HTTP request.
    /// `start` is inclusive, `end` is exclusive (byte past the last byte needed).
    /// Returns the raw bytes for the entire range.
    pub async fn fetch_batch(&self, start: u64, end: u64) -> Result<Bytes, FetchError> {
        if start >= end {
            return Ok(Bytes::new());
        }
        // fetch_range uses inclusive end
        self.fetch_range(start, end - 1).await
    }

    /// Fetch a byte range (inclusive start and end). Returns the raw bytes.
    pub async fn fetch_range(&self, start: u64, end: u64) -> Result<Bytes, FetchError> {
        match &self.inner {
            FetcherInner::Http { client, url, .. } => {
                Self::fetch_range_http(client, url, start, end).await
            }
            #[cfg(feature = "s3")]
            FetcherInner::S3 {
                client,
                bucket,
                key,
                ..
            } => Self::fetch_range_s3(client, bucket, key, start, end).await,
            #[cfg(feature = "xet")]
            FetcherInner::Xet {
                cas_client,
                file_hash,
                ..
            } => xet::fetch_range_xet(cas_client, file_hash, start, end).await,
        }
    }

    /// HTTP range fetch implementation with retry logic for transient errors.
    ///
    /// Retries on:
    /// - HTTP 429, 500, 502, 503, 504 status codes
    /// - Connection errors (`reqwest::Error::is_connect()`)
    /// - Timeout errors (`reqwest::Error::is_timeout()`)
    ///
    /// Uses exponential backoff with deterministic jitter.
    async fn fetch_range_http(
        client: &Client,
        url: &str,
        start: u64,
        end: u64,
    ) -> Result<Bytes, FetchError> {
        let range = format!("bytes={start}-{end}");
        let mut last_error: Option<FetchError> = None;

        for attempt in 0..=MAX_RETRIES {
            if attempt > 0 {
                let delay = backoff_duration(attempt - 1);
                tracing::warn!(
                    url,
                    attempt,
                    delay_ms = delay.as_millis() as u64,
                    "retrying HTTP range fetch after transient error"
                );
                tokio::time::sleep(delay).await;
            }

            tracing::debug!(url, %range, attempt, "fetching byte range (HTTP)");

            let result = client
                .get(url)
                .header(reqwest::header::RANGE, &range)
                .send()
                .await;

            let response = match result {
                Ok(resp) => resp,
                Err(err) => {
                    if attempt < MAX_RETRIES && is_retryable_request_error(&err) {
                        tracing::warn!(
                            url,
                            attempt,
                            error = %err,
                            "transient request error, will retry"
                        );
                        last_error = Some(FetchError::Request(err));
                        continue;
                    }
                    return Err(FetchError::Request(err));
                }
            };

            let status = response.status().as_u16();

            if status == 206 {
                return response.bytes().await.map_err(FetchError::Body);
            }

            if status == 200 {
                // Server ignored Range header, returned full body. Slice it.
                tracing::warn!("server returned 200 instead of 206, slicing response");
                let full = response.bytes().await.map_err(FetchError::Body)?;
                let end_idx = std::cmp::min((end + 1) as usize, full.len());
                return Ok(full.slice(start as usize..end_idx));
            }

            if attempt < MAX_RETRIES && is_retryable_status(status) {
                tracing::warn!(
                    url,
                    attempt,
                    status,
                    "retryable HTTP status, will retry"
                );
                last_error = Some(FetchError::UnexpectedStatus(status));
                continue;
            }

            return Err(FetchError::UnexpectedStatus(status));
        }

        // All retries exhausted — return the last error.
        // This path is only reachable if every attempt hit a retryable condition.
        Err(last_error.unwrap_or(FetchError::UnexpectedStatus(0)))
    }

    /// S3 range fetch implementation.
    #[cfg(feature = "s3")]
    async fn fetch_range_s3(
        client: &aws_sdk_s3::Client,
        bucket: &str,
        key: &str,
        start: u64,
        end: u64,
    ) -> Result<Bytes, FetchError> {
        let range = format!("bytes={start}-{end}");
        tracing::debug!(bucket, key, %range, "fetching byte range (S3)");

        let resp = client
            .get_object()
            .bucket(bucket)
            .key(key)
            .range(&range)
            .send()
            .await
            .map_err(|e| FetchError::S3GetObject(e.to_string()))?;

        let body = resp
            .body
            .collect()
            .await
            .map_err(|e| FetchError::S3ByteStream(e.to_string()))?;

        Ok(body.into_bytes())
    }

    /// Fetch the safetensors header: first 8 bytes for the size prefix,
    /// then the JSON header bytes.
    ///
    /// Returns `(size_prefix_bytes, header_json_bytes)`.
    pub async fn fetch_header(&self) -> Result<(Bytes, Bytes), FetchError> {
        // First 8 bytes: little-endian u64 header length
        let size_bytes = self.fetch_range(0, 7).await?;
        if size_bytes.len() < 8 {
            return Err(FetchError::HeaderTooShort);
        }

        let header_len = u64::from_le_bytes(
            size_bytes[..8]
                .try_into()
                .map_err(|_| FetchError::HeaderTooShort)?,
        );

        tracing::debug!(header_len, "fetching safetensors header JSON");

        // Fetch the JSON header (starts at byte 8, length = header_len)
        let header_json = self.fetch_range(8, 8 + header_len - 1).await?;

        Ok((size_bytes, header_json))
    }

    /// Total file size in bytes.
    pub fn total_size(&self) -> u64 {
        match &self.inner {
            FetcherInner::Http { total_size, .. } => *total_size,
            #[cfg(feature = "s3")]
            FetcherInner::S3 { total_size, .. } => *total_size,
            #[cfg(feature = "xet")]
            FetcherInner::Xet { total_size, .. } => *total_size,
        }
    }

    /// The resolved URL (or `s3://bucket/key`) used for fetching.
    pub fn url(&self) -> String {
        match &self.inner {
            FetcherInner::Http { url, .. } => url.clone(),
            #[cfg(feature = "s3")]
            FetcherInner::S3 { bucket, key, .. } => format!("s3://{bucket}/{key}"),
            #[cfg(feature = "xet")]
            FetcherInner::Xet { file_hash, .. } => format!("xet://{file_hash}"),
        }
    }
}

fn content_length_from_response(resp: &reqwest::Response) -> Result<u64, FetchError> {
    // Try Content-Length first, then X-Linked-Size (HF specific)
    if let Some(cl) = resp.headers().get(reqwest::header::CONTENT_LENGTH) {
        let s = cl.to_str().map_err(FetchError::InvalidHeader)?;
        return s
            .parse::<u64>()
            .map_err(|_| FetchError::MissingContentLength);
    }

    if let Some(xls) = resp.headers().get("X-Linked-Size") {
        let s = xls.to_str().map_err(FetchError::InvalidHeader)?;
        return s
            .parse::<u64>()
            .map_err(|_| FetchError::MissingContentLength);
    }

    Err(FetchError::MissingContentLength)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── S3 URL parsing (no network) ──────────────────────────────────

    #[test]
    fn test_parse_s3_url_basic() {
        let parsed = parse_s3_url("s3://my-bucket/path/to/model.safetensors");
        assert!(parsed.is_ok());
        let s3 = parsed.unwrap();
        assert_eq!(s3.bucket, "my-bucket");
        assert_eq!(s3.key, "path/to/model.safetensors");
    }

    #[test]
    fn test_parse_s3_url_single_key() {
        let parsed = parse_s3_url("s3://bucket/file.bin");
        assert!(parsed.is_ok());
        let s3 = parsed.unwrap();
        assert_eq!(s3.bucket, "bucket");
        assert_eq!(s3.key, "file.bin");
    }

    #[test]
    fn test_parse_s3_url_nested_key() {
        let parsed = parse_s3_url("s3://b/a/b/c/d/e.safetensors");
        assert!(parsed.is_ok());
        let s3 = parsed.unwrap();
        assert_eq!(s3.bucket, "b");
        assert_eq!(s3.key, "a/b/c/d/e.safetensors");
    }

    #[test]
    fn test_parse_s3_url_not_s3() {
        let parsed = parse_s3_url("https://example.com/file.bin");
        assert!(parsed.is_err());
    }

    #[test]
    fn test_parse_s3_url_empty_bucket() {
        let parsed = parse_s3_url("s3:///key");
        assert!(parsed.is_err());
    }

    #[test]
    fn test_parse_s3_url_empty_key() {
        let parsed = parse_s3_url("s3://bucket/");
        assert!(parsed.is_err());
    }

    #[test]
    fn test_parse_s3_url_no_key() {
        let parsed = parse_s3_url("s3://bucket");
        assert!(parsed.is_err());
    }

    #[test]
    fn test_is_s3_url() {
        assert!(is_s3_url("s3://bucket/key"));
        assert!(!is_s3_url("https://example.com"));
        assert!(!is_s3_url("http://example.com"));
    }

    // ── Retry backoff calculation (no network) ─────────────────────────

    #[test]
    fn test_backoff_duration_increases_exponentially() {
        let d0 = backoff_duration(0);
        let d1 = backoff_duration(1);
        let d2 = backoff_duration(2);

        // Base durations: 100ms, 200ms, 400ms (plus jitter)
        // Each should be roughly double the previous (within jitter bounds)
        assert!(d0.as_millis() >= 100, "attempt 0 should be >= 100ms, got {}ms", d0.as_millis());
        assert!(d0.as_millis() < 200, "attempt 0 should be < 200ms, got {}ms", d0.as_millis());

        assert!(d1.as_millis() >= 200, "attempt 1 should be >= 200ms, got {}ms", d1.as_millis());
        assert!(d1.as_millis() < 300, "attempt 1 should be < 300ms, got {}ms", d1.as_millis());

        assert!(d2.as_millis() >= 400, "attempt 2 should be >= 400ms, got {}ms", d2.as_millis());
        assert!(d2.as_millis() < 500, "attempt 2 should be < 500ms, got {}ms", d2.as_millis());
    }

    #[test]
    fn test_backoff_includes_jitter() {
        // Jitter = (attempt + 1) * 37 % 100, so:
        // attempt 0: 1 * 37 % 100 = 37ms
        // attempt 1: 2 * 37 % 100 = 74ms
        // attempt 2: 3 * 37 % 100 = 11ms (wraps via modulo)
        let d0 = backoff_duration(0);
        let d1 = backoff_duration(1);
        let d2 = backoff_duration(2);

        assert_eq!(d0, Duration::from_millis(100 + 37));
        assert_eq!(d1, Duration::from_millis(200 + 74));
        assert_eq!(d2, Duration::from_millis(400 + 11));
    }

    #[test]
    fn test_retryable_status_codes() {
        assert!(is_retryable_status(429));
        assert!(is_retryable_status(500));
        assert!(is_retryable_status(502));
        assert!(is_retryable_status(503));
        assert!(is_retryable_status(504));
        // Non-retryable
        assert!(!is_retryable_status(400));
        assert!(!is_retryable_status(401));
        assert!(!is_retryable_status(403));
        assert!(!is_retryable_status(404));
        assert!(!is_retryable_status(200));
        assert!(!is_retryable_status(206));
    }

    // ── HTTP tests (network required) ────────────────────────────────

    const HF_MODEL_URL: &str = "https://huggingface.co/google/bert_uncased_L-2_H-128_A-2/resolve/main/model.safetensors";

    #[tokio::test]
    async fn test_resolve_hf_redirect() {
        let fetcher = RangeFetcher::new(HF_MODEL_URL).await.unwrap();
        // Should have resolved to a CDN URL, not the original HF URL
        assert!(
            !fetcher.url().contains("huggingface.co/google"),
            "URL should be resolved to CDN, got: {}",
            fetcher.url()
        );
        assert!(fetcher.total_size() > 0, "total_size should be > 0");
    }

    #[tokio::test]
    async fn test_fetch_header_size() {
        let fetcher = RangeFetcher::new(HF_MODEL_URL).await.unwrap();
        let size_bytes = fetcher.fetch_range(0, 7).await.unwrap();
        assert_eq!(size_bytes.len(), 8, "should get exactly 8 bytes");

        let header_len = u64::from_le_bytes(size_bytes[..8].try_into().unwrap());
        assert!(header_len > 0 && header_len < 10_000_000, "header length should be reasonable");
    }

    #[tokio::test]
    async fn test_fetch_full_header() {
        let fetcher = RangeFetcher::new(HF_MODEL_URL).await.unwrap();
        let (size_bytes, header_json) = fetcher.fetch_header().await.unwrap();

        assert_eq!(size_bytes.len(), 8);
        let header_len = u64::from_le_bytes(size_bytes[..8].try_into().unwrap());
        assert_eq!(header_json.len(), header_len as usize);

        // Parse the header JSON to verify it's valid
        let parsed: serde_json::Value = serde_json::from_slice(&header_json).unwrap();
        assert!(parsed.is_object(), "header should be a JSON object");

        // BERT model should have known tensor names
        let obj = parsed.as_object().unwrap();
        assert!(obj.len() > 0, "header should contain entries");
    }

    #[tokio::test]
    async fn test_fetch_tensor_range() {
        let fetcher = RangeFetcher::new(HF_MODEL_URL).await.unwrap();
        let (_size_bytes, header_json) = fetcher.fetch_header().await.unwrap();

        let parsed: serde_json::Value = serde_json::from_slice(&header_json).unwrap();
        let obj = parsed.as_object().unwrap();

        // Find first non-metadata tensor and fetch it
        let (tensor_name, tensor_meta) = obj
            .iter()
            .find(|(k, _)| *k != "__metadata__")
            .unwrap();

        let offsets = tensor_meta["data_offsets"].as_array().unwrap();
        let start = offsets[0].as_u64().unwrap();
        let end = offsets[1].as_u64().unwrap();
        let tensor_size = end - start;

        // Offset from file start: header_size(8) + header_json_len + data_offset
        let header_len = u64::from_le_bytes(
            fetcher.fetch_range(0, 7).await.unwrap()[..8].try_into().unwrap()
        );
        let data_start = 8 + header_len + start;
        let data_end = 8 + header_len + end - 1; // inclusive

        let tensor_bytes = fetcher.fetch_range(data_start, data_end).await.unwrap();
        assert_eq!(
            tensor_bytes.len() as u64, tensor_size,
            "fetched tensor size should match expected for '{tensor_name}'"
        );
    }

    #[tokio::test]
    async fn test_total_size_matches() {
        let fetcher = RangeFetcher::new(HF_MODEL_URL).await.unwrap();
        let total = fetcher.total_size();

        // Fetch header to compute expected minimum size
        let (size_bytes, header_json) = fetcher.fetch_header().await.unwrap();
        let header_len = u64::from_le_bytes(size_bytes[..8].try_into().unwrap());

        // Total file size must be at least 8 (prefix) + header_len
        assert!(
            total >= 8 + header_len,
            "total_size ({total}) should be >= 8 + header_len ({header_len})"
        );
        // Verify the header JSON size matches what we fetched
        assert_eq!(header_json.len() as u64, header_len);
    }

    // ── S3 integration test (requires real credentials) ──────────────

    #[cfg(feature = "s3")]
    #[tokio::test]
    #[ignore = "requires AWS credentials and a real S3 bucket"]
    async fn test_s3_fetch_header() {
        // Set S3_TEST_URL env var to an s3://bucket/key pointing to a .safetensors file
        let url = std::env::var("S3_TEST_URL")
            .unwrap_or_else(|_| "s3://test-bucket/model.safetensors".to_string());

        let fetcher = RangeFetcher::new(&url).await.unwrap();
        assert!(fetcher.total_size() > 0);

        let (size_bytes, header_json) = fetcher.fetch_header().await.unwrap();
        assert_eq!(size_bytes.len(), 8);

        let header_len = u64::from_le_bytes(size_bytes[..8].try_into().unwrap());
        assert_eq!(header_json.len(), header_len as usize);

        // Validate it's valid JSON
        let parsed: serde_json::Value = serde_json::from_slice(&header_json).unwrap();
        assert!(parsed.is_object());
    }
}
