pub mod error;

pub use error::FetchError;

use bytes::Bytes;
use reqwest::Client;
use std::time::Duration;

/// HTTP client for fetching byte ranges from safetensors files.
///
/// Handles HuggingFace redirect resolution (302 to CDN) and
/// standard HTTP range requests for any URL.
pub struct RangeFetcher {
    client: Client,
    url: String,
    total_size: u64,
}

/// Check if a URL is a HuggingFace resolve URL that needs redirect handling.
fn is_hf_resolve_url(url: &str) -> bool {
    url.contains("huggingface.co") && url.contains("/resolve/")
}

impl RangeFetcher {
    /// Create a new fetcher. For HuggingFace URLs, resolves the redirect
    /// to get the actual CDN URL. For other URLs, uses the URL directly.
    ///
    /// Fetches total file size via a HEAD request (or from the redirect response).
    pub async fn new(url: &str) -> Result<Self, FetchError> {
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

        tracing::info!(url = %resolved_url, total_size, "RangeFetcher ready");

        Ok(Self {
            client,
            url: resolved_url,
            total_size,
        })
    }

    /// Fetch a byte range (inclusive start and end). Returns the raw bytes.
    pub async fn fetch_range(&self, start: u64, end: u64) -> Result<Bytes, FetchError> {
        let range = format!("bytes={start}-{end}");
        tracing::debug!(url = %self.url, %range, "fetching byte range");

        let response = self
            .client
            .get(&self.url)
            .header(reqwest::header::RANGE, &range)
            .send()
            .await
            .map_err(FetchError::Request)?;

        let status = response.status();

        if status.as_u16() == 206 {
            // Partial content — expected
            return response.bytes().await.map_err(FetchError::Body);
        }

        if status.as_u16() == 200 {
            // Server ignored Range header, returned full body. Slice it.
            tracing::warn!("server returned 200 instead of 206, slicing response");
            let full = response.bytes().await.map_err(FetchError::Body)?;
            let end_idx = std::cmp::min((end + 1) as usize, full.len());
            return Ok(full.slice(start as usize..end_idx));
        }

        Err(FetchError::UnexpectedStatus(status.as_u16()))
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
        self.total_size
    }

    /// The resolved URL used for fetching.
    pub fn url(&self) -> &str {
        &self.url
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
        println!("Resolved URL: {}", fetcher.url());
        println!("Total size: {} bytes", fetcher.total_size());
    }

    #[tokio::test]
    async fn test_fetch_header_size() {
        let fetcher = RangeFetcher::new(HF_MODEL_URL).await.unwrap();
        let size_bytes = fetcher.fetch_range(0, 7).await.unwrap();
        assert_eq!(size_bytes.len(), 8, "should get exactly 8 bytes");

        let header_len = u64::from_le_bytes(size_bytes[..8].try_into().unwrap());
        println!("Header length: {} bytes", header_len);
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
        println!("Header contains {} entries", obj.len());
        for key in obj.keys().take(5) {
            println!("  tensor: {key}");
        }
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

        println!("Fetching tensor '{tensor_name}': bytes {start}..{end} ({tensor_size} bytes)");

        // Offset from file start: header_size(8) + header_json_len + data_offset
        let header_len = u64::from_le_bytes(
            fetcher.fetch_range(0, 7).await.unwrap()[..8].try_into().unwrap()
        );
        let data_start = 8 + header_len + start;
        let data_end = 8 + header_len + end - 1; // inclusive

        let tensor_bytes = fetcher.fetch_range(data_start, data_end).await.unwrap();
        assert_eq!(
            tensor_bytes.len() as u64, tensor_size,
            "fetched tensor size should match expected"
        );
        println!("Successfully fetched {tensor_size} bytes for tensor '{tensor_name}'");
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
        println!(
            "Total: {total}, header: {header_len}, data: {} bytes",
            total - 8 - header_len
        );
        // Verify the header JSON size matches what we fetched
        assert_eq!(header_json.len() as u64, header_len);
    }
}
