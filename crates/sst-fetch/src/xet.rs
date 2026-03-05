//! Xet-native fetch backend for HuggingFace Xet-backed files.
//!
//! When a HuggingFace resolve URL returns an `X-Xet-Hash` header,
//! this backend uses the xet-core `file_reconstruction` crate to
//! fetch byte ranges via the CAS reconstruction protocol, bypassing
//! the standard LFS HTTP Range request path.

use std::sync::Arc;

use bytes::Bytes;
use cas_client::RemoteClient;
use cas_types::FileRange;
use file_reconstruction::FileReconstructor;
use merklehash::MerkleHash;
use reqwest::Client;
use xet_core_utils::auth::AuthConfig;

use crate::error::FetchError;
use crate::{FetcherInner, RangeFetcher};

/// Response from HuggingFace's xet-read-token API.
#[derive(serde::Deserialize)]
struct XetTokenResponse {
    #[serde(rename = "accessToken")]
    access_token: String,
    exp: u64,
    #[serde(rename = "casUrl")]
    cas_url: String,
}

/// Extract repo_id and revision from a HuggingFace resolve URL.
///
/// Expected format: `https://huggingface.co/{namespace}/{repo}/resolve/{revision}/{filepath}`
fn parse_hf_resolve_url(url: &str) -> Option<(String, String)> {
    let url = url.strip_prefix("https://huggingface.co/")?;

    // Find /resolve/ separator
    let resolve_idx = url.find("/resolve/")?;
    let repo_id = &url[..resolve_idx];

    let after_resolve = &url[resolve_idx + "/resolve/".len()..];
    // revision is everything up to the next /
    let revision = after_resolve.split('/').next()?;

    Some((repo_id.to_string(), revision.to_string()))
}

/// Fetch a Xet read token from the HuggingFace Hub API.
async fn fetch_xet_token(
    client: &Client,
    repo_id: &str,
    revision: &str,
) -> Result<XetTokenResponse, FetchError> {
    let token_url = format!(
        "https://huggingface.co/api/models/{repo_id}/xet-read-token/{revision}"
    );

    tracing::debug!(token_url = %token_url, "fetching Xet read token");

    let mut request = client.get(&token_url);

    // Use HF_TOKEN if available for authentication
    if let Ok(hf_token) = std::env::var("HF_TOKEN") {
        request = request.header("Authorization", format!("Bearer {hf_token}"));
    } else if let Ok(hf_token) = std::env::var("HUGGING_FACE_HUB_TOKEN") {
        request = request.header("Authorization", format!("Bearer {hf_token}"));
    }

    let resp = request
        .send()
        .await
        .map_err(|e| FetchError::XetTokenFetch(e.to_string()))?;

    let status = resp.status().as_u16();
    if status != 200 {
        let body = resp.text().await.unwrap_or_default();
        return Err(FetchError::XetTokenFetch(format!(
            "HTTP {status}: {body}"
        )));
    }

    let body = resp.bytes().await
        .map_err(|e| FetchError::XetTokenFetch(e.to_string()))?;
    serde_json::from_slice::<XetTokenResponse>(&body)
        .map_err(|e| FetchError::XetTokenFetch(e.to_string()))
}

/// Create a Xet-backed RangeFetcher.
///
/// Called from `RangeFetcher::new_http` when an `X-Xet-Hash` header is detected.
pub(crate) async fn new_xet_fetcher(
    url: &str,
    xet_hash: &str,
    total_size: u64,
    http_client: &Client,
) -> Result<RangeFetcher, FetchError> {
    let file_hash = MerkleHash::from_hex(xet_hash)
        .map_err(|e| FetchError::XetHashParse(e.to_string()))?;

    // Parse repo_id and revision from the HF URL
    let (repo_id, revision) = parse_hf_resolve_url(url)
        .ok_or_else(|| FetchError::XetTokenFetch(
            "could not parse repo_id/revision from HuggingFace URL".into(),
        ))?;

    // Fetch Xet read token
    let token_resp = fetch_xet_token(http_client, &repo_id, &revision).await?;

    tracing::info!(
        cas_url = %token_resp.cas_url,
        file_hash = %file_hash,
        total_size,
        "RangeFetcher ready (Xet)"
    );

    // Create CAS client with authentication
    let auth = AuthConfig {
        token: token_resp.access_token,
        token_expiration: token_resp.exp,
        token_refresher: Arc::new(xet_core_utils::auth::ErrTokenRefresher),
    };

    let cas_client = RemoteClient::new(
        &token_resp.cas_url,
        &Some(auth),
        "safetensors-streaming",
        false,
        None,
    );

    Ok(RangeFetcher {
        inner: FetcherInner::Xet {
            cas_client,
            file_hash,
            total_size,
        },
    })
}

/// Fetch a byte range via Xet reconstruction.
///
/// Uses `FileReconstructor::reconstruct_to_writer` to fetch the exact byte range
/// and write it to a `Vec<u8>`.
pub(crate) async fn fetch_range_xet(
    client: &Arc<dyn cas_client::Client>,
    file_hash: &MerkleHash,
    start: u64,
    end: u64,
) -> Result<Bytes, FetchError> {
    let expected_len = (end - start + 1) as usize;

    // FileRange uses exclusive end
    let range = FileRange::new(start, end + 1);

    tracing::debug!(
        file_hash = %file_hash,
        start,
        end,
        expected_len,
        "fetching byte range (Xet)"
    );

    let mut stream = FileReconstructor::new(client, *file_hash)
        .with_byte_range(range)
        .reconstruct_to_stream();

    let mut result = Vec::with_capacity(expected_len);
    while let Some(chunk) = stream
        .next()
        .await
        .map_err(|e| FetchError::XetReconstruction(e.to_string()))?
    {
        result.extend_from_slice(&chunk);
    }

    Ok(Bytes::from(result))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_hf_url_basic() {
        let url = "https://huggingface.co/meta-llama/Llama-3.1-8B/resolve/main/model-00001-of-00004.safetensors";
        let (repo_id, revision) = parse_hf_resolve_url(url).unwrap();
        assert_eq!(repo_id, "meta-llama/Llama-3.1-8B");
        assert_eq!(revision, "main");
    }

    #[test]
    fn parse_hf_url_with_commit_hash() {
        let url = "https://huggingface.co/org/model/resolve/abc123def/weights.safetensors";
        let (repo_id, revision) = parse_hf_resolve_url(url).unwrap();
        assert_eq!(repo_id, "org/model");
        assert_eq!(revision, "abc123def");
    }

    #[test]
    fn parse_hf_url_not_hf() {
        let result = parse_hf_resolve_url("https://example.com/file.safetensors");
        assert!(result.is_none());
    }
}
