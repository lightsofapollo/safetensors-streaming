/// Errors from fetching operations (HTTP and S3).
#[derive(Debug, thiserror::Error)]
pub enum FetchError {
    #[error("failed to build HTTP client: {0}")]
    HttpClient(reqwest::Error),

    #[error("HTTP request failed: {0}")]
    Request(reqwest::Error),

    #[error("failed to read response body: {0}")]
    Body(reqwest::Error),

    #[error("unexpected HTTP status: {0}")]
    UnexpectedStatus(u16),

    #[error("header response too short")]
    HeaderTooShort,

    #[error("missing Content-Length header")]
    MissingContentLength,

    #[error("missing Location header on redirect")]
    MissingLocation,

    #[error("invalid header value: {0}")]
    InvalidHeader(reqwest::header::ToStrError),

    #[error("invalid S3 URL: {0}")]
    InvalidS3Url(String),

    #[cfg(feature = "s3")]
    #[error("S3 GetObject failed: {0}")]
    S3GetObject(String),

    #[cfg(feature = "s3")]
    #[error("S3 HeadObject failed: {0}")]
    S3HeadObject(String),

    #[cfg(feature = "s3")]
    #[error("S3 response missing content length")]
    S3MissingContentLength,

    #[cfg(feature = "s3")]
    #[error("S3 byte stream error: {0}")]
    S3ByteStream(String),

    #[cfg(feature = "xet")]
    #[error("Xet reconstruction error: {0}")]
    XetReconstruction(String),

    #[cfg(feature = "xet")]
    #[error("Xet token fetch error: {0}")]
    XetTokenFetch(String),

    #[cfg(feature = "xet")]
    #[error("Xet hash parse error: {0}")]
    XetHashParse(String),
}
