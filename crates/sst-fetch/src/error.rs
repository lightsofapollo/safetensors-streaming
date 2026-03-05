/// Errors from HTTP fetching operations.
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
}
