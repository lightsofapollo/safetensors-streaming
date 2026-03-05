/// Errors from buffer operations.
#[derive(Debug, thiserror::Error)]
pub enum BufferError {
    #[error("buffer channel closed — consumer was dropped")]
    Closed,
}
