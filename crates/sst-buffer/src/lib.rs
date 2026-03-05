// sst-buffer: Ring buffer for streaming tensor data
//
// Decouples the HTTP fetcher (producer) from the tensor consumer.
// Provides backpressure (producer blocks when full) and FIFO ordering.
// Uses tokio bounded channels internally — swappable to pinned CUDA memory later.

pub mod error;

pub use error::BufferError;

use bytes::Bytes;
use futures::Stream;
use sst_types::DType;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

/// A named chunk of tensor data flowing through the buffer.
#[derive(Debug, Clone)]
pub struct TensorChunk {
    pub name: String,
    pub data: Bytes,
    pub dtype: DType,
    pub shape: Vec<usize>,
}

/// Create a bounded producer/consumer pair.
///
/// `capacity` is the max number of `TensorChunk`s buffered before the
/// producer blocks (backpressure).
pub fn channel(capacity: usize) -> (Producer, Consumer) {
    let (tx, rx) = mpsc::channel(capacity);
    (Producer { tx }, Consumer { rx })
}

/// Producer half — pushes fetched tensor chunks into the buffer.
pub struct Producer {
    tx: mpsc::Sender<TensorChunk>,
}

impl Producer {
    /// Send a tensor chunk. Awaits if the buffer is full (backpressure).
    pub async fn send(&self, chunk: TensorChunk) -> Result<(), BufferError> {
        self.tx.send(chunk).await.map_err(|_| BufferError::Closed)
    }

    /// Signal completion by dropping the sender.
    /// This is the same as just dropping the `Producer`.
    pub fn close(self) {
        drop(self);
    }
}

/// Consumer half — pulls tensor chunks out in FIFO order.
pub struct Consumer {
    rx: mpsc::Receiver<TensorChunk>,
}

impl Consumer {
    /// Receive the next tensor chunk.
    /// Returns `None` when the producer is dropped and the buffer is empty.
    pub async fn recv(&mut self) -> Option<TensorChunk> {
        self.rx.recv().await
    }

    /// Convert into an async `Stream` for use with `StreamExt` combinators.
    pub fn into_stream(self) -> impl Stream<Item = TensorChunk> {
        ReceiverStream::new(self.rx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    fn make_chunk(name: &str, size: usize) -> TensorChunk {
        TensorChunk {
            name: name.to_string(),
            data: Bytes::from(vec![0xABu8; size]),
            dtype: DType::F32,
            shape: vec![size / 4],
        }
    }

    #[tokio::test]
    async fn basic_send_recv() {
        let (producer, mut consumer) = channel(8);

        for i in 0..5 {
            let chunk = make_chunk(&format!("tensor_{i}"), 64);
            producer.send(chunk).await.unwrap();
        }
        producer.close();

        for i in 0..5 {
            let chunk = consumer.recv().await.unwrap();
            assert_eq!(chunk.name, format!("tensor_{i}"));
            assert_eq!(chunk.data.len(), 64);
        }
        assert!(consumer.recv().await.is_none());
    }

    #[tokio::test]
    async fn backpressure() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let (producer, mut consumer) = channel(2);
        let sent = Arc::new(AtomicUsize::new(0));
        let sent_clone = sent.clone();

        let producer_task = tokio::spawn(async move {
            for i in 0..5 {
                let chunk = make_chunk(&format!("tensor_{i}"), 32);
                producer.send(chunk).await.unwrap();
                sent_clone.fetch_add(1, Ordering::SeqCst);
            }
        });

        // Give the producer time to fill the buffer
        tokio::task::yield_now().await;
        tokio::task::yield_now().await;

        // With capacity=2, at most 2 items can be buffered before backpressure kicks in.
        // The producer may have sent up to 3 (2 buffered + 1 being received).
        let sent_so_far = sent.load(Ordering::SeqCst);
        assert!(sent_so_far <= 3, "producer should be blocked by backpressure, but sent {sent_so_far}");

        // Drain all
        let mut received = 0;
        while consumer.recv().await.is_some() {
            received += 1;
        }
        assert_eq!(received, 5);
        producer_task.await.unwrap();
    }

    #[tokio::test]
    async fn consumer_gets_none_after_producer_drops() {
        let (producer, mut consumer) = channel(4);
        producer.send(make_chunk("a", 16)).await.unwrap();
        drop(producer);

        assert!(consumer.recv().await.is_some());
        assert!(consumer.recv().await.is_none());
    }

    #[tokio::test]
    async fn concurrent_producer_consumer() {
        let (producer, consumer) = channel(4);
        let count = 100;

        let producer_task = tokio::spawn(async move {
            for i in 0..count {
                producer.send(make_chunk(&format!("t_{i}"), 128)).await.unwrap();
            }
        });

        let consumer_task = tokio::spawn(async move {
            let items: Vec<TensorChunk> = consumer.into_stream().collect().await;
            items
        });

        producer_task.await.unwrap();
        let items = consumer_task.await.unwrap();
        assert_eq!(items.len(), count);

        // Verify FIFO order
        for (i, item) in items.iter().enumerate() {
            assert_eq!(item.name, format!("t_{i}"));
        }
    }

    #[tokio::test]
    async fn large_data_varying_sizes() {
        let sizes = [512, 4096, 65536, 1024 * 1024, 10 * 1024 * 1024];
        let (producer, mut consumer) = channel(2);

        let producer_task = tokio::spawn(async move {
            for (i, &size) in sizes.iter().enumerate() {
                producer.send(make_chunk(&format!("big_{i}"), size)).await.unwrap();
            }
        });

        let mut received = Vec::new();
        while let Some(chunk) = consumer.recv().await {
            received.push(chunk.data.len());
        }

        producer_task.await.unwrap();
        assert_eq!(received, sizes.to_vec());
    }
}
