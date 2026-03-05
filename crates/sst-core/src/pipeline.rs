use crate::error::CoreError;
use crate::header::{parse_header_json, parse_header_size};
use crate::types::Header;
#[cfg(test)]
use crate::types::TensorInfo;
use bytes::BytesMut;
use sst_buffer::{Consumer, Producer, TensorChunk};
use sst_fetch::RangeFetcher;
use std::sync::Arc;

/// Default batch size: 256 MiB. Tensors are grouped into contiguous batches
/// up to this size, each fetched with a single HTTP Range request.
/// Larger batches reduce per-request overhead at the cost of more memory.
const DEFAULT_BATCH_SIZE_BYTES: usize = 256 * 1024 * 1024;

/// Number of batches to prefetch concurrently ahead of consumption.
const PREFETCH_AHEAD: usize = 4;

/// Configuration for the streaming pipeline.
pub struct PipelineConfig {
    /// Ring buffer capacity in number of TensorChunks (default: 8).
    pub buffer_capacity: usize,
    /// Maximum bytes per batched Range request (default: 16 MiB).
    /// Set to 0 to disable batching (one request per tensor).
    pub batch_size_bytes: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            buffer_capacity: 8,
            batch_size_bytes: DEFAULT_BATCH_SIZE_BYTES,
        }
    }
}

/// A streaming pipeline that fetches tensors from a safetensors file via HTTP Range requests
/// and yields them through a bounded channel.
///
/// When `batch_size_bytes > 0`, consecutive tensors are grouped into batches and
/// fetched with a single Range request each. The next batch is prefetched concurrently
/// while the current batch is being split and sent to the consumer.
pub struct StreamingPipeline {
    fetcher: Arc<RangeFetcher>,
    header: Header,
    config: PipelineConfig,
}

impl StreamingPipeline {
    /// Create a pipeline from a URL. Resolves the URL and fetches+parses the header.
    pub async fn from_url(url: &str, config: PipelineConfig) -> Result<Self, CoreError> {
        let fetcher = RangeFetcher::new(url).await?;
        let (size_bytes, header_json) = fetcher.fetch_header().await?;
        let header_size = parse_header_size(&size_bytes)?;
        let header = parse_header_json(&header_json, header_size)?;

        tracing::info!(
            tensors = header.tensors.len(),
            data_start = header.data_start,
            batch_size_bytes = config.batch_size_bytes,
            "pipeline header parsed"
        );

        Ok(Self {
            fetcher: Arc::new(fetcher),
            header,
            config,
        })
    }

    /// Access the parsed header.
    pub fn header(&self) -> &Header {
        &self.header
    }

    /// Start streaming tensors. Spawns a background tokio task that fetches tensor data
    /// and pushes chunks through the returned Consumer.
    ///
    /// For HTTP backends, uses a single streaming GET request for the entire data section
    /// (matching curl-like throughput). For S3/Xet, falls back to batched Range requests.
    pub fn stream(self) -> Consumer {
        let (producer, consumer) = sst_buffer::channel(self.config.buffer_capacity);
        let header = self.header;
        let fetcher = self.fetcher;
        let batch_size_bytes = self.config.batch_size_bytes;

        tokio::spawn(async move {
            let result = if fetcher.is_http() {
                // Single curl connection — stream data and extract tensors as they arrive
                curl_streaming_fetch(&fetcher, &header, &producer).await
            } else if batch_size_bytes == 0 {
                fetch_all_tensors_unbatched(&fetcher, &header, &producer).await
            } else {
                fetch_all_tensors_batched(&fetcher, &header, &producer, batch_size_bytes).await
            };

            if let Err(e) = result {
                tracing::error!(error = %e, "pipeline fetcher task failed");
            }
            // producer is dropped here, signaling completion to consumer
        });

        consumer
    }
}

/// A group of consecutive tensors to fetch in a single Range request.
struct Batch {
    /// Absolute byte offset (inclusive) in the file.
    abs_start: u64,
    /// Absolute byte offset (exclusive) in the file.
    abs_end: u64,
    /// Tensors in this batch, with their absolute offsets.
    tensors: Vec<BatchTensor>,
}

struct BatchTensor {
    name: String,
    dtype: sst_types::DType,
    shape: Vec<usize>,
    /// Absolute start offset in the file.
    abs_start: u64,
    /// Absolute end offset in the file (exclusive).
    abs_end: u64,
}

/// Plan batches from sorted tensors.
fn plan_batches(header: &Header, max_batch_bytes: usize) -> Vec<Batch> {
    let mut batches: Vec<Batch> = Vec::new();

    for tensor in &header.tensors {
        let (abs_start, abs_end) = tensor.absolute_offsets(header.data_start);
        let abs_start = abs_start as u64;
        let abs_end = abs_end as u64;

        let bt = BatchTensor {
            name: tensor.name.clone(),
            dtype: tensor.dtype,
            shape: tensor.shape.clone(),
            abs_start,
            abs_end,
        };

        let extend_current = if let Some(current) = batches.last() {
            // Would adding this tensor keep the batch within the size limit?
            (abs_end - current.abs_start) as usize <= max_batch_bytes
        } else {
            false
        };

        if extend_current {
            let current = batches.last_mut().map(|b| {
                b.abs_end = abs_end;
                b.tensors.push(bt);
            });
            // Silence unused variable warning — the mutation already happened
            let _ = current;
        } else {
            batches.push(Batch {
                abs_start,
                abs_end,
                tensors: vec![bt],
            });
        }
    }

    batches
}

/// Fetch all tensors with batching and concurrent prefetch of the next batch.
async fn fetch_all_tensors_batched(
    fetcher: &Arc<RangeFetcher>,
    header: &Header,
    producer: &Producer,
    max_batch_bytes: usize,
) -> Result<(), CoreError> {
    let batches = plan_batches(header, max_batch_bytes);

    tracing::info!(
        total_tensors = header.tensors.len(),
        total_batches = batches.len(),
        max_batch_bytes,
        "batched fetch plan"
    );

    if batches.is_empty() {
        return Ok(());
    }

    // Launch up to PREFETCH_AHEAD concurrent fetches
    let mut pending: std::collections::VecDeque<tokio::task::JoinHandle<Result<bytes::Bytes, sst_fetch::FetchError>>> =
        std::collections::VecDeque::new();

    // Seed the prefetch queue
    let prefetch_count = batches.len().min(PREFETCH_AHEAD);
    for batch in batches.iter().take(prefetch_count) {
        let start = batch.abs_start;
        let end = batch.abs_end;
        let f = Arc::clone(fetcher);
        pending.push_back(tokio::spawn(async move { f.fetch_batch(start, end).await }));
    }
    let mut next_to_launch = prefetch_count;

    for (i, batch) in batches.iter().enumerate() {
        // Await the next completed fetch
        let batch_data = pending
            .pop_front()
            .expect("prefetch queue empty")
            .await
            .map_err(|e| CoreError::JoinError(e.to_string()))??;

        // Launch next fetch to keep the pipeline full
        if next_to_launch < batches.len() {
            let nb = &batches[next_to_launch];
            let start = nb.abs_start;
            let end = nb.abs_end;
            let f = Arc::clone(fetcher);
            pending.push_back(tokio::spawn(async move { f.fetch_batch(start, end).await }));
            next_to_launch += 1;
        }

        tracing::debug!(
            batch = i,
            tensors = batch.tensors.len(),
            bytes = batch_data.len(),
            "received batch"
        );

        // Split batch data into individual tensor chunks
        for bt in &batch.tensors {
            let offset_in_batch = (bt.abs_start - batch.abs_start) as usize;
            let length = (bt.abs_end - bt.abs_start) as usize;

            let data = if length == 0 {
                bytes::Bytes::new()
            } else {
                batch_data.slice(offset_in_batch..offset_in_batch + length)
            };

            tracing::debug!(
                name = %bt.name,
                bytes = data.len(),
                "split tensor from batch"
            );

            let chunk = TensorChunk {
                name: bt.name.clone(),
                data,
                dtype: bt.dtype,
                shape: bt.shape.clone(),
            };

            producer.send(chunk).await?;
        }
    }

    Ok(())
}

/// Download all tensor data via a single GET request, then slice into tensors.
/// Uses `response.bytes()` for maximum download throughput.
async fn fetch_all_tensors_streaming(
    fetcher: &Arc<RangeFetcher>,
    header: &Header,
    producer: &Producer,
) -> Result<(), CoreError> {
    let data_start = header.data_start as u64;
    let total_size = fetcher.total_size();

    tracing::info!(
        total_tensors = header.tensors.len(),
        data_start,
        total_size,
        "single GET download mode"
    );

    // Download entire data section at once — maximum throughput, single HTTP connection
    let data = fetcher
        .fetch_range(data_start, total_size - 1)
        .await?;

    tracing::info!(bytes = data.len(), "data section downloaded");

    // Slice into individual tensors (zero-copy via Bytes::slice)
    for tensor in &header.tensors {
        let rel_start = tensor.data_offsets.0;
        let rel_end = tensor.data_offsets.1;

        let tensor_data = if rel_start == rel_end {
            bytes::Bytes::new()
        } else {
            data.slice(rel_start..rel_end)
        };

        let chunk = TensorChunk {
            name: tensor.name.clone(),
            data: tensor_data,
            dtype: tensor.dtype,
            shape: tensor.shape.clone(),
        };

        producer.send(chunk).await?;
    }

    Ok(())
}

/// Stream data via a single curl connection and extract tensors as they arrive.
/// This avoids large memory allocations — we only buffer enough data to complete
/// the current tensor. Uses sync channel for backpressure between the curl
/// blocking thread and the async producer.
async fn curl_streaming_fetch(
    fetcher: &Arc<RangeFetcher>,
    header: &Header,
    producer: &Producer,
) -> Result<(), CoreError> {
    let url = fetcher.url();
    let data_start = header.data_start as u64;
    let total_size = fetcher.total_size();

    // Pre-compute tensor boundaries (relative to data section start)
    let tensor_info: Vec<(String, sst_types::DType, Vec<usize>, usize, usize)> = header
        .tensors
        .iter()
        .map(|t| {
            (
                t.name.clone(),
                t.dtype,
                t.shape.clone(),
                t.data_offsets.0,
                t.data_offsets.1,
            )
        })
        .collect();

    let tensor_count = tensor_info.len();

    tracing::info!(
        tensors = tensor_count,
        data_start,
        total_size,
        "curl streaming mode"
    );

    // Bounded sync channel — blocks the curl thread when consumer is slow
    let (tx, rx) = std::sync::mpsc::sync_channel::<TensorChunk>(8);

    let handle = tokio::task::spawn_blocking(move || -> Result<(), CoreError> {
        let mut easy = curl::easy::Easy::new();
        easy.url(&url)
            .map_err(|e| CoreError::Fetch(sst_fetch::FetchError::Curl(e.to_string())))?;
        let range = format!("{data_start}-{}", total_size - 1);
        easy.range(&range)
            .map_err(|e| CoreError::Fetch(sst_fetch::FetchError::Curl(e.to_string())))?;
        easy.follow_location(true)
            .map_err(|e| CoreError::Fetch(sst_fetch::FetchError::Curl(e.to_string())))?;
        easy.buffer_size(512 * 1024)
            .map_err(|e| CoreError::Fetch(sst_fetch::FetchError::Curl(e.to_string())))?;

        // State shared with the write callback
        let mut buffer = BytesMut::new();
        let mut tensor_idx: usize = 0;
        let mut bytes_received: usize = 0;

        {
            let tx = &tx;
            let tensor_info = &tensor_info;
            let buffer = &mut buffer;
            let tensor_idx = &mut tensor_idx;
            let bytes_received = &mut bytes_received;

            let mut transfer = easy.transfer();
            transfer
                .write_function(|chunk| {
                    buffer.extend_from_slice(chunk);
                    *bytes_received += chunk.len();

                    // Extract completed tensors
                    while *tensor_idx < tensor_info.len() {
                        let (ref name, dtype, ref shape, rel_start, rel_end) =
                            tensor_info[*tensor_idx];
                        let tensor_size = rel_end - rel_start;

                        // Check if we have enough data for this tensor
                        // rel_end is relative to data section start, bytes_received is how much
                        // we've downloaded from data section start
                        if *bytes_received < rel_end {
                            break;
                        }

                        // Extract tensor data from buffer
                        let data = if tensor_size == 0 {
                            bytes::Bytes::new()
                        } else {
                            // The tensor starts at rel_start in the data section.
                            // Our buffer starts at (bytes_received - buffer.len()) relative to data section.
                            let buf_start =
                                *bytes_received - buffer.len(); // offset of buffer[0] in data section
                            let offset_in_buf = rel_start - buf_start;
                            let data =
                                bytes::Bytes::copy_from_slice(&buffer[offset_in_buf..offset_in_buf + tensor_size]);
                            data
                        };

                        let chunk = TensorChunk {
                            name: name.clone(),
                            data,
                            dtype,
                            shape: shape.clone(),
                        };

                        // Send to async consumer — blocks if channel is full (backpressure)
                        if tx.send(chunk).is_err() {
                            // Consumer dropped — abort download
                            return Err(curl::easy::WriteError::Pause);
                        }

                        *tensor_idx += 1;

                        // Drain buffer up to the next tensor's start (or the end of current tensor)
                        // to keep memory usage low
                        if *tensor_idx < tensor_info.len() {
                            let next_start = tensor_info[*tensor_idx].3;
                            let buf_start = *bytes_received - buffer.len();
                            if next_start > buf_start {
                                let drain_to = next_start - buf_start;
                                if drain_to <= buffer.len() {
                                    let _ = buffer.split_to(drain_to);
                                }
                            }
                        } else {
                            // All tensors extracted, clear buffer
                            buffer.clear();
                        }
                    }

                    Ok(chunk.len())
                })
                .map_err(|e| {
                    CoreError::Fetch(sst_fetch::FetchError::Curl(e.to_string()))
                })?;

            transfer.perform().map_err(|e| {
                CoreError::Fetch(sst_fetch::FetchError::Curl(e.to_string()))
            })?;
        }

        let status = easy
            .response_code()
            .map_err(|e| CoreError::Fetch(sst_fetch::FetchError::Curl(e.to_string())))?;

        if status != 206 && status != 200 {
            return Err(CoreError::Fetch(
                sst_fetch::FetchError::UnexpectedStatus(status as u16),
            ));
        }

        Ok(())
    });

    // Bridge: read from sync channel and forward to async producer
    // This runs on the tokio runtime and awaits the producer.send()
    loop {
        // Use spawn_blocking to avoid blocking the async runtime while waiting on the channel
        let chunk = {
            let rx_ref = &rx;
            match rx_ref.try_recv() {
                Ok(chunk) => Some(chunk),
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    // Yield to tokio, then retry
                    tokio::task::yield_now().await;
                    // Try again, but this time block briefly
                    match rx.recv_timeout(std::time::Duration::from_millis(1)) {
                        Ok(chunk) => Some(chunk),
                        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                            // Check if the curl thread is done
                            if handle.is_finished() {
                                match rx.try_recv() {
                                    Ok(chunk) => Some(chunk),
                                    Err(_) => None,
                                }
                            } else {
                                continue;
                            }
                        }
                        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => None,
                    }
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => None,
            }
        };

        match chunk {
            Some(chunk) => {
                producer.send(chunk).await?;
            }
            None => {
                // Channel closed — curl thread is done
                break;
            }
        }
    }

    // Wait for the curl thread to finish and propagate errors
    handle
        .await
        .map_err(|e| CoreError::JoinError(e.to_string()))??;

    Ok(())
}

/// Fallback: fetch each tensor individually (batch_size_bytes = 0).
async fn fetch_all_tensors_unbatched(
    fetcher: &Arc<RangeFetcher>,
    header: &Header,
    producer: &Producer,
) -> Result<(), CoreError> {
    for tensor in &header.tensors {
        let (abs_start, abs_end) = tensor.absolute_offsets(header.data_start);

        let data = if abs_start == abs_end {
            bytes::Bytes::new()
        } else {
            fetcher
                .fetch_range(abs_start as u64, (abs_end - 1) as u64)
                .await?
        };

        tracing::debug!(
            name = %tensor.name,
            bytes = data.len(),
            "fetched tensor"
        );

        let chunk = TensorChunk {
            name: tensor.name.clone(),
            data,
            dtype: tensor.dtype,
            shape: tensor.shape.clone(),
        };

        producer.send(chunk).await?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const BERT_URL: &str = "https://huggingface.co/google/bert_uncased_L-2_H-128_A-2/resolve/main/model.safetensors";

    #[test]
    fn plan_batches_groups_small_tensors() {
        // Create a fake header with 5 small tensors (1000 bytes each)
        let tensors: Vec<TensorInfo> = (0..5)
            .map(|i| TensorInfo {
                name: format!("tensor_{i}"),
                dtype: sst_types::DType::F32,
                shape: vec![250],
                data_offsets: (i * 1000, (i + 1) * 1000),
            })
            .collect();

        let header = Header {
            header_size: 100,
            data_start: 108,
            tensors,
            metadata: Default::default(),
        };

        // With a 4000 byte batch limit, should get 2 batches: [0..3] and [3..4]
        let batches = plan_batches(&header, 4000);
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].tensors.len(), 4);
        assert_eq!(batches[1].tensors.len(), 1);
    }

    #[test]
    fn plan_batches_single_large_tensor() {
        let tensors = vec![TensorInfo {
            name: "big".into(),
            dtype: sst_types::DType::F32,
            shape: vec![1000000],
            data_offsets: (0, 4_000_000),
        }];

        let header = Header {
            header_size: 50,
            data_start: 58,
            tensors,
            metadata: Default::default(),
        };

        // Even with a small batch size, a single tensor must be in its own batch
        let batches = plan_batches(&header, 1000);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].tensors.len(), 1);
    }

    #[test]
    fn plan_batches_zero_size_tensor() {
        let tensors = vec![
            TensorInfo {
                name: "empty".into(),
                dtype: sst_types::DType::F32,
                shape: vec![0],
                data_offsets: (0, 0),
            },
            TensorInfo {
                name: "real".into(),
                dtype: sst_types::DType::F32,
                shape: vec![100],
                data_offsets: (0, 400),
            },
        ];

        let header = Header {
            header_size: 50,
            data_start: 58,
            tensors,
            metadata: Default::default(),
        };

        let batches = plan_batches(&header, 16 * 1024 * 1024);
        // Both should be in one batch since they fit
        assert_eq!(batches.len(), 1);
    }

    #[test]
    fn plan_batches_unlimited_groups_all() {
        let tensors: Vec<TensorInfo> = (0..100)
            .map(|i| TensorInfo {
                name: format!("t_{i}"),
                dtype: sst_types::DType::F32,
                shape: vec![256],
                data_offsets: (i * 1024, (i + 1) * 1024),
            })
            .collect();

        let header = Header {
            header_size: 200,
            data_start: 208,
            tensors,
            metadata: Default::default(),
        };

        // 16MB batch should fit all 100 tensors (100 * 1024 = ~100KB)
        let batches = plan_batches(&header, 16 * 1024 * 1024);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].tensors.len(), 100);
    }

    #[tokio::test]
    #[ignore] // hits network
    async fn full_pipeline_bert() {
        let config = PipelineConfig {
            buffer_capacity: 4,
            batch_size_bytes: DEFAULT_BATCH_SIZE_BYTES,
        };

        let pipeline = StreamingPipeline::from_url(BERT_URL, config).await.unwrap();
        let expected_count = pipeline.header().tensors.len();
        let mut consumer = pipeline.stream();

        let mut count = 0;
        let mut total_bytes = 0usize;

        while let Some(chunk) = consumer.recv().await {
            count += 1;
            total_bytes += chunk.data.len();
            println!(
                "  [{count}/{expected_count}] {} {:?} {:?} {} bytes",
                chunk.name, chunk.dtype, chunk.shape, chunk.data.len()
            );
        }

        assert_eq!(count, expected_count, "should receive all tensors");
        assert!(
            total_bytes > 17_000_000 && total_bytes < 19_000_000,
            "total bytes {total_bytes} should be ~17.7MB"
        );
        println!("Total: {count} tensors, {total_bytes} bytes");
    }

    #[tokio::test]
    #[ignore] // hits network
    async fn batched_matches_unbatched() {
        // Fetch with batching
        let batched_config = PipelineConfig {
            buffer_capacity: 64,
            batch_size_bytes: DEFAULT_BATCH_SIZE_BYTES,
        };
        let pipeline = StreamingPipeline::from_url(BERT_URL, batched_config).await.unwrap();
        let mut consumer = pipeline.stream();
        let mut batched_results: Vec<(String, Vec<u8>)> = Vec::new();
        while let Some(chunk) = consumer.recv().await {
            batched_results.push((chunk.name.clone(), chunk.data.to_vec()));
        }

        // Fetch without batching
        let unbatched_config = PipelineConfig {
            buffer_capacity: 64,
            batch_size_bytes: 0,
        };
        let pipeline = StreamingPipeline::from_url(BERT_URL, unbatched_config).await.unwrap();
        let mut consumer = pipeline.stream();
        let mut unbatched_results: Vec<(String, Vec<u8>)> = Vec::new();
        while let Some(chunk) = consumer.recv().await {
            unbatched_results.push((chunk.name.clone(), chunk.data.to_vec()));
        }

        // Same count
        assert_eq!(
            batched_results.len(),
            unbatched_results.len(),
            "tensor count mismatch"
        );

        // Same names and data
        for (batched, unbatched) in batched_results.iter().zip(unbatched_results.iter()) {
            assert_eq!(batched.0, unbatched.0, "tensor name mismatch");
            assert_eq!(
                batched.1, unbatched.1,
                "tensor data mismatch for {}",
                batched.0
            );
        }
    }
}
