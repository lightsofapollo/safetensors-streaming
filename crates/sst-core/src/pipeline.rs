use crate::error::CoreError;
use crate::header::{parse_header_json, parse_header_size};
use crate::types::Header;
use sst_buffer::{Consumer, Producer, TensorChunk};
use sst_fetch::RangeFetcher;

/// Configuration for the streaming pipeline.
pub struct PipelineConfig {
    /// How many tensors to fetch ahead (default: 3). Currently informational;
    /// actual prefetch is bounded by `buffer_capacity`.
    pub prefetch_ahead: usize,
    /// Ring buffer capacity in number of TensorChunks (default: 8).
    pub buffer_capacity: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            prefetch_ahead: 3,
            buffer_capacity: 8,
        }
    }
}

/// A streaming pipeline that fetches tensors from a safetensors file via HTTP Range requests
/// and yields them through a bounded channel.
pub struct StreamingPipeline {
    fetcher: RangeFetcher,
    header: Header,
    buffer_capacity: usize,
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
            "pipeline header parsed"
        );

        Ok(Self {
            fetcher,
            header,
            buffer_capacity: config.buffer_capacity,
        })
    }

    /// Access the parsed header.
    pub fn header(&self) -> &Header {
        &self.header
    }

    /// Start streaming tensors. Spawns a background tokio task that fetches tensor data
    /// and pushes chunks through the returned Consumer.
    pub fn stream(self) -> Consumer {
        let (producer, consumer) = sst_buffer::channel(self.buffer_capacity);
        let header = self.header;
        let fetcher = self.fetcher;

        tokio::spawn(async move {
            if let Err(e) = fetch_all_tensors(&fetcher, &header, &producer).await {
                tracing::error!(error = %e, "pipeline fetcher task failed");
            }
            // producer is dropped here, signaling completion to consumer
        });

        consumer
    }
}

async fn fetch_all_tensors(
    fetcher: &RangeFetcher,
    header: &Header,
    producer: &Producer,
) -> Result<(), CoreError> {
    for tensor in &header.tensors {
        let (abs_start, abs_end) = tensor.absolute_offsets(header.data_start);

        // fetch_range uses inclusive end, but abs_end from absolute_offsets is exclusive
        let data = if abs_start == abs_end {
            // Zero-size tensor
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

    #[tokio::test]
    #[ignore] // hits network
    async fn full_pipeline_bert() {
        let config = PipelineConfig {
            prefetch_ahead: 3,
            buffer_capacity: 4,
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
        // bert_uncased_L-2_H-128_A-2 model.safetensors is ~17.7MB of tensor data
        assert!(
            total_bytes > 17_000_000 && total_bytes < 19_000_000,
            "total bytes {total_bytes} should be ~17.7MB"
        );
        println!("Total: {count} tensors, {total_bytes} bytes");
    }
}
