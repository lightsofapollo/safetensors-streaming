// sst-cli: Command-line interface for safetensors streaming operations

use clap::{Parser, Subcommand};
use sst_core::{PipelineConfig, StreamingPipeline};
use std::time::Instant;

#[derive(Parser)]
#[command(name = "sst", about = "Safetensors streaming toolkit")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Fetch and stream a safetensors file from a URL
    Fetch {
        /// URL of the safetensors file
        url: String,

        /// Run benchmark: print per-tensor timing and throughput summary
        #[arg(long, default_value_t = false)]
        benchmark: bool,

        /// Number of tensors to prefetch ahead
        #[arg(long, default_value_t = 3)]
        prefetch: usize,

        /// Ring buffer capacity (max buffered chunks)
        #[arg(long, default_value_t = 8)]
        buffer_size: usize,
    },
}

fn format_size(bytes: usize) -> String {
    if bytes >= 1024 * 1024 {
        format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.1} KiB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

fn format_shape(shape: &[usize]) -> String {
    let parts: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    format!("[{}]", parts.join(", "))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Command::Fetch {
            url,
            benchmark,
            prefetch,
            buffer_size,
        } => {
            run_fetch(&url, benchmark, prefetch, buffer_size).await?;
        }
    }

    Ok(())
}

async fn run_fetch(
    url: &str,
    benchmark: bool,
    prefetch: usize,
    buffer_size: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let config = PipelineConfig {
        prefetch_ahead: prefetch,
        buffer_capacity: buffer_size,
    };

    println!("Fetching: {url}");

    let pipeline_start = Instant::now();
    let pipeline = StreamingPipeline::from_url(url, config).await?;

    let header = pipeline.header();
    let tensor_count = header.tensors.len();
    let data_start = header.data_start;
    println!("Header: {tensor_count} tensors, data starts at byte {data_start}");
    println!();

    let mut consumer = pipeline.stream();
    let mut count = 0usize;
    let mut total_bytes = 0usize;
    let stream_start = Instant::now();

    while let Some(chunk) = consumer.recv().await {
        count += 1;
        let size = chunk.data.len();
        total_bytes += size;

        if benchmark {
            let elapsed = stream_start.elapsed();
            println!(
                "  {:<50} {:?}  {:<16} {:>10}    {:.0}ms",
                chunk.name,
                chunk.dtype,
                format_shape(&chunk.shape),
                format_size(size),
                elapsed.as_millis(),
            );
        } else {
            println!(
                "  {:<50} {:>10}",
                chunk.name,
                format_size(size),
            );
        }
    }

    let total_elapsed = pipeline_start.elapsed();

    if benchmark {
        let throughput_mib =
            total_bytes as f64 / (1024.0 * 1024.0) / total_elapsed.as_secs_f64();

        println!();
        println!("Summary:");
        println!("  Tensors:    {count}");
        println!("  Total:      {}", format_size(total_bytes));
        println!("  Time:       {:.2}s", total_elapsed.as_secs_f64());
        println!("  Throughput: {throughput_mib:.1} MiB/s");
    } else {
        println!();
        println!("{count} tensors, {} total", format_size(total_bytes));
    }

    Ok(())
}
