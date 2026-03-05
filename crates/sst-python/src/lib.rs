use std::collections::HashMap;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyTuple};
use serde::Deserialize;
use sst_buffer::TensorChunk;
use sst_core::header::{parse_header, parse_header_json, parse_header_size};
use sst_core::pipeline::{PipelineConfig, StreamingPipeline};
use sst_core::types::{Header, TensorInfo};
use sst_fetch::RangeFetcher;
use sst_types::DType;

/// Parsed device specification.
#[derive(Debug, Clone)]
enum DeviceSpec {
    Cpu,
    Cuda(usize),
}

impl DeviceSpec {
    /// Return the PyTorch device string (e.g. "cpu", "cuda:0").
    fn to_torch_string(&self) -> String {
        match self {
            DeviceSpec::Cpu => "cpu".to_string(),
            DeviceSpec::Cuda(ordinal) => format!("cuda:{ordinal}"),
        }
    }
}

/// Parse a device string like "cpu", "cuda", "cuda:0", "cuda:3".
fn parse_device(device: &str) -> Result<DeviceSpec, SstPythonError> {
    let trimmed = device.trim();
    if trimmed == "cpu" {
        return Ok(DeviceSpec::Cpu);
    }
    if trimmed == "cuda" {
        return Ok(DeviceSpec::Cuda(0));
    }
    if let Some(ordinal_str) = trimmed.strip_prefix("cuda:") {
        let ordinal: usize = ordinal_str.parse().map_err(|_| {
            SstPythonError::InvalidDevice(format!(
                "invalid CUDA device ordinal: {ordinal_str:?}"
            ))
        })?;
        return Ok(DeviceSpec::Cuda(ordinal));
    }
    Err(SstPythonError::InvalidDevice(format!(
        "unsupported device: {trimmed:?} (expected \"cpu\", \"cuda\", or \"cuda:N\")"
    )))
}

/// Errors from the Python bindings.
#[derive(Debug, thiserror::Error)]
enum SstPythonError {
    #[error("core error: {0}")]
    Core(#[from] sst_core::CoreError),

    #[error("fetch error: {0}")]
    Fetch(#[from] sst_fetch::FetchError),

    #[error("tensor not found: {0}")]
    TensorNotFound(String),

    #[error("unsupported framework: {0} (only \"pt\" is supported)")]
    UnsupportedFramework(String),

    #[error("{0}")]
    InvalidDevice(String),

    #[error("{0}")]
    CudaNotCompiled(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("buffer error: {0}")]
    Buffer(#[from] sst_buffer::BufferError),

    #[error("Python error: {0}")]
    Python(#[from] PyErr),

    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("HTTP request error: {0}")]
    Reqwest(#[from] reqwest::Error),

    #[error("sharded index error: {0}")]
    ShardedIndex(String),

    #[error("join error: {0}")]
    Join(String),
}

impl From<SstPythonError> for PyErr {
    fn from(err: SstPythonError) -> PyErr {
        match err {
            SstPythonError::Python(py_err) => py_err,
            other => pyo3::exceptions::PyRuntimeError::new_err(other.to_string()),
        }
    }
}

impl From<std::convert::Infallible> for SstPythonError {
    fn from(e: std::convert::Infallible) -> Self {
        match e {}
    }
}

/// Check that CUDA support is compiled when a CUDA device is requested.
/// Returns Ok(()) for CPU devices, or for CUDA when the feature is enabled.
/// Returns an error for CUDA when compiled without the cuda feature.
fn require_cuda_compiled(spec: &DeviceSpec) -> Result<(), SstPythonError> {
    match spec {
        DeviceSpec::Cpu => Ok(()),
        DeviceSpec::Cuda(_) => {
            if cfg!(feature = "cuda") {
                Ok(())
            } else {
                Err(SstPythonError::CudaNotCompiled(
                    "CUDA support not compiled. Install safetensors-streaming-cu12 for GPU support."
                        .to_string(),
                ))
            }
        }
    }
}

/// Move a PyTorch tensor to the specified device (no-op for CPU).
fn tensor_to_device(
    py: Python<'_>,
    tensor: PyObject,
    spec: &DeviceSpec,
) -> PyResult<PyObject> {
    match spec {
        DeviceSpec::Cpu => Ok(tensor),
        DeviceSpec::Cuda(_) => {
            let device_str = spec.to_torch_string();
            let moved = tensor.call_method1(py, "to", (device_str,))?;
            Ok(moved)
        }
    }
}

/// Detect whether a string is a URL (http:// or https://).
fn is_url(s: &str) -> bool {
    s.starts_with("http://") || s.starts_with("https://")
}

/// Internal state for URL-based opening.
struct UrlMode {
    header: Header,
    fetcher: Arc<RangeFetcher>,
    runtime: tokio::runtime::Runtime,
}

/// Internal state for local file opening.
struct LocalMode {
    header: Header,
    data: bytes::Bytes,
}

enum OpenMode {
    Url(UrlMode),
    Local(LocalMode),
}

/// Context manager for streaming safetensors loading.
///
/// Supports both URLs (streamed via HTTP Range requests) and local files.
///
/// Usage:
///     with safe_open("https://..../model.safetensors", framework="pt") as f:
///         for key in f.keys():
///             tensor = f.get_tensor(key)
#[pyclass]
struct SafeOpen {
    mode: OpenMode,
    device: DeviceSpec,
}

impl SafeOpen {
    fn header(&self) -> &Header {
        match &self.mode {
            OpenMode::Url(u) => &u.header,
            OpenMode::Local(l) => &l.header,
        }
    }

    fn find_tensor(&self, name: &str) -> Result<&TensorInfo, SstPythonError> {
        self.header()
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| SstPythonError::TensorNotFound(name.to_string()))
    }

    fn fetch_tensor_bytes(&self, tensor: &TensorInfo) -> Result<bytes::Bytes, SstPythonError> {
        match &self.mode {
            OpenMode::Url(u) => {
                let (abs_start, abs_end) = tensor.absolute_offsets(u.header.data_start);
                if abs_start == abs_end {
                    return Ok(bytes::Bytes::new());
                }
                let data = u.runtime.block_on(
                    u.fetcher
                        .fetch_range(abs_start as u64, (abs_end - 1) as u64),
                )?;
                Ok(data)
            }
            OpenMode::Local(l) => {
                let (abs_start, abs_end) = tensor.absolute_offsets(l.header.data_start);
                Ok(l.data.slice(abs_start..abs_end))
            }
        }
    }
}

/// Convert a DType to the corresponding torch dtype attribute name.
fn torch_dtype_attr(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "float32",
        DType::F16 => "float16",
        DType::BF16 => "bfloat16",
        DType::F64 => "float64",
        DType::I8 => "int8",
        DType::I16 => "int16",
        DType::I32 => "int32",
        DType::I64 => "int64",
        DType::U8 => "uint8",
        DType::Bool => "bool",
    }
}

/// Create a torch.Tensor from raw bytes, dtype, and shape.
fn bytes_to_tensor(
    py: Python<'_>,
    data: &[u8],
    dtype: DType,
    shape: &[usize],
) -> PyResult<PyObject> {
    let torch = py.import("torch")?;
    let dtype_obj = torch.getattr(torch_dtype_attr(dtype))?;

    let data_bytes = PyByteArray::new(py, data);
    let kwargs = pyo3::types::PyDict::new(py);
    kwargs.set_item("dtype", dtype_obj)?;
    let tensor = torch.call_method("frombuffer", (data_bytes,), Some(&kwargs))?;

    let shape_tuple = PyTuple::new(py, shape.iter().map(|&s| s as i64))?;
    let reshaped = tensor.call_method1("reshape", (shape_tuple,))?;
    // Clone to own the data since frombuffer shares the buffer
    let owned = reshaped.call_method0("clone")?;
    Ok(owned.into_pyobject(py)?.into())
}

#[pymethods]
impl SafeOpen {
    #[new]
    #[pyo3(signature = (path_or_url, *, framework = "pt", device = "cpu"))]
    fn new(path_or_url: &str, framework: &str, device: &str) -> Result<Self, SstPythonError> {
        if framework != "pt" {
            return Err(SstPythonError::UnsupportedFramework(
                framework.to_string(),
            ));
        }
        let device_spec = parse_device(device)?;
        require_cuda_compiled(&device_spec)?;

        if is_url(path_or_url) {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(SstPythonError::Io)?;

            let (fetcher, header) = runtime.block_on(async {
                let fetcher = RangeFetcher::new(path_or_url).await?;
                let (size_bytes, header_json) = fetcher.fetch_header().await?;
                let header_size = parse_header_size(&size_bytes)?;
                let header = parse_header_json(&header_json, header_size)?;
                Ok::<_, SstPythonError>((fetcher, header))
            })?;

            Ok(Self {
                mode: OpenMode::Url(UrlMode {
                    header,
                    fetcher: Arc::new(fetcher),
                    runtime,
                }),
                device: device_spec,
            })
        } else {
            let file_bytes = std::fs::read(path_or_url)?;
            let data = bytes::Bytes::from(file_bytes);
            let header = parse_header(&data)?;

            Ok(Self {
                mode: OpenMode::Local(LocalMode { header, data }),
                device: device_spec,
            })
        }
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (_exc_type=None, _exc_val=None, _exc_tb=None))]
    fn __exit__(
        &mut self,
        _exc_type: Option<PyObject>,
        _exc_val: Option<PyObject>,
        _exc_tb: Option<PyObject>,
    ) -> bool {
        false
    }

    /// Return tensor names from the header (excluding __metadata__).
    fn keys(&self) -> Vec<String> {
        self.header()
            .tensors
            .iter()
            .map(|t| t.name.clone())
            .collect()
    }

    /// Fetch a tensor by name and return it as a torch.Tensor on the configured device.
    fn get_tensor(&self, py: Python<'_>, name: &str) -> PyResult<PyObject> {
        let tensor_info = self.find_tensor(name)?;
        let data = self.fetch_tensor_bytes(tensor_info)?;
        let cpu_tensor = bytes_to_tensor(py, &data, tensor_info.dtype, &tensor_info.shape)?;
        let result = tensor_to_device(py, cpu_tensor, &self.device)?;
        Ok(result)
    }

    /// Return metadata from the header.
    fn metadata(&self) -> HashMap<String, String> {
        self.header().metadata.clone()
    }
}

/// Open a safetensors file from a URL or local path.
///
/// Returns a context manager that provides `keys()`, `get_tensor()`, and `metadata()`.
#[pyfunction]
#[pyo3(signature = (path_or_url, *, framework = "pt", device = "cpu"))]
fn safe_open(path_or_url: &str, framework: &str, device: &str) -> Result<SafeOpen, SstPythonError> {
    SafeOpen::new(path_or_url, framework, device)
}

/// Load all tensors from a safetensors file and return them as a dict.
///
/// Drop-in replacement for `safetensors.torch.load_file()`.
/// Supports both local file paths and HTTP/HTTPS URLs.
///
/// For URLs, uses StreamingPipeline to fetch all tensors via range requests.
/// For local files, reads the file directly and parses the header.
///
/// Returns: Dict[str, torch.Tensor]
#[pyfunction]
#[pyo3(signature = (path_or_url, *, device = "cpu"))]
fn load_file(py: Python<'_>, path_or_url: &str, device: &str) -> Result<PyObject, SstPythonError> {
    let device_spec = parse_device(device)?;
    require_cuda_compiled(&device_spec)?;

    if is_url(path_or_url) {
        load_file_url(py, path_or_url, &device_spec)
    } else {
        load_file_local(py, path_or_url, &device_spec)
    }
}

/// Load all tensors from a URL using StreamingPipeline.
fn load_file_url(
    py: Python<'_>,
    url: &str,
    device: &DeviceSpec,
) -> Result<PyObject, SstPythonError> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(SstPythonError::Io)?;

    let chunks: Vec<TensorChunk> = runtime.block_on(async {
        let pipeline = StreamingPipeline::from_url(url, PipelineConfig::default()).await?;
        let mut consumer = pipeline.stream();
        let mut result = Vec::new();
        while let Some(chunk) = consumer.recv().await {
            result.push(chunk);
        }
        Ok::<_, SstPythonError>(result)
    })?;

    let dict = pyo3::types::PyDict::new(py);
    for chunk in &chunks {
        let cpu_tensor = bytes_to_tensor(py, &chunk.data, chunk.dtype, &chunk.shape)?;
        let tensor = tensor_to_device(py, cpu_tensor, device)?;
        dict.set_item(&chunk.name, tensor)?;
    }
    Ok(dict.into_pyobject(py)?.into())
}

/// Load all tensors from a local file.
fn load_file_local(
    py: Python<'_>,
    path: &str,
    device: &DeviceSpec,
) -> Result<PyObject, SstPythonError> {
    let file_bytes = std::fs::read(path)?;
    let data = bytes::Bytes::from(file_bytes);
    let header = parse_header(&data)?;

    let dict = pyo3::types::PyDict::new(py);
    for tensor in &header.tensors {
        let (abs_start, abs_end) = tensor.absolute_offsets(header.data_start);
        let tensor_data = &data[abs_start..abs_end];
        let cpu_tensor = bytes_to_tensor(py, tensor_data, tensor.dtype, &tensor.shape)?;
        let t = tensor_to_device(py, cpu_tensor, device)?;
        dict.set_item(&tensor.name, t)?;
    }
    Ok(dict.into_pyobject(py)?.into())
}

/// Internal state for the TensorStreamIterator, tracking which source we read from.
enum StreamSource {
    /// URL-based streaming using a consumer and tokio runtime.
    Url {
        consumer: sst_buffer::Consumer,
        runtime: tokio::runtime::Runtime,
    },
    /// Local file with pre-parsed header and data.
    Local {
        header: Header,
        data: bytes::Bytes,
        index: usize,
    },
}

/// Python iterator that yields (name, tensor) tuples as each tensor is fetched.
///
/// For URLs, tensors are streamed via the pipeline - each iteration may trigger a network fetch.
/// For local files, tensors are yielded from the parsed header in order.
#[pyclass]
struct TensorStreamIterator {
    source: StreamSource,
    device: DeviceSpec,
}

#[pymethods]
impl TensorStreamIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        match &mut self.source {
            StreamSource::Url { consumer, runtime } => {
                let chunk = runtime.block_on(consumer.recv());
                match chunk {
                    Some(c) => {
                        let cpu_tensor = bytes_to_tensor(py, &c.data, c.dtype, &c.shape)?;
                        let tensor = tensor_to_device(py, cpu_tensor, &self.device)?;
                        let tuple = PyTuple::new(py, &[c.name.into_pyobject(py)?.into_any(), tensor.bind(py).clone()])?;
                        Ok(Some(tuple.into_pyobject(py)?.into()))
                    }
                    None => Ok(None),
                }
            }
            StreamSource::Local {
                header,
                data,
                index,
            } => {
                if *index >= header.tensors.len() {
                    return Ok(None);
                }
                let tensor_info = &header.tensors[*index];
                *index += 1;
                let (abs_start, abs_end) = tensor_info.absolute_offsets(header.data_start);
                let tensor_data = &data[abs_start..abs_end];
                let cpu_tensor = bytes_to_tensor(py, tensor_data, tensor_info.dtype, &tensor_info.shape)?;
                let tensor = tensor_to_device(py, cpu_tensor, &self.device)?;
                let tuple = PyTuple::new(py, &[tensor_info.name.clone().into_pyobject(py)?.into_any(), tensor.bind(py).clone()])?;
                Ok(Some(tuple.into_pyobject(py)?.into()))
            }
        }
    }
}

/// Stream tensors from a safetensors file, yielding (name, tensor) tuples one at a time.
///
/// This is the streaming API - for URLs, tensors are fetched and yielded incrementally,
/// allowing processing to begin before all data is downloaded.
///
/// Usage:
///     for name, tensor in stream_tensors("https://..../model.safetensors"):
///         process(name, tensor)
#[pyfunction]
#[pyo3(signature = (path_or_url, *, device = "cpu"))]
fn stream_tensors(path_or_url: &str, device: &str) -> Result<TensorStreamIterator, SstPythonError> {
    let device_spec = parse_device(device)?;
    require_cuda_compiled(&device_spec)?;

    if is_url(path_or_url) {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(SstPythonError::Io)?;

        let consumer = runtime.block_on(async {
            let pipeline =
                StreamingPipeline::from_url(path_or_url, PipelineConfig::default()).await?;
            Ok::<_, SstPythonError>(pipeline.stream())
        })?;

        Ok(TensorStreamIterator {
            source: StreamSource::Url { consumer, runtime },
            device: device_spec,
        })
    } else {
        let file_bytes = std::fs::read(path_or_url)?;
        let data = bytes::Bytes::from(file_bytes);
        let header = parse_header(&data)?;

        Ok(TensorStreamIterator {
            source: StreamSource::Local {
                header,
                data,
                index: 0,
            },
            device: device_spec,
        })
    }
}

// ── Sharded model support ─────────────────────────────────────────────

/// Parsed `model.safetensors.index.json` structure.
#[derive(Debug, Deserialize)]
struct ShardedIndex {
    weight_map: HashMap<String, String>,
}

/// Fetch the index JSON from a URL or local path.
fn fetch_index(
    runtime: &tokio::runtime::Runtime,
    index_url_or_path: &str,
) -> Result<ShardedIndex, SstPythonError> {
    if is_url(index_url_or_path) {
        let json_bytes: bytes::Bytes = runtime.block_on(async {
            let client = reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .build()
                .map_err(SstPythonError::Reqwest)?;

            let resp = client
                .get(index_url_or_path)
                .send()
                .await
                .map_err(SstPythonError::Reqwest)?;

            let status = resp.status();
            if !status.is_success() {
                return Err(SstPythonError::ShardedIndex(format!(
                    "HTTP {status} fetching index: {index_url_or_path}"
                )));
            }

            resp.bytes().await.map_err(SstPythonError::Reqwest)
        })?;

        let index: ShardedIndex = serde_json::from_slice(&json_bytes)?;
        Ok(index)
    } else {
        let file_bytes = std::fs::read(index_url_or_path)?;
        let index: ShardedIndex = serde_json::from_slice(&file_bytes)?;
        Ok(index)
    }
}

/// Given the index URL/path and a shard filename, construct the full shard URL/path.
fn resolve_shard_path(index_url_or_path: &str, shard_filename: &str) -> String {
    if let Some(last_slash) = index_url_or_path.rfind('/') {
        format!("{}/{shard_filename}", &index_url_or_path[..last_slash])
    } else {
        shard_filename.to_string()
    }
}

/// Group the weight_map by shard file: returns map from shard filename to list of tensor names.
fn group_by_shard(weight_map: &HashMap<String, String>) -> HashMap<String, Vec<String>> {
    let mut shards: HashMap<String, Vec<String>> = HashMap::new();
    for (tensor_name, shard_file) in weight_map {
        shards
            .entry(shard_file.clone())
            .or_default()
            .push(tensor_name.clone());
    }
    shards
}

/// Load all tensors from a sharded safetensors model (multiple .safetensors files).
///
/// Accepts the URL or local path to a `model.safetensors.index.json` file.
/// Parses the index, discovers shard files, fetches all shards concurrently,
/// and returns a merged dict of all tensors.
///
/// Returns: Dict[str, torch.Tensor]
#[pyfunction]
#[pyo3(signature = (index_url_or_path, *, device = "cpu"))]
fn load_sharded(
    py: Python<'_>,
    index_url_or_path: &str,
    device: &str,
) -> Result<PyObject, SstPythonError> {
    let device_spec = parse_device(device)?;
    require_cuda_compiled(&device_spec)?;

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(SstPythonError::Io)?;

    let index = fetch_index(&runtime, index_url_or_path)?;
    let shard_groups = group_by_shard(&index.weight_map);

    tracing::info!(
        shards = shard_groups.len(),
        tensors = index.weight_map.len(),
        "loading sharded model"
    );

    // For each unique shard, resolve its full path/URL and fetch all tensors concurrently
    let all_chunks: Vec<TensorChunk> = if is_url(index_url_or_path) {
        runtime.block_on(async {
            let mut handles: Vec<tokio::task::JoinHandle<Result<Vec<TensorChunk>, SstPythonError>>> =
                Vec::with_capacity(shard_groups.len());

            for shard_filename in shard_groups.keys() {
                let shard_url = resolve_shard_path(index_url_or_path, shard_filename);
                let shard_filename_owned = shard_filename.clone();

                handles.push(tokio::spawn(async move {
                    tracing::info!(shard = %shard_filename_owned, "fetching shard");
                    let pipeline =
                        StreamingPipeline::from_url(&shard_url, PipelineConfig::default()).await?;
                    let mut consumer = pipeline.stream();
                    let mut chunks = Vec::new();
                    while let Some(chunk) = consumer.recv().await {
                        chunks.push(chunk);
                    }
                    Ok(chunks)
                }));
            }

            let mut all = Vec::new();
            for handle in handles {
                let chunks = handle
                    .await
                    .map_err(|e| SstPythonError::Join(e.to_string()))??;
                all.extend(chunks);
            }
            Ok::<_, SstPythonError>(all)
        })?
    } else {
        // Local: read each shard file
        let mut all = Vec::new();
        for shard_filename in shard_groups.keys() {
            let shard_path = resolve_shard_path(index_url_or_path, shard_filename);
            let file_bytes = std::fs::read(&shard_path)?;
            let data = bytes::Bytes::from(file_bytes);
            let header = parse_header(&data)?;
            for tensor in &header.tensors {
                let (abs_start, abs_end) = tensor.absolute_offsets(header.data_start);
                let tensor_data = data.slice(abs_start..abs_end);
                all.push(TensorChunk {
                    name: tensor.name.clone(),
                    data: tensor_data,
                    dtype: tensor.dtype,
                    shape: tensor.shape.clone(),
                });
            }
        }
        all
    };

    let dict = pyo3::types::PyDict::new(py);
    for chunk in &all_chunks {
        let cpu_tensor = bytes_to_tensor(py, &chunk.data, chunk.dtype, &chunk.shape)?;
        let tensor = tensor_to_device(py, cpu_tensor, &device_spec)?;
        dict.set_item(&chunk.name, tensor)?;
    }
    Ok(dict.into_pyobject(py)?.into())
}

/// Internal state for the sharded stream iterator.
enum ShardedStreamSource {
    /// URL-based: multi-threaded runtime with a merged consumer receiving from all shards.
    Url {
        consumer: tokio::sync::mpsc::Receiver<TensorChunk>,
        runtime: tokio::runtime::Runtime,
    },
    /// Local file-based: pre-collected chunks yielded one at a time.
    Local {
        chunks: Vec<TensorChunk>,
        index: usize,
    },
}

/// Python iterator that yields (name, tensor) tuples from a sharded model.
///
/// Tensors may arrive out of order since multiple shards are fetched concurrently.
#[pyclass]
struct ShardedTensorStreamIterator {
    source: ShardedStreamSource,
    device: DeviceSpec,
}

#[pymethods]
impl ShardedTensorStreamIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        match &mut self.source {
            ShardedStreamSource::Url { consumer, runtime } => {
                let chunk = runtime.block_on(consumer.recv());
                match chunk {
                    Some(c) => {
                        let cpu_tensor = bytes_to_tensor(py, &c.data, c.dtype, &c.shape)?;
                        let tensor = tensor_to_device(py, cpu_tensor, &self.device)?;
                        let tuple = PyTuple::new(
                            py,
                            &[
                                c.name.into_pyobject(py)?.into_any(),
                                tensor.bind(py).clone(),
                            ],
                        )?;
                        Ok(Some(tuple.into_pyobject(py)?.into()))
                    }
                    None => Ok(None),
                }
            }
            ShardedStreamSource::Local { chunks, index } => {
                if *index >= chunks.len() {
                    return Ok(None);
                }
                let chunk = &chunks[*index];
                *index += 1;
                let cpu_tensor = bytes_to_tensor(py, &chunk.data, chunk.dtype, &chunk.shape)?;
                let tensor = tensor_to_device(py, cpu_tensor, &self.device)?;
                let tuple = PyTuple::new(
                    py,
                    &[
                        chunk.name.clone().into_pyobject(py)?.into_any(),
                        tensor.bind(py).clone(),
                    ],
                )?;
                Ok(Some(tuple.into_pyobject(py)?.into()))
            }
        }
    }
}

/// Stream tensors from a sharded safetensors model, yielding (name, tensor) tuples.
///
/// Accepts the URL or local path to a `model.safetensors.index.json` file.
/// Tensors may arrive out of order since multiple shards are fetched concurrently.
///
/// Usage:
///     for name, tensor in stream_sharded("https://.../model.safetensors.index.json"):
///         process(name, tensor)
#[pyfunction]
#[pyo3(signature = (index_url_or_path, *, device = "cpu"))]
fn stream_sharded(
    index_url_or_path: &str,
    device: &str,
) -> Result<ShardedTensorStreamIterator, SstPythonError> {
    let device_spec = parse_device(device)?;
    require_cuda_compiled(&device_spec)?;

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(SstPythonError::Io)?;

    let index = fetch_index(&runtime, index_url_or_path)?;
    let shard_groups = group_by_shard(&index.weight_map);

    tracing::info!(
        shards = shard_groups.len(),
        tensors = index.weight_map.len(),
        "streaming sharded model"
    );

    if is_url(index_url_or_path) {
        // Create a merged channel: each shard spawns a task that sends chunks into it.
        let (merged_tx, merged_rx) =
            tokio::sync::mpsc::channel::<TensorChunk>(std::cmp::max(8, shard_groups.len() * 2));

        for shard_filename in shard_groups.keys() {
            let shard_url = resolve_shard_path(index_url_or_path, shard_filename);
            let tx = merged_tx.clone();
            let shard_filename_owned = shard_filename.clone();

            runtime.spawn(async move {
                let result: Result<(), SstPythonError> = async {
                    tracing::info!(shard = %shard_filename_owned, "streaming shard");
                    let pipeline =
                        StreamingPipeline::from_url(&shard_url, PipelineConfig::default()).await?;
                    let mut consumer = pipeline.stream();
                    while let Some(chunk) = consumer.recv().await {
                        if tx.send(chunk).await.is_err() {
                            // Receiver dropped, stop
                            break;
                        }
                    }
                    Ok(())
                }
                .await;
                if let Err(e) = result {
                    tracing::error!(shard = %shard_filename_owned, error = %e, "shard fetch failed");
                }
            });
        }

        // Drop the original sender so the channel closes when all shard tasks complete
        drop(merged_tx);

        Ok(ShardedTensorStreamIterator {
            source: ShardedStreamSource::Url {
                consumer: merged_rx,
                runtime,
            },
            device: device_spec,
        })
    } else {
        // Local: read all shard files and collect chunks
        let mut all_chunks = Vec::new();
        for shard_filename in shard_groups.keys() {
            let shard_path = resolve_shard_path(index_url_or_path, shard_filename);
            let file_bytes = std::fs::read(&shard_path)?;
            let data = bytes::Bytes::from(file_bytes);
            let header = parse_header(&data)?;
            for tensor in &header.tensors {
                let (abs_start, abs_end) = tensor.absolute_offsets(header.data_start);
                let tensor_data = data.slice(abs_start..abs_end);
                all_chunks.push(TensorChunk {
                    name: tensor.name.clone(),
                    data: tensor_data,
                    dtype: tensor.dtype,
                    shape: tensor.shape.clone(),
                });
            }
        }

        Ok(ShardedTensorStreamIterator {
            source: ShardedStreamSource::Local {
                chunks: all_chunks,
                index: 0,
            },
            device: device_spec,
        })
    }
}

/// Returns whether the package was compiled with CUDA support.
#[pyfunction]
fn cuda_available() -> bool {
    cfg!(feature = "cuda")
}

/// Returns the package version.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// A Python module for streaming safetensors loading.
#[pymodule]
fn safetensors_streaming(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_available, m)?)?;
    m.add_function(wrap_pyfunction!(safe_open, m)?)?;
    m.add_function(wrap_pyfunction!(load_file, m)?)?;
    m.add_function(wrap_pyfunction!(stream_tensors, m)?)?;
    m.add_function(wrap_pyfunction!(load_sharded, m)?)?;
    m.add_function(wrap_pyfunction!(stream_sharded, m)?)?;
    m.add_class::<SafeOpen>()?;
    m.add_class::<TensorStreamIterator>()?;
    m.add_class::<ShardedTensorStreamIterator>()?;
    Ok(())
}
