// sst-python: PyO3 bindings for safetensors streaming
//
// Modules (planned):
// - loader: Python-facing streaming loader class
// - tensor: Python tensor wrapper with zero-copy support
// - config: Configuration dataclasses

use pyo3::prelude::*;

/// A Python module for streaming safetensors loading.
#[pymodule]
fn sst_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
