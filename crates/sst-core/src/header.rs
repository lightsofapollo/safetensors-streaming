use std::collections::HashMap;
use std::str::FromStr;

use crate::error::CoreError;
use crate::types::{DType, Header, TensorInfo};

/// Parse the header size (u64) from the first 8 bytes of a safetensors file.
pub fn parse_header_size(bytes: &[u8]) -> Result<u64, CoreError> {
    if bytes.len() < 8 {
        return Err(CoreError::HeaderTooShort {
            got: bytes.len(),
            expected: 8,
        });
    }

    let size_bytes: [u8; 8] = bytes[..8]
        .try_into()
        .map_err(|_| CoreError::HeaderTooShort {
            got: bytes.len(),
            expected: 8,
        })?;

    Ok(u64::from_le_bytes(size_bytes))
}

/// Parse the JSON header bytes into a `Header` struct.
///
/// `json_bytes` should be the raw JSON (bytes 8..8+header_size).
/// `header_size` is the u64 value parsed from the first 8 bytes.
pub fn parse_header_json(json_bytes: &[u8], header_size: u64) -> Result<Header, CoreError> {
    let header_json: serde_json::Value =
        serde_json::from_slice(json_bytes).map_err(CoreError::InvalidJson)?;

    let map = header_json
        .as_object()
        .ok_or(CoreError::InvalidHeaderStructure)?;

    let mut tensors = Vec::new();
    let mut metadata = HashMap::new();

    for (name, value) in map {
        if name == "__metadata__" {
            if let Some(meta_obj) = value.as_object() {
                for (k, v) in meta_obj {
                    if let Some(s) = v.as_str() {
                        metadata.insert(k.clone(), s.to_string());
                    }
                }
            }
            continue;
        }

        let dtype_str = value
            .get("dtype")
            .and_then(|v| v.as_str())
            .ok_or_else(|| CoreError::MissingField {
                tensor: name.clone(),
                field: "dtype".to_string(),
            })?;

        let offsets = value
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| CoreError::MissingField {
                tensor: name.clone(),
                field: "data_offsets".to_string(),
            })?;

        let start = offsets
            .first()
            .and_then(|v| v.as_u64())
            .ok_or_else(|| CoreError::MissingField {
                tensor: name.clone(),
                field: "data_offsets[0]".to_string(),
            })? as usize;

        let end = offsets
            .get(1)
            .and_then(|v| v.as_u64())
            .ok_or_else(|| CoreError::MissingField {
                tensor: name.clone(),
                field: "data_offsets[1]".to_string(),
            })? as usize;

        let shape: Vec<usize> = value
            .get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| CoreError::MissingField {
                tensor: name.clone(),
                field: "shape".to_string(),
            })?
            .iter()
            .filter_map(|v| v.as_u64().map(|n| n as usize))
            .collect();

        tensors.push(TensorInfo {
            name: name.clone(),
            dtype: DType::from_str(dtype_str)?,
            shape,
            data_offsets: (start, end),
        });
    }

    // Sort by offset for sequential fetch planning
    tensors.sort_by_key(|t| t.data_offsets.0);

    let data_start = 8 + header_size as usize;

    Ok(Header {
        header_size,
        data_start,
        tensors,
        metadata,
    })
}

/// Parse a complete safetensors header from raw bytes (8-byte size prefix + JSON).
pub fn parse_header(bytes: &[u8]) -> Result<Header, CoreError> {
    let header_size = parse_header_size(bytes)?;

    let header_end = 8 + header_size as usize;
    if bytes.len() < header_end {
        return Err(CoreError::HeaderTooShort {
            got: bytes.len(),
            expected: header_end,
        });
    }

    parse_header_json(&bytes[8..header_end], header_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal safetensors header buffer with 2 tensors and metadata.
    fn make_test_buffer() -> Vec<u8> {
        let json = serde_json::json!({
            "__metadata__": {
                "format": "pt",
                "source": "test"
            },
            "weight": {
                "dtype": "F32",
                "shape": [3, 4],
                "data_offsets": [0, 48]
            },
            "bias": {
                "dtype": "F16",
                "shape": [4],
                "data_offsets": [48, 56]
            }
        });
        let json_bytes = serde_json::to_vec(&json).unwrap();
        let header_size = json_bytes.len() as u64;

        let mut buf = Vec::new();
        buf.extend_from_slice(&header_size.to_le_bytes());
        buf.extend_from_slice(&json_bytes);
        buf
    }

    #[test]
    fn parse_minimal_header_with_two_tensors() {
        let buf = make_test_buffer();
        let header = parse_header(&buf).unwrap();

        assert_eq!(header.tensors.len(), 2);

        // Tensors should be sorted by offset
        let first = &header.tensors[0];
        let second = &header.tensors[1];
        assert!(first.data_offsets.0 <= second.data_offsets.0);
    }

    #[test]
    fn dtype_byte_sizes() {
        assert_eq!(DType::Bool.byte_size(), 1);
        assert_eq!(DType::U8.byte_size(), 1);
        assert_eq!(DType::I8.byte_size(), 1);
        assert_eq!(DType::I16.byte_size(), 2);
        assert_eq!(DType::F16.byte_size(), 2);
        assert_eq!(DType::BF16.byte_size(), 2);
        assert_eq!(DType::I32.byte_size(), 4);
        assert_eq!(DType::F32.byte_size(), 4);
        assert_eq!(DType::I64.byte_size(), 8);
        assert_eq!(DType::F64.byte_size(), 8);
    }

    #[test]
    fn tensor_byte_size_matches_offsets() {
        let buf = make_test_buffer();
        let header = parse_header(&buf).unwrap();

        // Find the weight tensor (3x4 F32 = 48 bytes)
        let weight = header.tensors.iter().find(|t| t.name == "weight").unwrap();
        assert_eq!(weight.byte_size(), 3 * 4 * 4); // 48
        assert_eq!(
            weight.data_offsets.1 - weight.data_offsets.0,
            weight.byte_size()
        );

        // Find the bias tensor (4 F16 = 8 bytes)
        let bias = header.tensors.iter().find(|t| t.name == "bias").unwrap();
        assert_eq!(bias.byte_size(), 4 * 2); // 8
        assert_eq!(
            bias.data_offsets.1 - bias.data_offsets.0,
            bias.byte_size()
        );
    }

    #[test]
    fn tensors_sorted_by_offset() {
        let buf = make_test_buffer();
        let header = parse_header(&buf).unwrap();

        for window in header.tensors.windows(2) {
            assert!(window[0].data_offsets.0 <= window[1].data_offsets.0);
        }
    }

    #[test]
    fn metadata_extracted_separately() {
        let buf = make_test_buffer();
        let header = parse_header(&buf).unwrap();

        assert_eq!(header.metadata.get("format").map(String::as_str), Some("pt"));
        assert_eq!(header.metadata.get("source").map(String::as_str), Some("test"));
        // metadata keys should not appear as tensors
        assert!(header.tensors.iter().all(|t| t.name != "__metadata__"));
    }

    #[test]
    fn error_invalid_json() {
        let bad_json = b"not valid json at all";
        let header_size = bad_json.len() as u64;
        let mut buf = Vec::new();
        buf.extend_from_slice(&header_size.to_le_bytes());
        buf.extend_from_slice(bad_json);

        let result = parse_header(&buf);
        assert!(matches!(result, Err(CoreError::InvalidJson(_))));
    }

    #[test]
    fn error_header_too_short() {
        let buf = vec![0u8; 4]; // only 4 bytes, need at least 8
        let result = parse_header(&buf);
        assert!(matches!(
            result,
            Err(CoreError::HeaderTooShort {
                got: 4,
                expected: 8
            })
        ));
    }

    #[test]
    fn absolute_offsets_correct() {
        let buf = make_test_buffer();
        let header = parse_header(&buf).unwrap();

        let weight = header.tensors.iter().find(|t| t.name == "weight").unwrap();
        let (abs_start, abs_end) = weight.absolute_offsets(header.data_start);
        assert_eq!(abs_start, header.data_start + weight.data_offsets.0);
        assert_eq!(abs_end, header.data_start + weight.data_offsets.1);
    }

    #[test]
    fn parse_header_size_standalone() {
        let val: u64 = 12345;
        let bytes = val.to_le_bytes();
        assert_eq!(parse_header_size(&bytes).unwrap(), 12345);
    }
}
