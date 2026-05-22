use std::fmt;
use std::str;

const GGUF_MAGIC: u32 = 0x4655_4747;
const MAX_DIMS: usize = 8;
const DEFAULT_ALIGNMENT: u64 = 32;
pub const MAX_REPORTED_TENSOR_TYPE_ID: u32 = 30;

#[derive(Debug, Clone, PartialEq)]
pub struct Gguf {
    pub version: u32,
    pub metadata: Vec<MetadataEntry>,
    pub tensors: Vec<TensorInfo>,
    pub alignment: u64,
    pub tensor_data_offset: u64,
    pub file_size: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetadataEntry {
    pub key: String,
    pub value: MetadataValue,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    UInt8(u8),
    Int8(i8),
    UInt16(u16),
    Int16(i16),
    UInt32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array {
        element_type: u32,
        values: Vec<MetadataValue>,
    },
    UInt64(u64),
    Int64(i64),
    Float64(f64),
}

impl MetadataValue {
    pub fn type_id(&self) -> u32 {
        match self {
            Self::UInt8(_) => 0,
            Self::Int8(_) => 1,
            Self::UInt16(_) => 2,
            Self::Int16(_) => 3,
            Self::UInt32(_) => 4,
            Self::Int32(_) => 5,
            Self::Float32(_) => 6,
            Self::Bool(_) => 7,
            Self::String(_) => 8,
            Self::Array { .. } => 9,
            Self::UInt64(_) => 10,
            Self::Int64(_) => 11,
            Self::Float64(_) => 12,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub type_id: u32,
    pub rel_offset: u64,
    pub abs_offset: u64,
    pub elements: u64,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufError {
    message: String,
}

impl GgufError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for GgufError {}

pub fn parse_gguf(bytes: &[u8]) -> Result<Gguf, GgufError> {
    let mut cursor = Cursor::new(bytes);
    let magic = cursor.u32()?;
    if magic != GGUF_MAGIC {
        return Err(GgufError::new("model is not a GGUF file"));
    }

    let version = cursor.u32()?;
    if version != 3 {
        return Err(GgufError::new("only GGUF v3 is supported"));
    }

    let n_tensors = cursor.u64()?;
    let n_metadata = cursor.u64()?;

    let mut metadata = Vec::new();
    reserve_vec(
        &mut metadata,
        usize_len(n_metadata, "metadata count")?,
        "metadata table",
    )?;
    let mut alignment = DEFAULT_ALIGNMENT;
    for _ in 0..n_metadata {
        let key = cursor.string()?;
        let value_type = cursor.u32()?;
        let value = cursor.value(value_type, 0)?;
        if key == "general.alignment" {
            if let MetadataValue::UInt32(v) = value {
                if v != 0 {
                    alignment = u64::from(v);
                }
            }
        }
        metadata.push(MetadataEntry { key, value });
    }

    let mut tensors = Vec::new();
    reserve_vec(
        &mut tensors,
        usize_len(n_tensors, "tensor count")?,
        "tensor table",
    )?;
    for _ in 0..n_tensors {
        let name = cursor.string()?;
        let ndim = cursor.u32()?;
        if ndim == 0 || ndim as usize > MAX_DIMS {
            return Err(GgufError::new(
                "tensor has an unsupported number of dimensions",
            ));
        }

        let mut dims = Vec::with_capacity(ndim as usize);
        let mut elements = 1u64;
        for _ in 0..ndim {
            let dim = cursor.u64()?;
            if dim != 0 {
                elements = elements
                    .checked_mul(dim)
                    .ok_or_else(|| GgufError::new("tensor element count overflow"))?;
            } else {
                elements = 0;
            }
            dims.push(dim);
        }

        let type_id = cursor.u32()?;
        let rel_offset = cursor.u64()?;
        let bytes_len = tensor_nbytes(type_id, elements).unwrap_or(0);
        tensors.push(TensorInfo {
            name,
            dims,
            type_id,
            rel_offset,
            abs_offset: 0,
            elements,
            bytes: bytes_len,
        });
    }

    let tensor_data_offset = align_up(cursor.position() as u64, alignment)?;
    let file_size = bytes.len() as u64;
    for tensor in &mut tensors {
        tensor.abs_offset = tensor_data_offset
            .checked_add(tensor.rel_offset)
            .ok_or_else(|| GgufError::new("tensor offset overflow"))?;
        if tensor.bytes != 0
            && (tensor.abs_offset > file_size || tensor.bytes > file_size - tensor.abs_offset)
        {
            return Err(GgufError::new("tensor points outside GGUF file"));
        }
    }

    Ok(Gguf {
        version,
        metadata,
        tensors,
        alignment,
        tensor_data_offset,
        file_size,
    })
}

pub fn value_type_name(type_id: u32) -> &'static str {
    match type_id {
        0 => "uint8",
        1 => "int8",
        2 => "uint16",
        3 => "int16",
        4 => "uint32",
        5 => "int32",
        6 => "float32",
        7 => "bool",
        8 => "string",
        9 => "array",
        10 => "uint64",
        11 => "int64",
        12 => "float64",
        _ => "unknown",
    }
}

pub fn tensor_type_name(type_id: u32) -> &'static str {
    tensor_type(type_id)
        .map(|info| info.name)
        .unwrap_or("unknown")
}

pub fn tensor_nbytes(type_id: u32, elements: u64) -> Option<u64> {
    let info = tensor_type(type_id)?;
    let blocks = elements.checked_add(info.block_elems - 1)? / info.block_elems;
    blocks.checked_mul(info.block_bytes)
}

struct TensorType {
    name: &'static str,
    block_elems: u64,
    block_bytes: u64,
}

fn tensor_type(type_id: u32) -> Option<TensorType> {
    let info = match type_id {
        0 => TensorType {
            name: "f32",
            block_elems: 1,
            block_bytes: 4,
        },
        1 => TensorType {
            name: "f16",
            block_elems: 1,
            block_bytes: 2,
        },
        2 => TensorType {
            name: "q4_0",
            block_elems: 32,
            block_bytes: 18,
        },
        3 => TensorType {
            name: "q4_1",
            block_elems: 32,
            block_bytes: 20,
        },
        6 => TensorType {
            name: "q5_0",
            block_elems: 32,
            block_bytes: 22,
        },
        7 => TensorType {
            name: "q5_1",
            block_elems: 32,
            block_bytes: 24,
        },
        8 => TensorType {
            name: "q8_0",
            block_elems: 32,
            block_bytes: 34,
        },
        9 => TensorType {
            name: "q8_1",
            block_elems: 32,
            block_bytes: 40,
        },
        10 => TensorType {
            name: "q2_k",
            block_elems: 256,
            block_bytes: 84,
        },
        11 => TensorType {
            name: "q3_k",
            block_elems: 256,
            block_bytes: 110,
        },
        12 => TensorType {
            name: "q4_k",
            block_elems: 256,
            block_bytes: 144,
        },
        13 => TensorType {
            name: "q5_k",
            block_elems: 256,
            block_bytes: 176,
        },
        14 => TensorType {
            name: "q6_k",
            block_elems: 256,
            block_bytes: 210,
        },
        15 => TensorType {
            name: "q8_k",
            block_elems: 256,
            block_bytes: 292,
        },
        16 => TensorType {
            name: "iq2_xxs",
            block_elems: 256,
            block_bytes: 66,
        },
        17 => TensorType {
            name: "iq2_xs",
            block_elems: 256,
            block_bytes: 74,
        },
        18 => TensorType {
            name: "iq3_xxs",
            block_elems: 256,
            block_bytes: 98,
        },
        19 => TensorType {
            name: "iq1_s",
            block_elems: 256,
            block_bytes: 110,
        },
        20 => TensorType {
            name: "iq4_nl",
            block_elems: 256,
            block_bytes: 50,
        },
        21 => TensorType {
            name: "iq3_s",
            block_elems: 256,
            block_bytes: 110,
        },
        22 => TensorType {
            name: "iq2_s",
            block_elems: 256,
            block_bytes: 82,
        },
        23 => TensorType {
            name: "iq4_xs",
            block_elems: 256,
            block_bytes: 136,
        },
        24 => TensorType {
            name: "i8",
            block_elems: 1,
            block_bytes: 1,
        },
        25 => TensorType {
            name: "i16",
            block_elems: 1,
            block_bytes: 2,
        },
        26 => TensorType {
            name: "i32",
            block_elems: 1,
            block_bytes: 4,
        },
        27 => TensorType {
            name: "i64",
            block_elems: 1,
            block_bytes: 8,
        },
        28 => TensorType {
            name: "f64",
            block_elems: 1,
            block_bytes: 8,
        },
        29 => TensorType {
            name: "iq1_m",
            block_elems: 256,
            block_bytes: 56,
        },
        30 => TensorType {
            name: "bf16",
            block_elems: 1,
            block_bytes: 2,
        },
        _ => return None,
    };
    Some(info)
}

fn align_up(value: u64, alignment: u64) -> Result<u64, GgufError> {
    if alignment == 0 {
        return Err(GgufError::new("alignment is zero"));
    }
    let rem = value % alignment;
    if rem == 0 {
        Ok(value)
    } else {
        value
            .checked_add(alignment - rem)
            .ok_or_else(|| GgufError::new("alignment overflow"))
    }
}

fn usize_len(value: u64, label: &str) -> Result<usize, GgufError> {
    usize::try_from(value).map_err(|_| GgufError::new(format!("{label} is too large")))
}

fn reserve_vec<T>(vec: &mut Vec<T>, additional: usize, label: &str) -> Result<(), GgufError> {
    vec.try_reserve(additional)
        .map_err(|_| GgufError::new(format!("{label} is too large")))
}

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn position(&self) -> usize {
        self.pos
    }

    fn read<const N: usize>(&mut self) -> Result<[u8; N], GgufError> {
        let end = self
            .pos
            .checked_add(N)
            .ok_or_else(|| GgufError::new("cursor overflow"))?;
        if end > self.bytes.len() {
            return Err(GgufError::new("truncated GGUF file"));
        }
        let mut out = [0u8; N];
        out.copy_from_slice(&self.bytes[self.pos..end]);
        self.pos = end;
        Ok(out)
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], GgufError> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| GgufError::new("cursor overflow"))?;
        if end > self.bytes.len() {
            return Err(GgufError::new("truncated GGUF file"));
        }
        let out = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn u8(&mut self) -> Result<u8, GgufError> {
        Ok(self.read::<1>()?[0])
    }

    fn i8(&mut self) -> Result<i8, GgufError> {
        Ok(self.u8()? as i8)
    }

    fn u16(&mut self) -> Result<u16, GgufError> {
        Ok(u16::from_le_bytes(self.read()?))
    }

    fn i16(&mut self) -> Result<i16, GgufError> {
        Ok(i16::from_le_bytes(self.read()?))
    }

    fn u32(&mut self) -> Result<u32, GgufError> {
        Ok(u32::from_le_bytes(self.read()?))
    }

    fn i32(&mut self) -> Result<i32, GgufError> {
        Ok(i32::from_le_bytes(self.read()?))
    }

    fn u64(&mut self) -> Result<u64, GgufError> {
        Ok(u64::from_le_bytes(self.read()?))
    }

    fn i64(&mut self) -> Result<i64, GgufError> {
        Ok(i64::from_le_bytes(self.read()?))
    }

    fn f32(&mut self) -> Result<f32, GgufError> {
        Ok(f32::from_le_bytes(self.read()?))
    }

    fn f64(&mut self) -> Result<f64, GgufError> {
        Ok(f64::from_le_bytes(self.read()?))
    }

    fn string(&mut self) -> Result<String, GgufError> {
        let len = usize_len(self.u64()?, "string length")?;
        let bytes = self.take(len)?;
        let text = str::from_utf8(bytes).map_err(|_| GgufError::new("invalid utf-8 string"))?;
        Ok(text.to_owned())
    }

    fn value(&mut self, type_id: u32, depth: usize) -> Result<MetadataValue, GgufError> {
        if depth > 16 {
            return Err(GgufError::new("metadata array nesting is too deep"));
        }
        let value = match type_id {
            0 => MetadataValue::UInt8(self.u8()?),
            1 => MetadataValue::Int8(self.i8()?),
            2 => MetadataValue::UInt16(self.u16()?),
            3 => MetadataValue::Int16(self.i16()?),
            4 => MetadataValue::UInt32(self.u32()?),
            5 => MetadataValue::Int32(self.i32()?),
            6 => MetadataValue::Float32(self.f32()?),
            7 => MetadataValue::Bool(self.u8()? != 0),
            8 => MetadataValue::String(self.string()?),
            9 => {
                let element_type = self.u32()?;
                let len = usize_len(self.u64()?, "metadata array length")?;
                let mut values = Vec::new();
                reserve_vec(&mut values, len, "metadata array")?;
                for _ in 0..len {
                    values.push(self.value(element_type, depth + 1)?);
                }
                MetadataValue::Array {
                    element_type,
                    values,
                }
            }
            10 => MetadataValue::UInt64(self.u64()?),
            11 => MetadataValue::Int64(self.i64()?),
            12 => MetadataValue::Float64(self.f64()?),
            _ => return Err(GgufError::new("unknown GGUF metadata type")),
        };
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_gguf, MetadataValue};

    #[test]
    fn parses_header_metadata_and_tensor_directory() {
        let fixture = fixture_bytes();
        let gguf = parse_gguf(&fixture).expect("parse fixture");

        assert_eq!(gguf.version, 3);
        assert_eq!(gguf.alignment, 64);
        assert_eq!(gguf.metadata.len(), 4);
        assert_eq!(gguf.tensors.len(), 1);
        assert_eq!(gguf.tensor_data_offset % 64, 0);

        let tensor = &gguf.tensors[0];
        assert_eq!(tensor.name, "tok.weight");
        assert_eq!(tensor.dims, vec![4]);
        assert_eq!(tensor.type_id, 0);
        assert_eq!(tensor.elements, 4);
        assert_eq!(tensor.bytes, 16);
        assert_eq!(tensor.abs_offset, gguf.tensor_data_offset);

        let ratios = gguf
            .metadata
            .iter()
            .find(|entry| entry.key == "deepseek4.attention.compress_ratios")
            .expect("compress ratios");
        assert_eq!(
            ratios.value,
            MetadataValue::Array {
                element_type: 4,
                values: vec![MetadataValue::UInt32(0), MetadataValue::UInt32(4)]
            }
        );
    }

    #[test]
    fn rejects_bad_magic() {
        let mut fixture = fixture_bytes();
        fixture[0] = 0;
        let err = parse_gguf(&fixture).unwrap_err();
        assert_eq!(err.message(), "model is not a GGUF file");
    }

    #[test]
    fn rejects_out_of_file_tensor_data() {
        let mut fixture = fixture_bytes();
        fixture.truncate(fixture.len() - 8);
        let err = parse_gguf(&fixture).unwrap_err();
        assert_eq!(err.message(), "tensor points outside GGUF file");
    }

    #[test]
    fn rejects_huge_metadata_array_before_allocation() {
        let mut fixture = Vec::new();
        fixture.extend_from_slice(&0x4655_4747u32.to_le_bytes());
        fixture.extend_from_slice(&3u32.to_le_bytes());
        fixture.extend_from_slice(&0u64.to_le_bytes());
        fixture.extend_from_slice(&1u64.to_le_bytes());
        push_string(&mut fixture, "huge.array");
        fixture.extend_from_slice(&9u32.to_le_bytes());
        fixture.extend_from_slice(&4u32.to_le_bytes());
        fixture.extend_from_slice(&u64::MAX.to_le_bytes());

        let err = parse_gguf(&fixture).unwrap_err();
        assert!(err.message().contains("metadata array"));
    }

    fn fixture_bytes() -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&0x4655_4747u32.to_le_bytes());
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&1u64.to_le_bytes());
        out.extend_from_slice(&4u64.to_le_bytes());

        push_string_entry(&mut out, "general.name", "fixture");
        push_string_entry(&mut out, "general.architecture", "deepseek4");
        push_u32_entry(&mut out, "general.alignment", 64);
        push_u32_array_entry(&mut out, "deepseek4.attention.compress_ratios", &[0, 4]);

        push_string(&mut out, "tok.weight");
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&4u64.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());

        while out.len() % 64 != 0 {
            out.push(0);
        }
        out.extend_from_slice(&[0u8; 16]);
        out
    }

    fn push_string_entry(out: &mut Vec<u8>, key: &str, value: &str) {
        push_string(out, key);
        out.extend_from_slice(&8u32.to_le_bytes());
        push_string(out, value);
    }

    fn push_u32_entry(out: &mut Vec<u8>, key: &str, value: u32) {
        push_string(out, key);
        out.extend_from_slice(&4u32.to_le_bytes());
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u32_array_entry(out: &mut Vec<u8>, key: &str, values: &[u32]) {
        push_string(out, key);
        out.extend_from_slice(&9u32.to_le_bytes());
        out.extend_from_slice(&4u32.to_le_bytes());
        out.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }

    fn push_string(out: &mut Vec<u8>, value: &str) {
        out.extend_from_slice(&(value.len() as u64).to_le_bytes());
        out.extend_from_slice(value.as_bytes());
    }
}
