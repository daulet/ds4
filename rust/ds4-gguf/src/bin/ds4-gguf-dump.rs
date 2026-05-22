use ds4_gguf::{
    parse_gguf, tensor_type_name, value_type_name, Gguf, MetadataValue, MAX_REPORTED_TENSOR_TYPE_ID,
};
use std::collections::BTreeMap;
use std::env;
use std::ffi::CStr;
use std::fs;
use std::io::{self, Write};
use std::os::raw::{c_char, c_double, c_int};
use std::path::PathBuf;

unsafe extern "C" {
    fn snprintf(s: *mut c_char, n: usize, format: *const c_char, ...) -> c_int;
}

const SELECTED_METADATA_KEYS: &[&str] = &[
    "general.name",
    "general.architecture",
    "deepseek4.context_length",
    "deepseek4.block_count",
    "deepseek4.embedding_length",
    "deepseek4.vocab_size",
    "deepseek4.attention.head_count",
    "deepseek4.attention.head_count_kv",
    "deepseek4.attention.key_length",
    "deepseek4.attention.value_length",
    "deepseek4.attention.sliding_window",
    "deepseek4.attention.q_lora_rank",
    "deepseek4.attention.output_lora_rank",
    "deepseek4.attention.output_group_count",
    "deepseek4.attention.indexer.head_count",
    "deepseek4.attention.indexer.key_length",
    "deepseek4.attention.indexer.top_k",
    "deepseek4.attention.compress_rope_freq_base",
    "deepseek4.attention.compress_ratios",
    "deepseek4.rope.dimension_count",
    "deepseek4.rope.freq_base",
    "deepseek4.rope.scaling.factor",
    "deepseek4.rope.scaling.original_context_length",
    "deepseek4.rope.scaling.yarn_beta_fast",
    "deepseek4.rope.scaling.yarn_beta_slow",
    "deepseek4.expert_count",
    "deepseek4.expert_used_count",
    "deepseek4.expert_feed_forward_length",
    "deepseek4.expert_shared_count",
    "deepseek4.expert_group_count",
    "deepseek4.expert_group_used_count",
    "deepseek4.hash_layer_count",
    "deepseek4.expert_weights_scale",
    "deepseek4.expert_weights_norm",
    "deepseek4.hyper_connection.count",
    "deepseek4.hyper_connection.sinkhorn_iterations",
    "deepseek4.hyper_connection.epsilon",
    "deepseek4.swiglu_clamp_exp",
    "deepseek4.attention.layer_norm_rms_epsilon",
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = parse_args()?;
    let bytes = fs::read(&path)?;
    let gguf = parse_gguf(&bytes)?;

    let mut out = io::BufWriter::new(io::stdout());
    write_dump(&mut out, &path, &gguf)?;
    Ok(())
}

fn parse_args() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let mut args = env::args_os();
    let program = args.next().unwrap_or_default();
    let Some(path) = args.next() else {
        eprintln!("usage: {} FILE", PathBuf::from(program).display());
        std::process::exit(2);
    };
    if args.next().is_some() {
        eprintln!("usage: {} FILE", PathBuf::from(program).display());
        std::process::exit(2);
    }
    Ok(path.into())
}

fn write_dump<W: Write>(out: &mut W, path: &PathBuf, gguf: &Gguf) -> io::Result<()> {
    writeln!(out, "{{")?;
    writeln!(out, "  \"schema\": \"ds4.metadata.v1\",")?;
    writeln!(out, "  \"source\": \"rust-gguf-parser\",")?;
    writeln!(out, "  \"model\": {{")?;
    write!(out, "    \"path\": ")?;
    write_json_string(out, &path.display().to_string())?;
    writeln!(out, ",")?;
    writeln!(out, "    \"size\": {},", gguf.file_size)?;
    writeln!(out, "    \"gguf_version\": {},", gguf.version)?;
    writeln!(out, "    \"metadata_count\": {},", gguf.metadata.len())?;
    writeln!(out, "    \"tensor_count\": {},", gguf.tensors.len())?;
    writeln!(out, "    \"alignment\": {},", gguf.alignment)?;
    writeln!(
        out,
        "    \"tensor_data_offset\": {}",
        gguf.tensor_data_offset
    )?;
    writeln!(out, "  }},")?;
    writeln!(
        out,
        "  \"validation\": {{\"config\": \"skipped\", \"weights\": \"skipped\", \"mtp_weights\": \"skipped\"}},"
    )?;
    write_selected_metadata(out, gguf)?;
    write_tensor_types(out, gguf)?;
    write_tensors(out, gguf)?;
    writeln!(out, "  \"bound_tensors\": [")?;
    writeln!(out, "  ]")?;
    writeln!(out, "}}")?;
    Ok(())
}

fn write_selected_metadata<W: Write>(out: &mut W, gguf: &Gguf) -> io::Result<()> {
    writeln!(out, "  \"selected_metadata\": [")?;
    let mut first = true;
    for key in SELECTED_METADATA_KEYS {
        let Some(entry) = gguf.metadata.iter().find(|entry| entry.key == *key) else {
            continue;
        };
        write_comma(out, &mut first)?;
        write!(out, "    {{\"key\": ")?;
        write_json_string(out, &entry.key)?;
        write!(out, ", \"type\": ")?;
        write_json_string(out, value_type_name(entry.value.type_id()))?;
        match &entry.value {
            MetadataValue::Array {
                element_type,
                values,
            } => {
                write!(out, ", \"array_type\": ")?;
                write_json_string(out, value_type_name(*element_type))?;
                write!(out, ", \"len\": {}, \"values\": ", values.len())?;
                write_value_array(out, values)?;
            }
            value => {
                write!(out, ", \"value\": ")?;
                write_scalar_value(out, value)?;
            }
        }
        write!(out, "}}")?;
    }
    writeln!(out)?;
    writeln!(out, "  ],")?;
    Ok(())
}

fn write_tensor_types<W: Write>(out: &mut W, gguf: &Gguf) -> io::Result<()> {
    let mut counts: BTreeMap<u32, (u64, u64)> = BTreeMap::new();
    for tensor in &gguf.tensors {
        if tensor.type_id > MAX_REPORTED_TENSOR_TYPE_ID {
            continue;
        }
        let entry = counts.entry(tensor.type_id).or_insert((0, 0));
        entry.0 += 1;
        entry.1 += tensor.bytes;
    }

    writeln!(out, "  \"tensor_types\": [")?;
    let mut first = true;
    for (type_id, (count, bytes)) in counts {
        write_comma(out, &mut first)?;
        write!(out, "    {{\"type\": {type_id}, \"type_name\": ")?;
        write_json_string(out, tensor_type_name(type_id))?;
        write!(out, ", \"count\": {count}, \"bytes\": {bytes}}}")?;
    }
    writeln!(out)?;
    writeln!(out, "  ],")?;
    Ok(())
}

fn write_tensors<W: Write>(out: &mut W, gguf: &Gguf) -> io::Result<()> {
    writeln!(out, "  \"tensors\": [")?;
    for (idx, tensor) in gguf.tensors.iter().enumerate() {
        if idx != 0 {
            writeln!(out, ",")?;
        }
        write!(out, "    {{\"index\": {idx}, \"name\": ")?;
        write_json_string(out, &tensor.name)?;
        write!(out, ", \"type\": {}, \"type_name\": ", tensor.type_id)?;
        write_json_string(out, tensor_type_name(tensor.type_id))?;
        write!(out, ", \"ndim\": {}, \"dims\": [", tensor.dims.len())?;
        for (dim_idx, dim) in tensor.dims.iter().enumerate() {
            if dim_idx != 0 {
                write!(out, ", ")?;
            }
            write!(out, "{dim}")?;
        }
        write!(
            out,
            "], \"elements\": {}, \"bytes\": {}, \"rel_offset\": {}, \"abs_offset\": {}}}",
            tensor.elements, tensor.bytes, tensor.rel_offset, tensor.abs_offset
        )?;
    }
    writeln!(out)?;
    writeln!(out, "  ],")?;
    Ok(())
}

fn write_scalar_value<W: Write>(out: &mut W, value: &MetadataValue) -> io::Result<()> {
    match value {
        MetadataValue::String(value) => write_json_string(out, value),
        MetadataValue::UInt32(value) => write!(out, "{value}"),
        MetadataValue::Int32(value) => write!(out, "{value}"),
        MetadataValue::Float32(value) => write_c_float(out, *value as f64, b"%.9g\0"),
        MetadataValue::Bool(value) => write!(out, "{value}"),
        MetadataValue::UInt64(value) => write!(out, "{value}"),
        MetadataValue::Float64(value) => write_c_float(out, *value, b"%.17g\0"),
        _ => write!(out, "null"),
    }
}

fn write_array_value<W: Write>(out: &mut W, value: &MetadataValue) -> io::Result<()> {
    match value {
        MetadataValue::UInt32(value) => write!(out, "{value}"),
        MetadataValue::Int32(value) => write!(out, "{value}"),
        MetadataValue::Float32(value) => write_c_float(out, *value as f64, b"%.9g\0"),
        MetadataValue::Float64(value) => write_c_float(out, *value, b"%.17g\0"),
        _ => write!(out, "null"),
    }
}

fn write_value_array<W: Write>(out: &mut W, values: &[MetadataValue]) -> io::Result<()> {
    write!(out, "[")?;
    for (idx, value) in values.iter().enumerate() {
        if idx != 0 {
            write!(out, ", ")?;
        }
        write_array_value(out, value)?;
    }
    write!(out, "]")
}

fn write_c_float<W: Write>(out: &mut W, value: f64, format: &'static [u8]) -> io::Result<()> {
    let mut buf = [0 as c_char; 64];
    let len = unsafe {
        snprintf(
            buf.as_mut_ptr(),
            buf.len(),
            format.as_ptr().cast::<c_char>(),
            value as c_double,
        )
    };
    if len < 0 || len as usize >= buf.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "failed to format GGUF float metadata",
        ));
    }
    let text = unsafe { CStr::from_ptr(buf.as_ptr()) };
    out.write_all(text.to_bytes())
}

fn write_json_string<W: Write>(out: &mut W, value: &str) -> io::Result<()> {
    write!(out, "\"")?;
    for ch in value.chars() {
        match ch {
            '"' => write!(out, "\\\"")?,
            '\\' => write!(out, "\\\\")?,
            '\u{08}' => write!(out, "\\b")?,
            '\u{0c}' => write!(out, "\\f")?,
            '\n' => write!(out, "\\n")?,
            '\r' => write!(out, "\\r")?,
            '\t' => write!(out, "\\t")?,
            ch if ch < ' ' => write!(out, "\\u{:04x}", ch as u32)?,
            ch => write!(out, "{ch}")?,
        }
    }
    write!(out, "\"")
}

fn write_comma<W: Write>(out: &mut W, first: &mut bool) -> io::Result<()> {
    if *first {
        *first = false;
    } else {
        writeln!(out, ",")?;
    }
    Ok(())
}
