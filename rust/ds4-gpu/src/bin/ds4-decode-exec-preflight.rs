use ds4_gguf::{
    bind_ds4_weights, parse_gguf_allowing_missing_tensor_data, tensor_type_name, Ds4LayerWeights,
    Ds4Weights, Gguf, TensorInfo,
};
#[cfg(all(target_os = "linux", feature = "cuda-backend"))]
use ds4_gpu::decode_backend::{cache_model_range, cache_q8_f16_range};
use ds4_gpu::decode_backend::{set_model_fd, set_model_map_range, DecodeBackend, ModelMap};
use ds4_gpu::decode_execution::{
    preflight_layer_coverage, DECODE_EXECUTION_PREFLIGHT_CASE, DECODE_EXECUTION_PREFLIGHT_LAYERS,
    DECODE_EXECUTION_PREFLIGHT_SCHEMA, DECODE_EXECUTION_PREFLIGHT_SCOPE,
    PREFLIGHT_CHECKPOINT_TARGETS, REPRESENTATIVE_TENSORS,
};
use ds4_gpu::graph_plan::{GraphPlan, ModelGraphDims, TensorByteLen};
use ds4_gpu::graph_state::{GraphStateFieldPlan, DECODE_GRAPH_STATE_FIELDS};
use ds4_gpu::{initialize, Tensor};
use std::ffi::c_void;
#[cfg(all(target_os = "linux", feature = "cuda-backend"))]
use std::ffi::CString;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::os::raw::c_int;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};

const PROT_READ: c_int = 0x1;
const MAP_PRIVATE: c_int = 0x02;
const MAP_FAILED: *mut c_void = !0usize as *mut c_void;
const INITIAL_HEADER_READ: u64 = 8 * 1024 * 1024;
const MAX_HEADER_READ: u64 = 512 * 1024 * 1024;
const SMALL_MODEL_CACHE_LIMIT: u64 = 1024 * 1024;
const Q8_CACHE_LIMIT: u64 = 32 * 1024 * 1024;

unsafe extern "C" {
    fn mmap(
        addr: *mut c_void,
        length: usize,
        prot: c_int,
        flags: c_int,
        fd: c_int,
        offset: i64,
    ) -> *mut c_void;
    fn munmap(addr: *mut c_void, length: usize) -> c_int;
}

fn main() {
    if let Err(err) = run() {
        eprintln!("ds4-decode-exec-preflight: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse()?;
    let (gguf, header_bytes_read) = parse_header_prefix(&args.model)?;
    let weights = bind_ds4_weights(&gguf)?;
    let mapped = MappedModel::open(&args.model)?;
    // The B300 comparator pins this reported value to the captured model hash;
    // the preflight itself avoids streaming the full 86 GiB file to rehash it.
    if let Some(expected) = &args.model_sha256 {
        if expected.len() != 64 || !expected.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err("--model-sha256 must be a 64-character hex string".into());
        }
    }

    initialize().map_err(|err| format!("failed to initialize backend: {err}"))?;
    set_model_fd(mapped.file.as_raw_fd())
        .map_err(|err| format!("failed to set model fd: {err}"))?;
    let model = unsafe { ModelMap::from_raw_parts(mapped.ptr.cast_const(), mapped.size) };
    set_model_map_range(
        model,
        gguf.tensor_data_offset,
        mapped.size - gguf.tensor_data_offset,
    )
    .map_err(|err| format!("failed to set model map range: {err}"))?;
    let backend = DecodeBackend::new(model);

    let representative = allocate_representative_tensors()?;
    let selected = selected_weights(&weights);
    let cache = cache_representative_ranges(model, &selected)?;

    write_report(
        &args,
        &gguf,
        &weights,
        header_bytes_read,
        mapped.size,
        backend.model().size(),
        &selected,
        &representative,
        &cache,
    );

    drop(representative);
    unsafe {
        ds4_gpu::cleanup();
    }
    Ok(())
}

struct Args {
    model: PathBuf,
    model_sha256: Option<String>,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn std::error::Error>> {
        let mut model = None;
        let mut model_sha256 = None;
        let mut args = std::env::args_os().skip(1);
        while let Some(arg) = args.next() {
            if arg == "--model" {
                let Some(value) = args.next() else {
                    return Err("--model requires a path".into());
                };
                model = Some(PathBuf::from(value));
            } else if arg == "--model-sha256" {
                let Some(value) = args.next() else {
                    return Err("--model-sha256 requires a value".into());
                };
                model_sha256 = Some(value.into_string().map_err(|_| "model sha must be utf-8")?);
            } else {
                return Err(
                    "usage: ds4-decode-exec-preflight --model FILE [--model-sha256 HEX]".into(),
                );
            }
        }
        let Some(model) = model else {
            return Err(
                "usage: ds4-decode-exec-preflight --model FILE [--model-sha256 HEX]".into(),
            );
        };
        Ok(Self {
            model,
            model_sha256,
        })
    }
}

struct MappedModel {
    file: File,
    ptr: *mut c_void,
    size: u64,
}

impl MappedModel {
    fn open(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let size = file.metadata()?.len();
        if size == 0 || usize::try_from(size).is_err() {
            return Err("model file is empty or too large for mmap length".into());
        }
        let ptr = unsafe {
            mmap(
                std::ptr::null_mut(),
                size as usize,
                PROT_READ,
                MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr == MAP_FAILED {
            return Err(std::io::Error::last_os_error().into());
        }
        Ok(Self { file, ptr, size })
    }
}

impl Drop for MappedModel {
    fn drop(&mut self) {
        unsafe {
            let _ = munmap(self.ptr, self.size as usize);
        }
    }
}

struct SelectedWeight<'a> {
    role: String,
    tensor: &'a TensorInfo,
}

struct AllocatedTensorReport {
    field: &'static str,
    layer: Option<usize>,
    bytes: u64,
}

struct CacheReport {
    model_ranges: Vec<String>,
    q8_f16_ranges: Vec<String>,
}

fn parse_header_prefix(path: &Path) -> Result<(Gguf, u64), Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let file_size = file.metadata()?.len();
    let mut size = INITIAL_HEADER_READ.min(file_size);
    loop {
        file.seek(SeekFrom::Start(0))?;
        let mut bytes = vec![0u8; usize::try_from(size)?];
        file.read_exact(&mut bytes)?;
        match parse_gguf_allowing_missing_tensor_data(&bytes) {
            Ok(gguf) => return Ok((gguf, size)),
            Err(err) if err.message() == "truncated GGUF file" && size < file_size => {
                size = (size.saturating_mul(2)).min(file_size).min(MAX_HEADER_READ);
                if size == bytes.len() as u64 {
                    return Err(err.into());
                }
            }
            Err(err) => return Err(err.into()),
        }
    }
}

fn selected_weights(weights: &Ds4Weights) -> Vec<SelectedWeight<'_>> {
    let mut out = Vec::new();
    push_weight(&mut out, "base.token_embd", &weights.token_embd);
    push_weight(&mut out, "base.output_norm", &weights.output_norm);
    push_weight(&mut out, "base.output_hc_scale", &weights.output_hc_scale);
    push_weight(&mut out, "base.output_hc_base", &weights.output_hc_base);
    push_weight(&mut out, "base.output", &weights.output);
    push_layer_weights(&mut out, 0, &weights.layers[0]);
    push_layer_weights(&mut out, 2, &weights.layers[2]);
    push_layer_weights(&mut out, 3, &weights.layers[3]);
    out
}

fn push_layer_weights<'a>(
    out: &mut Vec<SelectedWeight<'a>>,
    layer: usize,
    weights: &'a Ds4LayerWeights,
) {
    let prefix = format!("base.layer.{layer}");
    push_weight(out, &format!("{prefix}.attn_norm"), &weights.attn_norm);
    push_weight(out, &format!("{prefix}.attn_q_a"), &weights.attn_q_a);
    push_weight(
        out,
        &format!("{prefix}.attn_q_a_norm"),
        &weights.attn_q_a_norm,
    );
    push_weight(out, &format!("{prefix}.attn_q_b"), &weights.attn_q_b);
    push_weight(out, &format!("{prefix}.attn_kv"), &weights.attn_kv);
    push_weight(
        out,
        &format!("{prefix}.attn_kv_a_norm"),
        &weights.attn_kv_a_norm,
    );
    push_weight(out, &format!("{prefix}.attn_sinks"), &weights.attn_sinks);
    push_weight(
        out,
        &format!("{prefix}.attn_output_a"),
        &weights.attn_output_a,
    );
    push_weight(
        out,
        &format!("{prefix}.attn_output_b"),
        &weights.attn_output_b,
    );
    if let Some(tensor) = &weights.attn_compressor_norm {
        push_weight(out, &format!("{prefix}.attn_compressor_norm"), tensor);
    }
    if let Some(tensor) = &weights.indexer_proj {
        push_weight(out, &format!("{prefix}.indexer_proj"), tensor);
    }
    if let Some(tensor) = &weights.indexer_compressor_norm {
        push_weight(out, &format!("{prefix}.indexer_compressor_norm"), tensor);
    }
    push_weight(out, &format!("{prefix}.ffn_norm"), &weights.ffn_norm);
    if let Some(tensor) = &weights.ffn_gate_tid2eid {
        push_weight(out, &format!("{prefix}.ffn_gate_tid2eid"), tensor);
    }
    push_weight(
        out,
        &format!("{prefix}.ffn_gate_inp"),
        &weights.ffn_gate_inp,
    );
    push_weight(
        out,
        &format!("{prefix}.ffn_gate_shexp"),
        &weights.ffn_gate_shexp,
    );
    push_weight(
        out,
        &format!("{prefix}.ffn_up_shexp"),
        &weights.ffn_up_shexp,
    );
    push_weight(
        out,
        &format!("{prefix}.ffn_down_shexp"),
        &weights.ffn_down_shexp,
    );
}

fn push_weight<'a>(out: &mut Vec<SelectedWeight<'a>>, role: &str, tensor: &'a TensorInfo) {
    out.push(SelectedWeight {
        role: role.to_owned(),
        tensor,
    });
}

fn allocate_representative_tensors(
) -> Result<Vec<AllocatedTensorReport>, Box<dyn std::error::Error>> {
    let plan = GraphPlan::for_context(32768, 32768, false);
    let dims = ModelGraphDims::DS4_FLASH;
    let mut tensors = Vec::new();
    let mut report = Vec::new();
    for spec in REPRESENTATIVE_TENSORS {
        let field = state_field(spec.field).ok_or("representative tensor field missing")?;
        let allocation = field
            .initial_allocation(plan, dims, spec.layer)
            .map_err(|err| format!("representative tensor plan failed: {err:?}"))?;
        let TensorByteLen::Known(bytes) = allocation.byte_len else {
            return Err("representative tensor must have known byte length".into());
        };
        let mut tensor = Tensor::allocate(usize::try_from(bytes)?)
            .map_err(|err| format!("failed to allocate representative tensor: {err}"))?;
        if bytes % 4 == 0 {
            tensor
                .fill_f32(0.0, usize::try_from(bytes / 4)?)
                .map_err(|err| format!("failed to clear representative tensor: {err}"))?;
        }
        tensors.push(tensor);
        report.push(AllocatedTensorReport {
            field: spec.field,
            layer: spec.layer,
            bytes,
        });
    }
    drop(tensors);
    Ok(report)
}

fn state_field(name: &str) -> Option<GraphStateFieldPlan> {
    DECODE_GRAPH_STATE_FIELDS
        .iter()
        .copied()
        .find(|field| field.name == name)
}

fn cache_representative_ranges(
    model: ModelMap<'_>,
    selected: &[SelectedWeight<'_>],
) -> Result<CacheReport, Box<dyn std::error::Error>> {
    let mut report = CacheReport {
        model_ranges: Vec::new(),
        q8_f16_ranges: Vec::new(),
    };
    for weight in selected {
        if weight.tensor.bytes <= SMALL_MODEL_CACHE_LIMIT {
            cache_model_range_for_report(model, &mut report, weight)?;
        }
    }
    for weight in selected {
        if tensor_type_name(weight.tensor.type_id) == "q8_0"
            && weight.tensor.dims.len() == 2
            && weight.tensor.bytes <= Q8_CACHE_LIMIT
        {
            cache_q8_f16_range_for_report(model, &mut report, weight)?;
            break;
        }
    }
    Ok(report)
}

#[cfg(all(target_os = "linux", feature = "cuda-backend"))]
fn cache_model_range_for_report(
    model: ModelMap<'_>,
    report: &mut CacheReport,
    weight: &SelectedWeight<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    let label = CString::new(weight.role.as_str())?;
    // DS4 tensor offsets are file-absolute GGUF offsets. The backend was
    // registered with the tensor-data subrange and still accepts file-absolute
    // tensor offsets, matching the C startup/decode path.
    cache_model_range(
        model,
        weight.tensor.abs_offset,
        weight.tensor.bytes,
        Some(label.as_c_str()),
    )
    .map_err(|err| format!("failed to cache model range {}: {err}", weight.role))?;
    report.model_ranges.push(weight.role.clone());
    Ok(())
}

#[cfg(not(all(target_os = "linux", feature = "cuda-backend")))]
fn cache_model_range_for_report(
    _model: ModelMap<'_>,
    report: &mut CacheReport,
    weight: &SelectedWeight<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    report
        .model_ranges
        .push(format!("{}:compile-only", weight.role));
    Ok(())
}

#[cfg(all(target_os = "linux", feature = "cuda-backend"))]
fn cache_q8_f16_range_for_report(
    model: ModelMap<'_>,
    report: &mut CacheReport,
    weight: &SelectedWeight<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    let label = CString::new(weight.role.as_str())?;
    // See cache_model_range_for_report: GGUF tensor offsets stay file-absolute.
    cache_q8_f16_range(
        model,
        weight.tensor.abs_offset,
        weight.tensor.bytes,
        weight.tensor.dims[0],
        weight.tensor.dims[1],
        Some(label.as_c_str()),
    )
    .map_err(|err| format!("failed to cache q8/f16 range {}: {err}", weight.role))?;
    report.q8_f16_ranges.push(weight.role.clone());
    Ok(())
}

#[cfg(not(all(target_os = "linux", feature = "cuda-backend")))]
fn cache_q8_f16_range_for_report(
    _model: ModelMap<'_>,
    report: &mut CacheReport,
    weight: &SelectedWeight<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    report
        .q8_f16_ranges
        .push(format!("{}:compile-only", weight.role));
    Ok(())
}

fn write_report(
    args: &Args,
    gguf: &Gguf,
    weights: &Ds4Weights,
    header_bytes_read: u64,
    mapped_model_size: u64,
    backend_model_size: u64,
    selected: &[SelectedWeight<'_>],
    representative: &[AllocatedTensorReport],
    cache: &CacheReport,
) {
    let coverage = preflight_layer_coverage();
    println!("{{");
    println!("  \"schema\": \"{}\",", DECODE_EXECUTION_PREFLIGHT_SCHEMA);
    println!("  \"scope\": \"{}\",", DECODE_EXECUTION_PREFLIGHT_SCOPE);
    println!("  \"case\": \"{}\",", DECODE_EXECUTION_PREFLIGHT_CASE);
    println!("  \"model\": {{");
    print!("    \"path\": ");
    write_json_str(&args.model.display().to_string());
    println!(",");
    print!("    \"sha256\": ");
    write_json_str(args.model_sha256.as_deref().unwrap_or(""));
    println!(",");
    println!("    \"mapped_size\": {},", mapped_model_size);
    println!("    \"backend_model_size\": {},", backend_model_size);
    println!("    \"header_bytes_read\": {},", header_bytes_read);
    println!("    \"tensor_count\": {},", gguf.tensors.len());
    println!("    \"tensor_data_offset\": {},", gguf.tensor_data_offset);
    println!("    \"bound_layers\": {}", weights.layers.len());
    println!("  }},");
    println!("  \"backend\": {{");
    println!("    \"initialized\": true,");
    println!("    \"set_model_fd\": true,");
    println!("    \"set_model_map_range\": {{");
    println!("      \"offset\": {},", gguf.tensor_data_offset);
    println!(
        "      \"bytes\": {}",
        mapped_model_size - gguf.tensor_data_offset
    );
    println!("    }}");
    println!("  }},");
    println!("  \"layer_coverage\": {{");
    println!("    \"dense\": {},", bool_json(coverage.dense));
    println!("    \"ratio4\": {},", bool_json(coverage.ratio4));
    println!("    \"ratio128\": {},", bool_json(coverage.ratio128));
    println!(
        "    \"covers_default_decode\": {}",
        bool_json(coverage.covers_default_decode())
    );
    println!("  }},");
    write_preflight_layers();
    write_checkpoints();
    write_representative_tensors(representative);
    write_selected_weights(selected);
    write_cache_report(cache);
    println!("}}");
}

fn write_preflight_layers() {
    println!("  \"preflight_layers\": [");
    for (idx, layer) in DECODE_EXECUTION_PREFLIGHT_LAYERS.iter().enumerate() {
        if idx != 0 {
            println!(",");
        }
        let compression = ds4_gpu::graph_plan::layer_compression(*layer).expect("layer");
        print!("    {{\"layer\": {layer}, \"compression\": ");
        write_json_str(match compression {
            ds4_gpu::graph_plan::LayerCompression::Dense => "dense",
            ds4_gpu::graph_plan::LayerCompression::Ratio4 => "ratio4",
            ds4_gpu::graph_plan::LayerCompression::Ratio128 => "ratio128",
        });
        print!("}}");
    }
    println!();
    println!("  ],");
}

fn write_checkpoints() {
    println!("  \"checkpoint_targets\": [");
    for (idx, target) in PREFLIGHT_CHECKPOINT_TARGETS.iter().enumerate() {
        if idx != 0 {
            println!(",");
        }
        print!("    {{\"name\": ");
        write_json_str(target.name);
        print!(", \"stage\": ");
        write_json_str(target.stage);
        print!(", \"boundary\": ");
        write_json_str(target.boundary);
        print!(", \"tensor\": ");
        write_json_str(target.tensor);
        print!(", \"layer\": ");
        write_option_usize(target.layer);
        print!(", \"hash_policy\": ");
        write_json_str(target.hash_policy);
        print!("}}");
    }
    println!();
    println!("  ],");
}

fn write_representative_tensors(tensors: &[AllocatedTensorReport]) {
    println!("  \"representative_tensors\": [");
    for (idx, tensor) in tensors.iter().enumerate() {
        if idx != 0 {
            println!(",");
        }
        print!("    {{\"field\": ");
        write_json_str(tensor.field);
        print!(", \"layer\": ");
        write_option_usize(tensor.layer);
        print!(", \"bytes\": {}, \"allocated\": true}}", tensor.bytes);
    }
    println!();
    println!("  ],");
}

fn write_selected_weights(selected: &[SelectedWeight<'_>]) {
    println!("  \"selected_weights\": [");
    for (idx, weight) in selected.iter().enumerate() {
        if idx != 0 {
            println!(",");
        }
        print!("    {{\"role\": ");
        write_json_str(&weight.role);
        print!(", \"tensor\": ");
        write_json_str(&weight.tensor.name);
        print!(", \"type\": ");
        write_json_str(tensor_type_name(weight.tensor.type_id));
        print!(", \"bytes\": {}", weight.tensor.bytes);
        print!(", \"abs_offset\": {}", weight.tensor.abs_offset);
        print!(", \"dims\": [");
        for (dim_idx, dim) in weight.tensor.dims.iter().enumerate() {
            if dim_idx != 0 {
                print!(", ");
            }
            print!("{dim}");
        }
        print!("]}}");
    }
    println!();
    println!("  ],");
}

fn write_cache_report(cache: &CacheReport) {
    println!("  \"cache\": {{");
    print!("    \"model_ranges\": [");
    write_string_array(&cache.model_ranges);
    println!("],");
    print!("    \"q8_f16_ranges\": [");
    write_string_array(&cache.q8_f16_ranges);
    println!("]");
    println!("  }}");
}

fn write_string_array(values: &[String]) {
    for (idx, value) in values.iter().enumerate() {
        if idx != 0 {
            print!(", ");
        }
        write_json_str(value);
    }
}

fn write_json_str(value: &str) {
    print!("\"");
    for ch in value.chars() {
        match ch {
            '"' => print!("\\\""),
            '\\' => print!("\\\\"),
            '\n' => print!("\\n"),
            '\r' => print!("\\r"),
            '\t' => print!("\\t"),
            ch if ch.is_control() => print!("\\u{:04x}", ch as u32),
            ch => print!("{ch}"),
        }
    }
    print!("\"");
}

fn write_option_usize(value: Option<usize>) {
    match value {
        Some(value) => print!("{value}"),
        None => print!("null"),
    }
}

fn bool_json(value: bool) -> &'static str {
    if value {
        "true"
    } else {
        "false"
    }
}
