use ds4_gguf::{
    bind_ds4_weights, parse_gguf_allowing_missing_tensor_data, tensor_type_name, Gguf, TensorInfo,
};
use ds4_gpu::decode_backend::{set_model_fd, set_model_map_range, DecodeBackend, ModelMap};
use ds4_gpu::graph_plan::{HC_EPS, N_EMBD, N_HC, N_HC_SINKHORN_ITER, N_VOCAB, RMS_EPS};
use ds4_gpu::{initialize, synchronize, CommandBatch, Tensor};
use std::ffi::c_void;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::os::raw::c_int;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};

const SCHEMA: &str = "ds4.decode_layer0_attn_hc_pre.v1";
const CASE: &str = "token0_layer0_attn_hc_pre";
const TOKEN: u32 = 0;
const LAYER: usize = 0;
const PROT_READ: c_int = 0x1;
const MAP_PRIVATE: c_int = 0x02;
const MAP_FAILED: *mut c_void = !0usize as *mut c_void;
const INITIAL_HEADER_READ: u64 = 8 * 1024 * 1024;
const MAX_HEADER_READ: u64 = 512 * 1024 * 1024;

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
        eprintln!("ds4-decode-layer0-attn-hc-pre: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse()?;
    let (gguf, header_bytes_read) = parse_header_prefix(&args.model)?;
    let weights = bind_ds4_weights(&gguf)?;
    let layer0 = weights
        .layers
        .get(LAYER)
        .ok_or("DS4 weight binding did not include layer 0")?;
    let mapped = MappedModel::open(&args.model)?;

    initialize().map_err(|err| format!("failed to initialize backend: {err}"))?;
    let _backend = BackendGuard;
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

    let hc_dim = u64::from(N_HC) * u64::from(N_EMBD);
    let hc_mix_dim = 2 * u64::from(N_HC) + u64::from(N_HC) * u64::from(N_HC);
    let mut cur_hc = Tensor::allocate(byte_len(hc_dim)?)
        .map_err(|err| format!("failed to allocate cur_hc: {err}"))?;
    let mut flat_hc = Tensor::allocate(byte_len(hc_dim)?)
        .map_err(|err| format!("failed to allocate flat_hc: {err}"))?;
    let mut hc_mix = Tensor::allocate(byte_len(hc_mix_dim)?)
        .map_err(|err| format!("failed to allocate hc_mix: {err}"))?;
    let mut hc_split = Tensor::allocate(byte_len(hc_mix_dim)?)
        .map_err(|err| format!("failed to allocate hc_split: {err}"))?;
    let mut attn_cur = Tensor::allocate(byte_len(u64::from(N_EMBD))?)
        .map_err(|err| format!("failed to allocate attn_cur: {err}"))?;
    let mut attn_norm = Tensor::allocate(byte_len(u64::from(N_EMBD))?)
        .map_err(|err| format!("failed to allocate attn_norm: {err}"))?;

    let command_batch = CommandBatch::begin().map_err(|err| format!("begin failed: {err}"))?;
    backend
        .embed_token_hc(
            cur_hc.as_tensor_mut(),
            weights.token_embd.abs_offset,
            N_VOCAB,
            TOKEN,
            N_EMBD,
            N_HC,
        )
        .map_err(|err| format!("embed_token_hc failed: {err}"))?;
    backend
        .rms_norm_plain(
            flat_hc.as_tensor_mut(),
            cur_hc.as_tensor_ref(),
            hc_dim as u32,
            RMS_EPS,
        )
        .map_err(|err| format!("rms_norm_plain failed: {err}"))?;
    backend
        .matmul_f16(
            hc_mix.as_tensor_mut(),
            layer0.hc_attn_fn.abs_offset,
            hc_dim,
            hc_mix_dim,
            flat_hc.as_tensor_ref(),
            1,
        )
        .map_err(|err| format!("matmul_f16 failed: {err}"))?;
    backend
        .hc_split_weighted_sum_norm(
            attn_cur.as_tensor_mut(),
            attn_norm.as_tensor_mut(),
            hc_split.as_tensor_mut(),
            hc_mix.as_tensor_ref(),
            cur_hc.as_tensor_ref(),
            layer0.hc_attn_scale.abs_offset,
            layer0.hc_attn_base.abs_offset,
            layer0.attn_norm.abs_offset,
            N_EMBD,
            N_HC,
            N_HC_SINKHORN_ITER,
            HC_EPS,
            RMS_EPS,
        )
        .map_err(|err| format!("hc_split_weighted_sum_norm failed: {err}"))?;
    command_batch
        .finish()
        .map_err(|err| format!("finish failed: {err}"))?;
    synchronize().map_err(|err| format!("synchronize failed: {err}"))?;

    let outputs = vec![
        read_tensor_output("cur_hc", &cur_hc, hc_dim)?,
        read_tensor_output("flat_hc", &flat_hc, hc_dim)?,
        read_tensor_output("hc_mix", &hc_mix, hc_mix_dim)?,
        read_tensor_output("hc_split", &hc_split, hc_mix_dim)?,
        read_tensor_output("attn_cur", &attn_cur, u64::from(N_EMBD))?,
        read_tensor_output("attn_norm", &attn_norm, u64::from(N_EMBD))?,
    ];
    write_report(
        &gguf,
        &weights,
        header_bytes_read,
        mapped.size,
        hc_dim,
        hc_mix_dim,
        &outputs,
    );
    Ok(())
}

struct Args {
    model: PathBuf,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn std::error::Error>> {
        let mut model = None;
        let mut args = std::env::args_os().skip(1);
        while let Some(arg) = args.next() {
            if arg == "--model" {
                let Some(value) = args.next() else {
                    return Err("--model requires a path".into());
                };
                model = Some(PathBuf::from(value));
            } else {
                return Err("usage: ds4-decode-layer0-attn-hc-pre --model FILE".into());
            }
        }
        let Some(model) = model else {
            return Err("usage: ds4-decode-layer0-attn-hc-pre --model FILE".into());
        };
        Ok(Self { model })
    }
}

struct BackendGuard;

impl Drop for BackendGuard {
    fn drop(&mut self) {
        unsafe {
            ds4_gpu::cleanup();
        }
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

struct TensorOutput {
    field: &'static str,
    bytes: u64,
    elements: u64,
    nonzero_elements: u64,
    fnv1a64: u64,
    samples: Vec<F32Sample>,
}

#[derive(Clone, Copy)]
struct F32Sample {
    index: u64,
    value: f32,
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

fn read_tensor_output(
    field: &'static str,
    tensor: &Tensor,
    elements: u64,
) -> Result<TensorOutput, Box<dyn std::error::Error>> {
    let bytes = elements
        .checked_mul(4)
        .ok_or("tensor byte length overflow")?;
    if tensor.byte_len() != bytes {
        return Err(format!(
            "{field} tensor length drift: got {}, expected {bytes}",
            tensor.byte_len()
        )
        .into());
    }
    let mut data = vec![0u8; usize::try_from(bytes)?];
    tensor
        .read_bytes(0, &mut data)
        .map_err(|err| format!("{field} readback failed: {err}"))?;
    Ok(TensorOutput {
        field,
        bytes,
        elements,
        nonzero_elements: count_nonzero_f32(&data)?,
        fnv1a64: fnv1a64(&data),
        samples: read_samples(&data, elements)?,
    })
}

fn read_samples(bytes: &[u8], elements: u64) -> Result<Vec<F32Sample>, String> {
    let mut samples = Vec::new();
    for index in sample_indices(elements) {
        let start = usize::try_from(index)
            .map_err(|_| "sample index too large".to_string())?
            .checked_mul(4)
            .ok_or_else(|| "sample index overflow".to_string())?;
        let chunk = bytes
            .get(start..start + 4)
            .ok_or_else(|| format!("sample index {index} out of range"))?;
        let value = f32::from_le_bytes(chunk.try_into().expect("four-byte chunk"));
        samples.push(F32Sample { index, value });
    }
    Ok(samples)
}

fn sample_indices(elements: u64) -> Vec<u64> {
    let raw = [
        0,
        1,
        elements / 2,
        if elements > 1 { elements - 2 } else { 0 },
        elements.saturating_sub(1),
    ];
    let mut out = Vec::new();
    for index in raw {
        if index >= elements || out.contains(&index) {
            continue;
        }
        out.push(index);
    }
    out
}

fn count_nonzero_f32(bytes: &[u8]) -> Result<u64, String> {
    if bytes.len() % 4 != 0 {
        return Err("tensor byte length is not f32-aligned".to_string());
    }
    let mut count = 0u64;
    for chunk in bytes.chunks_exact(4) {
        let value = f32::from_le_bytes(chunk.try_into().expect("four-byte chunk"));
        if value != 0.0 {
            count += 1;
        }
    }
    Ok(count)
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn byte_len(elements: u64) -> Result<usize, Box<dyn std::error::Error>> {
    Ok(usize::try_from(
        elements
            .checked_mul(4)
            .ok_or("tensor byte length overflow")?,
    )?)
}

fn write_report(
    gguf: &Gguf,
    weights: &ds4_gguf::Ds4Weights,
    header_bytes_read: u64,
    mapped_size: u64,
    hc_dim: u64,
    hc_mix_dim: u64,
    outputs: &[TensorOutput],
) {
    let layer0 = &weights.layers[LAYER];
    println!("{{");
    println!("  \"schema\": \"{SCHEMA}\",");
    println!("  \"case\": \"{CASE}\",");
    println!("  \"model\": {{");
    println!("    \"mapped_size\": {mapped_size},");
    println!("    \"header_bytes_read\": {header_bytes_read},");
    println!("    \"tensor_count\": {},", gguf.tensors.len());
    println!("    \"tensor_data_offset\": {},", gguf.tensor_data_offset);
    println!("    \"bound_layers\": {}", weights.layers.len());
    println!("  }},");
    println!("  \"operation\": {{");
    println!("    \"name\": \"ds4_gpu_layer0_attn_hc_pre_prefix\",");
    println!(
        "    \"method\": \"embed_token_hc+rms_norm_plain+matmul_f16+hc_split_weighted_sum_norm\","
    );
    println!("    \"command_batch\": true,");
    println!("    \"synchronized\": true,");
    println!("    \"token\": {TOKEN},");
    println!("    \"layer\": {LAYER},");
    println!("    \"n_vocab\": {N_VOCAB},");
    println!("    \"n_embd\": {N_EMBD},");
    println!("    \"n_hc\": {N_HC},");
    println!("    \"n_hc_sinkhorn_iter\": {N_HC_SINKHORN_ITER},");
    println!("    \"hc_dim\": {hc_dim},");
    println!("    \"hc_mix_dim\": {hc_mix_dim},");
    println!("    \"hc_eps\": {HC_EPS},");
    println!("    \"rms_eps\": {RMS_EPS}");
    println!("  }},");
    println!("  \"weights\": {{");
    write_weight("token_embd", "base.token_embd", &weights.token_embd, true);
    write_weight(
        "hc_attn_fn",
        "base.layer.0.hc_attn_fn",
        &layer0.hc_attn_fn,
        true,
    );
    write_weight(
        "hc_attn_scale",
        "base.layer.0.hc_attn_scale",
        &layer0.hc_attn_scale,
        true,
    );
    write_weight(
        "hc_attn_base",
        "base.layer.0.hc_attn_base",
        &layer0.hc_attn_base,
        true,
    );
    write_weight(
        "attn_norm",
        "base.layer.0.attn_norm",
        &layer0.attn_norm,
        false,
    );
    println!("  }},");
    println!("  \"outputs\": {{");
    for (idx, output) in outputs.iter().enumerate() {
        write_output(output, idx + 1 != outputs.len());
    }
    println!("  }}");
    println!("}}");
}

fn write_weight(key: &str, role: &str, tensor: &TensorInfo, trailing_comma: bool) {
    println!("    \"{key}\": {{");
    println!("      \"role\": \"{role}\",");
    println!("      \"abs_offset\": {},", tensor.abs_offset);
    println!("      \"bytes\": {},", tensor.bytes);
    println!("      \"type\": {},", tensor.type_id);
    println!(
        "      \"type_name\": \"{}\"",
        tensor_type_name(tensor.type_id)
    );
    print!("    }}");
    if trailing_comma {
        print!(",");
    }
    println!();
}

fn write_output(output: &TensorOutput, trailing_comma: bool) {
    println!("    \"{}\": {{", output.field);
    println!("      \"field\": \"{}\",", output.field);
    println!("      \"bytes\": {},", output.bytes);
    println!("      \"elements\": {},", output.elements);
    println!("      \"nonzero_elements\": {},", output.nonzero_elements);
    println!("      \"fnv1a64\": \"{:016x}\",", output.fnv1a64);
    println!("      \"samples\": [");
    for (idx, sample) in output.samples.iter().enumerate() {
        if idx != 0 {
            println!(",");
        }
        print!(
            "        {{\"index\": {}, \"value\": {}}}",
            sample.index, sample.value
        );
    }
    println!();
    println!("      ]");
    print!("    }}");
    if trailing_comma {
        print!(",");
    }
    println!();
}
