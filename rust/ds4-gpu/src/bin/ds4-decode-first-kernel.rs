use ds4_gguf::{bind_ds4_weights, parse_gguf_allowing_missing_tensor_data, tensor_type_name, Gguf};
use ds4_gpu::decode_backend::{set_model_fd, set_model_map_range, DecodeBackend, ModelMap};
use ds4_gpu::graph_plan::{N_EMBD, N_HC, N_VOCAB};
use ds4_gpu::{initialize, synchronize, CommandBatch, Tensor};
use std::ffi::c_void;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::os::raw::c_int;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};

const SCHEMA: &str = "ds4.decode_first_kernel.v1";
const CASE: &str = "embed_token_hc_token0";
const TOKEN: u32 = 0;
const PROT_READ: c_int = 0x1;
const MAP_PRIVATE: c_int = 0x02;
const MAP_FAILED: *mut c_void = !0usize as *mut c_void;
const INITIAL_HEADER_READ: u64 = 8 * 1024 * 1024;
const MAX_HEADER_READ: u64 = 512 * 1024 * 1024;
const SAMPLE_INDICES: &[usize] = &[0, 1, 8192, 16382, 16383];

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
        eprintln!("ds4-decode-first-kernel: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse()?;
    let (gguf, header_bytes_read) = parse_header_prefix(&args.model)?;
    let weights = bind_ds4_weights(&gguf)?;
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

    let mut cur_hc = Tensor::allocate((u64::from(N_HC) * u64::from(N_EMBD) * 4) as usize)
        .map_err(|err| format!("failed to allocate cur_hc: {err}"))?;
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
    command_batch
        .finish()
        .map_err(|err| format!("finish failed: {err}"))?;
    synchronize().map_err(|err| format!("synchronize failed: {err}"))?;

    let mut bytes = vec![0u8; cur_hc.byte_len() as usize];
    cur_hc
        .read_bytes(0, &mut bytes)
        .map_err(|err| format!("cur_hc readback failed: {err}"))?;
    let samples = read_samples(&bytes)?;
    let nonzero_elements = count_nonzero_f32(&bytes)?;
    let output_fnv1a64 = fnv1a64(&bytes);
    write_report(
        &gguf,
        &weights,
        header_bytes_read,
        mapped.size,
        cur_hc.byte_len(),
        nonzero_elements,
        output_fnv1a64,
        &samples,
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
                return Err("usage: ds4-decode-first-kernel --model FILE".into());
            }
        }
        let Some(model) = model else {
            return Err("usage: ds4-decode-first-kernel --model FILE".into());
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

#[derive(Clone, Copy)]
struct F32Sample {
    index: usize,
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

fn read_samples(bytes: &[u8]) -> Result<Vec<F32Sample>, String> {
    let mut samples = Vec::new();
    for &index in SAMPLE_INDICES {
        let start = index
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

fn count_nonzero_f32(bytes: &[u8]) -> Result<u32, String> {
    if bytes.len() % 4 != 0 {
        return Err("cur_hc byte length is not f32-aligned".to_string());
    }
    let mut count = 0u32;
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

#[allow(clippy::too_many_arguments)]
fn write_report(
    gguf: &Gguf,
    weights: &ds4_gguf::Ds4Weights,
    header_bytes_read: u64,
    mapped_size: u64,
    cur_hc_bytes: u64,
    nonzero_elements: u32,
    output_fnv1a64: u64,
    samples: &[F32Sample],
) {
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
    println!("    \"name\": \"ds4_gpu_embed_token_hc_tensor\",");
    println!("    \"method\": \"embed_token_hc\",");
    println!("    \"command_batch\": true,");
    println!("    \"synchronized\": true,");
    println!("    \"token\": {TOKEN},");
    println!("    \"n_vocab\": {N_VOCAB},");
    println!("    \"n_embd\": {N_EMBD},");
    println!("    \"n_hc\": {N_HC}");
    println!("  }},");
    println!("  \"weight\": {{");
    println!("    \"role\": \"base.token_embd\",");
    println!("    \"abs_offset\": {},", weights.token_embd.abs_offset);
    println!("    \"bytes\": {},", weights.token_embd.bytes);
    println!("    \"type\": {},", weights.token_embd.type_id);
    println!(
        "    \"type_name\": \"{}\"",
        tensor_type_name(weights.token_embd.type_id)
    );
    println!("  }},");
    println!("  \"output\": {{");
    println!("    \"field\": \"cur_hc\",");
    println!("    \"bytes\": {cur_hc_bytes},");
    println!(
        "    \"elements\": {},",
        cur_hc_bytes.checked_div(4).unwrap_or(0)
    );
    println!("    \"nonzero_elements\": {nonzero_elements},");
    println!("    \"fnv1a64\": \"{output_fnv1a64:016x}\",");
    println!("    \"samples\": [");
    for (idx, sample) in samples.iter().enumerate() {
        if idx != 0 {
            println!(",");
        }
        print!(
            "      {{\"index\": {}, \"value\": {}}}",
            sample.index, sample.value
        );
    }
    println!();
    println!("    ]");
    println!("  }}");
    println!("}}");
}
