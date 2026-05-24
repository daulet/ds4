use ds4_gguf::session_payload::{
    compress_ratio, default_graph_payload_runtime, layer_attn_state_bytes, layer_index_state_bytes,
    read_graph_payload, GraphPayloadRead, PayloadSections, HEADER_BYTES,
};
use ds4_gpu::graph_plan::{
    layer_compression, GraphPlan, LayerCompression, N_HEAD_DIM, N_INDEXER_HEAD_DIM, N_LAYER,
    N_VOCAB,
};
use ds4_gpu::{initialize, synchronize, Tensor};
use std::fs;
use std::io::{self, Write};

const SCHEMA: &str = "ds4.rust_graph_restore_readback.v1";
const SOURCE: &str = "rust-graph-restore-readback";
const CTX_SIZE: u32 = 32_768;
const USAGE: &str = "usage: ds4-graph-restore-readback --case <id:path> [--case <id:path>...]";

fn main() {
    if let Err(err) = run() {
        eprintln!("ds4-graph-restore-readback: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let inputs = parse_args()?;
    initialize().map_err(|err| format!("failed to initialize backend: {err}"))?;
    let _backend = BackendGuard;

    let mut reports = Vec::with_capacity(inputs.len());
    for input in &inputs {
        reports.push(process_case(input)?);
    }
    write_report(&mut io::BufWriter::new(io::stdout()), &reports)?;
    Ok(())
}

struct BackendGuard;

impl Drop for BackendGuard {
    fn drop(&mut self) {
        unsafe {
            ds4_gpu::cleanup();
        }
    }
}

#[derive(Debug, Clone)]
struct CaseInput {
    id: String,
    path: String,
}

fn parse_args() -> Result<Vec<CaseInput>, Box<dyn std::error::Error>> {
    let mut inputs = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg != "--case" {
            return Err(USAGE.into());
        }
        let spec = args.next().ok_or(USAGE)?;
        let (id, path) = spec
            .split_once(':')
            .ok_or("usage: --case requires <id:path>")?;
        if id.is_empty() || path.is_empty() {
            return Err("usage: --case requires non-empty <id:path>".into());
        }
        inputs.push(CaseInput {
            id: id.to_owned(),
            path: path.to_owned(),
        });
    }
    if inputs.is_empty() {
        return Err(USAGE.into());
    }
    Ok(inputs)
}

struct CaseReport {
    input: CaseInput,
    payload_bytes: u64,
    file_fnv1a64: u64,
    parsed: GraphPayloadRead,
    readback: ReadbackReport,
}

fn process_case(input: &CaseInput) -> Result<CaseReport, Box<dyn std::error::Error>> {
    let bytes =
        fs::read(&input.path).map_err(|err| format!("failed to read {}: {err}", input.path))?;
    let runtime = default_graph_payload_runtime(CTX_SIZE);
    let parsed = read_graph_payload(&bytes, runtime)
        .map_err(|err| format!("{}: graph payload parse failed: {err}", input.id))?;
    let mut state = RestoreState::allocate()?;
    restore_payload(&bytes, &parsed, &mut state)
        .map_err(|err| format!("{}: restore write failed: {err}", input.id))?;
    synchronize().map_err(|err| format!("{}: synchronize failed: {err}", input.id))?;
    let readback = readback_payload(&bytes, &parsed, &state)
        .map_err(|err| format!("{}: readback failed: {err}", input.id))?;
    Ok(CaseReport {
        input: input.clone(),
        payload_bytes: bytes.len() as u64,
        file_fnv1a64: fnv1a64(&bytes),
        parsed,
        readback,
    })
}

struct RestoreState {
    checkpoint: Vec<u8>,
    logits: Vec<u8>,
    layer_n_comp: [u32; N_LAYER],
    layer_n_index_comp: [u32; N_LAYER],
    graph: GraphRestoreState,
}

impl RestoreState {
    fn allocate() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            checkpoint: Vec::new(),
            logits: Vec::new(),
            layer_n_comp: [0; N_LAYER],
            layer_n_index_comp: [0; N_LAYER],
            graph: GraphRestoreState::allocate()?,
        })
    }
}

struct GraphRestoreState {
    layer_raw_cache: Vec<Tensor>,
    layer_attn_comp_cache: Vec<Option<Tensor>>,
    layer_attn_state_kv: Vec<Option<Tensor>>,
    layer_attn_state_score: Vec<Option<Tensor>>,
    layer_index_comp_cache: Vec<Option<Tensor>>,
    layer_index_state_kv: Vec<Option<Tensor>>,
    layer_index_state_score: Vec<Option<Tensor>>,
}

impl GraphRestoreState {
    fn allocate() -> Result<Self, Box<dyn std::error::Error>> {
        let plan = GraphPlan::for_context(CTX_SIZE, CTX_SIZE, false);
        let mut layer_raw_cache = Vec::with_capacity(N_LAYER);
        let mut layer_attn_comp_cache = Vec::with_capacity(N_LAYER);
        let mut layer_attn_state_kv = Vec::with_capacity(N_LAYER);
        let mut layer_attn_state_score = Vec::with_capacity(N_LAYER);
        let mut layer_index_comp_cache = Vec::with_capacity(N_LAYER);
        let mut layer_index_state_kv = Vec::with_capacity(N_LAYER);
        let mut layer_index_state_score = Vec::with_capacity(N_LAYER);

        for layer in 0..N_LAYER {
            layer_raw_cache.push(allocate_tensor(
                "layer_raw_cache",
                layer,
                u64::from(plan.allocated_raw_cap) * u64::from(N_HEAD_DIM) * 4,
            )?);
            match layer_compression(layer).ok_or("invalid DS4 layer")? {
                LayerCompression::Dense => {
                    layer_attn_comp_cache.push(None);
                    layer_attn_state_kv.push(None);
                    layer_attn_state_score.push(None);
                    layer_index_comp_cache.push(None);
                    layer_index_state_kv.push(None);
                    layer_index_state_score.push(None);
                }
                LayerCompression::Ratio4 | LayerCompression::Ratio128 => {
                    let ratio = compress_ratio(layer);
                    let comp_cap = plan.layer_comp_cap(layer_compression(layer).unwrap());
                    layer_attn_comp_cache.push(Some(allocate_tensor(
                        "layer_attn_comp_cache",
                        layer,
                        u64::from(comp_cap) * u64::from(N_HEAD_DIM) * 4,
                    )?));
                    layer_attn_state_kv.push(Some(allocate_tensor(
                        "layer_attn_state_kv",
                        layer,
                        layer_attn_state_bytes(ratio),
                    )?));
                    layer_attn_state_score.push(Some(allocate_tensor(
                        "layer_attn_state_score",
                        layer,
                        layer_attn_state_bytes(ratio),
                    )?));
                    if ratio == 4 {
                        layer_index_comp_cache.push(Some(allocate_tensor(
                            "layer_index_comp_cache",
                            layer,
                            u64::from(comp_cap) * u64::from(N_INDEXER_HEAD_DIM) * 4,
                        )?));
                        layer_index_state_kv.push(Some(allocate_tensor(
                            "layer_index_state_kv",
                            layer,
                            layer_index_state_bytes(ratio),
                        )?));
                        layer_index_state_score.push(Some(allocate_tensor(
                            "layer_index_state_score",
                            layer,
                            layer_index_state_bytes(ratio),
                        )?));
                    } else {
                        layer_index_comp_cache.push(None);
                        layer_index_state_kv.push(None);
                        layer_index_state_score.push(None);
                    }
                }
            }
        }

        Ok(Self {
            layer_raw_cache,
            layer_attn_comp_cache,
            layer_attn_state_kv,
            layer_attn_state_score,
            layer_index_comp_cache,
            layer_index_state_kv,
            layer_index_state_score,
        })
    }
}

fn allocate_tensor(
    field: &str,
    layer: usize,
    bytes: u64,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    Tensor::allocate(usize::try_from(bytes).map_err(|_| "tensor byte length overflow")?)
        .map_err(|err| format!("failed to allocate {field}[{layer}]: {err}").into())
}

fn restore_payload(
    bytes: &[u8],
    parsed: &GraphPayloadRead,
    state: &mut RestoreState,
) -> Result<(), String> {
    let mut pos = HEADER_BYTES;
    let token_bytes = u64::from(parsed.header.token_count) * 4;
    state.checkpoint = take(bytes, &mut pos, token_bytes)?.to_vec();
    state.logits = take(bytes, &mut pos, u64::from(N_VOCAB) * 4)?.to_vec();

    let n_comp_bytes = take(bytes, &mut pos, N_LAYER as u64 * 4)?;
    let n_index_comp_bytes = take(bytes, &mut pos, N_LAYER as u64 * 4)?;
    state.layer_n_comp = parsed.n_comp;
    state.layer_n_index_comp = parsed.n_index_comp;
    if u32_array_bytes(&state.layer_n_comp) != n_comp_bytes {
        return Err("n_comp table bytes do not match parsed counts".to_string());
    }
    if u32_array_bytes(&state.layer_n_index_comp) != n_index_comp_bytes {
        return Err("n_index_comp table bytes do not match parsed counts".to_string());
    }

    let row_bytes = u64::from(N_HEAD_DIM) * 4;
    let raw_first = parsed.header.token_count - parsed.header.raw_live_rows;
    for layer in 0..N_LAYER {
        for row in 0..parsed.header.raw_live_rows {
            let source = take(bytes, &mut pos, row_bytes)?;
            let phys = (raw_first + row) % parsed.header.raw_cap;
            state.graph.layer_raw_cache[layer]
                .write_bytes(u64::from(phys) * row_bytes, source)
                .map_err(|err| format!("layer {layer} raw row write failed: {err}"))?;
        }

        let ratio = compress_ratio(layer);
        if ratio == 0 {
            continue;
        }
        let comp_bytes = u64::from(parsed.n_comp[layer]) * u64::from(N_HEAD_DIM) * 4;
        write_tensor_span(
            &mut state.graph.layer_attn_comp_cache[layer],
            "layer_attn_comp_cache",
            layer,
            take(bytes, &mut pos, comp_bytes)?,
        )?;
        write_tensor_span(
            &mut state.graph.layer_attn_state_kv[layer],
            "layer_attn_state_kv",
            layer,
            take(bytes, &mut pos, layer_attn_state_bytes(ratio))?,
        )?;
        write_tensor_span(
            &mut state.graph.layer_attn_state_score[layer],
            "layer_attn_state_score",
            layer,
            take(bytes, &mut pos, layer_attn_state_bytes(ratio))?,
        )?;
        if ratio == 4 {
            let index_bytes =
                u64::from(parsed.n_index_comp[layer]) * u64::from(N_INDEXER_HEAD_DIM) * 4;
            write_tensor_span(
                &mut state.graph.layer_index_comp_cache[layer],
                "layer_index_comp_cache",
                layer,
                take(bytes, &mut pos, index_bytes)?,
            )?;
            write_tensor_span(
                &mut state.graph.layer_index_state_kv[layer],
                "layer_index_state_kv",
                layer,
                take(bytes, &mut pos, layer_index_state_bytes(ratio))?,
            )?;
            write_tensor_span(
                &mut state.graph.layer_index_state_score[layer],
                "layer_index_state_score",
                layer,
                take(bytes, &mut pos, layer_index_state_bytes(ratio))?,
            )?;
        }
    }
    if pos != bytes.len() {
        return Err("payload has unread trailing bytes after restore".to_string());
    }
    Ok(())
}

fn write_tensor_span(
    tensor: &mut Option<Tensor>,
    field: &str,
    layer: usize,
    data: &[u8],
) -> Result<(), String> {
    let tensor = tensor
        .as_mut()
        .ok_or_else(|| format!("{field}[{layer}] is not allocated"))?;
    tensor
        .write_bytes(0, data)
        .map_err(|err| format!("{field}[{layer}] write failed: {err}"))
}

struct ReadbackReport {
    checkpoint: SectionDigest,
    logits: SectionDigest,
    attn_counts: SectionDigest,
    index_counts: SectionDigest,
    raw_rows: SectionDigest,
    attn_compressed_rows: SectionDigest,
    attn_state_kv: SectionDigest,
    attn_state_score: SectionDigest,
    indexer_compressed_rows: SectionDigest,
    index_state_kv: SectionDigest,
    index_state_score: SectionDigest,
    samples: Vec<LayerSample>,
    layer_n_comp: [u32; N_LAYER],
    layer_n_index_comp: [u32; N_LAYER],
}

struct LayerSample {
    layer: usize,
    ratio: u32,
    raw: SectionDigest,
    attn_compressed_rows: Option<SectionDigest>,
    attn_state_kv: Option<SectionDigest>,
    attn_state_score: Option<SectionDigest>,
    indexer_compressed_rows: Option<SectionDigest>,
    index_state_kv: Option<SectionDigest>,
    index_state_score: Option<SectionDigest>,
}

fn readback_payload(
    bytes: &[u8],
    parsed: &GraphPayloadRead,
    state: &RestoreState,
) -> Result<ReadbackReport, String> {
    let mut pos = HEADER_BYTES;
    let checkpoint_source = take(bytes, &mut pos, u64::from(parsed.header.token_count) * 4)?;
    let checkpoint = SectionDigest::from_pair(checkpoint_source, &state.checkpoint);
    let logits_source = take(bytes, &mut pos, u64::from(N_VOCAB) * 4)?;
    let logits = SectionDigest::from_pair(logits_source, &state.logits);
    let n_comp_source = take(bytes, &mut pos, N_LAYER as u64 * 4)?;
    let n_comp_readback = u32_array_bytes(&state.layer_n_comp);
    let attn_counts = SectionDigest::from_pair(n_comp_source, &n_comp_readback);
    let n_index_source = take(bytes, &mut pos, N_LAYER as u64 * 4)?;
    let n_index_readback = u32_array_bytes(&state.layer_n_index_comp);
    let index_counts = SectionDigest::from_pair(n_index_source, &n_index_readback);

    let mut raw_rows = SectionHasher::default();
    let mut attn_compressed_rows = SectionHasher::default();
    let mut attn_state_kv = SectionHasher::default();
    let mut attn_state_score = SectionHasher::default();
    let mut indexer_compressed_rows = SectionHasher::default();
    let mut index_state_kv = SectionHasher::default();
    let mut index_state_score = SectionHasher::default();
    let mut samples = Vec::new();
    let raw_first = parsed.header.token_count - parsed.header.raw_live_rows;
    let row_bytes = u64::from(N_HEAD_DIM) * 4;

    for layer in 0..N_LAYER {
        let mut raw_sample = SectionHasher::default();
        for row in 0..parsed.header.raw_live_rows {
            let source = take(bytes, &mut pos, row_bytes)?;
            let phys = (raw_first + row) % parsed.header.raw_cap;
            let readback = read_tensor_bytes(
                &state.graph.layer_raw_cache[layer],
                u64::from(phys) * row_bytes,
                row_bytes,
                "layer_raw_cache",
                layer,
            )?;
            raw_rows.update(source, &readback);
            raw_sample.update(source, &readback);
        }

        let ratio = compress_ratio(layer);
        let mut sample = LayerSample {
            layer,
            ratio,
            raw: raw_sample.finish(),
            attn_compressed_rows: None,
            attn_state_kv: None,
            attn_state_score: None,
            indexer_compressed_rows: None,
            index_state_kv: None,
            index_state_score: None,
        };
        if ratio != 0 {
            let comp_bytes = u64::from(parsed.n_comp[layer]) * u64::from(N_HEAD_DIM) * 4;
            let digest = readback_optional_tensor(
                bytes,
                &mut pos,
                &state.graph.layer_attn_comp_cache[layer],
                "layer_attn_comp_cache",
                layer,
                comp_bytes,
                &mut attn_compressed_rows,
            )?;
            sample.attn_compressed_rows = Some(digest);

            let digest = readback_optional_tensor(
                bytes,
                &mut pos,
                &state.graph.layer_attn_state_kv[layer],
                "layer_attn_state_kv",
                layer,
                layer_attn_state_bytes(ratio),
                &mut attn_state_kv,
            )?;
            sample.attn_state_kv = Some(digest);

            let digest = readback_optional_tensor(
                bytes,
                &mut pos,
                &state.graph.layer_attn_state_score[layer],
                "layer_attn_state_score",
                layer,
                layer_attn_state_bytes(ratio),
                &mut attn_state_score,
            )?;
            sample.attn_state_score = Some(digest);
        }
        if ratio == 4 {
            let index_bytes =
                u64::from(parsed.n_index_comp[layer]) * u64::from(N_INDEXER_HEAD_DIM) * 4;
            let digest = readback_optional_tensor(
                bytes,
                &mut pos,
                &state.graph.layer_index_comp_cache[layer],
                "layer_index_comp_cache",
                layer,
                index_bytes,
                &mut indexer_compressed_rows,
            )?;
            sample.indexer_compressed_rows = Some(digest);

            let digest = readback_optional_tensor(
                bytes,
                &mut pos,
                &state.graph.layer_index_state_kv[layer],
                "layer_index_state_kv",
                layer,
                layer_index_state_bytes(ratio),
                &mut index_state_kv,
            )?;
            sample.index_state_kv = Some(digest);

            let digest = readback_optional_tensor(
                bytes,
                &mut pos,
                &state.graph.layer_index_state_score[layer],
                "layer_index_state_score",
                layer,
                layer_index_state_bytes(ratio),
                &mut index_state_score,
            )?;
            sample.index_state_score = Some(digest);
        }
        if matches!(layer, 0 | 2 | 3 | 42) {
            samples.push(sample);
        }
    }
    if pos != bytes.len() {
        return Err("payload has unread trailing bytes after readback".to_string());
    }

    Ok(ReadbackReport {
        checkpoint,
        logits,
        attn_counts,
        index_counts,
        raw_rows: raw_rows.finish(),
        attn_compressed_rows: attn_compressed_rows.finish(),
        attn_state_kv: attn_state_kv.finish(),
        attn_state_score: attn_state_score.finish(),
        indexer_compressed_rows: indexer_compressed_rows.finish(),
        index_state_kv: index_state_kv.finish(),
        index_state_score: index_state_score.finish(),
        samples,
        layer_n_comp: state.layer_n_comp,
        layer_n_index_comp: state.layer_n_index_comp,
    })
}

fn readback_optional_tensor(
    bytes: &[u8],
    pos: &mut usize,
    tensor: &Option<Tensor>,
    field: &str,
    layer: usize,
    byte_len: u64,
    aggregate: &mut SectionHasher,
) -> Result<SectionDigest, String> {
    let source = take(bytes, pos, byte_len)?;
    let tensor = tensor
        .as_ref()
        .ok_or_else(|| format!("{field}[{layer}] is not allocated"))?;
    let readback = read_tensor_bytes(tensor, 0, byte_len, field, layer)?;
    aggregate.update(source, &readback);
    Ok(SectionDigest::from_pair(source, &readback))
}

fn read_tensor_bytes(
    tensor: &Tensor,
    offset: u64,
    byte_len: u64,
    field: &str,
    layer: usize,
) -> Result<Vec<u8>, String> {
    let mut out =
        vec![0_u8; usize::try_from(byte_len).map_err(|_| "readback byte length overflow")?];
    tensor
        .read_bytes(offset, &mut out)
        .map_err(|err| format!("{field}[{layer}] read failed: {err}"))?;
    Ok(out)
}

#[derive(Clone, Copy)]
struct SectionDigest {
    bytes: u64,
    source_fnv1a64: u64,
    readback_fnv1a64: u64,
}

impl SectionDigest {
    fn from_pair(source: &[u8], readback: &[u8]) -> Self {
        Self {
            bytes: source.len() as u64,
            source_fnv1a64: fnv1a64(source),
            readback_fnv1a64: fnv1a64(readback),
        }
    }

    fn matched(self) -> bool {
        self.source_fnv1a64 == self.readback_fnv1a64
    }
}

struct SectionHasher {
    bytes: u64,
    source: Fnv1a64,
    readback: Fnv1a64,
}

impl Default for SectionHasher {
    fn default() -> Self {
        Self {
            bytes: 0,
            source: Fnv1a64::default(),
            readback: Fnv1a64::default(),
        }
    }
}

impl SectionHasher {
    fn update(&mut self, source: &[u8], readback: &[u8]) {
        self.bytes += source.len() as u64;
        self.source.update(source);
        self.readback.update(readback);
    }

    fn finish(self) -> SectionDigest {
        SectionDigest {
            bytes: self.bytes,
            source_fnv1a64: self.source.finish(),
            readback_fnv1a64: self.readback.finish(),
        }
    }
}

#[derive(Clone, Copy)]
struct Fnv1a64 {
    hash: u64,
}

impl Default for Fnv1a64 {
    fn default() -> Self {
        Self {
            hash: 0xcbf29ce484222325,
        }
    }
}

impl Fnv1a64 {
    fn update(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.hash ^= u64::from(byte);
            self.hash = self.hash.wrapping_mul(0x100000001b3);
        }
    }

    fn finish(self) -> u64 {
        self.hash
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = Fnv1a64::default();
    hash.update(bytes);
    hash.finish()
}

fn take<'a>(bytes: &'a [u8], pos: &mut usize, byte_len: u64) -> Result<&'a [u8], String> {
    let len = usize::try_from(byte_len).map_err(|_| "payload section byte length overflow")?;
    let end = pos
        .checked_add(len)
        .ok_or_else(|| "payload section offset overflow".to_string())?;
    if end > bytes.len() {
        return Err("payload section extends past end of file".to_string());
    }
    let out = &bytes[*pos..end];
    *pos = end;
    Ok(out)
}

fn u32_array_bytes(values: &[u32; N_LAYER]) -> Vec<u8> {
    let mut out = Vec::with_capacity(N_LAYER * 4);
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

fn write_report<W: Write>(out: &mut W, reports: &[CaseReport]) -> io::Result<()> {
    writeln!(out, "{{")?;
    writeln!(out, "  \"schema\": \"{SCHEMA}\",")?;
    writeln!(out, "  \"source\": \"{SOURCE}\",")?;
    writeln!(
        out,
        "  \"runtime\": {{\"ctx\": {CTX_SIZE}, \"kind\": \"default-graph\", \"backend\": \"ds4-gpu\"}},"
    )?;
    writeln!(out, "  \"cases\": [")?;
    for (idx, report) in reports.iter().enumerate() {
        if idx != 0 {
            writeln!(out, ",")?;
        }
        write_case_report(out, report)?;
    }
    writeln!(out, "\n  ]")?;
    writeln!(out, "}}")
}

fn write_case_report<W: Write>(out: &mut W, report: &CaseReport) -> io::Result<()> {
    write!(out, "    {{\"id\": ")?;
    write_json_string(out, &report.input.id)?;
    write!(out, ", \"path\": ")?;
    write_json_string(out, &report.input.path)?;
    write!(
        out,
        ", \"payload_bytes\": {}, \"file_fnv1a64\": \"{:016x}\", \"ok\": true, \"error\": \"\", ",
        report.payload_bytes, report.file_fnv1a64
    )?;
    write!(out, "\"parsed\": ")?;
    write_parsed(out, &report.parsed)?;
    write!(out, ", \"readback\": ")?;
    write_readback(out, &report.readback)?;
    write!(out, "}}")
}

fn write_parsed<W: Write>(out: &mut W, parsed: &GraphPayloadRead) -> io::Result<()> {
    write!(
        out,
        "{{\"token_count\": {}, \"raw_first_pos\": {}, \"raw_last_pos\": {}, \
         \"raw_first_phys\": {}, \"raw_last_phys\": {}, \"payload_bytes\": {}, \
         \"ratio4_rows\": {}, \"ratio128_rows\": {}, \"layer2_n_index_comp\": {}, \
         \"section_bytes\": ",
        parsed.header.token_count,
        parsed.raw_first_pos,
        parsed.raw_last_pos,
        parsed.raw_first_phys,
        parsed.raw_last_phys,
        parsed.payload_bytes,
        parsed.n_comp[2],
        parsed.n_comp[3],
        parsed.n_index_comp[2]
    )?;
    write_section_bytes(out, parsed.sections)?;
    write!(out, "}}")
}

fn write_section_bytes<W: Write>(out: &mut W, sections: PayloadSections) -> io::Result<()> {
    write!(
        out,
        "{{\"header\": {}, \"tokens\": {}, \"logits\": {}, \"attn_counts\": {}, \
         \"index_counts\": {}, \"raw_rows\": {}, \"attn_compressed_rows\": {}, \
         \"attn_state\": {}, \"indexer_compressed_rows\": {}, \"indexer_state\": {}}}",
        HEADER_BYTES,
        sections.token_bytes,
        sections.logits_bytes,
        sections.comp_count_bytes,
        sections.index_count_bytes,
        sections.raw_row_bytes,
        sections.attn_comp_row_bytes,
        sections.attn_state_bytes,
        sections.index_comp_row_bytes,
        sections.index_state_bytes
    )
}

fn write_readback<W: Write>(out: &mut W, readback: &ReadbackReport) -> io::Result<()> {
    write!(out, "{{\"checkpoint\": ")?;
    write_digest(out, &readback.checkpoint)?;
    write!(out, ", \"logits\": ")?;
    write_digest(out, &readback.logits)?;
    write!(out, ", \"attn_counts\": ")?;
    write_digest(out, &readback.attn_counts)?;
    write!(out, ", \"index_counts\": ")?;
    write_digest(out, &readback.index_counts)?;
    write!(out, ", \"raw_rows\": ")?;
    write_digest(out, &readback.raw_rows)?;
    write!(out, ", \"attn_compressed_rows\": ")?;
    write_digest(out, &readback.attn_compressed_rows)?;
    write!(out, ", \"attn_state_kv\": ")?;
    write_digest(out, &readback.attn_state_kv)?;
    write!(out, ", \"attn_state_score\": ")?;
    write_digest(out, &readback.attn_state_score)?;
    write!(out, ", \"indexer_compressed_rows\": ")?;
    write_digest(out, &readback.indexer_compressed_rows)?;
    write!(out, ", \"index_state_kv\": ")?;
    write_digest(out, &readback.index_state_kv)?;
    write!(out, ", \"index_state_score\": ")?;
    write_digest(out, &readback.index_state_score)?;
    write!(out, ", \"samples\": [")?;
    for (idx, sample) in readback.samples.iter().enumerate() {
        if idx != 0 {
            write!(out, ", ")?;
        }
        write_sample(out, sample)?;
    }
    write!(out, "], \"post_restore_state\": {{\"checkpoint_valid\": true, \"mtp_draft_valid\": false, \"mtp_n_raw\": 0, \"layer_n_comp\": ")?;
    write_u32_array(out, &readback.layer_n_comp)?;
    write!(out, ", \"layer_n_index_comp\": ")?;
    write_u32_array(out, &readback.layer_n_index_comp)?;
    write!(out, "}}}}")
}

fn write_sample<W: Write>(out: &mut W, sample: &LayerSample) -> io::Result<()> {
    write!(
        out,
        "{{\"layer\": {}, \"ratio\": {}, \"raw\": ",
        sample.layer, sample.ratio
    )?;
    write_digest(out, &sample.raw)?;
    write!(out, ", \"attn_compressed_rows\": ")?;
    write_optional_digest(out, sample.attn_compressed_rows.as_ref())?;
    write!(out, ", \"attn_state_kv\": ")?;
    write_optional_digest(out, sample.attn_state_kv.as_ref())?;
    write!(out, ", \"attn_state_score\": ")?;
    write_optional_digest(out, sample.attn_state_score.as_ref())?;
    write!(out, ", \"indexer_compressed_rows\": ")?;
    write_optional_digest(out, sample.indexer_compressed_rows.as_ref())?;
    write!(out, ", \"index_state_kv\": ")?;
    write_optional_digest(out, sample.index_state_kv.as_ref())?;
    write!(out, ", \"index_state_score\": ")?;
    write_optional_digest(out, sample.index_state_score.as_ref())?;
    write!(out, "}}")
}

fn write_optional_digest<W: Write>(out: &mut W, digest: Option<&SectionDigest>) -> io::Result<()> {
    if let Some(digest) = digest {
        write_digest(out, digest)
    } else {
        write!(out, "null")
    }
}

fn write_digest<W: Write>(out: &mut W, digest: &SectionDigest) -> io::Result<()> {
    write!(
        out,
        "{{\"bytes\": {}, \"source_fnv1a64\": \"{:016x}\", \
         \"readback_fnv1a64\": \"{:016x}\", \"matched\": {}}}",
        digest.bytes,
        digest.source_fnv1a64,
        digest.readback_fnv1a64,
        if digest.matched() { "true" } else { "false" }
    )
}

fn write_u32_array<W: Write>(out: &mut W, values: &[u32; N_LAYER]) -> io::Result<()> {
    write!(out, "[")?;
    for (idx, value) in values.iter().enumerate() {
        if idx != 0 {
            write!(out, ", ")?;
        }
        write!(out, "{value}")?;
    }
    write!(out, "]")
}

fn write_json_string<W: Write>(out: &mut W, value: &str) -> io::Result<()> {
    write!(out, "\"")?;
    for ch in value.chars() {
        match ch {
            '"' => write!(out, "\\\"")?,
            '\\' => write!(out, "\\\\")?,
            '\n' => write!(out, "\\n")?,
            '\r' => write!(out, "\\r")?,
            '\t' => write!(out, "\\t")?,
            ch if ch < ' ' => write!(out, "\\u{:04x}", ch as u32)?,
            ch => write!(out, "{ch}")?,
        }
    }
    write!(out, "\"")
}
