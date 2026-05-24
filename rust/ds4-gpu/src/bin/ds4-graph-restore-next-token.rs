use ds4_gguf::kv_policy::{
    continued_store_target, KvPolicyConfig, REASON_CONTINUED, REASON_SHUTDOWN,
};
use ds4_gguf::session_payload::{
    compress_ratio, default_graph_payload_runtime, layer_attn_state_bytes, layer_index_state_bytes,
    read_graph_payload, GraphPayloadRead, PayloadSections, HEADER_BYTES,
};
use ds4_gguf::{sample_argmax, top_logprobs, TokenScore};
use ds4_gpu::graph_plan::{
    layer_compression, GraphPlan, LayerCompression, N_HEAD_DIM, N_INDEXER_HEAD_DIM, N_LAYER,
    N_VOCAB,
};
use ds4_gpu::{initialize, synchronize, Tensor};
use std::fs;
use std::io::{self, Write};

const SCHEMA: &str = "ds4.rust_graph_restore_next_token.v1";
const SOURCE: &str = "rust-graph-restore-next-token";
const CTX_SIZE: u32 = 32_768;
const TOP_K: usize = 20;
const USAGE: &str = "usage: ds4-graph-restore-next-token --case <id:path> [--case <id:path>...]";

fn main() {
    if let Err(err) = run() {
        eprintln!("ds4-graph-restore-next-token: {err}");
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
    next_token: NextTokenReport,
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
    let next_token = next_token_report(&parsed, &state)
        .map_err(|err| format!("{}: next-token report failed: {err}", input.id))?;
    Ok(CaseReport {
        input: input.clone(),
        payload_bytes: bytes.len() as u64,
        file_fnv1a64: fnv1a64(&bytes),
        parsed,
        next_token,
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

struct NextTokenReport {
    checkpoint_tokens: u32,
    checkpoint_fnv1a64: u64,
    logits_fnv1a64: u64,
    selected_token: i32,
    top_logprobs: Vec<TokenScore>,
    layer_n_comp: [u32; N_LAYER],
    layer_n_index_comp: [u32; N_LAYER],
    frontier_projection: FrontierProjection,
}

struct FrontierProjection {
    policy: KvPolicyConfig,
    continued_step_tokens: i32,
    loaded_frontier: i32,
    current_live_target: i32,
    next_continued_tokens: i32,
    next_continued_target: i32,
    already_stored_boundary_target: i32,
}

fn next_token_report(
    parsed: &GraphPayloadRead,
    state: &RestoreState,
) -> Result<NextTokenReport, String> {
    let logits = f32_vec_from_le_bytes(&state.logits)?;
    let selected_token = sample_argmax(&logits);
    let (_, top) = top_logprobs(&logits, TOP_K);
    if top.first().map(|score| score.id) != Some(selected_token) {
        return Err("selected token does not match top-logprob head".to_string());
    }
    Ok(NextTokenReport {
        checkpoint_tokens: parsed.header.token_count,
        checkpoint_fnv1a64: fnv1a64(&state.checkpoint),
        logits_fnv1a64: fnv1a64(&state.logits),
        selected_token,
        top_logprobs: top,
        layer_n_comp: state.layer_n_comp,
        layer_n_index_comp: state.layer_n_index_comp,
        frontier_projection: frontier_projection(parsed.header.token_count),
    })
}

fn frontier_projection(restored_tokens: u32) -> FrontierProjection {
    let restored_tokens = i32::try_from(restored_tokens).expect("restored token count fits i32");
    let mut policy = KvPolicyConfig::default();
    policy.continued_last_store_tokens = restored_tokens;
    let continued_step_tokens = continued_step(policy);
    let next_continued_tokens = next_continued_probe_tokens(restored_tokens, continued_step_tokens);

    let current_live_target = continued_store_target(policy, restored_tokens);
    let next_continued_target = continued_store_target(policy, next_continued_tokens);
    let mut already_stored_policy = policy;
    already_stored_policy.continued_last_store_tokens = next_continued_tokens;
    let already_stored_boundary_target =
        continued_store_target(already_stored_policy, next_continued_tokens);

    FrontierProjection {
        policy,
        continued_step_tokens,
        loaded_frontier: restored_tokens,
        current_live_target,
        next_continued_tokens,
        next_continued_target,
        already_stored_boundary_target,
    }
}

fn next_continued_probe_tokens(restored_tokens: i32, continued_step_tokens: i32) -> i32 {
    if continued_step_tokens <= 0 {
        return 0;
    }
    let mut probe = ((restored_tokens + continued_step_tokens - 1) / continued_step_tokens)
        * continued_step_tokens;
    if probe <= restored_tokens {
        probe += continued_step_tokens;
    }
    probe
}

fn continued_step(policy: KvPolicyConfig) -> i32 {
    let interval = policy.options.continued_interval_tokens;
    if !policy.enabled || interval <= 0 {
        return 0;
    }
    let align = policy.options.boundary_align_tokens;
    if align <= 0 {
        return interval;
    }
    let step = ((interval + align - 1) / align) * align;
    if step <= 0 {
        align
    } else {
        step
    }
}

fn f32_vec_from_le_bytes(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if bytes.len() != N_VOCAB as usize * 4 {
        return Err("logits byte length drift".to_string());
    }
    let mut out = Vec::with_capacity(N_VOCAB as usize);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
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
    writeln!(out, "  \"top_k\": {TOP_K},")?;
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
    write!(out, ", \"next_token\": ")?;
    write_next_token(out, &report.next_token)?;
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

fn write_next_token<W: Write>(out: &mut W, report: &NextTokenReport) -> io::Result<()> {
    write!(
        out,
        "{{\"cache_source\": \"restored-graph-payload\", \
         \"next_token_source\": \"restored-session-logits\", \"graph_restored\": true, \
         \"checkpoint_tokens\": {}, \"checkpoint_fnv1a64\": \"{:016x}\", \
         \"logits_fnv1a64\": \"{:016x}\", \"selected_token\": {}, \"top_logprobs\": [",
        report.checkpoint_tokens,
        report.checkpoint_fnv1a64,
        report.logits_fnv1a64,
        report.selected_token
    )?;
    for (idx, score) in report.top_logprobs.iter().enumerate() {
        if idx != 0 {
            write!(out, ", ")?;
        }
        write_score(out, score)?;
    }
    write!(out, "], \"post_restore_state\": {{\"checkpoint_valid\": true, \"mtp_draft_valid\": false, \"mtp_n_raw\": 0, \"layer_n_comp\": ")?;
    write_u32_array(out, &report.layer_n_comp)?;
    write!(out, ", \"layer_n_index_comp\": ")?;
    write_u32_array(out, &report.layer_n_index_comp)?;
    write!(out, "}}, \"frontier_projection\": ")?;
    write_frontier_projection(out, &report.frontier_projection)?;
    write!(out, "}}")
}

fn write_frontier_projection<W: Write>(
    out: &mut W,
    projection: &FrontierProjection,
) -> io::Result<()> {
    write!(
        out,
        "{{\"source\": \"restored-token-count\", \
         \"policy\": {{\"min_tokens\": {}, \"cold_max_tokens\": {}, \
         \"continued_interval_tokens\": {}, \"boundary_trim_tokens\": {}, \
         \"boundary_align_tokens\": {}, \"continued_step_tokens\": {}}}, \
         \"loaded_frontier\": {}, \
         \"current_live_skip\": {{\"live_tokens\": {}, \"target\": {}, \
         \"reason\": \"restored-position-unaligned\"}}, \
         \"next_continued_store\": {{\"frontier_before\": {}, \"live_tokens\": {}, \
         \"target\": {}, \"reason_name\": \"continued\", \"reason\": {}}}, \
         \"already_stored_boundary\": {{\"frontier_before\": {}, \"live_tokens\": {}, \
         \"target\": {}}}, \
         \"post_restore_shutdown\": {{\"reason_name\": \"shutdown\", \"reason\": {}, \
         \"tokens_source\": \"restored-session-position\"}}}}",
        projection.policy.options.min_tokens,
        projection.policy.options.cold_max_tokens,
        projection.policy.options.continued_interval_tokens,
        projection.policy.options.boundary_trim_tokens,
        projection.policy.options.boundary_align_tokens,
        projection.continued_step_tokens,
        projection.loaded_frontier,
        projection.loaded_frontier,
        projection.current_live_target,
        projection.loaded_frontier,
        projection.next_continued_tokens,
        projection.next_continued_target,
        REASON_CONTINUED,
        projection.next_continued_tokens,
        projection.next_continued_tokens,
        projection.already_stored_boundary_target,
        REASON_SHUTDOWN,
    )
}

fn write_score<W: Write>(out: &mut W, score: &TokenScore) -> io::Result<()> {
    write!(out, "{{\"id\": {}, \"logit\": ", score.id)?;
    write_json_f32(out, score.logit)?;
    write!(out, ", \"logprob\": ")?;
    write_json_f32(out, score.logprob)?;
    write!(out, "}}")
}

fn write_json_f32<W: Write>(out: &mut W, value: f32) -> io::Result<()> {
    if value.is_nan() {
        write!(out, "\"nan\"")
    } else if value == f32::INFINITY {
        write!(out, "\"inf\"")
    } else if value == f32::NEG_INFINITY {
        write!(out, "\"-inf\"")
    } else {
        write!(out, "{value:.9}")
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frontier_projection_reenables_next_boundary_after_seed_restore() {
        let projection = frontier_projection(550);
        assert_eq!(projection.continued_step_tokens, 10_240);
        assert_eq!(projection.loaded_frontier, 550);
        assert_eq!(projection.current_live_target, 0);
        assert_eq!(projection.next_continued_tokens, 10_240);
        assert_eq!(projection.next_continued_target, 10_240);
        assert_eq!(projection.already_stored_boundary_target, 0);
    }

    #[test]
    fn frontier_projection_reenables_next_boundary_after_continuation_restore() {
        let projection = frontier_projection(561);
        assert_eq!(projection.loaded_frontier, 561);
        assert_eq!(projection.current_live_target, 0);
        assert_eq!(projection.next_continued_tokens, 10_240);
        assert_eq!(projection.next_continued_target, 10_240);
        assert_eq!(projection.already_stored_boundary_target, 0);
    }
}
