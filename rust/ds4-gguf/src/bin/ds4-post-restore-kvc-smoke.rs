use ds4_gguf::kv_policy::{
    continued_store_target, read_kvc_file, sha1_bytes_hex, write_kvc_file, KvHeader,
    KvPolicyConfig, REASON_CONTINUED, REASON_SHUTDOWN,
};
use ds4_gguf::session_payload::{
    default_graph_payload_runtime, read_graph_payload, GraphPayloadRead, PayloadSections,
    HEADER_BYTES,
};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

const SCHEMA: &str = "ds4.rust_post_restore_kvc_smoke.v1";
const SOURCE: &str = "rust-post-restore-kvc-smoke";
const CTX_SIZE: u32 = 32_768;
const USAGE: &str =
    "usage: ds4-post-restore-kvc-smoke [--output-dir DIR] --case <id:payload:text>...";

fn main() {
    if let Err(err) = run() {
        eprintln!("ds4-post-restore-kvc-smoke: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    if let Some(dir) = &args.output_dir {
        fs::create_dir_all(dir)?;
    }
    let mut reports = Vec::with_capacity(args.cases.len());
    for case in &args.cases {
        reports.push(process_case(case, args.output_dir.as_deref())?);
    }
    write_report(&mut io::BufWriter::new(io::stdout()), &reports)?;
    Ok(())
}

struct Args {
    output_dir: Option<PathBuf>,
    cases: Vec<CaseInput>,
}

#[derive(Debug, Clone)]
struct CaseInput {
    id: String,
    payload_path: PathBuf,
    text_path: PathBuf,
}

fn parse_args() -> Result<Args, Box<dyn std::error::Error>> {
    let mut output_dir = None;
    let mut cases = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--output-dir" => {
                let value = args.next().ok_or("missing --output-dir value")?;
                output_dir = Some(PathBuf::from(value));
            }
            "--case" => {
                let spec = args.next().ok_or("missing --case value")?;
                cases.push(parse_case(&spec)?);
            }
            _ => return Err(USAGE.into()),
        }
    }
    if cases.is_empty() {
        return Err(USAGE.into());
    }
    Ok(Args { output_dir, cases })
}

fn parse_case(spec: &str) -> Result<CaseInput, Box<dyn std::error::Error>> {
    let parts: Vec<&str> = spec.splitn(3, ':').collect();
    if parts.len() != 3 || parts.iter().any(|part| part.is_empty()) {
        return Err("usage: --case requires <id:payload:text>".into());
    }
    Ok(CaseInput {
        id: parts[0].to_owned(),
        payload_path: PathBuf::from(parts[1]),
        text_path: PathBuf::from(parts[2]),
    })
}

struct CaseReport {
    input: CaseInput,
    parsed: GraphPayloadRead,
    text: TextReport,
    decisions: DecisionReport,
    kvc: KvcReport,
}

struct TextReport {
    bytes: u64,
    sha1: String,
    fnv1a64: u64,
}

struct DecisionReport {
    loaded_frontier: i32,
    current_live_skip: SkipDecision,
    next_continued_store: ContinuedDecision,
    already_stored_boundary: BoundaryDecision,
    shutdown_write_header: HeaderReport,
}

struct SkipDecision {
    live_tokens: i32,
    target: i32,
    reason: &'static str,
}

struct ContinuedDecision {
    frontier_before: i32,
    live_tokens: i32,
    target: i32,
    reason_name: &'static str,
    reason: u8,
}

struct BoundaryDecision {
    frontier_before: i32,
    live_tokens: i32,
    target: i32,
}

#[derive(Clone)]
struct HeaderReport {
    quant_bits: u8,
    reason_name: &'static str,
    reason: u8,
    ext_flags: u8,
    tokens: u32,
    hits: u32,
    ctx_size: u32,
    created_at: u64,
    last_used: u64,
    payload_bytes: u64,
}

struct KvcReport {
    file_name: String,
    output_path: Option<PathBuf>,
    file_size: u64,
    file_fnv1a64: u64,
    header: HeaderReport,
    text_bytes: u64,
    text_sha1: String,
    payload_bytes: u64,
    payload_fnv1a64: u64,
    trailer_bytes: u64,
    readback: KvcReadback,
}

struct KvcReadback {
    file_size: u64,
    header: HeaderReport,
    text_bytes: u64,
    text_sha1: String,
    payload_bytes: u64,
    payload_fnv1a64: u64,
    trailer_bytes: u64,
}

fn process_case(
    input: &CaseInput,
    output_dir: Option<&Path>,
) -> Result<CaseReport, Box<dyn std::error::Error>> {
    let payload = fs::read(&input.payload_path).map_err(|err| {
        format!(
            "{}: failed to read payload: {err}",
            input.payload_path.display()
        )
    })?;
    let text = fs::read(&input.text_path)
        .map_err(|err| format!("{}: failed to read text: {err}", input.text_path.display()))?;
    let runtime = default_graph_payload_runtime(CTX_SIZE);
    let parsed = read_graph_payload(&payload, runtime)
        .map_err(|err| format!("{}: graph payload parse failed: {err}", input.id))?;
    let text_report = TextReport {
        bytes: text.len() as u64,
        sha1: sha1_bytes_hex(&text),
        fnv1a64: fnv1a64(&text),
    };
    let decisions = decisions(&parsed, payload.len() as u64);
    let kvc = write_shutdown_kvc(
        &text,
        &payload,
        &decisions.shutdown_write_header,
        output_dir,
    )?;
    Ok(CaseReport {
        input: input.clone(),
        parsed,
        text: text_report,
        decisions,
        kvc,
    })
}

fn decisions(parsed: &GraphPayloadRead, payload_bytes: u64) -> DecisionReport {
    let restored_tokens =
        i32::try_from(parsed.header.token_count).expect("restored token count fits i32");
    let mut policy = KvPolicyConfig::default();
    policy.continued_last_store_tokens = restored_tokens;
    let step = continued_step(policy);
    let next_tokens = next_continued_probe_tokens(restored_tokens, step);
    let current_live_target = continued_store_target(policy, restored_tokens);
    let next_continued_target = continued_store_target(policy, next_tokens);
    let mut already_stored_policy = policy;
    already_stored_policy.continued_last_store_tokens = next_tokens;
    let already_stored_target = continued_store_target(already_stored_policy, next_tokens);
    DecisionReport {
        loaded_frontier: restored_tokens,
        current_live_skip: SkipDecision {
            live_tokens: restored_tokens,
            target: current_live_target,
            reason: "restored-position-unaligned",
        },
        next_continued_store: ContinuedDecision {
            frontier_before: restored_tokens,
            live_tokens: next_tokens,
            target: next_continued_target,
            reason_name: "continued",
            reason: REASON_CONTINUED,
        },
        already_stored_boundary: BoundaryDecision {
            frontier_before: next_tokens,
            live_tokens: next_tokens,
            target: already_stored_target,
        },
        shutdown_write_header: HeaderReport {
            quant_bits: 2,
            reason_name: "shutdown",
            reason: REASON_SHUTDOWN,
            ext_flags: 0,
            tokens: parsed.header.token_count,
            hits: 0,
            ctx_size: CTX_SIZE,
            created_at: 0,
            last_used: 0,
            payload_bytes,
        },
    }
}

fn write_shutdown_kvc(
    text: &[u8],
    payload: &[u8],
    header: &HeaderReport,
    output_dir: Option<&Path>,
) -> Result<KvcReport, Box<dyn std::error::Error>> {
    let kv_header = header.to_kv_header();
    let bytes = write_kvc_file(&kv_header, text, payload, &[])?;
    let file_name = format!("{}.kv", sha1_bytes_hex(text));
    let output_path = if let Some(dir) = output_dir {
        let path = dir.join(&file_name);
        fs::write(&path, &bytes)?;
        Some(path)
    } else {
        None
    };
    let parsed = read_kvc_file(&bytes)?;
    let readback = KvcReadback {
        file_size: parsed.file_size,
        header: HeaderReport::from_kv_header(&parsed.header),
        text_bytes: parsed.text.len() as u64,
        text_sha1: sha1_bytes_hex(&parsed.text),
        payload_bytes: parsed.payload.len() as u64,
        payload_fnv1a64: fnv1a64(&parsed.payload),
        trailer_bytes: parsed.trailer.len() as u64,
    };
    Ok(KvcReport {
        file_name,
        output_path,
        file_size: bytes.len() as u64,
        file_fnv1a64: fnv1a64(&bytes),
        header: header.clone(),
        text_bytes: text.len() as u64,
        text_sha1: sha1_bytes_hex(text),
        payload_bytes: payload.len() as u64,
        payload_fnv1a64: fnv1a64(payload),
        trailer_bytes: 0,
        readback,
    })
}

impl HeaderReport {
    fn to_kv_header(&self) -> KvHeader {
        KvHeader {
            quant_bits: self.quant_bits,
            reason: self.reason,
            ext_flags: self.ext_flags,
            tokens: self.tokens,
            hits: self.hits,
            ctx_size: self.ctx_size,
            created_at: self.created_at,
            last_used: self.last_used,
            payload_bytes: self.payload_bytes,
        }
    }

    fn from_kv_header(header: &KvHeader) -> Self {
        Self {
            quant_bits: header.quant_bits,
            reason_name: reason_name(header.reason),
            reason: header.reason,
            ext_flags: header.ext_flags,
            tokens: header.tokens,
            hits: header.hits,
            ctx_size: header.ctx_size,
            created_at: header.created_at,
            last_used: header.last_used,
            payload_bytes: header.payload_bytes,
        }
    }
}

fn reason_name(reason: u8) -> &'static str {
    match reason {
        REASON_CONTINUED => "continued",
        REASON_SHUTDOWN => "shutdown",
        _ => "unknown",
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

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn write_report<W: Write>(out: &mut W, reports: &[CaseReport]) -> io::Result<()> {
    writeln!(out, "{{")?;
    writeln!(out, "  \"schema\": \"{SCHEMA}\",")?;
    writeln!(out, "  \"source\": \"{SOURCE}\",")?;
    writeln!(out, "  \"runtime\": {{\"ctx\": {CTX_SIZE}, \"kind\": \"default-graph-payload\", \"kvc_writer\": \"ds4_gguf::kv_policy\"}},")?;
    writeln!(out, "  \"cases\": [")?;
    for (idx, report) in reports.iter().enumerate() {
        if idx != 0 {
            writeln!(out, ",")?;
        }
        write_case(out, report)?;
    }
    writeln!(out, "\n  ]")?;
    writeln!(out, "}}")
}

fn write_case<W: Write>(out: &mut W, report: &CaseReport) -> io::Result<()> {
    write!(out, "    {{\"id\": ")?;
    write_json_string(out, &report.input.id)?;
    write!(out, ", \"payload_path\": ")?;
    write_json_string(out, &report.input.payload_path.display().to_string())?;
    write!(out, ", \"text_path\": ")?;
    write_json_string(out, &report.input.text_path.display().to_string())?;
    write!(out, ", \"parsed\": ")?;
    write_parsed(out, &report.parsed)?;
    write!(out, ", \"text\": ")?;
    write_text(out, &report.text)?;
    write!(out, ", \"decisions\": ")?;
    write_decisions(out, &report.decisions)?;
    write!(out, ", \"kvc\": ")?;
    write_kvc(out, &report.kvc)?;
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

fn write_text<W: Write>(out: &mut W, text: &TextReport) -> io::Result<()> {
    write!(out, "{{\"bytes\": {}, \"sha1\": ", text.bytes)?;
    write_json_string(out, &text.sha1)?;
    write!(out, ", \"fnv1a64\": \"{:016x}\"}}", text.fnv1a64)
}

fn write_decisions<W: Write>(out: &mut W, decisions: &DecisionReport) -> io::Result<()> {
    write!(
        out,
        "{{\"loaded_frontier\": {}, \"current_live_skip\": {{\"live_tokens\": {}, \
         \"target\": {}, \"reason\": ",
        decisions.loaded_frontier,
        decisions.current_live_skip.live_tokens,
        decisions.current_live_skip.target
    )?;
    write_json_string(out, decisions.current_live_skip.reason)?;
    write!(
        out,
        "}}, \"next_continued_store\": {{\"frontier_before\": {}, \"live_tokens\": {}, \
         \"target\": {}, \"reason_name\": ",
        decisions.next_continued_store.frontier_before,
        decisions.next_continued_store.live_tokens,
        decisions.next_continued_store.target,
    )?;
    write_json_string(out, decisions.next_continued_store.reason_name)?;
    write!(
        out,
        ", \"reason\": {}}}, \"already_stored_boundary\": {{\"frontier_before\": {}, \
         \"live_tokens\": {}, \"target\": {}}}, \"shutdown_write_header\": ",
        decisions.next_continued_store.reason,
        decisions.already_stored_boundary.frontier_before,
        decisions.already_stored_boundary.live_tokens,
        decisions.already_stored_boundary.target,
    )?;
    write_header(out, &decisions.shutdown_write_header)?;
    write!(out, "}}")
}

fn write_kvc<W: Write>(out: &mut W, kvc: &KvcReport) -> io::Result<()> {
    write!(out, "{{\"file_name\": ")?;
    write_json_string(out, &kvc.file_name)?;
    write!(out, ", \"output_path\": ")?;
    if let Some(path) = &kvc.output_path {
        write_json_string(out, &path.display().to_string())?;
    } else {
        write!(out, "null")?;
    }
    write!(
        out,
        ", \"file_size\": {}, \"file_fnv1a64\": \"{:016x}\", \"header\": ",
        kvc.file_size, kvc.file_fnv1a64
    )?;
    write_header(out, &kvc.header)?;
    write!(out, ", \"text_bytes\": {}, \"text_sha1\": ", kvc.text_bytes)?;
    write_json_string(out, &kvc.text_sha1)?;
    write!(
        out,
        ", \"payload_bytes\": {}, \"payload_fnv1a64\": \"{:016x}\", \"trailer_bytes\": {}, \"readback\": ",
        kvc.payload_bytes,
        kvc.payload_fnv1a64,
        kvc.trailer_bytes
    )?;
    write_readback(out, &kvc.readback)?;
    write!(out, "}}")
}

fn write_readback<W: Write>(out: &mut W, readback: &KvcReadback) -> io::Result<()> {
    write!(out, "{{\"file_size\": {}, \"header\": ", readback.file_size)?;
    write_header(out, &readback.header)?;
    write!(
        out,
        ", \"text_bytes\": {}, \"text_sha1\": ",
        readback.text_bytes
    )?;
    write_json_string(out, &readback.text_sha1)?;
    write!(
        out,
        ", \"payload_bytes\": {}, \"payload_fnv1a64\": \"{:016x}\", \"trailer_bytes\": {}}}",
        readback.payload_bytes, readback.payload_fnv1a64, readback.trailer_bytes
    )
}

fn write_header<W: Write>(out: &mut W, header: &HeaderReport) -> io::Result<()> {
    write!(
        out,
        "{{\"quant_bits\": {}, \"reason_name\": ",
        header.quant_bits
    )?;
    write_json_string(out, header.reason_name)?;
    write!(
        out,
        ", \"reason\": {}, \"ext_flags\": {}, \"tokens\": {}, \"hits\": {}, \
         \"ctx_size\": {}, \"created_at\": {}, \"last_used\": {}, \"payload_bytes\": {}}}",
        header.reason,
        header.ext_flags,
        header.tokens,
        header.hits,
        header.ctx_size,
        header.created_at,
        header.last_used,
        header.payload_bytes
    )
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
    fn decision_projection_matches_restored_seed_frontier() {
        let plan = ds4_gguf::session_payload::graph_payload_plan(
            ds4_gguf::session_payload::GraphPayloadFixture {
                name: "disk_seed_payload",
                ctx_size: CTX_SIZE,
                token_count: 550,
            },
        );
        let parsed = GraphPayloadRead {
            header: plan.header,
            sections: plan.sections,
            payload_bytes: plan.payload_bytes,
            raw_first_pos: plan.raw_first_pos,
            raw_last_pos: plan.raw_last_pos,
            raw_first_phys: plan.raw_first_phys,
            raw_last_phys: plan.raw_last_phys,
            n_comp: plan.n_comp,
            n_index_comp: plan.n_index_comp,
        };
        let decisions = decisions(&parsed, 31_526_948);
        assert_eq!(decisions.loaded_frontier, 550);
        assert_eq!(decisions.current_live_skip.target, 0);
        assert_eq!(decisions.next_continued_store.live_tokens, 10_240);
        assert_eq!(decisions.next_continued_store.target, 10_240);
        assert_eq!(decisions.already_stored_boundary.target, 0);
        assert_eq!(decisions.shutdown_write_header.reason, REASON_SHUTDOWN);
        assert_eq!(decisions.shutdown_write_header.payload_bytes, 31_526_948);
    }

    #[test]
    fn shutdown_kvc_roundtrip_keeps_payload_opaque() {
        let header = HeaderReport {
            quant_bits: 2,
            reason_name: "shutdown",
            reason: REASON_SHUTDOWN,
            ext_flags: 0,
            tokens: 561,
            hits: 0,
            ctx_size: CTX_SIZE,
            created_at: 0,
            last_used: 0,
            payload_bytes: 3,
        };
        let report = write_shutdown_kvc(b"cache-key", &[1, 2, 3], &header, None).unwrap();
        assert_eq!(
            report.file_name,
            "bee970f5df526ae2acd05b948c2569038e800ff1.kv"
        );
        assert_eq!(report.file_size, 48 + 4 + 9 + 3);
        assert_eq!(report.readback.header.tokens, 561);
        assert_eq!(report.readback.payload_fnv1a64, fnv1a64(&[1, 2, 3]));
        assert_eq!(report.readback.trailer_bytes, 0);
    }
}
