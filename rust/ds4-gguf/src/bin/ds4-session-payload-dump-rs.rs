use ds4_gguf::session_payload::{
    append_full_payload, append_graph_payload_plan, append_prefix_to_first_comp,
    append_prefix_to_first_index, compress_ratio, default_cpu_runtime,
    default_graph_payload_runtime, default_header, graph_payload_plan, layer_attn_state_bytes,
    layer_index_state_bytes, read_graph_payload, sections, validate_payload_cpu,
    GraphPayloadFixture, GraphPayloadPlan, GraphPayloadRead, PayloadError, PayloadHeader,
    PayloadSections, GRAPH_PAYLOAD_FIXTURES, HEADER_BYTES, IO_CHUNK_BYTES, MAGIC, N_HEAD_DIM,
    N_INDEXER_HEAD_DIM, N_LAYER, N_SWA, N_VOCAB, U32_FIELDS, VERSION,
};
use std::fs;
use std::io::{self, Write};

const USAGE: &str = "usage: ds4-session-payload-dump-rs [--graph-plan|--graph-probe|--restore-header-plan|--restore-target-plan|--graph-file-probe <id:path>...]";

#[derive(Debug, Clone, PartialEq, Eq)]
enum DumpMode {
    Shape,
    GraphPlan,
    GraphProbe,
    RestoreHeaderPlan,
    RestoreTargetPlan,
    GraphFileProbe(Vec<GraphFileProbeInput>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GraphFileProbeInput {
    id: String,
    path: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut out = io::BufWriter::new(io::stdout());
    let mut mode = DumpMode::Shape;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--graph-plan" {
            if mode != DumpMode::Shape {
                return Err(USAGE.into());
            }
            mode = DumpMode::GraphPlan;
        } else if arg == "--graph-probe" {
            if mode != DumpMode::Shape {
                return Err(USAGE.into());
            }
            mode = DumpMode::GraphProbe;
        } else if arg == "--restore-header-plan" {
            if mode != DumpMode::Shape {
                return Err(USAGE.into());
            }
            mode = DumpMode::RestoreHeaderPlan;
        } else if arg == "--restore-target-plan" {
            if mode != DumpMode::Shape {
                return Err(USAGE.into());
            }
            mode = DumpMode::RestoreTargetPlan;
        } else if arg == "--graph-file-probe" {
            let spec = args
                .next()
                .ok_or("usage: --graph-file-probe requires <id:path>")?;
            let input = parse_graph_file_probe_input(&spec)?;
            match &mut mode {
                DumpMode::Shape => mode = DumpMode::GraphFileProbe(vec![input]),
                DumpMode::GraphFileProbe(inputs) => inputs.push(input),
                _ => return Err(USAGE.into()),
            }
        } else {
            return Err(USAGE.into());
        }
    }
    match mode {
        DumpMode::Shape => write_dump(&mut out)?,
        DumpMode::GraphPlan => write_graph_plan_dump(&mut out)?,
        DumpMode::GraphProbe => write_graph_probe_dump(&mut out)?,
        DumpMode::RestoreHeaderPlan => write_restore_header_plan_dump(&mut out)?,
        DumpMode::RestoreTargetPlan => write_restore_target_plan_dump(&mut out)?,
        DumpMode::GraphFileProbe(inputs) => write_graph_file_probe_dump(&mut out, &inputs)?,
    }
    Ok(())
}

fn parse_graph_file_probe_input(
    spec: &str,
) -> Result<GraphFileProbeInput, Box<dyn std::error::Error>> {
    let (id, path) = spec
        .split_once(':')
        .ok_or("usage: --graph-file-probe requires <id:path>")?;
    if id.is_empty() || path.is_empty() {
        return Err("usage: --graph-file-probe requires non-empty <id:path>".into());
    }
    Ok(GraphFileProbeInput {
        id: id.to_owned(),
        path: path.to_owned(),
    })
}

fn write_dump<W: Write>(out: &mut W) -> io::Result<()> {
    let probe_ctx = 16;
    let probe_tokens = 3;
    let header = default_header(probe_ctx, probe_tokens);
    let n_comp = [0_u32; N_LAYER];
    let n_index = [0_u32; N_LAYER];
    let size = sections(&header, &n_comp, &n_index);

    writeln!(out, "{{")?;
    writeln!(
        out,
        "  \"schema\": \"ds4.rust_session_payload_shape_structural.v1\","
    )?;
    writeln!(out, "  \"source\": \"rust-session-payload-no-model\",")?;
    writeln!(out, "  \"model\": \"no model is loaded for this oracle\",")?;
    writeln!(
        out,
        "  \"constants\": {{\"magic_u32\": {}, \"magic_field_hex\": \"{:08x}\", \
         \"magic_bytes_hex\": \"{}\", \"version\": {}, \"u32_fields\": {}, \
         \"header_bytes\": {}, \"io_chunk_bytes\": {}, \"u32_bytes\": 4, \
         \"float_bytes\": 4}},",
        MAGIC,
        MAGIC,
        hex_string(&MAGIC.to_le_bytes()),
        VERSION,
        U32_FIELDS,
        HEADER_BYTES,
        IO_CHUNK_BYTES
    )?;
    writeln!(
        out,
        "  \"fixed_model_layout\": {{\"n_layer\": {}, \"n_head_dim\": {}, \
         \"n_indexer_head_dim\": {}, \"n_vocab\": {}, \"n_swa\": {}}},",
        N_LAYER, N_HEAD_DIM, N_INDEXER_HEAD_DIM, N_VOCAB, N_SWA
    )?;

    write!(out, "  \"compress_ratio_by_layer\": [")?;
    for layer in 0..N_LAYER {
        if layer != 0 {
            write!(out, ", ")?;
        }
        write!(out, "{}", compress_ratio(layer))?;
    }
    writeln!(out, "],")?;

    write_header_fields(out)?;
    write_body_order(out)?;
    writeln!(out, "  \"size_case\": {{")?;
    writeln!(
        out,
        "    \"name\": \"cpu_probe_ctx16_tokens3_zero_compressed_rows\", \
         \"ctx_size\": {}, \"prefill_cap\": {}, \"raw_cap\": {}, \
         \"raw_window\": {}, \"comp_cap\": {}, \"token_count\": {}, \
         \"raw_live_rows\": {},",
        header.ctx_size,
        header.prefill_cap,
        header.raw_cap,
        header.raw_window,
        header.comp_cap,
        header.token_count,
        header.raw_live_rows
    )?;
    writeln!(
        out,
        "    \"section_bytes\": {{\"header\": {}, \"tokens\": {}, \
         \"logits\": {}, \"attn_counts\": {}, \"index_counts\": {}, \
         \"raw_rows\": {}, \"attn_compressed_rows\": {}, \"attn_state\": {}, \
         \"indexer_compressed_rows\": {}, \"indexer_state\": {}}},",
        HEADER_BYTES,
        size.token_bytes,
        size.logits_bytes,
        size.comp_count_bytes,
        size.index_count_bytes,
        size.raw_row_bytes,
        size.attn_comp_row_bytes,
        size.attn_state_bytes,
        size.index_comp_row_bytes,
        size.index_state_bytes
    )?;
    writeln!(out, "    \"payload_bytes\": {}", size.total())?;
    writeln!(out, "  }},")?;

    write_header_rejection_cases(out)?;
    write_body_probe_cases(out)?;
    writeln!(out, "}}")
}

fn write_header_fields<W: Write>(out: &mut W) -> io::Result<()> {
    const NAMES: [&str; U32_FIELDS] = [
        "magic",
        "version",
        "ctx_size",
        "prefill_cap",
        "raw_cap",
        "raw_window",
        "comp_cap",
        "token_count",
        "n_layer",
        "n_head_dim",
        "n_indexer_head_dim",
        "n_vocab",
        "raw_live_rows",
    ];
    writeln!(out, "  \"header_fields\": [")?;
    for (idx, name) in NAMES.iter().enumerate() {
        write!(out, "    {{\"index\": {}, \"name\": ", idx)?;
        write_json_string(out, name)?;
        write!(out, "}}")?;
        writeln!(out, "{}", if idx + 1 == NAMES.len() { "" } else { "," })?;
    }
    writeln!(out, "  ],")
}

fn write_body_order<W: Write>(out: &mut W) -> io::Result<()> {
    writeln!(out, "  \"body_order\": [")?;
    writeln!(
        out,
        "    {{\"name\": \"checkpoint_tokens\", \"bytes\": \"token_count * 4\"}},"
    )?;
    writeln!(
        out,
        "    {{\"name\": \"last_logits\", \"bytes\": \"n_vocab * 4\"}},"
    )?;
    writeln!(
        out,
        "    {{\"name\": \"attn_compressed_row_counts\", \"bytes\": \"n_layer * 4\"}},"
    )?;
    writeln!(
        out,
        "    {{\"name\": \"indexer_compressed_row_counts\", \"bytes\": \"n_layer * 4\"}},"
    )?;
    writeln!(
        out,
        "    {{\"name\": \"per_layer_raw_rows\", \"bytes\": \"raw_live_rows * n_head_dim * 4\"}},"
    )?;
    writeln!(
        out,
        "    {{\"name\": \"per_layer_attn_compressed_rows\", \"bytes\": \
         \"n_comp[layer] * n_head_dim * 4 when ratio != 0\"}},"
    )?;
    writeln!(
        out,
        "    {{\"name\": \"per_layer_attn_state_kv_then_score\", \"bytes\": \
         \"2 * coff * n_head_dim * coff * ratio * 4 when ratio != 0\"}},"
    )?;
    writeln!(
        out,
        "    {{\"name\": \"per_layer_indexer_compressed_rows\", \"bytes\": \
         \"n_index_comp[layer] * n_indexer_head_dim * 4 when ratio == 4\"}},"
    )?;
    writeln!(
        out,
        "    {{\"name\": \"per_layer_indexer_state_kv_then_score\", \"bytes\": \
         \"2 * coff * n_indexer_head_dim * coff * ratio * 4 when ratio == 4\"}}"
    )?;
    writeln!(out, "  ],")
}

fn write_graph_plan_dump<W: Write>(out: &mut W) -> io::Result<()> {
    writeln!(out, "{{")?;
    writeln!(
        out,
        "  \"schema\": \"ds4.rust_graph_session_payload_plan.v1\","
    )?;
    writeln!(out, "  \"source\": \"rust-session-payload-plan-no-model\",")?;
    writeln!(out, "  \"scope\": \"graph-session-payload-layout\",")?;
    writeln!(
        out,
        "  \"env_policy\": \"default graph caps; env overrides are out of scope\","
    )?;
    writeln!(
        out,
        "  \"constants\": {{\"magic_u32\": {}, \"version\": {}, \"u32_fields\": {}, \
         \"header_bytes\": {}, \"io_chunk_bytes\": {}, \"n_layer\": {}, \
         \"n_head_dim\": {}, \"n_indexer_head_dim\": {}, \"n_vocab\": {}, \
         \"n_swa\": {}}},",
        MAGIC,
        VERSION,
        U32_FIELDS,
        HEADER_BYTES,
        IO_CHUNK_BYTES,
        N_LAYER,
        N_HEAD_DIM,
        N_INDEXER_HEAD_DIM,
        N_VOCAB,
        N_SWA
    )?;
    writeln!(
        out,
        "  \"body_order\": [\"header\", \"checkpoint_tokens\", \"last_logits\", \
         \"attn_compressed_row_counts\", \"indexer_compressed_row_counts\", \
         \"per_layer_raw_rows_logical_order\", \"per_layer_attn_compressed_rows\", \
         \"per_layer_attn_state_kv\", \"per_layer_attn_state_score\", \
         \"per_layer_indexer_compressed_rows\", \"per_layer_index_state_kv\", \
         \"per_layer_index_state_score\"],"
    )?;
    writeln!(out, "  \"cases\": [")?;
    for (idx, fixture) in GRAPH_PAYLOAD_FIXTURES.iter().enumerate() {
        if idx != 0 {
            writeln!(out, ",")?;
        }
        write_graph_plan_case(out, &graph_payload_plan(*fixture))?;
    }
    writeln!(out, "\n  ]")?;
    writeln!(out, "}}")
}

fn write_graph_plan_case<W: Write>(out: &mut W, plan: &GraphPayloadPlan) -> io::Result<()> {
    write!(out, "    {{\"name\": ")?;
    write_json_string(out, plan.fixture.name)?;
    writeln!(
        out,
        ", \"ctx_size\": {}, \"token_count\": {}, \"prefill_cap\": {}, \
         \"raw_cap\": {}, \"raw_window\": {}, \"comp_cap\": {}, \
         \"raw_live_rows\": {}, \"raw_first_pos\": {}, \"raw_last_pos\": {}, \
         \"raw_first_phys\": {}, \"raw_last_phys\": {},",
        plan.header.ctx_size,
        plan.header.token_count,
        plan.header.prefill_cap,
        plan.header.raw_cap,
        plan.header.raw_window,
        plan.header.comp_cap,
        plan.header.raw_live_rows,
        plan.raw_first_pos,
        plan.raw_last_pos,
        plan.raw_first_phys,
        plan.raw_last_phys
    )?;
    write!(out, "     \"section_bytes\": ")?;
    write_graph_section_bytes(out, plan.sections)?;
    writeln!(
        out,
        ",\n     \"payload_bytes\": {}, \"ratio4_rows\": {}, \"ratio128_rows\": {},",
        plan.payload_bytes, plan.ratio4_rows, plan.ratio128_rows
    )?;
    writeln!(out, "     \"layer_samples\": [")?;
    for (idx, layer) in [0_usize, 2, 3, 42].into_iter().enumerate() {
        if idx != 0 {
            writeln!(out, ",")?;
        }
        write_layer_sample(out, plan, layer)?;
    }
    writeln!(out, "\n     ]}}")
}

fn write_graph_section_bytes<W: Write>(out: &mut W, sections: PayloadSections) -> io::Result<()> {
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

fn write_layer_sample<W: Write>(
    out: &mut W,
    plan: &GraphPayloadPlan,
    layer: usize,
) -> io::Result<()> {
    let ratio = compress_ratio(layer);
    let raw_row_bytes = u64::from(plan.header.raw_live_rows) * u64::from(N_HEAD_DIM) * 4;
    let attn_compressed_row_bytes = if ratio == 0 {
        0
    } else {
        u64::from(plan.n_comp[layer]) * u64::from(N_HEAD_DIM) * 4
    };
    let attn_state_bytes = if ratio == 0 {
        0
    } else {
        2 * ds4_gguf::session_payload::layer_attn_state_bytes(ratio)
    };
    let indexer_compressed_row_bytes = if ratio == 4 {
        u64::from(plan.n_index_comp[layer]) * u64::from(N_INDEXER_HEAD_DIM) * 4
    } else {
        0
    };
    let indexer_state_bytes = if ratio == 4 {
        2 * ds4_gguf::session_payload::layer_index_state_bytes(ratio)
    } else {
        0
    };
    write!(
        out,
        "    {{\"layer\": {}, \"ratio\": {}, \"n_comp\": {}, \"n_index_comp\": {}, \
         \"raw_first_phys\": {}, \"raw_last_phys\": {}, \"raw_row_bytes\": {}, \
         \"attn_compressed_row_bytes\": {}, \"attn_state_bytes\": {}, \
         \"indexer_compressed_row_bytes\": {}, \"indexer_state_bytes\": {}}}",
        layer,
        ratio,
        plan.n_comp[layer],
        plan.n_index_comp[layer],
        plan.raw_first_phys,
        plan.raw_last_phys,
        raw_row_bytes,
        attn_compressed_row_bytes,
        attn_state_bytes,
        indexer_compressed_row_bytes,
        indexer_state_bytes
    )
}

fn write_graph_probe_dump<W: Write>(out: &mut W) -> io::Result<()> {
    let runtime = default_graph_payload_runtime(32_768);
    writeln!(out, "{{")?;
    writeln!(
        out,
        "  \"schema\": \"ds4.rust_graph_session_payload_rw.v1\","
    )?;
    writeln!(
        out,
        "  \"source\": \"rust-graph-session-payload-rw-no-model\","
    )?;
    writeln!(out, "  \"scope\": \"graph-session-payload-read-write\",")?;
    writeln!(
        out,
        "  \"runtime\": {{\"ctx_size\": {}, \"prefill_cap\": {}, \
         \"raw_cap\": {}, \"raw_window\": {}, \"comp_cap\": {}}},",
        runtime.ctx_size,
        runtime.prefill_cap,
        runtime.raw_cap,
        runtime.raw_window,
        runtime.comp_cap
    )?;
    writeln!(out, "  \"cases\": [")?;
    let cases = graph_probe_cases();
    for (idx, case) in cases.iter().enumerate() {
        if idx != 0 {
            writeln!(out, ",")?;
        }
        write_graph_probe_case(out, case, runtime)?;
    }
    writeln!(out, "\n  ]")?;
    writeln!(out, "}}")
}

struct GraphProbeCase {
    name: &'static str,
    build: &'static str,
    bytes: Vec<u8>,
}

fn graph_probe_cases() -> Vec<GraphProbeCase> {
    let short = graph_payload_plan(GRAPH_PAYLOAD_FIXTURES[0]);
    let wrap = graph_payload_plan(GRAPH_PAYLOAD_FIXTURES[3]);
    let mut cases = Vec::new();

    let mut bytes = Vec::new();
    append_graph_payload_plan(&mut bytes, &short);
    cases.push(GraphProbeCase {
        name: "valid_short_graph_payload",
        build: "graph plan short full zero body",
        bytes,
    });

    let mut bytes = Vec::new();
    append_graph_payload_plan(&mut bytes, &wrap);
    cases.push(GraphProbeCase {
        name: "valid_raw_wrap_graph_payload",
        build: "graph plan raw-ring wrap full zero body",
        bytes,
    });

    let mut bytes = Vec::new();
    append_graph_payload_plan(&mut bytes, &short);
    bytes.extend_from_slice(&0_u32.to_le_bytes());
    cases.push(GraphProbeCase {
        name: "trailing_payload_bytes",
        build: "valid graph body plus 4 trailing bytes",
        bytes,
    });

    let mut bytes = Vec::new();
    append_graph_payload_plan(&mut bytes, &short);
    bytes.pop();
    cases.push(GraphProbeCase {
        name: "truncated_tensor_body",
        build: "valid graph body minus 1 byte",
        bytes,
    });

    let mut bytes = Vec::new();
    append_prefix_to_first_comp(&mut bytes, &short.header, short.header.comp_cap + 1);
    cases.push(GraphProbeCase {
        name: "n_comp_over_cap",
        build: "graph header tokens logits first n_comp",
        bytes,
    });

    let mut bytes = Vec::new();
    append_prefix_to_first_index(
        &mut bytes,
        &short.header,
        &short.n_comp,
        short.header.comp_cap + 1,
    );
    cases.push(GraphProbeCase {
        name: "n_index_comp_over_cap",
        build: "graph header tokens logits n_comp_table first n_index_comp",
        bytes,
    });

    let mut header = short.header.clone();
    header.raw_live_rows += 1;
    cases.push(header_only_case(
        "raw_live_rows_not_expected",
        "graph header raw_live_rows exceeds expected live rows",
        &header,
    ));

    let mut header = short.header.clone();
    header.ctx_size += 1;
    cases.push(header_only_case(
        "ctx_too_large",
        "graph header ctx_size exceeds runtime context",
        &header,
    ));

    let mut header = short.header.clone();
    header.n_layer = N_LAYER as u32 + 1;
    cases.push(header_only_case(
        "layer_count_mismatch",
        "graph header fixed layer count mismatch",
        &header,
    ));

    let mut header = short.header.clone();
    header.prefill_cap += 1;
    cases.push(header_only_case(
        "prefill_cap_mismatch",
        "graph header prefill cap mismatch",
        &header,
    ));

    let mut header = short.header;
    header.comp_cap += 1;
    cases.push(header_only_case(
        "comp_cap_too_large",
        "graph header comp cap exceeds runtime graph comp cap",
        &header,
    ));

    cases
}

fn header_only_case(
    name: &'static str,
    build: &'static str,
    header: &PayloadHeader,
) -> GraphProbeCase {
    GraphProbeCase {
        name,
        build,
        bytes: header.to_bytes().to_vec(),
    }
}

fn write_graph_probe_case<W: Write>(
    out: &mut W,
    case: &GraphProbeCase,
    runtime: ds4_gguf::session_payload::GraphPayloadRuntime,
) -> io::Result<()> {
    let result = read_graph_payload(&case.bytes, runtime);
    let (ok, code, error) = result_fields_graph(result.as_ref().map(|_| ()), result.as_ref().err());
    write!(out, "    {{\"name\": ")?;
    write_json_string(out, case.name)?;
    write!(out, ", \"build\": ")?;
    write_json_string(out, case.build)?;
    write!(
        out,
        ", \"payload_bytes\": {}, \"fnv1a64\": ",
        case.bytes.len()
    )?;
    write_json_string(out, &fnv1a64_hex(&case.bytes))?;
    write!(out, ", \"ok\": {}, \"code\": ", ok)?;
    write_json_string(out, code)?;
    write!(out, ", \"error\": ")?;
    write_json_string(out, error)?;
    if let Ok(parsed) = result {
        write!(out, ", \"parsed\": ")?;
        write_graph_probe_parsed(out, &parsed)?;
    }
    write!(out, "}}")
}

fn result_fields_graph(
    result: Result<(), &PayloadError>,
    err: Option<&PayloadError>,
) -> (&'static str, &'static str, &'static str) {
    match result {
        Ok(()) => ("true", "ok", ""),
        Err(_) => {
            let err = err.expect("graph payload error is present");
            ("false", err.code(), err.c_error())
        }
    }
}

fn write_graph_probe_parsed<W: Write>(out: &mut W, parsed: &GraphPayloadRead) -> io::Result<()> {
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
    write_graph_section_bytes(out, parsed.sections)?;
    write!(out, "}}")
}

fn write_graph_file_probe_dump<W: Write>(
    out: &mut W,
    inputs: &[GraphFileProbeInput],
) -> io::Result<()> {
    writeln!(out, "{{")?;
    writeln!(
        out,
        "  \"schema\": \"ds4.rust_graph_payload_file_probe.v1\","
    )?;
    writeln!(out, "  \"source\": \"rust-graph-payload-file-probe\",")?;
    writeln!(
        out,
        "  \"runtime\": {{\"ctx\": 32768, \"kind\": \"default-graph\"}},"
    )?;
    writeln!(out, "  \"cases\": [")?;
    let runtime = default_graph_payload_runtime(32_768);
    for (idx, input) in inputs.iter().enumerate() {
        if idx != 0 {
            writeln!(out, ",")?;
        }
        write_graph_file_probe_case(out, input, runtime)?;
    }
    writeln!(out, "\n  ]")?;
    writeln!(out, "}}")
}

fn write_graph_file_probe_case<W: Write>(
    out: &mut W,
    input: &GraphFileProbeInput,
    runtime: ds4_gguf::session_payload::GraphPayloadRuntime,
) -> io::Result<()> {
    let bytes = fs::read(&input.path)?;
    let result = read_graph_payload(&bytes, runtime);
    let (ok, code, error) = result_fields_graph(result.as_ref().map(|_| ()), result.as_ref().err());
    write!(out, "    {{\"id\": ")?;
    write_json_string(out, &input.id)?;
    write!(out, ", \"path\": ")?;
    write_json_string(out, &input.path)?;
    write!(out, ", \"payload_bytes\": {}, \"fnv1a64\": ", bytes.len())?;
    write_json_string(out, &fnv1a64_hex(&bytes))?;
    write!(out, ", \"ok\": {}, \"code\": ", ok)?;
    write_json_string(out, code)?;
    write!(out, ", \"error\": ")?;
    write_json_string(out, error)?;
    if let Ok(parsed) = result {
        write!(out, ", \"parsed\": ")?;
        write_graph_probe_parsed(out, &parsed)?;
    }
    write!(out, "}}")
}

fn fnv1a64_hex(bytes: &[u8]) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

struct RestoreHeaderCase {
    id: &'static str,
    kind: &'static str,
    prompt_case: &'static str,
    token_count: u32,
}

const RESTORE_HEADER_CASES: &[RestoreHeaderCase] = &[
    RestoreHeaderCase {
        id: "disk_seed_payload",
        kind: "disk-payload",
        prompt_case: "seed",
        token_count: 550,
    },
    RestoreHeaderCase {
        id: "snapshot_seed",
        kind: "memory-snapshot",
        prompt_case: "seed",
        token_count: 550,
    },
    RestoreHeaderCase {
        id: "disk_continuation_payload",
        kind: "disk-payload",
        prompt_case: "continuation",
        token_count: 561,
    },
    RestoreHeaderCase {
        id: "snapshot_continuation",
        kind: "memory-snapshot",
        prompt_case: "continuation",
        token_count: 561,
    },
];

fn write_restore_header_plan_dump<W: Write>(out: &mut W) -> io::Result<()> {
    writeln!(out, "{{")?;
    writeln!(
        out,
        "  \"schema\": \"ds4.rust_restore_payload_header_plan.v1\","
    )?;
    writeln!(
        out,
        "  \"source\": \"rust-restore-payload-header-plan-no-raw-bodies\","
    )?;
    writeln!(
        out,
        "  \"oracle\": \"ds4-parity/baselines/kv/m7.8/current-c.json\","
    )?;
    writeln!(out, "  \"model_path\": \"/workspace/ds4/ds4flash.gguf\",")?;
    writeln!(
        out,
        "  \"model_sha256\": \"efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668\","
    )?;
    writeln!(out, "  \"raw_body_policy\": \"hash-only; raw restore bodies are not required for this header contract\",")?;
    writeln!(out, "  \"cases\": [")?;
    for (idx, case) in RESTORE_HEADER_CASES.iter().enumerate() {
        if idx != 0 {
            writeln!(out, ",")?;
        }
        write_restore_header_case(out, case)?;
    }
    writeln!(out, "\n  ]")?;
    writeln!(out, "}}")
}

fn write_restore_header_case<W: Write>(out: &mut W, case: &RestoreHeaderCase) -> io::Result<()> {
    let plan = graph_payload_plan(GraphPayloadFixture {
        name: case.id,
        ctx_size: 32_768,
        token_count: case.token_count,
    });
    write!(out, "    {{\"id\": ")?;
    write_json_string(out, case.id)?;
    write!(out, ", \"kind\": ")?;
    write_json_string(out, case.kind)?;
    write!(out, ", \"prompt_case\": ")?;
    write_json_string(out, case.prompt_case)?;
    write!(
        out,
        ", \"ctx\": {}, \"prompt_tokens\": {}, \"raw_payload_committed\": false, \
         \"header_prefix_hex\": ",
        plan.header.ctx_size, plan.header.token_count
    )?;
    write_json_string(out, &hex_string(&plan.header.to_bytes()))?;
    write!(
        out,
        ", \"payload_bytes\": {}, \"graph\": {{\"prefill_cap\": {}, \
         \"raw_cap\": {}, \"raw_window\": {}, \"comp_cap\": {}, \
         \"raw_live_rows\": {}, \"ratio4_rows\": {}, \"ratio128_rows\": {}}}}}",
        plan.payload_bytes,
        plan.header.prefill_cap,
        plan.header.raw_cap,
        plan.header.raw_window,
        plan.header.comp_cap,
        plan.header.raw_live_rows,
        plan.ratio4_rows,
        plan.ratio128_rows
    )
}

fn write_restore_target_plan_dump<W: Write>(out: &mut W) -> io::Result<()> {
    writeln!(out, "{{")?;
    writeln!(
        out,
        "  \"schema\": \"ds4.rust_graph_restore_target_plan.v1\","
    )?;
    writeln!(
        out,
        "  \"source\": \"rust-graph-restore-target-plan-no-tensor-writes\","
    )?;
    writeln!(
        out,
        "  \"oracle\": \"ds4-parity/baselines/kv/m7.8/current-c.json\","
    )?;
    writeln!(out, "  \"model_path\": \"/workspace/ds4/ds4flash.gguf\",")?;
    writeln!(
        out,
        "  \"model_sha256\": \"efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668\","
    )?;
    writeln!(
        out,
        "  \"restore_order_source\": \"ds4_session_load_payload graph path\","
    )?;
    writeln!(out, "  \"raw_body_policy\": \"hash-only; target mapping uses parsed restore metadata and does not read raw bodies\",")?;
    writeln!(out, "  \"cases\": [")?;
    for (idx, case) in RESTORE_HEADER_CASES.iter().enumerate() {
        if idx != 0 {
            writeln!(out, ",")?;
        }
        write_restore_target_case(out, case)?;
    }
    writeln!(out, "\n  ]")?;
    writeln!(out, "}}")
}

fn write_restore_target_case<W: Write>(out: &mut W, case: &RestoreHeaderCase) -> io::Result<()> {
    let plan = graph_payload_plan(GraphPayloadFixture {
        name: case.id,
        ctx_size: 32_768,
        token_count: case.token_count,
    });
    let row_bytes = u64::from(N_HEAD_DIM) * 4;
    write!(out, "    {{\"id\": ")?;
    write_json_string(out, case.id)?;
    write!(out, ", \"kind\": ")?;
    write_json_string(out, case.kind)?;
    write!(out, ", \"prompt_case\": ")?;
    write_json_string(out, case.prompt_case)?;
    write!(
        out,
        ", \"ctx\": {}, \"prompt_tokens\": {}, \"payload_bytes\": {}, ",
        plan.header.ctx_size, plan.header.token_count, plan.payload_bytes
    )?;
    write!(
        out,
        "\"checkpoint\": {{\"target\": \"s->checkpoint\", \"source\": \"payload token u32 stream\", \
         \"tokens\": {}, \"bytes\": {}, \"commit\": \"replace-after-success\"}}, ",
        plan.header.token_count,
        u64::from(plan.header.token_count) * 4
    )?;
    write!(
        out,
        "\"logits\": {{\"target\": \"s->logits\", \"source\": \"payload logits f32 stream\", \
         \"bytes\": {}}}, ",
        u64::from(N_VOCAB) * 4
    )?;
    write!(
        out,
        "\"count_tables\": {{\"n_comp_source\": \"payload n_comp table\", \
         \"n_index_comp_source\": \"payload n_index_comp table\", \"bytes_each\": {}, \
         \"post_restore_targets\": [\"g->layer_n_comp\", \"g->layer_n_index_comp\"]}}, ",
        N_LAYER * 4
    )?;
    write!(
        out,
        "\"raw_ring\": {{\"target\": \"g->layer_raw_cache[layer]\", \
         \"source_order\": \"logical-position-order\", \"row_bytes\": {}, \
         \"rows_per_layer\": {}, \"first_pos\": {}, \"last_pos\": {}, \
         \"physical_rows\": ",
        row_bytes, plan.header.raw_live_rows, plan.raw_first_pos, plan.raw_last_pos
    )?;
    write_raw_physical_rows(out, &plan)?;
    write!(out, "}}, ")?;
    write!(
        out,
        "\"layer_summary\": {{\"layer_count\": {}, \"raw_layer_spans\": {}, \
         \"attn_comp_layers\": {}, \"ratio4_layers\": {}, \"ratio128_layers\": {}, \
         \"index_layers\": {}}}, ",
        N_LAYER,
        N_LAYER,
        count_layers_with_ratio(false),
        count_layers_ratio(4),
        count_layers_ratio(128),
        count_layers_ratio(4)
    )?;
    write!(out, "\"layers\": [")?;
    for layer in 0..N_LAYER {
        if layer != 0 {
            write!(out, ", ")?;
        }
        write_restore_target_layer(out, &plan, layer)?;
    }
    write!(out, "], ")?;
    write!(out, "\"post_restore_state\": {{\"checkpoint_valid\": true, \"mtp_draft_valid\": false, \"mtp_n_raw\": 0, \"layer_n_comp\": ")?;
    write_u32_array(out, &plan.n_comp)?;
    write!(out, ", \"layer_n_index_comp\": ")?;
    write_u32_array(out, &plan.n_index_comp)?;
    write!(out, "}}}}")
}

fn write_restore_target_layer<W: Write>(
    out: &mut W,
    plan: &GraphPayloadPlan,
    layer: usize,
) -> io::Result<()> {
    let ratio = compress_ratio(layer);
    let raw_bytes = u64::from(plan.header.raw_live_rows) * u64::from(N_HEAD_DIM) * 4;
    write!(
        out,
        "{{\"layer\": {}, \"ratio\": {}, \"raw\": {{\"target\": \"g->layer_raw_cache[layer]\", \
         \"bytes\": {}}}, \"attn\": ",
        layer, ratio, raw_bytes
    )?;
    if ratio == 0 {
        write!(
            out,
            "{{\"n_comp\": 0, \"comp_cache_bytes\": 0, \"state_kv_bytes\": 0, \
             \"state_score_bytes\": 0, \"targets\": []}}"
        )?;
    } else {
        let state_bytes = layer_attn_state_bytes(ratio);
        write!(
            out,
            "{{\"n_comp\": {}, \"comp_cache_bytes\": {}, \"state_kv_bytes\": {}, \
             \"state_score_bytes\": {}, \"targets\": [\"g->layer_attn_comp_cache[layer]\", \
             \"g->layer_attn_state_kv[layer]\", \"g->layer_attn_state_score[layer]\"]}}",
            plan.n_comp[layer],
            u64::from(plan.n_comp[layer]) * u64::from(N_HEAD_DIM) * 4,
            state_bytes,
            state_bytes
        )?;
    }
    write!(out, ", \"index\": ")?;
    if ratio == 4 {
        let state_bytes = layer_index_state_bytes(ratio);
        write!(
            out,
            "{{\"n_index_comp\": {}, \"comp_cache_bytes\": {}, \"state_kv_bytes\": {}, \
             \"state_score_bytes\": {}, \"targets\": [\"g->layer_index_comp_cache[layer]\", \
             \"g->layer_index_state_kv[layer]\", \"g->layer_index_state_score[layer]\"]}}",
            plan.n_index_comp[layer],
            u64::from(plan.n_index_comp[layer]) * u64::from(N_INDEXER_HEAD_DIM) * 4,
            state_bytes,
            state_bytes
        )?;
    } else {
        write!(
            out,
            "{{\"n_index_comp\": 0, \"comp_cache_bytes\": 0, \"state_kv_bytes\": 0, \
             \"state_score_bytes\": 0, \"targets\": []}}"
        )?;
    }
    write!(out, "}}")
}

fn write_raw_physical_rows<W: Write>(out: &mut W, plan: &GraphPayloadPlan) -> io::Result<()> {
    write!(out, "[")?;
    for idx in 0..plan.header.raw_live_rows {
        if idx != 0 {
            write!(out, ", ")?;
        }
        let pos = plan.raw_first_pos + idx;
        write!(out, "{}", pos % plan.header.raw_cap)?;
    }
    write!(out, "]")
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

fn count_layers_ratio(ratio: u32) -> usize {
    (0..N_LAYER)
        .filter(|&layer| compress_ratio(layer) == ratio)
        .count()
}

fn count_layers_with_ratio(include_zero: bool) -> usize {
    (0..N_LAYER)
        .filter(|&layer| include_zero || compress_ratio(layer) != 0)
        .count()
}

fn write_header_rejection_cases<W: Write>(out: &mut W) -> io::Result<()> {
    let probe_ctx = 16;
    let probe_tokens = 3;
    writeln!(out, "  \"header_rejection_cases\": [")?;
    let mut cases: Vec<(&str, PayloadHeader, usize)> = Vec::new();

    cases.push((
        "truncated_header",
        default_header(probe_ctx, probe_tokens),
        U32_FIELDS - 1,
    ));
    let mut h = default_header(probe_ctx, probe_tokens);
    h.magic = 0;
    cases.push(("bad_magic", h, U32_FIELDS));
    let mut h = default_header(probe_ctx, probe_tokens);
    h.version = 2;
    cases.push(("bad_version", h, U32_FIELDS));
    let mut h = default_header(probe_ctx, probe_tokens);
    h.ctx_size = probe_ctx + 1;
    cases.push(("ctx_too_large", h, U32_FIELDS));
    let mut h = default_header(probe_ctx, probe_tokens);
    h.token_count = probe_ctx;
    h.raw_live_rows = h.raw_window;
    cases.push(("tokens_equal_current_context", h, U32_FIELDS));
    let mut h = default_header(probe_ctx, probe_tokens);
    h.n_layer = N_LAYER as u32 + 1;
    cases.push(("layer_count_mismatch", h, U32_FIELDS));
    let mut h = default_header(probe_ctx, probe_tokens);
    h.n_head_dim = N_HEAD_DIM + 1;
    cases.push(("head_dim_mismatch", h, U32_FIELDS));
    let mut h = default_header(probe_ctx, probe_tokens);
    h.prefill_cap += 1;
    cases.push(("prefill_cap_mismatch", h, U32_FIELDS));
    let mut h = default_header(probe_ctx, probe_tokens);
    h.raw_window += 1;
    cases.push(("raw_window_mismatch", h, U32_FIELDS));
    let mut h = default_header(probe_ctx, probe_tokens);
    h.raw_cap = 0;
    cases.push(("zero_raw_cap", h, U32_FIELDS));
    let mut h = default_header(probe_ctx, probe_tokens);
    h.raw_live_rows += 1;
    cases.push(("raw_live_rows_not_expected", h, U32_FIELDS));
    let mut h = default_header(probe_ctx, probe_tokens);
    h.comp_cap += 1;
    cases.push(("comp_cap_too_large", h, U32_FIELDS));

    for (idx, (name, header, fields)) in cases.iter().enumerate() {
        if idx != 0 {
            writeln!(out, ",")?;
        }
        write_header_case(out, name, header, *fields)?;
    }
    writeln!(out, "\n  ],")
}

fn write_header_case<W: Write>(
    out: &mut W,
    name: &str,
    header: &PayloadHeader,
    fields: usize,
) -> io::Result<()> {
    let mut bytes = Vec::new();
    for field in header.fields().iter().take(fields) {
        bytes.extend_from_slice(&field.to_le_bytes());
    }
    let result = validate_payload_cpu(&bytes, default_cpu_runtime(16));
    let (ok, code, error) = result_fields(result);
    write!(out, "    {{\"name\": ")?;
    write_json_string(out, name)?;
    write!(
        out,
        ", \"fields_written\": {}, \"payload_bytes\": {}, \"header_hex\": ",
        fields,
        bytes.len()
    )?;
    write_hex_string(out, &bytes)?;
    write!(out, ", \"ok\": {}, \"code\": ", ok)?;
    write_json_string(out, code)?;
    write!(out, ", \"error\": ")?;
    write_json_string(out, error)?;
    write!(out, "}}")
}

fn write_body_probe_cases<W: Write>(out: &mut W) -> io::Result<()> {
    let header = default_header(16, 3);
    let n_comp = [0_u32; N_LAYER];
    let n_index = [0_u32; N_LAYER];
    writeln!(out, "  \"body_probe_cases\": [")?;

    let mut cases: Vec<(&str, &str, Vec<u8>)> = Vec::new();
    let mut bytes = Vec::new();
    append_full_payload(&mut bytes, &header, &n_comp, &n_index);
    cases.push(("valid_cpu_payload", "full zero body", bytes.clone()));
    bytes.extend_from_slice(&0_u32.to_le_bytes());
    cases.push((
        "trailing_payload_bytes",
        "valid body plus 4 trailing bytes",
        bytes,
    ));

    let mut bytes = Vec::new();
    append_full_payload(&mut bytes, &header, &n_comp, &n_index);
    bytes.pop();
    cases.push(("truncated_tensor_body", "valid body minus 1 byte", bytes));

    let mut bytes = Vec::new();
    append_prefix_to_first_comp(&mut bytes, &header, header.comp_cap + 1);
    cases.push((
        "n_comp_over_cap",
        "header tokens logits first n_comp",
        bytes,
    ));

    let mut bytes = Vec::new();
    append_prefix_to_first_index(&mut bytes, &header, &n_comp, header.comp_cap + 1);
    cases.push((
        "n_index_comp_over_cap",
        "header tokens logits n_comp_table first n_index_comp",
        bytes,
    ));

    for (idx, (name, build, bytes)) in cases.iter().enumerate() {
        if idx != 0 {
            writeln!(out, ",")?;
        }
        let result = validate_payload_cpu(bytes, default_cpu_runtime(16));
        let (ok, code, error) = result_fields(result);
        write!(out, "    {{\"name\": ")?;
        write_json_string(out, name)?;
        write!(out, ", \"build\": ")?;
        write_json_string(out, build)?;
        write!(
            out,
            ", \"payload_bytes\": {}, \"ok\": {}, \"code\": ",
            bytes.len(),
            ok
        )?;
        write_json_string(out, code)?;
        write!(out, ", \"error\": ")?;
        write_json_string(out, error)?;
        write!(out, "}}")?;
    }
    writeln!(out, "\n  ]")
}

fn result_fields(result: Result<(), PayloadError>) -> (&'static str, &'static str, &'static str) {
    match result {
        Ok(()) => ("true", "ok", ""),
        Err(err) => ("false", err.code(), err.c_error()),
    }
}

fn hex_string(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 15) as usize] as char);
    }
    out
}

fn write_hex_string<W: Write>(out: &mut W, bytes: &[u8]) -> io::Result<()> {
    write_json_string(out, &hex_string(bytes))
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
