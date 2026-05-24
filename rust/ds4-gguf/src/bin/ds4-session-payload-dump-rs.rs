use ds4_gguf::session_payload::{
    append_full_payload, append_graph_payload_plan, append_prefix_to_first_comp,
    append_prefix_to_first_index, compress_ratio, default_cpu_runtime,
    default_graph_payload_runtime, default_header, graph_payload_plan, read_graph_payload,
    sections, validate_payload_cpu, GraphPayloadPlan, GraphPayloadRead, PayloadError,
    PayloadHeader, PayloadSections, GRAPH_PAYLOAD_FIXTURES, HEADER_BYTES, IO_CHUNK_BYTES, MAGIC,
    N_HEAD_DIM, N_INDEXER_HEAD_DIM, N_LAYER, N_SWA, N_VOCAB, U32_FIELDS, VERSION,
};
use std::io::{self, Write};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DumpMode {
    Shape,
    GraphPlan,
    GraphProbe,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut out = io::BufWriter::new(io::stdout());
    let mut mode = DumpMode::Shape;
    for arg in std::env::args().skip(1) {
        if arg == "--graph-plan" {
            if mode != DumpMode::Shape {
                return Err(
                    "usage: ds4-session-payload-dump-rs [--graph-plan|--graph-probe]".into(),
                );
            }
            mode = DumpMode::GraphPlan;
        } else if arg == "--graph-probe" {
            if mode != DumpMode::Shape {
                return Err(
                    "usage: ds4-session-payload-dump-rs [--graph-plan|--graph-probe]".into(),
                );
            }
            mode = DumpMode::GraphProbe;
        } else {
            return Err("usage: ds4-session-payload-dump-rs [--graph-plan|--graph-probe]".into());
        }
    }
    match mode {
        DumpMode::Shape => write_dump(&mut out)?,
        DumpMode::GraphPlan => write_graph_plan_dump(&mut out)?,
        DumpMode::GraphProbe => write_graph_probe_dump(&mut out)?,
    }
    Ok(())
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

fn fnv1a64_hex(bytes: &[u8]) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
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
