use ds4_gpu::mtp_draft_plan::{MtpDraftOrchestrationCase, MTP_DRAFT_ORCHESTRATION_CASES};
use std::io::{self, Write};

fn main() {
    let mut out = io::BufWriter::new(io::stdout());
    writeln!(out, "{{").unwrap();
    writeln!(out, "  \"schema\": \"ds4.rust_mtp_draft_plan.v1\",").unwrap();
    writeln!(
        out,
        "  \"source\": \"rust-model-free-mtp-draft-orchestration\","
    )
    .unwrap();
    writeln!(out, "  \"oracle\": \"metal_graph_eval_mtp_draft_from_hc\",").unwrap();
    writeln!(out, "  \"cases\": [").unwrap();
    for (index, case) in MTP_DRAFT_ORCHESTRATION_CASES.iter().copied().enumerate() {
        if index != 0 {
            writeln!(out, ",").unwrap();
        }
        write_case(&mut out, case).unwrap();
    }
    writeln!(out).unwrap();
    writeln!(out, "  ]").unwrap();
    writeln!(out, "}}").unwrap();
}

fn write_case(out: &mut impl Write, case: MtpDraftOrchestrationCase) -> io::Result<()> {
    writeln!(out, "    {{")?;
    write_str_field(out, "id", case.id)?;
    write_str_field(out, "source_function", case.source_function)?;
    write_str_field(out, "command_boundary", case.command_boundary)?;
    write_str_field(out, "prev_hc", case.prev_hc)?;
    write_str_field(out, "out_hc", case.out_hc)?;
    write_str_field(out, "token_source", case.token_source)?;
    write_str_field(out, "pos_source", case.pos_source)?;
    write_str_field(out, "logits_role", case.logits_role)?;
    write_str_field(out, "top_id_role", case.top_id_role)?;
    write_str_array_field(out, "command_steps", case.command_steps)?;
    write_str_array_field(out, "readbacks", case.readbacks)?;
    write_str_field(out, "mtp_n_raw_transition", case.mtp_n_raw_transition)?;
    write_str_array_field(out, "saved_state", case.saved_state)?;
    write_str_field(out, "failure_action", case.failure_action)?;
    write_str_field_last(out, "live_status", case.live_status)?;
    write!(out, "    }}")?;
    Ok(())
}

fn write_str_field(out: &mut impl Write, key: &str, value: &str) -> io::Result<()> {
    write!(out, "      \"{key}\": ")?;
    write_json_string(out, value)?;
    writeln!(out, ",")
}

fn write_str_field_last(out: &mut impl Write, key: &str, value: &str) -> io::Result<()> {
    write!(out, "      \"{key}\": ")?;
    write_json_string(out, value)?;
    writeln!(out)
}

fn write_str_array_field(out: &mut impl Write, key: &str, values: &[&str]) -> io::Result<()> {
    write!(out, "      \"{key}\": [")?;
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            write!(out, ", ")?;
        }
        write_json_string(out, value)?;
    }
    writeln!(out, "],")
}

fn write_json_string(out: &mut impl Write, value: &str) -> io::Result<()> {
    write!(out, "\"")?;
    for ch in value.chars() {
        match ch {
            '"' => write!(out, "\\\"")?,
            '\\' => write!(out, "\\\\")?,
            '\n' => write!(out, "\\n")?,
            '\r' => write!(out, "\\r")?,
            '\t' => write!(out, "\\t")?,
            _ => write!(out, "{ch}")?,
        }
    }
    write!(out, "\"")
}
