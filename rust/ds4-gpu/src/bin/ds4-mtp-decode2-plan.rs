use ds4_gpu::mtp_decode2_plan::{MtpDecode2VerifierCase, MTP_DECODE2_ORCHESTRATION_CASES};
use std::io::{self, Write};

fn main() {
    let mut out = io::BufWriter::new(io::stdout());
    writeln!(out, "{{").unwrap();
    writeln!(out, "  \"schema\": \"ds4.rust_mtp_decode2_plan.v1\",").unwrap();
    writeln!(
        out,
        "  \"source\": \"rust-model-free-mtp-decode2-orchestration\","
    )
    .unwrap();
    writeln!(out, "  \"oracle\": \"metal_graph_verify_decode2_exact\",").unwrap();
    writeln!(out, "  \"cases\": [").unwrap();
    for (index, case) in MTP_DECODE2_ORCHESTRATION_CASES.iter().copied().enumerate() {
        if index != 0 {
            writeln!(out, ",").unwrap();
        }
        write_case(&mut out, case).unwrap();
    }
    writeln!(out).unwrap();
    writeln!(out, "  ]").unwrap();
    writeln!(out, "}}").unwrap();
}

fn write_case(out: &mut impl Write, case: MtpDecode2VerifierCase) -> io::Result<()> {
    writeln!(out, "    {{")?;
    write_str_field(out, "id", case.id)?;
    write_str_field(out, "source_function", case.source_function)?;
    write_str_field(out, "command_boundary", case.command_boundary)?;
    write_str_array_field(out, "target_tokens", case.target_tokens)?;
    write_str_field(out, "start_source", case.start_source)?;
    write_str_array_field(out, "decode_command_steps", case.decode_command_steps)?;
    write_str_array_field(out, "readbacks", case.readbacks)?;
    write_str_array_field(out, "frontier_ops", case.frontier_ops)?;
    write_str_field(out, "accept_condition", case.accept_condition)?;
    write_str_field(out, "accepted_suffix", case.accepted_suffix)?;
    write_str_field(out, "checkpoint_action", case.checkpoint_action)?;
    write_str_field(out, "logits_source", case.logits_source)?;
    write_str_field(out, "mtp_n_raw_keep", case.mtp_n_raw_keep)?;
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
