use ds4_gpu::mtp_plan::MtpCount;
use ds4_gpu::mtp_stream_plan::{MtpStreamOutcomePlan, MTP_STREAM_OUTCOME_CASES};
use std::io::{self, Write};

fn main() {
    let mut out = io::BufWriter::new(io::stdout());
    writeln!(out, "{{").unwrap();
    writeln!(out, "  \"schema\": \"ds4.rust_mtp_stream_plan.v1\",").unwrap();
    writeln!(
        out,
        "  \"oracle_path\": \"ds4-parity/baselines/graph/m10.8g1/mtp-stream-parity-contract.json\","
    )
    .unwrap();
    writeln!(
        out,
        "  \"source\": \"rust-model-free-mtp-stream-outcome-planner\","
    )
    .unwrap();
    writeln!(out, "  \"cases\": [").unwrap();
    for (index, case) in MTP_STREAM_OUTCOME_CASES.iter().copied().enumerate() {
        if index != 0 {
            writeln!(out, ",").unwrap();
        }
        write_case(&mut out, case).unwrap();
    }
    writeln!(out).unwrap();
    writeln!(out, "  ]").unwrap();
    writeln!(out, "}}").unwrap();
}

fn write_case(out: &mut impl Write, case: MtpStreamOutcomePlan) -> io::Result<()> {
    writeln!(out, "    {{")?;
    write_str_field(out, "id", case.id)?;
    write_str_field(out, "source_case", case.source_case)?;
    write_str_field(out, "path", case.path)?;
    write_str_array_field(out, "selected_subplans", case.selected_subplans)?;
    write_count_field(out, "accepted_suffix", case.accepted_suffix)?;
    write_str_field(out, "accepted_stream_delta", case.accepted_stream_delta)?;
    write_str_field(out, "checkpoint_delta", case.checkpoint_delta)?;
    write_str_field(out, "logits_source", case.logits_source)?;
    write_str_array_field(out, "frontier_ops", case.frontier_ops)?;
    write_count_field(out, "mtp_n_raw_keep", case.mtp_n_raw_keep)?;
    write_str_field(out, "cache_kvc_visibility", case.cache_kvc_visibility)?;
    write_str_field(out, "fallback", case.fallback)?;
    write_str_field(out, "error", case.error)?;
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

fn write_count_field(out: &mut impl Write, key: &str, value: MtpCount) -> io::Result<()> {
    write!(out, "      \"{key}\": ")?;
    match value.exact() {
        Some(number) => write!(out, "{number}")?,
        None => write_json_string(out, value.contract_value())?,
    }
    writeln!(out, ",")
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
