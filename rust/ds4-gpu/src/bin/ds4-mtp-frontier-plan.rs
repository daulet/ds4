use ds4_gpu::mtp_frontier_plan::{MtpFrontierMutationCase, MTP_FRONTIER_MUTATION_CASES};
use std::io::{self, Write};

fn main() {
    let mut out = io::BufWriter::new(io::stdout());
    writeln!(out, "{{").unwrap();
    writeln!(out, "  \"schema\": \"ds4.rust_mtp_frontier_plan.v1\",").unwrap();
    writeln!(
        out,
        "  \"source\": \"rust-model-free-mtp-frontier-orchestration\","
    )
    .unwrap();
    writeln!(
        out,
        "  \"oracle\": \"spec_frontier_snapshot_restore_prefix1\","
    )
    .unwrap();
    writeln!(out, "  \"cases\": [").unwrap();
    for (index, case) in MTP_FRONTIER_MUTATION_CASES.iter().copied().enumerate() {
        if index != 0 {
            writeln!(out, ",").unwrap();
        }
        write_case(&mut out, case).unwrap();
    }
    writeln!(out).unwrap();
    writeln!(out, "  ]").unwrap();
    writeln!(out, "}}").unwrap();
}

fn write_case(out: &mut impl Write, case: MtpFrontierMutationCase) -> io::Result<()> {
    writeln!(out, "    {{")?;
    write_str_field(out, "id", case.id)?;
    write_str_field(out, "source_function", case.source_function)?;
    write_str_field(out, "ratio_family", case.ratio_family)?;
    write_str_array_field(out, "saved_counters", case.saved_counters)?;
    write_str_array_field(out, "counter_updates", case.counter_updates)?;
    write_str_array_field(out, "tensor_copies", case.tensor_copies)?;
    write_str_field(out, "mtp_n_raw_action", case.mtp_n_raw_action)?;
    write_str_field(out, "invisible_rows_policy", case.invisible_rows_policy)?;
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
