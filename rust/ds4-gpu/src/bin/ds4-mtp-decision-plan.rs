use ds4_gpu::mtp_plan::{plan_scenario, MtpCount, MtpDecisionPlan, MTP_SCENARIOS};
use std::io::{self, Write};

fn main() {
    let mut out = io::BufWriter::new(io::stdout());
    writeln!(out, "{{").unwrap();
    writeln!(out, "  \"schema\": \"ds4.rust_mtp_decision_plan.v1\",").unwrap();
    writeln!(
        out,
        "  \"oracle_path\": \"ds4-parity/baselines/graph/m10.8a/mtp-state-machine-contract.json\","
    )
    .unwrap();
    writeln!(
        out,
        "  \"source\": \"rust-model-free-mtp-decision-planner\","
    )
    .unwrap();
    writeln!(out, "  \"cases\": [").unwrap();
    for (index, scenario) in MTP_SCENARIOS.iter().copied().enumerate() {
        if index != 0 {
            writeln!(out, ",").unwrap();
        }
        write_case(&mut out, plan_scenario(scenario)).unwrap();
    }
    writeln!(out).unwrap();
    writeln!(out, "  ]").unwrap();
    writeln!(out, "}}").unwrap();
}

fn write_case(out: &mut impl Write, plan: MtpDecisionPlan) -> io::Result<()> {
    writeln!(out, "    {{")?;
    write_str_field(out, "id", plan.id, true)?;
    write_str_field(out, "path", plan.path, true)?;
    write_str_array_field(out, "frontier_ops", plan.frontier_ops, true)?;
    write_count_field(out, "accepted_suffix", plan.accepted_suffix, true)?;
    write_str_field(out, "checkpoint_action", plan.checkpoint_action, true)?;
    write_str_field(out, "logits_source", plan.logits_source, true)?;
    write_count_field(out, "mtp_n_raw_keep", plan.mtp_n_raw_keep, true)?;
    write_str_field(out, "fallback", plan.fallback, true)?;
    writeln!(
        out,
        "      \"fail_closed\": {}",
        if plan.fail_closed { "true" } else { "false" }
    )?;
    write!(out, "    }}")?;
    Ok(())
}

fn write_str_field(out: &mut impl Write, key: &str, value: &str, comma: bool) -> io::Result<()> {
    write!(out, "      \"{key}\": ")?;
    write_json_string(out, value)?;
    if comma {
        writeln!(out, ",")
    } else {
        writeln!(out)
    }
}

fn write_count_field(
    out: &mut impl Write,
    key: &str,
    value: MtpCount,
    comma: bool,
) -> io::Result<()> {
    write!(out, "      \"{key}\": ")?;
    match value.exact() {
        Some(number) => write!(out, "{number}")?,
        None => write_json_string(out, value.contract_value())?,
    }
    if comma {
        writeln!(out, ",")
    } else {
        writeln!(out)
    }
}

fn write_str_array_field(
    out: &mut impl Write,
    key: &str,
    values: &[&str],
    comma: bool,
) -> io::Result<()> {
    write!(out, "      \"{key}\": [")?;
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            write!(out, ", ")?;
        }
        write_json_string(out, value)?;
    }
    write!(out, "]")?;
    if comma {
        writeln!(out, ",")
    } else {
        writeln!(out)
    }
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
