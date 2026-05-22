use ds4_gguf::decode_policy::{policy_cases, run_policy_case, PolicyCase, PolicyResult};
use std::io::{self, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut out = io::BufWriter::new(io::stdout());
    write_dump(&mut out)?;
    Ok(())
}

fn write_dump<W: Write>(out: &mut W) -> io::Result<()> {
    let cases = policy_cases();
    writeln!(out, "{{")?;
    writeln!(out, "  \"schema\": \"ds4.rust_decode_policy_oracle.v1\",")?;
    writeln!(out, "  \"source\": \"rust-decode-stop-policy\",")?;
    writeln!(out, "  \"model\": \"no model is loaded for this oracle\",")?;
    writeln!(out, "  \"cases\": [")?;
    for (idx, case) in cases.iter().enumerate() {
        if idx != 0 {
            writeln!(out, ",")?;
        }
        write_case(out, case)?;
    }
    writeln!(out, "\n  ]")?;
    writeln!(out, "}}")
}

fn write_case<W: Write>(out: &mut W, case: &PolicyCase) -> io::Result<()> {
    let result = run_policy_case(case);
    write!(out, "    {{\"name\":")?;
    write_json_string(out, case.name)?;
    write!(out, ",\"source\":")?;
    write_json_string(out, case.source)?;
    write!(out, ",\"request\":{{\"surface\":")?;
    write_json_string(out, case.request.surface.as_str())?;
    write!(out, ",\"api\":")?;
    write_json_string(out, case.request.api.as_str())?;
    write!(out, ",\"kind\":")?;
    write_json_string(out, case.request.kind.as_str())?;
    write!(
        out,
        ",\"stream\":{},\"has_tools\":{},\"max_tokens\":{},\"stops\":[",
        case.request.stream, case.request.has_tools, case.request.max_tokens
    )?;
    for (idx, stop) in case.request.stops.iter().enumerate() {
        if idx != 0 {
            write!(out, ",")?;
        }
        write_json_string(out, stop)?;
    }
    write!(out, "]}},\"schedule\":[")?;
    for (idx, piece) in case.schedule.iter().enumerate() {
        if idx != 0 {
            write!(out, ",")?;
        }
        write!(
            out,
            "{{\"index\":{},\"eos\":{},\"text_hex\":",
            idx, piece.eos
        )?;
        write_hex_string(out, &piece.text)?;
        write!(out, "}}")?;
    }
    write!(out, "],\"result\":")?;
    write_result(out, &result)?;
    write!(out, "}}")
}

fn write_result<W: Write>(out: &mut W, result: &PolicyResult) -> io::Result<()> {
    write!(out, "{{\"finish_reason\":")?;
    write_json_string(out, result.finish_reason)?;
    write!(out, ",\"completion_tokens\":{}", result.completion_tokens)?;
    write!(out, ",\"raw_text_hex\":")?;
    write_hex_string(out, &result.raw_text)?;
    write!(out, ",\"visible_text_hex\":")?;
    write_hex_string(out, &result.visible_text)?;
    write!(out, ",\"reasoning_hex\":")?;
    write_hex_string(out, &result.reasoning)?;
    write!(out, ",\"streamed_text_hex\":")?;
    write_hex_string(out, &result.streamed_text)?;
    write!(
        out,
        ",\"session_invalidation_required\":{},\"transcript_eos_appended\":{}",
        result.session_invalidation_required, result.transcript_eos_appended
    )?;
    write!(
        out,
        ",\"stop_boundary\":{{\"pos\":{},\"len\":{}}}",
        result.stop_boundary.pos, result.stop_boundary.len
    )?;
    write!(
        out,
        ",\"tool_boundary\":{{\"saw_start\":{},\"saw_end\":{},\"tool_call_count\":{}}}",
        result.tool_boundary.saw_start,
        result.tool_boundary.saw_end,
        result.tool_boundary.tool_call_count
    )?;
    write!(out, ",\"api_finish\":{{\"openai_finish_reason\":")?;
    write_json_option(out, result.api_finish.openai_finish_reason)?;
    write!(out, ",\"anthropic_stop_reason\":")?;
    write_json_option(out, result.api_finish.anthropic_stop_reason)?;
    write!(out, ",\"responses_status\":")?;
    write_json_option(out, result.api_finish.responses_status)?;
    write!(out, ",\"responses_item_status\":")?;
    write_json_option(out, result.api_finish.responses_item_status)?;
    write!(out, ",\"responses_incomplete_reason\":")?;
    write_json_option(out, result.api_finish.responses_incomplete_reason)?;
    write!(out, "}},\"stream_steps\":[")?;
    for (idx, step) in result.stream_steps.iter().enumerate() {
        if idx != 0 {
            write!(out, ",")?;
        }
        write!(
            out,
            "{{\"step\":{},\"text_len\":{},\"stream_safe_len\":{},\"delta_hex\":",
            step.step, step.text_len, step.stream_safe_len
        )?;
        write_hex_string(out, &step.delta)?;
        write!(out, ",\"held_tail_hex\":")?;
        write_hex_string(out, &step.held_tail)?;
        write!(
            out,
            ",\"hit_stop\":{},\"stop_pos\":{},\"stop_len\":{}}}",
            step.hit_stop, step.stop_pos, step.stop_len
        )?;
    }
    write!(out, "]}}")
}

fn write_json_option<W: Write>(out: &mut W, value: Option<&str>) -> io::Result<()> {
    if let Some(value) = value {
        write_json_string(out, value)
    } else {
        write!(out, "null")
    }
}

fn write_hex_string<W: Write>(out: &mut W, bytes: &[u8]) -> io::Result<()> {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    write!(out, "\"")?;
    for byte in bytes {
        out.write_all(&[HEX[(byte >> 4) as usize], HEX[(byte & 0x0f) as usize]])?;
    }
    write!(out, "\"")
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
