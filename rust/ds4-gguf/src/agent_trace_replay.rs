use std::io::{self, Write};

const TOOL_DSML: &str = "<｜DSML｜tool_calls>\n\
<｜DSML｜invoke name=\"list\">\n\
<｜DSML｜parameter name=\"path\" string=\"true\">.</｜DSML｜parameter>\n\
</｜DSML｜invoke>\n\
</｜DSML｜tool_calls>";

const TOOL_OUTPUT: &str = "Tool result for list path=\".\":\nREADME.md\nds4_agent.c\n";
const FINAL_ANSWER: &str = "README.md and ds4_agent.c are visible.";

pub fn write_agent_trace_replay_oracle<W: Write>(out: &mut W) -> io::Result<()> {
    writeln!(out, "{{")?;
    writeln!(out, "  \"schema\": \"ds4.agent_trace_replay_oracle.v1\",")?;
    writeln!(out, "  \"milestone\": \"M11.1\",")?;
    writeln!(out, "  \"source\": \"current-c-agent-trace-oracle\",")?;
    writeln!(out, "  \"normalization\": {{")?;
    writeln!(out, "    \"path_root\": \"<WORKSPACE>\",")?;
    writeln!(
        out,
        "    \"fields\": [\"timestamp\", \"cwd\", \"duration_ms\", \"pid\", \"session_sha\"],"
    )?;
    writeln!(out, "    \"rules\": [")?;
    writeln!(
        out,
        "      \"absolute workspace paths are replaced with <WORKSPACE>\","
    )?;
    writeln!(
        out,
        "      \"command durations and process ids are omitted\","
    )?;
    writeln!(
        out,
        "      \"saved session digests are normalized to <SESSION:...>\""
    )?;
    writeln!(out, "    ]")?;
    writeln!(out, "  }},")?;
    writeln!(out, "  \"cases\": [")?;
    write_single_tool_round(out)?;
    writeln!(out, ",")?;
    write_session_switching_commands(out)?;
    writeln!(out, "\n  ]")?;
    writeln!(out, "}}")?;
    Ok(())
}

fn write_single_tool_round<W: Write>(out: &mut W) -> io::Result<()> {
    writeln!(out, "    {{")?;
    writeln!(out, "      \"id\": \"single_tool_round\",")?;
    writeln!(
        out,
        "      \"description\": \"one user turn, one DSML tool call, deterministic tool result, final answer\","
    )?;
    writeln!(
        out,
        "      \"fixture\": {{\"kind\": \"scripted_model\", \"cwd\": \"<WORKSPACE>\", \"ctx_size\": 8192, \"think_mode\": \"none\"}},"
    )?;
    writeln!(out, "      \"inputs\": [")?;
    writeln!(
        out,
        "        {{\"type\": \"user\", \"text\": \"List the root files, then answer with the two names you used.\"}}"
    )?;
    writeln!(out, "      ],")?;
    writeln!(out, "      \"model_events\": [")?;
    write!(out, "        {{\"round\": 0, \"text\": ")?;
    write_json_string(out, TOOL_DSML)?;
    writeln!(out, "}},")?;
    write!(out, "        {{\"round\": 1, \"text\": ")?;
    write_json_string(out, FINAL_ANSWER)?;
    writeln!(out, "}}")?;
    writeln!(out, "      ],")?;
    writeln!(out, "      \"tool_stubs\": [")?;
    write!(
        out,
        "        {{\"round\": 0, \"name\": \"list\", \"args\": [{{\"name\": \"path\", \"value\": \".\", \"is_string\": true}}], \"output\": "
    )?;
    write_json_string(out, TOOL_OUTPUT)?;
    writeln!(out, "}}")?;
    writeln!(out, "      ],")?;
    writeln!(out, "      \"expected\": {{")?;
    writeln!(
        out,
        "        \"tool_sequence\": [{{\"round\": 0, \"name\": \"list\", \"args\": [{{\"name\": \"path\", \"value\": \".\", \"is_string\": true}}]}}],"
    )?;
    writeln!(
        out,
        "        \"transcript_roles\": [\"system\", \"user\", \"assistant\", \"tool\", \"assistant\"],"
    )?;
    writeln!(out, "        \"session_operations\": [],")?;
    write!(out, "        \"final_visible_output\": ")?;
    write_json_string(out, FINAL_ANSWER)?;
    writeln!(out)?;
    writeln!(out, "      }}")?;
    write!(out, "    }}")?;
    Ok(())
}

fn write_session_switching_commands<W: Write>(out: &mut W) -> io::Result<()> {
    writeln!(out, "    {{")?;
    writeln!(out, "      \"id\": \"session_switching_commands\",")?;
    writeln!(
        out,
        "      \"description\": \"save, list, switch, history, and new-session control flow without live model sampling\","
    )?;
    writeln!(
        out,
        "      \"fixture\": {{\"kind\": \"session_commands\", \"cwd\": \"<WORKSPACE>\", \"ctx_size\": 8192, \"think_mode\": \"none\"}},"
    )?;
    writeln!(out, "      \"inputs\": [")?;
    writeln!(
        out,
        "        {{\"type\": \"user\", \"text\": \"Remember that alpha was inspected.\"}},"
    )?;
    writeln!(
        out,
        "        {{\"type\": \"command\", \"text\": \"/save\"}},"
    )?;
    writeln!(
        out,
        "        {{\"type\": \"command\", \"text\": \"/list\"}},"
    )?;
    writeln!(
        out,
        "        {{\"type\": \"command\", \"text\": \"/switch <SESSION:alpha>\"}},"
    )?;
    writeln!(
        out,
        "        {{\"type\": \"command\", \"text\": \"/history 2\"}},"
    )?;
    writeln!(out, "        {{\"type\": \"command\", \"text\": \"/new\"}}")?;
    writeln!(out, "      ],")?;
    writeln!(out, "      \"model_events\": [")?;
    writeln!(
        out,
        "        {{\"round\": 0, \"text\": \"Noted: alpha was inspected.\"}}"
    )?;
    writeln!(out, "      ],")?;
    writeln!(out, "      \"tool_stubs\": [],")?;
    writeln!(out, "      \"expected\": {{")?;
    writeln!(out, "        \"tool_sequence\": [],")?;
    writeln!(
        out,
        "        \"transcript_roles\": [\"system\", \"user\", \"assistant\"],"
    )?;
    writeln!(out, "        \"session_operations\": [")?;
    writeln!(
        out,
        "          {{\"command\": \"save\", \"session\": \"<SESSION:alpha>\", \"visible\": \"saved session <SESSION:alpha> (3 turns)\"}},"
    )?;
    writeln!(
        out,
        "          {{\"command\": \"list\", \"sessions\": [\"<SESSION:alpha>\"]}},"
    )?;
    writeln!(
        out,
        "          {{\"command\": \"switch\", \"session\": \"<SESSION:alpha>\", \"visible\": \"switched to <SESSION:alpha>\"}},"
    )?;
    writeln!(
        out,
        "          {{\"command\": \"history\", \"turns\": 2, \"visible\": \"user: Remember that alpha was inspected.\"}},"
    )?;
    writeln!(
        out,
        "          {{\"command\": \"new\", \"visible\": \"new session started from system prompt\"}}"
    )?;
    writeln!(out, "        ],")?;
    writeln!(
        out,
        "        \"final_visible_output\": \"new session started from system prompt\""
    )?;
    writeln!(out, "      }}")?;
    write!(out, "    }}")?;
    Ok(())
}

fn write_json_string<W: Write>(out: &mut W, value: &str) -> io::Result<()> {
    write!(out, "\"")?;
    for c in value.chars() {
        match c {
            '"' => write!(out, "\\\"")?,
            '\\' => write!(out, "\\\\")?,
            '\n' => write!(out, "\\n")?,
            '\r' => write!(out, "\\r")?,
            '\t' => write!(out, "\\t")?,
            c if c < ' ' => write!(out, "\\u{:04x}", c as u32)?,
            c => write!(out, "{c}")?,
        }
    }
    write!(out, "\"")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oracle_contains_replay_boundaries() {
        let mut bytes = Vec::new();
        write_agent_trace_replay_oracle(&mut bytes).expect("write oracle");
        let text = String::from_utf8(bytes).expect("utf8");
        assert!(text.contains("\"schema\": \"ds4.agent_trace_replay_oracle.v1\""));
        assert!(text.contains("\"id\": \"single_tool_round\""));
        assert!(text.contains("\"id\": \"session_switching_commands\""));
        assert!(text.contains("<SESSION:alpha>"));
    }

    #[test]
    fn oracle_normalizes_workspace_paths() {
        let mut bytes = Vec::new();
        write_agent_trace_replay_oracle(&mut bytes).expect("write oracle");
        let text = String::from_utf8(bytes).expect("utf8");
        assert!(text.contains("\"cwd\": \"<WORKSPACE>\""));
        assert!(!text.contains("/Users/"));
        assert!(!text.contains("/workspace/ds4"));
    }
}
