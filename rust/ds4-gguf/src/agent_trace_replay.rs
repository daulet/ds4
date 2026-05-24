use crate::agent_dsml::AgentDsmlParser;
use crate::prompt::{render_chat_prompt_text, ChatMessage, ThinkMode, ToolArgument, ToolCall};
use std::io::{self, Write};

const TOOL_DSML: &str = "<｜DSML｜tool_calls>\n\
<｜DSML｜invoke name=\"list\">\n\
<｜DSML｜parameter name=\"path\" string=\"true\">.</｜DSML｜parameter>\n\
</｜DSML｜invoke>\n\
</｜DSML｜tool_calls>";

const TOOL_OUTPUT: &str = "Tool result for list path=\".\":\nREADME.md\nds4_agent.c\n";
const FINAL_ANSWER: &str = "README.md and ds4_agent.c are visible.";
const SESSION_MODEL_ANSWER: &str = "Noted: alpha was inspected.";
const SESSION_FINAL_COMMAND_OUTPUT: &str = "new session started from system prompt";

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

pub fn write_agent_rendered_context_replay<W: Write>(out: &mut W) -> io::Result<()> {
    writeln!(out, "{{")?;
    writeln!(
        out,
        "  \"schema\": \"ds4.agent_rendered_context_replay.v1\","
    )?;
    writeln!(out, "  \"milestone\": \"M11.2\",")?;
    writeln!(out, "  \"source\": \"rust-agent-rendered-context-replay\",")?;
    writeln!(
        out,
        "  \"oracle\": \"M11.1 current-C trace replay fixture plus Rust prompt/DSML rendering contracts\","
    )?;
    writeln!(out, "  \"cases\": [")?;
    write_rendered_context_case(
        out,
        "single_tool_round",
        &single_tool_messages(),
        &["system", "user", "assistant", "tool", "assistant"],
        FINAL_ANSWER,
    )?;
    writeln!(out, ",")?;
    write_rendered_context_case(
        out,
        "session_switching_commands",
        &session_command_messages(),
        &["system", "user", "assistant"],
        "Noted: alpha was inspected.",
    )?;
    writeln!(out, "\n  ]")?;
    writeln!(out, "}}")?;
    Ok(())
}

pub fn write_agent_deterministic_replay<W: Write>(out: &mut W) -> io::Result<()> {
    writeln!(out, "{{")?;
    writeln!(out, "  \"schema\": \"ds4.agent_deterministic_replay.v1\",")?;
    writeln!(out, "  \"milestone\": \"M11.3\",")?;
    writeln!(out, "  \"source\": \"rust-agent-deterministic-replay\",")?;
    writeln!(
        out,
        "  \"oracle\": \"M11.1 current-C trace replay fixture plus M11.2 rendered context artifact\","
    )?;
    writeln!(out, "  \"live_execution\": false,")?;
    writeln!(out, "  \"model_sampling\": false,")?;
    writeln!(out, "  \"cases\": [")?;
    write_deterministic_single_tool_round(out)?;
    writeln!(out, ",")?;
    write_deterministic_session_commands(out)?;
    writeln!(out, "\n  ]")?;
    writeln!(out, "}}")?;
    Ok(())
}

pub fn write_agent_loop_smoke<W: Write>(out: &mut W) -> io::Result<()> {
    writeln!(out, "{{")?;
    writeln!(out, "  \"schema\": \"ds4.agent_loop_smoke.v1\",")?;
    writeln!(out, "  \"milestone\": \"M11.4\",")?;
    writeln!(out, "  \"source\": \"rust-agent-loop-smoke\",")?;
    writeln!(
        out,
        "  \"oracle\": \"M11.1 trace fixture, M11.2 rendered context, and M11.3 deterministic replay\","
    )?;
    writeln!(out, "  \"model_sampling\": false,")?;
    writeln!(out, "  \"live_tool_execution\": false,")?;
    writeln!(
        out,
        "  \"manual_smoke\": \"deferred until model-backed Rust agent loop is enabled\","
    )?;
    writeln!(out, "  \"cases\": [")?;
    write_loop_single_tool_round(out)?;
    writeln!(out, ",")?;
    write_loop_session_commands(out)?;
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

fn write_rendered_context_case<W: Write>(
    out: &mut W,
    id: &str,
    messages: &[ChatMessage],
    roles: &[&str],
    final_visible_output: &str,
) -> io::Result<()> {
    let prompt = render_chat_prompt_text(messages, None, ThinkMode::None);
    let dsml_count = count_matches(&prompt, "<｜DSML｜tool_calls>");
    let tool_result_count = count_matches(&prompt, "<tool_result>");
    writeln!(out, "    {{")?;
    write!(out, "      \"id\": ")?;
    write_json_string(out, id)?;
    writeln!(out, ",")?;
    writeln!(out, "      \"replay_source\": \"M11.1\",")?;
    writeln!(out, "      \"think_mode\": \"none\",")?;
    writeln!(out, "      \"message_roles\": [")?;
    for (idx, role) in roles.iter().enumerate() {
        if idx != 0 {
            writeln!(out, ",")?;
        }
        write!(out, "        ")?;
        write_json_string(out, role)?;
    }
    writeln!(out, "\n      ],")?;
    writeln!(out, "      \"markers\": {{")?;
    writeln!(
        out,
        "        \"begin_sentence\": {},",
        count_matches(&prompt, "<｜begin▁of▁sentence｜>")
    )?;
    writeln!(
        out,
        "        \"user\": {},",
        count_matches(&prompt, "<｜User｜>")
    )?;
    writeln!(
        out,
        "        \"assistant\": {},",
        count_matches(&prompt, "<｜Assistant｜>")
    )?;
    writeln!(
        out,
        "        \"end_sentence\": {},",
        count_matches(&prompt, "<｜end▁of▁sentence｜>")
    )?;
    writeln!(out, "        \"tool_result\": {tool_result_count},")?;
    writeln!(out, "        \"dsml_tool_calls\": {dsml_count}")?;
    writeln!(out, "      }},")?;
    writeln!(
        out,
        "      \"raw_tool_dsml_preserved\": {},",
        if id == "single_tool_round" {
            dsml_count == 1 && prompt.contains(TOOL_DSML)
        } else {
            false
        }
    )?;
    writeln!(
        out,
        "      \"contains_final_visible_output\": {},",
        prompt.contains(final_visible_output)
    )?;
    write!(out, "      \"final_visible_output\": ")?;
    write_json_string(out, final_visible_output)?;
    writeln!(out, ",")?;
    write!(out, "      \"prompt_text\": ")?;
    write_json_string(out, &prompt)?;
    writeln!(out)?;
    write!(out, "    }}")?;
    Ok(())
}

fn write_deterministic_single_tool_round<W: Write>(out: &mut W) -> io::Result<()> {
    writeln!(out, "    {{")?;
    writeln!(out, "      \"id\": \"single_tool_round\",")?;
    writeln!(out, "      \"replay_sources\": [\"M11.1\", \"M11.2\"],")?;
    writeln!(
        out,
        "      \"transcript_roles\": [\"system\", \"user\", \"assistant\", \"tool\", \"assistant\"],"
    )?;
    writeln!(out, "      \"tool_replay\": {{")?;
    writeln!(out, "        \"tool_sequence\": [")?;
    writeln!(
        out,
        "          {{\"round\": 0, \"name\": \"list\", \"args\": [{{\"name\": \"path\", \"value\": \".\", \"is_string\": true}}]}}"
    )?;
    writeln!(out, "        ],")?;
    writeln!(out, "        \"stubs\": [")?;
    write!(
        out,
        "          {{\"round\": 0, \"name\": \"list\", \"args\": [{{\"name\": \"path\", \"value\": \".\", \"is_string\": true}}], \"inserted_role\": \"tool\", \"inserted_after_round\": 0, \"output\": "
    )?;
    write_json_string(out, TOOL_OUTPUT)?;
    writeln!(out, "}}")?;
    writeln!(out, "        ],")?;
    writeln!(out, "        \"tool_result_messages\": [")?;
    write!(out, "          ")?;
    write_json_string(out, TOOL_OUTPUT)?;
    writeln!(out)?;
    writeln!(out, "        ],")?;
    writeln!(
        out,
        "        \"rendered_context_case\": \"single_tool_round\","
    )?;
    writeln!(
        out,
        "        \"rendered_context_contains_tool_result\": true"
    )?;
    writeln!(out, "      }},")?;
    writeln!(out, "      \"session_replay\": {{\"operations\": []}},")?;
    write!(out, "      \"final_visible_output\": ")?;
    write_json_string(out, FINAL_ANSWER)?;
    writeln!(out, ",")?;
    writeln!(
        out,
        "      \"final_output_source\": \"model_event_round_1\""
    )?;
    write!(out, "    }}")?;
    Ok(())
}

fn write_deterministic_session_commands<W: Write>(out: &mut W) -> io::Result<()> {
    writeln!(out, "    {{")?;
    writeln!(out, "      \"id\": \"session_switching_commands\",")?;
    writeln!(out, "      \"replay_sources\": [\"M11.1\", \"M11.2\"],")?;
    writeln!(
        out,
        "      \"transcript_roles\": [\"system\", \"user\", \"assistant\"],"
    )?;
    writeln!(
        out,
        "      \"command_inputs\": [\"/save\", \"/list\", \"/switch <SESSION:alpha>\", \"/history 2\", \"/new\"],"
    )?;
    writeln!(
        out,
        "      \"tool_replay\": {{\"tool_sequence\": [], \"stubs\": [], \"tool_result_messages\": []}},"
    )?;
    writeln!(out, "      \"session_replay\": {{")?;
    writeln!(
        out,
        "        \"normalized_sessions\": [\"<SESSION:alpha>\"],"
    )?;
    writeln!(out, "        \"operations\": [")?;
    writeln!(
        out,
        "          {{\"step\": 0, \"input\": \"/save\", \"command\": \"save\", \"session\": \"<SESSION:alpha>\", \"visible\": \"saved session <SESSION:alpha> (3 turns)\"}},"
    )?;
    writeln!(
        out,
        "          {{\"step\": 1, \"input\": \"/list\", \"command\": \"list\", \"sessions\": [\"<SESSION:alpha>\"]}},"
    )?;
    writeln!(
        out,
        "          {{\"step\": 2, \"input\": \"/switch <SESSION:alpha>\", \"command\": \"switch\", \"session\": \"<SESSION:alpha>\", \"visible\": \"switched to <SESSION:alpha>\"}},"
    )?;
    writeln!(
        out,
        "          {{\"step\": 3, \"input\": \"/history 2\", \"command\": \"history\", \"turns\": 2, \"visible\": \"user: Remember that alpha was inspected.\"}},"
    )?;
    writeln!(
        out,
        "          {{\"step\": 4, \"input\": \"/new\", \"command\": \"new\", \"visible\": \"new session started from system prompt\"}}"
    )?;
    writeln!(out, "        ]")?;
    writeln!(out, "      }},")?;
    write!(out, "      \"model_visible_output_before_commands\": ")?;
    write_json_string(out, SESSION_MODEL_ANSWER)?;
    writeln!(out, ",")?;
    write!(out, "      \"final_visible_output\": ")?;
    write_json_string(out, SESSION_FINAL_COMMAND_OUTPUT)?;
    writeln!(out, ",")?;
    writeln!(
        out,
        "      \"final_output_source\": \"session_command_new\""
    )?;
    write!(out, "    }}")?;
    Ok(())
}

fn write_loop_single_tool_round<W: Write>(out: &mut W) -> io::Result<()> {
    let messages = single_tool_messages();
    let before_tool_prompt = render_chat_prompt_text(&messages[..2], None, ThinkMode::None);
    let after_tool_prompt = render_chat_prompt_text(&messages[..4], None, ThinkMode::None);
    let mut parser = AgentDsmlParser::default();
    parser.feed(TOOL_DSML.as_bytes());
    let parsed_name = parser
        .calls
        .first()
        .and_then(|call| call.name.as_deref())
        .unwrap_or("");
    let parsed_arg = parser.calls.first().and_then(|call| call.args.first());

    writeln!(out, "    {{")?;
    writeln!(out, "      \"id\": \"single_tool_round\",")?;
    writeln!(
        out,
        "      \"replay_sources\": [\"M11.1\", \"M11.2\", \"M11.3\"],"
    )?;
    writeln!(out, "      \"loop_steps\": [")?;
    writeln!(
        out,
        "        {{\"step\": 0, \"phase\": \"render_prompt\", \"message_roles\": [\"system\", \"user\"], \"prompt_has_user_marker\": {}, \"prompt_has_tool_result\": {}}},",
        before_tool_prompt.contains("<｜User｜>"),
        before_tool_prompt.contains("<tool_result>")
    )?;
    writeln!(
        out,
        "        {{\"step\": 1, \"phase\": \"parse_model_event\", \"round\": 0, \"parser_state\": \"{}\", \"parsed_tool_calls\": {}, \"raw_dsml_preserved\": {}}},",
        parser.state.name(),
        parser.calls.len(),
        parser.raw == TOOL_DSML.as_bytes()
    )?;
    write!(
        out,
        "        {{\"step\": 2, \"phase\": \"tool_replay\", \"source\": \"deterministic_stub\", \"live_tool_execution\": false, \"round\": 0, \"name\": "
    )?;
    write_json_string(out, parsed_name)?;
    write!(out, ", \"args\": [{{\"name\": ")?;
    write_json_string(out, parsed_arg.map(|arg| arg.name.as_str()).unwrap_or(""))?;
    write!(out, ", \"value\": ")?;
    write_json_string(out, parsed_arg.map(|arg| arg.value.as_str()).unwrap_or(""))?;
    writeln!(
        out,
        ", \"is_string\": {}}}], \"inserted_role\": \"tool\", \"output\": {}}},",
        parsed_arg.map(|arg| arg.is_string).unwrap_or(false),
        json_string(TOOL_OUTPUT)
    )?;
    writeln!(
        out,
        "        {{\"step\": 3, \"phase\": \"render_after_tool\", \"message_roles\": [\"system\", \"user\", \"assistant\", \"tool\"], \"prompt_has_tool_result\": {}, \"prompt_has_tool_output\": {}}},",
        after_tool_prompt.contains("<tool_result>"),
        after_tool_prompt.contains(TOOL_OUTPUT)
    )?;
    write!(
        out,
        "        {{\"step\": 4, \"phase\": \"final_model_event\", \"round\": 1, \"visible\": "
    )?;
    write_json_string(out, FINAL_ANSWER)?;
    writeln!(out, "}}")?;
    writeln!(out, "      ],")?;
    writeln!(
        out,
        "      \"parsed_tool_sequence\": [{{\"round\": 0, \"name\": \"list\", \"args\": [{{\"name\": \"path\", \"value\": \".\", \"is_string\": true}}]}}],"
    )?;
    writeln!(
        out,
        "      \"final_transcript_roles\": [\"system\", \"user\", \"assistant\", \"tool\", \"assistant\"],"
    )?;
    write!(out, "      \"final_visible_output\": ")?;
    write_json_string(out, FINAL_ANSWER)?;
    writeln!(out)?;
    write!(out, "    }}")?;
    Ok(())
}

fn write_loop_session_commands<W: Write>(out: &mut W) -> io::Result<()> {
    writeln!(out, "    {{")?;
    writeln!(out, "      \"id\": \"session_switching_commands\",")?;
    writeln!(
        out,
        "      \"replay_sources\": [\"M11.1\", \"M11.2\", \"M11.3\"],"
    )?;
    writeln!(out, "      \"loop_steps\": [")?;
    write!(
        out,
        "        {{\"step\": 0, \"phase\": \"model_event\", \"round\": 0, \"visible\": "
    )?;
    write_json_string(out, SESSION_MODEL_ANSWER)?;
    writeln!(
        out,
        ", \"transcript_roles\": [\"system\", \"user\", \"assistant\"]}},"
    )?;
    writeln!(
        out,
        "        {{\"step\": 1, \"phase\": \"session_command\", \"input\": \"/save\", \"command\": \"save\", \"session\": \"<SESSION:alpha>\", \"visible\": \"saved session <SESSION:alpha> (3 turns)\"}},"
    )?;
    writeln!(
        out,
        "        {{\"step\": 2, \"phase\": \"session_command\", \"input\": \"/list\", \"command\": \"list\", \"sessions\": [\"<SESSION:alpha>\"]}},"
    )?;
    writeln!(
        out,
        "        {{\"step\": 3, \"phase\": \"session_command\", \"input\": \"/switch <SESSION:alpha>\", \"command\": \"switch\", \"session\": \"<SESSION:alpha>\", \"visible\": \"switched to <SESSION:alpha>\"}},"
    )?;
    writeln!(
        out,
        "        {{\"step\": 4, \"phase\": \"session_command\", \"input\": \"/history 2\", \"command\": \"history\", \"turns\": 2, \"visible\": \"user: Remember that alpha was inspected.\"}},"
    )?;
    writeln!(
        out,
        "        {{\"step\": 5, \"phase\": \"session_command\", \"input\": \"/new\", \"command\": \"new\", \"visible\": \"new session started from system prompt\", \"active_session\": \"<SESSION:new>\"}}"
    )?;
    writeln!(out, "      ],")?;
    writeln!(out, "      \"saved_sessions\": [\"<SESSION:alpha>\"],")?;
    writeln!(out, "      \"active_session\": \"<SESSION:new>\",")?;
    writeln!(out, "      \"final_transcript_roles\": [\"system\"],")?;
    write!(out, "      \"final_visible_output\": ")?;
    write_json_string(out, SESSION_FINAL_COMMAND_OUTPUT)?;
    writeln!(out)?;
    write!(out, "    }}")?;
    Ok(())
}

fn single_tool_messages() -> Vec<ChatMessage> {
    vec![
        ChatMessage::new(
            "system",
            "You are a helpful coding assistant running inside ds4-agent.",
        ),
        ChatMessage::new(
            "user",
            "List the root files, then answer with the two names you used.",
        ),
        ChatMessage::new("assistant", "")
            .with_raw_tool_calls_dsml(TOOL_DSML)
            .with_tool_calls(vec![ToolCall::new(
                "list",
                vec![ToolArgument::string("path", ".")],
            )]),
        ChatMessage::new("tool", TOOL_OUTPUT),
        ChatMessage::new("assistant", FINAL_ANSWER),
    ]
}

fn session_command_messages() -> Vec<ChatMessage> {
    vec![
        ChatMessage::new(
            "system",
            "You are a helpful coding assistant running inside ds4-agent.",
        ),
        ChatMessage::new("user", "Remember that alpha was inspected."),
        ChatMessage::new("assistant", "Noted: alpha was inspected."),
    ]
}

fn count_matches(text: &str, needle: &str) -> usize {
    text.match_indices(needle).count()
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

fn json_string(value: &str) -> String {
    let mut out = Vec::new();
    write_json_string(&mut out, value).expect("write to vec");
    String::from_utf8(out).expect("json string is utf8")
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

    #[test]
    fn rendered_context_preserves_tool_boundaries() {
        let mut bytes = Vec::new();
        write_agent_rendered_context_replay(&mut bytes).expect("write replay");
        let text = String::from_utf8(bytes).expect("utf8");
        assert!(text.contains("\"schema\": \"ds4.agent_rendered_context_replay.v1\""));
        assert!(text.contains("\"dsml_tool_calls\": 1"));
        assert!(text.contains("\"tool_result\": 1"));
        assert!(text.contains(FINAL_ANSWER));
    }

    #[test]
    fn deterministic_replay_preserves_tool_and_session_effects() {
        let mut bytes = Vec::new();
        write_agent_deterministic_replay(&mut bytes).expect("write replay");
        let text = String::from_utf8(bytes).expect("utf8");
        assert!(text.contains("\"schema\": \"ds4.agent_deterministic_replay.v1\""));
        assert!(text.contains("\"rendered_context_contains_tool_result\": true"));
        assert!(text.contains("\"command\": \"history\""));
        assert!(text.contains(SESSION_FINAL_COMMAND_OUTPUT));
    }

    #[test]
    fn loop_smoke_records_no_model_tool_and_session_flow() {
        let mut bytes = Vec::new();
        write_agent_loop_smoke(&mut bytes).expect("write smoke");
        let text = String::from_utf8(bytes).expect("utf8");
        assert!(text.contains("\"schema\": \"ds4.agent_loop_smoke.v1\""));
        assert!(text.contains("\"parser_state\": \"done\""));
        assert!(text.contains("\"live_tool_execution\": false"));
        assert!(text.contains("\"active_session\": \"<SESSION:new>\""));
    }
}
