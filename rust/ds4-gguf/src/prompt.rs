use crate::Ds4Tokenizer;

pub const THINK_MAX_PREFIX: &str =
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n\
You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n\
Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkMode {
    None,
    High,
    Max,
}

impl ThinkMode {
    pub fn enabled(self) -> bool {
        !matches!(self, Self::None)
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::High => "high",
            Self::Max => "max",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub reasoning: String,
    pub tool_calls: Vec<ToolCall>,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            reasoning: String::new(),
            tool_calls: Vec::new(),
        }
    }

    pub fn with_tool_calls(mut self, tool_calls: Vec<ToolCall>) -> Self {
        self.tool_calls = tool_calls;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Vec<ToolArgument>,
}

impl ToolCall {
    pub fn new(name: impl Into<String>, arguments: Vec<ToolArgument>) -> Self {
        Self {
            name: name.into(),
            arguments,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolArgument {
    pub name: String,
    pub value: String,
    pub is_string: bool,
}

impl ToolArgument {
    pub fn string(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: value.into(),
            is_string: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CliOp {
    Begin,
    MaxEffortPrefix,
    AppendMessage { role: String, content: String },
    AssistantPrefix { think_mode: ThinkMode },
}

pub fn render_chat_prompt_text(
    messages: &[ChatMessage],
    tool_schemas: Option<&str>,
    think_mode: ThinkMode,
) -> String {
    let think = think_mode.enabled();
    let tool_context = chat_history_uses_tool_context(messages, tool_schemas);
    let last_user_idx = messages
        .iter()
        .enumerate()
        .filter(|(_, message)| role_is_user_like(&message.role))
        .map(|(idx, _)| idx)
        .last();

    let mut system = String::new();
    if let Some(tool_schemas) = tool_schemas.filter(|schemas| !schemas.is_empty()) {
        append_tools_prompt_text(&mut system, tool_schemas);
    }
    for message in messages {
        if !role_is_system(&message.role) {
            continue;
        }
        if !system.is_empty() {
            system.push_str("\n\n");
        }
        system.push_str(&message.content);
    }

    let mut out = String::new();
    out.push_str("<｜begin▁of▁sentence｜>");
    if think_mode == ThinkMode::Max {
        out.push_str(THINK_MAX_PREFIX);
    }
    out.push_str(&system);

    let mut pending_assistant = false;
    let mut pending_tool_result = false;
    for (idx, message) in messages.iter().enumerate() {
        if role_is_system(&message.role) {
            continue;
        } else if message.role == "user" {
            out.push_str("<｜User｜>");
            out.push_str(&message.content);
            pending_assistant = true;
            pending_tool_result = false;
        } else if message.role == "tool" || message.role == "function" {
            if !pending_tool_result {
                out.push_str("<｜User｜>");
            }
            out.push_str("<tool_result>");
            append_tool_result_text(&mut out, &message.content);
            out.push_str("</tool_result>");
            pending_assistant = true;
            pending_tool_result = true;
        } else if message.role == "assistant" {
            if pending_assistant {
                out.push_str("<｜Assistant｜>");
                if think {
                    if tool_context || last_user_idx.is_some_and(|last| idx > last) {
                        out.push_str("<think>");
                        out.push_str(&message.reasoning);
                        out.push_str("</think>");
                    } else {
                        out.push_str("</think>");
                    }
                } else {
                    out.push_str("</think>");
                }
            }
            out.push_str(&message.content);
            append_dsml_tool_calls_text(&mut out, &message.tool_calls);
            out.push_str("<｜end▁of▁sentence｜>");
            pending_assistant = false;
            pending_tool_result = false;
        }
    }

    if pending_assistant {
        out.push_str("<｜Assistant｜>");
        out.push_str(if think { "<think>" } else { "</think>" });
    }
    out
}

pub fn apply_cli_ops(tokenizer: &Ds4Tokenizer, ops: &[CliOp]) -> Vec<u32> {
    let mut tokens = Vec::new();
    for op in ops {
        match op {
            CliOp::Begin => tokens.push(tokenizer.special_token_ids().bos),
            CliOp::MaxEffortPrefix => tokens.extend(tokenizer.tokenize_text(THINK_MAX_PREFIX)),
            CliOp::AppendMessage { role, content } => {
                append_chat_message_tokens(tokenizer, &mut tokens, role, content);
            }
            CliOp::AssistantPrefix { think_mode } => {
                let special = tokenizer.special_token_ids();
                tokens.push(special.assistant);
                tokens.push(if think_mode.enabled() {
                    special.think_start
                } else {
                    special.think_end
                });
            }
        }
    }
    tokens
}

fn append_chat_message_tokens(
    tokenizer: &Ds4Tokenizer,
    tokens: &mut Vec<u32>,
    role: &str,
    content: &str,
) {
    let special = tokenizer.special_token_ids();
    if role == "system" || role == "developer" {
        tokens.extend(tokenizer.tokenize_text(content));
    } else if role == "assistant" {
        tokens.push(special.assistant);
        if !content.starts_with("<think>") && !content.starts_with("</think>") {
            tokens.push(special.think_end);
        }
        tokens.extend(tokenizer.tokenize_text(content));
    } else {
        tokens.push(special.user);
        if role == "tool" || role == "function" {
            tokens.extend(tokenizer.tokenize_text("Tool: "));
        }
        tokens.extend(tokenizer.tokenize_text(content));
    }
}

fn role_is_system(role: &str) -> bool {
    role == "system" || role == "developer"
}

fn role_is_user_like(role: &str) -> bool {
    role == "user" || role == "tool" || role == "function"
}

fn chat_history_uses_tool_context(messages: &[ChatMessage], tool_schemas: Option<&str>) -> bool {
    if tool_schemas.is_some_and(|schemas| !schemas.is_empty()) {
        return true;
    }
    messages.iter().any(|message| {
        (message.role == "assistant" && !message.tool_calls.is_empty())
            || message.role == "tool"
            || message.role == "function"
    })
}

fn append_tools_prompt_text(out: &mut String, tool_schemas: &str) {
    if tool_schemas.is_empty() {
        return;
    }
    out.push_str(
        "## Tools\n\n\
You have access to a set of tools to help answer the user question. You can invoke tools by writing a \"<｜DSML｜tool_calls>\" block like the following:\n\n\
<｜DSML｜tool_calls>\n\
<｜DSML｜invoke name=\"$TOOL_NAME\">\n\
<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜parameter>\n\
...\n\
</｜DSML｜invoke>\n\
<｜DSML｜invoke name=\"$TOOL_NAME2\">\n\
...\n\
</｜DSML｜invoke>\n\
</｜DSML｜tool_calls>\n\n\
String parameters should be specified as raw text and set `string=\"true\"`. Preserve characters such as `>`, `&`, and `&&` exactly; never replace normal string characters with XML or HTML entity escapes. Only if a string value itself contains the exact closing parameter tag `</｜DSML｜parameter>`, write that tag as `&lt;/｜DSML｜parameter>` inside the value. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\n\
If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\n\
Otherwise, output directly after </think> with tool calls or final response.\n\n\
### Available Tool Schemas\n\n",
    );
    out.push_str(tool_schemas);
    out.push_str(
        "\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls. Use the exact parameter names from the schemas.",
    );
}

fn append_tool_result_text(out: &mut String, text: &str) {
    append_escaped_sentinel(out, text, "</tool_result>", "&lt;");
}

fn append_dsml_tool_calls_text(out: &mut String, calls: &[ToolCall]) {
    if calls.is_empty() {
        return;
    }
    out.push_str("\n\n<｜DSML｜tool_calls>\n");
    for call in calls {
        out.push_str("<｜DSML｜invoke name=\"");
        append_dsml_attr_escaped(out, &call.name);
        out.push_str("\">\n");
        for arg in &call.arguments {
            out.push_str("<｜DSML｜parameter name=\"");
            append_dsml_attr_escaped(out, &arg.name);
            out.push_str("\" string=\"");
            out.push_str(if arg.is_string { "true" } else { "false" });
            out.push_str("\">");
            if arg.is_string {
                append_dsml_parameter_text(out, &arg.value);
            } else {
                append_dsml_json_literal(out, &arg.value);
            }
            out.push_str("</｜DSML｜parameter>\n");
        }
        out.push_str("</｜DSML｜invoke>\n");
    }
    out.push_str("</｜DSML｜tool_calls>");
}

fn append_dsml_attr_escaped(out: &mut String, text: &str) {
    for ch in text.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            ch => out.push(ch),
        }
    }
}

fn append_dsml_parameter_text(out: &mut String, text: &str) {
    append_escaped_sentinel(out, text, "</｜DSML｜parameter>", "&lt;");
}

fn append_dsml_json_literal(out: &mut String, text: &str) {
    append_escaped_sentinel(out, text, "</｜DSML｜parameter>", "\\u003c");
}

fn append_escaped_sentinel(out: &mut String, text: &str, sentinel: &str, replacement: &str) {
    let mut rest = text;
    while !rest.is_empty() {
        if rest.starts_with(sentinel) {
            out.push_str(replacement);
            rest = &rest[1..];
        } else {
            let ch = rest.chars().next().expect("nonempty string has char");
            out.push(ch);
            rest = &rest[ch.len_utf8()..];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_result_escapes_only_closing_tag_start() {
        let mut out = String::new();
        append_tool_result_text(&mut out, "a </tool_result> & b");
        assert_eq!(out, "a &lt;/tool_result> & b");
    }

    #[test]
    fn renderer_keeps_tool_schema_before_system_text() {
        let prompt = render_chat_prompt_text(
            &[
                ChatMessage::new("system", "sys"),
                ChatMessage::new("user", "hi"),
            ],
            Some("{\"name\":\"tool\"}"),
            ThinkMode::None,
        );
        let tools = prompt.find("## Tools").expect("tools");
        let system = prompt.find("sys").expect("system");
        assert!(tools < system);
        assert!(prompt.ends_with("<｜Assistant｜></think>"));
    }
}
