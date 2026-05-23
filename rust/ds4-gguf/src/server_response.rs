use crate::decode_policy::utf8_stream_safe_len;
use crate::dsml::{normalize_json_object_or_empty, DsmlJsonCall};
use crate::server_chat::ToolSchemaOrder;
use crate::server_http::{append_cors_headers, format_http_response, json_escape_string};

const DS4_TOOL_CALLS_START: &[u8] = "<｜DSML｜tool_calls>".as_bytes();
const DS4_TOOL_CALLS_END: &[u8] = "</｜DSML｜tool_calls>".as_bytes();
const DS4_INVOKE_START: &[u8] = "<｜DSML｜invoke".as_bytes();
const DS4_INVOKE_END: &[u8] = "</｜DSML｜invoke>".as_bytes();
const DS4_PARAM_START: &[u8] = "<｜DSML｜parameter".as_bytes();
const DS4_PARAM_END: &[u8] = "</｜DSML｜parameter>".as_bytes();
const DS4_TOOL_CALLS_START_SHORT: &[u8] = "<DSML｜tool_calls>".as_bytes();
const DS4_TOOL_CALLS_END_SHORT: &[u8] = "</DSML｜tool_calls>".as_bytes();
const DS4_INVOKE_START_SHORT: &[u8] = "<DSML｜invoke".as_bytes();
const DS4_INVOKE_END_SHORT: &[u8] = "</DSML｜invoke>".as_bytes();
const DS4_PARAM_START_SHORT: &[u8] = "<DSML｜parameter".as_bytes();
const DS4_PARAM_END_SHORT: &[u8] = "</DSML｜parameter>".as_bytes();
const PLAIN_TOOL_CALLS_START: &[u8] = b"<tool_calls>";
const PLAIN_TOOL_CALLS_END: &[u8] = b"</tool_calls>";
const PLAIN_INVOKE_START: &[u8] = b"<invoke";
const PLAIN_INVOKE_END: &[u8] = b"</invoke>";
const PLAIN_PARAM_START: &[u8] = b"<parameter";
const PLAIN_PARAM_END: &[u8] = b"</parameter>";

const DSML_TOOL_SYNTAXES: &[DsmlToolSyntax] = &[
    DsmlToolSyntax {
        tool_calls_start: DS4_TOOL_CALLS_START,
        tool_calls_end: DS4_TOOL_CALLS_END,
        invoke_start: DS4_INVOKE_START,
        invoke_end: DS4_INVOKE_END,
        param_start: DS4_PARAM_START,
        param_end: DS4_PARAM_END,
    },
    DsmlToolSyntax {
        tool_calls_start: DS4_TOOL_CALLS_START_SHORT,
        tool_calls_end: DS4_TOOL_CALLS_END_SHORT,
        invoke_start: DS4_INVOKE_START_SHORT,
        invoke_end: DS4_INVOKE_END_SHORT,
        param_start: DS4_PARAM_START_SHORT,
        param_end: DS4_PARAM_END_SHORT,
    },
    DsmlToolSyntax {
        tool_calls_start: PLAIN_TOOL_CALLS_START,
        tool_calls_end: PLAIN_TOOL_CALLS_END,
        invoke_start: PLAIN_INVOKE_START,
        invoke_end: PLAIN_INVOKE_END,
        param_start: PLAIN_PARAM_START,
        param_end: PLAIN_PARAM_END,
    },
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpenAiUsage {
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub cache_read_tokens: i32,
    pub cache_write_tokens: i32,
}

impl OpenAiUsage {
    pub const fn new(
        prompt_tokens: i32,
        completion_tokens: i32,
        cache_read_tokens: i32,
        cache_write_tokens: i32,
    ) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            cache_read_tokens,
            cache_write_tokens,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpenAiChatCompletion<'a> {
    pub id: &'a str,
    pub created: i64,
    pub model: &'a str,
    pub content: &'a str,
    pub reasoning_content: Option<&'a str>,
    pub finish_reason: &'a str,
    pub usage: OpenAiUsage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpenAiChatStream<'a> {
    pub id: &'a str,
    pub created: i64,
    pub model: &'a str,
    pub content_deltas: &'a [&'a str],
    pub finish_reason: &'a str,
    pub usage: Option<OpenAiUsage>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpenAiChatToolStream<'a> {
    pub id: &'a str,
    pub created: i64,
    pub model: &'a str,
    pub events: &'a [OpenAiToolCallStreamEvent<'a>],
    pub finish_reason: &'a str,
    pub usage: Option<OpenAiUsage>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResponsesFinalResponse<'a> {
    pub id: &'a str,
    pub created_at: i64,
    pub model: &'a str,
    pub content: &'a str,
    pub reasoning: Option<&'a str>,
    pub reasoning_summary_emit: bool,
    pub finish_reason: &'a str,
    pub usage: OpenAiUsage,
    pub reasoning_id: &'a str,
    pub message_id: &'a str,
    pub function_call_id_prefix: &'a str,
    pub call_id_prefix: &'a str,
    pub tool_orders: &'a [ToolSchemaOrder],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResponsesStreamResponse<'a> {
    pub response: ResponsesFinalResponse<'a>,
    pub reasoning_closed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnthropicMessageResponse<'a> {
    pub id: &'a str,
    pub model: &'a str,
    pub content: &'a str,
    pub reasoning: Option<&'a str>,
    pub finish_reason: &'a str,
    pub usage: OpenAiUsage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAiToolCallStreamEvent<'a> {
    Content {
        delta: &'a str,
    },
    Start {
        index: usize,
        id: &'a str,
        name: &'a str,
    },
    Arguments {
        index: usize,
        fragment: &'a str,
    },
    FullCalls {
        calls: &'a [DsmlJsonCall],
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpenAiToolCallStreamEventOwned {
    Start {
        index: usize,
        id: String,
        name: String,
    },
    Arguments {
        index: usize,
        fragment: String,
    },
}

impl OpenAiToolCallStreamEventOwned {
    pub fn as_borrowed(&self) -> OpenAiToolCallStreamEvent<'_> {
        match self {
            Self::Start { index, id, name } => OpenAiToolCallStreamEvent::Start {
                index: *index,
                id,
                name,
            },
            Self::Arguments { index, fragment } => OpenAiToolCallStreamEvent::Arguments {
                index: *index,
                fragment,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenAiToolCallStreamTranslator {
    state: DsmlToolStreamState,
    raw: Vec<u8>,
    parse_pos: usize,
    syntax: Option<DsmlToolSyntax>,
    call_id_prefix: String,
    call_ids: Vec<String>,
    index: usize,
    args_open: bool,
    first_param: bool,
    param_is_string: bool,
    emitted_any: bool,
}

impl OpenAiToolCallStreamTranslator {
    pub fn new(call_id_prefix: impl Into<String>) -> Self {
        Self {
            state: DsmlToolStreamState::Search,
            raw: Vec::new(),
            parse_pos: 0,
            syntax: None,
            call_id_prefix: call_id_prefix.into(),
            call_ids: Vec::new(),
            index: 0,
            args_open: false,
            first_param: true,
            param_is_string: false,
            emitted_any: false,
        }
    }

    pub fn feed(&mut self, bytes: &[u8]) -> Vec<OpenAiToolCallStreamEventOwned> {
        if matches!(
            self.state,
            DsmlToolStreamState::Done | DsmlToolStreamState::Error
        ) {
            return Vec::new();
        }
        self.raw.extend_from_slice(bytes);
        let mut events = Vec::new();
        self.parse(&mut events);
        events
    }

    pub fn emitted_any(&self) -> bool {
        self.emitted_any
    }

    pub fn is_done(&self) -> bool {
        self.state == DsmlToolStreamState::Done
    }

    pub fn is_error(&self) -> bool {
        self.state == DsmlToolStreamState::Error
    }

    pub fn call_ids(&self) -> &[String] {
        &self.call_ids
    }

    fn parse(&mut self, events: &mut Vec<OpenAiToolCallStreamEventOwned>) {
        loop {
            match self.state {
                DsmlToolStreamState::Search => {
                    let Some((start, syntax)) = find_tool_start_from(&self.raw, self.parse_pos)
                    else {
                        self.parse_pos = self
                            .raw
                            .len()
                            .saturating_sub(max_tool_start_len().saturating_sub(1));
                        return;
                    };
                    self.syntax = Some(syntax);
                    self.parse_pos = start + syntax.tool_calls_start.len();
                    self.state = DsmlToolStreamState::BetweenInvokes;
                }
                DsmlToolStreamState::BetweenInvokes => {
                    let Some(syntax) = self.syntax else {
                        self.state = DsmlToolStreamState::Error;
                        return;
                    };
                    self.skip_ascii_ws();
                    if self.parse_pos >= self.raw.len() {
                        return;
                    }
                    if raw_full_lit(&self.raw, self.parse_pos, syntax.tool_calls_end) {
                        self.parse_pos += syntax.tool_calls_end.len();
                        self.state = DsmlToolStreamState::Done;
                        return;
                    }
                    if raw_partial_any(
                        &self.raw,
                        self.parse_pos,
                        syntax.tool_calls_end,
                        syntax.invoke_start,
                    ) {
                        return;
                    }
                    if raw_full_lit(&self.raw, self.parse_pos, syntax.invoke_start) {
                        if !self.start_invoke(events) {
                            return;
                        }
                        continue;
                    }
                    self.state = DsmlToolStreamState::Error;
                    return;
                }
                DsmlToolStreamState::BetweenParams => {
                    let Some(syntax) = self.syntax else {
                        self.state = DsmlToolStreamState::Error;
                        return;
                    };
                    self.skip_ascii_ws();
                    if self.parse_pos >= self.raw.len() {
                        return;
                    }
                    if raw_full_lit(&self.raw, self.parse_pos, syntax.invoke_end) {
                        if self.args_open {
                            events.push(OpenAiToolCallStreamEventOwned::Arguments {
                                index: self.index,
                                fragment: "}".to_string(),
                            });
                        }
                        self.args_open = false;
                        self.parse_pos += syntax.invoke_end.len();
                        self.index += 1;
                        self.state = DsmlToolStreamState::BetweenInvokes;
                        continue;
                    }
                    if raw_partial_any(
                        &self.raw,
                        self.parse_pos,
                        syntax.invoke_end,
                        syntax.param_start,
                    ) {
                        return;
                    }
                    if raw_full_lit(&self.raw, self.parse_pos, syntax.param_start) {
                        if !self.start_param(events) {
                            return;
                        }
                        continue;
                    }
                    self.state = DsmlToolStreamState::Error;
                    return;
                }
                DsmlToolStreamState::ParamValue => {
                    let Some(syntax) = self.syntax else {
                        self.state = DsmlToolStreamState::Error;
                        return;
                    };
                    if let Some(rel) = find_bytes(&self.raw[self.parse_pos..], syntax.param_end) {
                        let value_end = self.parse_pos + rel;
                        self.emit_param_value(value_end, events);
                        if self.param_is_string {
                            events.push(OpenAiToolCallStreamEventOwned::Arguments {
                                index: self.index,
                                fragment: "\"".to_string(),
                            });
                        }
                        self.parse_pos = value_end + syntax.param_end.len();
                        self.state = DsmlToolStreamState::BetweenParams;
                        continue;
                    }

                    let limit = tool_param_value_stream_safe_len(
                        &self.raw,
                        self.parse_pos,
                        self.raw.len(),
                        syntax.param_end,
                        self.param_is_string,
                    );
                    if limit > self.parse_pos {
                        self.emit_param_value(limit, events);
                        self.parse_pos = limit;
                    }
                    return;
                }
                DsmlToolStreamState::Done | DsmlToolStreamState::Error => return,
            }
        }
    }

    fn start_invoke(&mut self, events: &mut Vec<OpenAiToolCallStreamEventOwned>) -> bool {
        let Some(tag_end_rel) = self.raw[self.parse_pos..]
            .iter()
            .position(|&byte| byte == b'>')
        else {
            return false;
        };
        let tag_end = self.parse_pos + tag_end_rel + 1;
        let Some(name) = parse_dsml_attr(&self.raw[self.parse_pos..tag_end], b"name") else {
            self.state = DsmlToolStreamState::Error;
            return false;
        };
        let id = self.tool_id(self.index);
        events.push(OpenAiToolCallStreamEventOwned::Start {
            index: self.index,
            id,
            name,
        });
        events.push(OpenAiToolCallStreamEventOwned::Arguments {
            index: self.index,
            fragment: "{".to_string(),
        });
        self.emitted_any = true;
        self.args_open = true;
        self.first_param = true;
        self.parse_pos = tag_end;
        self.state = DsmlToolStreamState::BetweenParams;
        true
    }

    fn start_param(&mut self, events: &mut Vec<OpenAiToolCallStreamEventOwned>) -> bool {
        let Some(tag_end_rel) = self.raw[self.parse_pos..]
            .iter()
            .position(|&byte| byte == b'>')
        else {
            return false;
        };
        let tag_end = self.parse_pos + tag_end_rel + 1;
        let tag = &self.raw[self.parse_pos..tag_end];
        let Some(name) = parse_dsml_attr(tag, b"name") else {
            self.state = DsmlToolStreamState::Error;
            return false;
        };
        let Some(is_string) = parse_dsml_attr(tag, b"string") else {
            self.state = DsmlToolStreamState::Error;
            return false;
        };
        self.param_is_string = is_string == "true";

        let mut fragment = String::new();
        if self.first_param {
            self.first_param = false;
        } else {
            fragment.push(',');
        }
        fragment.push_str(&json_escape_string(&name));
        fragment.push(':');
        if self.param_is_string {
            fragment.push('"');
        }
        events.push(OpenAiToolCallStreamEventOwned::Arguments {
            index: self.index,
            fragment,
        });
        self.parse_pos = tag_end;
        self.state = DsmlToolStreamState::ParamValue;
        true
    }

    fn emit_param_value(&self, value_end: usize, events: &mut Vec<OpenAiToolCallStreamEventOwned>) {
        if value_end <= self.parse_pos {
            return;
        }
        let value = String::from_utf8_lossy(&self.raw[self.parse_pos..value_end]);
        let fragment = if self.param_is_string {
            json_escape_fragment(&dsml_unescape_text(&value))
        } else {
            value.into_owned()
        };
        if !fragment.is_empty() {
            events.push(OpenAiToolCallStreamEventOwned::Arguments {
                index: self.index,
                fragment,
            });
        }
    }

    fn skip_ascii_ws(&mut self) {
        while self
            .raw
            .get(self.parse_pos)
            .is_some_and(|byte| byte.is_ascii_whitespace())
        {
            self.parse_pos += 1;
        }
    }

    fn tool_id(&mut self, index: usize) -> String {
        while self.call_ids.len() <= index {
            let next = self.call_ids.len();
            self.call_ids
                .push(format!("{}{:016x}", self.call_id_prefix, next));
        }
        self.call_ids[index].clone()
    }
}

pub fn format_openai_chat_completion_json(response: &OpenAiChatCompletion<'_>) -> String {
    format_openai_chat_completion_body_json(response, &[])
}

pub fn format_openai_chat_tool_completion_json(
    response: &OpenAiChatCompletion<'_>,
    tool_calls: &[DsmlJsonCall],
) -> String {
    format_openai_chat_completion_body_json(response, tool_calls)
}

fn format_openai_chat_completion_body_json(
    response: &OpenAiChatCompletion<'_>,
    tool_calls: &[DsmlJsonCall],
) -> String {
    let mut out = String::new();
    out.push_str("{\"id\":");
    out.push_str(&json_escape_string(response.id));
    out.push_str(",\"object\":\"chat.completion\",\"created\":");
    out.push_str(&response.created.to_string());
    out.push_str(",\"model\":");
    out.push_str(&json_escape_string(response.model));
    out.push_str(",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
    out.push_str(&json_escape_string(response.content));
    if let Some(reasoning) = response
        .reasoning_content
        .filter(|reasoning| !reasoning.is_empty())
    {
        out.push_str(",\"reasoning_content\":");
        out.push_str(&json_escape_string(reasoning));
    }
    if !tool_calls.is_empty() {
        out.push_str(",\"tool_calls\":");
        append_openai_tool_calls_json(&mut out, response.id, tool_calls);
    }
    out.push_str("},\"finish_reason\":");
    out.push_str(&json_escape_string(response.finish_reason));
    out.push_str("}],\"usage\":");
    append_openai_usage_json(&mut out, response.usage);
    out.push_str("}\n");
    out
}

pub fn format_openai_chat_completion_http(
    enable_cors: bool,
    response: &OpenAiChatCompletion<'_>,
) -> String {
    let body = format_openai_chat_completion_json(response);
    format_http_response(enable_cors, 200, Some("application/json"), &body)
}

pub fn format_openai_chat_tool_completion_http(
    enable_cors: bool,
    response: &OpenAiChatCompletion<'_>,
    tool_calls: &[DsmlJsonCall],
) -> String {
    let body = format_openai_chat_tool_completion_json(response, tool_calls);
    format_http_response(enable_cors, 200, Some("application/json"), &body)
}

pub fn format_responses_final_response_json(
    response: &ResponsesFinalResponse<'_>,
    tool_calls: &[DsmlJsonCall],
) -> String {
    let status = responses_status_for_finish(response.finish_reason);
    let item_status = responses_item_status_for_finish(response.finish_reason);
    let mut out = String::new();
    out.push_str("{\"id\":");
    out.push_str(&json_escape_string(response.id));
    out.push_str(",\"object\":\"response\",\"created_at\":");
    out.push_str(&response.created_at.to_string());
    out.push_str(",\"status\":");
    out.push_str(&json_escape_string(status));
    out.push_str(",\"model\":");
    out.push_str(&json_escape_string(response.model));
    match response.finish_reason {
        "error" => {
            out.push_str(",\"error\":{\"code\":\"server_error\",\"message\":\"generation failed\"}")
        }
        "length" => out.push_str(",\"incomplete_details\":{\"reason\":\"max_tokens\"}"),
        _ => {}
    }
    out.push_str(",\"output\":[");
    let mut wrote = false;
    if let Some(reasoning) = response
        .reasoning
        .filter(|reasoning| response.reasoning_summary_emit && !reasoning.is_empty())
    {
        append_separator(&mut out, &mut wrote);
        out.push_str("{\"id\":");
        out.push_str(&json_escape_string(response.reasoning_id));
        out.push_str(",\"type\":\"reasoning\",\"status\":");
        out.push_str(&json_escape_string(item_status));
        out.push_str(",\"summary\":[{\"type\":\"summary_text\",\"text\":");
        out.push_str(&json_escape_string(reasoning));
        out.push_str("}]}");
    }
    if !response.content.is_empty() {
        append_separator(&mut out, &mut wrote);
        out.push_str("{\"id\":");
        out.push_str(&json_escape_string(response.message_id));
        out.push_str(",\"type\":\"message\",\"status\":");
        out.push_str(&json_escape_string(item_status));
        out.push_str(",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":");
        out.push_str(&json_escape_string(response.content));
        out.push_str(",\"annotations\":[]}]}");
    }
    for (index, call) in tool_calls.iter().enumerate() {
        append_separator(&mut out, &mut wrote);
        let item = ResponsesToolItem {
            function_call_id: format!("{}{}", response.function_call_id_prefix, index),
            call_id: call
                .id
                .clone()
                .unwrap_or_else(|| format!("{}{}", response.call_id_prefix, index)),
        };
        append_responses_function_call_item(
            &mut out,
            call,
            &item,
            item_status,
            true,
            response.tool_orders,
        );
    }
    out.push_str("],\"usage\":");
    append_responses_usage_json(&mut out, response.usage);
    out.push('}');
    out
}

pub fn format_responses_final_response_http(
    enable_cors: bool,
    response: &ResponsesFinalResponse<'_>,
    tool_calls: &[DsmlJsonCall],
) -> String {
    let body = format_responses_final_response_json(response, tool_calls);
    format_http_response(enable_cors, 200, Some("application/json"), &body)
}

pub fn format_anthropic_message_json(
    response: &AnthropicMessageResponse<'_>,
    tool_calls: &[DsmlJsonCall],
) -> String {
    let mut out = String::new();
    out.push_str("{\"id\":");
    out.push_str(&json_escape_string(response.id));
    out.push_str(",\"type\":\"message\",\"role\":\"assistant\",\"model\":");
    out.push_str(&json_escape_string(response.model));
    out.push_str(",\"content\":");
    append_anthropic_content(
        &mut out,
        response.content,
        response.reasoning,
        tool_calls,
        response.id,
    );
    out.push_str(",\"stop_reason\":");
    out.push_str(&json_escape_string(anthropic_stop_reason(
        response.finish_reason,
    )));
    out.push_str(",\"stop_sequence\":null,\"usage\":");
    append_anthropic_usage_json(&mut out, response.usage);
    out.push_str("}\n");
    out
}

pub fn format_anthropic_message_http(
    enable_cors: bool,
    response: &AnthropicMessageResponse<'_>,
    tool_calls: &[DsmlJsonCall],
) -> String {
    let body = format_anthropic_message_json(response, tool_calls);
    format_http_response(enable_cors, 200, Some("application/json"), &body)
}

pub fn format_responses_stream_sse(
    stream: &ResponsesStreamResponse<'_>,
    tool_calls: &[DsmlJsonCall],
) -> String {
    let response = &stream.response;
    let item_status = responses_item_status_for_finish(response.finish_reason);
    let mut out = String::new();
    let mut sequence = 0;
    let mut next_output_index = 0usize;
    let mut reasoning_index = None;
    let mut message_index = None;

    let mut body = String::new();
    body.push_str("{\"type\":\"response.created\",\"response\":{\"id\":");
    body.push_str(&json_escape_string(response.id));
    body.push_str(",\"object\":\"response\",\"created_at\":");
    body.push_str(&response.created_at.to_string());
    body.push_str(",\"status\":\"in_progress\",\"model\":");
    body.push_str(&json_escape_string(response.model));
    body.push_str(",\"output\":[]}}");
    append_responses_sse_event_body(&mut out, &mut sequence, &body);

    if let Some(reasoning) = response
        .reasoning
        .filter(|reasoning| response.reasoning_summary_emit && !reasoning.is_empty())
    {
        let index = next_output_index;
        next_output_index += 1;
        reasoning_index = Some(index);
        append_responses_reasoning_events(
            &mut out,
            &mut sequence,
            response,
            index,
            reasoning,
            stream.reasoning_closed,
        );
    }

    if !response.content.is_empty() {
        let index = next_output_index;
        next_output_index += 1;
        message_index = Some(index);
        append_responses_message_events(&mut out, &mut sequence, response, index, item_status);
    }

    let tool_output_index = next_output_index;
    for (index, call) in tool_calls.iter().enumerate() {
        let output_index = tool_output_index + index;
        let item = responses_tool_item(response, call, index);
        append_responses_function_call_event(
            &mut out,
            &mut sequence,
            call,
            &item,
            output_index,
            "in_progress",
            false,
            response.tool_orders,
        );
        append_responses_function_call_argument_events(
            &mut out,
            &mut sequence,
            call,
            &item,
            output_index,
            response.tool_orders,
        );
        append_responses_function_call_event(
            &mut out,
            &mut sequence,
            call,
            &item,
            output_index,
            item_status,
            true,
            response.tool_orders,
        );
    }

    append_responses_terminal_event(
        &mut out,
        &mut sequence,
        response,
        stream.reasoning_closed,
        reasoning_index,
        message_index,
        tool_calls,
    );
    out
}

pub fn format_responses_stream_http(
    enable_cors: bool,
    stream: &ResponsesStreamResponse<'_>,
    tool_calls: &[DsmlJsonCall],
) -> String {
    let body = format_responses_stream_sse(stream, tool_calls);
    format_openai_chat_stream_http_body(enable_cors, &body)
}

pub fn format_anthropic_message_stream_sse(
    response: &AnthropicMessageResponse<'_>,
    tool_calls: &[DsmlJsonCall],
) -> String {
    let mut out = String::new();
    let mut next_index = 0usize;
    let mut sent_text = false;

    let mut body = String::new();
    body.push_str("{\"type\":\"message_start\",\"message\":{\"id\":");
    body.push_str(&json_escape_string(response.id));
    body.push_str(",\"type\":\"message\",\"role\":\"assistant\",\"model\":");
    body.push_str(&json_escape_string(response.model));
    body.push_str(",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":");
    append_anthropic_start_usage_json(&mut body, response.usage);
    body.push_str("}}");
    append_sse_event(&mut out, "message_start", &body);

    if let Some(reasoning) = response.reasoning.filter(|reasoning| !reasoning.is_empty()) {
        append_anthropic_thinking_stream_block(&mut out, next_index, response.id, reasoning);
        next_index += 1;
    }

    if !response.content.is_empty() {
        append_anthropic_text_stream_block(&mut out, next_index, response.content);
        next_index += 1;
        sent_text = true;
    }

    for (call_index, call) in tool_calls.iter().enumerate() {
        append_anthropic_tool_stream_block(&mut out, next_index, response.id, call_index, call);
        next_index += 1;
    }

    if response
        .reasoning
        .is_some_and(|reasoning| !reasoning.is_empty())
        && !sent_text
        && tool_calls.is_empty()
    {
        append_anthropic_empty_text_stream_block(&mut out, next_index);
    }

    body.clear();
    body.push_str("{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":");
    body.push_str(&json_escape_string(anthropic_stop_reason(
        response.finish_reason,
    )));
    body.push_str(",\"stop_sequence\":null},\"usage\":{\"output_tokens\":");
    body.push_str(&response.usage.completion_tokens.to_string());
    body.push_str("}}");
    append_sse_event(&mut out, "message_delta", &body);
    append_sse_event(&mut out, "message_stop", "{\"type\":\"message_stop\"}");
    out
}

pub fn format_anthropic_message_stream_http(
    enable_cors: bool,
    response: &AnthropicMessageResponse<'_>,
    tool_calls: &[DsmlJsonCall],
) -> String {
    let body = format_anthropic_message_stream_sse(response, tool_calls);
    format_openai_chat_stream_http_body(enable_cors, &body)
}

pub fn format_openai_chat_stream_sse(response: &OpenAiChatStream<'_>) -> String {
    let mut out = String::new();
    append_openai_chat_stream_role_chunk(&mut out, response.id, response.created, response.model);
    for delta in response.content_deltas {
        if !delta.is_empty() {
            append_openai_chat_stream_content_chunk(&mut out, response, delta);
        }
    }
    append_openai_chat_stream_finish_chunk(&mut out, response);
    if let Some(usage) = response.usage {
        append_openai_chat_stream_usage_chunk(&mut out, response, usage);
    }
    out.push_str("data: [DONE]\n");
    out
}

pub fn format_openai_chat_tool_stream_sse(response: &OpenAiChatToolStream<'_>) -> String {
    let mut out = String::new();
    append_openai_chat_stream_role_chunk(&mut out, response.id, response.created, response.model);
    for event in response.events {
        match event {
            OpenAiToolCallStreamEvent::Content { delta } => {
                if !delta.is_empty() {
                    append_openai_chat_stream_content_fields_chunk(
                        &mut out,
                        response.id,
                        response.created,
                        response.model,
                        delta,
                    );
                }
            }
            OpenAiToolCallStreamEvent::Start { index, id, name } => {
                append_openai_chat_tool_stream_start_chunk(&mut out, response, *index, id, name);
            }
            OpenAiToolCallStreamEvent::Arguments { index, fragment } => {
                if !fragment.is_empty() {
                    append_openai_chat_tool_stream_arguments_chunk(
                        &mut out, response, *index, fragment,
                    );
                }
            }
            OpenAiToolCallStreamEvent::FullCalls { calls } => {
                if !calls.is_empty() {
                    append_openai_chat_tool_stream_full_calls_chunk(&mut out, response, calls);
                }
            }
        }
    }
    append_openai_chat_stream_finish_fields_chunk(
        &mut out,
        response.id,
        response.created,
        response.model,
        response.finish_reason,
    );
    if let Some(usage) = response.usage {
        append_openai_chat_stream_usage_fields_chunk(
            &mut out,
            response.id,
            response.created,
            response.model,
            usage,
        );
    }
    out.push_str("data: [DONE]\n");
    out
}

pub fn format_openai_chat_stream_http(
    enable_cors: bool,
    response: &OpenAiChatStream<'_>,
) -> String {
    let body = format_openai_chat_stream_sse(response);
    format_openai_chat_stream_http_body(enable_cors, &body)
}

pub fn format_openai_chat_tool_stream_http(
    enable_cors: bool,
    response: &OpenAiChatToolStream<'_>,
) -> String {
    let body = format_openai_chat_tool_stream_sse(response);
    format_openai_chat_stream_http_body(enable_cors, &body)
}

fn format_openai_chat_stream_http_body(enable_cors: bool, body: &str) -> String {
    let mut out = String::new();
    out.push_str(
        "HTTP/1.1 200 OK\r\n\
Content-Type: text/event-stream\r\n\
Cache-Control: no-cache\r\n",
    );
    if enable_cors {
        append_cors_headers(&mut out);
    }
    out.push_str("Connection: close\r\n\r\n");
    out.push_str(body);
    out
}

fn append_openai_chat_stream_prefix(out: &mut String, id: &str, created: i64, model: &str) {
    out.push_str("data: {\"id\":");
    out.push_str(&json_escape_string(id));
    out.push_str(",\"object\":\"chat.completion.chunk\",\"created\":");
    out.push_str(&created.to_string());
    out.push_str(",\"model\":");
    out.push_str(&json_escape_string(model));
}

fn append_openai_chat_stream_role_chunk(out: &mut String, id: &str, created: i64, model: &str) {
    append_openai_chat_stream_prefix(out, id, created, model);
    out.push_str(
        ",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n",
    );
}

fn append_openai_chat_stream_content_chunk(
    out: &mut String,
    response: &OpenAiChatStream<'_>,
    delta: &str,
) {
    append_openai_chat_stream_content_fields_chunk(
        out,
        response.id,
        response.created,
        response.model,
        delta,
    );
}

fn append_openai_chat_stream_content_fields_chunk(
    out: &mut String,
    id: &str,
    created: i64,
    model: &str,
    delta: &str,
) {
    append_openai_chat_stream_prefix(out, id, created, model);
    out.push_str(",\"choices\":[{\"index\":0,\"delta\":{\"content\":");
    out.push_str(&json_escape_string(delta));
    out.push_str("},\"finish_reason\":null}]}\n\n");
}

fn append_openai_chat_stream_finish_chunk(out: &mut String, response: &OpenAiChatStream<'_>) {
    append_openai_chat_stream_finish_fields_chunk(
        out,
        response.id,
        response.created,
        response.model,
        response.finish_reason,
    );
}

fn append_openai_chat_stream_finish_fields_chunk(
    out: &mut String,
    id: &str,
    created: i64,
    model: &str,
    finish_reason: &str,
) {
    append_openai_chat_stream_prefix(out, id, created, model);
    out.push_str(",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":");
    out.push_str(&json_escape_string(finish_reason));
    out.push_str("}]}\n\n");
}

fn append_openai_chat_stream_usage_chunk(
    out: &mut String,
    response: &OpenAiChatStream<'_>,
    usage: OpenAiUsage,
) {
    append_openai_chat_stream_usage_fields_chunk(
        out,
        response.id,
        response.created,
        response.model,
        usage,
    );
}

fn append_openai_chat_stream_usage_fields_chunk(
    out: &mut String,
    id: &str,
    created: i64,
    model: &str,
    usage: OpenAiUsage,
) {
    append_openai_chat_stream_prefix(out, id, created, model);
    out.push_str(",\"choices\":[],\"usage\":");
    append_openai_usage_json(out, usage);
    out.push_str("}\n\n");
}

fn append_openai_chat_tool_stream_start_chunk(
    out: &mut String,
    response: &OpenAiChatToolStream<'_>,
    index: usize,
    id: &str,
    name: &str,
) {
    append_openai_chat_stream_prefix(out, response.id, response.created, response.model);
    out.push_str(",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":");
    out.push_str(&index.to_string());
    out.push_str(",\"id\":");
    out.push_str(&json_escape_string(id));
    out.push_str(",\"type\":\"function\",\"function\":{\"name\":");
    out.push_str(&json_escape_string(name));
    out.push_str(",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\n");
}

fn append_openai_chat_tool_stream_arguments_chunk(
    out: &mut String,
    response: &OpenAiChatToolStream<'_>,
    index: usize,
    fragment: &str,
) {
    append_openai_chat_stream_prefix(out, response.id, response.created, response.model);
    out.push_str(",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":");
    out.push_str(&index.to_string());
    out.push_str(",\"function\":{\"arguments\":");
    out.push_str(&json_escape_string(fragment));
    out.push_str("}}]},\"finish_reason\":null}]}\n\n");
}

fn append_openai_chat_tool_stream_full_calls_chunk(
    out: &mut String,
    response: &OpenAiChatToolStream<'_>,
    calls: &[DsmlJsonCall],
) {
    append_openai_chat_stream_prefix(out, response.id, response.created, response.model);
    out.push_str(",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":");
    append_openai_tool_call_deltas_json(out, response.id, calls);
    out.push_str("},\"finish_reason\":null}]}\n\n");
}

fn append_openai_usage_json(out: &mut String, usage: OpenAiUsage) {
    let cached_tokens = clamp_usage_tokens(usage.cache_read_tokens, usage.prompt_tokens);
    let cache_write_tokens = clamp_usage_tokens(
        usage.cache_write_tokens,
        usage.prompt_tokens - cached_tokens,
    );
    out.push_str("{\"prompt_tokens\":");
    out.push_str(&usage.prompt_tokens.to_string());
    out.push_str(",\"completion_tokens\":");
    out.push_str(&usage.completion_tokens.to_string());
    out.push_str(",\"total_tokens\":");
    out.push_str(&(usage.prompt_tokens + usage.completion_tokens).to_string());
    out.push_str(",\"prompt_tokens_details\":{\"cached_tokens\":");
    out.push_str(&cached_tokens.to_string());
    out.push_str(",\"cache_write_tokens\":");
    out.push_str(&cache_write_tokens.to_string());
    out.push_str("}}");
}

fn append_responses_usage_json(out: &mut String, usage: OpenAiUsage) {
    let cached_tokens = clamp_usage_tokens(usage.cache_read_tokens, usage.prompt_tokens);
    let cache_write_tokens = clamp_usage_tokens(
        usage.cache_write_tokens,
        usage.prompt_tokens - cached_tokens,
    );
    out.push_str("{\"input_tokens\":");
    out.push_str(&usage.prompt_tokens.to_string());
    out.push_str(",\"input_tokens_details\":{\"cached_tokens\":");
    out.push_str(&cached_tokens.to_string());
    out.push_str(",\"cache_write_tokens\":");
    out.push_str(&cache_write_tokens.to_string());
    out.push_str("},\"output_tokens\":");
    out.push_str(&usage.completion_tokens.to_string());
    out.push_str(",\"output_tokens_details\":{\"reasoning_tokens\":0},\"total_tokens\":");
    out.push_str(&(usage.prompt_tokens + usage.completion_tokens).to_string());
    out.push('}');
}

fn append_anthropic_usage_json(out: &mut String, usage: OpenAiUsage) {
    let cache_read_tokens = clamp_usage_tokens(usage.cache_read_tokens, usage.prompt_tokens);
    let cache_write_tokens = clamp_usage_tokens(
        usage.cache_write_tokens,
        usage.prompt_tokens - cache_read_tokens,
    );
    let input_tokens = (usage.prompt_tokens - cache_read_tokens - cache_write_tokens).max(0);
    out.push_str("{\"input_tokens\":");
    out.push_str(&input_tokens.to_string());
    out.push_str(",\"output_tokens\":");
    out.push_str(&usage.completion_tokens.to_string());
    out.push_str(",\"cache_read_input_tokens\":");
    out.push_str(&cache_read_tokens.to_string());
    out.push_str(",\"cache_creation_input_tokens\":");
    out.push_str(&cache_write_tokens.to_string());
    out.push('}');
}

fn append_openai_tool_calls_json(out: &mut String, id_prefix: &str, tool_calls: &[DsmlJsonCall]) {
    out.push('[');
    for (index, call) in tool_calls.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        let default_id = format!("{id_prefix}_tool_{index}");
        out.push_str("{\"id\":");
        out.push_str(&json_escape_string(
            call.id.as_deref().unwrap_or(&default_id),
        ));
        out.push_str(",\"type\":\"function\",\"function\":{\"name\":");
        out.push_str(&json_escape_string(&call.name));
        out.push_str(",\"arguments\":");
        out.push_str(&json_escape_string(&normalize_json_object_or_empty(
            &call.arguments,
        )));
        out.push_str("}}");
    }
    out.push(']');
}

fn append_openai_tool_call_deltas_json(
    out: &mut String,
    id_prefix: &str,
    tool_calls: &[DsmlJsonCall],
) {
    out.push('[');
    for (index, call) in tool_calls.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        let default_id = format!("{id_prefix}_tool_{index}");
        out.push_str("{\"index\":");
        out.push_str(&index.to_string());
        out.push_str(",\"id\":");
        out.push_str(&json_escape_string(
            call.id.as_deref().unwrap_or(&default_id),
        ));
        out.push_str(",\"type\":\"function\",\"function\":{\"name\":");
        out.push_str(&json_escape_string(&call.name));
        out.push_str(",\"arguments\":");
        out.push_str(&json_escape_string(&normalize_json_object_or_empty(
            &call.arguments,
        )));
        out.push_str("}}");
    }
    out.push(']');
}

struct ResponsesToolItem {
    function_call_id: String,
    call_id: String,
}

fn responses_tool_item(
    response: &ResponsesFinalResponse<'_>,
    call: &DsmlJsonCall,
    index: usize,
) -> ResponsesToolItem {
    ResponsesToolItem {
        function_call_id: format!("{}{}", response.function_call_id_prefix, index),
        call_id: call
            .id
            .clone()
            .unwrap_or_else(|| format!("{}{}", response.call_id_prefix, index)),
    }
}

fn append_responses_function_call_item(
    out: &mut String,
    call: &DsmlJsonCall,
    item: &ResponsesToolItem,
    item_status: &str,
    with_args: bool,
    tool_orders: &[ToolSchemaOrder],
) {
    let order = tool_order_for_name(tool_orders, &call.name);
    if responses_tool_call_is_tool_search(call, order) {
        out.push_str("{\"id\":");
        out.push_str(&json_escape_string(&item.function_call_id));
        out.push_str(",\"type\":\"tool_search_call\",\"status\":");
        out.push_str(&json_escape_string(item_status));
        out.push_str(",\"call_id\":");
        out.push_str(&json_escape_string(&item.call_id));
        out.push_str(",\"execution\":\"client\",\"arguments\":");
        if with_args {
            out.push_str(&normalize_json_object_or_empty(&call.arguments));
        } else {
            out.push_str("{}");
        }
        out.push('}');
        return;
    }

    out.push_str("{\"id\":");
    out.push_str(&json_escape_string(&item.function_call_id));
    out.push_str(",\"type\":\"function_call\",\"status\":");
    out.push_str(&json_escape_string(item_status));
    out.push_str(",\"name\":");
    out.push_str(&json_escape_string(
        order
            .and_then(|order| order.wire_name.as_deref())
            .unwrap_or(&call.name),
    ));
    if let Some(namespace) = order.and_then(|order| order.namespace.as_deref()) {
        out.push_str(",\"namespace\":");
        out.push_str(&json_escape_string(namespace));
    }
    out.push_str(",\"call_id\":");
    out.push_str(&json_escape_string(&item.call_id));
    out.push_str(",\"arguments\":");
    if with_args {
        out.push_str(&json_escape_string(&normalize_json_object_or_empty(
            &call.arguments,
        )));
    } else {
        out.push_str("\"\"");
    }
    out.push('}');
}

fn append_responses_sse_event_body(out: &mut String, sequence: &mut i32, body: &str) {
    out.push_str("data: ");
    if let Some(type_close) = responses_event_type_close(body) {
        out.push_str(&body[..type_close]);
        out.push_str(",\"sequence_number\":");
        out.push_str(&sequence.to_string());
        out.push_str(&body[type_close..]);
    } else {
        out.push_str(body);
    }
    out.push_str("\n\n");
    *sequence += 1;
}

fn responses_event_type_close(body: &str) -> Option<usize> {
    let tail = body.strip_prefix("{\"type\":\"")?;
    let mut escaped = false;
    for (offset, ch) in tail.char_indices() {
        if escaped {
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == '"' {
            return Some("{\"type\":\"".len() + offset + 1);
        }
    }
    None
}

fn append_responses_reasoning_events(
    out: &mut String,
    sequence: &mut i32,
    response: &ResponsesFinalResponse<'_>,
    output_index: usize,
    reasoning: &str,
    reasoning_closed: bool,
) {
    let mut body = String::new();
    body.push_str("{\"type\":\"response.output_item.added\",\"output_index\":");
    body.push_str(&output_index.to_string());
    body.push_str(",\"item\":{\"id\":");
    body.push_str(&json_escape_string(response.reasoning_id));
    body.push_str(",\"type\":\"reasoning\",\"status\":\"in_progress\",\"summary\":[]}}");
    append_responses_sse_event_body(out, sequence, &body);

    body.clear();
    body.push_str("{\"type\":\"response.reasoning_summary_part.added\",\"item_id\":");
    body.push_str(&json_escape_string(response.reasoning_id));
    body.push_str(",\"output_index\":");
    body.push_str(&output_index.to_string());
    body.push_str(",\"summary_index\":0,\"part\":{\"type\":\"summary_text\",\"text\":\"\"}}");
    append_responses_sse_event_body(out, sequence, &body);

    body.clear();
    body.push_str("{\"type\":\"response.reasoning_summary_text.delta\",\"item_id\":");
    body.push_str(&json_escape_string(response.reasoning_id));
    body.push_str(",\"output_index\":");
    body.push_str(&output_index.to_string());
    body.push_str(",\"summary_index\":0,\"delta\":");
    body.push_str(&json_escape_string(reasoning));
    body.push('}');
    append_responses_sse_event_body(out, sequence, &body);

    body.clear();
    body.push_str("{\"type\":\"response.reasoning_summary_text.done\",\"item_id\":");
    body.push_str(&json_escape_string(response.reasoning_id));
    body.push_str(",\"output_index\":");
    body.push_str(&output_index.to_string());
    body.push_str(",\"summary_index\":0,\"text\":");
    body.push_str(&json_escape_string(reasoning));
    body.push('}');
    append_responses_sse_event_body(out, sequence, &body);

    body.clear();
    body.push_str("{\"type\":\"response.reasoning_summary_part.done\",\"item_id\":");
    body.push_str(&json_escape_string(response.reasoning_id));
    body.push_str(",\"output_index\":");
    body.push_str(&output_index.to_string());
    body.push_str(",\"summary_index\":0,\"part\":{\"type\":\"summary_text\",\"text\":");
    body.push_str(&json_escape_string(reasoning));
    body.push_str("}}");
    append_responses_sse_event_body(out, sequence, &body);

    let reasoning_status = if reasoning_closed {
        "completed"
    } else {
        "incomplete"
    };
    body.clear();
    body.push_str("{\"type\":\"response.output_item.done\",\"output_index\":");
    body.push_str(&output_index.to_string());
    body.push_str(",\"item\":{\"id\":");
    body.push_str(&json_escape_string(response.reasoning_id));
    body.push_str(",\"type\":\"reasoning\",\"status\":");
    body.push_str(&json_escape_string(reasoning_status));
    body.push_str(",\"summary\":[{\"type\":\"summary_text\",\"text\":");
    body.push_str(&json_escape_string(reasoning));
    body.push_str("}]}}");
    append_responses_sse_event_body(out, sequence, &body);
}

fn append_responses_message_events(
    out: &mut String,
    sequence: &mut i32,
    response: &ResponsesFinalResponse<'_>,
    output_index: usize,
    item_status: &str,
) {
    let mut body = String::new();
    body.push_str("{\"type\":\"response.output_item.added\",\"output_index\":");
    body.push_str(&output_index.to_string());
    body.push_str(",\"item\":{\"id\":");
    body.push_str(&json_escape_string(response.message_id));
    body.push_str(
        ",\"type\":\"message\",\"status\":\"in_progress\",\"role\":\"assistant\",\"content\":[]}}",
    );
    append_responses_sse_event_body(out, sequence, &body);

    body.clear();
    body.push_str("{\"type\":\"response.content_part.added\",\"item_id\":");
    body.push_str(&json_escape_string(response.message_id));
    body.push_str(",\"output_index\":");
    body.push_str(&output_index.to_string());
    body.push_str(
        ",\"content_index\":0,\"part\":{\"type\":\"output_text\",\"text\":\"\",\"annotations\":[]}}",
    );
    append_responses_sse_event_body(out, sequence, &body);

    body.clear();
    body.push_str("{\"type\":\"response.output_text.delta\",\"item_id\":");
    body.push_str(&json_escape_string(response.message_id));
    body.push_str(",\"output_index\":");
    body.push_str(&output_index.to_string());
    body.push_str(",\"content_index\":0,\"delta\":");
    body.push_str(&json_escape_string(response.content));
    body.push('}');
    append_responses_sse_event_body(out, sequence, &body);

    body.clear();
    body.push_str("{\"type\":\"response.output_text.done\",\"item_id\":");
    body.push_str(&json_escape_string(response.message_id));
    body.push_str(",\"output_index\":");
    body.push_str(&output_index.to_string());
    body.push_str(",\"content_index\":0,\"text\":");
    body.push_str(&json_escape_string(response.content));
    body.push('}');
    append_responses_sse_event_body(out, sequence, &body);

    body.clear();
    body.push_str("{\"type\":\"response.content_part.done\",\"item_id\":");
    body.push_str(&json_escape_string(response.message_id));
    body.push_str(",\"output_index\":");
    body.push_str(&output_index.to_string());
    body.push_str(",\"content_index\":0,\"part\":{\"type\":\"output_text\",\"text\":");
    body.push_str(&json_escape_string(response.content));
    body.push_str(",\"annotations\":[]}}");
    append_responses_sse_event_body(out, sequence, &body);

    body.clear();
    body.push_str("{\"type\":\"response.output_item.done\",\"output_index\":");
    body.push_str(&output_index.to_string());
    body.push_str(",\"item\":{\"id\":");
    body.push_str(&json_escape_string(response.message_id));
    body.push_str(",\"type\":\"message\",\"status\":");
    body.push_str(&json_escape_string(item_status));
    body.push_str(",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":");
    body.push_str(&json_escape_string(response.content));
    body.push_str(",\"annotations\":[]}]}}");
    append_responses_sse_event_body(out, sequence, &body);
}

fn append_responses_function_call_event(
    out: &mut String,
    sequence: &mut i32,
    call: &DsmlJsonCall,
    item: &ResponsesToolItem,
    output_index: usize,
    item_status: &str,
    with_args: bool,
    tool_orders: &[ToolSchemaOrder],
) {
    let mut body = String::new();
    body.push_str("{\"type\":\"response.output_item.");
    body.push_str(if with_args { "done" } else { "added" });
    body.push_str("\",\"output_index\":");
    body.push_str(&output_index.to_string());
    body.push_str(",\"item\":");
    append_responses_function_call_item(&mut body, call, item, item_status, with_args, tool_orders);
    body.push('}');
    append_responses_sse_event_body(out, sequence, &body);
}

fn append_responses_function_call_argument_events(
    out: &mut String,
    sequence: &mut i32,
    call: &DsmlJsonCall,
    item: &ResponsesToolItem,
    output_index: usize,
    tool_orders: &[ToolSchemaOrder],
) {
    let order = tool_order_for_name(tool_orders, &call.name);
    if responses_tool_call_is_tool_search(call, order) {
        return;
    }
    let arguments = json_escape_string(&normalize_json_object_or_empty(&call.arguments));
    let mut body = String::new();
    body.push_str("{\"type\":\"response.function_call_arguments.delta\",\"item_id\":");
    body.push_str(&json_escape_string(&item.function_call_id));
    body.push_str(",\"output_index\":");
    body.push_str(&output_index.to_string());
    body.push_str(",\"delta\":");
    body.push_str(&arguments);
    body.push('}');
    append_responses_sse_event_body(out, sequence, &body);

    body.clear();
    body.push_str("{\"type\":\"response.function_call_arguments.done\",\"item_id\":");
    body.push_str(&json_escape_string(&item.function_call_id));
    body.push_str(",\"output_index\":");
    body.push_str(&output_index.to_string());
    body.push_str(",\"name\":");
    body.push_str(&json_escape_string(
        order
            .and_then(|order| order.wire_name.as_deref())
            .unwrap_or(&call.name),
    ));
    if let Some(namespace) = order.and_then(|order| order.namespace.as_deref()) {
        body.push_str(",\"namespace\":");
        body.push_str(&json_escape_string(namespace));
    }
    body.push_str(",\"arguments\":");
    body.push_str(&arguments);
    body.push('}');
    append_responses_sse_event_body(out, sequence, &body);
}

fn append_responses_terminal_event(
    out: &mut String,
    sequence: &mut i32,
    response: &ResponsesFinalResponse<'_>,
    reasoning_closed: bool,
    reasoning_index: Option<usize>,
    message_index: Option<usize>,
    tool_calls: &[DsmlJsonCall],
) {
    let event_type = responses_terminal_event_type(response.finish_reason);
    let status = responses_status_for_finish(response.finish_reason);
    let item_status = responses_item_status_for_finish(response.finish_reason);
    let mut body = String::new();
    body.push_str("{\"type\":");
    body.push_str(&json_escape_string(event_type));
    body.push_str(",\"response\":{\"id\":");
    body.push_str(&json_escape_string(response.id));
    body.push_str(",\"object\":\"response\",\"created_at\":");
    body.push_str(&response.created_at.to_string());
    body.push_str(",\"status\":");
    body.push_str(&json_escape_string(status));
    body.push_str(",\"model\":");
    body.push_str(&json_escape_string(response.model));
    match response.finish_reason {
        "error" => body
            .push_str(",\"error\":{\"code\":\"server_error\",\"message\":\"generation failed\"}"),
        "length" => body.push_str(",\"incomplete_details\":{\"reason\":\"max_tokens\"}"),
        _ => {}
    }
    body.push_str(",\"output\":[");
    let mut wrote = false;
    if let (Some(_), Some(reasoning)) = (
        reasoning_index,
        response
            .reasoning
            .filter(|reasoning| response.reasoning_summary_emit && !reasoning.is_empty()),
    ) {
        append_separator(&mut body, &mut wrote);
        let reasoning_status = if reasoning_closed {
            "completed"
        } else {
            "incomplete"
        };
        body.push_str("{\"id\":");
        body.push_str(&json_escape_string(response.reasoning_id));
        body.push_str(",\"type\":\"reasoning\",\"status\":");
        body.push_str(&json_escape_string(reasoning_status));
        body.push_str(",\"summary\":[{\"type\":\"summary_text\",\"text\":");
        body.push_str(&json_escape_string(reasoning));
        body.push_str("}]}");
    }
    if message_index.is_some() {
        append_separator(&mut body, &mut wrote);
        body.push_str("{\"id\":");
        body.push_str(&json_escape_string(response.message_id));
        body.push_str(",\"type\":\"message\",\"status\":");
        body.push_str(&json_escape_string(item_status));
        body.push_str(",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":");
        body.push_str(&json_escape_string(response.content));
        body.push_str(",\"annotations\":[]}]}");
    }
    for (index, call) in tool_calls.iter().enumerate() {
        append_separator(&mut body, &mut wrote);
        let item = responses_tool_item(response, call, index);
        append_responses_function_call_item(
            &mut body,
            call,
            &item,
            item_status,
            true,
            response.tool_orders,
        );
    }
    body.push_str("],\"usage\":");
    append_responses_usage_json(&mut body, response.usage);
    body.push_str("}}");
    append_responses_sse_event_body(out, sequence, &body);
}

fn responses_terminal_event_type(finish_reason: &str) -> &'static str {
    match finish_reason {
        "error" => "response.failed",
        "length" => "response.incomplete",
        _ => "response.completed",
    }
}

fn append_anthropic_content(
    out: &mut String,
    content: &str,
    reasoning: Option<&str>,
    tool_calls: &[DsmlJsonCall],
    id_prefix: &str,
) {
    out.push('[');
    let mut wrote = false;
    let mut wrote_after_thinking = false;
    if let Some(reasoning) = reasoning.filter(|reasoning| !reasoning.is_empty()) {
        append_separator(out, &mut wrote);
        out.push_str("{\"type\":\"thinking\",\"thinking\":");
        out.push_str(&json_escape_string(reasoning));
        out.push_str(",\"signature\":");
        out.push_str(&json_escape_string(id_prefix));
        out.push('}');
    }
    if !content.is_empty() {
        append_separator(out, &mut wrote);
        out.push_str("{\"type\":\"text\",\"text\":");
        out.push_str(&json_escape_string(content));
        out.push('}');
        wrote_after_thinking = true;
    }
    for (index, call) in tool_calls.iter().enumerate() {
        append_separator(out, &mut wrote);
        append_anthropic_tool_use(out, call, id_prefix, index);
        wrote_after_thinking = true;
    }
    if !wrote || (reasoning.is_some_and(|reasoning| !reasoning.is_empty()) && !wrote_after_thinking)
    {
        append_separator(out, &mut wrote);
        out.push_str("{\"type\":\"text\",\"text\":\"\"}");
    }
    out.push(']');
}

fn append_anthropic_start_usage_json(out: &mut String, usage: OpenAiUsage) {
    let start_usage = OpenAiUsage {
        completion_tokens: 0,
        ..usage
    };
    append_anthropic_usage_json(out, start_usage);
}

fn append_anthropic_thinking_stream_block(
    out: &mut String,
    index: usize,
    signature: &str,
    reasoning: &str,
) {
    let mut body = String::new();
    body.push_str("{\"type\":\"content_block_start\",\"index\":");
    body.push_str(&index.to_string());
    body.push_str(
        ",\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"\"}}",
    );
    append_sse_event(out, "content_block_start", &body);

    body.clear();
    body.push_str("{\"type\":\"content_block_delta\",\"index\":");
    body.push_str(&index.to_string());
    body.push_str(",\"delta\":{\"type\":\"thinking_delta\",\"thinking\":");
    body.push_str(&json_escape_string(reasoning));
    body.push_str("}}");
    append_sse_event(out, "content_block_delta", &body);

    body.clear();
    body.push_str("{\"type\":\"content_block_delta\",\"index\":");
    body.push_str(&index.to_string());
    body.push_str(",\"delta\":{\"type\":\"signature_delta\",\"signature\":");
    body.push_str(&json_escape_string(signature));
    body.push_str("}}");
    append_sse_event(out, "content_block_delta", &body);

    append_anthropic_content_block_stop(out, index);
}

fn append_anthropic_text_stream_block(out: &mut String, index: usize, text: &str) {
    let mut body = String::new();
    body.push_str("{\"type\":\"content_block_start\",\"index\":");
    body.push_str(&index.to_string());
    body.push_str(",\"content_block\":{\"type\":\"text\",\"text\":\"\"}}");
    append_sse_event(out, "content_block_start", &body);

    body.clear();
    body.push_str("{\"type\":\"content_block_delta\",\"index\":");
    body.push_str(&index.to_string());
    body.push_str(",\"delta\":{\"type\":\"text_delta\",\"text\":");
    body.push_str(&json_escape_string(text));
    body.push_str("}}");
    append_sse_event(out, "content_block_delta", &body);

    append_anthropic_content_block_stop(out, index);
}

fn append_anthropic_empty_text_stream_block(out: &mut String, index: usize) {
    let mut body = String::new();
    body.push_str("{\"type\":\"content_block_start\",\"index\":");
    body.push_str(&index.to_string());
    body.push_str(",\"content_block\":{\"type\":\"text\",\"text\":\"\"}}");
    append_sse_event(out, "content_block_start", &body);
    append_anthropic_content_block_stop(out, index);
}

fn append_anthropic_tool_stream_block(
    out: &mut String,
    block_index: usize,
    id_prefix: &str,
    call_index: usize,
    call: &DsmlJsonCall,
) {
    let default_id = format!("toolu_{id_prefix}_{call_index}");
    let mut body = String::new();
    body.push_str("{\"type\":\"content_block_start\",\"index\":");
    body.push_str(&block_index.to_string());
    body.push_str(",\"content_block\":{\"type\":\"tool_use\",\"id\":");
    body.push_str(&json_escape_string(
        call.id.as_deref().unwrap_or(&default_id),
    ));
    body.push_str(",\"name\":");
    body.push_str(&json_escape_string(&call.name));
    body.push_str(",\"input\":{}}}");
    append_sse_event(out, "content_block_start", &body);

    body.clear();
    body.push_str("{\"type\":\"content_block_delta\",\"index\":");
    body.push_str(&block_index.to_string());
    body.push_str(",\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":");
    body.push_str(&json_escape_string(&normalize_json_object_or_empty(
        &call.arguments,
    )));
    body.push_str("}}");
    append_sse_event(out, "content_block_delta", &body);

    append_anthropic_content_block_stop(out, block_index);
}

fn append_anthropic_content_block_stop(out: &mut String, index: usize) {
    let mut body = String::new();
    body.push_str("{\"type\":\"content_block_stop\",\"index\":");
    body.push_str(&index.to_string());
    body.push('}');
    append_sse_event(out, "content_block_stop", &body);
}

fn append_sse_event(out: &mut String, event: &str, data: &str) {
    out.push_str("event: ");
    out.push_str(event);
    out.push_str("\ndata: ");
    out.push_str(data);
    out.push_str("\n\n");
}

fn append_anthropic_tool_use(out: &mut String, call: &DsmlJsonCall, id_prefix: &str, index: usize) {
    let default_id = format!("toolu_{id_prefix}_{index}");
    out.push_str("{\"type\":\"tool_use\",\"id\":");
    out.push_str(&json_escape_string(
        call.id.as_deref().unwrap_or(&default_id),
    ));
    out.push_str(",\"name\":");
    out.push_str(&json_escape_string(&call.name));
    out.push_str(",\"input\":");
    out.push_str(&normalize_json_object_or_empty(&call.arguments));
    out.push('}');
}

fn responses_tool_call_is_tool_search(
    call: &DsmlJsonCall,
    order: Option<&ToolSchemaOrder>,
) -> bool {
    call.name == "tool_search" && order.is_none_or(|order| order.responses_tool_search)
}

fn tool_order_for_name<'a>(
    tool_orders: &'a [ToolSchemaOrder],
    name: &str,
) -> Option<&'a ToolSchemaOrder> {
    tool_orders.iter().find(|order| order.name == name)
}

fn responses_status_for_finish(finish_reason: &str) -> &'static str {
    match finish_reason {
        "length" => "incomplete",
        "error" => "failed",
        _ => "completed",
    }
}

fn responses_item_status_for_finish(finish_reason: &str) -> &'static str {
    match finish_reason {
        "length" | "error" => "incomplete",
        _ => "completed",
    }
}

fn anthropic_stop_reason(finish_reason: &str) -> &'static str {
    match finish_reason {
        "tool_calls" => "tool_use",
        "length" => "max_tokens",
        _ => "end_turn",
    }
}

fn append_separator(out: &mut String, wrote: &mut bool) {
    if *wrote {
        out.push(',');
    }
    *wrote = true;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DsmlToolSyntax {
    tool_calls_start: &'static [u8],
    tool_calls_end: &'static [u8],
    invoke_start: &'static [u8],
    invoke_end: &'static [u8],
    param_start: &'static [u8],
    param_end: &'static [u8],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DsmlToolStreamState {
    Search,
    BetweenInvokes,
    BetweenParams,
    ParamValue,
    Done,
    Error,
}

fn find_tool_start_from(raw: &[u8], start: usize) -> Option<(usize, DsmlToolSyntax)> {
    let mut best: Option<(usize, DsmlToolSyntax)> = None;
    for syntax in DSML_TOOL_SYNTAXES {
        if let Some(rel) = find_bytes(raw.get(start..)?, syntax.tool_calls_start) {
            let pos = start + rel;
            if best.is_none_or(|(best_pos, _)| pos < best_pos) {
                best = Some((pos, *syntax));
            }
        }
    }
    best
}

fn max_tool_start_len() -> usize {
    DSML_TOOL_SYNTAXES
        .iter()
        .map(|syntax| syntax.tool_calls_start.len())
        .max()
        .unwrap_or(0)
}

fn raw_full_lit(raw: &[u8], pos: usize, lit: &[u8]) -> bool {
    pos <= raw.len() && raw[pos..].starts_with(lit)
}

fn raw_partial_lit(raw: &[u8], pos: usize, lit: &[u8]) -> bool {
    if pos > raw.len() || raw.len() - pos >= lit.len() {
        return false;
    }
    lit.starts_with(&raw[pos..])
}

fn raw_partial_any(raw: &[u8], pos: usize, a: &[u8], b: &[u8]) -> bool {
    raw_partial_lit(raw, pos, a) || raw_partial_lit(raw, pos, b)
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn parse_dsml_attr(tag: &[u8], name: &[u8]) -> Option<String> {
    let mut pattern = Vec::with_capacity(name.len() + 2);
    pattern.extend_from_slice(name);
    pattern.extend_from_slice(b"=\"");
    let start = find_bytes(tag, &pattern)? + pattern.len();
    let end = tag[start..].iter().position(|&byte| byte == b'"')? + start;
    let raw = String::from_utf8_lossy(&tag[start..end]);
    Some(dsml_unescape_text(&raw))
}

fn tool_param_value_stream_safe_len(
    raw: &[u8],
    start: usize,
    raw_len: usize,
    param_end: &[u8],
    is_string: bool,
) -> usize {
    let raw_len = raw_len.min(raw.len());
    let mut limit = raw_len;
    let scan = if raw_len > start + param_end.len() {
        raw_len - param_end.len()
    } else {
        start
    };
    for i in (scan + 1..=raw_len).rev() {
        if raw[i - 1] != b'<' {
            continue;
        }
        let marker = i - 1;
        let tail = raw_len - marker;
        if tail < param_end.len() && param_end.starts_with(&raw[marker..raw_len]) {
            limit = marker;
        }
        break;
    }
    if is_string {
        limit = dsml_entity_stream_safe_len(raw, start, limit);
    }
    utf8_stream_safe_len(raw, start, limit, false)
}

fn dsml_entity_stream_safe_len(raw: &[u8], start: usize, limit: usize) -> usize {
    const ENTITIES: &[&[u8]] = &[b"&amp;", b"&lt;", b"&gt;", b"&quot;", b"&apos;"];
    let scan = limit.saturating_sub(6).max(start);
    for i in (scan + 1..=limit).rev() {
        if raw[i - 1] != b'&' {
            continue;
        }
        let amp = i - 1;
        let tail = limit - amp;
        if ENTITIES
            .iter()
            .any(|entity| tail < entity.len() && entity.starts_with(&raw[amp..limit]))
        {
            return amp;
        }
        break;
    }
    limit
}

fn dsml_unescape_text(value: &str) -> String {
    let mut out = String::new();
    let mut rest = value;
    while !rest.is_empty() {
        if let Some(tail) = rest.strip_prefix("&amp;") {
            out.push('&');
            rest = tail;
        } else if let Some(tail) = rest.strip_prefix("&lt;") {
            out.push('<');
            rest = tail;
        } else if let Some(tail) = rest.strip_prefix("&gt;") {
            out.push('>');
            rest = tail;
        } else if let Some(tail) = rest.strip_prefix("&quot;") {
            out.push('"');
            rest = tail;
        } else if let Some(tail) = rest.strip_prefix("&apos;") {
            out.push('\'');
            rest = tail;
        } else {
            let mut chars = rest.chars();
            let ch = chars.next().expect("non-empty string");
            out.push(ch);
            rest = chars.as_str();
        }
    }
    out
}

fn json_escape_fragment(value: &str) -> String {
    let escaped = json_escape_string(value);
    escaped[1..escaped.len() - 1].to_string()
}

fn clamp_usage_tokens(value: i32, max: i32) -> i32 {
    if value < 0 {
        0
    } else if max >= 0 && value > max {
        max
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHAT_BASIC: &str =
        include_str!("../../../ds4-parity/baselines/server-traces/m0.4/responses/chat_basic.json");
    const CHAT_THINKING_DISABLED: &str = include_str!(
        "../../../ds4-parity/baselines/server-traces/m0.4/responses/chat_thinking_disabled.json"
    );
    const CHAT_CACHE_SEED: &str = include_str!(
        "../../../ds4-parity/baselines/server-traces/m0.4/responses/chat_cache_seed.json"
    );
    const CHAT_CACHE_CONTINUATION: &str = include_str!(
        "../../../ds4-parity/baselines/server-traces/m0.4/responses/chat_cache_continuation.json"
    );
    const CHAT_BASIC_HEADERS: &str = include_str!(
        "../../../ds4-parity/baselines/server-traces/m0.4/headers/chat_basic.headers.txt"
    );
    const CHAT_STREAM: &str =
        include_str!("../../../ds4-parity/baselines/server-traces/m0.4/responses/chat_stream.sse");
    const CHAT_STREAM_HEADERS: &str = include_str!(
        "../../../ds4-parity/baselines/server-traces/m0.4/headers/chat_stream.headers.txt"
    );
    const CHAT_TOOL_CALL: &str = include_str!(
        "../../../ds4-parity/baselines/server-traces/m0.4/responses/chat_tool_call.json"
    );
    const CHAT_TOOL_CALL_HEADERS: &str = include_str!(
        "../../../ds4-parity/baselines/server-traces/m0.4/headers/chat_tool_call.headers.txt"
    );

    fn assert_ordered(haystack: &str, needles: &[&str]) {
        let mut last = 0;
        for needle in needles {
            let rel = haystack[last..]
                .find(needle)
                .unwrap_or_else(|| panic!("missing ordered needle {needle:?}"));
            last += rel + needle.len();
        }
    }

    #[test]
    fn formats_m04_non_streaming_chat_response_bodies() {
        assert_eq!(
            format_openai_chat_completion_json(&OpenAiChatCompletion {
                id: "chatcmpl-1",
                created: 1_779_416_174,
                model: "deepseek-chat",
                content: "baseline ready",
                reasoning_content: None,
                finish_reason: "stop",
                usage: OpenAiUsage::new(11, 3, 0, 11),
            }),
            CHAT_BASIC
        );
        assert_eq!(
            format_openai_chat_completion_json(&OpenAiChatCompletion {
                id: "chatcmpl-4",
                created: 1_779_416_176,
                model: "deepseek-v4-flash",
                content: "2",
                reasoning_content: None,
                finish_reason: "stop",
                usage: OpenAiUsage::new(15, 1, 0, 15),
            }),
            CHAT_THINKING_DISABLED
        );
    }

    #[test]
    fn formats_m04_cache_usage_details_from_explicit_inputs() {
        assert_eq!(
            format_openai_chat_completion_json(&OpenAiChatCompletion {
                id: "chatcmpl-5",
                created: 1_779_416_176,
                model: "deepseek-chat",
                content: "cache ready",
                reasoning_content: None,
                finish_reason: "stop",
                usage: OpenAiUsage::new(39, 2, 0, 39),
            }),
            CHAT_CACHE_SEED
        );
        assert_eq!(
            format_openai_chat_completion_json(&OpenAiChatCompletion {
                id: "chatcmpl-6",
                created: 1_779_416_176,
                model: "deepseek-chat",
                content: "cache continued",
                reasoning_content: None,
                finish_reason: "stop",
                usage: OpenAiUsage::new(50, 2, 41, 9),
            }),
            CHAT_CACHE_CONTINUATION
        );
    }

    #[test]
    fn formats_http_headers_with_c_content_length() {
        let response = format_openai_chat_completion_http(
            false,
            &OpenAiChatCompletion {
                id: "chatcmpl-1",
                created: 1_779_416_174,
                model: "deepseek-chat",
                content: "baseline ready",
                reasoning_content: None,
                finish_reason: "stop",
                usage: OpenAiUsage::new(11, 3, 0, 11),
            },
        );
        let (headers, body) = response.split_once("\r\n\r\n").expect("HTTP response");
        assert_eq!(headers.replace("\r\n", "\n") + "\n", CHAT_BASIC_HEADERS);
        assert_eq!(body, CHAT_BASIC);
    }

    #[test]
    fn formats_m04_tool_call_response_body() {
        let calls = [DsmlJsonCall {
            id: Some("call_74afa558e9694448bc8aef7aae54150d".to_string()),
            name: "list_files".to_string(),
            arguments: "{\"path\": \".\"}".to_string(),
        }];
        assert_eq!(
            format_openai_chat_tool_completion_json(
                &OpenAiChatCompletion {
                    id: "chatcmpl-3",
                    created: 1_779_416_175,
                    model: "deepseek-v4-flash",
                    content: "",
                    reasoning_content: None,
                    finish_reason: "tool_calls",
                    usage: OpenAiUsage::new(394, 42, 0, 394),
                },
                &calls
            ),
            CHAT_TOOL_CALL
        );
    }

    #[test]
    fn formats_m04_tool_call_http_headers() {
        let calls = [DsmlJsonCall {
            id: Some("call_74afa558e9694448bc8aef7aae54150d".to_string()),
            name: "list_files".to_string(),
            arguments: "{\"path\":\".\"}".to_string(),
        }];
        let response = format_openai_chat_tool_completion_http(
            false,
            &OpenAiChatCompletion {
                id: "chatcmpl-3",
                created: 1_779_416_175,
                model: "deepseek-v4-flash",
                content: "",
                reasoning_content: None,
                finish_reason: "tool_calls",
                usage: OpenAiUsage::new(394, 42, 0, 394),
            },
            &calls,
        );
        let (headers, body) = response.split_once("\r\n\r\n").expect("HTTP response");
        assert_eq!(headers.replace("\r\n", "\n") + "\n", CHAT_TOOL_CALL_HEADERS);
        assert_eq!(body, CHAT_TOOL_CALL);
    }

    #[test]
    fn tool_call_response_generates_ids_and_normalizes_arguments() {
        let calls = [
            DsmlJsonCall {
                id: None,
                name: "search \"docs\"".to_string(),
                arguments: "{\"query\":\"line\\nquote\",\"limit\": 2}".to_string(),
            },
            DsmlJsonCall {
                id: Some("call_exact".to_string()),
                name: "bad_args".to_string(),
                arguments: "not json".to_string(),
            },
        ];
        let json = format_openai_chat_tool_completion_json(
            &OpenAiChatCompletion {
                id: "chatcmpl-tool",
                created: 7,
                model: "model \"x\"",
                content: "before",
                reasoning_content: Some("why"),
                finish_reason: "tool_calls",
                usage: OpenAiUsage::new(5, 2, 0, 5),
            },
            &calls,
        );
        let generated = json.find("\"id\":\"chatcmpl-tool_tool_0\"").unwrap();
        let explicit = json.find("\"id\":\"call_exact\"").unwrap();
        assert!(generated < explicit);
        assert!(json.contains("\"name\":\"search \\\"docs\\\"\""));
        assert!(json
            .contains("\"arguments\":\"{\\\"query\\\":\\\"line\\\\nquote\\\",\\\"limit\\\":2}\""));
        assert!(json.contains("\"function\":{\"name\":\"bad_args\",\"arguments\":\"{}\"}"));
        assert!(
            json.contains("\"content\":\"before\",\"reasoning_content\":\"why\",\"tool_calls\"")
        );
    }

    #[test]
    fn formats_responses_final_response_body() {
        let orders = [ToolSchemaOrder {
            name: "bash".to_string(),
            wire_name: None,
            namespace: None,
            responses_tool_search: false,
            properties: vec!["command".to_string(), "description".to_string()],
        }];
        let calls = [DsmlJsonCall {
            id: None,
            name: "bash".to_string(),
            arguments: "{\"description\":\"list files\",\"command\":\"ls -la\",\"timeout\":10}"
                .to_string(),
        }];

        assert_eq!(
            format_responses_final_response_json(
                &ResponsesFinalResponse {
                    id: "resp_test",
                    created_at: 1234,
                    model: "deepseek-v4-flash",
                    content: "Hello.",
                    reasoning: Some("need a tool"),
                    reasoning_summary_emit: true,
                    finish_reason: "tool_calls",
                    usage: OpenAiUsage::new(10, 2, 7, 3),
                    reasoning_id: "rs_test",
                    message_id: "msg_test",
                    function_call_id_prefix: "fc_test_",
                    call_id_prefix: "call_test_",
                    tool_orders: &orders,
                },
                &calls,
            ),
            concat!(
                r#"{"id":"resp_test","object":"response","created_at":1234,"status":"completed","model":"deepseek-v4-flash","output":["#,
                r#"{"id":"rs_test","type":"reasoning","status":"completed","summary":[{"type":"summary_text","text":"need a tool"}]},"#,
                r#"{"id":"msg_test","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"Hello.","annotations":[]}]},"#,
                r#"{"id":"fc_test_0","type":"function_call","status":"completed","name":"bash","call_id":"call_test_0","arguments":"{\"description\":\"list files\",\"command\":\"ls -la\",\"timeout\":10}"}"#,
                r#"],"usage":{"input_tokens":10,"input_tokens_details":{"cached_tokens":7,"cache_write_tokens":3},"output_tokens":2,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":12}}"#
            )
        );
    }

    #[test]
    fn responses_final_restores_namespace_and_tool_search_shapes() {
        let orders = [
            ToolSchemaOrder {
                name: "mcp__perplexity__perplexity_search".to_string(),
                wire_name: Some("perplexity_search".to_string()),
                namespace: Some("mcp__perplexity__".to_string()),
                responses_tool_search: false,
                properties: vec![],
            },
            ToolSchemaOrder {
                name: "tool_search".to_string(),
                wire_name: None,
                namespace: None,
                responses_tool_search: true,
                properties: vec![],
            },
        ];
        let calls = [
            DsmlJsonCall {
                id: Some("call_ns".to_string()),
                name: "mcp__perplexity__perplexity_search".to_string(),
                arguments: "{\"query\":\"deepseek\",\"recency\":7}".to_string(),
            },
            DsmlJsonCall {
                id: Some("call_search".to_string()),
                name: "tool_search".to_string(),
                arguments: "{\"limit\":3,\"query\":\"perplexity\"}".to_string(),
            },
        ];
        let json = format_responses_final_response_json(
            &ResponsesFinalResponse {
                id: "resp_tools",
                created_at: 1234,
                model: "deepseek-chat",
                content: "",
                reasoning: Some("hidden"),
                reasoning_summary_emit: false,
                finish_reason: "length",
                usage: OpenAiUsage::new(6, 4, 0, 9),
                reasoning_id: "rs_unused",
                message_id: "msg_unused",
                function_call_id_prefix: "fc_",
                call_id_prefix: "call_",
                tool_orders: &orders,
            },
            &calls,
        );

        assert!(json.contains(r#""status":"incomplete""#));
        assert!(json.contains(r#""incomplete_details":{"reason":"max_tokens"}"#));
        assert!(!json.contains("hidden"));
        assert!(json.contains(
            r#""type":"function_call","status":"incomplete","name":"perplexity_search","namespace":"mcp__perplexity__","call_id":"call_ns""#
        ));
        assert!(!json.contains("mcp__perplexity__perplexity_search"));
        assert!(json.contains(
            r#""type":"tool_search_call","status":"incomplete","call_id":"call_search","execution":"client","arguments":{"limit":3,"query":"perplexity"}"#
        ));
        assert!(
            json.contains(r#""input_tokens_details":{"cached_tokens":0,"cache_write_tokens":6}"#)
        );
    }

    #[test]
    fn responses_function_named_tool_search_stays_function_call_with_plain_order() {
        let orders = [ToolSchemaOrder {
            name: "tool_search".to_string(),
            wire_name: None,
            namespace: None,
            responses_tool_search: false,
            properties: vec![],
        }];
        let calls = [DsmlJsonCall {
            id: Some("call_user_tool_search".to_string()),
            name: "tool_search".to_string(),
            arguments: "{\"query\":\"plain function\"}".to_string(),
        }];
        let json = format_responses_final_response_json(
            &ResponsesFinalResponse {
                id: "resp_user_tool_search",
                created_at: 1234,
                model: "deepseek-chat",
                content: "",
                reasoning: None,
                reasoning_summary_emit: true,
                finish_reason: "stop",
                usage: OpenAiUsage::new(1, 1, 0, 0),
                reasoning_id: "rs_unused",
                message_id: "msg_unused",
                function_call_id_prefix: "fc_",
                call_id_prefix: "call_",
                tool_orders: &orders,
            },
            &calls,
        );

        assert!(json.contains(r#""type":"function_call""#));
        assert!(!json.contains(r#""type":"tool_search_call""#));
    }

    #[test]
    fn formats_anthropic_message_body() {
        let calls = [DsmlJsonCall {
            id: None,
            name: "bash".to_string(),
            arguments: "{\"description\":\"list files\",\"command\":\"ls -la\",\"timeout\":10}"
                .to_string(),
        }];

        assert_eq!(
            format_anthropic_message_json(
                &AnthropicMessageResponse {
                    id: "msg_test",
                    model: "deepseek-v4-flash",
                    content: "done",
                    reasoning: Some("thinking text"),
                    finish_reason: "tool_calls",
                    usage: OpenAiUsage::new(10, 2, 7, 3),
                },
                &calls,
            ),
            concat!(
                r#"{"id":"msg_test","type":"message","role":"assistant","model":"deepseek-v4-flash","content":["#,
                r#"{"type":"thinking","thinking":"thinking text","signature":"msg_test"},"#,
                r#"{"type":"text","text":"done"},"#,
                r#"{"type":"tool_use","id":"toolu_msg_test_0","name":"bash","input":{"description":"list files","command":"ls -la","timeout":10}}"#,
                r#"],"stop_reason":"tool_use","stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":2,"cache_read_input_tokens":7,"cache_creation_input_tokens":3}}"#,
                "\n"
            )
        );
    }

    #[test]
    fn anthropic_empty_or_reasoning_only_content_keeps_text_block() {
        let empty = format_anthropic_message_json(
            &AnthropicMessageResponse {
                id: "msg_empty",
                model: "deepseek-chat",
                content: "",
                reasoning: None,
                finish_reason: "stop",
                usage: OpenAiUsage::new(3, 1, 0, 0),
            },
            &[],
        );
        assert!(empty.contains(r#""content":[{"type":"text","text":""}]"#));
        assert!(empty.contains(r#""stop_reason":"end_turn""#));

        let reasoning_only = format_anthropic_message_json(
            &AnthropicMessageResponse {
                id: "msg_reasoning",
                model: "deepseek-chat",
                content: "",
                reasoning: Some("hidden"),
                finish_reason: "length",
                usage: OpenAiUsage::new(3, 1, 0, 0),
            },
            &[],
        );
        let thinking = reasoning_only.find(r#""type":"thinking""#).unwrap();
        let text = reasoning_only.find(r#""type":"text","text":"""#).unwrap();
        assert!(thinking < text);
        assert!(reasoning_only.contains(r#""stop_reason":"max_tokens""#));
    }

    #[test]
    fn formats_responses_stream_sse_lifecycle() {
        let orders = [ToolSchemaOrder {
            name: "bash".to_string(),
            wire_name: None,
            namespace: None,
            responses_tool_search: false,
            properties: vec![],
        }];
        let calls = [DsmlJsonCall {
            id: None,
            name: "bash".to_string(),
            arguments: "{\"description\":\"list files\",\"command\":\"ls -la\"}".to_string(),
        }];
        let body = format_responses_stream_sse(
            &ResponsesStreamResponse {
                response: ResponsesFinalResponse {
                    id: "resp_stream",
                    created_at: 1234,
                    model: "deepseek-chat",
                    content: "Hello.",
                    reasoning: Some("need a tool"),
                    reasoning_summary_emit: true,
                    finish_reason: "tool_calls",
                    usage: OpenAiUsage::new(10, 8, 7, 3),
                    reasoning_id: "rs_stream",
                    message_id: "msg_stream",
                    function_call_id_prefix: "fc_stream_",
                    call_id_prefix: "call_stream_",
                    tool_orders: &orders,
                },
                reasoning_closed: true,
            },
            &calls,
        );

        for sequence in 0..=17 {
            assert!(
                body.contains(&format!(r#""sequence_number":{sequence}"#)),
                "missing sequence {sequence}"
            );
        }
        assert_ordered(
            &body,
            &[
                r#""type":"response.created","sequence_number":0"#,
                r#""type":"response.output_item.added","sequence_number":1"#,
                r#""type":"response.reasoning_summary_part.added","sequence_number":2"#,
                r#""type":"response.reasoning_summary_text.delta","sequence_number":3"#,
                r#""type":"response.reasoning_summary_text.done","sequence_number":4"#,
                r#""type":"response.reasoning_summary_part.done","sequence_number":5"#,
                r#""type":"response.output_item.done","sequence_number":6"#,
                r#""type":"response.output_item.added","sequence_number":7"#,
                r#""type":"response.content_part.added","sequence_number":8"#,
                r#""type":"response.output_text.delta","sequence_number":9"#,
                r#""type":"response.output_text.done","sequence_number":10"#,
                r#""type":"response.content_part.done","sequence_number":11"#,
                r#""type":"response.output_item.done","sequence_number":12"#,
                r#""type":"response.output_item.added","sequence_number":13"#,
                r#""type":"response.function_call_arguments.delta","sequence_number":14"#,
                r#""type":"response.function_call_arguments.done","sequence_number":15"#,
                r#""type":"response.output_item.done","sequence_number":16"#,
                r#""type":"response.completed","sequence_number":17"#,
            ],
        );
        assert!(
            body.contains(r#""delta":"{\"description\":\"list files\",\"command\":\"ls -la\"}""#)
        );
        assert!(body
            .contains(r#""arguments":"{\"description\":\"list files\",\"command\":\"ls -la\"}""#));
        assert!(body.contains(r#""usage":{"input_tokens":10"#));
        assert!(!body.contains("[DONE]"));
    }

    #[test]
    fn responses_stream_tool_search_skips_argument_events_and_maps_length() {
        let orders = [ToolSchemaOrder {
            name: "tool_search".to_string(),
            wire_name: None,
            namespace: None,
            responses_tool_search: true,
            properties: vec![],
        }];
        let calls = [DsmlJsonCall {
            id: Some("call_search".to_string()),
            name: "tool_search".to_string(),
            arguments: "{\"limit\":3,\"query\":\"perplexity\"}".to_string(),
        }];
        let body = format_responses_stream_sse(
            &ResponsesStreamResponse {
                response: ResponsesFinalResponse {
                    id: "resp_search",
                    created_at: 1234,
                    model: "deepseek-chat",
                    content: "",
                    reasoning: None,
                    reasoning_summary_emit: true,
                    finish_reason: "length",
                    usage: OpenAiUsage::new(6, 4, 0, 9),
                    reasoning_id: "rs_unused",
                    message_id: "msg_unused",
                    function_call_id_prefix: "fc_",
                    call_id_prefix: "call_",
                    tool_orders: &orders,
                },
                reasoning_closed: true,
            },
            &calls,
        );

        assert_ordered(
            &body,
            &[
                r#""type":"response.created","sequence_number":0"#,
                r#""type":"response.output_item.added","sequence_number":1"#,
                r#""type":"response.output_item.done","sequence_number":2"#,
                r#""type":"response.incomplete","sequence_number":3"#,
            ],
        );
        assert!(!body.contains("response.function_call_arguments."));
        assert!(body.contains(r#""type":"tool_search_call","status":"in_progress""#));
        assert!(body.contains(r#""type":"tool_search_call","status":"incomplete""#));
        assert!(body.contains(r#""incomplete_details":{"reason":"max_tokens"}"#));
        assert!(
            body.contains(r#""input_tokens_details":{"cached_tokens":0,"cache_write_tokens":6}"#)
        );
    }

    #[test]
    fn responses_stream_marks_unclosed_reasoning_incomplete() {
        let body = format_responses_stream_sse(
            &ResponsesStreamResponse {
                response: ResponsesFinalResponse {
                    id: "resp_reasoning",
                    created_at: 1234,
                    model: "deepseek-chat",
                    content: "",
                    reasoning: Some("partial hidden"),
                    reasoning_summary_emit: true,
                    finish_reason: "stop",
                    usage: OpenAiUsage::new(3, 1, 0, 0),
                    reasoning_id: "rs_partial",
                    message_id: "msg_unused",
                    function_call_id_prefix: "fc_",
                    call_id_prefix: "call_",
                    tool_orders: &[],
                },
                reasoning_closed: false,
            },
            &[],
        );

        assert!(body.contains(r#""type":"response.completed""#));
        assert!(body.contains(r#""id":"rs_partial","type":"reasoning","status":"incomplete""#));
    }

    #[test]
    fn formats_anthropic_message_stream_sse_lifecycle() {
        let calls = [DsmlJsonCall {
            id: None,
            name: "bash".to_string(),
            arguments: "{\"description\":\"list files\",\"command\":\"ls -la\"}".to_string(),
        }];
        let body = format_anthropic_message_stream_sse(
            &AnthropicMessageResponse {
                id: "msg_stream",
                model: "deepseek-v4-flash",
                content: "done",
                reasoning: Some("thinking text"),
                finish_reason: "tool_calls",
                usage: OpenAiUsage::new(10, 2, 7, 3),
            },
            &calls,
        );

        assert_ordered(
            &body,
            &[
                "event: message_start",
                r#""usage":{"input_tokens":0,"output_tokens":0,"cache_read_input_tokens":7,"cache_creation_input_tokens":3}"#,
                r#""content_block":{"type":"thinking","thinking":"","signature":""}"#,
                r#""delta":{"type":"thinking_delta","thinking":"thinking text"}"#,
                r#""delta":{"type":"signature_delta","signature":"msg_stream"}"#,
                r#""type":"content_block_stop","index":0"#,
                r#""content_block":{"type":"text","text":""}"#,
                r#""delta":{"type":"text_delta","text":"done"}"#,
                r#""type":"content_block_stop","index":1"#,
                r#""content_block":{"type":"tool_use","id":"toolu_msg_stream_0","name":"bash","input":{}}"#,
                r#""delta":{"type":"input_json_delta","partial_json":"{\"description\":\"list files\",\"command\":\"ls -la\"}"}"#,
                r#""type":"content_block_stop","index":2"#,
                r#""type":"message_delta","delta":{"stop_reason":"tool_use""#,
                r#""usage":{"output_tokens":2}"#,
                "event: message_stop",
            ],
        );
        assert!(!body.contains("[DONE]"));
    }

    #[test]
    fn anthropic_stream_reasoning_only_adds_empty_text_block() {
        let body = format_anthropic_message_stream_sse(
            &AnthropicMessageResponse {
                id: "msg_reasoning",
                model: "deepseek-chat",
                content: "",
                reasoning: Some("hidden"),
                finish_reason: "length",
                usage: OpenAiUsage::new(3, 1, 0, 0),
            },
            &[],
        );

        assert_ordered(
            &body,
            &[
                r#""content_block":{"type":"thinking","thinking":"","signature":""}"#,
                r#""delta":{"type":"thinking_delta","thinking":"hidden"}"#,
                r#""type":"content_block_stop","index":0"#,
                r#""content_block":{"type":"text","text":""}"#,
                r#""type":"content_block_stop","index":1"#,
                r#""type":"message_delta","delta":{"stop_reason":"max_tokens""#,
            ],
        );
    }

    #[test]
    fn protocol_stream_http_wrappers_use_sse_headers() {
        let responses_stream = ResponsesStreamResponse {
            response: ResponsesFinalResponse {
                id: "resp_http",
                created_at: 1234,
                model: "deepseek-chat",
                content: "ok",
                reasoning: None,
                reasoning_summary_emit: true,
                finish_reason: "stop",
                usage: OpenAiUsage::new(3, 1, 0, 0),
                reasoning_id: "rs_unused",
                message_id: "msg_http",
                function_call_id_prefix: "fc_",
                call_id_prefix: "call_",
                tool_orders: &[],
            },
            reasoning_closed: true,
        };
        let response_http = format_responses_stream_http(false, &responses_stream, &[]);
        let (headers, body) = response_http.split_once("\r\n\r\n").expect("HTTP response");
        assert_eq!(headers.replace("\r\n", "\n") + "\n", CHAT_STREAM_HEADERS);
        assert_eq!(body, format_responses_stream_sse(&responses_stream, &[]));

        let anthropic = AnthropicMessageResponse {
            id: "msg_http",
            model: "deepseek-chat",
            content: "ok",
            reasoning: None,
            finish_reason: "stop",
            usage: OpenAiUsage::new(3, 1, 0, 0),
        };
        let anthropic_http = format_anthropic_message_stream_http(false, &anthropic, &[]);
        let (headers, body) = anthropic_http
            .split_once("\r\n\r\n")
            .expect("HTTP response");
        assert_eq!(headers.replace("\r\n", "\n") + "\n", CHAT_STREAM_HEADERS);
        assert_eq!(body, format_anthropic_message_stream_sse(&anthropic, &[]));
    }

    #[test]
    fn formats_m04_chat_stream_sse_body() {
        assert_eq!(
            format_openai_chat_stream_sse(&OpenAiChatStream {
                id: "chatcmpl-2",
                created: 1_779_416_174,
                model: "deepseek-chat",
                content_deltas: &["stream", " baseline"],
                finish_reason: "stop",
                usage: Some(OpenAiUsage::new(11, 2, 0, 11)),
            }),
            CHAT_STREAM
        );
    }

    #[test]
    fn formats_m04_chat_stream_http_headers() {
        let response = format_openai_chat_stream_http(
            false,
            &OpenAiChatStream {
                id: "chatcmpl-2",
                created: 1_779_416_174,
                model: "deepseek-chat",
                content_deltas: &["stream", " baseline"],
                finish_reason: "stop",
                usage: Some(OpenAiUsage::new(11, 2, 0, 11)),
            },
        );
        let (headers, body) = response.split_once("\r\n\r\n").expect("HTTP response");
        assert_eq!(headers.replace("\r\n", "\n") + "\n", CHAT_STREAM_HEADERS);
        assert_eq!(body, CHAT_STREAM);
    }

    #[test]
    fn tool_stream_emits_start_argument_finish_and_usage_chunks() {
        let events = [
            OpenAiToolCallStreamEvent::Start {
                index: 0,
                id: "call_stream_0",
                name: "search \"docs\"",
            },
            OpenAiToolCallStreamEvent::Arguments {
                index: 0,
                fragment: "{\"query\":\"line\nquote\",",
            },
            OpenAiToolCallStreamEvent::Arguments {
                index: 0,
                fragment: "\"limit\":2}",
            },
        ];
        assert_eq!(
            format_openai_chat_tool_stream_sse(&OpenAiChatToolStream {
                id: "chatcmpl-tool-stream",
                created: 7,
                model: "model \"x\"",
                events: &events,
                finish_reason: "tool_calls",
                usage: Some(OpenAiUsage::new(5, 3, 1, 4)),
            }),
            concat!(
                r#"data: {"id":"chatcmpl-tool-stream","object":"chat.completion.chunk","created":7,"model":"model \"x\"","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#,
                "\n\n",
                r#"data: {"id":"chatcmpl-tool-stream","object":"chat.completion.chunk","created":7,"model":"model \"x\"","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_stream_0","type":"function","function":{"name":"search \"docs\"","arguments":""}}]},"finish_reason":null}]}"#,
                "\n\n",
                r#"data: {"id":"chatcmpl-tool-stream","object":"chat.completion.chunk","created":7,"model":"model \"x\"","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"query\":\"line\nquote\","}}]},"finish_reason":null}]}"#,
                "\n\n",
                r#"data: {"id":"chatcmpl-tool-stream","object":"chat.completion.chunk","created":7,"model":"model \"x\"","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"limit\":2}"}}]},"finish_reason":null}]}"#,
                "\n\n",
                r#"data: {"id":"chatcmpl-tool-stream","object":"chat.completion.chunk","created":7,"model":"model \"x\"","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
                "\n\n",
                r#"data: {"id":"chatcmpl-tool-stream","object":"chat.completion.chunk","created":7,"model":"model \"x\"","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8,"prompt_tokens_details":{"cached_tokens":1,"cache_write_tokens":4}}}"#,
                "\n\n",
                "data: [DONE]\n"
            )
        );
    }

    #[test]
    fn tool_stream_full_call_delta_generates_ids_and_normalizes_arguments() {
        let calls = [
            DsmlJsonCall {
                id: None,
                name: "list_files".to_string(),
                arguments: "{\"path\": \".\"}".to_string(),
            },
            DsmlJsonCall {
                id: Some("call_exact".to_string()),
                name: "bad_args".to_string(),
                arguments: "not json".to_string(),
            },
        ];
        let events = [OpenAiToolCallStreamEvent::FullCalls { calls: &calls }];
        assert_eq!(
            format_openai_chat_tool_stream_sse(&OpenAiChatToolStream {
                id: "chatcmpl-fallback",
                created: 11,
                model: "deepseek-chat",
                events: &events,
                finish_reason: "tool_calls",
                usage: None,
            }),
            concat!(
                r#"data: {"id":"chatcmpl-fallback","object":"chat.completion.chunk","created":11,"model":"deepseek-chat","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#,
                "\n\n",
                r#"data: {"id":"chatcmpl-fallback","object":"chat.completion.chunk","created":11,"model":"deepseek-chat","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"chatcmpl-fallback_tool_0","type":"function","function":{"name":"list_files","arguments":"{\"path\":\".\"}"}},{"index":1,"id":"call_exact","type":"function","function":{"name":"bad_args","arguments":"{}"}}]},"finish_reason":null}]}"#,
                "\n\n",
                r#"data: {"id":"chatcmpl-fallback","object":"chat.completion.chunk","created":11,"model":"deepseek-chat","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
                "\n\n",
                "data: [DONE]\n"
            )
        );
    }

    #[test]
    fn formats_tool_stream_http_headers() {
        let events = [OpenAiToolCallStreamEvent::Start {
            index: 0,
            id: "call_stream_0",
            name: "list_files",
        }];
        let stream = OpenAiChatToolStream {
            id: "chatcmpl-tool-stream",
            created: 7,
            model: "deepseek-chat",
            events: &events,
            finish_reason: "tool_calls",
            usage: None,
        };
        let response = format_openai_chat_tool_stream_http(false, &stream);
        let (headers, body) = response.split_once("\r\n\r\n").expect("HTTP response");
        assert_eq!(headers.replace("\r\n", "\n") + "\n", CHAT_STREAM_HEADERS);
        assert_eq!(body, format_openai_chat_tool_stream_sse(&stream));
    }

    #[test]
    fn tool_stream_can_emit_content_before_tool_deltas() {
        let events = [
            OpenAiToolCallStreamEvent::Content { delta: "Before." },
            OpenAiToolCallStreamEvent::Start {
                index: 0,
                id: "call_stream_0",
                name: "list_files",
            },
        ];
        let body = format_openai_chat_tool_stream_sse(&OpenAiChatToolStream {
            id: "chatcmpl-tool-stream",
            created: 7,
            model: "deepseek-chat",
            events: &events,
            finish_reason: "tool_calls",
            usage: None,
        });
        let role = body.find("\"role\":\"assistant\"").unwrap();
        let content = body.find("\"content\":\"Before.\"").unwrap();
        let tool = body.find("\"tool_calls\"").unwrap();
        assert!(role < content);
        assert!(content < tool);
    }

    #[test]
    fn tool_stream_translator_emits_string_argument_fragments_incrementally() {
        let mut translator = OpenAiToolCallStreamTranslator::new("call_0000000000000001");
        let mut events = Vec::new();
        events.extend(
            translator.feed("<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"bash\">\n".as_bytes()),
        );
        events.extend(
            translator
                .feed("<｜DSML｜parameter name=\"command\" string=\"true\">echo &l".as_bytes()),
        );
        events.extend(
            translator.feed(
                "t;ok</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>".as_bytes(),
            ),
        );

        assert_eq!(
            events,
            vec![
                OpenAiToolCallStreamEventOwned::Start {
                    index: 0,
                    id: "call_00000000000000010000000000000000".to_string(),
                    name: "bash".to_string(),
                },
                OpenAiToolCallStreamEventOwned::Arguments {
                    index: 0,
                    fragment: "{".to_string(),
                },
                OpenAiToolCallStreamEventOwned::Arguments {
                    index: 0,
                    fragment: "\"command\":\"".to_string(),
                },
                OpenAiToolCallStreamEventOwned::Arguments {
                    index: 0,
                    fragment: "echo ".to_string(),
                },
                OpenAiToolCallStreamEventOwned::Arguments {
                    index: 0,
                    fragment: "<ok".to_string(),
                },
                OpenAiToolCallStreamEventOwned::Arguments {
                    index: 0,
                    fragment: "\"".to_string(),
                },
                OpenAiToolCallStreamEventOwned::Arguments {
                    index: 0,
                    fragment: "}".to_string(),
                },
            ]
        );
        assert!(translator.emitted_any());
        assert!(translator.is_done());
        assert_eq!(
            translator.call_ids(),
            ["call_00000000000000010000000000000000"]
        );
    }

    #[test]
    fn tool_stream_translator_preserves_raw_json_arguments_and_commas() {
        let mut translator = OpenAiToolCallStreamTranslator::new("call_");
        let events = translator.feed(
            concat!(
                "<tool_calls>\n",
                "<invoke name=\"calc\">\n",
                "<parameter name=\"query\" string=\"true\">sum</parameter>\n",
                "<parameter name=\"config\" string=\"false\">{\"x\": 1}</parameter>\n",
                "</invoke>\n",
                "</tool_calls>",
            )
            .as_bytes(),
        );

        assert_eq!(
            events,
            vec![
                OpenAiToolCallStreamEventOwned::Start {
                    index: 0,
                    id: "call_0000000000000000".to_string(),
                    name: "calc".to_string(),
                },
                OpenAiToolCallStreamEventOwned::Arguments {
                    index: 0,
                    fragment: "{".to_string(),
                },
                OpenAiToolCallStreamEventOwned::Arguments {
                    index: 0,
                    fragment: "\"query\":\"".to_string(),
                },
                OpenAiToolCallStreamEventOwned::Arguments {
                    index: 0,
                    fragment: "sum".to_string(),
                },
                OpenAiToolCallStreamEventOwned::Arguments {
                    index: 0,
                    fragment: "\"".to_string(),
                },
                OpenAiToolCallStreamEventOwned::Arguments {
                    index: 0,
                    fragment: ",\"config\":".to_string(),
                },
                OpenAiToolCallStreamEventOwned::Arguments {
                    index: 0,
                    fragment: "{\"x\": 1}".to_string(),
                },
                OpenAiToolCallStreamEventOwned::Arguments {
                    index: 0,
                    fragment: "}".to_string(),
                },
            ]
        );
    }

    #[test]
    fn tool_stream_translator_holds_partial_tags_and_split_utf8() {
        let mut translator = OpenAiToolCallStreamTranslator::new("call_");
        assert!(translator
            .feed("<｜DSML｜tool_calls>\n<｜DSML｜inv".as_bytes())
            .is_empty());
        let start = translator.feed("oke name=\"note\">\n".as_bytes());
        assert_eq!(
            start,
            vec![
                OpenAiToolCallStreamEventOwned::Start {
                    index: 0,
                    id: "call_0000000000000000".to_string(),
                    name: "note".to_string(),
                },
                OpenAiToolCallStreamEventOwned::Arguments {
                    index: 0,
                    fragment: "{".to_string(),
                },
            ]
        );

        let mut param_chunk = "<｜DSML｜parameter name=\"text\" string=\"true\">caf"
            .as_bytes()
            .to_vec();
        param_chunk.push(0xc3);
        let param = translator.feed(&param_chunk);
        assert_eq!(
            param,
            vec![
                OpenAiToolCallStreamEventOwned::Arguments {
                    index: 0,
                    fragment: "\"text\":\"".to_string(),
                },
                OpenAiToolCallStreamEventOwned::Arguments {
                    index: 0,
                    fragment: "caf".to_string(),
                },
            ]
        );

        let mut close_start = vec![0xa9];
        close_start.extend_from_slice("</｜DSML｜par".as_bytes());
        assert_eq!(
            translator.feed(&close_start),
            vec![OpenAiToolCallStreamEventOwned::Arguments {
                index: 0,
                fragment: "é".to_string(),
            }]
        );
        let finish =
            translator.feed("ameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>".as_bytes());
        assert_eq!(
            finish,
            vec![
                OpenAiToolCallStreamEventOwned::Arguments {
                    index: 0,
                    fragment: "\"".to_string(),
                },
                OpenAiToolCallStreamEventOwned::Arguments {
                    index: 0,
                    fragment: "}".to_string(),
                },
            ]
        );
        assert!(translator.is_done());
        assert!(!translator.is_error());
    }

    #[test]
    fn stream_usage_chunk_is_optional() {
        let body = format_openai_chat_stream_sse(&OpenAiChatStream {
            id: "chatcmpl-no-usage",
            created: 1,
            model: "deepseek-chat",
            content_deltas: &["x"],
            finish_reason: "stop",
            usage: None,
        });
        assert!(!body.contains("\"usage\""));
        assert!(body.ends_with("data: [DONE]\n"));
    }

    #[test]
    fn stream_chunks_escape_model_delta_and_finish_reason() {
        assert_eq!(
            format_openai_chat_stream_sse(&OpenAiChatStream {
                id: "chatcmpl-stream-escape",
                created: 1,
                model: "model \"x\"",
                content_deltas: &["line\n\"quoted\""],
                finish_reason: "stop",
                usage: None,
            }),
            "data: {\"id\":\"chatcmpl-stream-escape\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"model \\\"x\\\"\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"chatcmpl-stream-escape\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"model \\\"x\\\"\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"line\\n\\\"quoted\\\"\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"chatcmpl-stream-escape\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"model \\\"x\\\"\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
data: [DONE]\n"
        );
    }

    #[test]
    fn usage_clamps_cache_details_like_c() {
        let json = format_openai_chat_completion_json(&OpenAiChatCompletion {
            id: "chatcmpl-clamp",
            created: 1,
            model: "deepseek-chat",
            content: "x",
            reasoning_content: None,
            finish_reason: "stop",
            usage: OpenAiUsage::new(5, 2, 7, 7),
        });
        assert!(json.contains(
            "\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2,\"total_tokens\":7,\
\"prompt_tokens_details\":{\"cached_tokens\":5,\"cache_write_tokens\":0}}"
        ));

        let json = format_openai_chat_completion_json(&OpenAiChatCompletion {
            id: "chatcmpl-clamp",
            created: 1,
            model: "deepseek-chat",
            content: "x",
            reasoning_content: None,
            finish_reason: "stop",
            usage: OpenAiUsage::new(5, 2, -1, 99),
        });
        assert!(json.contains(
            "\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2,\"total_tokens\":7,\
\"prompt_tokens_details\":{\"cached_tokens\":0,\"cache_write_tokens\":5}}"
        ));
    }

    #[test]
    fn escapes_content_and_optional_reasoning() {
        assert_eq!(
            format_openai_chat_completion_json(&OpenAiChatCompletion {
                id: "chatcmpl-escape",
                created: 1,
                model: "deepseek-chat",
                content: "line\n\"quoted\"",
                reasoning_content: Some("why\tok"),
                finish_reason: "length",
                usage: OpenAiUsage::new(1, 2, 0, 1),
            }),
            "{\"id\":\"chatcmpl-escape\",\"object\":\"chat.completion\",\"created\":1,\"model\":\"deepseek-chat\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"line\\n\\\"quoted\\\"\",\"reasoning_content\":\"why\\tok\"},\"finish_reason\":\"length\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2,\"total_tokens\":3,\"prompt_tokens_details\":{\"cached_tokens\":0,\"cache_write_tokens\":1}}}\n"
        );
    }
}
