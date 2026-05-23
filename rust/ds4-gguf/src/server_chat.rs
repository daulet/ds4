use std::fmt;

use crate::{
    render_chat_prompt_text, ChatMessage, SamplingParams, ThinkMode, ToolArgument, ToolCall,
};

const DEFAULT_MODEL: &str = "deepseek-v4-flash";
const THINK_MAX_MIN_CONTEXT: i32 = 393_216;

#[derive(Debug, Clone, PartialEq)]
pub struct OpenAiChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub has_tools: bool,
    pub tool_schemas: Option<String>,
    pub tool_orders: Vec<ToolSchemaOrder>,
    pub max_tokens: i32,
    pub sampling: SamplingParams,
    pub seed: u64,
    pub stream: bool,
    pub stream_include_usage: bool,
    pub think_mode: ThinkMode,
    pub stops: Vec<String>,
    pub prompt_text: String,
    pub prompt_preserves_reasoning: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolSchemaOrder {
    pub name: String,
    pub properties: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerRequestErrorCategory {
    InvalidJson,
    MissingMessages,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerRequestError {
    category: ServerRequestErrorCategory,
    message: &'static str,
}

impl ServerRequestError {
    fn invalid_json() -> Self {
        Self {
            category: ServerRequestErrorCategory::InvalidJson,
            message: "invalid JSON request",
        }
    }

    fn missing_messages() -> Self {
        Self {
            category: ServerRequestErrorCategory::MissingMessages,
            message: "missing messages",
        }
    }

    pub fn category(&self) -> ServerRequestErrorCategory {
        self.category
    }

    pub fn message(&self) -> &'static str {
        self.message
    }
}

impl fmt::Display for ServerRequestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.message)
    }
}

impl std::error::Error for ServerRequestError {}

pub fn parse_openai_chat_request(
    body: &str,
    def_tokens: i32,
    ctx_size: i32,
) -> Result<OpenAiChatRequest, ServerRequestError> {
    let fields = JsonParser::new(body)
        .parse_root_object_fields_raw()
        .map_err(|_| ServerRequestError::invalid_json())?;

    let mut model = DEFAULT_MODEL.to_string();
    let mut messages = None;
    let mut tool_schemas = None;
    let mut tool_orders = Vec::new();
    let mut tool_choice_none = false;
    let mut max_tokens = def_tokens;
    let mut sampling = SamplingParams::defaults();
    let mut seed = 0_u64;
    let mut stream = false;
    let mut stream_include_usage = false;
    let mut got_thinking = false;
    let mut thinking_enabled = true;
    let mut reasoning_effort = ThinkMode::High;
    let mut stops = Vec::new();

    for (key, raw, value) in fields {
        match key.as_str() {
            "messages" => {
                messages = Some(parse_messages(&value)?);
            }
            "tools" => {
                let parsed = parse_tools_value(&raw)?;
                tool_schemas = Some(parsed.schemas);
                tool_orders = parsed.orders;
            }
            "tool_choice" => {
                if let Some(choice) = value.as_str() {
                    tool_choice_none = choice == "none";
                }
            }
            "model" => {
                model = value
                    .as_str()
                    .ok_or_else(ServerRequestError::invalid_json)?
                    .to_string();
            }
            "max_tokens" | "max_completion_tokens" => {
                max_tokens = json_int(&value).ok_or_else(ServerRequestError::invalid_json)?;
            }
            "temperature" => {
                sampling.temperature =
                    json_number(&value).ok_or_else(ServerRequestError::invalid_json)? as f32;
            }
            "top_p" => {
                sampling.top_p =
                    json_number(&value).ok_or_else(ServerRequestError::invalid_json)? as f32;
            }
            "min_p" => {
                sampling.min_p =
                    json_number(&value).ok_or_else(ServerRequestError::invalid_json)? as f32;
            }
            "top_k" => {
                sampling.top_k = json_int(&value).ok_or_else(ServerRequestError::invalid_json)?;
            }
            "seed" => {
                let value = json_number(&value).ok_or_else(ServerRequestError::invalid_json)?;
                seed = if value > 0.0 { value as u64 } else { 0 };
            }
            "stream" => {
                stream = value
                    .as_bool()
                    .ok_or_else(ServerRequestError::invalid_json)?;
            }
            "stream_options" => {
                parse_stream_options(&value, &mut stream_include_usage)?;
            }
            "thinking" => {
                parse_thinking_control(&value, &mut thinking_enabled)?;
                got_thinking = true;
            }
            "reasoning_effort" => {
                parse_reasoning_effort_value(&value, &mut reasoning_effort)?;
            }
            "think" => {
                thinking_enabled = value
                    .as_bool()
                    .ok_or_else(ServerRequestError::invalid_json)?;
                got_thinking = true;
            }
            "stop" => {
                stops = parse_stop(&value)?;
            }
            _ => {}
        }
    }

    let messages = messages.ok_or_else(ServerRequestError::missing_messages)?;
    if !got_thinking && model_alias_disables_thinking(&model) {
        thinking_enabled = false;
    }
    if !got_thinking && model_alias_enables_thinking(&model) {
        thinking_enabled = true;
    }
    let has_tools = tool_schemas
        .as_deref()
        .is_some_and(|schemas| !schemas.is_empty())
        && !tool_choice_none;
    let active_tool_schemas = if has_tools {
        tool_schemas.as_deref()
    } else {
        None
    };
    let think_mode = think_mode_for_context(
        think_mode_from_enabled(thinking_enabled, reasoning_effort),
        ctx_size,
    );
    let prompt_preserves_reasoning = chat_history_uses_tool_context(&messages, active_tool_schemas);
    let prompt_text = render_chat_prompt_text(&messages, active_tool_schemas, think_mode);

    Ok(OpenAiChatRequest {
        model,
        messages,
        has_tools,
        tool_schemas,
        tool_orders,
        max_tokens,
        sampling,
        seed,
        stream,
        stream_include_usage,
        think_mode,
        stops,
        prompt_text,
        prompt_preserves_reasoning,
    })
}

pub fn think_mode_for_context(mode: ThinkMode, ctx_size: i32) -> ThinkMode {
    if mode == ThinkMode::Max && ctx_size.max(0) < THINK_MAX_MIN_CONTEXT {
        ThinkMode::High
    } else {
        mode
    }
}

pub fn request_exceeds_context(n_prompt_tokens: usize, ctx_size: i32) -> bool {
    n_prompt_tokens >= ctx_size.max(0) as usize
}

pub fn openai_context_length_error_body(n_prompt_tokens: usize, ctx_size: i32) -> String {
    format!(
        "{{\"error\":{{\"message\":\"Prompt has {n_prompt_tokens} tokens, but the configured context size is {ctx_size} tokens\",\"type\":\"invalid_request_error\",\"param\":\"messages\",\"code\":\"context_length_exceeded\",\"n_prompt_tokens\":{n_prompt_tokens},\"n_ctx\":{ctx_size}}}}}\n"
    )
}

fn model_alias_disables_thinking(model: &str) -> bool {
    model == "deepseek-chat"
}

fn model_alias_enables_thinking(model: &str) -> bool {
    model == "deepseek-reasoner"
}

fn think_mode_from_enabled(enabled: bool, effort: ThinkMode) -> ThinkMode {
    if !enabled || effort == ThinkMode::None {
        ThinkMode::None
    } else if effort == ThinkMode::Max {
        ThinkMode::Max
    } else {
        ThinkMode::High
    }
}

fn parse_messages(value: &JsonValue) -> Result<Vec<ChatMessage>, ServerRequestError> {
    let JsonValue::Array(values) = value else {
        return Err(ServerRequestError::invalid_json());
    };

    let mut messages = Vec::with_capacity(values.len());
    for value in values {
        let JsonValue::Object(fields) = value else {
            return Err(ServerRequestError::invalid_json());
        };
        let mut role = None;
        let mut content = None;
        let mut reasoning = None;
        let mut tool_calls = Vec::new();
        for (key, value) in fields {
            match key.as_str() {
                "role" => {
                    role = Some(
                        value
                            .as_str()
                            .ok_or_else(ServerRequestError::invalid_json)?
                            .to_string(),
                    );
                }
                "content" => {
                    content = Some(json_content(value)?);
                }
                "reasoning_content" => {
                    reasoning = Some(json_content(value)?);
                }
                "tool_calls" => {
                    tool_calls = parse_tool_calls_value(value)?;
                }
                _ => {}
            }
        }
        let mut message = ChatMessage::new(
            role.unwrap_or_else(|| "user".to_string()),
            content.unwrap_or_default(),
        );
        message.reasoning = reasoning.unwrap_or_default();
        message.tool_calls = tool_calls;
        messages.push(message);
    }
    Ok(messages)
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct ToolParseResult {
    schemas: String,
    orders: Vec<ToolSchemaOrder>,
}

fn parse_tools_value(raw: &str) -> Result<ToolParseResult, ServerRequestError> {
    let value = JsonParser::new(raw)
        .parse()
        .map_err(|_| ServerRequestError::invalid_json())?;
    if matches!(value, JsonValue::Null) {
        return Ok(ToolParseResult {
            schemas: String::new(),
            orders: Vec::new(),
        });
    }
    let raw_tools = JsonParser::new(raw)
        .parse_root_array_values_raw()
        .map_err(|_| ServerRequestError::invalid_json())?;
    let mut schemas = String::new();
    let mut orders = Vec::new();
    for raw_tool in raw_tools {
        let function = openai_function_schema_from_tool(&raw_tool)?;
        let schema = function.unwrap_or(raw_tool);
        append_raw_json_line(&mut schemas, schema.trim());
        if let Some(order) = tool_schema_order_from_json(schema.trim())? {
            push_tool_schema_order(&mut orders, order);
        }
    }
    Ok(ToolParseResult { schemas, orders })
}

fn append_raw_json_line(out: &mut String, json: &str) {
    if json.is_empty() {
        return;
    }
    if !out.is_empty() {
        out.push('\n');
    }
    out.push_str(json);
}

fn openai_function_schema_from_tool(raw: &str) -> Result<Option<String>, ServerRequestError> {
    if !raw.trim_start().starts_with('{') {
        return Ok(None);
    }
    let fields = JsonParser::new(raw)
        .parse_root_object_fields_raw()
        .map_err(|_| ServerRequestError::invalid_json())?;
    for (key, raw, _) in fields {
        if key == "function" {
            return Ok(Some(raw.trim().to_string()));
        }
    }
    Ok(None)
}

fn tool_schema_order_from_json(raw: &str) -> Result<Option<ToolSchemaOrder>, ServerRequestError> {
    if !raw.trim_start().starts_with('{') {
        return Ok(None);
    }
    let fields = JsonParser::new(raw)
        .parse_root_object_fields_raw()
        .map_err(|_| ServerRequestError::invalid_json())?;
    let mut name = None;
    let mut properties = Vec::new();
    for (key, raw, value) in fields {
        match key.as_str() {
            "name" => {
                name = Some(
                    value
                        .as_str()
                        .ok_or_else(ServerRequestError::invalid_json)?
                        .to_string(),
                );
            }
            "parameters" | "input_schema" => {
                properties = parse_schema_properties(&raw)?;
            }
            _ => {}
        }
    }
    Ok(name.map(|name| ToolSchemaOrder { name, properties }))
}

fn parse_schema_properties(raw: &str) -> Result<Vec<String>, ServerRequestError> {
    let value = JsonParser::new(raw)
        .parse()
        .map_err(|_| ServerRequestError::invalid_json())?;
    let JsonValue::Object(fields) = value else {
        return Ok(Vec::new());
    };
    for (key, value) in fields {
        if key != "properties" {
            continue;
        }
        let JsonValue::Object(properties) = value else {
            return Ok(Vec::new());
        };
        return Ok(properties.into_iter().map(|(name, _)| name).collect());
    }
    Ok(Vec::new())
}

fn push_tool_schema_order(orders: &mut Vec<ToolSchemaOrder>, order: ToolSchemaOrder) {
    if let Some(existing) = orders
        .iter_mut()
        .find(|existing| existing.name == order.name)
    {
        *existing = order;
    } else {
        orders.push(order);
    }
}

fn parse_tool_calls_value(value: &JsonValue) -> Result<Vec<ToolCall>, ServerRequestError> {
    if matches!(value, JsonValue::Null) {
        return Ok(Vec::new());
    }
    let JsonValue::Array(values) = value else {
        return Err(ServerRequestError::invalid_json());
    };
    let mut calls = Vec::new();
    for value in values {
        let JsonValue::Object(fields) = value else {
            return Err(ServerRequestError::invalid_json());
        };
        let mut name = None;
        let mut arguments = None;
        for (key, value) in fields {
            if key != "function" {
                continue;
            }
            let JsonValue::Object(function_fields) = value else {
                return Err(ServerRequestError::invalid_json());
            };
            for (key, value) in function_fields {
                match key.as_str() {
                    "name" => {
                        name = Some(
                            value
                                .as_str()
                                .ok_or_else(ServerRequestError::invalid_json)?
                                .to_string(),
                        );
                    }
                    "arguments" => {
                        arguments = Some(match value {
                            JsonValue::String(arguments) => arguments.clone(),
                            _ => minify_json_value(value),
                        });
                    }
                    _ => {}
                }
            }
        }
        if let (Some(name), Some(arguments)) = (name, arguments) {
            calls.push(ToolCall::new(
                name,
                tool_arguments_from_json(&arguments)
                    .unwrap_or_else(|| vec![ToolArgument::string("arguments", arguments.as_str())]),
            ));
        }
    }
    Ok(calls)
}

fn tool_arguments_from_json(json: &str) -> Option<Vec<ToolArgument>> {
    let fields = JsonParser::new(json).parse_root_object_fields_raw().ok()?;
    let mut args = Vec::new();
    for (name, raw, value) in fields {
        match value {
            JsonValue::String(value) => args.push(ToolArgument::string(name, value)),
            _ => args.push(ToolArgument {
                name,
                value: minify_json_raw_value(&raw),
                is_string: false,
            }),
        }
    }
    Some(args)
}

fn json_content(value: &JsonValue) -> Result<String, ServerRequestError> {
    match value {
        JsonValue::String(value) => Ok(value.clone()),
        JsonValue::Null => Ok(String::new()),
        JsonValue::Array(values) => {
            let mut out = String::new();
            for value in values {
                match value {
                    JsonValue::String(value) => out.push_str(value),
                    JsonValue::Object(fields) => {
                        for (key, value) in fields {
                            if key == "text" {
                                out.push_str(
                                    value
                                        .as_str()
                                        .ok_or_else(ServerRequestError::invalid_json)?,
                                );
                            }
                        }
                    }
                    _ => {}
                }
            }
            Ok(out)
        }
        _ => Ok(String::new()),
    }
}

fn parse_stream_options(
    value: &JsonValue,
    include_usage: &mut bool,
) -> Result<(), ServerRequestError> {
    let JsonValue::Object(fields) = value else {
        return Ok(());
    };
    for (key, value) in fields {
        if key == "include_usage" {
            *include_usage = value
                .as_bool()
                .ok_or_else(ServerRequestError::invalid_json)?;
        }
    }
    Ok(())
}

fn parse_thinking_control(
    value: &JsonValue,
    thinking_enabled: &mut bool,
) -> Result<(), ServerRequestError> {
    match value {
        JsonValue::Null => Ok(()),
        JsonValue::Bool(value) => {
            *thinking_enabled = *value;
            Ok(())
        }
        JsonValue::Object(fields) => {
            for (key, value) in fields {
                if key == "type" {
                    match value
                        .as_str()
                        .ok_or_else(ServerRequestError::invalid_json)?
                    {
                        "enabled" => *thinking_enabled = true,
                        "disabled" => *thinking_enabled = false,
                        _ => {}
                    }
                }
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn parse_reasoning_effort_value(
    value: &JsonValue,
    effort: &mut ThinkMode,
) -> Result<(), ServerRequestError> {
    if matches!(value, JsonValue::Null) {
        return Ok(());
    }
    let name = value
        .as_str()
        .ok_or_else(ServerRequestError::invalid_json)?;
    *effort = parse_reasoning_effort_name(name).ok_or_else(ServerRequestError::invalid_json)?;
    Ok(())
}

fn parse_reasoning_effort_name(name: &str) -> Option<ThinkMode> {
    match name {
        "max" => Some(ThinkMode::Max),
        "xhigh" | "high" | "medium" | "low" | "minimal" => Some(ThinkMode::High),
        "none" => Some(ThinkMode::None),
        _ => None,
    }
}

fn parse_stop(value: &JsonValue) -> Result<Vec<String>, ServerRequestError> {
    let mut stops = Vec::new();
    match value {
        JsonValue::String(value) => push_stop(&mut stops, value),
        JsonValue::Array(values) => {
            for value in values {
                if let JsonValue::String(value) = value {
                    push_stop(&mut stops, value);
                }
            }
        }
        _ => {}
    }
    Ok(stops)
}

fn push_stop(stops: &mut Vec<String>, value: &str) {
    if !value.is_empty() {
        stops.push(value.to_string());
    }
}

fn json_int(value: &JsonValue) -> Option<i32> {
    let value = json_number(value)?;
    if value < 0.0 {
        Some(0)
    } else if value > i32::MAX as f64 {
        Some(i32::MAX)
    } else {
        Some(value as i32)
    }
}

fn json_number(value: &JsonValue) -> Option<f64> {
    match value {
        JsonValue::Number(value) => Some(*value),
        _ => None,
    }
}

fn minify_json_value(value: &JsonValue) -> String {
    match value {
        JsonValue::Null => "null".to_string(),
        JsonValue::Bool(value) => value.to_string(),
        JsonValue::Number(value) => {
            let mut out = value.to_string();
            if out.ends_with(".0") {
                out.truncate(out.len() - 2);
            }
            out
        }
        JsonValue::String(value) => json_escape_string(value),
        JsonValue::Array(values) => {
            let mut out = String::from("[");
            for (idx, value) in values.iter().enumerate() {
                if idx > 0 {
                    out.push(',');
                }
                out.push_str(&minify_json_value(value));
            }
            out.push(']');
            out
        }
        JsonValue::Object(fields) => {
            let mut out = String::from("{");
            for (idx, (key, value)) in fields.iter().enumerate() {
                if idx > 0 {
                    out.push(',');
                }
                out.push_str(&json_escape_string(key));
                out.push(':');
                out.push_str(&minify_json_value(value));
            }
            out.push('}');
            out
        }
    }
}

fn minify_json_raw_value(raw: &str) -> String {
    let mut out = String::new();
    let mut in_string = false;
    let mut escape = false;
    for ch in raw.trim().chars() {
        if in_string {
            out.push(ch);
            if escape {
                escape = false;
            } else if ch == '\\' {
                escape = true;
            } else if ch == '"' {
                in_string = false;
            }
        } else if ch == '"' {
            in_string = true;
            out.push(ch);
        } else if !ch.is_ascii_whitespace() {
            out.push(ch);
        }
    }
    out
}

fn json_escape_string(value: &str) -> String {
    let mut out = String::from("\"");
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{0008}' => out.push_str("\\b"),
            '\u{000c}' => out.push_str("\\f"),
            ch if ch <= '\u{001f}' => {
                out.push_str(&format!("\\u{:04x}", ch as u32));
            }
            ch => out.push(ch),
        }
    }
    out.push('"');
    out
}

#[derive(Debug, Clone, PartialEq)]
enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<JsonValue>),
    Object(Vec<(String, JsonValue)>),
}

impl JsonValue {
    fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(value) => Some(*value),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct JsonParseError;

struct JsonParser<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn new(input: &'a str) -> Self {
        Self { input, pos: 0 }
    }

    fn parse(mut self) -> Result<JsonValue, JsonParseError> {
        let value = self.parse_value(0)?;
        self.skip_ws();
        if self.pos == self.input.len() {
            Ok(value)
        } else {
            Err(JsonParseError)
        }
    }

    fn parse_root_object_fields_raw(
        mut self,
    ) -> Result<Vec<(String, String, JsonValue)>, JsonParseError> {
        self.skip_ws();
        self.expect_byte(b'{')?;
        self.skip_ws();
        let mut fields = Vec::new();
        if self.consume_byte(b'}') {
            self.skip_ws();
            return if self.pos == self.input.len() {
                Ok(fields)
            } else {
                Err(JsonParseError)
            };
        }
        loop {
            let key = self.parse_string()?;
            self.skip_ws();
            self.expect_byte(b':')?;
            self.skip_ws();
            let raw_start = self.pos;
            let value = self.parse_value(1)?;
            let raw = self.input[raw_start..self.pos].to_string();
            fields.push((key, raw, value));
            self.skip_ws();
            if self.consume_byte(b'}') {
                self.skip_ws();
                return if self.pos == self.input.len() {
                    Ok(fields)
                } else {
                    Err(JsonParseError)
                };
            }
            self.expect_byte(b',')?;
            self.skip_ws();
        }
    }

    fn parse_root_array_values_raw(mut self) -> Result<Vec<String>, JsonParseError> {
        self.skip_ws();
        self.expect_byte(b'[')?;
        self.skip_ws();
        let mut values = Vec::new();
        if self.consume_byte(b']') {
            self.skip_ws();
            return if self.pos == self.input.len() {
                Ok(values)
            } else {
                Err(JsonParseError)
            };
        }
        loop {
            self.skip_ws();
            let raw_start = self.pos;
            self.parse_value(1)?;
            values.push(self.input[raw_start..self.pos].to_string());
            self.skip_ws();
            if self.consume_byte(b']') {
                self.skip_ws();
                return if self.pos == self.input.len() {
                    Ok(values)
                } else {
                    Err(JsonParseError)
                };
            }
            self.expect_byte(b',')?;
            self.skip_ws();
        }
    }

    fn parse_value(&mut self, depth: usize) -> Result<JsonValue, JsonParseError> {
        if depth >= 256 {
            return Err(JsonParseError);
        }
        self.skip_ws();
        match self.peek() {
            Some(b'n') => {
                self.expect_lit("null")?;
                Ok(JsonValue::Null)
            }
            Some(b't') => {
                self.expect_lit("true")?;
                Ok(JsonValue::Bool(true))
            }
            Some(b'f') => {
                self.expect_lit("false")?;
                Ok(JsonValue::Bool(false))
            }
            Some(b'"') => Ok(JsonValue::String(self.parse_string()?)),
            Some(b'[') => self.parse_array(depth),
            Some(b'{') => self.parse_object(depth),
            Some(b'-' | b'0'..=b'9') => Ok(JsonValue::Number(self.parse_number()?)),
            _ => Err(JsonParseError),
        }
    }

    fn parse_array(&mut self, depth: usize) -> Result<JsonValue, JsonParseError> {
        self.expect_byte(b'[')?;
        self.skip_ws();
        let mut values = Vec::new();
        if self.consume_byte(b']') {
            return Ok(JsonValue::Array(values));
        }
        loop {
            values.push(self.parse_value(depth + 1)?);
            self.skip_ws();
            if self.consume_byte(b']') {
                return Ok(JsonValue::Array(values));
            }
            self.expect_byte(b',')?;
            self.skip_ws();
        }
    }

    fn parse_object(&mut self, depth: usize) -> Result<JsonValue, JsonParseError> {
        self.expect_byte(b'{')?;
        self.skip_ws();
        let mut fields = Vec::new();
        if self.consume_byte(b'}') {
            return Ok(JsonValue::Object(fields));
        }
        loop {
            let key = self.parse_string()?;
            self.skip_ws();
            self.expect_byte(b':')?;
            let value = self.parse_value(depth + 1)?;
            fields.push((key, value));
            self.skip_ws();
            if self.consume_byte(b'}') {
                return Ok(JsonValue::Object(fields));
            }
            self.expect_byte(b',')?;
            self.skip_ws();
        }
    }

    fn parse_string(&mut self) -> Result<String, JsonParseError> {
        self.expect_byte(b'"')?;
        let mut out = String::new();
        while let Some(byte) = self.bump() {
            match byte {
                b'"' => return Ok(out),
                b'\\' => match self.bump().ok_or(JsonParseError)? {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'b' => out.push('\u{0008}'),
                    b'f' => out.push('\u{000c}'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => {
                        let cp = self.parse_u16_escape()?;
                        let cp = if (0xd800..=0xdbff).contains(&cp) {
                            let saved = self.pos;
                            if self.consume_byte(b'\\') && self.consume_byte(b'u') {
                                let low = self.parse_u16_escape()?;
                                if (0xdc00..=0xdfff).contains(&low) {
                                    0x10000 + ((cp - 0xd800) << 10) + (low - 0xdc00)
                                } else {
                                    return Err(JsonParseError);
                                }
                            } else {
                                self.pos = saved;
                                cp
                            }
                        } else {
                            cp
                        };
                        let ch = char::from_u32(cp).ok_or(JsonParseError)?;
                        out.push(ch);
                    }
                    _ => return Err(JsonParseError),
                },
                0x00..=0x1f => return Err(JsonParseError),
                _ => {
                    let start = self.pos - 1;
                    let rest = &self.input[start..];
                    let ch = rest.chars().next().ok_or(JsonParseError)?;
                    self.pos = start + ch.len_utf8();
                    out.push(ch);
                }
            }
        }
        Err(JsonParseError)
    }

    fn parse_u16_escape(&mut self) -> Result<u32, JsonParseError> {
        let mut cp = 0_u32;
        for _ in 0..4 {
            let byte = self.bump().ok_or(JsonParseError)?;
            let value = match byte {
                b'0'..=b'9' => byte - b'0',
                b'a'..=b'f' => byte - b'a' + 10,
                b'A'..=b'F' => byte - b'A' + 10,
                _ => return Err(JsonParseError),
            };
            cp = (cp << 4) | u32::from(value);
        }
        Ok(cp)
    }

    fn parse_number(&mut self) -> Result<f64, JsonParseError> {
        let start = self.pos;
        self.consume_byte(b'-');
        match self.peek() {
            Some(b'0') => {
                self.pos += 1;
            }
            Some(b'1'..=b'9') => {
                self.pos += 1;
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.pos += 1;
                }
            }
            _ => return Err(JsonParseError),
        }
        if self.consume_byte(b'.') {
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(JsonParseError);
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(JsonParseError);
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        self.input[start..self.pos]
            .parse::<f64>()
            .map_err(|_| JsonParseError)
    }

    fn expect_lit(&mut self, value: &str) -> Result<(), JsonParseError> {
        if self.input[self.pos..].starts_with(value) {
            self.pos += value.len();
            Ok(())
        } else {
            Err(JsonParseError)
        }
    }

    fn expect_byte(&mut self, byte: u8) -> Result<(), JsonParseError> {
        if self.consume_byte(byte) {
            Ok(())
        } else {
            Err(JsonParseError)
        }
    }

    fn consume_byte(&mut self, byte: u8) -> bool {
        if self.peek() == Some(byte) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn bump(&mut self) -> Option<u8> {
        let byte = self.peek()?;
        self.pos += 1;
        Some(byte)
    }

    fn peek(&self) -> Option<u8> {
        self.input.as_bytes().get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\n' | b'\r' | b'\t')) {
            self.pos += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompt::THINK_MAX_PREFIX;

    const CHAT_BASIC: &str =
        include_str!("../../../ds4-parity/baselines/server-fixtures/m0.4/chat_basic.json");
    const CHAT_STREAM: &str =
        include_str!("../../../ds4-parity/baselines/server-fixtures/m0.4/chat_stream.json");
    const CHAT_THINKING_DISABLED: &str = include_str!(
        "../../../ds4-parity/baselines/server-fixtures/m0.4/chat_thinking_disabled.json"
    );
    const CHAT_TOOL_CALL: &str =
        include_str!("../../../ds4-parity/baselines/server-fixtures/m0.4/chat_tool_call.json");
    const CHAT_CACHE_SEED: &str =
        include_str!("../../../ds4-parity/baselines/server-fixtures/m0.4/chat_cache_seed.json");
    const CHAT_CACHE_CONTINUATION: &str = include_str!(
        "../../../ds4-parity/baselines/server-fixtures/m0.4/chat_cache_continuation.json"
    );
    const M04_SERVER_TRACE: &str =
        include_str!("../../../ds4-parity/baselines/server-traces/m0.4/traces/server.trace");

    fn parse_fixture(body: &str) -> OpenAiChatRequest {
        parse_openai_chat_request(body, 128, 32_768).expect("fixture parses")
    }

    fn rendered_prompt_from_trace(request: usize) -> &'static str {
        let marker = format!("===== request {request} ");
        let start = M04_SERVER_TRACE.find(&marker).expect("request marker");
        let prompt = M04_SERVER_TRACE[start..]
            .find("--- rendered prompt ---\n")
            .expect("prompt marker")
            + start
            + "--- rendered prompt ---\n".len();
        let end = M04_SERVER_TRACE[prompt..]
            .find("\n\n--- generated text ---")
            .expect("generated marker")
            + prompt;
        &M04_SERVER_TRACE[prompt..end]
    }

    fn raw_request_from_trace(request: usize) -> &'static str {
        let marker = format!("===== request {request} ");
        let start = M04_SERVER_TRACE.find(&marker).expect("request marker");
        let raw = M04_SERVER_TRACE[start..]
            .find("--- raw request json ---\n")
            .expect("raw marker")
            + start
            + "--- raw request json ---\n".len();
        let end = M04_SERVER_TRACE[raw..]
            .find("\n\n--- rendered prompt ---")
            .expect("prompt marker")
            + raw;
        &M04_SERVER_TRACE[raw..end]
    }

    #[test]
    fn m04_non_tool_fixtures_match_rendered_prompt_and_fields() {
        let basic = parse_fixture(CHAT_BASIC);
        assert_eq!(basic.model, "deepseek-chat");
        assert!(!basic.has_tools);
        assert!(basic.tool_schemas.is_none());
        assert_eq!(basic.max_tokens, 8);
        assert_eq!(basic.sampling.temperature, 0.0);
        assert_eq!(basic.sampling.top_k, 0);
        assert_eq!(basic.sampling.top_p, 1.0);
        assert_eq!(basic.sampling.min_p, 0.05);
        assert_eq!(basic.seed, 7);
        assert!(!basic.stream);
        assert!(!basic.stream_include_usage);
        assert_eq!(basic.think_mode, ThinkMode::None);
        assert_eq!(
            basic.prompt_text,
            "<｜begin▁of▁sentence｜><｜User｜>Return exactly this text: baseline ready<｜Assistant｜></think>"
        );

        let stream = parse_fixture(CHAT_STREAM);
        assert_eq!(stream.model, "deepseek-chat");
        assert!(stream.stream);
        assert!(stream.stream_include_usage);
        assert_eq!(stream.seed, 11);
        assert_eq!(stream.think_mode, ThinkMode::None);
        assert_eq!(
            stream.prompt_text,
            "<｜begin▁of▁sentence｜><｜User｜>Return exactly this text: stream baseline<｜Assistant｜></think>"
        );

        let disabled = parse_fixture(CHAT_THINKING_DISABLED);
        assert_eq!(disabled.model, "deepseek-v4-flash");
        assert_eq!(disabled.max_tokens, 4);
        assert_eq!(disabled.seed, 13);
        assert_eq!(disabled.think_mode, ThinkMode::None);
        assert_eq!(
            disabled.prompt_text,
            "<｜begin▁of▁sentence｜><｜User｜>What is two plus two? Answer with one digit.<｜Assistant｜></think>"
        );

        let seed = parse_fixture(CHAT_CACHE_SEED);
        assert_eq!(seed.model, "deepseek-chat");
        assert_eq!(seed.seed, 17);
        assert_eq!(seed.think_mode, ThinkMode::None);
        assert_eq!(
            seed.prompt_text,
            "<｜begin▁of▁sentence｜>You answer with the shortest exact phrase requested by the user.<｜User｜>Cache baseline prompt alpha beta gamma delta epsilon zeta eta theta iota kappa lambda. Return exactly: cache ready<｜Assistant｜></think>"
        );

        let continuation = parse_fixture(CHAT_CACHE_CONTINUATION);
        assert_eq!(continuation.model, "deepseek-chat");
        assert_eq!(continuation.seed, 19);
        assert_eq!(continuation.think_mode, ThinkMode::None);
        assert_eq!(
            continuation.prompt_text,
            "<｜begin▁of▁sentence｜>You answer with the shortest exact phrase requested by the user.<｜User｜>Cache baseline prompt alpha beta gamma delta epsilon zeta eta theta iota kappa lambda. Return exactly: cache ready<｜Assistant｜></think>cache ready<｜end▁of▁sentence｜><｜User｜>Return exactly: cache continued<｜Assistant｜></think>"
        );
    }

    #[test]
    fn defaults_match_c_request_init() {
        let req = parse_fixture(r#"{"messages":[{"content":"Hello"}]}"#);
        assert_eq!(req.model, "deepseek-v4-flash");
        assert_eq!(req.max_tokens, 128);
        assert_eq!(req.sampling, SamplingParams::defaults());
        assert_eq!(req.seed, 0);
        assert!(!req.stream);
        assert!(!req.stream_include_usage);
        assert_eq!(req.think_mode, ThinkMode::High);
        assert_eq!(
            req.prompt_text,
            "<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜><think>"
        );
    }

    #[test]
    fn parses_core_generation_controls_and_stop_lists() {
        let req = parse_fixture(
            r#"{
                "messages":[{"role":"user","content":["a ",{"type":"text","text":"b"},9,null]}],
                "max_completion_tokens":-4,
                "temperature":0.25,
                "top_p":0.75,
                "min_p":0.12,
                "top_k":2048,
                "seed":42.9,
                "stop":["END","",{"ignored":true},"STOP"],
                "think":false
            }"#,
        );
        assert_eq!(req.max_tokens, 0);
        assert_eq!(req.sampling.temperature, 0.25);
        assert_eq!(req.sampling.top_p, 0.75);
        assert_eq!(req.sampling.min_p, 0.12);
        assert_eq!(req.sampling.top_k, 2048);
        assert_eq!(req.seed, 42);
        assert_eq!(req.stops, ["END", "STOP"]);
        assert_eq!(req.think_mode, ThinkMode::None);
        assert_eq!(
            req.prompt_text,
            "<｜begin▁of▁sentence｜><｜User｜>a b<｜Assistant｜></think>"
        );
    }

    #[test]
    fn m04_tool_fixture_matches_c_rendered_prompt_and_schema_order() {
        let req = parse_fixture(CHAT_TOOL_CALL);
        assert_eq!(req.model, "deepseek-v4-flash");
        assert!(req.has_tools);
        assert_eq!(req.think_mode, ThinkMode::None);
        assert_eq!(req.max_tokens, 192);
        assert_eq!(req.seed, 23);
        assert!(req.prompt_preserves_reasoning);
        assert_eq!(req.tool_orders.len(), 1);
        assert_eq!(req.tool_orders[0].name, "list_files");
        assert_eq!(req.tool_orders[0].properties, ["path"]);

        let trace_req = parse_fixture(raw_request_from_trace(3));
        assert_eq!(trace_req.prompt_text, rendered_prompt_from_trace(3));
    }

    #[test]
    fn tool_choice_none_parses_schemas_but_disables_tool_prompt() {
        let req = parse_fixture(
            r#"{
                "messages":[{"role":"user","content":"No tool prompt"}],
                "tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"query":{"type":"string"}}}}}],
                "tool_choice":"none"
            }"#,
        );
        assert!(!req.has_tools);
        assert!(req
            .tool_schemas
            .as_deref()
            .is_some_and(|s| s.contains("\"name\":\"lookup\"")));
        assert_eq!(req.tool_orders[0].name, "lookup");
        assert_eq!(req.tool_orders[0].properties, ["query"]);
        assert_eq!(
            req.prompt_text,
            "<｜begin▁of▁sentence｜><｜User｜>No tool prompt<｜Assistant｜><think>"
        );
        assert!(!req.prompt_preserves_reasoning);
    }

    #[test]
    fn assistant_tool_calls_render_dsml_arguments_in_request_history() {
        let req = parse_fixture(
            r#"{
                "messages":[
                    {"role":"user","content":"run"},
                    {"role":"assistant","reasoning_content":"need lookup","tool_calls":[{"id":"call_1","function":{"name":"lookup","arguments":"{\"query\":\"ds4\",\"limit\":2,\"ratio\":1.0,\"nested\":{\"x\":true}}"}}]},
                    {"role":"tool","content":"result </tool_result> & raw"},
                    {"role":"user","content":"continue"}
                ],
                "tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"query":{"type":"string"},"limit":{"type":"integer"},"ratio":{"type":"number"},"nested":{"type":"object"}}}}}]
            }"#,
        );
        assert!(req.has_tools);
        assert!(req.prompt_preserves_reasoning);
        assert!(req.prompt_text.contains("<think>need lookup</think>"));
        assert!(req
            .prompt_text
            .contains("<｜DSML｜invoke name=\"lookup\">\n<｜DSML｜parameter name=\"query\" string=\"true\">ds4</｜DSML｜parameter>\n<｜DSML｜parameter name=\"limit\" string=\"false\">2</｜DSML｜parameter>\n<｜DSML｜parameter name=\"ratio\" string=\"false\">1.0</｜DSML｜parameter>\n<｜DSML｜parameter name=\"nested\" string=\"false\">{\"x\":true}</｜DSML｜parameter>"));
        assert!(req
            .prompt_text
            .contains("<tool_result>result &lt;/tool_result> & raw</tool_result>"));
    }

    #[test]
    fn thinking_controls_match_c_mapping() {
        let max_small_ctx = parse_openai_chat_request(
            r#"{"messages":[{"content":"Hello"}],"reasoning_effort":"max"}"#,
            128,
            32_768,
        )
        .expect("max parses");
        assert_eq!(max_small_ctx.think_mode, ThinkMode::High);
        assert!(!max_small_ctx.prompt_text.contains(THINK_MAX_PREFIX));
        assert!(max_small_ctx
            .prompt_text
            .ends_with("<｜Assistant｜><think>"));

        let max_large_ctx = parse_openai_chat_request(
            r#"{"messages":[{"content":"Hello"}],"reasoning_effort":"max"}"#,
            128,
            393_216,
        )
        .expect("max parses");
        assert_eq!(max_large_ctx.think_mode, ThinkMode::Max);
        assert!(max_large_ctx
            .prompt_text
            .starts_with(&format!("<｜begin▁of▁sentence｜>{THINK_MAX_PREFIX}")));

        let disabled = parse_fixture(
            r#"{"model":"deepseek-reasoner","messages":[{"content":"Hello"}],"thinking":{"type":"disabled","budget_tokens":1024}}"#,
        );
        assert_eq!(disabled.think_mode, ThinkMode::None);
        assert!(disabled.prompt_text.ends_with("<｜Assistant｜></think>"));

        let none = parse_fixture(r#"{"messages":[{"content":"Hello"}],"reasoning_effort":"none"}"#);
        assert_eq!(none.think_mode, ThinkMode::None);

        let bad = parse_openai_chat_request(
            r#"{"messages":[{"content":"Hello"}],"reasoning_effort":"banana"}"#,
            128,
            32_768,
        )
        .expect_err("invalid effort fails");
        assert_eq!(bad.category(), ServerRequestErrorCategory::InvalidJson);
    }

    #[test]
    fn parse_errors_match_stable_categories() {
        let bad = parse_openai_chat_request("{", 128, 32_768).expect_err("bad JSON");
        assert_eq!(bad.category(), ServerRequestErrorCategory::InvalidJson);
        assert_eq!(bad.message(), "invalid JSON request");

        let missing = parse_openai_chat_request(r#"{"model":"deepseek-chat"}"#, 128, 32_768)
            .expect_err("missing messages");
        assert_eq!(
            missing.category(),
            ServerRequestErrorCategory::MissingMessages
        );
        assert_eq!(missing.message(), "missing messages");
    }

    #[test]
    fn context_length_helpers_match_openai_error_shape() {
        assert!(request_exceeds_context(16, 16));
        assert!(!request_exceeds_context(16, 17));
        assert_eq!(
            openai_context_length_error_body(16, 16),
            "{\"error\":{\"message\":\"Prompt has 16 tokens, but the configured context size is 16 tokens\",\"type\":\"invalid_request_error\",\"param\":\"messages\",\"code\":\"context_length_exceeded\",\"n_prompt_tokens\":16,\"n_ctx\":16}}\n"
        );
    }

    #[test]
    fn json_string_escapes_match_c_request_parser() {
        let req = parse_fixture(
            r#"{"messages":[{"role":"user","content":"line\nslash\/unicode \u003c\ud83d\ude00"}],"think":false}"#,
        );
        assert_eq!(req.messages[0].content, "line\nslash/unicode <😀");
        assert_eq!(
            req.prompt_text,
            "<｜begin▁of▁sentence｜><｜User｜>line\nslash/unicode <😀<｜Assistant｜></think>"
        );
    }
}
