use std::fmt;

use crate::{
    render_chat_prompt_text, render_live_tool_tail_text, ChatMessage, SamplingParams, ThinkMode,
    ToolArgument, ToolCall,
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

#[derive(Debug, Clone, PartialEq)]
pub struct ResponsesRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub has_tools: bool,
    pub tool_schemas: Option<String>,
    pub tool_orders: Vec<ToolSchemaOrder>,
    pub max_tokens: i32,
    pub sampling: SamplingParams,
    pub stream: bool,
    pub reasoning_summary_emit: bool,
    pub think_mode: ThinkMode,
    pub prompt_text: String,
    pub prompt_preserves_reasoning: bool,
    pub responses_requires_live_tool_state: bool,
    pub responses_requires_live_reasoning: bool,
    pub responses_live_call_ids: Vec<String>,
    pub responses_live_suffix_text: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ResponsesLiveState {
    pub call_ids: Vec<String>,
}

impl ResponsesLiveState {
    pub fn with_call_ids(call_ids: impl IntoIterator<Item = impl Into<String>>) -> Self {
        let mut state = Self::default();
        for call_id in call_ids {
            push_unique_id(&mut state.call_ids, call_id.into());
        }
        state
    }

    fn has_call_id(&self, call_id: &str) -> bool {
        self.call_ids.iter().any(|id| id == call_id)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolSchemaOrder {
    pub name: String,
    pub wire_name: Option<String>,
    pub namespace: Option<String>,
    pub responses_tool_search: bool,
    pub properties: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerRequestErrorCategory {
    InvalidJson,
    MissingMessages,
    MissingInput,
    UnsupportedDurableState,
    UnsupportedToolChoice,
    MissingResponsesContinuationState,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerRequestError {
    category: ServerRequestErrorCategory,
    message: String,
}

impl ServerRequestError {
    fn invalid_json() -> Self {
        Self {
            category: ServerRequestErrorCategory::InvalidJson,
            message: "invalid JSON request".to_string(),
        }
    }

    fn missing_messages() -> Self {
        Self {
            category: ServerRequestErrorCategory::MissingMessages,
            message: "missing messages".to_string(),
        }
    }

    fn missing_input() -> Self {
        Self {
            category: ServerRequestErrorCategory::MissingInput,
            message: "missing input".to_string(),
        }
    }

    fn unsupported_durable_state(key: &str) -> Self {
        Self {
            category: ServerRequestErrorCategory::UnsupportedDurableState,
            message: format!("{key} is not supported; replay full input instead"),
        }
    }

    fn unsupported_tool_choice(message: impl Into<String>) -> Self {
        Self {
            category: ServerRequestErrorCategory::UnsupportedToolChoice,
            message: message.into(),
        }
    }

    fn missing_responses_continuation_state(call_id: &str) -> Self {
        Self {
            category: ServerRequestErrorCategory::MissingResponsesContinuationState,
            message: format!(
                "Responses continuation state is not available for call_id {call_id}; retry by replaying the full input history"
            ),
        }
    }

    pub fn category(&self) -> ServerRequestErrorCategory {
        self.category
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for ServerRequestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
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

pub fn parse_responses_core_request(
    body: &str,
    def_tokens: i32,
    ctx_size: i32,
) -> Result<ResponsesRequest, ServerRequestError> {
    parse_responses_core_request_with_live_state(
        body,
        def_tokens,
        ctx_size,
        &ResponsesLiveState::default(),
    )
}

pub fn parse_responses_core_request_with_live_state(
    body: &str,
    def_tokens: i32,
    ctx_size: i32,
    live_state: &ResponsesLiveState,
) -> Result<ResponsesRequest, ServerRequestError> {
    let fields = JsonParser::new(body)
        .parse_root_object_fields_raw()
        .map_err(|_| ServerRequestError::invalid_json())?;

    let mut model = DEFAULT_MODEL.to_string();
    let mut messages = None;
    let mut instructions = None;
    let mut tool_schemas = None;
    let mut loaded_tool_schemas = String::new();
    let mut tool_orders = Vec::new();
    let mut tool_choice_none = false;
    let mut max_tokens = def_tokens;
    let mut sampling = SamplingParams::defaults();
    let mut stream = false;
    let mut got_thinking = false;
    let mut thinking_enabled = true;
    let mut reasoning_effort = ThinkMode::High;
    let mut reasoning_summary_emit = false;

    for (key, raw, value) in fields {
        match key.as_str() {
            "input" => {
                let parsed = parse_responses_core_input(&raw)?;
                messages = Some(parsed.messages);
                loaded_tool_schemas = parsed.loaded_tool_schemas;
                merge_tool_schema_orders(&mut tool_orders, parsed.tool_orders);
            }
            "instructions" => {
                instructions = Some(match value {
                    JsonValue::Null => String::new(),
                    JsonValue::String(value) => value,
                    _ => return Err(ServerRequestError::invalid_json()),
                });
            }
            "tools" => {
                let parsed = parse_tools_value(&raw)?;
                tool_schemas = Some(parsed.schemas);
                merge_tool_schema_orders(&mut tool_orders, parsed.orders);
            }
            "tool_choice" => match value {
                JsonValue::String(choice) => {
                    if choice == "none" {
                        tool_choice_none = true;
                    } else if choice != "auto" {
                        return Err(ServerRequestError::unsupported_tool_choice(format!(
                            "tool_choice={choice} not supported"
                        )));
                    }
                }
                JsonValue::Object(_) => {
                    return Err(ServerRequestError::unsupported_tool_choice(
                        "forced tool_choice not supported",
                    ));
                }
                _ => {}
            },
            "model" => {
                model = value
                    .as_str()
                    .ok_or_else(ServerRequestError::invalid_json)?
                    .to_string();
            }
            "max_output_tokens" | "max_tokens" => {
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
            "stream" => {
                stream = value
                    .as_bool()
                    .ok_or_else(ServerRequestError::invalid_json)?;
            }
            "reasoning" => {
                if let Some(effort) =
                    parse_responses_reasoning(&value, &mut reasoning_summary_emit)?
                {
                    reasoning_effort = effort;
                    got_thinking = true;
                    if reasoning_effort == ThinkMode::None {
                        thinking_enabled = false;
                    }
                }
            }
            "previous_response_id" | "conversation" => {
                if !matches!(value, JsonValue::Null) {
                    return Err(ServerRequestError::unsupported_durable_state(&key));
                }
            }
            _ => {}
        }
    }

    let mut messages = messages.ok_or_else(ServerRequestError::missing_input)?;
    if let Some(instructions) = instructions.filter(|instructions| !instructions.is_empty()) {
        messages.insert(0, ChatMessage::new("system", instructions));
    }
    if !got_thinking && model_alias_disables_thinking(&model) {
        thinking_enabled = false;
    }
    if !got_thinking && model_alias_enables_thinking(&model) {
        thinking_enabled = true;
    }
    let has_any_tool_schema_source = tool_schemas.is_some() || !loaded_tool_schemas.is_empty();
    let combined_tool_schemas = combine_tool_schemas(tool_schemas.as_deref(), &loaded_tool_schemas);
    let has_tools = !combined_tool_schemas.is_empty() && !tool_choice_none;
    let active_tool_schemas = if has_tools {
        Some(combined_tool_schemas.as_str())
    } else {
        None
    };
    let think_mode = think_mode_for_context(
        think_mode_from_enabled(thinking_enabled, reasoning_effort),
        ctx_size,
    );
    let live_validation = validate_responses_tool_outputs(&messages, think_mode, live_state)?;
    let live_continuation = prepare_responses_live_continuation(&messages, think_mode);
    let prompt_preserves_reasoning = chat_history_uses_tool_context(&messages, active_tool_schemas);
    let prompt_text = render_chat_prompt_text(&messages, active_tool_schemas, think_mode);

    Ok(ResponsesRequest {
        model,
        messages,
        has_tools,
        tool_schemas: has_any_tool_schema_source.then_some(combined_tool_schemas),
        tool_orders,
        max_tokens,
        sampling,
        stream,
        reasoning_summary_emit,
        think_mode,
        prompt_text,
        prompt_preserves_reasoning,
        responses_requires_live_tool_state: live_validation.requires_live_tool_state,
        responses_requires_live_reasoning: live_validation.requires_live_reasoning,
        responses_live_call_ids: live_continuation.call_ids,
        responses_live_suffix_text: live_continuation.suffix_text,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct ResponsesLiveValidation {
    requires_live_tool_state: bool,
    requires_live_reasoning: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct ResponsesLiveContinuation {
    call_ids: Vec<String>,
    suffix_text: Option<String>,
}

fn validate_responses_tool_outputs(
    messages: &[ChatMessage],
    think_mode: ThinkMode,
    live_state: &ResponsesLiveState,
) -> Result<ResponsesLiveValidation, ServerRequestError> {
    let mut validation = ResponsesLiveValidation::default();
    let needs_reasoning = think_mode.enabled();
    for (idx, message) in messages.iter().enumerate() {
        if message.role != "tool" && message.role != "function" {
            continue;
        }
        let ids = collect_tool_result_call_ids(message);
        for id in ids {
            let live_known = live_state.has_call_id(&id);
            let prior = find_prior_assistant_call_message(messages, idx, &id);
            if !live_known && prior.is_none() {
                return Err(ServerRequestError::missing_responses_continuation_state(
                    &id,
                ));
            }
            let Some(prior) = prior else {
                validation.requires_live_tool_state = true;
                continue;
            };
            if needs_reasoning && prior.reasoning.is_empty() {
                validation.requires_live_reasoning = true;
            }
        }
    }
    Ok(validation)
}

fn prepare_responses_live_continuation(
    messages: &[ChatMessage],
    think_mode: ThinkMode,
) -> ResponsesLiveContinuation {
    if messages.is_empty() {
        return ResponsesLiveContinuation::default();
    }

    let mut tail_start = messages.len();
    while tail_start > 0 {
        let message = &messages[tail_start - 1];
        if message.role != "tool" && message.role != "function" {
            break;
        }
        tail_start -= 1;
    }
    if tail_start == messages.len() {
        return ResponsesLiveContinuation::default();
    }

    let mut call_ids = Vec::new();
    if tail_start > 0 {
        let assistant = &messages[tail_start - 1];
        if assistant.role != "assistant" || assistant.tool_calls.is_empty() {
            return ResponsesLiveContinuation::default();
        }
        for call in &assistant.tool_calls {
            push_unique_id(&mut call_ids, call.id.clone());
        }
    } else {
        for message in &messages[tail_start..] {
            for id in collect_tool_result_call_ids(message) {
                push_unique_id(&mut call_ids, id);
            }
        }
    }
    if call_ids.is_empty() {
        return ResponsesLiveContinuation::default();
    }

    ResponsesLiveContinuation {
        call_ids,
        suffix_text: Some(render_live_tool_tail_text(messages, tail_start, think_mode)),
    }
}

fn collect_tool_result_call_ids(message: &ChatMessage) -> Vec<String> {
    let mut ids = Vec::new();
    for id in &message.tool_call_ids {
        push_unique_id(&mut ids, id.clone());
    }
    ids
}

fn find_prior_assistant_call_message<'a>(
    messages: &'a [ChatMessage],
    before: usize,
    id: &str,
) -> Option<&'a ChatMessage> {
    if id.is_empty() {
        return None;
    }
    messages[..before].iter().rev().find(|message| {
        message.role == "assistant" && message.tool_calls.iter().any(|call| call.id == id)
    })
}

fn push_unique_id(ids: &mut Vec<String>, id: String) {
    if id.is_empty() || ids.iter().any(|existing| existing == &id) {
        return;
    }
    ids.push(id);
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResponsesInputParse {
    messages: Vec<ChatMessage>,
    loaded_tool_schemas: String,
    tool_orders: Vec<ToolSchemaOrder>,
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
        if let Some(function) = function {
            append_raw_json_line(&mut schemas, function.trim());
            if let Some(order) = tool_schema_order_from_json(function.trim())? {
                push_tool_schema_order(&mut orders, order);
            }
        } else if let Some(parsed) = responses_namespace_tool_schemas(&raw_tool)? {
            append_raw_json_line(&mut schemas, parsed.schemas.trim());
            merge_tool_schema_orders(&mut orders, parsed.orders);
        } else if let Some(special) = responses_special_schema_from_tool(&raw_tool)? {
            append_raw_json_line(&mut schemas, special.trim());
            if let Some(order) = tool_schema_order_from_json_wire(special.trim(), None, None, true)?
            {
                push_tool_schema_order(&mut orders, order);
            }
        } else {
            append_raw_json_line(&mut schemas, raw_tool.trim());
            if let Some(order) = tool_schema_order_from_json(raw_tool.trim())? {
                push_tool_schema_order(&mut orders, order);
            }
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

fn responses_special_schema_from_tool(raw: &str) -> Result<Option<String>, ServerRequestError> {
    if !raw.trim_start().starts_with('{') {
        return Ok(None);
    }
    let fields = JsonParser::new(raw)
        .parse_root_object_fields_raw()
        .map_err(|_| ServerRequestError::invalid_json())?;
    let mut tool_type = None;
    let mut description = None;
    let mut parameters = None;
    for (key, raw, value) in fields {
        match key.as_str() {
            "type" => {
                if let Some(value) = value.as_str() {
                    tool_type = Some(value.to_string());
                } else {
                    return Ok(None);
                }
            }
            "description" => {
                if let Some(value) = value.as_str() {
                    description = Some(value.to_string());
                } else {
                    return Ok(None);
                }
            }
            "parameters" => {
                parameters = Some(raw);
            }
            _ => {}
        }
    }
    if tool_type.as_deref() != Some("tool_search") {
        return Ok(None);
    }

    Ok(Some(format!(
        "{{\"name\":\"tool_search\",\"description\":{},\"parameters\":{}}}",
        json_escape_string(description.as_deref().unwrap_or("Search available tools.")),
        parameters
            .as_deref()
            .unwrap_or("{\"type\":\"object\",\"properties\":{}}")
    )))
}

fn responses_namespace_tool_schemas(
    raw: &str,
) -> Result<Option<ToolParseResult>, ServerRequestError> {
    if !raw.trim_start().starts_with('{') {
        return Ok(None);
    }
    let fields = JsonParser::new(raw)
        .parse_root_object_fields_raw()
        .map_err(|_| ServerRequestError::invalid_json())?;
    let mut tool_type = None;
    let mut namespace = None;
    let mut tools = None;
    for (key, raw, value) in fields {
        match key.as_str() {
            "type" => {
                if let Some(value) = value.as_str() {
                    tool_type = Some(value.to_string());
                } else {
                    return Ok(None);
                }
            }
            "name" => {
                if let Some(value) = value.as_str() {
                    namespace = Some(value.to_string());
                } else {
                    return Ok(None);
                }
            }
            "tools" => {
                tools = Some(raw);
            }
            _ => {}
        }
    }
    let Some(namespace) = namespace else {
        return Ok(None);
    };
    if tool_type.as_deref() != Some("namespace") {
        return Ok(None);
    }
    let Some(tools) = tools else {
        return Ok(None);
    };
    let Ok(raw_tools) = JsonParser::new(&tools).parse_root_array_values_raw() else {
        return Ok(None);
    };

    let mut schemas = String::new();
    let mut orders = Vec::new();
    for raw_tool in raw_tools {
        if let Some((schema, wire_name)) =
            responses_namespace_function_schema_from_tool(&raw_tool, &namespace)?
        {
            append_raw_json_line(&mut schemas, &schema);
            if let Some(order) = tool_schema_order_from_json_wire(
                &schema,
                Some(namespace.clone()),
                Some(wire_name),
                false,
            )? {
                push_tool_schema_order(&mut orders, order);
            }
        }
    }
    if schemas.is_empty() {
        Ok(None)
    } else {
        Ok(Some(ToolParseResult { schemas, orders }))
    }
}

fn responses_namespace_function_schema_from_tool(
    raw: &str,
    namespace: &str,
) -> Result<Option<(String, String)>, ServerRequestError> {
    if !raw.trim_start().starts_with('{') {
        return Ok(None);
    }
    let fields = JsonParser::new(raw)
        .parse_root_object_fields_raw()
        .map_err(|_| ServerRequestError::invalid_json())?;
    let mut tool_type = None;
    let mut name = None;
    let mut description = None;
    let mut parameters = None;
    for (key, raw, value) in fields {
        match key.as_str() {
            "type" => {
                if let Some(value) = value.as_str() {
                    tool_type = Some(value.to_string());
                } else {
                    return Ok(None);
                }
            }
            "name" => {
                if let Some(value) = value.as_str() {
                    name = Some(value.to_string());
                } else {
                    return Ok(None);
                }
            }
            "description" => {
                if let Some(value) = value.as_str() {
                    description = Some(value.to_string());
                } else {
                    return Ok(None);
                }
            }
            "parameters" | "input_schema" => {
                parameters = Some(raw);
            }
            _ => {}
        }
    }
    if tool_type
        .as_deref()
        .is_some_and(|tool_type| tool_type != "function")
    {
        return Ok(None);
    }
    let Some(name) = name.filter(|name| !name.is_empty()) else {
        return Ok(None);
    };
    let prompt_name = format!("{namespace}{name}");
    let schema = format!(
        "{{\"name\":{},\"description\":{},\"parameters\":{}}}",
        json_escape_string(&prompt_name),
        json_escape_string(description.as_deref().unwrap_or("")),
        parameters
            .as_deref()
            .unwrap_or("{\"type\":\"object\",\"properties\":{}}")
    );
    Ok(Some((schema, name)))
}

fn tool_schema_order_from_json(raw: &str) -> Result<Option<ToolSchemaOrder>, ServerRequestError> {
    tool_schema_order_from_json_wire(raw, None, None, false)
}

fn tool_schema_order_from_json_wire(
    raw: &str,
    namespace: Option<String>,
    wire_name: Option<String>,
    responses_tool_search: bool,
) -> Result<Option<ToolSchemaOrder>, ServerRequestError> {
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
    Ok(name.map(|name| ToolSchemaOrder {
        name,
        wire_name,
        namespace,
        responses_tool_search,
        properties,
    }))
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

fn merge_tool_schema_orders(orders: &mut Vec<ToolSchemaOrder>, new_orders: Vec<ToolSchemaOrder>) {
    for order in new_orders {
        push_tool_schema_order(orders, order);
    }
}

fn combine_tool_schemas(top_level: Option<&str>, loaded: &str) -> String {
    let mut out = String::new();
    if let Some(top_level) = top_level.filter(|schemas| !schemas.is_empty()) {
        out.push_str(top_level);
    }
    if !loaded.is_empty() {
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(loaded);
    }
    out
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

fn parse_responses_core_input(raw: &str) -> Result<ResponsesInputParse, ServerRequestError> {
    let value = JsonParser::new(raw)
        .parse()
        .map_err(|_| ServerRequestError::invalid_json())?;
    if let JsonValue::String(value) = value {
        return Ok(ResponsesInputParse {
            messages: vec![ChatMessage::new("user", value.clone())],
            loaded_tool_schemas: String::new(),
            tool_orders: Vec::new(),
        });
    }
    if !matches!(value, JsonValue::Array(_)) {
        return Err(ServerRequestError::invalid_json());
    };
    let items = JsonParser::new(raw)
        .parse_root_array_values_raw()
        .map_err(|_| ServerRequestError::invalid_json())?;

    let mut messages = Vec::new();
    let mut loaded_tool_schemas = String::new();
    let mut tool_orders = Vec::new();
    let mut pending_reasoning = String::new();
    for item in items {
        let fields = JsonParser::new(&item)
            .parse_root_object_fields_raw()
            .map_err(|_| ServerRequestError::invalid_json())?;
        let mut item_type = None;
        let mut role = None;
        let mut content = None;
        let mut name = None;
        let mut namespace = None;
        let mut call_id = None;
        let mut item_id = None;
        let mut arguments = None;
        let mut output = None;
        let mut input = None;
        let mut summary = None;
        let mut action = None;
        let mut result = None;
        let mut tools_json = None;
        let mut status = None;
        for (key, raw, value) in fields {
            match key.as_str() {
                "type" => {
                    item_type = Some(
                        value
                            .as_str()
                            .ok_or_else(ServerRequestError::invalid_json)?
                            .to_string(),
                    );
                }
                "role" => {
                    role = Some(
                        value
                            .as_str()
                            .ok_or_else(ServerRequestError::invalid_json)?
                            .to_string(),
                    );
                }
                "content" => {
                    content = Some(parse_responses_content_array(&value)?);
                }
                "name" => {
                    name = Some(
                        value
                            .as_str()
                            .ok_or_else(ServerRequestError::invalid_json)?
                            .to_string(),
                    );
                }
                "namespace" => {
                    namespace = Some(
                        value
                            .as_str()
                            .ok_or_else(ServerRequestError::invalid_json)?
                            .to_string(),
                    );
                }
                "call_id" => {
                    call_id = Some(
                        value
                            .as_str()
                            .ok_or_else(ServerRequestError::invalid_json)?
                            .to_string(),
                    );
                }
                "id" => {
                    item_id = Some(
                        value
                            .as_str()
                            .ok_or_else(ServerRequestError::invalid_json)?
                            .to_string(),
                    );
                }
                "arguments" => {
                    arguments = Some(json_string_or_raw(&raw, &value));
                }
                "output" => {
                    output = Some(parse_responses_output_value(&raw, &value)?);
                }
                "input" => {
                    input = Some(json_string_or_raw(&raw, &value));
                }
                "summary" => {
                    summary = Some(parse_responses_content_array(&value)?);
                }
                "action" => {
                    action = Some(raw);
                }
                "result" => {
                    result = Some(json_string_or_raw(&raw, &value));
                }
                "status" => {
                    status = Some(
                        value
                            .as_str()
                            .ok_or_else(ServerRequestError::invalid_json)?
                            .to_string(),
                    );
                }
                "tools" => {
                    tools_json = Some(raw);
                }
                _ => {}
            }
        }
        if status
            .as_deref()
            .is_some_and(|status| !status.is_empty() && status != "completed")
        {
            return Err(ServerRequestError::invalid_json());
        }
        let item_type = item_type.as_deref().unwrap_or("message");
        let consumes_reasoning = responses_item_consumes_reasoning(item_type, role.as_deref());
        let is_bookkeeping = item_type == "compaction" || item_type == "context_compaction";
        if !consumes_reasoning && !is_bookkeeping && !pending_reasoning.is_empty() {
            let mut msg = ChatMessage::new("assistant", "");
            msg.reasoning = std::mem::take(&mut pending_reasoning);
            messages.push(msg);
        }
        match item_type {
            "message" => {
                let mut msg = ChatMessage::new(
                    role.unwrap_or_else(|| "user".to_string()),
                    content.unwrap_or_default(),
                );
                if msg.role == "assistant" && !pending_reasoning.is_empty() {
                    msg.reasoning = std::mem::take(&mut pending_reasoning);
                }
                messages.push(msg);
            }
            "function_call" | "custom_tool_call" => {
                let args = arguments.as_deref().or(input.as_deref()).unwrap_or("{}");
                let mut tool_name = name.unwrap_or_default();
                if item_type != "custom_tool_call" && !tool_name.is_empty() {
                    if let Some(namespace) = namespace.filter(|namespace| !namespace.is_empty()) {
                        tool_name = format!("{namespace}{tool_name}");
                    }
                }
                let call =
                    response_tool_call(tool_name, args, call_id.as_deref().or(item_id.as_deref()));
                push_responses_assistant_tool_call(&mut messages, call, &mut pending_reasoning);
            }
            "function_call_output" | "custom_tool_call_output" => {
                let mut msg = ChatMessage::new("tool", output.unwrap_or_default());
                if let Some(id) = call_id.or(item_id) {
                    msg.add_tool_call_id(id);
                }
                messages.push(msg);
            }
            "reasoning" => {
                if let Some(summary) = summary.filter(|summary| !summary.is_empty()) {
                    if !pending_reasoning.is_empty() {
                        pending_reasoning.push('\n');
                    }
                    pending_reasoning.push_str(&summary);
                }
                if let Some(content) = content.filter(|content| !content.is_empty()) {
                    if !pending_reasoning.is_empty() {
                        pending_reasoning.push('\n');
                    }
                    pending_reasoning.push_str(&content);
                }
            }
            "local_shell_call"
            | "web_search_call"
            | "tool_search_call"
            | "image_generation_call" => {
                let tool_name = match item_type {
                    "tool_search_call" => "tool_search",
                    "local_shell_call" => "local_shell",
                    _ => item_type,
                };
                let args = action
                    .as_deref()
                    .or(arguments.as_deref())
                    .or(input.as_deref())
                    .unwrap_or("{}");
                let call =
                    response_tool_call(tool_name, args, call_id.as_deref().or(item_id.as_deref()));
                push_responses_assistant_tool_call(&mut messages, call, &mut pending_reasoning);
            }
            "local_shell_call_output"
            | "web_search_call_output"
            | "tool_search_output"
            | "tool_search_call_output"
            | "image_generation_call_output" => {
                if item_type == "tool_search_output" {
                    if let Some(tools_json) = tools_json.as_deref() {
                        let parsed = parse_tools_value(tools_json)?;
                        append_raw_json_line(&mut loaded_tool_schemas, &parsed.schemas);
                        merge_tool_schema_orders(&mut tool_orders, parsed.orders);
                    }
                }
                let body = output.or(result).or(tools_json).unwrap_or_default();
                let mut msg = ChatMessage::new("tool", body);
                if let Some(id) = call_id.or(item_id) {
                    msg.add_tool_call_id(id);
                }
                messages.push(msg);
            }
            "compaction" | "context_compaction" => {}
            _ => return Err(ServerRequestError::invalid_json()),
        }
    }
    if !pending_reasoning.is_empty() {
        let mut msg = ChatMessage::new("assistant", "");
        msg.reasoning = pending_reasoning;
        messages.push(msg);
    }
    Ok(ResponsesInputParse {
        messages,
        loaded_tool_schemas,
        tool_orders,
    })
}

fn responses_item_consumes_reasoning(item_type: &str, role: Option<&str>) -> bool {
    (item_type == "message" && role.is_some_and(|role| role == "assistant"))
        || matches!(
            item_type,
            "function_call"
                | "custom_tool_call"
                | "local_shell_call"
                | "web_search_call"
                | "tool_search_call"
                | "image_generation_call"
        )
}

fn response_tool_call(name: impl Into<String>, args: &str, id: Option<&str>) -> ToolCall {
    ToolCall::new(
        name,
        tool_arguments_from_json(args)
            .unwrap_or_else(|| vec![ToolArgument::string("arguments", args)]),
    )
    .with_id(id.unwrap_or_default())
}

fn push_responses_assistant_tool_call(
    messages: &mut Vec<ChatMessage>,
    call: ToolCall,
    pending_reasoning: &mut String,
) {
    if let Some(last) = messages
        .last_mut()
        .filter(|message| message.role == "assistant")
    {
        if !pending_reasoning.is_empty() && last.reasoning.is_empty() {
            last.reasoning = std::mem::take(pending_reasoning);
        }
        last.tool_calls.push(call);
    } else {
        let mut msg = ChatMessage::new("assistant", "");
        if !pending_reasoning.is_empty() {
            msg.reasoning = std::mem::take(pending_reasoning);
        }
        msg.tool_calls.push(call);
        messages.push(msg);
    }
}

fn json_string_or_raw(raw: &str, value: &JsonValue) -> String {
    match value {
        JsonValue::String(value) => value.clone(),
        _ => raw.to_string(),
    }
}

fn parse_responses_output_value(
    raw: &str,
    value: &JsonValue,
) -> Result<String, ServerRequestError> {
    match value {
        JsonValue::Array(_) => parse_responses_content_array(value),
        JsonValue::String(value) => Ok(value.clone()),
        _ => Ok(raw.to_string()),
    }
}

fn parse_responses_content_array(value: &JsonValue) -> Result<String, ServerRequestError> {
    match value {
        JsonValue::String(value) => Ok(value.clone()),
        JsonValue::Null => Ok(String::new()),
        JsonValue::Array(items) => {
            let mut out = String::new();
            for item in items {
                match item {
                    JsonValue::String(value) => out.push_str(value),
                    JsonValue::Object(fields) => {
                        let mut block_type = None;
                        let mut text = None;
                        for (key, value) in fields {
                            match key.as_str() {
                                "type" => {
                                    block_type = Some(
                                        value
                                            .as_str()
                                            .ok_or_else(ServerRequestError::invalid_json)?
                                            .to_string(),
                                    );
                                }
                                "text" => {
                                    text = Some(match value {
                                        JsonValue::Null => String::new(),
                                        JsonValue::String(value) => value.clone(),
                                        _ => return Err(ServerRequestError::invalid_json()),
                                    });
                                }
                                _ => {}
                            }
                        }
                        let is_text_block = block_type.as_deref().is_some_and(|block_type| {
                            matches!(
                                block_type,
                                "input_text"
                                    | "output_text"
                                    | "text"
                                    | "summary_text"
                                    | "reasoning_text"
                            )
                        });
                        if !is_text_block || text.is_none() {
                            return Err(ServerRequestError::invalid_json());
                        }
                        out.push_str(&text.unwrap_or_default());
                    }
                    _ => return Err(ServerRequestError::invalid_json()),
                }
            }
            Ok(out)
        }
        _ => Err(ServerRequestError::invalid_json()),
    }
}

fn parse_responses_reasoning(
    value: &JsonValue,
    summary_opted_in: &mut bool,
) -> Result<Option<ThinkMode>, ServerRequestError> {
    if matches!(value, JsonValue::Null) {
        return Ok(None);
    }
    let JsonValue::Object(fields) = value else {
        return Ok(None);
    };
    let mut effort = None;
    for (key, value) in fields {
        match key.as_str() {
            "effort" => {
                if !matches!(value, JsonValue::Null) {
                    let name = value
                        .as_str()
                        .ok_or_else(ServerRequestError::invalid_json)?;
                    effort = Some(
                        parse_reasoning_effort_name(name)
                            .ok_or_else(ServerRequestError::invalid_json)?,
                    );
                }
            }
            "summary" => {
                if let JsonValue::String(mode) = value {
                    if matches!(mode.as_str(), "auto" | "concise" | "detailed") {
                        *summary_opted_in = true;
                    }
                } else if !matches!(value, JsonValue::Null) {
                    continue;
                }
            }
            _ => {}
        }
    }
    Ok(effort)
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

    fn parse_responses_fixture(body: &str) -> ResponsesRequest {
        parse_responses_core_request(body, 128, 32_768).expect("fixture parses")
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
    fn responses_string_input_instructions_and_controls_match_c_core_surface() {
        let req = parse_responses_fixture(
            r#"{
                "model":"deepseek-chat",
                "instructions":"sys",
                "input":"hello",
                "max_output_tokens":7,
                "temperature":0.25,
                "top_p":0.75,
                "stream":true
            }"#,
        );
        assert_eq!(req.model, "deepseek-chat");
        assert_eq!(
            req.messages,
            [
                ChatMessage::new("system", "sys"),
                ChatMessage::new("user", "hello"),
            ]
        );
        assert_eq!(req.max_tokens, 7);
        assert_eq!(req.sampling.temperature, 0.25);
        assert_eq!(req.sampling.top_p, 0.75);
        assert!(req.stream);
        assert_eq!(req.think_mode, ThinkMode::None);
        assert!(!req.reasoning_summary_emit);
        assert_eq!(
            req.prompt_text,
            "<｜begin▁of▁sentence｜>sys<｜User｜>hello<｜Assistant｜></think>"
        );
    }

    #[test]
    fn responses_reasoning_items_merge_into_next_assistant_message() {
        let req = parse_responses_core_request(
            r#"{
                "input":[
                    {"type":"message","role":"user","content":[{"type":"input_text","text":"Q"}]},
                    {"type":"reasoning","summary":[{"type":"summary_text","text":"brief"}],"content":[{"type":"reasoning_text","text":"deep"}]},
                    {"type":"message","role":"assistant","content":[{"type":"output_text","text":"A"}]}
                ],
                "reasoning":{"effort":"max","summary":"concise"}
            }"#,
            128,
            393_216,
        )
        .expect("reasoning parses");

        let mut assistant = ChatMessage::new("assistant", "A");
        assistant.reasoning = "brief\ndeep".to_string();
        assert_eq!(req.messages, [ChatMessage::new("user", "Q"), assistant]);
        assert_eq!(req.think_mode, ThinkMode::Max);
        assert!(req.reasoning_summary_emit);
        assert!(req.prompt_text.starts_with(&format!(
            "<｜begin▁of▁sentence｜>{THINK_MAX_PREFIX}<｜User｜>Q<｜Assistant｜>"
        )));
        assert!(req
            .prompt_text
            .ends_with("<think>brief\ndeep</think>A<｜end▁of▁sentence｜>"));
    }

    #[test]
    fn responses_top_level_tools_render_before_instructions() {
        let req = parse_responses_fixture(
            r#"{
                "instructions":"sys",
                "input":[{"type":"message","role":"user","content":"use it"}],
                "tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"query":{"type":"string"},"limit":{"type":"integer"}}}}}],
                "tool_choice":"auto"
            }"#,
        );
        assert!(req.has_tools);
        assert!(req.prompt_preserves_reasoning);
        assert_eq!(req.tool_orders.len(), 1);
        assert_eq!(req.tool_orders[0].name, "lookup");
        assert_eq!(req.tool_orders[0].properties, ["query", "limit"]);
        let tools = req.prompt_text.find("## Tools").expect("tools prompt");
        let system = req.prompt_text.find("sys").expect("instructions");
        let user = req.prompt_text.find("<｜User｜>use it").expect("user");
        assert!(tools < system);
        assert!(system < user);
        assert!(req
            .tool_schemas
            .as_deref()
            .is_some_and(|schemas| schemas.contains("\"name\":\"lookup\"")));
    }

    #[test]
    fn responses_function_call_items_merge_with_assistant_and_render_dsml() {
        let req = parse_responses_fixture(
            r#"{
                "input":[
                    {"type":"message","role":"user","content":"run lookup"},
                    {"type":"reasoning","content":[{"type":"reasoning_text","text":"need lookup"}]},
                    {"type":"message","role":"assistant","content":"checking"},
                    {"type":"function_call","call_id":"call_lookup","name":"lookup","arguments":{"query":"ds4","limit":2,"ratio":1.0}}
                ]
            }"#,
        );
        assert_eq!(req.messages.len(), 2);
        let assistant = &req.messages[1];
        assert_eq!(assistant.role, "assistant");
        assert_eq!(assistant.content, "checking");
        assert_eq!(assistant.reasoning, "need lookup");
        assert_eq!(assistant.tool_calls.len(), 1);
        assert_eq!(assistant.tool_calls[0].id, "call_lookup");
        assert_eq!(assistant.tool_calls[0].name, "lookup");
        assert_eq!(
            assistant.tool_calls[0].arguments,
            [
                ToolArgument::string("query", "ds4"),
                ToolArgument {
                    name: "limit".to_string(),
                    value: "2".to_string(),
                    is_string: false,
                },
                ToolArgument {
                    name: "ratio".to_string(),
                    value: "1.0".to_string(),
                    is_string: false,
                },
            ]
        );
        assert!(req
            .prompt_text
            .contains("<think>need lookup</think>checking\n\n<｜DSML｜tool_calls>"));
        assert!(req.prompt_text.contains("<｜DSML｜invoke name=\"lookup\">"));
        assert!(req.prompt_text.contains(
            "<｜DSML｜parameter name=\"ratio\" string=\"false\">1.0</｜DSML｜parameter>"
        ));
    }

    #[test]
    fn responses_custom_and_hosted_tool_calls_keep_ids_names_and_arguments() {
        let req = parse_responses_fixture(
            r#"{
                "input":[
                    {"type":"reasoning","summary":"tool plan"},
                    {"type":"custom_tool_call","call_id":"call_custom","name":"bash","input":"ls -la"},
                    {"type":"local_shell_call","id":"call_shell","action":{"cmd":"pwd","timeout":1.0}},
                    {"type":"web_search_call","call_id":"call_web","arguments":{"query":"ds4"}}
                ]
            }"#,
        );
        assert_eq!(req.messages.len(), 1);
        let assistant = &req.messages[0];
        assert_eq!(assistant.reasoning, "tool plan");
        assert_eq!(assistant.tool_calls.len(), 3);
        assert_eq!(assistant.tool_calls[0].id, "call_custom");
        assert_eq!(assistant.tool_calls[0].name, "bash");
        assert_eq!(
            assistant.tool_calls[0].arguments,
            [ToolArgument::string("arguments", "ls -la")]
        );
        assert_eq!(assistant.tool_calls[1].id, "call_shell");
        assert_eq!(assistant.tool_calls[1].name, "local_shell");
        assert_eq!(
            assistant.tool_calls[1].arguments,
            [
                ToolArgument::string("cmd", "pwd"),
                ToolArgument {
                    name: "timeout".to_string(),
                    value: "1.0".to_string(),
                    is_string: false,
                },
            ]
        );
        assert_eq!(assistant.tool_calls[2].id, "call_web");
        assert_eq!(assistant.tool_calls[2].name, "web_search_call");
        assert_eq!(
            assistant.tool_calls[2].arguments,
            [ToolArgument::string("query", "ds4")]
        );
    }

    #[test]
    fn responses_tool_outputs_preserve_call_ids_and_render_prompt_tail() {
        let req = parse_responses_fixture(
            r#"{
                "input":[
                    {"type":"message","role":"user","content":"run"},
                    {"type":"function_call","call_id":"call_lookup","name":"lookup","arguments":{"query":"ds4"}},
                    {"type":"local_shell_call","id":"call_shell","action":{"cmd":"pwd"}},
                    {"type":"tool_search_call","call_id":"call_search","arguments":{"query":"tools"}},
                    {"type":"function_call_output","call_id":"call_lookup","output":[{"type":"output_text","text":"result </tool_result> & raw"}]},
                    {"type":"local_shell_call_output","id":"call_shell","result":{"ok":true}},
                    {"type":"tool_search_output","call_id":"call_search","tools":[{"type":"namespace","name":"mcp__demo__","tools":[]}]}
                ]
            }"#,
        );
        assert_eq!(req.messages.len(), 5);
        assert_eq!(req.messages[2].role, "tool");
        assert_eq!(req.messages[2].content, "result </tool_result> & raw");
        assert_eq!(req.messages[2].tool_call_ids, ["call_lookup"]);
        assert_eq!(req.messages[3].content, "{\"ok\":true}");
        assert_eq!(req.messages[3].tool_call_ids, ["call_shell"]);
        assert!(req.messages[4].content.contains("\"type\":\"namespace\""));
        assert_eq!(req.messages[4].tool_call_ids, ["call_search"]);
        assert!(req
            .prompt_text
            .contains("<tool_result>result &lt;/tool_result> & raw</tool_result>"));
        assert!(req
            .prompt_text
            .contains("<tool_result>{\"ok\":true}</tool_result>"));
        assert!(req.prompt_text.ends_with("<｜Assistant｜><think>"));
    }

    #[test]
    fn responses_hosted_tool_search_schema_is_distinct_from_plain_function() {
        let req = parse_responses_fixture(
            r#"{
                "input":"search",
                "tools":[{"type":"tool_search","execution":"client","description":"Search deferred tools","parameters":{"type":"object","properties":{"query":{"type":"string"},"limit":{"type":"number"}}}}]
            }"#,
        );
        let schemas = req.tool_schemas.as_deref().expect("schemas");
        assert!(schemas.contains("\"name\":\"tool_search\""));
        assert!(schemas.contains("\"description\":\"Search deferred tools\""));
        assert_eq!(req.tool_orders.len(), 1);
        assert_eq!(req.tool_orders[0].name, "tool_search");
        assert!(req.tool_orders[0].responses_tool_search);
        assert_eq!(req.tool_orders[0].properties, ["query", "limit"]);

        let function = parse_responses_fixture(
            r#"{
                "input":"search",
                "tools":[{"type":"function","function":{"name":"tool_search","description":"plain function","parameters":{"type":"object","properties":{"query":{"type":"string"}}}}}]
            }"#,
        );
        assert_eq!(function.tool_orders.len(), 1);
        assert_eq!(function.tool_orders[0].name, "tool_search");
        assert!(!function.tool_orders[0].responses_tool_search);
    }

    #[test]
    fn responses_namespace_tools_flatten_prompt_name_and_keep_wire_metadata() {
        let req = parse_responses_fixture(
            r#"{
                "input":"use namespace",
                "tools":[{"type":"namespace","name":"mcp__perplexity__","description":"Perplexity tools","tools":[{"type":"function","name":"perplexity_search","description":"Search the web","parameters":{"type":"object","properties":{"query":{"type":"string"},"recency":{"type":"number"}}}}]}]
            }"#,
        );
        let schemas = req.tool_schemas.as_deref().expect("schemas");
        assert!(schemas.contains("\"name\":\"mcp__perplexity__perplexity_search\""));
        assert!(!schemas.contains("\"name\":\"perplexity_search\""));
        assert_eq!(req.tool_orders.len(), 1);
        assert_eq!(
            req.tool_orders[0].name,
            "mcp__perplexity__perplexity_search"
        );
        assert_eq!(
            req.tool_orders[0].namespace.as_deref(),
            Some("mcp__perplexity__")
        );
        assert_eq!(
            req.tool_orders[0].wire_name.as_deref(),
            Some("perplexity_search")
        );
        assert_eq!(req.tool_orders[0].properties, ["query", "recency"]);
        assert!(req
            .prompt_text
            .find("mcp__perplexity__perplexity_search")
            .is_some());
    }

    #[test]
    fn responses_tool_search_output_loads_dynamic_schemas_after_top_level_tools() {
        let req = parse_responses_fixture(
            r#"{
                "input":[
                    {"type":"tool_search_call","call_id":"call_search","arguments":{"query":"perplexity"}},
                    {"type":"tool_search_output","call_id":"call_search","status":"completed","tools":[{"type":"namespace","name":"mcp__perplexity__","tools":[{"type":"function","name":"perplexity_search","parameters":{"type":"object","properties":{"query":{"type":"string"}}}}]}]}
                ],
                "tools":[{"type":"function","function":{"name":"top_level","parameters":{"type":"object","properties":{"arg":{"type":"string"}}}}}]
            }"#,
        );
        let schemas = req.tool_schemas.as_deref().expect("schemas");
        let top = schemas
            .find("\"name\":\"top_level\"")
            .expect("top-level schema");
        let loaded = schemas
            .find("\"name\":\"mcp__perplexity__perplexity_search\"")
            .expect("loaded schema");
        assert!(top < loaded);
        assert_eq!(req.tool_orders.len(), 2);
        assert_eq!(
            req.tool_orders[0].name,
            "mcp__perplexity__perplexity_search"
        );
        assert_eq!(req.tool_orders[1].name, "top_level");
        assert!(req.has_tools);
        assert!(req.prompt_text.contains("## Tools"));
        assert!(req.messages[1].content.contains("mcp__perplexity__"));

        let bad = parse_responses_core_request(
            r#"{"input":[{"type":"tool_search_output","call_id":"call_search","status":"completed","tools":{"not":"a tool array"}}]}"#,
            128,
            32_768,
        )
        .expect_err("malformed dynamic tool list rejected");
        assert_eq!(bad.category(), ServerRequestErrorCategory::InvalidJson);
    }

    #[test]
    fn responses_tool_output_only_requires_live_state_or_prior_call() {
        let missing = parse_responses_core_request(
            r#"{"input":[{"type":"function_call_output","call_id":"call_missing","output":"out"}]}"#,
            128,
            32_768,
        )
        .expect_err("missing live continuation state rejected");
        assert_eq!(
            missing.category(),
            ServerRequestErrorCategory::MissingResponsesContinuationState
        );
        assert_eq!(
            missing.message(),
            "Responses continuation state is not available for call_id call_missing; retry by replaying the full input history"
        );

        let live_state = ResponsesLiveState::with_call_ids(["call_missing"]);
        let req = parse_responses_core_request_with_live_state(
            r#"{"input":[{"type":"function_call_output","call_id":"call_missing","output":"out </tool_result>"}]}"#,
            128,
            32_768,
            &live_state,
        )
        .expect("live-known tool output parses");
        assert!(req.responses_requires_live_tool_state);
        assert!(!req.responses_requires_live_reasoning);
        assert_eq!(req.responses_live_call_ids, ["call_missing"]);
        let suffix = req.responses_live_suffix_text.as_deref().expect("suffix");
        assert!(suffix.starts_with("<｜end▁of▁sentence｜><｜User｜><tool_result>"));
        assert!(suffix.contains("out &lt;/tool_result></tool_result>"));
        assert!(suffix.ends_with("<｜Assistant｜><think>"));
    }

    #[test]
    fn responses_stateless_tool_replay_marks_missing_reasoning_only_in_thinking_mode() {
        let req = parse_responses_fixture(
            r#"{
                "input":[
                    {"type":"function_call","call_id":"call_replay","name":"lookup","arguments":{"query":"ds4"}},
                    {"type":"function_call_output","call_id":"call_replay","output":"ok"}
                ]
            }"#,
        );
        assert!(!req.responses_requires_live_tool_state);
        assert!(req.responses_requires_live_reasoning);
        assert_eq!(req.responses_live_call_ids, ["call_replay"]);
        let suffix = req.responses_live_suffix_text.as_deref().expect("suffix");
        assert!(suffix.contains("<tool_result>ok</tool_result>"));
        assert!(!suffix.contains("lookup"));

        let replayed_reasoning = parse_responses_fixture(
            r#"{
                "input":[
                    {"type":"reasoning","content":[{"type":"reasoning_text","text":"hidden"}]},
                    {"type":"function_call","call_id":"call_replay","name":"lookup","arguments":{"query":"ds4"}},
                    {"type":"function_call_output","call_id":"call_replay","output":"ok"}
                ]
            }"#,
        );
        assert!(!replayed_reasoning.responses_requires_live_tool_state);
        assert!(!replayed_reasoning.responses_requires_live_reasoning);

        let no_thinking = parse_responses_fixture(
            r#"{
                "model":"deepseek-chat",
                "input":[
                    {"type":"function_call","call_id":"call_replay","name":"lookup","arguments":{"query":"ds4"}},
                    {"type":"function_call_output","call_id":"call_replay","output":"ok"}
                ]
            }"#,
        );
        assert_eq!(no_thinking.think_mode, ThinkMode::None);
        assert!(!no_thinking.responses_requires_live_reasoning);
        assert!(no_thinking
            .responses_live_suffix_text
            .as_deref()
            .is_some_and(|suffix| suffix.ends_with("<｜Assistant｜></think>")));
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

        let missing = parse_responses_core_request(r#"{"model":"deepseek-chat"}"#, 128, 32_768)
            .expect_err("missing input");
        assert_eq!(missing.category(), ServerRequestErrorCategory::MissingInput);
        assert_eq!(missing.message(), "missing input");
    }

    #[test]
    fn responses_rejects_durable_state_and_unsupported_tool_choice() {
        let previous = parse_responses_core_request(
            r#"{"input":"hi","previous_response_id":"resp_1"}"#,
            128,
            32_768,
        )
        .expect_err("previous response rejected");
        assert_eq!(
            previous.category(),
            ServerRequestErrorCategory::UnsupportedDurableState
        );
        assert_eq!(
            previous.message(),
            "previous_response_id is not supported; replay full input instead"
        );

        let conversation =
            parse_responses_core_request(r#"{"input":"hi","conversation":{}}"#, 128, 32_768)
                .expect_err("conversation rejected");
        assert_eq!(
            conversation.category(),
            ServerRequestErrorCategory::UnsupportedDurableState
        );
        assert_eq!(
            conversation.message(),
            "conversation is not supported; replay full input instead"
        );

        let required =
            parse_responses_core_request(r#"{"input":"hi","tool_choice":"required"}"#, 128, 32_768)
                .expect_err("required tool choice rejected");
        assert_eq!(
            required.category(),
            ServerRequestErrorCategory::UnsupportedToolChoice
        );
        assert_eq!(required.message(), "tool_choice=required not supported");

        let forced = parse_responses_core_request(
            r#"{"input":"hi","tool_choice":{"type":"function"}}"#,
            128,
            32_768,
        )
        .expect_err("forced tool choice rejected");
        assert_eq!(
            forced.category(),
            ServerRequestErrorCategory::UnsupportedToolChoice
        );
        assert_eq!(forced.message(), "forced tool_choice not supported");
    }

    #[test]
    fn responses_rejects_incomplete_or_non_text_core_input_items() {
        let in_progress = parse_responses_core_request(
            r#"{"input":[{"type":"message","role":"user","status":"in_progress","content":"hi"}]}"#,
            128,
            32_768,
        )
        .expect_err("in-progress message rejected");
        assert_eq!(
            in_progress.category(),
            ServerRequestErrorCategory::InvalidJson
        );

        let unknown = parse_responses_core_request(
            r#"{"input":[{"type":"unknown_call","status":"completed"}]}"#,
            128,
            32_768,
        )
        .expect_err("unknown input item rejected");
        assert_eq!(unknown.category(), ServerRequestErrorCategory::InvalidJson);

        let bad_content = parse_responses_core_request(
            r#"{"input":[{"type":"message","role":"user","content":[{"type":"input_text"}]}]}"#,
            128,
            32_768,
        )
        .expect_err("content object without text rejected");
        assert_eq!(
            bad_content.category(),
            ServerRequestErrorCategory::InvalidJson
        );
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
