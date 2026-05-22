const DS4_TOOL_CALLS_START: &str = "<｜DSML｜tool_calls>";
const DS4_TOOL_CALLS_END: &str = "</｜DSML｜tool_calls>";
const DS4_INVOKE_START: &str = "<｜DSML｜invoke";
const DS4_INVOKE_END: &str = "</｜DSML｜invoke>";
const DS4_PARAM_START: &str = "<｜DSML｜parameter";
const DS4_PARAM_END: &str = "</｜DSML｜parameter>";
const DS4_TOOL_CALLS_START_SHORT: &str = "<DSML｜tool_calls>";
const DS4_TOOL_CALLS_END_SHORT: &str = "</DSML｜tool_calls>";
const DS4_INVOKE_START_SHORT: &str = "<DSML｜invoke";
const DS4_INVOKE_END_SHORT: &str = "</DSML｜invoke>";
const DS4_PARAM_START_SHORT: &str = "<DSML｜parameter";
const DS4_PARAM_END_SHORT: &str = "</DSML｜parameter>";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DsmlArgument {
    pub name: String,
    pub value: String,
    pub is_string: bool,
}

impl DsmlArgument {
    pub fn string(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: value.into(),
            is_string: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DsmlRenderCall {
    pub name: String,
    pub arguments: Vec<DsmlArgument>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DsmlJsonCall {
    pub id: Option<String>,
    pub name: String,
    pub arguments: String,
}

impl DsmlJsonCall {
    pub fn new(name: impl Into<String>, arguments: impl Into<String>) -> Self {
        Self {
            id: None,
            name: name.into(),
            arguments: arguments.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedGeneratedMessage {
    pub content: String,
    pub reasoning: Option<String>,
    pub raw_dsml: Option<String>,
    pub calls: Vec<DsmlJsonCall>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResponseParse {
    pub parse_ok: bool,
    pub recovered: bool,
    pub finish: String,
    pub error: String,
    pub message: ParsedGeneratedMessage,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DsmlParseError;

pub fn render_dsml_tool_calls(raw_dsml: Option<&str>, calls: &[DsmlRenderCall]) -> String {
    if let Some(raw) = raw_dsml.filter(|raw| !raw.is_empty()) {
        return raw.to_string();
    }
    if calls.is_empty() {
        return String::new();
    }

    let mut out = String::new();
    out.push_str("\n\n");
    out.push_str(DS4_TOOL_CALLS_START);
    out.push('\n');
    for call in calls {
        out.push_str("<｜DSML｜invoke name=\"");
        append_dsml_attr_escaped(&mut out, &call.name);
        out.push_str("\">\n");
        for arg in &call.arguments {
            append_dsml_arg(&mut out, arg);
        }
        out.push_str(DS4_INVOKE_END);
        out.push('\n');
    }
    out.push_str(DS4_TOOL_CALLS_END);
    out
}

pub fn render_dsml_tool_calls_from_json(raw_dsml: Option<&str>, calls: &[DsmlJsonCall]) -> String {
    if let Some(raw) = raw_dsml.filter(|raw| !raw.is_empty()) {
        return raw.to_string();
    }
    if calls.is_empty() {
        return String::new();
    }

    let mut out = String::new();
    out.push_str("\n\n");
    out.push_str(DS4_TOOL_CALLS_START);
    out.push('\n');
    for call in calls {
        out.push_str("<｜DSML｜invoke name=\"");
        append_dsml_attr_escaped(&mut out, &call.name);
        out.push_str("\">\n");
        if let Some(args) = parse_json_arguments(&call.arguments) {
            for arg in args {
                append_dsml_arg(&mut out, &arg);
            }
        } else {
            append_dsml_arg(
                &mut out,
                &DsmlArgument::string("arguments", call.arguments.as_str()),
            );
        }
        out.push_str(DS4_INVOKE_END);
        out.push('\n');
    }
    out.push_str(DS4_TOOL_CALLS_END);
    out
}

pub fn render_tool_result_text(text: &str) -> String {
    let mut out = String::new();
    append_escaped_sentinel(&mut out, text, "</tool_result>", "&lt;");
    out
}

pub fn parse_generated_message(
    text: &str,
    require_thinking_closed: bool,
) -> Result<ParsedGeneratedMessage, DsmlParseError> {
    let tool_search_offset = if require_thinking_closed {
        if let Some(pos) = text.rfind("</think>") {
            pos + "</think>".len()
        } else {
            let (content, reasoning) = split_reasoning_content(text, text.len());
            return Ok(ParsedGeneratedMessage {
                content,
                reasoning,
                raw_dsml: None,
                calls: Vec::new(),
            });
        }
    } else {
        0
    };

    let Some((rel_start, style)) = find_tool_start(&text[tool_search_offset..]) else {
        let (content, reasoning) = split_reasoning_content(text, text.len());
        return Ok(ParsedGeneratedMessage {
            content,
            reasoning,
            raw_dsml: None,
            calls: Vec::new(),
        });
    };
    let raw_block_start = tool_search_offset + rel_start;
    let content_len = trim_ascii_ws_end(text, raw_block_start);

    let tags = match style {
        ToolStyle::Canonical => Tags {
            tool_calls_start: DS4_TOOL_CALLS_START,
            tool_calls_end: DS4_TOOL_CALLS_END,
            invoke_start: DS4_INVOKE_START,
            invoke_end: DS4_INVOKE_END,
            param_start: DS4_PARAM_START,
            param_end: DS4_PARAM_END,
        },
        ToolStyle::PlainXml => Tags {
            tool_calls_start: "<tool_calls>",
            tool_calls_end: "</tool_calls>",
            invoke_start: "<invoke",
            invoke_end: "</invoke>",
            param_start: "<parameter",
            param_end: "</parameter>",
        },
        ToolStyle::Short => Tags {
            tool_calls_start: DS4_TOOL_CALLS_START_SHORT,
            tool_calls_end: DS4_TOOL_CALLS_END_SHORT,
            invoke_start: DS4_INVOKE_START_SHORT,
            invoke_end: DS4_INVOKE_END_SHORT,
            param_start: DS4_PARAM_START_SHORT,
            param_end: DS4_PARAM_END_SHORT,
        },
    };

    let start_rel = text[raw_block_start..]
        .find(tags.tool_calls_start)
        .ok_or(DsmlParseError)?;
    let mut pos = raw_block_start + start_rel + tags.tool_calls_start.len();
    let mut calls = Vec::new();
    let raw_dsml;
    loop {
        pos = skip_ascii_ws(text, pos);
        if text[pos..].starts_with(tags.tool_calls_end) {
            let raw_end = pos + tags.tool_calls_end.len();
            raw_dsml = Some(text[raw_block_start..raw_end].to_string());
            break;
        }
        if !text[pos..].starts_with(tags.invoke_start) {
            return Err(DsmlParseError);
        }
        let tag_end = find_from(text, pos, ">").ok_or(DsmlParseError)?;
        let tag = &text[pos..tag_end + 1];
        let name = dsml_attr(tag, "name").ok_or(DsmlParseError)?;
        pos = tag_end + 1;

        let mut args = String::new();
        loop {
            pos = skip_ascii_ws(text, pos);
            if text[pos..].starts_with(tags.invoke_end) {
                pos += tags.invoke_end.len();
                break;
            }
            if !text[pos..].starts_with(tags.param_start) {
                return Err(DsmlParseError);
            }
            let tag_end = find_from(text, pos, ">").ok_or(DsmlParseError)?;
            let tag = &text[pos..tag_end + 1];
            let param_name = dsml_attr(tag, "name").ok_or(DsmlParseError)?;
            let param_is_string = dsml_attr(tag, "string");
            let value_start = tag_end + 1;
            if param_is_string.is_none()
                && text[skip_ascii_ws(text, value_start)..].starts_with(tags.param_start)
            {
                let mut nested_pos = value_start;
                let nested = parse_nested_params_object(text, &mut nested_pos, tags)?;
                tool_call_json_args_add(&mut args, &param_name, &nested, "false");
                pos = skip_ascii_ws(text, nested_pos);
                if text[pos..].starts_with(tags.param_end) {
                    pos += tags.param_end.len();
                }
                continue;
            }

            let value_end = find_from(text, value_start, tags.param_end).ok_or(DsmlParseError)?;
            let raw_value = &text[value_start..value_end];
            let ty = param_is_string.as_deref().unwrap_or("true");
            let value = if ty == "true" {
                dsml_unescape_text(raw_value)
            } else {
                raw_value.to_string()
            };
            tool_call_json_args_add(&mut args, &param_name, &value, ty);
            pos = value_end + tags.param_end.len();
        }

        calls.push(DsmlJsonCall {
            id: None,
            name,
            arguments: format!("{{{args}}}"),
        });
    }

    let (content, reasoning) = split_reasoning_content(text, content_len);
    Ok(ParsedGeneratedMessage {
        content,
        reasoning,
        raw_dsml,
        calls,
    })
}

pub fn parse_generated_message_for_response(
    text: &str,
    has_tools: bool,
    saw_tool_start: bool,
    require_thinking_closed: bool,
    finish: &str,
) -> ResponseParse {
    match parse_generated_message(text, require_thinking_closed) {
        Ok(message) => ResponseParse {
            parse_ok: true,
            recovered: false,
            finish: finish.to_string(),
            error: String::new(),
            message,
        },
        Err(_) => {
            let mut recovered = false;
            let mut finish_after = finish.to_string();
            let mut error = String::new();
            if has_tools && saw_tool_start && finish != "error" {
                recovered = true;
                finish_after = if finish == "length" {
                    "length".to_string()
                } else {
                    "stop".to_string()
                };
                error = "invalid tool call".to_string();
            }
            ResponseParse {
                parse_ok: false,
                recovered,
                finish: finish_after,
                error,
                message: ParsedGeneratedMessage {
                    content: text.to_string(),
                    reasoning: None,
                    raw_dsml: None,
                    calls: Vec::new(),
                },
            }
        }
    }
}

fn append_dsml_arg(out: &mut String, arg: &DsmlArgument) {
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
    out.push_str(DS4_PARAM_END);
    out.push('\n');
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
    append_escaped_sentinel(out, text, DS4_PARAM_END, "&lt;");
}

fn append_dsml_json_literal(out: &mut String, text: &str) {
    append_escaped_sentinel(out, text, DS4_PARAM_END, "\\u003c");
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

fn parse_json_arguments(json: &str) -> Option<Vec<DsmlArgument>> {
    let mut pos = skip_json_ws(json, 0);
    if json.as_bytes().get(pos) != Some(&b'{') {
        return None;
    }
    pos += 1;
    let mut args = Vec::new();
    pos = skip_json_ws(json, pos);
    while pos < json.len() && json.as_bytes()[pos] != b'}' {
        let (key, next) = parse_json_string(json, pos)?;
        pos = skip_json_ws(json, next);
        if json.as_bytes().get(pos) != Some(&b':') {
            return None;
        }
        pos = skip_json_ws(json, pos + 1);
        if json.as_bytes().get(pos) == Some(&b'"') {
            let (value, next) = parse_json_string(json, pos)?;
            args.push(DsmlArgument {
                name: key,
                value,
                is_string: true,
            });
            pos = next;
        } else {
            let (raw, next) = raw_json_value(json, pos)?;
            args.push(DsmlArgument {
                name: key,
                value: minify_json_raw_value(raw),
                is_string: false,
            });
            pos = next;
        }
        pos = skip_json_ws(json, pos);
        if json.as_bytes().get(pos) == Some(&b',') {
            pos = skip_json_ws(json, pos + 1);
        }
    }
    if json.as_bytes().get(pos) != Some(&b'}') {
        return None;
    }
    Some(args)
}

fn tool_call_json_args_add(out: &mut String, name: &str, value: &str, is_string: &str) {
    if !out.is_empty() {
        out.push_str(", ");
    }
    out.push_str(&json_escape(name));
    out.push_str(": ");
    if is_string == "true" {
        out.push_str(&json_escape(value));
    } else {
        let minified = minify_json_raw_value(value);
        if minified.is_empty() {
            out.push_str("null");
        } else {
            out.push_str(&minified);
        }
    }
}

fn parse_nested_params_object(
    text: &str,
    pos: &mut usize,
    tags: Tags,
) -> Result<String, DsmlParseError> {
    let mut members = String::new();
    let mut any = false;
    loop {
        *pos = skip_ascii_ws(text, *pos);
        if !text[*pos..].starts_with(tags.param_start) {
            break;
        }
        parse_leaf_param_json(text, pos, tags.param_start, tags.param_end, &mut members)?;
        any = true;
    }
    if !any {
        return Err(DsmlParseError);
    }
    Ok(format!("{{{members}}}"))
}

fn parse_leaf_param_json(
    text: &str,
    pos: &mut usize,
    param_start: &str,
    param_end: &str,
    out: &mut String,
) -> Result<(), DsmlParseError> {
    if !text[*pos..].starts_with(param_start) {
        return Err(DsmlParseError);
    }
    let tag_end = find_from(text, *pos, ">").ok_or(DsmlParseError)?;
    let tag = &text[*pos..tag_end + 1];
    let name = dsml_attr(tag, "name").ok_or(DsmlParseError)?;
    let is_string = dsml_attr(tag, "string");
    let value_start = tag_end + 1;
    let value_end = find_from(text, value_start, param_end).ok_or(DsmlParseError)?;
    let raw_value = &text[value_start..value_end];
    let ty = is_string.as_deref().unwrap_or("true");
    let value = if ty == "true" {
        dsml_unescape_text(raw_value)
    } else {
        raw_value.to_string()
    };
    tool_call_json_args_add(out, &name, &value, ty);
    *pos = value_end + param_end.len();
    Ok(())
}

fn dsml_attr(tag: &str, name: &str) -> Option<String> {
    let pat = format!("{name}=\"");
    let start = tag.find(&pat)? + pat.len();
    let end = tag[start..].find('"')? + start;
    Some(dsml_unescape_text(&tag[start..end]))
}

fn dsml_unescape_text(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while !rest.is_empty() {
        if let Some(after) = rest.strip_prefix("&amp;") {
            out.push('&');
            rest = after;
        } else if let Some(after) = rest.strip_prefix("&lt;") {
            out.push('<');
            rest = after;
        } else if let Some(after) = rest.strip_prefix("&gt;") {
            out.push('>');
            rest = after;
        } else if let Some(after) = rest.strip_prefix("&quot;") {
            out.push('"');
            rest = after;
        } else if let Some(after) = rest.strip_prefix("&apos;") {
            out.push('\'');
            rest = after;
        } else {
            let ch = rest.chars().next().expect("nonempty string has char");
            out.push(ch);
            rest = &rest[ch.len_utf8()..];
        }
    }
    out
}

fn split_reasoning_content(text: &str, len: usize) -> (String, Option<String>) {
    let s = &text[..len];
    let body = s.strip_prefix("<think>").unwrap_or(s);
    if let Some(end) = body.find("</think>") {
        (
            body[end + "</think>".len()..].to_string(),
            Some(body[..end].to_string()),
        )
    } else {
        (s.to_string(), None)
    }
}

#[derive(Clone, Copy)]
struct Tags {
    tool_calls_start: &'static str,
    tool_calls_end: &'static str,
    invoke_start: &'static str,
    invoke_end: &'static str,
    param_start: &'static str,
    param_end: &'static str,
}

#[derive(Clone, Copy)]
enum ToolStyle {
    Canonical,
    PlainXml,
    Short,
}

fn find_tool_start(s: &str) -> Option<(usize, ToolStyle)> {
    if let Some(pos) = s.find(&format!("\n\n{DS4_TOOL_CALLS_START}")) {
        return Some((pos, ToolStyle::Canonical));
    }
    if let Some(pos) = s.find(DS4_TOOL_CALLS_START) {
        return Some((pos, ToolStyle::Canonical));
    }
    if let Some(pos) = s.find(&format!("\n\n{DS4_TOOL_CALLS_START_SHORT}")) {
        return Some((pos, ToolStyle::Short));
    }
    if let Some(pos) = s.find(DS4_TOOL_CALLS_START_SHORT) {
        return Some((pos, ToolStyle::Short));
    }
    if let Some(pos) = s.find("\n\n<tool_calls>") {
        return Some((pos, ToolStyle::PlainXml));
    }
    if let Some(pos) = s.find("<tool_calls>") {
        return Some((pos, ToolStyle::PlainXml));
    }
    None
}

fn trim_ascii_ws_end(text: &str, mut end: usize) -> usize {
    while end > 0 && text.as_bytes()[end - 1].is_ascii_whitespace() {
        end -= 1;
    }
    end
}

fn skip_ascii_ws(text: &str, mut pos: usize) -> usize {
    while pos < text.len() && text.as_bytes()[pos].is_ascii_whitespace() {
        pos += 1;
    }
    pos
}

fn find_from(text: &str, pos: usize, pat: &str) -> Option<usize> {
    text[pos..].find(pat).map(|rel| pos + rel)
}

fn skip_json_ws(text: &str, mut pos: usize) -> usize {
    while pos < text.len() && text.as_bytes()[pos].is_ascii_whitespace() {
        pos += 1;
    }
    pos
}

fn parse_json_string(text: &str, mut pos: usize) -> Option<(String, usize)> {
    pos = skip_json_ws(text, pos);
    let bytes = text.as_bytes();
    if bytes.get(pos) != Some(&b'"') {
        return None;
    }
    pos += 1;
    let mut out = String::new();
    while pos < bytes.len() && bytes[pos] != b'"' {
        if bytes[pos] != b'\\' {
            let ch = text[pos..].chars().next()?;
            out.push(ch);
            pos += ch.len_utf8();
            continue;
        }
        pos += 1;
        let esc = *bytes.get(pos)?;
        pos += 1;
        match esc {
            b'"' => out.push('"'),
            b'\\' => out.push('\\'),
            b'/' => out.push('/'),
            b'b' => out.push('\u{0008}'),
            b'f' => out.push('\u{000c}'),
            b'n' => out.push('\n'),
            b'r' => out.push('\r'),
            b't' => out.push('\t'),
            b'u' => {
                pos -= 2;
                let (mut cp, next) = parse_json_u16(text, pos)?;
                pos = next;
                if (0xd800..=0xdbff).contains(&cp) {
                    if let Some((lo, next)) = parse_json_u16(text, pos) {
                        if (0xdc00..=0xdfff).contains(&lo) {
                            cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00);
                            pos = next;
                        }
                    }
                }
                out.push(char::from_u32(cp).unwrap_or('\u{fffd}'));
            }
            _ => return None,
        }
    }
    if bytes.get(pos) != Some(&b'"') {
        return None;
    }
    Some((out, pos + 1))
}

fn parse_json_u16(text: &str, pos: usize) -> Option<(u32, usize)> {
    let bytes = text.as_bytes();
    if bytes.get(pos) != Some(&b'\\') || bytes.get(pos + 1) != Some(&b'u') {
        return None;
    }
    let mut cp = 0u32;
    for idx in pos + 2..pos + 6 {
        let h = *bytes.get(idx)?;
        let digit = match h {
            b'0'..=b'9' => (h - b'0') as u32,
            b'a'..=b'f' => 10 + (h - b'a') as u32,
            b'A'..=b'F' => 10 + (h - b'A') as u32,
            _ => return None,
        };
        cp = (cp << 4) | digit;
    }
    Some((cp, pos + 6))
}

fn raw_json_value(text: &str, pos: usize) -> Option<(&str, usize)> {
    let start = skip_json_ws(text, pos);
    let end = skip_json_value(text, start, 0)?;
    Some((&text[start..end], end))
}

fn skip_json_value(text: &str, pos: usize, depth: usize) -> Option<usize> {
    let pos = skip_json_ws(text, pos);
    match text.as_bytes().get(pos)? {
        b'"' => skip_json_string(text, pos),
        b'{' => skip_json_object(text, pos, depth),
        b'[' => skip_json_array(text, pos, depth),
        b't' if text[pos..].starts_with("true") => Some(pos + 4),
        b'f' if text[pos..].starts_with("false") => Some(pos + 5),
        b'n' if text[pos..].starts_with("null") => Some(pos + 4),
        _ => skip_json_number(text, pos),
    }
}

fn skip_json_string(text: &str, mut pos: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    if bytes.get(pos) != Some(&b'"') {
        return None;
    }
    pos += 1;
    while pos < bytes.len() {
        match bytes[pos] {
            b'"' => return Some(pos + 1),
            b'\\' => {
                pos += 1;
                match *bytes.get(pos)? {
                    b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => pos += 1,
                    b'u' => {
                        for idx in pos + 1..pos + 5 {
                            if !bytes.get(idx)?.is_ascii_hexdigit() {
                                return None;
                            }
                        }
                        pos += 5;
                    }
                    _ => return None,
                }
            }
            _ => {
                let ch = text[pos..].chars().next()?;
                pos += ch.len_utf8();
            }
        }
    }
    None
}

fn skip_json_object(text: &str, mut pos: usize, depth: usize) -> Option<usize> {
    if depth >= 256 || text.as_bytes().get(pos) != Some(&b'{') {
        return None;
    }
    pos = skip_json_ws(text, pos + 1);
    if text.as_bytes().get(pos) == Some(&b'}') {
        return Some(pos + 1);
    }
    loop {
        pos = skip_json_string(text, pos)?;
        pos = skip_json_ws(text, pos);
        if text.as_bytes().get(pos) != Some(&b':') {
            return None;
        }
        pos = skip_json_value(text, pos + 1, depth + 1)?;
        pos = skip_json_ws(text, pos);
        match text.as_bytes().get(pos)? {
            b'}' => return Some(pos + 1),
            b',' => pos = skip_json_ws(text, pos + 1),
            _ => return None,
        }
    }
}

fn skip_json_array(text: &str, mut pos: usize, depth: usize) -> Option<usize> {
    if depth >= 256 || text.as_bytes().get(pos) != Some(&b'[') {
        return None;
    }
    pos = skip_json_ws(text, pos + 1);
    if text.as_bytes().get(pos) == Some(&b']') {
        return Some(pos + 1);
    }
    loop {
        pos = skip_json_value(text, pos, depth + 1)?;
        pos = skip_json_ws(text, pos);
        match text.as_bytes().get(pos)? {
            b']' => return Some(pos + 1),
            b',' => pos = skip_json_ws(text, pos + 1),
            _ => return None,
        }
    }
}

fn skip_json_number(text: &str, mut pos: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    if bytes.get(pos) == Some(&b'-') {
        pos += 1;
    }
    let int_start = pos;
    while pos < bytes.len() && bytes[pos].is_ascii_digit() {
        pos += 1;
    }
    if pos == int_start {
        return None;
    }
    if bytes.get(pos) == Some(&b'.') {
        pos += 1;
        let frac_start = pos;
        while pos < bytes.len() && bytes[pos].is_ascii_digit() {
            pos += 1;
        }
        if pos == frac_start {
            return None;
        }
    }
    if matches!(bytes.get(pos), Some(b'e' | b'E')) {
        pos += 1;
        if matches!(bytes.get(pos), Some(b'+' | b'-')) {
            pos += 1;
        }
        let exp_start = pos;
        while pos < bytes.len() && bytes[pos].is_ascii_digit() {
            pos += 1;
        }
        if pos == exp_start {
            return None;
        }
    }
    Some(pos)
}

fn minify_json_raw_value(json: &str) -> String {
    let Some((raw, _)) = raw_json_value(json, 0) else {
        return json.to_string();
    };
    let mut out = String::new();
    let mut in_string = false;
    let mut escape = false;
    for ch in raw.chars() {
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

fn json_escape(text: &str) -> String {
    let mut out = String::new();
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if ch < ' ' => out.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => out.push(ch),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_result_escapes_only_closing_tag_start() {
        assert_eq!(
            render_tool_result_text("a </tool_result> & b"),
            "a &lt;/tool_result> & b"
        );
    }

    #[test]
    fn parser_ignores_tool_calls_before_think_close_when_required() {
        let parsed = parse_generated_message(
            "<think>x\n\n<｜DSML｜tool_calls>\n</｜DSML｜tool_calls>\n</think>done",
            true,
        )
        .expect("parse");
        assert_eq!(parsed.content, "done");
        assert_eq!(
            parsed.reasoning.as_deref(),
            Some("x\n\n<｜DSML｜tool_calls>\n</｜DSML｜tool_calls>\n")
        );
        assert!(parsed.calls.is_empty());
    }

    #[test]
    fn parser_minifies_json_parameters() {
        let parsed = parse_generated_message(
            "done\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"edit\">\n<｜DSML｜parameter name=\"edits\" string=\"false\">[{\"old\": \"x\", \"new\": \"y\"}]</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
            false,
        )
        .expect("parse");
        assert_eq!(
            parsed.calls[0].arguments,
            "{\"edits\": [{\"old\":\"x\",\"new\":\"y\"}]}"
        );
    }
}
