const MAX_HEADER_BYTES: usize = 64 * 1024;
const MAX_BODY_BYTES: usize = 64 * 1024 * 1024;
pub const DS4_MODEL_ID: &str = "deepseek-v4-flash";
const DS4_MODEL_ROUTE_PATH: &str = "/v1/models/deepseek-v4-flash";
const MODEL_NAME: &str = "DeepSeek V4 Flash";
const MODEL_CREATED: i64 = 1_767_225_600;
const MODEL_OWNER: &str = "ds4.c";
const MODEL_SUPPORTED_PARAMETERS_JSON: &str = "[\"tools\",\"tool_choice\",\"max_tokens\",\"temperature\",\"top_p\",\"top_k\",\"min_p\",\"stop\",\"seed\",\"stream\",\"reasoning_effort\"]";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpRequest {
    pub method: String,
    pub path: String,
    pub body: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NoModelRouteConfig {
    pub enable_cors: bool,
    pub context_length: i32,
    pub default_tokens: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HttpRequestParseError {
    Incomplete,
    Malformed,
    HeaderTooLarge,
    BodyTooLarge,
}

pub fn parse_http_request(input: &[u8]) -> Result<HttpRequest, HttpRequestParseError> {
    let header_end = header_end(input).ok_or_else(|| {
        if input.len() >= MAX_HEADER_BYTES {
            HttpRequestParseError::HeaderTooLarge
        } else {
            HttpRequestParseError::Incomplete
        }
    })?;
    if header_end > MAX_HEADER_BYTES {
        return Err(HttpRequestParseError::HeaderTooLarge);
    }

    let header =
        std::str::from_utf8(&input[..header_end]).map_err(|_| HttpRequestParseError::Malformed)?;
    let request_line = header
        .lines()
        .next()
        .map(|line| line.trim_end_matches('\r'))
        .ok_or(HttpRequestParseError::Malformed)?;
    let (method, path) = parse_request_line(request_line)?;

    let body_len = content_length(header)?;
    if body_len > MAX_BODY_BYTES {
        return Err(HttpRequestParseError::BodyTooLarge);
    }
    let body_end = header_end
        .checked_add(body_len)
        .ok_or(HttpRequestParseError::BodyTooLarge)?;
    if input.len() < body_end {
        return Err(HttpRequestParseError::Incomplete);
    }
    let body = std::str::from_utf8(&input[header_end..body_end])
        .map_err(|_| HttpRequestParseError::Malformed)?
        .to_string();

    Ok(HttpRequest { method, path, body })
}

pub fn format_model_metadata_json(context_length: i32, default_tokens: i32) -> String {
    let max_completion_tokens = if default_tokens < context_length {
        default_tokens
    } else {
        context_length
    };
    format!(
        "{{\"id\":\"{DS4_MODEL_ID}\",\
\"object\":\"model\",\
\"created\":{MODEL_CREATED},\
\"owned_by\":\"{MODEL_OWNER}\",\
\"name\":\"{MODEL_NAME}\",\
\"context_length\":{context_length},\
\"top_provider\":{{\
\"context_length\":{context_length},\
\"max_completion_tokens\":{max_completion_tokens},\
\"is_moderated\":false}},\
\"supported_parameters\":{MODEL_SUPPORTED_PARAMETERS_JSON}}}"
    )
}

pub fn route_no_model_http(input: &[u8], config: NoModelRouteConfig) -> String {
    match parse_http_request(input) {
        Ok(request) => route_no_model_request(&request, config),
        Err(_) => format_http_error(config.enable_cors, 400, "bad HTTP request"),
    }
}

pub fn route_no_model_request(request: &HttpRequest, config: NoModelRouteConfig) -> String {
    if request.method == "OPTIONS" {
        return format_http_response(config.enable_cors, 204, None, "");
    }

    if request.method == "GET" && request.path == "/v1/models" {
        return format_http_response(
            config.enable_cors,
            200,
            Some("application/json"),
            &format_models_body(config),
        );
    }

    if request.method == "GET" && request.path == DS4_MODEL_ROUTE_PATH {
        return format_http_response(
            config.enable_cors,
            200,
            Some("application/json"),
            &format_model_body(config),
        );
    }

    format_http_error(config.enable_cors, 404, "unknown endpoint")
}

pub fn format_http_response(
    enable_cors: bool,
    code: u16,
    content_type: Option<&str>,
    body: &str,
) -> String {
    let mut out = String::new();
    out.push_str("HTTP/1.1 ");
    out.push_str(&code.to_string());
    out.push(' ');
    out.push_str(reason_phrase(code));
    out.push_str("\r\nContent-Length: ");
    out.push_str(&body.len().to_string());
    out.push_str("\r\n");
    if let Some(content_type) = content_type.filter(|content_type| !content_type.is_empty()) {
        out.push_str("Content-Type: ");
        out.push_str(content_type);
        out.push_str("\r\n");
    }
    if enable_cors {
        append_cors_headers(&mut out);
    }
    out.push_str("Connection: close\r\n\r\n");
    out.push_str(body);
    out
}

pub fn format_http_error(enable_cors: bool, code: u16, message: &str) -> String {
    let body = format!(
        "{{\"error\":{{\"message\":{},\"type\":\"invalid_request_error\"}}}}\n",
        json_escape_string(message)
    );
    format_http_response(enable_cors, code, Some("application/json"), &body)
}

fn parse_request_line(line: &str) -> Result<(String, String), HttpRequestParseError> {
    let mut parts = line.split_whitespace();
    let method = parts.next().ok_or(HttpRequestParseError::Malformed)?;
    let path = parts.next().ok_or(HttpRequestParseError::Malformed)?;
    if method.is_empty() || method.len() > 7 || path.is_empty() || path.len() > 255 {
        return Err(HttpRequestParseError::Malformed);
    }
    let path = path.split_once('?').map_or(path, |(path, _)| path);
    Ok((method.to_string(), path.to_string()))
}

fn format_model_body(config: NoModelRouteConfig) -> String {
    let mut body = format_model_metadata_json(config.context_length, config.default_tokens);
    body.push('\n');
    body
}

fn format_models_body(config: NoModelRouteConfig) -> String {
    let model = format_model_metadata_json(config.context_length, config.default_tokens);
    format!("{{\"object\":\"list\",\"data\":[{model}]}}\n")
}

fn header_end(input: &[u8]) -> Option<usize> {
    for idx in 3..input.len() {
        if input[idx - 3] == b'\r'
            && input[idx - 2] == b'\n'
            && input[idx - 1] == b'\r'
            && input[idx] == b'\n'
        {
            return Some(idx + 1);
        }
    }
    for idx in 1..input.len() {
        if input[idx - 1] == b'\n' && input[idx] == b'\n' {
            return Some(idx + 1);
        }
    }
    None
}

fn content_length(header: &str) -> Result<usize, HttpRequestParseError> {
    for line in header.lines() {
        let line = line.trim_end_matches('\r');
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        if !name.eq_ignore_ascii_case("Content-Length") {
            continue;
        }
        let value = value.trim_start();
        if value.starts_with('-') {
            return Err(HttpRequestParseError::Malformed);
        }
        let digits: String = value.chars().take_while(|ch| ch.is_ascii_digit()).collect();
        if digits.is_empty() {
            return Ok(0);
        }
        return digits
            .parse::<usize>()
            .map_err(|_| HttpRequestParseError::BodyTooLarge);
    }
    Ok(0)
}

fn reason_phrase(code: u16) -> &'static str {
    match code {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        404 => "Not Found",
        409 => "Conflict",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "Error",
    }
}

fn append_cors_headers(out: &mut String) {
    out.push_str(
        "Access-Control-Allow-Origin: *\r\n\
Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n\
Access-Control-Allow-Headers: *\r\n",
    );
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
            ch if ch <= '\u{001f}' => out.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => out.push(ch),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODEL_JSON_32768_64: &str = "{\"id\":\"deepseek-v4-flash\",\"object\":\"model\",\"created\":1767225600,\"owned_by\":\"ds4.c\",\"name\":\"DeepSeek V4 Flash\",\"context_length\":32768,\"top_provider\":{\"context_length\":32768,\"max_completion_tokens\":64,\"is_moderated\":false},\"supported_parameters\":[\"tools\",\"tool_choice\",\"max_tokens\",\"temperature\",\"top_p\",\"top_k\",\"min_p\",\"stop\",\"seed\",\"stream\",\"reasoning_effort\"]}";
    const M04_MODELS_JSON: &str =
        include_str!("../../../ds4-parity/baselines/server-traces/m0.4/responses/models.json");

    fn route_config(enable_cors: bool) -> NoModelRouteConfig {
        NoModelRouteConfig {
            enable_cors,
            context_length: 32768,
            default_tokens: 64,
        }
    }

    #[test]
    fn parses_request_line_query_and_exact_body() {
        let req = parse_http_request(
            b"POST /v1/chat/completions?trace=1 HTTP/1.1\r\nHost: x\r\nContent-Length: 7\r\n\r\n{\"a\":1}ignored",
        )
        .expect("request parses");
        assert_eq!(req.method, "POST");
        assert_eq!(req.path, "/v1/chat/completions");
        assert_eq!(req.body, "{\"a\":1}");
    }

    #[test]
    fn parses_lf_headers_and_case_insensitive_content_length() {
        let req = parse_http_request(b"GET /v1/models HTTP/1.1\ncontent-length: 0\n\n")
            .expect("request parses");
        assert_eq!(req.method, "GET");
        assert_eq!(req.path, "/v1/models");
        assert_eq!(req.body, "");
    }

    #[test]
    fn rejects_incomplete_malformed_and_too_large_requests() {
        assert_eq!(
            parse_http_request(b"GET / HTTP/1.1\r\n").unwrap_err(),
            HttpRequestParseError::Incomplete
        );
        assert_eq!(
            parse_http_request(b"GET\r\n\r\n").unwrap_err(),
            HttpRequestParseError::Malformed
        );
        assert_eq!(
            parse_http_request(b"POST / HTTP/1.1\r\nContent-Length: -1\r\n\r\n").unwrap_err(),
            HttpRequestParseError::Malformed
        );
        assert_eq!(
            parse_http_request(b"POST / HTTP/1.1\r\nContent-Length: 4\r\n\r\nxy").unwrap_err(),
            HttpRequestParseError::Incomplete
        );
    }

    #[test]
    fn formats_json_response_without_cors() {
        assert_eq!(
            format_http_response(false, 200, Some("application/json"), "{}"),
            "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{}"
        );
    }

    #[test]
    fn formats_cors_response_with_c_header_order() {
        assert_eq!(
            format_http_response(true, 200, Some("application/json"), "{}"),
            "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: *\r\nConnection: close\r\n\r\n{}"
        );
    }

    #[test]
    fn formats_preflight_no_content_without_content_type() {
        assert_eq!(
            format_http_response(true, 204, None, ""),
            "HTTP/1.1 204 No Content\r\nContent-Length: 0\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: *\r\nConnection: close\r\n\r\n"
        );
    }

    #[test]
    fn formats_json_error_body_like_c_helper() {
        assert_eq!(
            format_http_error(false, 400, "bad \"request\""),
            "HTTP/1.1 400 Bad Request\r\nContent-Length: 71\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{\"error\":{\"message\":\"bad \\\"request\\\"\",\"type\":\"invalid_request_error\"}}\n"
        );
    }

    #[test]
    fn formats_model_metadata_like_c_helper() {
        assert_eq!(format_model_metadata_json(32768, 64), MODEL_JSON_32768_64);
        assert_eq!(
            format_model_metadata_json(16, 64),
            "{\"id\":\"deepseek-v4-flash\",\"object\":\"model\",\"created\":1767225600,\"owned_by\":\"ds4.c\",\"name\":\"DeepSeek V4 Flash\",\"context_length\":16,\"top_provider\":{\"context_length\":16,\"max_completion_tokens\":16,\"is_moderated\":false},\"supported_parameters\":[\"tools\",\"tool_choice\",\"max_tokens\",\"temperature\",\"top_p\",\"top_k\",\"min_p\",\"stop\",\"seed\",\"stream\",\"reasoning_effort\"]}"
        );
    }

    #[test]
    fn routes_options_to_no_content_with_cors() {
        assert_eq!(
            route_no_model_http(
                b"OPTIONS /anything HTTP/1.1\r\nHost: x\r\n\r\n",
                route_config(true)
            ),
            "HTTP/1.1 204 No Content\r\nContent-Length: 0\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: *\r\nConnection: close\r\n\r\n"
        );
    }

    #[test]
    fn routes_model_list_and_single_model() {
        let models_body = format!("{{\"object\":\"list\",\"data\":[{MODEL_JSON_32768_64}]}}\n");
        assert_eq!(models_body, M04_MODELS_JSON);
        assert_eq!(
            route_no_model_http(b"GET /v1/models HTTP/1.1\r\nHost: x\r\n\r\n", route_config(false)),
            format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{}",
                models_body.len(),
                models_body
            )
        );

        let model_body = format!("{MODEL_JSON_32768_64}\n");
        assert_eq!(
            route_no_model_http(
                b"GET /v1/models/deepseek-v4-flash HTTP/1.1\r\nHost: x\r\n\r\n",
                route_config(false),
            ),
            format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{}",
                model_body.len(),
                model_body
            )
        );
    }

    #[test]
    fn routes_query_stripped_model_path() {
        let response = route_no_model_http(
            b"GET /v1/models?trace=1 HTTP/1.1\r\nHost: x\r\n\r\n",
            route_config(true),
        );
        assert!(response.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(response.contains("Access-Control-Allow-Origin: *\r\n"));
        assert!(response.ends_with(&format!(
            "{{\"object\":\"list\",\"data\":[{MODEL_JSON_32768_64}]}}\n"
        )));
    }

    #[test]
    fn routes_bad_http_unknown_endpoint_and_wrong_method() {
        assert_eq!(
            route_no_model_http(b"GET\r\n\r\n", route_config(false)),
            "HTTP/1.1 400 Bad Request\r\nContent-Length: 72\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{\"error\":{\"message\":\"bad HTTP request\",\"type\":\"invalid_request_error\"}}\n"
        );
        assert_eq!(
            route_no_model_http(b"GET /unknown HTTP/1.1\r\nHost: x\r\n\r\n", route_config(false)),
            "HTTP/1.1 404 Not Found\r\nContent-Length: 72\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{\"error\":{\"message\":\"unknown endpoint\",\"type\":\"invalid_request_error\"}}\n"
        );
        assert_eq!(
            route_no_model_http(
                b"POST /v1/models HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n\r\n",
                route_config(false),
            ),
            "HTTP/1.1 404 Not Found\r\nContent-Length: 72\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{\"error\":{\"message\":\"unknown endpoint\",\"type\":\"invalid_request_error\"}}\n"
        );
    }
}
