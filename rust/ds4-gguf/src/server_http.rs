const MAX_HEADER_BYTES: usize = 64 * 1024;
const MAX_BODY_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpRequest {
    pub method: String,
    pub path: String,
    pub body: String,
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
}
