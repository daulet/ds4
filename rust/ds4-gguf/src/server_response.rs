use crate::server_http::{format_http_response, json_escape_string};

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

pub fn format_openai_chat_completion_json(response: &OpenAiChatCompletion<'_>) -> String {
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
