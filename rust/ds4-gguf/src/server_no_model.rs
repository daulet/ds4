use crate::server_chat::{
    anthropic_context_length_error_body, openai_context_length_error_body_for_param,
    parse_anthropic_core_request, parse_completion_core_request, parse_openai_chat_request,
    parse_responses_core_request, request_exceeds_context, ServerRequestError,
};
use crate::server_http::{
    format_http_error, format_http_response, parse_http_request, route_no_model_request,
    HttpRequest, NoModelRouteConfig,
};

const NO_MODEL_GENERATION_MESSAGE: &str = "generation is not available in no-model server";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GenerationRoute {
    ChatCompletions,
    Responses,
    Completions,
    AnthropicMessages,
}

struct ParsedGenerationRequest {
    route: GenerationRoute,
    prompt_text: String,
}

impl ParsedGenerationRequest {
    fn context_error_body(&self, n_prompt_tokens: usize, ctx_size: i32) -> String {
        match self.route {
            GenerationRoute::ChatCompletions => {
                openai_context_length_error_body_for_param(n_prompt_tokens, ctx_size, "messages")
            }
            GenerationRoute::Responses => {
                openai_context_length_error_body_for_param(n_prompt_tokens, ctx_size, "input")
            }
            GenerationRoute::Completions => {
                openai_context_length_error_body_for_param(n_prompt_tokens, ctx_size, "prompt")
            }
            GenerationRoute::AnthropicMessages => {
                anthropic_context_length_error_body(n_prompt_tokens, ctx_size)
            }
        }
    }
}

pub fn route_no_model_server_http(input: &[u8], config: NoModelRouteConfig) -> String {
    route_no_model_server_http_with_prompt_tokens(input, config, |_| 0)
}

pub fn route_no_model_server_http_with_prompt_tokens(
    input: &[u8],
    config: NoModelRouteConfig,
    prompt_tokens: impl FnMut(&str) -> usize,
) -> String {
    route_no_model_server_http_with_generation_message(
        input,
        config,
        prompt_tokens,
        NO_MODEL_GENERATION_MESSAGE,
    )
}

pub fn route_no_model_server_http_with_generation_message(
    input: &[u8],
    config: NoModelRouteConfig,
    prompt_tokens: impl FnMut(&str) -> usize,
    generation_message: &str,
) -> String {
    match parse_http_request(input) {
        Ok(request) => route_no_model_server_request_with_generation_message(
            &request,
            config,
            prompt_tokens,
            generation_message,
        ),
        Err(_) => format_http_error(config.enable_cors, 400, "bad HTTP request"),
    }
}

pub fn route_no_model_server_request(request: &HttpRequest, config: NoModelRouteConfig) -> String {
    route_no_model_server_request_with_prompt_tokens(request, config, |_| 0)
}

pub fn route_no_model_server_request_with_prompt_tokens(
    request: &HttpRequest,
    config: NoModelRouteConfig,
    mut prompt_tokens: impl FnMut(&str) -> usize,
) -> String {
    route_no_model_server_request_with_generation_message(
        request,
        config,
        &mut prompt_tokens,
        NO_MODEL_GENERATION_MESSAGE,
    )
}

pub fn route_no_model_server_request_with_generation_message(
    request: &HttpRequest,
    config: NoModelRouteConfig,
    mut prompt_tokens: impl FnMut(&str) -> usize,
    generation_message: &str,
) -> String {
    let Some(route) = generation_route(request) else {
        return route_no_model_request(request, config);
    };

    let parsed = match parse_generation_request(route, request, config) {
        Ok(parsed) => parsed,
        Err(error) => return format_request_error(config, &error),
    };

    let n_prompt_tokens = prompt_tokens(&parsed.prompt_text);
    if request_exceeds_context(n_prompt_tokens, config.context_length) {
        let body = parsed.context_error_body(n_prompt_tokens, config.context_length);
        return format_http_response(config.enable_cors, 400, Some("application/json"), &body);
    }

    format_http_error(config.enable_cors, 503, generation_message)
}

fn generation_route(request: &HttpRequest) -> Option<GenerationRoute> {
    if request.method != "POST" {
        return None;
    }
    match request.path.as_str() {
        "/v1/chat/completions" => Some(GenerationRoute::ChatCompletions),
        "/v1/responses" => Some(GenerationRoute::Responses),
        "/v1/completions" => Some(GenerationRoute::Completions),
        "/v1/messages" => Some(GenerationRoute::AnthropicMessages),
        _ => None,
    }
}

fn parse_generation_request(
    route: GenerationRoute,
    request: &HttpRequest,
    config: NoModelRouteConfig,
) -> Result<ParsedGenerationRequest, ServerRequestError> {
    let prompt_text = match route {
        GenerationRoute::ChatCompletions => {
            parse_openai_chat_request(&request.body, config.default_tokens, config.context_length)?
                .prompt_text
        }
        GenerationRoute::Responses => {
            parse_responses_core_request(
                &request.body,
                config.default_tokens,
                config.context_length,
            )?
            .prompt_text
        }
        GenerationRoute::Completions => {
            parse_completion_core_request(
                &request.body,
                config.default_tokens,
                config.context_length,
            )?
            .prompt_text
        }
        GenerationRoute::AnthropicMessages => {
            parse_anthropic_core_request(
                &request.body,
                config.default_tokens,
                config.context_length,
            )?
            .prompt_text
        }
    };
    Ok(ParsedGenerationRequest { route, prompt_text })
}

fn format_request_error(config: NoModelRouteConfig, error: &ServerRequestError) -> String {
    format_http_error(config.enable_cors, 400, error.message())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(enable_cors: bool) -> NoModelRouteConfig {
        NoModelRouteConfig {
            enable_cors,
            context_length: 32768,
            default_tokens: 64,
        }
    }

    fn post(path: &str, body: &str) -> Vec<u8> {
        format!(
            "POST {path} HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .into_bytes()
    }

    #[test]
    fn dispatch_reuses_no_model_routes_and_bad_http_error() {
        assert_eq!(
            route_no_model_server_http(b"GET\r\n\r\n", config(false)),
            "HTTP/1.1 400 Bad Request\r\nContent-Length: 72\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{\"error\":{\"message\":\"bad HTTP request\",\"type\":\"invalid_request_error\"}}\n"
        );

        let response = route_no_model_server_http(
            b"OPTIONS /v1/chat/completions HTTP/1.1\r\nHost: x\r\n\r\n",
            config(true),
        );
        assert_eq!(
            response,
            "HTTP/1.1 204 No Content\r\nContent-Length: 0\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: *\r\nConnection: close\r\n\r\n"
        );

        let response = route_no_model_server_http(
            b"GET /v1/models?probe=1 HTTP/1.1\r\nHost: x\r\n\r\n",
            config(false),
        );
        assert!(response.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(response.ends_with("]}\n"));
    }

    #[test]
    fn dispatch_maps_generation_parser_errors() {
        assert_eq!(
            route_no_model_server_http(&post("/v1/chat/completions", "{}"), config(false)),
            format_http_error(false, 400, "missing messages")
        );
        assert_eq!(
            route_no_model_server_http(&post("/v1/responses", "{}"), config(false)),
            format_http_error(false, 400, "missing input")
        );
        assert_eq!(
            route_no_model_server_http(&post("/v1/completions", "{}"), config(false)),
            format_http_error(false, 400, "missing prompt")
        );
        assert_eq!(
            route_no_model_server_http(&post("/v1/messages", "{"), config(false)),
            format_http_error(false, 400, "invalid JSON request")
        );
    }

    #[test]
    fn dispatch_maps_responses_durable_state_and_tool_choice_errors() {
        assert_eq!(
            route_no_model_server_http(
                &post(
                    "/v1/responses",
                    r#"{"input":"hi","previous_response_id":"resp_1"}"#
                ),
                config(false)
            ),
            format_http_error(
                false,
                400,
                "previous_response_id is not supported; replay full input instead"
            )
        );
        assert_eq!(
            route_no_model_server_http(
                &post(
                    "/v1/responses",
                    r#"{"input":"hi","tool_choice":"required"}"#
                ),
                config(true)
            ),
            format_http_error(true, 400, "tool_choice=required not supported")
        );
    }

    #[test]
    fn dispatch_returns_context_length_errors_for_each_api_shape() {
        let chat = post(
            "/v1/chat/completions",
            r#"{"messages":[{"role":"user","content":"hi"}]}"#,
        );
        assert_eq!(
            route_no_model_server_http_with_prompt_tokens(&chat, config(false), |_| 32768),
            format_http_response(
                false,
                400,
                Some("application/json"),
                &openai_context_length_error_body_for_param(32768, 32768, "messages")
            )
        );

        let responses = post("/v1/responses", r#"{"input":"hi"}"#);
        assert_eq!(
            route_no_model_server_http_with_prompt_tokens(&responses, config(false), |_| 32768),
            format_http_response(
                false,
                400,
                Some("application/json"),
                &openai_context_length_error_body_for_param(32768, 32768, "input")
            )
        );

        let completion = post("/v1/completions", r#"{"prompt":"hi"}"#);
        let mut completion_prompt_text = String::new();
        assert_eq!(
            route_no_model_server_http_with_prompt_tokens(&completion, config(false), |prompt| {
                completion_prompt_text = prompt.to_string();
                32768
            }),
            format_http_response(
                false,
                400,
                Some("application/json"),
                &openai_context_length_error_body_for_param(32768, 32768, "prompt")
            )
        );
        assert_eq!(
            completion_prompt_text,
            "<｜begin▁of▁sentence｜>You are a helpful assistant<｜User｜>hi<｜Assistant｜><think>"
        );

        let anthropic = post(
            "/v1/messages",
            r#"{"messages":[{"role":"user","content":"hi"}]}"#,
        );
        assert_eq!(
            route_no_model_server_http_with_prompt_tokens(&anthropic, config(false), |_| 32768),
            format_http_response(
                false,
                400,
                Some("application/json"),
                &anthropic_context_length_error_body(32768, 32768)
            )
        );
    }

    #[test]
    fn dispatch_rejects_successful_generation_without_model() {
        let response = route_no_model_server_http(
            &post(
                "/v1/chat/completions",
                r#"{"messages":[{"role":"user","content":"hi"}]}"#,
            ),
            config(false),
        );
        assert_eq!(
            response,
            format_http_error(false, 503, NO_MODEL_GENERATION_MESSAGE)
        );
    }

    #[test]
    fn dispatch_allows_model_backed_boundary_generation_message() {
        let response = route_no_model_server_http_with_generation_message(
            &post(
                "/v1/chat/completions",
                r#"{"messages":[{"role":"user","content":"hi"}]}"#,
            ),
            config(false),
            |_| 0,
            "model-backed chat generation is not implemented yet",
        );
        assert_eq!(
            response,
            format_http_error(
                false,
                503,
                "model-backed chat generation is not implemented yet"
            )
        );
    }
}
