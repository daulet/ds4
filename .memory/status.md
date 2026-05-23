# DS4 Rust Port Status

- Date: 2026-05-23 UTC
- Branch: `main`
- Starting oracle commit: `6975b57c196255e8ac4a22bb3be4dca18b92ebba`
- Active item: M9.6b Model-Backed Tool-Call Replay
- Last validated source commit: M9.6a OpenAI tool-call response formatter in
  this commit; prior pushed source commit
  `5b02e5e7452a1109cdb3e735e08da35c5d769bf1`
- Active debugging ledger: none
- B300 context: `hou2-prod1`
- B300 namespace: `default`
- B300 pod: `ds4-rust-port-b300`
- B300 node: `c1v17-b300n1-nic1`
- B300 temp kubeconfig: `/tmp/ds4-hou2-prod1.kubeconfig` for this local
  session; regenerate a temp copy in future sessions instead of treating this
  path as durable, and pass `--context hou2-prod1` explicitly because the temp
  kubeconfig can contain other contexts.
- Known local validation constraint: `ds4flash.gguf` is not present in the
  workspace, so model-backed tests and benchmark baselines need a model path or
  remote B300 execution.
- B300 model path: `/workspace/ds4/ds4flash.gguf`
- B300 model SHA256:
  `efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668`
- B300 model size: 86,720,111,488 bytes.

## Last Evidence

- M9.6a added pure OpenAI chat tool-call response formatting in
  `ds4_gguf::server_response`, reusing the existing chat completion response
  struct while adding explicit tool-call JSON/HTTP helpers.
- M9.6a normalizes parser-produced tool-call argument objects through the DSML
  JSON argument parser before escaping them as OpenAI `function.arguments`
  strings, matching C's `append_json_object_string` behavior and falling back
  to `{}` for invalid argument JSON.
- M9.6a tests compare the exact M0.4 `chat_tool_call` JSON body and HTTP
  headers, generated call IDs (`{chat_id}_tool_{index}`), explicit call IDs,
  multiple-call ordering, escaped names, normalized argument strings, and
  invalid-argument fallback without touching model execution or runtime
  routing.
- M9.6a local validation passed for targeted
  `cargo test -p ds4-gguf server_response -- --nocapture`, targeted
  `cargo test -p ds4-gguf dsml -- --nocapture`, full
  `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- M9.6a Claude review returned no blockers after checking response field
  order, argument normalization/escaping, call-ID fallback semantics,
  usage/cache preservation, and the absence of runtime/model routing changes.
- M9.6 was split before implementation because the remaining server tool-call
  work spans independent oracle surfaces: pure final-response JSON formatting,
  model-backed non-streaming replay, streaming tool-call deltas, and
  tool-quality regression wiring.
- M9.6a is active first because M5.6 already owns generated DSML parsing and
  M9.2b already owns model-free tool schema prompt rendering; a pure
  `tool_calls` response formatter is the smallest remaining prerequisite for
  the runtime replay.
- M9.6 split validation passed for `git diff --check`; implementation
  validation remains owned by M9.6a and later sub-items.
- M9.5 was split before implementation because pure SSE byte formatting and
  model-backed per-token streaming replay have distinct oracles, comparators,
  and validation gates.
- M9.5a added pure OpenAI chat SSE helpers for headers,
  role/content/final/usage chunks, optional usage omission, JSON escaping, and
  `[DONE]` formatting with injected IDs/timestamps and fixed deltas.
- M9.5a formatter tests compare exact M0.4 `chat_stream.sse` bytes and stream
  headers; the committed fixture ends with one newline after `[DONE]`, and the
  formatter follows that oracle byte-for-byte.
- M9.5a validation passed for targeted
  `cargo test -p ds4-gguf server_response -- --nocapture`, decode-policy
  streaming hold tests `cargo test -p ds4-gguf decode_policy -- --nocapture`,
  full `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- M9.5b added model-backed streaming routing for supported OpenAI chat
  requests, captured raw per-token text chunks from `ServerSession`, and fed
  those chunks through the M9.5a formatter for SSE responses.
- M9.5b converts token chunks into SSE deltas through the existing
  `utf8_stream_safe_len` helper, preserving ordinary token boundaries while
  holding split multi-byte UTF-8 bytes until they are safe to emit.
- M9.5b keeps tools, thinking, and stop-list requests outside the streaming
  path while preserving the existing non-streaming response behavior.
- M9.5b local validation passed for targeted
  `cargo test -p ds4-engine --bin ds4-server-runtime-rs -- --nocapture`,
  targeted `cargo test -p ds4-gguf server_response -- --nocapture`,
  decode-policy streaming hold tests
  `cargo test -p ds4-gguf decode_policy -- --nocapture`, full
  `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- M9.5b B300 validation used `/workspace/ds4-m95b` and model
  `/workspace/ds4/ds4flash.gguf`; targeted runtime tests passed, and the
  server replay on port `18198` normalized only IDs/timestamps while matching
  M0.4 `chat_stream` SSE headers/body, deltas `stream` and ` baseline`, finish
  `stop`, usage `11/2/13`, cache `0/11`, and one newline after `[DONE]`.
- M9.5b B300 trace validation checked `stream: 1`, `prompt_tokens: 11`,
  `stream_include_usage: 1`, `cache_source: none`, `generated_tokens: 2`, and
  final content `stream baseline`. Artifacts remain in the pod at
  `/tmp/ds4-m95b-server.trace`, `/tmp/ds4-m95b-server.stderr`, and
  `/tmp/ds4-m95b-chat_stream.*`; the server process was stopped by the
  validation script.
- M9.4d added a reusable `ServerSession` path for model-backed server
  generation so `/v1/chat/completions` requests in one Rust server process can
  reuse the live token prefix from prior completions.
- M9.4d reports server-generation cache read/write counts,
  `live_tokens_before`, and `live_prompt_common` from the Rust session, then
  forwards those counts into OpenAI usage details and `--trace` cache-decision
  fields.
- M9.4d local validation passed for targeted
  `cargo test -p ds4-engine --bin ds4-server-runtime-rs -- --nocapture`, full
  `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- M9.4d B300 validation used `/workspace/ds4-m94d` and model
  `/workspace/ds4/ds4flash.gguf`; targeted runtime tests passed, and a single
  server replay on port `18197` normalized only IDs/timestamps while matching
  `chat_cache_seed` content `cache ready`, finish `stop`, usage `39/2/41`,
  cache `0/39`, and `chat_cache_continuation` content `cache continued`,
  finish `stop`, usage `50/2/52`, cache `41/9`.
- M9.4d B300 trace validation checked rendered prompts, `cache_source: none`
  for the seed, `cache_source: memory-token`, `cached_tokens: 41`,
  `live_tokens_before: 41`, `live_prompt_common: 41`,
  `memory_token_reusable: 1`, generated token counts, and final content for
  the continuation. Artifacts remain in the pod at
  `/tmp/ds4-m94d-server.trace`, `/tmp/ds4-m94d-server.stderr`, and
  `/tmp/ds4-m94d-*.json`; the server process was stopped by the validation
  script.
- M9.4c added server-generation runtime support in `ds4-engine` that syncs a
  rendered prompt into a fresh session, samples raw token text without the CLI
  trailing-newline printer, returns prompt/completion token counts, and reports
  `stop`, `length`, or `error` finish reasons.
- M9.4c routes model-backed OpenAI `/v1/chat/completions` requests through the
  runtime only for the no-cache M0.4 surface: non-streaming, no tools,
  non-thinking, no stop-list requests. Streaming, tools, thinking, and stops
  remain explicit 503 boundaries for later M9 items.
- M9.4c uses the M9.4b response builder for successful chat responses and
  writes a `--trace` file with request metadata, no-cache decision fields,
  rendered prompt, generated text, finish reason, and generated-token count.
- M9.4c local validation passed for targeted
  `cargo test -p ds4-engine --bin ds4-server-runtime-rs -- --nocapture`, full
  `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- M9.4c B300 validation used `/workspace/ds4-m94c` and model
  `/workspace/ds4/ds4flash.gguf`; targeted runtime tests passed, and the
  server replay on port `18196` normalized only IDs/timestamps while matching
  `chat_basic` content `baseline ready`, finish `stop`, usage `11/3/14`, and
  `chat_thinking_disabled` content `2`, finish `stop`, usage `15/1/16`.
- M9.4c B300 trace validation checked rendered prompts, `cache_source: none`,
  prompt token counts, generated token counts, and final content for both
  no-cache fixtures. Artifacts remain in the pod at
  `/tmp/ds4-m94c-server.trace`, `/tmp/ds4-m94c-server.stderr`, and
  `/tmp/ds4-m94c-*.json`; the server process was stopped by the validation
  script.
- M9.4b added `rust/ds4-gguf/src/server_response.rs` with pure formatting
  helpers for OpenAI non-streaming chat-completion JSON, HTTP response
  wrapping, usage details, finish reasons, optional reasoning content, and
  C-compatible cache read/write clamping.
- M9.4b response-builder tests compare exact M0.4 `chat_basic`,
  `chat_thinking_disabled`, `chat_cache_seed`, and `chat_cache_continuation`
  JSON bodies using injected IDs/timestamps and explicit usage/cache counts.
- M9.4b header tests compare the `chat_basic` `Content-Length`,
  `Content-Type`, and `Connection: close` header surface through the existing
  C-shaped HTTP formatter; escaping tests cover visible content and optional
  reasoning content.
- M9.4b validation passed for targeted
  `cargo test -p ds4-gguf server_response -- --nocapture`, full
  `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- M9.4a added `ds4-server-runtime-rs` under `rust/ds4-engine` so the
  model-backed server boundary lives in the runtime crate and depends on
  `ds4-gguf` helpers without creating a dependency cycle.
- M9.4a parses the C server startup subset for model path, backend selection,
  MTP options, threads, directional steering, warm/quality flags, host, port,
  context length, default tokens, and CORS.
- M9.4a opens `Engine`, creates a session, uses `Engine::encode_chat_prompt`
  for tokenizer-backed prompt-token counts, preserves M9.3 no-model
  route/error behavior, and returns a distinct 503 JSON error for valid
  generation while model-backed chat generation remains unimplemented.
- M9.4a local validation passed for targeted
  `cargo test -p ds4-engine --bin ds4-server-runtime-rs -- --nocapture`,
  targeted `cargo test -p ds4-gguf server_no_model -- --nocapture`, full
  `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- M9.4a B300 validation used a copied source snapshot at `/workspace/ds4-m94a`
  and model `/workspace/ds4/ds4flash.gguf`; targeted runtime tests passed and
  the server smoke on port `18194` loaded CUDA, answered `/v1/models` with
  `context_length=16`, returned `missing messages` for bad chat JSON,
  returned a tokenizer-backed completion context error with
  `n_prompt_tokens=29` and `n_ctx=16`, and rejected valid chat generation with
  `model-backed chat generation is not implemented yet`.
- M9.4a B300 smoke artifacts remain in the pod at
  `/tmp/ds4-m94a-server.stderr` and `/tmp/ds4-m94a-*.out`; the server process
  was stopped by the validation script.
- M9.4 was split before implementation because model-backed server ownership,
  non-streaming response formatting, no-cache B300 generation replay, and
  memory-token cache continuation have distinct oracles and validation gates.
- M9.4a now owns the model-backed Rust server runtime boundary, engine/session
  lifetime, model-load smoke replay, tokenizer-backed prompt-token counting,
  and preservation of M9.3 no-model route/error behavior.
- M9.4b now owns pure OpenAI non-streaming chat response/usage/header
  formatting, with IDs and timestamps injected or normalized outside the
  builder.
- M9.4c now owns B300 no-cache non-streaming `/v1/chat/completions` replay for
  `chat_basic` and `chat_thinking_disabled`.
- M9.4d now owns live memory-token cache seed/continuation replay for
  `chat_cache_seed` and `chat_cache_continuation`, leaving disk KV/tool-memory
  behavior to M9.8.
- M9.3c2 added the `ds4-server-rs` binary with model-free `--host`, `--port`,
  `--ctx`, `--tokens`, and `--cors` startup flags plus C-style localhost
  binding behavior.
- M9.3c2 wires accepted TCP sockets through the M9.3c1 no-model dispatcher,
  reads until a complete request or stable parser failure, writes C-shaped
  HTTP responses, and closes the client socket.
- M9.3c2 added real process/socket replay coverage for OPTIONS, model list,
  unknown route, malformed HTTP, missing messages, unsupported durable state,
  unsupported tool choice, no-model context-limit response, and valid
  generation rejection through 503.
- M9.3c2 validation passed for local no-model HTTP comparator
  `cargo test -p ds4-gguf --test no_model_server -- --nocapture`, targeted
  binary tests `cargo test -p ds4-gguf --bin ds4-server-rs -- --nocapture`,
  full `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- M9.3c1 added `rust/ds4-gguf/src/server_no_model.rs` as a socket-free
  dispatcher that combines M9.3a/M9.3b HTTP helpers with generation-route
  parse/error handling.
- M9.3c1 added Rust completion request parsing for `/v1/completions` negative
  and no-model dispatch paths, including missing-prompt handling and C-shaped
  completion prompt rendering.
- M9.3c1 added API-specific context-length response bodies for OpenAI chat,
  Responses, completions, and Anthropic messages, with injectable prompt-token
  counting for future tokenizer-backed or replay-specific checks.
- M9.3c1 route tests cover bad HTTP, preflight, model route reuse, bad JSON,
  missing `messages`/`input`/`prompt`, unsupported durable state, unsupported
  tool choice, per-API context-length bodies, CORS propagation, and valid
  generation rejection through a 503 no-model JSON error.
- M9.3c1 validation passed for targeted
  `cargo test -p ds4-gguf server_no_model -- --nocapture`, full
  `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- M9.3c was split before implementation because in-memory generation-route
  parser/error dispatch and socket/process replay have distinct oracles,
  fixture families, and validation gates.
- M9.3c1 now owns the no-model dispatcher that combines M9.3a/M9.3b helpers
  with generation-route parse/error mapping for bad HTTP, bad JSON, missing
  messages/input/model/prompt, unsupported durable state/tool choice, and
  context length errors without sockets or model loading.
- M9.3c2 now owns the `ds4-server-rs` binary, CLI flag parsing, TCP bind/listen
  loop, accepted-socket dispatch, local replay comparator, and deterministic
  shutdown behavior.
- M9.3b added a no-model HTTP route dispatch surface over the M9.3a in-memory
  parser/formatter helpers.
- M9.3b covers C `client_main` behavior for `OPTIONS`, `GET /v1/models`,
  `GET /v1/models/deepseek-v4-flash`, unknown endpoint rejection, and bad HTTP
  request rejection without opening sockets or loading a model.
- M9.3b added deterministic model metadata JSON matching C
  `append_model_json_values`, including `deepseek-v4-flash`, created timestamp
  `1767225600`, owner `ds4.c`, provider context length, capped
  `max_completion_tokens`, and supported parameter ordering.
- M9.3b route tests compare exact response bytes for preflight, model list,
  single-model, bad HTTP, unknown endpoint, wrong method, CORS propagation, and
  parser-level query stripping; the model-list body is also compared with the
  captured M0.4 `models.json` fixture.
- M9.3b validation passed for targeted
  `cargo test -p ds4-gguf server_http -- --nocapture`, full
  `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- M9.3a added `rust/ds4-gguf/src/server_http.rs` with in-memory HTTP request
  parsing plus C-shaped response/error formatting and exported the helper
  surface from `ds4_gguf`.
- M9.3a covers C-style CRLF and LF-only header terminators, query stripping,
  case-insensitive `Content-Length`, exact body slicing, malformed/incomplete
  request categories, 200 JSON response bytes, 204 no-content preflight bytes,
  opt-in CORS header order, and JSON error body escaping.
- M9.3a validation passed for targeted
  `cargo test -p ds4-gguf server_http -- --nocapture`, full
  `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- M9.3 was split before implementation because byte-level HTTP
  framing/CORS, no-model route/model metadata dispatch, and socket/process
  replay have distinct C oracles, fixture families, and validation gates.
- M9.3a now owns in-memory HTTP request parsing, response/error byte
  formatting, CORS header parity, query stripping, content-length handling, and
  malformed request categories without opening sockets.
- M9.3b now owns OPTIONS and model metadata route dispatch on top of M9.3a,
  including `/v1/models`, `/v1/models/deepseek-v4-flash`, unknown endpoint
  behavior, and deterministic model JSON.
- M9.3c now owns the local `ds4-server-rs` no-model binary, socket loop,
  startup flags, no-generation negative replay, and local HTTP comparator.
- M9.2c3c added exported Rust `AnthropicLiveState` plus
  `parse_anthropic_core_request_with_live_state` so parser tests can exercise
  live-known Anthropic tool-result continuations without touching server KV or
  tool-memory side effects.
- M9.2c3c validates Anthropic `tool_result.tool_use_id` like C: missing
  tool-result-only state returns the exact Anthropic continuation error,
  live-known IDs set `anthropic_requires_live_tool_state`, and replayed prior
  assistant `tool_use` blocks avoid the live-state requirement.
- M9.2c3c collects trailing Anthropic tool-result IDs, renders the visible
  live suffix from EOS through user tool results to the next assistant prefix,
  and ignores appended system messages when locating the final tool-result
  tail.
- M9.2c3c validation covered missing live state, live-known tool-result-only
  continuation, replayed prior `tool_use` with `content` before `role`, exact
  error text, delimiter escaping, collected IDs, and live-tail prompt bytes.
- M9.2c3c validation passed for targeted
  `cargo test -p ds4-gguf server_chat -- --nocapture`, full
  `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- M9.2 request parse and prompt-render surface is now complete enough for the
  roadmap to move to M9.3 HTTP skeleton work.
- M9.2c3b extended Rust Anthropic request parsing for top-level `tools`,
  `tool_choice.type`, active tool-schema prompting, assistant `tool_use`
  blocks, user `tool_result` blocks, tool-use IDs, and DSML request-history
  rendering.
- M9.2c3b preserves raw nested `input` JSON for Anthropic `tool_use` blocks so
  DSML parameter rendering keeps object-field order and numeric spelling such
  as `2.0`, while still using stable fallback `arguments` text for non-object
  input.
- M9.2c3b validation covered direct Anthropic schemas, OpenAI-compatible
  wrapped tools, `tool_choice.type` auto/none behavior, Anthropic string
  `tool_choice` skip behavior, role-after-content `tool_use`, content-array
  `tool_result`, call-id preservation, delimiter escaping, and exact prompt
  bytes for visible tool history.
- M9.2c3b validation passed for targeted
  `cargo test -p ds4-gguf server_chat -- --nocapture`, full
  `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- M9.2c3a added exported Rust `AnthropicRequest` and
  `parse_anthropic_core_request` for model-free Anthropic core request
  parsing.
- M9.2c3a covers `messages`, `system` string/array/object parsing, private
  `x-anthropic-*` system filtering, text and thinking content blocks, scalar
  generation controls, stop sequences, stream flag, `thinking`,
  `output_config.effort`, bare `reasoning_effort`, model alias fallbacks, and
  prompt rendering without tools.
- M9.2c3a validation covered missing messages, invalid messages,
  system-private filtering and newline joining, content arrays, thinking block
  rendering, effort precedence, disabled thinking, invalid effort rejection,
  and prompt bytes.
- M9.2c3a validation passed for targeted
  `cargo test -p ds4-gguf server_chat -- --nocapture`, full
  `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- M9.2c3 was split before implementation because Anthropic core
  message/system/control parsing, tool schema/tool history parsing, and live
  tool-result continuation validation have distinct C branches, fixture
  families, and validation comparators.
- M9.2c3a now owns Anthropic `messages`, `system`, text content, private system
  filtering, scalar controls, stop sequences, stream flag, thinking/effort
  controls, model alias fallbacks, and non-tool prompt rendering.
- M9.2c3b now owns Anthropic tool schemas, `tool_choice.type`, assistant
  `tool_use`, user `tool_result`, tool-use IDs, tool-result prompt rendering,
  and DSML request-history rendering.
- M9.2c3c now owns Anthropic missing/live `tool_use_id` validation, live-state
  requirement flags, live tool-use ID collection, and live suffix rendering.
- M9.2c2c added a Rust `ResponsesLiveState` stub plus request metadata for
  `responses_requires_live_tool_state`, `responses_requires_live_reasoning`,
  `responses_live_call_ids`, and `responses_live_suffix_text`.
- M9.2c2c added model-free validation matching C
  `responses_validate_tool_outputs`: tool-output-only requests without live or
  replayed prior call IDs now return the same stable error string, live-known
  tool-output-only requests set `requires_live_tool_state`, and stateless
  thinking-mode replays without prior reasoning set `requires_live_reasoning`.
- M9.2c2c added a Rust live-tool-tail renderer matching C
  `render_live_tool_tail`, including the leading EOS, tool-result delimiter
  escaping, and next assistant prefix for thinking and non-thinking modes.
- M9.2c2c validation covered missing live state, live-known tool-output-only
  continuations, replayed prior tool calls with and without reasoning,
  non-thinking replay, call-id collection, and suffix text excluding prior tool
  call DSML.
- M9.2c2c validation passed for targeted
  `cargo test -p ds4-gguf server_chat -- --nocapture`, full
  `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- M9.2c2b extended `ToolSchemaOrder` with namespace, wire-name, and hosted
  `tool_search` metadata needed by Responses schema parsing.
- M9.2c2b extended Rust `parse_tools_value` to match C handling for top-level
  hosted `tool_search`, normal functions named `tool_search`, namespace tool
  flattening, property order capture, and dynamic `tool_search_output.tools`
  schema loading.
- M9.2c2b combines top-level tool schemas before dynamically loaded schemas in
  prompt text while preserving parser field-order replacement semantics in
  `tool_orders`.
- M9.2c2b validation covered hosted `tool_search` distinction, namespace
  prompt-name flattening with namespace/wire metadata, dynamic schema loading
  after top-level schemas, and malformed dynamic tool-list rejection.
- M9.2c2b validation passed for targeted
  `cargo test -p ds4-gguf server_chat -- --nocapture`, full
  `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- M9.2c2a extended the Rust prompt data model with stable tool-call IDs on
  assistant calls and tool result messages without changing existing prompt
  rendering for non-tool histories.
- M9.2c2a extended Responses `input` parsing to preserve raw item fields where
  C preserves raw JSON, parse `function_call`, `custom_tool_call`,
  `local_shell_call`, `web_search_call`, `tool_search_call`,
  `image_generation_call`, function/custom outputs, and hosted tool outputs
  into `ChatMessage`/`ToolCall` history.
- M9.2c2a validation covered assistant text plus split tool-call merging,
  reasoning attachment, DSML argument ordering and numeric spelling,
  custom-tool free-text fallback, hosted tool names/actions, output/result/tool
  body selection, tool-result delimiter escaping, and call-id preservation.
- M9.2c2a validation passed for targeted
  `cargo test -p ds4-gguf server_chat -- --nocapture`, full
  `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- M9.2c2 was split before implementation because Responses tool-call input
  parsing, dynamic tool schema loading, and live continuation validation have
  distinct C branches, fixture families, and validation comparators.
- M9.2c2a now owns function/custom/hosted call inputs, tool output inputs, call
  IDs, pending-reasoning merge rules, and DSML prompt rendering.
- M9.2c2b now owns hosted `tool_search`, namespace schema flattening,
  tool-search-output dynamic schema loading, combined schema ordering, and
  namespace/wire-name metadata.
- M9.2c2c now owns missing/live call-id validation, live-state requirement
  flags, reasoning replay requirement flags, call-id collection, and live-tail
  suffix text.
- M9.2c1 added the exported Rust `ResponsesRequest` and
  `parse_responses_core_request` surface for model-free Responses API core
  request parsing.
- M9.2c1 covers bare string and array `input`, `instructions` system prepend,
  scalar generation controls, `reasoning.effort`, reasoning summary opt-in,
  model-alias thinking fallbacks, prompt rendering, top-level tool prompt
  participation, durable-state rejection for `previous_response_id` and
  `conversation`, unsupported `tool_choice` categories, and strict text-content
  shape checks.
- M9.2c1 validation passed for targeted
  `cargo test -p ds4-gguf server_chat -- --nocapture`, full
  `cargo test --workspace`, `cargo fmt --all -- --check`, and
  `git diff --check`.
- M9.2c was split before implementation because Responses core input/reasoning,
  Responses tool-output/live-tail validation, and Anthropic content/tool-result
  parsing have distinct oracle surfaces, fixtures, validation categories, and
  prompt/live-tail comparators.
- M9.2c split review initially found boundary gaps; the split now explicitly
  defers KV/tool-memory replay side effects to M9.8, makes M9.2c2 depend on
  M9.2c1 for combined Responses tool schemas, names `parse_anthropic_system`
  and `parse_anthropic_system_object`, and includes Anthropic bare
  `reasoning_effort` coverage.
- M9.2c1 now owns Responses core `input`, `instructions`, scalar generation
  controls, reasoning effort/summary flags, durable-state rejection, and prompt
  rendering without live tool-state validation.
- M9.2c2 now owns Responses function/tool output inputs, tool-search schema
  loading, namespace tool schema restoration, and live-tail validation.
- M9.2c3 now owns Anthropic content/system blocks, tools/tool_choice,
  stop/thinking controls, tool-use/tool-result messages, and live-tail
  validation.
- M9.2b extended `rust/ds4-gguf/src/server_chat.rs` to parse OpenAI `tools`,
  `tool_choice`, assistant `tool_calls`, tool schema property order, tool role
  prompt rendering, and DSML request-history rendering without loading a model
  or opening sockets.
- M9.2b exact prompt-byte coverage uses the raw request body captured in the
  M0.4 `chat_tool_call` trace because C preserves raw tool-schema whitespace in
  the prompt; the pretty committed JSON fixture is still used for semantic
  parser checks.
- M9.2b validation covered schema order, `tool_choice: "none"` suppression,
  DSML argument ordering, raw JSON argument minification, numeric spelling
  preservation, tool-result delimiter escaping, and reasoning preservation when
  tools are active.
- M9.2a added a dependency-free Rust OpenAI chat request parser in
  `rust/ds4-gguf/src/server_chat.rs`, exported through `ds4-gguf`, for
  model-free `/v1/chat/completions` core fields.
- M9.2a parser coverage includes the M0.4 non-tool OpenAI fixtures
  `chat_basic`, `chat_stream`, `chat_thinking_disabled`, `chat_cache_seed`,
  and `chat_cache_continuation`, matching rendered prompt bytes, stream flags,
  sampling defaults/options, seeds, max-token fields, thinking mode, stop lists,
  error categories, and OpenAI context-length error body shape.
- M9.2a Claude review returned PASS on the material parser parity checks after
  reviewing defaults, duplicate-field order, stream options, thinking/reasoning
  mapping, stop lists, context-length helper shape, prompt bytes, and fixture
  coverage.
- M9.2a intentionally excludes tool schema/tool-call payload handling and
  alternate protocols; M9.2b owns OpenAI tool schema plus DSML prompt rendering,
  while M9.2c owns Responses and Anthropic request inputs.
- `git status --short` was clean before M0.1 edits.
- `AGENT.md`, `CONTRIBUTING.md`, and `RUST_PORT_ROADMAP.md` were read before
  creating the protocol.
- M0.1 validation passed with `git diff --name-only` and `git diff --check`.
- M0.1 Claude review returned PASS before commit.
- M0.2 local arm64 validation captured `arch -arm64 make` exit 0,
  `arch -arm64 make cpu` exit 0, `./ds4_test --server` exit 0, and
  `./ds4_test --metal-kernels` exit 0.
- M0.2 local default `make` and model-backed `make test` failures are recorded
  in `ds4-parity/baselines/manifest.md` with exact logs.
- M0.2 B300 validation captured `make cuda-generic` exit 0,
  `make cuda-regression` exit 0, `./ds4_test --server` exit 0, and
  `./ds4_test --metal-kernels` exit 0 on `ds4-rust-port-b300`.
- M0.3 B300 validation downloaded q2-imatrix, recorded model hash/size, built
  `ds4_test`, and captured `./ds4_test --logprob-vectors` exit 0 with
  `logprob-vectors: OK`.
- M0.4 B300 validation refreshed source commit
  `3d87577962abeac1ab0d80e9c21d0012bfc53afb`, built `ds4-server`, and replayed
  six server fixtures from `ds4-parity/baselines/server-fixtures/m0.4/` with
  HTTP 200 for all requests.
- M0.4 artifacts live under `ds4-parity/baselines/server-traces/m0.4/`; the
  final trace records non-streaming chat, SSE chat, DSML-to-OpenAI tool calls,
  explicit thinking-disabled chat, and cache continuation with
  `cache_source=memory-token`, `cached_tokens=41`, `cache_write_tokens=9`.
- M0.5 B300 validation refreshed source commit
  `0623bbb4d97d056a58e208e324216f97abed685e`, built `ds4-server`, and replayed
  three disk-KV server lifetimes from
  `ds4-parity/baselines/kv-fixtures/m0.5/` with HTTP 200 for all requests.
- M0.5 artifacts live under `ds4-parity/baselines/kv-artifacts/m0.5/`; the
  replay records a cold 550-token cache write, a fresh-process 550-token
  `disk-text` restore, and a fresh-process continuation restore of the
  552-token shutdown prefix with a 9-token suffix write.
- M0.5 raw `.kv` files are not checked in; committed comparator artifacts
  include full raw hashes, timestamp-normalized hashes, parsed KVC headers, and
  extracted rendered cache text.
- M0.6 B300 validation refreshed source commit
  `add2c507f81aa2e363809213771134c282c50bf2`, built `ds4-bench`, and captured
  short-context and long-context CSV baselines using
  `speed-bench/promessi_sposi.txt` with SHA256
  `f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f`.
- M0.6 artifacts live under `ds4-parity/baselines/bench/m0.6/`; the short CSV
  covers 2048 through 8192 tokens and the long CSV covers 16384 through 32768
  tokens, both with 32 greedy generation tokens per frontier.
- M1.1 documented the Milestone 1 implementation work items in
  `RUST_PORT_ROADMAP.md`; the next executable item is a static verifier for the
  committed Milestone 0 artifacts.
- M1.2 added `python3 ds4-parity/verify_baselines.py`, which verifies M0.2
  through M0.6 artifact families locally without rerunning model-backed
  commands. Its negative test corrupts a copied benchmark CSV and requires the
  verifier to detect the drift.
- M1.3 added `python3 ds4-parity/compare_server_kv.py`, which self-compares
  committed M0.4 server and M0.5 KV artifacts with only documented
  normalizations. Its negative test covers finish reason, cached token count,
  cache source, KV reason, and rendered text drift.
- M1.4 added `python3 ds4-parity/compare_logprob_numeric.py`, which parses the
  compact official-vector fixture, audits it against raw official API JSON,
  verifies the M0.3 B300 pass markers, and compares candidate vector files with
  exact selected tokens plus a reported 4.0 absolute logprob tolerance. Its
  negative test covers selected-token drift and numeric drift outside tolerance.
- M1.5 added `python3 ds4-parity/compare_bench_csv.py`, which self-compares
  committed M0.6 benchmark CSV artifacts, validates capture metadata for
  threshold use, requires exact workload shape and KV byte counts, and applies
  the documented 10% throughput regression threshold. Its negative test covers
  schema, context frontier, generation-token, cache-byte, and throughput drift.
- M1.6 added `python3 ds4-parity/run_parity_report.py`, which runs local
  no-model C oracles, invokes the M1.2 through M1.5 comparator commands, and
  reports skipped B300/model-backed oracle refreshes with explicit
  `--kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1` rerun
  commands. The local report passed with 9 executed items and 4 B300 refreshes
  skipped by design.
- M2.1 added a Rust workspace with `ds4-gpu-sys` and `ds4-gpu`, seeded core
  tensor/command/model-map FFI declarations, added smoke-only safe status
  wrappers, and wired `make rust-test`. Validation passed for `cargo fmt`,
  `cargo test --workspace`, `make rust-test`, sequential `arch -arm64 make`,
  sequential `arch -arm64 make cpu`, and the unified parity report.
- M3.1 added safe Rust `Tensor`, `TensorView`, and `CommandBatch` wrappers over
  the existing `ds4_gpu.h` tensor/command ABI without changing the C ABI. The
  macOS `ds4-gpu` build script compiles the current `ds4.c` and `ds4_metal.m`
  backend objects into a test archive so Rust tests call the real Metal
  implementation rather than a mock.
- M3.1 Rust ABI parity validation passed with
  `cargo test -p ds4-gpu safe_tensor_wrapper_matches_direct_c_abi -- --nocapture`;
  the test compares safe-wrapper and direct-C paths for allocation, byte-size
  queries, write/read, fill, view writes, command-batched copy, flush/end,
  synchronize, and out-of-bounds write/copy failures.
- M3.1 full validation passed for `cargo fmt --all -- --check`,
  `cargo test --workspace`, `make rust-test`, `git diff --check`, sequential
  `arch -arm64 make clean`, sequential `arch -arm64 make`, sequential
  `arch -arm64 make clean`, sequential `arch -arm64 make cpu`, and
  `python3 ds4-parity/run_parity_report.py` with 9 passed, 4 skipped, 0 failed.
- M4.1 split Milestone 4 into concrete GGUF/model-metadata work items after
  reading `ds4.c` loader, summary, metadata validation, base tensor binding,
  and MTP tensor binding surfaces. The next executable item is the current-C
  metadata dump oracle.
- M4.2 added `./ds4-metadata-dump`, which opens the model through the current C
  GGUF loader, runs `config_validate_model` and `weights_bind`, and emits
  deterministic `ds4.metadata.v1` JSON with selected metadata values, tensor
  type histograms, all tensor descriptors, and bound semantic tensor roles.
- M4.2 added `python3 ds4-parity/check_metadata_dump.py`, whose schema checker
  validates the dump and whose negative test detects tensor-count drift, a
  missing required bound role, and a missing required metadata key.
- M4.2 B300 validation copied the M4.2 source files into
  `/workspace/ds4`, built with `make clean ds4-metadata-dump CUDA_ARCH=native`,
  dumped `/workspace/ds4/ds4flash.gguf`, and passed
  `python3 ds4-parity/check_metadata_dump.py /tmp/ds4-metadata.json --negative-test`.
  The generated B300 dump had 633,297 bytes, SHA256
  `39ad79574b19421e2c470a055376258b9415eb1f429188426cfd2860688a2a2f`,
  1,328 tensors, and 1,511 bound tensor roles.
- M4.2 local validation passed for `arch -arm64 make ds4-metadata-dump`,
  `./ds4-metadata-dump --help`, local schema/negative checks against the copied
  B300 dump, `python3 -m py_compile ds4-parity/check_metadata_dump.py`,
  sequential `arch -arm64 make clean`, sequential `arch -arm64 make`,
  sequential `arch -arm64 make clean`, sequential `arch -arm64 make cpu`,
  `python3 ds4-parity/run_parity_report.py` with 9 passed, 4 skipped, 0 failed,
  `cargo test --workspace`, and `git diff --check`.
- M4.3 added a dependency-free `ds4-gguf` Rust crate and `ds4-gguf-dump` CLI
  that parse GGUF v3 metadata and tensor directory records, compute C-equivalent
  tensor byte sizes and aligned absolute offsets, and emit the same
  `ds4.metadata.v1` directory surface as the C metadata dump.
- M4.3 added `./ds4-metadata-dump --directory-only` so local synthetic GGUF
  fixtures can compare the current C GGUF directory parser against Rust without
  requiring the full DS4 model or semantic tensor binding.
- M4.3 added `python3 ds4-parity/compare_gguf_directory.py`, whose synthetic
  fixture covers scalar metadata, array metadata, non-power-of-two
  `general.alignment=48`, F32 byte sizing, Q8_0 block byte sizing, relative and
  absolute offsets, C-compatible float metadata formatting, unsupported scalar
  metadata emission as `null`, and C/Rust rejection of corrupted magic,
  truncated metadata, truncated tensor data, and tensor offset overflow.
- M4.3 B300 check confirmed the pod does not currently have `rustc` or `cargo`,
  so this item used local synthetic C-vs-Rust directory comparison instead of a
  B300 Rust run. Real supported-model Rust comparison remains deferred to the
  roadmap item that provides Rust on the model host or transfers feasible dump
  artifacts.
- M4.3 local validation passed for `arch -arm64 make ds4-metadata-dump`,
  `cargo test -p ds4-gguf` with 4 tests,
  `python3 ds4-parity/compare_gguf_directory.py --negative-test` with 14
  checks, `cargo fmt --all -- --check`, `python3 -m py_compile
  ds4-parity/compare_gguf_directory.py ds4-parity/check_metadata_dump.py`,
  local schema/negative checks against the copied B300 M4.2 dump,
  `cargo test --workspace`, `git diff --check`, sequential
  `arch -arm64 make clean`, sequential `arch -arm64 make`, sequential
  `arch -arm64 make clean`, sequential `arch -arm64 make cpu`, and
  `python3 ds4-parity/run_parity_report.py` with 9 passed, 4 skipped, 0 failed.
- M4.4 added `./ds4-metadata-dump --validate-config-only`, which runs current C
  `config_validate_model` after GGUF parsing but skips tensor binding, making
  local metadata-only validation fixtures possible without the full model
  tensor table.
- M4.4 added `validate_ds4_metadata` in `ds4-gguf`, matching C required-key
  behavior, `u64` and `f32` coercion rules, optional expert group defaults,
  fixed DeepSeek4 constants, compression-ratio arrays, SwiGLU clamp arrays,
  RoPE constants, HC constants, RMS epsilon, expert weight scale, and expert
  weight normalization.
- M4.4 added `python3 ds4-parity/compare_metadata_validation.py`, whose
  synthetic fixtures compare C and Rust pass/fail behavior and normalized first
  failures for baseline metadata, C-compatible numeric coercions, missing keys,
  wrong scalar types, wrong scalar values, short arrays, negative compression
  ratios, wrong compression ratios, float tolerance failures, non-integer
  `u64` inputs, non-float `f32` inputs, and boolean drift.
- M4.4 focused validation passed for `arch -arm64 make ds4-metadata-dump`,
  `cargo test -p ds4-gguf` with 4 tests, `python3
  ds4-parity/compare_metadata_validation.py --negative-test` with 41 checks,
  and `python3 -m py_compile ds4-parity/compare_metadata_validation.py`.
- M4.5 added `./ds4-metadata-dump --validate-layout-only`, which runs current C
  metadata validation plus base/MTP tensor binding and layout validation from
  GGUF directories while skipping tensor payload range checks for synthetic
  local fixtures.
- M4.5 added Rust base and MTP tensor binding/layout validation in `ds4-gguf`,
  including required, optional, compression-ratio-dependent, hash-layer-only,
  plain F16/F32 MTP, routed expert quant-type, routed gate/up type equality,
  and fixed tensor dimension rules.
- M4.5 added `python3 ds4-parity/compare_tensor_bindings.py`, whose synthetic
  fixtures compare C and Rust layout dumps for base plus MTP bindings and
  negative cases for missing required tensors, wrong types, wrong dimensions,
  optional tensor type drift, routed expert type drift, routed gate/up type
  mismatch, missing compressor/indexer tensors, MTP plain-type rejection, and
  missing required MTP tensors.
- M4.5 focused validation passed for `arch -arm64 make ds4-metadata-dump`,
  `cargo test -p ds4-gguf` with 4 tests, `python3
  ds4-parity/compare_tensor_bindings.py --negative-test` with 33 checks, and
  `python3 -m py_compile ds4-parity/compare_tensor_bindings.py`.
- M4.6 recaptured the supported-model metadata baseline on B300
  `ds4-rust-port-b300` in `hou2-prod1` after refreshing `ds4.c`, `ds4.h`, and
  `ds4_metadata_dump.c` from source commit
  `58bad019226499d5b294340093f77c70b7250b79`.
- M4.6 committed `ds4-parity/baselines/metadata/m4.6/current-c.json` for
  `/workspace/ds4/ds4flash.gguf`, whose resolved path is
  `/workspace/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`,
  model size is 86,720,111,488 bytes, model SHA256 is
  `efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668`,
  dump size is 633,297 bytes, and dump SHA256 is
  `39ad79574b19421e2c470a055376258b9415eb1f429188426cfd2860688a2a2f`.
- M4.6 added `python3 ds4-parity/compare_metadata_baseline.py --negative-test`,
  which schema-checks the committed baseline, verifies manifest artifact hashes,
  normalizes model paths/source for candidate comparisons, and detects scalar
  metadata, array metadata, tensor shape, tensor type, binding, and offset
  drift.
- M4.6 wired the metadata baseline comparator into
  `python3 ds4-parity/run_parity_report.py` and added a B300 skip entry with
  exact source-refresh, capture, hash, schema-check, and copy-back commands.
- M4.7 added `python3 ds4-parity/compare_gguf_failures.py`, a generated
  malformed-GGUF matrix that compares C and Rust rejection status plus
  normalized first-error categories for invalid magic, unsupported version,
  truncated metadata, unknown metadata type, bad tensor dimension, out-of-file
  tensor data, tensor offset overflow, missing required metadata, wrong
  metadata type, bad metadata array length, and unsupported DS4 tensor type.
- M4.7 validation passed for `arch -arm64 make ds4-metadata-dump`,
  `python3 ds4-parity/compare_gguf_failures.py` with 55 checks,
  `python3 ds4-parity/compare_gguf_failures.py --list-cases`, M4.3 through
  M4.5 comparators (`compare_gguf_directory.py --negative-test`,
  `compare_metadata_validation.py --negative-test`, and
  `compare_tensor_bindings.py --negative-test`), `python3 -m py_compile` for
  all involved comparators, and `cargo test --workspace`.
- M5.1 split Milestone 5 into M5.2 through M5.7 after reading tokenizer source
  (`vocab_load`, JoyAI `bpe_tokenize_text`, rendered-chat special tokenization,
  `ds4_token_text`, and `ds4_dump_text_tokenization`), CLI prompt paths
  (`--dump-tokens`, `build_prompt`, and REPL append functions), server prompt
  and API paths (`parse_chat_request`, `render_chat_prompt_text`,
  `render_live_tool_tail`, and DSML formatting/parsing helpers), the agent DSML
  streaming parser, and existing M0.3/M0.4/M0.5 fixtures.
- M5.1 validation passed for `git diff --check` and non-interactive Claude
  review after tightening tokenizer identity, server-vs-CLI prompt oracles,
  token decoding ownership, DSML chunk/EOF parser schedules, tool-schema
  fixture variants, and request body hashing; final Claude review returned
  `NO BLOCKERS`.
- M5.2 added current-C tokenizer and prompt oracle dumping through
  `./ds4-server --dump-token-oracle`, with tokenizer identity hashing in
  `ds4_engine_dump_tokenizer_identity_json`, shared `ds4_sha256_hex`, raw
  request-body hashing, server prompt/token fixtures, and CLI `ds4_chat_*`
  operation/token-stream fixtures. The dump mode opens the model through the
  existing engine path but exits before session/listener/worker startup, and
  advisory token text emission preserves valid UTF-8 while escaping invalid
  raw bytes so future byte-fallback fixtures still produce valid JSON.
- M5.2 committed
  `ds4-parity/baselines/tokenization/m5.2/current-c.json` captured on B300
  `ds4-rust-port-b300` from `/workspace/ds4/ds4flash.gguf`; dump size is
  124,497 bytes and dump SHA256 is
  `b0689f47abe63750ab3191772d5661d5f0f433e954bcfd0de6a0e55a747489e9`.
  The tokenizer identity records 129,280 tokens, token-bytes SHA256
  `c92251fc634ff01cc6767d2e3ce1d368e72b5f02b647983d4410eb0c46693fa3`,
  127,741 merge records, merge-pairs SHA256
  `8100a9693dc10b8aad79abbe20b172545ff5e1e6051e0705cc91e73b88e3751f`,
  the seven rendered-control specials, and 863 literal-special tokens.
- M5.2 B300 validation passed after copying the changed source/checker into
  `/workspace/ds4`, building with `make clean ds4-server CUDA_ARCH=native`,
  dumping the oracle from the q2-imatrix model, and running
  `python3 ds4-parity/check_tokenization_dump.py
  /tmp/ds4-tokenization-m5.2-current-c.json --negative-test`; the final B300
  checker reported `tokenization schema: PASS, 13409 checks` and
  `tokenization negative tests: PASS, 12 checks`.
- M5.2 local validation passed for
  `python3 ds4-parity/check_tokenization_dump.py
  ds4-parity/baselines/tokenization/m5.2/current-c.json --manifest
  ds4-parity/baselines/tokenization/m5.2/manifest.json --negative-test`,
  with `tokenization schema: PASS, 13409 checks`, `tokenization manifest:
  PASS, 11 checks`, and `tokenization negative tests: PASS, 12 checks`;
  `python3 -m py_compile ds4-parity/check_tokenization_dump.py`,
  `./ds4_test --server`, `git diff --check`, `arch -arm64 make ds4-server`,
  `cargo test --workspace`, and `arch -arm64 make cpu` also passed.
- M5.2 Claude review returned `NO BLOCKERS`; after hardening invalid UTF-8
  token text escaping and checker pinning for exact special/server semantics,
  the follow-up Claude review also returned `NO BLOCKERS`.
- M5.3 added `Ds4Tokenizer` to `ds4-gguf`, loading
  `tokenizer.ggml.tokens` and `tokenizer.ggml.merges` from GGUF metadata,
  computing the same canonical token/merge SHA256 identity as C, validating
  required DS4 special token IDs, porting JoyAI plain-text pre-tokenization and
  byte-level BPE merge ranking, and decoding ordinary token pieces through the
  GPT-2 byte mapping used by `ds4_token_text`.
- M5.3 added `ds4-tokenizer-dump` for fixed plain-text cases and
  `python3 ds4-parity/compare_tokenizer_text.py`, which compares Rust token
  IDs and decoded token-piece bytes against the M5.2 current-C `text_cases`.
  Its negative tests cover missing token table, missing merges, token-bytes
  hash drift, merge hash drift, missing required special token, invalid UTF-8
  token strings, and merge-rank drift.
- M5.3 B300 extraction copied `ds4-parity/extract_tokenizer_fixture.py` to
  `ds4-rust-port-b300` and wrote
  `/tmp/ds4-tokenization-m5.3/tokenizer.gguf` from
  `/workspace/ds4/ds4flash.gguf`. The committed tokenizer-only GGUF fixture has
  129,280 tokens, 127,741 merges, size 4,722,720 bytes, and SHA256
  `b1e0d128bde9ea996fee335c9662e93707d2a68decaeb47a8dc5fb902bdbb025`.
- M5.3 local validation passed for `cargo fmt --all -- --check`,
  `cargo test -p ds4-gguf` with 8 tests,
  `python3 -m py_compile ds4-parity/extract_tokenizer_fixture.py
  ds4-parity/compare_tokenizer_text.py`, and
  `python3 ds4-parity/compare_tokenizer_text.py --manifest
  ds4-parity/baselines/tokenization/m5.3/manifest.json --negative-test` with
  `tokenizer text comparison: PASS, 51 checks`, `tokenizer manifest: PASS, 4
  checks`, and `tokenizer negative tests: PASS, 7 checks`.
- M5.3 Claude review returned `NO BLOCKERS` after checking the Rust tokenizer
  against the C byte encoding, JoyAI split rules, BPE merge loop, token text
  decoding, and comparator scope.
- M5.4 added Rust rendered-chat tokenization over the exact C
  `special_token_at` rendered-control table. `tokenize_rendered_chat` scans
  trusted rendered prompt bytes for BOS, EOS, User, Assistant, `<think>`,
  `</think>`, and `｜DSML｜`, emits those special token IDs, and tokenizes
  intervening spans through the existing JoyAI BPE path; plain `tokenize_text`
  remains separate so special-looking user text is not trusted as control text.
- M5.4 extended `ds4-tokenizer-dump` and
  `python3 ds4-parity/compare_tokenizer_text.py` to compare the M5.2
  `rendered_chat_cases` exactly for rendered prompt bytes, token IDs, and
  decoded token-piece bytes. Negative checks now include rendered special-token
  ID drift and rendered ordinary-piece drift.
- M5.4 local validation passed for `cargo fmt --all -- --check`,
  `cargo test --workspace` with 9 `ds4-gguf` tests,
  `python3 -m py_compile ds4-parity/compare_tokenizer_text.py`, and
  `python3 ds4-parity/compare_tokenizer_text.py --manifest
  ds4-parity/baselines/tokenization/m5.3/manifest.json --negative-test` with
  `tokenizer text comparison: PASS, 71 checks`, `tokenizer manifest: PASS, 4
  checks`, and `tokenizer negative tests: PASS, 9 checks`.
- M5.4 Claude review returned `NO BLOCKERS`; after adding Rust dump `mode`
  fields and pinning them in the comparator, the follow-up Claude review also
  returned `NO BLOCKERS`.
- M5.5 added a Rust prompt renderer matching C `render_chat_prompt_text` for
  the committed M5.2 server prompt cases: tool schemas before system text,
  system/developer aggregation, user/tool/function message handling, assistant
  history turns, thinking disabled/high/max prefixes, DSML tool-call rendering,
  escaped tool-result closing tags, and pending assistant prefixes.
- M5.5 added direct Rust CLI token construction for the M5.2 `ds4_chat_*`
  operation fixtures, covering begin, Think Max prefix append, system/developer
  direct text, user/tool/function messages, assistant content, and assistant
  prefixes for high/max/none thinking modes.
- M5.5 extended `ds4-tokenizer-dump` and
  `python3 ds4-parity/compare_tokenizer_text.py` to compare every M5.2
  `server_request_cases` prompt byte string, rendered token IDs, decoded token
  pieces, CLI operation sequence, and CLI token stream. Negative checks now
  include server prompt-byte drift, server token-ID drift, CLI operation drift,
  and CLI token-piece drift.
- M5.5 local validation passed for `cargo fmt --all -- --check`,
  `cargo test -p ds4-gguf` with 11 tests, `cargo test --workspace`,
  `./ds4_test --server`, `python3 -m py_compile
  ds4-parity/compare_tokenizer_text.py`, `git diff --check`, and
  `python3 ds4-parity/compare_tokenizer_text.py --manifest
  ds4-parity/baselines/tokenization/m5.3/manifest.json --negative-test` with
  `tokenizer text comparison: PASS, 154 checks`, `tokenizer manifest: PASS, 4
  checks`, and `tokenizer negative tests: PASS, 13 checks`.
- M5.5 Claude review returned `NO BLOCKERS` after checking Rust prompt
  rendering against C role handling, thinking branches, DSML/tool-result
  escaping, CLI token construction, and comparator coverage.
- M5.6 was split into M5.6a and M5.6b before implementation because server
  generated-message DSML parsing and agent incremental DSML streaming have
  different oracle surfaces and comparator shapes. M5.6a owns server DSML
  formatting plus `parse_generated_message_ex`; M5.6b owns `agent_dsml_parse`
  chunk schedules and streaming state/event parity.
- M5.6 split validation passed for docs-only `git diff --name-only`, `git
  diff --check`, and non-interactive Claude review. Claude returned
  `NO BLOCKERS`.
- M5.6a added `./ds4-server --dump-dsml-oracle`, a no-model current-C DSML
  oracle covering rendered tool-call blocks, raw sampled DSML replay, JSON and
  string parameters, sentinel escaping, tool-result escaping,
  `parse_generated_message_ex`, and recoverable response parsing. The committed
  baseline lives at `ds4-parity/baselines/dsml/m5.6a/current-c.json` with size
  17,016 bytes and SHA256
  `3f20b4869a2035deab709e3299de91ccf151f46fa3524a8b389814ebbf880442`.
- M5.6a added Rust DSML formatting/parsing in `ds4_gguf::dsml`, routed the Rust
  prompt renderer's DSML and tool-result escaping through that module, and added
  `ds4-dsml-dump` plus `python3 ds4-parity/compare_dsml.py`.
- M5.6a validation passed for `arch -arm64 make ds4-server`,
  `./ds4-server --dump-dsml-oracle /tmp/ds4-dsml-final-c.json` compared
  byte-for-byte with the committed baseline,
  `python3 ds4-parity/compare_dsml.py --manifest
  ds4-parity/baselines/dsml/m5.6a/manifest.json --negative-test` with
  `DSML comparison: PASS, 410 checks`, `python3 -m py_compile
  ds4-parity/compare_dsml.py`, `cargo fmt --all -- --check`,
  `cargo test -p ds4-gguf` with 14 tests, `cargo test --workspace`,
  `./ds4_test --server`, `python3 ds4-parity/compare_tokenizer_text.py
  --manifest ds4-parity/baselines/tokenization/m5.3/manifest.json
  --negative-test`, and `git diff --check`.
- M5.6a Claude review returned `NO BLOCKERS` after checking the Rust DSML
  parser/formatter against C tool-start ordering, raw block boundaries,
  sentinel escaping, entity escaping, JSON minification, response recovery,
  raw DSML replay, prompt-renderer routing, and comparator coverage.
- M5.6a implementation commit:
  `aaab1818710384e1c0b754c94f63dbf408ddb724`.
- M5.6b added `./ds4-agent --dump-agent-dsml-oracle`, a no-model current-C
  oracle for the agent incremental DSML parser. The fixture records whole,
  one-byte, marker-prefix, and parameter-boundary schedules where applicable,
  including raw/search buffer hex, parser states, current call, completed calls,
  parameter state, and error text after each chunk.
- M5.6b added Rust `ds4_gguf::agent_dsml`, `ds4-agent-dsml-dump`, and
  `python3 ds4-parity/compare_agent_dsml.py`. The committed C baseline lives at
  `ds4-parity/baselines/dsml/m5.6b/current-c.json` with size 887,559 bytes and
  SHA256
  `0b0f21728b0f5230dcbae5d3d2a99e272347ecdeac04fa57ca07ec00b9f00618`.
- M5.6b validation passed for `arch -arm64 make ds4-agent`,
  `./ds4-agent --dump-agent-dsml-oracle /tmp/agent-dsml-final-c.json` compared
  byte-for-byte with the committed baseline,
  `python3 ds4-parity/compare_agent_dsml.py --manifest
  ds4-parity/baselines/dsml/m5.6b/manifest.json --negative-test` with
  `agent DSML comparison: PASS, 37873 checks`, `python3 -m py_compile
  ds4-parity/compare_agent_dsml.py ds4-parity/compare_dsml.py`,
  `cargo fmt --all -- --check`, `cargo test -p ds4-gguf` with 16 tests,
  `cargo test --workspace`, `python3 ds4-parity/compare_dsml.py --manifest
  ds4-parity/baselines/dsml/m5.6a/manifest.json --negative-test`,
  `./ds4_test --server`, and `git diff --check`.
- M5.6b Claude review returned `NO BLOCKERS` after checking byte-vs-UTF-8
  behavior, mid-chunk done/error accumulation, close-tag variants, search-tail
  behavior, raw buffer accumulation, current/completed call transitions,
  fixture coverage, and no-model oracle startup.
- M5.6b implementation commit:
  `d6bade1d5bde4c72280bed0395322d85dfc30d5e`.
- M5.7 added `python3 ds4-parity/run_text_parity_report.py`, which runs the
  M5.2 token/prompt schema checker, M5.3-M5.5 Rust tokenizer/prompt
  comparator, M5.6a server DSML comparator, and M5.6b agent DSML comparator
  from committed fixtures without requiring the model locally.
- M5.7 records model-backed refreshes as skipped report items using exact
  `refresh_commands` from the M5.2 and M5.3 manifests, preserving the
  `--kubeconfig /tmp/ds4-hou2-prod1.kubeconfig --context hou2-prod1` B300
  command path for future recapture.
- M5.7 wired the text report into
  `python3 ds4-parity/run_parity_report.py`, so the unified parity report now
  includes Milestone 5 text parity alongside earlier baseline comparators.
- M5.7 validation passed for `python3 -m py_compile
  ds4-parity/run_text_parity_report.py ds4-parity/run_parity_report.py`,
  `python3 ds4-parity/run_text_parity_report.py` with `summary: 4 passed, 2
  skipped, 0 failed`, JSON mode output with `ok: true`,
  `python3 ds4-parity/run_parity_report.py --skip-local-oracles` with
  `summary: 6 passed, 10 skipped, 0 failed`, `cargo test --workspace`, and
  `git diff --check`.
- M5.7 Claude review returned `NO BLOCKERS` after checking report
  integration, failure output, B300 refresh command fidelity, JSON/text output
  shape, status/TODO consistency, and accidental local model dependencies.
- M5.7 implementation commit:
  `3223f6e3a09f066873c5b8afc1b855adabad068d`.
- M6.1 split Milestone 6 into M6.2 through M6.7, including M6.6a/M6.6b,
  after reading the public sampling/logprob APIs in `ds4.h`, current C sampler
  and logprob math
  (`sample_argmax`, `sample_rng_next`, `sample_top_p_min_p`,
  `ds4_session_top_logprobs`, and `ds4_session_token_logprob`), CLI
  `--dump-logprobs` and perplexity surfaces, server decode stop handling,
  agent sampling, M0.3 official-vector tests, and M1.4 numeric comparator
  conventions.
- M6.1 defined separate oracle surfaces for no-model fixed-logits sampler math,
  Rust sampler/logprob math, B300 current-C session logits capture, Rust
  fixed-logits model-slice comparison, C decode stop policy fixtures, Rust
  decode stop policy comparison, and report integration.
- M6.1 validation passed for `git diff --check`; Claude review returned
  `NO BLOCKERS` after tightening M6.2 fixture ownership for source-named
  request-surface sampling tuples and splitting decode stop policy into M6.6a
  C oracle fixtures plus M6.6b Rust policy comparison.
- M6.1 implementation commit:
  `4d401ecf2a2f13e214927ab8ec05dc931d5e796e`.
- M6.2 added `./ds4-sampling-dump`, a no-model current-C fixed-logits sampler
  and logprob oracle that records selected token, actual sampler selection,
  consumed RNG state, effective sampling parameters, filtered candidate sets,
  input logits, top-logprob slices, token-logprob requests, and source-named
  request-surface sampling tuples.
- M6.2 committed
  `ds4-parity/baselines/sampling/m6.2/current-c.json` with size 16,556 bytes
  and SHA256
  `f3740560d562960ed3960f7aa07f50793b7b4338a31114b67f827ee9706493e0`.
- M6.2 routes oracle trace fields through the same helper used by
  `ds4_session_sample`, and request-surface sampling tuples now resolve through
  shared `ds4_sampling_params_*` helpers used by server, CLI, and agent
  defaults.
- M6.2 added `python3 ds4-parity/check_sampling_dump.py`, whose schema checker
  validates coverage for greedy ties, non-finite logits, temperature
  normalization, `top_p` clamping, `top_k` caps, `min_p` thresholds,
  full-vocab sampling, seeded RNG draws, top-logprob ordering, token-logprob
  requests, and request-surface parameter tuples. Its negative tests catch
  selected-token drift, missing request cases, candidate-list drift,
  top-logprob ordering drift, token-logprob schema drift, and manifest hash
  drift.
- M6.2 validation passed for `arch -arm64 make ds4-sampling-dump`,
  `./ds4-sampling-dump /tmp/ds4-sampling-m6.2-refresh.json` compared
  byte-for-byte with the committed baseline,
  `python3 ds4-parity/check_sampling_dump.py
  ds4-parity/baselines/sampling/m6.2/current-c.json --manifest
  ds4-parity/baselines/sampling/m6.2/manifest.json --negative-test` with
  `sampling schema: PASS, 1243 checks`, `sampling manifest: PASS, 7 checks`,
  and `sampling negative tests: PASS, 6 checks`, `python3 -m py_compile
  ds4-parity/check_sampling_dump.py`, `arch -arm64 make ds4_test`,
  `./ds4_test --server`, `arch -arm64 make cpu`, CPU
  `./ds4-sampling-dump` compared byte-for-byte with the committed baseline, and
  `git diff --check`.
- M6.2 Claude review returned `NO BLOCKERS` after checking sampler helper
  sharing, RNG bookkeeping, candidate ordering, request-surface helper
  plumbing, fake-session logprob safety, manifest checks, and negative-test
  coverage. Non-blocking notes: `matches_actual` now compares two calls through
  the same helper, and the schema checker is mostly shape/coverage while
  byte-for-byte baseline comparison carries M6.2 drift detection.
- M6.2 implementation commit:
  `b1b637978779700fb6ce7250e67eaa3eb23c19c6`.
- M6.3 added Rust no-model sampler/logprob math in `ds4_gguf::sampling`,
  including argmax, xorshift RNG, top-p/min-p/top-k filtering, full-vocab
  sampling, top-logprob slices, token-logprob scoring, and shared sampling
  parameter defaults.
- M6.3 added `cargo run --quiet -p ds4-gguf --bin ds4-sampling-dump-rs`, which
  emits the same fixed-logits case set as the M6.2 C oracle with selected
  tokens, RNG states, filtered candidates, and logprob scores.
- M6.3 added `python3 ds4-parity/compare_sampling.py --negative-test`, whose
  C/Rust comparator enforces exact selected token, RNG state, candidate IDs,
  candidate counts, request case coverage, top-logprob order, and token-logprob
  request shape, with `1e-5` ordinary absolute float tolerance and `1e-6`
  relative tolerance for large sentinel values. Negative tests catch selected
  token drift, RNG drift, candidate-list drift, logprob drift, and request
  coverage drift.
- M6.3 validation passed for `cargo test -p ds4-gguf sampling --quiet` with 3
  sampling tests passing, `python3 -m py_compile
  ds4-parity/compare_sampling.py`, `python3 ds4-parity/compare_sampling.py
  --negative-test --write-rust-dump /tmp/ds4-sampling-rust-from-comparator.json`
  with `sampling C/Rust comparator: PASS, 3241 checks` and `sampling C/Rust
  negative tests: PASS, 5 checks`, `cargo fmt --all -- --check`,
  `cargo test --workspace` with all workspace tests passing, and
  `git diff --check`.
- M6.3 Claude review returned `NO BLOCKERS` after checking Rust numeric edge
  cases, RNG semantics, candidate filtering order, top-logprob tie order,
  non-finite handling, request fixture coverage, and comparator negative tests.
  Non-blocking notes: top-p/full-vocab tied-logit fixture coverage is latent,
  Rust faithfully recomputes full-vocab weights during roulette like C, and
  greedy mode intentionally leaves effective params unclamped to match C.
- M6.3 implementation commit:
  `fea2ea3de57a260474d349d2536527bf2c16927a`.
- M6.4 added `./ds4-logits-dump`, a current-C model-backed oracle helper that
  runs official-vector prompts through `ds4_session_sync`,
  `ds4_session_argmax`, `ds4_session_top_logprobs`,
  `ds4_session_token_logprob`, and `ds4_session_eval`, then records selected
  tokens, token bytes, top-logprob slices, official-top deltas, and per-step
  full-logits SHA256s. The helper requires a 64-character lowercase
  `--model-sha256` and verifies the actual model file via `sha256sum` or
  `shasum -a 256` before opening the engine.
- M6.4 exposes `ds4_session_logits_data` so the dump helper can write a
  contiguous f32 logits blob without moving model execution into the helper.
- M6.4 captured B300 current-C artifacts on `ds4-rust-port-b300` in
  `hou2-prod1/default` after refreshing source into `/workspace/ds4` and
  building `make ds4-logits-dump CUDA_ARCH=native`. Capture command wrote
  `ds4-parity/baselines/sampling/m6.4/current-c.json` with size 19,535 bytes
  and SHA256
  `5343e5aa855305ca2092943e155a359db50a28216d44927d450d2e0cce82efd0`,
  plus `ds4-parity/baselines/sampling/m6.4/logits.f32le` with size
  4,654,080 bytes and SHA256
  `972636c24ff63534d3a7fb7b1360e78786dee0bdd111f1fde813aa758e1f1928`.
- M6.4 fixture contains 9 scored steps across `short_italian_fact`,
  `short_code_completion`, and `short_reasoning_plain`. `long_memory_archive`
  remains explicitly skipped for the existing API/official-graph mismatch, and
  `long_code_audit` is explicitly skipped because repeated B300 CUDA captures
  produced byte-different long-context logits even with deterministic-kernel
  probes.
- M6.4 added `python3 ds4-parity/check_session_logits_dump.py`, whose schema
  and hash checker validates model/backend identity, case coverage, prompt
  hashes, selected-token matches, top-logprob shape, selected/top scores
  recomputed from the f32le logits blob, official-top local matches and delta
  tolerances, contiguous per-step logits ranges, per-step logits SHA256s,
  whole-blob manifest SHA256 plus n_vocab/step counts, and exact
  temp-kubeconfig/context refresh commands.
- M6.4 validation passed for `arch -arm64 make ds4-logits-dump`,
  `python3 -m py_compile ds4-parity/check_session_logits_dump.py`, B300
  `make ds4-logits-dump CUDA_ARCH=native`, B300 capture with
  `./ds4-logits-dump --backend cuda -m /workspace/ds4/ds4flash.gguf -v
  tests/test-vectors/official.vec -o
  ds4-parity/baselines/sampling/m6.4/current-c.json -l
  ds4-parity/baselines/sampling/m6.4/logits.f32le --model-sha256
  efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668`,
  local and B300 checker runs with `session logits schema: PASS, 2356 checks`,
  `session logits manifest: PASS, 20 checks`, and `session logits negative
  tests: PASS, 11 checks`, `python3 ds4-parity/compare_logprob_numeric.py`
  with `summary: 5/5 sections passed, 528 checks`, `arch -arm64 make cpu`,
  and `git diff --check`.
- M6.5 adds `ds4-model-logits-dump-rs`, which reads the committed M6.4
  `logits.f32le` blob as contiguous f32 vocab slices, loads the M5.3
  tokenizer GGUF, runs Rust `sample_argmax` and `top_logprobs`, and emits a
  flat per-slice JSON dump with selected token IDs, rendered token bytes, and
  top-logprob scores.
- M6.5 adds `python3 ds4-parity/compare_model_logits.py`, which maps those
  flat Rust slices back to the M6.4 current-C case/step records and compares
  selected token, selected bytes, expected bytes, logits offsets, top-logprob
  ordering, top token IDs, token bytes, logits, and logprobs.
- M6.5 validation passed with `python3 -m py_compile
  ds4-parity/compare_model_logits.py`, `python3
  ds4-parity/compare_model_logits.py --negative-test` (`model logits C/Rust
  comparator: PASS, 2982 checks, max_abs_logit_delta=5.00000041e-08,
  max_abs_logprob_delta=5.00000006e-08`; negative tests `PASS, 6 checks`),
  `cargo fmt --all -- --check`, `cargo test -p ds4-gguf --bin
  ds4-model-logits-dump-rs`, `cargo test --workspace`, and
  `git diff --check`.
- M6.6a adds `./ds4-decode-policy-dump`, a no-model current-C decode stop
  policy oracle. The helper includes the `ds4_server.c` test surface so the
  fixture uses the real C stop-list, UTF-8 stream-hold, DSML marker, generated
  message parse, Anthropic stop-reason, and Responses status mapping helpers.
- M6.6a fixture `ds4-parity/baselines/sampling/m6.6a/current-c.json` covers
  CLI EOS/length, server OpenAI EOS/length/user-stop/stream-stop-tail/
  streaming-stop-hit/partial-UTF-8/stop-at-mid-UTF-8-boundary/
  tool-call-boundary, server Responses length mapping, server Anthropic tool
  mapping, and agent EOS/length defaults. The artifact is 17,000 bytes with
  SHA256
  `9d11d90a12e1ee4d16ac1d4aa8c971efe775a86202004db91aff8d452081a2b5`.
- M6.6a adds `python3 ds4-parity/check_decode_policy_dump.py`, whose schema
  and negative checks validate case coverage, request option records,
  generated text schedules, finish reason, visible bytes, streamed bytes,
  held-tail bytes, session invalidation, stop boundary offsets, tool-call
  boundary flags, and API finish mappings.
- M6.6a validation passed with `arch -arm64 make ds4-decode-policy-dump`,
  `./ds4-decode-policy-dump ds4-parity/baselines/sampling/m6.6a/current-c.json`,
  `python3 -m py_compile ds4-parity/check_decode_policy_dump.py`, `python3
  ds4-parity/check_decode_policy_dump.py --negative-test` (`decode policy
  schema: PASS, 969 checks`; manifest `PASS, 5 checks`; negative tests `PASS,
  10 checks`), `arch -arm64 make ds4_test`, `./ds4_test --server`, and
  `git diff --check`.
- M6.6b adds the Rust byte-oriented decode stop policy in
  `rust/ds4-gguf/src/decode_policy.rs` plus `ds4-decode-policy-dump-rs`.
  It mirrors the M6.6a generated-token schedules without introducing a Rust
  CLI/server runtime or reimplementing M5 DSML parsing; the tool case only
  observes complete tool-call marker boundaries.
- M6.6b adds `python3 ds4-parity/compare_decode_policy.py`, which runs the
  Rust dump and compares request records, schedules, finish reason, raw and
  visible bytes, streamed bytes, held tails, session invalidation, stop
  boundaries, tool-boundary flags, API finish mappings, and per-step streaming
  metadata against the committed M6.6a C oracle.
- M6.6b validation passed with `python3 -m py_compile
  ds4-parity/compare_decode_policy.py`, `python3
  ds4-parity/compare_decode_policy.py --negative-test` (`decode policy C/Rust
  comparator: PASS, 1059 checks`; negative tests `PASS, 10 checks`), `cargo
  fmt --all -- --check`, `cargo test -p ds4-gguf decode_policy`, `cargo test
  -p ds4-gguf --bin ds4-decode-policy-dump-rs`, `cargo test --workspace`, and
  `git diff --check`.
- M6.7 adds `python3 ds4-parity/run_sampling_parity_report.py`, which runs the
  M6.2 current-C sampler/logprob checker, M6.3 Rust sampler comparator, M6.4
  committed session-logits fixture checker, M6.5 Rust model-logits comparator,
  M6.6a current-C decode policy checker, and M6.6b Rust decode-policy
  comparator.
- M6.7 records the model-backed M6.4 B300 session-logits recapture as a
  skipped report item using the exact `refresh_commands` from
  `ds4-parity/baselines/sampling/m6.4/manifest.json`; no other M6 local
  comparator is skipped by the M6 report.
- M6.7 wires the sampling/logprob report into
  `python3 ds4-parity/run_parity_report.py`. Validation passed with `python3
  -m py_compile ds4-parity/run_sampling_parity_report.py
  ds4-parity/run_parity_report.py`, `python3
  ds4-parity/run_sampling_parity_report.py` (`summary: 6 passed, 1 skipped, 0
  failed`), `python3 ds4-parity/run_sampling_parity_report.py --json |
  python3 -m json.tool`, `python3 ds4-parity/run_parity_report.py
  --skip-local-oracles` (`summary: 7 passed, 10 skipped, 0 failed`), `python3
  ds4-parity/run_parity_report.py --skip-local-oracles --json | python3 -m
  json.tool`, `cargo test --workspace`, and `git diff --check`.
- M7.1 split Milestone 7 into C KV header/policy oracle, Rust KV
  parser/policy, generic full-file round-trip coverage, per-extension trailer
  coverage, C on-disk session payload shape oracle, Rust payload header reader,
  KV replay/prefix decision comparator, B300 disk-KV and in-memory snapshot
  restore oracle, and report integration items. The first executable item is
  the no-model C KV header and policy oracle; the C on-disk session payload
  shape oracle is independently eligible because it depends on session payload
  code rather than KV header/policy work.
- M7.2 added `./ds4-kv-policy-dump`, a deterministic no-model current-C
  oracle for KVC header bytes, decoded fields, reason/key-kind mapping,
  little-endian helpers, SHA/path helpers, size budgeting, store-boundary
  selection, chat-anchor selection, continued-store targets, byte-prefix
  matching, eviction scoring with explicit `now`, text-prefix lookup, and M0.5
  parsed header fixture references.
- M7.2 added `python3 ds4-parity/check_kv_policy_dump.py`, whose schema,
  manifest, and negative checks validate the C oracle dump and the committed
  M0.5 `kv-header.tsv` row references.
- M7.2 local validation passed for `arch -arm64 make ds4-kv-policy-dump`,
  `./ds4-kv-policy-dump ds4-parity/baselines/kv/m7.2/current-c.json`,
  `python3 -m json.tool ds4-parity/baselines/kv/m7.2/current-c.json`,
  `python3 -m py_compile ds4-parity/check_kv_policy_dump.py`, and
  `python3 ds4-parity/check_kv_policy_dump.py --negative-test` (`451` schema
  checks, `11` manifest checks, `7` negative checks), `arch -arm64 make`,
  `arch -arm64 make cpu`, deterministic CPU-regenerated dump comparison
  against the committed M7.2 artifact, `arch -arm64 make ds4_test`,
  `./ds4_test --server`, and `git diff --check`.
- M7.3 adds `rust/ds4-gguf/src/kv_policy.rs` for no-model KVC header
  parsing/writing, reason/key-kind helpers, SHA/path helpers, file-size
  budgeting, store-boundary selection, chat-anchor selection,
  continued-store target selection, byte-prefix matching, eviction scoring,
  and text-prefix entry selection.
- M7.3 adds `ds4-kv-policy-dump-rs`, which emits the same deterministic
  synthetic no-model policy fixture as the M7.2 C oracle with a Rust schema and
  source label.
- M7.3 adds `python3 ds4-parity/compare_kv_policy.py`, which runs the Rust
  dump and recursively compares it to the committed M7.2 C oracle while
  allowing only the schema/source labels to differ. It checks header bytes,
  decoded fields, reason and extension flags, SHA/path helpers, policy
  decisions, eviction scores, text-prefix selections, and M0.5 header rows.
- M7.3 local validation passed for `python3 -m py_compile
  ds4-parity/compare_kv_policy.py`, `python3
  ds4-parity/compare_kv_policy.py --negative-test` (`KV policy C/Rust
  comparator: PASS, 1488 checks`; negative tests `PASS, 8 checks`), `python3
  ds4-parity/check_kv_policy_dump.py --negative-test` (`451` schema checks,
  `11` manifest checks, `7` negative checks), `cargo fmt --all -- --check`,
  `cargo test -p ds4-gguf kv_policy`, `cargo test -p ds4-gguf --bin
  ds4-kv-policy-dump-rs`, and `cargo test --workspace`.
- M7.4a adds `./ds4-kvc-file-dump`, a deterministic no-model current-C oracle
  for complete generic KVC file bytes: fixed header, text length, rendered-text
  bytes, opaque payload bytes, and opaque trailer bytes.
- M7.4a fixture `ds4-parity/baselines/kv/m7.4a/current-c.json` covers
  no-trailer, opaque trailer, visible-transcript flag without payload, empty
  text with trailer, no-budget/fitting-budget/over-budget size decisions, and
  malformed header/text/payload/trailer boundary records. The artifact is
  6,445 bytes with SHA256
  `ff37ba4a359b10d66199928a1936b10ec0adc43a17ceb7ba49c0ad3e02c8b7d7`.
- M7.4a adds Rust generic KVC full-file helpers in
  `rust/ds4-gguf/src/kv_policy.rs`; the reader keeps payload and trailer bytes
  opaque and treats all bytes after fixed header, text, and declared payload as
  generic trailer data.
- M7.4a adds `python3 ds4-parity/compare_kvc_file.py`, which runs
  `ds4-kvc-file-dump-rs` and compares complete file hex, read metadata,
  file-size budget records, malformed case outcomes, and trailer-size records
  against the committed C oracle.
- M7.4a local validation passed for `arch -arm64 make ds4-kvc-file-dump`,
  `./ds4-kvc-file-dump ds4-parity/baselines/kv/m7.4a/current-c.json`,
  `python3 -m json.tool ds4-parity/baselines/kv/m7.4a/current-c.json`,
  `python3 -m py_compile ds4-parity/compare_kvc_file.py`, `python3
  ds4-parity/compare_kvc_file.py --negative-test` (`KVC file C/Rust
  comparator: PASS, 277 checks`; negative tests `PASS, 8 checks`), `cargo
  fmt --all -- --check`, `cargo test -p ds4-gguf kvc`, `cargo test -p
  ds4-gguf --bin ds4-kvc-file-dump-rs`, `cargo test --workspace`, `arch
  -arm64 make cpu`, CPU-regenerated `./ds4-kvc-file-dump` comparison against
  the committed M7.4a artifact, and `git diff --check`.
- M7.4b adds `./ds4-kv-trailer-dump`, a deterministic no-model current-C
  oracle for server-owned KVC trailer payloads using the real
  `kv_tool_map_serialized_size`, `kv_tool_map_write`, and
  `kv_tool_map_load_from_pos` helpers from `ds4_server.c`.
- M7.4b fixture `ds4-parity/baselines/kv/m7.4b/current-c.json` covers empty
  tool-map output, single-block output, text filtering, duplicate-block
  suppression, multiple IDs for one DSML block, UTF-8 bytes with a long ID,
  disabled exact replay, visible-transcript extension flags without payload
  bytes, and malformed trailer load/decode boundaries. The artifact is 13,232
  bytes with SHA256
  `c5f73f2ea0f712e5fa1f2ee57666e1907304324d5334f398356c90ca40401d73`.
- M7.4b adds Rust tool-map trailer helpers in
  `rust/ds4-gguf/src/kv_policy.rs`; the writer scans DSML tool-call blocks,
  suppresses duplicate blocks, mirrors C's reverse insertion order for
  multiple IDs on one block, and the reader preserves partial decoded entries
  on malformed trailers.
- M7.4b adds `python3 ds4-parity/compare_kv_trailer.py`, which runs
  `ds4-kv-trailer-dump-rs` and compares trailer bytes, decoded entries,
  load-count behavior, wanted-ID filtering, extension flag records, and
  malformed trailer categories against the committed C oracle.
- M7.4b local validation passed for `arch -arm64 make ds4-kv-trailer-dump`,
  `./ds4-kv-trailer-dump ds4-parity/baselines/kv/m7.4b/current-c.json`,
  `python3 -m json.tool ds4-parity/baselines/kv/m7.4b/current-c.json`,
  `python3 -m py_compile ds4-parity/compare_kv_trailer.py`, `python3
  ds4-parity/compare_kv_trailer.py --negative-test` (`KV trailer C/Rust
  comparator: PASS, 432 checks`; negative tests `PASS, 8 checks`), `cargo
  fmt --all -- --check`, `cargo test -p ds4-gguf tool_map`, `cargo test -p
  ds4-gguf --bin ds4-kv-trailer-dump-rs`, `cargo test --workspace`, `arch
  -arm64 make cpu`, CPU-regenerated `./ds4-kv-trailer-dump` comparison against
  the committed M7.4b artifact, and `git diff --check`.
- M7.5 adds `./ds4-session-payload-dump`, a deterministic no-model current-C
  oracle for the DSV4 session payload shape using the real payload constants,
  fixed DS4 model layout constants, size formula, and `ds4_session_load_payload`
  rejection behavior on synthetic CPU fixtures.
- M7.5 fixture `ds4-parity/baselines/kv/m7.5/current-c.json` records the
  13-u32 DSV4 header, little-endian magic bytes, field order, body section
  order, synthetic payload byte accounting, header rejection categories,
  body/trailing/truncated/row-count rejection categories, and M0.5 B300
  model-backed payload size/hash records. The artifact is 19,774 bytes with
  SHA256 `479d05d7274fde43ea5a2676895637639113534ee3f7bbb2723d032756b10806`.
- M7.5 adds `python3 ds4-parity/check_session_payload_shape.py`, which reruns
  the C payload dump, compares it to the committed fixture, verifies the M0.5
  payload records against committed logs and hashes, and checks that exact B300
  recapture commands preserve the temp kubeconfig, explicit context, pod, and
  model path.
- M7.5 local validation passed for `arch -arm64 make ds4-session-payload-dump`,
  `./ds4-session-payload-dump | python3 -m json.tool`, baseline generation via
  `python3 ds4-parity/check_session_payload_shape.py --write-baseline
  ds4-parity/baselines/kv/m7.5/current-c.json`, `python3 -m json.tool
  ds4-parity/baselines/kv/m7.5/current-c.json`, `python3 -m py_compile
  ds4-parity/check_session_payload_shape.py`, and `python3
  ds4-parity/check_session_payload_shape.py --negative-test` (`Session payload
  shape oracle: PASS, 552 checks`; negative tests `PASS, 8 checks`), `arch
  -arm64 make cpu`, and `git diff --check`.
- M7.6 adds `rust/ds4-gguf/src/session_payload.rs`, a no-runtime-restore Rust
  reader for DSV4 payload headers and structural body boundaries. It mirrors
  C's combined bad-magic/bad-version `unsupported-version` behavior, fixed DS4
  layout checks, CPU layout/cap checks, row-count validation, truncated body
  rejection, and trailing-payload rejection.
- M7.6 adds `ds4-session-payload-dump-rs`, which emits the Rust structural
  surface for the same synthetic M7.5 cases without loading a model or claiming
  tensor/session restore support.
- M7.6 adds `python3 ds4-parity/compare_session_payload.py`, which compares
  Rust output to the M7.5 current-C structural oracle and checks the M0.5
  payload/hash/B300-command records as fixture preconditions.
- M7.6 local validation passed for `cargo fmt --all -- --check`, `python3 -m
  py_compile ds4-parity/compare_session_payload.py`, `python3
  ds4-parity/compare_session_payload.py --negative-test` (`Session payload
  C/Rust structural comparator: PASS, 347 checks`; M0.5 fixture contract
  `PASS, 17 checks`; negative tests `PASS, 8 checks`), `cargo test -p
  ds4-gguf session_payload`, `cargo test -p ds4-gguf --bin
  ds4-session-payload-dump-rs`, `cargo test --workspace`, and
  `git diff --check`.
- M7.7 adds Rust cache-replay helpers in `rust/ds4-gguf/src/kv_policy.rs`
  for live-prefix reuse, disk-text restore accounting, cache write token
  calculation, and byte-prefix effective prompt suffix construction.
- M7.7 adds `ds4-kv-replay-dump-rs`, a no-model Rust replay fixture for the
  committed M0.5 cold/disk restore cases and M0.4 DSML/cache-continuation
  cases.
- M7.7 adds `ds4-parity/baselines/kv/m7.7/current-c.json`, derived from the
  committed M0.4/M0.5 traces and responses. It records M5 prompt-rendering
  artifact hashes as fixture preconditions, six replay cases, disk-cache
  reason/key/rendered-text fields, token-window hashes for mismatch cases,
  DSML tool-call records, and effective prompt suffix byte hex for disk and
  memory-prefix restores.
- M7.7 adds `python3 ds4-parity/compare_kv_replay.py`, which regenerates the
  C replay oracle from committed artifacts, fails M5 hash drift as a
  precondition, compares Rust replay output, and checks the M7.3 Rust policy
  dump's M0.5 KVC header rows.
- M7.7 local validation passed for `cargo fmt --all -- --check`, `python3 -m
  py_compile ds4-parity/compare_kv_replay.py`, JSON validation for
  `ds4-parity/baselines/kv/m7.7/current-c.json` and `manifest.json`, `python3
  ds4-parity/compare_kv_replay.py --negative-test` (`KV replay C fixture
  preconditions: PASS, 333 checks`; `KV replay C/Rust comparator: PASS, 273
  checks`; `KV replay Rust policy precondition: PASS, 14 checks`; manifest
  `PASS, 6 checks`; negative tests `PASS, 6 checks`), `cargo test -p
  ds4-gguf kv_policy`, `cargo test -p ds4-gguf --bin
  ds4-kv-replay-dump-rs`, `cargo test --workspace`, and `git diff --check`.
- M7.8 adds `./ds4-restore-dump`, a current-C model-backed restore oracle
  helper for the recorded B300 model. It captures disk DSV4 payload restore and
  in-memory `ds4_session_snapshot` restore for seed and continuation prompts,
  recording selected tokens, top-20 logprob slices, max score deltas, payload
  sizes, payload/snapshot hashes, header prefixes, fixture hashes, backend
  identity, and raw-payload non-commit policy.
- M7.8 B300 validation on `ds4-rust-port-b300` in `hou2-prod1` refreshed the
  uncommitted M7.8 source delta into `/workspace/ds4`, built
  `ds4-restore-dump` with `CUDA_ARCH=native`, opened
  `/workspace/ds4/ds4flash.gguf` with SHA256
  `efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668`, and
  captured `ds4-parity/baselines/kv/m7.8/current-c.json`.
- M7.8 committed `current-c.json` is 15,715 bytes with SHA256
  `5a50459507e7750179f187a1ea177ac8f0f44c8e2c41ea6a08ee922e861e7574`;
  raw restore bodies remain uncommitted on the capture workspace and are
  represented by hashes plus exact B300 refresh commands in
  `ds4-parity/baselines/kv/m7.8/manifest.json`.
- M7.8 validation passed on B300 for `python3
  ds4-parity/check_restore_dump.py
  ds4-parity/baselines/kv/m7.8/current-c.json --negative-test` (`restore
  oracle schema: PASS, 1448 checks`; negative tests `PASS, 6 checks`).
- M7.8 local validation passed for `arch -arm64 make ds4-restore-dump`,
  `python3 -m py_compile ds4-parity/check_restore_dump.py`, manifest generation
  with `--write-manifest`, `python3 ds4-parity/check_restore_dump.py
  ds4-parity/baselines/kv/m7.8/current-c.json --manifest
  ds4-parity/baselines/kv/m7.8/manifest.json --negative-test` (`restore oracle
  schema: PASS, 1448 checks`; manifest `PASS, 13 checks`; negative tests
  `PASS, 6 checks`), `python3 -m json.tool` for both committed M7.8 JSON files,
  `arch -arm64 make cpu`, and `git diff --check`.
- M7.9 adds `python3 ds4-parity/run_kv_parity_report.py`, which builds the
  local no-model `ds4-session-payload-dump` helper, runs M7.2 through M7.8
  KV/snapshot comparator commands, emits text or machine-readable JSON, and
  skips only the model-backed M7.8 B300 restore recapture with the manifest
  refresh commands.
- M7.9 wires the Milestone 7 report into
  `python3 ds4-parity/run_parity_report.py` as the `M7.9 KV/snapshot parity
  report` comparator item, so the unified report now covers M1, M4, M5, M6, and
  M7 comparator families.
- M7.9 validation passed for `python3 -m py_compile
  ds4-parity/run_kv_parity_report.py ds4-parity/run_parity_report.py`,
  `python3 ds4-parity/run_kv_parity_report.py` (`summary: 9 passed, 1 skipped,
  0 failed`), `python3 ds4-parity/run_kv_parity_report.py --json | python3 -m
  json.tool >/dev/null`, `python3 ds4-parity/run_parity_report.py` (`summary:
  13 passed, 5 skipped, 0 failed`), `cargo test --workspace`, and
  `git diff --check`.
- M8.1 split Milestone 8 into CLI parity work items after inspecting
  `ds4_cli.c` usage text, parser branches, prompt building, one-shot
  generation, `--dump-tokens`, `--dump-logprobs`, `--perplexity-file`,
  `--inspect`, imatrix capture, debug/test mode flags, thinking controls, and
  REPL command handling.
- M8.1 added roadmap items M8.2 through M8.16, covering current-C parse/error
  oracle, Rust parse/error parity, token/prompt diagnostics, logprob/perplexity
  diagnostics, inspect output, imatrix capture, one-shot generation,
  interactive PTY transcripts, and CLI report integration. The next executable
  item is M8.2 current-C CLI parse/error oracle.
- M8.1 validation passed for docs/state-only diff inspection and
  `git diff --check`.
- M8.2 adds `python3 ds4-parity/check_cli_parse_dump.py`, a local no-model
  current-C CLI parser/error oracle checker. It captures 20 cases for `--help`,
  missing option values, unknown options, invalid numeric/float/backend values,
  duplicate prompt sources, missing prompt files, `--server`,
  `--metal-graph-generate`, `--dump-tokens` without a prompt, imatrix option
  coupling, and `--perplexity-file` prompt-source rejection.
- M8.2 committed `ds4-parity/baselines/cli/m8.2/current-c.json` is 23,626 bytes
  with SHA256 `d395a55e92957b84deb4cb43d4b70c5a2e78bac363b7d11be1200f1d3601fa22`.
  The checker stores stdout/stderr bytes and hashes while asserting stable help
  anchors, error categories, exact option names, exit status, and no model-load
  markers.
- M8.2 validation passed for `arch -arm64 make ds4`, baseline generation with
  `python3 ds4-parity/check_cli_parse_dump.py --write-baseline
  ds4-parity/baselines/cli/m8.2/current-c.json --write-manifest
  ds4-parity/baselines/cli/m8.2/manifest.json`, `python3 -m py_compile
  ds4-parity/check_cli_parse_dump.py`, `python3
  ds4-parity/check_cli_parse_dump.py
  ds4-parity/baselines/cli/m8.2/current-c.json --manifest
  ds4-parity/baselines/cli/m8.2/manifest.json --negative-test` (`CLI parse
  oracle: PASS, 369 checks`; manifest `PASS, 11 checks`; negative tests `PASS,
  7 checks`), `python3 -m json.tool` for both M8.2 JSON files, and
  `git diff --check`.
- M8.3 adds `rust/ds4-gguf/src/cli_parse.rs` and `ds4-cli-parse-rs`, a
  parser-only Rust CLI surface for the committed M8.2 no-model argument matrix.
  It emits the same exit categories for help, parser errors, removed/deprecated
  flags, imatrix/perplexity coupling, and `--dump-tokens` without a prompt, and
  deliberately exits with an unsupported parser-only status if a model-backed
  path is reached outside the M8.3 fixture.
- M8.3 adds `python3 ds4-parity/compare_cli_parse.py`, which builds
  `ds4-cli-parse-rs`, validates the M8.2 C fixture preconditions, and compares
  Rust exit status, stdout/stderr emptiness, stable help anchors, stderr
  category anchors, and no-model-load markers.
- M8.3 validation passed for `cargo fmt --all -- --check`, `python3 -m
  py_compile ds4-parity/compare_cli_parse.py`, `cargo test -p ds4-gguf
  cli_parse` (3 parser tests passed), `python3
  ds4-parity/compare_cli_parse.py --negative-test` (`CLI parse C fixture
  preconditions: PASS, 224 checks`; `CLI parse C/Rust comparator: PASS, 244
  checks`; negative tests `PASS, 5 checks`), `cargo test --workspace`, and
  `git diff --check`.
- M8.4 adds `python3 ds4-parity/check_cli_token_dump.py`, B300 fixtures under
  `ds4-parity/baselines/cli-fixtures/m8.4/`, and the committed current-C
  `--dump-tokens` oracle at `ds4-parity/baselines/cli/m8.4/current-c.json`.
  The checker captures raw stdout/stderr bytes as base64 plus hashes, parses
  the first-line token ID list, records prompt bytes and prompt-file hashes, and
  asserts that `--system`, empty `--system`, `--think`, both `--think-max`
  context thresholds, and `--nothink` are ignored by the early dump-token exit.
- M8.4 committed `ds4-parity/baselines/cli/m8.4/current-c.json` is 18,870 bytes
  with SHA256 `87d427fb88563c15a07e859618fd585c6cb847bc77add556da89edf504bfb51c`.
- M8.4 B300 validation passed after refreshing `/workspace/ds4` on
  `ds4-rust-port-b300`, building `make ds4 CUDA_ARCH=native`, capturing the
  baseline, and running `python3 ds4-parity/check_cli_token_dump.py
  ds4-parity/baselines/cli/m8.4/current-c.json --manifest
  ds4-parity/baselines/cli/m8.4/manifest.json --negative-test` (`CLI token dump
  oracle: PASS, 306 checks`; manifest `PASS, 18 checks`; negative tests `PASS,
  8 checks`).
- M8.5 refactors `rust/ds4-gguf/src/cli_parse.rs` to expose the parsed model
  path, prompt text, and `--dump-tokens` flag while preserving the M8.3
  parser-only surface. It adds `ds4-cli-token-dump-rs`, which loads the M5.3
  tokenizer GGUF, tokenizes the raw prompt with `tokenize_rendered_chat`, and
  writes the C diagnostic format.
- M8.5 adds `Ds4Tokenizer::token_text_bytes` for diagnostics that need raw
  tokenizer table bytes. The C `dump_tokens_fp` output uses those raw token text
  bytes (`Ġtoken` style), not decoded token bytes (` token` style); existing
  decoded `token_bytes` behavior is unchanged.
- M8.5 adds `python3 ds4-parity/compare_cli_token_dump.py`, which validates the
  M8.4 C fixture, checks the M5.3 tokenizer fixture hash, substitutes that
  tokenizer fixture for the B300 model path, and compares Rust/C exit status,
  stdout bytes, stderr bytes, and token IDs exactly.
- M8.5 validation passed for `cargo fmt --all -- --check`, `cargo test -p
  ds4-gguf cli_parse` (4 parser tests passed), `cargo test -p ds4-gguf
  token_text_decodes_gpt2_byte_mapping` (1 tokenizer diagnostic test passed),
  `python3 -m py_compile ds4-parity/compare_cli_token_dump.py`, `python3
  ds4-parity/compare_cli_token_dump.py --skip-build --negative-test` (`CLI
  token dump tokenizer fixture: PASS, 3 checks`; C fixture preconditions `PASS,
  166 checks`; C/Rust comparator `PASS, 65 checks`; negative tests `PASS, 5
  checks`), `python3 ds4-parity/compare_cli_parse.py --skip-build
  --negative-test` (`CLI parse C fixture preconditions: PASS, 224 checks`;
  C/Rust comparator `PASS, 244 checks`; negative tests `PASS, 5 checks`), and
  `cargo test --workspace`.
- M8.6 adds `python3 ds4-parity/check_cli_diagnostics_dump.py`, fixtures under
  `ds4-parity/baselines/cli-fixtures/m8.6/`, and the current-C diagnostic
  artifact at `ds4-parity/baselines/cli/m8.6/current-c.json`.
- M8.6 captures four B300 CLI diagnostic cases: inline `--dump-logprobs`
  (`top_k=3`, 2 generated steps, selected IDs 2581 and 1309), prompt-file
  `--dump-logprobs` (`top_k=5`, 1 generated step, selected ID 2581), a bad
  logprob output path error, and `--perplexity-file` (`tokens=69`, `scored=4`,
  `nll=0.158310216`, `avg_nll=0.039577554`, `ppl=1.040371181`).
- M8.6 committed `ds4-parity/baselines/cli/m8.6/current-c.json` is 16,161 bytes
  with SHA256 `838646513c85069db6ecc34ae5b8729257ecd89e7a6b28002e5e6e4f3edc429c`.
- M8.6 B300 validation passed after refreshing `/workspace/ds4` on
  `ds4-rust-port-b300`, building `make ds4 CUDA_ARCH=native`, capturing the
  baseline, and running `python3 ds4-parity/check_cli_diagnostics_dump.py
  ds4-parity/baselines/cli/m8.6/current-c.json --manifest
  ds4-parity/baselines/cli/m8.6/manifest.json --negative-test` (`CLI
  diagnostics oracle: PASS, 267 checks`; manifest `PASS, 12 checks`; negative
  tests `PASS, 7 checks`). Local revalidation of the copied artifact reported
  the same PASS counts; `python3 -m py_compile
  ds4-parity/check_cli_diagnostics_dump.py`, `python3 -m json.tool` for both
  M8.6 JSON files, and `git diff --check` also passed.
- M8.7 is not implemented as model-backed parity because the Rust tree does not
  yet expose a model/session execution boundary. Current Rust evidence is
  tokenizer/prompt/fixed-logits support in `rust/ds4-gguf`, a model-logits
  replay binary over captured logits, and low-level GPU tensor wrappers in
  `rust/ds4-gpu`; there is no Rust `ds4_engine`/`ds4_session` equivalent that
  can run the M8.6 CLI prompts.
- M8.7 has been split in `RUST_PORT_ROADMAP.md` into a runtime-boundary
  prerequisite and the actual CLI diagnostic output parity item. The original
  M8.7 is blocked until that runtime-boundary prerequisite exists; skipping to
  M8.8 avoids claiming replay-only artifact handling as execution parity.
- M8.8 adds `ds4-parity/check_cli_inspect_dump.py` and the current-C inspect
  artifact at `ds4-parity/baselines/cli/m8.8/current-c.json`.
- M8.8 captures two B300 CLI inspect cases: plain `--cuda --inspect` and an
  inspect case with prompt/context/generation controls that must keep the same
  summary stdout and avoid context-buffer, think-max, generation, perplexity,
  or imatrix logs.
- M8.8 committed `ds4-parity/baselines/cli/m8.8/current-c.json` is 9,087 bytes
  with SHA256 `613a03b604204831a04d5c74be15f3c4ecdf33f990aef43fcb3e92e6fe894ca1`.
- M8.8 B300 validation passed after refreshing `/workspace/ds4` on
  `ds4-rust-port-b300`, building `make ds4 CUDA_ARCH=native`, capturing the
  baseline, and running `python3 ds4-parity/check_cli_inspect_dump.py
  ds4-parity/baselines/cli/m8.8/current-c.json --manifest
  ds4-parity/baselines/cli/m8.8/manifest.json --negative-test` (`CLI inspect
  oracle: PASS, 112 checks`; manifest `PASS, 20 checks`; negative tests `PASS,
  8 checks`). Local revalidation of the copied artifact reported the same PASS
  counts; `python3 -m py_compile ds4-parity/check_cli_inspect_dump.py`,
  `python3 -m json.tool` for both M8.8 JSON files, and `git diff --check` also
  passed.
- M8.9 is not implemented as inspect parity because the Rust tree only accepts
  `--inspect` as a recognized parser option and still returns the parser-only
  model-backed-path stub from `parse_cli`; there is no Rust engine-open or
  engine-summary boundary.
- M8.9 has been split in `RUST_PORT_ROADMAP.md` into an inspect runtime-boundary
  prerequisite and the actual CLI inspect output surface item. The original
  M8.9 is blocked until the runtime-boundary prerequisite exists.
- M8.9a adds `rust/ds4-engine`, a Rust `Engine` wrapper over
  `ds4_engine_open`/`ds4_engine_summary`/`ds4_engine_close`, the
  `ds4-inspect-runtime-rs` runtime binary, and
  `ds4-parity/compare_cli_inspect_runtime.py`.
- M8.9a local validation passed for `cargo fmt --all -- --check`, `cargo test
  --workspace`, `python3 -m py_compile
  ds4-parity/compare_cli_inspect_runtime.py`, and `git diff --check`.
- M8.9a B300 validation used a temporary Rust 1.95.0 toolchain under
  `/tmp/ds4-cargo` and `/tmp/ds4-rustup`, built the CUDA-backed Rust binary,
  and passed `python3 ds4-parity/compare_cli_inspect_runtime.py
  ds4-parity/baselines/cli/m8.8/current-c.json --candidate-binary
  target/debug/ds4-inspect-runtime-rs --negative-test` (`CLI inspect runtime
  comparator: PASS, 68 checks`; negative tests `PASS, 5 checks`).
- M8.9b adds `ds4-cli-inspect-rs`, which routes parsed Rust CLI `--inspect`
  configuration through the M8.9a `Engine` boundary instead of replaying the
  M8.8 JSON artifact.
- M8.9b extends `CliConfig` so the Rust parser preserves inspect dispatch,
  backend selection, `--warm-weights`, and `--quality` for runtime-boundary
  consumers while keeping non-inspect model-backed paths stubbed.
- M8.9b updates `ds4-parity/compare_cli_inspect_runtime.py` with
  `--use-case-argv`, so the comparator can run the exact committed M8.8 CLI
  argv through the Rust binary, including prompt/control flags that current C
  ignores in inspect mode.
- M8.9b local validation passed for `cargo fmt --all -- --check`, `cargo test
  -p ds4-gguf cli_parse::tests::config_retains_inspect_backend_and_runtime_flags`,
  `cargo test -p ds4-engine`, `cargo test --workspace`, `python3 -m py_compile
  ds4-parity/compare_cli_inspect_runtime.py`, and `git diff --check`.
- M8.9b B300 validation used the temporary Rust 1.95.0 toolchain under
  `/tmp/ds4-cargo` and `/tmp/ds4-rustup`, built `ds4-cli-inspect-rs`, and
  passed `python3 ds4-parity/compare_cli_inspect_runtime.py
  ds4-parity/baselines/cli/m8.8/current-c.json --candidate-binary
  target/debug/ds4-cli-inspect-rs --use-case-argv --negative-test` (`CLI
  inspect comparator: PASS, 68 checks`; negative tests `PASS, 5 checks`).
- M8.10 is not implemented as an output-hash oracle because current C forces
  `--imatrix-out` to the Metal backend in `ds4_cli.c`, and
  `ds4_engine_collect_imatrix` requires `DS4_BACKEND_METAL` plus `metal_ready`.
  The B300 model host is a CUDA build, so it cannot write a valid imatrix
  `.dat` artifact today.
- M8.10 has been split in `RUST_PORT_ROADMAP.md` into M8.10a, the completed
  feasibility guard, and M8.10b, the blocked current-C imatrix output oracle.
- M8.10a B300 proof refreshed `/workspace/ds4` to
  `bfd96275d077e33970d368a92a99963451e3384d`, built `make ds4
  CUDA_ARCH=native`, wrote a tiny `/tmp/m8.10-imatrix-dataset.txt`, and ran
  `./ds4 -m /workspace/ds4/ds4flash.gguf --ctx 64 --imatrix-dataset
  /tmp/m8.10-imatrix-dataset.txt --imatrix-out /tmp/m8.10-imatrix.dat
  --imatrix-max-prompts 1 --imatrix-max-tokens 16`.
- M8.10a B300 proof returned exit 1, stdout bytes 0, stderr
  `ds4: context buffers 22.51 MiB (ctx=64, backend=metal, prefill_chunk=64,
  raw_kv_rows=256, compressed_kv_rows=18)` followed by
  `ds4: Metal backend requested but this build is linked with CUDA, not Metal`,
  and no `/tmp/m8.10-imatrix.dat` output file.
- M8.10a local availability check found no `ds4flash.gguf` or imatrix GGUF in
  the workspace on this `x86_64` host with `51539607552` bytes of RAM, so a
  local Metal capture of the recorded q2-imatrix model is not currently
  feasible.
- M8.11 is blocked because it requires the committed M8.10b current-C imatrix
  output fixture. It should not be implemented against the M8.10a failure proof
  or a synthetic `.dat` substitute.
- M8.12 is the next runnable roadmap item because it captures current-C
  one-shot generation transcripts on the B300 CUDA model host and does not
  depend on the blocked imatrix output oracle.
- M8.12 has been split in `RUST_PORT_ROADMAP.md` into M8.12a core prompt,
  thinking-control, seeded-sampling, and context transcript capture, followed by
  M8.12b advanced runtime-control coverage for MTP, directional steering,
  quality, warm-weights, threads, and backend-option behavior.
- M8.12a adds `ds4-parity/check_cli_generation_dump.py`, fixture
  `ds4-parity/baselines/cli-fixtures/m8.12a/prompt_file.txt`, and the
  current-C transcript artifact at
  `ds4-parity/baselines/cli/m8.12a/current-c.json`.
- M8.12a committed current-C artifact size is 25,651 bytes with SHA256
  `d56ab4566471731abdb55769b23dceb9c9e42ad1d40ffc18bbb201a343861628`.
- M8.12a captures five B300 CLI one-shot cases: greedy inline `--nothink`,
  prompt-file `--think`, low-context `--think-max` downgrade warning, seeded
  non-greedy sampling with seed `12345`, and a too-small-context error case.
- M8.12a case stdout hashes are: `greedy_inline_nothink`
  `862550215bb33a4e6f591f4c1c52fcd03dc98022f1acad87ce807f7d58b8c03c`,
  `prompt_file_think`
  `e566cf8e60978ac10a300c2503a68e03edd6d162fb11e2496057d57346660af0`,
  `think_max_downgrade`
  `c36d4b240ac10dcf300ba8d9d5aafc33957b8c1976c7f5ae2b26e654701e1b74`,
  `seeded_sampling_nothink`
  `c72869b348ae66d5a5267de18ed40cd032dc13a908bc32ad10d30f4d1b550c39`,
  and `ctx_too_small`
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- M8.12a B300 validation passed after copying the uncommitted checker/fixture to
  `/workspace/ds4`, running the capture/checker command, and then copying
  `current-c.json` and `manifest.json` back. Local revalidation passed
  `python3 ds4-parity/check_cli_generation_dump.py
  ds4-parity/baselines/cli/m8.12a/current-c.json --manifest
  ds4-parity/baselines/cli/m8.12a/manifest.json --negative-test` with oracle
  `PASS, 156 checks`, manifest `PASS, 17 checks`, and negative tests `PASS, 5
  checks`.
- M8.12b adds `ds4-parity/check_cli_runtime_controls_dump.py` and the
  current-C runtime-control transcript artifact at
  `ds4-parity/baselines/cli/m8.12b/current-c.json`.
- M8.12b committed current-C artifact size is 22,815 bytes with SHA256
  `b9e5aca6f745ce846d1daceed820f5c6e3aa06d80f87674b24017df285fe221e`.
- M8.12b captures five B300 CLI runtime-control cases:
  `--backend cuda --quality -t 2`, `--warm-weights`, directional steering via
  `dir-steering/out/verbosity.f32`, blocked `--backend metal` on the CUDA
  build, and blocked `--mtp /workspace/ds4/missing-mtp.gguf`.
- M8.12b records the directional steering support artifact
  `dir-steering/out/verbosity.f32` as 704,512 bytes with SHA256
  `6414573b7d88822e16e6fe5972386ef2f1e51fc8502fe5849c4a611afad50cdd`,
  and records that no MTP GGUF is present in the B300 workspace.
- M8.12b B300 validation passed after refreshing `/workspace/ds4` to
  `dad2b7d95cb0ad8fcdce6044dac860ae9bf68a44`, copying the uncommitted
  checker, rebuilding `ds4`, running the capture/checker command, and copying
  `current-c.json` and `manifest.json` back. Local revalidation passed
  `python3 ds4-parity/check_cli_runtime_controls_dump.py
  ds4-parity/baselines/cli/m8.12b/current-c.json --manifest
  ds4-parity/baselines/cli/m8.12b/manifest.json --negative-test` with oracle
  `PASS, 158 checks`, manifest `PASS, 16 checks`, and negative tests `PASS, 5
  checks`.
- M8.13 has been split in `RUST_PORT_ROADMAP.md` into M8.13a argmax one-shot
  runtime boundary, M8.13b session sampling runtime boundary, M8.13c core CLI
  transcript surface, and M8.13d runtime-control CLI transcript surface.
- M8.13 source inspection found that `rust/ds4-engine/src/lib.rs` currently
  wraps `ds4_engine_open`, `ds4_engine_summary`, and `ds4_engine_close`, but
  does not expose prompt encoding, generated-token text, argmax generation, or
  session sampling.
- M8.13 source inspection found that
  `rust/ds4-engine/src/bin/ds4-cli-inspect-rs.rs` still exits 99 for the
  non-inspect model-backed path, so it cannot produce M8.12a/M8.12b one-shot
  transcripts.
- M8.13 source inspection found the needed current-C runtime APIs in `ds4.h`:
  `ds4_encode_chat_prompt`, `ds4_tokenize_rendered_chat`,
  `ds4_engine_generate_argmax`, `ds4_token_text`, `ds4_tokens_free`,
  `ds4_session_create`, `ds4_session_sync`, `ds4_session_sample`, and
  `ds4_session_eval`.
- M8.13a adds safe Rust ownership for `ds4_tokens`, `ThinkMode`, C context
  memory estimates, prompt encoding through `ds4_encode_chat_prompt` or
  `ds4_tokenize_rendered_chat`, and argmax generation through
  `ds4_engine_generate_argmax` with Rust callbacks that convert generated token
  IDs through `ds4_token_text` and free the C-allocated pieces.
- M8.13a adds `ds4-argmax-runtime-rs`, a narrow runtime-boundary binary for
  greedy one-shot generation. It accepts the M8.12a greedy/error argv surface,
  logs context memory and Think Max downgrade warnings, but rejects nonzero
  `--temp` so seeded sampling remains in M8.13b.
- M8.13a adds `ds4-parity/compare_cli_argmax_runtime.py`, which runs
  `target/debug/ds4-argmax-runtime-rs` against the M8.12a current-C
  `greedy_inline_nothink`, `prompt_file_think`, `think_max_downgrade`, and
  `ctx_too_small` cases. It excludes `seeded_sampling_nothink` for M8.13b.
- M8.13a B300 validation over `/workspace/ds4/ds4flash.gguf` passed after
  overlaying the M8.13a files on the pushed M8.13 split commit, building with
  `CARGO_HOME=/tmp/ds4-cargo RUSTUP_HOME=/tmp/ds4-rustup
  PATH=/tmp/ds4-cargo/bin:$PATH CUDA_ARCH=native cargo build -p ds4-engine
  --bin ds4-argmax-runtime-rs`, and running
  `python3 ds4-parity/compare_cli_argmax_runtime.py
  ds4-parity/baselines/cli/m8.12a/current-c.json --candidate-binary
  target/debug/ds4-argmax-runtime-rs --negative-test`.
- M8.13a B300 comparator reported `CLI argmax runtime comparator: PASS, 109
  checks` and `CLI argmax runtime negative tests: PASS, 4 checks`.
- M8.13a local validation passed for `cargo fmt --all -- --check`, focused
  `cargo test -p ds4-engine token_printer -- --nocapture`, `cargo build -p
  ds4-engine --bin ds4-argmax-runtime-rs`, full `cargo test --workspace`,
  `python3 -m py_compile ds4-parity/compare_cli_argmax_runtime.py`, and
  `git diff --check`.
- M8.13b adds `SamplingOptions` and a Rust session-backed generation path that
  calls current-C `ds4_session_create`, `ds4_session_sync`,
  `ds4_session_sample`, `ds4_session_eval`, `ds4_session_ctx`,
  `ds4_session_pos`, and `ds4_token_eos`. It preserves Rust ownership of the
  generated stdout buffer and frees the C session with `ds4_session_free`.
- M8.13b adds `ds4-session-runtime-rs`, a narrow seeded non-greedy runtime
  binary. It accepts the M8.12a seeded sampling argv surface, requires `--seed`,
  and rejects `--temp 0` so greedy generation remains owned by M8.13a.
- M8.13b adds `ds4-parity/compare_cli_session_runtime.py`, which runs
  `target/debug/ds4-session-runtime-rs` against the M8.12a
  `seeded_sampling_nothink` current-C case with seed `12345`; it does not cover
  the M8.12a greedy cases already owned by M8.13a.
- M8.13b B300 validation over `/workspace/ds4/ds4flash.gguf` passed after
  overlaying the M8.13b files on the pushed M8.13a commit, building with
  `CARGO_HOME=/tmp/ds4-cargo RUSTUP_HOME=/tmp/ds4-rustup
  PATH=/tmp/ds4-cargo/bin:$PATH CUDA_ARCH=native cargo build -p ds4-engine
  --bin ds4-session-runtime-rs`, and running
  `python3 ds4-parity/compare_cli_session_runtime.py
  ds4-parity/baselines/cli/m8.12a/current-c.json --candidate-binary
  target/debug/ds4-session-runtime-rs --negative-test`.
- M8.13b B300 comparator reported `CLI session runtime comparator: PASS, 28
  checks` and `CLI session runtime negative tests: PASS, 5 checks`.
- M8.13b local validation passed for `cargo fmt --all -- --check`, `cargo test
  -p ds4-engine`, `cargo build -p ds4-engine --bin ds4-session-runtime-rs`,
  full `cargo test --workspace`, `python3 -m py_compile
  ds4-parity/compare_cli_session_runtime.py
  ds4-parity/compare_cli_argmax_runtime.py`, and `git diff --check`.
- M8.13c extends `rust/ds4-gguf/src/cli_parse.rs` so `CliConfig` retains the
  core one-shot generation surface: system prompt, context, token limit,
  temperature, top-p, min-p, optional seed, and thinking mode.
- M8.13c adds `ds4-cli-one-shot-rs`, which parses the exact M8.12a argv through
  the shared Rust CLI parser, routes `--temp 0` cases through the M8.13a argmax
  boundary, routes the seeded non-greedy case through the M8.13b session
  boundary, and rejects non-generation modes outside this milestone.
- M8.13c adds `ds4-parity/compare_cli_one_shot_runtime.py`, which runs
  `target/debug/ds4-cli-one-shot-rs` against all five M8.12a current-C cases:
  `greedy_inline_nothink`, `prompt_file_think`, `think_max_downgrade`,
  `seeded_sampling_nothink`, and `ctx_too_small`.
- M8.13c B300 validation over `/workspace/ds4/ds4flash.gguf` passed after
  overlaying the M8.13c files on the pushed M8.13b commit, building with
  `CARGO_HOME=/tmp/ds4-cargo RUSTUP_HOME=/tmp/ds4-rustup
  PATH=/tmp/ds4-cargo/bin:$PATH CUDA_ARCH=native cargo build -p ds4-engine
  --bin ds4-cli-one-shot-rs`, and running
  `python3 ds4-parity/compare_cli_one_shot_runtime.py
  ds4-parity/baselines/cli/m8.12a/current-c.json --candidate-binary
  target/debug/ds4-cli-one-shot-rs --negative-test`.
- M8.13c B300 comparator reported `CLI one-shot runtime comparator: PASS, 144
  checks` and `CLI one-shot runtime negative tests: PASS, 5 checks`.
- M8.13c local validation passed for `cargo fmt --all -- --check`, `cargo test
  -p ds4-gguf cli_parse -- --nocapture`, `cargo build -p ds4-engine --bin
  ds4-cli-one-shot-rs`, `cargo test -p ds4-engine`, full
  `cargo test --workspace`, `python3 -m py_compile
  ds4-parity/compare_cli_one_shot_runtime.py
  ds4-parity/compare_cli_argmax_runtime.py
  ds4-parity/compare_cli_session_runtime.py`, and `git diff --check`.
- M8.13d extends `EngineOptions` and `ds4-cli-one-shot-rs` so Rust one-shot
  generation passes through M8.12b runtime controls: optional MTP path, thread
  count, MTP draft tokens and margin, directional steering file and scales,
  warm weights, and quality.
- M8.13d extends `rust/ds4-gguf/src/cli_parse.rs` so the shared Rust CLI parser
  retains `--mtp`, `--mtp-draft`, `--mtp-margin`, `-t`/`--threads`,
  `--dir-steering-file`, `--dir-steering-ffn`, and `--dir-steering-attn`; it
  also preserves the current-C default of `--dir-steering-file` without an
  explicit scale implying FFN scale `1.0`.
- M8.13d makes `ds4-cli-one-shot-rs` return the C-side blocked startup exit
  path for `ds4_engine_open` failures, avoiding an extra Rust stderr wrapper for
  the M8.12b blocked `--backend metal` and missing-MTP cases.
- M8.13d adds `ds4-parity/compare_cli_runtime_controls_runtime.py`, which runs
  `target/debug/ds4-cli-one-shot-rs` against all five M8.12b current-C cases:
  `backend_name_cuda_quality_threads`, `warm_weights`, `directional_steering`,
  `backend_metal_error`, and `mtp_missing_model`.
- M8.13d B300 validation over `/workspace/ds4/ds4flash.gguf` passed after
  copying the changed files one by one to `/workspace/ds4`, verifying SHA256
  matches, building with `CARGO_HOME=/tmp/ds4-cargo
  RUSTUP_HOME=/tmp/ds4-rustup PATH=/tmp/ds4-cargo/bin:$PATH CUDA_ARCH=native
  cargo build -p ds4-engine --bin ds4-cli-one-shot-rs`, and running
  `python3 ds4-parity/compare_cli_runtime_controls_runtime.py
  ds4-parity/baselines/cli/m8.12b/current-c.json --candidate-binary
  target/debug/ds4-cli-one-shot-rs --negative-test`.
- M8.13d B300 comparator reported `CLI runtime-controls runtime comparator:
  PASS, 154 checks` and `CLI runtime-controls runtime negative tests: PASS, 6
  checks`.
- M8.13d local validation passed for `cargo fmt --all -- --check`, `cargo test
  -p ds4-gguf cli_parse -- --nocapture`, `cargo build -p ds4-engine --bin
  ds4-cli-one-shot-rs`, `cargo test -p ds4-engine`, full
  `cargo test --workspace`, `python3 -m py_compile
  ds4-parity/compare_cli_runtime_controls_runtime.py
  ds4-parity/compare_cli_one_shot_runtime.py
  ds4-parity/compare_cli_argmax_runtime.py
  ds4-parity/compare_cli_session_runtime.py`, and `git diff --check`.
- M8.14 adds `ds4-parity/check_cli_interactive_dump.py`, a current-C PTY
  capture/checker for interactive CLI transcripts. It sets an explicit PTY
  window size so `linenoise` does not block on cursor-position probing and sends
  carriage returns to match terminal Enter behavior.
- M8.14 adds fixture
  `ds4-parity/baselines/cli-fixtures/m8.14/read_prompt.txt` with SHA256
  `418e80cf0af232690c1cdb12b0ca015953f0f14d24c2c8ba40052464387b49b3`.
- M8.14 captures two current-C B300 PTY cases in
  `ds4-parity/baselines/cli/m8.14/current-c.json`: `command_suite` and
  `ctrl_c_at_prompt`. The command suite covers empty input, `/help`, `/think`,
  `/think-max`, `/nothink`, `/ctx 128`, `/read`, an unknown command, one direct
  model-backed prompt, and `/quit`; the Ctrl+C case covers deterministic
  Ctrl+C behavior at the prompt.
- M8.14 committed current-C artifact size is 34,582 bytes with SHA256
  `223939b68a2791d79b2b7bac207e1e2a89db71f3073d0e4ab885a34e08c65a9f`.
  The manifest size is 2,472 bytes with SHA256
  `d63181e48f0c381308401692c60672a3d6bf1dfe5e142036df728ee100cd40f4`.
- M8.14 B300 validation passed after copying the checker and fixture to
  `/workspace/ds4`, rebuilding `ds4` with `make ds4 CUDA_ARCH=native`, running
  the PTY capture/checker, and copying `current-c.json` and `manifest.json`
  back from the pod.
- M8.14 B300 and local revalidation passed
  `python3 ds4-parity/check_cli_interactive_dump.py
  ds4-parity/baselines/cli/m8.14/current-c.json --manifest
  ds4-parity/baselines/cli/m8.14/manifest.json --negative-test` with oracle
  `PASS, 89 checks`, manifest `PASS, 15 checks`, and negative tests `PASS, 6
  checks`.
- M8.14 local validation also passed `python3 -m json.tool` for both M8.14 JSON
  files, `python3 -m py_compile ds4-parity/check_cli_interactive_dump.py`, and
  `git diff --check`.
- M8.15 was split before implementation because the current Rust runtime has
  one-shot generation but not reusable sessions, chat transcript mutation,
  session progress callbacks, or REPL command state.
- M8.15 source inspection found the needed current-C APIs in `ds4.h`:
  `ds4_chat_begin`, `ds4_chat_append_message`,
  `ds4_chat_append_assistant_prefix`, `ds4_chat_append_max_effort_prefix`,
  `ds4_tokens_push`, reusable `ds4_session_*` APIs,
  `ds4_session_set_progress`, `ds4_session_common_prefix`,
  `ds4_session_invalidate`, `ds4_session_pos`, and `ds4_session_ctx`.
- M8.15 source inspection found that `rust/ds4-engine/src/lib.rs` currently
  exposes one-shot prompt encoding plus argmax/session generation helpers, but
  `Session` and `TokenPrinter` are private and there is no public reusable
  chat transcript/session boundary for interactive turns.
- M8.15 has been split in `RUST_PORT_ROADMAP.md` into M8.15a reusable
  interactive session boundary, M8.15b REPL command-state surface, and M8.15c
  final interactive PTY transcript surface.
- M8.15a extends `rust/ds4-engine/src/lib.rs` with a borrowed `ChatSession`
  wrapper over the current C chat/session APIs, including chat transcript
  creation, user/assistant append, reusable session creation/reset,
  session-sync progress callbacks, token append/eos, session position/context
  handling, and two-turn generation.
- M8.15a adds `ds4-interactive-runtime-rs`, a narrow non-PTY runtime-boundary
  binary that simulates the M8.14 model-backed `/read` turn followed by the
  direct prompt `Answer with one short noun: glacier.` and emits explicit
  `read`/`direct` turn blocks.
- M8.15a adds `ds4-parity/compare_cli_interactive_runtime.py`, which extracts
  the M8.14 generated turn bytes from the committed PTY transcript and compares
  them against `target/debug/ds4-interactive-runtime-rs` while also checking
  runtime stderr anchors and forbidden unsupported paths.
- M8.15a B300 validation over `/workspace/ds4/ds4flash.gguf` passed after
  copying the changed files to `/workspace/ds4`, verifying SHA256 matches,
  building with `CARGO_HOME=/tmp/ds4-cargo RUSTUP_HOME=/tmp/ds4-rustup
  PATH=/tmp/ds4-cargo/bin:$PATH CUDA_ARCH=native cargo build -p ds4-engine
  --bin ds4-interactive-runtime-rs`, and running
  `python3 ds4-parity/compare_cli_interactive_runtime.py
  ds4-parity/baselines/cli/m8.14/current-c.json --candidate-binary
  target/debug/ds4-interactive-runtime-rs --negative-test`.
- M8.15a B300 comparator reported `CLI interactive runtime comparator: PASS, 19
  checks` and `CLI interactive runtime negative tests: PASS, 4 checks`.
- M8.15a local validation passed for `cargo fmt --all -- --check`, `cargo build
  -p ds4-engine --bin ds4-interactive-runtime-rs`, `cargo test -p
  ds4-engine`, full `cargo test --workspace`, `python3 -m py_compile
  ds4-parity/compare_cli_interactive_runtime.py
  ds4-parity/check_cli_interactive_dump.py`, and `git diff --check`.
- M8.15b adds `rust/ds4-engine/src/interactive_cli.rs`, a model-free REPL
  command-state surface covering empty input, `/help`, `/think`, `/think-max`,
  `/nothink`, `/ctx`, `/read`, unknown slash commands, `/quit`, `/exit`,
  normal prompt dispatch, and Ctrl+C-at-prompt recovery.
- M8.15b exports the REPL command-state module from `ds4-engine` while leaving
  PTY line editing and model-backed turn execution for M8.15c/M8.15a.
- M8.15b command matching uses command-boundary checks so short or extended
  slash commands such as `/c`, `/ctxx`, `/rea`, and `/readx` route to the C
  CLI unknown-command category instead of matching or panicking.
- M8.15b local validation passed for `cargo fmt --all -- --check`, `cargo test
  -p ds4-engine interactive_cli -- --nocapture` (5 REPL command tests), `cargo
  test -p ds4-engine` (11 tests), full `cargo test --workspace`, and `git diff
  --check`.
- M8.15c adds `ds4-cli-interactive-rs`, a no-prompt Rust REPL binary that uses
  the shared CLI parser, M8.15b `ReplState`, and M8.15a `ChatSession` to cover
  the M8.14 `command_suite` and `ctrl_c_at_prompt` PTY cases.
- M8.15c adds `ChatSession::run_turn_to_writer` so PTY generation writes model
  bytes before the timing line, matching the current C merged stdout/stderr
  transcript order while preserving the existing buffered `run_turn` API.
- M8.15c adds `ds4-parity/compare_cli_interactive_pty.py`, which reuses the
  M8.14 PTY driver and explicitly normalizes linenoise redraw-only frames while
  comparing committed prompts, command responses, generated bytes, timing and
  progress categories, exit status, normalized transcript hashes, and
  Ctrl+C-at-prompt recovery.
- M8.15c B300 validation over `/workspace/ds4/ds4flash.gguf` passed after
  copying the changed files to `/workspace/ds4`, building with `CARGO_HOME=/tmp/ds4-cargo
  RUSTUP_HOME=/tmp/ds4-rustup PATH=/tmp/ds4-cargo/bin:$PATH CUDA_ARCH=native
  cargo build -p ds4-engine --bin ds4-cli-interactive-rs`, and running
  `python3 ds4-parity/compare_cli_interactive_pty.py
  ds4-parity/baselines/cli/m8.14/current-c.json --candidate-binary
  target/debug/ds4-cli-interactive-rs --write-candidate
  /tmp/ds4-m8.15c-rust-pty.json --negative-test`.
- M8.15c B300 comparator reported `CLI interactive PTY comparator: PASS, 59
  checks` and `CLI interactive PTY negative tests: PASS, 4 checks`.
- M8.15c local validation passed for `cargo fmt --all -- --check`, `cargo build
  -p ds4-engine --bin ds4-cli-interactive-rs`, `cargo test -p ds4-engine`, full
  `cargo test --workspace`, `python3 -m py_compile
  ds4-parity/compare_cli_interactive_pty.py
  ds4-parity/check_cli_interactive_dump.py`, and `git diff --check`.
- M8.16 adds `ds4-parity/run_cli_parity_report.py`, which executes local M8 CLI
  artifact validators/comparators and records model-backed B300 current-C
  refreshes plus Rust runtime/PTY checks as skipped items with exact rerun
  commands.
- M8.16 wires the CLI report into `ds4-parity/run_parity_report.py` as the
  `M8.16 CLI parity report` comparator item.
- M8.16 local CLI report validation passed with `summary: 9 passed, 13 skipped,
  0 failed`; JSON output from `python3 ds4-parity/run_cli_parity_report.py
  --json` parsed with `python3 -m json.tool`.
- M8.16 unified report validation passed with `summary: 14 passed, 5 skipped,
  0 failed`, including the nested `M8.16 CLI parity report`.
- M8.16 local validation also passed `python3 -m py_compile
  ds4-parity/run_cli_parity_report.py ds4-parity/run_parity_report.py`, full
  `cargo test --workspace`, and `git diff --check`.
- M9.1 split Milestone 9 into concrete server parity items in
  `RUST_PORT_ROADMAP.md` and `.memory/TODO.md`: request parse/render, HTTP
  skeleton and model metadata, non-streaming chat runtime, streaming SSE,
  OpenAI tool/DSML server surface, Responses/Anthropic protocols, cache/KV/tool
  memory, and server report integration.
- M9.1 source inspection found that `ds4_server.c` server scope includes
  OpenAI chat, Responses, Anthropic, streaming deltas, CORS/preflight behavior,
  thinking controls, stop lists, DSML/tool parsing, live tool-tail validation,
  usage accounting, cache/KV restore, tool-memory replay, and eviction policy.
- M9.1 baseline inspection confirmed M0.4 covers `models`, non-streaming chat,
  streaming chat, tool calls, thinking-disabled chat, and memory-token cache
  continuation, while M0.5 covers disk-KV seed miss/restore and continuation
  restore with KV headers, rendered text, traces, and cache decisions.
- M9.1 validation passed with source/fixture inspection, roadmap/board diff, and
  `git diff --check`.
- M9.2 was split before implementation because the full request parse/render
  surface spans OpenAI core chat, OpenAI tool/DSML prompt rendering, and
  Responses/Anthropic protocol inputs.
- M9.2a is the next executable item and is intentionally limited to model-free
  OpenAI `/v1/chat/completions` core fields and prompt rendering, excluding
  tool-call payloads and alternate protocols.
- M9.2b covers OpenAI tool schema parsing and DSML prompt rendering, while M9.2c
  covers Responses and Anthropic request parsing/rendering inputs; later M9.6
  and M9.7 remain responsible for model-backed tool/protocol response behavior.
- M9.2 split validation passed with roadmap/board diff and `git diff --check`.
