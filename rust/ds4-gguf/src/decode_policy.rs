const DS4_TOOL_CALLS_START: &str = "<｜DSML｜tool_calls>";
const DS4_TOOL_CALLS_END: &str = "</｜DSML｜tool_calls>";
const DS4_INVOKE_START: &str = "<｜DSML｜invoke";
const DS4_INVOKE_END: &str = "</｜DSML｜invoke>";
const DS4_PARAM_START: &str = "<｜DSML｜parameter";
const DS4_PARAM_END: &str = "</｜DSML｜parameter>";
const DS4_TOOL_CALLS_START_SHORT: &str = "<DSML｜tool_calls>";
const DS4_TOOL_CALLS_END_SHORT: &str = "</DSML｜tool_calls>";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicySurface {
    Cli,
    Server,
    Agent,
}

impl PolicySurface {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cli => "cli",
            Self::Server => "server",
            Self::Agent => "agent",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiStyle {
    OpenAi,
    Anthropic,
    Responses,
}

impl ApiStyle {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::OpenAi => "openai",
            Self::Anthropic => "anthropic",
            Self::Responses => "responses",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyKind {
    Chat,
    Completion,
}

impl PolicyKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::Completion => "completion",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyRequest {
    pub surface: PolicySurface,
    pub api: ApiStyle,
    pub kind: PolicyKind,
    pub stream: bool,
    pub has_tools: bool,
    pub max_tokens: usize,
    pub stops: Vec<&'static str>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyPiece {
    pub text: Vec<u8>,
    pub eos: bool,
}

impl PolicyPiece {
    fn text(text: impl Into<Vec<u8>>) -> Self {
        Self {
            text: text.into(),
            eos: false,
        }
    }

    fn eos() -> Self {
        Self {
            text: Vec::new(),
            eos: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyCase {
    pub name: &'static str,
    pub source: &'static str,
    pub request: PolicyRequest,
    pub schedule: Vec<PolicyPiece>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StopBoundary {
    pub pos: isize,
    pub len: usize,
}

impl Default for StopBoundary {
    fn default() -> Self {
        Self { pos: -1, len: 0 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ToolBoundary {
    pub saw_start: bool,
    pub saw_end: bool,
    pub tool_call_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApiFinish {
    pub openai_finish_reason: Option<&'static str>,
    pub anthropic_stop_reason: Option<&'static str>,
    pub responses_status: Option<&'static str>,
    pub responses_item_status: Option<&'static str>,
    pub responses_incomplete_reason: Option<&'static str>,
}

impl ApiFinish {
    fn null() -> Self {
        Self {
            openai_finish_reason: None,
            anthropic_stop_reason: None,
            responses_status: None,
            responses_item_status: None,
            responses_incomplete_reason: None,
        }
    }

    fn server(finish: &'static str) -> Self {
        Self {
            openai_finish_reason: Some(finish),
            anthropic_stop_reason: Some(anthropic_stop_reason(finish)),
            responses_status: Some(responses_status_for_finish(finish)),
            responses_item_status: Some(responses_item_status_for_finish(finish)),
            responses_incomplete_reason: responses_incomplete_reason(finish),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamStep {
    pub step: usize,
    pub text_len: usize,
    pub stream_safe_len: usize,
    pub delta: Vec<u8>,
    pub held_tail: Vec<u8>,
    pub hit_stop: bool,
    pub stop_pos: isize,
    pub stop_len: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyResult {
    pub finish_reason: &'static str,
    pub completion_tokens: usize,
    pub raw_text: Vec<u8>,
    pub visible_text: Vec<u8>,
    pub reasoning: Vec<u8>,
    pub streamed_text: Vec<u8>,
    pub session_invalidation_required: bool,
    pub transcript_eos_appended: bool,
    pub stop_boundary: StopBoundary,
    pub tool_boundary: ToolBoundary,
    pub api_finish: ApiFinish,
    pub stream_steps: Vec<StreamStep>,
}

pub fn policy_cases() -> Vec<PolicyCase> {
    let tool_schedule = tool_call_schedule_piece();
    vec![
        PolicyCase {
            name: "cli_eos_stop",
            source: "ds4_cli.c:run_sampled_generation EOS",
            request: request(
                PolicySurface::Cli,
                ApiStyle::OpenAi,
                PolicyKind::Completion,
                4,
            ),
            schedule: vec![PolicyPiece::text("cli hello"), PolicyPiece::eos()],
        },
        PolicyCase {
            name: "cli_max_tokens_length",
            source: "ds4_cli.c:run_sampled_generation max_tokens",
            request: request(
                PolicySurface::Cli,
                ApiStyle::OpenAi,
                PolicyKind::Completion,
                2,
            ),
            schedule: vec![
                PolicyPiece::text("a"),
                PolicyPiece::text("b"),
                PolicyPiece::text("c"),
            ],
        },
        PolicyCase {
            name: "server_openai_eos_stop",
            source: "ds4_server.c:generate_job EOS",
            request: request(PolicySurface::Server, ApiStyle::OpenAi, PolicyKind::Chat, 4),
            schedule: vec![PolicyPiece::text("server hello"), PolicyPiece::eos()],
        },
        PolicyCase {
            name: "server_openai_max_tokens_length",
            source: "ds4_server.c:generate_job max_tokens",
            request: request(PolicySurface::Server, ApiStyle::OpenAi, PolicyKind::Chat, 2),
            schedule: vec![
                PolicyPiece::text("one"),
                PolicyPiece::text(" two"),
                PolicyPiece::text(" three"),
            ],
        },
        PolicyCase {
            name: "server_openai_user_stop_sequence",
            source: "ds4_server.c:generate_job stop_list_find_from",
            request: request_with_stops(
                PolicySurface::Server,
                ApiStyle::OpenAi,
                PolicyKind::Chat,
                4,
                vec!["STOP"],
            ),
            schedule: vec![
                PolicyPiece::text("answer ST"),
                PolicyPiece::text("OP hidden"),
            ],
        },
        PolicyCase {
            name: "server_openai_stream_holds_stop_tail",
            source: "ds4_server.c:stop_list_stream_safe_len",
            request: request_with_stream_stops(ApiStyle::OpenAi, 4, vec!["</END>"]),
            schedule: vec![PolicyPiece::text("hello </"), PolicyPiece::eos()],
        },
        PolicyCase {
            name: "server_openai_stream_stop_hit_discards_tail",
            source: "ds4_server.c:generate_job streaming stop hit",
            request: request_with_stream_stops(ApiStyle::OpenAi, 4, vec!["STOP"]),
            schedule: vec![PolicyPiece::text("pre ST"), PolicyPiece::text("OP after")],
        },
        PolicyCase {
            name: "server_openai_stream_holds_partial_utf8",
            source: "ds4_server.c:utf8_stream_safe_len",
            request: request_with_stream(ApiStyle::OpenAi, 4),
            schedule: vec![
                PolicyPiece::text(vec![0xe2, 0x82]),
                PolicyPiece::text(vec![0xac, b' ', b'o', b'k']),
                PolicyPiece::eos(),
            ],
        },
        PolicyCase {
            name: "server_openai_stop_mid_utf8_boundary",
            source: "ds4_server.c:utf8_stream_safe_len hit_stop",
            request: request_with_stream_stops(ApiStyle::OpenAi, 4, vec!["STOP"]),
            schedule: vec![
                PolicyPiece::text(vec![0xe2, b'S', b'T']),
                PolicyPiece::text("OP tail"),
            ],
        },
        PolicyCase {
            name: "server_openai_tool_call_boundary",
            source: "ds4_server.c:observe_tool_markers tool_calls",
            request: request_with_tools(ApiStyle::OpenAi, 8),
            schedule: vec![
                PolicyPiece::text("I will call.\n\n"),
                PolicyPiece::text(tool_schedule.clone()),
            ],
        },
        PolicyCase {
            name: "server_responses_length_mapping",
            source: "ds4_server.c:responses_status_for_finish length",
            request: request(
                PolicySurface::Server,
                ApiStyle::Responses,
                PolicyKind::Chat,
                2,
            ),
            schedule: vec![
                PolicyPiece::text("one"),
                PolicyPiece::text(" two"),
                PolicyPiece::text(" three"),
            ],
        },
        PolicyCase {
            name: "server_anthropic_tool_mapping",
            source: "ds4_server.c:anthropic_stop_reason tool_calls",
            request: request_with_tools(ApiStyle::Anthropic, 8),
            schedule: vec![
                PolicyPiece::text("I will call.\n\n"),
                PolicyPiece::text(tool_schedule),
            ],
        },
        PolicyCase {
            name: "agent_eos_stop",
            source: "ds4_agent.c:agent_worker_run EOS",
            request: request(PolicySurface::Agent, ApiStyle::OpenAi, PolicyKind::Chat, 4),
            schedule: vec![PolicyPiece::text("agent hello"), PolicyPiece::eos()],
        },
        PolicyCase {
            name: "agent_max_tokens_length",
            source: "ds4_agent.c:agent_worker_run max_tokens",
            request: request(PolicySurface::Agent, ApiStyle::OpenAi, PolicyKind::Chat, 2),
            schedule: vec![
                PolicyPiece::text("x"),
                PolicyPiece::text("y"),
                PolicyPiece::text("z"),
            ],
        },
    ]
}

pub fn run_policy_case(case: &PolicyCase) -> PolicyResult {
    match case.request.surface {
        PolicySurface::Server => run_server_case(case),
        PolicySurface::Cli | PolicySurface::Agent => run_simple_case(case),
    }
}

pub fn find_stop_from(stops: &[&str], text: &[u8], from: usize) -> Option<(usize, usize)> {
    if stops.is_empty() || from > text.len() {
        return None;
    }
    let mut best: Option<(usize, usize)> = None;
    for stop in stops {
        let needle = stop.as_bytes();
        if needle.is_empty() {
            continue;
        }
        if let Some(rel) = find_bytes(&text[from..], needle) {
            let pos = from + rel;
            if best.is_none_or(|(best_pos, _)| pos < best_pos) {
                best = Some((pos, needle.len()));
            }
        }
    }
    best
}

pub fn stop_list_stream_safe_len(stops: &[&str], text_len: usize) -> usize {
    let max_len = stops.iter().map(|stop| stop.len()).max().unwrap_or(0);
    if stops.is_empty() || max_len <= 1 {
        text_len
    } else {
        text_len.saturating_sub(max_len - 1)
    }
}

pub fn utf8_stream_safe_len(bytes: &[u8], start: usize, limit: usize, final_chunk: bool) -> usize {
    let limit = limit.min(bytes.len());
    let start = start.min(limit);
    if final_chunk || limit <= start {
        return limit;
    }

    let mut p = limit;
    let mut continuation = 0;
    while p > start && continuation < 4 && (bytes[p - 1] & 0xc0) == 0x80 {
        p -= 1;
        continuation += 1;
    }

    if p == limit {
        return if utf8_expected_len(bytes[limit - 1]) > 1 {
            limit - 1
        } else {
            limit
        };
    }
    if p == start && (bytes[p] & 0xc0) == 0x80 {
        return start;
    }

    let lead = p - 1;
    let need = utf8_expected_len(bytes[lead]);
    if limit - lead < need {
        lead
    } else {
        limit
    }
}

fn run_simple_case(case: &PolicyCase) -> PolicyResult {
    let mut raw_text = Vec::new();
    let mut visible_text = Vec::new();
    let mut stream_steps = Vec::new();
    let mut finish_reason = "length";
    let mut generated = 0;

    for (idx, piece) in case.schedule.iter().enumerate() {
        if generated >= case.request.max_tokens {
            break;
        }
        if piece.eos {
            finish_reason = "stop";
            break;
        }
        raw_text.extend_from_slice(&piece.text);
        visible_text.extend_from_slice(&piece.text);
        generated += 1;
        stream_steps.push(StreamStep {
            step: idx,
            text_len: raw_text.len(),
            stream_safe_len: raw_text.len(),
            delta: piece.text.clone(),
            held_tail: Vec::new(),
            hit_stop: false,
            stop_pos: -1,
            stop_len: 0,
        });
    }

    PolicyResult {
        finish_reason,
        completion_tokens: generated,
        raw_text,
        visible_text,
        reasoning: Vec::new(),
        streamed_text: Vec::new(),
        session_invalidation_required: false,
        transcript_eos_appended: case.request.surface == PolicySurface::Agent,
        stop_boundary: StopBoundary::default(),
        tool_boundary: ToolBoundary::default(),
        api_finish: ApiFinish::null(),
        stream_steps,
    }
}

fn run_server_case(case: &PolicyCase) -> PolicyResult {
    let request = &case.request;
    let mut raw_text = Vec::new();
    let mut streamed_text = Vec::new();
    let mut stream_steps = Vec::new();
    let mut finish_reason = "length";
    let mut completion_tokens = 0;
    let mut plain_stream_pos = 0;
    let mut stop_scan_from = 0;
    let mut tool_scan_from = 0;
    let mut tool_boundary = ToolBoundary::default();
    let mut stop_boundary = StopBoundary::default();
    let mut session_invalidation_required = false;

    for (idx, piece) in case.schedule.iter().enumerate() {
        if completion_tokens >= request.max_tokens {
            break;
        }
        if piece.eos {
            finish_reason = "stop";
            break;
        }

        raw_text.extend_from_slice(&piece.text);
        completion_tokens += 1;

        if request.kind == PolicyKind::Chat && request.has_tools {
            if tool_scan_from > raw_text.len() {
                tool_scan_from = raw_text.len();
            }
            observe_tool_markers(&raw_text[tool_scan_from..], &mut tool_boundary);
            let hold_from = raw_text.len().saturating_sub(80);
            if hold_from > tool_scan_from {
                tool_scan_from = hold_from;
            }
        }

        let hit = find_stop_from(&request.stops, &raw_text, stop_scan_from);
        let (hit_stop, stop_pos, stop_len) = if let Some((pos, len)) = hit {
            (true, pos, len)
        } else {
            (false, 0, 0)
        };
        let mut stream_len = if hit_stop {
            stop_pos
        } else {
            stop_list_stream_safe_len(&request.stops, raw_text.len())
        };
        stream_len = stream_len.min(raw_text.len());
        stream_len = utf8_stream_safe_len(&raw_text, plain_stream_pos, stream_len, hit_stop);

        if !hit_stop {
            let max_stop_len = request
                .stops
                .iter()
                .map(|stop| stop.len())
                .max()
                .unwrap_or(0);
            if max_stop_len > 1 {
                let hold = max_stop_len - 1;
                stop_scan_from = raw_text.len().saturating_sub(hold);
            }
        }

        let mut delta = Vec::new();
        if request.stream && stream_len > plain_stream_pos {
            delta.extend_from_slice(&raw_text[plain_stream_pos..stream_len]);
            streamed_text.extend_from_slice(&delta);
            plain_stream_pos = stream_len;
        }

        let held_tail = if request.stream && !hit_stop && raw_text.len() >= plain_stream_pos {
            raw_text[plain_stream_pos..].to_vec()
        } else {
            Vec::new()
        };

        stream_steps.push(StreamStep {
            step: idx,
            text_len: raw_text.len(),
            stream_safe_len: stream_len,
            delta,
            held_tail,
            hit_stop,
            stop_pos: if hit_stop { stop_pos as isize } else { -1 },
            stop_len,
        });

        if hit_stop {
            finish_reason = "stop";
            raw_text.truncate(stop_pos);
            session_invalidation_required = true;
            stop_boundary = StopBoundary {
                pos: stop_pos as isize,
                len: stop_len,
            };
            break;
        }

        if request.kind == PolicyKind::Chat && request.has_tools && tool_boundary.saw_end {
            finish_reason = "tool_calls";
            break;
        }
    }

    if request.stream && raw_text.len() > plain_stream_pos {
        streamed_text.extend_from_slice(&raw_text[plain_stream_pos..]);
    }

    let (visible_text, reasoning, tool_call_count) = parse_response_boundary(
        &raw_text,
        request.has_tools,
        tool_boundary.saw_start,
        tool_boundary.saw_end,
    );
    tool_boundary.tool_call_count = tool_call_count;
    if tool_call_count > 0 {
        finish_reason = "tool_calls";
    }

    PolicyResult {
        finish_reason,
        completion_tokens,
        raw_text,
        visible_text,
        reasoning,
        streamed_text,
        session_invalidation_required,
        transcript_eos_appended: false,
        stop_boundary,
        tool_boundary,
        api_finish: ApiFinish::server(finish_reason),
        stream_steps,
    }
}

fn parse_response_boundary(
    raw: &[u8],
    has_tools: bool,
    saw_tool_start: bool,
    saw_tool_end: bool,
) -> (Vec<u8>, Vec<u8>, usize) {
    if has_tools && saw_tool_start {
        if let Some(start) = find_any_tool_start(raw) {
            let visible_end = trim_ascii_ws_end(raw, 0, start);
            let complete = saw_tool_end && find_any_tool_end(&raw[start..]).is_some();
            return (
                raw[..visible_end].to_vec(),
                Vec::new(),
                usize::from(complete),
            );
        }
    }
    (raw.to_vec(), Vec::new(), 0)
}

fn observe_tool_markers(scan: &[u8], boundary: &mut ToolBoundary) {
    let had_start = boundary.saw_start;
    let start = find_any_tool_start(scan);
    if start.is_some() {
        boundary.saw_start = true;
    }

    let end_scan = if had_start {
        Some(scan)
    } else {
        start.map(|pos| &scan[pos..])
    };
    if end_scan.and_then(find_any_tool_end).is_some() {
        boundary.saw_end = true;
    }
}

fn find_any_tool_start(bytes: &[u8]) -> Option<usize> {
    find_earliest(
        bytes,
        &[
            DS4_TOOL_CALLS_START.as_bytes(),
            DS4_TOOL_CALLS_START_SHORT.as_bytes(),
            b"<tool_calls>",
        ],
    )
}

fn find_any_tool_end(bytes: &[u8]) -> Option<usize> {
    find_earliest(
        bytes,
        &[
            DS4_TOOL_CALLS_END.as_bytes(),
            DS4_TOOL_CALLS_END_SHORT.as_bytes(),
            b"</tool_calls>",
        ],
    )
}

fn find_earliest(bytes: &[u8], needles: &[&[u8]]) -> Option<usize> {
    let mut best = None;
    for needle in needles {
        if let Some(pos) = find_bytes(bytes, needle) {
            if best.is_none_or(|best_pos| pos < best_pos) {
                best = Some(pos);
            }
        }
    }
    best
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn trim_ascii_ws_end(raw: &[u8], start: usize, mut limit: usize) -> usize {
    while limit > start && raw[limit - 1].is_ascii_whitespace() {
        limit -= 1;
    }
    limit
}

fn utf8_expected_len(byte: u8) -> usize {
    match byte {
        0x00..=0x7f => 1,
        0xc2..=0xdf => 2,
        0xe0..=0xef => 3,
        0xf0..=0xf4 => 4,
        _ => 1,
    }
}

fn request(
    surface: PolicySurface,
    api: ApiStyle,
    kind: PolicyKind,
    max_tokens: usize,
) -> PolicyRequest {
    request_with_stops(surface, api, kind, max_tokens, Vec::new())
}

fn request_with_stops(
    surface: PolicySurface,
    api: ApiStyle,
    kind: PolicyKind,
    max_tokens: usize,
    stops: Vec<&'static str>,
) -> PolicyRequest {
    PolicyRequest {
        surface,
        api,
        kind,
        stream: false,
        has_tools: false,
        max_tokens,
        stops,
    }
}

fn request_with_stream(api: ApiStyle, max_tokens: usize) -> PolicyRequest {
    PolicyRequest {
        stream: true,
        ..request(PolicySurface::Server, api, PolicyKind::Chat, max_tokens)
    }
}

fn request_with_stream_stops(
    api: ApiStyle,
    max_tokens: usize,
    stops: Vec<&'static str>,
) -> PolicyRequest {
    PolicyRequest {
        stream: true,
        ..request_with_stops(
            PolicySurface::Server,
            api,
            PolicyKind::Chat,
            max_tokens,
            stops,
        )
    }
}

fn request_with_tools(api: ApiStyle, max_tokens: usize) -> PolicyRequest {
    PolicyRequest {
        has_tools: true,
        ..request(PolicySurface::Server, api, PolicyKind::Chat, max_tokens)
    }
}

fn tool_call_schedule_piece() -> Vec<u8> {
    let text = format!(
        "{DS4_TOOL_CALLS_START}\n\
         {DS4_INVOKE_START} name=\"bash\">\n\
         {DS4_PARAM_START} name=\"command\" string=\"true\">echo hi{DS4_PARAM_END}\n\
         {DS4_INVOKE_END}\n\
         {DS4_TOOL_CALLS_END}"
    );
    text.into_bytes()
}

fn anthropic_stop_reason(finish: &'static str) -> &'static str {
    match finish {
        "tool_calls" => "tool_use",
        "length" => "max_tokens",
        _ => "end_turn",
    }
}

fn responses_status_for_finish(finish: &'static str) -> &'static str {
    match finish {
        "length" => "incomplete",
        "error" => "failed",
        _ => "completed",
    }
}

fn responses_item_status_for_finish(finish: &'static str) -> &'static str {
    match finish {
        "length" | "error" => "incomplete",
        _ => "completed",
    }
}

fn responses_incomplete_reason(finish: &'static str) -> Option<&'static str> {
    (finish == "length").then_some("max_tokens")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(name: &str) -> PolicyResult {
        let case = policy_cases()
            .into_iter()
            .find(|case| case.name == name)
            .expect("case exists");
        run_policy_case(&case)
    }

    #[test]
    fn stop_finder_matches_c_tie_and_scan_semantics() {
        assert_eq!(
            find_stop_from(&["END", "STOP"], b"aa STOP END", 0),
            Some((3, 4))
        );
        assert_eq!(find_stop_from(&["STOP", "ST"], b"aa STOP", 0), Some((3, 4)));
        assert_eq!(find_stop_from(&["STOP"], b"aa STOP", 4), None);
        assert_eq!(stop_list_stream_safe_len(&["</END>"], 8), 3);
    }

    #[test]
    fn utf8_stream_safe_len_holds_incomplete_tail() {
        assert_eq!(utf8_stream_safe_len(&[0xe2, 0x82], 0, 2, false), 0);
        assert_eq!(utf8_stream_safe_len(&[0xe2, 0x82, 0xac], 0, 3, false), 3);
        assert_eq!(utf8_stream_safe_len(&[0xe2], 0, 1, true), 1);
    }

    #[test]
    fn streaming_stop_hit_discards_held_tail() {
        let got = result("server_openai_stream_stop_hit_discards_tail");
        assert_eq!(got.finish_reason, "stop");
        assert_eq!(got.visible_text, b"pre ");
        assert_eq!(got.streamed_text, b"pre ");
        assert!(got.session_invalidation_required);
        assert_eq!(got.stop_boundary, StopBoundary { pos: 4, len: 4 });
        assert_eq!(got.stream_steps[0].held_tail, b" ST");
        assert!(got.stream_steps[1].hit_stop);
        assert!(got.stream_steps[1].held_tail.is_empty());
    }

    #[test]
    fn mid_utf8_stop_policy_stays_byte_oriented() {
        let got = result("server_openai_stop_mid_utf8_boundary");
        assert_eq!(got.visible_text, vec![0xe2]);
        assert_eq!(got.streamed_text, vec![0xe2]);
        assert_eq!(got.stop_boundary, StopBoundary { pos: 1, len: 4 });
    }

    #[test]
    fn complete_tool_boundary_trims_visible_prefix_without_runtime_parser() {
        let got = result("server_openai_tool_call_boundary");
        assert_eq!(got.finish_reason, "tool_calls");
        assert_eq!(got.visible_text, b"I will call.");
        assert_eq!(
            got.tool_boundary,
            ToolBoundary {
                saw_start: true,
                saw_end: true,
                tool_call_count: 1,
            }
        );
        assert_eq!(got.api_finish.openai_finish_reason, Some("tool_calls"));
    }
}
