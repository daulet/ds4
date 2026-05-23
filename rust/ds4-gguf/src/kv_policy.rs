use std::fmt;

pub const FIXED_HEADER: usize = 48;
pub const DEFAULT_MB: u64 = 4096;
pub const HIT_HALF_LIFE_SECONDS: u64 = 6 * 60 * 60;
pub const MIN_EFFECTIVE_HITS: f64 = 0.01;

pub const EXT_TOOL_MAP: u8 = 1 << 0;
pub const EXT_RESPONSES_VISIBLE: u8 = 1 << 1;
pub const EXT_THINKING_VISIBLE: u8 = 1 << 2;
pub const TOOL_MAP_HEADER: usize = 8;
pub const TOOL_MAP_VERSION: u8 = 1;
pub const TOOL_MAP_MAX_ID_LEN: usize = 256;
pub const TOOL_MAP_DEFAULT_MAX_ENTRIES: usize = 100_000;
pub const TOOL_MAP_MAX_DSML_LEN: usize = 512 * 1024 * 1024;

pub const REASON_UNKNOWN: u8 = 0;
pub const REASON_COLD: u8 = 1;
pub const REASON_CONTINUED: u8 = 2;
pub const REASON_EVICT: u8 = 3;
pub const REASON_SHUTDOWN: u8 = 4;
pub const REASON_AGENT_SYSTEM: u8 = 5;
pub const REASON_AGENT_SESSION: u8 = 6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvOptions {
    pub min_tokens: i32,
    pub cold_max_tokens: i32,
    pub continued_interval_tokens: i32,
    pub boundary_trim_tokens: i32,
    pub boundary_align_tokens: i32,
}

impl Default for KvOptions {
    fn default() -> Self {
        Self {
            min_tokens: 512,
            cold_max_tokens: 30_000,
            continued_interval_tokens: 10_000,
            boundary_trim_tokens: 32,
            boundary_align_tokens: 2048,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvPolicyConfig {
    pub enabled: bool,
    pub budget_bytes: u64,
    pub reject_different_quant: bool,
    pub options: KvOptions,
    pub continued_last_store_tokens: i32,
}

impl Default for KvPolicyConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            budget_bytes: 0,
            reject_different_quant: false,
            options: KvOptions::default(),
            continued_last_store_tokens: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvHeader {
    pub quant_bits: u8,
    pub reason: u8,
    pub ext_flags: u8,
    pub tokens: u32,
    pub hits: u32,
    pub ctx_size: u32,
    pub created_at: u64,
    pub last_used: u64,
    pub payload_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedHeader {
    pub header: KvHeader,
    pub text_bytes: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvHeaderError {
    Truncated,
    BadMagic,
    BadVersion,
    EmptyTokens,
    BadQuant,
}

impl fmt::Display for KvHeaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => f.write_str("truncated header"),
            Self::BadMagic => f.write_str("bad KVC magic"),
            Self::BadVersion => f.write_str("bad KVC version"),
            Self::EmptyTokens => f.write_str("empty token count"),
            Self::BadQuant => f.write_str("bad quant bits"),
        }
    }
}

impl std::error::Error for KvHeaderError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvEntry {
    pub sha: String,
    pub quant_bits: u8,
    pub reason: u8,
    pub ext_flags: u8,
    pub tokens: u32,
    pub hits: u32,
    pub ctx_size: u32,
    pub created_at: u64,
    pub last_used: u64,
    pub payload_bytes: u64,
    pub text_bytes: u64,
    pub file_size: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileSizeDecision {
    pub fits: bool,
    pub file_bytes: u64,
    pub required_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheReplayDecision {
    pub cache_source: &'static str,
    pub cached_tokens: u32,
    pub cache_write_tokens: u32,
    pub disk_cached_tokens: u32,
    pub memory_token_reusable: bool,
    pub memory_miss_reason: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvcFile {
    pub header: KvHeader,
    pub text: Vec<u8>,
    pub payload: Vec<u8>,
    pub trailer: Vec<u8>,
    pub file_size: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvcFileError {
    Header(KvHeaderError),
    TextTooLarge,
    PayloadLengthMismatch,
    SizeOverflow,
    PayloadTooLarge,
    TruncatedText,
    TruncatedPayload,
}

impl fmt::Display for KvcFileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Header(err) => write!(f, "{err}"),
            Self::TextTooLarge => f.write_str("KVC text too large"),
            Self::PayloadLengthMismatch => f.write_str("KVC payload length mismatch"),
            Self::SizeOverflow => f.write_str("KVC file size overflow"),
            Self::PayloadTooLarge => f.write_str("KVC payload too large"),
            Self::TruncatedText => f.write_str("truncated KVC text"),
            Self::TruncatedPayload => f.write_str("truncated KVC payload"),
        }
    }
}

impl std::error::Error for KvcFileError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolMapEntry {
    pub id: String,
    pub dsml: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolMapDecode {
    pub entries: Vec<ToolMapEntry>,
    pub declared_count: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolMapError {
    ShortHeader,
    BadHeader,
    CountLimit,
    ShortEntryHeader,
    BadIdLen,
    BadDsmlLen,
    TruncatedId,
    TruncatedDsml,
}

impl ToolMapError {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ShortHeader => "short-header",
            Self::BadHeader => "bad-header",
            Self::CountLimit => "count-limit",
            Self::ShortEntryHeader => "short-entry-header",
            Self::BadIdLen => "bad-id-len",
            Self::BadDsmlLen => "bad-dsml-len",
            Self::TruncatedId => "truncated-id",
            Self::TruncatedDsml => "truncated-dsml",
        }
    }
}

impl fmt::Display for ToolMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::error::Error for ToolMapError {}

pub fn reason_code(reason: Option<&str>) -> u8 {
    match reason {
        Some("cold") => REASON_COLD,
        Some("continued") => REASON_CONTINUED,
        Some("evict") => REASON_EVICT,
        Some("shutdown") => REASON_SHUTDOWN,
        Some("agent-system") => REASON_AGENT_SYSTEM,
        Some("agent-session") => REASON_AGENT_SESSION,
        _ => REASON_UNKNOWN,
    }
}

pub fn key_kind(ext_flags: u8) -> &'static str {
    if ext_flags & EXT_RESPONSES_VISIBLE != 0 {
        "responses-visible"
    } else if ext_flags & EXT_THINKING_VISIBLE != 0 {
        "thinking-visible"
    } else {
        "token-text"
    }
}

pub fn le_put32(value: u32) -> [u8; 4] {
    value.to_le_bytes()
}

pub fn le_get32(bytes: [u8; 4]) -> u32 {
    u32::from_le_bytes(bytes)
}

impl KvHeader {
    pub fn to_bytes(&self) -> [u8; FIXED_HEADER] {
        let mut h = [0_u8; FIXED_HEADER];
        h[0] = b'K';
        h[1] = b'V';
        h[2] = b'C';
        h[3] = 1;
        h[4] = self.quant_bits;
        h[5] = self.reason;
        h[6] = self.ext_flags;
        h[8..12].copy_from_slice(&self.tokens.to_le_bytes());
        h[12..16].copy_from_slice(&self.hits.to_le_bytes());
        h[16..20].copy_from_slice(&self.ctx_size.to_le_bytes());
        h[24..32].copy_from_slice(&self.created_at.to_le_bytes());
        h[32..40].copy_from_slice(&self.last_used.to_le_bytes());
        h[40..48].copy_from_slice(&self.payload_bytes.to_le_bytes());
        h
    }
}

pub fn read_header(bytes: &[u8]) -> Result<DecodedHeader, KvHeaderError> {
    if bytes.len() < FIXED_HEADER + 4 {
        return Err(KvHeaderError::Truncated);
    }
    if bytes[0] != b'K' || bytes[1] != b'V' || bytes[2] != b'C' {
        return Err(KvHeaderError::BadMagic);
    }
    if bytes[3] != 1 {
        return Err(KvHeaderError::BadVersion);
    }
    let quant_bits = bytes[4];
    let raw_reason = bytes[5];
    let reason = if raw_reason <= REASON_AGENT_SESSION {
        raw_reason
    } else {
        REASON_UNKNOWN
    };
    let tokens = u32::from_le_bytes(bytes[8..12].try_into().expect("slice length"));
    if tokens == 0 {
        return Err(KvHeaderError::EmptyTokens);
    }
    if quant_bits != 2 && quant_bits != 4 {
        return Err(KvHeaderError::BadQuant);
    }
    Ok(DecodedHeader {
        header: KvHeader {
            quant_bits,
            reason,
            ext_flags: bytes[6],
            tokens,
            hits: u32::from_le_bytes(bytes[12..16].try_into().expect("slice length")),
            ctx_size: u32::from_le_bytes(bytes[16..20].try_into().expect("slice length")),
            created_at: u64::from_le_bytes(bytes[24..32].try_into().expect("slice length")),
            last_used: u64::from_le_bytes(bytes[32..40].try_into().expect("slice length")),
            payload_bytes: u64::from_le_bytes(bytes[40..48].try_into().expect("slice length")),
        },
        text_bytes: u32::from_le_bytes(bytes[48..52].try_into().expect("slice length")),
    })
}

pub fn write_kvc_file(
    header: &KvHeader,
    text: &[u8],
    payload: &[u8],
    trailer: &[u8],
) -> Result<Vec<u8>, KvcFileError> {
    if text.len() > u32::MAX as usize {
        return Err(KvcFileError::TextTooLarge);
    }
    if payload.len() as u64 != header.payload_bytes {
        return Err(KvcFileError::PayloadLengthMismatch);
    }
    let total = FIXED_HEADER
        .checked_add(4)
        .and_then(|n| n.checked_add(text.len()))
        .and_then(|n| n.checked_add(payload.len()))
        .and_then(|n| n.checked_add(trailer.len()))
        .ok_or(KvcFileError::SizeOverflow)?;
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&header.to_bytes());
    out.extend_from_slice(&(text.len() as u32).to_le_bytes());
    out.extend_from_slice(text);
    out.extend_from_slice(payload);
    out.extend_from_slice(trailer);
    Ok(out)
}

pub fn read_kvc_file(bytes: &[u8]) -> Result<KvcFile, KvcFileError> {
    let decoded = read_header(bytes).map_err(KvcFileError::Header)?;
    let text_len = decoded.text_bytes as usize;
    let text_start = FIXED_HEADER + 4;
    let text_end = text_start
        .checked_add(text_len)
        .ok_or(KvcFileError::SizeOverflow)?;
    if bytes.len() < text_end {
        return Err(KvcFileError::TruncatedText);
    }
    let payload_len =
        usize::try_from(decoded.header.payload_bytes).map_err(|_| KvcFileError::PayloadTooLarge)?;
    let payload_end = text_end
        .checked_add(payload_len)
        .ok_or(KvcFileError::SizeOverflow)?;
    if bytes.len() < payload_end {
        return Err(KvcFileError::TruncatedPayload);
    }
    Ok(KvcFile {
        header: decoded.header,
        text: bytes[text_start..text_end].to_vec(),
        payload: bytes[text_end..payload_end].to_vec(),
        trailer: bytes[payload_end..].to_vec(),
        file_size: bytes.len() as u64,
    })
}

pub fn find_dsml_tool_blocks(text: &[u8]) -> Vec<&[u8]> {
    const FORMS: [(&[u8], &[u8]); 6] = [
        (
            "\n\n<｜DSML｜tool_calls>".as_bytes(),
            "</｜DSML｜tool_calls>".as_bytes(),
        ),
        (
            "<｜DSML｜tool_calls>".as_bytes(),
            "</｜DSML｜tool_calls>".as_bytes(),
        ),
        (
            "\n\n<DSML｜tool_calls>".as_bytes(),
            "</DSML｜tool_calls>".as_bytes(),
        ),
        (
            "<DSML｜tool_calls>".as_bytes(),
            "</DSML｜tool_calls>".as_bytes(),
        ),
        (b"\n\n<tool_calls>", b"</tool_calls>"),
        (b"<tool_calls>", b"</tool_calls>"),
    ];
    let mut blocks = Vec::new();
    let mut pos = 0;
    while pos < text.len() {
        let mut best: Option<(usize, usize, &[u8])> = None;
        for (start, end) in FORMS {
            if let Some(rel_start) = find_bytes(&text[pos..], start) {
                let start_abs = pos + rel_start;
                if best.map_or(false, |(best_start, _, _)| start_abs >= best_start) {
                    continue;
                }
                let after_start = start_abs + start.len();
                if let Some(rel_end) = find_bytes(&text[after_start..], end) {
                    best = Some((start_abs, after_start + rel_end + end.len(), end));
                }
            }
        }
        if let Some((start, end, _)) = best {
            blocks.push(&text[start..end]);
            pos = end;
        } else {
            break;
        }
    }
    blocks
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

pub fn write_tool_map_trailer(
    text: &[u8],
    entries: &[ToolMapEntry],
    disabled: bool,
) -> Option<Vec<u8>> {
    if disabled || text.is_empty() {
        return Some(Vec::new());
    }
    let blocks = find_dsml_tool_blocks(text);
    let mut selected: Vec<&ToolMapEntry> = Vec::new();
    let mut seen_blocks: Vec<Vec<u8>> = Vec::new();
    for block in blocks {
        if seen_blocks.iter().any(|seen| seen.as_slice() == block) {
            continue;
        }
        seen_blocks.push(block.to_vec());
        let block_entries: Vec<&ToolMapEntry> = entries
            .iter()
            .filter(|entry| entry.dsml.as_slice() == block)
            .collect();
        for entry in block_entries.into_iter().rev() {
            if entry.id.len() > u32::MAX as usize || entry.dsml.len() > u32::MAX as usize {
                continue;
            }
            selected.push(entry);
        }
    }
    if selected.is_empty() {
        return Some(Vec::new());
    }
    if selected.len() > u32::MAX as usize {
        return None;
    }
    let mut len = TOOL_MAP_HEADER;
    for entry in &selected {
        len = len
            .checked_add(8)?
            .checked_add(entry.id.len())?
            .checked_add(entry.dsml.len())?;
    }
    let mut out = Vec::with_capacity(len);
    out.extend_from_slice(b"KTM");
    out.push(TOOL_MAP_VERSION);
    out.extend_from_slice(&(selected.len() as u32).to_le_bytes());
    for entry in selected {
        out.extend_from_slice(&(entry.id.len() as u32).to_le_bytes());
        out.extend_from_slice(&(entry.dsml.len() as u32).to_le_bytes());
        out.extend_from_slice(entry.id.as_bytes());
        out.extend_from_slice(&entry.dsml);
    }
    Some(out)
}

pub fn read_tool_map_trailer(
    bytes: &[u8],
    max_entries: usize,
) -> Result<ToolMapDecode, (ToolMapError, ToolMapDecode)> {
    if bytes.is_empty() {
        return Ok(ToolMapDecode {
            entries: Vec::new(),
            declared_count: 0,
        });
    }
    if bytes.len() < TOOL_MAP_HEADER {
        return Err((
            ToolMapError::ShortHeader,
            ToolMapDecode {
                entries: Vec::new(),
                declared_count: 0,
            },
        ));
    }
    if &bytes[0..3] != b"KTM" || bytes[3] != TOOL_MAP_VERSION {
        return Err((
            ToolMapError::BadHeader,
            ToolMapDecode {
                entries: Vec::new(),
                declared_count: 0,
            },
        ));
    }
    let declared_count = u32::from_le_bytes(bytes[4..8].try_into().expect("slice length"));
    let mut decode = ToolMapDecode {
        entries: Vec::new(),
        declared_count,
    };
    if declared_count as usize > max_entries.saturating_mul(4) {
        return Err((ToolMapError::CountLimit, decode));
    }
    let mut pos = TOOL_MAP_HEADER;
    for _ in 0..declared_count {
        if bytes.len() - pos < 8 {
            return Err((ToolMapError::ShortEntryHeader, decode));
        }
        let id_len = u32::from_le_bytes(bytes[pos..pos + 4].try_into().expect("slice length"));
        let dsml_len =
            u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().expect("slice length"));
        pos += 8;
        if id_len == 0 || id_len as usize > TOOL_MAP_MAX_ID_LEN {
            return Err((ToolMapError::BadIdLen, decode));
        }
        if dsml_len == 0 || dsml_len as usize > TOOL_MAP_MAX_DSML_LEN {
            return Err((ToolMapError::BadDsmlLen, decode));
        }
        let id_len = id_len as usize;
        let dsml_len = dsml_len as usize;
        if bytes.len() - pos < id_len {
            return Err((ToolMapError::TruncatedId, decode));
        }
        let id = String::from_utf8_lossy(&bytes[pos..pos + id_len]).into_owned();
        pos += id_len;
        if bytes.len() - pos < dsml_len {
            return Err((ToolMapError::TruncatedDsml, decode));
        }
        let dsml = bytes[pos..pos + dsml_len].to_vec();
        pos += dsml_len;
        decode.entries.push(ToolMapEntry { id, dsml });
    }
    Ok(decode)
}

pub fn sha1_bytes_hex(bytes: &[u8]) -> String {
    let mut sha1 = Sha1::new();
    sha1.update(bytes);
    hex_bytes(&sha1.finish())
}

pub fn sha_hex_name(name: &str) -> Option<String> {
    if name.len() != 43 || !name.ends_with(".kv") {
        return None;
    }
    let sha = &name[..40];
    if !sha.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    Some(sha.to_ascii_lowercase())
}

pub fn path_join(dir: &str, name: &str) -> String {
    let mut out = String::with_capacity(dir.len() + name.len() + 1);
    out.push_str(dir);
    if out.is_empty() || !out.ends_with('/') {
        out.push('/');
    }
    out.push_str(name);
    out
}

pub fn path_for_sha(dir: &str, sha: &str) -> String {
    path_join(dir, &format!("{sha}.kv"))
}

pub fn store_len(options: KvOptions, tokens: i32) -> i32 {
    let trim = options.boundary_trim_tokens;
    let align = options.boundary_align_tokens;
    if tokens > options.min_tokens + trim {
        let mut stable = tokens - trim;
        if align > 0 {
            stable -= stable % align;
        }
        if stable >= options.min_tokens {
            return stable;
        }
    }
    tokens
}

pub fn chat_anchor_pos(
    options: KvOptions,
    prompt: &[i32],
    user_token_id: i32,
    assistant_token_id: i32,
) -> i32 {
    if user_token_id < 0 || assistant_token_id < 0 {
        return -1;
    }
    let mut last_user = -1;
    for (idx, &token) in prompt.iter().enumerate() {
        if token == assistant_token_id {
            break;
        }
        if token == user_token_id {
            last_user = idx as i32;
        }
    }
    if last_user >= options.min_tokens {
        last_user
    } else {
        -1
    }
}

fn continued_step(config: KvPolicyConfig) -> i32 {
    if !config.enabled || config.options.continued_interval_tokens <= 0 {
        return 0;
    }
    let mut step = config.options.continued_interval_tokens;
    let align = config.options.boundary_align_tokens;
    if align > 0 {
        step = ((step + align - 1) / align) * align;
        if step <= 0 {
            step = align;
        }
    }
    step
}

pub fn continued_store_target(config: KvPolicyConfig, live_tokens: i32) -> i32 {
    let step = continued_step(config);
    if step <= 0 {
        return 0;
    }
    if live_tokens < config.options.min_tokens {
        return 0;
    }
    if live_tokens % step != 0 {
        return 0;
    }
    if live_tokens <= config.continued_last_store_tokens {
        return 0;
    }
    live_tokens
}

pub fn note_store(config: &mut KvPolicyConfig, tokens: i32) {
    if tokens > config.continued_last_store_tokens {
        config.continued_last_store_tokens = tokens;
    }
}

pub fn suppress_continued_store(config: &mut KvPolicyConfig, tokens: i32) -> i32 {
    if continued_store_target(*config, tokens) != tokens {
        return -1;
    }
    let old = config.continued_last_store_tokens;
    note_store(config, tokens);
    old
}

pub fn restore_suppressed_continued(
    config: &mut KvPolicyConfig,
    old_tokens: i32,
    suppressed_tokens: i32,
) {
    if old_tokens >= 0 && config.continued_last_store_tokens == suppressed_tokens {
        config.continued_last_store_tokens = old_tokens;
    }
}

pub fn file_size_fits(
    budget_bytes: u64,
    text_bytes: u64,
    payload_bytes: u64,
    trailer_bytes: u64,
) -> Option<FileSizeDecision> {
    let file_bytes = (FIXED_HEADER as u64)
        .checked_add(4)?
        .checked_add(text_bytes)?
        .checked_add(payload_bytes)?
        .checked_add(trailer_bytes)?;
    if budget_bytes == 0 {
        return Some(FileSizeDecision {
            fits: true,
            file_bytes,
            required_bytes: 0,
        });
    }
    let slack = file_bytes / 100 + u64::from(file_bytes % 100 != 0);
    let required_bytes = file_bytes.checked_add(slack)?;
    Some(FileSizeDecision {
        fits: required_bytes <= budget_bytes,
        file_bytes,
        required_bytes,
    })
}

pub fn byte_prefix_match(text: &[u8], prefix: &[u8]) -> bool {
    prefix.len() <= text.len() && text.starts_with(prefix)
}

pub fn effective_prompt_suffix<'a>(prompt_text: &'a [u8], cached_text: &[u8]) -> Option<&'a [u8]> {
    if byte_prefix_match(prompt_text, cached_text) {
        Some(&prompt_text[cached_text.len()..])
    } else {
        None
    }
}

pub fn cache_replay_decision(
    live_tokens_before: u32,
    prompt_tokens: u32,
    live_prompt_common: u32,
    disk_cached_tokens: u32,
) -> CacheReplayDecision {
    let memory_token_reusable = live_tokens_before > 0
        && live_prompt_common == live_tokens_before
        && prompt_tokens >= live_tokens_before;
    let memory_miss_reason = if memory_token_reusable {
        "live-prefix-match"
    } else if live_tokens_before == 0 {
        "no-live-checkpoint"
    } else {
        "token-mismatch"
    };

    if memory_token_reusable {
        return CacheReplayDecision {
            cache_source: "memory-token",
            cached_tokens: live_tokens_before,
            cache_write_tokens: prompt_tokens.saturating_sub(live_tokens_before),
            disk_cached_tokens: 0,
            memory_token_reusable,
            memory_miss_reason,
        };
    }

    if disk_cached_tokens > 0 {
        return CacheReplayDecision {
            cache_source: "disk-text",
            cached_tokens: disk_cached_tokens,
            cache_write_tokens: prompt_tokens.saturating_sub(disk_cached_tokens),
            disk_cached_tokens,
            memory_token_reusable,
            memory_miss_reason,
        };
    }

    CacheReplayDecision {
        cache_source: "none",
        cached_tokens: 0,
        cache_write_tokens: prompt_tokens,
        disk_cached_tokens: 0,
        memory_token_reusable,
        memory_miss_reason,
    }
}

pub fn entry_eviction_score(entry: &KvEntry, protected_sha: Option<&str>, now: u64) -> f64 {
    if entry.file_size == 0 {
        return 0.0;
    }
    if protected_sha == Some(entry.sha.as_str()) {
        return f64::MAX;
    }
    let mut effective_hits = entry.hits as f64;
    let used_at = if entry.last_used != 0 {
        entry.last_used
    } else {
        entry.created_at
    };
    if used_at == 0 {
        effective_hits = 0.0;
    } else if now > used_at {
        let elapsed = (now - used_at) as f64;
        effective_hits *= f64::exp2(-elapsed / HIT_HALF_LIFE_SECONDS as f64);
        if effective_hits < MIN_EFFECTIVE_HITS {
            effective_hits = 0.0;
        }
    }
    (effective_hits + 1.0) * entry.tokens as f64 / entry.file_size as f64
}

pub fn find_text_prefix(
    entries: &[KvEntry],
    prompt_text: &str,
    config: KvPolicyConfig,
    quant_bits: u8,
    ctx_size: u32,
) -> Option<usize> {
    let prompt = prompt_text.as_bytes();
    let mut best: Option<usize> = None;
    for (idx, entry) in entries.iter().enumerate() {
        if entry.text_bytes > prompt.len() as u64 || entry.text_bytes > usize::MAX as u64 {
            continue;
        }
        if entry.tokens < config.options.min_tokens as u32 {
            continue;
        }
        if ctx_size < entry.ctx_size {
            continue;
        }
        if config.reject_different_quant && entry.quant_bits != quant_bits {
            continue;
        }
        if let Some(best_idx) = best {
            let current = &entries[best_idx];
            if entry.text_bytes < current.text_bytes {
                continue;
            }
            if entry.text_bytes == current.text_bytes && entry.tokens <= current.tokens {
                continue;
            }
        }
        let text_len = entry.text_bytes as usize;
        if sha1_bytes_hex(&prompt[..text_len]) == entry.sha {
            best = Some(idx);
        }
    }
    best
}

pub fn hex_bytes(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

#[derive(Clone)]
struct Sha1 {
    h: [u32; 5],
    bytes: u64,
    block: [u8; 64],
    used: usize,
}

impl Sha1 {
    fn new() -> Self {
        Self {
            h: [
                0x6745_2301,
                0xefcd_ab89,
                0x98ba_dcfe,
                0x1032_5476,
                0xc3d2_e1f0,
            ],
            bytes: 0,
            block: [0; 64],
            used: 0,
        }
    }

    fn update(&mut self, mut bytes: &[u8]) {
        self.bytes += bytes.len() as u64;
        while !bytes.is_empty() {
            let n = (64 - self.used).min(bytes.len());
            self.block[self.used..self.used + n].copy_from_slice(&bytes[..n]);
            self.used += n;
            bytes = &bytes[n..];
            if self.used == 64 {
                let block = self.block;
                self.transform(&block);
                self.used = 0;
            }
        }
    }

    fn finish(mut self) -> [u8; 20] {
        let bits = self.bytes * 8;
        self.update(&[0x80]);
        while self.used != 56 {
            self.update(&[0]);
        }
        self.update(&bits.to_be_bytes());
        let mut out = [0_u8; 20];
        for (idx, word) in self.h.iter().enumerate() {
            out[idx * 4..idx * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }

    fn transform(&mut self, block: &[u8; 64]) {
        let mut w = [0_u32; 80];
        for idx in 0..16 {
            w[idx] = u32::from_be_bytes([
                block[idx * 4],
                block[idx * 4 + 1],
                block[idx * 4 + 2],
                block[idx * 4 + 3],
            ]);
        }
        for idx in 16..80 {
            w[idx] = (w[idx - 3] ^ w[idx - 8] ^ w[idx - 14] ^ w[idx - 16]).rotate_left(1);
        }

        let mut a = self.h[0];
        let mut b = self.h[1];
        let mut c = self.h[2];
        let mut d = self.h[3];
        let mut e = self.h[4];
        for (idx, word) in w.iter().enumerate() {
            let (f, k) = if idx < 20 {
                ((b & c) | ((!b) & d), 0x5a82_7999)
            } else if idx < 40 {
                (b ^ c ^ d, 0x6ed9_eba1)
            } else if idx < 60 {
                ((b & c) | (b & d) | (c & d), 0x8f1b_bcdc)
            } else {
                (b ^ c ^ d, 0xca62_c1d6)
            };
            let tmp = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(*word);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = tmp;
        }
        self.h[0] = self.h[0].wrapping_add(a);
        self.h[1] = self.h[1].wrapping_add(b);
        self.h[2] = self.h[2].wrapping_add(c);
        self.h[3] = self.h[3].wrapping_add(d);
        self.h[4] = self.h[4].wrapping_add(e);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha1_matches_known_vectors() {
        assert_eq!(
            sha1_bytes_hex(b""),
            "da39a3ee5e6b4b0d3255bfef95601890afd80709"
        );
        assert_eq!(
            sha1_bytes_hex(b"abc"),
            "a9993e364706816aba3e25717850c26c9cd0d89d"
        );
    }

    #[test]
    fn header_roundtrip_matches_kvc_layout() {
        let header = KvHeader {
            quant_bits: 2,
            reason: REASON_COLD,
            ext_flags: 0,
            tokens: 550,
            hits: 1,
            ctx_size: 32768,
            created_at: 1779417499,
            last_used: 1779417514,
            payload_bytes: 31526948,
        };
        let mut bytes = Vec::from(header.to_bytes());
        bytes.extend_from_slice(&2520_u32.to_le_bytes());
        let decoded = read_header(&bytes).expect("valid header");
        assert_eq!(decoded.header, header);
        assert_eq!(decoded.text_bytes, 2520);
    }

    #[test]
    fn policy_edges_match_current_c_expectations() {
        let options = KvOptions::default();
        assert_eq!(store_len(options, 4096), 2048);
        assert_eq!(
            continued_store_target(KvPolicyConfig::default(), 10240),
            10240
        );
        let chat_options = KvOptions {
            min_tokens: 2,
            ..options
        };
        assert_eq!(
            chat_anchor_pos(chat_options, &[10, 1, 11, 1, 20, 2], 1, 2),
            3
        );
        assert_eq!(
            file_size_fits(1024, 100, 200, 30),
            Some(FileSizeDecision {
                fits: true,
                file_bytes: 382,
                required_bytes: 386,
            })
        );
    }

    #[test]
    fn cache_replay_decision_matches_server_cache_accounting() {
        assert_eq!(
            cache_replay_decision(0, 550, 0, 0),
            CacheReplayDecision {
                cache_source: "none",
                cached_tokens: 0,
                cache_write_tokens: 550,
                disk_cached_tokens: 0,
                memory_token_reusable: false,
                memory_miss_reason: "no-live-checkpoint",
            }
        );
        assert_eq!(
            cache_replay_decision(0, 561, 0, 552),
            CacheReplayDecision {
                cache_source: "disk-text",
                cached_tokens: 552,
                cache_write_tokens: 9,
                disk_cached_tokens: 552,
                memory_token_reusable: false,
                memory_miss_reason: "no-live-checkpoint",
            }
        );
        assert_eq!(
            cache_replay_decision(41, 50, 41, 0),
            CacheReplayDecision {
                cache_source: "memory-token",
                cached_tokens: 41,
                cache_write_tokens: 9,
                disk_cached_tokens: 0,
                memory_token_reusable: true,
                memory_miss_reason: "live-prefix-match",
            }
        );
        assert_eq!(
            cache_replay_decision(16, 39, 1, 0).memory_miss_reason,
            "token-mismatch"
        );
    }

    #[test]
    fn continued_store_note_suppress_and_restore_match_c_policy() {
        let mut config = KvPolicyConfig::default();
        note_store(&mut config, 4096);
        assert_eq!(config.continued_last_store_tokens, 4096);
        note_store(&mut config, 2048);
        assert_eq!(config.continued_last_store_tokens, 4096);
        assert_eq!(continued_store_target(config, 10240), 10240);

        let old = suppress_continued_store(&mut config, 10240);
        assert_eq!(old, 4096);
        assert_eq!(config.continued_last_store_tokens, 10240);
        assert_eq!(continued_store_target(config, 10240), 0);

        restore_suppressed_continued(&mut config, old, 10240);
        assert_eq!(config.continued_last_store_tokens, 4096);
        assert_eq!(continued_store_target(config, 10240), 10240);
    }

    #[test]
    fn continued_store_restore_ignores_non_suppressed_frontiers() {
        let mut config = KvPolicyConfig {
            continued_last_store_tokens: 10240,
            ..KvPolicyConfig::default()
        };
        assert_eq!(suppress_continued_store(&mut config, 10240), -1);
        assert_eq!(config.continued_last_store_tokens, 10240);
        assert_eq!(suppress_continued_store(&mut config, 18432), -1);
        assert_eq!(config.continued_last_store_tokens, 10240);

        restore_suppressed_continued(&mut config, -1, 10240);
        assert_eq!(config.continued_last_store_tokens, 10240);
        restore_suppressed_continued(&mut config, 4096, 20480);
        assert_eq!(config.continued_last_store_tokens, 10240);
    }

    #[test]
    fn effective_prompt_suffix_requires_byte_prefix() {
        assert_eq!(
            effective_prompt_suffix(b"abcdef", b"abc"),
            Some(&b"def"[..])
        );
        assert_eq!(effective_prompt_suffix(b"abcdef", b"abd"), None);
    }

    #[test]
    fn kvc_full_file_roundtrip_keeps_trailer_opaque() {
        let header = KvHeader {
            quant_bits: 4,
            reason: REASON_CONTINUED,
            ext_flags: EXT_TOOL_MAP,
            tokens: 2048,
            hits: 7,
            ctx_size: 65536,
            created_at: 1700000100,
            last_used: 1700000200,
            payload_bytes: 3,
        };
        let bytes = write_kvc_file(&header, b"text", &[1, 2, 3], b"xyz").expect("write KVC");
        let parsed = read_kvc_file(&bytes).expect("read KVC");
        assert_eq!(parsed.header, header);
        assert_eq!(parsed.text, b"text");
        assert_eq!(parsed.payload, [1, 2, 3]);
        assert_eq!(parsed.trailer, b"xyz");
        assert_eq!(parsed.file_size, bytes.len() as u64);
    }

    #[test]
    fn kvc_full_file_rejects_truncated_payload() {
        let header = KvHeader {
            quant_bits: 2,
            reason: REASON_COLD,
            ext_flags: 0,
            tokens: 1,
            hits: 0,
            ctx_size: 4096,
            created_at: 1,
            last_used: 1,
            payload_bytes: 4,
        };
        let mut bytes = Vec::from(header.to_bytes());
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(b"abc");
        bytes.extend_from_slice(&[1, 2]);
        assert_eq!(read_kvc_file(&bytes), Err(KvcFileError::TruncatedPayload));
    }

    #[test]
    fn tool_map_writer_matches_server_block_order() {
        let dsml = b"\n\n<tool_calls>\n</tool_calls>";
        let entries = vec![
            ToolMapEntry {
                id: "call_a".to_string(),
                dsml: dsml.to_vec(),
            },
            ToolMapEntry {
                id: "call_b".to_string(),
                dsml: dsml.to_vec(),
            },
        ];
        let trailer = write_tool_map_trailer(dsml, &entries, false).expect("write trailer");
        let decoded =
            read_tool_map_trailer(&trailer, TOOL_MAP_DEFAULT_MAX_ENTRIES).expect("valid trailer");
        assert_eq!(decoded.declared_count, 2);
        assert_eq!(decoded.entries[0].id, "call_b");
        assert_eq!(decoded.entries[1].id, "call_a");
    }

    #[test]
    fn tool_map_reader_keeps_partial_success_count() {
        let mut trailer = Vec::new();
        trailer.extend_from_slice(b"KTM");
        trailer.push(1);
        trailer.extend_from_slice(&2_u32.to_le_bytes());
        trailer.extend_from_slice(&6_u32.to_le_bytes());
        trailer.extend_from_slice(&4_u32.to_le_bytes());
        trailer.extend_from_slice(b"call_a");
        trailer.extend_from_slice(b"dsml");
        trailer.extend_from_slice(&4_u32.to_le_bytes());
        trailer.extend_from_slice(&5_u32.to_le_bytes());
        trailer.extend_from_slice(b"ca");
        let (err, decode) =
            read_tool_map_trailer(&trailer, TOOL_MAP_DEFAULT_MAX_ENTRIES).unwrap_err();
        assert_eq!(err, ToolMapError::TruncatedId);
        assert_eq!(decode.declared_count, 2);
        assert_eq!(decode.entries.len(), 1);
    }
}
