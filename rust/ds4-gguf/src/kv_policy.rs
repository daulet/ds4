use std::fmt;

pub const FIXED_HEADER: usize = 48;
pub const DEFAULT_MB: u64 = 4096;
pub const HIT_HALF_LIFE_SECONDS: u64 = 6 * 60 * 60;
pub const MIN_EFFECTIVE_HITS: f64 = 0.01;

pub const EXT_TOOL_MAP: u8 = 1 << 0;
pub const EXT_RESPONSES_VISIBLE: u8 = 1 << 1;
pub const EXT_THINKING_VISIBLE: u8 = 1 << 2;

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
}
