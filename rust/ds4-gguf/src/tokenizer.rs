use crate::{Gguf, MetadataValue};
use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizerIdentity {
    pub token_count: usize,
    pub token_bytes_sha256: String,
    pub merge_count: usize,
    pub merge_pairs_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpecialTokenIds {
    pub bos: u32,
    pub eos: u32,
    pub user: u32,
    pub assistant: u32,
    pub think_start: u32,
    pub think_end: u32,
    pub dsml: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizerError {
    message: String,
}

impl TokenizerError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for TokenizerError {}

#[derive(Debug, Clone)]
pub struct Ds4Tokenizer {
    tokens: Vec<Vec<u8>>,
    token_to_id: HashMap<Vec<u8>, u32>,
    merge_rank: HashMap<Vec<u8>, i32>,
    identity: TokenizerIdentity,
    special: SpecialTokenIds,
}

impl Ds4Tokenizer {
    pub fn from_gguf(gguf: &Gguf) -> Result<Self, TokenizerError> {
        let tokens = required_string_array(gguf, "tokenizer.ggml.tokens")?;
        let merges = required_string_array(gguf, "tokenizer.ggml.merges")?;
        Self::from_token_and_merge_bytes(tokens, merges)
    }

    pub fn identity(&self) -> &TokenizerIdentity {
        &self.identity
    }

    pub fn special_token_ids(&self) -> SpecialTokenIds {
        self.special
    }

    pub fn tokenize_text(&self, text: &str) -> Vec<u32> {
        let bytes = text.as_bytes();
        let mut out = Vec::new();
        let mut pos = 0usize;
        while pos < bytes.len() {
            let start = pos;
            let c = bytes[pos];
            if ascii_digit(c) {
                let mut ndigits = 0;
                while pos < bytes.len() && ascii_digit(bytes[pos]) && ndigits < 3 {
                    pos += 1;
                    ndigits += 1;
                }
            } else if joyai_cjk_at(bytes, pos) {
                loop {
                    pos = next_utf8_char(bytes, pos);
                    if pos >= bytes.len() || !joyai_cjk_at(bytes, pos) {
                        break;
                    }
                }
            } else if joyai_ascii_punct_symbol(c)
                && pos + 1 < bytes.len()
                && ascii_alpha(bytes[pos + 1])
            {
                pos += 1;
                while pos < bytes.len() && ascii_alpha(bytes[pos]) {
                    pos += 1;
                }
            } else if joyai_letter_like_at(bytes, pos) {
                pos = joyai_consume_letters(bytes, pos);
            } else if !ascii_newline(c)
                && !joyai_ascii_punct_symbol(c)
                && pos + 1 < bytes.len()
                && joyai_letter_like_at(bytes, pos + 1)
            {
                pos += 1;
                pos = joyai_consume_letters(bytes, pos);
            } else if c == b' ' && pos + 1 < bytes.len() && joyai_ascii_punct_symbol(bytes[pos + 1])
            {
                pos += 1;
                while pos < bytes.len() && joyai_ascii_punct_symbol(bytes[pos]) {
                    pos += 1;
                }
                while pos < bytes.len() && ascii_newline(bytes[pos]) {
                    pos += 1;
                }
            } else if joyai_ascii_punct_symbol(c) {
                while pos < bytes.len() && joyai_ascii_punct_symbol(bytes[pos]) {
                    pos += 1;
                }
                while pos < bytes.len() && ascii_newline(bytes[pos]) {
                    pos += 1;
                }
            } else if ascii_space(c) {
                let mut p = pos;
                let mut last_newline_end = 0usize;
                while p < bytes.len() && ascii_space(bytes[p]) {
                    let sc = bytes[p];
                    p += 1;
                    if ascii_newline(sc) {
                        last_newline_end = p;
                    }
                }
                if last_newline_end != 0 {
                    pos = last_newline_end;
                } else if p < bytes.len()
                    && p > pos + 1
                    && (joyai_letter_like_at(bytes, p) || joyai_ascii_punct_symbol(bytes[p]))
                {
                    pos = p - 1;
                } else {
                    pos = p;
                }
            } else {
                pos = next_utf8_char(bytes, pos);
            }
            if pos == start {
                pos = next_utf8_char(bytes, pos);
            }
            self.bpe_emit_piece(&bytes[start..pos], &mut out);
        }
        out
    }

    pub fn tokenize_rendered_chat(&self, text: &str) -> Vec<u32> {
        let bytes = text.as_bytes();
        let mut out = Vec::new();
        let mut span = 0usize;
        let mut pos = 0usize;
        while pos < bytes.len() {
            if let Some((token, len)) = self.special_token_at(bytes, pos) {
                self.tokenize_span(&bytes[span..pos], &mut out);
                out.push(token);
                pos += len;
                span = pos;
            } else {
                pos += 1;
            }
        }
        self.tokenize_span(&bytes[span..], &mut out);
        out
    }

    pub fn token_bytes(&self, token: u32) -> Vec<u8> {
        let Some(raw) = self.tokens.get(token as usize) else {
            return Vec::new();
        };
        if token_is_literal_special(raw) {
            return raw.clone();
        }

        let mut out = Vec::with_capacity(raw.len());
        let mut pos = 0usize;
        while pos < raw.len() {
            let cp = utf8_decode_one(raw, &mut pos);
            if let Some(byte) = gpt2_codepoint_to_byte(cp) {
                out.push(byte);
            }
        }
        out
    }

    fn special_token_at(&self, bytes: &[u8], pos: usize) -> Option<(u32, usize)> {
        let special = self.special;
        let specials: [(&[u8], u32); 7] = [
            ("<｜begin▁of▁sentence｜>".as_bytes(), special.bos),
            ("<｜end▁of▁sentence｜>".as_bytes(), special.eos),
            ("<｜User｜>".as_bytes(), special.user),
            ("<｜Assistant｜>".as_bytes(), special.assistant),
            ("<think>".as_bytes(), special.think_start),
            ("</think>".as_bytes(), special.think_end),
            ("｜DSML｜".as_bytes(), special.dsml),
        ];
        for (text, token) in specials {
            if bytes[pos..].starts_with(text) {
                return Some((token, text.len()));
            }
        }
        None
    }

    fn tokenize_span(&self, bytes: &[u8], out: &mut Vec<u32>) {
        if bytes.is_empty() {
            return;
        }
        let text =
            std::str::from_utf8(bytes).expect("rendered chat spans must preserve UTF-8 boundaries");
        out.extend(self.tokenize_text(text));
    }

    fn from_token_and_merge_bytes(
        tokens: Vec<Vec<u8>>,
        merges: Vec<Vec<u8>>,
    ) -> Result<Self, TokenizerError> {
        if tokens.len() > i32::MAX as usize {
            return Err(TokenizerError::new("tokenizer token table is too large"));
        }

        let token_bytes_sha256 = canonical_string_table_sha256(&tokens);
        let merge_pairs_sha256 = canonical_string_table_sha256(&merges);

        let mut token_to_id = HashMap::with_capacity(tokens.len());
        for (idx, token) in tokens.iter().enumerate() {
            token_to_id.insert(token.clone(), idx as u32);
        }

        let mut merge_rank = HashMap::with_capacity(merges.len());
        for (idx, merge) in merges.iter().enumerate() {
            let rank = i32::try_from(idx).map_err(|_| {
                TokenizerError::new("tokenizer merge table is too large for C-compatible ranks")
            })?;
            merge_rank.insert(merge.clone(), rank);
        }

        let special = SpecialTokenIds {
            bos: lookup_required(&token_to_id, "<｜begin▁of▁sentence｜>")?,
            eos: lookup_required(&token_to_id, "<｜end▁of▁sentence｜>")?,
            user: lookup_required(&token_to_id, "<｜User｜>")?,
            assistant: lookup_required(&token_to_id, "<｜Assistant｜>")?,
            think_start: lookup_required(&token_to_id, "<think>")?,
            think_end: lookup_required(&token_to_id, "</think>")?,
            dsml: lookup_required(&token_to_id, "｜DSML｜")?,
        };

        Ok(Self {
            identity: TokenizerIdentity {
                token_count: tokens.len(),
                token_bytes_sha256,
                merge_count: merges.len(),
                merge_pairs_sha256,
            },
            tokens,
            token_to_id,
            merge_rank,
            special,
        })
    }

    fn bpe_emit_piece(&self, raw_piece: &[u8], out: &mut Vec<u32>) {
        let encoded = byte_encode(raw_piece);
        let mut symbols = Vec::new();
        let mut off = 0usize;
        while off < encoded.len() {
            let mut n = utf8_len_from_first_byte(encoded[off]);
            if off + n > encoded.len() {
                n = 1;
            }
            symbols.push(encoded[off..off + n].to_vec());
            off += n;
        }

        loop {
            let mut best_i = None;
            let mut best_rank = i32::MAX;
            for i in 0..symbols.len().saturating_sub(1) {
                if let Some(rank) = self.bpe_rank(&symbols[i], &symbols[i + 1]) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_i = Some(i);
                    }
                }
            }
            let Some(i) = best_i else {
                break;
            };
            let mut merged = Vec::with_capacity(symbols[i].len() + symbols[i + 1].len());
            merged.extend_from_slice(&symbols[i]);
            merged.extend_from_slice(&symbols[i + 1]);
            symbols[i] = merged;
            symbols.remove(i + 1);
        }

        for symbol in symbols {
            if let Some(token) = self.token_to_id.get(&symbol) {
                out.push(*token);
            } else {
                for byte in symbol {
                    if let Some(token) = self.token_to_id.get(&[byte][..]) {
                        out.push(*token);
                    }
                }
            }
        }
    }

    fn bpe_rank(&self, a: &[u8], b: &[u8]) -> Option<i32> {
        let mut key = Vec::with_capacity(a.len() + 1 + b.len());
        key.extend_from_slice(a);
        key.push(b' ');
        key.extend_from_slice(b);
        self.merge_rank.get(&key).copied()
    }

    #[cfg(test)]
    fn from_parts_for_test(tokens: &[&str], merges: &[&str]) -> Self {
        let tokens = tokens.iter().map(|s| s.as_bytes().to_vec()).collect();
        let merges = merges.iter().map(|s| s.as_bytes().to_vec()).collect();
        Self::from_token_and_merge_bytes(tokens, merges).expect("test tokenizer")
    }
}

fn required_string_array(gguf: &Gguf, key: &str) -> Result<Vec<Vec<u8>>, TokenizerError> {
    let Some(entry) = gguf.metadata.iter().find(|entry| entry.key == key) else {
        return Err(TokenizerError::new(format!("GGUF {key} is missing")));
    };
    match &entry.value {
        MetadataValue::Array {
            element_type: 8,
            values,
        } => {
            let mut out = Vec::with_capacity(values.len());
            for value in values {
                let MetadataValue::String(text) = value else {
                    return Err(TokenizerError::new(format!(
                        "GGUF {key} has a non-string entry"
                    )));
                };
                out.push(text.as_bytes().to_vec());
            }
            Ok(out)
        }
        _ => Err(TokenizerError::new(format!(
            "GGUF {key} is missing or invalid"
        ))),
    }
}

fn lookup_required(token_to_id: &HashMap<Vec<u8>, u32>, text: &str) -> Result<u32, TokenizerError> {
    token_to_id
        .get(text.as_bytes())
        .copied()
        .ok_or_else(|| TokenizerError::new(format!("required tokenizer token is missing: {text}")))
}

fn canonical_string_table_sha256(values: &[Vec<u8>]) -> String {
    let mut sha = Sha256::new();
    for (idx, value) in values.iter().enumerate() {
        sha.update(&(idx as u64).to_le_bytes());
        sha.update(&(value.len() as u64).to_le_bytes());
        sha.update(value);
    }
    hex_lower(&sha.finish())
}

fn byte_encode(input: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(input.len() * 4);
    for byte in input {
        utf8_put(&mut out, gpt2_byte_to_codepoint(*byte));
    }
    out
}

fn utf8_put(out: &mut Vec<u8>, cp: u32) {
    if cp <= 0x7f {
        out.push(cp as u8);
    } else if cp <= 0x7ff {
        out.push((0xc0 | (cp >> 6)) as u8);
        out.push((0x80 | (cp & 0x3f)) as u8);
    } else if cp <= 0xffff {
        out.push((0xe0 | (cp >> 12)) as u8);
        out.push((0x80 | ((cp >> 6) & 0x3f)) as u8);
        out.push((0x80 | (cp & 0x3f)) as u8);
    } else {
        out.push((0xf0 | (cp >> 18)) as u8);
        out.push((0x80 | ((cp >> 12) & 0x3f)) as u8);
        out.push((0x80 | ((cp >> 6) & 0x3f)) as u8);
        out.push((0x80 | (cp & 0x3f)) as u8);
    }
}

fn gpt2_byte_to_codepoint(byte: u8) -> u32 {
    if (33..=126).contains(&byte) || (161..=172).contains(&byte) || byte >= 174 {
        return byte as u32;
    }

    let mut n = 0u32;
    for candidate in 0u32..256 {
        if (33..=126).contains(&candidate) || (161..=172).contains(&candidate) || candidate >= 174 {
            continue;
        }
        if candidate == byte as u32 {
            return 256 + n;
        }
        n += 1;
    }
    byte as u32
}

fn gpt2_codepoint_to_byte(cp: u32) -> Option<u8> {
    if (33..=126).contains(&cp) || (161..=172).contains(&cp) || (174..=255).contains(&cp) {
        return Some(cp as u8);
    }

    let mut n = 0u32;
    for byte in 0u32..256 {
        if (33..=126).contains(&byte) || (161..=172).contains(&byte) || byte >= 174 {
            continue;
        }
        if cp == 256 + n {
            return Some(byte as u8);
        }
        n += 1;
    }
    None
}

fn utf8_len_from_first_byte(c: u8) -> usize {
    if c < 0x80 {
        1
    } else if (c & 0xe0) == 0xc0 {
        2
    } else if (c & 0xf0) == 0xe0 {
        3
    } else if (c & 0xf8) == 0xf0 {
        4
    } else {
        1
    }
}

fn next_utf8_char(bytes: &[u8], pos: usize) -> usize {
    let n = utf8_len_from_first_byte(bytes[pos]);
    if pos + n > bytes.len() {
        pos + 1
    } else {
        pos + n
    }
}

fn ascii_alpha(c: u8) -> bool {
    c.is_ascii_alphabetic()
}

fn ascii_digit(c: u8) -> bool {
    c.is_ascii_digit()
}

fn ascii_space(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}

fn ascii_newline(c: u8) -> bool {
    matches!(c, b'\n' | b'\r')
}

fn joyai_ascii_punct_symbol(c: u8) -> bool {
    (b'!'..=b'/').contains(&c)
        || (b':'..=b'@').contains(&c)
        || (b'['..=b'`').contains(&c)
        || (b'{'..=b'~').contains(&c)
}

fn utf8_is_cjk_hira_kata(cp: u32) -> bool {
    (0x4e00..=0x9fa5).contains(&cp)
        || (0x3040..=0x309f).contains(&cp)
        || (0x30a0..=0x30ff).contains(&cp)
}

fn utf8_peek_one(bytes: &[u8], pos: usize) -> (u32, usize) {
    let c0 = bytes[pos];
    let mut n = utf8_len_from_first_byte(c0);
    if pos + n > bytes.len() {
        n = 1;
    }
    let next = pos + n;

    let cp = if n == 1 {
        c0 as u32
    } else if n == 2 {
        ((c0 & 0x1f) as u32) << 6 | ((bytes[pos + 1] & 0x3f) as u32)
    } else if n == 3 {
        ((c0 & 0x0f) as u32) << 12
            | ((bytes[pos + 1] & 0x3f) as u32) << 6
            | ((bytes[pos + 2] & 0x3f) as u32)
    } else {
        ((c0 & 0x07) as u32) << 18
            | ((bytes[pos + 1] & 0x3f) as u32) << 12
            | ((bytes[pos + 2] & 0x3f) as u32) << 6
            | ((bytes[pos + 3] & 0x3f) as u32)
    };
    (cp, next)
}

fn joyai_letter_like_at(bytes: &[u8], pos: usize) -> bool {
    let c = bytes[pos];
    if c < 128 {
        ascii_alpha(c)
    } else {
        true
    }
}

fn joyai_consume_letters(bytes: &[u8], mut pos: usize) -> usize {
    while pos < bytes.len() && joyai_letter_like_at(bytes, pos) {
        pos = next_utf8_char(bytes, pos);
    }
    pos
}

fn joyai_cjk_at(bytes: &[u8], pos: usize) -> bool {
    if bytes[pos] < 128 {
        return false;
    }
    let (cp, _) = utf8_peek_one(bytes, pos);
    utf8_is_cjk_hira_kata(cp)
}

fn utf8_decode_one(bytes: &[u8], pos: &mut usize) -> u32 {
    let c = bytes[*pos];
    if c < 0x80 || *pos + 1 >= bytes.len() {
        *pos += 1;
        return c as u32;
    }
    if (c & 0xe0) == 0xc0 && *pos + 1 < bytes.len() {
        let cp = ((c & 0x1f) as u32) << 6 | ((bytes[*pos + 1] & 0x3f) as u32);
        *pos += 2;
        return cp;
    }
    if (c & 0xf0) == 0xe0 && *pos + 2 < bytes.len() {
        let cp = ((c & 0x0f) as u32) << 12
            | ((bytes[*pos + 1] & 0x3f) as u32) << 6
            | ((bytes[*pos + 2] & 0x3f) as u32);
        *pos += 3;
        return cp;
    }
    if (c & 0xf8) == 0xf0 && *pos + 3 < bytes.len() {
        let cp = ((c & 0x07) as u32) << 18
            | ((bytes[*pos + 1] & 0x3f) as u32) << 12
            | ((bytes[*pos + 2] & 0x3f) as u32) << 6
            | ((bytes[*pos + 3] & 0x3f) as u32);
        *pos += 4;
        return cp;
    }
    *pos += 1;
    c as u32
}

fn token_is_literal_special(bytes: &[u8]) -> bool {
    bytes.windows(3).any(|window| window == [0xef, 0xbd, 0x9c])
}

fn hex_lower(bytes: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(64);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

struct Sha256 {
    state: [u32; 8],
    block: [u8; 64],
    block_len: usize,
    bytes: u64,
}

impl Sha256 {
    fn new() -> Self {
        Self {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            block: [0; 64],
            block_len: 0,
            bytes: 0,
        }
    }

    fn update(&mut self, data: &[u8]) {
        let mut off = 0usize;
        self.bytes = self.bytes.wrapping_add(data.len() as u64);
        while off < data.len() {
            let take = (64 - self.block_len).min(data.len() - off);
            self.block[self.block_len..self.block_len + take]
                .copy_from_slice(&data[off..off + take]);
            self.block_len += take;
            off += take;
            if self.block_len == 64 {
                let block = self.block;
                self.compress(&block);
                self.block_len = 0;
            }
        }
    }

    fn finish(mut self) -> [u8; 32] {
        let bit_len = self.bytes.wrapping_mul(8);
        self.block[self.block_len] = 0x80;
        self.block_len += 1;
        if self.block_len > 56 {
            for byte in &mut self.block[self.block_len..] {
                *byte = 0;
            }
            let block = self.block;
            self.compress(&block);
            self.block_len = 0;
        }
        for byte in &mut self.block[self.block_len..56] {
            *byte = 0;
        }
        self.block[56..64].copy_from_slice(&bit_len.to_be_bytes());
        let block = self.block;
        self.compress(&block);

        let mut out = [0u8; 32];
        for (idx, word) in self.state.iter().enumerate() {
            out[idx * 4..idx * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }

    fn compress(&mut self, block: &[u8; 64]) {
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2,
        ];
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let mut a = self.state[0];
        let mut b = self.state[1];
        let mut c = self.state[2];
        let mut d = self.state[3];
        let mut e = self.state[4];
        let mut f = self.state[5];
        let mut g = self.state[6];
        let mut h = self.state[7];

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        self.state[0] = self.state[0].wrapping_add(a);
        self.state[1] = self.state[1].wrapping_add(b);
        self.state[2] = self.state[2].wrapping_add(c);
        self.state[3] = self.state[3].wrapping_add(d);
        self.state[4] = self.state[4].wrapping_add(e);
        self.state[5] = self.state[5].wrapping_add(f);
        self.state[6] = self.state[6].wrapping_add(g);
        self.state[7] = self.state[7].wrapping_add(h);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BOS: &str = "<｜begin▁of▁sentence｜>";
    const EOS: &str = "<｜end▁of▁sentence｜>";
    const USER: &str = "<｜User｜>";
    const ASSISTANT: &str = "<｜Assistant｜>";
    const THINK_START: &str = "<think>";
    const THINK_END: &str = "</think>";
    const DSML: &str = "｜DSML｜";

    fn toy_tokenizer() -> Ds4Tokenizer {
        Ds4Tokenizer::from_parts_for_test(
            &[
                BOS,
                EOS,
                USER,
                ASSISTANT,
                THINK_START,
                THINK_END,
                DSML,
                "a",
                "b",
                "ab",
                "1",
                "2",
                "3",
                "4",
                "12",
                "123",
                "\u{0120}",
            ],
            &["a b", "1 2", "12 3"],
        )
    }

    #[test]
    fn byte_level_bpe_merges_lowest_rank_pairs() {
        let tokenizer = toy_tokenizer();
        assert_eq!(tokenizer.tokenize_text("ab"), vec![9]);
    }

    #[test]
    fn joyai_digits_split_in_groups_of_three() {
        let tokenizer = toy_tokenizer();
        assert_eq!(tokenizer.tokenize_text("1234"), vec![15, 13]);
    }

    #[test]
    fn token_text_decodes_gpt2_byte_mapping() {
        let tokenizer = toy_tokenizer();
        assert_eq!(tokenizer.token_bytes(16), b" ");
    }

    #[test]
    fn rendered_chat_scans_specials_but_plain_text_does_not() {
        let tokenizer = toy_tokenizer();
        assert_eq!(tokenizer.tokenize_rendered_chat("<｜User｜>ab"), vec![2, 9]);
        assert_ne!(
            tokenizer.tokenize_text("<｜User｜>ab"),
            tokenizer.tokenize_rendered_chat("<｜User｜>ab"),
            "plain text remains BPE-tokenized and does not trust rendered controls"
        );
    }

    #[test]
    fn sha256_matches_known_vector() {
        let mut sha = Sha256::new();
        sha.update(b"abc");
        assert_eq!(
            hex_lower(&sha.finish()),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
