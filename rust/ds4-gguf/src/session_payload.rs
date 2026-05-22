use std::fmt;

pub const MAGIC: u32 = 0x3456_5344;
pub const VERSION: u32 = 1;
pub const U32_FIELDS: usize = 13;
pub const HEADER_BYTES: usize = U32_FIELDS * 4;
pub const IO_CHUNK_BYTES: usize = 8 * 1024 * 1024;
pub const N_LAYER: usize = 43;
pub const N_HEAD_DIM: u32 = 512;
pub const N_INDEXER_HEAD_DIM: u32 = 128;
pub const N_VOCAB: u32 = 129_280;
pub const N_SWA: u32 = 128;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PayloadHeader {
    pub magic: u32,
    pub version: u32,
    pub ctx_size: u32,
    pub prefill_cap: u32,
    pub raw_cap: u32,
    pub raw_window: u32,
    pub comp_cap: u32,
    pub token_count: u32,
    pub n_layer: u32,
    pub n_head_dim: u32,
    pub n_indexer_head_dim: u32,
    pub n_vocab: u32,
    pub raw_live_rows: u32,
}

impl PayloadHeader {
    pub fn fields(&self) -> [u32; U32_FIELDS] {
        [
            self.magic,
            self.version,
            self.ctx_size,
            self.prefill_cap,
            self.raw_cap,
            self.raw_window,
            self.comp_cap,
            self.token_count,
            self.n_layer,
            self.n_head_dim,
            self.n_indexer_head_dim,
            self.n_vocab,
            self.raw_live_rows,
        ]
    }

    pub fn to_bytes(&self) -> [u8; HEADER_BYTES] {
        let mut out = [0_u8; HEADER_BYTES];
        for (idx, field) in self.fields().iter().enumerate() {
            out[idx * 4..idx * 4 + 4].copy_from_slice(&field.to_le_bytes());
        }
        out
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PayloadRuntime {
    pub ctx_size: u32,
    pub prefill_cap: u32,
    pub raw_cap: u32,
    pub comp_cap: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PayloadSections {
    pub token_bytes: u64,
    pub logits_bytes: u64,
    pub comp_count_bytes: u64,
    pub index_count_bytes: u64,
    pub raw_row_bytes: u64,
    pub attn_comp_row_bytes: u64,
    pub attn_state_bytes: u64,
    pub index_comp_row_bytes: u64,
    pub index_state_bytes: u64,
}

impl PayloadSections {
    pub fn total(self) -> u64 {
        HEADER_BYTES as u64
            + self.token_bytes
            + self.logits_bytes
            + self.comp_count_bytes
            + self.index_count_bytes
            + self.raw_row_bytes
            + self.attn_comp_row_bytes
            + self.attn_state_bytes
            + self.index_comp_row_bytes
            + self.index_state_bytes
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PayloadError {
    TruncatedPayload,
    UnsupportedVersion,
    ContextFit,
    LayoutMismatch,
    ChunkLayoutMismatch,
    RawRingMismatch,
    CompressedCapTooLarge,
    InvalidCompressedRowCount,
    InvalidIndexerRowCount,
    TrailingPayloadBytes,
    SizeOverflow,
}

impl PayloadError {
    pub fn code(self) -> &'static str {
        match self {
            Self::TruncatedPayload => "truncated-payload",
            Self::UnsupportedVersion => "unsupported-version",
            Self::ContextFit => "context-fit",
            Self::LayoutMismatch => "layout-mismatch",
            Self::ChunkLayoutMismatch => "chunk-layout-mismatch",
            Self::RawRingMismatch => "raw-ring-mismatch",
            Self::CompressedCapTooLarge => "compressed-cap-too-large",
            Self::InvalidCompressedRowCount => "invalid-compressed-row-count",
            Self::InvalidIndexerRowCount => "invalid-indexer-row-count",
            Self::TrailingPayloadBytes => "trailing-payload-bytes",
            Self::SizeOverflow => "size-overflow",
        }
    }

    pub fn c_error(self) -> &'static str {
        match self {
            Self::TruncatedPayload => "truncated session payload",
            Self::UnsupportedVersion => "unsupported session payload version",
            Self::ContextFit => "KV checkpoint does not fit current context",
            Self::LayoutMismatch => "KV checkpoint was written for a different DS4 layout",
            Self::ChunkLayoutMismatch => {
                "KV checkpoint graph chunk layout does not match current runtime"
            }
            Self::RawRingMismatch => "KV checkpoint raw ring layout does not match current context",
            Self::CompressedCapTooLarge => {
                "KV checkpoint compressed cache is larger than current context"
            }
            Self::InvalidCompressedRowCount => "KV checkpoint has invalid compressed row count",
            Self::InvalidIndexerRowCount => "KV checkpoint has invalid indexer row count",
            Self::TrailingPayloadBytes => "KV checkpoint has trailing payload bytes",
            Self::SizeOverflow => "session payload size overflow",
        }
    }
}

impl fmt::Display for PayloadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.c_error())
    }
}

impl std::error::Error for PayloadError {}

pub fn compress_ratio(layer: usize) -> u32 {
    assert!(
        layer < N_LAYER,
        "DeepSeek4 layer index is outside the fixed model layout"
    );
    if layer < 2 {
        0
    } else if layer & 1 == 0 {
        4
    } else {
        128
    }
}

pub fn default_raw_cap(ctx_size: u32) -> u32 {
    let raw_cap = N_SWA.min(ctx_size);
    raw_cap.max(1)
}

pub fn default_prefill_cap(ctx_size: u32) -> u32 {
    if ctx_size == 0 {
        1
    } else {
        ctx_size.min(2048)
    }
}

pub fn cpu_comp_cap(ctx_size: u32) -> u32 {
    (ctx_size / 4 + 2).max(2)
}

pub fn default_cpu_runtime(ctx_size: u32) -> PayloadRuntime {
    PayloadRuntime {
        ctx_size,
        prefill_cap: default_prefill_cap(ctx_size),
        raw_cap: default_raw_cap(ctx_size),
        comp_cap: cpu_comp_cap(ctx_size),
    }
}

pub fn default_header(ctx_size: u32, tokens: u32) -> PayloadHeader {
    let raw_cap = default_raw_cap(ctx_size);
    PayloadHeader {
        magic: MAGIC,
        version: VERSION,
        ctx_size,
        prefill_cap: default_prefill_cap(ctx_size),
        raw_cap,
        raw_window: raw_cap,
        comp_cap: cpu_comp_cap(ctx_size),
        token_count: tokens,
        n_layer: N_LAYER as u32,
        n_head_dim: N_HEAD_DIM,
        n_indexer_head_dim: N_INDEXER_HEAD_DIM,
        n_vocab: N_VOCAB,
        raw_live_rows: tokens.min(raw_cap),
    }
}

pub fn read_header(bytes: &[u8]) -> Result<PayloadHeader, PayloadError> {
    if bytes.len() < HEADER_BYTES {
        return Err(PayloadError::TruncatedPayload);
    }
    let mut fields = [0_u32; U32_FIELDS];
    for (idx, field) in fields.iter_mut().enumerate() {
        *field = u32::from_le_bytes(
            bytes[idx * 4..idx * 4 + 4]
                .try_into()
                .expect("header u32 slice length"),
        );
    }
    let header = PayloadHeader {
        magic: fields[0],
        version: fields[1],
        ctx_size: fields[2],
        prefill_cap: fields[3],
        raw_cap: fields[4],
        raw_window: fields[5],
        comp_cap: fields[6],
        token_count: fields[7],
        n_layer: fields[8],
        n_head_dim: fields[9],
        n_indexer_head_dim: fields[10],
        n_vocab: fields[11],
        raw_live_rows: fields[12],
    };
    if header.magic != MAGIC || header.version != VERSION {
        return Err(PayloadError::UnsupportedVersion);
    }
    Ok(header)
}

pub fn validate_header_cpu(
    header: &PayloadHeader,
    runtime: PayloadRuntime,
) -> Result<(), PayloadError> {
    if header.ctx_size > runtime.ctx_size || header.token_count >= runtime.ctx_size {
        return Err(PayloadError::ContextFit);
    }
    if header.n_layer != N_LAYER as u32
        || header.n_head_dim != N_HEAD_DIM
        || header.n_indexer_head_dim != N_INDEXER_HEAD_DIM
        || header.n_vocab != N_VOCAB
    {
        return Err(PayloadError::LayoutMismatch);
    }
    if header.prefill_cap != runtime.prefill_cap || header.raw_window != runtime.raw_cap {
        return Err(PayloadError::ChunkLayoutMismatch);
    }
    let expected_raw_live = header.token_count.min(header.raw_window);
    if header.raw_cap == 0
        || header.raw_live_rows != expected_raw_live
        || header.raw_live_rows > header.raw_cap
        || header.raw_live_rows > runtime.raw_cap
    {
        return Err(PayloadError::RawRingMismatch);
    }
    if header.comp_cap > runtime.comp_cap {
        return Err(PayloadError::CompressedCapTooLarge);
    }
    Ok(())
}

pub fn layer_attn_state_bytes(ratio: u32) -> u64 {
    let coff = if ratio == 4 { 2_u64 } else { 1_u64 };
    coff * u64::from(N_HEAD_DIM) * coff * u64::from(ratio) * 4
}

pub fn layer_index_state_bytes(ratio: u32) -> u64 {
    let coff = if ratio == 4 { 2_u64 } else { 1_u64 };
    coff * u64::from(N_INDEXER_HEAD_DIM) * coff * u64::from(ratio) * 4
}

pub fn sections(
    header: &PayloadHeader,
    n_comp: &[u32; N_LAYER],
    n_index_comp: &[u32; N_LAYER],
) -> PayloadSections {
    let mut out = PayloadSections {
        token_bytes: u64::from(header.token_count) * 4,
        logits_bytes: u64::from(N_VOCAB) * 4,
        comp_count_bytes: N_LAYER as u64 * 4,
        index_count_bytes: N_LAYER as u64 * 4,
        raw_row_bytes: 0,
        attn_comp_row_bytes: 0,
        attn_state_bytes: 0,
        index_comp_row_bytes: 0,
        index_state_bytes: 0,
    };
    for layer in 0..N_LAYER {
        out.raw_row_bytes += u64::from(header.raw_live_rows) * u64::from(N_HEAD_DIM) * 4;
        let ratio = compress_ratio(layer);
        if ratio == 0 {
            continue;
        }
        out.attn_comp_row_bytes += u64::from(n_comp[layer]) * u64::from(N_HEAD_DIM) * 4;
        out.attn_state_bytes += 2 * layer_attn_state_bytes(ratio);
        if ratio == 4 {
            out.index_comp_row_bytes +=
                u64::from(n_index_comp[layer]) * u64::from(N_INDEXER_HEAD_DIM) * 4;
            out.index_state_bytes += 2 * layer_index_state_bytes(ratio);
        }
    }
    out
}

pub fn validate_payload_cpu(bytes: &[u8], runtime: PayloadRuntime) -> Result<(), PayloadError> {
    let header = read_header(bytes)?;
    validate_header_cpu(&header, runtime)?;
    let mut pos = HEADER_BYTES;
    consume(&mut pos, bytes.len(), u64::from(header.token_count) * 4)?;
    consume(&mut pos, bytes.len(), u64::from(N_VOCAB) * 4)?;

    let mut n_comp = [0_u32; N_LAYER];
    let mut n_index_comp = [0_u32; N_LAYER];
    for value in &mut n_comp {
        *value = read_u32(bytes, &mut pos)?;
        if *value > header.comp_cap || *value > runtime.comp_cap {
            return Err(PayloadError::InvalidCompressedRowCount);
        }
    }
    for value in &mut n_index_comp {
        *value = read_u32(bytes, &mut pos)?;
        if *value > header.comp_cap || *value > runtime.comp_cap {
            return Err(PayloadError::InvalidIndexerRowCount);
        }
    }

    for layer in 0..N_LAYER {
        consume(
            &mut pos,
            bytes.len(),
            u64::from(header.raw_live_rows) * u64::from(N_HEAD_DIM) * 4,
        )?;
        let ratio = compress_ratio(layer);
        if ratio == 0 {
            continue;
        }
        consume(
            &mut pos,
            bytes.len(),
            u64::from(n_comp[layer]) * u64::from(N_HEAD_DIM) * 4,
        )?;
        consume(&mut pos, bytes.len(), layer_attn_state_bytes(ratio))?;
        consume(&mut pos, bytes.len(), layer_attn_state_bytes(ratio))?;
        if ratio == 4 {
            consume(
                &mut pos,
                bytes.len(),
                u64::from(n_index_comp[layer]) * u64::from(N_INDEXER_HEAD_DIM) * 4,
            )?;
            consume(&mut pos, bytes.len(), layer_index_state_bytes(ratio))?;
            consume(&mut pos, bytes.len(), layer_index_state_bytes(ratio))?;
        }
    }
    if pos != bytes.len() {
        return Err(PayloadError::TrailingPayloadBytes);
    }
    Ok(())
}

fn read_u32(bytes: &[u8], pos: &mut usize) -> Result<u32, PayloadError> {
    let end = pos.checked_add(4).ok_or(PayloadError::SizeOverflow)?;
    if end > bytes.len() {
        return Err(PayloadError::TruncatedPayload);
    }
    let value = u32::from_le_bytes(bytes[*pos..end].try_into().expect("u32 slice length"));
    *pos = end;
    Ok(value)
}

fn consume(pos: &mut usize, len: usize, bytes: u64) -> Result<(), PayloadError> {
    let n = usize::try_from(bytes).map_err(|_| PayloadError::SizeOverflow)?;
    let end = pos.checked_add(n).ok_or(PayloadError::SizeOverflow)?;
    if end > len {
        return Err(PayloadError::TruncatedPayload);
    }
    *pos = end;
    Ok(())
}

pub fn append_full_payload(
    out: &mut Vec<u8>,
    header: &PayloadHeader,
    n_comp: &[u32; N_LAYER],
    n_index_comp: &[u32; N_LAYER],
) {
    out.extend_from_slice(&header.to_bytes());
    for idx in 0..header.token_count {
        out.extend_from_slice(&(1000_u32 + idx).to_le_bytes());
    }
    append_zeros(out, u64::from(N_VOCAB) * 4);
    for value in n_comp {
        out.extend_from_slice(&value.to_le_bytes());
    }
    for value in n_index_comp {
        out.extend_from_slice(&value.to_le_bytes());
    }
    for layer in 0..N_LAYER {
        append_zeros(
            out,
            u64::from(header.raw_live_rows) * u64::from(N_HEAD_DIM) * 4,
        );
        let ratio = compress_ratio(layer);
        if ratio == 0 {
            continue;
        }
        append_zeros(out, u64::from(n_comp[layer]) * u64::from(N_HEAD_DIM) * 4);
        append_zeros(out, layer_attn_state_bytes(ratio));
        append_zeros(out, layer_attn_state_bytes(ratio));
        if ratio == 4 {
            append_zeros(
                out,
                u64::from(n_index_comp[layer]) * u64::from(N_INDEXER_HEAD_DIM) * 4,
            );
            append_zeros(out, layer_index_state_bytes(ratio));
            append_zeros(out, layer_index_state_bytes(ratio));
        }
    }
}

pub fn append_prefix_to_first_comp(out: &mut Vec<u8>, header: &PayloadHeader, first_comp: u32) {
    out.extend_from_slice(&header.to_bytes());
    for idx in 0..header.token_count {
        out.extend_from_slice(&(1000_u32 + idx).to_le_bytes());
    }
    append_zeros(out, u64::from(N_VOCAB) * 4);
    out.extend_from_slice(&first_comp.to_le_bytes());
}

pub fn append_prefix_to_first_index(
    out: &mut Vec<u8>,
    header: &PayloadHeader,
    n_comp: &[u32; N_LAYER],
    first_index: u32,
) {
    out.extend_from_slice(&header.to_bytes());
    for idx in 0..header.token_count {
        out.extend_from_slice(&(1000_u32 + idx).to_le_bytes());
    }
    append_zeros(out, u64::from(N_VOCAB) * 4);
    for value in n_comp {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out.extend_from_slice(&first_index.to_le_bytes());
}

fn append_zeros(out: &mut Vec<u8>, bytes: u64) {
    let n = usize::try_from(bytes).expect("fixture payload fits usize");
    out.resize(out.len() + n, 0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_probe_payload_matches_size_formula() {
        let header = default_header(16, 3);
        let n_comp = [0_u32; N_LAYER];
        let n_index = [0_u32; N_LAYER];
        let mut bytes = Vec::new();
        append_full_payload(&mut bytes, &header, &n_comp, &n_index);
        assert_eq!(
            bytes.len() as u64,
            sections(&header, &n_comp, &n_index).total()
        );
        assert_eq!(
            validate_payload_cpu(&bytes, default_cpu_runtime(16)),
            Ok(())
        );
    }

    #[test]
    fn bad_magic_and_bad_version_match_c_conflation() {
        let mut bytes = default_header(16, 3).to_bytes();
        bytes[0] = 0;
        assert_eq!(
            validate_payload_cpu(&bytes, default_cpu_runtime(16)).unwrap_err(),
            PayloadError::UnsupportedVersion
        );

        let mut bytes = default_header(16, 3).to_bytes();
        bytes[4] = 2;
        assert_eq!(
            validate_payload_cpu(&bytes, default_cpu_runtime(16)).unwrap_err(),
            PayloadError::UnsupportedVersion
        );
    }

    #[test]
    fn trailing_payload_bytes_reject_after_full_body() {
        let header = default_header(16, 3);
        let n_comp = [0_u32; N_LAYER];
        let n_index = [0_u32; N_LAYER];
        let mut bytes = Vec::new();
        append_full_payload(&mut bytes, &header, &n_comp, &n_index);
        bytes.extend_from_slice(&0_u32.to_le_bytes());
        assert_eq!(
            validate_payload_cpu(&bytes, default_cpu_runtime(16)).unwrap_err(),
            PayloadError::TrailingPayloadBytes
        );
    }
}
