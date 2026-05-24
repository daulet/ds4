use ds4_gguf::{
    bind_ds4_weights, parse_gguf_allowing_missing_tensor_data, tensor_nbytes, tensor_type_name,
    Ds4LayerWeights, Ds4Weights, Gguf, TensorInfo,
};
use ds4_gpu::decode_backend::{set_model_fd, set_model_map_range, DecodeBackend, ModelMap};
use ds4_gpu::decode_plan::{raw_span_for_batch, raw_start_for_span};
use ds4_gpu::graph_plan::{
    layer_compression, GraphPlan, LayerCompression, HC_EPS, N_EMBD, N_EXPERT, N_EXPERT_USED,
    N_FF_EXP, N_HC, N_HC_SINKHORN_ITER, N_HEAD, N_HEAD_DIM, N_HEAD_KV, N_INDEXER_HEAD,
    N_INDEXER_HEAD_DIM, N_INDEXER_TOP_K, N_LAYER, N_LORA_O, N_LORA_Q, N_OUT_GROUP, N_ROT, N_VOCAB,
    RMS_EPS, ROPE_FREQ_BASE, ROPE_YARN_BETA_FAST, ROPE_YARN_BETA_SLOW,
};
use ds4_gpu::{initialize, synchronize, CommandBatch, Tensor};
use std::ffi::c_void;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::os::raw::c_int;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};

const SCHEMA: &str = "ds4.decode_long_indexed_attention.v1";
const CASE: &str = "tokens0_2051_long_indexed_attention";
const SEQUENCE_LEN: u32 = N_INDEXER_TOP_K * 4 + 4;
const FINAL_POSITION: u32 = SEQUENCE_LEN - 1;
const SPLIT_AFTER_LAYER: usize = 3;
const CTX_SIZE: u32 = 32_768;
const INDEXED_LAYER: usize = 2;
const PROT_READ: c_int = 0x1;
const MAP_PRIVATE: c_int = 0x02;
const MAP_FAILED: *mut c_void = !0usize as *mut c_void;
const INITIAL_HEADER_READ: u64 = 8 * 1024 * 1024;
const MAX_HEADER_READ: u64 = 512 * 1024 * 1024;
const SWIGLU_CLAMP_EXP: f32 = 10.0;
const COMPRESS_ROPE_FREQ_BASE: f32 = 160_000.0;
const COMPRESS_ROPE_FREQ_SCALE: f32 = 1.0 / 16.0;
const COMPRESSOR_SCORE_INIT: f32 = -1.0e30;
const ROPE_ORIG_CTX: u32 = 65_536;

unsafe extern "C" {
    fn mmap(
        addr: *mut c_void,
        length: usize,
        prot: c_int,
        flags: c_int,
        fd: c_int,
        offset: i64,
    ) -> *mut c_void;
    fn munmap(addr: *mut c_void, length: usize) -> c_int;
}

fn main() {
    if let Err(err) = run() {
        eprintln!("ds4-decode-long-indexed-attention: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse()?;
    let (gguf, header_bytes_read) = parse_header_prefix(&args.model)?;
    let weights = bind_ds4_weights(&gguf)?;
    let mapped = MappedModel::open(&args.model)?;

    initialize().map_err(|err| format!("failed to initialize backend: {err}"))?;
    let _backend = BackendGuard;
    set_model_fd(mapped.file.as_raw_fd())
        .map_err(|err| format!("failed to set model fd: {err}"))?;
    let model = unsafe { ModelMap::from_raw_parts(mapped.ptr.cast_const(), mapped.size) };
    set_model_map_range(
        model,
        gguf.tensor_data_offset,
        mapped.size - gguf.tensor_data_offset,
    )
    .map_err(|err| format!("failed to set model map range: {err}"))?;
    let backend = DecodeBackend::new(model);

    let plan = GraphPlan::for_context(CTX_SIZE, CTX_SIZE, false);
    let raw_cap = plan.allocated_raw_cap;
    let final_raw_row = FINAL_POSITION % raw_cap;
    let final_n_raw = raw_span_for_batch(plan.raw_window, raw_cap, FINAL_POSITION, 1);
    let final_raw_start = raw_start_for_span(FINAL_POSITION, final_n_raw, raw_cap);
    let dims = Dims::new(&weights.layers[0])?;
    let mut state = DecodeState::allocate(plan, dims)?;
    let mut after_layer2_hc = Tensor::allocate(byte_len(dims.hc_dim)?)
        .map_err(|err| format!("failed to allocate after_layer2_hc: {err}"))?;
    let mut layer2_raw_cache_row = Tensor::allocate(byte_len(u64::from(N_HEAD_DIM))?)
        .map_err(|err| format!("failed to allocate layer2_raw_cache_row: {err}"))?;
    let mut layer2_attn_comp_row = Tensor::allocate(byte_len(u64::from(N_HEAD_DIM))?)
        .map_err(|err| format!("failed to allocate layer2_attn_comp_row: {err}"))?;
    let mut layer2_index_comp_row = Tensor::allocate(byte_len(u64::from(N_INDEXER_HEAD_DIM))?)
        .map_err(|err| format!("failed to allocate layer2_index_comp_row: {err}"))?;

    for position in 0..SEQUENCE_LEN {
        let token = position;
        let raw_row = position % raw_cap;
        let n_raw = raw_span_for_batch(plan.raw_window, raw_cap, position, 1);
        let raw_start = raw_start_for_span(position, n_raw, raw_cap);
        let mut command_batch = CommandBatch::begin()
            .map_err(|err| format!("begin position {position} failed: {err}"))?;
        backend
            .embed_token_hc(
                state.cur_hc.as_tensor_mut(),
                weights.token_embd.abs_offset,
                N_VOCAB,
                token,
                N_EMBD,
                N_HC,
            )
            .map_err(|err| format!("position {position} embed_token_hc failed: {err}"))?;
        let layer_limit = if position == FINAL_POSITION {
            INDEXED_LAYER + 1
        } else {
            N_LAYER
        };
        for layer in 0..layer_limit {
            execute_layer(
                backend,
                &weights.layers[layer],
                layer,
                token,
                position,
                raw_cap,
                raw_row,
                n_raw,
                raw_start,
                &mut state,
                dims,
            )?;
            std::mem::swap(&mut state.cur_hc, &mut state.after_ffn_hc);
            if layer == SPLIT_AFTER_LAYER {
                command_batch
                    .flush()
                    .map_err(|err| format!("position {position} split flush failed: {err}"))?;
            }
            if position == FINAL_POSITION && layer == INDEXED_LAYER {
                after_layer2_hc
                    .copy_from(
                        &state.cur_hc,
                        0,
                        0,
                        byte_len(dims.hc_dim)?,
                        &mut command_batch,
                    )
                    .map_err(|err| format!("after_layer2_hc copy failed: {err}"))?;
            }
        }
        if position == FINAL_POSITION {
            copy_cache_checkpoints(
                &state,
                &mut layer2_raw_cache_row,
                &mut layer2_attn_comp_row,
                &mut layer2_index_comp_row,
                &mut command_batch,
            )?;
        }
        command_batch
            .finish()
            .map_err(|err| format!("finish position {position} failed: {err}"))?;
    }
    synchronize().map_err(|err| format!("synchronize failed: {err}"))?;

    let outputs = vec![
        Output::F32(read_tensor_output(
            "after_layer2_hc",
            &after_layer2_hc,
            dims.hc_dim,
        )?),
        Output::F32(read_tensor_output(
            "layer2_heads",
            &state.heads,
            dims.q_dim,
        )?),
        Output::F32(read_tensor_output(
            "layer2_attn_out",
            &state.attn_out,
            u64::from(N_EMBD),
        )?),
        Output::F32(read_tensor_output(
            "layer2_after_attn_hc",
            &state.after_attn_hc,
            dims.hc_dim,
        )?),
        Output::F32(read_tensor_output(
            "layer2_indexer_q",
            &state.indexer_q,
            u64::from(N_INDEXER_HEAD) * u64::from(N_INDEXER_HEAD_DIM),
        )?),
        Output::F32(read_tensor_output(
            "layer2_indexer_weights",
            &state.indexer_weights,
            u64::from(N_INDEXER_HEAD),
        )?),
        Output::F32(read_tensor_output(
            "layer2_indexer_scores",
            &state.indexer_scores,
            u64::from(state.layer_n_index_comp[INDEXED_LAYER]),
        )?),
        Output::I32(read_tensor_i32_output(
            "layer2_comp_selected",
            &state.comp_selected,
            u64::from(N_INDEXER_TOP_K),
        )?),
        Output::F32(read_tensor_output(
            "layer2_raw_cache_row",
            &layer2_raw_cache_row,
            u64::from(N_HEAD_DIM),
        )?),
        Output::F32(read_tensor_output(
            "layer2_attn_comp_row512",
            &layer2_attn_comp_row,
            u64::from(N_HEAD_DIM),
        )?),
        Output::F32(read_tensor_output(
            "layer2_index_comp_row512",
            &layer2_index_comp_row,
            u64::from(N_INDEXER_HEAD_DIM),
        )?),
    ];

    write_report(
        &gguf,
        &weights,
        header_bytes_read,
        mapped.size,
        plan,
        final_raw_row,
        final_n_raw,
        final_raw_start,
        dims,
        &state,
        &outputs,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn copy_cache_checkpoints(
    state: &DecodeState,
    layer2_raw_cache_row: &mut Tensor,
    layer2_attn_comp_row: &mut Tensor,
    layer2_index_comp_row: &mut Tensor,
    command_batch: &mut CommandBatch,
) -> Result<(), Box<dyn std::error::Error>> {
    const FINAL_COMP_ROW: u64 = (N_INDEXER_TOP_K as u64 * 4 + 4) / 4 - 1;
    let raw_offset = u64::from(FINAL_POSITION) * u64::from(N_HEAD_DIM) * 4;
    let attn_comp_offset = FINAL_COMP_ROW * u64::from(N_HEAD_DIM) * 4;
    let index_comp_offset = FINAL_COMP_ROW * u64::from(N_INDEXER_HEAD_DIM) * 4;
    let raw_bytes = byte_len(u64::from(N_HEAD_DIM))?;
    let index_bytes = byte_len(u64::from(N_INDEXER_HEAD_DIM))?;

    layer2_raw_cache_row
        .copy_from(
            &state.raw_cache[INDEXED_LAYER],
            0,
            raw_offset,
            raw_bytes,
            command_batch,
        )
        .map_err(|err| format!("layer2 raw cache row copy failed: {err}"))?;
    layer2_attn_comp_row
        .copy_from(
            state.layer_attn_comp_cache[INDEXED_LAYER]
                .as_ref()
                .ok_or("layer2 attn comp cache missing")?,
            0,
            attn_comp_offset,
            raw_bytes,
            command_batch,
        )
        .map_err(|err| format!("layer2 attn comp row copy failed: {err}"))?;
    layer2_index_comp_row
        .copy_from(
            state.layer_index_comp_cache[INDEXED_LAYER]
                .as_ref()
                .ok_or("layer2 index comp cache missing")?,
            0,
            index_comp_offset,
            index_bytes,
            command_batch,
        )
        .map_err(|err| format!("layer2 index comp row copy failed: {err}"))?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn execute_layer(
    backend: DecodeBackend<'_>,
    layer_weights: &Ds4LayerWeights,
    layer: usize,
    token: u32,
    position: u32,
    raw_cap: u32,
    raw_row: u32,
    n_raw: u32,
    raw_start: u32,
    state: &mut DecodeState,
    dims: Dims,
) -> Result<(), Box<dyn std::error::Error>> {
    let compression = layer_compression(layer).ok_or("invalid layer")?;
    let compressed = compression != LayerCompression::Dense;
    let (freq_base, freq_scale, n_ctx_orig, ext_factor, attn_factor) = if compressed {
        (
            COMPRESS_ROPE_FREQ_BASE,
            COMPRESS_ROPE_FREQ_SCALE,
            ROPE_ORIG_CTX,
            1.0,
            compressed_rope_attn_factor(),
        )
    } else {
        (ROPE_FREQ_BASE, 1.0, 0, 0.0, 1.0)
    };

    backend
        .rms_norm_plain(
            state.flat_hc.as_tensor_mut(),
            state.cur_hc.as_tensor_ref(),
            u32::try_from(dims.hc_dim)?,
            RMS_EPS,
        )
        .map_err(|err| format!("layer{layer} attn rms_norm_plain failed: {err}"))?;
    backend
        .matmul_f16(
            state.hc_mix.as_tensor_mut(),
            layer_weights.hc_attn_fn.abs_offset,
            dims.hc_dim,
            dims.hc_mix_dim,
            state.flat_hc.as_tensor_ref(),
            1,
        )
        .map_err(|err| format!("layer{layer} hc_attn_fn matmul_f16 failed: {err}"))?;
    backend
        .hc_split_weighted_sum_norm(
            state.attn_cur.as_tensor_mut(),
            state.attn_norm.as_tensor_mut(),
            state.hc_split.as_tensor_mut(),
            state.hc_mix.as_tensor_ref(),
            state.cur_hc.as_tensor_ref(),
            layer_weights.hc_attn_scale.abs_offset,
            layer_weights.hc_attn_base.abs_offset,
            layer_weights.attn_norm.abs_offset,
            N_EMBD,
            N_HC,
            N_HC_SINKHORN_ITER,
            HC_EPS,
            RMS_EPS,
        )
        .map_err(|err| format!("layer{layer} attn hc_split_weighted_sum_norm failed: {err}"))?;
    backend
        .matmul_q8_0(
            state.qr.as_tensor_mut(),
            layer_weights.attn_q_a.abs_offset,
            u64::from(N_EMBD),
            dims.q_rank,
            state.attn_norm.as_tensor_ref(),
            1,
        )
        .map_err(|err| format!("layer{layer} attn_q_a matmul_q8_0 failed: {err}"))?;
    backend
        .matmul_q8_0(
            state.kv_raw.as_tensor_mut(),
            layer_weights.attn_kv.abs_offset,
            u64::from(N_EMBD),
            dims.kv_dim,
            state.attn_norm.as_tensor_ref(),
            1,
        )
        .map_err(|err| format!("layer{layer} attn_kv matmul_q8_0 failed: {err}"))?;
    backend
        .dsv4_qkv_rms_norm_rows(
            state.qr_norm.as_tensor_mut(),
            state.qr.as_tensor_ref(),
            layer_weights.attn_q_a_norm.abs_offset,
            N_LORA_Q,
            state.kv.as_tensor_mut(),
            state.kv_raw.as_tensor_ref(),
            layer_weights.attn_kv_a_norm.abs_offset,
            N_HEAD_DIM,
            1,
            RMS_EPS,
        )
        .map_err(|err| format!("layer{layer} dsv4_qkv_rms_norm_rows failed: {err}"))?;
    backend
        .matmul_q8_0(
            state.q.as_tensor_mut(),
            layer_weights.attn_q_b.abs_offset,
            dims.q_rank,
            dims.q_dim,
            state.qr_norm.as_tensor_ref(),
            1,
        )
        .map_err(|err| format!("layer{layer} attn_q_b matmul_q8_0 failed: {err}"))?;
    backend
        .head_rms_norm(state.q.as_tensor_mut(), 1, N_HEAD, N_HEAD_DIM, RMS_EPS)
        .map_err(|err| format!("layer{layer} head_rms_norm failed: {err}"))?;
    backend
        .rope_tail(
            state.q.as_tensor_mut(),
            1,
            N_HEAD,
            N_HEAD_DIM,
            N_ROT,
            position,
            n_ctx_orig,
            false,
            freq_base,
            freq_scale,
            ext_factor,
            attn_factor,
            ROPE_YARN_BETA_FAST,
            ROPE_YARN_BETA_SLOW,
        )
        .map_err(|err| format!("layer{layer} q rope_tail failed: {err}"))?;
    backend
        .rope_tail(
            state.kv.as_tensor_mut(),
            1,
            N_HEAD_KV,
            N_HEAD_DIM,
            N_ROT,
            position,
            n_ctx_orig,
            false,
            freq_base,
            freq_scale,
            ext_factor,
            attn_factor,
            ROPE_YARN_BETA_FAST,
            ROPE_YARN_BETA_SLOW,
        )
        .map_err(|err| format!("layer{layer} kv rope_tail failed: {err}"))?;
    backend
        .kv_fp8_store_raw(
            state.kv.as_tensor_mut(),
            state.raw_cache[layer].as_tensor_mut(),
            raw_cap,
            raw_row,
            N_HEAD_DIM,
            N_ROT,
        )
        .map_err(|err| format!("layer{layer} kv_fp8_store_raw failed: {err}"))?;

    let mut n_comp = 0u32;
    let mut n_selected = 0u32;
    let mut use_indexed_attention = false;
    if compressed {
        let ratio = compression.ratio();
        let coeff = compression_coeff(compression);
        let comp_width = u64::from(coeff * N_HEAD_DIM);
        backend
            .matmul_f16_pair(
                state.comp_kv_cur.as_tensor_mut(),
                state.comp_sc_cur.as_tensor_mut(),
                layer_weights
                    .attn_compressor_kv
                    .as_ref()
                    .ok_or("attn_compressor_kv missing")?
                    .abs_offset,
                layer_weights
                    .attn_compressor_gate
                    .as_ref()
                    .ok_or("attn_compressor_gate missing")?
                    .abs_offset,
                u64::from(N_EMBD),
                comp_width,
                state.attn_norm.as_tensor_ref(),
                1,
            )
            .map_err(|err| {
                format!("layer{layer} attention compressor matmul_f16_pair failed: {err}")
            })?;
        backend
            .compressor_update(
                state.comp_kv_cur.as_tensor_ref(),
                state.comp_sc_cur.as_tensor_ref(),
                state.layer_attn_state_kv[layer]
                    .as_mut()
                    .ok_or("attn state kv missing")?
                    .as_tensor_mut(),
                state.layer_attn_state_score[layer]
                    .as_mut()
                    .ok_or("attn state score missing")?
                    .as_tensor_mut(),
                state.layer_attn_comp_cache[layer]
                    .as_mut()
                    .ok_or("attn comp cache missing")?
                    .as_tensor_mut(),
                layer_weights
                    .attn_compressor_ape
                    .as_ref()
                    .ok_or("attn_compressor_ape missing")?
                    .abs_offset,
                layer_weights
                    .attn_compressor_ape
                    .as_ref()
                    .ok_or("attn_compressor_ape missing")?
                    .type_id,
                layer_weights
                    .attn_compressor_norm
                    .as_ref()
                    .ok_or("attn_compressor_norm missing")?
                    .abs_offset,
                layer_weights
                    .attn_compressor_norm
                    .as_ref()
                    .ok_or("attn_compressor_norm missing")?
                    .type_id,
                N_HEAD_DIM,
                ratio,
                position,
                state.layer_n_comp[layer],
                N_ROT,
                n_ctx_orig,
                freq_base,
                freq_scale,
                ext_factor,
                attn_factor,
                ROPE_YARN_BETA_FAST,
                ROPE_YARN_BETA_SLOW,
                RMS_EPS,
            )
            .map_err(|err| format!("layer{layer} attention compressor_update failed: {err}"))?;
        if (position + 1) % ratio == 0 {
            let comp_row = state.layer_n_comp[layer];
            let mut comp_row_view = state.layer_attn_comp_cache[layer]
                .as_mut()
                .ok_or("attn comp cache missing")?
                .view(
                    u64::from(comp_row) * u64::from(N_HEAD_DIM) * 4,
                    byte_len(u64::from(N_HEAD_DIM))?,
                )
                .map_err(|err| format!("layer{layer} attn comp row view failed: {err}"))?;
            backend
                .dsv4_fp8_kv_quantize(comp_row_view.as_tensor_mut(), 1, N_HEAD_DIM, N_ROT)
                .map_err(|err| format!("layer{layer} attn comp row quantize failed: {err}"))?;
            state.layer_n_comp[layer] += 1;
        }
        if compression == LayerCompression::Ratio4 {
            let index_width = u64::from(coeff * N_INDEXER_HEAD_DIM);
            backend
                .matmul_f16_pair(
                    state.comp_kv_cur.as_tensor_mut(),
                    state.comp_sc_cur.as_tensor_mut(),
                    layer_weights
                        .indexer_compressor_kv
                        .as_ref()
                        .ok_or("indexer_compressor_kv missing")?
                        .abs_offset,
                    layer_weights
                        .indexer_compressor_gate
                        .as_ref()
                        .ok_or("indexer_compressor_gate missing")?
                        .abs_offset,
                    u64::from(N_EMBD),
                    index_width,
                    state.attn_norm.as_tensor_ref(),
                    1,
                )
                .map_err(|err| {
                    format!("layer{layer} indexer compressor matmul_f16_pair failed: {err}")
                })?;
            backend
                .compressor_update(
                    state.comp_kv_cur.as_tensor_ref(),
                    state.comp_sc_cur.as_tensor_ref(),
                    state.layer_index_state_kv[layer]
                        .as_mut()
                        .ok_or("index state kv missing")?
                        .as_tensor_mut(),
                    state.layer_index_state_score[layer]
                        .as_mut()
                        .ok_or("index state score missing")?
                        .as_tensor_mut(),
                    state.layer_index_comp_cache[layer]
                        .as_mut()
                        .ok_or("index comp cache missing")?
                        .as_tensor_mut(),
                    layer_weights
                        .indexer_compressor_ape
                        .as_ref()
                        .ok_or("indexer_compressor_ape missing")?
                        .abs_offset,
                    layer_weights
                        .indexer_compressor_ape
                        .as_ref()
                        .ok_or("indexer_compressor_ape missing")?
                        .type_id,
                    layer_weights
                        .indexer_compressor_norm
                        .as_ref()
                        .ok_or("indexer_compressor_norm missing")?
                        .abs_offset,
                    layer_weights
                        .indexer_compressor_norm
                        .as_ref()
                        .ok_or("indexer_compressor_norm missing")?
                        .type_id,
                    N_INDEXER_HEAD_DIM,
                    ratio,
                    position,
                    state.layer_n_index_comp[layer],
                    N_ROT,
                    n_ctx_orig,
                    freq_base,
                    freq_scale,
                    ext_factor,
                    attn_factor,
                    ROPE_YARN_BETA_FAST,
                    ROPE_YARN_BETA_SLOW,
                    RMS_EPS,
                )
                .map_err(|err| format!("layer{layer} indexer compressor_update failed: {err}"))?;
            if (position + 1) % ratio == 0 {
                let index_row = state.layer_n_index_comp[layer];
                let mut index_row_view = state.layer_index_comp_cache[layer]
                    .as_mut()
                    .ok_or("index comp cache missing")?
                    .view(
                        u64::from(index_row) * u64::from(N_INDEXER_HEAD_DIM) * 4,
                        byte_len(u64::from(N_INDEXER_HEAD_DIM))?,
                    )
                    .map_err(|err| format!("layer{layer} index comp row view failed: {err}"))?;
                backend
                    .dsv4_indexer_qat(index_row_view.as_tensor_mut(), 1, N_INDEXER_HEAD_DIM)
                    .map_err(|err| format!("layer{layer} index comp row qat failed: {err}"))?;
                state.layer_n_index_comp[layer] += 1;
            }
        }
        n_comp = state.layer_n_comp[layer];
        if n_comp > N_INDEXER_TOP_K {
            let indexer_q_dim = u64::from(N_INDEXER_HEAD) * u64::from(N_INDEXER_HEAD_DIM);
            backend
                .matmul_f16(
                    state.indexer_q.as_tensor_mut(),
                    layer_weights
                        .indexer_attn_q_b
                        .as_ref()
                        .ok_or("indexer_attn_q_b missing")?
                        .abs_offset,
                    dims.q_rank,
                    indexer_q_dim,
                    state.qr_norm.as_tensor_ref(),
                    1,
                )
                .map_err(|err| format!("layer{layer} indexer q matmul_f16 failed: {err}"))?;
            backend
                .rope_tail(
                    state.indexer_q.as_tensor_mut(),
                    1,
                    N_INDEXER_HEAD,
                    N_INDEXER_HEAD_DIM,
                    N_ROT,
                    position,
                    n_ctx_orig,
                    false,
                    freq_base,
                    freq_scale,
                    ext_factor,
                    attn_factor,
                    ROPE_YARN_BETA_FAST,
                    ROPE_YARN_BETA_SLOW,
                )
                .map_err(|err| format!("layer{layer} indexer q rope_tail failed: {err}"))?;
            backend
                .dsv4_indexer_qat(
                    state.indexer_q.as_tensor_mut(),
                    N_INDEXER_HEAD,
                    N_INDEXER_HEAD_DIM,
                )
                .map_err(|err| format!("layer{layer} indexer q qat failed: {err}"))?;
            backend
                .matmul_f16(
                    state.indexer_weights.as_tensor_mut(),
                    layer_weights
                        .indexer_proj
                        .as_ref()
                        .ok_or("indexer_proj missing")?
                        .abs_offset,
                    u64::from(N_EMBD),
                    u64::from(N_INDEXER_HEAD),
                    state.attn_norm.as_tensor_ref(),
                    1,
                )
                .map_err(|err| format!("layer{layer} indexer weights matmul_f16 failed: {err}"))?;
            let index_scale = 1.0f32 / ((N_INDEXER_HEAD_DIM * N_INDEXER_HEAD) as f32).sqrt();
            backend
                .indexer_score_one(
                    state.indexer_scores.as_tensor_mut(),
                    state.indexer_q.as_tensor_ref(),
                    state.indexer_weights.as_tensor_ref(),
                    state.layer_index_comp_cache[layer]
                        .as_ref()
                        .ok_or("index comp cache missing")?
                        .as_tensor_ref(),
                    state.layer_n_index_comp[layer],
                    N_INDEXER_HEAD,
                    N_INDEXER_HEAD_DIM,
                    index_scale,
                )
                .map_err(|err| format!("layer{layer} indexer_score_one failed: {err}"))?;
            backend
                .indexer_topk(
                    state.comp_selected.as_tensor_mut(),
                    state.indexer_scores.as_tensor_ref(),
                    state.layer_n_index_comp[layer],
                    1,
                    N_INDEXER_TOP_K,
                )
                .map_err(|err| format!("layer{layer} indexer_topk failed: {err}"))?;
            n_selected = N_INDEXER_TOP_K.min(state.layer_n_index_comp[layer]);
            use_indexed_attention = n_selected != 0;
        }
    }

    if use_indexed_attention {
        backend
            .attention_indexed_mixed_batch_heads(
                state.heads.as_tensor_mut(),
                layer_weights.attn_sinks.abs_offset,
                state.q.as_tensor_ref(),
                state.raw_cache[layer].as_tensor_ref(),
                state.layer_attn_comp_cache[layer]
                    .as_ref()
                    .ok_or("attn comp cache missing")?
                    .as_tensor_ref(),
                state.comp_selected.as_tensor_ref(),
                1,
                position,
                n_raw,
                raw_cap,
                raw_start,
                n_comp,
                n_selected,
                128,
                compression.ratio(),
                N_HEAD,
                N_HEAD_DIM,
            )
            .map_err(|err| {
                format!("layer{layer} attention_indexed_mixed_batch_heads failed: {err}")
            })?;
    } else {
        backend
            .attention_decode_heads(
                state.heads.as_tensor_mut(),
                layer_weights.attn_sinks.abs_offset,
                state.q.as_tensor_ref(),
                state.raw_cache[layer].as_tensor_ref(),
                n_raw,
                raw_cap,
                raw_start,
                state.layer_attn_comp_cache[layer]
                    .as_ref()
                    .map(|tensor| tensor.as_tensor_ref()),
                n_comp,
                None,
                0,
                N_HEAD,
                N_HEAD_DIM,
            )
            .map_err(|err| format!("layer{layer} attention_decode_heads failed: {err}"))?;
    }
    backend
        .rope_tail(
            state.heads.as_tensor_mut(),
            1,
            N_HEAD,
            N_HEAD_DIM,
            N_ROT,
            position,
            n_ctx_orig,
            true,
            freq_base,
            freq_scale,
            ext_factor,
            attn_factor,
            ROPE_YARN_BETA_FAST,
            ROPE_YARN_BETA_SLOW,
        )
        .map_err(|err| format!("layer{layer} heads inverse rope_tail failed: {err}"))?;
    backend
        .attention_output_low_q8(
            state.attn_low.as_tensor_mut(),
            layer_weights.attn_output_a.abs_offset,
            dims.group_dim,
            dims.rank,
            N_OUT_GROUP,
            state.heads.as_tensor_ref(),
        )
        .map_err(|err| format!("layer{layer} attention_output_low_q8 failed: {err}"))?;
    backend
        .matmul_q8_0_hc_expand(
            state.after_attn_hc.as_tensor_mut(),
            state.attn_out.as_tensor_mut(),
            layer_weights.attn_output_b.abs_offset,
            dims.low_dim,
            u64::from(N_EMBD),
            state.attn_low.as_tensor_ref(),
            state.cur_hc.as_tensor_ref(),
            state.hc_split.as_tensor_ref(),
            N_EMBD,
            N_HC,
        )
        .map_err(|err| format!("layer{layer} matmul_q8_0_hc_expand failed: {err}"))?;
    backend
        .rms_norm_plain(
            state.flat_hc.as_tensor_mut(),
            state.after_attn_hc.as_tensor_ref(),
            u32::try_from(dims.hc_dim)?,
            RMS_EPS,
        )
        .map_err(|err| format!("layer{layer} ffn rms_norm_plain failed: {err}"))?;
    backend
        .matmul_f16(
            state.hc_mix.as_tensor_mut(),
            layer_weights.hc_ffn_fn.abs_offset,
            dims.hc_dim,
            dims.hc_mix_dim,
            state.flat_hc.as_tensor_ref(),
            1,
        )
        .map_err(|err| format!("layer{layer} hc_ffn_fn matmul_f16 failed: {err}"))?;
    backend
        .hc_split_weighted_sum_norm(
            state.ffn_cur.as_tensor_mut(),
            state.ffn_norm.as_tensor_mut(),
            state.hc_split.as_tensor_mut(),
            state.hc_mix.as_tensor_ref(),
            state.after_attn_hc.as_tensor_ref(),
            layer_weights.hc_ffn_scale.abs_offset,
            layer_weights.hc_ffn_base.abs_offset,
            layer_weights.ffn_norm.abs_offset,
            N_EMBD,
            N_HC,
            N_HC_SINKHORN_ITER,
            HC_EPS,
            RMS_EPS,
        )
        .map_err(|err| format!("layer{layer} ffn hc_split_weighted_sum_norm failed: {err}"))?;
    backend
        .matmul_f16(
            state.router_logits.as_tensor_mut(),
            layer_weights.ffn_gate_inp.abs_offset,
            u64::from(N_EMBD),
            u64::from(N_EXPERT),
            state.ffn_norm.as_tensor_ref(),
            1,
        )
        .map_err(|err| format!("layer{layer} ffn_gate_inp matmul_f16 failed: {err}"))?;
    let router_hash_rows = layer_weights
        .ffn_gate_tid2eid
        .as_ref()
        .and_then(|tensor| tensor.dims.get(1).copied())
        .unwrap_or(0);
    backend
        .router_select(
            state.router_selected.as_tensor_mut(),
            state.router_weights.as_tensor_mut(),
            state.router_probs.as_tensor_mut(),
            layer_weights
                .ffn_exp_probs_b
                .as_ref()
                .map(|tensor| tensor.abs_offset)
                .unwrap_or(0),
            layer_weights
                .ffn_gate_tid2eid
                .as_ref()
                .map(|tensor| tensor.abs_offset)
                .unwrap_or(0),
            u32::try_from(router_hash_rows)?,
            token,
            0,
            0,
            layer_weights.ffn_exp_probs_b.is_some(),
            layer_weights.ffn_gate_tid2eid.is_some(),
            state.router_logits.as_tensor_ref(),
        )
        .map_err(|err| format!("layer{layer} router_select failed: {err}"))?;
    let gate_row_bytes = routed_expert_row_bytes(&layer_weights.ffn_gate_exps)?;
    let gate_expert_bytes = dims.expert_mid_dim * gate_row_bytes;
    let down_row_bytes = routed_expert_row_bytes(&layer_weights.ffn_down_exps)?;
    let down_expert_bytes = dims.routed_out_dim * down_row_bytes;
    backend
        .routed_moe_one(
            state.routed_out.as_tensor_mut(),
            state.routed_gate.as_tensor_mut(),
            state.routed_up.as_tensor_mut(),
            state.routed_mid.as_tensor_mut(),
            state.routed_down.as_tensor_mut(),
            layer_weights.ffn_gate_exps.abs_offset,
            layer_weights.ffn_up_exps.abs_offset,
            layer_weights.ffn_down_exps.abs_offset,
            layer_weights.ffn_gate_exps.type_id,
            layer_weights.ffn_down_exps.type_id,
            gate_expert_bytes,
            gate_row_bytes,
            down_expert_bytes,
            down_row_bytes,
            u32::try_from(dims.expert_in_dim)?,
            u32::try_from(dims.down_in_dim)?,
            u32::try_from(dims.routed_out_dim)?,
            state.router_selected.as_tensor_ref(),
            state.router_weights.as_tensor_ref(),
            N_EXPERT_USED,
            SWIGLU_CLAMP_EXP,
            state.ffn_norm.as_tensor_ref(),
        )
        .map_err(|err| format!("layer{layer} routed_moe_one failed: {err}"))?;
    backend
        .shared_gate_up_swiglu_q8_0(
            state.shared_gate.as_tensor_mut(),
            state.shared_up.as_tensor_mut(),
            state.shared_mid.as_tensor_mut(),
            layer_weights.ffn_gate_shexp.abs_offset,
            layer_weights.ffn_up_shexp.abs_offset,
            u64::from(N_EMBD),
            dims.shared_dim,
            state.ffn_norm.as_tensor_ref(),
            SWIGLU_CLAMP_EXP,
        )
        .map_err(|err| format!("layer{layer} shared_gate_up_swiglu_q8_0 failed: {err}"))?;
    backend
        .shared_down_hc_expand_q8_0(
            state.after_ffn_hc.as_tensor_mut(),
            state.shared_out.as_tensor_mut(),
            layer_weights.ffn_down_shexp.abs_offset,
            dims.shared_dim,
            u64::from(N_EMBD),
            state.shared_mid.as_tensor_ref(),
            state.routed_out.as_tensor_ref(),
            state.after_attn_hc.as_tensor_ref(),
            state.hc_split.as_tensor_ref(),
            N_EMBD,
            N_HC,
        )
        .map_err(|err| format!("layer{layer} shared_down_hc_expand_q8_0 failed: {err}"))?;

    Ok(())
}

struct Args {
    model: PathBuf,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn std::error::Error>> {
        let mut model = None;
        let mut args = std::env::args_os().skip(1);
        while let Some(arg) = args.next() {
            if arg == "--model" {
                let Some(value) = args.next() else {
                    return Err("--model requires a path".into());
                };
                model = Some(PathBuf::from(value));
            } else {
                return Err("usage: ds4-decode-long-indexed-attention --model FILE".into());
            }
        }
        let Some(model) = model else {
            return Err("usage: ds4-decode-long-indexed-attention --model FILE".into());
        };
        Ok(Self { model })
    }
}

struct BackendGuard;

impl Drop for BackendGuard {
    fn drop(&mut self) {
        unsafe {
            ds4_gpu::cleanup();
        }
    }
}

#[derive(Clone, Copy)]
struct Dims {
    hc_dim: u64,
    hc_mix_dim: u64,
    q_rank: u64,
    q_dim: u64,
    kv_dim: u64,
    group_dim: u64,
    rank: u64,
    low_dim: u64,
    shared_dim: u64,
    expert_in_dim: u64,
    expert_mid_dim: u64,
    down_in_dim: u64,
    routed_out_dim: u64,
}

impl Dims {
    fn new(layer0: &Ds4LayerWeights) -> Result<Self, Box<dyn std::error::Error>> {
        let _ = layer0;
        Ok(Self {
            hc_dim: u64::from(N_HC) * u64::from(N_EMBD),
            hc_mix_dim: 2 * u64::from(N_HC) + u64::from(N_HC) * u64::from(N_HC),
            q_rank: u64::from(N_LORA_Q),
            q_dim: u64::from(N_HEAD) * u64::from(N_HEAD_DIM),
            kv_dim: u64::from(N_HEAD_DIM),
            group_dim: u64::from(N_HEAD_DIM) * u64::from(N_HEAD / N_OUT_GROUP),
            rank: u64::from(N_LORA_O),
            low_dim: u64::from(N_OUT_GROUP) * u64::from(N_LORA_O),
            shared_dim: u64::from(N_FF_EXP),
            expert_in_dim: u64::from(N_EMBD),
            expert_mid_dim: u64::from(N_FF_EXP),
            down_in_dim: u64::from(N_FF_EXP),
            routed_out_dim: u64::from(N_EMBD),
        })
    }
}

struct DecodeState {
    cur_hc: Tensor,
    flat_hc: Tensor,
    hc_mix: Tensor,
    hc_split: Tensor,
    attn_cur: Tensor,
    attn_norm: Tensor,
    qr: Tensor,
    kv_raw: Tensor,
    qr_norm: Tensor,
    q: Tensor,
    kv: Tensor,
    raw_cache: Vec<Tensor>,
    layer_attn_comp_cache: Vec<Option<Tensor>>,
    layer_attn_state_kv: Vec<Option<Tensor>>,
    layer_attn_state_score: Vec<Option<Tensor>>,
    layer_index_comp_cache: Vec<Option<Tensor>>,
    layer_index_state_kv: Vec<Option<Tensor>>,
    layer_index_state_score: Vec<Option<Tensor>>,
    layer_n_comp: Vec<u32>,
    layer_n_index_comp: Vec<u32>,
    comp_kv_cur: Tensor,
    comp_sc_cur: Tensor,
    indexer_q: Tensor,
    indexer_weights: Tensor,
    indexer_scores: Tensor,
    comp_selected: Tensor,
    heads: Tensor,
    attn_low: Tensor,
    attn_out: Tensor,
    after_attn_hc: Tensor,
    ffn_cur: Tensor,
    ffn_norm: Tensor,
    router_logits: Tensor,
    router_selected: Tensor,
    router_weights: Tensor,
    router_probs: Tensor,
    routed_out: Tensor,
    routed_gate: Tensor,
    routed_up: Tensor,
    routed_mid: Tensor,
    routed_down: Tensor,
    shared_gate: Tensor,
    shared_up: Tensor,
    shared_mid: Tensor,
    shared_out: Tensor,
    after_ffn_hc: Tensor,
}

impl DecodeState {
    fn allocate(plan: GraphPlan, dims: Dims) -> Result<Self, Box<dyn std::error::Error>> {
        let raw_cache_bytes = byte_len(u64::from(plan.allocated_raw_cap) * dims.kv_dim)?;
        let mut raw_cache = Vec::with_capacity(N_LAYER);
        let mut layer_attn_comp_cache = Vec::with_capacity(N_LAYER);
        let mut layer_attn_state_kv = Vec::with_capacity(N_LAYER);
        let mut layer_attn_state_score = Vec::with_capacity(N_LAYER);
        let mut layer_index_comp_cache = Vec::with_capacity(N_LAYER);
        let mut layer_index_state_kv = Vec::with_capacity(N_LAYER);
        let mut layer_index_state_score = Vec::with_capacity(N_LAYER);
        for layer in 0..N_LAYER {
            raw_cache.push(
                Tensor::allocate(raw_cache_bytes)
                    .map_err(|err| format!("failed to allocate raw cache layer{layer}: {err}"))?,
            );
            match layer_compression(layer).ok_or("invalid layer")? {
                LayerCompression::Dense => {
                    layer_attn_comp_cache.push(None);
                    layer_attn_state_kv.push(None);
                    layer_attn_state_score.push(None);
                    layer_index_comp_cache.push(None);
                    layer_index_state_kv.push(None);
                    layer_index_state_score.push(None);
                }
                compression => {
                    let comp_cache_elements =
                        u64::from(plan.layer_comp_cap(compression)) * u64::from(N_HEAD_DIM);
                    let mut attn_comp_cache = Tensor::allocate(byte_len(comp_cache_elements)?)
                        .map_err(|err| {
                            format!("failed to allocate attn comp cache layer{layer}: {err}")
                        })?;
                    attn_comp_cache
                        .fill_f32(0.0, usize::try_from(comp_cache_elements)?)
                        .map_err(|err| {
                            format!("failed to zero attn comp cache layer{layer}: {err}")
                        })?;
                    let state_dim = attn_state_dim(compression);
                    let mut attn_state_kv =
                        Tensor::allocate(byte_len(state_dim)?).map_err(|err| {
                            format!("failed to allocate attn state kv layer{layer}: {err}")
                        })?;
                    attn_state_kv
                        .fill_f32(0.0, usize::try_from(state_dim)?)
                        .map_err(|err| {
                            format!("failed to zero attn state kv layer{layer}: {err}")
                        })?;
                    let mut attn_state_score =
                        Tensor::allocate(byte_len(state_dim)?).map_err(|err| {
                            format!("failed to allocate attn state score layer{layer}: {err}")
                        })?;
                    attn_state_score
                        .fill_f32(COMPRESSOR_SCORE_INIT, usize::try_from(state_dim)?)
                        .map_err(|err| {
                            format!("failed to fill attn state score layer{layer}: {err}")
                        })?;
                    layer_attn_comp_cache.push(Some(attn_comp_cache));
                    layer_attn_state_kv.push(Some(attn_state_kv));
                    layer_attn_state_score.push(Some(attn_state_score));
                    if compression == LayerCompression::Ratio4 {
                        let index_cache_elements = u64::from(plan.layer_comp_cap(compression))
                            * u64::from(N_INDEXER_HEAD_DIM);
                        let mut index_comp_cache =
                            Tensor::allocate(byte_len(index_cache_elements)?).map_err(|err| {
                                format!("failed to allocate index comp cache layer{layer}: {err}")
                            })?;
                        index_comp_cache
                            .fill_f32(0.0, usize::try_from(index_cache_elements)?)
                            .map_err(|err| {
                                format!("failed to zero index comp cache layer{layer}: {err}")
                            })?;
                        let index_dim = index_state_dim(compression);
                        let mut index_state_kv =
                            Tensor::allocate(byte_len(index_dim)?).map_err(|err| {
                                format!("failed to allocate index state kv layer{layer}: {err}")
                            })?;
                        index_state_kv
                            .fill_f32(0.0, usize::try_from(index_dim)?)
                            .map_err(|err| {
                                format!("failed to zero index state kv layer{layer}: {err}")
                            })?;
                        let mut index_state_score = Tensor::allocate(byte_len(index_dim)?)
                            .map_err(|err| {
                                format!("failed to allocate index state score layer{layer}: {err}")
                            })?;
                        index_state_score
                            .fill_f32(COMPRESSOR_SCORE_INIT, usize::try_from(index_dim)?)
                            .map_err(|err| {
                                format!("failed to fill index state score layer{layer}: {err}")
                            })?;
                        layer_index_comp_cache.push(Some(index_comp_cache));
                        layer_index_state_kv.push(Some(index_state_kv));
                        layer_index_state_score.push(Some(index_state_score));
                    } else {
                        layer_index_comp_cache.push(None);
                        layer_index_state_kv.push(None);
                        layer_index_state_score.push(None);
                    }
                }
            }
        }

        let down_in_dim = dims.down_in_dim;
        Ok(Self {
            cur_hc: alloc("cur_hc", dims.hc_dim)?,
            flat_hc: alloc("flat_hc", dims.hc_dim)?,
            hc_mix: alloc("hc_mix", dims.hc_mix_dim)?,
            hc_split: alloc("hc_split", dims.hc_mix_dim)?,
            attn_cur: alloc("attn_cur", u64::from(N_EMBD))?,
            attn_norm: alloc("attn_norm", u64::from(N_EMBD))?,
            qr: alloc("qr", dims.q_rank)?,
            kv_raw: alloc("kv_raw", dims.kv_dim)?,
            qr_norm: alloc("qr_norm", dims.q_rank)?,
            q: alloc("q", dims.q_dim)?,
            kv: alloc("kv", dims.kv_dim)?,
            raw_cache,
            layer_attn_comp_cache,
            layer_attn_state_kv,
            layer_attn_state_score,
            layer_index_comp_cache,
            layer_index_state_kv,
            layer_index_state_score,
            layer_n_comp: vec![0; N_LAYER],
            layer_n_index_comp: vec![0; N_LAYER],
            comp_kv_cur: alloc("comp_kv_cur", u64::from(2 * N_HEAD_DIM))?,
            comp_sc_cur: alloc("comp_sc_cur", u64::from(2 * N_HEAD_DIM))?,
            indexer_q: alloc(
                "indexer_q",
                u64::from(N_INDEXER_HEAD) * u64::from(N_INDEXER_HEAD_DIM),
            )?,
            indexer_weights: alloc("indexer_weights", u64::from(N_INDEXER_HEAD))?,
            indexer_scores: alloc("indexer_scores", u64::from(plan.comp_cap))?,
            comp_selected: Tensor::allocate(byte_len_i32(u64::from(N_INDEXER_TOP_K))?)
                .map_err(|err| format!("failed to allocate comp_selected: {err}"))?,
            heads: alloc("heads", dims.q_dim)?,
            attn_low: alloc("attn_low", dims.low_dim)?,
            attn_out: alloc("attn_out", u64::from(N_EMBD))?,
            after_attn_hc: alloc("after_attn_hc", dims.hc_dim)?,
            ffn_cur: alloc("ffn_cur", u64::from(N_EMBD))?,
            ffn_norm: alloc("ffn_norm", u64::from(N_EMBD))?,
            router_logits: alloc("router_logits", u64::from(N_EXPERT))?,
            router_selected: alloc("router_selected", u64::from(N_EXPERT_USED))?,
            router_weights: alloc("router_weights", u64::from(N_EXPERT_USED))?,
            router_probs: alloc("router_probs", u64::from(N_EXPERT))?,
            routed_out: alloc("routed_out", u64::from(N_EMBD))?,
            routed_gate: alloc("routed_gate", u64::from(N_EXPERT_USED) * down_in_dim)?,
            routed_up: alloc("routed_up", u64::from(N_EXPERT_USED) * down_in_dim)?,
            routed_mid: alloc("routed_mid", u64::from(N_EXPERT_USED) * down_in_dim)?,
            routed_down: alloc("routed_down", u64::from(N_EXPERT_USED) * u64::from(N_EMBD))?,
            shared_gate: alloc("shared_gate", dims.shared_dim)?,
            shared_up: alloc("shared_up", dims.shared_dim)?,
            shared_mid: alloc("shared_mid", dims.shared_dim)?,
            shared_out: alloc("shared_out", u64::from(N_EMBD))?,
            after_ffn_hc: alloc("after_ffn_hc", dims.hc_dim)?,
        })
    }
}

struct MappedModel {
    file: File,
    ptr: *mut c_void,
    size: u64,
}

impl MappedModel {
    fn open(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let size = file.metadata()?.len();
        if size == 0 || usize::try_from(size).is_err() {
            return Err("model file is empty or too large for mmap length".into());
        }
        let ptr = unsafe {
            mmap(
                std::ptr::null_mut(),
                size as usize,
                PROT_READ,
                MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr == MAP_FAILED {
            return Err(std::io::Error::last_os_error().into());
        }
        Ok(Self { file, ptr, size })
    }
}

impl Drop for MappedModel {
    fn drop(&mut self) {
        unsafe {
            let _ = munmap(self.ptr, self.size as usize);
        }
    }
}

struct TensorOutput {
    field: &'static str,
    bytes: u64,
    elements: u64,
    nonzero_elements: u64,
    fnv1a64: u64,
    samples: Vec<F32Sample>,
}

struct I32TensorOutput {
    field: &'static str,
    bytes: u64,
    elements: u64,
    nonzero_elements: u64,
    fnv1a64: u64,
    samples: Vec<I32Sample>,
}

enum Output {
    F32(TensorOutput),
    I32(I32TensorOutput),
}

#[derive(Clone, Copy)]
struct F32Sample {
    index: u64,
    value: f32,
}

#[derive(Clone, Copy)]
struct I32Sample {
    index: u64,
    value: i32,
}

fn parse_header_prefix(path: &Path) -> Result<(Gguf, u64), Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let file_size = file.metadata()?.len();
    let mut size = INITIAL_HEADER_READ.min(file_size);
    loop {
        file.seek(SeekFrom::Start(0))?;
        let mut bytes = vec![0u8; usize::try_from(size)?];
        file.read_exact(&mut bytes)?;
        match parse_gguf_allowing_missing_tensor_data(&bytes) {
            Ok(gguf) => return Ok((gguf, size)),
            Err(err) if err.message() == "truncated GGUF file" && size < file_size => {
                size = (size.saturating_mul(2)).min(file_size).min(MAX_HEADER_READ);
                if size == bytes.len() as u64 {
                    return Err(err.into());
                }
            }
            Err(err) => return Err(err.into()),
        }
    }
}

fn read_tensor_output(
    field: &'static str,
    tensor: &Tensor,
    elements: u64,
) -> Result<TensorOutput, Box<dyn std::error::Error>> {
    read_tensor_output_offset(field, tensor, 0, elements)
}

fn read_tensor_output_offset(
    field: &'static str,
    tensor: &Tensor,
    offset: u64,
    elements: u64,
) -> Result<TensorOutput, Box<dyn std::error::Error>> {
    let bytes = elements
        .checked_mul(4)
        .ok_or("tensor byte length overflow")?;
    if offset > tensor.byte_len() || bytes > tensor.byte_len() - offset {
        return Err(format!(
            "{field} tensor range drift: offset {offset}, bytes {bytes}, tensor {}",
            tensor.byte_len()
        )
        .into());
    }
    let mut data = vec![0u8; usize::try_from(bytes)?];
    tensor
        .read_bytes(offset, &mut data)
        .map_err(|err| format!("{field} readback failed: {err}"))?;
    Ok(TensorOutput {
        field,
        bytes,
        elements,
        nonzero_elements: count_nonzero_f32(&data)?,
        fnv1a64: fnv1a64(&data),
        samples: read_samples(&data, elements)?,
    })
}

fn read_tensor_i32_output(
    field: &'static str,
    tensor: &Tensor,
    elements: u64,
) -> Result<I32TensorOutput, Box<dyn std::error::Error>> {
    let bytes = elements
        .checked_mul(4)
        .ok_or("tensor byte length overflow")?;
    if bytes > tensor.byte_len() {
        return Err(format!(
            "{field} tensor range drift: bytes {bytes}, tensor {}",
            tensor.byte_len()
        )
        .into());
    }
    let mut data = vec![0u8; usize::try_from(bytes)?];
    tensor
        .read_bytes(0, &mut data)
        .map_err(|err| format!("{field} readback failed: {err}"))?;
    Ok(I32TensorOutput {
        field,
        bytes,
        elements,
        nonzero_elements: count_nonzero_i32(&data)?,
        fnv1a64: fnv1a64(&data),
        samples: read_i32_samples(&data, elements)?,
    })
}

fn read_samples(bytes: &[u8], elements: u64) -> Result<Vec<F32Sample>, String> {
    let mut samples = Vec::new();
    for index in sample_indices(elements) {
        let start = usize::try_from(index)
            .map_err(|_| "sample index too large".to_string())?
            .checked_mul(4)
            .ok_or_else(|| "sample index overflow".to_string())?;
        let chunk = bytes
            .get(start..start + 4)
            .ok_or_else(|| format!("sample index {index} out of range"))?;
        let value = f32::from_le_bytes(chunk.try_into().expect("four-byte chunk"));
        samples.push(F32Sample { index, value });
    }
    Ok(samples)
}

fn read_i32_samples(bytes: &[u8], elements: u64) -> Result<Vec<I32Sample>, String> {
    let mut samples = Vec::new();
    for index in sample_indices(elements) {
        let start = usize::try_from(index)
            .map_err(|_| "sample index too large".to_string())?
            .checked_mul(4)
            .ok_or_else(|| "sample index overflow".to_string())?;
        let chunk = bytes
            .get(start..start + 4)
            .ok_or_else(|| format!("sample index {index} out of range"))?;
        let value = i32::from_le_bytes(chunk.try_into().expect("four-byte chunk"));
        samples.push(I32Sample { index, value });
    }
    Ok(samples)
}

fn sample_indices(elements: u64) -> Vec<u64> {
    let raw = [
        0,
        1,
        elements / 2,
        if elements > 1 { elements - 2 } else { 0 },
        elements.saturating_sub(1),
    ];
    let mut out = Vec::new();
    for index in raw {
        if index >= elements || out.contains(&index) {
            continue;
        }
        out.push(index);
    }
    out
}

fn count_nonzero_f32(bytes: &[u8]) -> Result<u64, String> {
    if bytes.len() % 4 != 0 {
        return Err("tensor byte length is not f32-aligned".to_string());
    }
    let mut count = 0u64;
    for chunk in bytes.chunks_exact(4) {
        let value = f32::from_le_bytes(chunk.try_into().expect("four-byte chunk"));
        if value != 0.0 {
            count += 1;
        }
    }
    Ok(count)
}

fn count_nonzero_i32(bytes: &[u8]) -> Result<u64, String> {
    if bytes.len() % 4 != 0 {
        return Err("tensor byte length is not i32-aligned".to_string());
    }
    let mut count = 0u64;
    for chunk in bytes.chunks_exact(4) {
        let value = i32::from_le_bytes(chunk.try_into().expect("four-byte chunk"));
        if value != 0 {
            count += 1;
        }
    }
    Ok(count)
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn byte_len(elements: u64) -> Result<usize, Box<dyn std::error::Error>> {
    Ok(usize::try_from(
        elements
            .checked_mul(4)
            .ok_or("tensor byte length overflow")?,
    )?)
}

fn byte_len_i32(elements: u64) -> Result<usize, Box<dyn std::error::Error>> {
    byte_len(elements)
}

fn alloc(name: &str, elements: u64) -> Result<Tensor, Box<dyn std::error::Error>> {
    Tensor::allocate(byte_len(elements)?)
        .map_err(|err| format!("failed to allocate {name}: {err}").into())
}

fn routed_expert_row_bytes(tensor: &TensorInfo) -> Result<u64, Box<dyn std::error::Error>> {
    let dim0 = *tensor
        .dims
        .first()
        .ok_or("routed expert tensor has no input dimension")?;
    tensor_nbytes(tensor.type_id, dim0)
        .ok_or_else(|| format!("unsupported routed expert tensor type {}", tensor.type_id).into())
}

fn compression_coeff(compression: LayerCompression) -> u32 {
    match compression {
        LayerCompression::Dense => 0,
        LayerCompression::Ratio4 => 2,
        LayerCompression::Ratio128 => 1,
    }
}

fn attn_state_dim(compression: LayerCompression) -> u64 {
    let coeff = compression_coeff(compression);
    u64::from(coeff * N_HEAD_DIM) * u64::from(coeff * compression.ratio())
}

fn index_state_dim(compression: LayerCompression) -> u64 {
    let coeff = compression_coeff(compression);
    u64::from(coeff * N_INDEXER_HEAD_DIM) * u64::from(coeff * compression.ratio())
}

fn compressed_rope_attn_factor() -> f32 {
    1.0 / (1.0 + 0.1 * (1.0f32 / COMPRESS_ROPE_FREQ_SCALE).ln())
}

fn write_report(
    gguf: &Gguf,
    weights: &Ds4Weights,
    header_bytes_read: u64,
    mapped_size: u64,
    plan: GraphPlan,
    raw_row: u32,
    n_raw: u32,
    raw_start: u32,
    dims: Dims,
    state: &DecodeState,
    outputs: &[Output],
) {
    println!("{{");
    println!("  \"schema\": \"{SCHEMA}\",");
    println!("  \"case\": \"{CASE}\",");
    println!("  \"model\": {{");
    println!("    \"mapped_size\": {mapped_size},");
    println!("    \"header_bytes_read\": {header_bytes_read},");
    println!("    \"tensor_count\": {},", gguf.tensors.len());
    println!("    \"tensor_data_offset\": {},", gguf.tensor_data_offset);
    println!("    \"bound_layers\": {}", weights.layers.len());
    println!("  }},");
    println!("  \"operation\": {{");
    println!("    \"name\": \"rust_gpu_long_indexed_attention\",");
    println!(
        "    \"method\": \"decode_backend_execute_layer_x2051x43+final_layer2_indexed_mixed_attention\","
    );
    println!("    \"command_batch\": true,");
    println!("    \"synchronized\": true,");
    println!("    \"first_token\": 0,");
    println!("    \"last_token\": {FINAL_POSITION},");
    println!("    \"sequence_len\": {SEQUENCE_LEN},");
    println!("    \"final_position\": {FINAL_POSITION},");
    println!("    \"first_layer\": 0,");
    println!("    \"last_layer\": {INDEXED_LAYER},");
    println!("    \"indexed_layer\": {INDEXED_LAYER},");
    println!("    \"full_decode_tokens\": {},", SEQUENCE_LEN - 1);
    println!("    \"decoded_layers_per_full_token\": {N_LAYER},");
    println!("    \"final_decoded_layers\": {},", INDEXED_LAYER + 1);
    println!(
        "    \"total_decode_layer_calls\": {},",
        (SEQUENCE_LEN - 1) * N_LAYER as u32 + (INDEXED_LAYER as u32 + 1)
    );
    println!("    \"dense_layers\": 2,");
    println!("    \"ratio4_layers\": 21,");
    println!("    \"ratio128_layers\": 20,");
    println!("    \"allow_split_flush\": 1,");
    println!("    \"split_after_layer\": {SPLIT_AFTER_LAYER},");
    println!("    \"ctx_size\": {},", plan.ctx_size);
    println!("    \"prefill_cap\": {},", plan.prefill_cap);
    println!("    \"raw_cap\": {},", plan.allocated_raw_cap);
    println!("    \"raw_window\": {},", plan.raw_window);
    println!("    \"raw_row\": {raw_row},");
    println!("    \"raw_start\": {raw_start},");
    println!("    \"n_raw\": {n_raw},");
    println!("    \"indexer_top_k\": {N_INDEXER_TOP_K},");
    println!("    \"n_selected\": {N_INDEXER_TOP_K},");
    println!("    \"use_indexed_attention\": 1,");
    println!("    \"emit_compressed_row\": 1,");
    println!("    \"head_dim\": {N_HEAD_DIM},");
    println!("    \"heads_dim\": {},", dims.q_dim);
    println!("    \"n_embd\": {N_EMBD},");
    println!("    \"n_hc\": {N_HC},");
    println!("    \"hc_dim\": {},", dims.hc_dim);
    println!("    \"indexer_head\": {N_INDEXER_HEAD},");
    println!("    \"indexer_head_dim\": {N_INDEXER_HEAD_DIM},");
    println!(
        "    \"indexer_q_dim\": {},",
        u64::from(N_INDEXER_HEAD) * u64::from(N_INDEXER_HEAD_DIM)
    );
    println!(
        "    \"indexer_scores_dim\": {},",
        state.layer_n_index_comp[INDEXED_LAYER]
    );
    println!(
        "    \"layer2_comp_cap\": {},",
        plan.layer_comp_cap(LayerCompression::Ratio4)
    );
    println!(
        "    \"layer2_n_comp\": {},",
        state.layer_n_comp[INDEXED_LAYER]
    );
    println!(
        "    \"layer2_n_index_comp\": {},",
        state.layer_n_index_comp[INDEXED_LAYER]
    );
    println!(
        "    \"layer2_final_comp_row\": {},",
        state.layer_n_comp[INDEXED_LAYER] - 1
    );
    println!("    \"rms_eps\": {RMS_EPS},");
    println!("    \"hc_eps\": {HC_EPS}");
    println!("  }},");
    println!("  \"weights\": {{");
    write_weight("token_embd", "base.token_embd", &weights.token_embd, true);
    let layer2 = &weights.layers[INDEXED_LAYER];
    write_weight(
        "layer2_attn_sinks",
        "layer2.attn_sinks",
        &layer2.attn_sinks,
        true,
    );
    write_weight(
        "layer2_indexer_attn_q_b",
        "layer2.indexer_attn_q_b",
        layer2
            .indexer_attn_q_b
            .as_ref()
            .expect("layer2 indexer_attn_q_b"),
        true,
    );
    write_weight(
        "layer2_indexer_proj",
        "layer2.indexer_proj",
        layer2.indexer_proj.as_ref().expect("layer2 indexer_proj"),
        true,
    );
    write_weight(
        "layer2_attn_compressor_kv",
        "layer2.attn_compressor_kv",
        layer2
            .attn_compressor_kv
            .as_ref()
            .expect("layer2 attn_compressor_kv"),
        true,
    );
    write_weight(
        "layer2_attn_compressor_gate",
        "layer2.attn_compressor_gate",
        layer2
            .attn_compressor_gate
            .as_ref()
            .expect("layer2 attn_compressor_gate"),
        true,
    );
    write_weight(
        "layer2_indexer_compressor_kv",
        "layer2.indexer_compressor_kv",
        layer2
            .indexer_compressor_kv
            .as_ref()
            .expect("layer2 indexer_compressor_kv"),
        true,
    );
    write_weight(
        "layer2_indexer_compressor_gate",
        "layer2.indexer_compressor_gate",
        layer2
            .indexer_compressor_gate
            .as_ref()
            .expect("layer2 indexer_compressor_gate"),
        false,
    );
    println!("  }},");
    println!("  \"outputs\": {{");
    for (idx, output) in outputs.iter().enumerate() {
        write_any_output(output, idx + 1 != outputs.len());
    }
    println!("  }}");
    println!("}}");
}

fn write_weight(key: &str, role: &str, tensor: &TensorInfo, trailing_comma: bool) {
    println!("    \"{key}\": {{");
    println!("      \"role\": \"{role}\",");
    println!("      \"abs_offset\": {},", tensor.abs_offset);
    println!("      \"bytes\": {},", tensor.bytes);
    println!("      \"type\": {},", tensor.type_id);
    println!(
        "      \"type_name\": \"{}\"",
        tensor_type_name(tensor.type_id)
    );
    print!("    }}");
    if trailing_comma {
        print!(",");
    }
    println!();
}

fn write_output(output: &TensorOutput, trailing_comma: bool) {
    println!("    \"{}\": {{", output.field);
    println!("      \"field\": \"{}\",", output.field);
    println!("      \"bytes\": {},", output.bytes);
    println!("      \"elements\": {},", output.elements);
    println!("      \"nonzero_elements\": {},", output.nonzero_elements);
    println!("      \"fnv1a64\": \"{:016x}\",", output.fnv1a64);
    println!("      \"samples\": [");
    for (idx, sample) in output.samples.iter().enumerate() {
        if idx != 0 {
            println!(",");
        }
        print!("        {{\"index\": {}, \"value\": ", sample.index);
        print_json_f32(sample.value);
        print!("}}");
    }
    println!();
    println!("      ]");
    print!("    }}");
    if trailing_comma {
        print!(",");
    }
    println!();
}

fn write_i32_output(output: &I32TensorOutput, trailing_comma: bool) {
    println!("    \"{}\": {{", output.field);
    println!("      \"field\": \"{}\",", output.field);
    println!("      \"dtype\": \"i32\",");
    println!("      \"bytes\": {},", output.bytes);
    println!("      \"elements\": {},", output.elements);
    println!("      \"nonzero_elements\": {},", output.nonzero_elements);
    println!("      \"fnv1a64\": \"{:016x}\",", output.fnv1a64);
    println!("      \"samples\": [");
    for (idx, sample) in output.samples.iter().enumerate() {
        if idx != 0 {
            println!(",");
        }
        print!(
            "        {{\"index\": {}, \"value\": {}}}",
            sample.index, sample.value
        );
    }
    println!();
    println!("      ]");
    print!("    }}");
    if trailing_comma {
        print!(",");
    }
    println!();
}

fn write_any_output(output: &Output, trailing_comma: bool) {
    match output {
        Output::F32(output) => write_output(output, trailing_comma),
        Output::I32(output) => write_i32_output(output, trailing_comma),
    }
}

fn print_json_f32(value: f32) {
    if value.is_nan() {
        print!("\"nan\"");
    } else if value == f32::INFINITY {
        print!("\"inf\"");
    } else if value == f32::NEG_INFINITY {
        print!("\"-inf\"");
    } else {
        print!("{value:.9}");
    }
}
