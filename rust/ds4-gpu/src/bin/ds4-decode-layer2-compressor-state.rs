use ds4_gguf::{
    bind_ds4_weights, parse_gguf_allowing_missing_tensor_data, tensor_nbytes, tensor_type_name,
    Gguf, TensorInfo,
};
use ds4_gpu::decode_backend::{set_model_fd, set_model_map_range, DecodeBackend, ModelMap};
use ds4_gpu::graph_plan::{
    GraphPlan, HC_EPS, N_EMBD, N_EXPERT, N_EXPERT_USED, N_FF_EXP, N_HC, N_HC_SINKHORN_ITER, N_HEAD,
    N_HEAD_DIM, N_HEAD_KV, N_INDEXER_HEAD_DIM, N_LORA_O, N_LORA_Q, N_OUT_GROUP, N_ROT, N_VOCAB,
    RMS_EPS, ROPE_FREQ_BASE, ROPE_YARN_BETA_FAST, ROPE_YARN_BETA_SLOW,
};
use ds4_gpu::{initialize, synchronize, CommandBatch, Tensor};
use std::ffi::c_void;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::os::raw::c_int;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};

const SCHEMA: &str = "ds4.decode_layer2_compressor_state.v1";
const CASE: &str = "token0_layer2_compressor_state";
const TOKEN: u32 = 0;
const FIRST_LAYER: usize = 0;
const LAST_DENSE_LAYER: usize = 1;
const COMPRESSED_LAYER: usize = 2;
const COMPRESSION_RATIO: u32 = 4;
const COMPRESSOR_COEFFICIENT: u32 = 2;
const POSITION: u32 = 0;
const CTX_SIZE: u32 = 32_768;
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
        eprintln!("ds4-decode-layer2-compressor-state: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse()?;
    let (gguf, header_bytes_read) = parse_header_prefix(&args.model)?;
    let weights = bind_ds4_weights(&gguf)?;
    let layer0 = weights
        .layers
        .get(FIRST_LAYER)
        .ok_or("DS4 weight binding did not include layer 0")?;
    let layer1 = weights
        .layers
        .get(LAST_DENSE_LAYER)
        .ok_or("DS4 weight binding did not include layer 1")?;
    let layer2 = weights
        .layers
        .get(COMPRESSED_LAYER)
        .ok_or("DS4 weight binding did not include layer 2")?;
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
    let raw_row = POSITION % raw_cap;
    let n_raw = 1;
    let raw_start = 0;
    let hc_dim = u64::from(N_HC) * u64::from(N_EMBD);
    let hc_dim_u32 = u32::try_from(hc_dim).map_err(|_| "hc_dim does not fit in u32")?;
    let hc_mix_dim = 2 * u64::from(N_HC) + u64::from(N_HC) * u64::from(N_HC);
    let q_rank = u64::from(N_LORA_Q);
    let q_dim = u64::from(N_HEAD) * u64::from(N_HEAD_DIM);
    let kv_dim = u64::from(N_HEAD_DIM);
    let comp_width = u64::from(COMPRESSOR_COEFFICIENT) * u64::from(N_HEAD_DIM);
    let index_width = u64::from(COMPRESSOR_COEFFICIENT) * u64::from(N_INDEXER_HEAD_DIM);
    let attn_state_dim = comp_width * u64::from(COMPRESSOR_COEFFICIENT * COMPRESSION_RATIO);
    let index_state_dim = index_width * u64::from(COMPRESSOR_COEFFICIENT * COMPRESSION_RATIO);
    let layer_comp_cap = plan.layer_comp_cap(ds4_gpu::graph_plan::LayerCompression::Ratio4);
    let emit_compressed_row = (POSITION + 1) % COMPRESSION_RATIO == 0;
    let layer_n_comp = 0u32;
    let layer_n_index_comp = 0u32;
    let group_heads = N_HEAD / N_OUT_GROUP;
    let group_dim = u64::from(N_HEAD_DIM) * u64::from(group_heads);
    let rank = u64::from(N_LORA_O);
    let low_dim = u64::from(N_OUT_GROUP) * rank;
    let shared_dim = u64::from(N_FF_EXP);
    let expert_in_dim = u64::from(N_EMBD);
    let expert_mid_dim = u64::from(N_FF_EXP);
    let down_in_dim = u64::from(N_FF_EXP);
    let routed_out_dim = u64::from(N_EMBD);
    let gate_row_bytes = routed_expert_row_bytes(&layer0.ffn_gate_exps)?;
    let gate_expert_bytes = expert_mid_dim * gate_row_bytes;
    let down_row_bytes = routed_expert_row_bytes(&layer0.ffn_down_exps)?;
    let down_expert_bytes = routed_out_dim * down_row_bytes;
    let router_hash_rows = layer0
        .ffn_gate_tid2eid
        .as_ref()
        .and_then(|tensor| tensor.dims.get(1).copied())
        .unwrap_or(0);
    let layer1_router_hash_rows = layer1
        .ffn_gate_tid2eid
        .as_ref()
        .and_then(|tensor| tensor.dims.get(1).copied())
        .unwrap_or(0);
    let mut cur_hc = Tensor::allocate(byte_len(hc_dim)?)
        .map_err(|err| format!("failed to allocate cur_hc: {err}"))?;
    let mut flat_hc = Tensor::allocate(byte_len(hc_dim)?)
        .map_err(|err| format!("failed to allocate flat_hc: {err}"))?;
    let mut hc_mix = Tensor::allocate(byte_len(hc_mix_dim)?)
        .map_err(|err| format!("failed to allocate hc_mix: {err}"))?;
    let mut hc_split = Tensor::allocate(byte_len(hc_mix_dim)?)
        .map_err(|err| format!("failed to allocate hc_split: {err}"))?;
    let mut attn_cur = Tensor::allocate(byte_len(u64::from(N_EMBD))?)
        .map_err(|err| format!("failed to allocate attn_cur: {err}"))?;
    let mut attn_norm = Tensor::allocate(byte_len(u64::from(N_EMBD))?)
        .map_err(|err| format!("failed to allocate attn_norm: {err}"))?;
    let mut qr = Tensor::allocate(byte_len(q_rank)?)
        .map_err(|err| format!("failed to allocate qr: {err}"))?;
    let mut kv_raw = Tensor::allocate(byte_len(kv_dim)?)
        .map_err(|err| format!("failed to allocate kv_raw: {err}"))?;
    let mut qr_norm = Tensor::allocate(byte_len(q_rank)?)
        .map_err(|err| format!("failed to allocate qr_norm: {err}"))?;
    let mut q =
        Tensor::allocate(byte_len(q_dim)?).map_err(|err| format!("failed to allocate q: {err}"))?;
    let mut kv = Tensor::allocate(byte_len(kv_dim)?)
        .map_err(|err| format!("failed to allocate kv: {err}"))?;
    let mut raw_cache = Tensor::allocate(byte_len(u64::from(raw_cap) * kv_dim)?)
        .map_err(|err| format!("failed to allocate raw_cache: {err}"))?;
    let mut raw_cache_layer1 = Tensor::allocate(byte_len(u64::from(raw_cap) * kv_dim)?)
        .map_err(|err| format!("failed to allocate raw_cache_layer1: {err}"))?;
    let mut raw_cache_layer2 = Tensor::allocate(byte_len(u64::from(raw_cap) * kv_dim)?)
        .map_err(|err| format!("failed to allocate raw_cache_layer2: {err}"))?;
    let mut layer2_attn_comp_cache =
        Tensor::allocate(byte_len(u64::from(layer_comp_cap) * u64::from(N_HEAD_DIM))?)
            .map_err(|err| format!("failed to allocate layer2_attn_comp_cache: {err}"))?;
    let mut layer2_attn_state_kv = Tensor::allocate(byte_len(attn_state_dim)?)
        .map_err(|err| format!("failed to allocate layer2_attn_state_kv: {err}"))?;
    let mut layer2_attn_state_score = Tensor::allocate(byte_len(attn_state_dim)?)
        .map_err(|err| format!("failed to allocate layer2_attn_state_score: {err}"))?;
    let mut layer2_index_comp_cache = Tensor::allocate(byte_len(
        u64::from(layer_comp_cap) * u64::from(N_INDEXER_HEAD_DIM),
    )?)
    .map_err(|err| format!("failed to allocate layer2_index_comp_cache: {err}"))?;
    let mut layer2_index_state_kv = Tensor::allocate(byte_len(index_state_dim)?)
        .map_err(|err| format!("failed to allocate layer2_index_state_kv: {err}"))?;
    let mut layer2_index_state_score = Tensor::allocate(byte_len(index_state_dim)?)
        .map_err(|err| format!("failed to allocate layer2_index_state_score: {err}"))?;
    let mut comp_kv_cur = Tensor::allocate(byte_len(comp_width)?)
        .map_err(|err| format!("failed to allocate comp_kv_cur: {err}"))?;
    let mut comp_sc_cur = Tensor::allocate(byte_len(comp_width)?)
        .map_err(|err| format!("failed to allocate comp_sc_cur: {err}"))?;
    let mut heads = Tensor::allocate(byte_len(q_dim)?)
        .map_err(|err| format!("failed to allocate heads: {err}"))?;
    let mut attn_low = Tensor::allocate(byte_len(low_dim)?)
        .map_err(|err| format!("failed to allocate attn_low: {err}"))?;
    let mut attn_out = Tensor::allocate(byte_len(u64::from(N_EMBD))?)
        .map_err(|err| format!("failed to allocate attn_out: {err}"))?;
    let mut after_attn_hc = Tensor::allocate(byte_len(hc_dim)?)
        .map_err(|err| format!("failed to allocate after_attn_hc: {err}"))?;
    let mut ffn_cur = Tensor::allocate(byte_len(u64::from(N_EMBD))?)
        .map_err(|err| format!("failed to allocate ffn_cur: {err}"))?;
    let mut ffn_norm = Tensor::allocate(byte_len(u64::from(N_EMBD))?)
        .map_err(|err| format!("failed to allocate ffn_norm: {err}"))?;
    let mut router_logits = Tensor::allocate(byte_len(u64::from(N_EXPERT))?)
        .map_err(|err| format!("failed to allocate router_logits: {err}"))?;
    let mut router_selected = Tensor::allocate(byte_len(u64::from(N_EXPERT_USED))?)
        .map_err(|err| format!("failed to allocate router_selected: {err}"))?;
    let mut router_weights = Tensor::allocate(byte_len(u64::from(N_EXPERT_USED))?)
        .map_err(|err| format!("failed to allocate router_weights: {err}"))?;
    let mut router_probs = Tensor::allocate(byte_len(u64::from(N_EXPERT))?)
        .map_err(|err| format!("failed to allocate router_probs: {err}"))?;
    let mut routed_out = Tensor::allocate(byte_len(u64::from(N_EMBD))?)
        .map_err(|err| format!("failed to allocate routed_out: {err}"))?;
    let mut routed_gate = Tensor::allocate(byte_len(u64::from(N_EXPERT_USED) * down_in_dim)?)
        .map_err(|err| format!("failed to allocate routed_gate: {err}"))?;
    let mut routed_up = Tensor::allocate(byte_len(u64::from(N_EXPERT_USED) * down_in_dim)?)
        .map_err(|err| format!("failed to allocate routed_up: {err}"))?;
    let mut routed_mid = Tensor::allocate(byte_len(u64::from(N_EXPERT_USED) * down_in_dim)?)
        .map_err(|err| format!("failed to allocate routed_mid: {err}"))?;
    let mut routed_down = Tensor::allocate(byte_len(u64::from(N_EXPERT_USED) * u64::from(N_EMBD))?)
        .map_err(|err| format!("failed to allocate routed_down: {err}"))?;
    let mut shared_gate = Tensor::allocate(byte_len(shared_dim)?)
        .map_err(|err| format!("failed to allocate shared_gate: {err}"))?;
    let mut shared_up = Tensor::allocate(byte_len(shared_dim)?)
        .map_err(|err| format!("failed to allocate shared_up: {err}"))?;
    let mut shared_mid = Tensor::allocate(byte_len(shared_dim)?)
        .map_err(|err| format!("failed to allocate shared_mid: {err}"))?;
    let mut shared_out = Tensor::allocate(byte_len(u64::from(N_EMBD))?)
        .map_err(|err| format!("failed to allocate shared_out: {err}"))?;
    let mut after_ffn_hc = Tensor::allocate(byte_len(hc_dim)?)
        .map_err(|err| format!("failed to allocate after_ffn_hc: {err}"))?;
    layer2_attn_comp_cache
        .fill_f32(
            0.0,
            usize::try_from(u64::from(layer_comp_cap) * u64::from(N_HEAD_DIM))?,
        )
        .map_err(|err| format!("failed to zero layer2_attn_comp_cache: {err}"))?;
    layer2_attn_state_kv
        .fill_f32(0.0, usize::try_from(attn_state_dim)?)
        .map_err(|err| format!("failed to zero layer2_attn_state_kv: {err}"))?;
    layer2_attn_state_score
        .fill_f32(COMPRESSOR_SCORE_INIT, usize::try_from(attn_state_dim)?)
        .map_err(|err| format!("failed to fill layer2_attn_state_score: {err}"))?;
    layer2_index_comp_cache
        .fill_f32(
            0.0,
            usize::try_from(u64::from(layer_comp_cap) * u64::from(N_INDEXER_HEAD_DIM))?,
        )
        .map_err(|err| format!("failed to zero layer2_index_comp_cache: {err}"))?;
    layer2_index_state_kv
        .fill_f32(0.0, usize::try_from(index_state_dim)?)
        .map_err(|err| format!("failed to zero layer2_index_state_kv: {err}"))?;
    layer2_index_state_score
        .fill_f32(COMPRESSOR_SCORE_INIT, usize::try_from(index_state_dim)?)
        .map_err(|err| format!("failed to fill layer2_index_state_score: {err}"))?;

    let command_batch = CommandBatch::begin().map_err(|err| format!("begin failed: {err}"))?;
    backend
        .embed_token_hc(
            cur_hc.as_tensor_mut(),
            weights.token_embd.abs_offset,
            N_VOCAB,
            TOKEN,
            N_EMBD,
            N_HC,
        )
        .map_err(|err| format!("embed_token_hc failed: {err}"))?;
    backend
        .rms_norm_plain(
            flat_hc.as_tensor_mut(),
            cur_hc.as_tensor_ref(),
            hc_dim as u32,
            RMS_EPS,
        )
        .map_err(|err| format!("rms_norm_plain failed: {err}"))?;
    backend
        .matmul_f16(
            hc_mix.as_tensor_mut(),
            layer0.hc_attn_fn.abs_offset,
            hc_dim,
            hc_mix_dim,
            flat_hc.as_tensor_ref(),
            1,
        )
        .map_err(|err| format!("matmul_f16 failed: {err}"))?;
    backend
        .hc_split_weighted_sum_norm(
            attn_cur.as_tensor_mut(),
            attn_norm.as_tensor_mut(),
            hc_split.as_tensor_mut(),
            hc_mix.as_tensor_ref(),
            cur_hc.as_tensor_ref(),
            layer0.hc_attn_scale.abs_offset,
            layer0.hc_attn_base.abs_offset,
            layer0.attn_norm.abs_offset,
            N_EMBD,
            N_HC,
            N_HC_SINKHORN_ITER,
            HC_EPS,
            RMS_EPS,
        )
        .map_err(|err| format!("hc_split_weighted_sum_norm failed: {err}"))?;
    backend
        .matmul_q8_0(
            qr.as_tensor_mut(),
            layer0.attn_q_a.abs_offset,
            u64::from(N_EMBD),
            q_rank,
            attn_norm.as_tensor_ref(),
            1,
        )
        .map_err(|err| format!("attn_q_a matmul_q8_0 failed: {err}"))?;
    backend
        .matmul_q8_0(
            kv_raw.as_tensor_mut(),
            layer0.attn_kv.abs_offset,
            u64::from(N_EMBD),
            kv_dim,
            attn_norm.as_tensor_ref(),
            1,
        )
        .map_err(|err| format!("attn_kv matmul_q8_0 failed: {err}"))?;
    backend
        .dsv4_qkv_rms_norm_rows(
            qr_norm.as_tensor_mut(),
            qr.as_tensor_ref(),
            layer0.attn_q_a_norm.abs_offset,
            N_LORA_Q,
            kv.as_tensor_mut(),
            kv_raw.as_tensor_ref(),
            layer0.attn_kv_a_norm.abs_offset,
            N_HEAD_DIM,
            1,
            RMS_EPS,
        )
        .map_err(|err| format!("dsv4_qkv_rms_norm_rows failed: {err}"))?;
    backend
        .matmul_q8_0(
            q.as_tensor_mut(),
            layer0.attn_q_b.abs_offset,
            q_rank,
            q_dim,
            qr_norm.as_tensor_ref(),
            1,
        )
        .map_err(|err| format!("attn_q_b matmul_q8_0 failed: {err}"))?;
    backend
        .head_rms_norm(q.as_tensor_mut(), 1, N_HEAD, N_HEAD_DIM, RMS_EPS)
        .map_err(|err| format!("head_rms_norm failed: {err}"))?;
    backend
        .rope_tail(
            q.as_tensor_mut(),
            1,
            N_HEAD,
            N_HEAD_DIM,
            N_ROT,
            POSITION,
            0,
            false,
            ROPE_FREQ_BASE,
            1.0,
            0.0,
            1.0,
            ROPE_YARN_BETA_FAST,
            ROPE_YARN_BETA_SLOW,
        )
        .map_err(|err| format!("q rope_tail failed: {err}"))?;
    backend
        .rope_tail(
            kv.as_tensor_mut(),
            1,
            N_HEAD_KV,
            N_HEAD_DIM,
            N_ROT,
            POSITION,
            0,
            false,
            ROPE_FREQ_BASE,
            1.0,
            0.0,
            1.0,
            ROPE_YARN_BETA_FAST,
            ROPE_YARN_BETA_SLOW,
        )
        .map_err(|err| format!("kv rope_tail failed: {err}"))?;
    backend
        .kv_fp8_store_raw(
            kv.as_tensor_mut(),
            raw_cache.as_tensor_mut(),
            raw_cap,
            raw_row,
            N_HEAD_DIM,
            N_ROT,
        )
        .map_err(|err| format!("kv_fp8_store_raw failed: {err}"))?;
    backend
        .attention_decode_heads(
            heads.as_tensor_mut(),
            layer0.attn_sinks.abs_offset,
            q.as_tensor_ref(),
            raw_cache.as_tensor_ref(),
            n_raw,
            raw_cap,
            raw_start,
            None,
            0,
            None,
            0,
            N_HEAD,
            N_HEAD_DIM,
        )
        .map_err(|err| format!("attention_decode_heads failed: {err}"))?;
    backend
        .rope_tail(
            heads.as_tensor_mut(),
            1,
            N_HEAD,
            N_HEAD_DIM,
            N_ROT,
            POSITION,
            0,
            true,
            ROPE_FREQ_BASE,
            1.0,
            0.0,
            1.0,
            ROPE_YARN_BETA_FAST,
            ROPE_YARN_BETA_SLOW,
        )
        .map_err(|err| format!("heads inverse rope_tail failed: {err}"))?;
    backend
        .attention_output_low_q8(
            attn_low.as_tensor_mut(),
            layer0.attn_output_a.abs_offset,
            group_dim,
            rank,
            N_OUT_GROUP,
            heads.as_tensor_ref(),
        )
        .map_err(|err| format!("attention_output_low_q8 failed: {err}"))?;
    backend
        .matmul_q8_0_hc_expand(
            after_attn_hc.as_tensor_mut(),
            attn_out.as_tensor_mut(),
            layer0.attn_output_b.abs_offset,
            low_dim,
            u64::from(N_EMBD),
            attn_low.as_tensor_ref(),
            cur_hc.as_tensor_ref(),
            hc_split.as_tensor_ref(),
            N_EMBD,
            N_HC,
        )
        .map_err(|err| format!("matmul_q8_0_hc_expand failed: {err}"))?;
    backend
        .rms_norm_plain(
            flat_hc.as_tensor_mut(),
            after_attn_hc.as_tensor_ref(),
            hc_dim as u32,
            RMS_EPS,
        )
        .map_err(|err| format!("ffn rms_norm_plain failed: {err}"))?;
    backend
        .matmul_f16(
            hc_mix.as_tensor_mut(),
            layer0.hc_ffn_fn.abs_offset,
            hc_dim,
            hc_mix_dim,
            flat_hc.as_tensor_ref(),
            1,
        )
        .map_err(|err| format!("hc_ffn_fn matmul_f16 failed: {err}"))?;
    backend
        .hc_split_weighted_sum_norm(
            ffn_cur.as_tensor_mut(),
            ffn_norm.as_tensor_mut(),
            hc_split.as_tensor_mut(),
            hc_mix.as_tensor_ref(),
            after_attn_hc.as_tensor_ref(),
            layer0.hc_ffn_scale.abs_offset,
            layer0.hc_ffn_base.abs_offset,
            layer0.ffn_norm.abs_offset,
            N_EMBD,
            N_HC,
            N_HC_SINKHORN_ITER,
            HC_EPS,
            RMS_EPS,
        )
        .map_err(|err| format!("ffn hc_split_weighted_sum_norm failed: {err}"))?;
    backend
        .matmul_f16(
            router_logits.as_tensor_mut(),
            layer0.ffn_gate_inp.abs_offset,
            u64::from(N_EMBD),
            u64::from(N_EXPERT),
            ffn_norm.as_tensor_ref(),
            1,
        )
        .map_err(|err| format!("ffn_gate_inp matmul_f16 failed: {err}"))?;
    backend
        .router_select(
            router_selected.as_tensor_mut(),
            router_weights.as_tensor_mut(),
            router_probs.as_tensor_mut(),
            layer0
                .ffn_exp_probs_b
                .as_ref()
                .map(|tensor| tensor.abs_offset)
                .unwrap_or(0),
            layer0
                .ffn_gate_tid2eid
                .as_ref()
                .map(|tensor| tensor.abs_offset)
                .unwrap_or(0),
            u32::try_from(router_hash_rows)?,
            TOKEN,
            0,
            0,
            layer0.ffn_exp_probs_b.is_some(),
            layer0.ffn_gate_tid2eid.is_some(),
            router_logits.as_tensor_ref(),
        )
        .map_err(|err| format!("router_select failed: {err}"))?;
    backend
        .routed_moe_one(
            routed_out.as_tensor_mut(),
            routed_gate.as_tensor_mut(),
            routed_up.as_tensor_mut(),
            routed_mid.as_tensor_mut(),
            routed_down.as_tensor_mut(),
            layer0.ffn_gate_exps.abs_offset,
            layer0.ffn_up_exps.abs_offset,
            layer0.ffn_down_exps.abs_offset,
            layer0.ffn_gate_exps.type_id,
            layer0.ffn_down_exps.type_id,
            gate_expert_bytes,
            gate_row_bytes,
            down_expert_bytes,
            down_row_bytes,
            u32::try_from(expert_in_dim)?,
            u32::try_from(down_in_dim)?,
            u32::try_from(routed_out_dim)?,
            router_selected.as_tensor_ref(),
            router_weights.as_tensor_ref(),
            N_EXPERT_USED,
            SWIGLU_CLAMP_EXP,
            ffn_norm.as_tensor_ref(),
        )
        .map_err(|err| format!("routed_moe_one failed: {err}"))?;
    backend
        .shared_gate_up_swiglu_q8_0(
            shared_gate.as_tensor_mut(),
            shared_up.as_tensor_mut(),
            shared_mid.as_tensor_mut(),
            layer0.ffn_gate_shexp.abs_offset,
            layer0.ffn_up_shexp.abs_offset,
            u64::from(N_EMBD),
            shared_dim,
            ffn_norm.as_tensor_ref(),
            SWIGLU_CLAMP_EXP,
        )
        .map_err(|err| format!("shared_gate_up_swiglu_q8_0 failed: {err}"))?;
    backend
        .shared_down_hc_expand_q8_0(
            after_ffn_hc.as_tensor_mut(),
            shared_out.as_tensor_mut(),
            layer0.ffn_down_shexp.abs_offset,
            shared_dim,
            u64::from(N_EMBD),
            shared_mid.as_tensor_ref(),
            routed_out.as_tensor_ref(),
            after_attn_hc.as_tensor_ref(),
            hc_split.as_tensor_ref(),
            N_EMBD,
            N_HC,
        )
        .map_err(|err| format!("shared_down_hc_expand_q8_0 failed: {err}"))?;
    std::mem::swap(&mut cur_hc, &mut after_ffn_hc);
    backend
        .rms_norm_plain(
            flat_hc.as_tensor_mut(),
            cur_hc.as_tensor_ref(),
            hc_dim as u32,
            RMS_EPS,
        )
        .map_err(|err| format!("rms_norm_plain failed: {err}"))?;
    backend
        .matmul_f16(
            hc_mix.as_tensor_mut(),
            layer1.hc_attn_fn.abs_offset,
            hc_dim,
            hc_mix_dim,
            flat_hc.as_tensor_ref(),
            1,
        )
        .map_err(|err| format!("matmul_f16 failed: {err}"))?;
    backend
        .hc_split_weighted_sum_norm(
            attn_cur.as_tensor_mut(),
            attn_norm.as_tensor_mut(),
            hc_split.as_tensor_mut(),
            hc_mix.as_tensor_ref(),
            cur_hc.as_tensor_ref(),
            layer1.hc_attn_scale.abs_offset,
            layer1.hc_attn_base.abs_offset,
            layer1.attn_norm.abs_offset,
            N_EMBD,
            N_HC,
            N_HC_SINKHORN_ITER,
            HC_EPS,
            RMS_EPS,
        )
        .map_err(|err| format!("hc_split_weighted_sum_norm failed: {err}"))?;
    backend
        .matmul_q8_0(
            qr.as_tensor_mut(),
            layer1.attn_q_a.abs_offset,
            u64::from(N_EMBD),
            q_rank,
            attn_norm.as_tensor_ref(),
            1,
        )
        .map_err(|err| format!("attn_q_a matmul_q8_0 failed: {err}"))?;
    backend
        .matmul_q8_0(
            kv_raw.as_tensor_mut(),
            layer1.attn_kv.abs_offset,
            u64::from(N_EMBD),
            kv_dim,
            attn_norm.as_tensor_ref(),
            1,
        )
        .map_err(|err| format!("attn_kv matmul_q8_0 failed: {err}"))?;
    backend
        .dsv4_qkv_rms_norm_rows(
            qr_norm.as_tensor_mut(),
            qr.as_tensor_ref(),
            layer1.attn_q_a_norm.abs_offset,
            N_LORA_Q,
            kv.as_tensor_mut(),
            kv_raw.as_tensor_ref(),
            layer1.attn_kv_a_norm.abs_offset,
            N_HEAD_DIM,
            1,
            RMS_EPS,
        )
        .map_err(|err| format!("dsv4_qkv_rms_norm_rows failed: {err}"))?;
    backend
        .matmul_q8_0(
            q.as_tensor_mut(),
            layer1.attn_q_b.abs_offset,
            q_rank,
            q_dim,
            qr_norm.as_tensor_ref(),
            1,
        )
        .map_err(|err| format!("attn_q_b matmul_q8_0 failed: {err}"))?;
    backend
        .head_rms_norm(q.as_tensor_mut(), 1, N_HEAD, N_HEAD_DIM, RMS_EPS)
        .map_err(|err| format!("head_rms_norm failed: {err}"))?;
    backend
        .rope_tail(
            q.as_tensor_mut(),
            1,
            N_HEAD,
            N_HEAD_DIM,
            N_ROT,
            POSITION,
            0,
            false,
            ROPE_FREQ_BASE,
            1.0,
            0.0,
            1.0,
            ROPE_YARN_BETA_FAST,
            ROPE_YARN_BETA_SLOW,
        )
        .map_err(|err| format!("q rope_tail failed: {err}"))?;
    backend
        .rope_tail(
            kv.as_tensor_mut(),
            1,
            N_HEAD_KV,
            N_HEAD_DIM,
            N_ROT,
            POSITION,
            0,
            false,
            ROPE_FREQ_BASE,
            1.0,
            0.0,
            1.0,
            ROPE_YARN_BETA_FAST,
            ROPE_YARN_BETA_SLOW,
        )
        .map_err(|err| format!("kv rope_tail failed: {err}"))?;
    backend
        .kv_fp8_store_raw(
            kv.as_tensor_mut(),
            raw_cache_layer1.as_tensor_mut(),
            raw_cap,
            raw_row,
            N_HEAD_DIM,
            N_ROT,
        )
        .map_err(|err| format!("kv_fp8_store_raw failed: {err}"))?;
    backend
        .attention_decode_heads(
            heads.as_tensor_mut(),
            layer1.attn_sinks.abs_offset,
            q.as_tensor_ref(),
            raw_cache_layer1.as_tensor_ref(),
            n_raw,
            raw_cap,
            raw_start,
            None,
            0,
            None,
            0,
            N_HEAD,
            N_HEAD_DIM,
        )
        .map_err(|err| format!("attention_decode_heads failed: {err}"))?;
    backend
        .rope_tail(
            heads.as_tensor_mut(),
            1,
            N_HEAD,
            N_HEAD_DIM,
            N_ROT,
            POSITION,
            0,
            true,
            ROPE_FREQ_BASE,
            1.0,
            0.0,
            1.0,
            ROPE_YARN_BETA_FAST,
            ROPE_YARN_BETA_SLOW,
        )
        .map_err(|err| format!("heads inverse rope_tail failed: {err}"))?;
    backend
        .attention_output_low_q8(
            attn_low.as_tensor_mut(),
            layer1.attn_output_a.abs_offset,
            group_dim,
            rank,
            N_OUT_GROUP,
            heads.as_tensor_ref(),
        )
        .map_err(|err| format!("attention_output_low_q8 failed: {err}"))?;
    backend
        .matmul_q8_0_hc_expand(
            after_attn_hc.as_tensor_mut(),
            attn_out.as_tensor_mut(),
            layer1.attn_output_b.abs_offset,
            low_dim,
            u64::from(N_EMBD),
            attn_low.as_tensor_ref(),
            cur_hc.as_tensor_ref(),
            hc_split.as_tensor_ref(),
            N_EMBD,
            N_HC,
        )
        .map_err(|err| format!("matmul_q8_0_hc_expand failed: {err}"))?;
    backend
        .rms_norm_plain(
            flat_hc.as_tensor_mut(),
            after_attn_hc.as_tensor_ref(),
            hc_dim as u32,
            RMS_EPS,
        )
        .map_err(|err| format!("ffn rms_norm_plain failed: {err}"))?;
    backend
        .matmul_f16(
            hc_mix.as_tensor_mut(),
            layer1.hc_ffn_fn.abs_offset,
            hc_dim,
            hc_mix_dim,
            flat_hc.as_tensor_ref(),
            1,
        )
        .map_err(|err| format!("hc_ffn_fn matmul_f16 failed: {err}"))?;
    backend
        .hc_split_weighted_sum_norm(
            ffn_cur.as_tensor_mut(),
            ffn_norm.as_tensor_mut(),
            hc_split.as_tensor_mut(),
            hc_mix.as_tensor_ref(),
            after_attn_hc.as_tensor_ref(),
            layer1.hc_ffn_scale.abs_offset,
            layer1.hc_ffn_base.abs_offset,
            layer1.ffn_norm.abs_offset,
            N_EMBD,
            N_HC,
            N_HC_SINKHORN_ITER,
            HC_EPS,
            RMS_EPS,
        )
        .map_err(|err| format!("ffn hc_split_weighted_sum_norm failed: {err}"))?;
    backend
        .matmul_f16(
            router_logits.as_tensor_mut(),
            layer1.ffn_gate_inp.abs_offset,
            u64::from(N_EMBD),
            u64::from(N_EXPERT),
            ffn_norm.as_tensor_ref(),
            1,
        )
        .map_err(|err| format!("ffn_gate_inp matmul_f16 failed: {err}"))?;
    backend
        .router_select(
            router_selected.as_tensor_mut(),
            router_weights.as_tensor_mut(),
            router_probs.as_tensor_mut(),
            layer1
                .ffn_exp_probs_b
                .as_ref()
                .map(|tensor| tensor.abs_offset)
                .unwrap_or(0),
            layer1
                .ffn_gate_tid2eid
                .as_ref()
                .map(|tensor| tensor.abs_offset)
                .unwrap_or(0),
            u32::try_from(layer1_router_hash_rows)?,
            TOKEN,
            0,
            0,
            layer1.ffn_exp_probs_b.is_some(),
            layer1.ffn_gate_tid2eid.is_some(),
            router_logits.as_tensor_ref(),
        )
        .map_err(|err| format!("router_select failed: {err}"))?;
    backend
        .routed_moe_one(
            routed_out.as_tensor_mut(),
            routed_gate.as_tensor_mut(),
            routed_up.as_tensor_mut(),
            routed_mid.as_tensor_mut(),
            routed_down.as_tensor_mut(),
            layer1.ffn_gate_exps.abs_offset,
            layer1.ffn_up_exps.abs_offset,
            layer1.ffn_down_exps.abs_offset,
            layer1.ffn_gate_exps.type_id,
            layer1.ffn_down_exps.type_id,
            gate_expert_bytes,
            gate_row_bytes,
            down_expert_bytes,
            down_row_bytes,
            u32::try_from(expert_in_dim)?,
            u32::try_from(down_in_dim)?,
            u32::try_from(routed_out_dim)?,
            router_selected.as_tensor_ref(),
            router_weights.as_tensor_ref(),
            N_EXPERT_USED,
            SWIGLU_CLAMP_EXP,
            ffn_norm.as_tensor_ref(),
        )
        .map_err(|err| format!("routed_moe_one failed: {err}"))?;
    backend
        .shared_gate_up_swiglu_q8_0(
            shared_gate.as_tensor_mut(),
            shared_up.as_tensor_mut(),
            shared_mid.as_tensor_mut(),
            layer1.ffn_gate_shexp.abs_offset,
            layer1.ffn_up_shexp.abs_offset,
            u64::from(N_EMBD),
            shared_dim,
            ffn_norm.as_tensor_ref(),
            SWIGLU_CLAMP_EXP,
        )
        .map_err(|err| format!("shared_gate_up_swiglu_q8_0 failed: {err}"))?;
    backend
        .shared_down_hc_expand_q8_0(
            after_ffn_hc.as_tensor_mut(),
            shared_out.as_tensor_mut(),
            layer1.ffn_down_shexp.abs_offset,
            shared_dim,
            u64::from(N_EMBD),
            shared_mid.as_tensor_ref(),
            routed_out.as_tensor_ref(),
            after_attn_hc.as_tensor_ref(),
            hc_split.as_tensor_ref(),
            N_EMBD,
            N_HC,
        )
        .map_err(|err| format!("shared_down_hc_expand_q8_0 failed: {err}"))?;
    std::mem::swap(&mut cur_hc, &mut after_ffn_hc);
    let compressed_rope_attn_factor =
        1.0f32 / (1.0 + 0.1 * (1.0f32 / COMPRESS_ROPE_FREQ_SCALE).ln());
    backend
        .rms_norm_plain(
            flat_hc.as_tensor_mut(),
            cur_hc.as_tensor_ref(),
            hc_dim_u32,
            RMS_EPS,
        )
        .map_err(|err| format!("layer2 rms_norm_plain failed: {err}"))?;
    backend
        .matmul_f16(
            hc_mix.as_tensor_mut(),
            layer2.hc_attn_fn.abs_offset,
            hc_dim,
            hc_mix_dim,
            flat_hc.as_tensor_ref(),
            1,
        )
        .map_err(|err| format!("layer2 hc_attn_fn matmul_f16 failed: {err}"))?;
    backend
        .hc_split_weighted_sum_norm(
            attn_cur.as_tensor_mut(),
            attn_norm.as_tensor_mut(),
            hc_split.as_tensor_mut(),
            hc_mix.as_tensor_ref(),
            cur_hc.as_tensor_ref(),
            layer2.hc_attn_scale.abs_offset,
            layer2.hc_attn_base.abs_offset,
            layer2.attn_norm.abs_offset,
            N_EMBD,
            N_HC,
            N_HC_SINKHORN_ITER,
            HC_EPS,
            RMS_EPS,
        )
        .map_err(|err| format!("layer2 hc_split_weighted_sum_norm failed: {err}"))?;
    backend
        .matmul_q8_0(
            qr.as_tensor_mut(),
            layer2.attn_q_a.abs_offset,
            u64::from(N_EMBD),
            q_rank,
            attn_norm.as_tensor_ref(),
            1,
        )
        .map_err(|err| format!("layer2 attn_q_a matmul_q8_0 failed: {err}"))?;
    backend
        .matmul_q8_0(
            kv_raw.as_tensor_mut(),
            layer2.attn_kv.abs_offset,
            u64::from(N_EMBD),
            kv_dim,
            attn_norm.as_tensor_ref(),
            1,
        )
        .map_err(|err| format!("layer2 attn_kv matmul_q8_0 failed: {err}"))?;
    backend
        .dsv4_qkv_rms_norm_rows(
            qr_norm.as_tensor_mut(),
            qr.as_tensor_ref(),
            layer2.attn_q_a_norm.abs_offset,
            N_LORA_Q,
            kv.as_tensor_mut(),
            kv_raw.as_tensor_ref(),
            layer2.attn_kv_a_norm.abs_offset,
            N_HEAD_DIM,
            1,
            RMS_EPS,
        )
        .map_err(|err| format!("layer2 dsv4_qkv_rms_norm_rows failed: {err}"))?;
    backend
        .matmul_q8_0(
            q.as_tensor_mut(),
            layer2.attn_q_b.abs_offset,
            q_rank,
            q_dim,
            qr_norm.as_tensor_ref(),
            1,
        )
        .map_err(|err| format!("layer2 attn_q_b matmul_q8_0 failed: {err}"))?;
    backend
        .head_rms_norm(q.as_tensor_mut(), 1, N_HEAD, N_HEAD_DIM, RMS_EPS)
        .map_err(|err| format!("layer2 head_rms_norm failed: {err}"))?;
    backend
        .rope_tail(
            q.as_tensor_mut(),
            1,
            N_HEAD,
            N_HEAD_DIM,
            N_ROT,
            POSITION,
            ROPE_ORIG_CTX,
            false,
            COMPRESS_ROPE_FREQ_BASE,
            COMPRESS_ROPE_FREQ_SCALE,
            1.0,
            compressed_rope_attn_factor,
            ROPE_YARN_BETA_FAST,
            ROPE_YARN_BETA_SLOW,
        )
        .map_err(|err| format!("layer2 q rope_tail failed: {err}"))?;
    backend
        .rope_tail(
            kv.as_tensor_mut(),
            1,
            N_HEAD_KV,
            N_HEAD_DIM,
            N_ROT,
            POSITION,
            ROPE_ORIG_CTX,
            false,
            COMPRESS_ROPE_FREQ_BASE,
            COMPRESS_ROPE_FREQ_SCALE,
            1.0,
            compressed_rope_attn_factor,
            ROPE_YARN_BETA_FAST,
            ROPE_YARN_BETA_SLOW,
        )
        .map_err(|err| format!("layer2 kv rope_tail failed: {err}"))?;
    backend
        .kv_fp8_store_raw(
            kv.as_tensor_mut(),
            raw_cache_layer2.as_tensor_mut(),
            raw_cap,
            raw_row,
            N_HEAD_DIM,
            N_ROT,
        )
        .map_err(|err| format!("layer2 kv_fp8_store_raw failed: {err}"))?;
    backend
        .matmul_f16_pair(
            comp_kv_cur.as_tensor_mut(),
            comp_sc_cur.as_tensor_mut(),
            layer2
                .attn_compressor_kv
                .as_ref()
                .ok_or("layer2 attn_compressor_kv missing")?
                .abs_offset,
            layer2
                .attn_compressor_gate
                .as_ref()
                .ok_or("layer2 attn_compressor_gate missing")?
                .abs_offset,
            u64::from(N_EMBD),
            comp_width,
            attn_norm.as_tensor_ref(),
            1,
        )
        .map_err(|err| format!("layer2 attention compressor matmul_f16_pair failed: {err}"))?;
    backend
        .compressor_update(
            comp_kv_cur.as_tensor_ref(),
            comp_sc_cur.as_tensor_ref(),
            layer2_attn_state_kv.as_tensor_mut(),
            layer2_attn_state_score.as_tensor_mut(),
            layer2_attn_comp_cache.as_tensor_mut(),
            layer2
                .attn_compressor_ape
                .as_ref()
                .ok_or("layer2 attn_compressor_ape missing")?
                .abs_offset,
            layer2
                .attn_compressor_ape
                .as_ref()
                .ok_or("layer2 attn_compressor_ape missing")?
                .type_id,
            layer2
                .attn_compressor_norm
                .as_ref()
                .ok_or("layer2 attn_compressor_norm missing")?
                .abs_offset,
            layer2
                .attn_compressor_norm
                .as_ref()
                .ok_or("layer2 attn_compressor_norm missing")?
                .type_id,
            N_HEAD_DIM,
            COMPRESSION_RATIO,
            POSITION,
            layer_n_comp,
            N_ROT,
            ROPE_ORIG_CTX,
            COMPRESS_ROPE_FREQ_BASE,
            COMPRESS_ROPE_FREQ_SCALE,
            1.0,
            compressed_rope_attn_factor,
            ROPE_YARN_BETA_FAST,
            ROPE_YARN_BETA_SLOW,
            RMS_EPS,
        )
        .map_err(|err| format!("layer2 attention compressor_update failed: {err}"))?;
    backend
        .matmul_f16_pair(
            comp_kv_cur.as_tensor_mut(),
            comp_sc_cur.as_tensor_mut(),
            layer2
                .indexer_compressor_kv
                .as_ref()
                .ok_or("layer2 indexer_compressor_kv missing")?
                .abs_offset,
            layer2
                .indexer_compressor_gate
                .as_ref()
                .ok_or("layer2 indexer_compressor_gate missing")?
                .abs_offset,
            u64::from(N_EMBD),
            index_width,
            attn_norm.as_tensor_ref(),
            1,
        )
        .map_err(|err| format!("layer2 indexer compressor matmul_f16_pair failed: {err}"))?;
    backend
        .compressor_update(
            comp_kv_cur.as_tensor_ref(),
            comp_sc_cur.as_tensor_ref(),
            layer2_index_state_kv.as_tensor_mut(),
            layer2_index_state_score.as_tensor_mut(),
            layer2_index_comp_cache.as_tensor_mut(),
            layer2
                .indexer_compressor_ape
                .as_ref()
                .ok_or("layer2 indexer_compressor_ape missing")?
                .abs_offset,
            layer2
                .indexer_compressor_ape
                .as_ref()
                .ok_or("layer2 indexer_compressor_ape missing")?
                .type_id,
            layer2
                .indexer_compressor_norm
                .as_ref()
                .ok_or("layer2 indexer_compressor_norm missing")?
                .abs_offset,
            layer2
                .indexer_compressor_norm
                .as_ref()
                .ok_or("layer2 indexer_compressor_norm missing")?
                .type_id,
            N_INDEXER_HEAD_DIM,
            COMPRESSION_RATIO,
            POSITION,
            layer_n_index_comp,
            N_ROT,
            ROPE_ORIG_CTX,
            COMPRESS_ROPE_FREQ_BASE,
            COMPRESS_ROPE_FREQ_SCALE,
            1.0,
            compressed_rope_attn_factor,
            ROPE_YARN_BETA_FAST,
            ROPE_YARN_BETA_SLOW,
            RMS_EPS,
        )
        .map_err(|err| format!("layer2 indexer compressor_update failed: {err}"))?;
    command_batch
        .finish()
        .map_err(|err| format!("finish failed: {err}"))?;
    synchronize().map_err(|err| format!("synchronize failed: {err}"))?;

    let outputs = vec![
        read_tensor_output("after_layer1_hc", &cur_hc, hc_dim)?,
        read_tensor_output_offset(
            "layer2_raw_cache_row",
            &raw_cache_layer2,
            u64::from(raw_row) * kv_dim * 4,
            kv_dim,
        )?,
        read_tensor_output(
            "layer2_attn_state_kv",
            &layer2_attn_state_kv,
            attn_state_dim,
        )?,
        read_tensor_output(
            "layer2_attn_state_score",
            &layer2_attn_state_score,
            attn_state_dim,
        )?,
        read_tensor_output(
            "layer2_index_state_kv",
            &layer2_index_state_kv,
            index_state_dim,
        )?,
        read_tensor_output(
            "layer2_index_state_score",
            &layer2_index_state_score,
            index_state_dim,
        )?,
    ];
    write_report(
        &gguf,
        &weights,
        header_bytes_read,
        mapped.size,
        plan,
        raw_row,
        n_raw,
        raw_start,
        hc_dim,
        q_rank,
        q_dim,
        kv_dim,
        group_heads,
        group_dim,
        rank,
        low_dim,
        hc_mix_dim,
        shared_dim,
        expert_in_dim,
        expert_mid_dim,
        down_in_dim,
        routed_out_dim,
        gate_row_bytes,
        gate_expert_bytes,
        down_row_bytes,
        down_expert_bytes,
        router_hash_rows,
        layer_comp_cap,
        emit_compressed_row,
        layer_n_comp,
        layer_n_index_comp,
        comp_width,
        index_width,
        attn_state_dim,
        index_state_dim,
        compressed_rope_attn_factor,
        &outputs,
    );
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
                return Err("usage: ds4-decode-layer2-compressor-state --model FILE".into());
            }
        }
        let Some(model) = model else {
            return Err("usage: ds4-decode-layer2-compressor-state --model FILE".into());
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

#[derive(Clone, Copy)]
struct F32Sample {
    index: u64,
    value: f32,
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

fn routed_expert_row_bytes(tensor: &TensorInfo) -> Result<u64, Box<dyn std::error::Error>> {
    let dim0 = *tensor
        .dims
        .first()
        .ok_or("routed expert tensor has no input dimension")?;
    tensor_nbytes(tensor.type_id, dim0)
        .ok_or_else(|| format!("unsupported routed expert tensor type {}", tensor.type_id).into())
}

#[allow(clippy::too_many_arguments)]
fn write_report(
    gguf: &Gguf,
    weights: &ds4_gguf::Ds4Weights,
    header_bytes_read: u64,
    mapped_size: u64,
    plan: GraphPlan,
    raw_row: u32,
    n_raw: u32,
    raw_start: u32,
    hc_dim: u64,
    q_rank: u64,
    q_dim: u64,
    kv_dim: u64,
    group_heads: u32,
    group_dim: u64,
    rank: u64,
    low_dim: u64,
    hc_mix_dim: u64,
    shared_dim: u64,
    expert_in_dim: u64,
    expert_mid_dim: u64,
    down_in_dim: u64,
    routed_out_dim: u64,
    gate_row_bytes: u64,
    gate_expert_bytes: u64,
    down_row_bytes: u64,
    down_expert_bytes: u64,
    router_hash_rows: u64,
    layer_comp_cap: u32,
    emit_compressed_row: bool,
    layer_n_comp: u32,
    layer_n_index_comp: u32,
    comp_width: u64,
    index_width: u64,
    attn_state_dim: u64,
    index_state_dim: u64,
    compressed_rope_attn_factor: f32,
    outputs: &[TensorOutput],
) {
    let layer0 = &weights.layers[FIRST_LAYER];
    let layer2 = &weights.layers[COMPRESSED_LAYER];
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
    println!("    \"name\": \"ds4_gpu_layer2_compressor_state\",");
    println!("    \"method\": \"dense_layer0+swap+dense_layer1+swap+layer2_qkv_rope+kv_fp8_store_raw+attn_compressor_update+indexer_compressor_update\",");
    println!("    \"command_batch\": true,");
    println!("    \"synchronized\": true,");
    println!("    \"token\": {TOKEN},");
    println!("    \"first_layer\": {FIRST_LAYER},");
    println!("    \"last_dense_layer\": {LAST_DENSE_LAYER},");
    println!("    \"compressed_layer\": {COMPRESSED_LAYER},");
    println!("    \"position\": {POSITION},");
    println!("    \"decoded_layers\": 3,");
    println!("    \"dense_layers\": 2,");
    println!("    \"compression\": \"ratio4\",");
    println!("    \"compression_ratio\": {COMPRESSION_RATIO},");
    println!("    \"compressor_coefficient\": {COMPRESSOR_COEFFICIENT},");
    println!(
        "    \"emit_compressed_row\": {},",
        u8::from(emit_compressed_row)
    );
    println!("    \"ctx_size\": {},", plan.ctx_size);
    println!("    \"prefill_cap\": {},", plan.prefill_cap);
    println!("    \"raw_cap\": {},", plan.allocated_raw_cap);
    println!("    \"raw_window\": {},", plan.raw_window);
    println!("    \"raw_row\": {raw_row},");
    println!("    \"raw_start\": {raw_start},");
    println!("    \"n_raw\": {n_raw},");
    println!("    \"layer_comp_cap\": {layer_comp_cap},");
    println!("    \"layer_n_comp\": {layer_n_comp},");
    println!("    \"layer_n_index_comp\": {layer_n_index_comp},");
    println!("    \"n_embd\": {N_EMBD},");
    println!("    \"n_hc\": {N_HC},");
    println!("    \"hc_dim\": {hc_dim},");
    println!("    \"q_rank\": {q_rank},");
    println!("    \"q_dim\": {q_dim},");
    println!("    \"head_dim\": {kv_dim},");
    println!("    \"indexer_head_dim\": {N_INDEXER_HEAD_DIM},");
    println!("    \"comp_width\": {comp_width},");
    println!("    \"index_width\": {index_width},");
    println!("    \"attn_state_dim\": {attn_state_dim},");
    println!("    \"index_state_dim\": {index_state_dim},");
    println!("    \"n_head\": {N_HEAD},");
    println!("    \"n_head_kv\": {N_HEAD_KV},");
    println!("    \"n_rot\": {N_ROT},");
    println!("    \"n_groups\": {N_OUT_GROUP},");
    println!("    \"group_heads\": {group_heads},");
    println!("    \"group_dim\": {group_dim},");
    println!("    \"rank\": {rank},");
    println!("    \"low_dim\": {low_dim},");
    println!("    \"rope_freq_base\": {COMPRESS_ROPE_FREQ_BASE},");
    println!("    \"rope_freq_scale\": {COMPRESS_ROPE_FREQ_SCALE},");
    println!("    \"rope_ext_factor\": 1,");
    println!("    \"rope_attn_factor\": {compressed_rope_attn_factor},");
    println!("    \"rope_yarn_beta_fast\": {ROPE_YARN_BETA_FAST},");
    println!("    \"rope_yarn_beta_slow\": {ROPE_YARN_BETA_SLOW},");
    println!("    \"rope_orig_ctx\": {ROPE_ORIG_CTX},");
    println!("    \"hc_mix_dim\": {hc_mix_dim},");
    println!("    \"shared_dim\": {shared_dim},");
    println!("    \"expert_in_dim\": {expert_in_dim},");
    println!("    \"expert_mid_dim\": {expert_mid_dim},");
    println!("    \"down_in_dim\": {down_in_dim},");
    println!("    \"routed_out_dim\": {routed_out_dim},");
    println!("    \"n_expert\": {N_EXPERT},");
    println!("    \"n_expert_used\": {N_EXPERT_USED},");
    println!("    \"gate_row_bytes\": {gate_row_bytes},");
    println!("    \"gate_expert_bytes\": {gate_expert_bytes},");
    println!("    \"down_row_bytes\": {down_row_bytes},");
    println!("    \"down_expert_bytes\": {down_expert_bytes},");
    println!(
        "    \"router_has_bias\": {},",
        layer0.ffn_exp_probs_b.is_some()
    );
    println!(
        "    \"router_hash_mode\": {},",
        layer0.ffn_gate_tid2eid.is_some()
    );
    println!("    \"router_hash_rows\": {router_hash_rows},");
    println!("    \"router_n_expert_groups\": 0,");
    println!("    \"router_n_group_used\": 0,");
    println!("    \"swiglu_clamp_exp\": {SWIGLU_CLAMP_EXP},");
    println!("    \"rms_eps\": {RMS_EPS},");
    println!("    \"hc_eps\": {HC_EPS}");
    println!("  }},");
    println!("  \"weights\": {{");
    write_weight("token_embd", "base.token_embd", &weights.token_embd, true);
    write_weight(
        "layer2_hc_attn_fn",
        "layer2.hc_attn_fn",
        &layer2.hc_attn_fn,
        true,
    );
    write_weight(
        "layer2_hc_attn_scale",
        "layer2.hc_attn_scale",
        &layer2.hc_attn_scale,
        true,
    );
    write_weight(
        "layer2_hc_attn_base",
        "layer2.hc_attn_base",
        &layer2.hc_attn_base,
        true,
    );
    write_weight(
        "layer2_attn_norm",
        "layer2.attn_norm",
        &layer2.attn_norm,
        true,
    );
    write_weight("layer2_attn_q_a", "layer2.attn_q_a", &layer2.attn_q_a, true);
    write_weight("layer2_attn_kv", "layer2.attn_kv", &layer2.attn_kv, true);
    write_weight(
        "layer2_attn_q_a_norm",
        "layer2.attn_q_a_norm",
        &layer2.attn_q_a_norm,
        true,
    );
    write_weight(
        "layer2_attn_kv_a_norm",
        "layer2.attn_kv_a_norm",
        &layer2.attn_kv_a_norm,
        true,
    );
    write_weight("layer2_attn_q_b", "layer2.attn_q_b", &layer2.attn_q_b, true);
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
        "layer2_attn_compressor_ape",
        "layer2.attn_compressor_ape",
        layer2
            .attn_compressor_ape
            .as_ref()
            .expect("layer2 attn_compressor_ape"),
        true,
    );
    write_weight(
        "layer2_attn_compressor_norm",
        "layer2.attn_compressor_norm",
        layer2
            .attn_compressor_norm
            .as_ref()
            .expect("layer2 attn_compressor_norm"),
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
        true,
    );
    write_weight(
        "layer2_indexer_compressor_ape",
        "layer2.indexer_compressor_ape",
        layer2
            .indexer_compressor_ape
            .as_ref()
            .expect("layer2 indexer_compressor_ape"),
        true,
    );
    write_weight(
        "layer2_indexer_compressor_norm",
        "layer2.indexer_compressor_norm",
        layer2
            .indexer_compressor_norm
            .as_ref()
            .expect("layer2 indexer_compressor_norm"),
        false,
    );
    println!("  }},");
    println!("  \"outputs\": {{");
    for (idx, output) in outputs.iter().enumerate() {
        write_output(output, idx + 1 != outputs.len());
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

fn print_json_f32(value: f32) {
    if value.is_nan() {
        print!("\"nan\"");
    } else if value.is_infinite() {
        if value.is_sign_negative() {
            print!("\"-inf\"");
        } else {
            print!("\"inf\"");
        }
    } else {
        print!("{value}");
    }
}
