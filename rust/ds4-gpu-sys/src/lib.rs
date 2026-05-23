#![no_std]

use core::ffi::{c_char, c_int, c_void};
use core::marker::PhantomData;

#[repr(C)]
pub struct Ds4GpuTensor {
    _private: [u8; 0],
    _marker: PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

unsafe extern "C" {
    pub fn ds4_gpu_init() -> c_int;
    pub fn ds4_gpu_cleanup();

    pub fn ds4_gpu_tensor_alloc(bytes: u64) -> *mut Ds4GpuTensor;
    pub fn ds4_gpu_tensor_alloc_managed(bytes: u64) -> *mut Ds4GpuTensor;
    pub fn ds4_gpu_tensor_view(
        base: *const Ds4GpuTensor,
        offset: u64,
        bytes: u64,
    ) -> *mut Ds4GpuTensor;
    pub fn ds4_gpu_tensor_free(tensor: *mut Ds4GpuTensor);
    pub fn ds4_gpu_tensor_bytes(tensor: *const Ds4GpuTensor) -> u64;
    pub fn ds4_gpu_tensor_contents(tensor: *mut Ds4GpuTensor) -> *mut c_void;
    pub fn ds4_gpu_tensor_fill_f32(tensor: *mut Ds4GpuTensor, value: f32, count: u64) -> c_int;
    pub fn ds4_gpu_tensor_write(
        tensor: *mut Ds4GpuTensor,
        offset: u64,
        data: *const c_void,
        bytes: u64,
    ) -> c_int;
    pub fn ds4_gpu_tensor_read(
        tensor: *const Ds4GpuTensor,
        offset: u64,
        data: *mut c_void,
        bytes: u64,
    ) -> c_int;
    pub fn ds4_gpu_tensor_copy(
        dst: *mut Ds4GpuTensor,
        dst_offset: u64,
        src: *const Ds4GpuTensor,
        src_offset: u64,
        bytes: u64,
    ) -> c_int;

    pub fn ds4_gpu_begin_commands() -> c_int;
    pub fn ds4_gpu_flush_commands() -> c_int;
    pub fn ds4_gpu_end_commands() -> c_int;
    pub fn ds4_gpu_synchronize() -> c_int;

    pub fn ds4_gpu_set_model_map(model_map: *const c_void, model_size: u64) -> c_int;
    pub fn ds4_gpu_set_model_fd(fd: c_int) -> c_int;
    pub fn ds4_gpu_set_model_map_range(
        model_map: *const c_void,
        model_size: u64,
        map_offset: u64,
        map_size: u64,
    ) -> c_int;
    pub fn ds4_gpu_cache_model_range(
        model_map: *const c_void,
        model_size: u64,
        offset: u64,
        bytes: u64,
        label: *const c_char,
    ) -> c_int;
    pub fn ds4_gpu_cache_q8_f16_range(
        model_map: *const c_void,
        model_size: u64,
        offset: u64,
        bytes: u64,
        in_dim: u64,
        out_dim: u64,
        label: *const c_char,
    ) -> c_int;
    pub fn ds4_gpu_should_use_managed_kv_cache(kv_cache_bytes: u64, context_bytes: u64) -> c_int;
    pub fn ds4_gpu_set_quality(quality: bool);
    pub fn ds4_gpu_print_memory_report(label: *const c_char);

    pub fn ds4_gpu_embed_token_hc_tensor(
        out_hc: *mut Ds4GpuTensor,
        model_map: *const c_void,
        model_size: u64,
        weight_offset: u64,
        n_vocab: u32,
        token: u32,
        n_embd: u32,
        n_hc: u32,
    ) -> c_int;

    pub fn ds4_gpu_embed_tokens_hc_tensor(
        out_hc: *mut Ds4GpuTensor,
        tokens: *const Ds4GpuTensor,
        model_map: *const c_void,
        model_size: u64,
        weight_offset: u64,
        n_vocab: u32,
        n_tokens: u32,
        n_embd: u32,
        n_hc: u32,
    ) -> c_int;

    pub fn ds4_gpu_indexer_score_one_tensor(
        scores: *mut Ds4GpuTensor,
        q: *const Ds4GpuTensor,
        weights: *const Ds4GpuTensor,
        index_comp: *const Ds4GpuTensor,
        n_comp: u32,
        n_head: u32,
        head_dim: u32,
        scale: f32,
    ) -> c_int;

    pub fn ds4_gpu_indexer_scores_prefill_tensor(
        scores: *mut Ds4GpuTensor,
        q: *const Ds4GpuTensor,
        weights: *const Ds4GpuTensor,
        index_comp: *const Ds4GpuTensor,
        n_comp: u32,
        n_tokens: u32,
        n_head: u32,
        head_dim: u32,
        ratio: u32,
        scale: f32,
    ) -> c_int;

    pub fn ds4_gpu_indexer_scores_decode_batch_tensor(
        scores: *mut Ds4GpuTensor,
        q: *const Ds4GpuTensor,
        weights: *const Ds4GpuTensor,
        index_comp: *const Ds4GpuTensor,
        n_comp: u32,
        n_tokens: u32,
        pos0: u32,
        n_head: u32,
        head_dim: u32,
        ratio: u32,
        scale: f32,
    ) -> c_int;

    pub fn ds4_gpu_indexer_topk_tensor(
        selected: *mut Ds4GpuTensor,
        scores: *const Ds4GpuTensor,
        n_comp: u32,
        n_tokens: u32,
        top_k: u32,
    ) -> c_int;

    pub fn ds4_gpu_dsv4_topk_mask_tensor(
        mask: *mut Ds4GpuTensor,
        topk: *const Ds4GpuTensor,
        n_comp: u32,
        n_tokens: u32,
        top_k: u32,
    ) -> c_int;

    pub fn ds4_gpu_matmul_q8_0_tensor(
        out: *mut Ds4GpuTensor,
        model_map: *const c_void,
        model_size: u64,
        weight_offset: u64,
        in_dim: u64,
        out_dim: u64,
        x: *const Ds4GpuTensor,
        n_tok: u64,
    ) -> c_int;

    pub fn ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
        gate: *mut Ds4GpuTensor,
        up: *mut Ds4GpuTensor,
        mid: *mut Ds4GpuTensor,
        model_map: *const c_void,
        model_size: u64,
        gate_offset: u64,
        up_offset: u64,
        in_dim: u64,
        out_dim: u64,
        x: *const Ds4GpuTensor,
        clamp: f32,
    ) -> c_int;

    pub fn ds4_gpu_matmul_f16_tensor(
        out: *mut Ds4GpuTensor,
        model_map: *const c_void,
        model_size: u64,
        weight_offset: u64,
        in_dim: u64,
        out_dim: u64,
        x: *const Ds4GpuTensor,
        n_tok: u64,
    ) -> c_int;

    pub fn ds4_gpu_matmul_f16_pair_tensor(
        out_a: *mut Ds4GpuTensor,
        out_b: *mut Ds4GpuTensor,
        model_map: *const c_void,
        model_size: u64,
        weight_a_offset: u64,
        weight_b_offset: u64,
        in_dim: u64,
        out_dim: u64,
        x: *const Ds4GpuTensor,
        n_tok: u64,
    ) -> c_int;

    pub fn ds4_gpu_matmul_f32_tensor(
        out: *mut Ds4GpuTensor,
        model_map: *const c_void,
        model_size: u64,
        weight_offset: u64,
        in_dim: u64,
        out_dim: u64,
        x: *const Ds4GpuTensor,
        n_tok: u64,
    ) -> c_int;

    pub fn ds4_gpu_repeat_hc_tensor(
        out: *mut Ds4GpuTensor,
        row: *const Ds4GpuTensor,
        n_embd: u32,
        n_hc: u32,
    ) -> c_int;

    pub fn ds4_gpu_rms_norm_plain_tensor(
        out: *mut Ds4GpuTensor,
        x: *const Ds4GpuTensor,
        n: u32,
        eps: f32,
    ) -> c_int;

    pub fn ds4_gpu_rms_norm_plain_rows_tensor(
        out: *mut Ds4GpuTensor,
        x: *const Ds4GpuTensor,
        n: u32,
        rows: u32,
        eps: f32,
    ) -> c_int;

    pub fn ds4_gpu_rms_norm_weight_tensor(
        out: *mut Ds4GpuTensor,
        x: *const Ds4GpuTensor,
        model_map: *const c_void,
        model_size: u64,
        weight_offset: u64,
        n: u32,
        eps: f32,
    ) -> c_int;

    pub fn ds4_gpu_rms_norm_weight_rows_tensor(
        out: *mut Ds4GpuTensor,
        x: *const Ds4GpuTensor,
        model_map: *const c_void,
        model_size: u64,
        weight_offset: u64,
        n: u32,
        rows: u32,
        eps: f32,
    ) -> c_int;

    pub fn ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
        q_out: *mut Ds4GpuTensor,
        q: *const Ds4GpuTensor,
        model_map: *const c_void,
        model_size: u64,
        q_weight_offset: u64,
        q_n: u32,
        kv_out: *mut Ds4GpuTensor,
        kv: *const Ds4GpuTensor,
        kv_weight_offset: u64,
        kv_n: u32,
        rows: u32,
        eps: f32,
    ) -> c_int;

    pub fn ds4_gpu_head_rms_norm_tensor(
        x: *mut Ds4GpuTensor,
        n_tok: u32,
        n_head: u32,
        head_dim: u32,
        eps: f32,
    ) -> c_int;

    pub fn ds4_gpu_dsv4_fp8_kv_quantize_tensor(
        x: *mut Ds4GpuTensor,
        n_tok: u32,
        head_dim: u32,
        n_rot: u32,
    ) -> c_int;

    pub fn ds4_gpu_dsv4_indexer_qat_tensor(
        x: *mut Ds4GpuTensor,
        n_rows: u32,
        head_dim: u32,
    ) -> c_int;

    pub fn ds4_gpu_rope_tail_tensor(
        x: *mut Ds4GpuTensor,
        n_tok: u32,
        n_head: u32,
        head_dim: u32,
        n_rot: u32,
        pos0: u32,
        n_ctx_orig: u32,
        inverse: bool,
        freq_base: f32,
        freq_scale: f32,
        ext_factor: f32,
        attn_factor: f32,
        beta_fast: f32,
        beta_slow: f32,
    ) -> c_int;

    pub fn ds4_gpu_kv_fp8_store_raw_tensor(
        kv: *mut Ds4GpuTensor,
        raw_cache: *mut Ds4GpuTensor,
        raw_cap: u32,
        row: u32,
        head_dim: u32,
        n_rot: u32,
    ) -> c_int;

    pub fn ds4_gpu_store_raw_kv_tensor(
        raw_cache: *mut Ds4GpuTensor,
        kv: *const Ds4GpuTensor,
        raw_cap: u32,
        row: u32,
        head_dim: u32,
    ) -> c_int;

    pub fn ds4_gpu_store_raw_kv_batch_tensor(
        raw_cache: *mut Ds4GpuTensor,
        kv: *const Ds4GpuTensor,
        raw_cap: u32,
        pos0: u32,
        n_tokens: u32,
        head_dim: u32,
    ) -> c_int;

    pub fn ds4_gpu_compressor_update_tensor(
        kv_cur: *const Ds4GpuTensor,
        sc_cur: *const Ds4GpuTensor,
        state_kv: *mut Ds4GpuTensor,
        state_score: *mut Ds4GpuTensor,
        comp_cache: *mut Ds4GpuTensor,
        model_map: *const c_void,
        model_size: u64,
        ape_offset: u64,
        ape_type: u32,
        norm_offset: u64,
        norm_type: u32,
        head_dim: u32,
        ratio: u32,
        pos: u32,
        comp_row: u32,
        n_rot: u32,
        n_ctx_orig: u32,
        freq_base: f32,
        freq_scale: f32,
        ext_factor: f32,
        attn_factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        rms_eps: f32,
    ) -> c_int;

    pub fn ds4_gpu_compressor_store_batch_tensor(
        kv: *const Ds4GpuTensor,
        sc: *const Ds4GpuTensor,
        state_kv: *mut Ds4GpuTensor,
        state_score: *mut Ds4GpuTensor,
        model_map: *const c_void,
        model_size: u64,
        ape_offset: u64,
        ape_type: u32,
        head_dim: u32,
        ratio: u32,
        pos0: u32,
        n_tokens: u32,
    ) -> c_int;

    pub fn ds4_gpu_compressor_prefill_tensor(
        comp_cache: *mut Ds4GpuTensor,
        state_kv: *mut Ds4GpuTensor,
        state_score: *mut Ds4GpuTensor,
        kv: *const Ds4GpuTensor,
        sc: *const Ds4GpuTensor,
        model_map: *const c_void,
        model_size: u64,
        ape_offset: u64,
        ape_type: u32,
        norm_offset: u64,
        norm_type: u32,
        head_dim: u32,
        ratio: u32,
        pos0: u32,
        n_tokens: u32,
        n_rot: u32,
        n_ctx_orig: u32,
        quantize_fp8: bool,
        freq_base: f32,
        freq_scale: f32,
        ext_factor: f32,
        attn_factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        rms_eps: f32,
    ) -> c_int;

    pub fn ds4_gpu_compressor_prefill_ratio4_replay_tensor(
        comp_cache: *mut Ds4GpuTensor,
        state_kv: *mut Ds4GpuTensor,
        state_score: *mut Ds4GpuTensor,
        kv: *const Ds4GpuTensor,
        sc: *const Ds4GpuTensor,
        model_map: *const c_void,
        model_size: u64,
        ape_offset: u64,
        ape_type: u32,
        norm_offset: u64,
        norm_type: u32,
        head_dim: u32,
        pos0: u32,
        n_tokens: u32,
        n_rot: u32,
        n_ctx_orig: u32,
        quantize_fp8: bool,
        freq_base: f32,
        freq_scale: f32,
        ext_factor: f32,
        attn_factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        rms_eps: f32,
    ) -> c_int;

    pub fn ds4_gpu_compressor_prefill_state_ratio4_tensor(
        state_kv: *mut Ds4GpuTensor,
        state_score: *mut Ds4GpuTensor,
        kv_tail: *const Ds4GpuTensor,
        sc_tail: *const Ds4GpuTensor,
        model_map: *const c_void,
        model_size: u64,
        ape_offset: u64,
        ape_type: u32,
        head_dim: u32,
        pos0: u32,
    ) -> c_int;

    pub fn ds4_gpu_attention_decode_heads_tensor(
        heads: *mut Ds4GpuTensor,
        model_map: *const c_void,
        model_size: u64,
        sinks_offset: u64,
        q: *const Ds4GpuTensor,
        raw_kv: *const Ds4GpuTensor,
        n_raw: u32,
        raw_cap: u32,
        raw_start: u32,
        comp_kv: *const Ds4GpuTensor,
        n_comp: u32,
        comp_mask: *const Ds4GpuTensor,
        use_mask: u32,
        n_head: u32,
        head_dim: u32,
    ) -> c_int;

    pub fn ds4_gpu_attention_prefill_raw_heads_tensor(
        heads: *mut Ds4GpuTensor,
        model_map: *const c_void,
        model_size: u64,
        sinks_offset: u64,
        q: *const Ds4GpuTensor,
        raw_kv: *const Ds4GpuTensor,
        n_tokens: u32,
        window: u32,
        n_head: u32,
        head_dim: u32,
    ) -> c_int;

    pub fn ds4_gpu_attention_decode_raw_batch_heads_tensor(
        heads: *mut Ds4GpuTensor,
        model_map: *const c_void,
        model_size: u64,
        sinks_offset: u64,
        q: *const Ds4GpuTensor,
        raw_kv: *const Ds4GpuTensor,
        n_tokens: u32,
        pos0: u32,
        n_raw: u32,
        raw_cap: u32,
        raw_start: u32,
        window: u32,
        n_head: u32,
        head_dim: u32,
    ) -> c_int;

    pub fn ds4_gpu_attention_decode_mixed_batch_heads_tensor(
        heads: *mut Ds4GpuTensor,
        model_map: *const c_void,
        model_size: u64,
        sinks_offset: u64,
        q: *const Ds4GpuTensor,
        raw_kv: *const Ds4GpuTensor,
        comp_kv: *const Ds4GpuTensor,
        comp_mask: *const Ds4GpuTensor,
        use_comp_mask: u32,
        n_tokens: u32,
        pos0: u32,
        n_raw: u32,
        raw_cap: u32,
        raw_start: u32,
        n_comp: u32,
        window: u32,
        ratio: u32,
        n_head: u32,
        head_dim: u32,
    ) -> c_int;

    pub fn ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
        heads: *mut Ds4GpuTensor,
        model_map: *const c_void,
        model_size: u64,
        sinks_offset: u64,
        q: *const Ds4GpuTensor,
        raw_kv: *const Ds4GpuTensor,
        comp_kv: *const Ds4GpuTensor,
        topk: *const Ds4GpuTensor,
        n_tokens: u32,
        pos0: u32,
        n_raw: u32,
        raw_cap: u32,
        raw_start: u32,
        n_comp: u32,
        top_k: u32,
        window: u32,
        ratio: u32,
        n_head: u32,
        head_dim: u32,
    ) -> c_int;

    pub fn ds4_gpu_attention_prefill_static_mixed_heads_tensor(
        heads: *mut Ds4GpuTensor,
        model_map: *const c_void,
        model_size: u64,
        sinks_offset: u64,
        q: *const Ds4GpuTensor,
        raw_kv: *const Ds4GpuTensor,
        comp_kv: *const Ds4GpuTensor,
        n_tokens: u32,
        n_comp: u32,
        window: u32,
        ratio: u32,
        n_head: u32,
        head_dim: u32,
    ) -> c_int;

    pub fn ds4_gpu_attention_prefill_masked_mixed_heads_tensor(
        heads: *mut Ds4GpuTensor,
        model_map: *const c_void,
        model_size: u64,
        sinks_offset: u64,
        q: *const Ds4GpuTensor,
        raw_kv: *const Ds4GpuTensor,
        comp_kv: *const Ds4GpuTensor,
        comp_mask: *const Ds4GpuTensor,
        n_tokens: u32,
        n_comp: u32,
        window: u32,
        ratio: u32,
        n_head: u32,
        head_dim: u32,
    ) -> c_int;

    pub fn ds4_gpu_attention_output_q8_batch_tensor(
        out: *mut Ds4GpuTensor,
        low: *mut Ds4GpuTensor,
        group_tmp: *mut Ds4GpuTensor,
        low_tmp: *mut Ds4GpuTensor,
        model_map: *const c_void,
        model_size: u64,
        out_a_offset: u64,
        out_b_offset: u64,
        group_dim: u64,
        rank: u64,
        n_groups: u32,
        out_dim: u64,
        heads: *const Ds4GpuTensor,
        n_tokens: u32,
    ) -> c_int;

    pub fn ds4_gpu_attention_output_low_q8_tensor(
        low: *mut Ds4GpuTensor,
        model_map: *const c_void,
        model_size: u64,
        out_a_offset: u64,
        group_dim: u64,
        rank: u64,
        n_groups: u32,
        heads: *const Ds4GpuTensor,
    ) -> c_int;

    pub fn ds4_gpu_swiglu_tensor(
        out: *mut Ds4GpuTensor,
        gate: *const Ds4GpuTensor,
        up: *const Ds4GpuTensor,
        n: u32,
        clamp: f32,
        weight: f32,
    ) -> c_int;

    pub fn ds4_gpu_add_tensor(
        out: *mut Ds4GpuTensor,
        a: *const Ds4GpuTensor,
        b: *const Ds4GpuTensor,
        n: u32,
    ) -> c_int;

    pub fn ds4_gpu_directional_steering_project_tensor(
        x: *mut Ds4GpuTensor,
        directions: *const Ds4GpuTensor,
        layer: u32,
        width: u32,
        rows: u32,
        scale: f32,
    ) -> c_int;

    pub fn ds4_gpu_router_select_tensor(
        selected: *mut Ds4GpuTensor,
        weights: *mut Ds4GpuTensor,
        probs: *mut Ds4GpuTensor,
        model_map: *const c_void,
        model_size: u64,
        bias_offset: u64,
        hash_offset: u64,
        hash_rows: u32,
        token: u32,
        n_expert_groups: u32,
        n_group_used: u32,
        has_bias: bool,
        hash_mode: bool,
        logits: *const Ds4GpuTensor,
    ) -> c_int;

    pub fn ds4_gpu_router_select_batch_tensor(
        selected: *mut Ds4GpuTensor,
        weights: *mut Ds4GpuTensor,
        probs: *mut Ds4GpuTensor,
        model_map: *const c_void,
        model_size: u64,
        bias_offset: u64,
        hash_offset: u64,
        hash_rows: u32,
        n_expert_groups: u32,
        n_group_used: u32,
        has_bias: bool,
        hash_mode: bool,
        logits: *const Ds4GpuTensor,
        tokens: *const Ds4GpuTensor,
        n_tokens: u32,
    ) -> c_int;

    pub fn ds4_gpu_routed_moe_one_tensor(
        out: *mut Ds4GpuTensor,
        gate: *mut Ds4GpuTensor,
        up: *mut Ds4GpuTensor,
        mid: *mut Ds4GpuTensor,
        experts: *mut Ds4GpuTensor,
        model_map: *const c_void,
        model_size: u64,
        gate_offset: u64,
        up_offset: u64,
        down_offset: u64,
        gate_type: u32,
        down_type: u32,
        gate_expert_bytes: u64,
        gate_row_bytes: u64,
        down_expert_bytes: u64,
        down_row_bytes: u64,
        expert_in_dim: u32,
        expert_mid_dim: u32,
        out_dim: u32,
        selected: *const Ds4GpuTensor,
        weights: *const Ds4GpuTensor,
        n_expert: u32,
        clamp: f32,
        x: *const Ds4GpuTensor,
    ) -> c_int;

    pub fn ds4_gpu_routed_moe_batch_tensor(
        out: *mut Ds4GpuTensor,
        gate: *mut Ds4GpuTensor,
        up: *mut Ds4GpuTensor,
        mid: *mut Ds4GpuTensor,
        experts: *mut Ds4GpuTensor,
        model_map: *const c_void,
        model_size: u64,
        gate_offset: u64,
        up_offset: u64,
        down_offset: u64,
        gate_type: u32,
        down_type: u32,
        gate_expert_bytes: u64,
        gate_row_bytes: u64,
        down_expert_bytes: u64,
        down_row_bytes: u64,
        expert_in_dim: u32,
        expert_mid_dim: u32,
        out_dim: u32,
        selected: *const Ds4GpuTensor,
        weights: *const Ds4GpuTensor,
        n_expert: u32,
        clamp: f32,
        x: *const Ds4GpuTensor,
        n_tokens: u32,
        mid_is_f16: *mut bool,
    ) -> c_int;

    pub fn ds4_gpu_hc_split_sinkhorn_tensor(
        out: *mut Ds4GpuTensor,
        mix: *const Ds4GpuTensor,
        model_map: *const c_void,
        model_size: u64,
        scale_offset: u64,
        base_offset: u64,
        n_hc: u32,
        sinkhorn_iters: u32,
        eps: f32,
    ) -> c_int;

    pub fn ds4_gpu_hc_weighted_sum_tensor(
        out: *mut Ds4GpuTensor,
        residual_hc: *const Ds4GpuTensor,
        weights: *const Ds4GpuTensor,
        n_embd: u32,
        n_hc: u32,
    ) -> c_int;

    pub fn ds4_gpu_hc_weighted_sum_split_tensor(
        out: *mut Ds4GpuTensor,
        residual_hc: *const Ds4GpuTensor,
        split: *const Ds4GpuTensor,
        n_embd: u32,
        n_hc: u32,
    ) -> c_int;

    pub fn ds4_gpu_hc_split_weighted_sum_tensor(
        out: *mut Ds4GpuTensor,
        split: *mut Ds4GpuTensor,
        mix: *const Ds4GpuTensor,
        residual_hc: *const Ds4GpuTensor,
        model_map: *const c_void,
        model_size: u64,
        scale_offset: u64,
        base_offset: u64,
        n_embd: u32,
        n_hc: u32,
        sinkhorn_iters: u32,
        eps: f32,
    ) -> c_int;

    pub fn ds4_gpu_hc_split_weighted_sum_norm_tensor(
        out: *mut Ds4GpuTensor,
        norm_out: *mut Ds4GpuTensor,
        split: *mut Ds4GpuTensor,
        mix: *const Ds4GpuTensor,
        residual_hc: *const Ds4GpuTensor,
        model_map: *const c_void,
        model_size: u64,
        scale_offset: u64,
        base_offset: u64,
        norm_weight_offset: u64,
        n_embd: u32,
        n_hc: u32,
        sinkhorn_iters: u32,
        eps: f32,
        norm_eps: f32,
    ) -> c_int;

    pub fn ds4_gpu_output_hc_weights_tensor(
        out: *mut Ds4GpuTensor,
        pre: *const Ds4GpuTensor,
        model_map: *const c_void,
        model_size: u64,
        scale_offset: u64,
        base_offset: u64,
        n_hc: u32,
        eps: f32,
    ) -> c_int;

    pub fn ds4_gpu_hc_expand_tensor(
        out_hc: *mut Ds4GpuTensor,
        block_out: *const Ds4GpuTensor,
        residual_hc: *const Ds4GpuTensor,
        post: *const Ds4GpuTensor,
        comb: *const Ds4GpuTensor,
        n_embd: u32,
        n_hc: u32,
    ) -> c_int;

    pub fn ds4_gpu_hc_expand_split_tensor(
        out_hc: *mut Ds4GpuTensor,
        block_out: *const Ds4GpuTensor,
        residual_hc: *const Ds4GpuTensor,
        split: *const Ds4GpuTensor,
        n_embd: u32,
        n_hc: u32,
    ) -> c_int;

    pub fn ds4_gpu_hc_expand_add_split_tensor(
        out_hc: *mut Ds4GpuTensor,
        block_out: *const Ds4GpuTensor,
        block_add: *const Ds4GpuTensor,
        residual_hc: *const Ds4GpuTensor,
        split: *const Ds4GpuTensor,
        n_embd: u32,
        n_hc: u32,
    ) -> c_int;

    pub fn ds4_gpu_shared_down_hc_expand_q8_0_tensor(
        out_hc: *mut Ds4GpuTensor,
        shared_out: *mut Ds4GpuTensor,
        model_map: *const c_void,
        model_size: u64,
        weight_offset: u64,
        in_dim: u64,
        out_dim: u64,
        shared_mid: *const Ds4GpuTensor,
        routed_out: *const Ds4GpuTensor,
        residual_hc: *const Ds4GpuTensor,
        split: *const Ds4GpuTensor,
        n_embd: u32,
        n_hc: u32,
    ) -> c_int;

    pub fn ds4_gpu_matmul_q8_0_hc_expand_tensor(
        out_hc: *mut Ds4GpuTensor,
        block_out: *mut Ds4GpuTensor,
        model_map: *const c_void,
        model_size: u64,
        weight_offset: u64,
        in_dim: u64,
        out_dim: u64,
        x: *const Ds4GpuTensor,
        residual_hc: *const Ds4GpuTensor,
        split: *const Ds4GpuTensor,
        n_embd: u32,
        n_hc: u32,
    ) -> c_int;
}

#[cfg(test)]
mod tests {
    use super::Ds4GpuTensor;

    #[test]
    fn opaque_tensor_marker_is_zero_sized() {
        assert_eq!(core::mem::size_of::<Ds4GpuTensor>(), 0);
        assert_eq!(core::mem::align_of::<Ds4GpuTensor>(), 1);
    }
}
