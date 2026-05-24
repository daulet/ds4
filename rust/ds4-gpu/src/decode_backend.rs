//! Safe wrappers for the default one-token decode backend primitives.
//!
//! M10.5c3 keeps scheduling out of Rust.  This module only encapsulates the
//! FFI calls that the default fused decode path needs so the scheduler added
//! later can call named safe methods instead of raw `ds4_gpu_*` symbols.

use core::ffi::{c_int, c_void};
use core::marker::PhantomData;
use core::ptr;

#[cfg(all(target_os = "linux", feature = "cuda-backend"))]
use core::ffi::CStr;

use crate::{sys, GpuError, GpuStatus, TensorMut, TensorRef};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeFacadeOperation {
    pub operation: &'static str,
    pub method: &'static str,
    pub tensor_args: &'static [&'static str],
}

pub const DEFAULT_DECODE_FACADE_OPERATIONS: &[DecodeFacadeOperation] = &[
    DecodeFacadeOperation {
        operation: "ds4_gpu_embed_token_hc_tensor",
        method: "embed_token_hc",
        tensor_args: &["out_hc"],
    },
    DecodeFacadeOperation {
        operation: "ds4_gpu_rms_norm_plain_tensor",
        method: "rms_norm_plain",
        tensor_args: &["out", "x"],
    },
    DecodeFacadeOperation {
        operation: "ds4_gpu_matmul_f16_tensor",
        method: "matmul_f16",
        tensor_args: &["out", "x"],
    },
    DecodeFacadeOperation {
        operation: "ds4_gpu_hc_split_weighted_sum_norm_tensor",
        method: "hc_split_weighted_sum_norm",
        tensor_args: &["out", "norm_out", "split", "mix", "residual_hc"],
    },
    DecodeFacadeOperation {
        operation: "ds4_gpu_rms_norm_weight_tensor",
        method: "rms_norm_weight",
        tensor_args: &["out", "x"],
    },
    DecodeFacadeOperation {
        operation: "ds4_gpu_matmul_q8_0_tensor",
        method: "matmul_q8_0",
        tensor_args: &["out", "x"],
    },
    DecodeFacadeOperation {
        operation: "ds4_gpu_dsv4_qkv_rms_norm_rows_tensor",
        method: "dsv4_qkv_rms_norm_rows",
        tensor_args: &["q_out", "q", "kv_out", "kv"],
    },
    DecodeFacadeOperation {
        operation: "ds4_gpu_head_rms_norm_tensor",
        method: "head_rms_norm",
        tensor_args: &["x"],
    },
    DecodeFacadeOperation {
        operation: "ds4_gpu_rope_tail_tensor",
        method: "rope_tail",
        tensor_args: &["x"],
    },
    DecodeFacadeOperation {
        operation: "ds4_gpu_kv_fp8_store_raw_tensor",
        method: "kv_fp8_store_raw",
        tensor_args: &["kv", "raw_cache"],
    },
    DecodeFacadeOperation {
        operation: "ds4_gpu_matmul_f16_pair_tensor",
        method: "matmul_f16_pair",
        tensor_args: &["out_a", "out_b", "x"],
    },
    DecodeFacadeOperation {
        operation: "ds4_gpu_compressor_update_tensor",
        method: "compressor_update",
        tensor_args: &["kv_cur", "sc_cur", "state_kv", "state_score", "comp_cache"],
    },
    DecodeFacadeOperation {
        operation: "ds4_gpu_dsv4_fp8_kv_quantize_tensor",
        method: "dsv4_fp8_kv_quantize",
        tensor_args: &["x"],
    },
    DecodeFacadeOperation {
        operation: "ds4_gpu_dsv4_indexer_qat_tensor",
        method: "dsv4_indexer_qat",
        tensor_args: &["x"],
    },
    DecodeFacadeOperation {
        operation: "ds4_gpu_indexer_score_one_tensor",
        method: "indexer_score_one",
        tensor_args: &["scores", "q", "weights", "index_comp"],
    },
    DecodeFacadeOperation {
        operation: "ds4_gpu_indexer_topk_tensor",
        method: "indexer_topk",
        tensor_args: &["selected", "scores"],
    },
    DecodeFacadeOperation {
        operation: "ds4_gpu_attention_indexed_mixed_batch_heads_tensor",
        method: "attention_indexed_mixed_batch_heads",
        tensor_args: &["heads", "q", "raw_kv", "comp_kv", "topk"],
    },
    DecodeFacadeOperation {
        operation: "ds4_gpu_attention_decode_heads_tensor",
        method: "attention_decode_heads",
        tensor_args: &["heads", "q", "raw_kv", "comp_kv", "comp_mask"],
    },
    DecodeFacadeOperation {
        operation: "ds4_gpu_attention_output_low_q8_tensor",
        method: "attention_output_low_q8",
        tensor_args: &["low", "heads"],
    },
    DecodeFacadeOperation {
        operation: "ds4_gpu_matmul_q8_0_hc_expand_tensor",
        method: "matmul_q8_0_hc_expand",
        tensor_args: &["out_hc", "block_out", "x", "residual_hc", "split"],
    },
    DecodeFacadeOperation {
        operation: "ds4_gpu_router_select_tensor",
        method: "router_select",
        tensor_args: &["selected", "weights", "probs", "logits"],
    },
    DecodeFacadeOperation {
        operation: "ds4_gpu_routed_moe_one_tensor",
        method: "routed_moe_one",
        tensor_args: &[
            "out", "gate", "up", "mid", "experts", "selected", "weights", "x",
        ],
    },
    DecodeFacadeOperation {
        operation: "ds4_gpu_shared_gate_up_swiglu_q8_0_tensor",
        method: "shared_gate_up_swiglu_q8_0",
        tensor_args: &["gate", "up", "mid", "x"],
    },
    DecodeFacadeOperation {
        operation: "ds4_gpu_shared_down_hc_expand_q8_0_tensor",
        method: "shared_down_hc_expand_q8_0",
        tensor_args: &[
            "out_hc",
            "shared_out",
            "shared_mid",
            "routed_out",
            "residual_hc",
            "split",
        ],
    },
    DecodeFacadeOperation {
        operation: "ds4_gpu_output_hc_weights_tensor",
        method: "output_hc_weights",
        tensor_args: &["out", "pre"],
    },
    DecodeFacadeOperation {
        operation: "ds4_gpu_hc_weighted_sum_tensor",
        method: "hc_weighted_sum",
        tensor_args: &["out", "residual_hc", "weights"],
    },
];

pub const DIRECTIONAL_STEERING_DECODE_FACADE_OPERATIONS: &[DecodeFacadeOperation] = &[
    DecodeFacadeOperation {
        operation: "ds4_gpu_attention_output_q8_batch_tensor",
        method: "attention_output_q8_batch",
        tensor_args: &["out", "low", "group_tmp", "low_tmp", "heads"],
    },
    DecodeFacadeOperation {
        operation: "ds4_gpu_add_tensor",
        method: "add",
        tensor_args: &["out", "a", "b"],
    },
    DecodeFacadeOperation {
        operation: "ds4_gpu_directional_steering_project_tensor",
        method: "directional_steering_project",
        tensor_args: &["x", "directions"],
    },
    DecodeFacadeOperation {
        operation: "ds4_gpu_hc_expand_split_tensor",
        method: "hc_expand_split",
        tensor_args: &["out_hc", "block_out", "residual_hc", "split"],
    },
];

pub const PREFILL_FACADE_OPERATIONS: &[DecodeFacadeOperation] = &[
    DecodeFacadeOperation {
        operation: "ds4_gpu_embed_tokens_hc_tensor",
        method: "embed_tokens_hc",
        tensor_args: &["out_hc", "tokens"],
    },
    DecodeFacadeOperation {
        operation: "ds4_gpu_rms_norm_plain_rows_tensor",
        method: "rms_norm_plain_rows",
        tensor_args: &["out", "x"],
    },
    DecodeFacadeOperation {
        operation: "ds4_gpu_rms_norm_weight_rows_tensor",
        method: "rms_norm_weight_rows",
        tensor_args: &["out", "x"],
    },
    DecodeFacadeOperation {
        operation: "ds4_gpu_store_raw_kv_batch_tensor",
        method: "store_raw_kv_batch",
        tensor_args: &["raw_cache", "kv"],
    },
    DecodeFacadeOperation {
        operation: "ds4_gpu_compressor_prefill_tensor",
        method: "compressor_prefill",
        tensor_args: &["comp_cache", "state_kv", "state_score", "kv", "sc"],
    },
    DecodeFacadeOperation {
        operation: "ds4_gpu_compressor_prefill_ratio4_replay_tensor",
        method: "compressor_prefill_ratio4_replay",
        tensor_args: &["comp_cache", "state_kv", "state_score", "kv", "sc"],
    },
    DecodeFacadeOperation {
        operation: "ds4_gpu_compressor_prefill_state_ratio4_tensor",
        method: "compressor_prefill_state_ratio4",
        tensor_args: &["state_kv", "state_score", "kv_tail", "sc_tail"],
    },
    DecodeFacadeOperation {
        operation: "ds4_gpu_indexer_scores_decode_batch_tensor",
        method: "indexer_scores_decode_batch",
        tensor_args: &["scores", "q", "weights", "index_comp"],
    },
    DecodeFacadeOperation {
        operation: "ds4_gpu_attention_prefill_raw_heads_tensor",
        method: "attention_prefill_raw_heads",
        tensor_args: &["heads", "q", "raw_kv"],
    },
    DecodeFacadeOperation {
        operation: "ds4_gpu_attention_decode_raw_batch_heads_tensor",
        method: "attention_decode_raw_batch_heads",
        tensor_args: &["heads", "q", "raw_kv"],
    },
    DecodeFacadeOperation {
        operation: "ds4_gpu_attention_decode_mixed_batch_heads_tensor",
        method: "attention_decode_mixed_batch_heads",
        tensor_args: &["heads", "q", "raw_kv", "comp_kv", "comp_mask"],
    },
    DecodeFacadeOperation {
        operation: "ds4_gpu_attention_prefill_static_mixed_heads_tensor",
        method: "attention_prefill_static_mixed_heads",
        tensor_args: &["heads", "q", "raw_kv", "comp_kv"],
    },
    DecodeFacadeOperation {
        operation: "ds4_gpu_router_select_batch_tensor",
        method: "router_select_batch",
        tensor_args: &["selected", "weights", "probs", "logits", "tokens"],
    },
    DecodeFacadeOperation {
        operation: "ds4_gpu_routed_moe_batch_tensor",
        method: "routed_moe_batch",
        tensor_args: &[
            "out", "gate", "up", "mid", "experts", "selected", "weights", "x",
        ],
    },
    DecodeFacadeOperation {
        operation: "ds4_gpu_swiglu_tensor",
        method: "swiglu",
        tensor_args: &["out", "gate", "up"],
    },
    DecodeFacadeOperation {
        operation: "ds4_gpu_hc_split_weighted_sum_tensor",
        method: "hc_split_weighted_sum",
        tensor_args: &["out", "split", "mix", "residual_hc"],
    },
    DecodeFacadeOperation {
        operation: "ds4_gpu_hc_expand_add_split_tensor",
        method: "hc_expand_add_split",
        tensor_args: &["out_hc", "block_out", "block_add", "residual_hc", "split"],
    },
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExistingDecodeOperation {
    pub operation: &'static str,
    pub wrapper: &'static str,
}

pub const EXISTING_DECODE_OPERATIONS: &[ExistingDecodeOperation] = &[
    ExistingDecodeOperation {
        operation: "ds4_gpu_begin_commands",
        wrapper: "CommandBatch::begin",
    },
    ExistingDecodeOperation {
        operation: "ds4_gpu_flush_commands",
        wrapper: "CommandBatch::flush",
    },
    ExistingDecodeOperation {
        operation: "ds4_gpu_end_commands",
        wrapper: "CommandBatch::finish",
    },
    ExistingDecodeOperation {
        operation: "ds4_gpu_synchronize",
        wrapper: "synchronize",
    },
    ExistingDecodeOperation {
        operation: "ds4_gpu_tensor_read",
        wrapper: "Tensor::read_bytes",
    },
    ExistingDecodeOperation {
        operation: "ds4_gpu_tensor_view",
        wrapper: "Tensor::view",
    },
    ExistingDecodeOperation {
        operation: "ds4_gpu_tensor_free",
        wrapper: "Drop",
    },
];

pub const MODEL_MAP_BACKEND_OPERATIONS: &[ExistingDecodeOperation] = &[
    ExistingDecodeOperation {
        operation: "ds4_gpu_set_model_map",
        wrapper: "set_model_map",
    },
    ExistingDecodeOperation {
        operation: "ds4_gpu_set_model_fd",
        wrapper: "set_model_fd",
    },
    ExistingDecodeOperation {
        operation: "ds4_gpu_set_model_map_range",
        wrapper: "set_model_map_range",
    },
    // Inventory parity includes the CUDA cache calls on every host; the safe
    // wrappers are gated to Linux CUDA builds below.
    ExistingDecodeOperation {
        operation: "ds4_gpu_cache_model_range",
        wrapper: "cache_model_range",
    },
    ExistingDecodeOperation {
        operation: "ds4_gpu_cache_q8_f16_range",
        wrapper: "cache_q8_f16_range",
    },
];

#[derive(Clone, Copy, Debug)]
pub struct ModelMap<'a> {
    ptr: *const c_void,
    size: u64,
    _lifetime: PhantomData<&'a [u8]>,
}

impl<'a> ModelMap<'a> {
    pub fn from_bytes(bytes: &'a [u8]) -> Self {
        Self {
            ptr: bytes.as_ptr().cast::<c_void>(),
            size: bytes.len() as u64,
            _lifetime: PhantomData,
        }
    }

    /// Creates a borrowed model mapping from an existing C-side mapping.
    ///
    /// # Safety
    ///
    /// `ptr` must be valid for reads of `size` bytes for the lifetime `'a`.
    /// The backend may read from this mapping while any call using this
    /// `ModelMap` is executing.
    pub unsafe fn from_raw_parts(ptr: *const c_void, size: u64) -> Self {
        Self {
            ptr,
            size,
            _lifetime: PhantomData,
        }
    }

    pub const fn as_ptr(self) -> *const c_void {
        self.ptr
    }

    pub const fn size(self) -> u64 {
        self.size
    }
}

pub fn set_model_map(model: ModelMap<'_>) -> Result<(), GpuError> {
    unsafe {
        GpuStatus::from_raw(sys::ds4_gpu_set_model_map(model.as_ptr(), model.size())).into_result()
    }
}

pub fn set_model_fd(fd: c_int) -> Result<(), GpuError> {
    unsafe { GpuStatus::from_raw(sys::ds4_gpu_set_model_fd(fd)).into_result() }
}

pub fn set_model_map_range(
    model: ModelMap<'_>,
    map_offset: u64,
    map_size: u64,
) -> Result<(), GpuError> {
    validate_model_range(model, map_offset, map_size)?;
    unsafe {
        GpuStatus::from_raw(sys::ds4_gpu_set_model_map_range(
            model.as_ptr(),
            model.size(),
            map_offset,
            map_size,
        ))
        .into_result()
    }
}

#[cfg(all(target_os = "linux", feature = "cuda-backend"))]
pub fn cache_model_range(
    model: ModelMap<'_>,
    offset: u64,
    bytes: u64,
    label: Option<&CStr>,
) -> Result<(), GpuError> {
    validate_model_cache_range(model, offset, bytes)?;
    let label = label.map_or(ptr::null(), CStr::as_ptr);
    unsafe {
        GpuStatus::from_raw(sys::ds4_gpu_cache_model_range(
            model.as_ptr(),
            model.size(),
            offset,
            bytes,
            label,
        ))
        .into_result()
    }
}

#[cfg(all(target_os = "linux", feature = "cuda-backend"))]
pub fn cache_q8_f16_range(
    model: ModelMap<'_>,
    offset: u64,
    bytes: u64,
    in_dim: u64,
    out_dim: u64,
    label: Option<&CStr>,
) -> Result<(), GpuError> {
    validate_model_cache_range(model, offset, bytes)?;
    let label = label.map_or(ptr::null(), CStr::as_ptr);
    unsafe {
        GpuStatus::from_raw(sys::ds4_gpu_cache_q8_f16_range(
            model.as_ptr(),
            model.size(),
            offset,
            bytes,
            in_dim,
            out_dim,
            label,
        ))
        .into_result()
    }
}

#[cfg(all(target_os = "linux", feature = "cuda-backend"))]
fn validate_model_cache_range(
    model: ModelMap<'_>,
    offset: u64,
    bytes: u64,
) -> Result<(), GpuError> {
    if bytes == 0 {
        Ok(())
    } else {
        validate_model_range(model, offset, bytes)
    }
}

fn validate_model_range(model: ModelMap<'_>, offset: u64, bytes: u64) -> Result<(), GpuError> {
    if bytes == 0 || offset > model.size() || bytes > model.size() - offset {
        Err(GpuError::invalid_range())
    } else {
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DecodeBackend<'a> {
    model: ModelMap<'a>,
}

impl<'a> DecodeBackend<'a> {
    pub const fn new(model: ModelMap<'a>) -> Self {
        Self { model }
    }

    pub const fn model(self) -> ModelMap<'a> {
        self.model
    }

    pub fn embed_token_hc(
        self,
        out_hc: TensorMut<'_>,
        weight_offset: u64,
        n_vocab: u32,
        token: u32,
        n_embd: u32,
        n_hc: u32,
    ) -> Result<(), GpuError> {
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_embed_token_hc_tensor(
                out_hc.raw(),
                self.model.as_ptr(),
                self.model.size(),
                weight_offset,
                n_vocab,
                token,
                n_embd,
                n_hc,
            ))
            .into_result()
        }
    }

    pub fn embed_tokens_hc(
        self,
        out_hc: TensorMut<'_>,
        tokens: TensorRef<'_>,
        weight_offset: u64,
        n_vocab: u32,
        n_tokens: u32,
        n_embd: u32,
        n_hc: u32,
    ) -> Result<(), GpuError> {
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_embed_tokens_hc_tensor(
                out_hc.raw(),
                tokens.raw(),
                self.model.as_ptr(),
                self.model.size(),
                weight_offset,
                n_vocab,
                n_tokens,
                n_embd,
                n_hc,
            ))
            .into_result()
        }
    }

    pub fn rms_norm_plain(
        self,
        out: TensorMut<'_>,
        x: TensorRef<'_>,
        n: u32,
        eps: f32,
    ) -> Result<(), GpuError> {
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_rms_norm_plain_tensor(
                out.raw(),
                x.raw(),
                n,
                eps,
            ))
            .into_result()
        }
    }

    pub fn rms_norm_plain_rows(
        self,
        out: TensorMut<'_>,
        x: TensorRef<'_>,
        n: u32,
        rows: u32,
        eps: f32,
    ) -> Result<(), GpuError> {
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_rms_norm_plain_rows_tensor(
                out.raw(),
                x.raw(),
                n,
                rows,
                eps,
            ))
            .into_result()
        }
    }

    pub fn matmul_f16(
        self,
        out: TensorMut<'_>,
        weight_offset: u64,
        in_dim: u64,
        out_dim: u64,
        x: TensorRef<'_>,
        n_tok: u64,
    ) -> Result<(), GpuError> {
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_matmul_f16_tensor(
                out.raw(),
                self.model.as_ptr(),
                self.model.size(),
                weight_offset,
                in_dim,
                out_dim,
                x.raw(),
                n_tok,
            ))
            .into_result()
        }
    }

    pub fn hc_split_weighted_sum_norm(
        self,
        out: TensorMut<'_>,
        norm_out: TensorMut<'_>,
        split: TensorMut<'_>,
        mix: TensorRef<'_>,
        residual_hc: TensorRef<'_>,
        scale_offset: u64,
        base_offset: u64,
        norm_weight_offset: u64,
        n_embd: u32,
        n_hc: u32,
        sinkhorn_iters: u32,
        eps: f32,
        norm_eps: f32,
    ) -> Result<(), GpuError> {
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_hc_split_weighted_sum_norm_tensor(
                out.raw(),
                norm_out.raw(),
                split.raw(),
                mix.raw(),
                residual_hc.raw(),
                self.model.as_ptr(),
                self.model.size(),
                scale_offset,
                base_offset,
                norm_weight_offset,
                n_embd,
                n_hc,
                sinkhorn_iters,
                eps,
                norm_eps,
            ))
            .into_result()
        }
    }

    pub fn rms_norm_weight(
        self,
        out: TensorMut<'_>,
        x: TensorRef<'_>,
        weight_offset: u64,
        n: u32,
        eps: f32,
    ) -> Result<(), GpuError> {
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_rms_norm_weight_tensor(
                out.raw(),
                x.raw(),
                self.model.as_ptr(),
                self.model.size(),
                weight_offset,
                n,
                eps,
            ))
            .into_result()
        }
    }

    pub fn rms_norm_weight_rows(
        self,
        out: TensorMut<'_>,
        x: TensorRef<'_>,
        weight_offset: u64,
        n: u32,
        rows: u32,
        eps: f32,
    ) -> Result<(), GpuError> {
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_rms_norm_weight_rows_tensor(
                out.raw(),
                x.raw(),
                self.model.as_ptr(),
                self.model.size(),
                weight_offset,
                n,
                rows,
                eps,
            ))
            .into_result()
        }
    }

    pub fn matmul_q8_0(
        self,
        out: TensorMut<'_>,
        weight_offset: u64,
        in_dim: u64,
        out_dim: u64,
        x: TensorRef<'_>,
        n_tok: u64,
    ) -> Result<(), GpuError> {
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_matmul_q8_0_tensor(
                out.raw(),
                self.model.as_ptr(),
                self.model.size(),
                weight_offset,
                in_dim,
                out_dim,
                x.raw(),
                n_tok,
            ))
            .into_result()
        }
    }

    pub fn dsv4_qkv_rms_norm_rows(
        self,
        q_out: TensorMut<'_>,
        q: TensorRef<'_>,
        q_weight_offset: u64,
        q_n: u32,
        kv_out: TensorMut<'_>,
        kv: TensorRef<'_>,
        kv_weight_offset: u64,
        kv_n: u32,
        rows: u32,
        eps: f32,
    ) -> Result<(), GpuError> {
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
                q_out.raw(),
                q.raw(),
                self.model.as_ptr(),
                self.model.size(),
                q_weight_offset,
                q_n,
                kv_out.raw(),
                kv.raw(),
                kv_weight_offset,
                kv_n,
                rows,
                eps,
            ))
            .into_result()
        }
    }

    pub fn head_rms_norm(
        self,
        x: TensorMut<'_>,
        n_tok: u32,
        n_head: u32,
        head_dim: u32,
        eps: f32,
    ) -> Result<(), GpuError> {
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_head_rms_norm_tensor(
                x.raw(),
                n_tok,
                n_head,
                head_dim,
                eps,
            ))
            .into_result()
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rope_tail(
        self,
        x: TensorMut<'_>,
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
    ) -> Result<(), GpuError> {
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_rope_tail_tensor(
                x.raw(),
                n_tok,
                n_head,
                head_dim,
                n_rot,
                pos0,
                n_ctx_orig,
                inverse,
                freq_base,
                freq_scale,
                ext_factor,
                attn_factor,
                beta_fast,
                beta_slow,
            ))
            .into_result()
        }
    }

    pub fn kv_fp8_store_raw(
        self,
        kv: TensorMut<'_>,
        raw_cache: TensorMut<'_>,
        raw_cap: u32,
        row: u32,
        head_dim: u32,
        n_rot: u32,
    ) -> Result<(), GpuError> {
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_kv_fp8_store_raw_tensor(
                kv.raw(),
                raw_cache.raw(),
                raw_cap,
                row,
                head_dim,
                n_rot,
            ))
            .into_result()
        }
    }

    pub fn store_raw_kv_batch(
        self,
        raw_cache: TensorMut<'_>,
        kv: TensorRef<'_>,
        raw_cap: u32,
        pos0: u32,
        n_tokens: u32,
        head_dim: u32,
    ) -> Result<(), GpuError> {
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_store_raw_kv_batch_tensor(
                raw_cache.raw(),
                kv.raw(),
                raw_cap,
                pos0,
                n_tokens,
                head_dim,
            ))
            .into_result()
        }
    }

    pub fn matmul_f16_pair(
        self,
        out_a: TensorMut<'_>,
        out_b: TensorMut<'_>,
        weight_a_offset: u64,
        weight_b_offset: u64,
        in_dim: u64,
        out_dim: u64,
        x: TensorRef<'_>,
        n_tok: u64,
    ) -> Result<(), GpuError> {
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_matmul_f16_pair_tensor(
                out_a.raw(),
                out_b.raw(),
                self.model.as_ptr(),
                self.model.size(),
                weight_a_offset,
                weight_b_offset,
                in_dim,
                out_dim,
                x.raw(),
                n_tok,
            ))
            .into_result()
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn compressor_update(
        self,
        kv_cur: TensorRef<'_>,
        sc_cur: TensorRef<'_>,
        state_kv: TensorMut<'_>,
        state_score: TensorMut<'_>,
        comp_cache: TensorMut<'_>,
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
    ) -> Result<(), GpuError> {
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_compressor_update_tensor(
                kv_cur.raw(),
                sc_cur.raw(),
                state_kv.raw(),
                state_score.raw(),
                comp_cache.raw(),
                self.model.as_ptr(),
                self.model.size(),
                ape_offset,
                ape_type,
                norm_offset,
                norm_type,
                head_dim,
                ratio,
                pos,
                comp_row,
                n_rot,
                n_ctx_orig,
                freq_base,
                freq_scale,
                ext_factor,
                attn_factor,
                beta_fast,
                beta_slow,
                rms_eps,
            ))
            .into_result()
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn compressor_prefill(
        self,
        comp_cache: TensorMut<'_>,
        state_kv: TensorMut<'_>,
        state_score: TensorMut<'_>,
        kv: TensorRef<'_>,
        sc: TensorRef<'_>,
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
    ) -> Result<(), GpuError> {
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_compressor_prefill_tensor(
                comp_cache.raw(),
                state_kv.raw(),
                state_score.raw(),
                kv.raw(),
                sc.raw(),
                self.model.as_ptr(),
                self.model.size(),
                ape_offset,
                ape_type,
                norm_offset,
                norm_type,
                head_dim,
                ratio,
                pos0,
                n_tokens,
                n_rot,
                n_ctx_orig,
                quantize_fp8,
                freq_base,
                freq_scale,
                ext_factor,
                attn_factor,
                beta_fast,
                beta_slow,
                rms_eps,
            ))
            .into_result()
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn compressor_prefill_ratio4_replay(
        self,
        comp_cache: TensorMut<'_>,
        state_kv: TensorMut<'_>,
        state_score: TensorMut<'_>,
        kv: TensorRef<'_>,
        sc: TensorRef<'_>,
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
    ) -> Result<(), GpuError> {
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_compressor_prefill_ratio4_replay_tensor(
                comp_cache.raw(),
                state_kv.raw(),
                state_score.raw(),
                kv.raw(),
                sc.raw(),
                self.model.as_ptr(),
                self.model.size(),
                ape_offset,
                ape_type,
                norm_offset,
                norm_type,
                head_dim,
                pos0,
                n_tokens,
                n_rot,
                n_ctx_orig,
                quantize_fp8,
                freq_base,
                freq_scale,
                ext_factor,
                attn_factor,
                beta_fast,
                beta_slow,
                rms_eps,
            ))
            .into_result()
        }
    }

    pub fn compressor_prefill_state_ratio4(
        self,
        state_kv: TensorMut<'_>,
        state_score: TensorMut<'_>,
        kv_tail: TensorRef<'_>,
        sc_tail: TensorRef<'_>,
        ape_offset: u64,
        ape_type: u32,
        head_dim: u32,
        pos0: u32,
    ) -> Result<(), GpuError> {
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_compressor_prefill_state_ratio4_tensor(
                state_kv.raw(),
                state_score.raw(),
                kv_tail.raw(),
                sc_tail.raw(),
                self.model.as_ptr(),
                self.model.size(),
                ape_offset,
                ape_type,
                head_dim,
                pos0,
            ))
            .into_result()
        }
    }

    pub fn dsv4_fp8_kv_quantize(
        self,
        x: TensorMut<'_>,
        n_tok: u32,
        head_dim: u32,
        n_rot: u32,
    ) -> Result<(), GpuError> {
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_dsv4_fp8_kv_quantize_tensor(
                x.raw(),
                n_tok,
                head_dim,
                n_rot,
            ))
            .into_result()
        }
    }

    pub fn dsv4_indexer_qat(
        self,
        x: TensorMut<'_>,
        n_rows: u32,
        head_dim: u32,
    ) -> Result<(), GpuError> {
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_dsv4_indexer_qat_tensor(
                x.raw(),
                n_rows,
                head_dim,
            ))
            .into_result()
        }
    }

    pub fn indexer_score_one(
        self,
        scores: TensorMut<'_>,
        q: TensorRef<'_>,
        weights: TensorRef<'_>,
        index_comp: TensorRef<'_>,
        n_comp: u32,
        n_head: u32,
        head_dim: u32,
        scale: f32,
    ) -> Result<(), GpuError> {
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_indexer_score_one_tensor(
                scores.raw(),
                q.raw(),
                weights.raw(),
                index_comp.raw(),
                n_comp,
                n_head,
                head_dim,
                scale,
            ))
            .into_result()
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn indexer_scores_decode_batch(
        self,
        scores: TensorMut<'_>,
        q: TensorRef<'_>,
        weights: TensorRef<'_>,
        index_comp: TensorRef<'_>,
        n_comp: u32,
        n_tokens: u32,
        pos0: u32,
        n_head: u32,
        head_dim: u32,
        ratio: u32,
        scale: f32,
    ) -> Result<(), GpuError> {
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_indexer_scores_decode_batch_tensor(
                scores.raw(),
                q.raw(),
                weights.raw(),
                index_comp.raw(),
                n_comp,
                n_tokens,
                pos0,
                n_head,
                head_dim,
                ratio,
                scale,
            ))
            .into_result()
        }
    }

    pub fn indexer_topk(
        self,
        selected: TensorMut<'_>,
        scores: TensorRef<'_>,
        n_comp: u32,
        n_tokens: u32,
        top_k: u32,
    ) -> Result<(), GpuError> {
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_indexer_topk_tensor(
                selected.raw(),
                scores.raw(),
                n_comp,
                n_tokens,
                top_k,
            ))
            .into_result()
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention_indexed_mixed_batch_heads(
        self,
        heads: TensorMut<'_>,
        sinks_offset: u64,
        q: TensorRef<'_>,
        raw_kv: TensorRef<'_>,
        comp_kv: TensorRef<'_>,
        topk: TensorRef<'_>,
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
    ) -> Result<(), GpuError> {
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                heads.raw(),
                self.model.as_ptr(),
                self.model.size(),
                sinks_offset,
                q.raw(),
                raw_kv.raw(),
                comp_kv.raw(),
                topk.raw(),
                n_tokens,
                pos0,
                n_raw,
                raw_cap,
                raw_start,
                n_comp,
                top_k,
                window,
                ratio,
                n_head,
                head_dim,
            ))
            .into_result()
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention_decode_raw_batch_heads(
        self,
        heads: TensorMut<'_>,
        sinks_offset: u64,
        q: TensorRef<'_>,
        raw_kv: TensorRef<'_>,
        n_tokens: u32,
        pos0: u32,
        n_raw: u32,
        raw_cap: u32,
        raw_start: u32,
        window: u32,
        n_head: u32,
        head_dim: u32,
    ) -> Result<(), GpuError> {
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_attention_decode_raw_batch_heads_tensor(
                heads.raw(),
                self.model.as_ptr(),
                self.model.size(),
                sinks_offset,
                q.raw(),
                raw_kv.raw(),
                n_tokens,
                pos0,
                n_raw,
                raw_cap,
                raw_start,
                window,
                n_head,
                head_dim,
            ))
            .into_result()
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention_decode_mixed_batch_heads(
        self,
        heads: TensorMut<'_>,
        sinks_offset: u64,
        q: TensorRef<'_>,
        raw_kv: TensorRef<'_>,
        comp_kv: TensorRef<'_>,
        comp_mask: Option<TensorRef<'_>>,
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
    ) -> Result<(), GpuError> {
        let comp_mask = optional_tensor_ref(comp_mask, use_comp_mask != 0)?;
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_attention_decode_mixed_batch_heads_tensor(
                heads.raw(),
                self.model.as_ptr(),
                self.model.size(),
                sinks_offset,
                q.raw(),
                raw_kv.raw(),
                comp_kv.raw(),
                comp_mask,
                use_comp_mask,
                n_tokens,
                pos0,
                n_raw,
                raw_cap,
                raw_start,
                n_comp,
                window,
                ratio,
                n_head,
                head_dim,
            ))
            .into_result()
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention_decode_heads(
        self,
        heads: TensorMut<'_>,
        sinks_offset: u64,
        q: TensorRef<'_>,
        raw_kv: TensorRef<'_>,
        n_raw: u32,
        raw_cap: u32,
        raw_start: u32,
        comp_kv: Option<TensorRef<'_>>,
        n_comp: u32,
        comp_mask: Option<TensorRef<'_>>,
        use_mask: u32,
        n_head: u32,
        head_dim: u32,
    ) -> Result<(), GpuError> {
        let comp_kv = optional_tensor_ref(comp_kv, n_comp != 0)?;
        let comp_mask = optional_tensor_ref(comp_mask, use_mask != 0)?;
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_attention_decode_heads_tensor(
                heads.raw(),
                self.model.as_ptr(),
                self.model.size(),
                sinks_offset,
                q.raw(),
                raw_kv.raw(),
                n_raw,
                raw_cap,
                raw_start,
                comp_kv,
                n_comp,
                comp_mask,
                use_mask,
                n_head,
                head_dim,
            ))
            .into_result()
        }
    }

    pub fn attention_prefill_raw_heads(
        self,
        heads: TensorMut<'_>,
        sinks_offset: u64,
        q: TensorRef<'_>,
        raw_kv: TensorRef<'_>,
        n_tokens: u32,
        window: u32,
        n_head: u32,
        head_dim: u32,
    ) -> Result<(), GpuError> {
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_attention_prefill_raw_heads_tensor(
                heads.raw(),
                self.model.as_ptr(),
                self.model.size(),
                sinks_offset,
                q.raw(),
                raw_kv.raw(),
                n_tokens,
                window,
                n_head,
                head_dim,
            ))
            .into_result()
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention_prefill_static_mixed_heads(
        self,
        heads: TensorMut<'_>,
        sinks_offset: u64,
        q: TensorRef<'_>,
        raw_kv: TensorRef<'_>,
        comp_kv: TensorRef<'_>,
        n_tokens: u32,
        n_comp: u32,
        window: u32,
        ratio: u32,
        n_head: u32,
        head_dim: u32,
    ) -> Result<(), GpuError> {
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_attention_prefill_static_mixed_heads_tensor(
                heads.raw(),
                self.model.as_ptr(),
                self.model.size(),
                sinks_offset,
                q.raw(),
                raw_kv.raw(),
                comp_kv.raw(),
                n_tokens,
                n_comp,
                window,
                ratio,
                n_head,
                head_dim,
            ))
            .into_result()
        }
    }

    pub fn attention_output_low_q8(
        self,
        low: TensorMut<'_>,
        out_a_offset: u64,
        group_dim: u64,
        rank: u64,
        n_groups: u32,
        heads: TensorRef<'_>,
    ) -> Result<(), GpuError> {
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_attention_output_low_q8_tensor(
                low.raw(),
                self.model.as_ptr(),
                self.model.size(),
                out_a_offset,
                group_dim,
                rank,
                n_groups,
                heads.raw(),
            ))
            .into_result()
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention_output_q8_batch(
        self,
        out: TensorMut<'_>,
        low: TensorMut<'_>,
        group_tmp: TensorMut<'_>,
        low_tmp: TensorMut<'_>,
        out_a_offset: u64,
        out_b_offset: u64,
        group_dim: u64,
        rank: u64,
        n_groups: u32,
        out_dim: u64,
        heads: TensorRef<'_>,
        n_tokens: u32,
    ) -> Result<(), GpuError> {
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_attention_output_q8_batch_tensor(
                out.raw(),
                low.raw(),
                group_tmp.raw(),
                low_tmp.raw(),
                self.model.as_ptr(),
                self.model.size(),
                out_a_offset,
                out_b_offset,
                group_dim,
                rank,
                n_groups,
                out_dim,
                heads.raw(),
                n_tokens,
            ))
            .into_result()
        }
    }

    pub fn matmul_q8_0_hc_expand(
        self,
        out_hc: TensorMut<'_>,
        block_out: TensorMut<'_>,
        weight_offset: u64,
        in_dim: u64,
        out_dim: u64,
        x: TensorRef<'_>,
        residual_hc: TensorRef<'_>,
        split: TensorRef<'_>,
        n_embd: u32,
        n_hc: u32,
    ) -> Result<(), GpuError> {
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_matmul_q8_0_hc_expand_tensor(
                out_hc.raw(),
                block_out.raw(),
                self.model.as_ptr(),
                self.model.size(),
                weight_offset,
                in_dim,
                out_dim,
                x.raw(),
                residual_hc.raw(),
                split.raw(),
                n_embd,
                n_hc,
            ))
            .into_result()
        }
    }

    pub fn hc_expand_split(
        self,
        out_hc: TensorMut<'_>,
        block_out: TensorRef<'_>,
        residual_hc: TensorRef<'_>,
        split: TensorRef<'_>,
        n_embd: u32,
        n_hc: u32,
    ) -> Result<(), GpuError> {
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_hc_expand_split_tensor(
                out_hc.raw(),
                block_out.raw(),
                residual_hc.raw(),
                split.raw(),
                n_embd,
                n_hc,
            ))
            .into_result()
        }
    }

    pub fn hc_split_weighted_sum(
        self,
        out: TensorMut<'_>,
        split: TensorMut<'_>,
        mix: TensorRef<'_>,
        residual_hc: TensorRef<'_>,
        scale_offset: u64,
        base_offset: u64,
        n_embd: u32,
        n_hc: u32,
        sinkhorn_iters: u32,
        eps: f32,
    ) -> Result<(), GpuError> {
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_hc_split_weighted_sum_tensor(
                out.raw(),
                split.raw(),
                mix.raw(),
                residual_hc.raw(),
                self.model.as_ptr(),
                self.model.size(),
                scale_offset,
                base_offset,
                n_embd,
                n_hc,
                sinkhorn_iters,
                eps,
            ))
            .into_result()
        }
    }

    pub fn hc_expand_add_split(
        self,
        out_hc: TensorMut<'_>,
        block_out: TensorRef<'_>,
        block_add: TensorRef<'_>,
        residual_hc: TensorRef<'_>,
        split: TensorRef<'_>,
        n_embd: u32,
        n_hc: u32,
    ) -> Result<(), GpuError> {
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_hc_expand_add_split_tensor(
                out_hc.raw(),
                block_out.raw(),
                block_add.raw(),
                residual_hc.raw(),
                split.raw(),
                n_embd,
                n_hc,
            ))
            .into_result()
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn router_select(
        self,
        selected: TensorMut<'_>,
        weights: TensorMut<'_>,
        probs: TensorMut<'_>,
        bias_offset: u64,
        hash_offset: u64,
        hash_rows: u32,
        token: u32,
        n_expert_groups: u32,
        n_group_used: u32,
        has_bias: bool,
        hash_mode: bool,
        logits: TensorRef<'_>,
    ) -> Result<(), GpuError> {
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_router_select_tensor(
                selected.raw(),
                weights.raw(),
                probs.raw(),
                self.model.as_ptr(),
                self.model.size(),
                bias_offset,
                hash_offset,
                hash_rows,
                token,
                n_expert_groups,
                n_group_used,
                has_bias,
                hash_mode,
                logits.raw(),
            ))
            .into_result()
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn router_select_batch(
        self,
        selected: TensorMut<'_>,
        weights: TensorMut<'_>,
        probs: TensorMut<'_>,
        bias_offset: u64,
        hash_offset: u64,
        hash_rows: u32,
        n_expert_groups: u32,
        n_group_used: u32,
        has_bias: bool,
        hash_mode: bool,
        logits: TensorRef<'_>,
        tokens: TensorRef<'_>,
        n_tokens: u32,
    ) -> Result<(), GpuError> {
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_router_select_batch_tensor(
                selected.raw(),
                weights.raw(),
                probs.raw(),
                self.model.as_ptr(),
                self.model.size(),
                bias_offset,
                hash_offset,
                hash_rows,
                n_expert_groups,
                n_group_used,
                has_bias,
                hash_mode,
                logits.raw(),
                tokens.raw(),
                n_tokens,
            ))
            .into_result()
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn routed_moe_one(
        self,
        out: TensorMut<'_>,
        gate: TensorMut<'_>,
        up: TensorMut<'_>,
        mid: TensorMut<'_>,
        experts: TensorMut<'_>,
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
        selected: TensorRef<'_>,
        weights: TensorRef<'_>,
        n_expert: u32,
        clamp: f32,
        x: TensorRef<'_>,
    ) -> Result<(), GpuError> {
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_routed_moe_one_tensor(
                out.raw(),
                gate.raw(),
                up.raw(),
                mid.raw(),
                experts.raw(),
                self.model.as_ptr(),
                self.model.size(),
                gate_offset,
                up_offset,
                down_offset,
                gate_type,
                down_type,
                gate_expert_bytes,
                gate_row_bytes,
                down_expert_bytes,
                down_row_bytes,
                expert_in_dim,
                expert_mid_dim,
                out_dim,
                selected.raw(),
                weights.raw(),
                n_expert,
                clamp,
                x.raw(),
            ))
            .into_result()
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn routed_moe_batch(
        self,
        out: TensorMut<'_>,
        gate: TensorMut<'_>,
        up: TensorMut<'_>,
        mid: TensorMut<'_>,
        experts: TensorMut<'_>,
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
        selected: TensorRef<'_>,
        weights: TensorRef<'_>,
        n_expert: u32,
        clamp: f32,
        x: TensorRef<'_>,
        n_tokens: u32,
        mid_is_f16: &mut bool,
    ) -> Result<(), GpuError> {
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_routed_moe_batch_tensor(
                out.raw(),
                gate.raw(),
                up.raw(),
                mid.raw(),
                experts.raw(),
                self.model.as_ptr(),
                self.model.size(),
                gate_offset,
                up_offset,
                down_offset,
                gate_type,
                down_type,
                gate_expert_bytes,
                gate_row_bytes,
                down_expert_bytes,
                down_row_bytes,
                expert_in_dim,
                expert_mid_dim,
                out_dim,
                selected.raw(),
                weights.raw(),
                n_expert,
                clamp,
                x.raw(),
                n_tokens,
                mid_is_f16 as *mut bool,
            ))
            .into_result()
        }
    }

    pub fn add(
        self,
        out: TensorMut<'_>,
        a: TensorRef<'_>,
        b: TensorRef<'_>,
        n: u32,
    ) -> Result<(), GpuError> {
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_add_tensor(out.raw(), a.raw(), b.raw(), n))
                .into_result()
        }
    }

    pub fn directional_steering_project(
        self,
        x: TensorMut<'_>,
        directions: TensorRef<'_>,
        layer: u32,
        width: u32,
        rows: u32,
        scale: f32,
    ) -> Result<(), GpuError> {
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_directional_steering_project_tensor(
                x.raw(),
                directions.raw(),
                layer,
                width,
                rows,
                scale,
            ))
            .into_result()
        }
    }

    pub fn swiglu(
        self,
        out: TensorMut<'_>,
        gate: TensorRef<'_>,
        up: TensorRef<'_>,
        n: u32,
        clamp: f32,
        weight: f32,
    ) -> Result<(), GpuError> {
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_swiglu_tensor(
                out.raw(),
                gate.raw(),
                up.raw(),
                n,
                clamp,
                weight,
            ))
            .into_result()
        }
    }

    pub fn shared_gate_up_swiglu_q8_0(
        self,
        gate: TensorMut<'_>,
        up: TensorMut<'_>,
        mid: TensorMut<'_>,
        gate_offset: u64,
        up_offset: u64,
        in_dim: u64,
        out_dim: u64,
        x: TensorRef<'_>,
        clamp: f32,
    ) -> Result<(), GpuError> {
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
                gate.raw(),
                up.raw(),
                mid.raw(),
                self.model.as_ptr(),
                self.model.size(),
                gate_offset,
                up_offset,
                in_dim,
                out_dim,
                x.raw(),
                clamp,
            ))
            .into_result()
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn shared_down_hc_expand_q8_0(
        self,
        out_hc: TensorMut<'_>,
        shared_out: TensorMut<'_>,
        weight_offset: u64,
        in_dim: u64,
        out_dim: u64,
        shared_mid: TensorRef<'_>,
        routed_out: TensorRef<'_>,
        residual_hc: TensorRef<'_>,
        split: TensorRef<'_>,
        n_embd: u32,
        n_hc: u32,
    ) -> Result<(), GpuError> {
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_shared_down_hc_expand_q8_0_tensor(
                out_hc.raw(),
                shared_out.raw(),
                self.model.as_ptr(),
                self.model.size(),
                weight_offset,
                in_dim,
                out_dim,
                shared_mid.raw(),
                routed_out.raw(),
                residual_hc.raw(),
                split.raw(),
                n_embd,
                n_hc,
            ))
            .into_result()
        }
    }

    pub fn output_hc_weights(
        self,
        out: TensorMut<'_>,
        pre: TensorRef<'_>,
        scale_offset: u64,
        base_offset: u64,
        n_hc: u32,
        eps: f32,
    ) -> Result<(), GpuError> {
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_output_hc_weights_tensor(
                out.raw(),
                pre.raw(),
                self.model.as_ptr(),
                self.model.size(),
                scale_offset,
                base_offset,
                n_hc,
                eps,
            ))
            .into_result()
        }
    }

    pub fn hc_weighted_sum(
        self,
        out: TensorMut<'_>,
        residual_hc: TensorRef<'_>,
        weights: TensorRef<'_>,
        n_embd: u32,
        n_hc: u32,
    ) -> Result<(), GpuError> {
        unsafe {
            GpuStatus::from_raw(sys::ds4_gpu_hc_weighted_sum_tensor(
                out.raw(),
                residual_hc.raw(),
                weights.raw(),
                n_embd,
                n_hc,
            ))
            .into_result()
        }
    }
}

fn optional_tensor_ref(
    tensor: Option<TensorRef<'_>>,
    required: bool,
) -> Result<*const sys::Ds4GpuTensor, GpuError> {
    match tensor {
        Some(tensor) => Ok(tensor.raw()),
        None if required => Err(GpuError::null_tensor()),
        None => Ok(ptr::null()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXPECTED_DEFAULT_OPS: &[&str] = &[
        "ds4_gpu_embed_token_hc_tensor",
        "ds4_gpu_rms_norm_plain_tensor",
        "ds4_gpu_matmul_f16_tensor",
        "ds4_gpu_hc_split_weighted_sum_norm_tensor",
        "ds4_gpu_rms_norm_weight_tensor",
        "ds4_gpu_matmul_q8_0_tensor",
        "ds4_gpu_dsv4_qkv_rms_norm_rows_tensor",
        "ds4_gpu_head_rms_norm_tensor",
        "ds4_gpu_rope_tail_tensor",
        "ds4_gpu_kv_fp8_store_raw_tensor",
        "ds4_gpu_matmul_f16_pair_tensor",
        "ds4_gpu_compressor_update_tensor",
        "ds4_gpu_dsv4_fp8_kv_quantize_tensor",
        "ds4_gpu_dsv4_indexer_qat_tensor",
        "ds4_gpu_indexer_score_one_tensor",
        "ds4_gpu_indexer_topk_tensor",
        "ds4_gpu_attention_indexed_mixed_batch_heads_tensor",
        "ds4_gpu_attention_decode_heads_tensor",
        "ds4_gpu_attention_output_low_q8_tensor",
        "ds4_gpu_matmul_q8_0_hc_expand_tensor",
        "ds4_gpu_router_select_tensor",
        "ds4_gpu_routed_moe_one_tensor",
        "ds4_gpu_shared_gate_up_swiglu_q8_0_tensor",
        "ds4_gpu_shared_down_hc_expand_q8_0_tensor",
        "ds4_gpu_output_hc_weights_tensor",
        "ds4_gpu_hc_weighted_sum_tensor",
    ];

    const EXPECTED_DIRECTIONAL_STEERING_OPS: &[&str] = &[
        "ds4_gpu_attention_output_q8_batch_tensor",
        "ds4_gpu_add_tensor",
        "ds4_gpu_directional_steering_project_tensor",
        "ds4_gpu_hc_expand_split_tensor",
    ];

    const EXPECTED_PREFILL_OPS: &[&str] = &[
        "ds4_gpu_embed_tokens_hc_tensor",
        "ds4_gpu_rms_norm_plain_rows_tensor",
        "ds4_gpu_rms_norm_weight_rows_tensor",
        "ds4_gpu_store_raw_kv_batch_tensor",
        "ds4_gpu_compressor_prefill_tensor",
        "ds4_gpu_compressor_prefill_ratio4_replay_tensor",
        "ds4_gpu_compressor_prefill_state_ratio4_tensor",
        "ds4_gpu_indexer_scores_decode_batch_tensor",
        "ds4_gpu_attention_prefill_raw_heads_tensor",
        "ds4_gpu_attention_decode_raw_batch_heads_tensor",
        "ds4_gpu_attention_decode_mixed_batch_heads_tensor",
        "ds4_gpu_attention_prefill_static_mixed_heads_tensor",
        "ds4_gpu_router_select_batch_tensor",
        "ds4_gpu_routed_moe_batch_tensor",
        "ds4_gpu_swiglu_tensor",
        "ds4_gpu_hc_split_weighted_sum_tensor",
        "ds4_gpu_hc_expand_add_split_tensor",
    ];

    #[test]
    fn default_decode_facade_operations_are_pinned() {
        assert_eq!(DEFAULT_DECODE_FACADE_OPERATIONS.len(), 26);
        for (spec, expected) in DEFAULT_DECODE_FACADE_OPERATIONS
            .iter()
            .zip(EXPECTED_DEFAULT_OPS)
        {
            assert_eq!(spec.operation, *expected);
            assert!(!spec.method.is_empty());
            assert!(!spec.tensor_args.is_empty());
        }
    }

    #[test]
    fn default_decode_facade_methods_are_unique() {
        for (i, left) in DEFAULT_DECODE_FACADE_OPERATIONS.iter().enumerate() {
            for right in &DEFAULT_DECODE_FACADE_OPERATIONS[i + 1..] {
                assert_ne!(left.operation, right.operation);
                assert_ne!(left.method, right.method);
            }
        }
    }

    #[test]
    fn directional_steering_decode_facade_operations_are_pinned() {
        assert_eq!(DIRECTIONAL_STEERING_DECODE_FACADE_OPERATIONS.len(), 4);
        for (spec, expected) in DIRECTIONAL_STEERING_DECODE_FACADE_OPERATIONS
            .iter()
            .zip(EXPECTED_DIRECTIONAL_STEERING_OPS)
        {
            assert_eq!(spec.operation, *expected);
            assert!(!spec.method.is_empty());
            assert!(!spec.tensor_args.is_empty());
        }
    }

    #[test]
    fn prefill_facade_operations_are_pinned() {
        assert_eq!(PREFILL_FACADE_OPERATIONS.len(), 17);
        for (spec, expected) in PREFILL_FACADE_OPERATIONS.iter().zip(EXPECTED_PREFILL_OPS) {
            assert_eq!(spec.operation, *expected);
            assert!(!spec.method.is_empty());
            assert!(!spec.tensor_args.is_empty());
        }
    }

    #[test]
    fn existing_decode_operations_cover_command_read_and_sync() {
        assert_eq!(EXISTING_DECODE_OPERATIONS.len(), 7);
        assert!(EXISTING_DECODE_OPERATIONS
            .iter()
            .any(|spec| spec.operation == "ds4_gpu_flush_commands"));
        assert!(EXISTING_DECODE_OPERATIONS
            .iter()
            .any(|spec| spec.operation == "ds4_gpu_tensor_read"));
        assert!(EXISTING_DECODE_OPERATIONS
            .iter()
            .any(|spec| spec.operation == "ds4_gpu_synchronize"));
    }

    #[test]
    fn model_map_backend_operations_are_wrapped() {
        assert_eq!(MODEL_MAP_BACKEND_OPERATIONS.len(), 5);
        for expected in [
            "ds4_gpu_set_model_map",
            "ds4_gpu_set_model_fd",
            "ds4_gpu_set_model_map_range",
            "ds4_gpu_cache_model_range",
            "ds4_gpu_cache_q8_f16_range",
        ] {
            assert!(MODEL_MAP_BACKEND_OPERATIONS
                .iter()
                .any(|spec| spec.operation == expected));
        }
    }

    #[test]
    fn model_map_from_bytes_keeps_borrowed_pointer_and_size() {
        let bytes = [1_u8, 2, 3, 4];
        let model = ModelMap::from_bytes(&bytes);
        assert_eq!(model.as_ptr(), bytes.as_ptr().cast::<c_void>());
        assert_eq!(model.size(), 4);
    }

    #[test]
    fn optional_tensor_ref_rejects_required_nulls() {
        assert_eq!(
            optional_tensor_ref(None, true).unwrap_err().kind(),
            crate::GpuErrorKind::NullTensor
        );
        assert_eq!(optional_tensor_ref(None, false).unwrap(), ptr::null());
    }
}
