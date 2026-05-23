use std::fmt;
use std::str;

pub mod agent_dsml;
pub mod cli_parse;
pub mod cli_token_dump;
pub mod decode_policy;
pub mod dsml;
pub mod kv_policy;
pub mod prompt;
pub mod sampling;
pub mod server_chat;
pub mod server_http;
pub mod server_no_model;
pub mod server_response;
pub mod session_payload;
mod tokenizer;

pub use agent_dsml::{
    hex as agent_dsml_hex, AgentDsmlParser, AgentDsmlState, AgentToolArg, AgentToolCall,
};
pub use decode_policy::{
    find_stop_from, policy_cases, run_policy_case, stop_list_stream_safe_len, utf8_stream_safe_len,
    ApiFinish, ApiStyle, PolicyCase, PolicyKind, PolicyPiece, PolicyRequest, PolicyResult,
    PolicySurface, StopBoundary, StreamStep, ToolBoundary,
};
pub use dsml::{
    parse_generated_message, parse_generated_message_for_response, render_dsml_tool_calls,
    render_dsml_tool_calls_from_json, render_tool_result_text, DsmlArgument, DsmlJsonCall,
    DsmlParseError, DsmlRenderCall, ParsedGeneratedMessage, ResponseParse,
};
pub use prompt::{
    apply_cli_ops, render_chat_prompt_text, render_live_tool_tail_text, ChatMessage, CliOp,
    ThinkMode, ToolArgument, ToolCall,
};
pub use sampling::{
    sample_argmax, sample_rng_f32, sample_rng_next, sample_top_p_min_p, token_logprob,
    top_logprobs, SamplingParams, SamplingTrace, TokenScore,
};
pub use server_chat::{
    anthropic_context_length_error_body, openai_context_length_error_body,
    openai_context_length_error_body_for_param, parse_anthropic_core_request,
    parse_anthropic_core_request_with_live_state, parse_completion_core_request,
    parse_openai_chat_request, parse_responses_core_request,
    parse_responses_core_request_with_live_state, request_exceeds_context, think_mode_for_context,
    AnthropicLiveState, AnthropicRequest, CompletionRequest, OpenAiChatRequest, ResponsesLiveState,
    ResponsesRequest, ServerRequestError, ServerRequestErrorCategory, ToolSchemaOrder,
};
pub use server_http::{
    format_http_error, format_http_response, format_model_metadata_json, parse_http_request,
    route_no_model_http, route_no_model_request, HttpRequest, HttpRequestParseError,
    NoModelRouteConfig, DS4_MODEL_ID,
};
pub use server_no_model::{
    route_no_model_server_http, route_no_model_server_http_with_generation_message,
    route_no_model_server_http_with_prompt_tokens, route_no_model_server_request,
    route_no_model_server_request_with_generation_message,
    route_no_model_server_request_with_prompt_tokens,
};
pub use server_response::{
    format_openai_chat_completion_http, format_openai_chat_completion_json,
    format_openai_chat_stream_http, format_openai_chat_stream_sse, OpenAiChatCompletion,
    OpenAiChatStream, OpenAiUsage,
};
pub use tokenizer::{Ds4Tokenizer, SpecialTokenIds, TokenizerError, TokenizerIdentity};

const GGUF_MAGIC: u32 = 0x4655_4747;
const MAX_DIMS: usize = 8;
const DEFAULT_ALIGNMENT: u64 = 32;
pub const MAX_REPORTED_TENSOR_TYPE_ID: u32 = 30;

#[derive(Debug, Clone, PartialEq)]
pub struct Gguf {
    pub version: u32,
    pub metadata: Vec<MetadataEntry>,
    pub tensors: Vec<TensorInfo>,
    pub alignment: u64,
    pub tensor_data_offset: u64,
    pub file_size: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetadataEntry {
    pub key: String,
    pub value: MetadataValue,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    UInt8(u8),
    Int8(i8),
    UInt16(u16),
    Int16(i16),
    UInt32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array {
        element_type: u32,
        values: Vec<MetadataValue>,
    },
    UInt64(u64),
    Int64(i64),
    Float64(f64),
}

impl MetadataValue {
    pub fn type_id(&self) -> u32 {
        match self {
            Self::UInt8(_) => 0,
            Self::Int8(_) => 1,
            Self::UInt16(_) => 2,
            Self::Int16(_) => 3,
            Self::UInt32(_) => 4,
            Self::Int32(_) => 5,
            Self::Float32(_) => 6,
            Self::Bool(_) => 7,
            Self::String(_) => 8,
            Self::Array { .. } => 9,
            Self::UInt64(_) => 10,
            Self::Int64(_) => 11,
            Self::Float64(_) => 12,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub type_id: u32,
    pub rel_offset: u64,
    pub abs_offset: u64,
    pub elements: u64,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoundTensor {
    pub role: String,
    pub tensor: Option<TensorInfo>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufError {
    message: String,
}

impl GgufError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for GgufError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ds4ValidationError {
    message: String,
}

impl Ds4ValidationError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for Ds4ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Ds4ValidationError {}

const DS4_N_LAYER: u32 = 43;
const DS4_N_EMBD: u32 = 4096;
const DS4_N_VOCAB: u32 = 129280;
const DS4_N_HEAD: u32 = 64;
const DS4_N_HEAD_KV: u32 = 1;
const DS4_N_HEAD_DIM: u32 = 512;
const DS4_N_VALUE_DIM: u32 = 512;
const DS4_N_ROT: u32 = 64;
const DS4_N_OUT_GROUP: u32 = 8;
const DS4_N_LORA_Q: u32 = 1024;
const DS4_N_LORA_O: u32 = 1024;
const DS4_N_EXPERT: u32 = 256;
const DS4_N_EXPERT_USED: u32 = 6;
const DS4_N_EXPERT_SHARED: u32 = 1;
const DS4_N_FF_EXP: u32 = 2048;
const DS4_N_HASH_LAYER: u32 = 3;
const DS4_N_SWA: u32 = 128;
const DS4_N_INDEXER_HEAD: u32 = 64;
const DS4_N_INDEXER_HEAD_DIM: u32 = 128;
const DS4_N_INDEXER_TOP_K: u32 = 512;
const DS4_N_HC: u32 = 4;
const DS4_N_HC_SINKHORN_ITER: u32 = 20;
const DS4_RMS_EPS: f32 = 1.0e-6;
const DS4_HC_EPS: f32 = 1.0e-6;
const DS4_EXPERT_WEIGHT_SCALE: f32 = 1.5;
const DS4_SWIGLU_CLAMP_EXP: f32 = 10.0;
const DS4_ROPE_FREQ_BASE: f32 = 10000.0;
const DS4_ROPE_SCALE_FACTOR: f32 = 16.0;
const DS4_ROPE_YARN_BETA_FAST: f32 = 32.0;
const DS4_ROPE_YARN_BETA_SLOW: f32 = 1.0;
const DS4_COMPRESS_ROPE_FREQ_BASE: f32 = 160000.0;
const DS4_ROPE_ORIG_CTX: u64 = 65536;
const DS4_TENSOR_F32: u32 = 0;
const DS4_TENSOR_F16: u32 = 1;
const DS4_TENSOR_Q8_0: u32 = 8;
const DS4_TENSOR_Q2_K: u32 = 10;
const DS4_TENSOR_Q4_K: u32 = 12;
const DS4_TENSOR_IQ2_XXS: u32 = 16;
const DS4_TENSOR_I32: u32 = 26;

pub fn validate_ds4_metadata(gguf: &Gguf) -> Result<(), Ds4ValidationError> {
    let n_layer = required_u32(gguf, "deepseek4.block_count")?;
    let n_embd = required_u32(gguf, "deepseek4.embedding_length")?;
    let n_vocab = required_u32(gguf, "deepseek4.vocab_size")?;
    let n_head = required_u32(gguf, "deepseek4.attention.head_count")?;
    let n_head_kv = required_u32(gguf, "deepseek4.attention.head_count_kv")?;
    let n_head_dim = required_u32(gguf, "deepseek4.attention.key_length")?;
    let n_value_dim = required_u32(gguf, "deepseek4.attention.value_length")?;
    let n_rot = required_u32(gguf, "deepseek4.rope.dimension_count")?;
    let n_lora_q = required_u32(gguf, "deepseek4.attention.q_lora_rank")?;
    let n_lora_o = required_u32(gguf, "deepseek4.attention.output_lora_rank")?;
    let n_out_group = required_u32(gguf, "deepseek4.attention.output_group_count")?;
    let n_expert = required_u32(gguf, "deepseek4.expert_count")?;
    let n_expert_used = required_u32(gguf, "deepseek4.expert_used_count")?;
    let n_ff_exp = required_u32(gguf, "deepseek4.expert_feed_forward_length")?;
    let n_expert_shared = required_u32(gguf, "deepseek4.expert_shared_count")?;
    let n_hash_layer = required_u32(gguf, "deepseek4.hash_layer_count")?;
    let n_expert_groups = optional_u32(gguf, "deepseek4.expert_group_count").unwrap_or(0);
    let n_group_used = optional_u32(gguf, "deepseek4.expert_group_used_count").unwrap_or(0);

    expect_u32("embedding_length", n_embd, DS4_N_EMBD)?;
    expect_u32("vocab_size", n_vocab, DS4_N_VOCAB)?;
    expect_u32("attention.head_count", n_head, DS4_N_HEAD)?;
    expect_u32("attention.key_length", n_head_dim, DS4_N_HEAD_DIM)?;
    expect_u32("attention.head_count_kv", n_head_kv, DS4_N_HEAD_KV)?;
    expect_u32("attention.value_length", n_value_dim, DS4_N_VALUE_DIM)?;
    expect_u32("rope.dimension_count", n_rot, DS4_N_ROT)?;
    expect_u32("attention.output_group_count", n_out_group, DS4_N_OUT_GROUP)?;
    expect_u32("attention.q_lora_rank", n_lora_q, DS4_N_LORA_Q)?;
    expect_u32("attention.output_lora_rank", n_lora_o, DS4_N_LORA_O)?;
    expect_u32("expert_count", n_expert, DS4_N_EXPERT)?;
    expect_u32("expert_used_count", n_expert_used, DS4_N_EXPERT_USED)?;
    expect_u32("expert_feed_forward_length", n_ff_exp, DS4_N_FF_EXP)?;
    expect_u32("expert_shared_count", n_expert_shared, DS4_N_EXPERT_SHARED)?;
    expect_u32("hash_layer_count", n_hash_layer, DS4_N_HASH_LAYER)?;
    expect_u32("expert_group_count", n_expert_groups, 0)?;
    expect_u32("expert_group_used_count", n_group_used, 0)?;

    let n_swa = required_u32(gguf, "deepseek4.attention.sliding_window")?;
    expect_u32("attention.sliding_window", n_swa, DS4_N_SWA)?;
    let n_indexer_head = required_u32(gguf, "deepseek4.attention.indexer.head_count")?;
    let n_indexer_head_dim = required_u32(gguf, "deepseek4.attention.indexer.key_length")?;
    let n_indexer_top_k = required_u32(gguf, "deepseek4.attention.indexer.top_k")?;
    expect_u32(
        "attention.indexer.head_count",
        n_indexer_head,
        DS4_N_INDEXER_HEAD,
    )?;
    expect_u32(
        "attention.indexer.key_length",
        n_indexer_head_dim,
        DS4_N_INDEXER_HEAD_DIM,
    )?;
    expect_u32(
        "attention.indexer.top_k",
        n_indexer_top_k,
        DS4_N_INDEXER_TOP_K,
    )?;
    let n_hc = required_u32(gguf, "deepseek4.hyper_connection.count")?;
    expect_u32("hyper_connection.count", n_hc, DS4_N_HC)?;
    let n_hc_sinkhorn_iter = required_u32(gguf, "deepseek4.hyper_connection.sinkhorn_iterations")?;
    expect_u32(
        "hyper_connection.sinkhorn_iterations",
        n_hc_sinkhorn_iter,
        DS4_N_HC_SINKHORN_ITER,
    )?;

    expect_u32("block_count", n_layer, DS4_N_LAYER)?;
    validate_compress_ratio_metadata(gguf)?;
    validate_swiglu_clamp_metadata(gguf)?;

    let rope_orig_ctx = required_u64(gguf, "deepseek4.rope.scaling.original_context_length")?;
    if rope_orig_ctx != DS4_ROPE_ORIG_CTX {
        return Err(Ds4ValidationError::new(format!(
            "ds4: expected rope.scaling.original_context_length={} for DeepSeek4 Flash, got {}",
            DS4_ROPE_ORIG_CTX, rope_orig_ctx
        )));
    }

    let rope_freq_base = required_f32(gguf, "deepseek4.rope.freq_base")?;
    expect_f32("rope.freq_base", rope_freq_base, DS4_ROPE_FREQ_BASE)?;
    let rope_scale_factor = required_f32(gguf, "deepseek4.rope.scaling.factor")?;
    expect_f32(
        "rope.scaling.factor",
        rope_scale_factor,
        DS4_ROPE_SCALE_FACTOR,
    )?;
    let rope_yarn_beta_fast = required_f32(gguf, "deepseek4.rope.scaling.yarn_beta_fast")?;
    expect_f32(
        "rope.scaling.yarn_beta_fast",
        rope_yarn_beta_fast,
        DS4_ROPE_YARN_BETA_FAST,
    )?;
    let rope_yarn_beta_slow = required_f32(gguf, "deepseek4.rope.scaling.yarn_beta_slow")?;
    expect_f32(
        "rope.scaling.yarn_beta_slow",
        rope_yarn_beta_slow,
        DS4_ROPE_YARN_BETA_SLOW,
    )?;
    let compress_rope_freq_base =
        required_f32(gguf, "deepseek4.attention.compress_rope_freq_base")?;
    expect_f32(
        "attention.compress_rope_freq_base",
        compress_rope_freq_base,
        DS4_COMPRESS_ROPE_FREQ_BASE,
    )?;
    let expert_weight_scale = required_f32(gguf, "deepseek4.expert_weights_scale")?;
    expect_f32(
        "expert_weights_scale",
        expert_weight_scale,
        DS4_EXPERT_WEIGHT_SCALE,
    )?;
    let rms_eps = required_f32(gguf, "deepseek4.attention.layer_norm_rms_epsilon")?;
    expect_f32("attention.layer_norm_rms_epsilon", rms_eps, DS4_RMS_EPS)?;
    let hc_eps = required_f32(gguf, "deepseek4.hyper_connection.epsilon")?;
    expect_f32("hyper_connection.epsilon", hc_eps, DS4_HC_EPS)?;
    let expert_weight_norm = required_bool(gguf, "deepseek4.expert_weights_norm")?;
    expect_bool("expert_weights_norm", expert_weight_norm, true)?;
    Ok(())
}

pub fn bind_ds4_tensors(gguf: &Gguf) -> Result<Vec<BoundTensor>, Ds4ValidationError> {
    validate_ds4_metadata(gguf)?;

    let hc_dim = u64::from(DS4_N_EMBD) * u64::from(DS4_N_HC);
    let hc_mix_dim = 2 * u64::from(DS4_N_HC) + u64::from(DS4_N_HC) * u64::from(DS4_N_HC);
    let q_dim = u64::from(DS4_N_HEAD) * u64::from(DS4_N_HEAD_DIM);
    let out_low_dim = u64::from(DS4_N_OUT_GROUP) * u64::from(DS4_N_LORA_O);
    let mut out = Vec::new();

    bind_layout(
        &mut out,
        gguf,
        "base.token_embd",
        "token_embd.weight",
        DS4_TENSOR_F16,
        &[u64::from(DS4_N_EMBD), u64::from(DS4_N_VOCAB)],
    )?;
    bind_layout(
        &mut out,
        gguf,
        "base.output_hc_base",
        "output_hc_base.weight",
        DS4_TENSOR_F32,
        &[u64::from(DS4_N_HC)],
    )?;
    bind_layout(
        &mut out,
        gguf,
        "base.output_hc_fn",
        "output_hc_fn.weight",
        DS4_TENSOR_F16,
        &[hc_dim, u64::from(DS4_N_HC)],
    )?;
    bind_layout(
        &mut out,
        gguf,
        "base.output_hc_scale",
        "output_hc_scale.weight",
        DS4_TENSOR_F32,
        &[1],
    )?;
    bind_layout(
        &mut out,
        gguf,
        "base.output_norm",
        "output_norm.weight",
        DS4_TENSOR_F32,
        &[u64::from(DS4_N_EMBD)],
    )?;
    bind_layout(
        &mut out,
        gguf,
        "base.output",
        "output.weight",
        DS4_TENSOR_Q8_0,
        &[u64::from(DS4_N_EMBD), u64::from(DS4_N_VOCAB)],
    )?;

    for layer in 0..DS4_N_LAYER {
        bind_layer_tensors(
            &mut out,
            gguf,
            "base",
            layer,
            hc_dim,
            hc_mix_dim,
            q_dim,
            out_low_dim,
        )?;
    }

    Ok(out)
}

pub fn bind_ds4_mtp_tensors(gguf: &Gguf) -> Result<Vec<BoundTensor>, Ds4ValidationError> {
    let hc_dim = u64::from(DS4_N_EMBD) * u64::from(DS4_N_HC);
    let hc_mix_dim = 2 * u64::from(DS4_N_HC) + u64::from(DS4_N_HC) * u64::from(DS4_N_HC);
    let q_dim = u64::from(DS4_N_HEAD) * u64::from(DS4_N_HEAD_DIM);
    let out_low_dim = u64::from(DS4_N_OUT_GROUP) * u64::from(DS4_N_LORA_O);
    let mut out = Vec::new();

    bind_layout(
        &mut out,
        gguf,
        "mtp.e_proj",
        "mtp.0.e_proj.weight",
        DS4_TENSOR_Q8_0,
        &[u64::from(DS4_N_EMBD), u64::from(DS4_N_EMBD)],
    )?;
    bind_layout(
        &mut out,
        gguf,
        "mtp.h_proj",
        "mtp.0.h_proj.weight",
        DS4_TENSOR_Q8_0,
        &[u64::from(DS4_N_EMBD), u64::from(DS4_N_EMBD)],
    )?;
    bind_layout(
        &mut out,
        gguf,
        "mtp.enorm",
        "mtp.0.enorm.weight",
        DS4_TENSOR_F32,
        &[u64::from(DS4_N_EMBD)],
    )?;
    bind_layout(
        &mut out,
        gguf,
        "mtp.hnorm",
        "mtp.0.hnorm.weight",
        DS4_TENSOR_F32,
        &[u64::from(DS4_N_EMBD)],
    )?;
    bind_layout(
        &mut out,
        gguf,
        "mtp.norm",
        "mtp.0.norm.weight",
        DS4_TENSOR_F32,
        &[u64::from(DS4_N_EMBD)],
    )?;
    bind_layout(
        &mut out,
        gguf,
        "mtp.hc_head_base",
        "mtp.0.hc_head_base.weight",
        DS4_TENSOR_F32,
        &[u64::from(DS4_N_HC)],
    )?;
    bind_plain_layout(
        &mut out,
        gguf,
        "mtp.hc_head_fn",
        "mtp.0.hc_head_fn.weight",
        &[hc_dim, u64::from(DS4_N_HC)],
    )?;
    bind_layout(
        &mut out,
        gguf,
        "mtp.hc_head_scale",
        "mtp.0.hc_head_scale.weight",
        DS4_TENSOR_F32,
        &[1],
    )?;

    bind_mtp_block_tensors(
        &mut out,
        gguf,
        "mtp.block",
        hc_dim,
        hc_mix_dim,
        q_dim,
        out_low_dim,
    )?;
    Ok(out)
}

pub fn parse_gguf(bytes: &[u8]) -> Result<Gguf, GgufError> {
    parse_gguf_inner(bytes, false)
}

pub fn parse_gguf_allowing_missing_tensor_data(bytes: &[u8]) -> Result<Gguf, GgufError> {
    parse_gguf_inner(bytes, true)
}

fn parse_gguf_inner(bytes: &[u8], skip_tensor_data_bounds: bool) -> Result<Gguf, GgufError> {
    let mut cursor = Cursor::new(bytes);
    let magic = cursor.u32()?;
    if magic != GGUF_MAGIC {
        return Err(GgufError::new("model is not a GGUF file"));
    }

    let version = cursor.u32()?;
    if version != 3 {
        return Err(GgufError::new("only GGUF v3 is supported"));
    }

    let n_tensors = cursor.u64()?;
    let n_metadata = cursor.u64()?;

    let mut metadata = Vec::new();
    reserve_vec(
        &mut metadata,
        usize_len(n_metadata, "metadata count")?,
        "metadata table",
    )?;
    let mut alignment = DEFAULT_ALIGNMENT;
    for _ in 0..n_metadata {
        let key = cursor.string()?;
        let value_type = cursor.u32()?;
        let value = cursor.value(value_type, 0)?;
        if key == "general.alignment" {
            if let MetadataValue::UInt32(v) = value {
                if v != 0 {
                    alignment = u64::from(v);
                }
            }
        }
        metadata.push(MetadataEntry { key, value });
    }

    let mut tensors = Vec::new();
    reserve_vec(
        &mut tensors,
        usize_len(n_tensors, "tensor count")?,
        "tensor table",
    )?;
    for _ in 0..n_tensors {
        let name = cursor.string()?;
        let ndim = cursor.u32()?;
        if ndim == 0 || ndim as usize > MAX_DIMS {
            return Err(GgufError::new(
                "tensor has an unsupported number of dimensions",
            ));
        }

        let mut dims = Vec::with_capacity(ndim as usize);
        let mut elements = 1u64;
        for _ in 0..ndim {
            let dim = cursor.u64()?;
            if dim != 0 {
                elements = elements
                    .checked_mul(dim)
                    .ok_or_else(|| GgufError::new("tensor element count overflow"))?;
            } else {
                elements = 0;
            }
            dims.push(dim);
        }

        let type_id = cursor.u32()?;
        let rel_offset = cursor.u64()?;
        let bytes_len = tensor_nbytes(type_id, elements).unwrap_or(0);
        tensors.push(TensorInfo {
            name,
            dims,
            type_id,
            rel_offset,
            abs_offset: 0,
            elements,
            bytes: bytes_len,
        });
    }

    let tensor_data_offset = align_up(cursor.position() as u64, alignment)?;
    let file_size = bytes.len() as u64;
    for tensor in &mut tensors {
        tensor.abs_offset = tensor_data_offset
            .checked_add(tensor.rel_offset)
            .ok_or_else(|| GgufError::new("tensor offset overflow"))?;
        if !skip_tensor_data_bounds
            && tensor.bytes != 0
            && (tensor.abs_offset > file_size || tensor.bytes > file_size - tensor.abs_offset)
        {
            return Err(GgufError::new("tensor points outside GGUF file"));
        }
    }

    Ok(Gguf {
        version,
        metadata,
        tensors,
        alignment,
        tensor_data_offset,
        file_size,
    })
}

pub fn value_type_name(type_id: u32) -> &'static str {
    match type_id {
        0 => "uint8",
        1 => "int8",
        2 => "uint16",
        3 => "int16",
        4 => "uint32",
        5 => "int32",
        6 => "float32",
        7 => "bool",
        8 => "string",
        9 => "array",
        10 => "uint64",
        11 => "int64",
        12 => "float64",
        _ => "unknown",
    }
}

pub fn tensor_type_name(type_id: u32) -> &'static str {
    tensor_type(type_id)
        .map(|info| info.name)
        .unwrap_or("unknown")
}

pub fn tensor_nbytes(type_id: u32, elements: u64) -> Option<u64> {
    let info = tensor_type(type_id)?;
    let blocks = elements.checked_add(info.block_elems - 1)? / info.block_elems;
    blocks.checked_mul(info.block_bytes)
}

struct TensorType {
    name: &'static str,
    block_elems: u64,
    block_bytes: u64,
}

fn tensor_type(type_id: u32) -> Option<TensorType> {
    let info = match type_id {
        0 => TensorType {
            name: "f32",
            block_elems: 1,
            block_bytes: 4,
        },
        1 => TensorType {
            name: "f16",
            block_elems: 1,
            block_bytes: 2,
        },
        2 => TensorType {
            name: "q4_0",
            block_elems: 32,
            block_bytes: 18,
        },
        3 => TensorType {
            name: "q4_1",
            block_elems: 32,
            block_bytes: 20,
        },
        6 => TensorType {
            name: "q5_0",
            block_elems: 32,
            block_bytes: 22,
        },
        7 => TensorType {
            name: "q5_1",
            block_elems: 32,
            block_bytes: 24,
        },
        8 => TensorType {
            name: "q8_0",
            block_elems: 32,
            block_bytes: 34,
        },
        9 => TensorType {
            name: "q8_1",
            block_elems: 32,
            block_bytes: 40,
        },
        10 => TensorType {
            name: "q2_k",
            block_elems: 256,
            block_bytes: 84,
        },
        11 => TensorType {
            name: "q3_k",
            block_elems: 256,
            block_bytes: 110,
        },
        12 => TensorType {
            name: "q4_k",
            block_elems: 256,
            block_bytes: 144,
        },
        13 => TensorType {
            name: "q5_k",
            block_elems: 256,
            block_bytes: 176,
        },
        14 => TensorType {
            name: "q6_k",
            block_elems: 256,
            block_bytes: 210,
        },
        15 => TensorType {
            name: "q8_k",
            block_elems: 256,
            block_bytes: 292,
        },
        16 => TensorType {
            name: "iq2_xxs",
            block_elems: 256,
            block_bytes: 66,
        },
        17 => TensorType {
            name: "iq2_xs",
            block_elems: 256,
            block_bytes: 74,
        },
        18 => TensorType {
            name: "iq3_xxs",
            block_elems: 256,
            block_bytes: 98,
        },
        19 => TensorType {
            name: "iq1_s",
            block_elems: 256,
            block_bytes: 110,
        },
        20 => TensorType {
            name: "iq4_nl",
            block_elems: 256,
            block_bytes: 50,
        },
        21 => TensorType {
            name: "iq3_s",
            block_elems: 256,
            block_bytes: 110,
        },
        22 => TensorType {
            name: "iq2_s",
            block_elems: 256,
            block_bytes: 82,
        },
        23 => TensorType {
            name: "iq4_xs",
            block_elems: 256,
            block_bytes: 136,
        },
        24 => TensorType {
            name: "i8",
            block_elems: 1,
            block_bytes: 1,
        },
        25 => TensorType {
            name: "i16",
            block_elems: 1,
            block_bytes: 2,
        },
        26 => TensorType {
            name: "i32",
            block_elems: 1,
            block_bytes: 4,
        },
        27 => TensorType {
            name: "i64",
            block_elems: 1,
            block_bytes: 8,
        },
        28 => TensorType {
            name: "f64",
            block_elems: 1,
            block_bytes: 8,
        },
        29 => TensorType {
            name: "iq1_m",
            block_elems: 256,
            block_bytes: 56,
        },
        30 => TensorType {
            name: "bf16",
            block_elems: 1,
            block_bytes: 2,
        },
        _ => return None,
    };
    Some(info)
}

fn bind_layer_tensors(
    out: &mut Vec<BoundTensor>,
    gguf: &Gguf,
    prefix: &str,
    layer: u32,
    hc_dim: u64,
    hc_mix_dim: u64,
    q_dim: u64,
    out_low_dim: u64,
) -> Result<(), Ds4ValidationError> {
    let ratio = ds4_layer_compress_ratio(layer);
    let role = |field: &str| format!("{prefix}.layer.{layer}.{field}");
    let name = |field: &str| format!("blk.{layer}.{field}.weight");

    bind_layout(
        out,
        gguf,
        &role("hc_attn_fn"),
        &name("hc_attn_fn"),
        DS4_TENSOR_F16,
        &[hc_dim, hc_mix_dim],
    )?;
    bind_layout(
        out,
        gguf,
        &role("hc_attn_scale"),
        &name("hc_attn_scale"),
        DS4_TENSOR_F32,
        &[3],
    )?;
    bind_layout(
        out,
        gguf,
        &role("hc_attn_base"),
        &name("hc_attn_base"),
        DS4_TENSOR_F32,
        &[hc_mix_dim],
    )?;
    bind_layout(
        out,
        gguf,
        &role("attn_norm"),
        &name("attn_norm"),
        DS4_TENSOR_F32,
        &[u64::from(DS4_N_EMBD)],
    )?;
    bind_layout(
        out,
        gguf,
        &role("attn_q_a"),
        &name("attn_q_a"),
        DS4_TENSOR_Q8_0,
        &[u64::from(DS4_N_EMBD), u64::from(DS4_N_LORA_Q)],
    )?;
    bind_layout(
        out,
        gguf,
        &role("attn_q_a_norm"),
        &name("attn_q_a_norm"),
        DS4_TENSOR_F32,
        &[u64::from(DS4_N_LORA_Q)],
    )?;
    bind_layout(
        out,
        gguf,
        &role("attn_q_b"),
        &name("attn_q_b"),
        DS4_TENSOR_Q8_0,
        &[u64::from(DS4_N_LORA_Q), q_dim],
    )?;
    bind_layout(
        out,
        gguf,
        &role("attn_kv"),
        &name("attn_kv"),
        DS4_TENSOR_Q8_0,
        &[u64::from(DS4_N_EMBD), u64::from(DS4_N_HEAD_DIM)],
    )?;
    bind_layout(
        out,
        gguf,
        &role("attn_kv_a_norm"),
        &name("attn_kv_a_norm"),
        DS4_TENSOR_F32,
        &[u64::from(DS4_N_HEAD_DIM)],
    )?;
    bind_layout(
        out,
        gguf,
        &role("attn_sinks"),
        &name("attn_sinks"),
        DS4_TENSOR_F32,
        &[u64::from(DS4_N_HEAD)],
    )?;
    bind_layout(
        out,
        gguf,
        &role("attn_output_a"),
        &name("attn_output_a"),
        DS4_TENSOR_Q8_0,
        &[
            u64::from(DS4_N_HEAD_DIM) * (u64::from(DS4_N_HEAD) / u64::from(DS4_N_OUT_GROUP)),
            out_low_dim,
        ],
    )?;
    bind_layout(
        out,
        gguf,
        &role("attn_output_b"),
        &name("attn_output_b"),
        DS4_TENSOR_Q8_0,
        &[out_low_dim, u64::from(DS4_N_EMBD)],
    )?;

    if ratio == 0 {
        push_absent(out, role("attn_compressor_ape"));
        push_absent(out, role("attn_compressor_kv"));
        push_absent(out, role("attn_compressor_gate"));
        push_absent(out, role("attn_compressor_norm"));
    } else {
        let coff = if ratio == 4 { 2u64 } else { 1u64 };
        let comp_width = coff * u64::from(DS4_N_HEAD_DIM);
        bind_layout(
            out,
            gguf,
            &role("attn_compressor_ape"),
            &name("attn_compressor_ape"),
            DS4_TENSOR_F16,
            &[comp_width, u64::from(ratio)],
        )?;
        bind_layout(
            out,
            gguf,
            &role("attn_compressor_kv"),
            &name("attn_compressor_kv"),
            DS4_TENSOR_F16,
            &[u64::from(DS4_N_EMBD), comp_width],
        )?;
        bind_layout(
            out,
            gguf,
            &role("attn_compressor_gate"),
            &name("attn_compressor_gate"),
            DS4_TENSOR_F16,
            &[u64::from(DS4_N_EMBD), comp_width],
        )?;
        bind_layout(
            out,
            gguf,
            &role("attn_compressor_norm"),
            &name("attn_compressor_norm"),
            DS4_TENSOR_F32,
            &[u64::from(DS4_N_HEAD_DIM)],
        )?;
    }

    if ratio == 4 {
        let index_q_dim = u64::from(DS4_N_INDEXER_HEAD) * u64::from(DS4_N_INDEXER_HEAD_DIM);
        let index_width = 2 * u64::from(DS4_N_INDEXER_HEAD_DIM);
        bind_layout(
            out,
            gguf,
            &role("indexer_attn_q_b"),
            &format!("blk.{layer}.indexer.attn_q_b.weight"),
            DS4_TENSOR_F16,
            &[u64::from(DS4_N_LORA_Q), index_q_dim],
        )?;
        bind_layout(
            out,
            gguf,
            &role("indexer_proj"),
            &format!("blk.{layer}.indexer.proj.weight"),
            DS4_TENSOR_F16,
            &[u64::from(DS4_N_EMBD), u64::from(DS4_N_INDEXER_HEAD)],
        )?;
        bind_layout(
            out,
            gguf,
            &role("indexer_compressor_ape"),
            &name("indexer_compressor_ape"),
            DS4_TENSOR_F16,
            &[index_width, u64::from(ratio)],
        )?;
        bind_layout(
            out,
            gguf,
            &role("indexer_compressor_kv"),
            &name("indexer_compressor_kv"),
            DS4_TENSOR_F16,
            &[u64::from(DS4_N_EMBD), index_width],
        )?;
        bind_layout(
            out,
            gguf,
            &role("indexer_compressor_gate"),
            &name("indexer_compressor_gate"),
            DS4_TENSOR_F16,
            &[u64::from(DS4_N_EMBD), index_width],
        )?;
        bind_layout(
            out,
            gguf,
            &role("indexer_compressor_norm"),
            &name("indexer_compressor_norm"),
            DS4_TENSOR_F32,
            &[u64::from(DS4_N_INDEXER_HEAD_DIM)],
        )?;
    } else {
        push_absent(out, role("indexer_attn_q_b"));
        push_absent(out, role("indexer_proj"));
        push_absent(out, role("indexer_compressor_ape"));
        push_absent(out, role("indexer_compressor_kv"));
        push_absent(out, role("indexer_compressor_gate"));
        push_absent(out, role("indexer_compressor_norm"));
    }

    bind_layout(
        out,
        gguf,
        &role("hc_ffn_fn"),
        &name("hc_ffn_fn"),
        DS4_TENSOR_F16,
        &[hc_dim, hc_mix_dim],
    )?;
    bind_layout(
        out,
        gguf,
        &role("hc_ffn_scale"),
        &name("hc_ffn_scale"),
        DS4_TENSOR_F32,
        &[3],
    )?;
    bind_layout(
        out,
        gguf,
        &role("hc_ffn_base"),
        &name("hc_ffn_base"),
        DS4_TENSOR_F32,
        &[hc_mix_dim],
    )?;
    bind_layout(
        out,
        gguf,
        &role("ffn_norm"),
        &name("ffn_norm"),
        DS4_TENSOR_F32,
        &[u64::from(DS4_N_EMBD)],
    )?;
    if layer < DS4_N_HASH_LAYER {
        bind_layout(
            out,
            gguf,
            &role("ffn_gate_tid2eid"),
            &name("ffn_gate_tid2eid"),
            DS4_TENSOR_I32,
            &[u64::from(DS4_N_EXPERT_USED), u64::from(DS4_N_VOCAB)],
        )?;
    } else {
        push_absent(out, role("ffn_gate_tid2eid"));
    }
    bind_layout(
        out,
        gguf,
        &role("ffn_gate_inp"),
        &name("ffn_gate_inp"),
        DS4_TENSOR_F16,
        &[u64::from(DS4_N_EMBD), u64::from(DS4_N_EXPERT)],
    )?;
    bind_optional_layout(
        out,
        gguf,
        &role("ffn_exp_probs_b"),
        &format!("blk.{layer}.exp_probs_b.bias"),
        DS4_TENSOR_F32,
        &[u64::from(DS4_N_EXPERT)],
    )?;
    let gate_type = bind_routed_layout(
        out,
        gguf,
        &role("ffn_gate_exps"),
        &name("ffn_gate_exps"),
        &[
            u64::from(DS4_N_EMBD),
            u64::from(DS4_N_FF_EXP),
            u64::from(DS4_N_EXPERT),
        ],
    )?;
    let up_type = bind_routed_layout(
        out,
        gguf,
        &role("ffn_up_exps"),
        &name("ffn_up_exps"),
        &[
            u64::from(DS4_N_EMBD),
            u64::from(DS4_N_FF_EXP),
            u64::from(DS4_N_EXPERT),
        ],
    )?;
    bind_routed_layout(
        out,
        gguf,
        &role("ffn_down_exps"),
        &name("ffn_down_exps"),
        &[
            u64::from(DS4_N_FF_EXP),
            u64::from(DS4_N_EMBD),
            u64::from(DS4_N_EXPERT),
        ],
    )?;
    if gate_type != up_type {
        return Err(Ds4ValidationError::new(format!(
            "ds4: routed gate/up experts use different quant types in layer {layer}"
        )));
    }
    bind_layout(
        out,
        gguf,
        &role("ffn_gate_shexp"),
        &name("ffn_gate_shexp"),
        DS4_TENSOR_Q8_0,
        &[u64::from(DS4_N_EMBD), u64::from(DS4_N_FF_EXP)],
    )?;
    bind_layout(
        out,
        gguf,
        &role("ffn_up_shexp"),
        &name("ffn_up_shexp"),
        DS4_TENSOR_Q8_0,
        &[u64::from(DS4_N_EMBD), u64::from(DS4_N_FF_EXP)],
    )?;
    bind_layout(
        out,
        gguf,
        &role("ffn_down_shexp"),
        &name("ffn_down_shexp"),
        DS4_TENSOR_Q8_0,
        &[u64::from(DS4_N_FF_EXP), u64::from(DS4_N_EMBD)],
    )?;
    Ok(())
}

fn bind_mtp_block_tensors(
    out: &mut Vec<BoundTensor>,
    gguf: &Gguf,
    prefix: &str,
    hc_dim: u64,
    hc_mix_dim: u64,
    q_dim: u64,
    out_low_dim: u64,
) -> Result<(), Ds4ValidationError> {
    let role = |field: &str| format!("{prefix}.{field}");
    let name = |field: &str| format!("mtp.0.{field}.weight");

    bind_plain_layout(
        out,
        gguf,
        &role("hc_attn_fn"),
        &name("hc_attn_fn"),
        &[hc_dim, hc_mix_dim],
    )?;
    bind_layout(
        out,
        gguf,
        &role("hc_attn_scale"),
        &name("hc_attn_scale"),
        DS4_TENSOR_F32,
        &[3],
    )?;
    bind_layout(
        out,
        gguf,
        &role("hc_attn_base"),
        &name("hc_attn_base"),
        DS4_TENSOR_F32,
        &[hc_mix_dim],
    )?;
    bind_layout(
        out,
        gguf,
        &role("attn_norm"),
        &name("attn_norm"),
        DS4_TENSOR_F32,
        &[u64::from(DS4_N_EMBD)],
    )?;
    bind_layout(
        out,
        gguf,
        &role("attn_q_a"),
        &name("attn_q_a"),
        DS4_TENSOR_Q8_0,
        &[u64::from(DS4_N_EMBD), u64::from(DS4_N_LORA_Q)],
    )?;
    bind_layout(
        out,
        gguf,
        &role("attn_q_a_norm"),
        &name("attn_q_a_norm"),
        DS4_TENSOR_F32,
        &[u64::from(DS4_N_LORA_Q)],
    )?;
    bind_layout(
        out,
        gguf,
        &role("attn_q_b"),
        &name("attn_q_b"),
        DS4_TENSOR_Q8_0,
        &[u64::from(DS4_N_LORA_Q), q_dim],
    )?;
    bind_layout(
        out,
        gguf,
        &role("attn_kv"),
        &name("attn_kv"),
        DS4_TENSOR_Q8_0,
        &[u64::from(DS4_N_EMBD), u64::from(DS4_N_HEAD_DIM)],
    )?;
    bind_layout(
        out,
        gguf,
        &role("attn_kv_a_norm"),
        &name("attn_kv_a_norm"),
        DS4_TENSOR_F32,
        &[u64::from(DS4_N_HEAD_DIM)],
    )?;
    bind_layout(
        out,
        gguf,
        &role("attn_sinks"),
        &name("attn_sinks"),
        DS4_TENSOR_F32,
        &[u64::from(DS4_N_HEAD)],
    )?;
    bind_layout(
        out,
        gguf,
        &role("attn_output_a"),
        &name("attn_output_a"),
        DS4_TENSOR_Q8_0,
        &[
            u64::from(DS4_N_HEAD_DIM) * (u64::from(DS4_N_HEAD) / u64::from(DS4_N_OUT_GROUP)),
            out_low_dim,
        ],
    )?;
    bind_layout(
        out,
        gguf,
        &role("attn_output_b"),
        &name("attn_output_b"),
        DS4_TENSOR_Q8_0,
        &[out_low_dim, u64::from(DS4_N_EMBD)],
    )?;
    bind_plain_layout(
        out,
        gguf,
        &role("hc_ffn_fn"),
        &name("hc_ffn_fn"),
        &[hc_dim, hc_mix_dim],
    )?;
    bind_layout(
        out,
        gguf,
        &role("hc_ffn_scale"),
        &name("hc_ffn_scale"),
        DS4_TENSOR_F32,
        &[3],
    )?;
    bind_layout(
        out,
        gguf,
        &role("hc_ffn_base"),
        &name("hc_ffn_base"),
        DS4_TENSOR_F32,
        &[hc_mix_dim],
    )?;
    bind_layout(
        out,
        gguf,
        &role("ffn_norm"),
        &name("ffn_norm"),
        DS4_TENSOR_F32,
        &[u64::from(DS4_N_EMBD)],
    )?;
    bind_plain_layout(
        out,
        gguf,
        &role("ffn_gate_inp"),
        &name("ffn_gate_inp"),
        &[u64::from(DS4_N_EMBD), u64::from(DS4_N_EXPERT)],
    )?;
    bind_layout(
        out,
        gguf,
        &role("ffn_exp_probs_b"),
        "mtp.0.exp_probs_b.bias",
        DS4_TENSOR_F32,
        &[u64::from(DS4_N_EXPERT)],
    )?;
    let gate_type = bind_routed_layout(
        out,
        gguf,
        &role("ffn_gate_exps"),
        &name("ffn_gate_exps"),
        &[
            u64::from(DS4_N_EMBD),
            u64::from(DS4_N_FF_EXP),
            u64::from(DS4_N_EXPERT),
        ],
    )?;
    let up_type = bind_routed_layout(
        out,
        gguf,
        &role("ffn_up_exps"),
        &name("ffn_up_exps"),
        &[
            u64::from(DS4_N_EMBD),
            u64::from(DS4_N_FF_EXP),
            u64::from(DS4_N_EXPERT),
        ],
    )?;
    bind_routed_layout(
        out,
        gguf,
        &role("ffn_down_exps"),
        &name("ffn_down_exps"),
        &[
            u64::from(DS4_N_FF_EXP),
            u64::from(DS4_N_EMBD),
            u64::from(DS4_N_EXPERT),
        ],
    )?;
    if gate_type != up_type {
        return Err(Ds4ValidationError::new(
            "ds4: MTP routed gate/up experts use different quant types",
        ));
    }
    bind_layout(
        out,
        gguf,
        &role("ffn_gate_shexp"),
        &name("ffn_gate_shexp"),
        DS4_TENSOR_Q8_0,
        &[u64::from(DS4_N_EMBD), u64::from(DS4_N_FF_EXP)],
    )?;
    bind_layout(
        out,
        gguf,
        &role("ffn_up_shexp"),
        &name("ffn_up_shexp"),
        DS4_TENSOR_Q8_0,
        &[u64::from(DS4_N_EMBD), u64::from(DS4_N_FF_EXP)],
    )?;
    bind_layout(
        out,
        gguf,
        &role("ffn_down_shexp"),
        &name("ffn_down_shexp"),
        DS4_TENSOR_Q8_0,
        &[u64::from(DS4_N_FF_EXP), u64::from(DS4_N_EMBD)],
    )?;
    Ok(())
}

fn bind_layout(
    out: &mut Vec<BoundTensor>,
    gguf: &Gguf,
    role: &str,
    name: &str,
    type_id: u32,
    dims: &[u64],
) -> Result<(), Ds4ValidationError> {
    let tensor = required_tensor(gguf, name)?;
    expect_tensor_layout(tensor, type_id, dims)?;
    out.push(BoundTensor {
        role: role.to_owned(),
        tensor: Some(tensor.clone()),
    });
    Ok(())
}

fn bind_optional_layout(
    out: &mut Vec<BoundTensor>,
    gguf: &Gguf,
    role: &str,
    name: &str,
    type_id: u32,
    dims: &[u64],
) -> Result<(), Ds4ValidationError> {
    if let Some(tensor) = tensor_by_name(gguf, name) {
        expect_tensor_layout(tensor, type_id, dims)?;
        out.push(BoundTensor {
            role: role.to_owned(),
            tensor: Some(tensor.clone()),
        });
    } else {
        push_absent(out, role.to_owned());
    }
    Ok(())
}

fn bind_plain_layout(
    out: &mut Vec<BoundTensor>,
    gguf: &Gguf,
    role: &str,
    name: &str,
    dims: &[u64],
) -> Result<(), Ds4ValidationError> {
    let tensor = required_tensor(gguf, name)?;
    expect_plain_layout(tensor, dims)?;
    out.push(BoundTensor {
        role: role.to_owned(),
        tensor: Some(tensor.clone()),
    });
    Ok(())
}

fn bind_routed_layout(
    out: &mut Vec<BoundTensor>,
    gguf: &Gguf,
    role: &str,
    name: &str,
    dims: &[u64],
) -> Result<u32, Ds4ValidationError> {
    let tensor = required_tensor(gguf, name)?;
    expect_routed_layout(tensor, dims)?;
    let type_id = tensor.type_id;
    out.push(BoundTensor {
        role: role.to_owned(),
        tensor: Some(tensor.clone()),
    });
    Ok(type_id)
}

fn push_absent(out: &mut Vec<BoundTensor>, role: String) {
    out.push(BoundTensor { role, tensor: None });
}

fn tensor_by_name<'a>(gguf: &'a Gguf, name: &str) -> Option<&'a TensorInfo> {
    gguf.tensors.iter().find(|tensor| tensor.name == name)
}

fn required_tensor<'a>(gguf: &'a Gguf, name: &str) -> Result<&'a TensorInfo, Ds4ValidationError> {
    tensor_by_name(gguf, name)
        .ok_or_else(|| Ds4ValidationError::new(format!("ds4: required tensor is missing: {name}")))
}

fn expect_tensor_layout(
    tensor: &TensorInfo,
    type_id: u32,
    dims: &[u64],
) -> Result<(), Ds4ValidationError> {
    if tensor.type_id != type_id {
        return Err(Ds4ValidationError::new(format!(
            "ds4: tensor {} has type {}, expected {}",
            tensor.name,
            tensor_type_name(tensor.type_id),
            tensor_type_name(type_id)
        )));
    }
    expect_tensor_dims(tensor, dims)
}

fn expect_plain_layout(tensor: &TensorInfo, dims: &[u64]) -> Result<(), Ds4ValidationError> {
    if tensor.type_id != DS4_TENSOR_F16 && tensor.type_id != DS4_TENSOR_F32 {
        return Err(Ds4ValidationError::new(format!(
            "ds4: tensor {} has type {}, expected F16 or F32",
            tensor.name,
            tensor_type_name(tensor.type_id)
        )));
    }
    expect_tensor_dims(tensor, dims)
}

fn expect_routed_layout(tensor: &TensorInfo, dims: &[u64]) -> Result<(), Ds4ValidationError> {
    if !matches!(
        tensor.type_id,
        DS4_TENSOR_IQ2_XXS | DS4_TENSOR_Q2_K | DS4_TENSOR_Q4_K
    ) {
        return Err(Ds4ValidationError::new(format!(
            "ds4: tensor {} has type {} ({}), expected a routed expert quant type",
            tensor.name,
            tensor.type_id,
            tensor_type_name(tensor.type_id)
        )));
    }
    expect_tensor_dims(tensor, dims)
}

fn expect_tensor_dims(tensor: &TensorInfo, dims: &[u64]) -> Result<(), Ds4ValidationError> {
    if tensor.dims.len() != dims.len() {
        return Err(Ds4ValidationError::new(format!(
            "ds4: tensor {} has {} dimensions, expected {}",
            tensor.name,
            tensor.dims.len(),
            dims.len()
        )));
    }
    for (idx, (got, expected)) in tensor.dims.iter().zip(dims.iter()).enumerate() {
        if got != expected {
            return Err(Ds4ValidationError::new(format!(
                "ds4: tensor {} has dim[{}]={}, expected {}",
                tensor.name, idx, got, expected
            )));
        }
    }
    Ok(())
}

fn validate_compress_ratio_metadata(gguf: &Gguf) -> Result<(), Ds4ValidationError> {
    let key = "deepseek4.attention.compress_ratios";
    let (element_type, values) = required_array(gguf, key)?;
    if element_type != 4 && element_type != 5 {
        return Err(Ds4ValidationError::new(format!(
            "ds4: required int32/uint32 array metadata key is missing: {key}"
        )));
    }
    if values.len() < DS4_N_LAYER as usize {
        return Err(Ds4ValidationError::new(
            "ds4: deepseek4.attention.compress_ratios is shorter than the layer count",
        ));
    }

    for (il, value) in values.iter().take(DS4_N_LAYER as usize).enumerate() {
        let got = match value {
            MetadataValue::UInt32(v) if element_type == 4 => *v,
            MetadataValue::Int32(v) if element_type == 5 => {
                if *v < 0 {
                    return Err(Ds4ValidationError::new(
                        "ds4: metadata array contains a negative value",
                    ));
                }
                *v as u32
            }
            _ => {
                return Err(Ds4ValidationError::new(format!(
                    "ds4: required int32/uint32 array metadata key is missing: {key}"
                )));
            }
        };
        let expected = ds4_layer_compress_ratio(il as u32);
        if got != expected {
            return Err(Ds4ValidationError::new(format!(
                "ds4: unexpected DeepSeek4 compression ratio at layer {il}: got {got}, expected {expected}"
            )));
        }
    }
    Ok(())
}

fn validate_swiglu_clamp_metadata(gguf: &Gguf) -> Result<(), Ds4ValidationError> {
    let key = "deepseek4.swiglu_clamp_exp";
    let (element_type, values) = required_array(gguf, key)?;
    if element_type != 6 && element_type != 12 {
        return Err(Ds4ValidationError::new(format!(
            "ds4: required float array metadata key is missing: {key}"
        )));
    }
    if values.len() < DS4_N_LAYER as usize {
        return Err(Ds4ValidationError::new(
            "ds4: deepseek4.swiglu_clamp_exp is shorter than the layer count",
        ));
    }

    for value in values.iter().take(DS4_N_LAYER as usize) {
        let got = match value {
            MetadataValue::Float32(v) if element_type == 6 => *v,
            MetadataValue::Float64(v) if element_type == 12 => *v as f32,
            _ => {
                return Err(Ds4ValidationError::new(format!(
                    "ds4: required float array metadata key is missing: {key}"
                )));
            }
        };
        expect_f32("swiglu_clamp_exp", got, DS4_SWIGLU_CLAMP_EXP)?;
    }
    Ok(())
}

fn ds4_layer_compress_ratio(layer: u32) -> u32 {
    if layer < 2 {
        0
    } else if (layer & 1) == 0 {
        4
    } else {
        128
    }
}

fn metadata_value<'a>(gguf: &'a Gguf, key: &str) -> Option<&'a MetadataValue> {
    gguf.metadata
        .iter()
        .find(|entry| entry.key == key)
        .map(|entry| &entry.value)
}

fn required_u32(gguf: &Gguf, key: &str) -> Result<u32, Ds4ValidationError> {
    match metadata_value(gguf, key) {
        Some(MetadataValue::UInt32(v)) => Ok(*v),
        _ => Err(Ds4ValidationError::new(format!(
            "ds4: required metadata key is missing: {key}"
        ))),
    }
}

fn optional_u32(gguf: &Gguf, key: &str) -> Option<u32> {
    match metadata_value(gguf, key) {
        Some(MetadataValue::UInt32(v)) => Some(*v),
        _ => None,
    }
}

fn required_u64(gguf: &Gguf, key: &str) -> Result<u64, Ds4ValidationError> {
    match metadata_value(gguf, key) {
        Some(MetadataValue::UInt64(v)) => Ok(*v),
        Some(MetadataValue::UInt32(v)) => Ok(u64::from(*v)),
        Some(_) => Err(Ds4ValidationError::new(format!(
            "ds4: metadata key has a non-integer type: {key}"
        ))),
        None => Err(Ds4ValidationError::new(format!(
            "ds4: required metadata key is missing: {key}"
        ))),
    }
}

fn required_f32(gguf: &Gguf, key: &str) -> Result<f32, Ds4ValidationError> {
    match metadata_value(gguf, key) {
        Some(MetadataValue::Float32(v)) => Ok(*v),
        Some(MetadataValue::Float64(v)) => Ok(*v as f32),
        Some(MetadataValue::UInt32(v)) => Ok(*v as f32),
        Some(MetadataValue::Int32(v)) => Ok(*v as f32),
        Some(value) => Err(Ds4ValidationError::new(format!(
            "ds4: metadata key has a non-float type {}: {key}",
            value.type_id()
        ))),
        None => Err(Ds4ValidationError::new(format!(
            "ds4: required metadata key is missing: {key}"
        ))),
    }
}

fn required_bool(gguf: &Gguf, key: &str) -> Result<bool, Ds4ValidationError> {
    match metadata_value(gguf, key) {
        Some(MetadataValue::Bool(v)) => Ok(*v),
        _ => Err(Ds4ValidationError::new(format!(
            "ds4: required metadata key is missing: {key}"
        ))),
    }
}

fn required_array<'a>(
    gguf: &'a Gguf,
    key: &str,
) -> Result<(u32, &'a [MetadataValue]), Ds4ValidationError> {
    match metadata_value(gguf, key) {
        Some(MetadataValue::Array {
            element_type,
            values,
        }) => Ok((*element_type, values)),
        _ => Err(Ds4ValidationError::new(format!(
            "ds4: required array metadata key is missing: {key}"
        ))),
    }
}

fn expect_u32(name: &str, got: u32, expected: u32) -> Result<(), Ds4ValidationError> {
    if got == expected {
        Ok(())
    } else {
        Err(Ds4ValidationError::new(format!(
            "ds4: expected {name}={expected} for DeepSeek4 Flash, got {got}"
        )))
    }
}

fn expect_f32(name: &str, got: f32, expected: f32) -> Result<(), Ds4ValidationError> {
    let scale = if expected.abs() > 1.0 {
        expected.abs()
    } else {
        1.0
    };
    if (got - expected).abs() <= scale * 1.0e-6 {
        Ok(())
    } else {
        Err(Ds4ValidationError::new(format!(
            "ds4: expected {name}={expected} for DeepSeek4 Flash, got {got}"
        )))
    }
}

fn expect_bool(name: &str, got: bool, expected: bool) -> Result<(), Ds4ValidationError> {
    if got == expected {
        Ok(())
    } else {
        Err(Ds4ValidationError::new(format!(
            "ds4: expected {name}={} for DeepSeek4 Flash, got {}",
            if expected { "true" } else { "false" },
            if got { "true" } else { "false" }
        )))
    }
}

fn align_up(value: u64, alignment: u64) -> Result<u64, GgufError> {
    if alignment == 0 {
        return Err(GgufError::new("alignment is zero"));
    }
    let rem = value % alignment;
    if rem == 0 {
        Ok(value)
    } else {
        value
            .checked_add(alignment - rem)
            .ok_or_else(|| GgufError::new("alignment overflow"))
    }
}

fn usize_len(value: u64, label: &str) -> Result<usize, GgufError> {
    usize::try_from(value).map_err(|_| GgufError::new(format!("{label} is too large")))
}

fn reserve_vec<T>(vec: &mut Vec<T>, additional: usize, label: &str) -> Result<(), GgufError> {
    vec.try_reserve(additional)
        .map_err(|_| GgufError::new(format!("{label} is too large")))
}

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn position(&self) -> usize {
        self.pos
    }

    fn read<const N: usize>(&mut self) -> Result<[u8; N], GgufError> {
        let end = self
            .pos
            .checked_add(N)
            .ok_or_else(|| GgufError::new("cursor overflow"))?;
        if end > self.bytes.len() {
            return Err(GgufError::new("truncated GGUF file"));
        }
        let mut out = [0u8; N];
        out.copy_from_slice(&self.bytes[self.pos..end]);
        self.pos = end;
        Ok(out)
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], GgufError> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| GgufError::new("cursor overflow"))?;
        if end > self.bytes.len() {
            return Err(GgufError::new("truncated GGUF file"));
        }
        let out = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn u8(&mut self) -> Result<u8, GgufError> {
        Ok(self.read::<1>()?[0])
    }

    fn i8(&mut self) -> Result<i8, GgufError> {
        Ok(self.u8()? as i8)
    }

    fn u16(&mut self) -> Result<u16, GgufError> {
        Ok(u16::from_le_bytes(self.read()?))
    }

    fn i16(&mut self) -> Result<i16, GgufError> {
        Ok(i16::from_le_bytes(self.read()?))
    }

    fn u32(&mut self) -> Result<u32, GgufError> {
        Ok(u32::from_le_bytes(self.read()?))
    }

    fn i32(&mut self) -> Result<i32, GgufError> {
        Ok(i32::from_le_bytes(self.read()?))
    }

    fn u64(&mut self) -> Result<u64, GgufError> {
        Ok(u64::from_le_bytes(self.read()?))
    }

    fn i64(&mut self) -> Result<i64, GgufError> {
        Ok(i64::from_le_bytes(self.read()?))
    }

    fn f32(&mut self) -> Result<f32, GgufError> {
        Ok(f32::from_le_bytes(self.read()?))
    }

    fn f64(&mut self) -> Result<f64, GgufError> {
        Ok(f64::from_le_bytes(self.read()?))
    }

    fn string(&mut self) -> Result<String, GgufError> {
        let len = usize_len(self.u64()?, "string length")?;
        let bytes = self.take(len)?;
        let text = str::from_utf8(bytes).map_err(|_| GgufError::new("invalid utf-8 string"))?;
        Ok(text.to_owned())
    }

    fn value(&mut self, type_id: u32, depth: usize) -> Result<MetadataValue, GgufError> {
        if depth > 16 {
            return Err(GgufError::new("metadata array nesting is too deep"));
        }
        let value = match type_id {
            0 => MetadataValue::UInt8(self.u8()?),
            1 => MetadataValue::Int8(self.i8()?),
            2 => MetadataValue::UInt16(self.u16()?),
            3 => MetadataValue::Int16(self.i16()?),
            4 => MetadataValue::UInt32(self.u32()?),
            5 => MetadataValue::Int32(self.i32()?),
            6 => MetadataValue::Float32(self.f32()?),
            7 => MetadataValue::Bool(self.u8()? != 0),
            8 => MetadataValue::String(self.string()?),
            9 => {
                let element_type = self.u32()?;
                let len = usize_len(self.u64()?, "metadata array length")?;
                let mut values = Vec::new();
                reserve_vec(&mut values, len, "metadata array")?;
                for _ in 0..len {
                    values.push(self.value(element_type, depth + 1)?);
                }
                MetadataValue::Array {
                    element_type,
                    values,
                }
            }
            10 => MetadataValue::UInt64(self.u64()?),
            11 => MetadataValue::Int64(self.i64()?),
            12 => MetadataValue::Float64(self.f64()?),
            _ => return Err(GgufError::new("unknown GGUF metadata type")),
        };
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_gguf, MetadataValue};

    #[test]
    fn parses_header_metadata_and_tensor_directory() {
        let fixture = fixture_bytes();
        let gguf = parse_gguf(&fixture).expect("parse fixture");

        assert_eq!(gguf.version, 3);
        assert_eq!(gguf.alignment, 64);
        assert_eq!(gguf.metadata.len(), 4);
        assert_eq!(gguf.tensors.len(), 1);
        assert_eq!(gguf.tensor_data_offset % 64, 0);

        let tensor = &gguf.tensors[0];
        assert_eq!(tensor.name, "tok.weight");
        assert_eq!(tensor.dims, vec![4]);
        assert_eq!(tensor.type_id, 0);
        assert_eq!(tensor.elements, 4);
        assert_eq!(tensor.bytes, 16);
        assert_eq!(tensor.abs_offset, gguf.tensor_data_offset);

        let ratios = gguf
            .metadata
            .iter()
            .find(|entry| entry.key == "deepseek4.attention.compress_ratios")
            .expect("compress ratios");
        assert_eq!(
            ratios.value,
            MetadataValue::Array {
                element_type: 4,
                values: vec![MetadataValue::UInt32(0), MetadataValue::UInt32(4)]
            }
        );
    }

    #[test]
    fn rejects_bad_magic() {
        let mut fixture = fixture_bytes();
        fixture[0] = 0;
        let err = parse_gguf(&fixture).unwrap_err();
        assert_eq!(err.message(), "model is not a GGUF file");
    }

    #[test]
    fn rejects_out_of_file_tensor_data() {
        let mut fixture = fixture_bytes();
        fixture.truncate(fixture.len() - 8);
        let err = parse_gguf(&fixture).unwrap_err();
        assert_eq!(err.message(), "tensor points outside GGUF file");
    }

    #[test]
    fn rejects_huge_metadata_array_before_allocation() {
        let mut fixture = Vec::new();
        fixture.extend_from_slice(&0x4655_4747u32.to_le_bytes());
        fixture.extend_from_slice(&3u32.to_le_bytes());
        fixture.extend_from_slice(&0u64.to_le_bytes());
        fixture.extend_from_slice(&1u64.to_le_bytes());
        push_string(&mut fixture, "huge.array");
        fixture.extend_from_slice(&9u32.to_le_bytes());
        fixture.extend_from_slice(&4u32.to_le_bytes());
        fixture.extend_from_slice(&u64::MAX.to_le_bytes());

        let err = parse_gguf(&fixture).unwrap_err();
        assert!(err.message().contains("metadata array"));
    }

    fn fixture_bytes() -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&0x4655_4747u32.to_le_bytes());
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&1u64.to_le_bytes());
        out.extend_from_slice(&4u64.to_le_bytes());

        push_string_entry(&mut out, "general.name", "fixture");
        push_string_entry(&mut out, "general.architecture", "deepseek4");
        push_u32_entry(&mut out, "general.alignment", 64);
        push_u32_array_entry(&mut out, "deepseek4.attention.compress_ratios", &[0, 4]);

        push_string(&mut out, "tok.weight");
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&4u64.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());

        while out.len() % 64 != 0 {
            out.push(0);
        }
        out.extend_from_slice(&[0u8; 16]);
        out
    }

    fn push_string_entry(out: &mut Vec<u8>, key: &str, value: &str) {
        push_string(out, key);
        out.extend_from_slice(&8u32.to_le_bytes());
        push_string(out, value);
    }

    fn push_u32_entry(out: &mut Vec<u8>, key: &str, value: u32) {
        push_string(out, key);
        out.extend_from_slice(&4u32.to_le_bytes());
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u32_array_entry(out: &mut Vec<u8>, key: &str, values: &[u32]) {
        push_string(out, key);
        out.extend_from_slice(&9u32.to_le_bytes());
        out.extend_from_slice(&4u32.to_le_bytes());
        out.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }

    fn push_string(out: &mut Vec<u8>, value: &str) {
        out.extend_from_slice(&(value.len() as u64).to_le_bytes());
        out.extend_from_slice(value.as_bytes());
    }
}
