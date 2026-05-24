//! Rust-side prefill scheduling plan for DS4 graph execution.
//!
//! This module does not execute GPU kernels. It mirrors the current-C routing
//! and chunk-boundary decisions around `metal_graph_prefill_layer_major`,
//! `metal_graph_prefill_chunked_range`, and resumed checkpoint extension so
//! later Rust prefill execution can fail closed before touching backend state.

use crate::graph_plan::{GraphPlan, N_LAYER};

pub const PREFILL_PLAN_SCHEMA: &str = "ds4.prefill_plan.v1";
pub const PREFILL_PLAN_SCOPE: &str = "m10.6a";
pub const RESUME_PREFILL_MIN_TOKENS: u32 = 4;
pub const MAX_PREFILL_PLAN_CHUNKS: usize = 8;
pub const MAX_PREFILL_PROGRESS_POINTS: usize = MAX_PREFILL_PLAN_CHUNKS + 1;

const EMPTY_CHUNK: PrefillChunk = PrefillChunk {
    start: 0,
    tokens: 0,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrefillPlanInput {
    pub ctx_size: u32,
    pub prompt_len: u32,
    pub start: u32,
    pub n_tokens: u32,
    pub checkpoint_valid: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrefillRoute {
    WholeLayerMajor,
    ChunkedRange,
    DecodeSuffix,
    CacheHit,
    Invalid,
}

impl PrefillRoute {
    pub const fn name(self) -> &'static str {
        match self {
            Self::WholeLayerMajor => "whole_layer_major",
            Self::ChunkedRange => "chunked_range",
            Self::DecodeSuffix => "decode_suffix",
            Self::CacheHit => "cache_hit",
            Self::Invalid => "invalid",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrefillChunk {
    pub start: u32,
    pub tokens: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrefillPlan {
    pub route: PrefillRoute,
    pub prefill_cap: u32,
    pub raw_cap: u32,
    pub chunk_cap: u32,
    pub first_chunk_tokens: u32,
    pub chunk_count: u32,
    pub chunks: [PrefillChunk; MAX_PREFILL_PLAN_CHUNKS],
    pub final_output_batch_row: Option<u32>,
    pub output_absolute_pos: Option<u32>,
    pub progress_point_count: u32,
    pub progress_points: [u32; MAX_PREFILL_PROGRESS_POINTS],
    pub layer_batch_calls: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrefillPlanCaseOracle {
    pub name: &'static str,
    pub input: PrefillPlanInput,
    pub expected_route: PrefillRoute,
    pub expected_prefill_cap: u32,
    pub expected_raw_cap: u32,
    pub expected_chunk_cap: u32,
    pub expected_first_chunk_tokens: u32,
    pub expected_chunk_count: u32,
    pub expected_final_output_batch_row: Option<u32>,
    pub expected_output_absolute_pos: Option<u32>,
    pub expected_progress_point_count: u32,
    pub expected_layer_batch_calls: u32,
    pub expected_chunks: [PrefillChunk; MAX_PREFILL_PLAN_CHUNKS],
    pub expected_progress_points: [u32; MAX_PREFILL_PROGRESS_POINTS],
}

impl PrefillPlanCaseOracle {
    pub fn computed(self) -> PrefillPlan {
        prefill_plan(self.input)
    }

    pub const fn expected(self) -> PrefillPlan {
        PrefillPlan {
            route: self.expected_route,
            prefill_cap: self.expected_prefill_cap,
            raw_cap: self.expected_raw_cap,
            chunk_cap: self.expected_chunk_cap,
            first_chunk_tokens: self.expected_first_chunk_tokens,
            chunk_count: self.expected_chunk_count,
            chunks: self.expected_chunks,
            final_output_batch_row: self.expected_final_output_batch_row,
            output_absolute_pos: self.expected_output_absolute_pos,
            progress_point_count: self.expected_progress_point_count,
            progress_points: self.expected_progress_points,
            layer_batch_calls: self.expected_layer_batch_calls,
        }
    }
}

macro_rules! chunks {
    () => {
        [EMPTY_CHUNK; MAX_PREFILL_PLAN_CHUNKS]
    };
    ($(($start:literal, $tokens:literal)),+ $(,)?) => {{
        let mut chunks = [EMPTY_CHUNK; MAX_PREFILL_PLAN_CHUNKS];
        let mut index = 0usize;
        $(
            chunks[index] = PrefillChunk {
                start: $start,
                tokens: $tokens,
            };
            index += 1;
        )+
        let _ = index;
        chunks
    }};
}

macro_rules! progress {
    () => {
        [0; MAX_PREFILL_PROGRESS_POINTS]
    };
    ($($pos:literal),+ $(,)?) => {{
        let mut points = [0; MAX_PREFILL_PROGRESS_POINTS];
        let mut index = 0usize;
        $(
            points[index] = $pos;
            index += 1;
        )+
        let _ = index;
        points
    }};
}

macro_rules! case {
    (
        $name:literal,
        $ctx_size:literal,
        $prompt_len:literal,
        $start:literal,
        $n_tokens:literal,
        $checkpoint_valid:literal,
        $route:ident,
        $prefill_cap:literal,
        $raw_cap:literal,
        $chunk_cap:literal,
        $first_chunk:literal,
        $chunk_count:literal,
        $final_row:expr,
        $output_pos:expr,
        $progress_count:literal,
        $layer_calls:literal,
        [$($chunk:tt)*],
        [$($progress:tt)*]
    ) => {
        PrefillPlanCaseOracle {
            name: $name,
            input: PrefillPlanInput {
                ctx_size: $ctx_size,
                prompt_len: $prompt_len,
                start: $start,
                n_tokens: $n_tokens,
                checkpoint_valid: $checkpoint_valid,
            },
            expected_route: PrefillRoute::$route,
            expected_prefill_cap: $prefill_cap,
            expected_raw_cap: $raw_cap,
            expected_chunk_cap: $chunk_cap,
            expected_first_chunk_tokens: $first_chunk,
            expected_chunk_count: $chunk_count,
            expected_final_output_batch_row: $final_row,
            expected_output_absolute_pos: $output_pos,
            expected_progress_point_count: $progress_count,
            expected_layer_batch_calls: $layer_calls,
            expected_chunks: chunks![$($chunk)*],
            expected_progress_points: progress![$($progress)*],
        }
    };
}

pub const M106A_PREFILL_PLAN_CASE_ORACLE: &[PrefillPlanCaseOracle] = &[
    case!(
        "cold_whole_prompt_22",
        32768,
        22,
        0,
        22,
        false,
        WholeLayerMajor,
        22,
        256,
        22,
        22,
        1,
        Some(21),
        Some(21),
        0,
        43,
        [(0, 22)],
        []
    ),
    case!(
        "cold_whole_prefill_cap_boundary",
        32768,
        2048,
        0,
        2048,
        false,
        WholeLayerMajor,
        2048,
        2304,
        2048,
        2048,
        1,
        Some(2047),
        Some(2047),
        0,
        43,
        [(0, 2048)],
        []
    ),
    case!(
        "cold_chunked_2052_crosses_prefill_cap",
        32768,
        2052,
        0,
        2052,
        false,
        ChunkedRange,
        2048,
        2304,
        2048,
        2048,
        2,
        Some(3),
        Some(2051),
        3,
        86,
        [(0, 2048), (2048, 4)],
        [0, 2048, 2052]
    ),
    case!(
        "resume_suffix_aligns_to_prefill_boundary",
        32768,
        4096,
        1537,
        800,
        true,
        ChunkedRange,
        2048,
        2304,
        2048,
        511,
        2,
        Some(288),
        Some(2336),
        3,
        86,
        [(1537, 511), (2048, 289)],
        [1537, 2048, 2337]
    ),
    case!(
        "resume_short_suffix_uses_decode",
        32768,
        4096,
        512,
        2,
        true,
        DecodeSuffix,
        2048,
        2304,
        0,
        0,
        0,
        None,
        None,
        0,
        86,
        [],
        []
    ),
    case!(
        "checkpoint_exact_prefix_cache_hit",
        32768,
        4096,
        4096,
        0,
        true,
        CacheHit,
        2048,
        2304,
        0,
        0,
        0,
        None,
        None,
        0,
        0,
        [],
        []
    ),
];

pub fn prefill_plan(input: PrefillPlanInput) -> PrefillPlan {
    let graph = GraphPlan::for_context(input.ctx_size, input.prompt_len, false);
    let mut plan = PrefillPlan {
        route: PrefillRoute::Invalid,
        prefill_cap: graph.prefill_cap,
        raw_cap: graph.allocated_raw_cap,
        chunk_cap: 0,
        first_chunk_tokens: 0,
        chunk_count: 0,
        chunks: [EMPTY_CHUNK; MAX_PREFILL_PLAN_CHUNKS],
        final_output_batch_row: None,
        output_absolute_pos: None,
        progress_point_count: 0,
        progress_points: [0; MAX_PREFILL_PROGRESS_POINTS],
        layer_batch_calls: 0,
    };

    if input.start > input.prompt_len
        || input.n_tokens > input.prompt_len.saturating_sub(input.start)
    {
        return plan;
    }
    if input.n_tokens == 0 {
        plan.route = if input.checkpoint_valid {
            PrefillRoute::CacheHit
        } else {
            PrefillRoute::Invalid
        };
        return plan;
    }

    if input.checkpoint_valid && input.start != 0 && input.n_tokens < RESUME_PREFILL_MIN_TOKENS {
        plan.route = PrefillRoute::DecodeSuffix;
        plan.layer_batch_calls = input.n_tokens * N_LAYER as u32;
        return plan;
    }

    if input.start == 0 && input.n_tokens <= plan.prefill_cap {
        plan.route = PrefillRoute::WholeLayerMajor;
        plan.chunk_cap = plan.prefill_cap;
        plan.first_chunk_tokens = input.n_tokens;
        plan.chunk_count = 1;
        plan.chunks[0] = PrefillChunk {
            start: 0,
            tokens: input.n_tokens,
        };
        plan.final_output_batch_row = Some(input.n_tokens - 1);
        plan.output_absolute_pos = Some(input.n_tokens - 1);
        plan.layer_batch_calls = N_LAYER as u32;
        return plan;
    }

    fill_chunked_plan(input, &mut plan);
    plan
}

fn fill_chunked_plan(input: PrefillPlanInput, plan: &mut PrefillPlan) {
    if plan.prefill_cap == 0 {
        return;
    }

    let mut chunk_cap = plan.prefill_cap;
    if input.start != 0 && chunk_cap > plan.raw_cap {
        chunk_cap = plan.raw_cap;
    }
    if chunk_cap == 0 {
        return;
    }
    plan.route = PrefillRoute::ChunkedRange;
    plan.chunk_cap = chunk_cap;

    let mut pos = input.start;
    let end = input.start + input.n_tokens;
    plan.progress_points[0] = input.start;
    plan.progress_point_count = 1;

    while pos < end {
        let remaining = end - pos;
        let mut local_cap = chunk_cap;
        if input.start != 0 {
            let boundary_offset = pos % plan.prefill_cap;
            if boundary_offset != 0 {
                let to_boundary = plan.prefill_cap - boundary_offset;
                if to_boundary < local_cap {
                    local_cap = to_boundary;
                }
            }
        }

        let chunk = if remaining < local_cap {
            remaining
        } else {
            local_cap
        };
        if chunk == 0 || plan.chunk_count as usize == MAX_PREFILL_PLAN_CHUNKS {
            plan.route = PrefillRoute::Invalid;
            return;
        }

        let chunk_index = plan.chunk_count as usize;
        plan.chunks[chunk_index] = PrefillChunk {
            start: pos,
            tokens: chunk,
        };
        if plan.chunk_count == 0 {
            plan.first_chunk_tokens = chunk;
        }
        plan.chunk_count += 1;
        plan.layer_batch_calls += N_LAYER as u32;
        pos += chunk;

        let progress_index = plan.progress_point_count as usize;
        if progress_index < MAX_PREFILL_PROGRESS_POINTS {
            plan.progress_points[progress_index] = pos;
            plan.progress_point_count += 1;
        }
    }

    let last = plan.chunks[(plan.chunk_count - 1) as usize];
    plan.final_output_batch_row = Some(last.tokens - 1);
    plan.output_absolute_pos = Some(input.start + input.n_tokens - 1);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn m106a_prefill_plan_cases_match_oracle() {
        for case in M106A_PREFILL_PLAN_CASE_ORACLE {
            assert_eq!(
                case.computed(),
                case.expected(),
                "prefill plan case drift: {}",
                case.name
            );
        }
    }

    #[test]
    fn resumed_chunks_align_to_absolute_prefill_boundaries() {
        let plan = prefill_plan(PrefillPlanInput {
            ctx_size: 32768,
            prompt_len: 4096,
            start: 1537,
            n_tokens: 800,
            checkpoint_valid: true,
        });
        assert_eq!(plan.route, PrefillRoute::ChunkedRange);
        assert_eq!(plan.chunk_count, 2);
        assert_eq!(
            plan.chunks[0],
            PrefillChunk {
                start: 1537,
                tokens: 511
            }
        );
        assert_eq!(
            plan.chunks[1],
            PrefillChunk {
                start: 2048,
                tokens: 289
            }
        );
        assert_eq!(plan.final_output_batch_row, Some(288));
    }

    #[test]
    fn short_resume_suffix_stays_on_decode_path() {
        let plan = prefill_plan(PrefillPlanInput {
            ctx_size: 32768,
            prompt_len: 4096,
            start: 512,
            n_tokens: RESUME_PREFILL_MIN_TOKENS - 1,
            checkpoint_valid: true,
        });
        assert_eq!(plan.route, PrefillRoute::DecodeSuffix);
        assert_eq!(plan.chunk_count, 0);
        assert_eq!(
            plan.layer_batch_calls,
            (RESUME_PREFILL_MIN_TOKENS - 1) * N_LAYER as u32
        );
    }
}
