use ds4_gguf::sampling::{
    fill_full_logits, sample_top_p_min_p, token_logprob, top_logprobs, SamplingParams,
    SamplingTrace, TokenScore, DS4_DEFAULT_MIN_P, DS4_NEG_INF, DS4_N_VOCAB,
};
use std::io::{self, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut out = io::BufWriter::new(io::stdout());
    write_dump(&mut out)?;
    Ok(())
}

#[derive(Clone, Copy)]
struct SamplingCase {
    name: &'static str,
    source: &'static str,
    logits: &'static [f32],
    params: SamplingParams,
    seed: u64,
}

fn write_dump<W: Write>(out: &mut W) -> io::Result<()> {
    let request_defaults = SamplingParams::defaults();
    let mut thinking_defaults = request_defaults;
    thinking_defaults.apply_thinking_defaults();
    let mut dsml_structural = request_defaults;
    dsml_structural.apply_dsml_structural();
    let cases = sampling_cases(request_defaults, thinking_defaults, dsml_structural);

    writeln!(out, "{{")?;
    writeln!(out, "  \"schema\": \"ds4.rust_sampling_oracle.v1\",")?;
    writeln!(out, "  \"source\": \"rust-fixed-logits\",")?;
    writeln!(out, "  \"n_vocab_full\": {DS4_N_VOCAB},")?;
    write!(out, "  \"defaults\": {{\"temperature\": ")?;
    write_json_f32(out, request_defaults.temperature)?;
    write!(out, ", \"top_k\": {}, \"top_p\": ", request_defaults.top_k)?;
    write_json_f32(out, request_defaults.top_p)?;
    write!(out, ", \"min_p\": ")?;
    write_json_f32(out, request_defaults.min_p)?;
    writeln!(out, "}},")?;

    writeln!(out, "  \"sampling_cases\": [")?;
    for (idx, case) in cases.iter().enumerate() {
        if idx != 0 {
            writeln!(out, ",")?;
        }
        write_sampling_case(out, case)?;
    }
    writeln!(out, "\n  ],")?;

    writeln!(out, "  \"logprob_cases\": [")?;
    write_logprob_case(
        out,
        "top_logprobs_sparse",
        "rust/ds4-gguf/src/sampling.rs:top_logprobs sparse",
        &[0.0, 1.25, -2.0, 3.5, 2.0, -0.5, f32::NEG_INFINITY, 3.5],
        6,
        &[3, 7, 6, 120],
        true,
    )?;
    write_logprob_case(
        out,
        "top_logprobs_tie_order",
        "rust/ds4-gguf/src/sampling.rs:top_logprobs tie order",
        &[2.0, 2.0, 1.0, 0.0],
        4,
        &[0, 1, 3],
        false,
    )?;
    write_logprob_case(
        out,
        "top_logprobs_nonfinite",
        "rust/ds4-gguf/src/sampling.rs:top_logprobs nonfinite",
        &[f32::NEG_INFINITY, f32::INFINITY, f32::NEG_INFINITY],
        3,
        &[0, 2],
        false,
    )?;
    writeln!(out, "\n  ]")?;
    writeln!(out, "}}")?;
    Ok(())
}

fn sampling_cases(
    request_defaults: SamplingParams,
    thinking_defaults: SamplingParams,
    dsml_structural: SamplingParams,
) -> Vec<SamplingCase> {
    const BASE: &[f32] = &[0.0, 1.25, 0.25, 3.5, 2.0, -0.5];
    const TIE: &[f32] = &[1.0, 2.0, 2.0, -3.0];
    const NONFINITE: &[f32] = &[f32::NEG_INFINITY, 0.25, -1.0, f32::INFINITY, 0.5];
    const WIDE: &[f32] = &[4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0];
    vec![
        SamplingCase {
            name: "greedy_tie_first_max",
            source: "rust/ds4-gguf/src/sampling.rs:sample_argmax",
            logits: TIE,
            params: SamplingParams {
                temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                min_p: DS4_DEFAULT_MIN_P,
            },
            seed: 0x1111_1111_1111_1111,
        },
        SamplingCase {
            name: "non_finite_logits",
            source: "rust/ds4-gguf/src/sampling.rs:sample_top_p_min_p",
            logits: NONFINITE,
            params: SamplingParams {
                temperature: 0.7,
                top_k: 0,
                top_p: 1.0,
                min_p: 0.0,
            },
            seed: 0x2222_2222_2222_2222,
        },
        SamplingCase {
            name: "full_vocab_min_p",
            source: "rust/ds4-gguf/src/sampling.rs:sample_full_vocab top_p>=1",
            logits: BASE,
            params: SamplingParams {
                temperature: 0.75,
                top_k: 0,
                top_p: 1.0,
                min_p: 0.2,
            },
            seed: 0x3333_3333_3333_3333,
        },
        SamplingCase {
            name: "full_vocab_top_p",
            source: "rust/ds4-gguf/src/sampling.rs:sample_full_vocab top_p<1",
            logits: BASE,
            params: SamplingParams {
                temperature: 0.9,
                top_k: 0,
                top_p: 0.65,
                min_p: DS4_DEFAULT_MIN_P,
            },
            seed: 0x4444_4444_4444_4444,
        },
        SamplingCase {
            name: "top_p_clamped_zero",
            source: "rust/ds4-gguf/src/sampling.rs:sample_top_p_min_p top_p clamp",
            logits: BASE,
            params: SamplingParams {
                temperature: 0.9,
                top_k: 0,
                top_p: 0.0,
                min_p: DS4_DEFAULT_MIN_P,
            },
            seed: 0x5555_5555_5555_5555,
        },
        SamplingCase {
            name: "negative_min_p_clamped",
            source: "rust/ds4-gguf/src/sampling.rs:sample_top_p_min_p min_p clamp",
            logits: BASE,
            params: SamplingParams {
                temperature: 0.9,
                top_k: 0,
                top_p: 1.0,
                min_p: -0.5,
            },
            seed: 0x6666_6666_6666_6666,
        },
        SamplingCase {
            name: "top_k_filter",
            source: "rust/ds4-gguf/src/sampling.rs:sample_top_p_min_p top_k",
            logits: BASE,
            params: SamplingParams {
                temperature: 0.8,
                top_k: 3,
                top_p: 0.8,
                min_p: DS4_DEFAULT_MIN_P,
            },
            seed: 0x7777_7777_7777_7777,
        },
        SamplingCase {
            name: "top_k_capped_to_vocab",
            source: "rust/ds4-gguf/src/sampling.rs:sample_top_p_min_p top_k cap",
            logits: BASE,
            params: SamplingParams {
                temperature: 0.8,
                top_k: 99,
                top_p: 1.0,
                min_p: 0.0,
            },
            seed: 0x8888_8888_8888_8888,
        },
        SamplingCase {
            name: "seeded_rng_draw",
            source: "rust/ds4-gguf/src/sampling.rs:sample_rng_next/sample_rng_f32",
            logits: WIDE,
            params: SamplingParams {
                temperature: 1.1,
                top_k: 4,
                top_p: 0.95,
                min_p: 0.0,
            },
            seed: 0x0123_4567_89ab_cdef,
        },
        request_case(
            "request_cli_default_ds4_cli_c",
            "ds4_cli.c:run_sampled_generation",
            BASE,
            request_defaults,
            0x0c11_0000_0000_0001,
        ),
        request_case(
            "request_openai_chat_default_ds4_server_c",
            "ds4_server.c:request_init/openai chat",
            BASE,
            request_defaults,
            0x0c11_0000_0000_0002,
        ),
        request_case(
            "request_openai_responses_default_ds4_server_c",
            "ds4_server.c:request_init/responses",
            BASE,
            request_defaults,
            0x0c11_0000_0000_0003,
        ),
        request_case(
            "request_anthropic_default_ds4_server_c",
            "ds4_server.c:request_init/anthropic",
            BASE,
            request_defaults,
            0x0c11_0000_0000_0004,
        ),
        request_case(
            "request_agent_default_ds4_agent_c",
            "ds4_agent.c:agent_config defaults",
            BASE,
            request_defaults,
            0x0c11_0000_0000_0005,
        ),
        request_case(
            "request_thinking_default_ds4_server_c",
            "ds4_server.c:thinking sampling override",
            BASE,
            thinking_defaults,
            0x0c11_0000_0000_0006,
        ),
        request_case(
            "request_dsml_structural_greedy_ds4_server_c",
            "ds4_server.c:DSML structural sampling override",
            BASE,
            dsml_structural,
            0x0c11_0000_0000_0007,
        ),
    ]
}

fn request_case(
    name: &'static str,
    source: &'static str,
    logits: &'static [f32],
    params: SamplingParams,
    seed: u64,
) -> SamplingCase {
    SamplingCase {
        name,
        source,
        logits,
        params,
        seed,
    }
}

fn write_sampling_case<W: Write>(out: &mut W, case: &SamplingCase) -> io::Result<()> {
    let mut rng = case.seed;
    let mut trace = SamplingTrace::new(case.seed, case.params);
    let selected = sample_top_p_min_p(case.logits, case.params, &mut rng, Some(&mut trace));
    let mut actual_rng = case.seed;
    let actual_selected = sample_top_p_min_p(case.logits, case.params, &mut actual_rng, None);
    trace.actual_selected = actual_selected;
    trace.actual_rng_after = actual_rng;
    trace.matches_actual = selected == actual_selected && trace.rng_after == actual_rng;

    write!(out, "    {{\"name\": ")?;
    write_json_string(out, case.name)?;
    write!(out, ", \"source\": ")?;
    write_json_string(out, case.source)?;
    write!(
        out,
        ", \"n_vocab\": {}, \"params\": {{\"temperature\": ",
        case.logits.len()
    )?;
    write_json_f32(out, case.params.temperature)?;
    write!(out, ", \"top_k\": {}, \"top_p\": ", case.params.top_k)?;
    write_json_f32(out, case.params.top_p)?;
    write!(out, ", \"min_p\": ")?;
    write_json_f32(out, case.params.min_p)?;
    write!(out, ", \"seed\": {}}}", case.seed)?;
    write!(
        out,
        ", \"effective\": {{\"top_k\": {}, \"top_p\": ",
        trace.effective_top_k
    )?;
    write_json_f32(out, trace.effective_top_p)?;
    write!(out, ", \"min_p\": ")?;
    write_json_f32(out, trace.effective_min_p)?;
    write!(out, "}}, \"logits\": ")?;
    write_logits(out, case.logits)?;
    write!(
        out,
        ", \"selected\": {}, \"actual_selected\": {}, \"matches_actual\": {}, \"rng_before\": {}, \"rng_after\": {}, \"actual_rng_after\": {}, \"greedy\": {}, \"finite_count\": {}, \"filtered_count\": {}, \"max_logit\": ",
        trace.selected,
        trace.actual_selected,
        trace.matches_actual,
        trace.rng_before,
        trace.rng_after,
        trace.actual_rng_after,
        trace.greedy,
        trace.finite_count,
        trace.filtered.len()
    )?;
    write_json_f32(out, trace.max_logit)?;
    write!(out, ", \"sum\": ")?;
    write_json_f32(out, trace.sum)?;
    write!(out, ", \"filtered_sum\": ")?;
    write_json_f32(out, trace.filtered_sum)?;
    write!(out, ", \"rng_unit\": ")?;
    write_json_f32(out, trace.rng_unit)?;
    write!(out, ", \"filtered_candidates\": [")?;
    for (idx, candidate) in trace.filtered.iter().enumerate() {
        if idx != 0 {
            write!(out, ", ")?;
        }
        write!(out, "{{\"id\": {}, \"logit\": ", candidate.id)?;
        write_json_f32(out, candidate.logit)?;
        write!(out, ", \"weight\": ")?;
        write_json_f32(out, candidate.weight)?;
        write!(out, ", \"normalized_prob\": ")?;
        write_json_f32(out, candidate.normalized_prob)?;
        write!(out, "}}")?;
    }
    write!(out, "]}}")
}

fn write_logprob_case<W: Write>(
    out: &mut W,
    name: &str,
    source: &str,
    logits: &[f32],
    k: usize,
    token_queries: &[usize],
    first: bool,
) -> io::Result<()> {
    if !first {
        writeln!(out, ",")?;
    }
    let full_logits = fill_full_logits(logits);
    let (returned, scores) = top_logprobs(&full_logits, k);
    write!(out, "    {{\"name\": ")?;
    write_json_string(out, name)?;
    write!(out, ", \"source\": ")?;
    write_json_string(out, source)?;
    write!(
        out,
        ", \"n_vocab\": {}, \"background_logit\": ",
        logits.len()
    )?;
    write_json_f32(out, DS4_NEG_INF)?;
    write!(
        out,
        ", \"top_k\": {k}, \"returned\": {returned}, \"logits\": "
    )?;
    write_logits(out, logits)?;
    write!(out, ", \"scores\": [")?;
    for (idx, score) in scores.iter().enumerate() {
        if idx != 0 {
            write!(out, ", ")?;
        }
        write_score(out, score)?;
    }
    write!(out, "], \"token_logprobs\": [")?;
    for (idx, &token) in token_queries.iter().enumerate() {
        if idx != 0 {
            write!(out, ", ")?;
        }
        let score = token_logprob(&full_logits, token).unwrap_or(TokenScore {
            id: 0,
            logit: 0.0,
            logprob: 0.0,
        });
        write!(
            out,
            "{{\"token\": {token}, \"ok\": {}, \"score\": ",
            token < DS4_N_VOCAB
        )?;
        write_score(out, &score)?;
        write!(out, "}}")?;
    }
    write!(out, "]}}")
}

fn write_logits<W: Write>(out: &mut W, logits: &[f32]) -> io::Result<()> {
    write!(out, "[")?;
    for (idx, value) in logits.iter().enumerate() {
        if idx != 0 {
            write!(out, ", ")?;
        }
        write!(out, "{{\"id\": {idx}, \"value\": ")?;
        write_json_f32(out, *value)?;
        write!(out, "}}")?;
    }
    write!(out, "]")
}

fn write_score<W: Write>(out: &mut W, score: &TokenScore) -> io::Result<()> {
    write!(out, "{{\"id\": {}, \"logit\": ", score.id)?;
    write_json_f32(out, score.logit)?;
    write!(out, ", \"logprob\": ")?;
    write_json_f32(out, score.logprob)?;
    write!(out, "}}")
}

fn write_json_f32<W: Write>(out: &mut W, value: f32) -> io::Result<()> {
    if value.is_nan() {
        write!(out, "\"nan\"")
    } else if value.is_infinite() {
        if value.is_sign_negative() {
            write!(out, "\"-inf\"")
        } else {
            write!(out, "\"inf\"")
        }
    } else {
        write!(out, "{value:.9e}")
    }
}

fn write_json_string<W: Write>(out: &mut W, value: &str) -> io::Result<()> {
    write!(out, "\"")?;
    for ch in value.chars() {
        match ch {
            '"' => write!(out, "\\\"")?,
            '\\' => write!(out, "\\\\")?,
            '\n' => write!(out, "\\n")?,
            '\r' => write!(out, "\\r")?,
            '\t' => write!(out, "\\t")?,
            ch if ch < ' ' => write!(out, "\\u{:04x}", ch as u32)?,
            ch => write!(out, "{ch}")?,
        }
    }
    write!(out, "\"")
}
