//! Model-free MTP runtime guard plan.
//!
//! This module pins the Rust runtime surfaces that can reach MTP options before
//! any model-backed MTP runtime execution is introduced. It composes the
//! unavailable stream outcomes from M10.8g2 with static Rust/C dispatch anchors.

use crate::mtp_plan::MtpCount;
use crate::mtp_stream_plan::stream_case_by_id;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MtpRuntimeGuardPlan {
    pub id: &'static str,
    pub surface: &'static str,
    pub runtime_entry: &'static str,
    pub source_stream_case: &'static str,
    pub guard_condition: &'static str,
    pub dispatch: &'static str,
    pub selected_stream_plan: &'static str,
    pub accepted_stream_delta: &'static str,
    pub checkpoint_delta: &'static str,
    pub logits_source: &'static str,
    pub mtp_n_raw_keep: MtpCount,
    pub cache_kvc_visibility: &'static str,
    pub target_stream_visibility: &'static str,
    pub fallback: &'static str,
    pub error: &'static str,
    pub live_status: &'static str,
    pub source_anchors: &'static [&'static str],
}

pub const MTP_RUNTIME_GUARD_CASES: &[MtpRuntimeGuardPlan] = &[
    MtpRuntimeGuardPlan {
        id: "engine_options_default_mtp_off",
        surface: "ds4_engine::EngineOptions",
        runtime_entry: "EngineOptions::new",
        source_stream_case: "mtp_disabled_after_first_token",
        guard_condition: "mtp_path_none_and_mtp_draft_tokens_one",
        dispatch: "target_only_runtime_default",
        selected_stream_plan: "mtp_stream_plan:mtp_disabled_after_first_token",
        accepted_stream_delta: "first_token",
        checkpoint_delta: "1",
        logits_source: "target first-token logits",
        mtp_n_raw_keep: MtpCount::Exact(0),
        cache_kvc_visibility: "first_token checkpoint only",
        target_stream_visibility: "target_only",
        fallback: "return first-token accept",
        error: "none",
        live_status: "model_free",
        source_anchors: &[
            "rust/ds4-engine/src/lib.rs::pub mtp_path: Option<&'a str>,",
            "rust/ds4-engine/src/lib.rs::mtp_path: None,",
            "rust/ds4-engine/src/lib.rs::mtp_draft_tokens: 1,",
            "rust/ds4-engine/src/lib.rs::options_default_runtime_flags_match_c_cli_inspect",
        ],
    },
    MtpRuntimeGuardPlan {
        id: "one_shot_runtime_mtp_off",
        surface: "ds4-cli-one-shot-rs",
        runtime_entry: "parse_cli_config_to_generate_argmax_text",
        source_stream_case: "mtp_disabled_after_first_token",
        guard_condition: "mtp_path_none_or_mtp_draft_tokens_one",
        dispatch: "target_only_argmax_runtime",
        selected_stream_plan: "mtp_stream_plan:mtp_disabled_after_first_token",
        accepted_stream_delta: "first_token",
        checkpoint_delta: "1",
        logits_source: "target first-token logits",
        mtp_n_raw_keep: MtpCount::Exact(0),
        cache_kvc_visibility: "first_token checkpoint only",
        target_stream_visibility: "target_only",
        fallback: "return first-token accept",
        error: "none",
        live_status: "model_free",
        source_anchors: &[
            "rust/ds4-gguf/src/cli_parse.rs::\"--mtp\" => state.config.mtp_path = Some(value),",
            "rust/ds4-gguf/src/cli_parse.rs::if arg == \"--mtp-draft\"",
            "rust/ds4-engine/src/bin/ds4-cli-one-shot-rs.rs::options.mtp_path = config.mtp_path.as_deref();",
            "rust/ds4-engine/src/bin/ds4-cli-one-shot-rs.rs::options.mtp_draft_tokens = config.mtp_draft_tokens;",
            "rust/ds4-engine/src/bin/ds4-cli-one-shot-rs.rs::engine.generate_argmax_text(",
        ],
    },
    MtpRuntimeGuardPlan {
        id: "interactive_runtime_mtp_off",
        surface: "ds4-cli-interactive-rs",
        runtime_entry: "parse_cli_config_to_chat_session",
        source_stream_case: "mtp_disabled_after_first_token",
        guard_condition: "mtp_path_none_or_mtp_draft_tokens_one",
        dispatch: "target_only_interactive_runtime",
        selected_stream_plan: "mtp_stream_plan:mtp_disabled_after_first_token",
        accepted_stream_delta: "first_token",
        checkpoint_delta: "1",
        logits_source: "target first-token logits",
        mtp_n_raw_keep: MtpCount::Exact(0),
        cache_kvc_visibility: "first_token checkpoint only",
        target_stream_visibility: "target_only",
        fallback: "return first-token accept",
        error: "none",
        live_status: "model_free",
        source_anchors: &[
            "rust/ds4-engine/src/bin/ds4-cli-interactive-rs.rs::options.mtp_path = config.mtp_path.as_deref();",
            "rust/ds4-engine/src/bin/ds4-cli-interactive-rs.rs::options.mtp_draft_tokens = config.mtp_draft_tokens;",
            "rust/ds4-engine/src/bin/ds4-cli-interactive-rs.rs::engine.create_chat_session(",
            "rust/ds4-engine/src/bin/ds4-cli-interactive-rs.rs::chat.run_turn_to_writer(",
        ],
    },
    MtpRuntimeGuardPlan {
        id: "server_runtime_mtp_off",
        surface: "ds4-server-runtime-rs",
        runtime_entry: "engine_options_from_config",
        source_stream_case: "mtp_disabled_after_first_token",
        guard_condition: "mtp_path_none_or_mtp_draft_tokens_one",
        dispatch: "target_only_server_runtime",
        selected_stream_plan: "mtp_stream_plan:mtp_disabled_after_first_token",
        accepted_stream_delta: "first_token",
        checkpoint_delta: "1",
        logits_source: "target first-token logits",
        mtp_n_raw_keep: MtpCount::Exact(0),
        cache_kvc_visibility: "first_token checkpoint only",
        target_stream_visibility: "target_only_with_runtime_cache_ledger",
        fallback: "return first-token accept",
        error: "none",
        live_status: "model_free",
        source_anchors: &[
            "rust/ds4-engine/src/bin/ds4-server-runtime-rs.rs::fn engine_options_from_config(config: &ServerConfig) -> EngineOptions<'_> {",
            "rust/ds4-engine/src/bin/ds4-server-runtime-rs.rs::options.mtp_path = config.mtp_path.as_deref();",
            "rust/ds4-engine/src/bin/ds4-server-runtime-rs.rs::options.mtp_draft_tokens = config.mtp_draft_tokens;",
            "rust/ds4-engine/src/bin/ds4-server-runtime-rs.rs::RuntimeCacheDecision::from_runtime(",
        ],
    },
    MtpRuntimeGuardPlan {
        id: "argmax_session_runtime_non_mtp",
        surface: "ds4-argmax-runtime-rs+ds4-session-runtime-rs",
        runtime_entry: "runtime_specific_parsers_without_mtp_flags",
        source_stream_case: "mtp_disabled_after_first_token",
        guard_condition: "no_mtp_cli_surface",
        dispatch: "target_only_legacy_runtime",
        selected_stream_plan: "mtp_stream_plan:mtp_disabled_after_first_token",
        accepted_stream_delta: "first_token",
        checkpoint_delta: "1",
        logits_source: "target first-token logits",
        mtp_n_raw_keep: MtpCount::Exact(0),
        cache_kvc_visibility: "first_token checkpoint only",
        target_stream_visibility: "target_only",
        fallback: "return first-token accept",
        error: "none",
        live_status: "model_free",
        source_anchors: &[
            "rust/ds4-engine/src/bin/ds4-argmax-runtime-rs.rs::let mut engine_options = EngineOptions::new(&config.model_path, config.backend);",
            "rust/ds4-engine/src/bin/ds4-argmax-runtime-rs.rs::engine.generate_argmax_text(",
            "rust/ds4-engine/src/bin/ds4-session-runtime-rs.rs::let mut engine_options = EngineOptions::new(&config.model_path, config.backend);",
            "rust/ds4-engine/src/bin/ds4-session-runtime-rs.rs::engine.generate_sampled_text(",
        ],
    },
    MtpRuntimeGuardPlan {
        id: "first_draft_miss_no_drift",
        surface: "current-C speculative dispatch guard",
        runtime_entry: "ds4_session_eval_speculative_argmax",
        source_stream_case: "first_draft_miss",
        guard_condition: "spec_dispatch_allowed_but_no_valid_draft",
        dispatch: "target_first_token_only",
        selected_stream_plan: "mtp_stream_plan:first_draft_miss",
        accepted_stream_delta: "first_token",
        checkpoint_delta: "1",
        logits_source: "target first-token logits",
        mtp_n_raw_keep: MtpCount::Exact(0),
        cache_kvc_visibility: "first_token checkpoint only",
        target_stream_visibility: "target_only",
        fallback: "skip speculative work",
        error: "none",
        live_status: "blocked_missing_mtp_model",
        source_anchors: &[
            "ds4_cli.c::ds4_engine_mtp_draft_tokens(engine) > 1",
            "ds4_cli.c::getenv(\"DS4_MTP_SPEC_DISABLE\") == NULL",
            "ds4.c::int ds4_engine_mtp_draft_tokens(ds4_engine *e) {",
            "ds4.c::return e && e->backend != DS4_BACKEND_CPU && e->mtp_ready ? e->mtp_draft_tokens : 0;",
        ],
    },
    MtpRuntimeGuardPlan {
        id: "b300_missing_mtp_support_runtime_blocker",
        surface: "B300 support-artifact guard",
        runtime_entry: "missing_support_artifact_check",
        source_stream_case: "b300_missing_mtp_support_model",
        guard_condition: "mtp_path_configured_but_support_artifact_absent",
        dispatch: "block_before_runtime_stream",
        selected_stream_plan: "mtp_stream_plan:b300_missing_mtp_support_model",
        accepted_stream_delta: "blocked_before_stream",
        checkpoint_delta: "0",
        logits_source: "none",
        mtp_n_raw_keep: MtpCount::Exact(0),
        cache_kvc_visibility: "none",
        target_stream_visibility: "blocked_before_stream",
        fallback: "blocked_missing_mtp_model",
        error: "blocked_missing_mtp_model",
        live_status: "blocked_missing_mtp_model",
        source_anchors: &[
            "ds4-parity/README.md::test ! -e /workspace/ds4/missing-mtp.gguf",
            "ds4-parity/README.md::printf \"mtp_candidates=%s\\n\" \"$candidates\"; test -z \"$candidates\"",
            "rust/ds4-engine/src/bin/ds4-cli-one-shot-rs.rs::Err(err) if err.open_failed_code().is_some() => return Ok(1),",
            "rust/ds4-engine/src/bin/ds4-server-runtime-rs.rs::Err(err) if err.open_failed_code().is_some() => return Ok(1),",
        ],
    },
];

pub fn runtime_guard_case_by_id(id: &str) -> Option<MtpRuntimeGuardPlan> {
    MTP_RUNTIME_GUARD_CASES
        .iter()
        .copied()
        .find(|case| case.id == id)
}

pub fn runtime_guard_case_matches_stream(case: MtpRuntimeGuardPlan) -> bool {
    let Some(stream) = stream_case_by_id(case.source_stream_case) else {
        return false;
    };
    case.accepted_stream_delta == stream.accepted_stream_delta
        && case.checkpoint_delta == stream.checkpoint_delta
        && case.logits_source == stream.logits_source
        && case.mtp_n_raw_keep == stream.mtp_n_raw_keep
        && case.cache_kvc_visibility == stream.cache_kvc_visibility
        && case.fallback == stream.fallback
        && case.error == stream.error
        && case.live_status == stream.live_status
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_guard_cases_are_ordered_and_complete() {
        let ids = [
            "engine_options_default_mtp_off",
            "one_shot_runtime_mtp_off",
            "interactive_runtime_mtp_off",
            "server_runtime_mtp_off",
            "argmax_session_runtime_non_mtp",
            "first_draft_miss_no_drift",
            "b300_missing_mtp_support_runtime_blocker",
        ];
        assert_eq!(MTP_RUNTIME_GUARD_CASES.len(), ids.len());
        for (case, expected) in MTP_RUNTIME_GUARD_CASES.iter().zip(ids) {
            assert_eq!(case.id, expected);
        }
    }

    #[test]
    fn runtime_guard_cases_match_selected_stream_outcomes() {
        for case in MTP_RUNTIME_GUARD_CASES {
            assert!(
                runtime_guard_case_matches_stream(*case),
                "{} does not match {}",
                case.id,
                case.source_stream_case
            );
        }
    }

    #[test]
    fn disabled_runtime_surfaces_expose_only_target_stream() {
        for id in [
            "engine_options_default_mtp_off",
            "one_shot_runtime_mtp_off",
            "interactive_runtime_mtp_off",
            "argmax_session_runtime_non_mtp",
        ] {
            let case = runtime_guard_case_by_id(id).expect("case");
            assert_eq!(case.source_stream_case, "mtp_disabled_after_first_token");
            assert_eq!(case.mtp_n_raw_keep, MtpCount::Exact(0));
            assert_eq!(case.target_stream_visibility, "target_only");
            assert_eq!(case.error, "none");
        }
        assert_eq!(
            runtime_guard_case_by_id("server_runtime_mtp_off")
                .expect("server")
                .target_stream_visibility,
            "target_only_with_runtime_cache_ledger"
        );
    }

    #[test]
    fn first_draft_miss_preserves_first_token_only_state() {
        let case = runtime_guard_case_by_id("first_draft_miss_no_drift").expect("case");
        assert_eq!(case.source_stream_case, "first_draft_miss");
        assert_eq!(case.accepted_stream_delta, "first_token");
        assert_eq!(case.checkpoint_delta, "1");
        assert_eq!(case.cache_kvc_visibility, "first_token checkpoint only");
        assert_eq!(case.fallback, "skip speculative work");
    }

    #[test]
    fn missing_support_blocks_before_stream_mutation() {
        let case =
            runtime_guard_case_by_id("b300_missing_mtp_support_runtime_blocker").expect("case");
        assert_eq!(case.source_stream_case, "b300_missing_mtp_support_model");
        assert_eq!(case.accepted_stream_delta, "blocked_before_stream");
        assert_eq!(case.checkpoint_delta, "0");
        assert_eq!(case.logits_source, "none");
        assert_eq!(case.cache_kvc_visibility, "none");
        assert_eq!(case.error, "blocked_missing_mtp_model");
    }
}
