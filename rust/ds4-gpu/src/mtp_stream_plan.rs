//! Model-free MTP speculative stream outcome planner.
//!
//! This module composes the M10.8 draft, verifier, suffix, and frontier plans
//! into the final stream-level outcomes pinned by the M10.8g1 contract. It
//! describes visible target-stream state only and does not execute GPU kernels.

use crate::mtp_plan::MtpCount;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MtpStreamOutcomePlan {
    pub id: &'static str,
    pub source_case: &'static str,
    pub path: &'static str,
    pub selected_subplans: &'static [&'static str],
    pub accepted_suffix: MtpCount,
    pub accepted_stream_delta: &'static str,
    pub checkpoint_delta: &'static str,
    pub logits_source: &'static str,
    pub frontier_ops: &'static [&'static str],
    pub mtp_n_raw_keep: MtpCount,
    pub cache_kvc_visibility: &'static str,
    pub fallback: &'static str,
    pub error: &'static str,
    pub live_status: &'static str,
}

pub const MTP_STREAM_OUTCOME_CASES: &[MtpStreamOutcomePlan] = &[
    MtpStreamOutcomePlan {
        id: "b300_missing_mtp_support_model",
        source_case: "b300_missing_mtp_support_model",
        path: "availability_blocker",
        selected_subplans: &[
            "mtp_plan:b300_missing_mtp_support_model",
            "mtp_draft_plan:b300_missing_mtp_support_model",
            "mtp_decode2_plan:b300_missing_mtp_support_model",
            "mtp_suffix_plan:b300_missing_mtp_support_model",
            "mtp_frontier_plan:b300_missing_mtp_support_model",
        ],
        accepted_suffix: MtpCount::Exact(0),
        accepted_stream_delta: "blocked_before_stream",
        checkpoint_delta: "0",
        logits_source: "none",
        frontier_ops: &[],
        mtp_n_raw_keep: MtpCount::Exact(0),
        cache_kvc_visibility: "none",
        fallback: "blocked_missing_mtp_model",
        error: "blocked_missing_mtp_model",
        live_status: "blocked_missing_mtp_model",
    },
    MtpStreamOutcomePlan {
        id: "mtp_disabled_after_first_token",
        source_case: "mtp_disabled_after_first_token",
        path: "guard",
        selected_subplans: &["mtp_plan:mtp_disabled_after_first_token"],
        accepted_suffix: MtpCount::Exact(0),
        accepted_stream_delta: "first_token",
        checkpoint_delta: "1",
        logits_source: "target first-token logits",
        frontier_ops: &[],
        mtp_n_raw_keep: MtpCount::Exact(0),
        cache_kvc_visibility: "first_token checkpoint only",
        fallback: "return first-token accept",
        error: "none",
        live_status: "model_free",
    },
    MtpStreamOutcomePlan {
        id: "first_draft_miss",
        source_case: "first_draft_miss",
        path: "draft_miss",
        selected_subplans: &[
            "mtp_plan:first_draft_miss",
            "mtp_draft_plan:first_draft_from_current_hc",
        ],
        accepted_suffix: MtpCount::Exact(0),
        accepted_stream_delta: "first_token",
        checkpoint_delta: "1",
        logits_source: "target first-token logits",
        frontier_ops: &[],
        mtp_n_raw_keep: MtpCount::Exact(0),
        cache_kvc_visibility: "first_token checkpoint only",
        fallback: "skip speculative work",
        error: "none",
        live_status: "blocked_missing_mtp_model",
    },
    MtpStreamOutcomePlan {
        id: "margin_skip_single_target_replay",
        source_case: "margin_skip_single_target_replay",
        path: "margin_skip",
        selected_subplans: &[
            "mtp_plan:margin_skip_single_target_replay",
            "mtp_draft_plan:first_draft_from_current_hc",
        ],
        accepted_suffix: MtpCount::Exact(1),
        accepted_stream_delta: "first_token + drafts[0]",
        checkpoint_delta: "2",
        logits_source: "target decode logits for drafts[0]",
        frontier_ops: &["keep_accepted"],
        mtp_n_raw_keep: MtpCount::Exact(1),
        cache_kvc_visibility: "two-token target checkpoint",
        fallback: "margin-skip",
        error: "none",
        live_status: "blocked_missing_mtp_model",
    },
    MtpStreamOutcomePlan {
        id: "exact_decode2_full_accept",
        source_case: "exact_decode2_full_accept",
        path: "exact_decode2",
        selected_subplans: &[
            "mtp_plan:exact_decode2_full_accept",
            "mtp_decode2_plan:exact_decode2_full_accept",
            "mtp_frontier_plan:snapshot_compressed_attn_frontier",
        ],
        accepted_suffix: MtpCount::Exact(2),
        accepted_stream_delta: "first_token + drafts[0..1]",
        checkpoint_delta: "3",
        logits_source: "decode2 logits1",
        frontier_ops: &["snapshot", "keep_accepted"],
        mtp_n_raw_keep: MtpCount::Exact(2),
        cache_kvc_visibility: "three-token target checkpoint",
        fallback: "none",
        error: "none",
        live_status: "blocked_missing_mtp_model",
    },
    MtpStreamOutcomePlan {
        id: "exact_decode2_prefix1_accept",
        source_case: "exact_decode2_prefix1_accept",
        path: "exact_decode2",
        selected_subplans: &[
            "mtp_plan:exact_decode2_prefix1_accept",
            "mtp_decode2_plan:exact_decode2_prefix1_accept",
            "mtp_frontier_plan:snapshot_compressed_attn_frontier",
            "mtp_frontier_plan:prefix1_commit_compressed_attn_frontier",
            "mtp_frontier_plan:prefix1_commit_ratio4_index_frontier",
        ],
        accepted_suffix: MtpCount::Exact(1),
        accepted_stream_delta: "first_token + drafts[0]",
        checkpoint_delta: "2",
        logits_source: "decode2 logits0",
        frontier_ops: &["snapshot", "commit_prefix1", "keep_accepted"],
        mtp_n_raw_keep: MtpCount::Exact(1),
        cache_kvc_visibility: "two-token target checkpoint",
        fallback: "none",
        error: "none",
        live_status: "blocked_missing_mtp_model",
    },
    MtpStreamOutcomePlan {
        id: "exact_decode2_failure_restore_then_sequential",
        source_case: "exact_decode2_failure_restore_then_sequential",
        path: "exact_decode2_failure",
        selected_subplans: &[
            "mtp_plan:exact_decode2_failure_restore_then_sequential",
            "mtp_decode2_plan:exact_decode2_failure_restore_then_sequential",
            "mtp_frontier_plan:restore_compressed_attn_frontier",
            "mtp_frontier_plan:restore_ratio4_index_frontier",
        ],
        accepted_suffix: MtpCount::VerifiedBySequentialFallback,
        accepted_stream_delta: "first_token + sequentially verified drafts",
        checkpoint_delta: "1 + verified_by_sequential_fallback",
        logits_source: "sequential target decode",
        frontier_ops: &["snapshot", "restore"],
        mtp_n_raw_keep: MtpCount::VerifiedBySequentialFallback,
        cache_kvc_visibility: "sequential target checkpoint only",
        fallback: "sequential safety fallback",
        error: "none unless sequential decode fails",
        live_status: "blocked_missing_mtp_model",
    },
    MtpStreamOutcomePlan {
        id: "suffix_full_accept",
        source_case: "suffix_full_accept",
        path: "suffix_verifier",
        selected_subplans: &[
            "mtp_plan:suffix_full_accept",
            "mtp_suffix_plan:suffix_full_accept",
        ],
        accepted_suffix: MtpCount::DraftN,
        accepted_stream_delta: "first_token + drafts[0..draft_n-1]",
        checkpoint_delta: "1 + draft_n",
        logits_source: "spec logits row draft_n - 1",
        frontier_ops: &["keep_accepted"],
        mtp_n_raw_keep: MtpCount::DraftN,
        cache_kvc_visibility: "verified suffix checkpoint only",
        fallback: "none",
        error: "none",
        live_status: "blocked_missing_mtp_model",
    },
    MtpStreamOutcomePlan {
        id: "suffix_prefix1_accept",
        source_case: "suffix_prefix1_accept",
        path: "suffix_verifier",
        selected_subplans: &[
            "mtp_plan:suffix_prefix1_accept",
            "mtp_suffix_plan:suffix_prefix1_accept",
            "mtp_frontier_plan:prefix1_commit_compressed_attn_frontier",
            "mtp_frontier_plan:prefix1_commit_ratio4_index_frontier",
        ],
        accepted_suffix: MtpCount::Exact(1),
        accepted_stream_delta: "first_token + drafts[0]",
        checkpoint_delta: "2",
        logits_source: "spec logits row 0",
        frontier_ops: &["commit_prefix1", "keep_accepted"],
        mtp_n_raw_keep: MtpCount::Exact(1),
        cache_kvc_visibility: "two-token target checkpoint",
        fallback: "none",
        error: "none",
        live_status: "blocked_missing_mtp_model",
    },
    MtpStreamOutcomePlan {
        id: "suffix_restore_replay_accept",
        source_case: "suffix_restore_replay_accept",
        path: "suffix_verifier_replay",
        selected_subplans: &[
            "mtp_plan:suffix_restore_replay_accept",
            "mtp_suffix_plan:suffix_restore_replay_accept",
            "mtp_frontier_plan:snapshot_compressed_attn_frontier",
            "mtp_frontier_plan:restore_compressed_attn_frontier",
            "mtp_frontier_plan:restore_ratio4_index_frontier",
        ],
        accepted_suffix: MtpCount::CommitDrafts,
        accepted_stream_delta: "first_token + drafts[0..commit_drafts-1]",
        checkpoint_delta: "1 + commit_drafts",
        logits_source: "target replay logits",
        frontier_ops: &["snapshot", "restore", "keep_accepted"],
        mtp_n_raw_keep: MtpCount::CommitDrafts,
        cache_kvc_visibility: "restored then replayed target checkpoint only",
        fallback: "target replay",
        error: "none unless replay fails",
        live_status: "blocked_missing_mtp_model",
    },
    MtpStreamOutcomePlan {
        id: "suffix_failure_restore_or_error",
        source_case: "suffix_failure_restore_or_error",
        path: "suffix_verifier_failure",
        selected_subplans: &[
            "mtp_plan:suffix_failure_restore_or_error",
            "mtp_suffix_plan:suffix_failure_restore_or_error",
            "mtp_frontier_plan:restore_compressed_attn_frontier",
            "mtp_frontier_plan:restore_ratio4_index_frontier",
        ],
        accepted_suffix: MtpCount::VerifiedBySequentialFallbackOrError,
        accepted_stream_delta: "first_token + sequential fallback or hard error",
        checkpoint_delta: "1 + verified_by_sequential_fallback_or_error",
        logits_source: "sequential target decode or none on hard error",
        frontier_ops: &["restore_or_error"],
        mtp_n_raw_keep: MtpCount::VerifiedBySequentialFallbackOrZero,
        cache_kvc_visibility: "restored checkpoint, sequential fallback, or invalidated session",
        fallback: "sequential safety fallback or MTP verifier failed",
        error: "MTP verifier failed if mutated state lacks a snapshot",
        live_status: "blocked_missing_mtp_model",
    },
    MtpStreamOutcomePlan {
        id: "sequential_safety_fallback",
        source_case: "sequential_safety_fallback",
        path: "sequential_fallback",
        selected_subplans: &["mtp_plan:sequential_safety_fallback"],
        accepted_suffix: MtpCount::Verified,
        accepted_stream_delta: "first_token + verified drafts",
        checkpoint_delta: "1 + verified",
        logits_source: "normal target decode logits",
        frontier_ops: &["keep_accepted"],
        mtp_n_raw_keep: MtpCount::Verified,
        cache_kvc_visibility: "sequential target checkpoint only",
        fallback: "target sequential verifier",
        error: "none unless target decode or logits readback fails",
        live_status: "blocked_missing_mtp_model",
    },
];

pub fn stream_case_by_id(id: &str) -> Option<MtpStreamOutcomePlan> {
    MTP_STREAM_OUTCOME_CASES
        .iter()
        .copied()
        .find(|case| case.id == id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_cases_are_ordered_and_complete() {
        let ids = [
            "b300_missing_mtp_support_model",
            "mtp_disabled_after_first_token",
            "first_draft_miss",
            "margin_skip_single_target_replay",
            "exact_decode2_full_accept",
            "exact_decode2_prefix1_accept",
            "exact_decode2_failure_restore_then_sequential",
            "suffix_full_accept",
            "suffix_prefix1_accept",
            "suffix_restore_replay_accept",
            "suffix_failure_restore_or_error",
            "sequential_safety_fallback",
        ];
        assert_eq!(MTP_STREAM_OUTCOME_CASES.len(), ids.len());
        for (case, expected) in MTP_STREAM_OUTCOME_CASES.iter().zip(ids) {
            assert_eq!(case.id, expected);
        }
    }

    #[test]
    fn unavailable_paths_expose_only_first_token_or_blocker() {
        for id in [
            "b300_missing_mtp_support_model",
            "mtp_disabled_after_first_token",
            "first_draft_miss",
        ] {
            let case = stream_case_by_id(id).expect("case");
            assert_eq!(case.accepted_suffix, MtpCount::Exact(0));
            assert!(case.frontier_ops.is_empty());
        }
    }

    #[test]
    fn exact_decode2_prefix_paths_compose_frontier_plans() {
        let full = stream_case_by_id("exact_decode2_full_accept").expect("full");
        assert_eq!(full.accepted_suffix, MtpCount::Exact(2));
        assert!(full
            .selected_subplans
            .contains(&"mtp_decode2_plan:exact_decode2_full_accept"));
        let prefix = stream_case_by_id("exact_decode2_prefix1_accept").expect("prefix");
        assert_eq!(
            prefix.frontier_ops,
            ["snapshot", "commit_prefix1", "keep_accepted"]
        );
        assert!(prefix
            .selected_subplans
            .contains(&"mtp_frontier_plan:prefix1_commit_ratio4_index_frontier"));
    }

    #[test]
    fn suffix_replay_and_failure_restore_before_fallback() {
        let replay = stream_case_by_id("suffix_restore_replay_accept").expect("replay");
        assert_eq!(
            replay.frontier_ops,
            ["snapshot", "restore", "keep_accepted"]
        );
        assert_eq!(replay.mtp_n_raw_keep, MtpCount::CommitDrafts);
        let failure = stream_case_by_id("suffix_failure_restore_or_error").expect("failure");
        assert_eq!(failure.frontier_ops, ["restore_or_error"]);
        assert_eq!(
            failure.mtp_n_raw_keep,
            MtpCount::VerifiedBySequentialFallbackOrZero
        );
    }

    #[test]
    fn sequential_fallback_keeps_only_verified_drafts_visible() {
        let case = stream_case_by_id("sequential_safety_fallback").expect("seq");
        assert_eq!(case.accepted_suffix, MtpCount::Verified);
        assert_eq!(case.mtp_n_raw_keep, MtpCount::Verified);
        assert_eq!(
            case.cache_kvc_visibility,
            "sequential target checkpoint only"
        );
    }
}
