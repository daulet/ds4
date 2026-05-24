//! Model-free MTP suffix verifier orchestration plan.
//!
//! The live suffix-verifier smoke is blocked until the B300 workspace has an MTP
//! support GGUF. This plan pins the Rust-owned microbatch verifier decisions
//! against the current C suffix path before GPU execution is ported.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MtpSuffixVerifierCase {
    pub id: &'static str,
    pub source_functions: &'static [&'static str],
    pub command_boundaries: &'static [&'static str],
    pub verifier_input: &'static str,
    pub capture_prefix1: &'static str,
    pub snapshot_requirement: &'static str,
    pub commit_rule: &'static str,
    pub readbacks: &'static [&'static str],
    pub frontier_ops: &'static [&'static str],
    pub checkpoint_action: &'static str,
    pub accepted_suffix: &'static str,
    pub logits_source: &'static str,
    pub mtp_n_raw_keep: &'static str,
    pub fallback: &'static str,
    pub failure_action: &'static str,
    pub live_status: &'static str,
}

pub const MTP_SUFFIX_BASE_FUNCTIONS: &[&str] = &[
    "metal_graph_verify_suffix_tops",
    "metal_graph_read_spec_logits_row",
];

pub const MTP_SUFFIX_VERIFIER_READBACKS: &[&str] =
    &["row_tops[0..draft_n-2]", "spec_logits selected row"];

pub const MTP_SUFFIX_ORCHESTRATION_CASES: &[MtpSuffixVerifierCase] = &[
    MtpSuffixVerifierCase {
        id: "b300_missing_mtp_support_model",
        source_functions: &[],
        command_boundaries: &[],
        verifier_input: "none",
        capture_prefix1: "none",
        snapshot_requirement: "none",
        commit_rule: "blocked_missing_mtp_model",
        readbacks: &[],
        frontier_ops: &[],
        checkpoint_action: "no session is created",
        accepted_suffix: "0",
        logits_source: "none",
        mtp_n_raw_keep: "0",
        fallback: "blocked_missing_mtp_model",
        failure_action: "blocked_missing_mtp_model",
        live_status: "blocked_missing_mtp_model",
    },
    MtpSuffixVerifierCase {
        id: "suffix_full_accept",
        source_functions: MTP_SUFFIX_BASE_FUNCTIONS,
        command_boundaries: &["mtp_suffix_tops"],
        verifier_input: "checkpoint with all draft tokens appended",
        capture_prefix1: "branch-computed capture_prefix1",
        snapshot_requirement: "optional when snapshot_required is true",
        commit_rule: "commit_drafts == draft_n",
        readbacks: &["row_tops[0..draft_n-2]", "spec_logits[draft_n - 1]"],
        frontier_ops: &["keep_accepted"],
        checkpoint_action: "checkpoint already contains all draft tokens",
        accepted_suffix: "draft_n",
        logits_source: "spec logits row draft_n - 1",
        mtp_n_raw_keep: "draft_n",
        fallback: "none",
        failure_action: "none",
        live_status: "blocked_missing_mtp_model",
    },
    MtpSuffixVerifierCase {
        id: "suffix_prefix1_accept",
        source_functions: &[
            "metal_graph_verify_suffix_tops",
            "spec_frontier_commit_prefix1",
            "metal_graph_read_spec_logits_row",
        ],
        command_boundaries: &["mtp_suffix_tops", "spec_frontier_commit_prefix1"],
        verifier_input: "checkpoint with drafts[0] and drafts[1] appended",
        capture_prefix1: "draft_n == 2 && capture_prefix1",
        snapshot_requirement: "snapshot not required unless forced",
        commit_rule: "commit_drafts == 1",
        readbacks: &["row_tops[0]", "spec_logits[0]"],
        frontier_ops: &["commit_prefix1", "keep_accepted"],
        checkpoint_action: "reset to start then push drafts[0]",
        accepted_suffix: "1",
        logits_source: "spec logits row 0",
        mtp_n_raw_keep: "1",
        fallback: "none",
        failure_action: "none",
        live_status: "blocked_missing_mtp_model",
    },
    MtpSuffixVerifierCase {
        id: "suffix_restore_replay_accept",
        source_functions: &[
            "spec_frontier_snapshot",
            "metal_graph_verify_suffix_tops",
            "spec_frontier_restore",
            "metal_graph_eval_token_raw_swa",
            "metal_graph_read_spec_logits_row",
        ],
        command_boundaries: &[
            "spec_frontier_snapshot",
            "mtp_suffix_tops",
            "spec_frontier_restore",
        ],
        verifier_input: "checkpoint with all draft tokens appended",
        capture_prefix1: "!capture_prefix1 or replay-required suffix",
        snapshot_requirement: "snapshot_required",
        commit_rule: "commit_drafts < draft_n",
        readbacks: &["target replay logits", "or spec_logits[commit_drafts - 1]"],
        frontier_ops: &["snapshot", "restore", "keep_accepted"],
        checkpoint_action: "restore to start then replay accepted drafts",
        accepted_suffix: "commit_drafts",
        logits_source: "target replay logits",
        mtp_n_raw_keep: "commit_drafts",
        fallback: "target replay",
        failure_action: "restore_before_replay",
        live_status: "blocked_missing_mtp_model",
    },
    MtpSuffixVerifierCase {
        id: "suffix_exact_replay_debug_accept",
        source_functions: &[
            "spec_frontier_snapshot",
            "metal_graph_verify_suffix_tops",
            "spec_frontier_restore",
            "metal_graph_eval_token_raw_swa",
        ],
        command_boundaries: &[
            "spec_frontier_snapshot",
            "mtp_suffix_tops",
            "spec_frontier_restore",
        ],
        verifier_input: "checkpoint with all draft tokens appended",
        capture_prefix1: "any",
        snapshot_requirement: "required by DS4_MTP_EXACT_REPLAY",
        commit_rule: "exact_replay_debug && have_frontier",
        readbacks: &["target replay logits"],
        frontier_ops: &["snapshot", "restore", "keep_accepted"],
        checkpoint_action: "restore to start then exact-replay committed drafts",
        accepted_suffix: "commit_drafts",
        logits_source: "target replay logits",
        mtp_n_raw_keep: "commit_drafts",
        fallback: "exact replay debug",
        failure_action: "restore_failure_falls_through",
        live_status: "blocked_missing_mtp_model",
    },
    MtpSuffixVerifierCase {
        id: "suffix_failure_restore_or_error",
        source_functions: &["metal_graph_verify_suffix_tops", "spec_frontier_restore"],
        command_boundaries: &["mtp_suffix_tops", "spec_frontier_restore"],
        verifier_input: "checkpoint reset to start after verifier attempt",
        capture_prefix1: "any",
        snapshot_requirement:
            "restore if have_frontier; error if verifier mutated without frontier",
        commit_rule: "verifier, prefix commit, replay, or logits read failed",
        readbacks: &[],
        frontier_ops: &["restore_or_error"],
        checkpoint_action:
            "reset to start; restore if snapshot exists; otherwise error if verifier mutated",
        accepted_suffix: "verified_by_sequential_fallback_or_error",
        logits_source: "sequential target decode or none on hard error",
        mtp_n_raw_keep: "verified_by_sequential_fallback_or_zero",
        fallback: "sequential safety fallback or MTP verifier failed",
        failure_action: "restore_or_error",
        live_status: "blocked_missing_mtp_model",
    },
];

pub fn suffix_case_by_id(id: &str) -> Option<MtpSuffixVerifierCase> {
    MTP_SUFFIX_ORCHESTRATION_CASES
        .iter()
        .copied()
        .find(|case| case.id == id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suffix_cases_are_ordered_and_complete() {
        let ids = [
            "b300_missing_mtp_support_model",
            "suffix_full_accept",
            "suffix_prefix1_accept",
            "suffix_restore_replay_accept",
            "suffix_exact_replay_debug_accept",
            "suffix_failure_restore_or_error",
        ];
        assert_eq!(MTP_SUFFIX_ORCHESTRATION_CASES.len(), ids.len());
        for (case, expected) in MTP_SUFFIX_ORCHESTRATION_CASES.iter().zip(ids) {
            assert_eq!(case.id, expected);
        }
    }

    #[test]
    fn full_accept_reads_last_spec_logits_row() {
        let case = suffix_case_by_id("suffix_full_accept").expect("full");
        assert_eq!(case.commit_rule, "commit_drafts == draft_n");
        assert_eq!(
            case.readbacks,
            ["row_tops[0..draft_n-2]", "spec_logits[draft_n - 1]"]
        );
        assert_eq!(case.logits_source, "spec logits row draft_n - 1");
        assert_eq!(case.frontier_ops, ["keep_accepted"]);
    }

    #[test]
    fn prefix1_accept_commits_captured_prefix_state() {
        let case = suffix_case_by_id("suffix_prefix1_accept").expect("prefix1");
        assert_eq!(case.capture_prefix1, "draft_n == 2 && capture_prefix1");
        assert_eq!(case.readbacks, ["row_tops[0]", "spec_logits[0]"]);
        assert_eq!(case.frontier_ops, ["commit_prefix1", "keep_accepted"]);
        assert_eq!(case.checkpoint_action, "reset to start then push drafts[0]");
    }

    #[test]
    fn replay_paths_restore_before_accepting_tokens() {
        for id in [
            "suffix_restore_replay_accept",
            "suffix_exact_replay_debug_accept",
        ] {
            let case = suffix_case_by_id(id).expect("replay");
            assert!(case.frontier_ops.starts_with(&["snapshot", "restore"]));
            assert_eq!(case.accepted_suffix, "commit_drafts");
            assert_eq!(case.logits_source, "target replay logits");
        }
    }

    #[test]
    fn failure_case_restores_or_errors_fail_closed() {
        let case = suffix_case_by_id("suffix_failure_restore_or_error").expect("failure");
        assert_eq!(case.frontier_ops, ["restore_or_error"]);
        assert_eq!(case.failure_action, "restore_or_error");
        assert_eq!(
            case.accepted_suffix,
            "verified_by_sequential_fallback_or_error"
        );
    }
}
