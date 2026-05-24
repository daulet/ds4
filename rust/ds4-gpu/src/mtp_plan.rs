//! Model-free MTP speculative decode decision planner.
//!
//! This module mirrors the M10.8a current-C state-machine contract without
//! executing GPU kernels. Later M10.8 stages replace the static scenario rows
//! with runtime graph calls while keeping these decision names stable.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MtpCount {
    Exact(u8),
    DraftN,
    CommitDrafts,
    Verified,
    VerifiedBySequentialFallback,
    VerifiedBySequentialFallbackOrError,
    VerifiedBySequentialFallbackOrZero,
}

impl MtpCount {
    pub const fn contract_value(self) -> &'static str {
        match self {
            Self::Exact(0) => "0",
            Self::Exact(1) => "1",
            Self::Exact(2) => "2",
            Self::Exact(_) => "unsupported",
            Self::DraftN => "draft_n",
            Self::CommitDrafts => "commit_drafts",
            Self::Verified => "verified",
            Self::VerifiedBySequentialFallback => "verified_by_sequential_fallback",
            Self::VerifiedBySequentialFallbackOrError => "verified_by_sequential_fallback_or_error",
            Self::VerifiedBySequentialFallbackOrZero => "verified_by_sequential_fallback_or_zero",
        }
    }

    pub const fn exact(self) -> Option<u8> {
        match self {
            Self::Exact(value) => Some(value),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MtpDecisionPlan {
    pub id: &'static str,
    pub path: &'static str,
    pub frontier_ops: &'static [&'static str],
    pub accepted_suffix: MtpCount,
    pub checkpoint_action: &'static str,
    pub logits_source: &'static str,
    pub mtp_n_raw_keep: MtpCount,
    pub fallback: &'static str,
    pub fail_closed: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MtpScenario {
    B300MissingSupportModel,
    DisabledAfterFirstToken,
    FirstDraftMiss,
    MarginSkipSingleTargetReplay,
    ExactDecode2FullAccept,
    ExactDecode2Prefix1Accept,
    ExactDecode2FailureRestoreThenSequential,
    SuffixFullAccept,
    SuffixPrefix1Accept,
    SuffixRestoreReplayAccept,
    SuffixFailureRestoreOrError,
    SequentialSafetyFallback,
}

pub const MTP_SCENARIOS: &[MtpScenario] = &[
    MtpScenario::B300MissingSupportModel,
    MtpScenario::DisabledAfterFirstToken,
    MtpScenario::FirstDraftMiss,
    MtpScenario::MarginSkipSingleTargetReplay,
    MtpScenario::ExactDecode2FullAccept,
    MtpScenario::ExactDecode2Prefix1Accept,
    MtpScenario::ExactDecode2FailureRestoreThenSequential,
    MtpScenario::SuffixFullAccept,
    MtpScenario::SuffixPrefix1Accept,
    MtpScenario::SuffixRestoreReplayAccept,
    MtpScenario::SuffixFailureRestoreOrError,
    MtpScenario::SequentialSafetyFallback,
];

pub const fn plan_scenario(scenario: MtpScenario) -> MtpDecisionPlan {
    match scenario {
        MtpScenario::B300MissingSupportModel => MtpDecisionPlan {
            id: "b300_missing_mtp_support_model",
            path: "availability_blocker",
            frontier_ops: &[],
            accepted_suffix: MtpCount::Exact(0),
            checkpoint_action: "no session is created",
            logits_source: "none",
            mtp_n_raw_keep: MtpCount::Exact(0),
            fallback: "blocked_missing_mtp_model",
            fail_closed: true,
        },
        MtpScenario::DisabledAfterFirstToken => MtpDecisionPlan {
            id: "mtp_disabled_after_first_token",
            path: "guard",
            frontier_ops: &[],
            accepted_suffix: MtpCount::Exact(0),
            checkpoint_action: "first target token only",
            logits_source: "target first-token logits",
            mtp_n_raw_keep: MtpCount::Exact(0),
            fallback: "return first-token accept",
            fail_closed: true,
        },
        MtpScenario::FirstDraftMiss => MtpDecisionPlan {
            id: "first_draft_miss",
            path: "draft_miss",
            frontier_ops: &[],
            accepted_suffix: MtpCount::Exact(0),
            checkpoint_action: "first target token only",
            logits_source: "target first-token logits",
            mtp_n_raw_keep: MtpCount::Exact(0),
            fallback: "skip speculative work",
            fail_closed: true,
        },
        MtpScenario::MarginSkipSingleTargetReplay => MtpDecisionPlan {
            id: "margin_skip_single_target_replay",
            path: "margin_skip",
            frontier_ops: &["keep_accepted"],
            accepted_suffix: MtpCount::Exact(1),
            checkpoint_action: "push drafts[0] after one exact target decode",
            logits_source: "target decode logits for drafts[0]",
            mtp_n_raw_keep: MtpCount::Exact(1),
            fallback: "margin-skip",
            fail_closed: true,
        },
        MtpScenario::ExactDecode2FullAccept => MtpDecisionPlan {
            id: "exact_decode2_full_accept",
            path: "exact_decode2",
            frontier_ops: &["snapshot", "keep_accepted"],
            accepted_suffix: MtpCount::Exact(2),
            checkpoint_action: "push drafts[0] and drafts[1]",
            logits_source: "decode2 logits1",
            mtp_n_raw_keep: MtpCount::Exact(2),
            fallback: "none",
            fail_closed: false,
        },
        MtpScenario::ExactDecode2Prefix1Accept => MtpDecisionPlan {
            id: "exact_decode2_prefix1_accept",
            path: "exact_decode2",
            frontier_ops: &["snapshot", "commit_prefix1", "keep_accepted"],
            accepted_suffix: MtpCount::Exact(1),
            checkpoint_action: "reset to start then push drafts[0]",
            logits_source: "decode2 logits0",
            mtp_n_raw_keep: MtpCount::Exact(1),
            fallback: "none",
            fail_closed: false,
        },
        MtpScenario::ExactDecode2FailureRestoreThenSequential => MtpDecisionPlan {
            id: "exact_decode2_failure_restore_then_sequential",
            path: "exact_decode2_failure",
            frontier_ops: &["snapshot", "restore"],
            accepted_suffix: MtpCount::VerifiedBySequentialFallback,
            checkpoint_action: "reset to start before sequential fallback",
            logits_source: "sequential target decode",
            mtp_n_raw_keep: MtpCount::VerifiedBySequentialFallback,
            fallback: "sequential safety fallback",
            fail_closed: true,
        },
        MtpScenario::SuffixFullAccept => MtpDecisionPlan {
            id: "suffix_full_accept",
            path: "suffix_verifier",
            frontier_ops: &["keep_accepted"],
            accepted_suffix: MtpCount::DraftN,
            checkpoint_action: "checkpoint already contains all draft tokens",
            logits_source: "spec logits row draft_n - 1",
            mtp_n_raw_keep: MtpCount::DraftN,
            fallback: "none",
            fail_closed: false,
        },
        MtpScenario::SuffixPrefix1Accept => MtpDecisionPlan {
            id: "suffix_prefix1_accept",
            path: "suffix_verifier",
            frontier_ops: &["commit_prefix1", "keep_accepted"],
            accepted_suffix: MtpCount::Exact(1),
            checkpoint_action: "reset to start then push drafts[0]",
            logits_source: "spec logits row 0",
            mtp_n_raw_keep: MtpCount::Exact(1),
            fallback: "none",
            fail_closed: false,
        },
        MtpScenario::SuffixRestoreReplayAccept => MtpDecisionPlan {
            id: "suffix_restore_replay_accept",
            path: "suffix_verifier_replay",
            frontier_ops: &["snapshot", "restore", "keep_accepted"],
            accepted_suffix: MtpCount::CommitDrafts,
            checkpoint_action: "restore to start then replay accepted drafts",
            logits_source: "target replay logits",
            mtp_n_raw_keep: MtpCount::CommitDrafts,
            fallback: "target replay",
            fail_closed: true,
        },
        MtpScenario::SuffixFailureRestoreOrError => MtpDecisionPlan {
            id: "suffix_failure_restore_or_error",
            path: "suffix_verifier_failure",
            frontier_ops: &["restore_or_error"],
            accepted_suffix: MtpCount::VerifiedBySequentialFallbackOrError,
            checkpoint_action:
                "reset to start; restore if snapshot exists; otherwise error if verifier mutated",
            logits_source: "sequential target decode or none on hard error",
            mtp_n_raw_keep: MtpCount::VerifiedBySequentialFallbackOrZero,
            fallback: "sequential safety fallback or MTP verifier failed",
            fail_closed: true,
        },
        MtpScenario::SequentialSafetyFallback => MtpDecisionPlan {
            id: "sequential_safety_fallback",
            path: "sequential_fallback",
            frontier_ops: &["keep_accepted"],
            accepted_suffix: MtpCount::Verified,
            checkpoint_action: "push each verified draft token in order",
            logits_source: "normal target decode logits",
            mtp_n_raw_keep: MtpCount::Verified,
            fallback: "target sequential verifier",
            fail_closed: true,
        },
    }
}

pub fn plan_by_id(id: &str) -> Option<MtpDecisionPlan> {
    MTP_SCENARIOS
        .iter()
        .copied()
        .map(plan_scenario)
        .find(|plan| plan.id == id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planner_covers_contract_cases() {
        let ids: [&str; 12] = [
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
        assert_eq!(MTP_SCENARIOS.len(), ids.len());
        for (scenario, expected) in MTP_SCENARIOS.iter().zip(ids) {
            assert_eq!(plan_scenario(*scenario).id, expected);
        }
    }

    #[test]
    fn missing_support_and_guards_fail_closed() {
        for id in [
            "b300_missing_mtp_support_model",
            "mtp_disabled_after_first_token",
            "first_draft_miss",
        ] {
            let plan = plan_by_id(id).expect("known plan");
            assert!(plan.fail_closed);
            assert_eq!(plan.accepted_suffix.exact(), Some(0));
            assert!(plan.frontier_ops.is_empty());
        }
    }

    #[test]
    fn exact_decode2_paths_keep_row_contracts() {
        let full = plan_by_id("exact_decode2_full_accept").expect("full");
        assert_eq!(full.accepted_suffix.exact(), Some(2));
        assert_eq!(full.logits_source, "decode2 logits1");
        assert_eq!(full.frontier_ops, ["snapshot", "keep_accepted"]);

        let prefix = plan_by_id("exact_decode2_prefix1_accept").expect("prefix");
        assert_eq!(prefix.accepted_suffix.exact(), Some(1));
        assert_eq!(prefix.logits_source, "decode2 logits0");
        assert!(prefix.frontier_ops.contains(&"commit_prefix1"));
    }

    #[test]
    fn suffix_paths_record_commit_restore_edges() {
        let prefix = plan_by_id("suffix_prefix1_accept").expect("prefix");
        assert_eq!(prefix.logits_source, "spec logits row 0");
        assert!(prefix.frontier_ops.contains(&"commit_prefix1"));

        let replay = plan_by_id("suffix_restore_replay_accept").expect("replay");
        assert_eq!(replay.accepted_suffix, MtpCount::CommitDrafts);
        assert!(replay.frontier_ops.contains(&"restore"));
        assert_eq!(replay.logits_source, "target replay logits");
    }

    #[test]
    fn unknown_case_is_rejected() {
        assert!(plan_by_id("not_a_real_mtp_case").is_none());
    }
}
