//! Model-free MTP exact N=2 verifier orchestration plan.
//!
//! The live exact-N=2 verifier smoke is blocked until the B300 workspace has an
//! MTP support GGUF. This plan pins the Rust-owned orchestration shape against
//! the current C exact verifier before GPU execution is ported.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MtpDecode2VerifierCase {
    pub id: &'static str,
    pub source_function: &'static str,
    pub command_boundary: &'static str,
    pub target_tokens: &'static [&'static str],
    pub start_source: &'static str,
    pub decode_command_steps: &'static [&'static str],
    pub readbacks: &'static [&'static str],
    pub frontier_ops: &'static [&'static str],
    pub accept_condition: &'static str,
    pub accepted_suffix: &'static str,
    pub checkpoint_action: &'static str,
    pub logits_source: &'static str,
    pub mtp_n_raw_keep: &'static str,
    pub failure_action: &'static str,
    pub live_status: &'static str,
}

pub const MTP_DECODE2_TARGET_TOKENS: &[&str] = &["drafts[0]", "drafts[1]"];

pub const MTP_DECODE2_COMMAND_STEPS: &[&str] = &[
    "row_view_batch_cur_hc_0",
    "row_view_batch_cur_hc_1",
    "row_view_batch_next_hc_0",
    "row_view_batch_next_hc_1",
    "embed_token0_hc",
    "embed_token1_hc",
    "save_cur_after_and_capture_flag",
    "enable_spec_capture_prefix1",
    "begin_decode2_commands",
    "for_each_layer_decode_token0_at_start",
    "capture_prefix1_attn_state",
    "capture_prefix1_index_state",
    "for_each_layer_decode_token1_at_start_plus_one",
    "swap_token0_cur_next_hc",
    "swap_token1_cur_next_hc",
    "end_decode2_commands",
    "restore_cur_after_and_capture_flag",
    "output_head_token0",
    "top1_readback_token0",
    "optional_logits0_readback",
    "output_head_token1",
    "logits1_readback",
    "free_decode2_row_views",
];

pub const MTP_DECODE2_READBACKS: &[&str] = &["top0", "optional_logits0", "logits1"];

pub const MTP_DECODE2_ORCHESTRATION_CASES: &[MtpDecode2VerifierCase] = &[
    MtpDecode2VerifierCase {
        id: "b300_missing_mtp_support_model",
        source_function: "none",
        command_boundary: "none",
        target_tokens: &[],
        start_source: "none",
        decode_command_steps: &[],
        readbacks: &[],
        frontier_ops: &[],
        accept_condition: "blocked_missing_mtp_model",
        accepted_suffix: "0",
        checkpoint_action: "no session is created",
        logits_source: "none",
        mtp_n_raw_keep: "0",
        failure_action: "blocked_missing_mtp_model",
        live_status: "blocked_missing_mtp_model",
    },
    MtpDecode2VerifierCase {
        id: "exact_decode2_full_accept",
        source_function: "metal_graph_verify_decode2_exact",
        command_boundary: "mtp_decode2_exact",
        target_tokens: MTP_DECODE2_TARGET_TOKENS,
        start_source: "checkpoint.len",
        decode_command_steps: MTP_DECODE2_COMMAND_STEPS,
        readbacks: MTP_DECODE2_READBACKS,
        frontier_ops: &["snapshot", "keep_accepted"],
        accept_condition: "row0_top == drafts[1]",
        accepted_suffix: "2",
        checkpoint_action: "push drafts[0] and drafts[1]",
        logits_source: "decode2 logits1",
        mtp_n_raw_keep: "2",
        failure_action: "none",
        live_status: "blocked_missing_mtp_model",
    },
    MtpDecode2VerifierCase {
        id: "exact_decode2_prefix1_accept",
        source_function: "metal_graph_verify_decode2_exact",
        command_boundary: "mtp_decode2_exact",
        target_tokens: MTP_DECODE2_TARGET_TOKENS,
        start_source: "checkpoint.len",
        decode_command_steps: MTP_DECODE2_COMMAND_STEPS,
        readbacks: MTP_DECODE2_READBACKS,
        frontier_ops: &["snapshot", "commit_prefix1", "keep_accepted"],
        accept_condition: "row0_top != drafts[1] && decode2 ok",
        accepted_suffix: "1",
        checkpoint_action: "reset to start then push drafts[0]",
        logits_source: "decode2 logits0",
        mtp_n_raw_keep: "1",
        failure_action: "none",
        live_status: "blocked_missing_mtp_model",
    },
    MtpDecode2VerifierCase {
        id: "exact_decode2_failure_restore_then_sequential",
        source_function: "metal_graph_verify_decode2_exact",
        command_boundary: "mtp_decode2_exact",
        target_tokens: MTP_DECODE2_TARGET_TOKENS,
        start_source: "checkpoint.len",
        decode_command_steps: MTP_DECODE2_COMMAND_STEPS,
        readbacks: MTP_DECODE2_READBACKS,
        frontier_ops: &["snapshot", "restore"],
        accept_condition: "decode2 failed or prefix1 commit failed",
        accepted_suffix: "verified_by_sequential_fallback",
        checkpoint_action: "reset to start before sequential fallback",
        logits_source: "sequential target decode",
        mtp_n_raw_keep: "verified_by_sequential_fallback",
        failure_action: "restore_pre_verifier_frontier",
        live_status: "blocked_missing_mtp_model",
    },
];

pub fn decode2_case_by_id(id: &str) -> Option<MtpDecode2VerifierCase> {
    MTP_DECODE2_ORCHESTRATION_CASES
        .iter()
        .copied()
        .find(|case| case.id == id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode2_cases_are_ordered_and_complete() {
        let ids = [
            "b300_missing_mtp_support_model",
            "exact_decode2_full_accept",
            "exact_decode2_prefix1_accept",
            "exact_decode2_failure_restore_then_sequential",
        ];
        assert_eq!(MTP_DECODE2_ORCHESTRATION_CASES.len(), ids.len());
        for (case, expected) in MTP_DECODE2_ORCHESTRATION_CASES.iter().zip(ids) {
            assert_eq!(case.id, expected);
        }
    }

    #[test]
    fn exact_decode2_preserves_target_token_order() {
        for id in ["exact_decode2_full_accept", "exact_decode2_prefix1_accept"] {
            let case = decode2_case_by_id(id).expect("decode2 case");
            assert_eq!(case.target_tokens, ["drafts[0]", "drafts[1]"]);
            assert!(case
                .decode_command_steps
                .windows(2)
                .any(|pair| pair == ["embed_token0_hc", "embed_token1_hc"]));
            assert!(case.decode_command_steps.windows(3).any(|triple| triple
                == [
                    "for_each_layer_decode_token0_at_start",
                    "capture_prefix1_attn_state",
                    "capture_prefix1_index_state",
                ]));
            assert!(case
                .decode_command_steps
                .contains(&"for_each_layer_decode_token1_at_start_plus_one"));
        }
    }

    #[test]
    fn full_and_prefix_accept_use_distinct_logits_rows() {
        let full = decode2_case_by_id("exact_decode2_full_accept").expect("full");
        let prefix = decode2_case_by_id("exact_decode2_prefix1_accept").expect("prefix");
        assert_eq!(full.accept_condition, "row0_top == drafts[1]");
        assert_eq!(full.logits_source, "decode2 logits1");
        assert_eq!(full.accepted_suffix, "2");
        assert_eq!(
            prefix.accept_condition,
            "row0_top != drafts[1] && decode2 ok"
        );
        assert_eq!(prefix.logits_source, "decode2 logits0");
        assert_eq!(
            prefix.frontier_ops,
            ["snapshot", "commit_prefix1", "keep_accepted"]
        );
    }

    #[test]
    fn failure_case_restores_pre_verifier_frontier() {
        let case =
            decode2_case_by_id("exact_decode2_failure_restore_then_sequential").expect("failure");
        assert_eq!(case.frontier_ops, ["snapshot", "restore"]);
        assert_eq!(case.failure_action, "restore_pre_verifier_frontier");
        assert_eq!(case.logits_source, "sequential target decode");
        assert_eq!(case.mtp_n_raw_keep, "verified_by_sequential_fallback");
    }
}
