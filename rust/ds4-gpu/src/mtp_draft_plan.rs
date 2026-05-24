//! Model-free MTP draft graph orchestration plan.
//!
//! The live MTP draft smoke is blocked until the B300 workspace has an MTP
//! support GGUF. This plan still pins the Rust-owned orchestration boundary
//! against the current C draft functions before GPU execution is ported.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MtpDraftOrchestrationCase {
    pub id: &'static str,
    pub source_function: &'static str,
    pub command_boundary: &'static str,
    pub prev_hc: &'static str,
    pub out_hc: &'static str,
    pub token_source: &'static str,
    pub pos_source: &'static str,
    pub logits_role: &'static str,
    pub top_id_role: &'static str,
    pub command_steps: &'static [&'static str],
    pub readbacks: &'static [&'static str],
    pub mtp_n_raw_transition: &'static str,
    pub saved_state: &'static [&'static str],
    pub failure_action: &'static str,
    pub live_status: &'static str,
}

pub const MTP_DRAFT_COMMAND_STEPS: &[&str] = &[
    "begin_commands",
    "embed_token_hc",
    "rms_norm_embed",
    "matmul_e_proj",
    "repeat_e_proj_hc",
    "rms_norm_prev_hc",
    "matmul_h_proj",
    "add_mtp_input_hc",
    "encode_decode_layer_mtp_block",
    "set_cur_hc_to_out_hc",
    "encode_output_head_mtp",
    "optional_top1_indexer",
    "end_commands",
];

pub const MTP_DRAFT_READBACKS: &[&str] = &["optional_logits", "optional_top_id"];

pub const MTP_DRAFT_SAVED_STATE: &[&str] = &["cur_hc", "after_ffn_hc"];

pub const MTP_DRAFT_ORCHESTRATION_CASES: &[MtpDraftOrchestrationCase] = &[
    MtpDraftOrchestrationCase {
        id: "b300_missing_mtp_support_model",
        source_function: "none",
        command_boundary: "none",
        prev_hc: "none",
        out_hc: "none",
        token_source: "none",
        pos_source: "none",
        logits_role: "none",
        top_id_role: "none",
        command_steps: &[],
        readbacks: &[],
        mtp_n_raw_transition: "none",
        saved_state: &[],
        failure_action: "blocked_missing_mtp_model",
        live_status: "blocked_missing_mtp_model",
    },
    MtpDraftOrchestrationCase {
        id: "first_draft_from_current_hc",
        source_function: "metal_graph_eval_mtp_draft",
        command_boundary: "mtp_draft",
        prev_hc: "cur_hc",
        out_hc: "mtp_state_hc",
        token_source: "accepted target token",
        pos_source: "checkpoint.len - 1",
        logits_role: "optional_full_logits",
        top_id_role: "required_for_draft_token",
        command_steps: MTP_DRAFT_COMMAND_STEPS,
        readbacks: MTP_DRAFT_READBACKS,
        mtp_n_raw_transition: "increment_if_less_than_raw_window",
        saved_state: MTP_DRAFT_SAVED_STATE,
        failure_action: "leave_mtp_draft_valid_false_and_keep_target_decode",
        live_status: "blocked_missing_mtp_model",
    },
    MtpDraftOrchestrationCase {
        id: "recursive_draft_state_to_next",
        source_function: "metal_graph_eval_mtp_draft_from_hc",
        command_boundary: "mtp_draft",
        prev_hc: "mtp_state_hc",
        out_hc: "mtp_next_hc",
        token_source: "previous draft token",
        pos_source: "checkpoint.len + draft_n - 1",
        logits_role: "optional_need_logits",
        top_id_role: "required_for_next_draft",
        command_steps: MTP_DRAFT_COMMAND_STEPS,
        readbacks: MTP_DRAFT_READBACKS,
        mtp_n_raw_transition: "increment_if_less_than_raw_window",
        saved_state: MTP_DRAFT_SAVED_STATE,
        failure_action: "return_current_accepted_prefix",
        live_status: "blocked_missing_mtp_model",
    },
    MtpDraftOrchestrationCase {
        id: "recursive_draft_next_to_state",
        source_function: "metal_graph_eval_mtp_draft_from_hc",
        command_boundary: "mtp_draft",
        prev_hc: "mtp_next_hc",
        out_hc: "mtp_state_hc",
        token_source: "previous draft token",
        pos_source: "checkpoint.len + draft_n - 1",
        logits_role: "optional_need_logits",
        top_id_role: "required_for_next_draft",
        command_steps: MTP_DRAFT_COMMAND_STEPS,
        readbacks: MTP_DRAFT_READBACKS,
        mtp_n_raw_transition: "increment_if_less_than_raw_window",
        saved_state: MTP_DRAFT_SAVED_STATE,
        failure_action: "return_current_accepted_prefix",
        live_status: "blocked_missing_mtp_model",
    },
    MtpDraftOrchestrationCase {
        id: "draft_failure_restores_saved_graph_state",
        source_function: "metal_graph_eval_mtp_draft_from_hc",
        command_boundary: "mtp_draft",
        prev_hc: "any_prev_hc",
        out_hc: "any_out_hc",
        token_source: "draft input token",
        pos_source: "draft input position",
        logits_role: "not_committed_on_failure",
        top_id_role: "not_committed_on_failure",
        command_steps: &["synchronize_after_failure"],
        readbacks: &[],
        mtp_n_raw_transition: "no_increment_on_failure",
        saved_state: MTP_DRAFT_SAVED_STATE,
        failure_action: "restore_cur_hc_and_after_ffn_hc",
        live_status: "blocked_missing_mtp_model",
    },
];

pub fn draft_case_by_id(id: &str) -> Option<MtpDraftOrchestrationCase> {
    MTP_DRAFT_ORCHESTRATION_CASES
        .iter()
        .copied()
        .find(|case| case.id == id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draft_cases_are_ordered_and_complete() {
        let ids = [
            "b300_missing_mtp_support_model",
            "first_draft_from_current_hc",
            "recursive_draft_state_to_next",
            "recursive_draft_next_to_state",
            "draft_failure_restores_saved_graph_state",
        ];
        assert_eq!(MTP_DRAFT_ORCHESTRATION_CASES.len(), ids.len());
        for (case, expected) in MTP_DRAFT_ORCHESTRATION_CASES.iter().zip(ids) {
            assert_eq!(case.id, expected);
        }
    }

    #[test]
    fn first_draft_uses_current_hc_wrapper_shape() {
        let case = draft_case_by_id("first_draft_from_current_hc").expect("first draft");
        assert_eq!(case.source_function, "metal_graph_eval_mtp_draft");
        assert_eq!(case.prev_hc, "cur_hc");
        assert_eq!(case.out_hc, "mtp_state_hc");
        assert!(case.command_steps.contains(&"encode_output_head_mtp"));
        assert_eq!(
            case.mtp_n_raw_transition,
            "increment_if_less_than_raw_window"
        );
    }

    #[test]
    fn recursive_drafts_alternate_hc_buffers() {
        let state_to_next = draft_case_by_id("recursive_draft_state_to_next").expect("state");
        let next_to_state = draft_case_by_id("recursive_draft_next_to_state").expect("next");
        assert_eq!(state_to_next.prev_hc, "mtp_state_hc");
        assert_eq!(state_to_next.out_hc, "mtp_next_hc");
        assert_eq!(next_to_state.prev_hc, "mtp_next_hc");
        assert_eq!(next_to_state.out_hc, "mtp_state_hc");
        assert_eq!(state_to_next.command_steps, next_to_state.command_steps);
    }

    #[test]
    fn failure_case_restores_saved_graph_pointers() {
        let case = draft_case_by_id("draft_failure_restores_saved_graph_state").expect("failure");
        assert_eq!(case.command_steps, ["synchronize_after_failure"]);
        assert_eq!(case.mtp_n_raw_transition, "no_increment_on_failure");
        assert_eq!(case.saved_state, ["cur_hc", "after_ffn_hc"]);
        assert_eq!(case.failure_action, "restore_cur_hc_and_after_ffn_hc");
    }
}
