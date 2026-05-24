//! Model-free MTP speculative frontier mutation plan.
//!
//! The live frontier mutation smoke is blocked until the B300 workspace has an
//! MTP support GGUF. This plan pins the Rust-owned snapshot, restore, and
//! prefix-1 commit semantics against the current C frontier mutators before GPU
//! execution is ported.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MtpFrontierMutationCase {
    pub id: &'static str,
    pub source_function: &'static str,
    pub ratio_family: &'static str,
    pub saved_counters: &'static [&'static str],
    pub counter_updates: &'static [&'static str],
    pub tensor_copies: &'static [&'static str],
    pub mtp_n_raw_action: &'static str,
    pub invisible_rows_policy: &'static str,
    pub failure_action: &'static str,
    pub live_status: &'static str,
}

pub const FRONTIER_COUNTERS: &[&str] = &["n_comp", "n_index_comp", "mtp_n_raw"];

pub const MTP_FRONTIER_MUTATION_CASES: &[MtpFrontierMutationCase] = &[
    MtpFrontierMutationCase {
        id: "b300_missing_mtp_support_model",
        source_function: "none",
        ratio_family: "none",
        saved_counters: &[],
        counter_updates: &[],
        tensor_copies: &[],
        mtp_n_raw_action: "none",
        invisible_rows_policy: "blocked_missing_mtp_model",
        failure_action: "blocked_missing_mtp_model",
        live_status: "blocked_missing_mtp_model",
    },
    MtpFrontierMutationCase {
        id: "snapshot_dense_layer_counters_only",
        source_function: "spec_frontier_snapshot",
        ratio_family: "ratio0",
        saved_counters: FRONTIER_COUNTERS,
        counter_updates: &[
            "f.n_comp = g.layer_n_comp",
            "f.n_index_comp = g.layer_n_index_comp",
            "f.mtp_n_raw = g.mtp_n_raw",
        ],
        tensor_copies: &[],
        mtp_n_raw_action: "save",
        invisible_rows_policy: "none",
        failure_action: "spec_frontier_free_on_failure",
        live_status: "blocked_missing_mtp_model",
    },
    MtpFrontierMutationCase {
        id: "snapshot_compressed_attn_frontier",
        source_function: "spec_frontier_snapshot",
        ratio_family: "ratio4_or_ratio128",
        saved_counters: FRONTIER_COUNTERS,
        counter_updates: &[
            "f.n_comp = g.layer_n_comp",
            "f.n_index_comp = g.layer_n_index_comp",
            "f.mtp_n_raw = g.mtp_n_raw",
        ],
        tensor_copies: &[
            "spec_attn_state_kv <- layer_attn_state_kv",
            "spec_attn_state_score <- layer_attn_state_score",
        ],
        mtp_n_raw_action: "save",
        invisible_rows_policy: "none",
        failure_action: "spec_frontier_free_on_failure",
        live_status: "blocked_missing_mtp_model",
    },
    MtpFrontierMutationCase {
        id: "snapshot_ratio4_index_frontier",
        source_function: "spec_frontier_snapshot",
        ratio_family: "ratio4",
        saved_counters: FRONTIER_COUNTERS,
        counter_updates: &[
            "f.n_comp = g.layer_n_comp",
            "f.n_index_comp = g.layer_n_index_comp",
            "f.mtp_n_raw = g.mtp_n_raw",
        ],
        tensor_copies: &[
            "spec_index_state_kv <- layer_index_state_kv",
            "spec_index_state_score <- layer_index_state_score",
        ],
        mtp_n_raw_action: "save",
        invisible_rows_policy: "none",
        failure_action: "spec_frontier_free_on_failure",
        live_status: "blocked_missing_mtp_model",
    },
    MtpFrontierMutationCase {
        id: "restore_compressed_attn_frontier",
        source_function: "spec_frontier_restore",
        ratio_family: "ratio4_or_ratio128",
        saved_counters: FRONTIER_COUNTERS,
        counter_updates: &[
            "g.layer_n_comp = f.n_comp",
            "g.layer_n_index_comp = f.n_index_comp",
            "g.mtp_n_raw = f.mtp_n_raw",
        ],
        tensor_copies: &[
            "layer_attn_state_kv <- spec_attn_state_kv",
            "layer_attn_state_score <- spec_attn_state_score",
        ],
        mtp_n_raw_action: "restore",
        invisible_rows_policy: "append_only_rows_may_remain_beyond_restored_counters",
        failure_action: "return_false_after_synchronize",
        live_status: "blocked_missing_mtp_model",
    },
    MtpFrontierMutationCase {
        id: "restore_ratio4_index_frontier",
        source_function: "spec_frontier_restore",
        ratio_family: "ratio4",
        saved_counters: FRONTIER_COUNTERS,
        counter_updates: &[
            "g.layer_n_comp = f.n_comp",
            "g.layer_n_index_comp = f.n_index_comp",
            "g.mtp_n_raw = f.mtp_n_raw",
        ],
        tensor_copies: &[
            "layer_index_state_kv <- spec_index_state_kv",
            "layer_index_state_score <- spec_index_state_score",
        ],
        mtp_n_raw_action: "restore",
        invisible_rows_policy: "append_only_rows_may_remain_beyond_restored_counters",
        failure_action: "return_false_after_synchronize",
        live_status: "blocked_missing_mtp_model",
    },
    MtpFrontierMutationCase {
        id: "prefix1_commit_compressed_attn_frontier",
        source_function: "spec_frontier_commit_prefix1",
        ratio_family: "ratio4_or_ratio128",
        saved_counters: &["spec_prefix1_n_comp"],
        counter_updates: &["g.layer_n_comp = g.spec_prefix1_n_comp"],
        tensor_copies: &[
            "layer_attn_state_kv <- spec_prefix1_attn_state_kv",
            "layer_attn_state_score <- spec_prefix1_attn_state_score",
        ],
        mtp_n_raw_action: "unchanged",
        invisible_rows_policy: "second speculative row may remain invisible",
        failure_action: "return_false_after_synchronize",
        live_status: "blocked_missing_mtp_model",
    },
    MtpFrontierMutationCase {
        id: "prefix1_commit_ratio4_index_frontier",
        source_function: "spec_frontier_commit_prefix1",
        ratio_family: "ratio4",
        saved_counters: &["spec_prefix1_n_index_comp"],
        counter_updates: &["g.layer_n_index_comp = g.spec_prefix1_n_index_comp"],
        tensor_copies: &[
            "layer_index_state_kv <- spec_prefix1_index_state_kv",
            "layer_index_state_score <- spec_prefix1_index_state_score",
        ],
        mtp_n_raw_action: "unchanged",
        invisible_rows_policy: "second speculative row may remain invisible",
        failure_action: "return_false_after_synchronize",
        live_status: "blocked_missing_mtp_model",
    },
];

pub fn frontier_case_by_id(id: &str) -> Option<MtpFrontierMutationCase> {
    MTP_FRONTIER_MUTATION_CASES
        .iter()
        .copied()
        .find(|case| case.id == id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frontier_cases_are_ordered_and_complete() {
        let ids = [
            "b300_missing_mtp_support_model",
            "snapshot_dense_layer_counters_only",
            "snapshot_compressed_attn_frontier",
            "snapshot_ratio4_index_frontier",
            "restore_compressed_attn_frontier",
            "restore_ratio4_index_frontier",
            "prefix1_commit_compressed_attn_frontier",
            "prefix1_commit_ratio4_index_frontier",
        ];
        assert_eq!(MTP_FRONTIER_MUTATION_CASES.len(), ids.len());
        for (case, expected) in MTP_FRONTIER_MUTATION_CASES.iter().zip(ids) {
            assert_eq!(case.id, expected);
        }
    }

    #[test]
    fn snapshot_records_counters_and_raw_frontier() {
        for id in [
            "snapshot_dense_layer_counters_only",
            "snapshot_compressed_attn_frontier",
            "snapshot_ratio4_index_frontier",
        ] {
            let case = frontier_case_by_id(id).expect("snapshot");
            assert_eq!(case.source_function, "spec_frontier_snapshot");
            assert_eq!(case.saved_counters, FRONTIER_COUNTERS);
            assert_eq!(case.mtp_n_raw_action, "save");
            assert!(case.counter_updates.contains(&"f.mtp_n_raw = g.mtp_n_raw"));
        }
    }

    #[test]
    fn restore_rewinds_counters_and_raw_frontier() {
        for id in [
            "restore_compressed_attn_frontier",
            "restore_ratio4_index_frontier",
        ] {
            let case = frontier_case_by_id(id).expect("restore");
            assert_eq!(case.source_function, "spec_frontier_restore");
            assert_eq!(case.saved_counters, FRONTIER_COUNTERS);
            assert_eq!(case.mtp_n_raw_action, "restore");
            assert!(case.counter_updates.contains(&"g.mtp_n_raw = f.mtp_n_raw"));
            assert_eq!(
                case.invisible_rows_policy,
                "append_only_rows_may_remain_beyond_restored_counters"
            );
        }
    }

    #[test]
    fn prefix1_commit_keeps_speculative_rows_invisible() {
        for id in [
            "prefix1_commit_compressed_attn_frontier",
            "prefix1_commit_ratio4_index_frontier",
        ] {
            let case = frontier_case_by_id(id).expect("prefix1");
            assert_eq!(case.source_function, "spec_frontier_commit_prefix1");
            assert_eq!(case.mtp_n_raw_action, "unchanged");
            assert_eq!(
                case.invisible_rows_policy,
                "second speculative row may remain invisible"
            );
            assert!(case
                .counter_updates
                .iter()
                .any(|update| update.starts_with("g.layer_n_")));
        }
    }
}
