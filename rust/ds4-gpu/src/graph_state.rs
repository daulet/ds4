//! No-execute graph tensor state plan for DS4 decode.
//!
//! This module records the allocation shape that M10.5c4 will need before it
//! starts issuing decode kernels: owned tensors, views, lazy optional tensors,
//! and initialization obligations for persistent caches.

use crate::graph_plan::{
    tensor_spec, ElementType, GraphPlan, GraphPlanError, ModelGraphDims, TensorByteLen,
    TensorOwner, N_HC, N_LAYER,
};

const BYTES_F32: u64 = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphTensorStorage {
    Owned,
    LazyOwned,
    View {
        base: &'static str,
        offset_bytes: u64,
    },
    External,
}

impl GraphTensorStorage {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Owned => "owned",
            Self::LazyOwned => "lazy_owned",
            Self::View { .. } => "view",
            Self::External => "external",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphTensorInstances {
    Single,
    PerLayer,
}

impl GraphTensorInstances {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Single => "single",
            Self::PerLayer => "per_layer",
        }
    }

    pub const fn count(self) -> usize {
        match self {
            Self::Single => 1,
            Self::PerLayer => N_LAYER,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphTensorInitialFill {
    Unspecified,
    ZeroFullCapacity,
    ZeroState,
    NegativeInfinityState,
    ExternalInput,
}

impl GraphTensorInitialFill {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Unspecified => "unspecified",
            Self::ZeroFullCapacity => "zero_full_capacity",
            Self::ZeroState => "zero_state",
            Self::NegativeInfinityState => "negative_infinity_state",
            Self::ExternalInput => "external_input",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GraphStateFieldPlan {
    pub name: &'static str,
    pub owner: TensorOwner,
    pub element_type: ElementType,
    pub instances: GraphTensorInstances,
    pub storage: GraphTensorStorage,
    pub initial_fill: GraphTensorInitialFill,
}

impl GraphStateFieldPlan {
    pub fn byte_len(
        self,
        plan: GraphPlan,
        dims: ModelGraphDims,
        layer: Option<usize>,
    ) -> Result<TensorByteLen, GraphPlanError<'static>> {
        let spec = *tensor_spec(self.name).ok_or(GraphPlanError::MissingTensorField(self.name))?;
        spec.byte_len(plan, dims, layer)
    }

    pub fn initial_allocation(
        self,
        plan: GraphPlan,
        dims: ModelGraphDims,
        layer: Option<usize>,
    ) -> Result<GraphStateAllocation, GraphPlanError<'static>> {
        let byte_len = self.byte_len(plan, dims, layer)?;
        let allocated = matches!(self.storage, GraphTensorStorage::Owned)
            && matches!(byte_len, TensorByteLen::Known(bytes) if bytes != 0);
        Ok(GraphStateAllocation {
            field: self.name,
            owner: self.owner,
            storage: self.storage,
            initial_fill: self.initial_fill,
            layer,
            byte_len,
            initially_allocated: allocated,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GraphStateAllocation {
    pub field: &'static str,
    pub owner: TensorOwner,
    pub storage: GraphTensorStorage,
    pub initial_fill: GraphTensorInitialFill,
    pub layer: Option<usize>,
    pub byte_len: TensorByteLen,
    pub initially_allocated: bool,
}

macro_rules! state_field {
    ($name:literal, $owner:ident, $ty:ident, $instances:ident, $storage:expr, $fill:ident) => {
        GraphStateFieldPlan {
            name: $name,
            owner: TensorOwner::$owner,
            element_type: ElementType::$ty,
            instances: GraphTensorInstances::$instances,
            storage: $storage,
            initial_fill: GraphTensorInitialFill::$fill,
        }
    };
}

pub const DECODE_GRAPH_STATE_FIELDS: &[GraphStateFieldPlan] = &[
    state_field!(
        "cur_hc",
        GraphDecodeState,
        F32,
        Single,
        GraphTensorStorage::Owned,
        Unspecified
    ),
    state_field!(
        "flat_hc",
        GraphDecodeState,
        F32,
        Single,
        GraphTensorStorage::Owned,
        Unspecified
    ),
    state_field!(
        "hc_mix",
        GraphDecodeState,
        F32,
        Single,
        GraphTensorStorage::Owned,
        Unspecified
    ),
    state_field!(
        "hc_split",
        GraphDecodeState,
        F32,
        Single,
        GraphTensorStorage::Owned,
        Unspecified
    ),
    state_field!(
        "hc_pre",
        GraphDecodeState,
        F32,
        Single,
        GraphTensorStorage::View {
            base: "hc_split",
            offset_bytes: 0,
        },
        Unspecified
    ),
    state_field!(
        "hc_post",
        GraphDecodeState,
        F32,
        Single,
        GraphTensorStorage::View {
            base: "hc_split",
            offset_bytes: (N_HC as u64) * BYTES_F32,
        },
        Unspecified
    ),
    state_field!(
        "hc_comb",
        GraphDecodeState,
        F32,
        Single,
        GraphTensorStorage::View {
            base: "hc_split",
            offset_bytes: 2 * (N_HC as u64) * BYTES_F32,
        },
        Unspecified
    ),
    state_field!(
        "attn_cur",
        GraphDecodeState,
        F32,
        Single,
        GraphTensorStorage::Owned,
        Unspecified
    ),
    state_field!(
        "attn_norm",
        GraphDecodeState,
        F32,
        Single,
        GraphTensorStorage::Owned,
        Unspecified
    ),
    state_field!(
        "qr",
        GraphDecodeState,
        F32,
        Single,
        GraphTensorStorage::Owned,
        Unspecified
    ),
    state_field!(
        "qr_norm",
        GraphDecodeState,
        F32,
        Single,
        GraphTensorStorage::Owned,
        Unspecified
    ),
    state_field!(
        "q",
        GraphDecodeState,
        F32,
        Single,
        GraphTensorStorage::Owned,
        Unspecified
    ),
    state_field!(
        "kv_raw",
        GraphDecodeState,
        F32,
        Single,
        GraphTensorStorage::Owned,
        Unspecified
    ),
    state_field!(
        "kv",
        GraphDecodeState,
        F32,
        Single,
        GraphTensorStorage::Owned,
        Unspecified
    ),
    state_field!(
        "layer_raw_cache",
        GraphPersistentKvState,
        F32,
        PerLayer,
        GraphTensorStorage::Owned,
        ZeroFullCapacity
    ),
    state_field!(
        "layer_attn_comp_cache",
        GraphPersistentKvState,
        F32,
        PerLayer,
        GraphTensorStorage::Owned,
        ZeroFullCapacity
    ),
    state_field!(
        "layer_attn_state_kv",
        GraphPersistentKvState,
        F32,
        PerLayer,
        GraphTensorStorage::Owned,
        ZeroState
    ),
    state_field!(
        "layer_attn_state_score",
        GraphPersistentKvState,
        F32,
        PerLayer,
        GraphTensorStorage::Owned,
        NegativeInfinityState
    ),
    state_field!(
        "layer_index_comp_cache",
        GraphPersistentKvState,
        F32,
        PerLayer,
        GraphTensorStorage::Owned,
        ZeroFullCapacity
    ),
    state_field!(
        "layer_index_state_kv",
        GraphPersistentKvState,
        F32,
        PerLayer,
        GraphTensorStorage::Owned,
        ZeroState
    ),
    state_field!(
        "layer_index_state_score",
        GraphPersistentKvState,
        F32,
        PerLayer,
        GraphTensorStorage::Owned,
        NegativeInfinityState
    ),
    state_field!(
        "comp_kv_cur",
        GraphLayerWorkState,
        F32,
        Single,
        GraphTensorStorage::Owned,
        Unspecified
    ),
    state_field!(
        "comp_sc_cur",
        GraphLayerWorkState,
        F32,
        Single,
        GraphTensorStorage::Owned,
        Unspecified
    ),
    state_field!(
        "indexer_q",
        GraphLayerWorkState,
        F32,
        Single,
        GraphTensorStorage::Owned,
        Unspecified
    ),
    state_field!(
        "indexer_weights",
        GraphLayerWorkState,
        F32,
        Single,
        GraphTensorStorage::Owned,
        Unspecified
    ),
    state_field!(
        "indexer_scores",
        GraphLayerWorkState,
        F32,
        Single,
        GraphTensorStorage::Owned,
        Unspecified
    ),
    state_field!(
        "comp_mask",
        GraphLayerWorkState,
        F32,
        Single,
        GraphTensorStorage::Owned,
        Unspecified
    ),
    state_field!(
        "comp_selected",
        GraphLayerWorkState,
        U32,
        Single,
        GraphTensorStorage::Owned,
        Unspecified
    ),
    state_field!(
        "heads",
        GraphLayerWorkState,
        F32,
        Single,
        GraphTensorStorage::Owned,
        Unspecified
    ),
    state_field!(
        "attn_low",
        GraphLayerWorkState,
        F32,
        Single,
        GraphTensorStorage::Owned,
        Unspecified
    ),
    state_field!(
        "attn_out",
        GraphLayerWorkState,
        F32,
        Single,
        GraphTensorStorage::Owned,
        Unspecified
    ),
    state_field!(
        "after_attn_hc",
        GraphLayerWorkState,
        F32,
        Single,
        GraphTensorStorage::Owned,
        Unspecified
    ),
    state_field!(
        "ffn_cur",
        GraphLayerWorkState,
        F32,
        Single,
        GraphTensorStorage::Owned,
        Unspecified
    ),
    state_field!(
        "ffn_norm",
        GraphLayerWorkState,
        F32,
        Single,
        GraphTensorStorage::Owned,
        Unspecified
    ),
    state_field!(
        "shared_gate",
        GraphLayerWorkState,
        F32,
        Single,
        GraphTensorStorage::Owned,
        Unspecified
    ),
    state_field!(
        "shared_up",
        GraphLayerWorkState,
        F32,
        Single,
        GraphTensorStorage::Owned,
        Unspecified
    ),
    state_field!(
        "shared_mid",
        GraphLayerWorkState,
        F32,
        Single,
        GraphTensorStorage::Owned,
        Unspecified
    ),
    state_field!(
        "shared_out",
        GraphLayerWorkState,
        F32,
        Single,
        GraphTensorStorage::Owned,
        Unspecified
    ),
    state_field!(
        "router_logits",
        GraphLayerWorkState,
        F32,
        Single,
        GraphTensorStorage::Owned,
        Unspecified
    ),
    state_field!(
        "router_probs",
        GraphLayerWorkState,
        F32,
        Single,
        GraphTensorStorage::Owned,
        Unspecified
    ),
    state_field!(
        "router_selected",
        GraphLayerWorkState,
        I32,
        Single,
        GraphTensorStorage::Owned,
        Unspecified
    ),
    state_field!(
        "router_weights",
        GraphLayerWorkState,
        F32,
        Single,
        GraphTensorStorage::Owned,
        Unspecified
    ),
    state_field!(
        "routed_gate",
        GraphLayerWorkState,
        F32,
        Single,
        GraphTensorStorage::Owned,
        Unspecified
    ),
    state_field!(
        "routed_up",
        GraphLayerWorkState,
        F32,
        Single,
        GraphTensorStorage::Owned,
        Unspecified
    ),
    state_field!(
        "routed_mid",
        GraphLayerWorkState,
        F32,
        Single,
        GraphTensorStorage::Owned,
        Unspecified
    ),
    state_field!(
        "routed_down",
        GraphLayerWorkState,
        F32,
        Single,
        GraphTensorStorage::Owned,
        Unspecified
    ),
    state_field!(
        "routed_out",
        GraphLayerWorkState,
        F32,
        Single,
        GraphTensorStorage::Owned,
        Unspecified
    ),
    state_field!(
        "ffn_out",
        GraphLayerWorkState,
        F32,
        Single,
        GraphTensorStorage::LazyOwned,
        Unspecified
    ),
    state_field!(
        "after_ffn_hc",
        GraphLayerWorkState,
        F32,
        Single,
        GraphTensorStorage::Owned,
        Unspecified
    ),
    state_field!(
        "output_pre",
        GraphLayerWorkState,
        F32,
        Single,
        GraphTensorStorage::Owned,
        Unspecified
    ),
    state_field!(
        "output_weights",
        GraphLayerWorkState,
        F32,
        Single,
        GraphTensorStorage::Owned,
        Unspecified
    ),
    state_field!(
        "output_embd",
        GraphLayerWorkState,
        F32,
        Single,
        GraphTensorStorage::Owned,
        Unspecified
    ),
    state_field!(
        "output_norm",
        GraphLayerWorkState,
        F32,
        Single,
        GraphTensorStorage::Owned,
        Unspecified
    ),
    state_field!(
        "logits",
        GraphLayerWorkState,
        F32,
        Single,
        GraphTensorStorage::Owned,
        Unspecified
    ),
    state_field!(
        "directional_steering_dirs",
        GraphOptionalControlState,
        F32,
        Single,
        GraphTensorStorage::External,
        ExternalInput
    ),
];

pub fn decode_graph_state_field(name: &str) -> Option<&'static GraphStateFieldPlan> {
    DECODE_GRAPH_STATE_FIELDS
        .iter()
        .find(|field| field.name == name)
}

pub const fn is_decode_graph_state_owner(owner: TensorOwner) -> bool {
    matches!(
        owner,
        TensorOwner::GraphDecodeState
            | TensorOwner::GraphPersistentKvState
            | TensorOwner::GraphLayerWorkState
            | TensorOwner::GraphOptionalControlState
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_plan::{GRAPH_TENSOR_FIELDS, N_HEAD_DIM, N_INDEXER_HEAD_DIM};

    #[test]
    fn decode_state_fields_match_graph_inventory_order() {
        let mut index = 0usize;
        for spec in GRAPH_TENSOR_FIELDS {
            if is_decode_graph_state_owner(spec.owner) {
                assert_eq!(DECODE_GRAPH_STATE_FIELDS[index].name, spec.name);
                assert_eq!(DECODE_GRAPH_STATE_FIELDS[index].owner, spec.owner);
                assert_eq!(
                    DECODE_GRAPH_STATE_FIELDS[index].element_type,
                    spec.element_type
                );
                index += 1;
            }
        }
        assert_eq!(index, DECODE_GRAPH_STATE_FIELDS.len());
        assert_eq!(DECODE_GRAPH_STATE_FIELDS.len(), 55);
    }

    #[test]
    fn decode_state_marks_hc_split_views_and_lazy_ffn_out() {
        assert_eq!(
            decode_graph_state_field("hc_pre").map(|field| field.storage),
            Some(GraphTensorStorage::View {
                base: "hc_split",
                offset_bytes: 0,
            })
        );
        assert_eq!(
            decode_graph_state_field("hc_post").map(|field| field.storage),
            Some(GraphTensorStorage::View {
                base: "hc_split",
                offset_bytes: u64::from(N_HC) * BYTES_F32,
            })
        );
        assert_eq!(
            decode_graph_state_field("hc_comb").map(|field| field.storage),
            Some(GraphTensorStorage::View {
                base: "hc_split",
                offset_bytes: 2 * u64::from(N_HC) * BYTES_F32,
            })
        );
        assert_eq!(
            decode_graph_state_field("ffn_out").map(|field| field.storage),
            Some(GraphTensorStorage::LazyOwned)
        );
        assert_eq!(
            decode_graph_state_field("directional_steering_dirs").map(|field| field.storage),
            Some(GraphTensorStorage::External)
        );
    }

    #[test]
    fn decode_state_initial_allocation_summary_matches_c_shape() {
        let plan = GraphPlan::for_context(32768, 32768, false);
        let mut owned = 0u32;
        let mut views = 0u32;
        let mut lazy = 0u32;
        let mut external = 0u32;
        let mut zero_full = 0u32;
        let mut zero_state = 0u32;
        let mut neg_inf_state = 0u32;

        for field in DECODE_GRAPH_STATE_FIELDS {
            match field.instances {
                GraphTensorInstances::Single => count_instance(
                    *field,
                    plan,
                    None,
                    &mut owned,
                    &mut views,
                    &mut lazy,
                    &mut external,
                    &mut zero_full,
                    &mut zero_state,
                    &mut neg_inf_state,
                ),
                GraphTensorInstances::PerLayer => {
                    for layer in 0..N_LAYER {
                        count_instance(
                            *field,
                            plan,
                            Some(layer),
                            &mut owned,
                            &mut views,
                            &mut lazy,
                            &mut external,
                            &mut zero_full,
                            &mut zero_state,
                            &mut neg_inf_state,
                        );
                    }
                }
            }
        }

        assert_eq!(owned, 272);
        assert_eq!(views, 3);
        assert_eq!(lazy, 1);
        assert_eq!(external, 1);
        assert_eq!(zero_full, 105);
        assert_eq!(zero_state, 62);
        assert_eq!(neg_inf_state, 62);
    }

    #[test]
    fn decode_state_cache_byte_sizes_use_full_layer_capacity() {
        let plan = GraphPlan::for_context(32768, 32768, false);
        let raw = *decode_graph_state_field("layer_raw_cache").expect("raw cache");
        assert_eq!(
            raw.byte_len(plan, ModelGraphDims::DS4_FLASH, Some(0)),
            Ok(TensorByteLen::Known(
                2304 * u64::from(N_HEAD_DIM) * BYTES_F32
            ))
        );
        let attn = *decode_graph_state_field("layer_attn_comp_cache").expect("attn cache");
        assert_eq!(
            attn.byte_len(plan, ModelGraphDims::DS4_FLASH, Some(2)),
            Ok(TensorByteLen::Known(
                8194 * u64::from(N_HEAD_DIM) * BYTES_F32
            ))
        );
        assert_eq!(
            attn.byte_len(plan, ModelGraphDims::DS4_FLASH, Some(3)),
            Ok(TensorByteLen::Known(
                258 * u64::from(N_HEAD_DIM) * BYTES_F32
            ))
        );
        let index = *decode_graph_state_field("layer_index_comp_cache").expect("index cache");
        assert_eq!(
            index.byte_len(plan, ModelGraphDims::DS4_FLASH, Some(2)),
            Ok(TensorByteLen::Known(
                8194 * u64::from(N_INDEXER_HEAD_DIM) * BYTES_F32
            ))
        );
        assert_eq!(
            index.byte_len(plan, ModelGraphDims::DS4_FLASH, Some(3)),
            Ok(TensorByteLen::Known(0))
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn count_instance(
        field: GraphStateFieldPlan,
        plan: GraphPlan,
        layer: Option<usize>,
        owned: &mut u32,
        views: &mut u32,
        lazy: &mut u32,
        external: &mut u32,
        zero_full: &mut u32,
        zero_state: &mut u32,
        neg_inf_state: &mut u32,
    ) {
        let allocation = field
            .initial_allocation(plan, ModelGraphDims::DS4_FLASH, layer)
            .expect("allocation");
        match allocation.byte_len {
            TensorByteLen::Known(0) => return,
            TensorByteLen::Known(_) => {}
            TensorByteLen::External => {
                *external += 1;
                return;
            }
        }
        match allocation.storage {
            GraphTensorStorage::Owned => *owned += 1,
            GraphTensorStorage::View { .. } => *views += 1,
            GraphTensorStorage::LazyOwned => *lazy += 1,
            GraphTensorStorage::External => *external += 1,
        }
        match allocation.initial_fill {
            GraphTensorInitialFill::ZeroFullCapacity => *zero_full += 1,
            GraphTensorInitialFill::ZeroState => *zero_state += 1,
            GraphTensorInitialFill::NegativeInfinityState => *neg_inf_state += 1,
            GraphTensorInitialFill::Unspecified | GraphTensorInitialFill::ExternalInput => {}
        }
    }
}
