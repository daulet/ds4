use crate::allocation_policy::DeviceMemoryCapacity;

const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;
const LARGE_DEVICE_BYTES: u64 = 112 * GIB;
const LARGE_DEVICE_RESERVE_BYTES: u64 = 512 * MIB;
const SMALL_DEVICE_MIN_RESERVE_BYTES: u64 = 4096 * MIB;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BlasMathPolicy {
    Default,
    Tf32TensorOp,
}

pub fn quality_blas_math_policy(quality_mode: bool, no_tf32: bool) -> BlasMathPolicy {
    if quality_mode || no_tf32 {
        BlasMathPolicy::Default
    } else {
        BlasMathPolicy::Tf32TensorOp
    }
}

#[cfg(feature = "cuda-oxide-backend")]
pub fn apply_quality_blas_policy(
    blas: &cuda_core::Blas,
    quality_mode: bool,
    no_tf32: bool,
) -> Result<BlasMathPolicy, cuda_core::BlasError> {
    let policy = quality_blas_math_policy(quality_mode, no_tf32);
    let mode = match policy {
        BlasMathPolicy::Default => cuda_core::BlasMathMode::Default,
        BlasMathPolicy::Tf32TensorOp => cuda_core::BlasMathMode::Tf32TensorOp,
    };
    blas.set_math_mode(mode)?;
    Ok(policy)
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Q8CacheOptions {
    pub quality_mode: bool,
    pub no_q8_f16_cache: bool,
    pub q8_f16_all: bool,
    pub no_attention_output_f16_cache: bool,
    pub no_attn_q_b_f16_cache: bool,
    pub attention_output_preload: bool,
    pub q8_f16_limit_bytes: Option<u64>,
    pub q8_f16_reserve_bytes: Option<u64>,
    pub no_q8_f32_cache: bool,
    pub q8_f32_all: bool,
    pub attn_q_b_f32_cache: bool,
    pub q8_f32_large: bool,
    pub q8_f32_preload: bool,
    pub weight_cache_verbose: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Q8CacheState {
    pub f16_cached_bytes: u64,
    pub f16_disabled_after_failure: bool,
    pub f16_budget_notice_printed: bool,
    pub optional_preload_disabled: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Q8F16AdmissionReason {
    Admitted,
    QualityMode,
    DisabledAfterFailure,
    DisabledBySetting,
    PreloadSuppressed,
    NotEligible,
    LimitReached,
    MemoryQueryUnavailable,
    BudgetExhausted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Q8F16Admission {
    pub admitted: bool,
    pub reason: Q8F16AdmissionReason,
    pub reserve_bytes: u64,
    pub emit_budget_notice: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Q8PreloadFormat {
    F16,
    F32,
}

pub fn q8_f16_cache_reserve_bytes(total_bytes: u64, override_bytes: Option<u64>) -> u64 {
    if let Some(reserve_bytes) = override_bytes {
        return reserve_bytes;
    }
    if total_bytes >= LARGE_DEVICE_BYTES {
        return LARGE_DEVICE_RESERVE_BYTES;
    }
    (total_bytes / 20).max(SMALL_DEVICE_MIN_RESERVE_BYTES)
}

pub fn q8_f16_cache_allowed(
    options: Q8CacheOptions,
    label: Option<&str>,
    in_dim: u64,
    out_dim: u64,
    state: Q8CacheState,
) -> bool {
    if options.quality_mode
        || state.f16_disabled_after_failure
        || options.no_q8_f16_cache
        || options.q8_f16_limit_bytes == Some(0)
    {
        return false;
    }
    if options.q8_f16_all {
        return true;
    }
    let Some(label) = label else {
        return false;
    };
    if q8_label_is_attention_output(label) {
        return !options.no_attention_output_f16_cache;
    }
    if label.contains("attn_q_b") {
        return !options.no_attn_q_b_f16_cache;
    }
    if label.contains("ffn_gate_shexp")
        || label.contains("ffn_up_shexp")
        || label.contains("ffn_down_shexp")
    {
        return true;
    }
    matches!(
        (in_dim, out_dim),
        (4096, 2048) | (2048, 4096) | (4096, 1024) | (4096, 512)
    ) || (!options.no_attn_q_b_f16_cache && (in_dim, out_dim) == (1024, 32768))
}

pub fn q8_f16_preload_allowed(
    options: Q8CacheOptions,
    label: Option<&str>,
    in_dim: u64,
    out_dim: u64,
    state: Q8CacheState,
) -> bool {
    if label.is_some_and(q8_label_is_attention_output)
        && !options.attention_output_preload
        && !options.q8_f16_all
    {
        return false;
    }
    q8_f16_cache_allowed(options, label, in_dim, out_dim, state)
}

pub fn q8_f32_cache_allowed(
    options: Q8CacheOptions,
    label: Option<&str>,
    in_dim: u64,
    out_dim: u64,
) -> bool {
    if options.no_q8_f32_cache {
        return false;
    }
    if options.q8_f32_all {
        return true;
    }
    if label.is_some_and(|label| label.contains("attn_q_b")) {
        return options.attn_q_b_f32_cache;
    }
    options.q8_f32_large && (in_dim, out_dim) == (1024, 32768)
}

pub fn q8_preload_format(
    options: Q8CacheOptions,
    label: Option<&str>,
    in_dim: u64,
    out_dim: u64,
    state: Q8CacheState,
) -> Option<Q8PreloadFormat> {
    if state.optional_preload_disabled {
        return None;
    }
    if options.q8_f32_preload && q8_f32_cache_allowed(options, label, in_dim, out_dim) {
        return Some(Q8PreloadFormat::F32);
    }
    q8_f16_preload_allowed(options, label, in_dim, out_dim, state).then_some(Q8PreloadFormat::F16)
}

impl Q8CacheState {
    pub fn admit_f16_bytes(
        &mut self,
        options: Q8CacheOptions,
        label: Option<&str>,
        in_dim: u64,
        out_dim: u64,
        request_bytes: u64,
        memory: Option<DeviceMemoryCapacity>,
        preload: bool,
    ) -> Q8F16Admission {
        if options.quality_mode {
            return admission(false, Q8F16AdmissionReason::QualityMode, 0, false);
        }
        if self.f16_disabled_after_failure {
            return admission(false, Q8F16AdmissionReason::DisabledAfterFailure, 0, false);
        }
        if options.no_q8_f16_cache || options.q8_f16_limit_bytes == Some(0) {
            return admission(false, Q8F16AdmissionReason::DisabledBySetting, 0, false);
        }
        if preload
            && label.is_some_and(q8_label_is_attention_output)
            && !options.attention_output_preload
            && !options.q8_f16_all
        {
            return admission(false, Q8F16AdmissionReason::PreloadSuppressed, 0, false);
        }
        if !q8_f16_cache_allowed(options, label, in_dim, out_dim, *self) {
            return admission(false, Q8F16AdmissionReason::NotEligible, 0, false);
        }
        if let Some(limit_bytes) = options.q8_f16_limit_bytes {
            if self.f16_cached_bytes > limit_bytes
                || request_bytes > limit_bytes - self.f16_cached_bytes
            {
                let emit = self.note_budget_rejection(options);
                return admission(false, Q8F16AdmissionReason::LimitReached, 0, emit);
            }
        }
        let Some(memory) = memory else {
            return admission(
                false,
                Q8F16AdmissionReason::MemoryQueryUnavailable,
                0,
                false,
            );
        };
        let reserve_bytes =
            q8_f16_cache_reserve_bytes(memory.total_bytes, options.q8_f16_reserve_bytes);
        if request_bytes > memory.free_bytes || memory.free_bytes - request_bytes < reserve_bytes {
            let emit = self.note_budget_rejection(options);
            return admission(
                false,
                Q8F16AdmissionReason::BudgetExhausted,
                reserve_bytes,
                emit,
            );
        }
        admission(true, Q8F16AdmissionReason::Admitted, reserve_bytes, false)
    }

    pub fn record_f16_success(&mut self, cached_bytes: u64) {
        self.f16_cached_bytes = self.f16_cached_bytes.saturating_add(cached_bytes);
    }

    pub fn disable_f16_after_failure(&mut self) -> bool {
        let emit_notice = !self.f16_disabled_after_failure;
        self.f16_disabled_after_failure = true;
        self.f16_cached_bytes = 0;
        emit_notice
    }

    pub fn disable_optional_preload_after_failure(&mut self) {
        self.optional_preload_disabled = true;
    }

    fn note_budget_rejection(&mut self, options: Q8CacheOptions) -> bool {
        let emit = !self.f16_budget_notice_printed || options.weight_cache_verbose;
        self.f16_budget_notice_printed = true;
        emit
    }
}

fn q8_label_is_attention_output(label: &str) -> bool {
    label.contains("attn_output_a")
        || label.contains("attn_output_b")
        || label.contains("attention_output_a")
        || label.contains("attention_output_b")
}

const fn admission(
    admitted: bool,
    reason: Q8F16AdmissionReason,
    reserve_bytes: u64,
    emit_budget_notice: bool,
) -> Q8F16Admission {
    Q8F16Admission {
        admitted,
        reason,
        reserve_bytes,
        emit_budget_notice,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        q8_f16_cache_allowed, q8_f16_cache_reserve_bytes, q8_f16_preload_allowed,
        q8_f32_cache_allowed, q8_preload_format, quality_blas_math_policy, BlasMathPolicy,
        Q8CacheOptions, Q8CacheState, Q8F16AdmissionReason, Q8PreloadFormat, GIB, MIB,
    };
    use crate::allocation_policy::DeviceMemoryCapacity;

    #[test]
    fn quality_mode_and_tf32_override_match_current_c() {
        assert_eq!(
            quality_blas_math_policy(false, false),
            BlasMathPolicy::Tf32TensorOp
        );
        assert_eq!(
            quality_blas_math_policy(true, false),
            BlasMathPolicy::Default
        );
        assert_eq!(
            quality_blas_math_policy(false, true),
            BlasMathPolicy::Default
        );
    }

    #[test]
    fn q8_f16_eligibility_and_preload_exclusions_match_current_c() {
        let options = Q8CacheOptions::default();
        let state = Q8CacheState::default();
        assert!(q8_f16_cache_allowed(
            options,
            Some("ffn_gate_shexp"),
            1,
            1,
            state
        ));
        assert!(q8_f16_cache_allowed(
            options,
            Some("dense_projection"),
            4096,
            2048,
            state
        ));
        assert!(!q8_f16_cache_allowed(options, None, 4096, 2048, state));
        assert!(q8_f16_cache_allowed(options, Some("attn_q_b"), 1, 1, state));
        assert!(!q8_f16_preload_allowed(
            options,
            Some("attn_output_a"),
            1,
            1,
            state
        ));
        assert!(q8_f16_preload_allowed(
            Q8CacheOptions {
                attention_output_preload: true,
                ..options
            },
            Some("attn_output_a"),
            1,
            1,
            state
        ));
        assert!(!q8_f16_cache_allowed(
            Q8CacheOptions {
                quality_mode: true,
                ..options
            },
            Some("ffn_gate_shexp"),
            1,
            1,
            state
        ));
    }

    #[test]
    fn q8_f16_budget_reserve_and_failure_state_match_current_c() {
        assert_eq!(q8_f16_cache_reserve_bytes(112 * GIB, None), 512 * MIB);
        assert_eq!(q8_f16_cache_reserve_bytes(80 * GIB, None), 4 * GIB);
        assert_eq!(q8_f16_cache_reserve_bytes(128 * GIB, Some(17)), 17);

        let mut state = Q8CacheState::default();
        let options = Q8CacheOptions::default();
        let fits = state.admit_f16_bytes(
            options,
            Some("ffn_gate_shexp"),
            1,
            1,
            GIB,
            Some(DeviceMemoryCapacity {
                free_bytes: 16 * GIB,
                total_bytes: 80 * GIB,
            }),
            false,
        );
        assert!(fits.admitted);
        assert_eq!(fits.reserve_bytes, 4 * GIB);
        state.record_f16_success(GIB);

        let equal_reserve = state.admit_f16_bytes(
            options,
            Some("ffn_gate_shexp"),
            1,
            1,
            12 * GIB,
            Some(DeviceMemoryCapacity {
                free_bytes: 16 * GIB,
                total_bytes: 80 * GIB,
            }),
            false,
        );
        assert!(equal_reserve.admitted);
        let exhausted = state.admit_f16_bytes(
            options,
            Some("ffn_gate_shexp"),
            1,
            1,
            13 * GIB,
            Some(DeviceMemoryCapacity {
                free_bytes: 16 * GIB,
                total_bytes: 80 * GIB,
            }),
            false,
        );
        assert_eq!(exhausted.reason, Q8F16AdmissionReason::BudgetExhausted);
        assert!(exhausted.emit_budget_notice);
        let repeated = state.admit_f16_bytes(
            options,
            Some("ffn_gate_shexp"),
            1,
            1,
            13 * GIB,
            Some(DeviceMemoryCapacity {
                free_bytes: 16 * GIB,
                total_bytes: 80 * GIB,
            }),
            false,
        );
        assert!(!repeated.emit_budget_notice);
        assert!(state.disable_f16_after_failure());
        assert!(!state.disable_f16_after_failure());
        assert_eq!(state.f16_cached_bytes, 0);
    }

    #[test]
    fn q8_f32_and_preload_choice_match_current_c() {
        let state = Q8CacheState::default();
        let options = Q8CacheOptions {
            q8_f32_preload: true,
            attn_q_b_f32_cache: true,
            ..Q8CacheOptions::default()
        };
        assert!(q8_f32_cache_allowed(options, Some("attn_q_b"), 1, 1));
        assert_eq!(
            q8_preload_format(options, Some("attn_q_b"), 1, 1, state),
            Some(Q8PreloadFormat::F32)
        );
        let mut disabled = state;
        disabled.disable_optional_preload_after_failure();
        assert_eq!(
            q8_preload_format(options, Some("attn_q_b"), 1, 1, disabled),
            None
        );
    }
}
