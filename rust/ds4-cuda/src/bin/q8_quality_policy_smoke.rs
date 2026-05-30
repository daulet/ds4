use ds4_cuda::allocation_policy::DeviceMemoryCapacity;
use ds4_cuda::q8_policy::{
    apply_quality_blas_policy, q8_f16_cache_allowed, q8_f16_preload_allowed, q8_f32_cache_allowed,
    q8_preload_format, BlasMathPolicy, Q8CacheOptions, Q8CacheState, Q8F16AdmissionReason,
    Q8PreloadFormat,
};
use ds4_cuda::{substrate::CudaOxideSubstrate, M14_1B3B_SCOPE};

const GIB: u64 = 1024 * 1024 * 1024;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let substrate = CudaOxideSubstrate::open(0)?;
    let blas = substrate.blas_handle()?;
    assert_eq!(
        apply_quality_blas_policy(&blas, false, false)?,
        BlasMathPolicy::Tf32TensorOp
    );
    assert_eq!(
        apply_quality_blas_policy(&blas, true, false)?,
        BlasMathPolicy::Default
    );
    assert_eq!(
        apply_quality_blas_policy(&blas, false, true)?,
        BlasMathPolicy::Default
    );

    let options = Q8CacheOptions::default();
    let mut state = Q8CacheState::default();
    assert!(q8_f16_cache_allowed(
        options,
        Some("ffn_gate_shexp"),
        1,
        1,
        state
    ));
    assert!(!q8_f16_preload_allowed(
        options,
        Some("attn_output_a"),
        1,
        1,
        state
    ));

    let admitted = state.admit_f16_bytes(
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
    assert!(admitted.admitted);
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
    let budget_exhausted = state.admit_f16_bytes(
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
    assert_eq!(
        budget_exhausted.reason,
        Q8F16AdmissionReason::BudgetExhausted
    );
    assert!(budget_exhausted.emit_budget_notice);
    assert!(state.disable_f16_after_failure());
    assert!(!state.disable_f16_after_failure());

    let f32_options = Q8CacheOptions {
        q8_f32_preload: true,
        attn_q_b_f32_cache: true,
        ..Q8CacheOptions::default()
    };
    assert!(q8_f32_cache_allowed(f32_options, Some("attn_q_b"), 1, 1));
    assert_eq!(
        q8_preload_format(f32_options, Some("attn_q_b"), 1, 1, Q8CacheState::default()),
        Some(Q8PreloadFormat::F32)
    );
    state.disable_optional_preload_after_failure();
    assert!(state.optional_preload_disabled);

    println!(
        "{{\"milestone\":\"M14.1b3b\",\"device_name\":{:?},\"tf32_fast_mode_applied\":true,\"quality_default_math_applied\":true,\"no_tf32_default_math_applied\":true,\"f16_admission_policy\":true,\"attention_output_preload_suppression\":true,\"f16_budget_rejection\":true,\"f16_disable_after_failure\":true,\"f32_preload_policy\":true,\"optional_preload_disable_after_failure\":true,\"owns_q8_cache_admission_policy\":{},\"owns_q8_cache_failure_disable_policy\":{},\"owns_quality_blas_selection\":{},\"owns_converted_q8_buffers\":{},\"owns_dequant_kernels\":{},\"owns_ds4_kernels\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        M14_1B3B_SCOPE.owns_q8_cache_admission_policy,
        M14_1B3B_SCOPE.owns_q8_cache_failure_disable_policy,
        M14_1B3B_SCOPE.owns_quality_blas_selection,
        M14_1B3B_SCOPE.owns_converted_q8_buffers,
        M14_1B3B_SCOPE.owns_dequant_kernels,
        M14_1B3B_SCOPE.owns_ds4_kernels,
        M14_1B3B_SCOPE.changes_default_route,
    );
    Ok(())
}
