use std::path::PathBuf;

use ds4_cuda::model_map::{
    CacheOutcome, MappedModelFile, ModelRangeCache, ModelRangeStrategy, RegisteredRangeResolution,
};
use ds4_cuda::{substrate::CudaOxideSubstrate, M14_1B2B2_SCOPE};

const RANGE_OFFSET: u64 = 13;
const RANGE_BYTES: u64 = 4096;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: ds4-cuda-model-registered-range-smoke MODEL.gguf")?;
    let model = MappedModelFile::open(&model_path)?;
    let expected = model.range(RANGE_OFFSET, RANGE_BYTES)?.to_vec();
    let layout = model.registered_range_layout(RANGE_OFFSET, RANGE_BYTES)?;
    assert_eq!(layout.registered_offset % layout.page_size, 0);
    assert_eq!(layout.registered_bytes % layout.page_size, 0);
    assert_eq!(layout.device_offset, RANGE_OFFSET);
    assert!(layout.registered_bytes > RANGE_BYTES);

    let substrate = CudaOxideSubstrate::open(0)?;
    let mut cache = ModelRangeCache::default();
    let strategy = ModelRangeStrategy::ReadOnlyRegisteredOrMmapDeviceCopy;
    assert_eq!(
        cache.cache_range_with_strategy(&substrate, &model, RANGE_OFFSET, RANGE_BYTES, strategy)?,
        CacheOutcome::Inserted
    );
    assert_eq!(
        cache.cache_range_with_strategy(&substrate, &model, RANGE_OFFSET, RANGE_BYTES, strategy)?,
        CacheOutcome::Reused
    );
    assert_eq!(
        cache.readback_with_strategy(&substrate, RANGE_OFFSET, RANGE_BYTES, strategy)?,
        expected
    );
    assert_eq!(cache.len(), 1);

    let resolution = cache
        .registered_resolution(RANGE_OFFSET, RANGE_BYTES)
        .ok_or("registered strategy did not record a resolution")?;
    let (read_only_registration_supported, mmap_device_copy_fallback, error_code) = match resolution
    {
        RegisteredRangeResolution::ReadOnlyMapped => (true, false, 0),
        RegisteredRangeResolution::MmapDeviceCopyFallback(err) => (false, true, err.0),
    };
    assert!(!read_only_registration_supported);
    assert!(mmap_device_copy_fallback);
    assert_eq!(
        error_code,
        cuda_core::sys::cudaError_enum_CUDA_ERROR_NOT_SUPPORTED
    );

    println!(
        "{{\"milestone\":\"M14.1b2b2\",\"device_name\":{:?},\"model_size\":{},\"range_offset\":{},\"range_bytes\":{},\"registration_page_size\":{},\"registration_offset\":{},\"registration_bytes\":{},\"registration_device_offset\":{},\"read_only_registration_attempted\":true,\"read_only_registration_supported\":{},\"read_only_registration_error_code\":{},\"mmap_device_copy_fallback\":{},\"fallback_readback_matches\":true,\"strategy_cache_reused\":true,\"owns_page_aligned_read_only_registration_attempt\":{},\"owns_mmap_device_copy_fallback_after_registration_error\":{},\"owns_pageable_hmm_strategy\":{},\"owns_o_direct_staging\":{},\"owns_ds4_kernels\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        model.size(),
        RANGE_OFFSET,
        RANGE_BYTES,
        layout.page_size,
        layout.registered_offset,
        layout.registered_bytes,
        layout.device_offset,
        read_only_registration_supported,
        error_code,
        mmap_device_copy_fallback,
        M14_1B2B2_SCOPE.owns_page_aligned_read_only_registration_attempt,
        M14_1B2B2_SCOPE.owns_mmap_device_copy_fallback_after_registration_error,
        M14_1B2B2_SCOPE.owns_pageable_hmm_strategy,
        M14_1B2B2_SCOPE.owns_o_direct_staging,
        M14_1B2B2_SCOPE.owns_ds4_kernels,
        M14_1B2B2_SCOPE.changes_default_route
    );
    Ok(())
}
