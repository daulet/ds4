use std::path::PathBuf;

use ds4_cuda::model_map::{CacheOutcome, MappedModelFile, ModelRangeCache, ModelRangeStrategy};
use ds4_cuda::{substrate::CudaOxideSubstrate, M14_1B2B1_SCOPE};

const RANGE_OFFSET: u64 = 0;
const RANGE_BYTES: u64 = 4096;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: ds4-cuda-model-range-strategy-smoke MODEL.gguf")?;
    let model = MappedModelFile::open(&model_path)?;
    let expected = model.range(RANGE_OFFSET, RANGE_BYTES)?.to_vec();

    let substrate = CudaOxideSubstrate::open(0)?;
    let mut cache = ModelRangeCache::default();
    for strategy in [
        ModelRangeStrategy::MmapDeviceCopy,
        ModelRangeStrategy::FileStagedDeviceCopy,
    ] {
        assert_eq!(
            cache.cache_range_with_strategy(
                &substrate,
                &model,
                RANGE_OFFSET,
                RANGE_BYTES,
                strategy,
            )?,
            CacheOutcome::Inserted
        );
        assert_eq!(
            cache.cache_range_with_strategy(
                &substrate,
                &model,
                RANGE_OFFSET,
                RANGE_BYTES,
                strategy,
            )?,
            CacheOutcome::Reused
        );
        assert_eq!(
            cache.readback_with_strategy(&substrate, RANGE_OFFSET, RANGE_BYTES, strategy)?,
            expected
        );
    }
    assert_eq!(cache.len(), 2);

    println!(
        "{{\"milestone\":\"M14.1b2b1\",\"device_name\":{:?},\"model_size\":{},\"range_offset\":{},\"range_bytes\":{},\"mmap_device_copy\":true,\"file_staged_device_copy\":true,\"strategy_readbacks_equal\":true,\"strategy_cache_reused\":true,\"owns_explicit_file_staged_device_copy_strategy\":{},\"owns_registered_range_strategy\":{},\"owns_pageable_hmm_strategy\":{},\"owns_o_direct_staging\":{},\"owns_ds4_kernels\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        model.size(),
        RANGE_OFFSET,
        RANGE_BYTES,
        M14_1B2B1_SCOPE.owns_explicit_file_staged_device_copy_strategy,
        M14_1B2B1_SCOPE.owns_registered_range_strategy,
        M14_1B2B1_SCOPE.owns_pageable_hmm_strategy,
        M14_1B2B1_SCOPE.owns_o_direct_staging,
        M14_1B2B1_SCOPE.owns_ds4_kernels,
        M14_1B2B1_SCOPE.changes_default_route
    );
    Ok(())
}
