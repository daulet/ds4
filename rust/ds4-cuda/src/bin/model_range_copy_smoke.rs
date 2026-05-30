use std::path::PathBuf;

use ds4_cuda::model_map::{CacheOutcome, MappedModelFile, ModelRangeCache};
use ds4_cuda::{substrate::CudaOxideSubstrate, M14_1B2A_SCOPE};

const RANGE_OFFSET: u64 = 0;
const RANGE_BYTES: u64 = 4096;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: ds4-cuda-model-range-copy-smoke MODEL.gguf")?;
    let model = MappedModelFile::open(&model_path)?;
    let expected = model.range(RANGE_OFFSET, RANGE_BYTES)?.to_vec();
    assert!(model.range(model.size() + 1, 1).is_err());

    let substrate = CudaOxideSubstrate::open(0)?;
    let mut cache = ModelRangeCache::default();
    assert_eq!(
        cache.cache_range(&substrate, &model, RANGE_OFFSET, RANGE_BYTES)?,
        CacheOutcome::Inserted
    );
    assert_eq!(
        cache.cache_range(&substrate, &model, RANGE_OFFSET, RANGE_BYTES)?,
        CacheOutcome::Reused
    );
    assert_eq!(cache.len(), 1);
    assert_eq!(
        cache.readback(&substrate, RANGE_OFFSET, RANGE_BYTES)?,
        expected
    );

    println!(
        "{{\"milestone\":\"M14.1b2a\",\"device_name\":{:?},\"model_size\":{},\"range_offset\":{},\"range_bytes\":{},\"bounds_rejected\":true,\"range_copy_readback\":true,\"range_cache_reused\":true,\"owns_mapped_model_file_lifetime\":{},\"owns_device_range_copy_cache\":{},\"owns_range_strategy_selection\":{},\"owns_ds4_kernels\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        model.size(),
        RANGE_OFFSET,
        RANGE_BYTES,
        M14_1B2A_SCOPE.owns_mapped_model_file_lifetime,
        M14_1B2A_SCOPE.owns_device_range_copy_cache,
        M14_1B2A_SCOPE.owns_range_strategy_selection,
        M14_1B2A_SCOPE.owns_ds4_kernels,
        M14_1B2A_SCOPE.changes_default_route
    );
    Ok(())
}
