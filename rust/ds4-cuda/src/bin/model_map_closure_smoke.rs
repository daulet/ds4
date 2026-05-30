use std::path::PathBuf;

use ds4_cuda::model_map::{
    AsyncPinnedCacheConfig, AsyncPinnedCacheOutcome, AsyncPinnedRangeCache, MappedModelFile,
    ModelLoadProgressMode,
};
use ds4_cuda::{substrate::CudaOxideSubstrate, M14_1B2C_SCOPE};

const CHUNK_BYTES: u64 = 4096;
const CACHED_OFFSET: u64 = 13;
const CACHED_BYTES: u64 = CHUNK_BYTES * 2;
const CONTAINED_OFFSET: u64 = CACHED_OFFSET + CHUNK_BYTES + 7;
const CONTAINED_BYTES: u64 = 257;
const RETAINED_OFFSET: u64 = CACHED_OFFSET + CACHED_BYTES + CHUNK_BYTES;

fn config(keep_source_pages: bool, progress_mode: ModelLoadProgressMode) -> AsyncPinnedCacheConfig {
    AsyncPinnedCacheConfig {
        copy_chunk_bytes: CHUNK_BYTES,
        arena_chunk_bytes: 16384,
        cache_limit_bytes: 16384,
        keep_source_pages,
        progress_mode,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: ds4-cuda-model-map-closure-smoke MODEL.gguf")?;
    let model = MappedModelFile::open(&model_path)?;
    let substrate = CudaOxideSubstrate::open(0)?;
    let stats = {
        let mut cache =
            AsyncPinnedRangeCache::new(&model, config(false, ModelLoadProgressMode::NonTty))?;
        assert_eq!(
            cache.cache_range(&substrate, CACHED_OFFSET, CACHED_BYTES)?,
            AsyncPinnedCacheOutcome::Inserted
        );
        assert_eq!(
            cache.cache_range(&substrate, CACHED_OFFSET, CACHED_BYTES)?,
            AsyncPinnedCacheOutcome::Reused
        );
        assert_eq!(
            cache.cache_range(&substrate, CONTAINED_OFFSET, CONTAINED_BYTES)?,
            AsyncPinnedCacheOutcome::Reused
        );
        assert_eq!(
            cache.readback(&substrate, CONTAINED_OFFSET, CONTAINED_BYTES)?,
            model.range(CONTAINED_OFFSET, CONTAINED_BYTES)?
        );
        cache.stats()
    };
    assert_eq!(stats.chunks_uploaded, 2);
    assert_eq!(stats.exact_range_hits, 1);
    assert_eq!(stats.containing_range_hits, 1);
    assert_eq!(stats.source_file_discard_calls, 2);
    assert_eq!(stats.source_file_discard_bytes, CACHED_BYTES);
    assert_eq!(stats.source_mapping_discard_calls, 2);
    assert_eq!(stats.source_mapping_discard_bytes, CHUNK_BYTES * 4);
    assert_eq!(stats.progress_notes, 3);
    assert_eq!(stats.progress_messages, 1);

    let fresh = AsyncPinnedRangeCache::new(&model, config(true, ModelLoadProgressMode::Disabled))?;
    let fresh_stats = fresh.stats();
    assert_eq!(fresh_stats.range_count, 0);
    assert_eq!(fresh_stats.progress_messages, 0);
    drop(fresh);

    let mut retained =
        AsyncPinnedRangeCache::new(&model, config(true, ModelLoadProgressMode::Disabled))?;
    assert_eq!(
        retained.cache_range(&substrate, RETAINED_OFFSET, CHUNK_BYTES)?,
        AsyncPinnedCacheOutcome::Inserted
    );
    let retained_stats = retained.stats();
    assert_eq!(retained_stats.source_file_discard_calls, 0);
    assert_eq!(retained_stats.source_mapping_discard_calls, 0);
    assert_eq!(retained_stats.progress_messages, 0);

    println!(
        "{{\"milestone\":\"M14.1b2c\",\"device_name\":{:?},\"model_size\":{},\"cached_offset\":{},\"cached_bytes\":{},\"contained_offset\":{},\"contained_bytes\":{},\"chunks_uploaded\":{},\"exact_range_hits\":{},\"containing_range_hits\":{},\"contained_range_reused\":true,\"contained_readback_matches\":true,\"source_file_discard_calls\":{},\"source_file_discard_bytes\":{},\"source_mapping_discard_calls\":{},\"source_mapping_discard_bytes\":{},\"progress_notes\":{},\"progress_messages\":{},\"non_tty_progress_initial_message\":true,\"fresh_cache_starts_empty\":true,\"keep_source_pages_suppresses_advice\":true,\"disabled_progress_suppresses_messages\":true,\"owns_containing_range_reuse\":{},\"owns_source_page_discard_policy\":{},\"owns_progress_reporting\":{},\"owns_raii_cache_cleanup\":{},\"owns_ds4_kernels\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        model.size(),
        CACHED_OFFSET,
        CACHED_BYTES,
        CONTAINED_OFFSET,
        CONTAINED_BYTES,
        stats.chunks_uploaded,
        stats.exact_range_hits,
        stats.containing_range_hits,
        stats.source_file_discard_calls,
        stats.source_file_discard_bytes,
        stats.source_mapping_discard_calls,
        stats.source_mapping_discard_bytes,
        stats.progress_notes,
        stats.progress_messages,
        M14_1B2C_SCOPE.owns_containing_range_reuse,
        M14_1B2C_SCOPE.owns_source_page_discard_policy,
        M14_1B2C_SCOPE.owns_progress_reporting,
        M14_1B2C_SCOPE.owns_raii_cache_cleanup,
        M14_1B2C_SCOPE.owns_ds4_kernels,
        M14_1B2C_SCOPE.changes_default_route
    );
    Ok(())
}
