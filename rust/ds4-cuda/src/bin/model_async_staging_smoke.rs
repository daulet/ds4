use std::path::PathBuf;

use ds4_cuda::model_map::{
    AsyncPinnedCacheConfig, AsyncPinnedCacheOutcome, AsyncPinnedRangeCache, DirectIoPolicyState,
    MappedModelFile, ModelLoadProgressMode,
};
use ds4_cuda::{substrate::CudaOxideSubstrate, M14_1B2B3B2_SCOPE};

const FIRST_OFFSET: u64 = 13;
const CHUNK_BYTES: u64 = 4096;
const FIRST_BYTES: u64 = CHUNK_BYTES * 6;
const SECOND_OFFSET: u64 = FIRST_OFFSET + FIRST_BYTES;
const SECOND_BYTES: u64 = CHUNK_BYTES;
const REJECTED_OFFSET: u64 = SECOND_OFFSET + SECOND_BYTES;
const REJECTED_BYTES: u64 = 1;
const ARENA_CHUNK_BYTES: u64 = 32768;
const CACHE_LIMIT_BYTES: u64 = FIRST_BYTES + SECOND_BYTES;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: ds4-cuda-model-async-staging-smoke MODEL.gguf")?;
    let model = MappedModelFile::open(&model_path)?;
    let substrate = CudaOxideSubstrate::open(0)?;
    let mut cache = AsyncPinnedRangeCache::new(
        &model,
        AsyncPinnedCacheConfig {
            copy_chunk_bytes: CHUNK_BYTES,
            arena_chunk_bytes: ARENA_CHUNK_BYTES,
            cache_limit_bytes: CACHE_LIMIT_BYTES,
            keep_source_pages: true,
            progress_mode: ModelLoadProgressMode::Disabled,
        },
    )?;

    assert_eq!(
        cache.cache_range(&substrate, FIRST_OFFSET, FIRST_BYTES)?,
        AsyncPinnedCacheOutcome::Inserted
    );
    assert_eq!(
        cache.readback(&substrate, FIRST_OFFSET, FIRST_BYTES)?,
        model.range(FIRST_OFFSET, FIRST_BYTES)?
    );
    assert_eq!(
        cache.cache_range(&substrate, FIRST_OFFSET, FIRST_BYTES)?,
        AsyncPinnedCacheOutcome::Reused
    );
    assert_eq!(
        cache.cache_range(&substrate, SECOND_OFFSET, SECOND_BYTES)?,
        AsyncPinnedCacheOutcome::Inserted
    );
    assert_eq!(
        cache.readback(&substrate, SECOND_OFFSET, SECOND_BYTES)?,
        model.range(SECOND_OFFSET, SECOND_BYTES)?
    );
    assert_eq!(
        cache.cache_range(&substrate, REJECTED_OFFSET, REJECTED_BYTES)?,
        AsyncPinnedCacheOutcome::BudgetFallback
    );
    assert!(cache
        .readback(&substrate, REJECTED_OFFSET, REJECTED_BYTES)
        .is_err());

    let stats = cache.stats();
    let direct_io_alignment = match stats.direct_io_state {
        DirectIoPolicyState::Enabled { alignment } => alignment,
        state => return Err(format!("expected enabled direct I/O, got {state:?}").into()),
    };
    assert_eq!(stats.stage_slots, 4);
    assert_eq!(stats.chunks_uploaded, 7);
    assert_eq!(stats.stage_slot_reuse_waits, 2);
    assert_eq!(stats.events_recorded, stats.chunks_uploaded);
    assert_eq!(stats.direct_io_chunks, stats.chunks_uploaded);
    assert_eq!(stats.buffered_chunks, 0);
    assert_eq!(stats.arena_count, 1);
    assert_eq!(stats.arena_bytes, ARENA_CHUNK_BYTES);
    assert_eq!(stats.range_count, 2);
    assert_eq!(stats.range_bytes, CACHE_LIMIT_BYTES);
    assert_eq!(stats.budget_fallbacks, 1);

    let mut alignment_cache = AsyncPinnedRangeCache::new(
        &model,
        AsyncPinnedCacheConfig {
            copy_chunk_bytes: CHUNK_BYTES,
            arena_chunk_bytes: 256,
            cache_limit_bytes: 257,
            keep_source_pages: true,
            progress_mode: ModelLoadProgressMode::Disabled,
        },
    )?;
    assert_eq!(
        alignment_cache.cache_range(&substrate, 0, 256)?,
        AsyncPinnedCacheOutcome::Inserted
    );
    assert_eq!(
        alignment_cache.cache_range(&substrate, 256, 1)?,
        AsyncPinnedCacheOutcome::BudgetFallback
    );

    println!(
        "{{\"milestone\":\"M14.1b2b3b2\",\"device_name\":{:?},\"model_size\":{},\"copy_chunk_bytes\":{},\"first_range_offset\":{},\"first_range_bytes\":{},\"second_range_offset\":{},\"second_range_bytes\":{},\"rejected_range_offset\":{},\"rejected_range_bytes\":{},\"direct_io_alignment\":{},\"stage_slots\":{},\"chunks_uploaded\":{},\"stage_slot_reuse_waits\":{},\"events_recorded\":{},\"direct_io_chunks\":{},\"buffered_chunks\":{},\"arena_count\":{},\"arena_bytes\":{},\"range_count\":{},\"range_bytes\":{},\"cache_limit_bytes\":{},\"budget_fallbacks\":{},\"budget_fallback_not_cached\":true,\"aligned_new_arena_budget_fallback\":true,\"exact_readbacks_match\":true,\"direct_io_disable_after_error_policy_present\":{},\"direct_io_error_branch_live_exercised\":false,\"owns_four_slot_event_ring\":{},\"owns_arena_range_allocation\":{},\"owns_range_cache_budget_fallback\":{},\"owns_source_page_discard_policy\":{},\"owns_progress_reporting\":{},\"owns_ds4_kernels\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        model.size(),
        CHUNK_BYTES,
        FIRST_OFFSET,
        FIRST_BYTES,
        SECOND_OFFSET,
        SECOND_BYTES,
        REJECTED_OFFSET,
        REJECTED_BYTES,
        direct_io_alignment,
        stats.stage_slots,
        stats.chunks_uploaded,
        stats.stage_slot_reuse_waits,
        stats.events_recorded,
        stats.direct_io_chunks,
        stats.buffered_chunks,
        stats.arena_count,
        stats.arena_bytes,
        stats.range_count,
        stats.range_bytes,
        CACHE_LIMIT_BYTES,
        stats.budget_fallbacks,
        M14_1B2B3B2_SCOPE.owns_direct_io_disable_after_error_policy,
        M14_1B2B3B2_SCOPE.owns_four_slot_event_ring,
        M14_1B2B3B2_SCOPE.owns_arena_range_allocation,
        M14_1B2B3B2_SCOPE.owns_range_cache_budget_fallback,
        M14_1B2B3B2_SCOPE.owns_source_page_discard_policy,
        M14_1B2B3B2_SCOPE.owns_progress_reporting,
        M14_1B2B3B2_SCOPE.owns_ds4_kernels,
        M14_1B2B3B2_SCOPE.changes_default_route
    );
    Ok(())
}
