use std::path::PathBuf;

use ds4_cuda::model_map::{
    stage_pinned_model_range, MappedModelFile, PinnedStagePolicy, PinnedStageResolution,
};
use ds4_cuda::{substrate::CudaOxideSubstrate, M14_1B2B3B1_SCOPE};

const RANGE_OFFSET: u64 = 13;
const RANGE_BYTES: u64 = 4096;
const TAIL_BYTES: u64 = 13;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: ds4-cuda-model-direct-io-smoke MODEL.gguf")?;
    let model = MappedModelFile::open(&model_path)?;
    let substrate = CudaOxideSubstrate::open(0)?;

    let expected = model.range(RANGE_OFFSET, RANGE_BYTES)?.to_vec();
    let direct = stage_pinned_model_range(
        &substrate,
        &model,
        RANGE_OFFSET,
        RANGE_BYTES,
        PinnedStagePolicy::DirectIoOrBufferedFallback,
    )?;
    let (alignment, read_offset, read_bytes, payload_offset) = match direct.resolution() {
        PinnedStageResolution::DirectIo {
            alignment,
            read_offset,
            read_bytes,
            payload_offset,
        } => (alignment, read_offset, read_bytes, payload_offset),
        resolution => {
            return Err(format!("expected direct I/O resolution, got {resolution:?}").into())
        }
    };
    assert_eq!(direct.readback(&substrate)?, expected);

    let tail_offset = model.size() - TAIL_BYTES;
    let expected_tail = model.range(tail_offset, TAIL_BYTES)?.to_vec();
    let fallback = stage_pinned_model_range(
        &substrate,
        &model,
        tail_offset,
        TAIL_BYTES,
        PinnedStagePolicy::DirectIoOrBufferedFallback,
    )?;
    assert_eq!(
        fallback.resolution(),
        PinnedStageResolution::BufferedFallback
    );
    assert_eq!(fallback.readback(&substrate)?, expected_tail);

    println!(
        "{{\"milestone\":\"M14.1b2b3b1\",\"device_name\":{:?},\"model_size\":{},\"range_offset\":{},\"range_bytes\":{},\"direct_io_selected\":true,\"direct_io_alignment\":{},\"direct_io_read_offset\":{},\"direct_io_read_bytes\":{},\"direct_io_payload_offset\":{},\"direct_io_readback_matches\":true,\"tail_range_offset\":{},\"tail_range_bytes\":{},\"tail_buffered_fallback\":true,\"tail_fallback_readback_matches\":true,\"pinned_upload_synchronized_for_smoke\":true,\"owns_pinned_file_staging\":{},\"owns_o_direct_open_and_aligned_read\":{},\"owns_buffered_read_fallback\":{},\"owns_asynchronous_staging_ring\":{},\"owns_cache_budget_policy\":{},\"owns_ds4_kernels\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        model.size(),
        RANGE_OFFSET,
        RANGE_BYTES,
        alignment,
        read_offset,
        read_bytes,
        payload_offset,
        tail_offset,
        TAIL_BYTES,
        M14_1B2B3B1_SCOPE.owns_pinned_file_staging,
        M14_1B2B3B1_SCOPE.owns_o_direct_open_and_aligned_read,
        M14_1B2B3B1_SCOPE.owns_buffered_read_fallback,
        M14_1B2B3B1_SCOPE.owns_asynchronous_staging_ring,
        M14_1B2B3B1_SCOPE.owns_cache_budget_policy,
        M14_1B2B3B1_SCOPE.owns_ds4_kernels,
        M14_1B2B3B1_SCOPE.changes_default_route
    );
    Ok(())
}
