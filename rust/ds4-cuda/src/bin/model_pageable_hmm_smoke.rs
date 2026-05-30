use std::path::PathBuf;

use ds4_cuda::model_map::{prefetch_pageable_read_only_range, MappedModelFile};
use ds4_cuda::{substrate::CudaOxideSubstrate, M14_1B2B3A_SCOPE};

const RANGE_OFFSET: u64 = 13;
const RANGE_BYTES: u64 = 4096;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: ds4-cuda-model-pageable-hmm-smoke MODEL.gguf")?;
    let model = MappedModelFile::open(&model_path)?;
    let expected = model.range(RANGE_OFFSET, RANGE_BYTES)?.to_vec();
    let layout = model.registered_range_layout(RANGE_OFFSET, RANGE_BYTES)?;
    assert_eq!(layout.registered_offset % layout.page_size, 0);
    assert_eq!(layout.registered_bytes % layout.page_size, 0);
    assert_eq!(layout.device_offset, RANGE_OFFSET);
    assert!(layout.registered_bytes > RANGE_BYTES);

    let substrate = CudaOxideSubstrate::open(0)?;
    let pageable_memory_access = substrate.pageable_memory_access()?;
    let pageable_memory_access_uses_host_page_tables =
        substrate.pageable_memory_access_uses_host_page_tables()?;
    assert!(pageable_memory_access);
    assert!(!pageable_memory_access_uses_host_page_tables);

    let prefetched =
        prefetch_pageable_read_only_range(&substrate, &model, RANGE_OFFSET, RANGE_BYTES)?;
    assert_eq!(prefetched.layout(), layout);
    assert_eq!(prefetched.readback(&substrate)?, expected);

    println!(
        "{{\"milestone\":\"M14.1b2b3a\",\"device_name\":{:?},\"model_size\":{},\"range_offset\":{},\"range_bytes\":{},\"pageable_memory_access\":{},\"pageable_memory_access_uses_host_page_tables\":{},\"prefetch_page_size\":{},\"prefetch_offset\":{},\"prefetch_bytes\":{},\"prefetch_device_offset\":{},\"read_mostly_advice\":true,\"preferred_device_advice\":true,\"pageable_prefetch\":true,\"prefetch_synchronized_for_smoke\":true,\"hmm_direct_readback_matches\":true,\"owns_page_aligned_pageable_hmm_prefetch\":{},\"owns_hmm_direct_read_pointer\":{},\"owns_o_direct_staging\":{},\"owns_ds4_kernels\":{},\"changes_default_route\":{}}}",
        substrate.device_name()?,
        model.size(),
        RANGE_OFFSET,
        RANGE_BYTES,
        pageable_memory_access,
        pageable_memory_access_uses_host_page_tables,
        layout.page_size,
        layout.registered_offset,
        layout.registered_bytes,
        layout.device_offset,
        M14_1B2B3A_SCOPE.owns_page_aligned_pageable_hmm_prefetch,
        M14_1B2B3A_SCOPE.owns_hmm_direct_read_pointer,
        M14_1B2B3A_SCOPE.owns_o_direct_staging,
        M14_1B2B3A_SCOPE.owns_ds4_kernels,
        M14_1B2B3A_SCOPE.changes_default_route
    );
    Ok(())
}
