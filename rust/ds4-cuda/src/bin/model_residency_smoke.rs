use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

use ds4_cuda::{substrate::CudaOxideSubstrate, M14_1B1_SCOPE};

const MODEL_WINDOW_BYTES: usize = 4096;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: ds4-cuda-model-residency-smoke MODEL.gguf")?;
    let model_size = std::fs::metadata(&model_path)?.len();
    let mut model_window = vec![0_u8; MODEL_WINDOW_BYTES];
    File::open(&model_path)?.read_exact(&mut model_window)?;

    let substrate = CudaOxideSubstrate::open(0)?;
    let managed = substrate.managed_from_slice(&model_window)?;
    substrate.prefetch_read_mostly_to_device(&managed)?;
    substrate.return_managed_to_host(&managed)?;
    assert_eq!(managed.as_slice(), model_window);

    let mapped = substrate.mapped_from_slice(&model_window)?;
    assert_eq!(mapped.as_slice(), model_window);
    assert_ne!(mapped.cu_deviceptr(), 0);

    let mut registered_backing = model_window.clone();
    let registered = substrate.register_host_range(&mut registered_backing)?;
    assert_eq!(registered.as_slice(), model_window);
    assert_ne!(registered.cu_deviceptr(), 0);
    drop(registered);

    let device_name = substrate.device_name()?;
    println!(
        "{{\"milestone\":\"M14.1b1\",\"device_name\":{:?},\"model_size\":{},\"model_window_bytes\":{},\"managed_advice_prefetch\":true,\"mapped_device_pointer\":true,\"registered_host_pointer\":true,\"owns_complete_model_map\":{},\"owns_ds4_kernels\":{},\"changes_default_route\":{}}}",
        device_name,
        model_size,
        MODEL_WINDOW_BYTES,
        M14_1B1_SCOPE.owns_complete_model_map,
        M14_1B1_SCOPE.owns_ds4_kernels,
        M14_1B1_SCOPE.changes_default_route
    );
    Ok(())
}
