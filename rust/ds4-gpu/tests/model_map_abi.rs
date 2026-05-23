#![cfg(all(target_os = "linux", feature = "cuda-backend"))]

use ds4_gpu::decode_backend::{cache_model_range, cache_q8_f16_range};
use ds4_gpu::decode_backend::{set_model_fd, set_model_map, set_model_map_range, ModelMap};
use std::ffi::CStr;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};

#[test]
fn model_map_wrappers_accept_fd_and_mapped_ranges() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    std::env::set_current_dir(manifest_dir.join("../..")).expect("repo root cwd");

    let (path, bytes, file) = tiny_model_file();
    let model = ModelMap::from_bytes(&bytes);

    ds4_gpu::initialize().expect("initialize DS4 GPU backend");
    {
        set_model_fd(file.as_raw_fd()).expect("set model fd");
        set_model_map(model).expect("set model map");
        set_model_map_range(model, 64, 512).expect("set model map range");
        assert!(set_model_map_range(model, bytes.len() as u64 + 1, 1).is_err());

        let label = CStr::from_bytes_with_nul(b"tiny_model\0").expect("label");
        cache_model_range(model, 128, 256, Some(label)).expect("cache model range");
        cache_q8_f16_range(model, 256, 512, 16, 16, Some(label))
            .expect("optional q8/f16 cache range");
        cache_model_range(model, bytes.len() as u64 + 1, 0, None)
            .expect("zero-byte cache range follows C no-op");
        assert!(cache_model_range(model, bytes.len() as u64 + 1, 1, None).is_err());
    }
    unsafe {
        ds4_gpu::cleanup();
    }
    fs::remove_file(path).expect("remove tiny model file");
}

fn tiny_model_file() -> (PathBuf, Vec<u8>, File) {
    let path = std::env::temp_dir().join(format!(
        "ds4-model-map-{}-{}.bin",
        std::process::id(),
        unique_suffix()
    ));
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&path)
        .expect("create tiny model file");
    let mut bytes = vec![0_u8; 4096];
    for (idx, byte) in bytes.iter_mut().enumerate() {
        *byte = (idx as u8).wrapping_mul(17).wrapping_add(3);
    }
    file.write_all(&bytes).expect("write tiny model file");
    file.flush().expect("flush tiny model file");
    file.seek(SeekFrom::Start(0))
        .expect("rewind tiny model file");
    let mut readback = Vec::new();
    file.read_to_end(&mut readback)
        .expect("read tiny model file");
    assert_eq!(readback, bytes);
    (path, bytes, file)
}

fn unique_suffix() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time before epoch")
        .as_nanos()
}
