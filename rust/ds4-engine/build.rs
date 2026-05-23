use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").expect("target OS");
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let repo_root = manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("repo root")
        .to_path_buf();
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("out dir"));

    rerun_for_backend_sources(&repo_root);
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");

    let ds4_obj = out_dir.join("ds4_engine_core.o");
    let kvstore_obj = out_dir.join("ds4_engine_kvstore.o");
    let backend_obj = match target_os.as_str() {
        "macos" => {
            compile_c(&repo_root, &out_dir, "ds4.c", &ds4_obj, false);
            compile_c(&repo_root, &out_dir, "ds4_kvstore.c", &kvstore_obj, false);
            let obj = out_dir.join("ds4_engine_metal.o");
            compile_objc(&repo_root, &out_dir, "ds4_metal.m", &obj);
            link_macos(&out_dir, &[&ds4_obj, &kvstore_obj, &obj]);
            return;
        }
        "linux" => {
            let cuda_home = env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".to_owned());
            let nvcc = PathBuf::from(&cuda_home).join("bin/nvcc");
            if nvcc.is_file() {
                compile_c(&repo_root, &out_dir, "ds4.c", &ds4_obj, false);
                compile_c(&repo_root, &out_dir, "ds4_kvstore.c", &kvstore_obj, false);
                let obj = out_dir.join("ds4_engine_cuda.o");
                compile_cuda(&repo_root, &out_dir, &nvcc, "ds4_cuda.cu", &obj);
                link_linux_cuda(&out_dir, &cuda_home, &[&ds4_obj, &kvstore_obj, &obj]);
                return;
            }
            compile_c(&repo_root, &out_dir, "ds4.c", &ds4_obj, true);
            compile_c(&repo_root, &out_dir, "ds4_kvstore.c", &kvstore_obj, true);
            None
        }
        _ => {
            compile_c(&repo_root, &out_dir, "ds4.c", &ds4_obj, true);
            compile_c(&repo_root, &out_dir, "ds4_kvstore.c", &kvstore_obj, true);
            None
        }
    };

    let objects = match backend_obj.as_ref() {
        Some(obj) => vec![&ds4_obj, &kvstore_obj, obj],
        None => vec![&ds4_obj, &kvstore_obj],
    };
    link_cpu(&out_dir, &objects);
}

fn rerun_for_backend_sources(repo_root: &Path) {
    for path in [
        "ds4.c",
        "ds4.h",
        "ds4_gpu.h",
        "ds4_kvstore.c",
        "ds4_kvstore.h",
        "ds4_metal.m",
        "ds4_cuda.cu",
    ] {
        println!("cargo:rerun-if-changed={}", repo_root.join(path).display());
    }
    let metal_dir = repo_root.join("metal");
    if let Ok(entries) = fs::read_dir(&metal_dir) {
        for entry in entries {
            let path = entry.expect("metal source entry").path();
            if path.extension().is_some_and(|ext| ext == "metal") {
                println!("cargo:rerun-if-changed={}", path.display());
            }
        }
    }
    println!(
        "cargo:rerun-if-changed={}",
        repo_root.join("ds4_iq2_tables_cuda.inc").display()
    );
}

fn compile_c(repo_root: &Path, out_dir: &Path, source: &str, object: &Path, no_gpu: bool) {
    let mut command = compiler_command();
    command
        .current_dir(repo_root)
        .arg("-O3")
        .arg("-ffast-math")
        .arg("-g")
        .arg("-Wall")
        .arg("-Wextra")
        .arg("-std=c99")
        .arg("-I.")
        .arg("-D_GNU_SOURCE")
        .arg("-fno-finite-math-only");
    if no_gpu {
        command.arg("-DDS4_NO_GPU");
    }
    command
        .arg("-c")
        .arg(source)
        .arg("-o")
        .arg(object)
        .env("TMPDIR", out_dir);
    run(&mut command);
}

fn compile_objc(repo_root: &Path, out_dir: &Path, source: &str, object: &Path) {
    let mut command = compiler_command();
    command
        .current_dir(repo_root)
        .arg("-O3")
        .arg("-ffast-math")
        .arg("-g")
        .arg("-Wall")
        .arg("-Wextra")
        .arg("-fobjc-arc")
        .arg("-I.")
        .arg("-c")
        .arg(source)
        .arg("-o")
        .arg(object)
        .env("TMPDIR", out_dir);
    run(&mut command);
}

fn compile_cuda(repo_root: &Path, out_dir: &Path, nvcc: &Path, source: &str, object: &Path) {
    let arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "native".to_owned());
    let mut command = Command::new(nvcc);
    command
        .current_dir(repo_root)
        .arg("-O3")
        .arg("-g")
        .arg("-lineinfo")
        .arg("--use_fast_math")
        .arg(format!("-arch={arch}"))
        .arg("-Xcompiler")
        .arg("-march=native")
        .arg("-Xcompiler")
        .arg("-pthread")
        .arg("-c")
        .arg(source)
        .arg("-o")
        .arg(object)
        .env("TMPDIR", out_dir);
    run(&mut command);
}

fn link_macos(out_dir: &Path, objects: &[&PathBuf]) {
    archive(out_dir, objects);
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=ds4_engine");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=m");
    println!("cargo:rustc-link-lib=pthread");
}

fn link_linux_cuda(out_dir: &Path, cuda_home: &str, objects: &[&PathBuf]) {
    archive(out_dir, objects);
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=ds4_engine");
    println!("cargo:rustc-link-search=native={cuda_home}/targets/sbsa-linux/lib");
    println!("cargo:rustc-link-search=native={cuda_home}/lib64");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=m");
    println!("cargo:rustc-link-lib=pthread");
}

fn link_cpu(out_dir: &Path, objects: &[&PathBuf]) {
    archive(out_dir, objects);
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=ds4_engine");
    println!("cargo:rustc-link-lib=m");
    println!("cargo:rustc-link-lib=pthread");
}

fn archive(out_dir: &Path, objects: &[&PathBuf]) {
    let lib_path = out_dir.join("libds4_engine.a");
    let mut command = Command::new("ar");
    command.arg("crs").arg(&lib_path);
    for object in objects {
        command.arg(object);
    }
    run(&mut command);
}

fn compiler_command() -> Command {
    let compiler = env::var("CC").unwrap_or_else(|_| "cc".to_owned());
    match env::var("CARGO_CFG_TARGET_ARCH").as_deref() {
        Ok("aarch64") if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") => {
            let mut command = Command::new("arch");
            command.arg("-arm64").arg(compiler);
            command
        }
        Ok("x86_64") if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") => {
            let mut command = Command::new("arch");
            command.arg("-x86_64").arg(compiler);
            command
        }
        _ => Command::new(compiler),
    }
}

fn run(command: &mut Command) {
    let status = command.status().expect("run engine build command");
    if !status.success() {
        panic!("engine build command failed with {status}: {command:?}");
    }
}
