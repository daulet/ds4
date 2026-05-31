use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").expect("target OS");
    match target_os.as_str() {
        "macos" => build_macos_backend(),
        "linux" if env::var_os("CARGO_FEATURE_CUDA_RUST_BACKEND").is_some() => {
            build_linux_rust_cuda_backend()
        }
        "linux" if env::var_os("CARGO_FEATURE_CUDA_BACKEND").is_some() => {
            build_linux_cuda_backend()
        }
        _ => {}
    }
}

fn build_macos_backend() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let repo_root = manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("repo root")
        .to_path_buf();
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("out dir"));

    rerun_for_backend_sources(&repo_root);

    let ds4_obj = out_dir.join("ds4.o");
    compile_c(&repo_root, &out_dir, "ds4.c", &ds4_obj);

    let metal_obj = out_dir.join("ds4_metal.o");
    compile_objc(&repo_root, &out_dir, "ds4_metal.m", &metal_obj);

    let lib_path = out_dir.join("libds4_backend.a");
    run(Command::new("ar")
        .arg("crs")
        .arg(&lib_path)
        .arg(&ds4_obj)
        .arg(&metal_obj));

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=ds4_backend");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=m");
    println!("cargo:rustc-link-lib=pthread");
}

fn build_linux_cuda_backend() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let repo_root = manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("repo root")
        .to_path_buf();
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("out dir"));

    rerun_for_backend_sources(&repo_root);
    println!(
        "cargo:rerun-if-changed={}",
        repo_root.join("ds4_cuda.cu").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        repo_root.join("ds4_iq2_tables_cuda.inc").display()
    );
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=NVCC");

    let ds4_obj = out_dir.join("ds4.o");
    compile_c_linux(&repo_root, &out_dir, "ds4.c", &ds4_obj);

    let cuda_obj = out_dir.join("ds4_cuda.o");
    compile_cuda(&repo_root, &out_dir, &cuda_obj);

    let lib_path = out_dir.join("libds4_backend.a");
    run(Command::new("ar")
        .arg("crs")
        .arg(&lib_path)
        .arg(&ds4_obj)
        .arg(&cuda_obj));

    let cuda_home = cuda_home();
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!(
        "cargo:rustc-link-search=native={}/targets/sbsa-linux/lib",
        cuda_home.display()
    );
    println!(
        "cargo:rustc-link-search=native={}/lib64",
        cuda_home.display()
    );
    println!("cargo:rustc-link-lib=static=ds4_backend");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=m");
    println!("cargo:rustc-link-lib=pthread");
    println!("cargo:rustc-link-lib=dl");
    println!("cargo:rustc-link-lib=rt");
}

fn build_linux_rust_cuda_backend() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let repo_root = manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("repo root")
        .to_path_buf();
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("out dir"));
    let rust_dylib = rust_cuda_dylib();

    rerun_for_backend_sources(&repo_root);
    println!("cargo:rerun-if-env-changed=DS4_CUDA_RUST_DYLIB");
    println!("cargo:rerun-if-changed={}", rust_dylib.display());

    let ds4_obj = out_dir.join("ds4.o");
    compile_c_linux(&repo_root, &out_dir, "ds4.c", &ds4_obj);

    let lib_path = out_dir.join("libds4_backend.a");
    run(Command::new("ar").arg("crs").arg(&lib_path).arg(&ds4_obj));

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=ds4_backend");
    println!(
        "cargo:rustc-link-search=native={}",
        rust_dylib
            .parent()
            .expect("Rust CUDA dynamic library directory")
            .display()
    );
    println!("cargo:rustc-link-lib=dylib=ds4_cuda");
    println!(
        "cargo:rustc-link-arg=-Wl,-rpath,{}",
        rust_dylib
            .parent()
            .expect("Rust CUDA dynamic library directory")
            .display()
    );
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=m");
    println!("cargo:rustc-link-lib=pthread");
    println!("cargo:rustc-link-lib=dl");
    println!("cargo:rustc-link-lib=rt");
    println!("cargo:rustc-link-lib=util");
}

fn rerun_for_backend_sources(repo_root: &Path) {
    for path in ["ds4.c", "ds4.h", "ds4_gpu.h", "ds4_metal.m"] {
        println!("cargo:rerun-if-changed={}", repo_root.join(path).display());
    }

    let metal_dir = repo_root.join("metal");
    let entries = fs::read_dir(&metal_dir).expect("read metal source dir");
    for entry in entries {
        let path = entry.expect("metal source entry").path();
        if path.extension().is_some_and(|ext| ext == "metal") {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}

fn compile_c(repo_root: &Path, out_dir: &Path, source: &str, object: &Path) {
    let mut command = common_c_command(repo_root, out_dir, source, object);
    run(&mut command);
}

fn compile_c_linux(repo_root: &Path, out_dir: &Path, source: &str, object: &Path) {
    let mut command = common_c_command(repo_root, out_dir, source, object);
    command
        .arg("-D_GNU_SOURCE")
        .arg("-fno-finite-math-only")
        .arg("-march=native");
    run(&mut command);
}

fn common_c_command(repo_root: &Path, out_dir: &Path, source: &str, object: &Path) -> Command {
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
        .arg("-c")
        .arg(source)
        .arg("-o")
        .arg(object)
        .env("TMPDIR", out_dir);
    command
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

fn compile_cuda(repo_root: &Path, out_dir: &Path, object: &Path) {
    let mut command = Command::new(nvcc_path());
    command
        .current_dir(repo_root)
        .arg("-O3")
        .arg("-g")
        .arg("-lineinfo")
        .arg("--use_fast_math")
        .arg("-Xcompiler")
        .arg("-march=native")
        .arg("-Xcompiler")
        .arg("-pthread");

    if let Some(arch) = cuda_arch() {
        command.arg(format!("-arch={arch}"));
    }

    command
        .arg("-c")
        .arg("ds4_cuda.cu")
        .arg("-o")
        .arg(object)
        .env("TMPDIR", out_dir);
    run(&mut command);
}

fn compiler_command() -> Command {
    let compiler = env::var("CC").unwrap_or_else(|_| "cc".to_owned());
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return Command::new(compiler);
    }
    match env::var("CARGO_CFG_TARGET_ARCH").as_deref() {
        Ok("aarch64") => {
            let mut command = Command::new("arch");
            command.arg("-arm64").arg(compiler);
            command
        }
        Ok("x86_64") => {
            let mut command = Command::new("arch");
            command.arg("-x86_64").arg(compiler);
            command
        }
        _ => Command::new(compiler),
    }
}

fn cuda_home() -> PathBuf {
    env::var_os("CUDA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/usr/local/cuda"))
}

fn nvcc_path() -> PathBuf {
    env::var_os("NVCC")
        .map(PathBuf::from)
        .unwrap_or_else(|| cuda_home().join("bin/nvcc"))
}

fn rust_cuda_dylib() -> PathBuf {
    let path = env::var_os("DS4_CUDA_RUST_DYLIB")
        .map(PathBuf::from)
        .expect("DS4_CUDA_RUST_DYLIB must point to the prebuilt Rust CUDA dynamic library");
    assert!(
        path.is_file(),
        "DS4_CUDA_RUST_DYLIB is not a file: {}",
        path.display()
    );
    assert_eq!(
        path.file_name().and_then(|name| name.to_str()),
        Some("libds4_cuda.so"),
        "DS4_CUDA_RUST_DYLIB must name libds4_cuda.so"
    );
    path.canonicalize()
        .expect("canonicalize Rust CUDA dynamic library")
}

fn cuda_arch() -> Option<String> {
    match env::var("CUDA_ARCH") {
        Ok(value) if !value.trim().is_empty() => Some(value),
        Ok(_) => None,
        Err(_) => Some("native".to_owned()),
    }
}

fn run(command: &mut Command) {
    let status = command.status().expect("run backend build command");
    if !status.success() {
        panic!("backend build command failed with {status}: {command:?}");
    }
}
