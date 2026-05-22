use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").expect("target OS");
    if target_os != "macos" {
        return;
    }

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

fn compiler_command() -> Command {
    let compiler = env::var("CC").unwrap_or_else(|_| "cc".to_owned());
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

fn run(command: &mut Command) {
    let status = command.status().expect("run backend build command");
    if !status.success() {
        panic!("backend build command failed with {status}: {command:?}");
    }
}
