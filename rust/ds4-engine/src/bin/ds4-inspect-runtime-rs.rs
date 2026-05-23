use ds4_engine::{Backend, Engine, EngineOptions};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => {}
        Err(message) => {
            eprint!("{message}");
            std::process::exit(2);
        }
    }
}

fn run(args: &[String]) -> Result<(), String> {
    let mut model_path: Option<String> = None;
    let mut backend: Option<Backend> = None;
    let mut warm_weights = false;
    let mut quality = false;
    let mut i = 0usize;
    while i < args.len() {
        let arg = args[i].as_str();
        match arg {
            "-h" | "--help" => {
                print_usage();
                return Ok(());
            }
            "-m" | "--model" => {
                i += 1;
                model_path = Some(
                    args.get(i)
                        .ok_or_else(|| {
                            format!("ds4-inspect-runtime-rs: {arg} requires an argument\n")
                        })?
                        .clone(),
                );
            }
            "--backend" => {
                i += 1;
                let value = args.get(i).ok_or_else(|| {
                    "ds4-inspect-runtime-rs: --backend requires an argument\n".to_string()
                })?;
                backend = Backend::parse(value);
                if backend.is_none() {
                    return Err(format!(
                        "ds4-inspect-runtime-rs: invalid backend: {value}\n\
                         ds4-inspect-runtime-rs: valid backends are: metal, cuda, cpu\n"
                    ));
                }
            }
            "--metal" => backend = Some(Backend::Metal),
            "--cuda" => backend = Some(Backend::Cuda),
            "--cpu" => backend = Some(Backend::Cpu),
            "--warm-weights" => warm_weights = true,
            "--quality" => quality = true,
            "--inspect" => {}
            _ => return Err(format!("ds4-inspect-runtime-rs: unknown option: {arg}\n")),
        }
        i += 1;
    }

    let model_path =
        model_path.ok_or_else(|| "ds4-inspect-runtime-rs: --model is required\n".to_string())?;
    let backend = backend.ok_or_else(|| {
        "ds4-inspect-runtime-rs: --backend/--cuda/--metal/--cpu is required\n".to_string()
    })?;
    let mut options = EngineOptions::new(&model_path, backend);
    options.warm_weights = warm_weights;
    options.quality = quality;
    let engine =
        Engine::open(&options).map_err(|error| format!("ds4-inspect-runtime-rs: {error}\n"))?;
    engine.print_summary();
    Ok(())
}

fn print_usage() {
    print!(
        "Usage: ds4-inspect-runtime-rs --model FILE (--cuda | --metal | --cpu | --backend NAME) [--inspect]\n"
    );
}
