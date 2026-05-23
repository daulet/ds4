use ds4_engine::{Backend, Engine, EngineOptions};
use ds4_gguf::cli_parse::{parse_cli_config, CliBackend, CliParseResult};
use std::io::{self, Write};
use std::process;

fn main() {
    match run() {
        Ok(code) => process::exit(code),
        Err(err) => {
            eprintln!("{err}");
            process::exit(1);
        }
    }
}

fn run() -> Result<i32, Box<dyn std::error::Error>> {
    let config = match parse_cli_config(std::env::args().skip(1)) {
        Ok(config) => config,
        Err(result) => return Ok(write_parse_result(result)?),
    };
    if !config.inspect {
        eprintln!("ds4-rs: M8.9b inspect implementation reached non-inspect path");
        return Ok(99);
    }

    let mut options = EngineOptions::new(&config.model_path, map_backend(config.backend));
    options.warm_weights = config.warm_weights;
    options.quality = config.quality;
    let engine = Engine::open(&options)?;
    engine.print_summary();
    Ok(0)
}

fn map_backend(backend: CliBackend) -> Backend {
    match backend {
        CliBackend::Metal => Backend::Metal,
        CliBackend::Cuda => Backend::Cuda,
        CliBackend::Cpu => Backend::Cpu,
    }
}

fn write_parse_result(result: CliParseResult) -> io::Result<i32> {
    let mut stdout = io::stdout().lock();
    let mut stderr = io::stderr().lock();
    stdout.write_all(result.stdout.as_bytes())?;
    stderr.write_all(result.stderr.as_bytes())?;
    Ok(result.exit_code)
}
