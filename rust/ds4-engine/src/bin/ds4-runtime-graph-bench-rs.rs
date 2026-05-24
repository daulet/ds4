use ds4_engine::{
    context_memory_estimate, Backend, Engine, EngineOptions, RuntimeGraphRoute, SessionSnapshot,
    RUNTIME_GRAPH_ROUTE_UNSUPPORTED_CODE, RUNTIME_GRAPH_ROUTE_VALID_VALUES,
};
use std::error::Error;
use std::fs::{self, File};
use std::io::{self, Write};
use std::process;
use std::time::Instant;

const HELP: &str = "\
Usage: ds4-runtime-graph-bench-rs --prompt-file FILE [options]\n\
\n\
M10.9f Rust graph-runtime benchmark capture.\n\
\n\
Options:\n\
  -m, --model FILE\n\
  --prompt-file FILE\n\
  --backend NAME | --cuda | --metal | --cpu\n\
  --runtime-graph ROUTE\n\
  -t, --threads N\n\
  --ctx-start N\n\
  --ctx-max N\n\
  --ctx-alloc N\n\
  --step-incr N\n\
  --gen-tokens N | --tokens N | -n N\n\
  --quality\n\
  --warm-weights\n\
  --csv FILE\n\
  -h, --help\n";

#[derive(Debug)]
struct Config {
    model_path: String,
    prompt_path: String,
    csv_path: Option<String>,
    backend: Backend,
    route: RuntimeGraphRoute,
    threads: i32,
    ctx_start: i32,
    ctx_max: i32,
    ctx_alloc: i32,
    step_incr: i32,
    gen_tokens: i32,
    warm_weights: bool,
    quality: bool,
}

impl Default for Config {
    fn default() -> Self {
        let ctx_max = 32768;
        let gen_tokens = 128;
        Self {
            model_path: "ds4flash.gguf".to_string(),
            prompt_path: String::new(),
            csv_path: None,
            backend: Backend::Cuda,
            route: RuntimeGraphRoute::Graph,
            threads: 0,
            ctx_start: 2048,
            ctx_max,
            ctx_alloc: ctx_max + gen_tokens + 1,
            step_incr: 2048,
            gen_tokens,
            warm_weights: false,
            quality: false,
        }
    }
}

#[derive(Debug)]
struct CliExit {
    code: i32,
    stdout: String,
    stderr: String,
}

fn main() {
    match run() {
        Ok(code) => process::exit(code),
        Err(err) => {
            eprintln!("{err}");
            process::exit(1);
        }
    }
}

fn run() -> Result<i32, Box<dyn Error>> {
    let config = match parse_args(std::env::args().skip(1)) {
        Ok(config) => config,
        Err(exit) => return Ok(write_exit(exit)?),
    };
    if config.route == RuntimeGraphRoute::Graph && config.backend == Backend::Cpu {
        return Ok(write_exit(CliExit {
            code: RUNTIME_GRAPH_ROUTE_UNSUPPORTED_CODE,
            stdout: String::new(),
            stderr:
                "ds4-runtime-graph-bench-rs: --runtime-graph graph requires cuda or metal backend\n"
                    .to_string(),
        })?);
    }

    let prompt_text = fs::read_to_string(&config.prompt_path)?;
    log_context_memory(config.backend, config.ctx_alloc);
    let mut options = EngineOptions::new(&config.model_path, config.backend);
    options.n_threads = config.threads;
    options.warm_weights = config.warm_weights;
    options.quality = config.quality;
    let engine = Engine::open(&options)?;
    let prompt = engine.tokenize_text(&prompt_text)?;
    if prompt.len() < config.ctx_max {
        eprintln!(
            "ds4-runtime-graph-bench-rs: prompt has {} tokens, need at least --ctx-max={}",
            prompt.len(),
            config.ctx_max
        );
        return Ok(1);
    }

    let mut session = engine.create_server_session(config.ctx_alloc)?;
    let mut snapshot = SessionSnapshot::new();
    let mut out: Box<dyn Write> = match config.csv_path.as_deref() {
        Some(path) => Box::new(File::create(path)?),
        None => Box::new(io::stdout().lock()),
    };
    writeln!(
        out,
        "ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,kvcache_bytes"
    )?;

    let mut previous = 0;
    let mut frontier = config.ctx_start;
    loop {
        let prefill_start = Instant::now();
        if let Err(err) = session.sync_prompt_prefix(&prompt, frontier) {
            eprintln!("ds4-runtime-graph-bench-rs: prefill to {frontier} failed: {err}");
            return Ok(1);
        }
        let prefill_elapsed = prefill_start.elapsed();
        let prefill_tokens = frontier - previous;

        if let Err(err) = session.save_snapshot(&mut snapshot) {
            eprintln!("ds4-runtime-graph-bench-rs: snapshot at {frontier} failed: {err}");
            return Ok(1);
        }

        let decode_start = Instant::now();
        for _ in 0..config.gen_tokens {
            if session.position() + 1 >= session.context_size() {
                eprintln!(
                    "ds4-runtime-graph-bench-rs: generation would exceed allocated context at frontier {frontier}"
                );
                return Ok(1);
            }
            let token = session.argmax_excluding_eos();
            if token < 0 {
                eprintln!(
                    "ds4-runtime-graph-bench-rs: failed to choose non-EOS token at frontier {frontier}"
                );
                return Ok(1);
            }
            if let Err(err) = session.eval_token(token) {
                eprintln!(
                    "ds4-runtime-graph-bench-rs: decode at frontier {frontier} failed: {err}"
                );
                return Ok(1);
            }
        }
        let decode_elapsed = decode_start.elapsed();

        if let Err(err) = session.load_snapshot(&snapshot) {
            eprintln!("ds4-runtime-graph-bench-rs: restore at {frontier} failed: {err}");
            return Ok(1);
        }

        writeln!(
            out,
            "{},{},{:.2},{},{:.2},{}",
            frontier,
            prefill_tokens,
            rate(prefill_tokens, prefill_elapsed),
            config.gen_tokens,
            rate(config.gen_tokens, decode_elapsed),
            snapshot.len()
        )?;
        out.flush()?;

        previous = frontier;
        if frontier >= config.ctx_max {
            break;
        }
        frontier = next_frontier(&config, frontier);
    }

    Ok(0)
}

fn parse_args<I, S>(args: I) -> Result<Config, CliExit>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let argv: Vec<String> = args.into_iter().map(Into::into).collect();
    let mut config = Config::default();
    let mut ctx_alloc_was_set = false;
    let mut i = 0usize;
    while i < argv.len() {
        let arg = argv[i].as_str();
        match arg {
            "-h" | "--help" => return Err(exit(0, HELP, "")),
            "-m" | "--model" => config.model_path = need_arg(&argv, &mut i, arg)?.to_string(),
            "--prompt-file" => config.prompt_path = need_arg(&argv, &mut i, arg)?.to_string(),
            "--backend" => {
                let value = need_arg(&argv, &mut i, arg)?;
                config.backend = Backend::parse(value).ok_or_else(|| {
                    exit(
                        2,
                        "",
                        &format!(
                            "ds4-runtime-graph-bench-rs: invalid backend: {value}\n\
                             ds4-runtime-graph-bench-rs: valid backends are: metal, cuda, cpu\n"
                        ),
                    )
                })?;
            }
            "--runtime-graph" | "--runtime-graph-route" => {
                let value = need_arg(&argv, &mut i, arg)?;
                config.route = RuntimeGraphRoute::parse(value).ok_or_else(|| {
                    exit(
                        2,
                        "",
                        &format!(
                            "ds4-runtime-graph-bench-rs: invalid runtime graph route: {value}\n\
                             ds4-runtime-graph-bench-rs: valid runtime graph routes are: {RUNTIME_GRAPH_ROUTE_VALID_VALUES}\n"
                        ),
                    )
                })?;
            }
            "-t" | "--threads" => {
                config.threads = parse_nonnegative_i32(need_arg(&argv, &mut i, arg)?, arg)?
            }
            "--ctx-start" => {
                config.ctx_start = parse_positive_i32(need_arg(&argv, &mut i, arg)?, arg)?
            }
            "--ctx-max" => config.ctx_max = parse_positive_i32(need_arg(&argv, &mut i, arg)?, arg)?,
            "--ctx-alloc" => {
                config.ctx_alloc = parse_positive_i32(need_arg(&argv, &mut i, arg)?, arg)?;
                ctx_alloc_was_set = true;
            }
            "--step-incr" => {
                config.step_incr = parse_positive_i32(need_arg(&argv, &mut i, arg)?, arg)?
            }
            "--gen-tokens" | "--tokens" | "-n" => {
                config.gen_tokens = parse_positive_i32(need_arg(&argv, &mut i, arg)?, arg)?
            }
            "--csv" => config.csv_path = Some(need_arg(&argv, &mut i, arg)?.to_string()),
            "--quality" => config.quality = true,
            "--warm-weights" => config.warm_weights = true,
            "--cuda" => config.backend = Backend::Cuda,
            "--metal" => config.backend = Backend::Metal,
            "--cpu" => config.backend = Backend::Cpu,
            _ => {
                return Err(exit(
                    2,
                    "",
                    &format!("ds4-runtime-graph-bench-rs: unknown option: {arg}\n{HELP}"),
                ))
            }
        }
        i += 1;
    }
    validate_config(&mut config, ctx_alloc_was_set)?;
    Ok(config)
}

fn validate_config(config: &mut Config, ctx_alloc_was_set: bool) -> Result<(), CliExit> {
    if config.prompt_path.is_empty() {
        return Err(exit(
            2,
            "",
            "ds4-runtime-graph-bench-rs: specify --prompt-file\n",
        ));
    }
    if config.ctx_start > config.ctx_max {
        return Err(exit(
            2,
            "",
            "ds4-runtime-graph-bench-rs: --ctx-start must be <= --ctx-max\n",
        ));
    }
    if !ctx_alloc_was_set {
        config.ctx_alloc = config.ctx_max + config.gen_tokens + 1;
    }
    if config.ctx_alloc <= config.ctx_max + config.gen_tokens {
        return Err(exit(
            2,
            "",
            "ds4-runtime-graph-bench-rs: --ctx-alloc must be greater than ctx-max + gen-tokens\n",
        ));
    }
    Ok(())
}

fn need_arg<'a>(argv: &'a [String], idx: &mut usize, opt: &str) -> Result<&'a str, CliExit> {
    if *idx + 1 >= argv.len() {
        return Err(exit(
            2,
            "",
            &format!("ds4-runtime-graph-bench-rs: missing value for {opt}\n"),
        ));
    }
    *idx += 1;
    Ok(argv[*idx].as_str())
}

fn parse_positive_i32(value: &str, opt: &str) -> Result<i32, CliExit> {
    match value.parse::<i64>() {
        Ok(v) if (1..=i32::MAX as i64).contains(&v) => Ok(v as i32),
        _ => Err(exit(
            2,
            "",
            &format!("ds4-runtime-graph-bench-rs: invalid value for {opt}: {value}\n"),
        )),
    }
}

fn parse_nonnegative_i32(value: &str, opt: &str) -> Result<i32, CliExit> {
    match value.parse::<i64>() {
        Ok(v) if (0..=i32::MAX as i64).contains(&v) => Ok(v as i32),
        _ => Err(exit(
            2,
            "",
            &format!("ds4-runtime-graph-bench-rs: invalid value for {opt}: {value}\n"),
        )),
    }
}

fn next_frontier(config: &Config, current: i32) -> i32 {
    if current >= config.ctx_max {
        return config.ctx_max;
    }
    (current.saturating_add(config.step_incr)).min(config.ctx_max)
}

fn rate(tokens: i32, elapsed: std::time::Duration) -> f64 {
    let seconds = elapsed.as_secs_f64();
    if seconds > 0.0 {
        tokens as f64 / seconds
    } else {
        0.0
    }
}

fn log_context_memory(backend: Backend, ctx_size: i32) {
    let memory = context_memory_estimate(backend, ctx_size);
    eprintln!(
        "ds4-runtime-graph-bench-rs: context buffers {:.2} MiB (ctx={}, backend={}, prefill_chunk={}, raw_kv_rows={}, compressed_kv_rows={})",
        memory.total_bytes as f64 / (1024.0 * 1024.0),
        ctx_size,
        backend.name(),
        memory.prefill_cap,
        memory.raw_cap,
        memory.comp_cap
    );
}

fn write_exit(exit: CliExit) -> io::Result<i32> {
    let mut stdout = io::stdout().lock();
    let mut stderr = io::stderr().lock();
    stdout.write_all(exit.stdout.as_bytes())?;
    stderr.write_all(exit.stderr.as_bytes())?;
    Ok(exit.code)
}

fn exit(code: i32, stdout: &str, stderr: &str) -> CliExit {
    CliExit {
        code,
        stdout: stdout.to_string(),
        stderr: stderr.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_b300_benchmark_shape() {
        let config = parse_args([
            "--prompt-file",
            "speed-bench/promessi_sposi.txt",
            "-m",
            "/workspace/ds4/ds4flash.gguf",
            "--cuda",
            "--runtime-graph",
            "graph",
            "--ctx-start",
            "2048",
            "--ctx-max",
            "8192",
            "--step-incr",
            "2048",
            "--gen-tokens",
            "32",
        ])
        .expect("config");
        assert_eq!(config.ctx_alloc, 8192 + 32 + 1);
        assert_eq!(config.backend, Backend::Cuda);
        assert_eq!(config.route, RuntimeGraphRoute::Graph);
        assert_eq!(next_frontier(&config, 2048), 4096);
        assert_eq!(next_frontier(&config, 8192), 8192);
    }

    #[test]
    fn rejects_alloc_without_decode_room() {
        let err = parse_args([
            "--prompt-file",
            "prompt.txt",
            "--ctx-max",
            "8192",
            "--ctx-alloc",
            "8192",
            "--gen-tokens",
            "32",
        ])
        .expect_err("invalid");
        assert_eq!(err.code, 2);
        assert!(err.stderr.contains("--ctx-alloc"));
    }
}
