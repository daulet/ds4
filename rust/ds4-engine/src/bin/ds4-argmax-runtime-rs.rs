use ds4_engine::{
    context_memory_estimate, ArgmaxOptions, Backend, Engine, EngineOptions, ThinkMode,
};
use std::error::Error;
use std::fs;
use std::io::{self, Write};
use std::process;

const HELP: &str = "\
Usage: ds4-argmax-runtime-rs [(-p PROMPT | --prompt-file FILE)] [options]\n\
\n\
M8.13a runtime boundary for greedy one-shot generation.\n\
\n\
Options:\n\
  -m, --model FILE\n\
  --backend NAME | --cuda | --metal | --cpu\n\
  -c, --ctx N\n\
  -n, --tokens N\n\
  --temp 0\n\
  --think | --think-max | --nothink\n\
  -sys, --system TEXT\n";

#[derive(Debug)]
struct RuntimeConfig {
    model_path: String,
    backend: Backend,
    prompt: String,
    system: String,
    n_predict: i32,
    ctx_size: i32,
    think_mode: ThinkMode,
    warm_weights: bool,
    quality: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            model_path: "ds4flash.gguf".to_string(),
            backend: Backend::default_backend(),
            prompt: String::new(),
            system: "You are a helpful assistant".to_string(),
            n_predict: 50000,
            ctx_size: 32768,
            think_mode: ThinkMode::default_mode(),
            warm_weights: false,
            quality: false,
        }
    }
}

#[derive(Debug)]
struct RuntimeExit {
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
    log_context_memory(config.backend, config.ctx_size);

    let effective_think = config.think_mode.for_context(config.ctx_size);
    if config.think_mode == ThinkMode::Max && effective_think != ThinkMode::Max {
        eprintln!(
            "ds4: warning: --think-max needs --ctx >= {}; ctx={} uses normal thinking instead",
            ThinkMode::max_min_context(),
            config.ctx_size
        );
    }

    let mut engine_options = EngineOptions::new(&config.model_path, config.backend);
    engine_options.warm_weights = config.warm_weights;
    engine_options.quality = config.quality;
    let engine = Engine::open(&engine_options)?;
    let prompt = engine.encode_chat_prompt(&config.system, &config.prompt, effective_think)?;
    let generation = engine.generate_argmax_text(
        &prompt,
        ArgmaxOptions {
            n_predict: config.n_predict,
            ctx_size: config.ctx_size,
            think_mode: effective_think,
        },
    );
    io::stdout().lock().write_all(&generation.stdout)?;
    Ok(generation.exit_code)
}

fn parse_args<I, S>(args: I) -> Result<RuntimeConfig, RuntimeExit>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let argv: Vec<String> = args.into_iter().map(Into::into).collect();
    let mut config = RuntimeConfig::default();
    let mut prompt_seen = false;
    let mut i = 0usize;
    while i < argv.len() {
        let arg = argv[i].as_str();
        match arg {
            "-h" | "--help" => return Err(exit(0, HELP, "")),
            "-p" | "--prompt" => {
                let value = need_arg(&argv, &mut i, arg)?;
                if prompt_seen {
                    return Err(exit(2, "", "ds4: specify only one prompt source\n"));
                }
                config.prompt = value.to_string();
                prompt_seen = true;
            }
            "--prompt-file" => {
                let value = need_arg(&argv, &mut i, arg)?;
                if prompt_seen {
                    return Err(exit(2, "", "ds4: specify only one prompt source\n"));
                }
                config.prompt = read_prompt_file(value)?;
                prompt_seen = true;
            }
            "-m" | "--model" => {
                config.model_path = need_arg(&argv, &mut i, arg)?.to_string();
            }
            "-sys" | "--system" => {
                config.system = need_arg(&argv, &mut i, arg)?.to_string();
            }
            "-c" | "--ctx" => {
                config.ctx_size = parse_positive_i32(need_arg(&argv, &mut i, arg)?, arg)?;
            }
            "-n" | "--tokens" => {
                config.n_predict = parse_positive_i32(need_arg(&argv, &mut i, arg)?, arg)?;
            }
            "--temp" => {
                let value = parse_float_range(need_arg(&argv, &mut i, arg)?, arg, 0.0, 100.0)?;
                if value != 0.0 {
                    return Err(exit(
                        99,
                        "",
                        "ds4-rs: M8.13a argmax runtime supports only --temp 0\n",
                    ));
                }
            }
            "--backend" => {
                let value = need_arg(&argv, &mut i, arg)?;
                let Some(backend) = Backend::parse(value) else {
                    return Err(exit(
                        2,
                        "",
                        &format!(
                            "ds4: invalid backend: {value}\n\
                             ds4: valid backends are: metal, cuda, cpu\n"
                        ),
                    ));
                };
                config.backend = backend;
            }
            "--cuda" => config.backend = Backend::Cuda,
            "--metal" => config.backend = Backend::Metal,
            "--cpu" => config.backend = Backend::Cpu,
            "--think" => config.think_mode = ThinkMode::High,
            "--think-max" => config.think_mode = ThinkMode::Max,
            "--nothink" => config.think_mode = ThinkMode::None,
            "--warm-weights" => config.warm_weights = true,
            "--quality" => config.quality = true,
            _ => return Err(exit(2, "", &format!("ds4: unknown option: {arg}\n{HELP}"))),
        }
        i += 1;
    }
    if !prompt_seen {
        return Err(exit(
            2,
            "",
            "ds4-rs: M8.13a argmax runtime requires -p or --prompt-file\n",
        ));
    }
    Ok(config)
}

fn need_arg<'a>(argv: &'a [String], idx: &mut usize, opt: &str) -> Result<&'a str, RuntimeExit> {
    if *idx + 1 >= argv.len() {
        return Err(exit(2, "", &format!("ds4: missing value for {opt}\n")));
    }
    *idx += 1;
    Ok(argv[*idx].as_str())
}

fn read_prompt_file(path: &str) -> Result<String, RuntimeExit> {
    fs::read_to_string(path)
        .map_err(|_| exit(2, "", &format!("ds4: failed to read prompt file: {path}\n")))
}

fn parse_positive_i32(value: &str, opt: &str) -> Result<i32, RuntimeExit> {
    match value.parse::<i64>() {
        Ok(v) if (1..=i32::MAX as i64).contains(&v) => Ok(v as i32),
        _ => Err(exit(
            2,
            "",
            &format!("ds4: invalid value for {opt}: {value}\n"),
        )),
    }
}

fn parse_float_range(value: &str, opt: &str, min: f32, max: f32) -> Result<f32, RuntimeExit> {
    match value.parse::<f32>() {
        Ok(v) if v.is_finite() && v >= min && v <= max => Ok(v),
        _ => Err(exit(
            2,
            "",
            &format!("ds4: invalid value for {opt}: {value}\n"),
        )),
    }
}

fn log_context_memory(backend: Backend, ctx_size: i32) {
    let memory = context_memory_estimate(backend, ctx_size);
    eprintln!(
        "ds4: context buffers {:.2} MiB (ctx={}, backend={}, prefill_chunk={}, raw_kv_rows={}, compressed_kv_rows={})",
        memory.total_bytes as f64 / (1024.0 * 1024.0),
        ctx_size,
        backend.name(),
        memory.prefill_cap,
        memory.raw_cap,
        memory.comp_cap
    );
}

fn write_exit(exit: RuntimeExit) -> io::Result<i32> {
    let mut stdout = io::stdout().lock();
    let mut stderr = io::stderr().lock();
    stdout.write_all(exit.stdout.as_bytes())?;
    stderr.write_all(exit.stderr.as_bytes())?;
    Ok(exit.code)
}

fn exit(code: i32, stdout: &str, stderr: &str) -> RuntimeExit {
    RuntimeExit {
        code,
        stdout: stdout.to_string(),
        stderr: stderr.to_string(),
    }
}
