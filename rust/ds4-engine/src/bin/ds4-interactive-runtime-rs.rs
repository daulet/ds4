use ds4_engine::{
    context_memory_estimate, Backend, Engine, EngineOptions, InteractiveTurnOptions, ThinkMode,
};
use std::fs;
use std::io::{self, Write};
use std::process;

#[derive(Debug)]
struct Config {
    model_path: String,
    backend: Backend,
    ctx_size: i32,
    n_predict: i32,
    think_mode: ThinkMode,
    temperature: f32,
    top_p: f32,
    min_p: f32,
    seed: u64,
    system: String,
    read_prompt_file: String,
    next_prompt: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model_path: "ds4flash.gguf".to_string(),
            backend: Backend::default_backend(),
            ctx_size: 32768,
            n_predict: 50000,
            think_mode: ThinkMode::default_mode(),
            temperature: 1.0,
            top_p: 1.0,
            min_p: 0.05,
            seed: 1,
            system: "You are a helpful assistant".to_string(),
            read_prompt_file: String::new(),
            next_prompt: String::new(),
        }
    }
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

fn run() -> Result<i32, Box<dyn std::error::Error>> {
    let config = parse_args(std::env::args().skip(1))?;
    let backend = config.backend;
    log_context_memory(backend, config.ctx_size);

    let options = EngineOptions::new(&config.model_path, backend);
    let engine = match Engine::open(&options) {
        Ok(engine) => engine,
        Err(err) if err.open_failed_code().is_some() => return Ok(1),
        Err(err) => return Err(Box::new(err)),
    };
    let mut chat =
        engine.create_chat_session(&config.system, config.ctx_size, config.think_mode)?;
    chat.set_ctx(config.ctx_size)?;

    let turn_options = InteractiveTurnOptions {
        n_predict: config.n_predict,
        think_mode: config.think_mode,
        temperature: config.temperature,
        top_p: config.top_p,
        min_p: config.min_p,
        seed: config.seed,
    };
    let first_prompt = fs::read_to_string(&config.read_prompt_file)?;
    let first = chat.run_turn(&first_prompt, turn_options);
    write_turn("read", &first.stdout)?;
    if first.exit_code != 0 {
        return Ok(first.exit_code);
    }

    let second = chat.run_turn(&config.next_prompt, turn_options);
    write_turn("direct", &second.stdout)?;
    Ok(second.exit_code)
}

fn parse_args<I, S>(args: I) -> Result<Config, String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let argv: Vec<String> = args.into_iter().map(Into::into).collect();
    let mut config = Config::default();
    let mut i = 0usize;
    while i < argv.len() {
        let arg = argv[i].as_str();
        match arg {
            "-m" | "--model" => config.model_path = need_arg(&argv, &mut i, arg)?.to_string(),
            "-c" | "--ctx" => {
                config.ctx_size = parse_positive_i32(need_arg(&argv, &mut i, arg)?, arg)?
            }
            "-n" | "--tokens" => {
                config.n_predict = parse_positive_i32(need_arg(&argv, &mut i, arg)?, arg)?
            }
            "--temp" => {
                config.temperature =
                    parse_float_range(need_arg(&argv, &mut i, arg)?, arg, 0.0, 100.0)?
            }
            "--top-p" => {
                config.top_p = parse_float_range(need_arg(&argv, &mut i, arg)?, arg, 0.0, 1.0)?
            }
            "--min-p" => {
                config.min_p = parse_float_range(need_arg(&argv, &mut i, arg)?, arg, 0.0, 1.0)?
            }
            "--seed" => config.seed = parse_positive_u64(need_arg(&argv, &mut i, arg)?, arg)?,
            "-sys" | "--system" => config.system = need_arg(&argv, &mut i, arg)?.to_string(),
            "--backend" => {
                let value = need_arg(&argv, &mut i, arg)?;
                config.backend = Backend::parse(value).ok_or_else(|| {
                    format!(
                        "ds4: invalid backend: {value}\n\
                         ds4: valid backends are: metal, cuda, cpu"
                    )
                })?;
            }
            "--cuda" => config.backend = Backend::Cuda,
            "--metal" => config.backend = Backend::Metal,
            "--cpu" => config.backend = Backend::Cpu,
            "--think" => config.think_mode = ThinkMode::High,
            "--think-max" => config.think_mode = ThinkMode::Max,
            "--nothink" => config.think_mode = ThinkMode::None,
            "--read-prompt-file" => {
                config.read_prompt_file = need_arg(&argv, &mut i, arg)?.to_string();
            }
            "--next-prompt" => config.next_prompt = need_arg(&argv, &mut i, arg)?.to_string(),
            _ => return Err(format!("ds4: unknown option: {arg}")),
        }
        i += 1;
    }
    if config.read_prompt_file.is_empty() {
        return Err("ds4-rs: M8.15a runtime requires --read-prompt-file".to_string());
    }
    if config.next_prompt.is_empty() {
        return Err("ds4-rs: M8.15a runtime requires --next-prompt".to_string());
    }
    Ok(config)
}

fn need_arg<'a>(argv: &'a [String], idx: &mut usize, opt: &str) -> Result<&'a str, String> {
    if *idx + 1 >= argv.len() {
        return Err(format!("ds4: missing value for {opt}"));
    }
    *idx += 1;
    Ok(argv[*idx].as_str())
}

fn parse_positive_i32(value: &str, opt: &str) -> Result<i32, String> {
    match value.parse::<i64>() {
        Ok(v) if (1..=i32::MAX as i64).contains(&v) => Ok(v as i32),
        _ => Err(format!("ds4: invalid value for {opt}: {value}")),
    }
}

fn parse_positive_u64(value: &str, opt: &str) -> Result<u64, String> {
    match value.parse::<u64>() {
        Ok(v) if v > 0 => Ok(v),
        _ => Err(format!("ds4: invalid value for {opt}: {value}")),
    }
}

fn parse_float_range(value: &str, opt: &str, min: f32, max: f32) -> Result<f32, String> {
    match value.parse::<f32>() {
        Ok(v) if v.is_finite() && v >= min && v <= max => Ok(v),
        _ => Err(format!("ds4: invalid value for {opt}: {value}")),
    }
}

fn write_turn(label: &str, bytes: &[u8]) -> io::Result<()> {
    let mut stdout = io::stdout().lock();
    writeln!(stdout, "<<<ds4-rs-turn:{label}>>>")?;
    stdout.write_all(bytes)?;
    writeln!(stdout, "<<<ds4-rs-end:{label}>>>")?;
    Ok(())
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
