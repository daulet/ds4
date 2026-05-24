use ds4_engine::{
    Backend, Engine, EngineOptions, RuntimeGraphRoute, ServerGenerationOptions,
    RUNTIME_GRAPH_ROUTE_UNSUPPORTED_CODE, RUNTIME_GRAPH_ROUTE_VALID_VALUES,
};
use std::error::Error;
use std::fs;
use std::io::{self, Write};
use std::process;

const HELP: &str = "\
Usage: ds4-runtime-long-context-rs --model FILE [options]\n\
\n\
M10.9d Rust runtime long-context capture.\n\
\n\
Options:\n\
  -m, --model FILE\n\
  --prompt-file FILE\n\
  --backend NAME | --cuda | --metal | --cpu\n\
  --runtime-graph ROUTE\n\
  --ctx N\n\
  --tokens N\n\
  --seed N\n";

const DEFAULT_PROMPT_FILE: &str = "tests/long_context_story_prompt.txt";
const DEFAULT_CTX_SIZE: i32 = 100000;
const DEFAULT_TOKENS: i32 = 350;
const DEFAULT_SEED: u64 = 12345;

#[derive(Debug)]
struct Config {
    model_path: String,
    prompt_path: String,
    backend: Backend,
    route: RuntimeGraphRoute,
    ctx_size: i32,
    n_predict: i32,
    seed: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model_path: "ds4flash.gguf".to_string(),
            prompt_path: DEFAULT_PROMPT_FILE.to_string(),
            backend: Backend::Cuda,
            route: RuntimeGraphRoute::Graph,
            ctx_size: DEFAULT_CTX_SIZE,
            n_predict: DEFAULT_TOKENS,
            seed: DEFAULT_SEED,
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
                "ds4-runtime-long-context-rs: --runtime-graph graph requires cuda or metal backend\n"
                    .to_string(),
        })?);
    }

    let prompt_text = fs::read_to_string(&config.prompt_path)?;
    let engine_options = EngineOptions::new(&config.model_path, config.backend);
    let engine = Engine::open(&engine_options)?;
    let prompt = engine.encode_chat_prompt("", &prompt_text, ds4_engine::ThinkMode::None)?;
    let generation = engine.generate_server_text(
        &prompt,
        ServerGenerationOptions {
            n_predict: config.n_predict,
            ctx_size: config.ctx_size,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed: config.seed,
        },
    );

    let mut out = Vec::new();
    write_capture(&mut out, &config, &generation)?;
    io::stdout().lock().write_all(&out)?;
    Ok(generation.exit_code)
}

fn parse_args<I, S>(args: I) -> Result<Config, CliExit>
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
                            "ds4-runtime-long-context-rs: invalid backend: {value}\n\
                             ds4-runtime-long-context-rs: valid backends are: metal, cuda, cpu\n"
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
                            "ds4-runtime-long-context-rs: invalid runtime graph route: {value}\n\
                             ds4-runtime-long-context-rs: valid runtime graph routes are: {RUNTIME_GRAPH_ROUTE_VALID_VALUES}\n"
                        ),
                    )
                })?;
            }
            "--ctx" => config.ctx_size = parse_positive_i32(need_arg(&argv, &mut i, arg)?, arg)?,
            "--tokens" => {
                config.n_predict = parse_positive_i32(need_arg(&argv, &mut i, arg)?, arg)?
            }
            "--seed" => config.seed = parse_positive_u64(need_arg(&argv, &mut i, arg)?, arg)?,
            "--cuda" => config.backend = Backend::Cuda,
            "--metal" => config.backend = Backend::Metal,
            "--cpu" => config.backend = Backend::Cpu,
            _ => {
                return Err(exit(
                    2,
                    "",
                    &format!("ds4-runtime-long-context-rs: unknown option: {arg}\n{HELP}"),
                ))
            }
        }
        i += 1;
    }
    Ok(config)
}

fn need_arg<'a>(argv: &'a [String], idx: &mut usize, opt: &str) -> Result<&'a str, CliExit> {
    if *idx + 1 >= argv.len() {
        return Err(exit(
            2,
            "",
            &format!("ds4-runtime-long-context-rs: missing value for {opt}\n"),
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
            &format!("ds4-runtime-long-context-rs: invalid value for {opt}: {value}\n"),
        )),
    }
}

fn parse_positive_u64(value: &str, opt: &str) -> Result<u64, CliExit> {
    match value.parse::<u64>() {
        Ok(v) if v > 0 => Ok(v),
        _ => Err(exit(
            2,
            "",
            &format!("ds4-runtime-long-context-rs: invalid value for {opt}: {value}\n"),
        )),
    }
}

fn write_capture<W: Write>(
    out: &mut W,
    config: &Config,
    generation: &ds4_engine::ServerGenerationResult,
) -> io::Result<()> {
    writeln!(out, "{{")?;
    writeln!(
        out,
        "  \"schema\": \"ds4.runtime_graph_long_context.rust.v1\","
    )?;
    writeln!(out, "  \"source\": \"ds4-runtime-long-context-rs\",")?;
    write!(out, "  \"runtime_graph_route\": ")?;
    write_json_str(out, config.route.name())?;
    writeln!(out, ",")?;
    write!(out, "  \"backend\": ")?;
    write_json_str(out, config.backend.name())?;
    writeln!(out, ",")?;
    write!(out, "  \"model_path\": ")?;
    write_json_str(out, &config.model_path)?;
    writeln!(out, ",")?;
    write!(out, "  \"prompt_file\": ")?;
    write_json_str(out, &config.prompt_path)?;
    writeln!(out, ",")?;
    writeln!(out, "  \"ctx\": {},", config.ctx_size)?;
    writeln!(out, "  \"max_tokens\": {},", config.n_predict)?;
    writeln!(out, "  \"seed\": {},", config.seed)?;
    writeln!(out, "  \"temperature\": 0,")?;
    writeln!(out, "  \"top_k\": 0,")?;
    writeln!(out, "  \"top_p\": 1,")?;
    writeln!(out, "  \"min_p\": 0,")?;
    writeln!(out, "  \"exit_code\": {},", generation.exit_code)?;
    writeln!(out, "  \"prompt_tokens\": {},", generation.prompt_tokens)?;
    writeln!(
        out,
        "  \"completion_tokens\": {},",
        generation.completion_tokens
    )?;
    writeln!(
        out,
        "  \"cache_read_tokens\": {},",
        generation.cache_read_tokens
    )?;
    writeln!(
        out,
        "  \"cache_write_tokens\": {},",
        generation.cache_write_tokens
    )?;
    writeln!(
        out,
        "  \"live_tokens_before\": {},",
        generation.live_tokens_before
    )?;
    writeln!(
        out,
        "  \"live_prompt_common\": {},",
        generation.live_prompt_common
    )?;
    write!(out, "  \"finish_reason\": ")?;
    write_json_str(out, generation.finish_reason)?;
    writeln!(out, ",")?;
    write!(out, "  \"generated_text\": ")?;
    write_json_str(out, &String::from_utf8_lossy(&generation.text))?;
    writeln!(out)?;
    writeln!(out, "}}")?;
    Ok(())
}

fn write_json_str<W: Write>(out: &mut W, value: &str) -> io::Result<()> {
    write!(out, "\"")?;
    for ch in value.chars() {
        match ch {
            '"' => write!(out, "\\\"")?,
            '\\' => write!(out, "\\\\")?,
            '\n' => write!(out, "\\n")?,
            '\r' => write!(out, "\\r")?,
            '\t' => write!(out, "\\t")?,
            ch if ch < ' ' => write!(out, "\\u{:04x}", ch as u32)?,
            ch => write!(out, "{ch}")?,
        }
    }
    write!(out, "\"")?;
    Ok(())
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
    fn rejects_cpu_graph_before_model_open() {
        let config = parse_args([
            "--cpu",
            "--runtime-graph",
            "graph",
            "--model",
            "/tmp/missing.gguf",
        ])
        .unwrap();
        assert_eq!(config.backend, Backend::Cpu);
        assert_eq!(config.route, RuntimeGraphRoute::Graph);
    }

    #[test]
    fn retains_long_context_defaults() {
        let config = parse_args(["--model", "model.gguf"]).unwrap();
        assert_eq!(config.prompt_path, DEFAULT_PROMPT_FILE);
        assert_eq!(config.ctx_size, DEFAULT_CTX_SIZE);
        assert_eq!(config.n_predict, DEFAULT_TOKENS);
        assert_eq!(config.seed, DEFAULT_SEED);
        assert_eq!(config.route, RuntimeGraphRoute::Graph);
    }
}
