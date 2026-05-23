use ds4_engine::{
    context_memory_estimate, Backend, Engine, EngineOptions, ServerGenerationOptions, ThinkMode,
};
use ds4_gguf::{
    format_http_error, format_http_response, format_openai_chat_completion_http,
    openai_context_length_error_body, parse_http_request, parse_openai_chat_request,
    request_exceeds_context, route_no_model_server_request_with_generation_message, HttpRequest,
    HttpRequestParseError, NoModelRouteConfig, OpenAiChatCompletion, OpenAiChatRequest,
    OpenAiUsage,
};
use std::env;
use std::fs::File;
use std::io::{self, Read, Write};
use std::net::{Ipv4Addr, Shutdown, SocketAddrV4, TcpListener, TcpStream};
use std::process;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const READ_TIMEOUT: Duration = Duration::from_secs(10);
const WRITE_TIMEOUT: Duration = Duration::from_secs(10);
const MODEL_BACKED_GENERATION_MESSAGE: &str = "model-backed chat generation is not implemented yet";

#[derive(Debug, Clone, PartialEq)]
struct ServerConfig {
    model_path: String,
    mtp_path: Option<String>,
    backend: Backend,
    n_threads: i32,
    mtp_draft_tokens: i32,
    mtp_margin: f32,
    directional_steering_file: Option<String>,
    directional_steering_attn: f32,
    directional_steering_ffn: f32,
    warm_weights: bool,
    quality: bool,
    host: String,
    port: u16,
    trace_path: Option<String>,
    context_length: i32,
    default_tokens: i32,
    enable_cors: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            model_path: "ds4flash.gguf".to_string(),
            mtp_path: None,
            backend: Backend::default_backend(),
            n_threads: 0,
            mtp_draft_tokens: 1,
            mtp_margin: 3.0,
            directional_steering_file: None,
            directional_steering_attn: 0.0,
            directional_steering_ffn: 0.0,
            warm_weights: false,
            quality: false,
            host: "127.0.0.1".to_string(),
            port: 8000,
            trace_path: None,
            context_length: 32768,
            default_tokens: 393216,
            enable_cors: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
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

fn run() -> Result<i32, Box<dyn std::error::Error>> {
    let config = match parse_args(env::args().skip(1)) {
        Ok(Some(config)) => config,
        Ok(None) => return Ok(0),
        Err(exit) => return Ok(write_exit(exit)?),
    };

    let engine_options = engine_options_from_config(&config);
    let engine = match Engine::open(&engine_options) {
        Ok(engine) => engine,
        Err(err) if err.open_failed_code().is_some() => return Ok(1),
        Err(err) => return Err(Box::new(err)),
    };
    log_context_memory(config.backend, config.context_length);
    let _session = engine.create_chat_session("", config.context_length, ThinkMode::High)?;
    serve(config, &engine)?;
    Ok(0)
}

fn serve(config: ServerConfig, engine: &Engine) -> io::Result<()> {
    let host = bind_host(&config.host)?;
    let addr = SocketAddrV4::new(host, config.port);
    let listener = TcpListener::bind(addr)?;
    let actual_addr = listener.local_addr()?;
    eprintln!("ds4-server-runtime-rs: listening on http://{actual_addr}");

    let mut state = RuntimeState {
        sequence: 0,
        trace: match config.trace_path.as_deref() {
            Some(path) => Some(File::create(path)?),
            None => None,
        },
    };
    for stream in listener.incoming() {
        match stream {
            Ok(mut stream) => {
                if let Err(err) = handle_client(&mut stream, &config, engine, &mut state) {
                    eprintln!("ds4-server-runtime-rs: client error: {err}");
                }
            }
            Err(err) => {
                eprintln!("ds4-server-runtime-rs: accept failed: {err}");
            }
        }
    }
    Ok(())
}

struct RuntimeState {
    sequence: u64,
    trace: Option<File>,
}

fn handle_client(
    stream: &mut TcpStream,
    config: &ServerConfig,
    engine: &Engine,
    state: &mut RuntimeState,
) -> io::Result<()> {
    stream.set_read_timeout(Some(READ_TIMEOUT))?;
    stream.set_write_timeout(Some(WRITE_TIMEOUT))?;
    let request = read_request_bytes(stream)?;
    let response = route_runtime_http(&request, config, engine, state);
    stream.write_all(response.as_bytes())?;
    let _ = stream.shutdown(Shutdown::Both);
    Ok(())
}

fn route_runtime_http(
    input: &[u8],
    config: &ServerConfig,
    engine: &Engine,
    state: &mut RuntimeState,
) -> String {
    let route_config = route_config(config);
    let request = match parse_http_request(input) {
        Ok(request) => request,
        Err(_) => return format_http_error(config.enable_cors, 400, "bad HTTP request"),
    };
    if request.method == "POST" && request.path == "/v1/chat/completions" {
        return route_chat_completions(&request, config, engine, state);
    }
    route_no_model_server_request_with_generation_message(
        &request,
        route_config,
        |prompt| count_prompt_tokens(engine, prompt),
        MODEL_BACKED_GENERATION_MESSAGE,
    )
}

fn route_chat_completions(
    request: &HttpRequest,
    config: &ServerConfig,
    engine: &Engine,
    state: &mut RuntimeState,
) -> String {
    let parsed = match parse_openai_chat_request(
        &request.body,
        config.default_tokens,
        config.context_length,
    ) {
        Ok(parsed) => parsed,
        Err(err) => return format_http_error(config.enable_cors, 400, err.message()),
    };
    let prompt = match engine.encode_chat_prompt("", &parsed.prompt_text, ThinkMode::None) {
        Ok(prompt) => prompt,
        Err(err) => {
            eprintln!("ds4-server-runtime-rs: failed to tokenize chat prompt: {err}");
            return format_http_error(config.enable_cors, 400, "invalid prompt text");
        }
    };
    let prompt_tokens = prompt.len().max(0) as usize;
    if request_exceeds_context(prompt_tokens, config.context_length) {
        let body = openai_context_length_error_body(prompt_tokens, config.context_length);
        return format_http_response(config.enable_cors, 400, Some("application/json"), &body);
    }
    if let Some(message) = unsupported_chat_generation_message(&parsed) {
        return format_http_error(config.enable_cors, 503, message);
    }

    let generated = engine.generate_server_text(
        &prompt,
        ServerGenerationOptions {
            n_predict: parsed.max_tokens,
            ctx_size: config.context_length,
            temperature: parsed.sampling.temperature,
            top_k: parsed.sampling.top_k,
            top_p: parsed.sampling.top_p,
            min_p: parsed.sampling.min_p,
            seed: parsed.seed,
        },
    );
    if generated.exit_code != 0 || generated.finish_reason == "error" {
        return format_http_error(config.enable_cors, 500, "generation failed");
    }
    let content = String::from_utf8_lossy(&generated.text);
    state.sequence += 1;
    let id = format!("chatcmpl-{}", state.sequence);
    if let Some(trace) = state.trace.as_mut() {
        if let Err(err) = write_chat_trace(
            trace,
            state.sequence,
            &request.body,
            &parsed,
            generated.prompt_tokens,
            &generated,
            &content,
        ) {
            eprintln!("ds4-server-runtime-rs: failed to write trace: {err}");
        }
    }
    format_openai_chat_completion_http(
        config.enable_cors,
        &OpenAiChatCompletion {
            id: &id,
            created: unix_timestamp(),
            model: &parsed.model,
            content: &content,
            reasoning_content: None,
            finish_reason: generated.finish_reason,
            usage: OpenAiUsage::new(
                generated.prompt_tokens,
                generated.completion_tokens,
                0,
                generated.prompt_tokens,
            ),
        },
    )
}

fn unsupported_chat_generation_message(parsed: &OpenAiChatRequest) -> Option<&'static str> {
    if parsed.stream {
        Some("streaming chat generation is not implemented yet")
    } else if parsed.has_tools {
        Some("tool chat generation is not implemented yet")
    } else if map_think_mode(parsed.think_mode).enabled() {
        Some("thinking chat generation is not implemented yet")
    } else if !parsed.stops.is_empty() {
        Some("stop sequences are not implemented yet")
    } else {
        None
    }
}

fn write_chat_trace(
    trace: &mut File,
    sequence: u64,
    raw_body: &str,
    request: &OpenAiChatRequest,
    prompt_tokens: i32,
    generated: &ds4_engine::ServerGenerationResult,
    content: &str,
) -> io::Result<()> {
    writeln!(trace, "===== request {sequence} =====")?;
    writeln!(trace, "kind: chat")?;
    writeln!(trace, "model: {}", request.model)?;
    writeln!(trace, "stream: {}", if request.stream { 1 } else { 0 })?;
    writeln!(trace, "tools: {}", if request.has_tools { 1 } else { 0 })?;
    writeln!(trace, "think_mode: {}", request.think_mode.name())?;
    writeln!(trace, "prompt_tokens: {prompt_tokens}")?;
    writeln!(trace, "effective_prompt_tokens: {prompt_tokens}")?;
    writeln!(trace, "cached_tokens: 0")?;
    writeln!(trace, "max_tokens: {}", request.max_tokens)?;
    writeln!(trace, "temperature: {:.3}", request.sampling.temperature)?;
    writeln!(trace, "top_k: {}", request.sampling.top_k)?;
    writeln!(trace, "top_p: {:.3}", request.sampling.top_p)?;
    writeln!(trace, "min_p: {:.3}", request.sampling.min_p)?;
    writeln!(trace, "seed: {}", request.seed)?;
    writeln!(
        trace,
        "stream_include_usage: {}",
        if request.stream_include_usage { 1 } else { 0 }
    )?;
    writeln!(trace)?;
    writeln!(trace, "--- cache decision ---")?;
    writeln!(trace, "live_tokens_before: 0")?;
    writeln!(trace, "prompt_tokens: {prompt_tokens}")?;
    writeln!(trace, "live_prompt_common: 0")?;
    writeln!(trace, "memory_token_reusable: 0")?;
    writeln!(trace, "memory_miss_reason: no-live-checkpoint")?;
    writeln!(trace, "cache_source: none")?;
    writeln!(trace, "cached_tokens: 0")?;
    writeln!(trace, "disk_cached_tokens: 0")?;
    writeln!(trace)?;
    writeln!(trace, "--- raw request json ---")?;
    writeln!(trace, "{raw_body}")?;
    writeln!(trace)?;
    writeln!(trace, "--- rendered prompt ---")?;
    writeln!(trace, "{}", request.prompt_text)?;
    writeln!(trace)?;
    writeln!(trace, "--- generated text ---")?;
    writeln!(trace, "{content}")?;
    writeln!(trace)?;
    writeln!(trace, "--- parsed message ---")?;
    writeln!(trace, "finish: {}", generated.finish_reason)?;
    writeln!(trace, "generated_tokens: {}", generated.completion_tokens)?;
    writeln!(trace, "content:")?;
    writeln!(trace, "{content}")?;
    writeln!(trace)?;
    writeln!(trace, "===== end request {sequence} =====")?;
    writeln!(trace)?;
    trace.flush()
}

fn route_config(config: &ServerConfig) -> NoModelRouteConfig {
    NoModelRouteConfig {
        enable_cors: config.enable_cors,
        context_length: config.context_length,
        default_tokens: config.default_tokens,
    }
}

fn read_request_bytes(stream: &mut TcpStream) -> io::Result<Vec<u8>> {
    let mut request = Vec::new();
    loop {
        match parse_http_request(&request) {
            Ok(_) => return Ok(request),
            Err(HttpRequestParseError::Incomplete) => {}
            Err(_) => return Ok(request),
        }

        let mut chunk = [0_u8; 8192];
        match stream.read(&mut chunk) {
            Ok(0) => return Ok(request),
            Ok(n) => request.extend_from_slice(&chunk[..n]),
            Err(err)
                if err.kind() == io::ErrorKind::WouldBlock
                    || err.kind() == io::ErrorKind::TimedOut =>
            {
                return Ok(request);
            }
            Err(err) => return Err(err),
        }
    }
}

fn count_prompt_tokens(engine: &Engine, prompt_text: &str) -> usize {
    match engine.encode_chat_prompt("", prompt_text, ThinkMode::None) {
        Ok(tokens) => tokens.len().max(0) as usize,
        Err(err) => {
            eprintln!("ds4-server-runtime-rs: failed to tokenize prompt: {err}");
            usize::MAX
        }
    }
}

fn map_think_mode(mode: ds4_gguf::ThinkMode) -> ThinkMode {
    match mode {
        ds4_gguf::ThinkMode::None => ThinkMode::None,
        ds4_gguf::ThinkMode::High => ThinkMode::High,
        ds4_gguf::ThinkMode::Max => ThinkMode::Max,
    }
}

fn unix_timestamp() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs() as i64)
}

fn engine_options_from_config(config: &ServerConfig) -> EngineOptions<'_> {
    let mut options = EngineOptions::new(&config.model_path, config.backend);
    options.mtp_path = config.mtp_path.as_deref();
    options.n_threads = config.n_threads;
    options.mtp_draft_tokens = config.mtp_draft_tokens;
    options.mtp_margin = config.mtp_margin;
    options.directional_steering_file = config.directional_steering_file.as_deref();
    options.directional_steering_attn = config.directional_steering_attn;
    options.directional_steering_ffn = config.directional_steering_ffn;
    options.warm_weights = config.warm_weights;
    options.quality = config.quality;
    options
}

fn bind_host(host: &str) -> io::Result<Ipv4Addr> {
    let host = if host == "localhost" {
        "127.0.0.1"
    } else {
        host
    };
    host.parse::<Ipv4Addr>()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "invalid bind host"))
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Option<ServerConfig>, CliExit> {
    let mut config = ServerConfig::default();
    let mut directional_steering_scale_set = false;
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                return Err(CliExit {
                    code: 0,
                    stdout: usage().to_string(),
                    stderr: String::new(),
                });
            }
            "-m" | "--model" => {
                config.model_path = need_arg(&mut args, &arg)?;
            }
            "--mtp" => {
                config.mtp_path = Some(need_arg(&mut args, &arg)?);
            }
            "--mtp-draft" => {
                config.mtp_draft_tokens = parse_positive_i32(&need_arg(&mut args, &arg)?, &arg)?;
            }
            "--mtp-margin" => {
                config.mtp_margin =
                    parse_float_range(&need_arg(&mut args, &arg)?, &arg, 0.0, 1000.0)?;
            }
            "--backend" => {
                let value = need_arg(&mut args, &arg)?;
                config.backend = Backend::parse(&value).ok_or_else(|| CliExit {
                    code: 2,
                    stdout: String::new(),
                    stderr: format!(
                        "ds4-server-runtime-rs: invalid backend: {value}\n\
                         ds4-server-runtime-rs: valid backends are: metal, cuda, cpu\n"
                    ),
                })?;
            }
            "--cuda" => {
                config.backend = Backend::Cuda;
            }
            "--metal" => {
                config.backend = Backend::Metal;
            }
            "--cpu" => {
                config.backend = Backend::Cpu;
            }
            "-t" | "--threads" => {
                config.n_threads = parse_positive_i32(&need_arg(&mut args, &arg)?, &arg)?;
            }
            "--dir-steering-file" => {
                config.directional_steering_file = Some(need_arg(&mut args, &arg)?);
            }
            "--dir-steering-attn" => {
                config.directional_steering_attn =
                    parse_float_range(&need_arg(&mut args, &arg)?, &arg, -100.0, 100.0)?;
                directional_steering_scale_set = true;
            }
            "--dir-steering-ffn" => {
                config.directional_steering_ffn =
                    parse_float_range(&need_arg(&mut args, &arg)?, &arg, -100.0, 100.0)?;
                directional_steering_scale_set = true;
            }
            "--warm-weights" => {
                config.warm_weights = true;
            }
            "--quality" => {
                config.quality = true;
            }
            "--host" => {
                config.host = need_arg(&mut args, &arg)?;
            }
            "--port" => {
                config.port = parse_port(&need_arg(&mut args, &arg)?, &arg)?;
            }
            "--trace" => {
                config.trace_path = Some(need_arg(&mut args, &arg)?);
            }
            "-c" | "--ctx" => {
                config.context_length = parse_positive_i32(&need_arg(&mut args, &arg)?, &arg)?;
            }
            "-n" | "--tokens" => {
                config.default_tokens = parse_positive_i32(&need_arg(&mut args, &arg)?, &arg)?;
            }
            "--cors" => {
                config.enable_cors = true;
            }
            _ => {
                return Err(CliExit {
                    code: 2,
                    stdout: String::new(),
                    stderr: format!("ds4-server-runtime-rs: unknown option: {arg}\n{}", usage()),
                });
            }
        }
    }

    if config.directional_steering_file.is_some() && !directional_steering_scale_set {
        config.directional_steering_ffn = 1.0;
    }
    Ok(Some(config))
}

fn need_arg(args: &mut impl Iterator<Item = String>, option: &str) -> Result<String, CliExit> {
    args.next().ok_or_else(|| CliExit {
        code: 2,
        stdout: String::new(),
        stderr: format!("ds4-server-runtime-rs: missing value for {option}\n"),
    })
}

fn parse_positive_i32(value: &str, option: &str) -> Result<i32, CliExit> {
    match value.parse::<i64>() {
        Ok(value) if (1..=i32::MAX as i64).contains(&value) => Ok(value as i32),
        _ => Err(invalid_value(option, value)),
    }
}

fn parse_port(value: &str, option: &str) -> Result<u16, CliExit> {
    let parsed = value
        .parse::<u16>()
        .map_err(|_| invalid_value(option, value))?;
    if parsed == 0 {
        return Err(invalid_value(option, value));
    }
    Ok(parsed)
}

fn parse_float_range(value: &str, option: &str, min: f32, max: f32) -> Result<f32, CliExit> {
    match value.parse::<f32>() {
        Ok(value) if value.is_finite() && value >= min && value <= max => Ok(value),
        _ => Err(invalid_value(option, value)),
    }
}

fn invalid_value(option: &str, value: &str) -> CliExit {
    CliExit {
        code: 2,
        stdout: String::new(),
        stderr: format!("ds4-server-runtime-rs: invalid value for {option}: {value}\n"),
    }
}

fn write_exit(exit: CliExit) -> io::Result<i32> {
    let mut stdout = io::stdout().lock();
    let mut stderr = io::stderr().lock();
    stdout.write_all(exit.stdout.as_bytes())?;
    stderr.write_all(exit.stderr.as_bytes())?;
    Ok(exit.code)
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

fn usage() -> &'static str {
    "Usage: ds4-server-runtime-rs [options]\n\
\n\
Model runtime:\n\
  -m, --model FILE\n\
      Model path. Default: ds4flash.gguf\n\
  --backend NAME | --cuda | --metal | --cpu\n\
      Runtime backend. Default: platform default\n\
  --mtp FILE\n\
      Optional MTP model path\n\
  --mtp-draft N\n\
      MTP draft tokens. Default: 1\n\
  --mtp-margin F\n\
      MTP acceptance margin. Default: 3.0\n\
  -t, --threads N\n\
      CPU thread count\n\
  --warm-weights\n\
      Warm model weights at startup\n\
  --quality\n\
      Enable quality-oriented runtime settings\n\
  --dir-steering-file FILE\n\
      Directional steering vector file\n\
  --dir-steering-attn F\n\
      Directional steering attention scale\n\
  --dir-steering-ffn F\n\
      Directional steering FFN scale\n\
\n\
HTTP API:\n\
  --host HOST\n\
      Bind address. Default: 127.0.0.1\n\
  --port N\n\
      Bind port. Default: 8000\n\
  --cors\n\
      Add Access-Control-Allow-* headers for browser JS clients. Does not change --host.\n\
  --trace FILE\n\
      Write a human-readable no-cache chat trace for supported M9.4c requests.\n\
  -c, --ctx N\n\
      Context size used for request parsing and prompt-token limits. Default: 32768\n\
  -n, --tokens N\n\
      Default max output tokens when the client omits a limit. Default: 393216 (384K)\n\
\n\
  -h, --help\n\
      Show this help.\n"
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHAT_BASIC: &str =
        include_str!("../../../../ds4-parity/baselines/server-fixtures/m0.4/chat_basic.json");
    const CHAT_THINKING_DISABLED: &str = include_str!(
        "../../../../ds4-parity/baselines/server-fixtures/m0.4/chat_thinking_disabled.json"
    );

    fn parse(args: &[&str]) -> Result<Option<ServerConfig>, CliExit> {
        parse_args(args.iter().copied().map(str::to_string))
    }

    fn parse_chat(body: &str) -> ds4_gguf::OpenAiChatRequest {
        parse_openai_chat_request(body, 64, 32_768).expect("chat request parses")
    }

    #[test]
    fn parses_default_and_m94a_flags() {
        assert_eq!(parse(&[]).unwrap(), Some(ServerConfig::default()));
        assert_eq!(
            parse(&[
                "--model",
                "model.gguf",
                "--mtp",
                "mtp.gguf",
                "--mtp-draft",
                "2",
                "--mtp-margin",
                "4.5",
                "--backend",
                "cpu",
                "--threads",
                "8",
                "--host",
                "localhost",
                "--port",
                "18080",
                "--trace",
                "server.trace",
                "--ctx",
                "16",
                "--tokens",
                "64",
                "--cors",
                "--warm-weights",
                "--quality",
                "--dir-steering-file",
                "steer.bin",
            ])
            .unwrap(),
            Some(ServerConfig {
                model_path: "model.gguf".to_string(),
                mtp_path: Some("mtp.gguf".to_string()),
                backend: Backend::Cpu,
                n_threads: 8,
                mtp_draft_tokens: 2,
                mtp_margin: 4.5,
                directional_steering_file: Some("steer.bin".to_string()),
                directional_steering_attn: 0.0,
                directional_steering_ffn: 1.0,
                warm_weights: true,
                quality: true,
                host: "localhost".to_string(),
                port: 18080,
                trace_path: Some("server.trace".to_string()),
                context_length: 16,
                default_tokens: 64,
                enable_cors: true,
            })
        );
    }

    #[test]
    fn directional_steering_scale_matches_c_default_rule() {
        assert_eq!(
            parse(&[
                "--dir-steering-file",
                "steer.bin",
                "--dir-steering-attn",
                "0.5",
            ])
            .unwrap()
            .unwrap()
            .directional_steering_ffn,
            0.0
        );
        assert_eq!(
            parse(&["--dir-steering-file", "steer.bin"])
                .unwrap()
                .unwrap()
                .directional_steering_ffn,
            1.0
        );
    }

    #[test]
    fn engine_options_map_runtime_flags() {
        let config = parse(&[
            "--model",
            "model.gguf",
            "--mtp",
            "mtp.gguf",
            "--cuda",
            "--threads",
            "3",
            "--dir-steering-file",
            "steer.bin",
            "--dir-steering-ffn",
            "0.75",
            "--warm-weights",
            "--quality",
        ])
        .unwrap()
        .unwrap();
        let options = engine_options_from_config(&config);
        assert_eq!(options.model_path, "model.gguf");
        assert_eq!(options.mtp_path, Some("mtp.gguf"));
        assert_eq!(options.backend, Backend::Cuda);
        assert_eq!(options.n_threads, 3);
        assert_eq!(options.directional_steering_file, Some("steer.bin"));
        assert_eq!(options.directional_steering_ffn, 0.75);
        assert!(options.warm_weights);
        assert!(options.quality);
    }

    #[test]
    fn rejects_missing_invalid_and_unknown_args() {
        assert_eq!(
            parse(&["--ctx", "0"]).unwrap_err(),
            CliExit {
                code: 2,
                stdout: String::new(),
                stderr: "ds4-server-runtime-rs: invalid value for --ctx: 0\n".to_string(),
            }
        );
        assert_eq!(
            parse(&["--model"]).unwrap_err(),
            CliExit {
                code: 2,
                stdout: String::new(),
                stderr: "ds4-server-runtime-rs: missing value for --model\n".to_string(),
            }
        );
        let backend = parse(&["--backend", "bad"]).unwrap_err();
        assert_eq!(backend.code, 2);
        assert!(backend
            .stderr
            .contains("valid backends are: metal, cuda, cpu"));
        let unknown = parse(&["--bad"]).unwrap_err();
        assert_eq!(unknown.code, 2);
        assert!(unknown
            .stderr
            .starts_with("ds4-server-runtime-rs: unknown option: --bad\n"));
    }

    #[test]
    fn bind_host_matches_c_localhost_behavior() {
        assert_eq!(bind_host("localhost").unwrap(), Ipv4Addr::new(127, 0, 0, 1));
        assert_eq!(bind_host("127.0.0.1").unwrap(), Ipv4Addr::new(127, 0, 0, 1));
        assert!(bind_host("example.com").is_err());
    }

    #[test]
    fn m94c_allows_only_no_cache_non_streaming_non_thinking_chat() {
        assert_eq!(
            unsupported_chat_generation_message(&parse_chat(CHAT_BASIC)),
            None
        );
        assert_eq!(
            unsupported_chat_generation_message(&parse_chat(CHAT_THINKING_DISABLED)),
            None
        );
        assert_eq!(
            unsupported_chat_generation_message(&parse_chat(
                r#"{"messages":[{"role":"user","content":"hi"}],"stream":true}"#
            )),
            Some("streaming chat generation is not implemented yet")
        );
        assert_eq!(
            unsupported_chat_generation_message(&parse_chat(
                r#"{"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{}}}}]}"#
            )),
            Some("tool chat generation is not implemented yet")
        );
        assert_eq!(
            unsupported_chat_generation_message(&parse_chat(
                r#"{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"hi"}]}"#
            )),
            Some("thinking chat generation is not implemented yet")
        );
        assert_eq!(
            unsupported_chat_generation_message(&parse_chat(
                r#"{"model":"deepseek-chat","messages":[{"role":"user","content":"hi"}],"stop":"done"}"#
            )),
            Some("stop sequences are not implemented yet")
        );
    }
}
