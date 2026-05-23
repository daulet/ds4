use ds4_engine::{context_memory_estimate, Backend, Engine, EngineOptions, ThinkMode};
use ds4_gguf::{
    parse_http_request, route_no_model_server_http_with_generation_message, HttpRequestParseError,
    NoModelRouteConfig,
};
use std::env;
use std::io::{self, Read, Write};
use std::net::{Ipv4Addr, Shutdown, SocketAddrV4, TcpListener, TcpStream};
use std::process;
use std::time::Duration;

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

    for stream in listener.incoming() {
        match stream {
            Ok(mut stream) => {
                if let Err(err) = handle_client(&mut stream, &config, engine) {
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

fn handle_client(stream: &mut TcpStream, config: &ServerConfig, engine: &Engine) -> io::Result<()> {
    stream.set_read_timeout(Some(READ_TIMEOUT))?;
    stream.set_write_timeout(Some(WRITE_TIMEOUT))?;
    let request = read_request_bytes(stream)?;
    let route_config = NoModelRouteConfig {
        enable_cors: config.enable_cors,
        context_length: config.context_length,
        default_tokens: config.default_tokens,
    };
    let response = route_no_model_server_http_with_generation_message(
        &request,
        route_config,
        |prompt| count_prompt_tokens(engine, prompt),
        MODEL_BACKED_GENERATION_MESSAGE,
    );
    stream.write_all(response.as_bytes())?;
    let _ = stream.shutdown(Shutdown::Both);
    Ok(())
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

    fn parse(args: &[&str]) -> Result<Option<ServerConfig>, CliExit> {
        parse_args(args.iter().copied().map(str::to_string))
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
}
