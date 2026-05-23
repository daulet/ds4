use ds4_gguf::{
    parse_http_request, route_no_model_server_http_with_prompt_tokens, HttpRequestParseError,
    NoModelRouteConfig,
};
use std::env;
use std::io::{self, Read, Write};
use std::net::{Ipv4Addr, Shutdown, SocketAddrV4, TcpListener, TcpStream};
use std::time::Duration;

const READ_TIMEOUT: Duration = Duration::from_secs(10);
const WRITE_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Debug, Clone, PartialEq, Eq)]
struct ServerConfig {
    host: String,
    port: u16,
    context_length: i32,
    default_tokens: i32,
    enable_cors: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8000,
            context_length: 32768,
            default_tokens: 393216,
            enable_cors: false,
        }
    }
}

fn main() {
    let config = match parse_args(env::args().skip(1)) {
        Ok(Some(config)) => config,
        Ok(None) => return,
        Err(exit) => {
            let _ = io::stderr().write_all(exit.message.as_bytes());
            std::process::exit(exit.code);
        }
    };

    if let Err(err) = serve(config) {
        eprintln!("ds4-server-rs: {err}");
        std::process::exit(1);
    }
}

fn serve(config: ServerConfig) -> io::Result<()> {
    let host = bind_host(&config.host)?;
    let addr = SocketAddrV4::new(host, config.port);
    let listener = TcpListener::bind(addr)?;
    let actual_addr = listener.local_addr()?;
    eprintln!("ds4-server-rs: listening on http://{actual_addr}");

    for stream in listener.incoming() {
        match stream {
            Ok(mut stream) => {
                if let Err(err) = handle_client(&mut stream, &config) {
                    eprintln!("ds4-server-rs: client error: {err}");
                }
            }
            Err(err) => {
                eprintln!("ds4-server-rs: accept failed: {err}");
            }
        }
    }
    Ok(())
}

fn handle_client(stream: &mut TcpStream, config: &ServerConfig) -> io::Result<()> {
    stream.set_read_timeout(Some(READ_TIMEOUT))?;
    stream.set_write_timeout(Some(WRITE_TIMEOUT))?;
    let request = read_request_bytes(stream)?;
    let route_config = NoModelRouteConfig {
        enable_cors: config.enable_cors,
        context_length: config.context_length,
        default_tokens: config.default_tokens,
    };
    let response = route_no_model_server_http_with_prompt_tokens(
        &request,
        route_config,
        estimate_prompt_tokens,
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

fn estimate_prompt_tokens(prompt_text: &str) -> usize {
    // M9.3c2 is intentionally model-free; later milestones can inject the real
    // tokenizer through the dispatcher once the server owns a loaded model.
    prompt_text.split_whitespace().count().max(1)
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct CliExit {
    code: i32,
    message: String,
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Option<ServerConfig>, CliExit> {
    let mut config = ServerConfig::default();
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                let _ = io::stdout().write_all(usage().as_bytes());
                return Ok(None);
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
                    message: format!("ds4-server-rs: unknown option: {arg}\n{}", usage()),
                });
            }
        }
    }
    Ok(Some(config))
}

fn need_arg(args: &mut impl Iterator<Item = String>, option: &str) -> Result<String, CliExit> {
    args.next().ok_or_else(|| CliExit {
        code: 2,
        message: format!("ds4-server-rs: missing value for {option}\n"),
    })
}

fn parse_positive_i32(value: &str, option: &str) -> Result<i32, CliExit> {
    let parsed = value
        .parse::<i32>()
        .map_err(|_| invalid_value(option, value))?;
    if parsed <= 0 {
        return Err(invalid_value(option, value));
    }
    Ok(parsed)
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

fn invalid_value(option: &str, value: &str) -> CliExit {
    CliExit {
        code: 2,
        message: format!("ds4-server-rs: invalid value for {option}: {value}\n"),
    }
}

fn usage() -> &'static str {
    "Usage: ds4-server-rs [options]\n\
\n\
HTTP API:\n\
  --host HOST\n\
      Bind address. Default: 127.0.0.1\n\
  --port N\n\
      Bind port. Default: 8000\n\
  --cors\n\
      Add Access-Control-Allow-* headers for browser JS clients. Does not change --host.\n\
\n\
Model-free runtime:\n\
  -c, --ctx N\n\
      Context size used for no-model context-limit checks. Default: 32768\n\
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
    fn parses_default_and_m93_flags() {
        assert_eq!(parse(&[]).unwrap(), Some(ServerConfig::default()));
        assert_eq!(
            parse(&[
                "--host",
                "localhost",
                "--port",
                "18080",
                "--ctx",
                "16",
                "--tokens",
                "64",
                "--cors"
            ])
            .unwrap(),
            Some(ServerConfig {
                host: "localhost".to_string(),
                port: 18080,
                context_length: 16,
                default_tokens: 64,
                enable_cors: true,
            })
        );
    }

    #[test]
    fn rejects_missing_invalid_and_unknown_args() {
        assert_eq!(
            parse(&["--ctx", "0"]).unwrap_err(),
            CliExit {
                code: 2,
                message: "ds4-server-rs: invalid value for --ctx: 0\n".to_string(),
            }
        );
        assert_eq!(
            parse(&["--host"]).unwrap_err(),
            CliExit {
                code: 2,
                message: "ds4-server-rs: missing value for --host\n".to_string(),
            }
        );
        let unknown = parse(&["--bad"]).unwrap_err();
        assert_eq!(unknown.code, 2);
        assert!(unknown
            .message
            .starts_with("ds4-server-rs: unknown option: --bad\n"));
    }

    #[test]
    fn bind_host_matches_c_localhost_behavior() {
        assert_eq!(bind_host("localhost").unwrap(), Ipv4Addr::new(127, 0, 0, 1));
        assert_eq!(bind_host("127.0.0.1").unwrap(), Ipv4Addr::new(127, 0, 0, 1));
        assert!(bind_host("example.com").is_err());
    }
}
