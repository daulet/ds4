use std::io::{Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

struct TestServer {
    child: Child,
    port: u16,
}

impl Drop for TestServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl TestServer {
    fn spawn(args: &[&str]) -> Self {
        let port = free_port();
        let mut command = Command::new(env!("CARGO_BIN_EXE_ds4-server-rs"));
        command
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(port.to_string())
            .args(args)
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        let child = command.spawn().expect("spawn ds4-server-rs");
        wait_until_ready(port);
        Self { child, port }
    }

    fn send(&self, request: &str) -> String {
        let mut stream = TcpStream::connect(("127.0.0.1", self.port)).expect("connect server");
        stream.write_all(request.as_bytes()).expect("write request");
        stream.shutdown(Shutdown::Write).expect("shutdown write");
        let mut response = String::new();
        stream.read_to_string(&mut response).expect("read response");
        response
    }
}

fn free_port() -> u16 {
    TcpListener::bind(("127.0.0.1", 0))
        .expect("bind free port")
        .local_addr()
        .expect("local addr")
        .port()
}

fn wait_until_ready(port: u16) {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match TcpStream::connect(("127.0.0.1", port)) {
            Ok(stream) => {
                drop(stream);
                return;
            }
            Err(err) if Instant::now() < deadline => {
                let _ = err;
                thread::sleep(Duration::from_millis(20));
            }
            Err(err) => panic!("server did not start: {err}"),
        }
    }
}

fn post(path: &str, body: &str) -> String {
    format!(
        "POST {path} HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
}

#[test]
fn socket_replay_models_preflight_and_unknown_routes() {
    let server = TestServer::spawn(&["--ctx", "16", "--tokens", "64", "--cors"]);

    assert_eq!(
        server.send("OPTIONS /anything HTTP/1.1\r\nHost: x\r\n\r\n"),
        "HTTP/1.1 204 No Content\r\nContent-Length: 0\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: *\r\nConnection: close\r\n\r\n"
    );

    let models = server.send("GET /v1/models?probe=1 HTTP/1.1\r\nHost: x\r\n\r\n");
    assert!(models.starts_with("HTTP/1.1 200 OK\r\n"));
    assert!(models.contains("Content-Type: application/json\r\n"));
    assert!(models.contains("Access-Control-Allow-Origin: *\r\n"));
    assert!(models.contains("\"context_length\":16"));
    assert!(models.contains("\"max_completion_tokens\":16"));

    assert_eq!(
        server.send("GET /missing HTTP/1.1\r\nHost: x\r\n\r\n"),
        "HTTP/1.1 404 Not Found\r\nContent-Length: 72\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: *\r\nConnection: close\r\n\r\n{\"error\":{\"message\":\"unknown endpoint\",\"type\":\"invalid_request_error\"}}\n"
    );
}

#[test]
fn socket_replay_generation_errors_and_context_limit() {
    let server = TestServer::spawn(&["--ctx", "1"]);

    assert_eq!(
        server.send("GET\r\n\r\n"),
        "HTTP/1.1 400 Bad Request\r\nContent-Length: 72\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{\"error\":{\"message\":\"bad HTTP request\",\"type\":\"invalid_request_error\"}}\n"
    );

    assert_eq!(
        server.send(&post("/v1/chat/completions", "{}")),
        "HTTP/1.1 400 Bad Request\r\nContent-Length: 72\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{\"error\":{\"message\":\"missing messages\",\"type\":\"invalid_request_error\"}}\n"
    );

    assert_eq!(
        server.send(&post(
            "/v1/responses",
            r#"{"input":"hi","previous_response_id":"resp_1"}"#
        )),
        "HTTP/1.1 400 Bad Request\r\nContent-Length: 120\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{\"error\":{\"message\":\"previous_response_id is not supported; replay full input instead\",\"type\":\"invalid_request_error\"}}\n"
    );

    assert_eq!(
        server.send(&post(
            "/v1/responses",
            r#"{"input":"hi","tool_choice":"required"}"#
        )),
        "HTTP/1.1 400 Bad Request\r\nContent-Length: 90\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{\"error\":{\"message\":\"tool_choice=required not supported\",\"type\":\"invalid_request_error\"}}\n"
    );

    assert_eq!(
        server.send(&post("/v1/completions", r#"{"prompt":"hi"}"#)),
        "HTTP/1.1 400 Bad Request\r\nContent-Length: 200\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{\"error\":{\"message\":\"Prompt has 5 tokens, but the configured context size is 1 tokens\",\"type\":\"invalid_request_error\",\"param\":\"prompt\",\"code\":\"context_length_exceeded\",\"n_prompt_tokens\":5,\"n_ctx\":1}}\n"
    );
}

#[test]
fn socket_replay_rejects_valid_generation_without_model() {
    let server = TestServer::spawn(&["--ctx", "32768"]);
    assert_eq!(
        server.send(&post(
            "/v1/chat/completions",
            r#"{"messages":[{"role":"user","content":"hi"}]}"#
        )),
        "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 102\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{\"error\":{\"message\":\"generation is not available in no-model server\",\"type\":\"invalid_request_error\"}}\n"
    );
}
