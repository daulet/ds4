use ds4_engine::interactive_cli::{repl_help, ReplAction, ReplCommandResult, ReplState};
use ds4_engine::{
    context_memory_estimate, Backend, Engine, EngineOptions, InteractiveTurnOptions,
    RuntimeGraphRoute, ThinkMode,
};
use ds4_gguf::cli_parse::{parse_cli_config, CliBackend, CliParseResult, CliRuntimeGraphRoute};
use std::fs;
use std::io::{self, BufRead, Write};
use std::process;
#[cfg(unix)]
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(unix)]
static INTERRUPTED: AtomicBool = AtomicBool::new(false);

#[cfg(unix)]
const SIGINT: i32 = 2;

#[cfg(unix)]
extern "C" fn handle_sigint(_: i32) {
    INTERRUPTED.store(true, Ordering::SeqCst);
}

#[cfg(unix)]
extern "C" {
    fn signal(signum: i32, handler: usize) -> usize;
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
    install_interrupt_handler();
    let config = match parse_cli_config(std::env::args().skip(1)) {
        Ok(config) => config,
        Err(result) => return Ok(write_parse_result(result)?),
    };
    if config.prompt.is_some() || config.dump_tokens || config.inspect {
        eprintln!("ds4-rs: M8.15c interactive implementation supports no-prompt REPL only");
        return Ok(99);
    }
    let runtime_graph_route = map_runtime_graph_route(config.runtime_graph_route);
    if let Some(exit) = runtime_graph_route.fail_closed("ds4-cli-interactive-rs") {
        eprint!("{}", exit.stderr);
        return Ok(exit.code);
    }

    let backend = map_backend(config.backend);
    log_context_memory(backend, config.ctx_size);
    let mut options = EngineOptions::new(&config.model_path, backend);
    options.mtp_path = config.mtp_path.as_deref();
    options.n_threads = config.n_threads;
    options.mtp_draft_tokens = config.mtp_draft_tokens;
    options.mtp_margin = config.mtp_margin;
    options.directional_steering_file = config.directional_steering_file.as_deref();
    options.directional_steering_attn = config.directional_steering_attn;
    options.directional_steering_ffn = config.directional_steering_ffn;
    options.warm_weights = config.warm_weights;
    options.quality = config.quality;

    let engine = match Engine::open(&options) {
        Ok(engine) => engine,
        Err(err) if err.open_failed_code().is_some() => return Ok(1),
        Err(err) => return Err(Box::new(err)),
    };
    let initial_think = map_think_mode(config.think_mode);
    let mut state = ReplState::new(initial_think, config.ctx_size);
    let mut chat = engine.create_chat_session(&config.system, config.ctx_size, initial_think)?;
    chat.set_ctx(config.ctx_size)?;

    print!("{}", repl_help());
    io::stdout().flush()?;
    repl_loop(&mut chat, &mut state, &config, backend)
}

fn repl_loop(
    chat: &mut ds4_engine::ChatSession<'_>,
    state: &mut ReplState,
    config: &ds4_gguf::cli_parse::CliConfig,
    backend: Backend,
) -> Result<i32, Box<dyn std::error::Error>> {
    let stdin = io::stdin();
    let mut input = stdin.lock();
    loop {
        print!("ds4> ");
        io::stdout().flush()?;
        let mut line = String::new();
        match input.read_line(&mut line) {
            Ok(0) => return Ok(0),
            Ok(_) => {}
            Err(err) if err.kind() == io::ErrorKind::Interrupted => {
                take_interrupt();
                println!();
                continue;
            }
            Err(err) => return Err(Box::new(err)),
        }
        if take_interrupt() {
            println!();
            if line.trim().is_empty() {
                continue;
            }
            println!("ds4> {}", line.trim_end_matches(['\r', '\n']));
        }

        let command = state.handle_line(&line);
        match command.action.clone() {
            ReplAction::Continue => write_command_result(&command)?,
            ReplAction::Exit(code) => {
                write_command_result(&command)?;
                return Ok(code);
            }
            ReplAction::Quit => return Ok(0),
            ReplAction::SetContext(ctx_size) => {
                log_context_memory(backend, ctx_size);
                chat.set_ctx(ctx_size)?;
                write_command_result(&command)?;
            }
            ReplAction::ReadFile(path) => {
                write_command_result(&command)?;
                match fs::read_to_string(&path) {
                    Ok(prompt) => {
                        let rc = run_turn(chat, state.think_mode, &prompt, config)?;
                        if rc != 0 {
                            return Ok(rc);
                        }
                    }
                    Err(_) => {
                        eprintln!("ds4: failed to open prompt file: {path}");
                    }
                }
            }
            ReplAction::RunPrompt(prompt) => {
                write_command_result(&command)?;
                let rc = run_turn(chat, state.think_mode, &prompt, config)?;
                if rc != 0 {
                    return Ok(rc);
                }
            }
        }
    }
}

fn run_turn(
    chat: &mut ds4_engine::ChatSession<'_>,
    think_mode: ThinkMode,
    prompt: &str,
    config: &ds4_gguf::cli_parse::CliConfig,
) -> io::Result<i32> {
    let turn_options = InteractiveTurnOptions {
        n_predict: config.n_predict,
        think_mode,
        temperature: config.temperature,
        top_p: config.top_p,
        min_p: config.min_p,
        seed: config.seed.unwrap_or(1),
    };
    let mut stdout = io::stdout().lock();
    chat.run_turn_to_writer(prompt, turn_options, &mut stdout)
}

fn write_command_result(result: &ReplCommandResult) -> io::Result<()> {
    if !result.stderr.is_empty() {
        eprint!("{}", result.stderr);
        io::stderr().flush()?;
    }
    if !result.stdout.is_empty() {
        print!("{}", result.stdout);
        io::stdout().flush()?;
    }
    Ok(())
}

fn map_backend(backend: CliBackend) -> Backend {
    match backend {
        CliBackend::Metal => Backend::Metal,
        CliBackend::Cuda => Backend::Cuda,
        CliBackend::Cpu => Backend::Cpu,
    }
}

fn map_runtime_graph_route(route: CliRuntimeGraphRoute) -> RuntimeGraphRoute {
    match route {
        CliRuntimeGraphRoute::TargetStream => RuntimeGraphRoute::TargetStream,
        CliRuntimeGraphRoute::Graph => RuntimeGraphRoute::Graph,
    }
}

fn map_think_mode(mode: ds4_gguf::ThinkMode) -> ThinkMode {
    match mode {
        ds4_gguf::ThinkMode::None => ThinkMode::None,
        ds4_gguf::ThinkMode::High => ThinkMode::High,
        ds4_gguf::ThinkMode::Max => ThinkMode::Max,
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

fn write_parse_result(result: CliParseResult) -> io::Result<i32> {
    let mut stdout = io::stdout().lock();
    let mut stderr = io::stderr().lock();
    stdout.write_all(result.stdout.as_bytes())?;
    stderr.write_all(result.stderr.as_bytes())?;
    Ok(result.exit_code)
}

#[cfg(unix)]
fn install_interrupt_handler() {
    unsafe {
        signal(SIGINT, handle_sigint as *const () as usize);
    }
}

#[cfg(not(unix))]
fn install_interrupt_handler() {}

#[cfg(unix)]
fn take_interrupt() -> bool {
    INTERRUPTED.swap(false, Ordering::SeqCst)
}

#[cfg(not(unix))]
fn take_interrupt() -> bool {
    false
}
