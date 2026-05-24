use ds4_engine::{
    context_memory_estimate, ArgmaxOptions, Backend, Engine, EngineOptions, RuntimeGraphRoute,
    SamplingOptions, ThinkMode,
};
use ds4_gguf::cli_parse::{parse_cli_config, CliBackend, CliParseResult, CliRuntimeGraphRoute};
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
    let Some(prompt_text) = config.prompt.as_deref() else {
        eprintln!("ds4-rs: M8.13c one-shot implementation requires -p or --prompt-file");
        return Ok(99);
    };
    if config.dump_tokens || config.inspect {
        eprintln!("ds4-rs: M8.13c one-shot implementation supports generation only");
        return Ok(99);
    }
    let runtime_graph_route = map_runtime_graph_route(config.runtime_graph_route);
    if let Some(exit) = runtime_graph_route.fail_closed("ds4-cli-one-shot-rs") {
        eprint!("{}", exit.stderr);
        return Ok(exit.code);
    }

    let backend = map_backend(config.backend);
    log_context_memory(backend, config.ctx_size);
    let requested_think = map_think_mode(config.think_mode);
    let effective_think = requested_think.for_context(config.ctx_size);
    if requested_think == ThinkMode::Max && effective_think != ThinkMode::Max {
        eprintln!(
            "ds4: warning: --think-max needs --ctx >= {}; ctx={} uses normal thinking instead",
            ThinkMode::max_min_context(),
            config.ctx_size
        );
    }

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
    let prompt = engine.encode_chat_prompt(&config.system, prompt_text, effective_think)?;
    let generation = if config.temperature <= 0.0 {
        engine.generate_argmax_text(
            &prompt,
            ArgmaxOptions {
                n_predict: config.n_predict,
                ctx_size: config.ctx_size,
                think_mode: effective_think,
            },
        )
    } else {
        let Some(seed) = config.seed else {
            eprintln!("ds4-rs: M8.13c sampled one-shot implementation requires --seed");
            return Ok(99);
        };
        engine.generate_sampled_text(
            &prompt,
            SamplingOptions {
                n_predict: config.n_predict,
                ctx_size: config.ctx_size,
                think_mode: effective_think,
                temperature: config.temperature,
                top_p: config.top_p,
                min_p: config.min_p,
                seed,
            },
        )
    };
    io::stdout().lock().write_all(&generation.stdout)?;
    Ok(generation.exit_code)
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
