use ds4_engine::{
    context_memory_estimate, Backend, Engine, EngineOptions, KvDiskCache, KvDiskCacheLoad,
    KvDiskCacheOptions, KvDiskCacheTrailerHooks, RuntimeGraphRoute, ServerCacheProbe,
    ServerGenerationOptions, ServerSession, ThinkMode, RUNTIME_GRAPH_ROUTE_UNSUPPORTED_CODE,
    RUNTIME_GRAPH_ROUTE_VALID_VALUES,
};
use ds4_gguf::kv_policy::{
    write_tool_map_trailer, KvOptions, KvPolicyConfig, ToolMapEntry, DEFAULT_MB as KV_DEFAULT_MB,
    EXT_TOOL_MAP,
};
use ds4_gguf::{
    format_http_error, format_http_response, format_openai_chat_completion_http,
    format_openai_chat_stream_http, format_openai_chat_tool_completion_http,
    format_openai_chat_tool_stream_http, openai_context_length_error_body,
    parse_generated_message_for_response, parse_http_request, parse_openai_chat_request,
    render_chat_prompt_text, request_exceeds_context,
    route_no_model_server_request_with_generation_message, utf8_stream_safe_len, ChatMessage,
    DsmlJsonCall, HttpRequest, HttpRequestParseError, NoModelRouteConfig, OpenAiChatCompletion,
    OpenAiChatRequest, OpenAiChatStream, OpenAiChatToolStream, OpenAiToolCallStreamEvent,
    OpenAiToolCallStreamEventOwned, OpenAiToolCallStreamTranslator, OpenAiUsage, ToolMemory,
    ToolMemorySource, ToolReplayStats, TOOL_MEMORY_DEFAULT_MAX_IDS, TOOL_MEMORY_MAX_BYTES,
};
use std::env;
use std::ffi::{c_char, c_int, c_void, CStr};
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::net::{Ipv4Addr, Shutdown, SocketAddrV4, TcpListener, TcpStream};
use std::process;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const READ_TIMEOUT: Duration = Duration::from_secs(10);
const WRITE_TIMEOUT: Duration = Duration::from_secs(10);
const MODEL_BACKED_GENERATION_MESSAGE: &str = "model-backed chat generation is not implemented yet";
const SIGINT: c_int = 2;
const SIGTERM: c_int = 15;

static STOP_REQUESTED: AtomicBool = AtomicBool::new(false);

unsafe extern "C" {
    fn fwrite(ptr: *const c_void, size: usize, nmemb: usize, stream: *mut c_void) -> usize;
    fn signal(
        signum: c_int,
        handler: Option<unsafe extern "C" fn(c_int)>,
    ) -> Option<unsafe extern "C" fn(c_int)>;
}

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
    cache: RuntimeCacheConfig,
    runtime_graph_route: RuntimeGraphRoute,
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
            cache: RuntimeCacheConfig::default(),
            runtime_graph_route: RuntimeGraphRoute::TargetStream,
            enable_cors: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RuntimeCacheConfig {
    disk_dir: Option<String>,
    disk_space_mb: u64,
    policy: KvOptions,
    reject_different_quant: bool,
    disable_exact_dsml_tool_replay: bool,
    tool_memory_max_ids: usize,
}

impl Default for RuntimeCacheConfig {
    fn default() -> Self {
        Self {
            disk_dir: None,
            disk_space_mb: 0,
            policy: KvOptions::default(),
            reject_different_quant: false,
            disable_exact_dsml_tool_replay: false,
            tool_memory_max_ids: TOOL_MEMORY_DEFAULT_MAX_IDS,
        }
    }
}

impl RuntimeCacheConfig {
    fn disk_enabled(&self) -> bool {
        self.disk_dir.is_some()
    }

    fn effective_disk_space_mb(&self) -> u64 {
        if !self.disk_enabled() {
            0
        } else if self.disk_space_mb == 0 {
            KV_DEFAULT_MB
        } else {
            self.disk_space_mb
        }
    }

    fn policy_config(&self) -> KvPolicyConfig {
        KvPolicyConfig {
            enabled: self.disk_enabled(),
            budget_bytes: self.effective_disk_space_mb().saturating_mul(1024 * 1024),
            reject_different_quant: self.reject_different_quant,
            options: self.policy,
            continued_last_store_tokens: 0,
        }
    }

    fn open_disk_cache(&self) -> Option<KvDiskCache> {
        let dir = self.disk_dir.as_deref()?;
        let options = KvDiskCacheOptions {
            dir,
            budget_mb: self.effective_disk_space_mb(),
            reject_different_quant: self.reject_different_quant,
            min_tokens: self.policy.min_tokens,
            cold_max_tokens: self.policy.cold_max_tokens,
            continued_interval_tokens: self.policy.continued_interval_tokens,
            boundary_trim_tokens: self.policy.boundary_trim_tokens,
            boundary_align_tokens: self.policy.boundary_align_tokens,
        };
        match KvDiskCache::open(&options) {
            Ok(cache) => cache,
            Err(err) => {
                eprintln!("ds4-server-runtime-rs: failed to open disk KV cache: {err}");
                None
            }
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
    STOP_REQUESTED.store(false, Ordering::SeqCst);
    install_stop_signal_handlers();
    let config = match parse_args(env::args().skip(1)) {
        Ok(Some(config)) => config,
        Ok(None) => return Ok(0),
        Err(exit) => return Ok(write_exit(exit)?),
    };
    if config.runtime_graph_route == RuntimeGraphRoute::Graph && config.backend == Backend::Cpu {
        return Ok(write_exit(CliExit {
            code: RUNTIME_GRAPH_ROUTE_UNSUPPORTED_CODE,
            stdout: String::new(),
            stderr: "ds4-server-runtime-rs: --runtime-graph graph requires cuda or metal backend\n"
                .to_string(),
        })?);
    }

    let engine_options = engine_options_from_config(&config);
    let engine = match Engine::open(&engine_options) {
        Ok(engine) => engine,
        Err(err) if err.open_failed_code().is_some() => return Ok(1),
        Err(err) => return Err(Box::new(err)),
    };
    log_context_memory(config.backend, config.context_length);
    let session = engine.create_server_session(config.context_length)?;
    serve(config, &engine, session)?;
    Ok(0)
}

fn install_stop_signal_handlers() {
    unsafe {
        signal(SIGINT, Some(handle_stop_signal));
        signal(SIGTERM, Some(handle_stop_signal));
    }
}

unsafe extern "C" fn handle_stop_signal(_signal: c_int) {
    STOP_REQUESTED.store(true, Ordering::SeqCst);
}

fn serve<'a>(
    config: ServerConfig,
    engine: &'a Engine,
    session: ServerSession<'a>,
) -> io::Result<()> {
    let host = bind_host(&config.host)?;
    let addr = SocketAddrV4::new(host, config.port);
    let listener = TcpListener::bind(addr)?;
    listener.set_nonblocking(true)?;
    let actual_addr = listener.local_addr()?;
    eprintln!("ds4-server-runtime-rs: listening on http://{actual_addr}");

    let mut state = RuntimeState {
        sequence: 0,
        trace: match config.trace_path.as_deref() {
            Some(path) => Some(File::create(path)?),
            None => None,
        },
        session,
        cache: RuntimeCacheState::new(config.cache.clone()),
    };
    while !STOP_REQUESTED.load(Ordering::SeqCst) {
        match listener.accept() {
            Ok((mut stream, _addr)) => {
                if let Err(err) = handle_client(&mut stream, &config, engine, &mut state) {
                    eprintln!("ds4-server-runtime-rs: client error: {err}");
                }
            }
            Err(err) if err.kind() == io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(50));
            }
            Err(err) if err.kind() == io::ErrorKind::Interrupted => {}
            Err(err) => {
                eprintln!("ds4-server-runtime-rs: accept failed: {err}");
            }
        }
    }
    Ok(())
}

struct RuntimeState<'a> {
    sequence: u64,
    trace: Option<File>,
    session: ServerSession<'a>,
    cache: RuntimeCacheState,
}

impl Drop for RuntimeState<'_> {
    fn drop(&mut self) {
        self.cache.store_current(&mut self.session, "shutdown");
    }
}

#[derive(Debug)]
struct RuntimeCacheState {
    config: RuntimeCacheConfig,
    disk: Option<KvDiskCache>,
    tool_memory: ToolMemory,
    ledger: Vec<RuntimeCacheLedgerEvent>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RuntimeCacheLedgerEvent {
    name: &'static str,
    cache_source: Option<&'static str>,
    reason: Option<&'static str>,
    tokens: i32,
    cached_tokens: i32,
    cache_write_tokens: i32,
    disk_cached_tokens: i32,
    frontier_before: i32,
    frontier_after: i32,
    success: Option<bool>,
}

impl RuntimeCacheState {
    fn new(config: RuntimeCacheConfig) -> Self {
        let disk = config.open_disk_cache();
        let tool_memory =
            ToolMemory::with_limits(config.tool_memory_max_ids, TOOL_MEMORY_MAX_BYTES);
        Self {
            config,
            disk,
            tool_memory,
            ledger: Vec::new(),
        }
    }

    fn begin_request_ledger(&mut self) {
        self.ledger.clear();
    }

    fn ledger_events(&self) -> &[RuntimeCacheLedgerEvent] {
        &self.ledger
    }

    fn continued_frontier(&self) -> i32 {
        self.disk
            .as_ref()
            .map_or(0, KvDiskCache::continued_last_store_tokens)
    }

    fn push_ledger_event(&mut self, event: RuntimeCacheLedgerEvent) {
        self.ledger.push(event);
    }

    fn record_frontier_event(
        &mut self,
        name: &'static str,
        reason: Option<&'static str>,
        tokens: i32,
        frontier_before: i32,
        success: Option<bool>,
    ) {
        self.push_ledger_event(RuntimeCacheLedgerEvent {
            name,
            cache_source: None,
            reason,
            tokens,
            cached_tokens: -1,
            cache_write_tokens: -1,
            disk_cached_tokens: -1,
            frontier_before,
            frontier_after: self.continued_frontier(),
            success,
        });
    }

    fn record_cache_decision(
        &mut self,
        cache_source: &'static str,
        prompt_tokens: i32,
        cached_tokens: i32,
        cache_write_tokens: i32,
        disk_cached_tokens: i32,
    ) {
        let frontier = self.continued_frontier();
        self.push_ledger_event(RuntimeCacheLedgerEvent {
            name: "cache_decision",
            cache_source: Some(cache_source),
            reason: None,
            tokens: prompt_tokens,
            cached_tokens,
            cache_write_tokens,
            disk_cached_tokens,
            frontier_before: frontier,
            frontier_after: frontier,
            success: None,
        });
    }

    fn prepare_chat_prompt(&mut self, request: &mut OpenAiChatRequest) -> ToolReplayStats {
        if self.config.disable_exact_dsml_tool_replay {
            return ToolReplayStats::default();
        }
        self.restore_tool_maps_for_messages(&request.messages);
        let stats = self.tool_memory.attach_to_messages(&mut request.messages);
        let active_tool_schemas = if request.has_tools {
            request.tool_schemas.as_deref()
        } else {
            None
        };
        request.prompt_text =
            render_chat_prompt_text(&request.messages, active_tool_schemas, request.think_mode);
        stats
    }

    fn try_load_disk_text(
        &mut self,
        session: &mut ServerSession<'_>,
        prompt_text: &str,
    ) -> Option<KvDiskCacheLoad> {
        let cache = self.disk.as_mut()?;
        match session.try_load_text_cache(cache, prompt_text, false) {
            Ok(load) => load,
            Err(err) => {
                eprintln!("ds4-server-runtime-rs: failed to load disk KV cache: {err}");
                None
            }
        }
    }

    fn reset_continued_frontier(&mut self) {
        let before = self.continued_frontier();
        if let Some(cache) = self.disk.as_mut() {
            cache.reset_continued_frontier();
        }
        self.record_frontier_event("reset_continued_frontier", None, 0, before, Some(true));
    }

    fn store_current(&mut self, session: &mut ServerSession<'_>, reason: &'static str) -> bool {
        let before = self.continued_frontier();
        let tokens = session.position();
        let stored = {
            let RuntimeCacheState {
                config,
                disk,
                tool_memory,
                ledger: _,
            } = self;
            if let Some(cache) = disk.as_mut() {
                let mut ctx = ToolMapTrailerContext {
                    tool_memory,
                    disabled: config.disable_exact_dsml_tool_replay,
                };
                let hooks = tool_map_trailer_hooks(&mut ctx);
                match session.store_current(cache, reason, Some(&hooks)) {
                    Ok(stored) => stored,
                    Err(err) => {
                        eprintln!(
                            "ds4-server-runtime-rs: failed to store disk KV cache ({reason}): {err}"
                        );
                        false
                    }
                }
            } else {
                false
            }
        };
        self.record_frontier_event("store_current", Some(reason), tokens, before, Some(stored));
        stored
    }

    fn suppress_continued_store(&mut self, tokens: i32) -> i32 {
        let before = self.continued_frontier();
        let old = if let Some(cache) = self.disk.as_mut() {
            cache.suppress_continued_store(tokens)
        } else {
            -1
        };
        self.record_frontier_event(
            "suppress_continued_store",
            None,
            tokens,
            before,
            Some(old >= 0),
        );
        old
    }

    fn note_store(&mut self, tokens: i32) {
        let before = self.continued_frontier();
        if let Some(cache) = self.disk.as_mut() {
            cache.note_store(tokens);
        }
        self.record_frontier_event(
            "note_store",
            None,
            tokens,
            before,
            Some(self.continued_frontier() > before),
        );
    }

    fn restore_suppressed_continued(&mut self, old_tokens: i32, suppressed_tokens: i32) {
        let before = self.continued_frontier();
        if let Some(cache) = self.disk.as_mut() {
            cache.restore_suppressed_continued(old_tokens, suppressed_tokens);
        }
        self.record_frontier_event(
            "restore_suppressed_continued",
            None,
            suppressed_tokens,
            before,
            Some(self.continued_frontier() != before),
        );
    }

    fn store_live_prefix(
        &mut self,
        session: &mut ServerSession<'_>,
        prompt: &ds4_engine::Tokens,
        store_len: i32,
        reason: &'static str,
    ) -> bool {
        let before = self.continued_frontier();
        let stored = {
            let RuntimeCacheState {
                config,
                disk,
                tool_memory,
                ledger: _,
            } = self;
            if let Some(cache) = disk.as_mut() {
                let mut ctx = ToolMapTrailerContext {
                    tool_memory,
                    disabled: config.disable_exact_dsml_tool_replay,
                };
                let hooks = tool_map_trailer_hooks(&mut ctx);
                match session.store_live_prefix(cache, prompt, store_len, reason, Some(&hooks)) {
                    Ok(stored) => stored,
                    Err(err) => {
                        eprintln!(
                            "ds4-server-runtime-rs: failed to store disk KV cache ({reason}): {err}"
                        );
                        false
                    }
                }
            } else {
                false
            }
        };
        self.record_frontier_event(
            "store_live_prefix",
            Some(reason),
            store_len,
            before,
            Some(stored),
        );
        stored
    }

    fn maybe_store_continued(&mut self, session: &mut ServerSession<'_>) -> bool {
        let before = self.continued_frontier();
        let tokens = session.position();
        let stored = {
            let RuntimeCacheState {
                config,
                disk,
                tool_memory,
                ledger: _,
            } = self;
            if let Some(cache) = disk.as_mut() {
                let mut ctx = ToolMapTrailerContext {
                    tool_memory,
                    disabled: config.disable_exact_dsml_tool_replay,
                };
                let hooks = tool_map_trailer_hooks(&mut ctx);
                match session.maybe_store_continued(cache, Some(&hooks)) {
                    Ok(stored) => stored,
                    Err(err) => {
                        eprintln!(
                            "ds4-server-runtime-rs: failed to store continued disk KV cache: {err}"
                        );
                        false
                    }
                }
            } else {
                false
            }
        };
        self.record_frontier_event(
            "maybe_store_continued",
            Some("continued"),
            tokens,
            before,
            Some(stored),
        );
        stored
    }

    fn remember_generated_tool_calls(&mut self, parsed: &ParsedChatGeneration) {
        if self.config.disable_exact_dsml_tool_replay {
            return;
        }
        let Some(raw_dsml) = parsed.raw_dsml.as_deref().filter(|raw| !raw.is_empty()) else {
            return;
        };
        let ids: Vec<&str> = parsed
            .calls
            .iter()
            .filter_map(|call| call.id.as_deref())
            .filter(|id| !id.is_empty())
            .collect();
        self.tool_memory
            .remember_ids(ids, raw_dsml, ToolMemorySource::Ram);
    }

    fn sync_prompt_with_stores(
        &mut self,
        session: &mut ServerSession<'_>,
        engine: &Engine,
        prompt: &ds4_engine::Tokens,
        cached_tokens: i32,
    ) -> Result<(), String> {
        let cold_store_len = self.cold_store_len(engine, prompt, cached_tokens);
        let mut suppressed_continued_last = -1;
        if cold_store_len >= self.config.policy.min_tokens {
            suppressed_continued_last = self.suppress_continued_store(cold_store_len);
        }

        if cold_store_len >= self.config.policy.min_tokens && cold_store_len < prompt.len() {
            if let Err(err) = session.sync_prompt_prefix(prompt, cold_store_len) {
                self.restore_suppressed_continued(suppressed_continued_last, cold_store_len);
                return Err(err.to_string());
            }
            if self.store_live_prefix(session, prompt, cold_store_len, "cold") {
                self.note_store(cold_store_len);
                suppressed_continued_last = -1;
            } else {
                self.restore_suppressed_continued(suppressed_continued_last, cold_store_len);
                suppressed_continued_last = -1;
            }
        }

        if let Err(err) = session.sync_prompt(prompt) {
            self.restore_suppressed_continued(suppressed_continued_last, cold_store_len);
            return Err(err.to_string());
        }
        self.maybe_store_continued(session);

        if cold_store_len == prompt.len() {
            if self.store_live_prefix(session, prompt, cold_store_len, "cold") {
                self.note_store(cold_store_len);
            } else {
                self.restore_suppressed_continued(suppressed_continued_last, cold_store_len);
            }
        }
        Ok(())
    }

    fn cold_store_len(
        &self,
        engine: &Engine,
        prompt: &ds4_engine::Tokens,
        cached_tokens: i32,
    ) -> i32 {
        let Some(cache) = self.disk.as_ref() else {
            return 0;
        };
        let policy = self.config.policy;
        if cached_tokens != 0
            || prompt.len() < policy.min_tokens
            || policy.cold_max_tokens <= 0
            || prompt.len() > policy.cold_max_tokens
        {
            return 0;
        }
        let anchor = cache.chat_anchor_pos(engine, prompt);
        if anchor >= policy.min_tokens {
            anchor
        } else {
            cache.store_len(prompt.len())
        }
    }

    fn trace_decision(
        &self,
        prompt_tokens: i32,
        cache_probe: ServerCacheProbe,
        generated: &ds4_engine::ServerGenerationResult,
        tool_replay: ToolReplayStats,
        disk_load: Option<&KvDiskCacheLoad>,
    ) -> RuntimeCacheDecision {
        let mut decision = RuntimeCacheDecision::from_runtime(
            prompt_tokens,
            cache_probe,
            generated,
            tool_replay,
            disk_load,
        );
        if !self.config.policy_config().enabled {
            decision.disk_cached_tokens = 0;
            decision.disk_cache_file = None;
        }
        decision
    }

    fn restore_tool_maps_for_messages(&mut self, messages: &[ChatMessage]) {
        if self.disk.is_none() {
            return;
        }
        let Some(dir) = self.config.disk_dir.as_deref() else {
            return;
        };
        let Ok(entries) = fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if !is_kv_cache_file_name(name) {
                continue;
            }
            let Ok(bytes) = fs::read(&path) else {
                continue;
            };
            self.tool_memory
                .restore_tool_map_from_kvc_for_messages(&bytes, messages);
        }
    }
}

struct ToolMapTrailerContext<'a> {
    tool_memory: &'a ToolMemory,
    disabled: bool,
}

impl ToolMapTrailerContext<'_> {
    fn trailer_for_text(&self, text: *const c_char) -> Option<Vec<u8>> {
        if text.is_null() {
            return Some(Vec::new());
        }
        let text = unsafe { CStr::from_ptr(text).to_bytes() };
        let entries: Vec<ToolMapEntry> = self.tool_memory.tool_map_entries();
        write_tool_map_trailer(text, &entries, self.disabled)
    }
}

fn tool_map_trailer_hooks(ctx: &mut ToolMapTrailerContext<'_>) -> KvDiskCacheTrailerHooks {
    KvDiskCacheTrailerHooks::new(
        (ctx as *mut ToolMapTrailerContext<'_>).cast(),
        EXT_TOOL_MAP,
        Some(tool_map_trailer_serialized_size),
        Some(tool_map_trailer_write),
    )
}

unsafe extern "C" fn tool_map_trailer_serialized_size(
    ud: *mut c_void,
    text: *const c_char,
    bytes_out: *mut u64,
) -> bool {
    if !bytes_out.is_null() {
        *bytes_out = 0;
    }
    let Some(ctx) = (ud as *mut ToolMapTrailerContext<'_>).as_ref() else {
        return true;
    };
    let Some(trailer) = ctx.trailer_for_text(text) else {
        return false;
    };
    let Ok(len) = u64::try_from(trailer.len()) else {
        return false;
    };
    if !bytes_out.is_null() {
        *bytes_out = len;
    }
    true
}

unsafe extern "C" fn tool_map_trailer_write(
    ud: *mut c_void,
    fp: *mut c_void,
    text: *const c_char,
    written_bytes: *mut u64,
) -> bool {
    if !written_bytes.is_null() {
        *written_bytes = 0;
    }
    let Some(ctx) = (ud as *mut ToolMapTrailerContext<'_>).as_ref() else {
        return true;
    };
    let Some(trailer) = ctx.trailer_for_text(text) else {
        return false;
    };
    if trailer.is_empty() {
        return true;
    }
    if fp.is_null() {
        return false;
    }
    let wrote = fwrite(trailer.as_ptr().cast(), 1, trailer.len(), fp);
    if wrote != trailer.len() {
        return false;
    }
    let Ok(len) = u64::try_from(trailer.len()) else {
        return false;
    };
    if !written_bytes.is_null() {
        *written_bytes = len;
    }
    true
}

fn is_kv_cache_file_name(name: &str) -> bool {
    let Some(sha) = name.strip_suffix(".kv") else {
        return false;
    };
    sha.len() == 40 && sha.bytes().all(|b| b.is_ascii_hexdigit())
}

fn memory_cached_tokens(cache_probe: ServerCacheProbe, prompt_tokens: i32) -> i32 {
    if cache_probe.live_tokens_before > 0
        && cache_probe.live_prompt_common == cache_probe.live_tokens_before
        && prompt_tokens >= cache_probe.live_tokens_before
    {
        cache_probe.live_prompt_common
    } else {
        0
    }
}

fn should_try_disk_text_cache(memory_cached_tokens: i32) -> bool {
    memory_cached_tokens == 0
}

fn handle_client(
    stream: &mut TcpStream,
    config: &ServerConfig,
    engine: &Engine,
    state: &mut RuntimeState<'_>,
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
    state: &mut RuntimeState<'_>,
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
    state: &mut RuntimeState<'_>,
) -> String {
    let mut parsed = match parse_openai_chat_request(
        &request.body,
        config.default_tokens,
        config.context_length,
    ) {
        Ok(parsed) => parsed,
        Err(err) => return format_http_error(config.enable_cors, 400, err.message()),
    };
    state.cache.begin_request_ledger();
    let tool_replay = state.cache.prepare_chat_prompt(&mut parsed);
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

    let prompt_cache_probe = state.session.cache_probe(&prompt);
    let memory_cached_tokens = memory_cached_tokens(prompt_cache_probe, prompt.len());
    if memory_cached_tokens == 0 {
        state.cache.reset_continued_frontier();
        if prompt_cache_probe.live_tokens_before >= config.cache.policy.min_tokens {
            state.cache.store_current(&mut state.session, "evict");
        }
    }
    let disk_load = if should_try_disk_text_cache(memory_cached_tokens) {
        state
            .cache
            .try_load_disk_text(&mut state.session, &parsed.prompt_text)
    } else {
        None
    };
    let prompt_for_generation = disk_load
        .as_ref()
        .map(|load| &load.effective_prompt)
        .unwrap_or(&prompt);
    let cached_tokens = disk_load
        .as_ref()
        .map_or(memory_cached_tokens, |load| load.tokens.max(0));
    let disk_cached_tokens = disk_load.as_ref().map_or(0, |load| load.tokens.max(0));
    let cache_source = if disk_cached_tokens > 0 {
        "disk-text"
    } else if memory_cached_tokens > 0 {
        "memory-token"
    } else {
        "none"
    };
    let cache_write_tokens = (prompt_for_generation.len() - cached_tokens).max(0);
    state.cache.record_cache_decision(
        cache_source,
        prompt_for_generation.len(),
        cached_tokens,
        cache_write_tokens,
        disk_cached_tokens,
    );
    let generation_cache_probe = state.session.cache_probe(prompt_for_generation);
    if let Err(err) = state.cache.sync_prompt_with_stores(
        &mut state.session,
        engine,
        prompt_for_generation,
        cached_tokens,
    ) {
        eprintln!("ds4-server-runtime-rs: prompt processing failed: {err}");
        return format_http_error(config.enable_cors, 500, "generation failed");
    }
    let generation_options = ServerGenerationOptions {
        n_predict: parsed.max_tokens,
        ctx_size: config.context_length,
        temperature: parsed.sampling.temperature,
        top_k: parsed.sampling.top_k,
        top_p: parsed.sampling.top_p,
        min_p: parsed.sampling.min_p,
        seed: parsed.seed,
    };
    let generated = {
        let has_tools = parsed.has_tools;
        let cache = &mut state.cache;
        let session = &mut state.session;
        session.generate_synced(
            prompt_for_generation.len(),
            cached_tokens,
            cache_write_tokens,
            generation_cache_probe.live_tokens_before,
            generation_cache_probe.live_prompt_common,
            generation_options,
            |session, generated_text| {
                if has_tools && generated_saw_dsml_tool_start(generated_text) {
                    return;
                }
                cache.maybe_store_continued(session);
            },
        )
    };
    if generated.exit_code != 0 || generated.finish_reason == "error" {
        return format_http_error(config.enable_cors, 500, "generation failed");
    }
    let generated_text = String::from_utf8_lossy(&generated.text);
    state.sequence += 1;
    let id = format!("chatcmpl-{}", state.sequence);
    let parsed_generation =
        parse_chat_generation(&parsed, &generated, &generated_text, state.sequence);
    state
        .cache
        .remember_generated_tool_calls(&parsed_generation);
    let cache_decision = state.cache.trace_decision(
        prompt.len(),
        prompt_cache_probe,
        &generated,
        tool_replay,
        disk_load.as_ref(),
    );
    let ledger_events = state.cache.ledger_events().to_vec();
    if let Some(trace) = state.trace.as_mut() {
        if let Err(err) = write_chat_trace_with_cache_decision(
            trace,
            state.sequence,
            &request.body,
            &parsed,
            &generated,
            &generated_text,
            &parsed_generation,
            &cache_decision,
            &ledger_events,
        ) {
            eprintln!("ds4-server-runtime-rs: failed to write trace: {err}");
        }
    }
    let created = unix_timestamp();
    let usage = OpenAiUsage::new(
        generated.prompt_tokens,
        generated.completion_tokens,
        generated.cache_read_tokens,
        generated.cache_write_tokens,
    );
    if parsed.stream {
        if parsed.has_tools
            && (parsed_generation.saw_tool_start || !parsed_generation.calls.is_empty())
        {
            format_streaming_tool_chat_http(
                config.enable_cors,
                &id,
                created,
                &parsed.model,
                parsed.stream_include_usage,
                usage,
                &generated,
                state.sequence,
                &parsed_generation,
            )
        } else {
            let delta_strings = match stream_delta_strings(&generated) {
                Ok(delta_strings) => delta_strings,
                Err(()) => return format_http_error(config.enable_cors, 500, "generation failed"),
            };
            let content_deltas = delta_strings.iter().map(String::as_str).collect::<Vec<_>>();
            format_openai_chat_stream_http(
                config.enable_cors,
                &OpenAiChatStream {
                    id: &id,
                    created,
                    model: &parsed.model,
                    content_deltas: &content_deltas,
                    finish_reason: &parsed_generation.finish_reason,
                    usage: if parsed.stream_include_usage {
                        Some(usage)
                    } else {
                        None
                    },
                },
            )
        }
    } else if !parsed_generation.calls.is_empty() {
        format_openai_chat_tool_completion_http(
            config.enable_cors,
            &OpenAiChatCompletion {
                id: &id,
                created,
                model: &parsed.model,
                content: &parsed_generation.content,
                reasoning_content: parsed_generation.reasoning.as_deref(),
                finish_reason: &parsed_generation.finish_reason,
                usage,
            },
            &parsed_generation.calls,
        )
    } else {
        format_openai_chat_completion_http(
            config.enable_cors,
            &OpenAiChatCompletion {
                id: &id,
                created,
                model: &parsed.model,
                content: &parsed_generation.content,
                reasoning_content: parsed_generation.reasoning.as_deref(),
                finish_reason: &parsed_generation.finish_reason,
                usage,
            },
        )
    }
}

fn unsupported_chat_generation_message(parsed: &OpenAiChatRequest) -> Option<&'static str> {
    if map_think_mode(parsed.think_mode).enabled() {
        Some("thinking chat generation is not implemented yet")
    } else if !parsed.stops.is_empty() {
        Some("stop sequences are not implemented yet")
    } else {
        None
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParsedChatGeneration {
    content: String,
    reasoning: Option<String>,
    calls: Vec<DsmlJsonCall>,
    raw_dsml: Option<String>,
    finish_reason: String,
    saw_tool_start: bool,
    saw_tool_end: bool,
}

fn parse_chat_generation(
    request: &OpenAiChatRequest,
    generated: &ds4_engine::ServerGenerationResult,
    text: &str,
    sequence: u64,
) -> ParsedChatGeneration {
    let saw_tool_start = saw_dsml_tool_start(text);
    let saw_tool_end = saw_dsml_tool_end(text);
    let response = parse_generated_message_for_response(
        text,
        request.has_tools,
        saw_tool_start,
        map_think_mode(request.think_mode).enabled(),
        generated.finish_reason,
    );
    let mut message = response.message;
    let mut finish_reason = response.finish;
    if !message.calls.is_empty() {
        assign_openai_tool_call_ids(&mut message.calls, sequence);
        finish_reason = "tool_calls".to_string();
    }
    ParsedChatGeneration {
        content: message.content,
        reasoning: message.reasoning,
        calls: message.calls,
        raw_dsml: message.raw_dsml,
        finish_reason,
        saw_tool_start,
        saw_tool_end,
    }
}

fn assign_openai_tool_call_ids(calls: &mut [DsmlJsonCall], sequence: u64) {
    for (index, call) in calls.iter_mut().enumerate() {
        if call.id.as_ref().is_some_and(|id| !id.is_empty()) {
            continue;
        }
        call.id = Some(format!("call_{sequence:016x}{index:016x}"));
    }
}

fn saw_dsml_tool_start(text: &str) -> bool {
    text.contains("<｜DSML｜tool_calls>")
        || text.contains("<DSML｜tool_calls>")
        || text.contains("<tool_calls>")
}

fn generated_saw_dsml_tool_start(text: &[u8]) -> bool {
    contains_bytes(text, "<｜DSML｜tool_calls>".as_bytes())
        || contains_bytes(text, "<DSML｜tool_calls>".as_bytes())
        || contains_bytes(text, b"<tool_calls>")
}

fn saw_dsml_tool_end(text: &str) -> bool {
    text.contains("</｜DSML｜tool_calls>")
        || text.contains("</DSML｜tool_calls>")
        || text.contains("</tool_calls>")
}

fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}

fn stream_delta_strings(generated: &ds4_engine::ServerGenerationResult) -> Result<Vec<String>, ()> {
    let mut raw = Vec::new();
    let mut emitted = 0;
    let mut deltas = Vec::new();
    for (index, token_text) in generated.token_texts.iter().enumerate() {
        raw.extend_from_slice(token_text);
        let final_chunk = index + 1 == generated.token_texts.len();
        let safe_len = utf8_stream_safe_len(&raw, emitted, raw.len(), final_chunk);
        if safe_len > emitted {
            let delta = std::str::from_utf8(&raw[emitted..safe_len]).map_err(|_| ())?;
            deltas.push(delta.to_string());
            emitted = safe_len;
        }
    }
    Ok(deltas)
}

struct ToolStreamTranslation {
    events: Vec<OpenAiToolCallStreamEventOwned>,
    emitted_any: bool,
}

fn translate_tool_stream_events(
    generated: &ds4_engine::ServerGenerationResult,
    sequence: u64,
) -> ToolStreamTranslation {
    let mut translator = OpenAiToolCallStreamTranslator::new(format!("call_{sequence:016x}"));
    let mut events = Vec::new();
    if generated.token_texts.is_empty() {
        events.extend(translator.feed(&generated.text));
    } else {
        for token_text in &generated.token_texts {
            events.extend(translator.feed(token_text));
        }
    }
    ToolStreamTranslation {
        events,
        emitted_any: translator.emitted_any(),
    }
}

fn format_streaming_tool_chat_http(
    enable_cors: bool,
    id: &str,
    created: i64,
    model: &str,
    stream_include_usage: bool,
    usage: OpenAiUsage,
    generated: &ds4_engine::ServerGenerationResult,
    sequence: u64,
    parsed_generation: &ParsedChatGeneration,
) -> String {
    let translation = translate_tool_stream_events(generated, sequence);
    let mut events = Vec::new();
    if !parsed_generation.content.is_empty() {
        events.push(OpenAiToolCallStreamEvent::Content {
            delta: &parsed_generation.content,
        });
    }
    events.extend(
        translation
            .events
            .iter()
            .map(OpenAiToolCallStreamEventOwned::as_borrowed),
    );
    if !translation.emitted_any && !parsed_generation.calls.is_empty() {
        events.push(OpenAiToolCallStreamEvent::FullCalls {
            calls: &parsed_generation.calls,
        });
    }
    format_openai_chat_tool_stream_http(
        enable_cors,
        &OpenAiChatToolStream {
            id,
            created,
            model,
            events: &events,
            finish_reason: &parsed_generation.finish_reason,
            usage: if stream_include_usage {
                Some(usage)
            } else {
                None
            },
        },
    )
}

#[cfg(test)]
fn write_chat_trace<W: Write>(
    trace: &mut W,
    sequence: u64,
    raw_body: &str,
    request: &OpenAiChatRequest,
    prompt_tokens: i32,
    generated: &ds4_engine::ServerGenerationResult,
    generated_text: &str,
    parsed: &ParsedChatGeneration,
) -> io::Result<()> {
    let cache = RuntimeCacheDecision::from_generation(prompt_tokens, generated);
    write_chat_trace_with_cache_decision(
        trace,
        sequence,
        raw_body,
        request,
        generated,
        generated_text,
        parsed,
        &cache,
        &[],
    )
}

fn write_chat_trace_with_cache_decision<W: Write>(
    trace: &mut W,
    sequence: u64,
    raw_body: &str,
    request: &OpenAiChatRequest,
    generated: &ds4_engine::ServerGenerationResult,
    generated_text: &str,
    parsed: &ParsedChatGeneration,
    cache: &RuntimeCacheDecision,
    ledger_events: &[RuntimeCacheLedgerEvent],
) -> io::Result<()> {
    writeln!(trace, "===== request {sequence} =====")?;
    writeln!(trace, "kind: chat")?;
    writeln!(trace, "model: {}", request.model)?;
    writeln!(trace, "stream: {}", if request.stream { 1 } else { 0 })?;
    writeln!(trace, "tools: {}", if request.has_tools { 1 } else { 0 })?;
    writeln!(trace, "think_mode: {}", request.think_mode.name())?;
    writeln!(trace, "prompt_tokens: {}", cache.prompt_tokens)?;
    writeln!(
        trace,
        "effective_prompt_tokens: {}",
        cache.effective_prompt_tokens
    )?;
    writeln!(trace, "cached_tokens: {}", cache.cached_tokens)?;
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
    writeln!(trace, "live_tokens_before: {}", cache.live_tokens_before)?;
    writeln!(trace, "prompt_tokens: {}", cache.prompt_tokens)?;
    writeln!(trace, "live_prompt_common: {}", cache.live_prompt_common)?;
    writeln!(
        trace,
        "memory_token_reusable: {}",
        if cache.memory_token_reusable() { 1 } else { 0 }
    )?;
    writeln!(trace, "memory_miss_reason: {}", cache.memory_miss_reason())?;
    writeln!(
        trace,
        "tool_replay: mem={} disk={} canonical={} missing_ids={}",
        cache.tool_replay.mem,
        cache.tool_replay.disk,
        cache.tool_replay.canonical,
        cache.tool_replay.missing_ids
    )?;
    writeln!(trace, "cache_source: {}", cache.cache_source)?;
    writeln!(trace, "cached_tokens: {}", cache.cached_tokens)?;
    writeln!(trace, "disk_cached_tokens: {}", cache.disk_cached_tokens)?;
    if let Some(path) = cache
        .disk_cache_file
        .as_deref()
        .filter(|path| !path.is_empty())
    {
        writeln!(trace, "disk_cache_file: {path}")?;
    }
    writeln!(trace)?;
    writeln!(trace, "--- runtime cache ledger ---")?;
    for (index, event) in ledger_events.iter().enumerate() {
        writeln!(
            trace,
            "event[{index}]: name={} cache_source={} reason={} tokens={} cached_tokens={} cache_write_tokens={} disk_cached_tokens={} frontier_before={} frontier_after={} success={}",
            event.name,
            event.cache_source.unwrap_or(""),
            event.reason.unwrap_or(""),
            event.tokens,
            event.cached_tokens,
            event.cache_write_tokens,
            event.disk_cached_tokens,
            event.frontier_before,
            event.frontier_after,
            optional_bool_text(event.success),
        )?;
    }
    writeln!(trace)?;
    writeln!(trace, "--- raw request json ---")?;
    writeln!(trace, "{raw_body}")?;
    writeln!(trace)?;
    writeln!(trace, "--- rendered prompt ---")?;
    writeln!(trace, "{}", request.prompt_text)?;
    writeln!(trace)?;
    writeln!(trace, "--- generated text ---")?;
    writeln!(trace, "{generated_text}")?;
    writeln!(trace)?;
    writeln!(trace, "--- parsed message ---")?;
    writeln!(trace, "finish: {}", parsed.finish_reason)?;
    writeln!(trace, "generated_tokens: {}", generated.completion_tokens)?;
    writeln!(
        trace,
        "dsml_start: {}",
        if parsed.saw_tool_start { 1 } else { 0 }
    )?;
    writeln!(
        trace,
        "dsml_end: {}",
        if parsed.saw_tool_end { 1 } else { 0 }
    )?;
    if let Some(reasoning) = parsed
        .reasoning
        .as_deref()
        .filter(|reasoning| !reasoning.is_empty())
    {
        writeln!(trace)?;
        writeln!(trace, "reasoning:")?;
        writeln!(trace, "{reasoning}")?;
    }
    if !parsed.content.is_empty() {
        writeln!(trace)?;
        writeln!(trace, "content:")?;
        writeln!(trace, "{}", parsed.content)?;
    }
    for (index, call) in parsed.calls.iter().enumerate() {
        writeln!(trace)?;
        writeln!(trace, "tool_call[{index}]:")?;
        writeln!(trace, "id: {}", call.id.as_deref().unwrap_or(""))?;
        writeln!(trace, "name: {}", call.name)?;
        writeln!(trace, "arguments:")?;
        writeln!(trace, "{}", call.arguments)?;
    }
    writeln!(trace)?;
    writeln!(trace, "===== end request {sequence} =====")?;
    writeln!(trace)?;
    trace.flush()
}

fn optional_bool_text(value: Option<bool>) -> &'static str {
    match value {
        Some(true) => "true",
        Some(false) => "false",
        None => "",
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RuntimeCacheDecision {
    live_tokens_before: i32,
    prompt_tokens: i32,
    effective_prompt_tokens: i32,
    live_prompt_common: i32,
    tool_replay: ToolReplayStats,
    cache_source: &'static str,
    cached_tokens: i32,
    disk_cached_tokens: i32,
    disk_cache_file: Option<String>,
}

impl RuntimeCacheDecision {
    #[cfg(test)]
    fn from_generation(prompt_tokens: i32, generated: &ds4_engine::ServerGenerationResult) -> Self {
        let cache_probe = ServerCacheProbe {
            live_tokens_before: generated.live_tokens_before,
            live_prompt_common: generated.live_prompt_common,
        };
        Self::from_runtime(
            prompt_tokens,
            cache_probe,
            generated,
            ToolReplayStats::default(),
            None,
        )
    }

    fn from_runtime(
        prompt_tokens: i32,
        cache_probe: ServerCacheProbe,
        generated: &ds4_engine::ServerGenerationResult,
        tool_replay: ToolReplayStats,
        disk_load: Option<&KvDiskCacheLoad>,
    ) -> Self {
        let disk_cached_tokens = disk_load.map_or(0, |load| load.tokens.max(0));
        let cached_tokens = if disk_cached_tokens > 0 {
            disk_cached_tokens
        } else {
            memory_cached_tokens(cache_probe, prompt_tokens)
        };
        let cache_source = if disk_cached_tokens > 0 {
            "disk-text"
        } else if cached_tokens > 0 {
            "memory-token"
        } else {
            "none"
        };
        Self {
            live_tokens_before: cache_probe.live_tokens_before.max(0),
            prompt_tokens,
            effective_prompt_tokens: generated.prompt_tokens,
            live_prompt_common: cache_probe.live_prompt_common.max(0),
            tool_replay,
            cache_source,
            cached_tokens,
            disk_cached_tokens,
            disk_cache_file: disk_load.and_then(|load| load.path.clone()),
        }
    }

    fn memory_token_reusable(&self) -> bool {
        self.live_tokens_before > 0
            && self.live_prompt_common == self.live_tokens_before
            && self.prompt_tokens >= self.live_tokens_before
    }

    fn memory_miss_reason(&self) -> &'static str {
        if self.memory_token_reusable() {
            "live-prefix-match"
        } else if self.live_tokens_before > 0 {
            "token-mismatch"
        } else {
            "no-live-checkpoint"
        }
    }
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
            "--runtime-graph" | "--runtime-graph-route" => {
                let value = need_arg(&mut args, &arg)?;
                config.runtime_graph_route =
                    RuntimeGraphRoute::parse(&value).ok_or_else(|| CliExit {
                        code: 2,
                        stdout: String::new(),
                        stderr: format!(
                            "ds4-server-runtime-rs: invalid runtime graph route: {value}\n\
                             ds4-server-runtime-rs: valid runtime graph routes are: {RUNTIME_GRAPH_ROUTE_VALID_VALUES}\n"
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
            "--kv-disk-dir" => {
                config.cache.disk_dir = Some(need_arg(&mut args, &arg)?);
            }
            "--kv-disk-space-mb" => {
                config.cache.disk_space_mb =
                    parse_positive_i32(&need_arg(&mut args, &arg)?, &arg)? as u64;
            }
            "--kv-cache-min-tokens" => {
                config.cache.policy.min_tokens =
                    parse_positive_i32(&need_arg(&mut args, &arg)?, &arg)?;
            }
            "--kv-cache-cold-max-tokens" => {
                config.cache.policy.cold_max_tokens =
                    parse_nonnegative_i32(&need_arg(&mut args, &arg)?, &arg)?;
            }
            "--kv-cache-continued-interval-tokens" => {
                config.cache.policy.continued_interval_tokens =
                    parse_nonnegative_i32(&need_arg(&mut args, &arg)?, &arg)?;
            }
            "--kv-cache-boundary-trim-tokens" => {
                config.cache.policy.boundary_trim_tokens =
                    parse_nonnegative_i32(&need_arg(&mut args, &arg)?, &arg)?;
            }
            "--kv-cache-boundary-align-tokens" => {
                config.cache.policy.boundary_align_tokens =
                    parse_nonnegative_i32(&need_arg(&mut args, &arg)?, &arg)?;
            }
            "--kv-cache-reject-different-quant" => {
                config.cache.reject_different_quant = true;
            }
            "--disable-exact-dsml-tool-replay" => {
                config.cache.disable_exact_dsml_tool_replay = true;
            }
            "--tool-memory-max-ids" => {
                config.cache.tool_memory_max_ids =
                    parse_positive_i32(&need_arg(&mut args, &arg)?, &arg)? as usize;
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
    if config.cache.policy.cold_max_tokens > 0
        && config.cache.policy.cold_max_tokens < config.cache.policy.min_tokens
    {
        return Err(CliExit {
            code: 2,
            stdout: String::new(),
            stderr:
                "ds4-server-runtime-rs: --kv-cache-cold-max-tokens must be 0 or >= --kv-cache-min-tokens\n"
                    .to_string(),
        });
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

fn parse_nonnegative_i32(value: &str, option: &str) -> Result<i32, CliExit> {
    match value.parse::<i64>() {
        Ok(value) if (0..=i32::MAX as i64).contains(&value) => Ok(value as i32),
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
  --runtime-graph ROUTE\n\
      Runtime graph route. Values: target-stream, off, graph. Default: target-stream\n\
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
      Write a human-readable session trace: prompts, cache decisions, output, tool calls.\n\
  -c, --ctx N\n\
      Context size used for request parsing and prompt-token limits. Default: 32768\n\
  -n, --tokens N\n\
      Default max output tokens when the client omits a limit. Default: 393216 (384K)\n\
\n\
Disk KV cache:\n\
  --kv-disk-dir DIR\n\
      Enable disk KV checkpoints in DIR. The directory is created if needed.\n\
  --kv-disk-space-mb N\n\
      Disk budget for checkpoint files. Default when enabled: 4096\n\
  --kv-cache-min-tokens N\n\
      Do not save or load checkpoints shorter than N tokens. Default: 512\n\
  --kv-cache-cold-max-tokens N\n\
      Cold first prompts in [min,N] are saved automatically. 0 disables cold saves. Default: 30000\n\
  --kv-cache-continued-interval-tokens N\n\
      Save at absolute aligned frontiers spaced about N tokens apart. 0 disables. Default: 10000\n\
  --kv-cache-boundary-trim-tokens N\n\
      Trim this many tail tokens before cold boundary saves to avoid tokenizer boundary merges. Default: 32\n\
  --kv-cache-boundary-align-tokens N\n\
      Align cold boundary saves down to this token multiple. 0 disables alignment. Default: 2048\n\
  --kv-cache-reject-different-quant\n\
      Refuse checkpoints written by the same model with a different routed-expert quantization.\n\
  --disable-exact-dsml-tool-replay\n\
      Disable the tool-id -> exact sampled DSML map. Tool history falls back to canonical JSON rendering.\n\
  --tool-memory-max-ids N\n\
      Maximum exact tool-call IDs kept in RAM for replay. Default: 100000\n\
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
    const CHAT_STREAM: &str =
        include_str!("../../../../ds4-parity/baselines/server-fixtures/m0.4/chat_stream.json");
    const CHAT_TOOL_CALL: &str =
        include_str!("../../../../ds4-parity/baselines/server-fixtures/m0.4/chat_tool_call.json");
    const CHAT_CACHE_CONTINUATION: &str = include_str!(
        "../../../../ds4-parity/baselines/server-fixtures/m0.4/chat_cache_continuation.json"
    );
    const GENERATED_TOOL_CALL: &str = "<｜DSML｜tool_calls>\n\
<｜DSML｜invoke name=\"list_files\">\n\
<｜DSML｜parameter name=\"path\" string=\"true\">.</｜DSML｜parameter>\n\
</｜DSML｜invoke>\n\
</｜DSML｜tool_calls>";

    fn parse(args: &[&str]) -> Result<Option<ServerConfig>, CliExit> {
        parse_args(args.iter().copied().map(str::to_string))
    }

    fn parse_chat(body: &str) -> ds4_gguf::OpenAiChatRequest {
        parse_openai_chat_request(body, 64, 32_768).expect("chat request parses")
    }

    fn unique_temp_dir(label: &str) -> std::path::PathBuf {
        let mut dir = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time after epoch")
            .as_nanos();
        dir.push(format!(
            "ds4-server-runtime-rs-{label}-{}-{nanos}",
            std::process::id()
        ));
        fs::create_dir(&dir).expect("create temp dir");
        dir
    }

    fn write_tool_map_kvc(dir: &std::path::Path, id: &str, dsml: &str) {
        use ds4_gguf::kv_policy::{
            sha1_bytes_hex, write_kvc_file, write_tool_map_trailer, KvHeader, ToolMapEntry,
            EXT_TOOL_MAP, REASON_CONTINUED,
        };

        let text = b"tool map carrier";
        let trailer = write_tool_map_trailer(
            dsml.as_bytes(),
            &[ToolMapEntry {
                id: id.to_string(),
                dsml: dsml.as_bytes().to_vec(),
            }],
            false,
        )
        .expect("write tool map trailer");
        let header = KvHeader {
            quant_bits: 2,
            reason: REASON_CONTINUED,
            ext_flags: EXT_TOOL_MAP,
            tokens: 512,
            hits: 0,
            ctx_size: 32_768,
            created_at: 1,
            last_used: 1,
            payload_bytes: 0,
        };
        let bytes = write_kvc_file(&header, text, &[], &trailer).expect("write KVC");
        fs::write(dir.join(format!("{}.kv", sha1_bytes_hex(text))), bytes).expect("write KVC file");
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
                cache: RuntimeCacheConfig::default(),
                runtime_graph_route: RuntimeGraphRoute::TargetStream,
                enable_cors: true,
            })
        );
        assert_eq!(
            parse(&["--runtime-graph", "graph"])
                .unwrap()
                .unwrap()
                .runtime_graph_route,
            RuntimeGraphRoute::Graph
        );
        assert_eq!(
            parse(&["--runtime-graph-route", "off"])
                .unwrap()
                .unwrap()
                .runtime_graph_route,
            RuntimeGraphRoute::TargetStream
        );
    }

    #[test]
    fn parses_runtime_cache_config_flags() {
        let config = parse(&[
            "--kv-disk-dir",
            "/tmp/ds4-kv",
            "--kv-disk-space-mb",
            "8192",
            "--kv-cache-min-tokens",
            "1024",
            "--kv-cache-cold-max-tokens",
            "0",
            "--kv-cache-continued-interval-tokens",
            "20000",
            "--kv-cache-boundary-trim-tokens",
            "0",
            "--kv-cache-boundary-align-tokens",
            "4096",
            "--kv-cache-reject-different-quant",
            "--disable-exact-dsml-tool-replay",
            "--tool-memory-max-ids",
            "7",
        ])
        .unwrap()
        .unwrap();
        assert_eq!(
            config.cache,
            RuntimeCacheConfig {
                disk_dir: Some("/tmp/ds4-kv".to_string()),
                disk_space_mb: 8192,
                policy: KvOptions {
                    min_tokens: 1024,
                    cold_max_tokens: 0,
                    continued_interval_tokens: 20_000,
                    boundary_trim_tokens: 0,
                    boundary_align_tokens: 4096,
                },
                reject_different_quant: true,
                disable_exact_dsml_tool_replay: true,
                tool_memory_max_ids: 7,
            }
        );
        let policy = config.cache.policy_config();
        assert!(policy.enabled);
        assert_eq!(policy.budget_bytes, 8192 * 1024 * 1024);
        assert!(policy.reject_different_quant);
    }

    #[test]
    fn runtime_cache_config_uses_c_default_budget_when_enabled() {
        let mut cache = RuntimeCacheConfig {
            disk_dir: Some("kv".to_string()),
            ..RuntimeCacheConfig::default()
        };
        assert_eq!(cache.effective_disk_space_mb(), 4096);
        assert_eq!(cache.policy_config().budget_bytes, 4096 * 1024 * 1024);

        cache.disk_dir = None;
        assert_eq!(cache.effective_disk_space_mb(), 0);
        assert!(!cache.policy_config().enabled);
    }

    #[test]
    fn runtime_cache_state_resets_continued_frontier_after_miss() {
        let dir = unique_temp_dir("continued-frontier-reset");
        let mut cache = RuntimeCacheState::new(RuntimeCacheConfig {
            disk_dir: Some(dir.to_string_lossy().into_owned()),
            ..RuntimeCacheConfig::default()
        });

        cache.begin_request_ledger();
        cache.note_store(20_480);
        assert_eq!(
            cache
                .disk
                .as_ref()
                .expect("disk cache enabled")
                .continued_last_store_tokens(),
            20_480
        );
        cache.reset_continued_frontier();
        assert_eq!(
            cache
                .disk
                .as_ref()
                .expect("disk cache enabled")
                .continued_last_store_tokens(),
            0
        );
        assert_eq!(
            cache.ledger_events(),
            &[
                RuntimeCacheLedgerEvent {
                    name: "note_store",
                    cache_source: None,
                    reason: None,
                    tokens: 20_480,
                    cached_tokens: -1,
                    cache_write_tokens: -1,
                    disk_cached_tokens: -1,
                    frontier_before: 0,
                    frontier_after: 20_480,
                    success: Some(true),
                },
                RuntimeCacheLedgerEvent {
                    name: "reset_continued_frontier",
                    cache_source: None,
                    reason: None,
                    tokens: 0,
                    cached_tokens: -1,
                    cache_write_tokens: -1,
                    disk_cached_tokens: -1,
                    frontier_before: 20_480,
                    frontier_after: 0,
                    success: Some(true),
                },
            ]
        );

        fs::remove_dir_all(dir).expect("remove temp dir");
    }

    #[test]
    fn runtime_cache_ledger_records_suppress_restore_and_cache_decision() {
        let dir = unique_temp_dir("continued-frontier-ledger");
        let mut cache = RuntimeCacheState::new(RuntimeCacheConfig {
            disk_dir: Some(dir.to_string_lossy().into_owned()),
            ..RuntimeCacheConfig::default()
        });

        cache.begin_request_ledger();
        let old = cache.suppress_continued_store(10_240);
        assert_eq!(old, 0);
        cache.restore_suppressed_continued(old, 10_240);
        cache.record_cache_decision("disk-text", 561, 552, 9, 552);

        assert_eq!(
            cache.ledger_events(),
            &[
                RuntimeCacheLedgerEvent {
                    name: "suppress_continued_store",
                    cache_source: None,
                    reason: None,
                    tokens: 10_240,
                    cached_tokens: -1,
                    cache_write_tokens: -1,
                    disk_cached_tokens: -1,
                    frontier_before: 0,
                    frontier_after: 10_240,
                    success: Some(true),
                },
                RuntimeCacheLedgerEvent {
                    name: "restore_suppressed_continued",
                    cache_source: None,
                    reason: None,
                    tokens: 10_240,
                    cached_tokens: -1,
                    cache_write_tokens: -1,
                    disk_cached_tokens: -1,
                    frontier_before: 10_240,
                    frontier_after: 0,
                    success: Some(true),
                },
                RuntimeCacheLedgerEvent {
                    name: "cache_decision",
                    cache_source: Some("disk-text"),
                    reason: None,
                    tokens: 561,
                    cached_tokens: 552,
                    cache_write_tokens: 9,
                    disk_cached_tokens: 552,
                    frontier_before: 0,
                    frontier_after: 0,
                    success: None,
                },
            ]
        );

        fs::remove_dir_all(dir).expect("remove temp dir");
    }

    #[test]
    fn runtime_cache_ledger_records_failed_suppress_without_mutating_frontier() {
        let dir = unique_temp_dir("continued-frontier-ledger-failed-suppress");
        let mut cache = RuntimeCacheState::new(RuntimeCacheConfig {
            disk_dir: Some(dir.to_string_lossy().into_owned()),
            ..RuntimeCacheConfig::default()
        });

        cache.note_store(10_240);
        cache.begin_request_ledger();
        let old = cache.suppress_continued_store(18_432);

        assert_eq!(old, -1);
        assert_eq!(cache.continued_frontier(), 10_240);
        assert_eq!(
            cache.ledger_events(),
            &[RuntimeCacheLedgerEvent {
                name: "suppress_continued_store",
                cache_source: None,
                reason: None,
                tokens: 18_432,
                cached_tokens: -1,
                cache_write_tokens: -1,
                disk_cached_tokens: -1,
                frontier_before: 10_240,
                frontier_after: 10_240,
                success: Some(false),
            }]
        );

        fs::remove_dir_all(dir).expect("remove temp dir");
    }

    #[test]
    fn runtime_cache_state_restores_tool_map_before_prompt_render() {
        let dir = unique_temp_dir("tool-map-restore");
        let sampled_dsml = "\n\n<｜DSML｜tool_calls>\n\
<｜DSML｜invoke name=\"bash\">\n\
<｜DSML｜parameter name=\"command\" string=\"true\">pwd sampled</｜DSML｜parameter>\n\
</｜DSML｜invoke>\n\
</｜DSML｜tool_calls>";
        write_tool_map_kvc(&dir, "call_keep", sampled_dsml);

        let mut request = parse_chat(
            r#"{
  "model": "deepseek-v4-flash",
  "think": false,
  "messages": [
    {"role": "user", "content": "run a command"},
    {"role": "assistant", "tool_calls": [
      {"id": "call_keep", "type": "function", "function": {"name": "bash", "arguments": "{\"command\":\"pwd canonical\"}"}}
    ]},
    {"role": "tool", "tool_call_id": "call_keep", "content": "/tmp"},
    {"role": "user", "content": "continue"}
  ]
}"#,
        );
        assert!(request.prompt_text.contains("pwd canonical"));

        let mut cache = RuntimeCacheState::new(RuntimeCacheConfig {
            disk_dir: Some(dir.to_string_lossy().into_owned()),
            ..RuntimeCacheConfig::default()
        });
        let stats = cache.prepare_chat_prompt(&mut request);
        assert_eq!(stats.disk, 1);
        assert_eq!(stats.mem, 0);
        assert_eq!(stats.canonical, 0);
        assert!(request.prompt_text.contains("pwd sampled"));
        assert!(!request.prompt_text.contains("pwd canonical"));

        fs::remove_dir_all(dir).expect("remove temp dir");
    }

    #[test]
    fn runtime_cache_state_respects_disabled_exact_tool_replay() {
        let dir = unique_temp_dir("tool-map-disabled");
        let sampled_dsml = "\n\n<｜DSML｜tool_calls>\n\
<｜DSML｜invoke name=\"bash\">\n\
<｜DSML｜parameter name=\"command\" string=\"true\">pwd sampled</｜DSML｜parameter>\n\
</｜DSML｜invoke>\n\
</｜DSML｜tool_calls>";
        write_tool_map_kvc(&dir, "call_keep", sampled_dsml);
        let mut request = parse_chat(
            r#"{
  "model": "deepseek-v4-flash",
  "think": false,
  "messages": [
    {"role": "user", "content": "run a command"},
    {"role": "assistant", "tool_calls": [
      {"id": "call_keep", "type": "function", "function": {"name": "bash", "arguments": "{\"command\":\"pwd canonical\"}"}}
    ]}
  ]
}"#,
        );
        let mut cache = RuntimeCacheState::new(RuntimeCacheConfig {
            disk_dir: Some(dir.to_string_lossy().into_owned()),
            disable_exact_dsml_tool_replay: true,
            ..RuntimeCacheConfig::default()
        });
        let stats = cache.prepare_chat_prompt(&mut request);
        assert_eq!(stats, ToolReplayStats::default());
        assert!(request.prompt_text.contains("pwd canonical"));
        assert!(!request.prompt_text.contains("pwd sampled"));

        fs::remove_dir_all(dir).expect("remove temp dir");
    }

    #[test]
    fn kv_cache_file_name_requires_sha_hex_suffix() {
        assert!(is_kv_cache_file_name(
            "0123456789abcdef0123456789abcdef01234567.kv"
        ));
        assert!(!is_kv_cache_file_name(
            "0123456789abcdef0123456789abcdef0123456.kv"
        ));
        assert!(!is_kv_cache_file_name(
            "0123456789abcdef0123456789abcdef0123456z.kv"
        ));
        assert!(!is_kv_cache_file_name(
            "0123456789abcdef0123456789abcdef01234567.tmp"
        ));
    }

    #[test]
    fn disk_text_cache_restore_runs_on_cache_miss_after_store_stage() {
        assert!(should_try_disk_text_cache(0));
        assert!(!should_try_disk_text_cache(4));
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
        let route = parse(&["--runtime-graph", "fallback"]).unwrap_err();
        assert_eq!(route.code, 2);
        assert!(route
            .stderr
            .contains("invalid runtime graph route: fallback"));
        assert!(route.stderr.contains("target-stream, off, graph"));
        let unknown = parse(&["--bad"]).unwrap_err();
        assert_eq!(unknown.code, 2);
        assert!(unknown
            .stderr
            .starts_with("ds4-server-runtime-rs: unknown option: --bad\n"));
        assert_eq!(
            parse(&["--kv-cache-min-tokens", "1024", "--kv-cache-cold-max-tokens", "1"])
                .unwrap_err(),
            CliExit {
                code: 2,
                stdout: String::new(),
                stderr:
                    "ds4-server-runtime-rs: --kv-cache-cold-max-tokens must be 0 or >= --kv-cache-min-tokens\n"
                        .to_string(),
            }
        );
        assert_eq!(
            parse(&["--kv-cache-boundary-trim-tokens", "-1"]).unwrap_err(),
            CliExit {
                code: 2,
                stdout: String::new(),
                stderr:
                    "ds4-server-runtime-rs: invalid value for --kv-cache-boundary-trim-tokens: -1\n"
                        .to_string(),
            }
        );
        assert_eq!(
            parse(&["--kv-disk-space-mb", "2147483648"]).unwrap_err(),
            CliExit {
                code: 2,
                stdout: String::new(),
                stderr: "ds4-server-runtime-rs: invalid value for --kv-disk-space-mb: 2147483648\n"
                    .to_string(),
            }
        );
        assert_eq!(
            parse(&["--tool-memory-max-ids", "0"]).unwrap_err(),
            CliExit {
                code: 2,
                stdout: String::new(),
                stderr: "ds4-server-runtime-rs: invalid value for --tool-memory-max-ids: 0\n"
                    .to_string(),
            }
        );
        assert_eq!(
            parse(&["--tool-memory-max-ids", "2147483648"]).unwrap_err(),
            CliExit {
                code: 2,
                stdout: String::new(),
                stderr:
                    "ds4-server-runtime-rs: invalid value for --tool-memory-max-ids: 2147483648\n"
                        .to_string(),
            }
        );
    }

    #[test]
    fn bind_host_matches_c_localhost_behavior() {
        assert_eq!(bind_host("localhost").unwrap(), Ipv4Addr::new(127, 0, 0, 1));
        assert_eq!(bind_host("127.0.0.1").unwrap(), Ipv4Addr::new(127, 0, 0, 1));
        assert!(bind_host("example.com").is_err());
    }

    #[test]
    fn m96b_allows_supported_non_streaming_tool_chat() {
        assert_eq!(
            unsupported_chat_generation_message(&parse_chat(CHAT_BASIC)),
            None
        );
        assert_eq!(
            unsupported_chat_generation_message(&parse_chat(CHAT_THINKING_DISABLED)),
            None
        );
        assert_eq!(
            unsupported_chat_generation_message(&parse_chat(CHAT_STREAM)),
            None
        );
        assert_eq!(
            unsupported_chat_generation_message(&parse_chat(CHAT_TOOL_CALL)),
            None
        );
        let streaming_tool_call = CHAT_TOOL_CALL.replace(
            "\"stream\": false",
            "\"stream\": true,\n  \"stream_options\": {\"include_usage\": true}",
        );
        assert_eq!(
            unsupported_chat_generation_message(&parse_chat(&streaming_tool_call)),
            None
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

    #[test]
    fn stream_delta_strings_preserve_token_boundaries() {
        let generated = ds4_engine::ServerGenerationResult {
            exit_code: 0,
            text: b"stream baseline".to_vec(),
            token_texts: vec![b"stream".to_vec(), b" baseline".to_vec()],
            prompt_tokens: 11,
            cache_read_tokens: 0,
            cache_write_tokens: 11,
            live_tokens_before: 0,
            live_prompt_common: 0,
            completion_tokens: 2,
            finish_reason: "stop",
        };
        assert_eq!(
            stream_delta_strings(&generated).unwrap(),
            vec!["stream".to_string(), " baseline".to_string()]
        );

        let split_utf8 = ds4_engine::ServerGenerationResult {
            text: "€".as_bytes().to_vec(),
            token_texts: vec![vec![0xe2], vec![0x82, 0xac]],
            completion_tokens: 2,
            ..generated.clone()
        };
        assert_eq!(stream_delta_strings(&split_utf8).unwrap(), vec!["€"]);

        let invalid = ds4_engine::ServerGenerationResult {
            token_texts: vec![vec![0xff]],
            ..generated
        };
        assert!(stream_delta_strings(&invalid).is_err());
    }

    #[test]
    fn m96c3_formats_streaming_tool_chat_replay() {
        let streaming_tool_call = CHAT_TOOL_CALL.replace(
            "\"stream\": false",
            "\"stream\": true,\n  \"stream_options\": {\"include_usage\": true}",
        );
        let parsed = parse_chat(&streaming_tool_call);
        let generated = ds4_engine::ServerGenerationResult {
            exit_code: 0,
            text: GENERATED_TOOL_CALL.as_bytes().to_vec(),
            token_texts: vec![
                "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"list_files\">\n"
                    .as_bytes()
                    .to_vec(),
                "<｜DSML｜parameter name=\"path\" string=\"true\">.</｜DSML｜parameter>\n"
                    .as_bytes()
                    .to_vec(),
                "</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
                    .as_bytes()
                    .to_vec(),
            ],
            prompt_tokens: 394,
            cache_read_tokens: 0,
            cache_write_tokens: 394,
            live_tokens_before: 0,
            live_prompt_common: 0,
            completion_tokens: 42,
            finish_reason: "stop",
        };
        let parsed_generation = parse_chat_generation(&parsed, &generated, GENERATED_TOOL_CALL, 3);
        let response = format_streaming_tool_chat_http(
            false,
            "chatcmpl-3",
            1_779_416_175,
            &parsed.model,
            parsed.stream_include_usage,
            OpenAiUsage::new(394, 42, 0, 394),
            &generated,
            3,
            &parsed_generation,
        );
        let (headers, body) = response.split_once("\r\n\r\n").expect("HTTP response");
        assert_eq!(
            headers.replace("\r\n", "\n") + "\n",
            "HTTP/1.1 200 OK\nContent-Type: text/event-stream\nCache-Control: no-cache\nConnection: close\n"
        );
        let role = body.find("\"role\":\"assistant\"").unwrap();
        let start = body
            .find("\"tool_calls\":[{\"index\":0,\"id\":\"call_00000000000000030000000000000000\",\"type\":\"function\",\"function\":{\"name\":\"list_files\",\"arguments\":\"\"}}]")
            .unwrap();
        let object_open = body.find("\"arguments\":\"{\"").unwrap();
        let path_prefix = body.find("\\\"path\\\":\\\"").unwrap();
        let path_value = body.find("\"arguments\":\".\"").unwrap();
        let string_close = body.rfind("\"arguments\":\"\\\"\"").unwrap();
        let object_close = body.rfind("\"arguments\":\"}\"").unwrap();
        let finish = body.find("\"finish_reason\":\"tool_calls\"").unwrap();
        let usage = body.find("\"prompt_tokens\":394").unwrap();
        let done = body.find("data: [DONE]\n").unwrap();
        assert!(role < start);
        assert!(start < object_open);
        assert!(object_open < path_prefix);
        assert!(path_prefix < path_value);
        assert!(path_value < string_close);
        assert!(string_close < object_close);
        assert!(object_close < finish);
        assert!(finish < usage);
        assert!(usage < done);
        assert!(body.ends_with("data: [DONE]\n"));
    }

    #[test]
    fn trace_reports_memory_token_cache_reuse() {
        let parsed = parse_chat(CHAT_CACHE_CONTINUATION);
        let generated = ds4_engine::ServerGenerationResult {
            exit_code: 0,
            text: b"cache continued".to_vec(),
            token_texts: vec![b"cache".to_vec(), b" continued".to_vec()],
            prompt_tokens: 50,
            cache_read_tokens: 41,
            cache_write_tokens: 9,
            live_tokens_before: 41,
            live_prompt_common: 41,
            completion_tokens: 2,
            finish_reason: "stop",
        };
        let parsed_generation = parse_chat_generation(&parsed, &generated, "cache continued", 6);
        let mut trace = Vec::new();
        write_chat_trace(
            &mut trace,
            6,
            CHAT_CACHE_CONTINUATION,
            &parsed,
            generated.prompt_tokens,
            &generated,
            "cache continued",
            &parsed_generation,
        )
        .unwrap();
        let trace = String::from_utf8(trace).unwrap();
        assert!(trace.contains("cached_tokens: 41\n"));
        assert!(trace.contains("live_tokens_before: 41\n"));
        assert!(trace.contains("live_prompt_common: 41\n"));
        assert!(trace.contains("memory_token_reusable: 1\n"));
        assert!(trace.contains("memory_miss_reason: live-prefix-match\n"));
        assert!(trace.contains("cache_source: memory-token\n"));
        assert!(trace.contains("generated_tokens: 2\n"));
        assert!(trace.contains("dsml_start: 0\n"));
        assert!(trace.contains("dsml_end: 0\n"));
        assert!(trace.contains("\ncontent:\ncache continued\n"));
    }

    #[test]
    fn trace_contract_reports_disk_text_cache_and_tool_replay() {
        let parsed = parse_chat(CHAT_CACHE_CONTINUATION);
        let generated = ds4_engine::ServerGenerationResult {
            exit_code: 0,
            text: b"cache continued".to_vec(),
            token_texts: vec![b"cache".to_vec(), b" continued".to_vec()],
            prompt_tokens: 50,
            cache_read_tokens: 41,
            cache_write_tokens: 9,
            live_tokens_before: 0,
            live_prompt_common: 0,
            completion_tokens: 2,
            finish_reason: "stop",
        };
        let parsed_generation = parse_chat_generation(&parsed, &generated, "cache continued", 6);
        let cache = RuntimeCacheDecision {
            live_tokens_before: 0,
            prompt_tokens: generated.prompt_tokens,
            effective_prompt_tokens: 50,
            live_prompt_common: 0,
            tool_replay: ToolReplayStats {
                mem: 0,
                disk: 1,
                canonical: 2,
                missing_ids: 3,
            },
            cache_source: "disk-text",
            cached_tokens: 41,
            disk_cached_tokens: 41,
            disk_cache_file: Some("/tmp/ds4-kv/abc.kv".to_string()),
        };
        let mut trace = Vec::new();
        write_chat_trace_with_cache_decision(
            &mut trace,
            7,
            CHAT_CACHE_CONTINUATION,
            &parsed,
            &generated,
            "cache continued",
            &parsed_generation,
            &cache,
            &[RuntimeCacheLedgerEvent {
                name: "cache_decision",
                cache_source: Some("disk-text"),
                reason: None,
                tokens: 50,
                cached_tokens: 41,
                cache_write_tokens: 9,
                disk_cached_tokens: 41,
                frontier_before: 552,
                frontier_after: 552,
                success: None,
            }],
        )
        .unwrap();
        let trace = String::from_utf8(trace).unwrap();
        assert!(trace.contains("effective_prompt_tokens: 50\n"));
        assert!(trace.contains("memory_token_reusable: 0\n"));
        assert!(trace.contains("memory_miss_reason: no-live-checkpoint\n"));
        assert!(trace.contains("tool_replay: mem=0 disk=1 canonical=2 missing_ids=3\n"));
        assert!(trace.contains("cache_source: disk-text\n"));
        assert!(trace.contains("cached_tokens: 41\n"));
        assert!(trace.contains("disk_cached_tokens: 41\n"));
        assert!(trace.contains("disk_cache_file: /tmp/ds4-kv/abc.kv\n"));
        assert!(trace.contains("--- runtime cache ledger ---\n"));
        assert!(trace.contains(
            "event[0]: name=cache_decision cache_source=disk-text reason= tokens=50 cached_tokens=41 cache_write_tokens=9 disk_cached_tokens=41 frontier_before=552 frontier_after=552 success=\n"
        ));
    }

    #[test]
    fn parses_tool_generation_for_response_and_trace() {
        let parsed = parse_chat(CHAT_TOOL_CALL);
        let generated = ds4_engine::ServerGenerationResult {
            exit_code: 0,
            text: GENERATED_TOOL_CALL.as_bytes().to_vec(),
            token_texts: Vec::new(),
            prompt_tokens: 394,
            cache_read_tokens: 0,
            cache_write_tokens: 394,
            live_tokens_before: 0,
            live_prompt_common: 0,
            completion_tokens: 42,
            finish_reason: "stop",
        };
        let parsed_generation = parse_chat_generation(&parsed, &generated, GENERATED_TOOL_CALL, 3);
        assert_eq!(parsed_generation.finish_reason, "tool_calls");
        assert_eq!(parsed_generation.content, "");
        assert_eq!(parsed_generation.reasoning, None);
        assert!(parsed_generation.saw_tool_start);
        assert!(parsed_generation.saw_tool_end);
        assert_eq!(parsed_generation.calls.len(), 1);
        assert_eq!(
            parsed_generation.calls[0].id.as_deref(),
            Some("call_00000000000000030000000000000000")
        );
        assert_eq!(parsed_generation.calls[0].name, "list_files");
        assert_eq!(parsed_generation.calls[0].arguments, "{\"path\": \".\"}");
        assert_eq!(
            parsed_generation.raw_dsml.as_deref(),
            Some(GENERATED_TOOL_CALL)
        );

        let mut trace = Vec::new();
        write_chat_trace(
            &mut trace,
            3,
            CHAT_TOOL_CALL,
            &parsed,
            generated.prompt_tokens,
            &generated,
            GENERATED_TOOL_CALL,
            &parsed_generation,
        )
        .unwrap();
        let trace = String::from_utf8(trace).unwrap();
        assert!(trace.contains("tools: 1\n"));
        assert!(trace.contains("finish: tool_calls\n"));
        assert!(trace.contains("generated_tokens: 42\n"));
        assert!(trace.contains("dsml_start: 1\n"));
        assert!(trace.contains("dsml_end: 1\n"));
        assert!(trace.contains("tool_call[0]:\n"));
        assert!(trace.contains("id: call_00000000000000030000000000000000\n"));
        assert!(trace.contains("name: list_files\n"));
        assert!(trace.contains("arguments:\n{\"path\": \".\"}\n"));
    }

    #[test]
    fn runtime_remembers_generated_tool_ids_for_kv_trailers() {
        let parsed = parse_chat(CHAT_TOOL_CALL);
        let generated = ds4_engine::ServerGenerationResult {
            exit_code: 0,
            text: GENERATED_TOOL_CALL.as_bytes().to_vec(),
            token_texts: Vec::new(),
            prompt_tokens: 394,
            cache_read_tokens: 0,
            cache_write_tokens: 394,
            live_tokens_before: 0,
            live_prompt_common: 0,
            completion_tokens: 42,
            finish_reason: "stop",
        };
        let parsed_generation = parse_chat_generation(&parsed, &generated, GENERATED_TOOL_CALL, 3);
        let mut cache = RuntimeCacheState::new(RuntimeCacheConfig::default());
        cache.remember_generated_tool_calls(&parsed_generation);

        let entries = cache.tool_memory.tool_map_entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].id, "call_00000000000000030000000000000000");
        assert_eq!(entries[0].dsml, GENERATED_TOOL_CALL.as_bytes());
        let trailer =
            write_tool_map_trailer(GENERATED_TOOL_CALL.as_bytes(), &entries, false).unwrap();
        assert!(!trailer.is_empty());
    }

    #[test]
    fn generated_tool_start_scan_handles_utf8_markers() {
        assert!(generated_saw_dsml_tool_start(
            GENERATED_TOOL_CALL.as_bytes()
        ));
        assert!(generated_saw_dsml_tool_start(b"abc<tool_calls>"));
        assert!(!generated_saw_dsml_tool_start(b"abc</tool_calls>"));
    }
}
