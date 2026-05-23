use std::error::Error;
use std::ffi::{c_char, c_float, c_int, c_void, CStr, CString, NulError};
use std::fmt;
use std::ptr::NonNull;
use std::slice;
use std::time::Instant;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Backend {
    Metal,
    Cuda,
    Cpu,
}

impl Backend {
    pub const fn default_backend() -> Self {
        if cfg!(target_os = "macos") {
            Self::Metal
        } else {
            Self::Cuda
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "metal" => Some(Self::Metal),
            "cuda" => Some(Self::Cuda),
            "cpu" => Some(Self::Cpu),
            _ => None,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Metal => "metal",
            Self::Cuda => "cuda",
            Self::Cpu => "cpu",
        }
    }

    const fn as_raw(self) -> c_int {
        match self {
            Self::Metal => 0,
            Self::Cuda => 1,
            Self::Cpu => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ThinkMode {
    None,
    High,
    Max,
}

impl ThinkMode {
    pub const fn default_mode() -> Self {
        Self::High
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::High => "high",
            Self::Max => "max",
        }
    }

    pub fn for_context(self, ctx_size: i32) -> Self {
        Self::from_raw(unsafe { ds4_think_mode_for_context(self.as_raw(), ctx_size as c_int) })
    }

    pub fn enabled(self) -> bool {
        unsafe { ds4_think_mode_enabled(self.as_raw()) }
    }

    pub fn max_min_context() -> u32 {
        unsafe { ds4_think_max_min_context() }
    }

    const fn as_raw(self) -> c_int {
        match self {
            Self::None => 0,
            Self::High => 1,
            Self::Max => 2,
        }
    }

    const fn from_raw(value: c_int) -> Self {
        match value {
            0 => Self::None,
            2 => Self::Max,
            _ => Self::High,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EngineOptions<'a> {
    pub model_path: &'a str,
    pub mtp_path: Option<&'a str>,
    pub backend: Backend,
    pub n_threads: i32,
    pub mtp_draft_tokens: i32,
    pub mtp_margin: f32,
    pub directional_steering_file: Option<&'a str>,
    pub directional_steering_attn: f32,
    pub directional_steering_ffn: f32,
    pub warm_weights: bool,
    pub quality: bool,
}

impl<'a> EngineOptions<'a> {
    pub const fn new(model_path: &'a str, backend: Backend) -> Self {
        Self {
            model_path,
            mtp_path: None,
            backend,
            n_threads: 0,
            mtp_draft_tokens: 1,
            mtp_margin: 3.0,
            directional_steering_file: None,
            directional_steering_attn: 0.0,
            directional_steering_ffn: 0.0,
            warm_weights: false,
            quality: false,
        }
    }
}

#[derive(Debug)]
pub struct Engine {
    raw: NonNull<RawEngine>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ContextMemory {
    pub total_bytes: u64,
    pub raw_bytes: u64,
    pub compressed_bytes: u64,
    pub scratch_bytes: u64,
    pub prefill_cap: u32,
    pub raw_cap: u32,
    pub comp_cap: u32,
}

#[derive(Debug)]
pub struct Tokens {
    raw: RawTokens,
}

impl Tokens {
    pub fn len(&self) -> i32 {
        self.raw.len
    }

    pub fn is_empty(&self) -> bool {
        self.raw.len == 0
    }

    fn set_len(&mut self, len: i32) {
        self.raw.len = len.clamp(0, self.raw.len);
    }

    fn push(&mut self, token: i32) {
        unsafe {
            ds4_tokens_push(&mut self.raw, token as c_int);
        }
    }
}

impl Drop for Tokens {
    fn drop(&mut self) {
        unsafe {
            ds4_tokens_free(&mut self.raw);
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArgmaxOptions {
    pub n_predict: i32,
    pub ctx_size: i32,
    pub think_mode: ThinkMode,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingOptions {
    pub n_predict: i32,
    pub ctx_size: i32,
    pub think_mode: ThinkMode,
    pub temperature: f32,
    pub top_p: f32,
    pub min_p: f32,
    pub seed: u64,
}

#[derive(Debug)]
pub struct GenerationResult {
    pub exit_code: i32,
    pub stdout: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InteractiveTurnOptions {
    pub n_predict: i32,
    pub think_mode: ThinkMode,
    pub temperature: f32,
    pub top_p: f32,
    pub min_p: f32,
    pub seed: u64,
}

#[derive(Debug)]
pub struct ChatSession<'a> {
    engine: &'a Engine,
    session: Session,
    transcript: Tokens,
    ctx_size: i32,
}

impl Engine {
    pub fn open(options: &EngineOptions<'_>) -> Result<Self, EngineError> {
        let model_path = CString::new(options.model_path)?;
        let mtp_path = options.mtp_path.map(CString::new).transpose()?;
        let directional_steering_file = options
            .directional_steering_file
            .map(CString::new)
            .transpose()?;
        let raw_options = RawEngineOptions {
            model_path: model_path.as_ptr(),
            mtp_path: mtp_path
                .as_ref()
                .map_or(std::ptr::null(), |path| path.as_ptr()),
            backend: options.backend.as_raw(),
            n_threads: options.n_threads as c_int,
            mtp_draft_tokens: options.mtp_draft_tokens as c_int,
            mtp_margin: options.mtp_margin,
            directional_steering_file: directional_steering_file
                .as_ref()
                .map_or(std::ptr::null(), |path| path.as_ptr()),
            directional_steering_attn: options.directional_steering_attn,
            directional_steering_ffn: options.directional_steering_ffn,
            warm_weights: options.warm_weights,
            quality: options.quality,
        };
        let mut raw = std::ptr::null_mut();
        let rc = unsafe { ds4_engine_open(&mut raw, &raw_options) };
        if rc != 0 {
            return Err(EngineError::open_failed(rc));
        }
        Ok(Self {
            raw: NonNull::new(raw).ok_or(EngineError::null_engine())?,
        })
    }

    pub fn print_summary(&self) {
        unsafe {
            ds4_engine_summary(self.raw.as_ptr());
        }
    }

    pub fn encode_chat_prompt(
        &self,
        system: &str,
        prompt: &str,
        think_mode: ThinkMode,
    ) -> Result<Tokens, EngineError> {
        let system = CString::new(system)?;
        let prompt = CString::new(prompt)?;
        let mut raw = RawTokens::default();
        unsafe {
            if is_rendered_chat_prompt(prompt.as_bytes()) {
                ds4_tokenize_rendered_chat(self.raw.as_ptr(), prompt.as_ptr(), &mut raw);
            } else {
                ds4_encode_chat_prompt(
                    self.raw.as_ptr(),
                    system.as_ptr(),
                    prompt.as_ptr(),
                    think_mode.as_raw(),
                    &mut raw,
                );
            }
        }
        Ok(Tokens { raw })
    }

    pub fn generate_argmax_text(
        &self,
        prompt: &Tokens,
        options: ArgmaxOptions,
    ) -> GenerationResult {
        let mut state = EmitState {
            engine: self.raw.as_ptr(),
            printer: TokenPrinter::new(options.think_mode),
        };
        let rc = unsafe {
            ds4_engine_generate_argmax(
                self.raw.as_ptr(),
                &prompt.raw,
                options.n_predict as c_int,
                options.ctx_size as c_int,
                Some(emit_generated_token),
                Some(finish_generation),
                (&mut state as *mut EmitState).cast(),
                None,
                std::ptr::null_mut(),
            )
        };
        GenerationResult {
            exit_code: rc,
            stdout: state.printer.into_bytes(),
        }
    }

    pub fn generate_sampled_text(
        &self,
        prompt: &Tokens,
        options: SamplingOptions,
    ) -> GenerationResult {
        let mut session = std::ptr::null_mut();
        let rc = unsafe {
            ds4_session_create(&mut session, self.raw.as_ptr(), options.ctx_size as c_int)
        };
        let Some(session) = NonNull::new(session) else {
            eprintln!("ds4: sampled CLI generation requires a session backend");
            return GenerationResult {
                exit_code: if rc == 0 { 1 } else { rc },
                stdout: Vec::new(),
            };
        };
        let session = Session { raw: session };
        let mut printer = TokenPrinter::new(options.think_mode);

        let mut err = [0 as c_char; 160];
        let prefill_start = Instant::now();
        let sync_rc = unsafe {
            ds4_session_sync(
                session.raw.as_ptr(),
                &prompt.raw,
                err.as_mut_ptr(),
                err.len(),
            )
        };
        if sync_rc != 0 {
            eprintln!("ds4: prompt processing failed: {}", c_error(&err));
            return GenerationResult {
                exit_code: sync_rc,
                stdout: printer.into_bytes(),
            };
        }
        let prefill_elapsed = prefill_start.elapsed();

        let mut max_tokens = options.n_predict;
        let room = unsafe {
            ds4_session_ctx(session.raw.as_ptr()) - ds4_session_pos(session.raw.as_ptr())
        };
        if room <= 1 {
            max_tokens = 0;
        } else if max_tokens > room - 1 {
            max_tokens = room - 1;
        }

        let mut rng = options.seed;
        let eos = unsafe { ds4_token_eos(self.raw.as_ptr()) };
        let decode_start = Instant::now();
        let mut generated = 0;
        while generated < max_tokens {
            let token = unsafe {
                ds4_session_sample(
                    session.raw.as_ptr(),
                    options.temperature,
                    0,
                    options.top_p,
                    options.min_p,
                    &mut rng,
                )
            };
            if token == eos {
                break;
            }
            let eval_rc = unsafe {
                ds4_session_eval(session.raw.as_ptr(), token, err.as_mut_ptr(), err.len())
            };
            if eval_rc != 0 {
                eprintln!("ds4: decode failed: {}", c_error(&err));
                return GenerationResult {
                    exit_code: eval_rc,
                    stdout: printer.into_bytes(),
                };
            }
            unsafe {
                append_token_text(self.raw.as_ptr(), &mut printer, token);
            }
            generated += 1;
        }
        let decode_elapsed = decode_start.elapsed();
        printer.finish_generation();
        eprintln!(
            "ds4: prefill: {:.2} t/s, generation: {:.2} t/s",
            rate(prompt.len(), prefill_elapsed),
            rate(generated, decode_elapsed)
        );
        GenerationResult {
            exit_code: 0,
            stdout: printer.into_bytes(),
        }
    }

    pub fn create_chat_session(
        &self,
        system: &str,
        ctx_size: i32,
        think_mode: ThinkMode,
    ) -> Result<ChatSession<'_>, EngineError> {
        let mut transcript = Tokens {
            raw: RawTokens::default(),
        };
        unsafe {
            ds4_chat_begin(self.raw.as_ptr(), &mut transcript.raw);
        }
        if think_mode.for_context(ctx_size) == ThinkMode::Max {
            unsafe {
                ds4_chat_append_max_effort_prefix(self.raw.as_ptr(), &mut transcript.raw);
            }
        }
        if !system.is_empty() {
            self.append_chat_message(&mut transcript, "system", system)?;
        }
        let session = self.create_session(ctx_size)?;
        Ok(ChatSession {
            engine: self,
            session,
            transcript,
            ctx_size,
        })
    }

    fn create_session(&self, ctx_size: i32) -> Result<Session, EngineError> {
        let mut raw = std::ptr::null_mut();
        let rc = unsafe { ds4_session_create(&mut raw, self.raw.as_ptr(), ctx_size as c_int) };
        if rc != 0 {
            return Err(EngineError::message(format!(
                "ds4_session_create failed with {rc}"
            )));
        }
        Ok(Session {
            raw: NonNull::new(raw).ok_or_else(|| {
                EngineError::message("ds4_session_create returned a null session".to_string())
            })?,
        })
    }

    fn append_chat_message(
        &self,
        transcript: &mut Tokens,
        role: &str,
        content: &str,
    ) -> Result<(), EngineError> {
        let role = CString::new(role)?;
        let content = CString::new(content)?;
        unsafe {
            ds4_chat_append_message(
                self.raw.as_ptr(),
                &mut transcript.raw,
                role.as_ptr(),
                content.as_ptr(),
            );
        }
        Ok(())
    }

    fn append_assistant_prefix(&self, transcript: &mut Tokens, think_mode: ThinkMode) {
        unsafe {
            ds4_chat_append_assistant_prefix(
                self.raw.as_ptr(),
                &mut transcript.raw,
                think_mode.as_raw(),
            );
        }
    }

    fn token_eos(&self) -> i32 {
        unsafe { ds4_token_eos(self.raw.as_ptr()) as i32 }
    }
}

impl ChatSession<'_> {
    pub fn ctx_size(&self) -> i32 {
        self.ctx_size
    }

    pub fn set_ctx(&mut self, ctx_size: i32) -> Result<(), EngineError> {
        self.session = self.engine.create_session(ctx_size)?;
        self.ctx_size = ctx_size;
        Ok(())
    }

    pub fn run_turn(
        &mut self,
        user_text: &str,
        options: InteractiveTurnOptions,
    ) -> GenerationResult {
        let think_mode = options.think_mode.for_context(self.ctx_size);
        let rollback_len = self.transcript.len();
        if let Err(err) = self
            .engine
            .append_chat_message(&mut self.transcript, "user", user_text)
        {
            eprintln!("{err}");
            return GenerationResult {
                exit_code: 1,
                stdout: Vec::new(),
            };
        }
        self.engine
            .append_assistant_prefix(&mut self.transcript, think_mode);

        let old_pos = self.session.pos();
        let common = self.session.common_prefix(&self.transcript);
        let cached = if common == old_pos && self.transcript.len() >= old_pos {
            common
        } else {
            0
        };
        let suffix = self.transcript.len() - cached;
        let progress = ProgressState {
            base_tokens: cached,
            input_tokens: suffix,
        };
        let mut err = [0 as c_char; 160];
        let prefill_start = Instant::now();
        unsafe {
            ds4_session_set_progress(
                self.session.raw.as_ptr(),
                Some(session_progress),
                (&progress as *const ProgressState).cast_mut().cast(),
            );
        }
        let sync_rc = unsafe {
            ds4_session_sync(
                self.session.raw.as_ptr(),
                &self.transcript.raw,
                err.as_mut_ptr(),
                err.len(),
            )
        };
        unsafe {
            ds4_session_set_progress(self.session.raw.as_ptr(), None, std::ptr::null_mut());
        }
        if sync_rc != 0 {
            self.transcript.set_len(rollback_len);
            eprintln!("ds4: prompt processing failed: {}", c_error(&err));
            return GenerationResult {
                exit_code: sync_rc,
                stdout: Vec::new(),
            };
        }
        let prefill_elapsed = prefill_start.elapsed();

        let mut max_tokens = options.n_predict;
        let room = self.session.ctx() - self.session.pos();
        if room <= 1 {
            max_tokens = 0;
        } else if max_tokens > room - 1 {
            max_tokens = room - 1;
        }

        let mut rng = options.seed;
        let eos = self.engine.token_eos();
        let mut printer = TokenPrinter::new(think_mode);
        let decode_start = Instant::now();
        let mut generated = 0;
        while generated < max_tokens {
            let token = unsafe {
                ds4_session_sample(
                    self.session.raw.as_ptr(),
                    options.temperature,
                    0,
                    options.top_p,
                    options.min_p,
                    &mut rng,
                )
            };
            if token == eos {
                break;
            }
            let eval_rc = unsafe {
                ds4_session_eval(
                    self.session.raw.as_ptr(),
                    token,
                    err.as_mut_ptr(),
                    err.len(),
                )
            };
            if eval_rc != 0 {
                eprintln!("ds4: decode failed: {}", c_error(&err));
                return GenerationResult {
                    exit_code: eval_rc,
                    stdout: printer.into_bytes(),
                };
            }
            self.transcript.push(token);
            unsafe {
                append_token_text(self.engine.raw.as_ptr(), &mut printer, token);
            }
            generated += 1;
        }
        let decode_elapsed = decode_start.elapsed();
        printer.finish_generation();
        self.transcript.push(eos);
        eprintln!(
            "ds4: prefill: {:.2} t/s, generation: {:.2} t/s",
            rate(suffix, prefill_elapsed),
            rate(generated, decode_elapsed)
        );
        GenerationResult {
            exit_code: 0,
            stdout: printer.into_bytes(),
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe {
            ds4_engine_close(self.raw.as_ptr());
        }
    }
}

#[derive(Debug)]
pub struct EngineError {
    kind: EngineErrorKind,
}

#[derive(Debug)]
enum EngineErrorKind {
    InvalidString(NulError),
    OpenFailed(c_int),
    NullEngine,
    Message(String),
}

impl EngineError {
    pub fn open_failed_code(&self) -> Option<i32> {
        match self.kind {
            EngineErrorKind::OpenFailed(code) => Some(code),
            _ => None,
        }
    }

    fn open_failed(code: c_int) -> Self {
        Self {
            kind: EngineErrorKind::OpenFailed(code),
        }
    }

    fn null_engine() -> Self {
        Self {
            kind: EngineErrorKind::NullEngine,
        }
    }

    fn message(message: String) -> Self {
        Self {
            kind: EngineErrorKind::Message(message),
        }
    }
}

impl From<NulError> for EngineError {
    fn from(error: NulError) -> Self {
        Self {
            kind: EngineErrorKind::InvalidString(error),
        }
    }
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            EngineErrorKind::InvalidString(error) => write!(f, "invalid C string: {error}"),
            EngineErrorKind::OpenFailed(code) => write!(f, "ds4_engine_open failed with {code}"),
            EngineErrorKind::NullEngine => write!(f, "ds4_engine_open returned a null engine"),
            EngineErrorKind::Message(message) => f.write_str(message),
        }
    }
}

impl Error for EngineError {}

pub fn context_memory_estimate(backend: Backend, ctx_size: i32) -> ContextMemory {
    let raw = unsafe { ds4_context_memory_estimate(backend.as_raw(), ctx_size as c_int) };
    ContextMemory {
        total_bytes: raw.total_bytes,
        raw_bytes: raw.raw_bytes,
        compressed_bytes: raw.compressed_bytes,
        scratch_bytes: raw.scratch_bytes,
        prefill_cap: raw.prefill_cap,
        raw_cap: raw.raw_cap,
        comp_cap: raw.comp_cap,
    }
}

fn is_rendered_chat_prompt(prompt: &[u8]) -> bool {
    prompt.starts_with("<｜begin▁of▁sentence｜>".as_bytes())
}

struct EmitState {
    engine: *mut RawEngine,
    printer: TokenPrinter,
}

#[derive(Debug)]
struct ProgressState {
    base_tokens: i32,
    input_tokens: i32,
}

unsafe extern "C" fn session_progress(
    ud: *mut c_void,
    event: *const c_char,
    current: c_int,
    _total: c_int,
) {
    if ud.is_null() || event.is_null() {
        return;
    }
    let state = &*ud.cast::<ProgressState>();
    if state.input_tokens <= 0 {
        return;
    }
    let Ok(event) = CStr::from_ptr(event).to_str() else {
        return;
    };
    if event != "prefill_chunk" {
        return;
    }
    let mut processed = current - state.base_tokens;
    processed = processed.clamp(0, state.input_tokens);
    let pct = 100.0 * f64::from(processed) / f64::from(state.input_tokens);
    eprintln!(
        "processing {} input tokens: {}/{} ({pct:.1}%)",
        state.input_tokens, processed, state.input_tokens
    );
}

unsafe extern "C" fn emit_generated_token(ud: *mut c_void, token: c_int) {
    let Some(state) = ud.cast::<EmitState>().as_mut() else {
        return;
    };
    append_token_text(state.engine, &mut state.printer, token);
}

unsafe fn append_token_text(engine: *mut RawEngine, printer: &mut TokenPrinter, token: c_int) {
    let mut len = 0usize;
    let text = ds4_token_text(engine, token, &mut len);
    if text.is_null() {
        return;
    }
    let bytes = slice::from_raw_parts(text.cast::<u8>(), len);
    printer.write_token_text(bytes);
    free(text.cast());
}

unsafe extern "C" fn finish_generation(ud: *mut c_void) {
    let Some(state) = ud.cast::<EmitState>().as_mut() else {
        return;
    };
    state.printer.finish_generation();
}

#[derive(Debug)]
struct TokenPrinter {
    bytes: Vec<u8>,
    format_thinking: bool,
    in_think: bool,
    last_output_newline: bool,
    pending: Vec<u8>,
}

impl TokenPrinter {
    fn new(think_mode: ThinkMode) -> Self {
        let format_thinking = think_mode.enabled();
        Self {
            bytes: Vec::new(),
            format_thinking,
            in_think: format_thinking,
            last_output_newline: true,
            pending: Vec::new(),
        }
    }

    fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    fn write_token_text(&mut self, text: &[u8]) {
        if self.format_thinking {
            self.process_thinking_text(text, false);
        } else {
            self.bytes.extend_from_slice(text);
            if let Some(&last) = text.last() {
                self.last_output_newline = last == b'\n';
            }
        }
    }

    fn finish_generation(&mut self) {
        if self.format_thinking {
            self.process_thinking_text(&[], true);
        }
        if !self.last_output_newline {
            self.bytes.push(b'\n');
            self.last_output_newline = true;
        }
    }

    fn process_thinking_text(&mut self, text: &[u8], finish: bool) {
        const THINK_OPEN: &[u8] = b"<think>";
        const THINK_CLOSE: &[u8] = b"</think>";
        let mut buf = Vec::with_capacity(self.pending.len() + text.len());
        buf.extend_from_slice(&self.pending);
        buf.extend_from_slice(text);
        self.pending.clear();

        let mut i = 0usize;
        while i < buf.len() {
            let cur = &buf[i..];
            if cur.starts_with(THINK_OPEN) {
                self.in_think = true;
                i += THINK_OPEN.len();
                continue;
            }
            if cur.starts_with(THINK_CLOSE) {
                self.in_think = false;
                if !self.last_output_newline {
                    self.bytes.push(b'\n');
                    self.last_output_newline = true;
                }
                i += THINK_CLOSE.len();
                continue;
            }
            if !finish
                && cur[0] == b'<'
                && (is_partial_prefix(cur, THINK_OPEN) || is_partial_prefix(cur, THINK_CLOSE))
            {
                if cur.len() < 16 {
                    self.pending.extend_from_slice(cur);
                    break;
                }
            }
            self.bytes.push(cur[0]);
            self.last_output_newline = cur[0] == b'\n';
            i += 1;
        }
    }
}

fn is_partial_prefix(bytes: &[u8], prefix: &[u8]) -> bool {
    bytes.len() < prefix.len() && prefix.starts_with(bytes)
}

fn c_error(buf: &[c_char]) -> String {
    unsafe { CStr::from_ptr(buf.as_ptr()).to_string_lossy().into_owned() }
}

fn rate(tokens: i32, elapsed: std::time::Duration) -> f64 {
    let seconds = elapsed.as_secs_f64();
    if seconds > 0.0 {
        f64::from(tokens) / seconds
    } else {
        0.0
    }
}

#[repr(C)]
struct RawEngine {
    _private: [u8; 0],
}

#[repr(C)]
struct RawSession {
    _private: [u8; 0],
}

#[derive(Debug)]
struct Session {
    raw: NonNull<RawSession>,
}

impl Drop for Session {
    fn drop(&mut self) {
        unsafe {
            ds4_session_free(self.raw.as_ptr());
        }
    }
}

impl Session {
    fn common_prefix(&self, tokens: &Tokens) -> i32 {
        unsafe { ds4_session_common_prefix(self.raw.as_ptr(), &tokens.raw) }
    }

    fn pos(&self) -> i32 {
        unsafe { ds4_session_pos(self.raw.as_ptr()) }
    }

    fn ctx(&self) -> i32 {
        unsafe { ds4_session_ctx(self.raw.as_ptr()) }
    }
}

#[repr(C)]
#[derive(Debug)]
struct RawTokens {
    v: *mut c_int,
    len: c_int,
    cap: c_int,
}

impl Default for RawTokens {
    fn default() -> Self {
        Self {
            v: std::ptr::null_mut(),
            len: 0,
            cap: 0,
        }
    }
}

#[repr(C)]
struct RawEngineOptions {
    model_path: *const c_char,
    mtp_path: *const c_char,
    backend: c_int,
    n_threads: c_int,
    mtp_draft_tokens: c_int,
    mtp_margin: c_float,
    directional_steering_file: *const c_char,
    directional_steering_attn: c_float,
    directional_steering_ffn: c_float,
    warm_weights: bool,
    quality: bool,
}

#[repr(C)]
struct RawContextMemory {
    total_bytes: u64,
    raw_bytes: u64,
    compressed_bytes: u64,
    scratch_bytes: u64,
    prefill_cap: u32,
    raw_cap: u32,
    comp_cap: u32,
}

unsafe extern "C" {
    fn ds4_engine_open(out: *mut *mut RawEngine, opt: *const RawEngineOptions) -> c_int;
    fn ds4_engine_close(engine: *mut RawEngine);
    fn ds4_engine_summary(engine: *mut RawEngine);
    fn ds4_think_mode_enabled(mode: c_int) -> bool;
    fn ds4_think_mode_for_context(mode: c_int, ctx_size: c_int) -> c_int;
    fn ds4_think_max_min_context() -> u32;
    fn ds4_context_memory_estimate(backend: c_int, ctx_size: c_int) -> RawContextMemory;
    fn ds4_tokenize_rendered_chat(engine: *mut RawEngine, text: *const c_char, out: *mut RawTokens);
    fn ds4_encode_chat_prompt(
        engine: *mut RawEngine,
        system: *const c_char,
        prompt: *const c_char,
        think_mode: c_int,
        out: *mut RawTokens,
    );
    fn ds4_tokens_free(tokens: *mut RawTokens);
    fn ds4_tokens_push(tokens: *mut RawTokens, token: c_int);
    fn ds4_chat_begin(engine: *mut RawEngine, tokens: *mut RawTokens);
    fn ds4_chat_append_max_effort_prefix(engine: *mut RawEngine, tokens: *mut RawTokens);
    fn ds4_chat_append_message(
        engine: *mut RawEngine,
        tokens: *mut RawTokens,
        role: *const c_char,
        content: *const c_char,
    );
    fn ds4_chat_append_assistant_prefix(
        engine: *mut RawEngine,
        tokens: *mut RawTokens,
        think_mode: c_int,
    );
    fn ds4_token_text(engine: *mut RawEngine, token: c_int, len: *mut usize) -> *mut c_char;
    fn ds4_engine_generate_argmax(
        engine: *mut RawEngine,
        prompt: *const RawTokens,
        n_predict: c_int,
        ctx_size: c_int,
        emit: Option<unsafe extern "C" fn(*mut c_void, c_int)>,
        done: Option<unsafe extern "C" fn(*mut c_void)>,
        emit_ud: *mut c_void,
        progress: Option<unsafe extern "C" fn(*mut c_void, *const c_char, c_int, c_int)>,
        progress_ud: *mut c_void,
    ) -> c_int;
    fn ds4_token_eos(engine: *mut RawEngine) -> c_int;
    fn ds4_session_create(
        out: *mut *mut RawSession,
        engine: *mut RawEngine,
        ctx_size: c_int,
    ) -> c_int;
    fn ds4_session_free(session: *mut RawSession);
    fn ds4_session_set_progress(
        session: *mut RawSession,
        progress: Option<unsafe extern "C" fn(*mut c_void, *const c_char, c_int, c_int)>,
        progress_ud: *mut c_void,
    );
    fn ds4_session_sync(
        session: *mut RawSession,
        prompt: *const RawTokens,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;
    fn ds4_session_sample(
        session: *mut RawSession,
        temperature: c_float,
        top_k: c_int,
        top_p: c_float,
        min_p: c_float,
        rng: *mut u64,
    ) -> c_int;
    fn ds4_session_eval(
        session: *mut RawSession,
        token: c_int,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;
    fn ds4_session_common_prefix(session: *mut RawSession, prompt: *const RawTokens) -> c_int;
    fn ds4_session_pos(session: *mut RawSession) -> c_int;
    fn ds4_session_ctx(session: *mut RawSession) -> c_int;
    fn free(ptr: *mut c_void);
}

#[cfg(test)]
mod tests {
    use super::{Backend, EngineOptions, RawEngineOptions, ThinkMode, TokenPrinter};

    #[test]
    fn backend_values_match_c_enum() {
        assert_eq!(Backend::Metal.as_raw(), 0);
        assert_eq!(Backend::Cuda.as_raw(), 1);
        assert_eq!(Backend::Cpu.as_raw(), 2);
        assert_eq!(Backend::Cuda.name(), "cuda");
    }

    #[test]
    fn raw_options_keeps_c_bool_layout_small() {
        assert_eq!(std::mem::size_of::<bool>(), 1);
        assert!(std::mem::size_of::<RawEngineOptions>() >= 56);
    }

    #[test]
    fn options_default_runtime_flags_match_c_cli_inspect() {
        let options = EngineOptions::new("model.gguf", Backend::Cuda);
        assert_eq!(options.mtp_path, None);
        assert_eq!(options.n_threads, 0);
        assert_eq!(options.mtp_draft_tokens, 1);
        assert_eq!(options.mtp_margin, 3.0);
        assert_eq!(options.directional_steering_file, None);
        assert_eq!(options.directional_steering_attn, 0.0);
        assert_eq!(options.directional_steering_ffn, 0.0);
        assert!(!options.warm_weights);
        assert!(!options.quality);
    }

    #[test]
    fn think_mode_values_match_c_enum() {
        assert_eq!(ThinkMode::None.as_raw(), 0);
        assert_eq!(ThinkMode::High.as_raw(), 1);
        assert_eq!(ThinkMode::Max.as_raw(), 2);
        assert_eq!(ThinkMode::default_mode(), ThinkMode::High);
    }

    #[test]
    fn token_printer_matches_thinking_delimiter_rules() {
        let mut printer = TokenPrinter::new(ThinkMode::High);
        printer.write_token_text(b"reason");
        printer.write_token_text(b"</thi");
        printer.write_token_text(b"nk>answer");
        printer.finish_generation();
        assert_eq!(printer.into_bytes(), b"reason\nanswer\n");
    }

    #[test]
    fn token_printer_preserves_text_in_non_thinking_mode() {
        let mut printer = TokenPrinter::new(ThinkMode::None);
        printer.write_token_text(b"</think>answer");
        printer.finish_generation();
        assert_eq!(printer.into_bytes(), b"</think>answer\n");
    }
}
