use ds4_engine::{
    Backend, Engine, EngineOptions, RuntimeGraphRoute, TopLogprobScore,
    RUNTIME_GRAPH_ROUTE_UNSUPPORTED_CODE, RUNTIME_GRAPH_ROUTE_VALID_VALUES,
};
use std::error::Error;
use std::fs;
use std::io::{self, Write};
use std::process;

const HELP: &str = "\
Usage: ds4-runtime-official-vectors-rs --model FILE [options]\n\
\n\
M10.9c Rust runtime official-vector capture.\n\
\n\
Options:\n\
  -m, --model FILE\n\
  -v, --vectors FILE\n\
  --backend NAME | --cuda | --metal | --cpu\n\
  --runtime-graph ROUTE\n\
  --top-k N\n";

const DEFAULT_VECTOR_FILE: &str = "tests/test-vectors/official.vec";
const DEFAULT_TOP_K: usize = 20;
const LOGPROB_ABS_TOLERANCE: f32 = 4.0;

#[derive(Debug)]
struct Config {
    model_path: String,
    vector_path: String,
    backend: Backend,
    route: RuntimeGraphRoute,
    top_k: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model_path: "ds4flash.gguf".to_string(),
            vector_path: DEFAULT_VECTOR_FILE.to_string(),
            backend: Backend::Cuda,
            route: RuntimeGraphRoute::Graph,
            top_k: DEFAULT_TOP_K,
        }
    }
}

#[derive(Debug)]
struct CliExit {
    code: i32,
    stdout: String,
    stderr: String,
}

#[derive(Clone, Debug, PartialEq)]
struct VecTop {
    bytes: Vec<u8>,
    logprob: f32,
}

#[derive(Clone, Debug, PartialEq)]
struct VecStep {
    index: usize,
    selected: Vec<u8>,
    top_count: usize,
    top: Vec<VecTop>,
}

#[derive(Clone, Debug, PartialEq)]
struct VecCase {
    id: String,
    ctx: i32,
    nsteps: usize,
    prompt_path: String,
    steps: Vec<VecStep>,
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
            stderr: "ds4-runtime-official-vectors-rs: --runtime-graph graph requires cuda or metal backend\n"
                .to_string(),
        })?);
    }

    let vector_text = fs::read_to_string(&config.vector_path)?;
    let cases = parse_vec_text(&vector_text)?;

    let engine_options = EngineOptions::new(&config.model_path, config.backend);
    let engine = Engine::open(&engine_options)?;

    let mut out = Vec::new();
    write_capture(&mut out, &config, &engine, &cases)?;
    io::stdout().lock().write_all(&out)?;
    Ok(0)
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
            "-v" | "--vectors" => config.vector_path = need_arg(&argv, &mut i, arg)?.to_string(),
            "--backend" => {
                let value = need_arg(&argv, &mut i, arg)?;
                config.backend = Backend::parse(value).ok_or_else(|| {
                    exit(
                        2,
                        "",
                        &format!(
                            "ds4-runtime-official-vectors-rs: invalid backend: {value}\n\
                             ds4-runtime-official-vectors-rs: valid backends are: metal, cuda, cpu\n"
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
                            "ds4-runtime-official-vectors-rs: invalid runtime graph route: {value}\n\
                             ds4-runtime-official-vectors-rs: valid runtime graph routes are: {RUNTIME_GRAPH_ROUTE_VALID_VALUES}\n"
                        ),
                    )
                })?;
            }
            "--top-k" => {
                config.top_k = parse_top_k(need_arg(&argv, &mut i, arg)?, arg)?;
            }
            "--cuda" => config.backend = Backend::Cuda,
            "--metal" => config.backend = Backend::Metal,
            "--cpu" => config.backend = Backend::Cpu,
            _ => {
                return Err(exit(
                    2,
                    "",
                    &format!("ds4-runtime-official-vectors-rs: unknown option: {arg}\n{HELP}"),
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
            &format!("ds4-runtime-official-vectors-rs: missing value for {opt}\n"),
        ));
    }
    *idx += 1;
    Ok(argv[*idx].as_str())
}

fn parse_top_k(value: &str, opt: &str) -> Result<usize, CliExit> {
    match value.parse::<usize>() {
        Ok(v) if (1..=128).contains(&v) => Ok(v),
        _ => Err(exit(
            2,
            "",
            &format!("ds4-runtime-official-vectors-rs: invalid value for {opt}: {value}\n"),
        )),
    }
}

fn parse_vec_text(text: &str) -> Result<Vec<VecCase>, String> {
    let mut cases = Vec::new();
    let mut current: Option<VecCaseBuilder> = None;
    let mut current_step: Option<usize> = None;

    for (lineno, raw_line) in text.lines().enumerate() {
        let lineno = lineno + 1;
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        let kind = parts.next().unwrap_or_default();
        match kind {
            "case" => {
                if current.is_some() {
                    return Err(format!("{lineno}: nested case"));
                }
                let id = parts
                    .next()
                    .ok_or_else(|| format!("{lineno}: missing case id"))?;
                let ctx = parse_i32(parts.next(), lineno, "ctx")?;
                let nsteps = parse_usize(parts.next(), lineno, "nsteps")?;
                let prompt_path = parts
                    .next()
                    .ok_or_else(|| format!("{lineno}: missing prompt path"))?;
                if parts.next().is_some() {
                    return Err(format!("{lineno}: trailing case fields"));
                }
                if nsteps == 0 {
                    return Err(format!("{lineno}: nsteps must be positive"));
                }
                current = Some(VecCaseBuilder {
                    id: id.to_string(),
                    ctx,
                    nsteps,
                    prompt_path: prompt_path.to_string(),
                    steps: vec![None; nsteps],
                });
                current_step = None;
            }
            "step" => {
                let builder = current
                    .as_mut()
                    .ok_or_else(|| format!("{lineno}: step outside case"))?;
                let index = parse_usize(parts.next(), lineno, "step index")?;
                let selected = parts
                    .next()
                    .ok_or_else(|| format!("{lineno}: missing selected token hex"))
                    .and_then(|hex| parse_hex(hex, lineno))?;
                let top_count = parse_usize(parts.next(), lineno, "top count")?;
                if parts.next().is_some() {
                    return Err(format!("{lineno}: trailing step fields"));
                }
                if index >= builder.nsteps {
                    return Err(format!("{lineno}: step index out of range"));
                }
                builder.steps[index] = Some(VecStep {
                    index,
                    selected,
                    top_count,
                    top: Vec::new(),
                });
                current_step = Some(index);
            }
            "top" => {
                let builder = current
                    .as_mut()
                    .ok_or_else(|| format!("{lineno}: top outside case"))?;
                let step_index =
                    current_step.ok_or_else(|| format!("{lineno}: top before step"))?;
                let token = parts
                    .next()
                    .ok_or_else(|| format!("{lineno}: missing top token hex"))
                    .and_then(|hex| parse_hex(hex, lineno))?;
                let logprob = parse_f32(parts.next(), lineno, "top logprob")?;
                if parts.next().is_some() {
                    return Err(format!("{lineno}: trailing top fields"));
                }
                let step = builder.steps[step_index]
                    .as_mut()
                    .ok_or_else(|| format!("{lineno}: top for missing step"))?;
                if step.top.len() >= step.top_count {
                    return Err(format!("{lineno}: too many top entries"));
                }
                step.top.push(VecTop {
                    bytes: token,
                    logprob,
                });
            }
            "end" => {
                if parts.next().is_some() {
                    return Err(format!("{lineno}: trailing end fields"));
                }
                let builder = current
                    .take()
                    .ok_or_else(|| format!("{lineno}: end outside case"))?;
                cases.push(builder.finish(lineno)?);
                current_step = None;
            }
            _ => return Err(format!("{lineno}: unexpected vector line")),
        }
    }

    if current.is_some() {
        return Err("unterminated vector case".to_string());
    }
    Ok(cases)
}

#[derive(Debug)]
struct VecCaseBuilder {
    id: String,
    ctx: i32,
    nsteps: usize,
    prompt_path: String,
    steps: Vec<Option<VecStep>>,
}

impl VecCaseBuilder {
    fn finish(self, lineno: usize) -> Result<VecCase, String> {
        let mut steps = Vec::with_capacity(self.steps.len());
        for (index, step) in self.steps.into_iter().enumerate() {
            let step = step.ok_or_else(|| format!("{lineno}: missing step {index}"))?;
            if step.top.len() != step.top_count {
                return Err(format!(
                    "{lineno}: step {index} top-count {} != {}",
                    step.top.len(),
                    step.top_count
                ));
            }
            steps.push(step);
        }
        Ok(VecCase {
            id: self.id,
            ctx: self.ctx,
            nsteps: self.nsteps,
            prompt_path: self.prompt_path,
            steps,
        })
    }
}

fn parse_i32(value: Option<&str>, lineno: usize, label: &str) -> Result<i32, String> {
    value
        .ok_or_else(|| format!("{lineno}: missing {label}"))?
        .parse::<i32>()
        .map_err(|_| format!("{lineno}: invalid {label}"))
}

fn parse_usize(value: Option<&str>, lineno: usize, label: &str) -> Result<usize, String> {
    value
        .ok_or_else(|| format!("{lineno}: missing {label}"))?
        .parse::<usize>()
        .map_err(|_| format!("{lineno}: invalid {label}"))
}

fn parse_f32(value: Option<&str>, lineno: usize, label: &str) -> Result<f32, String> {
    value
        .ok_or_else(|| format!("{lineno}: missing {label}"))?
        .parse::<f32>()
        .map_err(|_| format!("{lineno}: invalid {label}"))
}

fn parse_hex(value: &str, lineno: usize) -> Result<Vec<u8>, String> {
    if value.len() % 2 != 0 {
        return Err(format!("{lineno}: odd-length hex"));
    }
    let mut out = Vec::with_capacity(value.len() / 2);
    let bytes = value.as_bytes();
    for pair in bytes.chunks_exact(2) {
        let hi = hex_nibble(pair[0]).ok_or_else(|| format!("{lineno}: invalid hex"))?;
        let lo = hex_nibble(pair[1]).ok_or_else(|| format!("{lineno}: invalid hex"))?;
        out.push((hi << 4) | lo);
    }
    Ok(out)
}

fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn write_capture<W: Write>(
    out: &mut W,
    config: &Config,
    engine: &Engine,
    cases: &[VecCase],
) -> io::Result<()> {
    writeln!(out, "{{")?;
    writeln!(
        out,
        "  \"schema\": \"ds4.runtime_graph_official_vectors.rust.v1\","
    )?;
    writeln!(out, "  \"source\": \"ds4-runtime-official-vectors-rs\",")?;
    write!(out, "  \"runtime_graph_route\": ")?;
    write_json_str(out, config.route.name())?;
    writeln!(out, ",")?;
    write!(out, "  \"backend\": ")?;
    write_json_str(out, config.backend.name())?;
    writeln!(out, ",")?;
    write!(out, "  \"model_path\": ")?;
    write_json_str(out, &config.model_path)?;
    writeln!(out, ",")?;
    write!(out, "  \"vector_file\": ")?;
    write_json_str(out, &config.vector_path)?;
    writeln!(out, ",")?;
    writeln!(out, "  \"top_k\": {},", config.top_k)?;
    writeln!(
        out,
        "  \"logprob_abs_tolerance\": {},",
        LOGPROB_ABS_TOLERANCE
    )?;
    writeln!(out, "  \"cases\": [")?;
    for (case_index, case) in cases.iter().enumerate() {
        if case_index > 0 {
            writeln!(out, ",")?;
        }
        write_case(out, engine, config, case)?;
    }
    writeln!(out, "\n  ]")?;
    writeln!(out, "}}")?;
    Ok(())
}

fn write_case<W: Write>(
    out: &mut W,
    engine: &Engine,
    config: &Config,
    case: &VecCase,
) -> io::Result<()> {
    write!(out, "    {{\"id\":")?;
    write_json_str(out, &case.id)?;
    write!(
        out,
        ",\"ctx\":{},\"nsteps\":{},\"prompt_path\":",
        case.ctx, case.nsteps
    )?;
    write_json_str(out, &case.prompt_path)?;

    if let Some(reason) = case_skip_reason(&case.id) {
        write!(out, ",\"skipped\":true,\"skip_reason\":")?;
        write_json_str(out, reason)?;
        write!(out, ",\"steps\":[]}}")?;
        return Ok(());
    }

    let prompt_text = fs::read_to_string(&case.prompt_path)?;
    let prompt = engine
        .encode_chat_prompt("", &prompt_text, ds4_engine::ThinkMode::None)
        .map_err(io::Error::other)?;
    let mut session = engine
        .create_server_session(case.ctx)
        .map_err(io::Error::other)?;
    session.sync_prompt(&prompt).map_err(io::Error::other)?;
    write!(
        out,
        ",\"skipped\":false,\"prompt_tokens\":{},\"steps\":[",
        prompt.len()
    )?;

    for (step_index, step) in case.steps.iter().enumerate() {
        if step_index > 0 {
            write!(out, ",")?;
        }
        let scores = session.top_logprobs(config.top_k as i32);
        let selected = session.argmax();
        let selected_bytes = engine.token_text(selected);
        write!(
            out,
            "{{\"step\":{},\"selected_token\":{},",
            step.index, selected
        )?;
        write!(out, "\"selected_bytes_hex\":\"")?;
        write_hex(out, &selected_bytes)?;
        write!(out, "\",\"expected_selected_hex\":\"")?;
        write_hex(out, &step.selected)?;
        write!(
            out,
            "\",\"selected_matches_expected\":{},\"top_logprobs\":[",
            selected_bytes == step.selected
        )?;
        for (score_index, score) in scores.iter().enumerate() {
            if score_index > 0 {
                write!(out, ",")?;
            }
            write_score(out, score)?;
        }
        write!(out, "],\"official_top\":[")?;
        for (top_index, top) in step.top.iter().enumerate() {
            if top_index > 0 {
                write!(out, ",")?;
            }
            write_official_top(out, &scores, top)?;
        }
        write!(out, "]}}")?;
        if step_index + 1 < case.steps.len() {
            session.eval_token(selected).map_err(io::Error::other)?;
        }
    }

    write!(out, "]}}")?;
    Ok(())
}

fn case_skip_reason(case_id: &str) -> Option<&'static str> {
    match case_id {
        "long_memory_archive" => Some("API/official graph mismatch"),
        _ => None,
    }
}

fn write_score<W: Write>(out: &mut W, score: &TopLogprobScore) -> io::Result<()> {
    write!(out, "{{\"id\":{},\"bytes_hex\":\"", score.id)?;
    write_hex(out, &score.bytes)?;
    write!(out, "\",\"logit\":")?;
    write_json_f32(out, score.logit)?;
    write!(out, ",\"logprob\":")?;
    write_json_f32(out, score.logprob)?;
    write!(out, "}}")?;
    Ok(())
}

fn write_official_top<W: Write>(
    out: &mut W,
    scores: &[TopLogprobScore],
    top: &VecTop,
) -> io::Result<()> {
    write!(out, "{{\"bytes_hex\":\"")?;
    write_hex(out, &top.bytes)?;
    write!(out, "\",\"official_logprob\":")?;
    write_json_f32(out, top.logprob)?;
    let local = scores.iter().find(|score| score.bytes == top.bytes);
    write!(out, ",\"found\":{}", local.is_some())?;
    if let Some(score) = local {
        write!(out, ",\"local_score\":")?;
        write_score(out, score)?;
        write!(out, ",\"abs_delta\":")?;
        write_json_f32(out, (score.logprob - top.logprob).abs())?;
    }
    write!(out, "}}")?;
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

fn write_json_f32<W: Write>(out: &mut W, value: f32) -> io::Result<()> {
    if value.is_nan() {
        write!(out, "\"nan\"")
    } else if value.is_infinite() {
        if value.is_sign_negative() {
            write!(out, "\"-inf\"")
        } else {
            write!(out, "\"inf\"")
        }
    } else {
        write!(out, "{value:.9}")
    }
}

fn write_hex<W: Write>(out: &mut W, bytes: &[u8]) -> io::Result<()> {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for &byte in bytes {
        out.write_all(&[HEX[(byte >> 4) as usize], HEX[(byte & 0xf) as usize]])?;
    }
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
    fn parses_compact_vector_fixture() {
        let text = "\
case short 16 1 prompts/short.txt
step 0 41 1
top 41 0
end
";
        let cases = parse_vec_text(text).unwrap();
        assert_eq!(cases.len(), 1);
        assert_eq!(cases[0].id, "short");
        assert_eq!(cases[0].ctx, 16);
        assert_eq!(cases[0].steps[0].selected, b"A");
        assert_eq!(cases[0].steps[0].top[0].bytes, b"A");
    }

    #[test]
    fn rejects_missing_top_entries() {
        let text = "\
case short 16 1 prompts/short.txt
step 0 41 1
end
";
        assert!(parse_vec_text(text).is_err());
    }
}
