use std::{fs, str};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CliParseAction {
    Exit,
    Continue,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CliParseResult {
    pub action: CliParseAction,
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CliConfig {
    pub model_path: String,
    pub prompt: Option<String>,
    pub dump_tokens: bool,
    pub inspect: bool,
    pub backend: CliBackend,
    pub warm_weights: bool,
    pub quality: bool,
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            model_path: "ds4flash.gguf".to_string(),
            prompt: None,
            dump_tokens: false,
            inspect: false,
            backend: CliBackend::default_backend(),
            warm_weights: false,
            quality: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CliBackend {
    Metal,
    Cuda,
    Cpu,
}

impl CliBackend {
    const fn default_backend() -> Self {
        if cfg!(target_os = "macos") {
            Self::Metal
        } else {
            Self::Cuda
        }
    }
}

#[derive(Debug, Default)]
struct CliState {
    config: CliConfig,
    imatrix_dataset: Option<String>,
    imatrix_out: Option<String>,
    perplexity_file: Option<String>,
}

const HELP: &str = "\
Usage: ds4 [(-p PROMPT | --prompt-file FILE)] [options]\n\
\n\
Invocation modes:\n\
  ds4\n\
  ds4 -p TEXT\n\
  ds4 --prompt-file FILE\n\
\n\
Model and runtime:\n\
  -m, --model FILE\n\
  --mtp FILE\n\
  --mtp-draft N\n\
  --mtp-margin F\n\
  -c, --ctx N\n\
  --metal\n\
  --cuda\n\
  --cpu\n\
  --backend NAME\n\
  -t, --threads N\n\
  --quality\n\
  --dir-steering-file FILE\n\
  --dir-steering-ffn F\n\
  --dir-steering-attn F\n\
  --warm-weights\n\
\n\
Prompt and generation:\n\
  -p, --prompt TEXT\n\
  --prompt-file FILE\n\
  -sys, --system TEXT\n\
  -n, --tokens N\n\
  --temp F\n\
  --top-p F\n\
  --min-p F\n\
  --seed N\n\
  --think\n\
  --think-max\n\
  --nothink\n\
\n\
Interactive commands:\n\
  /help\n\
  /think, /think-max, /nothink\n\
  /ctx N\n\
  /read FILE\n\
  /quit, /exit\n\
  Ctrl+C\n\
\n\
Diagnostics:\n\
  --inspect\n\
  --dump-tokens\n\
  --dump-logprobs FILE\n\
  --logprobs-top-k N\n\
  --perplexity-file FILE\n\
  --imatrix-dataset FILE\n\
  --imatrix-out FILE\n\
  --imatrix-max-prompts N\n\
  --imatrix-max-tokens N\n\
  --head-test\n\
  --first-token-test\n\
  --metal-graph-test\n\
  --metal-graph-full-test\n\
  --metal-graph-prompt-test\n\
\n\
Normal CLI commands:\n\
  ./ds4\n\
  ./ds4 -p \"Scrivi una storia su una papera scansafatiche\"\n\
  ./ds4 --think-max --prompt-file prompt.txt --ctx 393216\n\
\n\
Notes:\n\
  -h, --help\n";

pub fn parse_cli<I, S>(args: I) -> CliParseResult
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    match parse_cli_config(args) {
        Ok(_) => CliParseResult {
            action: CliParseAction::Continue,
            exit_code: 99,
            stdout: String::new(),
            stderr: "ds4-rs: M8.3 parser-only implementation reached model-backed path\n"
                .to_string(),
        },
        Err(result) => result,
    }
}

pub fn parse_cli_config<I, S>(args: I) -> Result<CliConfig, CliParseResult>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let argv: Vec<String> = args.into_iter().map(Into::into).collect();
    let mut state = CliState::default();
    let mut i = 0usize;
    while i < argv.len() {
        let arg = argv[i].as_str();
        match arg {
            "-h" | "--help" => return Err(exit(0, HELP, "")),
            "-p" | "--prompt" => {
                let value = match need_arg(&argv, &mut i, arg) {
                    Ok(value) => value,
                    Err(result) => return Err(result),
                };
                if state.config.prompt.is_some() {
                    return Err(exit(2, "", "ds4: specify only one prompt source\n"));
                }
                state.config.prompt = Some(value.to_string());
            }
            "--prompt-file" => {
                let value = match need_arg(&argv, &mut i, arg) {
                    Ok(value) => value,
                    Err(result) => return Err(result),
                };
                if state.config.prompt.is_some() {
                    return Err(exit(2, "", "ds4: specify only one prompt source\n"));
                }
                state.config.prompt = Some(read_prompt_file(value)?);
            }
            "-m" | "--model" => {
                let value = match need_arg(&argv, &mut i, arg) {
                    Ok(value) => value.to_string(),
                    Err(result) => return Err(result),
                };
                state.config.model_path = value;
            }
            "-sys"
            | "--system"
            | "--mtp"
            | "--dir-steering-file"
            | "--dump-logprobs"
            | "--perplexity-file"
            | "--imatrix-dataset"
            | "--imatrix-out" => {
                let value = match need_arg(&argv, &mut i, arg) {
                    Ok(value) => value.to_string(),
                    Err(result) => return Err(result),
                };
                match arg {
                    "--perplexity-file" => state.perplexity_file = Some(value),
                    "--imatrix-dataset" => state.imatrix_dataset = Some(value),
                    "--imatrix-out" => state.imatrix_out = Some(value),
                    _ => {}
                }
            }
            "--mtp-draft"
            | "-n"
            | "--tokens"
            | "-c"
            | "--ctx"
            | "-t"
            | "--threads"
            | "--logprobs-top-k"
            | "--imatrix-max-prompts"
            | "--imatrix-max-tokens" => {
                let value = match need_arg(&argv, &mut i, arg) {
                    Ok(value) => value,
                    Err(result) => return Err(result),
                };
                if parse_positive_i32(value).is_none() {
                    return Err(exit(
                        2,
                        "",
                        &format!("ds4: invalid value for {arg}: {value}\n"),
                    ));
                }
            }
            "--seed" => {
                let value = match need_arg(&argv, &mut i, arg) {
                    Ok(value) => value,
                    Err(result) => return Err(result),
                };
                if parse_positive_u64(value).is_none() {
                    return Err(exit(
                        2,
                        "",
                        &format!("ds4: invalid value for {arg}: {value}\n"),
                    ));
                }
            }
            "--mtp-margin" => {
                let value = match need_arg(&argv, &mut i, arg) {
                    Ok(value) => value,
                    Err(result) => return Err(result),
                };
                if parse_float_range(value, 0.0, 1000.0).is_none() {
                    return Err(exit(
                        2,
                        "",
                        &format!("ds4: invalid value for {arg}: {value}\n"),
                    ));
                }
            }
            "--temp" => {
                let value = match need_arg(&argv, &mut i, arg) {
                    Ok(value) => value,
                    Err(result) => return Err(result),
                };
                if parse_float_range(value, 0.0, 100.0).is_none() {
                    return Err(exit(
                        2,
                        "",
                        &format!("ds4: invalid value for {arg}: {value}\n"),
                    ));
                }
            }
            "--top-p" | "--min-p" => {
                let value = match need_arg(&argv, &mut i, arg) {
                    Ok(value) => value,
                    Err(result) => return Err(result),
                };
                if parse_float_range(value, 0.0, 1.0).is_none() {
                    return Err(exit(
                        2,
                        "",
                        &format!("ds4: invalid value for {arg}: {value}\n"),
                    ));
                }
            }
            "--dir-steering-ffn" | "--dir-steering-attn" => {
                let value = match need_arg(&argv, &mut i, arg) {
                    Ok(value) => value,
                    Err(result) => return Err(result),
                };
                if parse_float_range(value, -100.0, 100.0).is_none() {
                    return Err(exit(
                        2,
                        "",
                        &format!("ds4: invalid value for {arg}: {value}\n"),
                    ));
                }
            }
            "--backend" => {
                let value = match need_arg(&argv, &mut i, arg) {
                    Ok(value) => value,
                    Err(result) => return Err(result),
                };
                let Some(backend) = parse_backend(value) else {
                    return Err(exit(
                        2,
                        "",
                        &format!(
                            "ds4: invalid backend: {value}\n\
                             ds4: valid backends are: metal, cuda, cpu\n"
                        ),
                    ));
                };
                state.config.backend = backend;
            }
            "--cpu"
            | "--metal"
            | "--cuda"
            | "--quality"
            | "--dump-tokens"
            | "--think"
            | "--think-max"
            | "--nothink"
            | "--head-test"
            | "--first-token-test"
            | "--metal-graph-test"
            | "--metal-graph-full-test"
            | "--metal-graph-prompt-test"
            | "--inspect"
            | "--warm-weights" => match arg {
                "--cpu" => state.config.backend = CliBackend::Cpu,
                "--metal" => state.config.backend = CliBackend::Metal,
                "--cuda" => state.config.backend = CliBackend::Cuda,
                "--quality" => state.config.quality = true,
                "--dump-tokens" => state.config.dump_tokens = true,
                "--inspect" => state.config.inspect = true,
                "--warm-weights" => state.config.warm_weights = true,
                _ => {}
            },
            "--metal-graph-generate" => {
                return Err(exit(
                    2,
                    "",
                    "ds4: --metal-graph-generate was removed; --metal is the graph path\n",
                ));
            }
            "--server" => {
                return Err(exit(2, "", "ds4: use ds4-server for the HTTP server\n"));
            }
            _ => {
                return Err(exit(2, "", &format!("ds4: unknown option: {arg}\n{HELP}")));
            }
        }
        i += 1;
    }

    if state.imatrix_out.is_some() && state.imatrix_dataset.is_none() {
        return Err(exit(
            2,
            "",
            "ds4: --imatrix-out requires --imatrix-dataset\n",
        ));
    }
    if state.imatrix_dataset.is_some() && state.imatrix_out.is_none() {
        return Err(exit(
            2,
            "",
            "ds4: --imatrix-dataset requires --imatrix-out\n",
        ));
    }
    if state.perplexity_file.is_some() && state.config.prompt.is_some() {
        return Err(exit(
            2,
            "",
            "ds4: --perplexity-file does not use -p/--prompt-file\n",
        ));
    }
    if state.config.dump_tokens && state.config.prompt.is_none() {
        return Err(exit(
            2,
            "",
            "ds4: --dump-tokens requires -p or --prompt-file\n",
        ));
    }

    Ok(state.config)
}

fn need_arg<'a>(argv: &'a [String], idx: &mut usize, opt: &str) -> Result<&'a str, CliParseResult> {
    if *idx + 1 >= argv.len() {
        return Err(exit(2, "", &format!("ds4: missing value for {opt}\n")));
    }
    *idx += 1;
    Ok(argv[*idx].as_str())
}

fn read_prompt_file(path: &str) -> Result<String, CliParseResult> {
    let bytes = fs::read(path)
        .map_err(|_| exit(2, "", &format!("ds4: failed to open prompt file: {path}\n")))?;
    let text = str::from_utf8(&bytes)
        .map_err(|_| exit(2, "", &format!("ds4: failed to read prompt file: {path}\n")))?;
    Ok(text.to_string())
}

fn parse_positive_i32(s: &str) -> Option<i32> {
    let v = s.parse::<i64>().ok()?;
    if (1..=i32::MAX as i64).contains(&v) {
        Some(v as i32)
    } else {
        None
    }
}

fn parse_positive_u64(s: &str) -> Option<u64> {
    let v = s.parse::<u64>().ok()?;
    (v > 0).then_some(v)
}

fn parse_float_range(s: &str, min: f32, max: f32) -> Option<f32> {
    let v = s.parse::<f32>().ok()?;
    (v.is_finite() && v >= min && v <= max).then_some(v)
}

fn parse_backend(value: &str) -> Option<CliBackend> {
    match value {
        "metal" => Some(CliBackend::Metal),
        "cuda" => Some(CliBackend::Cuda),
        "cpu" => Some(CliBackend::Cpu),
        _ => None,
    }
}

fn exit(exit_code: i32, stdout: &str, stderr: &str) -> CliParseResult {
    CliParseResult {
        action: CliParseAction::Exit,
        exit_code,
        stdout: stdout.to_string(),
        stderr: stderr.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(args: &[&str]) -> CliParseResult {
        parse_cli(args.iter().copied())
    }

    #[test]
    fn help_exits_successfully_without_stderr() {
        let result = run(&["--help"]);
        assert_eq!(result.exit_code, 0);
        assert!(result.stderr.is_empty());
        assert!(result.stdout.contains("Usage: ds4"));
        assert!(result.stdout.contains("--dump-logprobs FILE"));
    }

    #[test]
    fn parser_errors_match_current_c_categories() {
        let result = run(&["--backend", "vulkan"]);
        assert_eq!(result.exit_code, 2);
        assert!(result.stderr.contains("ds4: invalid backend: vulkan"));
        assert!(result
            .stderr
            .contains("ds4: valid backends are: metal, cuda, cpu"));

        let result = run(&["--imatrix-out", "out.dat"]);
        assert_eq!(result.exit_code, 2);
        assert_eq!(
            result.stderr,
            "ds4: --imatrix-out requires --imatrix-dataset\n"
        );
    }

    #[test]
    fn dump_tokens_rejects_missing_prompt_before_model_path() {
        let result = run(&["--dump-tokens"]);
        assert_eq!(result.exit_code, 2);
        assert_eq!(
            result.stderr,
            "ds4: --dump-tokens requires -p or --prompt-file\n"
        );
    }

    #[test]
    fn config_retains_model_prompt_and_dump_tokens() {
        let config = parse_cli_config([
            "--dump-tokens",
            "-m",
            "tokenizer.gguf",
            "-p",
            "prompt text",
            "--think-max",
            "--ctx",
            "393216",
            "--system",
            "ignored",
        ])
        .expect("valid dump-token config");

        assert_eq!(config.model_path, "tokenizer.gguf");
        assert_eq!(config.prompt.as_deref(), Some("prompt text"));
        assert!(config.dump_tokens);
        assert!(!config.inspect);
    }

    #[test]
    fn config_retains_inspect_backend_and_runtime_flags() {
        let config = parse_cli_config([
            "--inspect",
            "--backend",
            "cuda",
            "--warm-weights",
            "--quality",
            "-m",
            "model.gguf",
        ])
        .expect("valid inspect config");

        assert_eq!(config.model_path, "model.gguf");
        assert_eq!(config.backend, CliBackend::Cuda);
        assert!(config.inspect);
        assert!(config.warm_weights);
        assert!(config.quality);
        assert!(!config.dump_tokens);
    }
}
