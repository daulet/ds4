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

#[derive(Debug, Default)]
struct CliState {
    prompt: Option<String>,
    dump_tokens: bool,
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
    let argv: Vec<String> = args.into_iter().map(Into::into).collect();
    let mut state = CliState::default();
    let mut i = 0usize;
    while i < argv.len() {
        let arg = argv[i].as_str();
        match arg {
            "-h" | "--help" => return exit(0, HELP, ""),
            "-p" | "--prompt" => {
                let value = match need_arg(&argv, &mut i, arg) {
                    Ok(value) => value,
                    Err(result) => return result,
                };
                if state.prompt.is_some() {
                    return exit(2, "", "ds4: specify only one prompt source\n");
                }
                state.prompt = Some(value.to_string());
            }
            "--prompt-file" => {
                let value = match need_arg(&argv, &mut i, arg) {
                    Ok(value) => value,
                    Err(result) => return result,
                };
                if state.prompt.is_some() {
                    return exit(2, "", "ds4: specify only one prompt source\n");
                }
                if std::fs::read_to_string(value).is_err() {
                    return exit(
                        2,
                        "",
                        &format!("ds4: failed to open prompt file: {value}\n"),
                    );
                }
                state.prompt = Some(String::new());
            }
            "-sys"
            | "--system"
            | "-m"
            | "--model"
            | "--mtp"
            | "--dir-steering-file"
            | "--dump-logprobs"
            | "--perplexity-file"
            | "--imatrix-dataset"
            | "--imatrix-out" => {
                let value = match need_arg(&argv, &mut i, arg) {
                    Ok(value) => value.to_string(),
                    Err(result) => return result,
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
                    Err(result) => return result,
                };
                if parse_positive_i32(value).is_none() {
                    return exit(2, "", &format!("ds4: invalid value for {arg}: {value}\n"));
                }
            }
            "--seed" => {
                let value = match need_arg(&argv, &mut i, arg) {
                    Ok(value) => value,
                    Err(result) => return result,
                };
                if parse_positive_u64(value).is_none() {
                    return exit(2, "", &format!("ds4: invalid value for {arg}: {value}\n"));
                }
            }
            "--mtp-margin" => {
                let value = match need_arg(&argv, &mut i, arg) {
                    Ok(value) => value,
                    Err(result) => return result,
                };
                if parse_float_range(value, 0.0, 1000.0).is_none() {
                    return exit(2, "", &format!("ds4: invalid value for {arg}: {value}\n"));
                }
            }
            "--temp" => {
                let value = match need_arg(&argv, &mut i, arg) {
                    Ok(value) => value,
                    Err(result) => return result,
                };
                if parse_float_range(value, 0.0, 100.0).is_none() {
                    return exit(2, "", &format!("ds4: invalid value for {arg}: {value}\n"));
                }
            }
            "--top-p" | "--min-p" => {
                let value = match need_arg(&argv, &mut i, arg) {
                    Ok(value) => value,
                    Err(result) => return result,
                };
                if parse_float_range(value, 0.0, 1.0).is_none() {
                    return exit(2, "", &format!("ds4: invalid value for {arg}: {value}\n"));
                }
            }
            "--dir-steering-ffn" | "--dir-steering-attn" => {
                let value = match need_arg(&argv, &mut i, arg) {
                    Ok(value) => value,
                    Err(result) => return result,
                };
                if parse_float_range(value, -100.0, 100.0).is_none() {
                    return exit(2, "", &format!("ds4: invalid value for {arg}: {value}\n"));
                }
            }
            "--backend" => {
                let value = match need_arg(&argv, &mut i, arg) {
                    Ok(value) => value,
                    Err(result) => return result,
                };
                if !matches!(value, "metal" | "cuda" | "cpu") {
                    return exit(
                        2,
                        "",
                        &format!(
                            "ds4: invalid backend: {value}\n\
                             ds4: valid backends are: metal, cuda, cpu\n"
                        ),
                    );
                }
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
            | "--warm-weights" => {
                if arg == "--dump-tokens" {
                    state.dump_tokens = true;
                }
            }
            "--metal-graph-generate" => {
                return exit(
                    2,
                    "",
                    "ds4: --metal-graph-generate was removed; --metal is the graph path\n",
                );
            }
            "--server" => return exit(2, "", "ds4: use ds4-server for the HTTP server\n"),
            _ => {
                return exit(2, "", &format!("ds4: unknown option: {arg}\n{HELP}"));
            }
        }
        i += 1;
    }

    if state.imatrix_out.is_some() && state.imatrix_dataset.is_none() {
        return exit(2, "", "ds4: --imatrix-out requires --imatrix-dataset\n");
    }
    if state.imatrix_dataset.is_some() && state.imatrix_out.is_none() {
        return exit(2, "", "ds4: --imatrix-dataset requires --imatrix-out\n");
    }
    if state.perplexity_file.is_some() && state.prompt.is_some() {
        return exit(
            2,
            "",
            "ds4: --perplexity-file does not use -p/--prompt-file\n",
        );
    }
    if state.dump_tokens && state.prompt.is_none() {
        return exit(2, "", "ds4: --dump-tokens requires -p or --prompt-file\n");
    }

    CliParseResult {
        action: CliParseAction::Continue,
        exit_code: 99,
        stdout: String::new(),
        stderr: "ds4-rs: M8.3 parser-only implementation reached model-backed path\n".to_string(),
    }
}

fn need_arg<'a>(argv: &'a [String], idx: &mut usize, opt: &str) -> Result<&'a str, CliParseResult> {
    if *idx + 1 >= argv.len() {
        return Err(exit(2, "", &format!("ds4: missing value for {opt}\n")));
    }
    *idx += 1;
    Ok(argv[*idx].as_str())
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
}
