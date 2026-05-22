use ds4_gguf::{
    cli_parse::{parse_cli_config, CliParseResult},
    cli_token_dump::write_cli_token_dump,
    parse_gguf, Ds4Tokenizer,
};
use std::fs;
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
    if !config.dump_tokens {
        eprintln!("ds4-rs: M8.5 token-dump implementation reached non-dump path");
        return Ok(99);
    }
    let prompt = config.prompt.unwrap_or_default();
    let bytes = fs::read(&config.model_path)?;
    let gguf = parse_gguf(&bytes)?;
    let tokenizer = Ds4Tokenizer::from_gguf(&gguf)?;
    let mut out = io::BufWriter::new(io::stdout().lock());
    write_cli_token_dump(&mut out, &tokenizer, &prompt)?;
    out.flush()?;
    Ok(0)
}

fn write_parse_result(result: CliParseResult) -> io::Result<i32> {
    let mut stdout = io::stdout().lock();
    let mut stderr = io::stderr().lock();
    stdout.write_all(result.stdout.as_bytes())?;
    stderr.write_all(result.stderr.as_bytes())?;
    Ok(result.exit_code)
}
