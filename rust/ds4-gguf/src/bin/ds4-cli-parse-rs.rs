use ds4_gguf::cli_parse::parse_cli;
use std::io::{self, Write};

fn main() {
    let result = parse_cli(std::env::args().skip(1));
    let mut stdout = io::stdout().lock();
    let mut stderr = io::stderr().lock();
    let _ = stdout.write_all(result.stdout.as_bytes());
    let _ = stderr.write_all(result.stderr.as_bytes());
    std::process::exit(result.exit_code);
}
