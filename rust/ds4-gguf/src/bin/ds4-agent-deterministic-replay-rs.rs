use ds4_gguf::write_agent_deterministic_replay;
use std::io;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut out = io::BufWriter::new(io::stdout());
    write_agent_deterministic_replay(&mut out)?;
    Ok(())
}
