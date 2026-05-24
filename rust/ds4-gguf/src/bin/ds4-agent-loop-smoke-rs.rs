use ds4_gguf::write_agent_loop_smoke;
use std::io;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut out = io::BufWriter::new(io::stdout());
    write_agent_loop_smoke(&mut out)?;
    Ok(())
}
