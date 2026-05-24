use ds4_gguf::write_agent_rendered_context_replay;
use std::io;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut out = io::BufWriter::new(io::stdout());
    write_agent_rendered_context_replay(&mut out)?;
    Ok(())
}
