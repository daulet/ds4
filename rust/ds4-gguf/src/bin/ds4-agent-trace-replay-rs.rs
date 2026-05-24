use ds4_gguf::write_agent_trace_replay_oracle;
use std::io;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut out = io::BufWriter::new(io::stdout());
    write_agent_trace_replay_oracle(&mut out)?;
    Ok(())
}
