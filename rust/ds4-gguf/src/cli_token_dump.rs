use crate::Ds4Tokenizer;
use std::io::{self, Write};

pub fn write_cli_token_dump<W: Write>(
    out: &mut W,
    tokenizer: &Ds4Tokenizer,
    prompt: &str,
) -> io::Result<()> {
    let tokens = tokenizer.tokenize_rendered_chat(prompt);
    write!(out, "[")?;
    for (idx, token) in tokens.iter().enumerate() {
        if idx != 0 {
            write!(out, ", ")?;
        }
        write!(out, "{token}")?;
    }
    writeln!(out, "]")?;
    for token in tokens {
        write!(out, "{token:6}  ")?;
        out.write_all(tokenizer.token_text_bytes(token))?;
        writeln!(out)?;
    }
    Ok(())
}
