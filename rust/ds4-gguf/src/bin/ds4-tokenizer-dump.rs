use ds4_gguf::{parse_gguf, Ds4Tokenizer};
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;

const TEXT_CASES: &[(&str, &str)] = &[
    ("ascii_basic", "Hello, world!"),
    ("numbers_and_spaces", "123456789 42 007"),
    (
        "code_newlines",
        "for (int i = 0; i < 10; i++) {\n  printf(\"%d\\n\", i);\n}\n",
    ),
    ("utf8_mixed", "Cafe\u{0301} 世界 カタカナ"),
    (
        "literal_special_looking_user_text",
        "<｜User｜> literal ｜DSML｜ marker </think>",
    ),
];

const RENDERED_CHAT_CASES: &[(&str, &str)] = &[
    (
        "rendered_specials",
        "<｜begin▁of▁sentence｜>System<｜User｜>Hello<｜Assistant｜><think>Reason</think>Answer｜DSML｜<｜end▁of▁sentence｜>",
    ),
    (
        "rendered_tool_result",
        "<｜begin▁of▁sentence｜><｜User｜><tool_result>a & b </tool_result><｜Assistant｜></think>",
    ),
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = parse_args()?;
    let bytes = fs::read(&path)?;
    let gguf = parse_gguf(&bytes)?;
    let tokenizer = match Ds4Tokenizer::from_gguf(&gguf) {
        Ok(tokenizer) => tokenizer,
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(1);
        }
    };
    let mut out = io::BufWriter::new(io::stdout());
    write_dump(&mut out, &tokenizer)?;
    Ok(())
}

fn parse_args() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let mut args = env::args_os();
    let program = args.next().unwrap_or_default();
    let path = match (args.next(), args.next()) {
        (Some(path), None) => PathBuf::from(path),
        _ => {
            eprintln!("usage: {} TOKENIZER_GGUF", PathBuf::from(program).display());
            std::process::exit(2);
        }
    };
    Ok(path)
}

fn write_dump<W: Write>(out: &mut W, tokenizer: &Ds4Tokenizer) -> io::Result<()> {
    let identity = tokenizer.identity();
    let special = tokenizer.special_token_ids();
    writeln!(out, "{{")?;
    writeln!(out, "  \"schema\": \"ds4.rust_tokenizer.v1\",")?;
    writeln!(out, "  \"tokenizer\": {{")?;
    writeln!(out, "    \"token_count\": {},", identity.token_count)?;
    write!(out, "    \"token_bytes_sha256\": ")?;
    write_json_string(out, &identity.token_bytes_sha256)?;
    writeln!(out, ",")?;
    writeln!(out, "    \"merge_count\": {},", identity.merge_count)?;
    write!(out, "    \"merge_pairs_sha256\": ")?;
    write_json_string(out, &identity.merge_pairs_sha256)?;
    writeln!(out, ",")?;
    writeln!(out, "    \"special_token_at\": [")?;
    write_special(out, "bos", "<｜begin▁of▁sentence｜>", special.bos, true)?;
    write_special(out, "eos", "<｜end▁of▁sentence｜>", special.eos, false)?;
    write_special(out, "user", "<｜User｜>", special.user, false)?;
    write_special(
        out,
        "assistant",
        "<｜Assistant｜>",
        special.assistant,
        false,
    )?;
    write_special(out, "think_start", "<think>", special.think_start, false)?;
    write_special(out, "think_end", "</think>", special.think_end, false)?;
    write_special(out, "dsml", "｜DSML｜", special.dsml, false)?;
    writeln!(out, "\n    ]")?;
    writeln!(out, "  }},")?;
    writeln!(out, "  \"text_cases\": [")?;
    for (idx, (name, input)) in TEXT_CASES.iter().enumerate() {
        if idx != 0 {
            writeln!(out, ",")?;
        }
        write_text_case(out, tokenizer, name, input, false)?;
    }
    writeln!(out, "\n  ],")?;
    writeln!(out, "  \"rendered_chat_cases\": [")?;
    for (idx, (name, input)) in RENDERED_CHAT_CASES.iter().enumerate() {
        if idx != 0 {
            writeln!(out, ",")?;
        }
        write_text_case(out, tokenizer, name, input, true)?;
    }
    writeln!(out, "\n  ]")?;
    writeln!(out, "}}")?;
    Ok(())
}

fn write_special<W: Write>(
    out: &mut W,
    name: &str,
    text: &str,
    id: u32,
    first: bool,
) -> io::Result<()> {
    if !first {
        writeln!(out, ",")?;
    }
    write!(out, "      {{\"name\": ")?;
    write_json_string(out, name)?;
    write!(out, ", \"text\": ")?;
    write_json_string(out, text)?;
    write!(out, ", \"id\": {id}}}")?;
    Ok(())
}

fn write_text_case<W: Write>(
    out: &mut W,
    tokenizer: &Ds4Tokenizer,
    name: &str,
    input: &str,
    rendered_chat: bool,
) -> io::Result<()> {
    let tokens = if rendered_chat {
        tokenizer.tokenize_rendered_chat(input)
    } else {
        tokenizer.tokenize_text(input)
    };
    let mode = if rendered_chat {
        "rendered_chat"
    } else {
        "plain_text"
    };
    write!(out, "    {{\"name\": ")?;
    write_json_string(out, name)?;
    write!(out, ", \"mode\": ")?;
    write_json_string(out, mode)?;
    write!(out, ", \"input\": ")?;
    write_json_string(out, input)?;
    write!(out, ", \"token_count\": {}, \"tokens\": [", tokens.len())?;
    for (idx, token) in tokens.iter().enumerate() {
        if idx != 0 {
            write!(out, ", ")?;
        }
        write!(out, "{{\"id\": {token}, \"bytes\": [")?;
        let bytes = tokenizer.token_bytes(*token);
        for (byte_idx, byte) in bytes.iter().enumerate() {
            if byte_idx != 0 {
                write!(out, ",")?;
            }
            write!(out, "{byte}")?;
        }
        write!(out, "]}}")?;
    }
    write!(out, "]}}")?;
    Ok(())
}

fn write_json_string<W: Write>(out: &mut W, value: &str) -> io::Result<()> {
    write!(out, "\"")?;
    for c in value.chars() {
        match c {
            '"' => write!(out, "\\\"")?,
            '\\' => write!(out, "\\\\")?,
            '\n' => write!(out, "\\n")?,
            '\r' => write!(out, "\\r")?,
            '\t' => write!(out, "\\t")?,
            c if c < ' ' => write!(out, "\\u{:04x}", c as u32)?,
            c => write!(out, "{c}")?,
        }
    }
    write!(out, "\"")?;
    Ok(())
}
