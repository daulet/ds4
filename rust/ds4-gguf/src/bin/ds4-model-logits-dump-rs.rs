use ds4_gguf::sampling::DS4_N_VOCAB;
use ds4_gguf::{parse_gguf, sample_argmax, top_logprobs, Ds4Tokenizer, TokenScore};
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;

const DEFAULT_TOP_K: usize = 20;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    let tokenizer_bytes = fs::read(&args.tokenizer)?;
    let gguf = parse_gguf(&tokenizer_bytes)?;
    let tokenizer = Ds4Tokenizer::from_gguf(&gguf)?;
    let logits_bytes = fs::read(&args.logits)?;
    let step_bytes = DS4_N_VOCAB
        .checked_mul(std::mem::size_of::<f32>())
        .expect("vocab byte length overflow");
    if logits_bytes.len() % step_bytes != 0 {
        return Err(format!(
            "{} byte logits blob is not a whole number of {step_bytes}-byte vocab slices",
            logits_bytes.len()
        )
        .into());
    }

    let mut out = io::BufWriter::new(io::stdout());
    write_dump(&mut out, &args, &tokenizer, &logits_bytes, step_bytes)?;
    Ok(())
}

#[derive(Debug)]
struct Args {
    logits: PathBuf,
    tokenizer: PathBuf,
    top_k: usize,
}

fn parse_args() -> Result<Args, Box<dyn std::error::Error>> {
    let mut args = env::args_os();
    let program = PathBuf::from(args.next().unwrap_or_default());
    let mut logits = None;
    let mut tokenizer = None;
    let mut top_k = DEFAULT_TOP_K;

    while let Some(arg) = args.next() {
        match arg.to_string_lossy().as_ref() {
            "--logits" => {
                logits = Some(PathBuf::from(
                    args.next().ok_or("--logits requires a path")?,
                ));
            }
            "--tokenizer" => {
                tokenizer = Some(PathBuf::from(
                    args.next().ok_or("--tokenizer requires a path")?,
                ));
            }
            "--top-k" => {
                let raw = args.next().ok_or("--top-k requires a value")?;
                top_k = raw
                    .to_str()
                    .ok_or("--top-k must be UTF-8")?
                    .parse::<usize>()?;
            }
            "--help" | "-h" => {
                print_usage(&program);
                std::process::exit(0);
            }
            other => {
                return Err(format!("unknown argument: {other}").into());
            }
        }
    }

    let Some(logits) = logits else {
        print_usage(&program);
        std::process::exit(2);
    };
    let Some(tokenizer) = tokenizer else {
        print_usage(&program);
        std::process::exit(2);
    };
    if top_k == 0 {
        return Err("--top-k must be greater than zero".into());
    }

    Ok(Args {
        logits,
        tokenizer,
        top_k,
    })
}

fn print_usage(program: &std::path::Path) {
    eprintln!(
        "usage: {} --logits LOGITS_F32LE --tokenizer TOKENIZER_GGUF [--top-k N]",
        program.display()
    );
}

fn write_dump<W: Write>(
    out: &mut W,
    args: &Args,
    tokenizer: &Ds4Tokenizer,
    logits_bytes: &[u8],
    step_bytes: usize,
) -> io::Result<()> {
    let identity = tokenizer.identity();
    let slice_count = logits_bytes.len() / step_bytes;

    writeln!(out, "{{")?;
    writeln!(out, "  \"schema\": \"ds4.rust_model_logits_slices.v1\",")?;
    writeln!(
        out,
        "  \"source\": \"rust-m6.5-fixed-logits-model-slices\","
    )?;
    write!(out, "  \"logits_path\": ")?;
    write_json_string(out, &args.logits.display().to_string())?;
    writeln!(out, ",")?;
    write!(out, "  \"tokenizer_path\": ")?;
    write_json_string(out, &args.tokenizer.display().to_string())?;
    writeln!(out, ",")?;
    writeln!(out, "  \"n_vocab_full\": {DS4_N_VOCAB},")?;
    writeln!(out, "  \"top_k\": {},", args.top_k)?;
    writeln!(out, "  \"slice_bytes\": {step_bytes},")?;
    writeln!(out, "  \"slice_count\": {slice_count},")?;
    writeln!(out, "  \"tokenizer\": {{")?;
    writeln!(out, "    \"token_count\": {},", identity.token_count)?;
    write!(out, "    \"token_bytes_sha256\": ")?;
    write_json_string(out, &identity.token_bytes_sha256)?;
    writeln!(out, ",")?;
    writeln!(out, "    \"merge_count\": {},", identity.merge_count)?;
    write!(out, "    \"merge_pairs_sha256\": ")?;
    write_json_string(out, &identity.merge_pairs_sha256)?;
    writeln!(out)?;
    writeln!(out, "  }},")?;
    writeln!(out, "  \"slices\": [")?;

    let mut logits = vec![0.0f32; DS4_N_VOCAB];
    for (idx, chunk) in logits_bytes.chunks_exact(step_bytes).enumerate() {
        if idx != 0 {
            writeln!(out, ",")?;
        }
        decode_f32le_slice(chunk, &mut logits);
        write_slice(out, idx, step_bytes, &logits, tokenizer, args.top_k)?;
    }

    writeln!(out, "\n  ]")?;
    writeln!(out, "}}")
}

fn decode_f32le_slice(bytes: &[u8], out: &mut [f32]) {
    for (dst, raw) in out.iter_mut().zip(bytes.chunks_exact(4)) {
        *dst = f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
    }
}

fn write_slice<W: Write>(
    out: &mut W,
    idx: usize,
    step_bytes: usize,
    logits: &[f32],
    tokenizer: &Ds4Tokenizer,
    top_k: usize,
) -> io::Result<()> {
    let selected = sample_argmax(logits);
    let selected_bytes = token_bytes_hex(tokenizer, selected);
    let (returned, scores) = top_logprobs(logits, top_k);
    write!(
        out,
        "    {{\"index\": {idx}, \"logits_offset\": {}, \"logits_bytes\": {step_bytes}, \"selected_token\": {selected}, \"selected_bytes_hex\": ",
        idx * step_bytes
    )?;
    write_json_string(out, &selected_bytes)?;
    write!(out, ", \"top_k_returned\": {returned}, \"top_logprobs\": [")?;
    for (score_idx, score) in scores.iter().enumerate() {
        if score_idx != 0 {
            write!(out, ", ")?;
        }
        write_score(out, score, tokenizer)?;
    }
    write!(out, "]}}")
}

fn write_score<W: Write>(
    out: &mut W,
    score: &TokenScore,
    tokenizer: &Ds4Tokenizer,
) -> io::Result<()> {
    write!(out, "{{\"id\": {}, \"bytes_hex\": ", score.id)?;
    write_json_string(out, &token_bytes_hex(tokenizer, score.id))?;
    write!(out, ", \"logit\": ")?;
    write_json_f32(out, score.logit)?;
    write!(out, ", \"logprob\": ")?;
    write_json_f32(out, score.logprob)?;
    write!(out, "}}")
}

fn token_bytes_hex(tokenizer: &Ds4Tokenizer, token: i32) -> String {
    if token < 0 {
        return String::new();
    }
    bytes_to_hex(&tokenizer.token_bytes(token as u32))
}

fn bytes_to_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

fn write_json_f32<W: Write>(out: &mut W, value: f32) -> io::Result<()> {
    if value.is_nan() {
        write!(out, "\"nan\"")
    } else if value.is_infinite() {
        if value.is_sign_negative() {
            write!(out, "\"-inf\"")
        } else {
            write!(out, "\"inf\"")
        }
    } else {
        write!(out, "{value:.9e}")
    }
}

fn write_json_string<W: Write>(out: &mut W, value: &str) -> io::Result<()> {
    write!(out, "\"")?;
    for ch in value.chars() {
        match ch {
            '"' => write!(out, "\\\"")?,
            '\\' => write!(out, "\\\\")?,
            '\n' => write!(out, "\\n")?,
            '\r' => write!(out, "\\r")?,
            '\t' => write!(out, "\\t")?,
            ch if ch < ' ' => write!(out, "\\u{:04x}", ch as u32)?,
            ch => write!(out, "{ch}")?,
        }
    }
    write!(out, "\"")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_f32le_slices() {
        let bytes = [
            0x00, 0x00, 0x80, 0x3f, // 1.0
            0x00, 0x00, 0x20, 0xc0, // -2.5
        ];
        let mut out = [0.0f32; 2];
        decode_f32le_slice(&bytes, &mut out);
        assert_eq!(out, [1.0, -2.5]);
    }

    #[test]
    fn hex_encoding_is_lowercase_and_dense() {
        assert_eq!(bytes_to_hex(&[0, 10, 15, 16, 255]), "000a0f10ff");
    }
}
