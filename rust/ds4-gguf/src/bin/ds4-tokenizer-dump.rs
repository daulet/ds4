use ds4_gguf::{
    apply_cli_ops, parse_gguf, render_chat_prompt_text, ChatMessage, CliOp, Ds4Tokenizer,
    ThinkMode, ToolArgument, ToolCall,
};
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
    writeln!(out, "\n  ],")?;
    writeln!(out, "  \"server_request_cases\": [")?;
    let server_cases = server_prompt_cases();
    for (idx, case) in server_cases.iter().enumerate() {
        if idx != 0 {
            writeln!(out, ",")?;
        }
        write_server_case(out, tokenizer, case)?;
    }
    writeln!(out, "\n  ],")?;
    writeln!(out, "  \"cli_chat_cases\": [")?;
    let cli_cases = cli_chat_cases();
    for (idx, case) in cli_cases.iter().enumerate() {
        if idx != 0 {
            writeln!(out, ",")?;
        }
        write_cli_case(out, tokenizer, case)?;
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

struct ServerPromptCase {
    name: &'static str,
    messages: Vec<ChatMessage>,
    tool_schemas: Option<&'static str>,
    think_mode: ThinkMode,
}

struct CliCase {
    name: &'static str,
    ops: Vec<CliOp>,
}

fn write_server_case<W: Write>(
    out: &mut W,
    tokenizer: &Ds4Tokenizer,
    case: &ServerPromptCase,
) -> io::Result<()> {
    let prompt = render_chat_prompt_text(&case.messages, case.tool_schemas, case.think_mode);
    let tokens = tokenizer.tokenize_rendered_chat(&prompt);
    write!(out, "    {{\"name\": ")?;
    write_json_string(out, case.name)?;
    write!(out, ", \"think_mode\": ")?;
    write_json_string(out, case.think_mode.name())?;
    write!(out, ", \"prompt_text\": ")?;
    write_json_string(out, &prompt)?;
    write!(out, ", \"token_count\": {}, \"tokens\": [", tokens.len())?;
    write_tokens(out, tokenizer, &tokens)?;
    write!(out, "]}}")?;
    Ok(())
}

fn write_cli_case<W: Write>(
    out: &mut W,
    tokenizer: &Ds4Tokenizer,
    case: &CliCase,
) -> io::Result<()> {
    let tokens = apply_cli_ops(tokenizer, &case.ops);
    write!(out, "    {{\"name\": ")?;
    write_json_string(out, case.name)?;
    write!(out, ", \"operations\": [")?;
    for (idx, op) in case.ops.iter().enumerate() {
        if idx != 0 {
            write!(out, ", ")?;
        }
        write_cli_op(out, op)?;
    }
    write!(out, "], \"token_count\": {}, \"tokens\": [", tokens.len())?;
    write_tokens(out, tokenizer, &tokens)?;
    write!(out, "]}}")?;
    Ok(())
}

fn write_cli_op<W: Write>(out: &mut W, op: &CliOp) -> io::Result<()> {
    match op {
        CliOp::Begin => write!(out, "{{\"op\": \"begin\"}}"),
        CliOp::MaxEffortPrefix => write!(out, "{{\"op\": \"max_effort_prefix\"}}"),
        CliOp::AppendMessage { role, content } => {
            write!(out, "{{\"op\": \"append_message\", \"role\": ")?;
            write_json_string(out, role)?;
            write!(out, ", \"content\": ")?;
            write_json_string(out, content)?;
            write!(out, "}}")
        }
        CliOp::AssistantPrefix { think_mode } => {
            write!(out, "{{\"op\": \"assistant_prefix\", \"think_mode\": ")?;
            write_json_string(out, think_mode.name())?;
            write!(out, "}}")
        }
    }
}

fn write_tokens<W: Write>(out: &mut W, tokenizer: &Ds4Tokenizer, tokens: &[u32]) -> io::Result<()> {
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
    Ok(())
}

fn server_prompt_cases() -> Vec<ServerPromptCase> {
    vec![
        ServerPromptCase {
            name: "m0.4/chat_basic",
            messages: vec![ChatMessage::new(
                "user",
                "Return exactly this text: baseline ready",
            )],
            tool_schemas: None,
            think_mode: ThinkMode::None,
        },
        ServerPromptCase {
            name: "m0.4/chat_stream",
            messages: vec![ChatMessage::new(
                "user",
                "Return exactly this text: stream baseline",
            )],
            tool_schemas: None,
            think_mode: ThinkMode::None,
        },
        ServerPromptCase {
            name: "m0.4/chat_tool_call",
            messages: vec![ChatMessage::new(
                "user",
                "List the files in the current directory. Use the provided tool; do not answer in prose.",
            )],
            tool_schemas: Some(LIST_FILES_TOOL_SCHEMA),
            think_mode: ThinkMode::None,
        },
        ServerPromptCase {
            name: "m0.4/chat_thinking_disabled",
            messages: vec![ChatMessage::new(
                "user",
                "What is two plus two? Answer with one digit.",
            )],
            tool_schemas: None,
            think_mode: ThinkMode::None,
        },
        ServerPromptCase {
            name: "m0.4/chat_cache_seed",
            messages: vec![
                ChatMessage::new(
                    "system",
                    "You answer with the shortest exact phrase requested by the user.",
                ),
                ChatMessage::new(
                    "user",
                    "Cache baseline prompt alpha beta gamma delta epsilon zeta eta theta iota kappa lambda. Return exactly: cache ready",
                ),
            ],
            tool_schemas: None,
            think_mode: ThinkMode::None,
        },
        ServerPromptCase {
            name: "m0.4/chat_cache_continuation",
            messages: vec![
                ChatMessage::new(
                    "system",
                    "You answer with the shortest exact phrase requested by the user.",
                ),
                ChatMessage::new(
                    "user",
                    "Cache baseline prompt alpha beta gamma delta epsilon zeta eta theta iota kappa lambda. Return exactly: cache ready",
                ),
                ChatMessage::new("assistant", "cache ready"),
                ChatMessage::new("user", "Return exactly: cache continued"),
            ],
            tool_schemas: None,
            think_mode: ThinkMode::None,
        },
        ServerPromptCase {
            name: "builtin_thinking_max_developer",
            messages: vec![
                ChatMessage::new("developer", "Use terse diagnostics."),
                ChatMessage::new("user", "Explain tokenizer boundaries."),
            ],
            tool_schemas: None,
            think_mode: ThinkMode::Max,
        },
        ServerPromptCase {
            name: "builtin_function_result",
            messages: vec![
                ChatMessage::new("user", "run tool"),
                ChatMessage::new("assistant", "").with_tool_calls(vec![ToolCall::new(
                    "lookup",
                    vec![ToolArgument::string("query", "ds4")],
                )]),
                ChatMessage::new("function", "result </tool_result> & raw"),
            ],
            tool_schemas: None,
            think_mode: ThinkMode::High,
        },
        ServerPromptCase {
            name: "builtin_empty_tools_arrays",
            messages: vec![ChatMessage::new("user", "no tools")],
            tool_schemas: None,
            think_mode: ThinkMode::High,
        },
    ]
}

fn cli_chat_cases() -> Vec<CliCase> {
    vec![
        CliCase {
            name: "cli_basic_high",
            ops: vec![
                CliOp::Begin,
                CliOp::AppendMessage {
                    role: "system".to_string(),
                    content: "You are terse.".to_string(),
                },
                CliOp::AppendMessage {
                    role: "user".to_string(),
                    content: "Hello".to_string(),
                },
                CliOp::AssistantPrefix {
                    think_mode: ThinkMode::High,
                },
            ],
        },
        CliCase {
            name: "cli_developer_max",
            ops: vec![
                CliOp::Begin,
                CliOp::MaxEffortPrefix,
                CliOp::AppendMessage {
                    role: "developer".to_string(),
                    content: "Prefer exact token IDs.".to_string(),
                },
                CliOp::AppendMessage {
                    role: "user".to_string(),
                    content: "Why do chunks matter?".to_string(),
                },
                CliOp::AssistantPrefix {
                    think_mode: ThinkMode::Max,
                },
            ],
        },
        CliCase {
            name: "cli_tool_function_none",
            ops: vec![
                CliOp::Begin,
                CliOp::AppendMessage {
                    role: "user".to_string(),
                    content: "Use the tool.".to_string(),
                },
                CliOp::AppendMessage {
                    role: "assistant".to_string(),
                    content: "done".to_string(),
                },
                CliOp::AppendMessage {
                    role: "tool".to_string(),
                    content: "tool output </tool_result>".to_string(),
                },
                CliOp::AppendMessage {
                    role: "function".to_string(),
                    content: "function output".to_string(),
                },
                CliOp::AssistantPrefix {
                    think_mode: ThinkMode::None,
                },
            ],
        },
    ]
}

const LIST_FILES_TOOL_SCHEMA: &str = r#"{
        "name": "list_files",
        "description": "List files in a directory.",
        "parameters": {
          "type": "object",
          "properties": {
            "path": {
              "type": "string",
              "description": "Directory path to list."
            }
          },
          "required": [
            "path"
          ]
        }
      }"#;

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
