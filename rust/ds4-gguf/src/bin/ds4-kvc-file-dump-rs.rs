use ds4_gguf::kv_policy::{
    file_size_fits, hex_bytes, read_kvc_file, sha1_bytes_hex, write_kvc_file, FileSizeDecision,
    KvHeader, KvcFile, EXT_RESPONSES_VISIBLE, EXT_THINKING_VISIBLE, EXT_TOOL_MAP, FIXED_HEADER,
    REASON_AGENT_SESSION, REASON_COLD, REASON_CONTINUED, REASON_SHUTDOWN,
};
use std::io::{self, Write};

struct FixtureCase<'a> {
    name: &'a str,
    header: KvHeader,
    text: &'a [u8],
    payload: &'a [u8],
    trailer: &'a [u8],
    budget_bytes: u64,
}

struct ReadProbe {
    parsed: Option<KvcFile>,
    expected_total_ok: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut out = io::BufWriter::new(io::stdout());
    write_dump(&mut out)?;
    Ok(())
}

fn write_dump<W: Write>(out: &mut W) -> io::Result<()> {
    writeln!(out, "{{")?;
    writeln!(out, "  \"schema\": \"ds4.rust_kvc_file_oracle.v1\",")?;
    writeln!(out, "  \"source\": \"rust-kvstore-file-no-model\",")?;
    writeln!(out, "  \"model\": \"no model is loaded for this oracle\",")?;
    writeln!(
        out,
        "  \"constants\": {{\"fixed_header\":{},\"ext_flags\":{{\"tool_map\":{},\
         \"responses_visible\":{},\"thinking_visible\":{}}}}},",
        FIXED_HEADER, EXT_TOOL_MAP, EXT_RESPONSES_VISIBLE, EXT_THINKING_VISIBLE
    )?;
    write_cases(out)?;
    write_malformed_cases(out)?;
    writeln!(out, "}}")
}

fn fixture_cases<'a>() -> Vec<FixtureCase<'a>> {
    vec![
        FixtureCase {
            name: "plain_no_trailer",
            header: KvHeader {
                quant_bits: 2,
                reason: REASON_COLD,
                ext_flags: 0,
                tokens: 600,
                hits: 0,
                ctx_size: 32768,
                created_at: 1_700_000_000,
                last_used: 1_700_000_000,
                payload_bytes: 8,
            },
            text: b"Alpha beta gamma",
            payload: &[0x00, 0x01, 0x02, 0x03, 0xff, 0x10, 0x20, 0x30],
            trailer: &[],
            budget_bytes: 0,
        },
        FixtureCase {
            name: "opaque_tool_trailer",
            header: KvHeader {
                quant_bits: 4,
                reason: REASON_CONTINUED,
                ext_flags: EXT_TOOL_MAP,
                tokens: 2048,
                hits: 7,
                ctx_size: 65536,
                created_at: 1_700_000_100,
                last_used: 1_700_000_200,
                payload_bytes: 9,
            },
            text: b"<tool-visible-prefix>",
            payload: &[0xaa, 0xbb, 0xcc, 0xdd, 0x11, 0x22, 0x33, 0x44, 0x55],
            trailer: &[b'O', b'P', b'Q', 0x00, 0x01, 0x02],
            budget_bytes: 92,
        },
        FixtureCase {
            name: "visible_flag_no_payload",
            header: KvHeader {
                quant_bits: 2,
                reason: REASON_AGENT_SESSION,
                ext_flags: EXT_RESPONSES_VISIBLE,
                tokens: 700,
                hits: 2,
                ctx_size: 32768,
                created_at: 1_700_000_300,
                last_used: 1_700_000_400,
                payload_bytes: 0,
            },
            text: b"visible transcript",
            payload: &[],
            trailer: &[],
            budget_bytes: 70,
        },
        FixtureCase {
            name: "empty_text_thinking_trailer",
            header: KvHeader {
                quant_bits: 2,
                reason: REASON_SHUTDOWN,
                ext_flags: EXT_THINKING_VISIBLE,
                tokens: 1,
                hits: 0,
                ctx_size: 4096,
                created_at: 1_700_000_500,
                last_used: 1_700_000_501,
                payload_bytes: 2,
            },
            text: b"",
            payload: &[0x42, 0x24],
            trailer: b"xyz",
            budget_bytes: 64,
        },
    ]
}

fn write_cases<W: Write>(out: &mut W) -> io::Result<()> {
    writeln!(out, "  \"cases\": [")?;
    for (idx, case) in fixture_cases().iter().enumerate() {
        if idx != 0 {
            writeln!(out, ",")?;
        }
        write_case(out, case)?;
    }
    writeln!(out, "\n  ],")
}

fn write_case<W: Write>(out: &mut W, case: &FixtureCase<'_>) -> io::Result<()> {
    let bytes = write_kvc_file(&case.header, case.text, case.payload, case.trailer)
        .expect("fixture KVC write");
    let text_len = (case.text.len() as u32).to_le_bytes();
    let probe = probe_bytes(&bytes, bytes.len() as u64);
    write!(out, "    {{\"name\":")?;
    write_json_string(out, case.name)?;
    write!(out, ",\"input\":")?;
    write_input(out, case)?;
    write!(out, ",\"sha1\":")?;
    write_json_string(out, &sha1_bytes_hex(case.text))?;
    write!(out, ",\"header_hex\":")?;
    write_hex_string(out, &case.header.to_bytes())?;
    write!(out, ",\"text_len_hex\":")?;
    write_hex_string(out, &text_len)?;
    write!(out, ",\"file_hex\":")?;
    write_hex_string(out, &bytes)?;
    write!(
        out,
        ",\"file_size\":{},\"expected_trailer_bytes\":{},\"budget\":",
        bytes.len(),
        case.trailer.len()
    )?;
    write_budget(out, case)?;
    write!(out, ",\"read_entry\":")?;
    write_read_probe(out, &probe)?;
    write!(out, "}}")
}

fn write_input<W: Write>(out: &mut W, case: &FixtureCase<'_>) -> io::Result<()> {
    write!(
        out,
        "{{\"quant_bits\":{},\"reason\":{},\"ext_flags\":{},\"tokens\":{},\
         \"hits\":{},\"ctx_size\":{},\"created_at\":{},\"last_used\":{},",
        case.header.quant_bits,
        case.header.reason,
        case.header.ext_flags,
        case.header.tokens,
        case.header.hits,
        case.header.ctx_size,
        case.header.created_at,
        case.header.last_used
    )?;
    write!(out, "\"text_hex\":")?;
    write_hex_string(out, case.text)?;
    write!(out, ",\"payload_hex\":")?;
    write_hex_string(out, case.payload)?;
    write!(out, ",\"trailer_hex\":")?;
    write_hex_string(out, case.trailer)?;
    write!(out, "}}")
}

fn write_budget<W: Write>(out: &mut W, case: &FixtureCase<'_>) -> io::Result<()> {
    let decision = file_size_fits(
        case.budget_bytes,
        case.text.len() as u64,
        case.payload.len() as u64,
        case.trailer.len() as u64,
    )
    .unwrap_or(FileSizeDecision {
        fits: false,
        file_bytes: 0,
        required_bytes: 0,
    });
    write!(
        out,
        "{{\"budget_bytes\":{},\"fits\":{},\"file_bytes\":{},\"required_bytes\":{}}}",
        case.budget_bytes, decision.fits, decision.file_bytes, decision.required_bytes
    )
}

fn probe_bytes(bytes: &[u8], expected_total: u64) -> ReadProbe {
    match read_kvc_file(bytes) {
        Ok(parsed) => ReadProbe {
            expected_total_ok: parsed.file_size == expected_total,
            parsed: Some(parsed),
        },
        Err(_) => ReadProbe {
            parsed: None,
            expected_total_ok: false,
        },
    }
}

fn write_read_probe<W: Write>(out: &mut W, probe: &ReadProbe) -> io::Result<()> {
    if let Some(parsed) = &probe.parsed {
        let trailer_bytes = parsed.trailer.len();
        write!(
            out,
            "{{\"ok\":true,\"quant_bits\":{},\"reason\":{},\"ext_flags\":{},\
             \"tokens\":{},\"hits\":{},\"ctx_size\":{},\"created_at\":{},\
             \"last_used\":{},\"payload_bytes\":{},\"text_bytes\":{},\
             \"file_size\":{},\"trailer_bytes\":{},\"expected_total_ok\":{}}}",
            parsed.header.quant_bits,
            parsed.header.reason,
            parsed.header.ext_flags,
            parsed.header.tokens,
            parsed.header.hits,
            parsed.header.ctx_size,
            parsed.header.created_at,
            parsed.header.last_used,
            parsed.header.payload_bytes,
            parsed.text.len(),
            parsed.file_size,
            trailer_bytes,
            probe.expected_total_ok
        )
    } else {
        write!(out, "{{\"ok\":false}}")
    }
}

fn write_malformed_cases<W: Write>(out: &mut W) -> io::Result<()> {
    let text = b"Alpha beta gamma";
    let payload = &[0x00, 0x01, 0x02, 0x03, 0xff, 0x10, 0x20, 0x30];
    let trailer = &[b'O', b'P', b'Q', 0x00, 0x01, 0x02];
    let header = KvHeader {
        quant_bits: 2,
        reason: REASON_COLD,
        ext_flags: 0,
        tokens: 600,
        hits: 0,
        ctx_size: 32768,
        created_at: 1_700_000_000,
        last_used: 1_700_000_000,
        payload_bytes: payload.len() as u64,
    };
    let base = write_kvc_file(&header, text, payload, trailer).expect("fixture KVC write");
    let cases: Vec<(&str, &str, Vec<u8>, u64)> = vec![
        (
            "truncated_header",
            "truncated-header",
            base[..20].to_vec(),
            base.len() as u64,
        ),
        (
            "bad_magic",
            "invalid-header",
            mutate_byte(&base, 0, b'X'),
            base.len() as u64,
        ),
        (
            "bad_version",
            "invalid-header",
            mutate_byte(&base, 3, 2),
            base.len() as u64,
        ),
        (
            "zero_tokens",
            "invalid-header",
            mutate_u32(&base, 8, 0),
            base.len() as u64,
        ),
        (
            "bad_quant",
            "invalid-header",
            mutate_byte(&base, 4, 3),
            base.len() as u64,
        ),
        (
            "declared_text_truncated",
            "truncated-text",
            mutate_u32(&base, FIXED_HEADER, text.len() as u32 + 100),
            base.len() as u64 + 100,
        ),
        (
            "declared_payload_truncated",
            "truncated-payload",
            mutate_u64(&base, 40, payload.len() as u64 + 99),
            base.len() as u64 + 99,
        ),
        (
            "declared_trailer_truncated",
            "truncated-trailer",
            base[..base.len() - 1].to_vec(),
            base.len() as u64,
        ),
    ];

    writeln!(out, "  \"malformed_cases\": [")?;
    for (idx, (name, expected_error, bytes, expected_total)) in cases.iter().enumerate() {
        if idx != 0 {
            writeln!(out, ",")?;
        }
        let probe = probe_bytes(bytes, *expected_total);
        write!(out, "    {{\"name\":")?;
        write_json_string(out, name)?;
        write!(out, ",\"expected_error\":")?;
        write_json_string(out, expected_error)?;
        write!(out, ",\"file_hex\":")?;
        write_hex_string(out, bytes)?;
        write!(out, ",\"read_entry\":")?;
        write_read_probe(out, &probe)?;
        write!(out, "}}")?;
    }
    writeln!(out, "\n  ]")?;
    Ok(())
}

fn mutate_byte(bytes: &[u8], offset: usize, value: u8) -> Vec<u8> {
    let mut out = bytes.to_vec();
    out[offset] = value;
    out
}

fn mutate_u32(bytes: &[u8], offset: usize, value: u32) -> Vec<u8> {
    let mut out = bytes.to_vec();
    out[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    out
}

fn mutate_u64(bytes: &[u8], offset: usize, value: u64) -> Vec<u8> {
    let mut out = bytes.to_vec();
    out[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    out
}

fn write_hex_string<W: Write>(out: &mut W, bytes: &[u8]) -> io::Result<()> {
    write!(out, "\"{}\"", hex_bytes(bytes))
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
