use ds4_gguf::kv_policy::{
    hex_bytes, key_kind, read_tool_map_trailer, write_tool_map_trailer, ToolMapDecode,
    ToolMapEntry, ToolMapError, EXT_RESPONSES_VISIBLE, EXT_THINKING_VISIBLE, EXT_TOOL_MAP,
    TOOL_MAP_DEFAULT_MAX_ENTRIES, TOOL_MAP_HEADER, TOOL_MAP_MAX_ID_LEN, TOOL_MAP_VERSION,
};
use std::io::{self, Write};

#[derive(Clone)]
struct TrailerCase {
    name: &'static str,
    text: Vec<u8>,
    entries: Vec<ToolMapEntry>,
    disabled: bool,
    wanted_ids: Vec<&'static str>,
}

const DSML_BASH: &[u8] = b"\n\n<tool_calls>\n<invoke name=\"bash\">\n<parameter name=\"command\" string=\"true\">pwd</parameter>\n</invoke>\n</tool_calls>";
const DSML_EDIT: &[u8] = b"\n\n<tool_calls>\n<invoke name=\"edit\">\n<parameter name=\"patch\" string=\"true\">alpha beta</parameter>\n</invoke>\n</tool_calls>";
const DSML_UTF8: &[u8] = b"\n\n<tool_calls>\n<invoke name=\"note\">\n<parameter name=\"text\" string=\"true\">caf\xc3\xa9 \\xe2\\x82\\xac</parameter>\n</invoke>\n</tool_calls>";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut out = io::BufWriter::new(io::stdout());
    write_dump(&mut out)?;
    Ok(())
}

fn write_dump<W: Write>(out: &mut W) -> io::Result<()> {
    writeln!(out, "{{")?;
    writeln!(out, "  \"schema\": \"ds4.rust_kv_trailer_oracle.v1\",")?;
    writeln!(out, "  \"source\": \"rust-server-kv-trailer-no-model\",")?;
    writeln!(out, "  \"model\": \"no model is loaded for this oracle\",")?;
    writeln!(
        out,
        "  \"constants\": {{\"tool_map_magic_hex\":\"4b544d\",\
         \"tool_map_version\":{},\"tool_map_header\":{},\"max_id_len\":{},\
         \"ext_flags\":{{\"tool_map\":{},\"responses_visible\":{},\
         \"thinking_visible\":{}}}}},",
        TOOL_MAP_VERSION,
        TOOL_MAP_HEADER,
        TOOL_MAP_MAX_ID_LEN,
        EXT_TOOL_MAP,
        EXT_RESPONSES_VISIBLE,
        EXT_THINKING_VISIBLE
    )?;
    write_tool_map_cases(out)?;
    write_extension_flag_cases(out)?;
    write_malformed_cases(out)?;
    writeln!(out, "}}")
}

fn entry(id: &str, dsml: &[u8]) -> ToolMapEntry {
    ToolMapEntry {
        id: id.to_string(),
        dsml: dsml.to_vec(),
    }
}

fn tool_map_cases() -> Vec<TrailerCase> {
    let mut duplicate_text = Vec::new();
    duplicate_text.extend_from_slice(DSML_BASH);
    duplicate_text.extend_from_slice(b" suffix ");
    duplicate_text.extend_from_slice(DSML_BASH);
    vec![
        TrailerCase {
            name: "empty_text",
            text: Vec::new(),
            entries: vec![entry("call_unused", DSML_BASH)],
            disabled: false,
            wanted_ids: Vec::new(),
        },
        TrailerCase {
            name: "single_block",
            text: DSML_BASH.to_vec(),
            entries: vec![entry("call_keep", DSML_BASH)],
            disabled: false,
            wanted_ids: Vec::new(),
        },
        TrailerCase {
            name: "filters_by_text",
            text: DSML_BASH.to_vec(),
            entries: vec![entry("call_keep", DSML_BASH), entry("call_drop", DSML_EDIT)],
            disabled: false,
            wanted_ids: vec!["call_keep"],
        },
        TrailerCase {
            name: "duplicate_block_once",
            text: duplicate_text,
            entries: vec![entry("call_keep", DSML_BASH)],
            disabled: false,
            wanted_ids: Vec::new(),
        },
        TrailerCase {
            name: "multiple_ids_same_block",
            text: DSML_BASH.to_vec(),
            entries: vec![entry("call_a", DSML_BASH), entry("call_b", DSML_BASH)],
            disabled: false,
            wanted_ids: vec!["call_b"],
        },
        TrailerCase {
            name: "utf8_and_long_id",
            text: DSML_UTF8.to_vec(),
            entries: vec![entry(
                "call_utf8_long_identifier_012345678901234567890123456789",
                DSML_UTF8,
            )],
            disabled: false,
            wanted_ids: Vec::new(),
        },
        TrailerCase {
            name: "disabled_replay",
            text: DSML_BASH.to_vec(),
            entries: vec![entry("call_keep", DSML_BASH)],
            disabled: true,
            wanted_ids: Vec::new(),
        },
    ]
}

fn write_tool_map_cases<W: Write>(out: &mut W) -> io::Result<()> {
    writeln!(out, "  \"tool_map_cases\": [")?;
    for (idx, case) in tool_map_cases().iter().enumerate() {
        if idx != 0 {
            writeln!(out, ",")?;
        }
        write_tool_map_case(out, case)?;
    }
    writeln!(out, "\n  ],")
}

fn write_tool_map_case<W: Write>(out: &mut W, case: &TrailerCase) -> io::Result<()> {
    let trailer = write_tool_map_trailer(&case.text, &case.entries, case.disabled)
        .expect("fixture trailer write");
    let loaded_all = load_count(&trailer, &[]);
    let loaded_wanted = load_count(&trailer, &case.wanted_ids);
    write!(out, "    {{\"name\":")?;
    write_json_string(out, case.name)?;
    write!(out, ",\"disabled\":{},\"text_hex\":", case.disabled)?;
    write_hex_string(out, &case.text)?;
    write!(out, ",\"memory_entries\":")?;
    write_entries(out, &case.entries)?;
    write!(out, ",\"wanted_ids\":")?;
    write_wanted_ids(out, &case.wanted_ids)?;
    write!(
        out,
        ",\"serialized_size\":{},\"write_ok\":true,\"written_bytes\":{},\"trailer_hex\":",
        trailer.len(),
        trailer.len()
    )?;
    write_hex_string(out, &trailer)?;
    write!(
        out,
        ",\"trailer_bytes\":{},\"load_all_count\":{},\"load_wanted_count\":{},\"decoded\":",
        trailer.len(),
        loaded_all,
        loaded_wanted
    )?;
    write_decode_result(out, &trailer)?;
    write!(out, "}}")
}

fn write_entries<W: Write>(out: &mut W, entries: &[ToolMapEntry]) -> io::Result<()> {
    write!(out, "[")?;
    for (idx, entry) in entries.iter().enumerate() {
        if idx != 0 {
            write!(out, ",")?;
        }
        write!(out, "{{\"id\":")?;
        write_json_string(out, &entry.id)?;
        write!(out, ",\"dsml_hex\":")?;
        write_hex_string(out, &entry.dsml)?;
        write!(out, "}}")?;
    }
    write!(out, "]")
}

fn write_wanted_ids<W: Write>(out: &mut W, ids: &[&str]) -> io::Result<()> {
    write!(out, "[")?;
    for (idx, id) in ids.iter().enumerate() {
        if idx != 0 {
            write!(out, ",")?;
        }
        write_json_string(out, id)?;
    }
    write!(out, "]")
}

fn load_count(bytes: &[u8], wanted: &[&str]) -> usize {
    let decode = match read_tool_map_trailer(bytes, TOOL_MAP_DEFAULT_MAX_ENTRIES) {
        Ok(decode) => decode,
        Err((_, decode)) => decode,
    };
    decode
        .entries
        .iter()
        .filter(|entry| wanted.is_empty() || wanted.iter().any(|id| *id == entry.id))
        .count()
}

fn write_decode_result<W: Write>(out: &mut W, bytes: &[u8]) -> io::Result<()> {
    match read_tool_map_trailer(bytes, TOOL_MAP_DEFAULT_MAX_ENTRIES) {
        Ok(decode) => write_decode(out, true, None, &decode),
        Err((err, decode)) => write_decode(out, false, Some(err), &decode),
    }
}

fn write_decode<W: Write>(
    out: &mut W,
    ok: bool,
    err: Option<ToolMapError>,
    decode: &ToolMapDecode,
) -> io::Result<()> {
    write!(
        out,
        "{{\"ok\":{},\"count\":{},\"decoded_entries\":{}",
        ok,
        decode.declared_count,
        decode.entries.len()
    )?;
    if let Some(err) = err {
        write!(out, ",\"error\":")?;
        write_json_string(out, err.as_str())?;
    }
    write!(out, ",\"entries\":")?;
    write_entries(out, &decode.entries)?;
    write!(out, "}}")
}

fn write_extension_flag_cases<W: Write>(out: &mut W) -> io::Result<()> {
    let cases = [
        ("token_text", 0, 0_u64),
        ("tool_map_payload", EXT_TOOL_MAP, 16),
        ("responses_visible_no_payload", EXT_RESPONSES_VISIBLE, 0),
        ("thinking_visible_no_payload", EXT_THINKING_VISIBLE, 0),
        (
            "responses_priority",
            EXT_TOOL_MAP | EXT_RESPONSES_VISIBLE | EXT_THINKING_VISIBLE,
            0,
        ),
    ];
    writeln!(out, "  \"extension_flag_cases\": [")?;
    for (idx, (name, ext_flags, trailer_bytes)) in cases.into_iter().enumerate() {
        if idx != 0 {
            writeln!(out, ",")?;
        }
        write!(out, "    {{\"name\":")?;
        write_json_string(out, name)?;
        write!(
            out,
            ",\"ext_flags\":{},\"tool_map_present\":{},\"trailer_bytes\":{},\"key_kind\":",
            ext_flags,
            ext_flags & EXT_TOOL_MAP != 0,
            trailer_bytes
        )?;
        write_json_string(out, key_kind(ext_flags))?;
        write!(out, "}}")?;
    }
    writeln!(out, "\n  ],")
}

fn write_malformed_cases<W: Write>(out: &mut W) -> io::Result<()> {
    let mut count_limit = vec![b'K', b'T', b'M', 1, 0, 0, 0, 0];
    count_limit[4..8]
        .copy_from_slice(&((TOOL_MAP_DEFAULT_MAX_ENTRIES as u32) * 4 + 1).to_le_bytes());
    let mut partial_second = Vec::new();
    partial_second.extend_from_slice(b"KTM");
    partial_second.push(1);
    partial_second.extend_from_slice(&2_u32.to_le_bytes());
    append_entry(&mut partial_second, "call_a", DSML_BASH);
    partial_second.extend_from_slice(&4_u32.to_le_bytes());
    partial_second.extend_from_slice(&5_u32.to_le_bytes());
    partial_second.extend_from_slice(b"ca");
    let cases: Vec<(&str, &str, Vec<u8>)> = vec![
        ("short_header", "short-header", b"KT".to_vec()),
        (
            "bad_magic",
            "bad-header",
            vec![b'X', b'T', b'M', 1, 0, 0, 0, 0],
        ),
        (
            "bad_version",
            "bad-header",
            vec![b'K', b'T', b'M', 2, 0, 0, 0, 0],
        ),
        ("count_limit", "count-limit", count_limit),
        (
            "zero_id_len",
            "bad-id-len",
            vec![
                b'K', b'T', b'M', 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, b'x',
            ],
        ),
        (
            "id_len_too_long",
            "bad-id-len",
            vec![
                b'K', b'T', b'M', 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, b'x',
            ],
        ),
        (
            "zero_dsml_len",
            "bad-dsml-len",
            vec![
                b'K', b'T', b'M', 1, 1, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, b'c', b'a', b'l', b'l',
            ],
        ),
        (
            "truncated_id",
            "truncated-id",
            vec![
                b'K', b'T', b'M', 1, 1, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0, b'c', b'a',
            ],
        ),
        (
            "truncated_dsml",
            "truncated-dsml",
            vec![
                b'K', b'T', b'M', 1, 1, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0, b'c', b'a', b'l', b'l',
                b'x', b'y',
            ],
        ),
        ("partial_second_entry", "truncated-id", partial_second),
    ];
    writeln!(out, "  \"malformed_cases\": [")?;
    for (idx, (name, expected_error, bytes)) in cases.iter().enumerate() {
        if idx != 0 {
            writeln!(out, ",")?;
        }
        write!(out, "    {{\"name\":")?;
        write_json_string(out, name)?;
        write!(out, ",\"expected_error\":")?;
        write_json_string(out, expected_error)?;
        write!(out, ",\"trailer_hex\":")?;
        write_hex_string(out, bytes)?;
        write!(
            out,
            ",\"load_count\":{},\"decoded\":",
            load_count(bytes, &[])
        )?;
        write_decode_result(out, bytes)?;
        write!(out, "}}")?;
    }
    writeln!(out, "\n  ]")
}

fn append_entry(out: &mut Vec<u8>, id: &str, dsml: &[u8]) {
    out.extend_from_slice(&(id.len() as u32).to_le_bytes());
    out.extend_from_slice(&(dsml.len() as u32).to_le_bytes());
    out.extend_from_slice(id.as_bytes());
    out.extend_from_slice(dsml);
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
