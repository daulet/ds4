use ds4_gguf::kv_policy::{
    byte_prefix_match, chat_anchor_pos, continued_store_target, entry_eviction_score,
    file_size_fits, find_text_prefix, hex_bytes, key_kind, le_get32, le_put32, path_for_sha,
    path_join, read_header, reason_code, sha1_bytes_hex, sha_hex_name, store_len, FileSizeDecision,
    KvEntry, KvHeader, KvOptions, KvPolicyConfig, DEFAULT_MB, EXT_RESPONSES_VISIBLE,
    EXT_THINKING_VISIBLE, EXT_TOOL_MAP, FIXED_HEADER, HIT_HALF_LIFE_SECONDS, REASON_COLD,
    REASON_CONTINUED,
};
use std::io::{self, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut out = io::BufWriter::new(io::stdout());
    write_dump(&mut out)?;
    Ok(())
}

fn write_dump<W: Write>(out: &mut W) -> io::Result<()> {
    writeln!(out, "{{")?;
    writeln!(out, "  \"schema\": \"ds4.rust_kv_policy_oracle.v1\",")?;
    writeln!(out, "  \"source\": \"rust-kvstore-no-model\",")?;
    writeln!(out, "  \"model\": \"no model is loaded for this oracle\",")?;
    writeln!(
        out,
        "  \"constants\": {{\"fixed_header\":{},\"default_mb\":{},\
         \"hit_half_life_seconds\":{},\"ext_flags\":{{\"tool_map\":{},\
         \"responses_visible\":{},\"thinking_visible\":{}}}}},",
        FIXED_HEADER,
        DEFAULT_MB,
        HIT_HALF_LIFE_SECONDS,
        EXT_TOOL_MAP,
        EXT_RESPONSES_VISIBLE,
        EXT_THINKING_VISIBLE
    )?;
    write!(out, "  \"defaults\": ")?;
    write_options(out, KvOptions::default())?;
    writeln!(out, ",")?;
    write_reason_cases(out)?;
    write_key_kind_cases(out)?;
    write_little_endian_cases(out)?;
    write_sha_cases(out)?;
    write_name_cases(out)?;
    write_path_cases(out)?;
    write_header_cases(out)?;
    write_policy_cases(out)?;
    write_m05_fixture(out)?;
    writeln!(out, "}}")
}

fn write_options<W: Write>(out: &mut W, options: KvOptions) -> io::Result<()> {
    write!(
        out,
        "{{\"min_tokens\":{},\"cold_max_tokens\":{},\
         \"continued_interval_tokens\":{},\"boundary_trim_tokens\":{},\
         \"boundary_align_tokens\":{}}}",
        options.min_tokens,
        options.cold_max_tokens,
        options.continued_interval_tokens,
        options.boundary_trim_tokens,
        options.boundary_align_tokens
    )
}

fn write_reason_cases<W: Write>(out: &mut W) -> io::Result<()> {
    let names = [
        "cold",
        "continued",
        "evict",
        "shutdown",
        "agent-system",
        "agent-session",
        "other",
    ];
    writeln!(out, "  \"reason_codes\": [")?;
    write!(out, "    {{\"input\":null,\"code\":{}}}", reason_code(None))?;
    for name in names {
        write!(out, ",\n    {{\"input\":")?;
        write_json_string(out, name)?;
        write!(out, ",\"code\":{}}}", reason_code(Some(name)))?;
    }
    writeln!(out, "\n  ],")
}

fn write_key_kind_cases<W: Write>(out: &mut W) -> io::Result<()> {
    let flags = [
        0,
        EXT_TOOL_MAP,
        EXT_RESPONSES_VISIBLE,
        EXT_THINKING_VISIBLE,
        EXT_TOOL_MAP | EXT_RESPONSES_VISIBLE,
        EXT_RESPONSES_VISIBLE | EXT_THINKING_VISIBLE,
    ];
    writeln!(out, "  \"key_kind_cases\": [")?;
    for (idx, flags) in flags.into_iter().enumerate() {
        if idx != 0 {
            writeln!(out, ",")?;
        }
        write!(out, "    {{\"ext_flags\":{},\"key_kind\":", flags)?;
        write_json_string(out, key_kind(flags))?;
        write!(out, "}}")?;
    }
    writeln!(out, "\n  ],")
}

fn write_little_endian_cases<W: Write>(out: &mut W) -> io::Result<()> {
    let values = [0_u32, 1, 0x1234_5678, 0xffff_ffff];
    writeln!(out, "  \"little_endian_cases\": [")?;
    for (idx, value) in values.into_iter().enumerate() {
        let bytes = le_put32(value);
        if idx != 0 {
            writeln!(out, ",")?;
        }
        write!(out, "    {{\"value\":{},\"hex\":", value)?;
        write_hex_string(out, &bytes)?;
        write!(out, ",\"roundtrip\":{}}}", le_get32(bytes))?;
    }
    writeln!(out, "\n  ],")
}

fn write_sha_cases<W: Write>(out: &mut W) -> io::Result<()> {
    let texts = [
        "",
        "cache prefix",
        "Alpha beta gamma",
        "<|tool|>{\"id\":\"abc\"}",
    ];
    writeln!(out, "  \"sha_cases\": [")?;
    for (idx, text) in texts.iter().enumerate() {
        if idx != 0 {
            writeln!(out, ",")?;
        }
        write!(out, "    {{\"name\":\"case{}\",\"text_hex\":", idx)?;
        write_hex_string(out, text.as_bytes())?;
        write!(out, ",\"sha1\":")?;
        write_json_string(out, &sha1_bytes_hex(text.as_bytes()))?;
        write!(out, "}}")?;
    }
    writeln!(out, "\n  ],")
}

fn write_name_cases<W: Write>(out: &mut W) -> io::Result<()> {
    let inputs = [
        "ABCDEF0123456789ABCDEF0123456789ABCDEF01.kv",
        "abc.kv",
        "abcdef0123456789abcdef0123456789abcdef01.bin",
    ];
    writeln!(out, "  \"name_cases\": [")?;
    for (idx, input) in inputs.iter().enumerate() {
        if idx != 0 {
            writeln!(out, ",")?;
        }
        let sha = sha_hex_name(input);
        write!(out, "    {{\"input\":")?;
        write_json_string(out, input)?;
        write!(out, ",\"valid\":{}", sha.is_some())?;
        if let Some(sha) = sha {
            write!(out, ",\"sha\":")?;
            write_json_string(out, &sha)?;
        }
        write!(out, "}}")?;
    }
    writeln!(out, "\n  ],")
}

fn write_path_cases<W: Write>(out: &mut W) -> io::Result<()> {
    let sha = sha1_bytes_hex(b"cache prefix");
    let path = path_for_sha("cache", &sha);
    let basename = path.rsplit('/').next().unwrap_or(&path);
    writeln!(out, "  \"path_cases\": [")?;
    write!(
        out,
        "    {{\"dir\":\"cache\",\"name\":\"file.kv\",\"joined\":"
    )?;
    write_json_string(out, &path_join("cache", "file.kv"))?;
    write!(
        out,
        "}},\n    {{\"dir\":\"cache/\",\"name\":\"file.kv\",\"joined\":"
    )?;
    write_json_string(out, &path_join("cache/", "file.kv"))?;
    write!(out, "}},\n    {{\"dir\":\"cache\",\"sha\":")?;
    write_json_string(out, &sha)?;
    write!(out, ",\"basename\":")?;
    write_json_string(out, basename)?;
    writeln!(out, "}}\n  ],")
}

fn write_header_cases<W: Write>(out: &mut W) -> io::Result<()> {
    writeln!(out, "  \"header_cases\": [")?;
    write_header_case(
        out,
        "cold_token_text",
        KvHeader {
            quant_bits: 2,
            reason: REASON_COLD,
            ext_flags: 0,
            tokens: 550,
            hits: 1,
            ctx_size: 32768,
            created_at: 1779417499,
            last_used: 1779417514,
            payload_bytes: 31526948,
        },
        2520,
        true,
    )?;
    write_header_case(
        out,
        "continued_tool_map",
        KvHeader {
            quant_bits: 4,
            reason: REASON_CONTINUED,
            ext_flags: EXT_TOOL_MAP,
            tokens: 2048,
            hits: 3,
            ctx_size: 65536,
            created_at: 1000,
            last_used: 1200,
            payload_bytes: 4096,
        },
        64,
        false,
    )?;
    write_header_case(
        out,
        "unknown_reason_normalized",
        KvHeader {
            quant_bits: 2,
            reason: 99,
            ext_flags: EXT_RESPONSES_VISIBLE,
            tokens: 9,
            hits: 0,
            ctx_size: 4096,
            created_at: 1,
            last_used: 2,
            payload_bytes: 3,
        },
        4,
        false,
    )?;
    writeln!(out, "\n  ],")?;

    writeln!(out, "  \"invalid_header_cases\": [")?;
    write_invalid_header_case(out, "bad_magic", 0, b'X', 2, 16, true)?;
    write_invalid_header_case(out, "bad_version", 3, 2, 2, 16, false)?;
    write_invalid_header_case(out, "zero_tokens", 8, 0, 2, 0, false)?;
    write_invalid_header_case(out, "bad_quant", 4, 3, 3, 16, false)?;
    writeln!(out, "\n  ],")
}

fn write_header_case<W: Write>(
    out: &mut W,
    name: &str,
    header: KvHeader,
    text_bytes: u32,
    first: bool,
) -> io::Result<()> {
    if !first {
        writeln!(out, ",")?;
    }
    let header_bytes = header.to_bytes();
    let mut bytes = Vec::from(header_bytes);
    bytes.extend_from_slice(&text_bytes.to_le_bytes());
    let decoded = read_header(&bytes).expect("fixture header should parse");
    write!(out, "    {{\"name\":")?;
    write_json_string(out, name)?;
    write!(out, ",\"input\":")?;
    write_header_input(out, &header, text_bytes)?;
    write!(out, ",\"header_hex\":")?;
    write_hex_string(out, &header_bytes)?;
    write!(out, ",\"text_len_hex\":")?;
    write_hex_string(out, &text_bytes.to_le_bytes())?;
    write!(out, ",\"read_ok\":true,\"decoded\":")?;
    write_decoded_header(out, &decoded.header, decoded.text_bytes)?;
    write!(out, "}}")
}

fn write_invalid_header_case<W: Write>(
    out: &mut W,
    name: &str,
    byte_index: usize,
    byte_value: u8,
    quant_bits: u8,
    tokens: u32,
    first: bool,
) -> io::Result<()> {
    if !first {
        writeln!(out, ",")?;
    }
    let header = KvHeader {
        quant_bits,
        reason: REASON_COLD,
        ext_flags: 0,
        tokens,
        hits: 0,
        ctx_size: 4096,
        created_at: 100,
        last_used: 101,
        payload_bytes: 256,
    };
    let mut bytes = header.to_bytes();
    bytes[byte_index] = byte_value;
    let mut with_text_len = Vec::from(bytes);
    with_text_len.extend_from_slice(&12_u32.to_le_bytes());
    write!(out, "    {{\"name\":")?;
    write_json_string(out, name)?;
    write!(out, ",\"header_hex\":")?;
    write_hex_string(out, &bytes)?;
    write!(
        out,
        ",\"read_ok\":{}}}",
        read_header(&with_text_len).is_ok()
    )
}

fn write_header_input<W: Write>(out: &mut W, header: &KvHeader, text_bytes: u32) -> io::Result<()> {
    write!(
        out,
        "{{\"quant_bits\":{},\"reason\":{},\"ext_flags\":{},\"tokens\":{},\
         \"hits\":{},\"ctx_size\":{},\"created_at\":{},\"last_used\":{},\
         \"payload_bytes\":{},\"text_bytes\":{}}}",
        header.quant_bits,
        header.reason,
        header.ext_flags,
        header.tokens,
        header.hits,
        header.ctx_size,
        header.created_at,
        header.last_used,
        header.payload_bytes,
        text_bytes
    )
}

fn write_decoded_header<W: Write>(
    out: &mut W,
    header: &KvHeader,
    text_bytes: u32,
) -> io::Result<()> {
    write_header_input(out, header, text_bytes)
}

fn write_policy_cases<W: Write>(out: &mut W) -> io::Result<()> {
    writeln!(out, "  \"policy_cases\": {{")?;
    write_store_len_cases(out)?;
    write_chat_anchor_cases(out)?;
    write_continued_cases(out)?;
    write_file_size_cases(out)?;
    write_prefix_cases(out)?;
    write_eviction_cases(out)?;
    write_find_prefix_cases(out)?;
    writeln!(out, "  }},")
}

fn write_store_len_cases<W: Write>(out: &mut W) -> io::Result<()> {
    let def = KvOptions::default();
    let align_zero = KvOptions {
        boundary_align_tokens: 0,
        ..def
    };
    let cases = [
        ("below_min", def, 500),
        ("at_min_plus_trim", def, 544),
        ("aligned_after_trim", def, 4096),
        ("larger_aligned_after_trim", def, 5000),
        ("align_zero_uses_trimmed_stable", align_zero, 1000),
    ];
    writeln!(out, "    \"store_len\": [")?;
    for (idx, (name, options, tokens)) in cases.into_iter().enumerate() {
        if idx != 0 {
            writeln!(out, ",")?;
        }
        write!(out, "      {{\"name\":")?;
        write_json_string(out, name)?;
        write!(out, ",\"options\":")?;
        write_options(out, options)?;
        write!(
            out,
            ",\"tokens\":{},\"store_len\":{}}}",
            tokens,
            store_len(options, tokens)
        )?;
    }
    writeln!(out, "\n    ],")
}

fn write_chat_anchor_cases<W: Write>(out: &mut W) -> io::Result<()> {
    let def = KvOptions {
        min_tokens: 2,
        ..KvOptions::default()
    };
    let strict = KvOptions {
        min_tokens: 4,
        ..KvOptions::default()
    };
    let cases: [(&str, KvOptions, &[i32], i32, i32); 7] = [
        (
            "last_user_before_assistant",
            def,
            &[10, 1, 11, 1, 20, 2, 30],
            1,
            2,
        ),
        ("user_below_min", strict, &[1, 10, 2], 1, 2),
        ("missing_markers", def, &[10, 11, 12], 1, 2),
        ("assistant_first", def, &[2, 1, 10], 1, 2),
        (
            "multiple_users_before_assistant",
            def,
            &[1, 10, 1, 20, 1, 2],
            1,
            2,
        ),
        ("exact_min_boundary", def, &[10, 11, 1, 2], 1, 2),
        ("same_user_and_assistant_id", def, &[1, 10, 1], 1, 1),
    ];
    writeln!(out, "    \"chat_anchor\": [")?;
    for (idx, (name, options, tokens, user_id, assistant_id)) in cases.into_iter().enumerate() {
        if idx != 0 {
            writeln!(out, ",")?;
        }
        write!(out, "      {{\"name\":")?;
        write_json_string(out, name)?;
        write!(out, ",\"tokens\":[")?;
        for (token_idx, token) in tokens.iter().enumerate() {
            if token_idx != 0 {
                write!(out, ",")?;
            }
            write!(out, "{token}")?;
        }
        write!(
            out,
            "],\"min_tokens\":{},\"user_token_id\":{},\
             \"assistant_token_id\":{},\"anchor_pos\":{}}}",
            options.min_tokens,
            user_id,
            assistant_id,
            chat_anchor_pos(options, tokens, user_id, assistant_id)
        )?;
    }
    writeln!(out, "\n    ],")
}

fn write_continued_cases<W: Write>(out: &mut W) -> io::Result<()> {
    let def = KvPolicyConfig::default();
    let already = KvPolicyConfig {
        continued_last_store_tokens: 10240,
        ..def
    };
    let disabled = KvPolicyConfig {
        enabled: false,
        ..def
    };
    let align_zero = KvPolicyConfig {
        options: KvOptions {
            boundary_align_tokens: 0,
            ..def.options
        },
        ..def
    };
    let no_interval = KvPolicyConfig {
        options: KvOptions {
            continued_interval_tokens: 0,
            ..def.options
        },
        ..def
    };
    let cases = [
        ("below_min", def, 511),
        ("unaligned_interval", def, 10000),
        ("aligned_interval", def, 10240),
        ("already_stored", already, 10240),
        ("disabled_store", disabled, 10240),
        ("align_zero_interval", align_zero, 10000),
        ("no_interval", no_interval, 10000),
    ];
    writeln!(out, "    \"continued_store_target\": [")?;
    for (idx, (name, config, live_tokens)) in cases.into_iter().enumerate() {
        if idx != 0 {
            writeln!(out, ",")?;
        }
        write!(out, "      {{\"name\":")?;
        write_json_string(out, name)?;
        write!(
            out,
            ",\"enabled\":{},\"live_tokens\":{},\
             \"continued_last_store_tokens\":{},\"options\":",
            config.enabled, live_tokens, config.continued_last_store_tokens
        )?;
        write_options(out, config.options)?;
        write!(
            out,
            ",\"target\":{}}}",
            continued_store_target(config, live_tokens)
        )?;
    }
    writeln!(out, "\n    ],")
}

fn write_file_size_cases<W: Write>(out: &mut W) -> io::Result<()> {
    let cases = [
        ("no_budget", 0, 100, 200, 30),
        ("under_budget_with_slack", 1024, 100, 200, 30),
        ("over_budget_with_slack", 385, 100, 200, 30),
        ("overflow_text", 0, u64::MAX, 1, 1),
    ];
    writeln!(out, "    \"file_size_fits\": [")?;
    for (idx, (name, budget, text, payload, trailer)) in cases.into_iter().enumerate() {
        if idx != 0 {
            writeln!(out, ",")?;
        }
        let decision = file_size_fits(budget, text, payload, trailer).unwrap_or(FileSizeDecision {
            fits: false,
            file_bytes: 0,
            required_bytes: 0,
        });
        write!(out, "      {{\"name\":")?;
        write_json_string(out, name)?;
        write!(
            out,
            ",\"budget_bytes\":{},\"text_bytes\":{},\"payload_bytes\":{},\
             \"trailer_bytes\":{},\"fits\":{},\"file_bytes\":{},\
             \"required_bytes\":{}}}",
            budget,
            text,
            payload,
            trailer,
            decision.fits,
            decision.file_bytes,
            decision.required_bytes
        )?;
    }
    writeln!(out, "\n    ],")
}

fn write_prefix_cases<W: Write>(out: &mut W) -> io::Result<()> {
    let cases = [
        ("matching_prefix", b"abcdef".as_slice(), b"abc".as_slice()),
        ("empty_prefix", b"abcdef".as_slice(), b"".as_slice()),
        ("mismatch", b"abcdef".as_slice(), b"abd".as_slice()),
        (
            "prefix_longer_than_text",
            b"abc".as_slice(),
            b"abcd".as_slice(),
        ),
    ];
    writeln!(out, "    \"byte_prefix_match\": [")?;
    for (idx, (name, text, prefix)) in cases.into_iter().enumerate() {
        if idx != 0 {
            writeln!(out, ",")?;
        }
        write!(out, "      {{\"name\":")?;
        write_json_string(out, name)?;
        write!(out, ",\"text_hex\":")?;
        write_hex_string(out, text)?;
        write!(out, ",\"prefix_hex\":")?;
        write_hex_string(out, prefix)?;
        write!(out, ",\"matches\":{}}}", byte_prefix_match(text, prefix))?;
    }
    writeln!(out, "\n    ],")
}

fn write_eviction_cases<W: Write>(out: &mut W) -> io::Result<()> {
    let now = 1_000_000;
    let mut entries = vec![
        (
            "fresh_hits",
            eviction_entry(
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                1000,
                3,
                1000,
                0,
                now,
            ),
        ),
        (
            "one_half_life",
            eviction_entry(
                "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                1000,
                3,
                1000,
                0,
                now - HIT_HALF_LIFE_SECONDS,
            ),
        ),
        (
            "stale_hits_floor",
            eviction_entry(
                "cccccccccccccccccccccccccccccccccccccccc",
                1000,
                3,
                1000,
                0,
                now - 10 * HIT_HALF_LIFE_SECONDS,
            ),
        ),
        (
            "zero_timestamp",
            eviction_entry(
                "dddddddddddddddddddddddddddddddddddddddd",
                1000,
                3,
                1000,
                0,
                0,
            ),
        ),
        (
            "zero_file_size",
            eviction_entry(
                "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
                1000,
                3,
                0,
                0,
                now,
            ),
        ),
    ];
    writeln!(out, "    \"eviction_score\": [")?;
    for (idx, (name, entry)) in entries.iter().enumerate() {
        if idx != 0 {
            writeln!(out, ",")?;
        }
        write_eviction_case(out, name, entry, None, now)?;
    }
    let protected = entries.remove(0).1;
    writeln!(out, ",")?;
    write!(
        out,
        "      {{\"name\":\"protected_sha\",\"now\":{},\"sha\":",
        now
    )?;
    write_json_string(out, &protected.sha)?;
    write!(out, ",\"protected_sha\":")?;
    write_json_string(out, &protected.sha)?;
    write!(
        out,
        ",\"score\":{}}}",
        entry_eviction_score(&protected, Some(&protected.sha), now)
    )?;
    writeln!(out, "\n    ],")
}

fn eviction_entry(
    sha: &str,
    tokens: u32,
    hits: u32,
    file_size: u64,
    created_at: u64,
    last_used: u64,
) -> KvEntry {
    KvEntry {
        sha: sha.to_string(),
        quant_bits: 2,
        reason: REASON_COLD,
        ext_flags: 0,
        tokens,
        hits,
        ctx_size: 32768,
        created_at,
        last_used,
        payload_bytes: 0,
        text_bytes: 0,
        file_size,
    }
}

fn write_eviction_case<W: Write>(
    out: &mut W,
    name: &str,
    entry: &KvEntry,
    protected_sha: Option<&str>,
    now: u64,
) -> io::Result<()> {
    write!(out, "      {{\"name\":")?;
    write_json_string(out, name)?;
    write!(
        out,
        ",\"now\":{},\"tokens\":{},\"hits\":{},\"file_size\":{},\
         \"created_at\":{},\"last_used\":{},\"score\":{}}}",
        now,
        entry.tokens,
        entry.hits,
        entry.file_size,
        entry.created_at,
        entry.last_used,
        entry_eviction_score(entry, protected_sha, now)
    )
}

fn write_find_prefix_cases<W: Write>(out: &mut W) -> io::Result<()> {
    let entries = vec![
        fake_entry("Alpha beta", 2, REASON_COLD, 600, 32768, 1000),
        fake_entry("Alpha beta gamma", 2, REASON_CONTINUED, 700, 32768, 1001),
        fake_entry("Alpha beta gamma delta", 4, REASON_COLD, 800, 65536, 1002),
        fake_entry("Short", 2, REASON_COLD, 100, 32768, 1003),
    ];
    let def = KvPolicyConfig {
        reject_different_quant: true,
        ..KvPolicyConfig::default()
    };
    let allow_cross_quant = KvPolicyConfig {
        reject_different_quant: false,
        ..KvPolicyConfig::default()
    };
    let cases = [
        ("longest_prefix", "Alpha beta gamma suffix", 2, 32768, def),
        ("reject_quant", "Alpha beta gamma suffix", 4, 32768, def),
        (
            "allow_cross_quant_when_config_accepts",
            "Alpha beta gamma suffix",
            4,
            32768,
            allow_cross_quant,
        ),
        (
            "reject_ctx_too_small",
            "Alpha beta gamma suffix",
            2,
            1024,
            def,
        ),
        ("reject_below_min", "Short suffix", 2, 32768, def),
        (
            "longest_with_large_context",
            "Alpha beta gamma delta suffix",
            4,
            65536,
            def,
        ),
    ];
    writeln!(out, "    \"find_text_prefix\": [")?;
    for (idx, (name, prompt, quant_bits, ctx_size, config)) in cases.into_iter().enumerate() {
        if idx != 0 {
            writeln!(out, ",")?;
        }
        write!(out, "      {{\"name\":")?;
        write_json_string(out, name)?;
        write!(out, ",\"prompt_hex\":")?;
        write_hex_string(out, prompt.as_bytes())?;
        write!(
            out,
            ",\"quant_bits\":{},\"ctx_size\":{},\
             \"reject_different_quant\":{}",
            quant_bits, ctx_size, config.reject_different_quant
        )?;
        if let Some(selected) = find_text_prefix(&entries, prompt, config, quant_bits, ctx_size) {
            let entry = &entries[selected];
            write!(out, ",\"found\":true,\"selected_sha\":")?;
            write_json_string(out, &entry.sha)?;
            write!(
                out,
                ",\"selected_tokens\":{},\"selected_text_bytes\":{},\
                 \"selected_quant_bits\":{},\"selected_ctx_size\":{}",
                entry.tokens, entry.text_bytes, entry.quant_bits, entry.ctx_size
            )?;
        } else {
            write!(out, ",\"found\":false")?;
        }
        write!(out, "}}")?;
    }
    writeln!(out, "\n    ]")
}

fn fake_entry(
    text: &str,
    quant_bits: u8,
    reason: u8,
    tokens: u32,
    ctx_size: u32,
    timestamp: u64,
) -> KvEntry {
    let payload_bytes = 8;
    KvEntry {
        sha: sha1_bytes_hex(text.as_bytes()),
        quant_bits,
        reason,
        ext_flags: 0,
        tokens,
        hits: 0,
        ctx_size,
        created_at: timestamp,
        last_used: timestamp,
        payload_bytes,
        text_bytes: text.len() as u64,
        file_size: FIXED_HEADER as u64 + 4 + text.len() as u64 + payload_bytes,
    }
}

fn write_m05_fixture<W: Write>(out: &mut W) -> io::Result<()> {
    writeln!(out, "  \"m0_5_header_fixture\": {{")?;
    writeln!(
        out,
        "    \"path\":\"ds4-parity/baselines/kv-artifacts/m0.5/logs/kv-header.tsv\","
    )?;
    writeln!(out, "    \"expected_rows\": [")?;
    writeln!(
        out,
        "      {{\"file\":\"0ab2314538b11686a11e296b7f697651fbd17e60.kv\",\
         \"quant\":2,\"reason\":1,\"reason_name\":\"cold\",\"ext_flags\":0,\
         \"tokens\":550,\"hits\":1,\"ctx\":32768,\"payload_bytes\":31526948,\
         \"rendered_text_bytes\":2520,\"trailer_bytes\":0}},"
    )?;
    writeln!(
        out,
        "      {{\"file\":\"4f149e59b256cc9d4ae7d1c828954ed07e2f3dcf.kv\",\
         \"quant\":2,\"reason\":4,\"reason_name\":\"shutdown\",\"ext_flags\":0,\
         \"tokens\":563,\"hits\":0,\"ctx\":32768,\"payload_bytes\":31688280,\
         \"rendered_text_bytes\":2632,\"trailer_bytes\":0}},"
    )?;
    writeln!(
        out,
        "      {{\"file\":\"a0cac6ff193696ccb5d7e9ae151d7255d39cf161.kv\",\
         \"quant\":2,\"reason\":4,\"reason_name\":\"shutdown\",\"ext_flags\":0,\
         \"tokens\":552,\"hits\":1,\"ctx\":32768,\"payload_bytes\":31580716,\
         \"rendered_text_bytes\":2528,\"trailer_bytes\":0}}"
    )?;
    writeln!(out, "    ]")?;
    writeln!(out, "  }}")
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
