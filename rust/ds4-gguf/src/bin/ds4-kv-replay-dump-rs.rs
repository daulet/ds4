use ds4_gguf::kv_policy::{cache_replay_decision, key_kind, reason_code};
use std::io::{self, Write};

struct ReplayCase {
    name: &'static str,
    prompt_tokens: u32,
    live_tokens_before: u32,
    live_prompt_common: u32,
    disk_cached_tokens: u32,
    disk_cache_file: Option<&'static str>,
    disk_reason_name: Option<&'static str>,
    disk_ext_flags: Option<u8>,
    rendered_text_sha256: Option<&'static str>,
    rendered_text_bytes: Option<u32>,
    rendered_prompt_sha256: &'static str,
    rendered_prompt_bytes: u32,
    effective_suffix_hex: Option<&'static str>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut out = io::BufWriter::new(io::stdout());
    write_dump(&mut out)?;
    Ok(())
}

fn write_dump<W: Write>(out: &mut W) -> io::Result<()> {
    writeln!(out, "{{")?;
    writeln!(out, "  \"schema\": \"ds4.rust_kv_replay_oracle.v1\",")?;
    writeln!(out, "  \"source\": \"rust-kv-replay-no-model\",")?;
    writeln!(out, "  \"replay_cases\": [")?;
    let cases = replay_cases();
    for (idx, case) in cases.iter().enumerate() {
        if idx != 0 {
            writeln!(out, ",")?;
        }
        write_replay_case(out, case)?;
    }
    writeln!(out, "\n  ],")?;
    writeln!(out, "  \"dsml_tool_call_cases\": [")?;
    write!(
        out,
        "    {{\"name\":\"m0_4_tool_call\",\"dsml_start\":1,\"dsml_end\":1,"
    )?;
    write!(out, "\"tool_call_count\":1,\"tool_call_id\":")?;
    write_json_string(out, "call_74afa558e9694448bc8aef7aae54150d")?;
    write!(out, ",\"tool_call_name\":\"list_files\",")?;
    write!(
        out,
        "\"tool_call_arguments_sha256\":\"4ae486c3a48f8dc732af672b138b438a1d96960304cc334d46bbc2687d169cbb\","
    )?;
    writeln!(out, "\"tool_call_arguments_bytes\":12}}")?;
    writeln!(out, "  ]")?;
    writeln!(out, "}}")
}

fn replay_cases() -> [ReplayCase; 6] {
    [
        ReplayCase {
            name: "m0_5_seed_miss",
            prompt_tokens: 550,
            live_tokens_before: 0,
            live_prompt_common: 0,
            disk_cached_tokens: 0,
            disk_cache_file: None,
            disk_reason_name: None,
            disk_ext_flags: None,
            rendered_text_sha256: None,
            rendered_text_bytes: None,
            rendered_prompt_sha256: "73adfe87d8950dde4b5a33d45b2fadb717c0370421ca05b3c4257360a8bbd1f1",
            rendered_prompt_bytes: 2520,
            effective_suffix_hex: None,
        },
        ReplayCase {
            name: "m0_5_seed_restore",
            prompt_tokens: 550,
            live_tokens_before: 0,
            live_prompt_common: 0,
            disk_cached_tokens: 550,
            disk_cache_file: Some("0ab2314538b11686a11e296b7f697651fbd17e60.kv"),
            disk_reason_name: Some("cold"),
            disk_ext_flags: Some(0),
            rendered_text_sha256: Some("73adfe87d8950dde4b5a33d45b2fadb717c0370421ca05b3c4257360a8bbd1f1"),
            rendered_text_bytes: Some(2520),
            rendered_prompt_sha256: "73adfe87d8950dde4b5a33d45b2fadb717c0370421ca05b3c4257360a8bbd1f1",
            rendered_prompt_bytes: 2520,
            effective_suffix_hex: Some(""),
        },
        ReplayCase {
            name: "m0_5_continuation_restore",
            prompt_tokens: 561,
            live_tokens_before: 0,
            live_prompt_common: 0,
            disk_cached_tokens: 552,
            disk_cache_file: Some("a0cac6ff193696ccb5d7e9ae151d7255d39cf161.kv"),
            disk_reason_name: Some("shutdown"),
            disk_ext_flags: Some(0),
            rendered_text_sha256: Some("f179cc9e845ef94838bc456b24567c32aaa5d60b0e28f7d016fd1db598262a22"),
            rendered_text_bytes: Some(2528),
            rendered_prompt_sha256: "72fefe4df49b22c4fd53ecfab918152b988bf39e231e1ccdd55ab1b4848a8e06",
            rendered_prompt_bytes: 2620,
            effective_suffix_hex: Some(
                "3cefbd9c656e64e296816f66e2968173656e74656e6365efbd9c3e3cefbd9c55736572efbd9c3e52657475726e2065786163746c793a206b7620636f6e74696e7565643cefbd9c417373697374616e74efbd9c3e3c2f7468696e6b3e",
            ),
        },
        ReplayCase {
            name: "m0_4_tool_call",
            prompt_tokens: 394,
            live_tokens_before: 13,
            live_prompt_common: 1,
            disk_cached_tokens: 0,
            disk_cache_file: None,
            disk_reason_name: None,
            disk_ext_flags: None,
            rendered_text_sha256: None,
            rendered_text_bytes: None,
            rendered_prompt_sha256: "8050b1e4f35f8d3f2779f14e905cca9cdf4bb47bb970e606a4d21aaa7b9a2c39",
            rendered_prompt_bytes: 1838,
            effective_suffix_hex: None,
        },
        ReplayCase {
            name: "m0_4_cache_seed",
            prompt_tokens: 39,
            live_tokens_before: 16,
            live_prompt_common: 1,
            disk_cached_tokens: 0,
            disk_cache_file: None,
            disk_reason_name: None,
            disk_ext_flags: None,
            rendered_text_sha256: None,
            rendered_text_bytes: None,
            rendered_prompt_sha256: "5f823dcf82c86cf44088e500260f0a289b1557d98ed639c81192b67f37a35f9f",
            rendered_prompt_bytes: 244,
            effective_suffix_hex: None,
        },
        ReplayCase {
            name: "m0_4_cache_continuation",
            prompt_tokens: 50,
            live_tokens_before: 41,
            live_prompt_common: 41,
            disk_cached_tokens: 0,
            disk_cache_file: None,
            disk_reason_name: None,
            disk_ext_flags: None,
            rendered_text_sha256: None,
            rendered_text_bytes: None,
            rendered_prompt_sha256: "98d37f51a0aaa4f43ceeb61a2b054bf5553817bfd09ba9155065f1896be722af",
            rendered_prompt_bytes: 350,
            effective_suffix_hex: Some(
                "63616368652072656164793cefbd9c656e64e296816f66e2968173656e74656e6365efbd9c3e3cefbd9c55736572efbd9c3e52657475726e2065786163746c793a20636163686520636f6e74696e7565643cefbd9c417373697374616e74efbd9c3e3c2f7468696e6b3e",
            ),
        },
    ]
}

fn write_replay_case<W: Write>(out: &mut W, case: &ReplayCase) -> io::Result<()> {
    let decision = cache_replay_decision(
        case.live_tokens_before,
        case.prompt_tokens,
        case.live_prompt_common,
        case.disk_cached_tokens,
    );
    write!(out, "    {{\"name\":")?;
    write_json_string(out, case.name)?;
    write!(
        out,
        ",\"prompt_tokens\":{},\"live_tokens_before\":{},\"live_prompt_common\":{},\
         \"memory_token_reusable\":{},\"memory_miss_reason\":",
        case.prompt_tokens,
        case.live_tokens_before,
        case.live_prompt_common,
        decision.memory_token_reusable,
    )?;
    write_json_string(out, decision.memory_miss_reason)?;
    write!(out, ",\"cache_source\":")?;
    write_json_string(out, decision.cache_source)?;
    write!(
        out,
        ",\"cached_tokens\":{},\"disk_cached_tokens\":{},\"cache_write_tokens\":{}",
        decision.cached_tokens, decision.disk_cached_tokens, decision.cache_write_tokens
    )?;
    write!(out, ",\"disk_cache_file\":")?;
    write_opt_str(out, case.disk_cache_file)?;
    write!(out, ",\"disk_cache_reason_name\":")?;
    write_opt_str(out, case.disk_reason_name)?;
    write!(out, ",\"disk_cache_reason_code\":")?;
    match case.disk_reason_name {
        Some(reason) => write!(out, "{}", reason_code(Some(reason)))?,
        None => write!(out, "null")?,
    }
    write!(out, ",\"disk_cache_ext_flags\":")?;
    match case.disk_ext_flags {
        Some(flags) => write!(out, "{flags}")?,
        None => write!(out, "null")?,
    }
    write!(out, ",\"disk_cache_key_kind\":")?;
    match case.disk_ext_flags {
        Some(flags) => write_json_string(out, key_kind(flags))?,
        None => write!(out, "null")?,
    }
    write!(out, ",\"rendered_text_sha256\":")?;
    write_opt_str(out, case.rendered_text_sha256)?;
    write!(out, ",\"rendered_text_bytes\":")?;
    match case.rendered_text_bytes {
        Some(bytes) => write!(out, "{bytes}")?,
        None => write!(out, "null")?,
    }
    write!(
        out,
        ",\"rendered_prompt_sha256\":\"{}\",\"rendered_prompt_bytes\":{}",
        case.rendered_prompt_sha256, case.rendered_prompt_bytes
    )?;
    write!(out, ",\"effective_suffix_hex\":")?;
    write_opt_str(out, case.effective_suffix_hex)?;
    write!(out, ",\"effective_suffix_bytes\":")?;
    match case.effective_suffix_hex {
        Some(hex) => write!(out, "{}", hex.len() / 2)?,
        None => write!(out, "null")?,
    }
    write!(out, "}}")
}

fn write_opt_str<W: Write>(out: &mut W, value: Option<&str>) -> io::Result<()> {
    match value {
        Some(value) => write_json_string(out, value),
        None => write!(out, "null"),
    }
}

fn write_json_string<W: Write>(out: &mut W, value: &str) -> io::Result<()> {
    write!(out, "\"")?;
    for byte in value.bytes() {
        match byte {
            b'"' => write!(out, "\\\"")?,
            b'\\' => write!(out, "\\\\")?,
            b'\n' => write!(out, "\\n")?,
            b'\r' => write!(out, "\\r")?,
            b'\t' => write!(out, "\\t")?,
            0x20..=0x7e => write!(out, "{}", byte as char)?,
            _ => write!(out, "\\u{byte:04x}")?,
        }
    }
    write!(out, "\"")
}
