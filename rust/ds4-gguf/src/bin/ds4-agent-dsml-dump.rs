use ds4_gguf::{agent_dsml_hex, AgentDsmlParser, AgentToolCall};
use std::io::{self, Write};

const START: &str = "<｜DSML｜tool_calls>";
const PARAM_START: &str = "<｜DSML｜parameter";

struct Case {
    name: &'static str,
    input: &'static str,
}

#[derive(Default)]
struct Splits {
    values: Vec<usize>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut out = io::BufWriter::new(io::stdout());
    write_dump(&mut out)?;
    Ok(())
}

fn write_dump<W: Write>(out: &mut W) -> io::Result<()> {
    writeln!(out, "{{")?;
    writeln!(out, "  \"schema\": \"ds4.agent_dsml_oracle.v1\",")?;
    writeln!(out, "  \"cases\": [")?;
    let cases = cases();
    for (idx, case) in cases.iter().enumerate() {
        if idx != 0 {
            writeln!(out, ",")?;
        }
        write!(out, "    {{\"name\": ")?;
        write_json_string(out, case.name)?;
        write!(out, ", \"input\": ")?;
        write_json_string(out, case.input)?;
        writeln!(out, ", \"schedules\": [")?;

        let (start, param) = standard_splits(case.input);
        write_schedule(out, "whole", case.input, &Splits::default(), false, true)?;
        write_schedule(out, "one_byte", case.input, &Splits::default(), true, false)?;
        if !start.values.is_empty() {
            write_schedule(out, "marker_prefix", case.input, &start, false, false)?;
        }
        if !param.values.is_empty() {
            write_schedule(out, "parameter_boundary", case.input, &param, false, false)?;
        }
        write!(out, "\n      ]}}")?;
    }
    writeln!(out, "\n  ]")?;
    writeln!(out, "}}")?;
    Ok(())
}

fn write_schedule<W: Write>(
    out: &mut W,
    name: &str,
    input: &str,
    splits: &Splits,
    one_byte: bool,
    first: bool,
) -> io::Result<()> {
    if !first {
        writeln!(out, ",")?;
    }
    write!(out, "        {{\"name\": ")?;
    write_json_string(out, name)?;
    write!(out, ", \"steps\": [")?;

    let bytes = input.as_bytes();
    let mut parser = AgentDsmlParser::default();
    let mut offset = 0usize;
    let mut chunk_index = 0usize;
    let mut first_step = true;
    if one_byte {
        for offset in 0..bytes.len() {
            parser.feed(&bytes[offset..offset + 1]);
            if !first_step {
                write!(out, ", ")?;
            }
            write_snapshot(out, &parser, Some((chunk_index, offset, 1)))?;
            chunk_index += 1;
            first_step = false;
        }
    } else {
        for &split in &splits.values {
            if split <= offset || split > bytes.len() {
                continue;
            }
            parser.feed(&bytes[offset..split]);
            if !first_step {
                write!(out, ", ")?;
            }
            write_snapshot(out, &parser, Some((chunk_index, offset, split - offset)))?;
            chunk_index += 1;
            first_step = false;
            offset = split;
        }
        if offset < bytes.len() || chunk_index == 0 {
            parser.feed(&bytes[offset..]);
            if !first_step {
                write!(out, ", ")?;
            }
            write_snapshot(
                out,
                &parser,
                Some((chunk_index, offset, bytes.len() - offset)),
            )?;
        }
    }
    write!(out, "], \"final\": ")?;
    write_snapshot(out, &parser, None)?;
    write!(out, "}}")?;
    Ok(())
}

fn write_snapshot<W: Write>(
    out: &mut W,
    parser: &AgentDsmlParser,
    chunk: Option<(usize, usize, usize)>,
) -> io::Result<()> {
    write!(out, "{{")?;
    if let Some((chunk_index, offset, len)) = chunk {
        write!(
            out,
            "\"chunk_index\": {chunk_index}, \"offset\": {offset}, \"len\": {len}, "
        )?;
    }
    write!(out, "\"state\": ")?;
    write_json_string(out, parser.state.name())?;
    write!(
        out,
        ", \"search_len\": {}, \"search_tail_hex\": ",
        parser.search_tail.len()
    )?;
    write_json_string(out, &agent_dsml_hex(&parser.search_tail))?;
    write!(out, ", \"raw_len\": {}, \"raw_hex\": ", parser.raw.len())?;
    write_json_string(out, &agent_dsml_hex(&parser.raw))?;
    write!(
        out,
        ", \"parse_pos\": {}, \"param_name\": ",
        parser.parse_pos
    )?;
    write_json_nullable(out, parser.param_name.as_deref())?;
    write!(out, ", \"param_is_string\": {}", parser.param_is_string)?;
    write!(
        out,
        ", \"param_value_start\": {}, \"current\": ",
        parser.param_value_start
    )?;
    write_call(out, &parser.current)?;
    write!(out, ", \"calls\": ")?;
    write_calls(out, &parser.calls)?;
    write!(out, ", \"error\": ")?;
    write_json_string(out, &parser.error)?;
    write!(out, "}}")?;
    Ok(())
}

fn write_calls<W: Write>(out: &mut W, calls: &[AgentToolCall]) -> io::Result<()> {
    write!(out, "[")?;
    for (idx, call) in calls.iter().enumerate() {
        if idx != 0 {
            write!(out, ", ")?;
        }
        write_call(out, call)?;
    }
    write!(out, "]")?;
    Ok(())
}

fn write_call<W: Write>(out: &mut W, call: &AgentToolCall) -> io::Result<()> {
    write!(out, "{{\"name\": ")?;
    write_json_nullable(out, call.name.as_deref())?;
    write!(out, ", \"args\": [")?;
    for (idx, arg) in call.args.iter().enumerate() {
        if idx != 0 {
            write!(out, ", ")?;
        }
        write!(out, "{{\"name\": ")?;
        write_json_string(out, &arg.name)?;
        write!(out, ", \"value\": ")?;
        write_json_string(out, &arg.value)?;
        write!(out, ", \"is_string\": {}", arg.is_string)?;
        write!(out, "}}")?;
    }
    write!(out, "]}}")?;
    Ok(())
}

fn standard_splits(input: &str) -> (Splits, Splits) {
    let mut start = Splits::default();
    let mut param = Splits::default();
    let input_len = input.len();
    if let Some(pos) = input.find(START) {
        start.add(pos + 1, input_len);
        start.add(pos + 8, input_len);
        start.add(pos + START.len() - 1, input_len);
        start.add(pos + START.len(), input_len);
    }
    if let Some(pos) = input.find(PARAM_START) {
        if let Some(tag_end) = input[pos..].find('>') {
            param.add(pos + tag_end + 1, input_len);
        }
        if let Some(close_rel) = input[pos..].find("</｜DSML｜parameter") {
            let close = pos + close_rel;
            param.add(close + 1, input_len);
            param.add(close + 8, input_len);
            if let Some(close_end) = input[close..].find('>') {
                param.add(close + close_end, input_len);
                param.add(close + close_end + 1, input_len);
            }
        }
    }
    (start, param)
}

impl Splits {
    fn add(&mut self, split: usize, input_len: usize) {
        if split == 0 || split >= input_len || self.values.contains(&split) {
            return;
        }
        self.values.push(split);
        self.values.sort_unstable();
    }
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "simple_tool",
            input: "prefix <｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"bash\">\n<｜DSML｜parameter name=\"command\" string=\"true\">pwd</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
        },
        Case {
            name: "multiple_invokes",
            input: "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"first\">\n<｜DSML｜parameter name=\"value\" string=\"true\">one</｜DSML｜parameter>\n</｜DSML｜invoke>\n<｜DSML｜invoke name=\"second\">\n<｜DSML｜parameter name=\"value\" string=\"false\">2</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
        },
        Case {
            name: "escaped_parameter_delimiter",
            input: "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"bash\">\n<｜DSML｜parameter name=\"command\" string=\"true\">echo &lt;/｜DSML｜parameter> keep</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
        },
        Case {
            name: "close_tag_variants",
            input: "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"bash\">\n<｜DSML｜parameter name=\"command\" string=\"true\">pwd</｜DSML｜parameter ｜ >\n</｜DSML｜invoke ｜ >\n</｜DSML｜tool_calls ｜ >",
        },
        Case {
            name: "malformed_missing_invoke_name",
            input: "<｜DSML｜tool_calls>\n<｜DSML｜invoke>\n</｜DSML｜tool_calls>",
        },
        Case {
            name: "malformed_missing_parameter_name",
            input: "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"bash\">\n<｜DSML｜parameter>value</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
        },
        Case {
            name: "unexpected_tag",
            input: "<｜DSML｜tool_calls>\n<｜DSML｜oops>\n</｜DSML｜tool_calls>",
        },
        Case {
            name: "truncated_tool_calls",
            input: "<｜DSML｜tool_calls>\n",
        },
        Case {
            name: "truncated_invoke",
            input: "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"bash\"",
        },
        Case {
            name: "truncated_parameter",
            input: "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"bash\">\n<｜DSML｜parameter name=\"command\" string=\"true\">unterminated",
        },
        Case {
            name: "short_marker_ignored",
            input: "<DSML｜tool_calls>\n<DSML｜invoke name=\"bash\"></DSML｜invoke>\n</DSML｜tool_calls>",
        },
        Case {
            name: "no_start_prose",
            input: "plain text with </｜DSML｜tool_calls> but no canonical opening marker",
        },
        Case {
            name: "think_wrapped_start",
            input: "<think>reason</think>\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"bash\">\n<｜DSML｜parameter name=\"command\" string=\"true\">pwd</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
        },
    ]
}

fn write_json_nullable<W: Write>(out: &mut W, value: Option<&str>) -> io::Result<()> {
    if let Some(value) = value {
        write_json_string(out, value)
    } else {
        write!(out, "null")
    }
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
