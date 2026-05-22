use ds4_gguf::{
    parse_generated_message, parse_generated_message_for_response,
    render_dsml_tool_calls_from_json, render_tool_result_text, DsmlJsonCall,
    ParsedGeneratedMessage,
};
use std::io::{self, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut out = io::BufWriter::new(io::stdout());
    write_dump(&mut out)?;
    Ok(())
}

fn write_dump<W: Write>(out: &mut W) -> io::Result<()> {
    writeln!(out, "{{")?;
    writeln!(out, "  \"schema\": \"ds4.dsml_oracle.v1\",")?;
    writeln!(out, "  \"format_cases\": [")?;
    let format_cases = format_cases();
    for (idx, case) in format_cases.iter().enumerate() {
        if idx != 0 {
            writeln!(out, ",")?;
        }
        write_format_case(out, case)?;
    }
    writeln!(out, "\n  ],")?;
    writeln!(out, "  \"parse_cases\": [")?;
    let parse_cases = parse_cases();
    for (idx, case) in parse_cases.iter().enumerate() {
        if idx != 0 {
            writeln!(out, ",")?;
        }
        write_parse_case(out, case)?;
    }
    writeln!(out, "\n  ]")?;
    writeln!(out, "}}")?;
    Ok(())
}

enum FormatCase {
    ToolCalls {
        name: &'static str,
        raw_dsml: Option<&'static str>,
        calls: Vec<DsmlJsonCall>,
    },
    ToolResult {
        name: &'static str,
        input: &'static str,
    },
}

struct ParseCase {
    name: &'static str,
    input: &'static str,
    require_thinking_closed: bool,
    has_tools: bool,
    saw_tool_start: bool,
    finish: &'static str,
}

fn write_format_case<W: Write>(out: &mut W, case: &FormatCase) -> io::Result<()> {
    match case {
        FormatCase::ToolCalls {
            name,
            raw_dsml,
            calls,
        } => {
            let rendered = render_dsml_tool_calls_from_json(*raw_dsml, calls);
            write!(out, "    {{\"name\": ")?;
            write_json_string(out, name)?;
            write!(out, ", \"kind\": \"tool_calls\", \"rendered\": ")?;
            write_json_string(out, &rendered)?;
            write!(out, "}}")
        }
        FormatCase::ToolResult { name, input } => {
            let rendered = render_tool_result_text(input);
            write!(out, "    {{\"name\": ")?;
            write_json_string(out, name)?;
            write!(out, ", \"kind\": \"tool_result\", \"input\": ")?;
            write_json_string(out, input)?;
            write!(out, ", \"rendered\": ")?;
            write_json_string(out, &rendered)?;
            write!(out, "}}")
        }
    }
}

fn write_parse_case<W: Write>(out: &mut W, case: &ParseCase) -> io::Result<()> {
    let parsed = parse_generated_message(case.input, case.require_thinking_closed);
    let response = parse_generated_message_for_response(
        case.input,
        case.has_tools,
        case.saw_tool_start,
        case.require_thinking_closed,
        case.finish,
    );
    write!(out, "    {{\"name\": ")?;
    write_json_string(out, case.name)?;
    write!(out, ", \"require_thinking_closed\": ")?;
    write!(out, "{}", case.require_thinking_closed)?;
    write!(out, ", \"input\": ")?;
    write_json_string(out, case.input)?;
    write!(out, ", \"parse_ok\": ")?;
    write!(out, "{}", parsed.is_ok())?;
    match parsed {
        Ok(message) => write_message_fields(out, &message, "", true)?,
        Err(_) => {
            write!(out, ", \"content\": null")?;
            write!(out, ", \"reasoning\": null")?;
            write!(out, ", \"raw_dsml\": null")?;
            write!(out, ", \"calls\": []")?;
        }
    }
    write!(out, ", \"response_parse_ok\": ")?;
    write!(out, "{}", response.parse_ok)?;
    write!(out, ", \"response_recovered\": ")?;
    write!(out, "{}", response.recovered)?;
    write!(out, ", \"response_finish\": ")?;
    write_json_string(out, &response.finish)?;
    write!(out, ", \"response_error\": ")?;
    write_json_string(out, &response.error)?;
    write_message_fields(out, &response.message, "response_", false)?;
    write!(out, "}}")
}

fn write_message_fields<W: Write>(
    out: &mut W,
    message: &ParsedGeneratedMessage,
    prefix: &str,
    include_leading_comma: bool,
) -> io::Result<()> {
    if include_leading_comma {
        write!(out, ", ")?;
    } else {
        write!(out, ", ")?;
    }
    write!(out, "\"{prefix}content\": ")?;
    write_json_string(out, &message.content)?;
    write!(out, ", \"{prefix}reasoning\": ")?;
    write_json_nullable(out, message.reasoning.as_deref())?;
    write!(out, ", \"{prefix}raw_dsml\": ")?;
    write_json_nullable(out, message.raw_dsml.as_deref())?;
    write!(out, ", \"{prefix}calls\": [")?;
    for (idx, call) in message.calls.iter().enumerate() {
        if idx != 0 {
            write!(out, ", ")?;
        }
        write!(out, "{{\"id\": ")?;
        write_json_nullable(out, call.id.as_deref())?;
        write!(out, ", \"name\": ")?;
        write_json_string(out, &call.name)?;
        write!(out, ", \"arguments\": ")?;
        write_json_string(out, &call.arguments)?;
        write!(out, "}}")?;
    }
    write!(out, "]")?;
    Ok(())
}

fn format_cases() -> Vec<FormatCase> {
    vec![
        FormatCase::ToolCalls {
            name: "ordered_string_and_json_parameters",
            raw_dsml: None,
            calls: vec![DsmlJsonCall::new(
                "bash",
                "{\"description\":\"list files\",\"command\":\"ls -la\",\"timeout\":10}",
            )],
        },
        FormatCase::ToolCalls {
            name: "attribute_and_json_sentinel_escape",
            raw_dsml: None,
            calls: vec![DsmlJsonCall::new(
                "bad<&\"name",
                "{\"text\":\"a < b && c > d\",\"payload\":{\"end\":\"</｜DSML｜parameter>\"}}",
            )],
        },
        FormatCase::ToolCalls {
            name: "json_key_escaping",
            raw_dsml: None,
            calls: vec![DsmlJsonCall::new(
                "keys",
                "{\"quote\\\"key\":\"line\\nvalue\",\"slash\\\\key\":true}",
            )],
        },
        FormatCase::ToolCalls {
            name: "invalid_arguments_fallback",
            raw_dsml: None,
            calls: vec![DsmlJsonCall::new(
                "fallback",
                "not-json </｜DSML｜parameter> tail",
            )],
        },
        FormatCase::ToolCalls {
            name: "raw_dsml_replay",
            raw_dsml: Some(
                "<DSML｜tool_calls>\n<DSML｜invoke name=\"raw\">\n</DSML｜invoke>\n</DSML｜tool_calls>",
            ),
            calls: vec![DsmlJsonCall::new("ignored", "{\"x\":1}")],
        },
        FormatCase::ToolResult {
            name: "tool_result_closing_sentinel",
            input: "console.log('<tag>');\n</tool_result>\n& raw",
        },
    ]
}

fn parse_cases() -> Vec<ParseCase> {
    vec![
        ParseCase {
            name: "canonical_after_think",
            input: "<think>need a shell check</think>\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"bash\">\n<｜DSML｜parameter name=\"command\" string=\"true\">pwd</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
            require_thinking_closed: true,
            has_tools: true,
            saw_tool_start: true,
            finish: "tool_calls",
        },
        ParseCase {
            name: "short_marker",
            input: "<think>need a tool</think><DSML｜tool_calls>\n<DSML｜invoke name=\"bash\">\n<DSML｜parameter name=\"description\" string=\"true\">list files</DSML｜parameter>\n<DSML｜parameter name=\"command\" string=\"true\">ls -la</DSML｜parameter>\n</DSML｜invoke>\n</DSML｜tool_calls>",
            require_thinking_closed: false,
            has_tools: true,
            saw_tool_start: true,
            finish: "tool_calls",
        },
        ParseCase {
            name: "plain_xml_marker",
            input: "done\n\n<tool_calls>\n<invoke name=\"plain\">\n<parameter name=\"value\" string=\"true\">ok</parameter>\n</invoke>\n</tool_calls>",
            require_thinking_closed: false,
            has_tools: true,
            saw_tool_start: true,
            finish: "tool_calls",
        },
        ParseCase {
            name: "loose_nested_parameters",
            input: "review done\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"edit\">\n<｜DSML｜parameter name=\"path\">/private/tmp/tetris.c</｜DSML｜parameter>\n<｜DSML｜parameter name=\"edits\">\n<｜DSML｜parameter name=\"oldText\" string=\"true\">old &lt;text&gt;</｜DSML｜parameter>\n<｜DSML｜parameter name=\"newText\" string=\"true\">new text</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
            require_thinking_closed: false,
            has_tools: true,
            saw_tool_start: true,
            finish: "tool_calls",
        },
        ParseCase {
            name: "thinking_dsml_ignored_before_close",
            input: "<think>I might mention a tool:\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"bash\">\n<｜DSML｜parameter name=\"command\" string=\"true\">true</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>\nBut it is reasoning.</think>Final answer.",
            require_thinking_closed: true,
            has_tools: true,
            saw_tool_start: true,
            finish: "stop",
        },
        ParseCase {
            name: "missing_think_close_ignored",
            input: "<think>unfinished thought\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"bash\">\n<｜DSML｜parameter name=\"command\" string=\"true\">true</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
            require_thinking_closed: true,
            has_tools: true,
            saw_tool_start: true,
            finish: "length",
        },
        ParseCase {
            name: "separator_whitespace_trim",
            input: "<think>need a tool</think>I will inspect.\n\n\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"bash\">\n<｜DSML｜parameter name=\"command\" string=\"true\">ls -la</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
            require_thinking_closed: false,
            has_tools: true,
            saw_tool_start: true,
            finish: "tool_calls",
        },
        ParseCase {
            name: "json_parameter_minified",
            input: "need edit</think>\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"edit\">\n<｜DSML｜parameter name=\"path\" string=\"true\">/tmp/file</｜DSML｜parameter>\n<｜DSML｜parameter name=\"edits\" string=\"false\">[{\"oldText\": \"status=created\", \"newText\": \"status=created\\nstatus2=resumed\"}]</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
            require_thinking_closed: false,
            has_tools: true,
            saw_tool_start: true,
            finish: "tool_calls",
        },
        ParseCase {
            name: "multiple_invokes",
            input: "done\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"first\">\n<｜DSML｜parameter name=\"value\" string=\"true\">one</｜DSML｜parameter>\n</｜DSML｜invoke>\n<｜DSML｜invoke name=\"second\">\n<｜DSML｜parameter name=\"value\" string=\"false\">2</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
            require_thinking_closed: false,
            has_tools: true,
            saw_tool_start: true,
            finish: "tool_calls",
        },
        ParseCase {
            name: "default_string_attribute",
            input: "done\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"bash\">\n<｜DSML｜parameter name=\"command\">echo &lt;ok&gt; &amp;&amp; true</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
            require_thinking_closed: false,
            has_tools: true,
            saw_tool_start: true,
            finish: "tool_calls",
        },
        ParseCase {
            name: "malformed_missing_invoke_name",
            input: "trying a tool\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke>\n</｜DSML｜tool_calls>",
            require_thinking_closed: false,
            has_tools: true,
            saw_tool_start: true,
            finish: "tool_calls",
        },
        ParseCase {
            name: "malformed_error_finish",
            input: "trying a tool\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke>\n</｜DSML｜tool_calls>",
            require_thinking_closed: false,
            has_tools: true,
            saw_tool_start: true,
            finish: "error",
        },
        ParseCase {
            name: "truncated_parameter_length_finish",
            input: "trying a tool\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"bash\">\n<｜DSML｜parameter name=\"command\" string=\"true\">unterminated",
            require_thinking_closed: false,
            has_tools: true,
            saw_tool_start: true,
            finish: "length",
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
