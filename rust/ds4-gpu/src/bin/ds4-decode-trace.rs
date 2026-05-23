use ds4_gpu::decode_trace::{
    decode_trace_case, for_each_trace_event, trace_summary, DecodeTraceEvent, DecodeTraceState,
    DecodeTraceSummary, DECODE_TRACE_SCHEMA, DECODE_TRACE_SCOPE,
};
use std::env;

fn main() {
    let args = Args::parse();
    println!("{{");
    println!("  \"schema\": \"{}\",", DECODE_TRACE_SCHEMA);
    println!("  \"scope\": \"{}\",", DECODE_TRACE_SCOPE);
    println!("  \"cases\": [");
    let mut first_case = true;
    for case in ds4_gpu::decode_plan::M105B_DECODE_CASE_ORACLE {
        if let Some(case_name) = args.case {
            if case.name != case_name {
                continue;
            }
        }
        if !first_case {
            println!(",");
        }
        first_case = false;
        write_case(*case);
    }
    if first_case {
        eprintln!("unknown M10.5b decode plan case");
        std::process::exit(2);
    }
    println!();
    println!("  ]");
    println!("}}");
}

struct Args {
    case: Option<&'static str>,
}

impl Args {
    fn parse() -> Self {
        let mut case = None;
        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            if arg == "--case" {
                let Some(value) = args.next() else {
                    eprintln!("--case requires a value");
                    std::process::exit(2);
                };
                if decode_trace_case(&value).is_none() {
                    eprintln!("unknown M10.5b decode plan case: {value}");
                    std::process::exit(2);
                }
                case = Some(leak_arg(value));
            } else {
                eprintln!("usage: ds4-decode-trace [--case NAME]");
                std::process::exit(2);
            }
        }
        Self { case }
    }
}

fn leak_arg(value: String) -> &'static str {
    Box::leak(value.into_boxed_str())
}

fn write_case(case: ds4_gpu::decode_plan::DecodePlanCaseOracle) {
    let input = case.input;
    let plan = case.computed();
    println!("    {{");
    println!("      \"name\": \"{}\",", case.name);
    println!("      \"ctx_size\": {},", input.ctx_size);
    println!("      \"prompt_len\": {},", input.prompt_len);
    println!("      \"pos\": {},", input.pos);
    println!("      \"need_logits\": {},", bool_json(input.need_logits));
    println!(
        "      \"allow_split_flush\": {},",
        bool_json(input.allow_split_flush)
    );
    println!("      \"raw_row\": {},", plan.raw_row);
    println!("      \"n_raw\": {},", plan.n_raw);
    println!("      \"raw_start\": {},", plan.raw_start);
    write_summary(trace_summary(case));
    println!("      \"events\": [");
    let mut first = true;
    for_each_trace_event(case, |event| {
        if !first {
            println!(",");
        }
        first = false;
        write_event(event);
    });
    println!();
    println!("      ]");
    println!("    }}");
}

fn write_summary(summary: DecodeTraceSummary) {
    println!("      \"summary\": {{");
    println!("        \"events\": {},", summary.events);
    println!("        \"stage_markers\": {},", summary.stage_markers);
    println!("        \"facade_calls\": {},", summary.facade_calls);
    println!("        \"existing_calls\": {},", summary.existing_calls);
    println!("        \"state_events\": {},", summary.state_events);
    println!("        \"layers\": {},", summary.layers);
    println!("        \"dense_layers\": {},", summary.dense_layers);
    println!("        \"ratio4_layers\": {},", summary.ratio4_layers);
    println!("        \"ratio128_layers\": {},", summary.ratio128_layers);
    println!(
        "        \"compressed_emit_layers\": {},",
        summary.compressed_emit_layers
    );
    println!(
        "        \"indexed_attention_layers\": {},",
        summary.indexed_attention_layers
    );
    println!("        \"split_flushes\": {},", summary.split_flushes);
    println!(
        "        \"output_head_calls\": {},",
        summary.output_head_calls
    );
    println!(
        "        \"read_logits_calls\": {},",
        summary.read_logits_calls
    );
    println!(
        "        \"synchronize_on_failure_calls\": {}",
        summary.synchronize_on_failure_calls
    );
    println!("      }},");
}

fn write_event(event: DecodeTraceEvent) {
    print!("        {{");
    print!("\"index\": {}", event.index);
    print!(", \"kind\": \"{}\"", event.kind.name());
    print!(", \"stage\": \"{}\"", event.stage);
    print!(", \"layer\": ");
    write_optional_u32(event.layer);
    print!(", \"operation\": \"{}\"", event.operation);
    print!(", \"method\": \"{}\"", event.method);
    print!(", \"tensor_args\": [");
    for (i, arg) in event.tensor_args.iter().enumerate() {
        if i != 0 {
            print!(", ");
        }
        print!("\"{arg}\"");
    }
    print!("]");
    write_state(event.state);
    print!("}}");
}

fn write_state(state: DecodeTraceState) {
    print!(", \"state\": {{");
    print!("\"compression\": \"{}\"", state.compression);
    print!(", \"pos\": ");
    write_optional_u32(state.pos);
    print!(", \"raw_row\": ");
    write_optional_u32(state.raw_row);
    print!(", \"n_raw\": ");
    write_optional_u32(state.n_raw);
    print!(", \"raw_start\": ");
    write_optional_u32(state.raw_start);
    print!(", \"comp_before\": ");
    write_optional_u32(state.comp_before);
    print!(", \"comp_after\": ");
    write_optional_u32(state.comp_after);
    print!(", \"index_before\": ");
    write_optional_u32(state.index_before);
    print!(", \"index_after\": ");
    write_optional_u32(state.index_after);
    print!(
        ", \"emit_compressed_row\": {}",
        bool_json(state.emit_compressed_row)
    );
    print!(
        ", \"indexed_attention\": {}",
        bool_json(state.indexed_attention)
    );
    print!(
        ", \"attention_operation\": \"{}\"",
        state.attention_operation
    );
    print!("}}");
}

fn write_optional_u32(value: Option<u32>) {
    match value {
        Some(value) => print!("{value}"),
        None => print!("null"),
    }
}

fn bool_json(value: bool) -> &'static str {
    if value {
        "true"
    } else {
        "false"
    }
}
