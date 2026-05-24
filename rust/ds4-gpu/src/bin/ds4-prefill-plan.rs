use ds4_gpu::prefill_plan::{
    prefill_plan, PrefillChunk, PrefillPlan, PrefillPlanCaseOracle, PrefillRoute,
    M106A_PREFILL_PLAN_CASE_ORACLE, PREFILL_PLAN_SCHEMA, PREFILL_PLAN_SCOPE,
};
use std::env;

fn main() {
    let args = Args::parse();
    println!("{{");
    println!("  \"schema\": \"{}\",", PREFILL_PLAN_SCHEMA);
    println!("  \"scope\": \"{}\",", PREFILL_PLAN_SCOPE);
    println!("  \"cases\": [");
    let mut first_case = true;
    for case in M106A_PREFILL_PLAN_CASE_ORACLE {
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
        eprintln!("unknown M10.6a prefill plan case");
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
                if !M106A_PREFILL_PLAN_CASE_ORACLE
                    .iter()
                    .any(|case| case.name == value)
                {
                    eprintln!("unknown M10.6a prefill plan case: {value}");
                    std::process::exit(2);
                }
                case = Some(leak_arg(value));
            } else {
                eprintln!("usage: ds4-prefill-plan [--case NAME]");
                std::process::exit(2);
            }
        }
        Self { case }
    }
}

fn leak_arg(value: String) -> &'static str {
    Box::leak(value.into_boxed_str())
}

fn write_case(case: PrefillPlanCaseOracle) {
    let input = case.input;
    let computed = prefill_plan(input);
    println!("    {{");
    println!("      \"name\": \"{}\",", case.name);
    println!("      \"input\": {{");
    println!("        \"ctx_size\": {},", input.ctx_size);
    println!("        \"prompt_len\": {},", input.prompt_len);
    println!("        \"start\": {},", input.start);
    println!("        \"n_tokens\": {},", input.n_tokens);
    println!(
        "        \"checkpoint_valid\": {}",
        bool_json(input.checkpoint_valid)
    );
    println!("      }},");
    println!("      \"computed\": {{");
    write_plan(computed);
    println!("      }},");
    println!("      \"expected\": {{");
    write_plan(case.expected());
    println!("      }}");
    println!("    }}");
}

fn write_plan(plan: PrefillPlan) {
    println!("        \"route\": \"{}\",", plan.route.name());
    println!("        \"prefill_cap\": {},", plan.prefill_cap);
    println!("        \"raw_cap\": {},", plan.raw_cap);
    println!("        \"chunk_cap\": {},", plan.chunk_cap);
    println!(
        "        \"first_chunk_tokens\": {},",
        plan.first_chunk_tokens
    );
    println!("        \"chunk_count\": {},", plan.chunk_count);
    print!("        \"final_output_batch_row\": ");
    write_optional_u32(plan.final_output_batch_row);
    println!(",");
    print!("        \"output_absolute_pos\": ");
    write_optional_u32(plan.output_absolute_pos);
    println!(",");
    println!(
        "        \"progress_point_count\": {},",
        plan.progress_point_count
    );
    println!("        \"layer_batch_calls\": {},", plan.layer_batch_calls);
    println!("        \"chunks\": [");
    for index in 0..plan.chunk_count as usize {
        if index != 0 {
            println!(",");
        }
        write_chunk(plan.chunks[index]);
    }
    println!();
    println!("        ],");
    print!("        \"progress_points\": [");
    for index in 0..plan.progress_point_count as usize {
        if index != 0 {
            print!(", ");
        }
        print!("{}", plan.progress_points[index]);
    }
    println!("]");
}

fn write_chunk(chunk: PrefillChunk) {
    print!(
        "          {{\"start\": {}, \"tokens\": {}}}",
        chunk.start, chunk.tokens
    );
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

#[allow(dead_code)]
fn route_name(route: PrefillRoute) -> &'static str {
    route.name()
}
