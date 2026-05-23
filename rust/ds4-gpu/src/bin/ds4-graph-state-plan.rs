use ds4_gpu::graph_plan::{GraphPlan, ModelGraphDims, TensorByteLen, M102_PLAN_CASE_ORACLE};
use ds4_gpu::graph_state::{
    GraphStateFieldPlan, GraphTensorInitialFill, GraphTensorInstances, GraphTensorStorage,
    DECODE_GRAPH_STATE_FIELDS,
};
use std::env;

fn main() {
    let args = Args::parse();
    let mut first_case = true;
    println!("{{");
    println!("  \"schema\": \"ds4.graph_state.v1\",");
    println!("  \"scope\": \"decode\",");
    println!("  \"cases\": [");
    for oracle in M102_PLAN_CASE_ORACLE {
        if let Some(case_name) = args.case {
            if oracle.name != case_name {
                continue;
            }
        }
        if !first_case {
            println!(",");
        }
        first_case = false;
        write_case(oracle.name, oracle.expected_plan());
    }
    if first_case {
        eprintln!("unknown M10.2 graph plan case");
        std::process::exit(2);
    }
    println!();
    println!("  ],");
    println!("  \"excluded_owners\": [");
    println!("    \"GraphSpeculativeFrontierState\",");
    println!("    \"GraphMtpState\",");
    println!("    \"GraphPrefillBatchState\"");
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
                case = Some(leak_arg(value));
            } else {
                eprintln!("usage: ds4-graph-state-plan [--case NAME]");
                std::process::exit(2);
            }
        }
        Self { case }
    }
}

fn leak_arg(value: String) -> &'static str {
    Box::leak(value.into_boxed_str())
}

fn write_case(name: &str, plan: GraphPlan) {
    let mut summary = Summary::default();
    println!("    {{");
    println!("      \"name\": \"{name}\",");
    println!("      \"ctx_size\": {},", plan.ctx_size);
    println!("      \"prompt_len\": {},", plan.prompt_len);
    println!("      \"mtp_enabled\": {},", bool_json(plan.mtp_enabled));
    println!("      \"allocations\": [");
    let mut first = true;
    for field in DECODE_GRAPH_STATE_FIELDS {
        match field.instances {
            GraphTensorInstances::Single => {
                write_allocation(&mut first, *field, plan, None, &mut summary);
            }
            GraphTensorInstances::PerLayer => {
                for layer in 0..ds4_gpu::graph_plan::N_LAYER {
                    write_allocation(&mut first, *field, plan, Some(layer), &mut summary);
                }
            }
        }
    }
    println!();
    println!("      ],");
    write_summary(&summary);
    println!("    }}");
}

fn write_allocation(
    first: &mut bool,
    field: GraphStateFieldPlan,
    plan: GraphPlan,
    layer: Option<usize>,
    summary: &mut Summary,
) {
    let allocation = field
        .initial_allocation(plan, ModelGraphDims::DS4_FLASH, layer)
        .expect("allocation plan");
    summary.logical_instances += 1;
    if allocation.initially_allocated {
        summary.initial_owned_allocations += 1;
    }
    match allocation.byte_len {
        TensorByteLen::Known(bytes) if allocation.initially_allocated => {
            summary.initial_owned_bytes += bytes;
        }
        TensorByteLen::Known(_) => {}
        TensorByteLen::External => summary.external_inputs += 1,
    }
    match allocation.storage {
        GraphTensorStorage::View { .. } => summary.views += 1,
        GraphTensorStorage::LazyOwned => summary.lazy_owned += 1,
        GraphTensorStorage::Owned | GraphTensorStorage::External => {}
    }
    if allocation.initially_allocated {
        match allocation.initial_fill {
            GraphTensorInitialFill::ZeroFullCapacity => summary.zero_full_capacity_fills += 1,
            GraphTensorInitialFill::ZeroState => summary.zero_state_fills += 1,
            GraphTensorInitialFill::NegativeInfinityState => summary.negative_infinity_fills += 1,
            GraphTensorInitialFill::Unspecified | GraphTensorInitialFill::ExternalInput => {}
        }
    }

    if !*first {
        println!(",");
    }
    *first = false;
    print!("        {{\"field\": \"{}\"", allocation.field);
    match allocation.layer {
        Some(layer) => print!(", \"layer\": {layer}"),
        None => print!(", \"layer\": null"),
    }
    print!(", \"owner\": \"{}\"", allocation.owner.name());
    print!(", \"storage\": \"{}\"", allocation.storage.name());
    if let GraphTensorStorage::View { base, offset_bytes } = allocation.storage {
        print!(", \"view_base\": \"{base}\", \"view_offset_bytes\": {offset_bytes}");
    }
    print!(", \"instances\": \"{}\"", field.instances.name());
    print!(", \"element_type\": \"{}\"", element_type_name(field));
    print!(", \"initial_fill\": \"{}\"", allocation.initial_fill.name());
    match allocation.byte_len {
        TensorByteLen::Known(bytes) => print!(", \"bytes\": {bytes}"),
        TensorByteLen::External => print!(", \"bytes\": null"),
    }
    print!(
        ", \"initially_allocated\": {}",
        bool_json(allocation.initially_allocated)
    );
    print!("}}");
}

fn write_summary(summary: &Summary) {
    println!("      \"summary\": {{");
    println!(
        "        \"logical_instances\": {},",
        summary.logical_instances
    );
    println!(
        "        \"initial_owned_allocations\": {},",
        summary.initial_owned_allocations
    );
    println!(
        "        \"initial_owned_bytes\": {},",
        summary.initial_owned_bytes
    );
    println!("        \"views\": {},", summary.views);
    println!("        \"lazy_owned\": {},", summary.lazy_owned);
    println!("        \"external_inputs\": {},", summary.external_inputs);
    println!(
        "        \"zero_full_capacity_fills\": {},",
        summary.zero_full_capacity_fills
    );
    println!(
        "        \"zero_state_fills\": {},",
        summary.zero_state_fills
    );
    println!(
        "        \"negative_infinity_fills\": {}",
        summary.negative_infinity_fills
    );
    println!("      }}");
}

fn element_type_name(field: GraphStateFieldPlan) -> &'static str {
    match field.element_type {
        ds4_gpu::graph_plan::ElementType::F32 => "f32",
        ds4_gpu::graph_plan::ElementType::I32 => "i32",
        ds4_gpu::graph_plan::ElementType::U32 => "u32",
    }
}

fn bool_json(value: bool) -> &'static str {
    if value {
        "true"
    } else {
        "false"
    }
}

#[derive(Default)]
struct Summary {
    logical_instances: u32,
    initial_owned_allocations: u32,
    initial_owned_bytes: u64,
    views: u32,
    lazy_owned: u32,
    external_inputs: u32,
    zero_full_capacity_fills: u32,
    zero_state_fills: u32,
    negative_infinity_fills: u32,
}
