use ds4_gpu::decode_runtime::{
    for_each_runtime_handle, layer_counters, layer_weight_presence, runtime_summary,
    BASE_WEIGHT_FIELDS, COMMON_LAYER_WEIGHT_FIELDS, COMPRESSED_LAYER_WEIGHT_FIELDS,
    DECODE_RUNTIME_CASE, DECODE_RUNTIME_SCHEMA, DECODE_RUNTIME_SCOPE, FACADE_TENSOR_ARG_BINDINGS,
    HASH_LAYER_WEIGHT_FIELDS, OPTIONAL_LAYER_WEIGHT_FIELDS, RATIO4_INDEXER_WEIGHT_FIELDS,
    WEIGHT_SLICE_LAYERS,
};
use ds4_gpu::graph_plan::{GraphPlan, LayerCompression, ModelGraphDims, TensorByteLen, N_LAYER};
use ds4_gpu::graph_state::{GraphStateAllocation, GraphTensorStorage};

fn main() {
    let plan = GraphPlan::for_context(32768, 32768, false);
    let dims = ModelGraphDims::DS4_FLASH;
    println!("{{");
    println!("  \"schema\": \"{}\",", DECODE_RUNTIME_SCHEMA);
    println!("  \"scope\": \"{}\",", DECODE_RUNTIME_SCOPE);
    println!("  \"case\": {{");
    println!("    \"name\": \"{}\",", DECODE_RUNTIME_CASE);
    println!("    \"ctx_size\": {},", plan.ctx_size);
    println!("    \"prompt_len\": {},", plan.prompt_len);
    println!("    \"mtp_enabled\": {},", bool_json(plan.mtp_enabled));
    write_summary(runtime_summary(plan, dims));
    write_handles(plan, dims);
    write_counters(plan);
    write_facade_arg_bindings();
    write_weight_requirements();
    println!("  }}");
    println!("}}");
}

fn write_summary(summary: ds4_gpu::decode_runtime::DecodeRuntimeSummary) {
    println!("    \"summary\": {{");
    println!("      \"logical_handles\": {},", summary.logical_handles);
    println!(
        "      \"initial_owned_allocations\": {},",
        summary.initial_owned_allocations
    );
    println!("      \"views\": {},", summary.views);
    println!("      \"lazy_owned\": {},", summary.lazy_owned);
    println!("      \"external_inputs\": {},", summary.external_inputs);
    println!(
        "      \"initial_layer_counters\": {}",
        summary.initial_layer_counters
    );
    println!("    }},");
}

fn write_handles(plan: GraphPlan, dims: ModelGraphDims) {
    println!("    \"handles\": [");
    let mut first = true;
    for_each_runtime_handle(plan, dims, |handle| {
        if !first {
            println!(",");
        }
        first = false;
        print!("      {{");
        write_allocation(handle.field, handle.layer, handle.allocation);
        print!("}}");
    });
    println!();
    println!("    ],");
}

fn write_allocation(field: &str, layer: Option<usize>, allocation: GraphStateAllocation) {
    print!("\"field\": \"{field}\"");
    print!(", \"layer\": ");
    write_optional_usize(layer);
    print!(", \"owner\": \"{}\"", allocation.owner.name());
    print!(", \"storage\": \"{}\"", allocation.storage.name());
    if let GraphTensorStorage::View { base, offset_bytes } = allocation.storage {
        print!(", \"view_base\": \"{base}\"");
        print!(", \"view_offset_bytes\": {offset_bytes}");
    }
    print!(", \"initial_fill\": \"{}\"", allocation.initial_fill.name());
    print!(", \"bytes\": ");
    write_byte_len(allocation.byte_len);
    print!(
        ", \"initially_allocated\": {}",
        bool_json(allocation.initially_allocated)
    );
}

fn write_counters(plan: GraphPlan) {
    println!("    \"initial_layer_counters\": [");
    for layer in 0..N_LAYER {
        if layer != 0 {
            println!(",");
        }
        let counters = layer_counters(plan, layer).expect("layer");
        print!("      {{");
        print!("\"layer\": {}", counters.layer);
        print!(
            ", \"compression\": \"{}\"",
            compression_name(counters.compression)
        );
        print!(", \"layer_comp_cap\": {}", counters.layer_comp_cap);
        print!(", \"layer_n_comp\": {}", counters.layer_n_comp);
        print!(", \"layer_n_index_comp\": {}", counters.layer_n_index_comp);
        print!(", \"indexer_top_k\": {}", counters.indexer_top_k);
        print!("}}");
    }
    println!();
    println!("    ],");
}

fn write_facade_arg_bindings() {
    println!("    \"facade_arg_bindings\": [");
    for (idx, binding) in FACADE_TENSOR_ARG_BINDINGS.iter().enumerate() {
        if idx != 0 {
            println!(",");
        }
        print!("      {{");
        print!("\"operation\": \"{}\"", binding.operation);
        print!(", \"arg\": \"{}\"", binding.arg);
        print!(", \"source\": \"{}\"", binding.source.name());
        print!(", \"candidates\": [");
        write_str_list(binding.candidates);
        print!("]}}");
    }
    println!();
    println!("    ],");
}

fn write_weight_requirements() {
    println!("    \"weight_requirements\": [");
    let mut first = true;
    for field in BASE_WEIGHT_FIELDS {
        write_weight_requirement(&mut first, "base", None, field, "required_present");
    }
    for layer in WEIGHT_SLICE_LAYERS {
        for field in COMMON_LAYER_WEIGHT_FIELDS {
            write_layer_weight_requirement(&mut first, *layer, field);
        }
        for field in COMPRESSED_LAYER_WEIGHT_FIELDS {
            write_layer_weight_requirement(&mut first, *layer, field);
        }
        for field in RATIO4_INDEXER_WEIGHT_FIELDS {
            write_layer_weight_requirement(&mut first, *layer, field);
        }
        for field in HASH_LAYER_WEIGHT_FIELDS {
            write_layer_weight_requirement(&mut first, *layer, field);
        }
        for field in OPTIONAL_LAYER_WEIGHT_FIELDS {
            write_layer_weight_requirement(&mut first, *layer, field);
        }
    }
    println!();
    println!("    ]");
}

fn write_layer_weight_requirement(first: &mut bool, layer: usize, field: &str) {
    let presence = layer_weight_presence(layer, field).expect("weight field");
    write_weight_requirement(first, "layer", Some(layer), field, presence.name());
}

fn write_weight_requirement(
    first: &mut bool,
    scope: &str,
    layer: Option<usize>,
    field: &str,
    presence: &str,
) {
    if !*first {
        println!(",");
    }
    *first = false;
    print!("      {{");
    print!("\"scope\": \"{scope}\"");
    print!(", \"layer\": ");
    write_optional_usize(layer);
    print!(", \"field\": \"{field}\"");
    print!(", \"presence\": \"{presence}\"");
    print!(", \"role\": \"");
    match layer {
        Some(layer) => print!("base.layer.{layer}.{field}"),
        None => print!("base.{field}"),
    }
    print!("\"}}");
}

fn write_str_list(values: &[&str]) {
    for (idx, value) in values.iter().enumerate() {
        if idx != 0 {
            print!(", ");
        }
        print!("\"{value}\"");
    }
}

fn write_byte_len(byte_len: TensorByteLen) {
    match byte_len {
        TensorByteLen::Known(bytes) => print!("{bytes}"),
        TensorByteLen::External => print!("null"),
    }
}

fn write_optional_usize(value: Option<usize>) {
    match value {
        Some(value) => print!("{value}"),
        None => print!("null"),
    }
}

fn compression_name(compression: LayerCompression) -> &'static str {
    match compression {
        LayerCompression::Dense => "dense",
        LayerCompression::Ratio4 => "ratio4",
        LayerCompression::Ratio128 => "ratio128",
    }
}

fn bool_json(value: bool) -> &'static str {
    if value {
        "true"
    } else {
        "false"
    }
}
