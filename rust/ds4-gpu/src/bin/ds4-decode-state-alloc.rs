use ds4_gpu::graph_plan::{GraphPlan, ModelGraphDims, TensorByteLen};
use ds4_gpu::graph_state::{
    GraphStateAllocation, GraphTensorInitialFill, GraphTensorInstances, GraphTensorStorage,
    DECODE_GRAPH_STATE_FIELDS,
};
use ds4_gpu::{initialize, Tensor};

const SCHEMA: &str = "ds4.decode_state_allocation.v1";
const CASE: &str = "ctx32768_mtp_off";
const MAX_REPORTED_ALLOCATIONS: usize = 12;

fn main() {
    if let Err(err) = run() {
        eprintln!("ds4-decode-state-alloc: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    initialize().map_err(|err| format!("failed to initialize backend: {err}"))?;
    let _backend = BackendGuard;
    let report = allocate_state()?;
    write_report(&report);
    Ok(())
}

struct BackendGuard;

impl Drop for BackendGuard {
    fn drop(&mut self) {
        unsafe {
            ds4_gpu::cleanup();
        }
    }
}

struct OwnedTensor {
    field: &'static str,
    layer: Option<usize>,
    tensor: Tensor,
}

#[derive(Clone, Copy)]
struct AllocationReport {
    field: &'static str,
    layer: Option<usize>,
    bytes: u64,
    fill: &'static str,
}

#[derive(Default)]
struct AllocationSummary {
    logical_instances: u32,
    initial_owned_allocations: u32,
    initial_owned_bytes: u64,
    views_created: u32,
    lazy_owned_deferred: u32,
    external_inputs: u32,
    zero_full_capacity_fills: u32,
    zero_state_fills: u32,
    negative_infinity_fills: u32,
}

struct StateAllocationReport {
    summary: AllocationSummary,
    largest_allocations: Vec<AllocationReport>,
    views: Vec<ViewReport>,
}

struct ViewReport {
    field: &'static str,
    base: &'static str,
    layer: Option<usize>,
    offset_bytes: u64,
    bytes: u64,
}

fn allocate_state() -> Result<StateAllocationReport, String> {
    let plan = GraphPlan::for_context(32768, 32768, false);
    let dims = ModelGraphDims::DS4_FLASH;
    let mut owned: Vec<OwnedTensor> = Vec::new();
    let mut allocations: Vec<AllocationReport> = Vec::new();
    let mut views: Vec<ViewReport> = Vec::new();
    let mut summary = AllocationSummary::default();

    for field in DECODE_GRAPH_STATE_FIELDS {
        match field.instances {
            GraphTensorInstances::Single => {
                process_instance(
                    *field,
                    plan,
                    dims,
                    None,
                    &mut owned,
                    &mut allocations,
                    &mut views,
                    &mut summary,
                )?;
            }
            GraphTensorInstances::PerLayer => {
                for layer in 0..ds4_gpu::graph_plan::N_LAYER {
                    process_instance(
                        *field,
                        plan,
                        dims,
                        Some(layer),
                        &mut owned,
                        &mut allocations,
                        &mut views,
                        &mut summary,
                    )?;
                }
            }
        }
    }

    allocations.sort_by(|left, right| {
        right
            .bytes
            .cmp(&left.bytes)
            .then_with(|| left.field.cmp(right.field))
            .then_with(|| left.layer.cmp(&right.layer))
    });
    allocations.truncate(MAX_REPORTED_ALLOCATIONS);
    drop(owned);

    Ok(StateAllocationReport {
        summary,
        largest_allocations: allocations,
        views,
    })
}

#[allow(clippy::too_many_arguments)]
fn process_instance(
    field: ds4_gpu::graph_state::GraphStateFieldPlan,
    plan: GraphPlan,
    dims: ModelGraphDims,
    layer: Option<usize>,
    owned: &mut Vec<OwnedTensor>,
    allocations: &mut Vec<AllocationReport>,
    views: &mut Vec<ViewReport>,
    summary: &mut AllocationSummary,
) -> Result<(), String> {
    let allocation = field
        .initial_allocation(plan, dims, layer)
        .map_err(|err| format!("allocation plan failed for {}: {err:?}", field.name))?;
    summary.logical_instances += 1;
    match allocation.storage {
        GraphTensorStorage::Owned if allocation.initially_allocated => {
            allocate_owned(allocation, owned, allocations, summary)?;
        }
        GraphTensorStorage::Owned => {}
        GraphTensorStorage::View { base, offset_bytes } => {
            create_view(allocation, base, offset_bytes, owned, views, summary)?;
        }
        GraphTensorStorage::LazyOwned => {
            summary.lazy_owned_deferred += 1;
        }
        GraphTensorStorage::External => {
            summary.external_inputs += 1;
        }
    }
    Ok(())
}

fn allocate_owned(
    allocation: GraphStateAllocation,
    owned: &mut Vec<OwnedTensor>,
    allocations: &mut Vec<AllocationReport>,
    summary: &mut AllocationSummary,
) -> Result<(), String> {
    let TensorByteLen::Known(bytes) = allocation.byte_len else {
        return Err(format!("{} has external byte length", allocation.field));
    };
    let mut tensor = Tensor::allocate(usize::try_from(bytes).map_err(|_| "byte length overflow")?)
        .map_err(|err| {
            format!(
                "failed to allocate {} {:?}: {err}",
                allocation.field, allocation.layer
            )
        })?;
    apply_initial_fill(&mut tensor, allocation.initial_fill, bytes, summary)?;
    summary.initial_owned_allocations += 1;
    summary.initial_owned_bytes += bytes;
    allocations.push(AllocationReport {
        field: allocation.field,
        layer: allocation.layer,
        bytes,
        fill: allocation.initial_fill.name(),
    });
    owned.push(OwnedTensor {
        field: allocation.field,
        layer: allocation.layer,
        tensor,
    });
    Ok(())
}

fn apply_initial_fill(
    tensor: &mut Tensor,
    fill: GraphTensorInitialFill,
    bytes: u64,
    summary: &mut AllocationSummary,
) -> Result<(), String> {
    if fill != GraphTensorInitialFill::Unspecified && bytes % 4 != 0 {
        return Err(format!(
            "{} fill byte length is not f32-aligned",
            fill.name()
        ));
    }
    let count = usize::try_from(bytes / 4).map_err(|_| "fill count overflow")?;
    match fill {
        GraphTensorInitialFill::ZeroFullCapacity => {
            tensor
                .fill_f32(0.0, count)
                .map_err(|err| format!("zero full-capacity fill failed: {err}"))?;
            summary.zero_full_capacity_fills += 1;
        }
        GraphTensorInitialFill::ZeroState => {
            tensor
                .fill_f32(0.0, count)
                .map_err(|err| format!("zero state fill failed: {err}"))?;
            summary.zero_state_fills += 1;
        }
        GraphTensorInitialFill::NegativeInfinityState => {
            tensor
                .fill_f32(f32::NEG_INFINITY, count)
                .map_err(|err| format!("negative-infinity fill failed: {err}"))?;
            summary.negative_infinity_fills += 1;
        }
        GraphTensorInitialFill::Unspecified => {}
        GraphTensorInitialFill::ExternalInput => {
            return Err("external input cannot be an allocated owned tensor".to_string());
        }
    }
    Ok(())
}

fn create_view(
    allocation: GraphStateAllocation,
    base: &'static str,
    offset_bytes: u64,
    owned: &mut [OwnedTensor],
    views: &mut Vec<ViewReport>,
    summary: &mut AllocationSummary,
) -> Result<(), String> {
    let TensorByteLen::Known(bytes) = allocation.byte_len else {
        return Err(format!(
            "{} view has external byte length",
            allocation.field
        ));
    };
    let base_tensor = owned
        .iter_mut()
        .find(|entry| entry.field == base && entry.layer == allocation.layer)
        .ok_or_else(|| format!("view {} missing base {base}", allocation.field))?;
    let view = base_tensor
        .tensor
        .view(
            offset_bytes,
            usize::try_from(bytes).map_err(|_| "view byte length overflow")?,
        )
        .map_err(|err| format!("failed to create view {}: {err}", allocation.field))?;
    if view.byte_len() != bytes {
        return Err(format!(
            "view {} byte length drift: expected {bytes}, got {}",
            allocation.field,
            view.byte_len()
        ));
    }
    drop(view);
    summary.views_created += 1;
    views.push(ViewReport {
        field: allocation.field,
        base,
        layer: allocation.layer,
        offset_bytes,
        bytes,
    });
    Ok(())
}

fn write_report(report: &StateAllocationReport) {
    println!("{{");
    println!("  \"schema\": \"{SCHEMA}\",");
    println!("  \"case\": \"{CASE}\",");
    println!("  \"ctx_size\": 32768,");
    println!("  \"prompt_len\": 32768,");
    write_summary(&report.summary);
    write_allocations(&report.largest_allocations);
    write_views(&report.views);
    println!("}}");
}

fn write_summary(summary: &AllocationSummary) {
    println!("  \"summary\": {{");
    println!("    \"logical_instances\": {},", summary.logical_instances);
    println!(
        "    \"initial_owned_allocations\": {},",
        summary.initial_owned_allocations
    );
    println!(
        "    \"initial_owned_bytes\": {},",
        summary.initial_owned_bytes
    );
    println!("    \"views_created\": {},", summary.views_created);
    println!(
        "    \"lazy_owned_deferred\": {},",
        summary.lazy_owned_deferred
    );
    println!("    \"external_inputs\": {},", summary.external_inputs);
    println!(
        "    \"zero_full_capacity_fills\": {},",
        summary.zero_full_capacity_fills
    );
    println!("    \"zero_state_fills\": {},", summary.zero_state_fills);
    println!(
        "    \"negative_infinity_fills\": {}",
        summary.negative_infinity_fills
    );
    println!("  }},");
}

fn write_allocations(allocations: &[AllocationReport]) {
    println!("  \"largest_allocations\": [");
    for (idx, allocation) in allocations.iter().enumerate() {
        if idx != 0 {
            println!(",");
        }
        print!("    {{\"field\": \"{}\"", allocation.field);
        print!(", \"layer\": ");
        write_option_usize(allocation.layer);
        print!(", \"bytes\": {}", allocation.bytes);
        print!(", \"fill\": \"{}\"", allocation.fill);
        print!("}}");
    }
    println!();
    println!("  ],");
}

fn write_views(views: &[ViewReport]) {
    println!("  \"views\": [");
    for (idx, view) in views.iter().enumerate() {
        if idx != 0 {
            println!(",");
        }
        print!("    {{\"field\": \"{}\"", view.field);
        print!(", \"base\": \"{}\"", view.base);
        print!(", \"layer\": ");
        write_option_usize(view.layer);
        print!(", \"offset_bytes\": {}", view.offset_bytes);
        print!(", \"bytes\": {}", view.bytes);
        print!("}}");
    }
    println!();
    println!("  ]");
}

fn write_option_usize(value: Option<usize>) {
    match value {
        Some(value) => print!("{value}"),
        None => print!("null"),
    }
}
