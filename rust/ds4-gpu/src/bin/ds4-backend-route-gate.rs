use ds4_gpu::backend_route_gate::{
    first_backend_runtime_route_gate, route_decision, RuntimeBackendRouteDecision,
    RuntimeBackendRouteError, RuntimeBackendRouteGateSpec,
};
use std::env;
use std::io::{self, Write};

fn main() {
    if let Err(err) = run() {
        let _ = writeln!(io::stderr(), "ds4-backend-route-gate: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    let spec = first_backend_runtime_route_gate();
    let request = parse_request(&args)?;
    match request {
        RouteRequest::Summary => print_summary(spec, None),
        RouteRequest::Check { route, backend } => match route_decision(spec, route, backend) {
            Ok(decision) => print_summary(spec, Some(decision)),
            Err(RuntimeBackendRouteError::UnsupportedRoute { requested }) => {
                print_error(spec, "unsupported-route", "requested_route", requested)?;
                std::process::exit(2);
            }
            Err(RuntimeBackendRouteError::UnsupportedBackend { requested }) => {
                print_error(spec, "unsupported-backend", "requested_backend", requested)?;
                std::process::exit(3);
            }
        },
    }
}

enum RouteRequest<'a> {
    Summary,
    Check { route: &'a str, backend: &'a str },
}

fn parse_request(args: &[String]) -> Result<RouteRequest<'_>, String> {
    if args.is_empty() {
        return Ok(RouteRequest::Summary);
    }
    let mut route = None;
    let mut backend = "cuda-b300";
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "--route" | "--runtime-backend-route" => {
                index += 1;
                route = Some(
                    args.get(index)
                        .ok_or_else(|| "missing route after --route".to_string())?
                        .as_str(),
                );
            }
            "--backend" => {
                index += 1;
                backend = args
                    .get(index)
                    .ok_or_else(|| "missing backend after --backend".to_string())?
                    .as_str();
            }
            _ => {
                return Err(
                    "usage: ds4-backend-route-gate [--route NAME] [--backend NAME]".to_string(),
                );
            }
        }
        index += 1;
    }
    let route = route.ok_or_else(|| "missing --route NAME".to_string())?;
    Ok(RouteRequest::Check { route, backend })
}

fn print_summary(
    spec: &RuntimeBackendRouteGateSpec,
    decision: Option<RuntimeBackendRouteDecision<'_>>,
) -> Result<(), String> {
    let mut out = io::BufWriter::new(io::stdout().lock());
    writeln!(out, "{{").map_err(|err| err.to_string())?;
    write_string(&mut out, "schema", spec.schema, true)?;
    write_string(&mut out, "milestone", spec.milestone, true)?;
    write_string(&mut out, "status", spec.status, true)?;
    write_string(&mut out, "id", spec.id, true)?;
    write_string(&mut out, "route_selector", spec.route_selector, true)?;
    write_string(&mut out, "default_route", spec.default_route, true)?;
    write_string(&mut out, "opt_in_route", spec.opt_in_route, true)?;
    write_string(&mut out, "selected_slice_id", spec.selected_slice_id, true)?;
    write_string(&mut out, "operation_family", spec.operation_family, true)?;
    write_string(&mut out, "operation", spec.operation, true)?;
    write_string(&mut out, "method", spec.method, true)?;
    write_string(
        &mut out,
        "replacement_slice_artifact",
        spec.replacement_slice_artifact,
        true,
    )?;
    write_string(
        &mut out,
        "runtime_graph_route",
        spec.runtime_graph_route,
        true,
    )?;
    write_string(&mut out, "graph_backend", spec.graph_backend, true)?;
    write_array(
        &mut out,
        "supported_backends",
        spec.supported_backends,
        true,
    )?;
    write_array(
        &mut out,
        "unsupported_backends",
        spec.unsupported_backends,
        true,
    )?;
    write_array(
        &mut out,
        "validation_artifacts",
        spec.validation_artifacts,
        true,
    )?;
    write_array(&mut out, "quality_gates", spec.quality_gates, true)?;
    write_string(&mut out, "benchmark_policy", spec.benchmark_policy, true)?;
    write_bool(
        &mut out,
        "default_route_unchanged",
        spec.default_route_unchanged,
        true,
    )?;
    write_bool(
        &mut out,
        "replacement_route_opt_in",
        spec.replacement_route_opt_in,
        true,
    )?;
    write_bool(
        &mut out,
        "default_route_replacement_active",
        spec.default_route_replacement_active,
        true,
    )?;
    write_bool(
        &mut out,
        "general_backend_replacement",
        spec.general_backend_replacement,
        true,
    )?;
    write_bool(
        &mut out,
        "kernel_replacement",
        spec.kernel_replacement,
        true,
    )?;
    write_string(
        &mut out,
        "next_required_gate",
        spec.next_required_gate,
        true,
    )?;
    match decision {
        Some(decision) => {
            write_string(&mut out, "route_check", "supported", true)?;
            write_string(&mut out, "checked_route", decision.route.name(), true)?;
            write_string(&mut out, "checked_backend", decision.backend, true)?;
            write_bool(
                &mut out,
                "decision_replacement_active",
                decision.replacement_active,
                false,
            )?;
        }
        None => {
            write_string(&mut out, "route_check", "not-requested", false)?;
        }
    }
    writeln!(out, "\n}}").map_err(|err| err.to_string())
}

fn print_error(
    spec: &RuntimeBackendRouteGateSpec,
    check: &str,
    requested_key: &str,
    requested: &str,
) -> Result<(), String> {
    let mut out = io::BufWriter::new(io::stdout().lock());
    writeln!(out, "{{").map_err(|err| err.to_string())?;
    write_string(
        &mut out,
        "schema",
        "ds4.backend_runtime_route_gate.error.v1",
        true,
    )?;
    write_string(&mut out, "milestone", spec.milestone, true)?;
    write_string(&mut out, "id", spec.id, true)?;
    write_string(&mut out, "route_check", check, true)?;
    write_string(&mut out, requested_key, requested, true)?;
    write_string(&mut out, "route_selector", spec.route_selector, true)?;
    write_string(&mut out, "default_route", spec.default_route, true)?;
    write_string(&mut out, "opt_in_route", spec.opt_in_route, true)?;
    write_array(
        &mut out,
        "supported_backends",
        spec.supported_backends,
        true,
    )?;
    write_array(
        &mut out,
        "unsupported_backends",
        spec.unsupported_backends,
        true,
    )?;
    write_string(
        &mut out,
        "error",
        "unsupported runtime backend route gate",
        false,
    )?;
    writeln!(out, "\n}}").map_err(|err| err.to_string())
}

fn write_string(out: &mut impl Write, key: &str, value: &str, comma: bool) -> Result<(), String> {
    write!(out, "  \"{key}\": \"").map_err(|err| err.to_string())?;
    write_json_string(out, value)?;
    write!(out, "\"").map_err(|err| err.to_string())?;
    if comma {
        write!(out, ",").map_err(|err| err.to_string())?;
    }
    writeln!(out).map_err(|err| err.to_string())
}

fn write_bool(out: &mut impl Write, key: &str, value: bool, comma: bool) -> Result<(), String> {
    write!(out, "  \"{key}\": {value}").map_err(|err| err.to_string())?;
    if comma {
        write!(out, ",").map_err(|err| err.to_string())?;
    }
    writeln!(out).map_err(|err| err.to_string())
}

fn write_array(
    out: &mut impl Write,
    key: &str,
    values: &[&str],
    comma: bool,
) -> Result<(), String> {
    write!(out, "  \"{key}\": [").map_err(|err| err.to_string())?;
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            write!(out, ", ").map_err(|err| err.to_string())?;
        }
        write!(out, "\"").map_err(|err| err.to_string())?;
        write_json_string(out, value)?;
        write!(out, "\"").map_err(|err| err.to_string())?;
    }
    write!(out, "]").map_err(|err| err.to_string())?;
    if comma {
        write!(out, ",").map_err(|err| err.to_string())?;
    }
    writeln!(out).map_err(|err| err.to_string())
}

fn write_json_string(out: &mut impl Write, value: &str) -> Result<(), String> {
    for ch in value.chars() {
        match ch {
            '"' => write!(out, "\\\"").map_err(|err| err.to_string())?,
            '\\' => write!(out, "\\\\").map_err(|err| err.to_string())?,
            '\n' => write!(out, "\\n").map_err(|err| err.to_string())?,
            '\r' => write!(out, "\\r").map_err(|err| err.to_string())?,
            '\t' => write!(out, "\\t").map_err(|err| err.to_string())?,
            _ => write!(out, "{ch}").map_err(|err| err.to_string())?,
        }
    }
    Ok(())
}
