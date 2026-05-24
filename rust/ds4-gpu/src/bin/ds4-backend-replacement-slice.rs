use ds4_gpu::replacement_slice::{
    ensure_supported_backend, first_backend_replacement_slice, replacement_slice_by_id,
    ReplacementSliceSpec,
};
use std::env;
use std::io::{self, Write};

fn main() {
    if let Err(err) = run() {
        let _ = writeln!(io::stderr(), "ds4-backend-replacement-slice: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    let (spec, backend) = parse_args(&args)?;
    let spec = spec.unwrap_or_else(first_backend_replacement_slice);
    match backend {
        Some(backend) => match ensure_supported_backend(spec, backend) {
            Ok(()) => {
                print_summary(spec, Some(("supported", backend)))?;
                Ok(())
            }
            Err(err) => {
                print_unsupported(spec, err.requested)?;
                std::process::exit(2);
            }
        },
        None => {
            print_summary(spec, None)?;
            Ok(())
        }
    }
}

fn parse_args<'a>(
    args: &'a [String],
) -> Result<(Option<&'static ReplacementSliceSpec>, Option<&'a str>), String> {
    let mut spec = None;
    let mut backend = None;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--backend" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| "missing value for --backend".to_string())?;
                backend = Some(value.as_str());
            }
            "--slice" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| "missing value for --slice".to_string())?;
                spec = Some(
                    replacement_slice_by_id(value)
                        .ok_or_else(|| format!("unknown replacement slice: {value}"))?,
                );
            }
            _ => return Err(usage()),
        }
        index += 1;
    }
    Ok((spec, backend))
}

fn usage() -> String {
    "usage: ds4-backend-replacement-slice [--slice ID] [--backend NAME]".to_string()
}

fn print_summary(
    spec: &ReplacementSliceSpec,
    backend_check: Option<(&str, &str)>,
) -> Result<(), String> {
    let mut out = io::BufWriter::new(io::stdout().lock());
    writeln!(out, "{{").map_err(|err| err.to_string())?;
    write_string(&mut out, "schema", spec.schema, true)?;
    write_string(&mut out, "milestone", spec.milestone, true)?;
    write_string(&mut out, "status", spec.status, true)?;
    write_string(&mut out, "id", spec.id, true)?;
    write_string(&mut out, "operation_family", spec.operation_family, true)?;
    write_string(&mut out, "fixture_id", spec.fixture_id, true)?;
    write_string(&mut out, "operation", spec.operation, true)?;
    write_string(&mut out, "method", spec.method, true)?;
    write_string(&mut out, "rust_module", spec.rust_module, true)?;
    write_string(&mut out, "facade_replay", spec.facade_replay, true)?;
    write_string(
        &mut out,
        "tensor_fixture_manifest",
        spec.tensor_fixture_manifest,
        true,
    )?;
    write_string(&mut out, "comparator", spec.comparator, true)?;
    write_array(&mut out, "output_fields", spec.output_fields, true)?;
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
    write_bool(
        &mut out,
        "runtime_route_change",
        spec.runtime_route_change,
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
    match backend_check {
        Some((status, backend)) => {
            write_string(&mut out, "backend_check", status, true)?;
            write_string(&mut out, "checked_backend", backend, false)?;
        }
        None => {
            write_string(&mut out, "backend_check", "not-requested", false)?;
        }
    }
    writeln!(out, "\n}}").map_err(|err| err.to_string())
}

fn print_unsupported(spec: &ReplacementSliceSpec, backend: &str) -> Result<(), String> {
    let mut out = io::BufWriter::new(io::stdout().lock());
    writeln!(out, "{{").map_err(|err| err.to_string())?;
    write_string(
        &mut out,
        "schema",
        "ds4.backend_replacement_slice.error.v1",
        true,
    )?;
    write_string(&mut out, "milestone", spec.milestone, true)?;
    write_string(&mut out, "id", spec.id, true)?;
    write_string(&mut out, "backend_check", "unsupported", true)?;
    write_string(&mut out, "requested_backend", backend, true)?;
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
    write_string(&mut out, "error", "unsupported replacement backend", false)?;
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
