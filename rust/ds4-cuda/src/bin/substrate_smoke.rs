use ds4_cuda::{substrate::CudaOxideSubstrate, M14_1A_SCOPE};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let substrate = CudaOxideSubstrate::open(0)?;
    let input = [1.25_f32, -2.5, 3.75, 0.0];

    let device = substrate.upload(&input)?;
    let output = substrate.download(&device)?;
    assert_eq!(output, input);

    let zeros = substrate.zeroed::<u32>(4)?;
    let zero_output = substrate.download(&zeros)?;
    assert_eq!(zero_output, [0_u32; 4]);

    let mut managed = substrate.managed_zeroed::<f32>(input.len())?;
    managed.as_mut_slice().copy_from_slice(&input);
    substrate.synchronize()?;
    assert_eq!(managed.as_slice(), input);

    let device_name = substrate.device_name()?;
    println!(
        "{{\"milestone\":\"M14.1a\",\"cuda_oxide_substrate\":true,\"device_ordinal\":{},\"device_name\":{:?},\"device_roundtrip\":true,\"zeroed_roundtrip\":true,\"managed_lifetime\":true,\"owns_ds4_kernels\":{},\"changes_default_route\":{}}}",
        substrate.device_ordinal(),
        device_name,
        M14_1A_SCOPE.owns_ds4_kernels,
        M14_1A_SCOPE.changes_default_route
    );
    Ok(())
}
