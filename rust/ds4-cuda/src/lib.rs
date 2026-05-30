pub const CUDA_OXIDE_REVISION: &str = "0ab9a13bfd7caf28d241fb5f42f76b90a4d1b200";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HostSubstrateScope {
    pub opt_in_only: bool,
    pub owns_context_and_stream: bool,
    pub owns_device_buffer_roundtrip: bool,
    pub owns_managed_buffer_lifetime: bool,
    pub owns_ds4_kernels: bool,
    pub changes_default_route: bool,
}

pub const M14_1A_SCOPE: HostSubstrateScope = HostSubstrateScope {
    opt_in_only: true,
    owns_context_and_stream: true,
    owns_device_buffer_roundtrip: true,
    owns_managed_buffer_lifetime: true,
    owns_ds4_kernels: false,
    changes_default_route: false,
};

#[cfg(feature = "cuda-oxide-backend")]
pub mod substrate;

#[cfg(test)]
mod tests {
    use super::{CUDA_OXIDE_REVISION, M14_1A_SCOPE};

    #[test]
    fn substrate_scope_does_not_overclaim_kernel_or_route_ownership() {
        assert_eq!(
            CUDA_OXIDE_REVISION,
            "0ab9a13bfd7caf28d241fb5f42f76b90a4d1b200"
        );
        assert!(M14_1A_SCOPE.opt_in_only);
        assert!(M14_1A_SCOPE.owns_context_and_stream);
        assert!(M14_1A_SCOPE.owns_device_buffer_roundtrip);
        assert!(M14_1A_SCOPE.owns_managed_buffer_lifetime);
        assert!(!M14_1A_SCOPE.owns_ds4_kernels);
        assert!(!M14_1A_SCOPE.changes_default_route);
    }
}
