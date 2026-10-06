//! Optional Apple Metal backend for Akita prover compute operations.

mod error;

pub use error::MetalCommitError;

/// Policy for selecting Metal at an operation boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MetalExecutionPolicy {
    /// Reject unavailable or unsupported Metal execution.
    RequireMetal,
    /// Select CPU before execution when Metal is unavailable or not profitable.
    PreferMetal,
}

/// Maximum packed-opening preprocessing retained after commitment.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum OpeningAccelerationPolicy {
    /// Retain every supported opening accelerator.
    #[default]
    Eager,
    /// Retain accelerators only when their combined allocation fits this limit.
    RetainUpToBytes(usize),
}

impl OpeningAccelerationPolicy {
    /// Whether an allocation fits this policy.
    #[must_use]
    pub const fn allows_retention(self, bytes: usize) -> bool {
        match self {
            Self::Eager => true,
            Self::RetainUpToBytes(limit) => bytes <= limit,
        }
    }
}

#[cfg(target_os = "macos")]
mod backend;
#[cfg(target_os = "macos")]
mod coefficient_packing;
#[cfg(target_os = "macos")]
mod field;
#[cfg(target_os = "macos")]
mod onehot;
#[cfg(target_os = "macos")]
mod packed_onehot;
#[cfg(target_os = "macos")]
mod packed_onehot_fp128_d128_rank3;
#[cfg(target_os = "macos")]
mod packed_onehot_fp128_d512;
#[cfg(target_os = "macos")]
mod prepared;
#[cfg(target_os = "macos")]
mod recursive_fold;

#[cfg(target_os = "macos")]
mod ring_switch;
#[cfg(target_os = "macos")]
#[expect(
    dead_code,
    clippy::too_many_arguments,
    reason = "runtime retains unported operation routes; dispatch methods bind each kernel ABI"
)]
mod runtime;

#[cfg(target_os = "macos")]
pub use backend::{CommandProgress, MetalBackend, MetalCommitMetrics, MetalOpeningMetrics};
#[cfg(target_os = "macos")]
pub use packed_onehot::PackedOneHotCommitView;
#[cfg(target_os = "macos")]
pub use prepared::MetalPreparedSetup;
#[cfg(target_os = "macos")]
pub use runtime::{MetalDeviceCapabilities, MetalOneHotKernel};

#[cfg(not(target_os = "macos"))]
mod unsupported {
    use super::{MetalCommitError, MetalExecutionPolicy};

    /// Unavailable Metal backend placeholder on non-macOS targets.
    #[derive(Clone, Copy, Debug)]
    pub struct MetalBackend;

    impl MetalBackend {
        /// Report that this target has no Metal runtime.
        pub const fn is_available() -> bool {
            false
        }

        /// Constructing the backend on a non-macOS target always fails.
        pub fn new(_policy: MetalExecutionPolicy) -> Result<Self, MetalCommitError> {
            Err(MetalCommitError::UnsupportedPlatform)
        }
    }
}

#[cfg(not(target_os = "macos"))]
pub use unsupported::MetalBackend;
