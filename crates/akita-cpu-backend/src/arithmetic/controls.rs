//! Application-facing cache controls and public resource diagnostics.
use super::{CpuBackend, PreparedCrtNttProfile, PreparedNttCacheMetric};
use crate::opaque::NttExecutionRequirements;
use akita_error::AkitaError;
use akita_params::FoldSchedule;

impl<F: jolt_field::Field + jolt_field::CanonicalEncoding, E> CpuBackend<F, E> {
    /// Prepare the CPU cache union for a complete commit-and-prove workload.
    pub fn prewarm(&self, schedule: &FoldSchedule) -> Result<(), AkitaError> {
        let prepared = self.prepared()?;
        let planned = NttExecutionRequirements::from_commit_and_prove_schedule(schedule)?;
        super::requirements::warm_joined_ntt_requirements(self, prepared, &planned)
    }

    /// Prepare the relation, recursive-witness commitment, opening and
    /// terminal transforms of one `batched_prove` over `schedule`. Root and
    /// setup-prefix commitments are excluded: the root is committed before
    /// proving and the prefixes are imported by `batched_prove` itself.
    ///
    /// With `parallel`, slot builds run Rayon work in the calling thread's
    /// registry, which is the global registry for a non-Rayon thread. A caller
    /// that overlaps this with other users of the same backend cache must call
    /// it from a non-Rayon thread and run every such consumer in a distinct
    /// dedicated registry, never in the global registry servicing the build: a
    /// worker waiting inside a build can execute a job that waits on the same
    /// pending slot. This is a caller contract; nothing here enforces it.
    pub fn prewarm_batched_prove(&self, schedule: &FoldSchedule) -> Result<(), AkitaError> {
        let prepared = self.prepared()?;
        let planned = NttExecutionRequirements::from_prove_schedule(schedule, false)?;
        super::requirements::warm_joined_ntt_requirements(self, prepared, &planned)
    }

    /// Initialized setup transforms for application profiling.
    pub fn shared_ntt_cache_metrics(&self) -> Result<Vec<PreparedNttCacheMetric>, AkitaError> {
        self.prepared()?.shared_ntt_cache_metrics()
    }

    /// Arithmetic capacity profile for an initialized ring degree.
    pub fn shared_ntt_profile(&self, ring_d: usize) -> Result<PreparedCrtNttProfile, AkitaError> {
        self.prepared()?.shared_ntt_profile(ring_d)
    }
}
