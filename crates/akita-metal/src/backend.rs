use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::split_eq::GruenSplitEq;
use akita_algebra::CyclotomicRing;
use akita_error::AkitaError;
use akita_prover::backend::{
    DenseBatchView, DenseView, MultilinearPolynomialBatchView, MultilinearPolynomialView,
    OneHotBatchView, OneHotView,
};
use akita_prover::compute::{
    CompressionComputeBackend, CompressionRowsProducts, DecomposeFoldBatchPlan, DecomposeFoldPlan,
    OpeningBatchKernel, OpeningFoldKernel, OpeningFoldOutput, OpeningFoldPlan,
    SubringCoefficientPackingBatchKernel, SubringCoefficientPackingPartials,
    SubringCoefficientPackingPlan,
};
use akita_prover::{
    BatchDecomposeFoldOutcome, ComputeBackendSetup, CpuBackend, CyclicRowsComputeBackend,
    DecomposeFoldWitness, DigitRowsComputeBackend, DirectDigitRangeProofBackend,
    DirectDigitRangeProofInput, DirectRelationRangePreparationInput,
    DirectRelationRangeProofBackend, DirectRelationRangeProofOutput, DirectRelationRangeProofState,
    NttCacheOwnerId, NttOperationCluster, OneHotIndex, RecursiveFoldBatchView,
    RelationRangeImageProver, RoutedNttRequirement, SuffixWitnessBatchView, SuffixWitnessView,
};
use akita_sumcheck::{
    CompressedUniPoly, EqFactoredSumcheckProof, EqFactoredUniPoly, SumcheckProof,
};
use akita_transcript::labels;
use akita_types::{AkitaExpandedSetup, AkitaStage1Proof, AkitaStage1StageProof, NttCacheKey};
use jolt_field::{ExtField, Zero};
use metal::Device;

use crate::field::F;
use crate::prepared::MetalPreparedSetup;
use crate::runtime::{
    DirectRangeRoundOutcome, DirectRelationAdditionalPair, DirectRelationLinearSegment,
    DirectRelationLinearSourceInput, DirectRelationRoundData, DirectRelationRoundOutcome,
    DirectRelationScalars, MetalDeviceCapabilities, MetalOneHotKernel, MetalRuntime,
};
use crate::{MetalCommitError, MetalExecutionPolicy};

/// Timings and structural counters from the most recent one-hot commitment.
#[derive(Clone, Debug, Default)]
pub struct MetalCommitMetrics {
    /// Kernel variant selected for the dispatch.
    pub kernel: MetalOneHotKernel,
    /// Independent source blocks processed by one threadgroup.
    pub blocks_per_threadgroup: usize,
    /// Number of committed sources.
    pub num_sources: usize,
    /// Number of nonzero one-hot chunks across all sources.
    pub hot_entries: usize,
    /// Number of field additions represented by the sparse operation.
    pub field_additions: u64,
    /// Conservative direct-gather A-matrix traffic.
    pub gathered_matrix_bytes: u64,
    /// Canonical output bytes returned by the dispatch.
    pub output_bytes: usize,
    /// Device-only transient bytes allocated by the dispatch.
    pub scratch_bytes: usize,
    /// Resident A-prefix bytes.
    pub matrix_bytes: usize,
    /// Whether the A-prefix was already resident.
    pub matrix_cache_hit: bool,
    /// A-prefix packing and allocation time on a miss.
    pub matrix_prepare_time: Duration,
    /// Host time spent packing sparse indices.
    pub index_pack_time: Duration,
    /// Per-call Metal input/output buffer setup.
    pub buffer_setup_time: Duration,
    /// Wall time from command commit through completion.
    pub command_wall_time: Duration,
    /// GPU timestamp interval, when reported by the device.
    pub gpu_time: Option<Duration>,
    /// Sum of panel-command GPU intervals, excluding the final reduction.
    pub panel_gpu_active_time: Option<Duration>,
    /// GPU interval from the first panel command through the last panel command.
    pub panel_gpu_span: Option<Duration>,
    /// Final partial-reduction GPU interval.
    pub reduction_gpu_time: Option<Duration>,
    /// Metal command buffers submitted by the commitment.
    pub command_buffers: usize,
    /// Matrix panel streams represented by the dispatch.
    pub matrix_block_streams: usize,
    /// Copy from shared output storage into owned Rust limbs.
    pub readback_copy_time: Duration,
    /// Canonical field reconstruction and witness assembly.
    pub output_reconstruction_time: Duration,
    /// Complete kernel-boundary wall time.
    pub total_time: Duration,
}

/// Timings and routing counters from the most recent opening proof.
#[derive(Clone, Debug, Default)]
pub struct MetalOpeningMetrics {
    /// Wall time from Metal command submission through completion.
    pub command_wall_time: Duration,
    /// Sum of GPU timestamp intervals reported by Metal.
    pub gpu_active_time: Duration,
    /// Host-side input and output buffer construction time.
    pub buffer_setup_time: Duration,
    /// Host-side copy and canonical field reconstruction time.
    pub readback_time: Duration,
    /// Transient bytes requested by opening dispatches.
    pub allocation_bytes: usize,
    /// Opening operations deliberately delegated to the CPU backend.
    pub cpu_fallback_calls: usize,
    /// Scalar source work represented by delegated operations.
    pub cpu_fallback_work_units: usize,
}

struct PreparedMetalDirectRelation {
    session: crate::runtime::DirectRelationSession,
    setup_timings: crate::runtime::DispatchTimings,
    domain_len: usize,
    coefficient_rounds: usize,
}

#[doc(hidden)]
pub struct MetalDirectRelationPreparation {
    prepared: Option<Box<PreparedMetalDirectRelation>>,
}

struct BackendInner {
    runtime: Option<Arc<MetalRuntime>>,
    policy: MetalExecutionPolicy,
    cpu: CpuBackend,
    initialization_time: Duration,
    last_commit_metrics: Mutex<Option<MetalCommitMetrics>>,
    last_opening_metrics: Mutex<Option<MetalOpeningMetrics>>,
}

/// Akita compute backend with explicit Metal admission and CPU delegation.
#[derive(Clone)]
pub struct MetalBackend {
    inner: Arc<BackendInner>,
}

impl MetalBackend {
    /// Whether a system Metal device is visible.
    pub fn is_available() -> bool {
        Device::system_default().is_some()
    }

    /// Build a backend from the system default device.
    pub fn new(policy: MetalExecutionPolicy) -> Result<Self, MetalCommitError> {
        Self::new_with_cpu_backend(policy, CpuBackend::DEFAULT)
    }

    /// Build a backend with explicit CPU resource limits for delegated work.
    pub fn new_with_cpu_backend(
        policy: MetalExecutionPolicy,
        cpu: CpuBackend,
    ) -> Result<Self, MetalCommitError> {
        let start = Instant::now();
        let runtime = match MetalRuntime::new() {
            Ok(runtime) => Some(Arc::new(runtime)),
            Err(_error) if policy == MetalExecutionPolicy::PreferMetal => None,
            Err(error) => return Err(error),
        };
        Ok(Self {
            inner: Arc::new(BackendInner {
                runtime,
                policy,
                cpu,
                initialization_time: start.elapsed(),
                last_commit_metrics: Mutex::new(None),
                last_opening_metrics: Mutex::new(None),
            }),
        })
    }

    /// Build a backend around an existing Metal device.
    pub fn with_device(
        device: Device,
        policy: MetalExecutionPolicy,
    ) -> Result<Self, MetalCommitError> {
        Self::with_device_and_cpu_backend(device, policy, CpuBackend::DEFAULT)
    }

    /// Build around an existing device with explicit CPU resource limits.
    pub fn with_device_and_cpu_backend(
        device: Device,
        policy: MetalExecutionPolicy,
        cpu: CpuBackend,
    ) -> Result<Self, MetalCommitError> {
        let start = Instant::now();
        let runtime = Arc::new(MetalRuntime::from_device(device)?);
        Ok(Self {
            inner: Arc::new(BackendInner {
                runtime: Some(runtime),
                policy,
                cpu,
                initialization_time: start.elapsed(),
                last_commit_metrics: Mutex::new(None),
                last_opening_metrics: Mutex::new(None),
            }),
        })
    }

    /// Runtime compilation plus pipeline creation time.
    pub fn initialization_time(&self) -> Duration {
        self.inner.initialization_time
    }

    /// Selected execution policy.
    pub fn policy(&self) -> MetalExecutionPolicy {
        self.inner.policy
    }

    /// CPU backend used for delegated setup and operations.
    pub fn cpu_backend(&self) -> CpuBackend {
        self.inner.cpu
    }

    /// Stable device properties, or `None` when preferred Metal is unavailable.
    pub fn capabilities(&self) -> Option<MetalDeviceCapabilities> {
        self.runtime().map(MetalRuntime::capabilities)
    }

    /// Metrics for the most recent successful Metal commitment dispatch.
    pub fn last_commit_metrics(&self) -> Result<Option<MetalCommitMetrics>, MetalCommitError> {
        Ok(self
            .inner
            .last_commit_metrics
            .lock()
            .map_err(|_| MetalCommitError::PoisonedLock)?
            .clone())
    }

    /// Reset opening metrics immediately before a proof begins.
    pub fn begin_opening_metrics(&self) -> Result<(), MetalCommitError> {
        *self
            .inner
            .last_opening_metrics
            .lock()
            .map_err(|_| MetalCommitError::PoisonedLock)? = Some(MetalOpeningMetrics::default());
        Ok(())
    }

    /// Metrics accumulated since the most recent opening reset.
    pub fn last_opening_metrics(&self) -> Result<Option<MetalOpeningMetrics>, MetalCommitError> {
        Ok(self
            .inner
            .last_opening_metrics
            .lock()
            .map_err(|_| MetalCommitError::PoisonedLock)?
            .clone())
    }

    pub(crate) fn update_opening_metrics(
        &self,
        update: impl FnOnce(&mut MetalOpeningMetrics),
    ) -> Result<(), MetalCommitError> {
        let mut metrics = self
            .inner
            .last_opening_metrics
            .lock()
            .map_err(|_| MetalCommitError::PoisonedLock)?;
        if let Some(metrics) = metrics.as_mut() {
            update(metrics);
        }
        Ok(())
    }

    /// Record a deliberate CPU route inside a Metal opening stack.
    pub fn record_opening_cpu_fallback(&self, work_units: usize) -> Result<(), MetalCommitError> {
        self.update_opening_metrics(|metrics| {
            metrics.cpu_fallback_calls = metrics.cpu_fallback_calls.saturating_add(1);
            metrics.cpu_fallback_work_units =
                metrics.cpu_fallback_work_units.saturating_add(work_units);
        })
    }

    pub(crate) fn runtime(&self) -> Option<&MetalRuntime> {
        self.inner.runtime.as_deref()
    }

    pub(crate) fn record_commit_metrics(
        &self,
        metrics: MetalCommitMetrics,
    ) -> Result<(), MetalCommitError> {
        *self
            .inner
            .last_commit_metrics
            .lock()
            .map_err(|_| MetalCommitError::PoisonedLock)? = Some(metrics);
        Ok(())
    }
}

fn direct_range_eq_tables(
    split_eq: &GruenSplitEq<F>,
) -> (Vec<crate::field::Fp128Limbs>, Vec<crate::field::Fp128Limbs>) {
    let (first, second) = split_eq.remaining_eq_tables();
    (
        first
            .iter()
            .copied()
            .map(crate::field::Fp128Limbs::from_field)
            .collect(),
        second
            .iter()
            .copied()
            .map(crate::field::Fp128Limbs::from_field)
            .collect(),
    )
}

fn direct_range_round_poly(
    outcome: &DirectRangeRoundOutcome,
    basis: usize,
) -> Result<EqFactoredUniPoly<F>, AkitaError> {
    let stored_coefficients = match basis {
        4 => 2,
        8 => 4,
        _ => {
            return Err(MetalCommitError::UnsupportedShape(
                "direct range Metal proof supports basis four or eight".into(),
            )
            .into_akita())
        }
    };
    // The kernel accumulates `[q_1, ..., q_d]`; q_0 is implied by the running claim.
    let coeffs_except_constant_term = outcome.coefficients[..stored_coefficients]
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| {
            value
                .into_field(index)
                .map_err(MetalCommitError::into_akita)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(EqFactoredUniPoly {
        coeffs_except_constant_term,
    })
}

struct DirectRelationHostRound {
    e_first: Vec<crate::field::Fp128Limbs>,
    e_second: Vec<crate::field::Fp128Limbs>,
    alpha: Vec<crate::field::Fp128Limbs>,
    additional_pairs: Vec<DirectRelationAdditionalPair>,
    scalars: DirectRelationScalars,
    live_lane_count: usize,
}

impl DirectRelationHostRound {
    fn as_runtime(&self) -> DirectRelationRoundData<'_> {
        DirectRelationRoundData {
            e_first: &self.e_first,
            e_second: &self.e_second,
            alpha: &self.alpha,
            additional_pairs: &self.additional_pairs,
            scalars: self.scalars,
            live_lane_count: self.live_lane_count,
        }
    }
}

#[tracing::instrument(skip_all, name = "direct_relation_host_round")]
fn direct_relation_host_round(
    state: &DirectRelationRangeProofState<F>,
) -> Result<DirectRelationHostRound, AkitaError> {
    let to_limbs = |values: &[F]| {
        values
            .iter()
            .copied()
            .map(crate::field::Fp128Limbs::from_field)
            .collect::<Vec<_>>()
    };
    let (e_first, e_second) = state.remaining_eq_tables();
    let additional = state.additional_round();
    let additional_pairs = additional
        .pairs
        .into_iter()
        .map(|pair| {
            Ok(DirectRelationAdditionalPair {
                parent: u64::try_from(pair.parent).map_err(|_| {
                    AkitaError::InvalidSetup(
                        "direct relation sparse parent does not fit the Metal ABI".into(),
                    )
                })?,
                reserved: 0,
                linear: pair.linear.map(crate::field::Fp128Limbs::from_field),
                binary: pair.binary.map(crate::field::Fp128Limbs::from_field),
            })
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;
    let (l_at_0, l_at_1) = state.current_linear_factor_evals();
    Ok(DirectRelationHostRound {
        e_first: to_limbs(e_first),
        e_second: to_limbs(e_second),
        alpha: to_limbs(state.common_alpha_factor()?),
        additional_pairs,
        scalars: DirectRelationScalars {
            l_at_0: crate::field::Fp128Limbs::from_field(l_at_0),
            l_at_1: crate::field::Fp128Limbs::from_field(l_at_1),
            binary_batching: crate::field::Fp128Limbs::from_field(additional.binary_batching),
        },
        live_lane_count: state.current_live_lane_count(),
    })
}

fn direct_relation_round_poly(
    outcome: &DirectRelationRoundOutcome,
) -> Result<CompressedUniPoly<F>, AkitaError> {
    let mut coeffs_except_linear_term = (0..3)
        .map(|index| {
            let main = outcome.coefficients[index]
                .into_field(index)
                .map_err(MetalCommitError::into_akita)?;
            let additional = outcome.additional_coefficients[index]
                .into_field(index + 3)
                .map_err(MetalCommitError::into_akita)?;
            Ok(main + additional)
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;
    while coeffs_except_linear_term.len() > 1
        && coeffs_except_linear_term.last().is_some_and(Zero::is_zero)
    {
        coeffs_except_linear_term.pop();
    }
    Ok(CompressedUniPoly {
        coeffs_except_linear_term,
    })
}

fn direct_relation_prefix_round_poly(
    main_coefficients: [F; 3],
    additional_coefficients: [crate::field::Fp128Limbs; 4],
) -> Result<CompressedUniPoly<F>, AkitaError> {
    let mut coeffs_except_linear_term = main_coefficients
        .into_iter()
        .zip(additional_coefficients)
        .enumerate()
        .map(|(index, (main, additional))| {
            additional
                .into_field(index)
                .map(|value| main + value)
                .map_err(MetalCommitError::into_akita)
        })
        .collect::<Result<Vec<_>, _>>()?;
    while coeffs_except_linear_term.len() > 1
        && coeffs_except_linear_term.last().is_some_and(Zero::is_zero)
    {
        coeffs_except_linear_term.pop();
    }
    Ok(CompressedUniPoly {
        coeffs_except_linear_term,
    })
}

fn direct_relation_prefix_fields<const N: usize>(
    limbs: [crate::field::Fp128Limbs; N],
) -> Result<[F; N], AkitaError> {
    limbs
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            value
                .into_field(index)
                .map_err(MetalCommitError::into_akita)
        })
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| AkitaError::InvalidProof)
}

fn direct_relation_prefix_point(values: [F; 4], point: usize) -> F {
    let [value_00, value_10, value_01, value_11] = values;
    match point {
        0 => value_00,
        1 => value_01,
        2 => value_01 - value_00,
        3 => value_10,
        4 => value_11,
        5 => value_11 - value_10,
        6 => value_10 - value_00,
        7 => value_11 - value_01,
        _ => value_11 - value_10 - value_01 + value_00,
    }
}

struct DirectRelationTwoRoundPrefixHost {
    equality_first: Vec<crate::field::Fp128Limbs>,
    equality_second: Vec<crate::field::Fp128Limbs>,
    alpha_points: Vec<crate::field::Fp128Limbs>,
    norm_omitted_corner: usize,
}

#[tracing::instrument(skip_all, name = "direct_relation_two_round_prefix_host")]
fn direct_relation_two_round_prefix_host(
    state: &DirectRelationRangeProofState<F>,
) -> Result<Option<DirectRelationTwoRoundPrefixHost>, AkitaError> {
    let Some(data) = state.two_round_prefix_data()? else {
        return Ok(None);
    };
    if !matches!(data.basis, 4 | 8)
        || data.coefficient_count < 4
        || !data.coefficient_count.is_power_of_two()
        || data.alpha.len() != data.coefficient_count
        || data.lane_weights.is_empty()
        || data.live_lane_count > data.lane_weights.len()
    {
        return Err(AkitaError::InvalidProof);
    }
    let y_quads = data.coefficient_count / 4;
    let mut alpha_points = Vec::with_capacity(8 * y_quads);
    for point in 1..9 {
        for quad in data.alpha.chunks_exact(4) {
            let values: [F; 4] = quad.try_into().map_err(|_| AkitaError::InvalidProof)?;
            alpha_points.push(crate::field::Fp128Limbs::from_field(
                direct_relation_prefix_point(values, point),
            ));
        }
    }
    Ok(Some(DirectRelationTwoRoundPrefixHost {
        equality_first: data
            .equality_first
            .into_iter()
            .map(crate::field::Fp128Limbs::from_field)
            .collect(),
        equality_second: data
            .equality_second
            .into_iter()
            .map(crate::field::Fp128Limbs::from_field)
            .collect(),
        alpha_points,
        norm_omitted_corner: data.norm_omitted_corner,
    }))
}

impl MetalBackend {
    fn record_direct_range_dispatch(
        &self,
        timings: crate::runtime::DispatchTimings,
        allocation_bytes: usize,
    ) -> Result<(), AkitaError> {
        self.update_opening_metrics(|metrics| {
            metrics.command_wall_time += timings.command_wall;
            metrics.gpu_active_time += timings.gpu.unwrap_or_default();
            metrics.buffer_setup_time += timings.buffer_setup;
            metrics.readback_time += timings.readback_copy;
            metrics.allocation_bytes = metrics.allocation_bytes.saturating_add(allocation_bytes);
        })
        .map_err(MetalCommitError::into_akita)
    }

    fn direct_range_cpu_fallback<T>(
        &self,
        prepared: &MetalPreparedSetup,
        input: DirectDigitRangeProofInput<F>,
        transcript: &mut T,
        level: u32,
    ) -> Result<(AkitaStage1Proof<F>, Vec<F>), AkitaError>
    where
        T: akita_types::ProverTranscriptGrinding<F>,
    {
        self.record_opening_cpu_fallback(input.digit_witness_len())
            .map_err(MetalCommitError::into_akita)?;
        <CpuBackend as DirectDigitRangeProofBackend<F, F>>::prove_direct_digit_range(
            &self.cpu_backend(),
            &prepared.cpu,
            input,
            transcript,
            level,
        )
    }
}

impl DirectDigitRangeProofBackend<F, F> for MetalBackend {
    fn prove_direct_digit_range<T>(
        &self,
        prepared: &Self::PreparedSetup,
        input: DirectDigitRangeProofInput<F>,
        transcript: &mut T,
        level: u32,
    ) -> Result<(AkitaStage1Proof<F>, Vec<F>), AkitaError>
    where
        T: akita_types::ProverTranscriptGrinding<F>,
    {
        let num_vars = input
            .column_variable_count()
            .checked_add(input.ring_variable_count())
            .ok_or_else(|| {
                AkitaError::InvalidSetup("direct range variable width overflow".into())
            })?;
        let domain_len = 1usize
            .checked_shl(u32::try_from(num_vars).map_err(|_| {
                AkitaError::InvalidSetup("direct range variable width overflow".into())
            })?)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("direct range domain length overflow".into())
            })?;
        let ring_len = 1usize
            .checked_shl(
                u32::try_from(input.ring_variable_count()).map_err(|_| {
                    AkitaError::InvalidSetup("direct range ring width overflow".into())
                })?,
            )
            .ok_or_else(|| AkitaError::InvalidSetup("direct range ring length overflow".into()))?;
        let expected_live_len = input
            .live_column_count()
            .checked_mul(ring_len)
            .ok_or_else(|| AkitaError::InvalidSetup("direct range live length overflow".into()))?;
        if input.equality_point().len() != num_vars
            || input.digit_witness_len() != expected_live_len
        {
            return Err(AkitaError::InvalidSize {
                expected: domain_len,
                actual: input.digit_witness_len(),
            });
        }
        let _direct_span = tracing::info_span!(
            "direct_digit_range_metal",
            domain_len,
            live_len = input.digit_witness_len(),
            num_vars,
            basis = input.plan().basis(),
        )
        .entered();
        let runtime = match self.runtime() {
            Some(runtime) => runtime,
            None if self.policy() == MetalExecutionPolicy::PreferMetal => {
                return self.direct_range_cpu_fallback(prepared, input, transcript, level)
            }
            None => return Err(MetalCommitError::DeviceUnavailable.into_akita()),
        };
        let decoded_witness = {
            let _span = tracing::info_span!("direct_range_decode_packed_digits").entered();
            input.decode_digit_witness()
        };
        const COMPACT_PREFIX_ROUNDS: usize = 3;
        let (mut session, setup_time) = {
            let _span = tracing::info_span!("direct_range_session_setup").entered();
            match runtime.begin_fp128_direct_range(
                &decoded_witness,
                domain_len,
                COMPACT_PREFIX_ROUNDS,
            ) {
                Ok(session) => session,
                Err(_error) if self.policy() == MetalExecutionPolicy::PreferMetal => {
                    return self.direct_range_cpu_fallback(prepared, input, transcript, level)
                }
                Err(error) => return Err(error.into_akita()),
            }
        };
        self.update_opening_metrics(|metrics| {
            metrics.buffer_setup_time += setup_time;
            metrics.allocation_bytes = metrics
                .allocation_bytes
                .saturating_add(runtime.direct_range_session_allocation_bytes(&session));
        })
        .map_err(MetalCommitError::into_akita)?;

        let mut split_eq = GruenSplitEq::new(input.equality_point())?;
        let (first_eq, second_eq) = direct_range_eq_tables(&split_eq);
        let initial = {
            let _span = tracing::info_span!("direct_range_initial").entered();
            runtime.dispatch_fp128_direct_range_initial(
                &session,
                &first_eq,
                &second_eq,
                input.plan().basis(),
            )
        }
        .map_err(MetalCommitError::into_akita)?;
        self.record_direct_range_dispatch(initial.timings, initial.allocation_bytes)?;

        let mut next_poly = direct_range_round_poly(&initial, input.plan().basis())?;
        let mut round_polys = Vec::with_capacity(num_vars);
        let mut challenges = Vec::with_capacity(num_vars);
        let mut final_evaluation = None;
        transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &F::zero());
        for round in 0..num_vars {
            transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, &next_poly);
            let challenge = akita_types::sample_grinded_sumcheck_challenge::<F, F, T>(
                transcript,
                akita_types::SumcheckProtocol::Stage1,
                level,
                0,
                u32::try_from(round).map_err(|_| {
                    AkitaError::InvalidSetup("Stage 1 sumcheck round overflow".into())
                })?,
            )?;
            challenges.push(challenge);
            round_polys.push(next_poly.clone());
            split_eq.bind(challenge);
            let prefix_weights = if challenges.len() <= COMPACT_PREFIX_ROUNDS {
                EqPolynomial::evals(&challenges)?
                    .into_iter()
                    .map(crate::field::Fp128Limbs::from_field)
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            };
            let next_eq_storage = if round + 1 < num_vars {
                let (first, second) = direct_range_eq_tables(&split_eq);
                Some((first, second))
            } else {
                None
            };
            let next_eq = next_eq_storage
                .as_ref()
                .map(|(first, second)| (first.as_slice(), second.as_slice()));
            let advance = {
                let _span = tracing::info_span!("direct_range_advance", round).entered();
                runtime.dispatch_fp128_direct_range_advance(
                    &mut session,
                    crate::field::Fp128Limbs::from_field(challenge),
                    next_eq,
                    &prefix_weights,
                    input.plan().basis(),
                )
            }
            .map_err(MetalCommitError::into_akita)?;
            self.record_direct_range_dispatch(advance.timings, advance.allocation_bytes)?;
            if let Some(coefficients) = advance.next_coefficients {
                next_poly = direct_range_round_poly(
                    &DirectRangeRoundOutcome {
                        coefficients,
                        timings: Default::default(),
                        allocation_bytes: 0,
                    },
                    input.plan().basis(),
                )?;
            }
            if let Some(evaluation) = advance.final_evaluation {
                final_evaluation = Some(
                    evaluation
                        .into_field(0)
                        .map_err(MetalCommitError::into_akita)?,
                );
            }
        }
        let range_image_evaluation = final_evaluation.ok_or(AkitaError::InvalidProof)?;
        Ok((
            AkitaStage1Proof {
                stages: vec![AkitaStage1StageProof {
                    sumcheck_proof: EqFactoredSumcheckProof { round_polys },
                    child_claims: Vec::new(),
                }],
                range_image_evaluation,
                norm_proof: None,
            },
            challenges,
        ))
    }
}

impl MetalBackend {
    fn record_direct_relation_dispatch(
        &self,
        timings: crate::runtime::DispatchTimings,
        allocation_bytes: usize,
    ) -> Result<(), AkitaError> {
        self.record_direct_range_dispatch(timings, allocation_bytes)
    }

    fn record_direct_relation_source_dispatch(
        &self,
        timings: crate::runtime::DispatchTimings,
    ) -> Result<(), AkitaError> {
        self.record_direct_range_dispatch(timings, 0)
    }

    fn direct_relation_cpu_fallback<T>(
        &self,
        prepared: &MetalPreparedSetup,
        prover: RelationRangeImageProver<F>,
        transcript: &mut T,
        level: u32,
    ) -> Result<DirectRelationRangeProofOutput<F>, AkitaError>
    where
        T: akita_types::ProverTranscriptGrinding<F>,
    {
        self.record_opening_cpu_fallback(1)
            .map_err(MetalCommitError::into_akita)?;
        <CpuBackend as DirectRelationRangeProofBackend<F, F>>::prove_direct_relation_range(
            &self.cpu_backend(),
            &prepared.cpu,
            prover,
            (),
            transcript,
            level,
        )
    }
}

impl DirectRelationRangeProofBackend<F, F> for MetalBackend {
    type Preparation = MetalDirectRelationPreparation;

    fn should_overlap_direct_relation_preparation(&self) -> bool {
        true
    }

    fn prepare_direct_relation_range(
        &self,
        _prepared: &Self::PreparedSetup,
        input: DirectRelationRangePreparationInput<'_, F>,
    ) -> Result<Self::Preparation, AkitaError> {
        let _prepare_span = tracing::info_span!("direct_relation_static_prepare").entered();
        const COMPACT_PREFIX_ROUNDS: usize = 3;
        if input.coefficient_rounds() < COMPACT_PREFIX_ROUNDS
            || input.domain_len() < 16
            || !input.domain_len().is_power_of_two()
        {
            if self.policy() == MetalExecutionPolicy::PreferMetal {
                return Ok(MetalDirectRelationPreparation { prepared: None });
            }
            return Err(MetalCommitError::UnsupportedShape(
                "direct relation Metal proof requires at least three coefficient rounds".into(),
            )
            .into_akita());
        }
        let runtime = match self.runtime() {
            Some(runtime) => runtime,
            None if self.policy() == MetalExecutionPolicy::PreferMetal => {
                return Ok(MetalDirectRelationPreparation { prepared: None });
            }
            None => return Err(MetalCommitError::DeviceUnavailable.into_akita()),
        };
        let _host_span = tracing::info_span!("direct_relation_host_prepare").entered();
        let linear_layout = input.linear_layout();
        let linear_segments = linear_layout
            .segments
            .iter()
            .map(|segment| {
                Ok(DirectRelationLinearSegment {
                    factor: crate::field::Fp128Limbs::from_field(segment.factor),
                    source_index: u32::try_from(segment.source_index).map_err(|_| {
                        AkitaError::InvalidSetup(
                            "direct relation source index does not fit the Metal ABI".into(),
                        )
                    })?,
                    target_lane_start: u32::try_from(segment.target_lane_start).map_err(|_| {
                        AkitaError::InvalidSetup(
                            "direct relation target lane does not fit the Metal ABI".into(),
                        )
                    })?,
                    target_lane_stride: u32::try_from(segment.target_lane_stride).map_err(
                        |_| {
                            AkitaError::InvalidSetup(
                                "direct relation target stride does not fit the Metal ABI".into(),
                            )
                        },
                    )?,
                    source_lane_start: u32::try_from(segment.source_lane_start).map_err(|_| {
                        AkitaError::InvalidSetup(
                            "direct relation source lane does not fit the Metal ABI".into(),
                        )
                    })?,
                    source_lane_stride: u32::try_from(segment.source_lane_stride).map_err(
                        |_| {
                            AkitaError::InvalidSetup(
                                "direct relation source stride does not fit the Metal ABI".into(),
                            )
                        },
                    )?,
                    lane_count: u32::try_from(segment.lane_count).map_err(|_| {
                        AkitaError::InvalidSetup(
                            "direct relation lane count does not fit the Metal ABI".into(),
                        )
                    })?,
                })
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        let lane_offsets = linear_layout
            .lane_offsets
            .iter()
            .copied()
            .map(|offset| {
                u32::try_from(offset).map_err(|_| {
                    AkitaError::InvalidSetup(
                        "direct relation lane offset does not fit the Metal ABI".into(),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let lane_segments = linear_layout
            .lane_segments
            .iter()
            .copied()
            .map(|segment| {
                u32::try_from(segment).map_err(|_| {
                    AkitaError::InvalidSetup(
                        "direct relation lane segment does not fit the Metal ABI".into(),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let lane_weights = input
            .relation_lane_weights()
            .iter()
            .copied()
            .map(crate::field::Fp128Limbs::from_field)
            .collect::<Vec<_>>();
        let linear_round = input.linear_round();
        let linear_dense_values = linear_round
            .dense_values
            .unwrap_or_default()
            .into_iter()
            .map(crate::field::Fp128Limbs::from_field)
            .collect::<Vec<_>>();
        let linear_sources = linear_round
            .sources
            .into_iter()
            .map(|values| {
                DirectRelationLinearSourceInput::Values(
                    values
                        .into_iter()
                        .map(crate::field::Fp128Limbs::from_field)
                        .collect(),
                )
            })
            .collect::<Vec<_>>();
        let decoded_witness = {
            let _span = tracing::info_span!("direct_relation_decode_packed_digits").entered();
            input.decode_compact_witness()
        };
        drop(_host_span);
        let _session_span = tracing::info_span!("direct_relation_session_setup").entered();
        let session = runtime.begin_fp128_direct_relation(
            &decoded_witness,
            input.domain_len(),
            COMPACT_PREFIX_ROUNDS,
            input.coefficient_rounds(),
            &lane_weights,
            &linear_segments,
            &lane_offsets,
            &lane_segments,
            &linear_sources,
            &linear_dense_values,
        );
        let (session, setup_timings) = match session {
            Ok(session) => session,
            Err(_error) if self.policy() == MetalExecutionPolicy::PreferMetal => {
                return Ok(MetalDirectRelationPreparation { prepared: None });
            }
            Err(error) => return Err(error.into_akita()),
        };
        Ok(MetalDirectRelationPreparation {
            prepared: Some(Box::new(PreparedMetalDirectRelation {
                session,
                setup_timings,
                domain_len: input.domain_len(),
                coefficient_rounds: input.coefficient_rounds(),
            })),
        })
    }

    fn prove_direct_relation_range<T>(
        &self,
        prepared: &Self::PreparedSetup,
        prover: RelationRangeImageProver<F>,
        preparation: Self::Preparation,
        transcript: &mut T,
        level: u32,
    ) -> Result<DirectRelationRangeProofOutput<F>, AkitaError>
    where
        T: akita_types::ProverTranscriptGrinding<F>,
    {
        const COMPACT_PREFIX_ROUNDS: usize = 3;
        let Some(preparation) = preparation.prepared else {
            return self.direct_relation_cpu_fallback(prepared, prover, transcript, level);
        };
        let PreparedMetalDirectRelation {
            mut session,
            setup_timings,
            domain_len: prepared_domain_len,
            coefficient_rounds,
        } = *preparation;
        let runtime = self
            .runtime()
            .ok_or_else(|| MetalCommitError::DeviceUnavailable.into_akita())?;
        let mut state = DirectRelationRangeProofState::new(prover);
        let num_rounds = state.num_rounds();
        let domain_len = state
            .current_coefficient_count()?
            .checked_mul(state.current_lane_capacity()?)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("direct relation domain length overflow".into())
            })?;
        if domain_len != prepared_domain_len || state.coefficient_bits() != coefficient_rounds {
            return Err(AkitaError::InvalidSetup(
                "prepared direct relation geometry changed before Stage 2".into(),
            ));
        }
        let two_round_prefix = direct_relation_two_round_prefix_host(&state)?;
        let initial_round = direct_relation_host_round(&state)?;
        let (mut next_poly, prefix_reconstruction, initial_timings, initial_allocation_bytes) =
            if let Some(prefix) = &two_round_prefix {
                let outcome = runtime
                    .dispatch_fp128_direct_relation_two_round_prefix(
                        &session,
                        &prefix.equality_first,
                        &prefix.equality_second,
                        &prefix.alpha_points,
                        prefix.norm_omitted_corner,
                        initial_round.as_runtime(),
                    )
                    .map_err(MetalCommitError::into_akita)?;
                let reconstruction = state.reconstruct_two_round_prefix(
                    direct_relation_prefix_fields(outcome.norm_evals_except_corner)?,
                    direct_relation_prefix_fields(outcome.relation_evals_except_corner)?,
                )?;
                let poly = direct_relation_prefix_round_poly(
                    reconstruction.round_zero_coefficients_except_linear(),
                    outcome.additional_coefficients,
                )?;
                (
                    poly,
                    Some(reconstruction),
                    outcome.timings,
                    outcome.allocation_bytes,
                )
            } else {
                let outcome = runtime
                    .dispatch_fp128_direct_relation_initial(&session, initial_round.as_runtime())
                    .map_err(MetalCommitError::into_akita)?;
                (
                    direct_relation_round_poly(&outcome)?,
                    None,
                    outcome.timings,
                    outcome.allocation_bytes,
                )
            };
        self.update_opening_metrics(|metrics| {
            metrics.allocation_bytes = metrics
                .allocation_bytes
                .saturating_add(runtime.direct_relation_session_allocation_bytes(&session));
        })
        .map_err(MetalCommitError::into_akita)?;
        self.record_direct_relation_source_dispatch(setup_timings)?;
        self.record_direct_relation_dispatch(initial_timings, initial_allocation_bytes)?;

        let mut claim = state.input_claim();
        let mut round_polys = Vec::with_capacity(num_rounds);
        let mut challenges = Vec::with_capacity(num_rounds);
        let mut final_evaluation = None;
        let mut final_linear_evaluation = None;
        transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &claim);
        for round in 0..num_rounds {
            transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, &next_poly);
            let challenge = akita_types::sample_grinded_sumcheck_challenge::<F, F, T>(
                transcript,
                akita_types::SumcheckProtocol::Stage2,
                level,
                0,
                u32::try_from(round).map_err(|_| {
                    AkitaError::InvalidSetup("Stage 2 sumcheck round overflow".into())
                })?,
            )?;
            claim = next_poly.eval_from_hint(&claim, &challenge);
            challenges.push(challenge);
            round_polys.push(next_poly.clone());
            state.bind_without_linear_terms(challenge)?;
            if round == 0 {
                if let Some(prefix_reconstruction) = &prefix_reconstruction {
                    let prefix_weights = EqPolynomial::evals(&challenges)?
                        .into_iter()
                        .map(crate::field::Fp128Limbs::from_field)
                        .collect::<Vec<_>>();
                    let next_round = direct_relation_host_round(&state)?;
                    let additional = runtime
                        .dispatch_fp128_direct_relation_additional_compact_only(
                            &mut session,
                            crate::field::Fp128Limbs::from_field(challenge),
                            domain_len / 2,
                            &prefix_weights,
                            next_round.as_runtime(),
                        )
                        .map_err(MetalCommitError::into_akita)?;
                    self.record_direct_relation_dispatch(
                        additional.timings,
                        additional.allocation_bytes,
                    )?;
                    next_poly = direct_relation_prefix_round_poly(
                        prefix_reconstruction.round_one_coefficients_except_linear(challenge),
                        additional.coefficients,
                    )?;
                    continue;
                }
            }
            if round == 1 && prefix_reconstruction.is_some() {
                let prefix_weights = EqPolynomial::evals(&challenges)?
                    .into_iter()
                    .map(crate::field::Fp128Limbs::from_field)
                    .collect::<Vec<_>>();
                let next_round = direct_relation_host_round(&state)?;
                let outcome = runtime
                    .dispatch_fp128_direct_relation_resume_after_two_round_prefix(
                        &mut session,
                        crate::field::Fp128Limbs::from_field(challenge),
                        &prefix_weights,
                        next_round.as_runtime(),
                    )
                    .map_err(MetalCommitError::into_akita)?;
                self.record_direct_relation_dispatch(outcome.timings, outcome.allocation_bytes)?;
                next_poly = direct_relation_round_poly(&outcome)?;
                continue;
            }
            let prefix_weights = if challenges.len() <= COMPACT_PREFIX_ROUNDS {
                EqPolynomial::evals(&challenges)?
                    .into_iter()
                    .map(crate::field::Fp128Limbs::from_field)
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            };
            let next_round = if round + 1 < num_rounds {
                Some(direct_relation_host_round(&state)?)
            } else {
                None
            };
            let advance = runtime
                .dispatch_fp128_direct_relation_advance(
                    &mut session,
                    crate::field::Fp128Limbs::from_field(challenge),
                    &prefix_weights,
                    next_round.as_ref().map(DirectRelationHostRound::as_runtime),
                )
                .map_err(MetalCommitError::into_akita)?;
            self.record_direct_relation_dispatch(advance.timings, advance.allocation_bytes)?;
            if let (Some(coefficients), Some(additional_coefficients)) = (
                advance.next_coefficients,
                advance.next_additional_coefficients,
            ) {
                next_poly = direct_relation_round_poly(&DirectRelationRoundOutcome {
                    coefficients,
                    additional_coefficients,
                    timings: Default::default(),
                    allocation_bytes: 0,
                })?;
            }
            if let Some(evaluation) = advance.final_evaluation {
                final_evaluation = Some(
                    evaluation
                        .into_field(0)
                        .map_err(MetalCommitError::into_akita)?,
                );
            }
            if let Some(evaluation) = advance.final_linear_evaluation {
                final_linear_evaluation = Some(
                    evaluation
                        .into_field(0)
                        .map_err(MetalCommitError::into_akita)?,
                );
            }
        }
        let final_evaluation = final_evaluation.ok_or(AkitaError::InvalidProof)?;
        let final_linear_evaluation = final_linear_evaluation.ok_or(AkitaError::InvalidProof)?;
        let (prover, expected_claim) =
            state.finish_with_linear_evaluation(final_evaluation, final_linear_evaluation)?;
        if claim != expected_claim {
            return Err(AkitaError::InvalidInput(
                "Metal stage-2 final claim disagrees with its folded oracle".into(),
            ));
        }
        Ok((SumcheckProof { round_polys }, challenges, prover))
    }
}

impl ComputeBackendSetup<F> for MetalBackend {
    type PreparedSetup = MetalPreparedSetup;

    fn prepare_expanded(
        &self,
        expanded: Arc<AkitaExpandedSetup<F>>,
    ) -> Result<Self::PreparedSetup, AkitaError> {
        let cpu = self.cpu_backend().prepare_expanded(expanded.clone())?;
        Ok(MetalPreparedSetup::new(cpu, expanded))
    }

    fn ensure_ntt_slot(
        &self,
        prepared: &Self::PreparedSetup,
        key: NttCacheKey,
    ) -> Result<(), AkitaError> {
        self.cpu_backend().ensure_ntt_slot(&prepared.cpu, key)
    }

    fn ntt_requirement_is_cached(
        &self,
        prepared: &Self::PreparedSetup,
        requirement: RoutedNttRequirement,
    ) -> Result<bool, AkitaError> {
        if requirement.cluster == NttOperationCluster::RingSwitch && requirement.key.ring_d == 512 {
            return Ok(false);
        }
        self.cpu_backend()
            .ntt_requirement_is_cached(&prepared.cpu, requirement)
    }

    fn ntt_cache_owner_id(&self, prepared: &Self::PreparedSetup) -> NttCacheOwnerId {
        self.cpu_backend().ntt_cache_owner_id(&prepared.cpu)
    }

    fn planned_ntt_cache_entry_bytes(
        &self,
        prepared: &Self::PreparedSetup,
        key: NttCacheKey,
    ) -> Result<usize, AkitaError> {
        self.cpu_backend()
            .planned_ntt_cache_entry_bytes(&prepared.cpu, key)
    }

    fn prepared_expanded_setup<'a>(
        &self,
        prepared: &'a Self::PreparedSetup,
    ) -> &'a AkitaExpandedSetup<F> {
        prepared.expanded.as_ref()
    }

    fn release_built_ntt_slots(&self, prepared: &Self::PreparedSetup) -> Result<usize, AkitaError> {
        self.cpu_backend().release_built_ntt_slots(&prepared.cpu)
    }
}

impl CompressionComputeBackend<F> for MetalBackend {
    fn compression_negacyclic_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        digit_vectors: &[&[[i8; D]]],
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
        self.cpu_backend()
            .compression_negacyclic_rows(&prepared.cpu, digit_vectors)
    }

    fn compression_cache_bytes(&self, prepared: &Self::PreparedSetup) -> Option<usize> {
        self.cpu_backend().compression_cache_bytes(&prepared.cpu)
    }

    fn compression_rows_products<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        digit_vectors: &[&[[i8; D]]],
    ) -> Result<Vec<CompressionRowsProducts<F, D>>, AkitaError> {
        self.cpu_backend()
            .compression_rows_products(&prepared.cpu, digit_vectors)
    }
}

impl DigitRowsComputeBackend<F> for MetalBackend {
    fn digit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        row_len: usize,
        digit_vectors: &[&[[i8; D]]],
        log_basis: u32,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
        self.cpu_backend()
            .digit_rows(&prepared.cpu, row_len, digit_vectors, log_basis)
    }
}

impl CyclicRowsComputeBackend<F> for MetalBackend {
    fn cyclic_digit_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        row_len: usize,
        digits: &[[i8; D]],
        log_basis: u32,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        self.cpu_backend()
            .cyclic_digit_rows(&prepared.cpu, row_len, digits, log_basis)
    }
}

macro_rules! delegate_opening_pair_to_cpu {
    ($fold:ty, $batch:ty) => {
        impl<const D: usize> OpeningFoldKernel<$fold, F, D> for MetalBackend {
            fn evaluate_and_fold(
                &self,
                prepared: Option<&Self::PreparedSetup>,
                source: $fold,
                plan: OpeningFoldPlan<'_, F>,
            ) -> Result<OpeningFoldOutput<F, D>, AkitaError> {
                let output = <CpuBackend as OpeningFoldKernel<$fold, F, D>>::evaluate_and_fold(
                    &self.cpu_backend(),
                    prepared.map(|prepared| &prepared.cpu),
                    source,
                    plan,
                )?;
                self.record_opening_cpu_fallback(1)
                    .map_err(MetalCommitError::into_akita)?;
                Ok(output)
            }

            fn decompose_fold(
                &self,
                prepared: Option<&Self::PreparedSetup>,
                source: $fold,
                plan: DecomposeFoldPlan<'_>,
            ) -> Result<DecomposeFoldWitness<F>, AkitaError> {
                let output = <CpuBackend as OpeningFoldKernel<$fold, F, D>>::decompose_fold(
                    &self.cpu_backend(),
                    prepared.map(|prepared| &prepared.cpu),
                    source,
                    plan,
                )?;
                self.record_opening_cpu_fallback(1)
                    .map_err(MetalCommitError::into_akita)?;
                Ok(output)
            }
        }

        impl<const D: usize> OpeningBatchKernel<$batch, F, D> for MetalBackend {
            fn decompose_fold_batch(
                &self,
                prepared: Option<&Self::PreparedSetup>,
                source: $batch,
                plan: DecomposeFoldBatchPlan<'_>,
            ) -> Result<BatchDecomposeFoldOutcome<F, D>, AkitaError> {
                let output =
                    <CpuBackend as OpeningBatchKernel<$batch, F, D>>::decompose_fold_batch(
                        &self.cpu_backend(),
                        prepared.map(|prepared| &prepared.cpu),
                        source,
                        plan,
                    )?;
                self.record_opening_cpu_fallback(1)
                    .map_err(MetalCommitError::into_akita)?;
                Ok(output)
            }
        }
    };
}

macro_rules! delegate_indexed_opening_pair_to_cpu {
    ($fold:ty, $batch:ty) => {
        impl<I: OneHotIndex, const D: usize> OpeningFoldKernel<$fold, F, D> for MetalBackend {
            fn evaluate_and_fold(
                &self,
                prepared: Option<&Self::PreparedSetup>,
                source: $fold,
                plan: OpeningFoldPlan<'_, F>,
            ) -> Result<OpeningFoldOutput<F, D>, AkitaError> {
                let output = <CpuBackend as OpeningFoldKernel<$fold, F, D>>::evaluate_and_fold(
                    &self.cpu_backend(),
                    prepared.map(|prepared| &prepared.cpu),
                    source,
                    plan,
                )?;
                self.record_opening_cpu_fallback(1)
                    .map_err(MetalCommitError::into_akita)?;
                Ok(output)
            }

            fn decompose_fold(
                &self,
                prepared: Option<&Self::PreparedSetup>,
                source: $fold,
                plan: DecomposeFoldPlan<'_>,
            ) -> Result<DecomposeFoldWitness<F>, AkitaError> {
                let output = <CpuBackend as OpeningFoldKernel<$fold, F, D>>::decompose_fold(
                    &self.cpu_backend(),
                    prepared.map(|prepared| &prepared.cpu),
                    source,
                    plan,
                )?;
                self.record_opening_cpu_fallback(1)
                    .map_err(MetalCommitError::into_akita)?;
                Ok(output)
            }
        }

        impl<I: OneHotIndex, const D: usize> OpeningBatchKernel<$batch, F, D> for MetalBackend {
            fn decompose_fold_batch(
                &self,
                prepared: Option<&Self::PreparedSetup>,
                source: $batch,
                plan: DecomposeFoldBatchPlan<'_>,
            ) -> Result<BatchDecomposeFoldOutcome<F, D>, AkitaError> {
                let output =
                    <CpuBackend as OpeningBatchKernel<$batch, F, D>>::decompose_fold_batch(
                        &self.cpu_backend(),
                        prepared.map(|prepared| &prepared.cpu),
                        source,
                        plan,
                    )?;
                self.record_opening_cpu_fallback(1)
                    .map_err(MetalCommitError::into_akita)?;
                Ok(output)
            }
        }
    };
}

delegate_opening_pair_to_cpu!(DenseView<'_, F, D>, DenseBatchView<'_, F, D>);
delegate_opening_pair_to_cpu!(
    SuffixWitnessView<'_, F, D>,
    SuffixWitnessBatchView<'_, F, D>
);
delegate_indexed_opening_pair_to_cpu!(OneHotView<'_, F, D, I>, OneHotBatchView<'_, F, D, I>);
delegate_indexed_opening_pair_to_cpu!(
    MultilinearPolynomialView<'_, F, D, I>,
    MultilinearPolynomialBatchView<'_, F, D, I>
);

macro_rules! delegate_coefficient_packing_to_cpu {
    ($view:ident) => {
        impl<E, const D: usize> SubringCoefficientPackingBatchKernel<$view<'_, F, D>, F, E, D>
            for MetalBackend
        where
            E: ExtField<F>,
            CpuBackend: for<'a> SubringCoefficientPackingBatchKernel<$view<'a, F, D>, F, E, D>,
        {
            fn coefficient_packing_partials_batch(
                &self,
                prepared: Option<&Self::PreparedSetup>,
                source: $view<'_, F, D>,
                plan: SubringCoefficientPackingPlan<'_, E>,
            ) -> Result<Vec<SubringCoefficientPackingPartials<F>>, AkitaError> {
                let output = self.cpu_backend().coefficient_packing_partials_batch(
                    prepared.map(|prepared| &prepared.cpu),
                    source,
                    plan,
                )?;
                self.record_opening_cpu_fallback(1)
                    .map_err(MetalCommitError::into_akita)?;
                Ok(output)
            }
        }
    };
}

delegate_coefficient_packing_to_cpu!(RecursiveFoldBatchView);
delegate_coefficient_packing_to_cpu!(SuffixWitnessBatchView);

#[cfg(test)]
mod tests;
