use akita_algebra::ring::cyclotomic::decompose_centering_threshold;
use akita_algebra::CyclotomicRing;
use akita_error::AkitaError;
use akita_prover::backend::poly_helpers::build_decompose_fold_witness;
use akita_prover::backend::{
    setup_prefix_decompose_rings, RecursiveFoldBatchView, RecursiveFoldView,
};
use akita_prover::compute::{
    BatchDecomposeFoldOutcome, DecomposeFoldBatchPlan, DecomposeFoldPlan, OpeningBatchKernel,
    OpeningFoldKernel, OpeningFoldOutput, OpeningFoldPlan,
};
use akita_prover::DecomposeFoldWitness;
use jolt_field::One;

use crate::field::{Fp128Limbs, F};
use crate::runtime::RecursiveDecomposeFoldParams;
use crate::{MetalBackend, MetalCommitError, MetalPreparedSetup};

impl<const D: usize> OpeningFoldKernel<RecursiveFoldView<'_, F, D>, F, D> for MetalBackend {
    fn evaluate_and_fold(
        &self,
        prepared: Option<&MetalPreparedSetup>,
        source: RecursiveFoldView<'_, F, D>,
        plan: OpeningFoldPlan<'_, F>,
    ) -> Result<OpeningFoldOutput<F, D>, AkitaError> {
        self.record_opening_cpu_fallback(1)
            .map_err(MetalCommitError::into_akita)?;
        self.cpu_backend()
            .evaluate_and_fold(prepared.map(|value| &value.cpu), source, plan)
    }

    fn decompose_fold(
        &self,
        prepared: Option<&MetalPreparedSetup>,
        source: RecursiveFoldView<'_, F, D>,
        plan: DecomposeFoldPlan<'_>,
    ) -> Result<DecomposeFoldWitness<F>, AkitaError> {
        if let RecursiveFoldView::SetupPrefix { expanded, slot } = source {
            let rings = setup_prefix_decompose_rings::<F, D>(expanded, slot, &plan)?;
            if let Some(witness) = self.decompose_recursive_rings(rings, &plan)? {
                return Ok(witness);
            }
        }
        self.record_opening_cpu_fallback(1)
            .map_err(MetalCommitError::into_akita)?;
        self.cpu_backend()
            .decompose_fold(prepared.map(|value| &value.cpu), source, plan)
    }
}

impl<const D: usize> OpeningBatchKernel<RecursiveFoldBatchView<'_, F, D>, F, D> for MetalBackend {
    fn decompose_fold_batch(
        &self,
        _prepared: Option<&MetalPreparedSetup>,
        _source: RecursiveFoldBatchView<'_, F, D>,
        _plan: DecomposeFoldBatchPlan<'_>,
    ) -> Result<BatchDecomposeFoldOutcome<F, D>, AkitaError> {
        Ok(BatchDecomposeFoldOutcome::FallbackPerPoly)
    }
}

impl MetalBackend {
    #[tracing::instrument(skip_all, name = "MetalRecursiveFold::decompose_fold", fields(
        ring_d = D, source_rings = source.len(), digits = plan.num_digits,
        log_basis = plan.log_basis, positions = plan.num_positions_per_block,
    ))]
    fn decompose_recursive_rings<const D: usize>(
        &self,
        source: &[CyclotomicRing<F, D>],
        plan: &DecomposeFoldPlan<'_>,
    ) -> Result<Option<DecomposeFoldWitness<F>>, AkitaError> {
        let Some(runtime) = self.runtime() else {
            return Ok(None);
        };
        if !matches!(D, 64 | 128)
            || !(1..=16).contains(&plan.log_basis)
            || plan.num_digits == 0
            || plan.num_positions_per_block == 0
            || source.is_empty()
            || source.len() > u32::MAX as usize / D
        {
            return Ok(None);
        }
        let Some(output_coefficients) = plan
            .num_positions_per_block
            .checked_mul(plan.num_digits)
            .and_then(|count| count.checked_mul(D))
            .filter(|count| *count <= u32::MAX as usize)
        else {
            return Ok(None);
        };
        let weight = plan
            .challenges
            .iter()
            .flat_map(|challenge| &challenge.coeffs)
            .try_fold(0u64, |total, value| {
                total.checked_add(u64::from(value.unsigned_abs()))
            });
        if weight
            .and_then(|weight| weight.checked_mul(1u64 << (plan.log_basis - 1)))
            .is_none_or(|bound| bound > i32::MAX as u64)
        {
            return Ok(None);
        }
        let mut offsets = vec![0u32];
        let mut positions = Vec::new();
        let mut coefficients = Vec::new();
        for challenge in plan.challenges {
            if challenge.positions.len() != challenge.coeffs.len()
                || challenge
                    .positions
                    .iter()
                    .any(|&position| position >= D as u32)
            {
                return Err(AkitaError::InvalidInput(
                    "invalid recursive sparse challenge".into(),
                ));
            }
            positions.extend_from_slice(&challenge.positions);
            coefficients.extend_from_slice(&challenge.coeffs);
            offsets.push(u32::try_from(positions.len()).map_err(|_| {
                AkitaError::InvalidInput("recursive challenge offset exceeds Metal grid".into())
            })?);
        }
        if positions.is_empty() {
            positions.push(0);
            coefficients.push(0);
        }
        let modulus = (-F::one()).to_canonical_u128() + 1;
        let params = RecursiveDecomposeFoldParams {
            source_rings: source.len() as u64,
            positions: plan.num_positions_per_block as u64,
            blocks: plan.challenges.len() as u64,
            digits: plan.num_digits as u64,
            ring_d: D as u32,
            log_basis: plan.log_basis,
            output_coefficients: output_coefficients as u64,
            threshold: Fp128Limbs::from_u128(decompose_centering_threshold(
                plan.num_digits,
                plan.log_basis,
                modulus,
            )),
        };
        let (centered, timings, allocation_bytes) = runtime
            .dispatch_recursive_decompose_fold(source, &offsets, &positions, &coefficients, params)
            .map_err(MetalCommitError::into_akita)?;
        self.update_opening_metrics(|metrics| {
            metrics.command_wall_time += timings.command_wall;
            metrics.gpu_active_time += timings.gpu.unwrap_or_default();
            metrics.buffer_setup_time += timings.buffer_setup;
            metrics.readback_time += timings.readback_copy;
            metrics.allocation_bytes = metrics.allocation_bytes.saturating_add(allocation_bytes);
        })
        .map_err(MetalCommitError::into_akita)?;
        Ok(Some(build_decompose_fold_witness(centered, modulus)))
    }
}

#[cfg(test)]
mod tests {
    use akita_challenges::SparseChallenge;
    use akita_prover::backend::{DensePoly, DenseView};
    use akita_prover::compute::RootOpeningSource;
    use akita_prover::CpuBackend;
    use jolt_field::CanonicalEncoding;

    use super::*;
    use crate::MetalExecutionPolicy;

    fn check_recursive_decompose<const D: usize>(backend: &MetalBackend) {
        let modulus = (-F::one()).to_canonical_u128() + 1;
        for log_basis in [1u32, 4, 10, 16] {
            let num_digits = 128usize.div_ceil(log_basis as usize);
            let threshold = decompose_centering_threshold(num_digits, log_basis, modulus);
            let boundary = [
                0,
                1,
                threshold,
                threshold + 1,
                modulus / 2,
                modulus / 2 + 1,
                modulus - 1,
                1u128 << 127,
            ];
            let rings = (0..16)
                .map(|ring| {
                    CyclotomicRing::from_coefficients(std::array::from_fn(|coefficient| {
                        let index = ring * D + coefficient;
                        let value = if index < boundary.len() {
                            boundary[index]
                        } else {
                            (index as u128).wrapping_mul(0x9e3779b97f4a7c15_bf58476d1ce4e5b9)
                        };
                        F::from_u128_reduced(value)
                    }))
                })
                .collect::<Vec<CyclotomicRing<F, D>>>();
            let source = DensePoly::from_ring_coeffs(rings.clone());
            let view = <DensePoly<F> as RootOpeningSource<F, D>>::opening_view(&source).unwrap();
            let challenges = (0..4)
                .map(|block| SparseChallenge {
                    positions: [0, 1, (D / 2) as u32, (D - 1) as u32].into_iter().collect(),
                    coeffs: [-1, 2, -3, if block % 2 == 0 { 1 } else { -1 }]
                        .into_iter()
                        .collect(),
                })
                .collect::<Vec<_>>();
            let plan = DecomposeFoldPlan {
                challenges: &challenges,
                num_positions_per_block: 4,
                num_digits,
                log_basis,
            };
            let expected =
                <CpuBackend as OpeningFoldKernel<DenseView<'_, F, D>, F, D>>::decompose_fold(
                    &CpuBackend::DEFAULT,
                    None,
                    view,
                    plan,
                )
                .unwrap();
            let actual = backend
                .decompose_recursive_rings(&rings, &plan)
                .unwrap()
                .unwrap();
            assert_eq!(actual, expected, "D={D}, log_basis={log_basis}");
        }
    }

    #[test]
    fn recursive_decompose_fold_matches_cpu() {
        let backend = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
        check_recursive_decompose::<64>(&backend);
        check_recursive_decompose::<128>(&backend);
    }
}
