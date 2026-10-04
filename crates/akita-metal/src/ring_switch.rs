use akita_algebra::CyclotomicRing;
use akita_error::AkitaError;
use akita_prover::backend::RingSwitchRelationView;
use akita_prover::compute::{
    RingSwitchRelationKernel, RingSwitchRelationPlan, RingSwitchRelationRows,
};

use crate::field::{MetalField, F};
use crate::runtime::{
    D512LinearRelationParams, DigitRowsParams, FP128_D64_DIGIT_ROWS_COLUMNS_PER_PARTIAL,
};
use crate::{MetalBackend, MetalCommitError, MetalPreparedSetup};
use std::time::Instant;

impl<const D: usize> RingSwitchRelationKernel<RingSwitchRelationView<'_, D>, F, D>
    for MetalBackend
{
    fn relation_rows(
        &self,
        prepared: &MetalPreparedSetup,
        source: RingSwitchRelationView<'_, D>,
        plan: RingSwitchRelationPlan,
    ) -> Result<RingSwitchRelationRows<F, D>, AkitaError> {
        let total_start = Instant::now();
        if D == 64
            && plan.n_d != 0
            && plan.n_b == 0
            && plan.n_a == 0
            && plan.log_basis_open == 3
            && source.t_hat.is_empty()
            && source.z_segment.is_empty()
            && self.runtime().is_some_and(|runtime| {
                runtime.supports_fp128_d64_digit_rows::<D>(1, plan.n_d, source.e_hat.len(), true)
            })
        {
            return self.digit_relation_rows(prepared, source.e_hat, plan.n_d);
        }
        let rhs_abs_bound = source
            .z_segment
            .iter()
            .flat_map(|row| row.iter())
            .map(|value| u64::from(value.unsigned_abs()))
            .max()
            .unwrap_or(0)
            .max(u64::from(source.z_folded_centered_inf_norm));
        let use_metal = D == 512
            && plan.n_d == 0
            && plan.n_b == 0
            && plan.n_a == 1
            && source.e_hat.is_empty()
            && source.t_hat.is_empty()
            && !source.z_segment.is_empty()
            && self.runtime().is_some_and(|runtime| {
                runtime.supports_fp128_d512_linear_relation(source.z_segment.len(), rhs_abs_bound)
            });
        if !use_metal {
            let work_units = plan
                .n_d
                .saturating_mul(source.e_hat.len())
                .saturating_add(plan.n_b.saturating_mul(source.t_hat.len()))
                .saturating_add(plan.n_a.saturating_mul(source.z_segment.len()))
                .saturating_mul(D);
            self.record_opening_cpu_fallback(work_units)
                .map_err(MetalCommitError::into_akita)?;
            let output = self
                .cpu_backend()
                .relation_rows(&prepared.cpu, source, plan)?;
            tracing::debug!(
                route = "cpu",
                ring_dimension = D,
                n_d = plan.n_d,
                n_b = plan.n_b,
                n_a = plan.n_a,
                z_rows = source.z_segment.len(),
                elapsed_s = total_start.elapsed().as_secs_f64(),
                "completed Metal ring-switch relation route"
            );
            return Ok(output);
        }

        let runtime = self
            .runtime()
            .ok_or_else(|| MetalCommitError::DeviceUnavailable.into_akita())?;
        let matrix = prepared.matrix(runtime, 512, 1, source.z_segment.len())?;
        let num_tiles = source.z_segment.len().div_ceil(64);
        let outcome = runtime
            .dispatch_fp128_d512_linear_relation(
                &matrix.buffer,
                source.z_segment,
                D512LinearRelationParams {
                    num_columns: source.z_segment.len() as u64,
                    columns_per_tile: 64,
                    num_tiles: num_tiles as u64,
                    num_primes: 6,
                    ntt_size: 1_024,
                    output_coefficients: D as u64,
                    rhs_abs_bound,
                },
            )
            .map_err(MetalCommitError::into_akita)?;
        let timings = outcome.timings;
        let coefficients = outcome
            .coefficients
            .into_iter()
            .enumerate()
            .map(|(index, value)| F::from_device(value, index))
            .collect::<Result<Vec<_>, _>>()
            .map_err(MetalCommitError::into_akita)?;
        if coefficients.len() != D {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: coefficients.len(),
            });
        }
        let quotient = CyclotomicRing::from_slice(&coefficients);
        self.update_opening_metrics(|metrics| {
            metrics.command_wall_time += timings.command_wall;
            metrics.gpu_active_time += timings.gpu.unwrap_or_default();
            metrics.buffer_setup_time += timings.buffer_setup + matrix.prepare_time;
            metrics.readback_time += timings.readback_copy;
            metrics.allocation_bytes = metrics
                .allocation_bytes
                .saturating_add(outcome.allocation_bytes)
                .saturating_add(matrix.bytes.saturating_mul(usize::from(!matrix.cache_hit)));
        })
        .map_err(MetalCommitError::into_akita)?;
        tracing::debug!(
            route = "metal",
            ring_dimension = D,
            n_d = plan.n_d,
            n_b = plan.n_b,
            n_a = plan.n_a,
            z_rows = source.z_segment.len(),
            gpu_s = timings.gpu.map(|duration| duration.as_secs_f64()),
            elapsed_s = total_start.elapsed().as_secs_f64(),
            "completed Metal ring-switch relation route"
        );
        Ok(RingSwitchRelationRows {
            d_negacyclic: Vec::new(),
            d_cyclic: Vec::new(),
            b_cyclic: Vec::new(),
            a_quotients: vec![quotient],
        })
    }
}

impl MetalBackend {
    fn digit_relation_rows<const D: usize>(
        &self,
        prepared: &MetalPreparedSetup,
        digits: &[[i8; D]],
        num_rows: usize,
    ) -> Result<RingSwitchRelationRows<F, D>, AkitaError> {
        let _span = tracing::info_span!(
            "MetalRingSwitch::digit_relation_rows",
            num_rows,
            num_columns = digits.len(),
        )
        .entered();
        if digits
            .iter()
            .flatten()
            .any(|digit| !(-4..=3).contains(digit))
        {
            return Err(AkitaError::InvalidInput(
                "D-role digits exceed the configured opening basis".into(),
            ));
        }
        let runtime = self
            .runtime()
            .ok_or_else(|| MetalCommitError::DeviceUnavailable.into_akita())?;
        let matrix = prepared.matrix(runtime, D, num_rows, digits.len())?;
        let output_coefficients = num_rows.checked_mul(D).ok_or_else(|| {
            MetalCommitError::ShapeOverflow("D-role output coefficients").into_akita()
        })?;
        let outcome = runtime
            .dispatch_fp128_d64_digit_rows(
                &matrix.buffer,
                &[digits],
                true,
                DigitRowsParams {
                    num_vectors: 1,
                    num_rows: num_rows as u64,
                    num_cols: digits.len() as u64,
                    ring_d: D as u64,
                    output_coefficients: output_coefficients as u64,
                    columns_per_partial: FP128_D64_DIGIT_ROWS_COLUMNS_PER_PARTIAL as u64,
                    column_partials: digits
                        .len()
                        .div_ceil(FP128_D64_DIGIT_ROWS_COLUMNS_PER_PARTIAL)
                        as u64,
                    retain_quotients: 1,
                },
            )
            .map_err(MetalCommitError::into_akita)?;
        let timings = outcome.timings;
        let coefficients = outcome
            .coefficients
            .into_iter()
            .enumerate()
            .map(|(index, value)| F::from_device(value, index))
            .collect::<Result<Vec<_>, _>>()
            .map_err(MetalCommitError::into_akita)?;
        let expected = output_coefficients.checked_mul(2).ok_or_else(|| {
            MetalCommitError::ShapeOverflow("D-role product coefficients").into_akita()
        })?;
        if coefficients.len() != expected {
            return Err(AkitaError::InvalidSize {
                expected,
                actual: coefficients.len(),
            });
        }
        let (negacyclic, quotients) = coefficients.split_at(output_coefficients);
        let d_negacyclic = negacyclic
            .chunks_exact(D)
            .map(CyclotomicRing::from_slice)
            .collect::<Vec<_>>();
        // The retained high product H converts L-H into the cyclic product L+H.
        let d_cyclic = d_negacyclic
            .iter()
            .zip(quotients.chunks_exact(D))
            .map(|(row, coefficients)| {
                let quotient = CyclotomicRing::from_slice(coefficients);
                *row + quotient + quotient
            })
            .collect();
        self.update_opening_metrics(|metrics| {
            metrics.command_wall_time += timings.command_wall;
            metrics.gpu_active_time += timings.gpu.unwrap_or_default();
            metrics.buffer_setup_time += timings.buffer_setup + matrix.prepare_time;
            metrics.readback_time += timings.readback_copy;
            metrics.allocation_bytes = metrics
                .allocation_bytes
                .saturating_add(outcome.allocation_bytes)
                .saturating_add(matrix.bytes.saturating_mul(usize::from(!matrix.cache_hit)));
        })
        .map_err(MetalCommitError::into_akita)?;
        Ok(RingSwitchRelationRows {
            d_negacyclic,
            d_cyclic,
            b_cyclic: Vec::new(),
            a_quotients: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use akita_prover::{AkitaProverSetup, ComputeBackendSetup, CpuBackend};
    use akita_types::SetupMatrixCapacity;

    use super::*;
    use crate::MetalExecutionPolicy;

    #[test]
    fn d64_digit_relation_matches_cpu_across_tiles() {
        const D: usize = 64;
        const COLUMNS: usize = 257;
        const ROWS: usize = 3;
        let setup = AkitaProverSetup::<F>::generate_with_capacity(
            20,
            1,
            SetupMatrixCapacity {
                num_field_elements: ROWS * COLUMNS * D,
            },
        )
        .unwrap();
        let cpu = CpuBackend::DEFAULT;
        let cpu_prepared = cpu.prepare_setup(&setup).unwrap();
        let metal = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
        let metal_prepared = metal.prepare_setup(&setup).unwrap();
        for columns in [1, 128, COLUMNS] {
            let digits = (0..columns)
                .map(|column| {
                    std::array::from_fn(|coefficient| {
                        if column % 7 == 0 {
                            0
                        } else {
                            const VALUES: [i8; 8] = [-4, -3, -2, -1, 0, 1, 2, 3];
                            VALUES[(column * 17 + coefficient * 7) % VALUES.len()]
                        }
                    })
                })
                .collect::<Vec<[i8; D]>>();
            let source = RingSwitchRelationView {
                e_hat: &digits,
                t_hat: &[],
                z_segment: &[],
                z_folded_centered_inf_norm: 0,
            };
            let plan = RingSwitchRelationPlan {
                n_d: ROWS,
                n_b: 0,
                n_a: 0,
                log_basis_open: 3,
                log_basis_outer: 3,
            };
            let expected = cpu.relation_rows(&cpu_prepared, source, plan).unwrap();
            metal.begin_opening_metrics().unwrap();
            let actual = metal.relation_rows(&metal_prepared, source, plan).unwrap();
            assert_eq!(actual, expected);
            let metrics = metal.last_opening_metrics().unwrap().unwrap();
            assert_eq!(metrics.cpu_fallback_calls, 0);
            assert!(metrics.gpu_active_time > std::time::Duration::ZERO);
        }
        let invalid = [[4i8; D]];
        let source = RingSwitchRelationView {
            e_hat: &invalid,
            t_hat: &[],
            z_segment: &[],
            z_folded_centered_inf_norm: 0,
        };
        let plan = RingSwitchRelationPlan {
            n_d: ROWS,
            n_b: 0,
            n_a: 0,
            log_basis_open: 3,
            log_basis_outer: 3,
        };
        assert!(matches!(
            metal.relation_rows(&metal_prepared, source, plan),
            Err(AkitaError::InvalidInput(_))
        ));
    }

    #[test]
    fn d512_linear_relation_matches_cpu_across_tiles() {
        const D: usize = 512;
        const COLUMNS: usize = 67;
        let setup = AkitaProverSetup::<F>::generate_with_capacity(
            20,
            1,
            SetupMatrixCapacity {
                num_field_elements: COLUMNS * D,
            },
        )
        .unwrap();
        let z_segment: Vec<[i32; D]> = (0..COLUMNS)
            .map(|column| {
                std::array::from_fn(|coefficient| {
                    const VALUES: [i32; 11] = [-9, -4, -2, -1, 0, 1, 2, 3, 5, 8, 13];
                    VALUES[(column * 17 + coefficient * 7) % VALUES.len()]
                })
            })
            .collect::<Vec<_>>();
        let source = RingSwitchRelationView {
            e_hat: &[],
            t_hat: &[],
            z_segment: &z_segment,
            z_folded_centered_inf_norm: 13,
        };
        let plan = RingSwitchRelationPlan {
            n_d: 0,
            n_b: 0,
            n_a: 1,
            log_basis_open: 3,
            log_basis_outer: 3,
        };

        let cpu = CpuBackend::DEFAULT;
        let cpu_prepared = cpu.prepare_setup(&setup).unwrap();
        let expected = cpu.relation_rows(&cpu_prepared, source, plan).unwrap();

        let metal = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
        let metal_prepared = metal.prepare_setup(&setup).unwrap();
        metal.begin_opening_metrics().unwrap();
        let actual = metal.relation_rows(&metal_prepared, source, plan).unwrap();
        assert_eq!(actual, expected);
        let metrics = metal.last_opening_metrics().unwrap().unwrap();
        assert_eq!(metrics.cpu_fallback_calls, 0);
        assert!(metrics.gpu_active_time > std::time::Duration::ZERO);
    }
}
