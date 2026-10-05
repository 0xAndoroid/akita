use akita_algebra::CyclotomicRing;
use akita_error::AkitaError;
use akita_prover::backend::RingSwitchRelationView;
use akita_prover::compute::{
    RingSwitchRelationKernel, RingSwitchRelationPlan, RingSwitchRelationRows,
};

use crate::field::{MetalField, F};
use crate::runtime::{digit_rows_columns_per_partial, D512LinearRelationParams, DigitRowsParams};
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
                runtime.supports_fp128_d64_digit_rows::<D>(
                    1,
                    plan.n_d,
                    source.e_hat.len(),
                    true,
                    plan.log_basis_open,
                )
            })
        {
            return self.digit_relation_rows(
                prepared,
                source.e_hat,
                plan.n_d,
                plan.log_basis_open,
                false,
            );
        }
        if D == 64
            && plan.n_d == 0
            && plan.n_a == 0
            && plan.n_b != 0
            && (1..=8).contains(&plan.log_basis_outer)
            && source.e_hat.is_empty()
            && source.z_segment.is_empty()
            && source.t_hat.len() >= 32_768
            && self.runtime().is_some_and(|runtime| {
                runtime.supports_fp128_d64_digit_rows::<D>(
                    1,
                    plan.n_b,
                    source.t_hat.len(),
                    false,
                    plan.log_basis_outer,
                )
            })
        {
            return self.digit_relation_rows(
                prepared,
                source.t_hat,
                plan.n_b,
                plan.log_basis_outer,
                true,
            );
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
        log_basis: u32,
        cyclic: bool,
    ) -> Result<RingSwitchRelationRows<F, D>, AkitaError> {
        let _span = tracing::info_span!(
            "MetalRingSwitch::digit_relation_rows",
            num_rows,
            num_columns = digits.len(),
            ring_d = D,
            cyclic,
        )
        .entered();
        let columns_per_partial = digit_rows_columns_per_partial(log_basis)
            .ok_or_else(|| AkitaError::InvalidInput("digit basis must be in 1..=8".into()))?;
        let bound = 1i16 << (log_basis - 1);
        if digits
            .iter()
            .flatten()
            .any(|&digit| !(-bound..bound).contains(&i16::from(digit)))
        {
            return Err(AkitaError::InvalidInput(
                "relation digits exceed the configured basis".into(),
            ));
        }
        let runtime = self
            .runtime()
            .ok_or_else(|| MetalCommitError::DeviceUnavailable.into_akita())?;
        let matrix_start = Instant::now();
        let fields = prepared
            .expanded
            .shared_matrix()
            .ring_view::<D>(num_rows, digits.len())?;
        let matrix = runtime
            .shared_slice_buffer(fields.as_slice())
            .map_err(MetalCommitError::into_akita)?;
        let matrix_prepare_time = matrix_start.elapsed();
        let output_coefficients = num_rows.checked_mul(D).ok_or_else(|| {
            MetalCommitError::ShapeOverflow("D-role output coefficients").into_akita()
        })?;
        let outcome = runtime
            .dispatch_fp128_d64_digit_rows(
                &matrix.buffer,
                &[digits],
                !cyclic,
                log_basis,
                DigitRowsParams {
                    num_vectors: 1,
                    num_rows: num_rows as u64,
                    num_cols: digits.len() as u64,
                    ring_d: D as u64,
                    output_coefficients: output_coefficients as u64,
                    columns_per_partial: columns_per_partial as u64,
                    column_partials: digits.len().div_ceil(columns_per_partial) as u64,
                    retain_quotients: u64::from(!cyclic),
                    cyclic: u64::from(cyclic),
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
        let expected = output_coefficients
            .checked_mul(if cyclic { 1 } else { 2 })
            .ok_or_else(|| {
                MetalCommitError::ShapeOverflow("D-role product coefficients").into_akita()
            })?;
        if coefficients.len() != expected {
            return Err(AkitaError::InvalidSize {
                expected,
                actual: coefficients.len(),
            });
        }
        let rows = coefficients[..output_coefficients]
            .chunks_exact(D)
            .map(CyclotomicRing::from_slice)
            .collect::<Vec<_>>();
        let output = if cyclic {
            RingSwitchRelationRows {
                d_negacyclic: Vec::new(),
                d_cyclic: Vec::new(),
                b_cyclic: rows,
                a_quotients: Vec::new(),
            }
        } else {
            // The retained high product H converts L-H into the cyclic product L+H.
            let d_cyclic = rows
                .iter()
                .zip(coefficients[output_coefficients..].chunks_exact(D))
                .map(|(row, coefficients)| {
                    let quotient = CyclotomicRing::from_slice(coefficients);
                    *row + quotient + quotient
                })
                .collect();
            RingSwitchRelationRows {
                d_negacyclic: rows,
                d_cyclic,
                b_cyclic: Vec::new(),
                a_quotients: Vec::new(),
            }
        };
        self.update_opening_metrics(|metrics| {
            metrics.command_wall_time += timings.command_wall;
            metrics.gpu_active_time += timings.gpu.unwrap_or_default();
            metrics.buffer_setup_time += timings.buffer_setup + matrix_prepare_time;
            metrics.readback_time += timings.readback_copy;
            metrics.allocation_bytes = metrics
                .allocation_bytes
                .saturating_add(outcome.allocation_bytes)
                .saturating_add(
                    size_of_val(fields.as_slice()).saturating_mul(usize::from(!matrix.zero_copy)),
                );
        })
        .map_err(MetalCommitError::into_akita)?;
        Ok(output)
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
    fn cyclic_digit_rows_match_cpu_d64_rank2() {
        const D: usize = 64;
        const ROWS: usize = 2;
        const COLUMNS: usize = 129;
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
        let prepared = metal.prepare_setup(&setup).unwrap();
        for log_basis in [1, 3, 8] {
            let bound = 1i16 << (log_basis - 1);
            for negative_only in [false, true] {
                for columns in [1, 128, COLUMNS] {
                    let digits = (0..columns)
                        .map(|column| {
                            std::array::from_fn(|coefficient| {
                                if negative_only {
                                    -bound as i8
                                } else if column % 7 == 0 {
                                    0
                                } else {
                                    ((column * 17 + coefficient * 7) as i16 % (2 * bound) - bound)
                                        as i8
                                }
                            })
                        })
                        .collect::<Vec<[i8; D]>>();
                    let expected = cpu
                        .relation_rows(
                            &cpu_prepared,
                            RingSwitchRelationView {
                                e_hat: &[],
                                t_hat: &digits,
                                z_segment: &[],
                                z_folded_centered_inf_norm: 0,
                            },
                            RingSwitchRelationPlan {
                                n_d: 0,
                                n_b: ROWS,
                                n_a: 0,
                                log_basis_open: 3,
                                log_basis_outer: log_basis,
                            },
                        )
                        .unwrap();
                    let actual = metal
                        .digit_relation_rows(&prepared, &digits, ROWS, log_basis, true)
                        .unwrap();
                    assert_eq!(
                        actual, expected,
                        "D={D}, basis={log_basis}, columns={columns}"
                    );
                }
            }
        }
        assert!(metal
            .digit_relation_rows(&prepared, &[[4i8; D]], ROWS, 3, true)
            .is_err());
        assert!(
            !metal.runtime().unwrap().supports_fp128_d64_digit_rows::<D>(
                1,
                1,
                (1 << 25) + 1,
                false,
                8,
            )
        );
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
