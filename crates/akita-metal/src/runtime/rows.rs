use super::*;

impl MetalRuntime {
    pub(crate) fn dispatch_fp128_d64_digit_rows<const D: usize>(
        &self,
        matrix: &Buffer,
        digit_vectors: &[&[[i8; D]]],
        retain_quotients: bool,
        params: DigitRowsParams,
    ) -> Result<DigitRowsDispatchOutcome, MetalCommitError> {
        autoreleasepool(|| {
            let expected_output = params
                .num_vectors
                .checked_mul(params.num_rows)
                .and_then(|count| count.checked_mul(params.ring_d))
                .ok_or(MetalCommitError::ShapeOverflow("digit-row output"))?;
            let product_count = 1u64 + u64::from(retain_quotients);
            let total_output = params
                .output_coefficients
                .checked_mul(product_count)
                .ok_or(MetalCommitError::ShapeOverflow("digit-row product output"))?;
            let expected_vector_width = usize::try_from(params.num_cols)
                .map_err(|_| MetalCommitError::ShapeOverflow("digit-row column count"))?;
            let expected_vector_count = usize::try_from(params.num_vectors)
                .map_err(|_| MetalCommitError::ShapeOverflow("digit-row vector count"))?;
            let expected_row_count = usize::try_from(params.num_rows)
                .map_err(|_| MetalCommitError::ShapeOverflow("digit-row row count"))?;
            let expected_column_partials = params
                .num_cols
                .div_ceil(FP128_D64_DIGIT_ROWS_COLUMNS_PER_PARTIAL as u64);
            let partial_count = params
                .num_vectors
                .checked_mul(params.num_rows)
                .and_then(|count| count.checked_mul(expected_column_partials))
                .and_then(|count| count.checked_mul(params.ring_d))
                .and_then(|count| count.checked_mul(product_count))
                .ok_or(MetalCommitError::ShapeOverflow("digit-row partials"))?;
            let expected_matrix_bytes = params
                .num_rows
                .checked_mul(params.num_cols)
                .and_then(|count| count.checked_mul(params.ring_d))
                .and_then(|count| count.checked_mul(size_of::<Fp128Limbs>() as u64))
                .ok_or(MetalCommitError::ShapeOverflow("digit-row matrix bytes"))?;
            if D != 64
                || params.ring_d != 64
                || params.num_vectors == 0
                || params.num_rows == 0
                || digit_vectors.len() != expected_vector_count
                || digit_vectors
                    .iter()
                    .any(|digits| digits.len() != expected_vector_width)
                || params.output_coefficients != expected_output
                || params.retain_quotients != u64::from(retain_quotients)
                || params.columns_per_partial != FP128_D64_DIGIT_ROWS_COLUMNS_PER_PARTIAL as u64
                || params.column_partials != expected_column_partials
                || total_output > u64::from(u32::MAX)
                || params
                    .num_vectors
                    .checked_mul(params.num_rows)
                    .and_then(|count| count.checked_mul(params.column_partials))
                    .is_none_or(|count| count > u64::from(u32::MAX))
                || matrix.length() < expected_matrix_bytes
                || !self.supports_fp128_d64_digit_rows::<D>(
                    expected_vector_count,
                    expected_row_count,
                    expected_vector_width,
                    retain_quotients,
                )
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "fp128 D64 digit rows exceed the kernel's index or device-buffer limits".into(),
                ));
            }

            let buffer_start = Instant::now();
            let (digit_buffer, output, partials, output_count) = {
                let input_bytes = digit_vectors.iter().try_fold(0usize, |total, digits| {
                    total
                        .checked_add(size_of_val(*digits))
                        .ok_or(MetalCommitError::ShapeOverflow("digit-row input bytes"))
                })?;
                let span = tracing::info_span!(
                    "MetalDigitRows::buffer_setup",
                    input_bytes,
                    num_vectors = expected_vector_count,
                    num_cols = expected_vector_width,
                    num_rows = expected_row_count,
                    ring_dimension = D,
                );
                let _entered = span.enter();
                let digit_buffer = self.shared_buffer_from_digit_rows(digit_vectors)?;
                let output_count = usize::try_from(total_output).map_err(|_| {
                    MetalCommitError::ShapeOverflow("digit-row output coefficients")
                })?;
                let output_bytes = output_count
                    .checked_mul(size_of::<Fp128Limbs>())
                    .ok_or(MetalCommitError::ShapeOverflow("digit-row output bytes"))?;
                let output = self.shared_buffer(output_bytes)?;
                let partial_count = usize::try_from(partial_count)
                    .map_err(|_| MetalCommitError::ShapeOverflow("digit-row partial count"))?;
                let partial_bytes = partial_count
                    .checked_mul(size_of::<Fp128Limbs>())
                    .ok_or(MetalCommitError::ShapeOverflow("digit-row partial bytes"))?;
                let partials = self.private_buffer(partial_bytes)?;
                (digit_buffer, output, partials, output_count)
            };
            let buffer_setup = buffer_start.elapsed();

            let (command_wall, gpu) = {
                let span = tracing::info_span!(
                    "MetalDigitRows::command",
                    num_vectors = expected_vector_count,
                    num_cols = expected_vector_width,
                    num_rows = expected_row_count,
                    ring_dimension = D,
                );
                let _entered = span.enter();
                let command = self.queue.new_command_buffer();
                command.set_label("Akita fp128 D64 digit rows");
                let encoder = command.new_compute_command_encoder();
                encoder.set_label("Akita fp128 D64 digit-row partials");
                encoder.set_compute_pipeline_state(&self.fp128_d64_digit_rows_partials_pipeline);
                encoder.set_buffer(0, Some(matrix), 0);
                encoder.set_buffer(1, Some(&digit_buffer), 0);
                encoder.set_buffer(2, Some(&partials), 0);
                set_inline_bytes(encoder, 3, &params);
                encoder.dispatch_thread_groups(
                    MTLSize::new(
                        params.num_vectors * params.num_rows * params.column_partials,
                        1,
                        1,
                    ),
                    MTLSize::new(FP128_D64_DIGIT_ROWS_PARTIAL_THREADS as u64, 1, 1),
                );
                encoder.end_encoding();
                let encoder = command.new_compute_command_encoder();
                encoder.set_label("Akita fp128 D64 digit-row reduction");
                encoder.set_compute_pipeline_state(&self.fp128_d64_digit_rows_reduce_pipeline);
                encoder.set_buffer(0, Some(&partials), 0);
                encoder.set_buffer(1, Some(&output), 0);
                set_inline_bytes(encoder, 2, &params);
                encoder.dispatch_thread_groups(
                    MTLSize::new(total_output, 1, 1),
                    MTLSize::new(FP128_D64_DIGIT_ROWS_THREADS as u64, 1, 1),
                );
                encoder.end_encoding();
                complete_command(command)?
            };

            let readback_start = Instant::now();
            let coefficients = {
                let span = tracing::info_span!(
                    "MetalDigitRows::readback",
                    output_count,
                    output_bytes = output_count * size_of::<Fp128Limbs>(),
                );
                let _entered = span.enter();
                // SAFETY: `output` is live shared storage for exactly `output_count`
                // aligned `Fp128Limbs` values.
                unsafe {
                    std::slice::from_raw_parts(output.contents().cast::<Fp128Limbs>(), output_count)
                        .to_vec()
                }
            };
            Ok(DigitRowsDispatchOutcome {
                coefficients,
                allocation_bytes: akita_error::checked::sum([
                    digit_buffer.length() as usize,
                    output.length() as usize,
                    partials.length() as usize,
                ])
                .ok_or(MetalCommitError::ShapeOverflow(
                    "digit-row allocation bytes",
                ))?,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy: readback_start.elapsed(),
                },
            })
        })
    }

    pub(crate) fn dispatch_fp128_d512_linear_relation<const D: usize>(
        &self,
        matrix: &Buffer,
        rhs: &[[i32; D]],
        params: D512LinearRelationParams,
    ) -> Result<D512LinearRelationDispatchOutcome, MetalCommitError> {
        autoreleasepool(|| {
            let num_columns = usize::try_from(params.num_columns)
                .map_err(|_| MetalCommitError::ShapeOverflow("D512 relation columns"))?;
            let num_tiles = usize::try_from(params.num_tiles)
                .map_err(|_| MetalCommitError::ShapeOverflow("D512 relation tiles"))?;
            let expected_tiles = num_columns.div_ceil(FP128_D512_LINEAR_RELATION_COLUMNS_PER_TILE);
            let expected_matrix_bytes = num_columns
                .checked_mul(512)
                .and_then(|count| count.checked_mul(size_of::<Fp128Limbs>()))
                .ok_or(MetalCommitError::ShapeOverflow("D512 relation matrix"))?;
            if D != 512
                || rhs.len() != num_columns
                || params.columns_per_tile != FP128_D512_LINEAR_RELATION_COLUMNS_PER_TILE as u64
                || num_tiles != expected_tiles
                || params.num_primes != FP128_D512_LINEAR_RELATION_NUM_PRIMES as u64
                || params.ntt_size != FP128_D512_LINEAR_RELATION_NTT_SIZE as u64
                || params.output_coefficients != 512
                || matrix.length() < expected_matrix_bytes as u64
                || !self.supports_fp128_d512_linear_relation(num_columns, params.rhs_abs_bound)
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "fp128 D512 linear relation exceeds the exact CRT or device limits".into(),
                ));
            }

            let buffer_start = Instant::now();
            let rhs_buffer = self.shared_slice_buffer(rhs)?;
            let partial_count = num_tiles
                .checked_mul(FP128_D512_LINEAR_RELATION_NUM_PRIMES)
                .and_then(|count| count.checked_mul(FP128_D512_LINEAR_RELATION_NTT_SIZE))
                .ok_or(MetalCommitError::ShapeOverflow("D512 relation partials"))?;
            let partial_bytes = partial_count.checked_mul(size_of::<i32>()).ok_or(
                MetalCommitError::ShapeOverflow("D512 relation partial bytes"),
            )?;
            let partials = self.private_buffer(partial_bytes)?;
            let residue_count = FP128_D512_LINEAR_RELATION_NUM_PRIMES
                .checked_mul(FP128_D512_LINEAR_RELATION_NTT_SIZE)
                .ok_or(MetalCommitError::ShapeOverflow("D512 relation residues"))?;
            let residue_bytes = residue_count.checked_mul(size_of::<u32>()).ok_or(
                MetalCommitError::ShapeOverflow("D512 relation residue bytes"),
            )?;
            let residues = self.private_buffer(residue_bytes)?;
            let output_bytes = 512usize
                .checked_mul(size_of::<Fp128Limbs>())
                .ok_or(MetalCommitError::ShapeOverflow("D512 relation output"))?;
            let output = self.shared_buffer(output_bytes)?;
            let buffer_setup = buffer_start.elapsed();

            let resources = &self.fp128_d512_linear_relation_resources;
            let command = self.queue.new_command_buffer();
            command.set_label("Akita fp128 D512 linear relation");
            let encoder = command.new_compute_command_encoder();
            encoder.set_label("Akita D512 linear relation tiled NTT");
            encoder.set_compute_pipeline_state(&self.fp128_d512_linear_relation_partials_pipeline);
            encoder.set_buffer(0, Some(matrix), 0);
            encoder.set_buffer(1, Some(&rhs_buffer.buffer), 0);
            encoder.set_buffer(2, Some(&partials), 0);
            encoder.set_buffer(3, Some(&resources.primes), 0);
            encoder.set_buffer(4, Some(&resources.limb_weights), 0);
            encoder.set_buffer(5, Some(&resources.field_moduli), 0);
            encoder.set_buffer(6, Some(&resources.fwd_twiddles), 0);
            set_inline_bytes(encoder, 7, &params);
            encoder.dispatch_thread_groups(
                MTLSize::new(params.num_tiles * params.num_primes, 1, 1),
                MTLSize::new(FP128_D512_LINEAR_RELATION_THREADS as u64, 1, 1),
            );
            encoder.end_encoding();

            let encoder = command.new_compute_command_encoder();
            encoder.set_label("Akita D512 linear relation reduction");
            encoder.set_compute_pipeline_state(&self.fp128_d512_linear_relation_reduce_pipeline);
            encoder.set_buffer(0, Some(&partials), 0);
            encoder.set_buffer(1, Some(&residues), 0);
            encoder.set_buffer(2, Some(&resources.primes), 0);
            encoder.set_buffer(3, Some(&resources.inv_twiddles), 0);
            encoder.set_buffer(4, Some(&resources.d_inv), 0);
            set_inline_bytes(encoder, 5, &params);
            encoder.dispatch_thread_groups(
                MTLSize::new(params.num_primes, 1, 1),
                MTLSize::new(FP128_D512_LINEAR_RELATION_THREADS as u64, 1, 1),
            );
            encoder.end_encoding();

            let encoder = command.new_compute_command_encoder();
            encoder.set_label("Akita D512 linear relation CRT reconstruction");
            encoder
                .set_compute_pipeline_state(&self.fp128_d512_linear_relation_reconstruct_pipeline);
            encoder.set_buffer(0, Some(&residues), 0);
            encoder.set_buffer(1, Some(&output), 0);
            encoder.set_buffer(2, Some(&resources.primes), 0);
            encoder.set_buffer(3, Some(&resources.garner_gamma), 0);
            encoder.set_buffer(4, Some(&resources.field_partial_products), 0);
            set_inline_bytes(encoder, 5, &params);
            encoder.dispatch_thread_groups(
                MTLSize::new(1, 1, 1),
                MTLSize::new(FP128_D512_LINEAR_RELATION_THREADS as u64, 1, 1),
            );
            encoder.end_encoding();
            let (command_wall, gpu) = complete_command(command)?;

            let readback_start = Instant::now();
            // SAFETY: the shared output contains exactly 512 initialized fp128 limbs.
            let coefficients = unsafe {
                std::slice::from_raw_parts(output.contents().cast::<Fp128Limbs>(), 512).to_vec()
            };
            let readback_copy = readback_start.elapsed();
            let rhs_bytes = size_of_val(rhs);
            let allocation_bytes = partial_bytes
                .checked_add(residue_bytes)
                .and_then(|bytes| bytes.checked_add(output_bytes))
                .and_then(|bytes| {
                    bytes.checked_add(if rhs_buffer.zero_copy { 0 } else { rhs_bytes })
                })
                .ok_or(MetalCommitError::ShapeOverflow(
                    "D512 relation allocation bytes",
                ))?;
            Ok(D512LinearRelationDispatchOutcome {
                coefficients,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy,
                },
                allocation_bytes,
            })
        })
    }
    pub(crate) fn prepare_fp128_recursive_commit_matrix<const D: usize>(
        &self,
        matrix: &Buffer,
        params: RecursiveCommitParams,
    ) -> Result<RecursiveCommitMatrixNttOutcome, MetalCommitError> {
        autoreleasepool(|| {
            let num_rows = usize::try_from(params.num_rows)
                .map_err(|_| MetalCommitError::ShapeOverflow("recursive commit rows"))?;
            let num_cols = usize::try_from(params.num_cols)
                .map_err(|_| MetalCommitError::ShapeOverflow("recursive commit columns"))?;
            let expected_matrix_rings =
                num_rows
                    .checked_mul(num_cols)
                    .ok_or(MetalCommitError::ShapeOverflow(
                        "recursive commit matrix rings",
                    ))?;
            let expected_matrix_bytes = expected_matrix_rings
                .checked_mul(D)
                .and_then(|count| count.checked_mul(size_of::<Fp128Limbs>()))
                .ok_or(MetalCommitError::ShapeOverflow(
                    "recursive commit matrix bytes",
                ))?;
            if params.ring_d != D as u64
                || params.num_primes != FP128_D512_LINEAR_RELATION_NUM_PRIMES as u64
                || params.matrix_rings != expected_matrix_rings as u64
                || matrix.length() < expected_matrix_bytes as u64
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "fp128 recursive commitment matrix geometry is unsupported".into(),
                ));
            }
            let resources = self.recursive_commit_resources(D).ok_or_else(|| {
                MetalCommitError::UnsupportedShape(format!(
                    "no recursive commitment resources for D={D}"
                ))
            })?;

            let buffer_start = Instant::now();
            let matrix_ntt_count = expected_matrix_rings
                .checked_mul(D)
                .and_then(|count| count.checked_mul(FP128_D512_LINEAR_RELATION_NUM_PRIMES))
                .ok_or(MetalCommitError::ShapeOverflow(
                    "recursive commit transformed matrix",
                ))?;
            let matrix_ntt_bytes = matrix_ntt_count.checked_mul(size_of::<i32>()).ok_or(
                MetalCommitError::ShapeOverflow("recursive commit transformed matrix bytes"),
            )?;
            let matrix_ntt = self.private_buffer(matrix_ntt_bytes)?;
            let buffer_setup = buffer_start.elapsed();

            let command = self.queue.new_command_buffer();
            command.set_label("Akita fp128 recursive commitment matrix NTT");
            let encoder = command.new_compute_command_encoder();
            encoder.set_label("Akita recursive commitment matrix NTT");
            encoder.set_compute_pipeline_state(&self.fp128_recursive_commit_matrix_ntt_pipeline);
            encoder.set_buffer(0, Some(matrix), 0);
            encoder.set_buffer(1, Some(&matrix_ntt), 0);
            encoder.set_buffer(2, Some(&resources.primes), 0);
            encoder.set_buffer(3, Some(&resources.limb_weights), 0);
            encoder.set_buffer(4, Some(&resources.field_moduli), 0);
            encoder.set_buffer(5, Some(&resources.fwd_twiddles), 0);
            encoder.set_buffer(6, Some(&resources.psi_pows), 0);
            set_inline_bytes(encoder, 7, &params);
            encoder.dispatch_thread_groups(
                MTLSize::new(params.matrix_rings * params.num_primes, 1, 1),
                MTLSize::new(params.ring_d, 1, 1),
            );
            encoder.end_encoding();
            let (command_wall, gpu) = complete_command(command)?;
            Ok(RecursiveCommitMatrixNttOutcome {
                buffer: matrix_ntt,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy: Duration::ZERO,
                },
                allocation_bytes: matrix_ntt_bytes,
            })
        })
    }

    pub(crate) fn dispatch_fp128_recursive_commit<const D: usize>(
        &self,
        matrix_ntt: &Buffer,
        digits: &[i8],
        params: RecursiveCommitParams,
    ) -> Result<RecursiveCommitDispatchOutcome, MetalCommitError> {
        autoreleasepool(|| {
            let num_blocks = usize::try_from(params.num_blocks)
                .map_err(|_| MetalCommitError::ShapeOverflow("recursive commit blocks"))?;
            let num_rows = usize::try_from(params.num_rows)
                .map_err(|_| MetalCommitError::ShapeOverflow("recursive commit rows"))?;
            let num_cols = usize::try_from(params.num_cols)
                .map_err(|_| MetalCommitError::ShapeOverflow("recursive commit columns"))?;
            let expected_matrix_rings =
                num_rows
                    .checked_mul(num_cols)
                    .ok_or(MetalCommitError::ShapeOverflow(
                        "recursive commit matrix rings",
                    ))?;
            let expected_source_bytes = num_blocks
                .checked_mul(num_cols)
                .and_then(|count| count.checked_mul(D))
                .ok_or(MetalCommitError::ShapeOverflow(
                    "recursive commit source bytes",
                ))?;
            let expected_output = num_blocks
                .checked_mul(num_rows)
                .and_then(|count| count.checked_mul(D))
                .ok_or(MetalCommitError::ShapeOverflow("recursive commit output"))?;
            let expected_matrix_ntt_bytes = expected_matrix_rings
                .checked_mul(D)
                .and_then(|count| count.checked_mul(FP128_D512_LINEAR_RELATION_NUM_PRIMES))
                .and_then(|count| count.checked_mul(size_of::<i32>()))
                .ok_or(MetalCommitError::ShapeOverflow(
                    "recursive commit transformed matrix bytes",
                ))?;
            let expected_block_groups =
                num_blocks.div_ceil(FP128_RECURSIVE_COMMIT_BLOCKS_PER_GROUP);
            if params.blocks_per_group != FP128_RECURSIVE_COMMIT_BLOCKS_PER_GROUP as u64
                || params.num_block_groups != expected_block_groups as u64
                || params.ring_d != D as u64
                || params.num_primes != FP128_D512_LINEAR_RELATION_NUM_PRIMES as u64
                || params.matrix_rings != expected_matrix_rings as u64
                || params.output_coefficients != expected_output as u64
                || digits.len() < expected_source_bytes
                || matrix_ntt.length() < expected_matrix_ntt_bytes as u64
                || !self.supports_fp128_recursive_commit::<D>(
                    num_blocks,
                    num_rows,
                    num_cols,
                    params.rhs_abs_bound,
                )
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "fp128 recursive commitment exceeds the exact CRT or device limits".into(),
                ));
            }
            let resources = self.recursive_commit_resources(D).ok_or_else(|| {
                MetalCommitError::UnsupportedShape(format!(
                    "no recursive commitment resources for D={D}"
                ))
            })?;

            let buffer_start = Instant::now();
            let digit_buffer = self.shared_slice_buffer(digits)?;
            let residue_count = expected_output
                .checked_mul(FP128_D512_LINEAR_RELATION_NUM_PRIMES)
                .ok_or(MetalCommitError::ShapeOverflow("recursive commit residues"))?;
            let residue_bytes = residue_count.checked_mul(size_of::<u32>()).ok_or(
                MetalCommitError::ShapeOverflow("recursive commit residue bytes"),
            )?;
            let residues = self.private_buffer(residue_bytes)?;
            let output_bytes = expected_output.checked_mul(size_of::<Fp128Limbs>()).ok_or(
                MetalCommitError::ShapeOverflow("recursive commit output bytes"),
            )?;
            let output = self.shared_buffer(output_bytes)?;
            let buffer_setup = buffer_start.elapsed();

            let command = self.queue.new_command_buffer();
            command.set_label("Akita fp128 recursive witness commitment");
            let encoder = command.new_compute_command_encoder();
            encoder.set_label("Akita recursive commitment exact matvec");
            encoder.set_compute_pipeline_state(&self.fp128_recursive_commit_matvec_pipeline);
            encoder.set_buffer(0, Some(&digit_buffer.buffer), 0);
            encoder.set_buffer(1, Some(matrix_ntt), 0);
            encoder.set_buffer(2, Some(&residues), 0);
            encoder.set_buffer(3, Some(&resources.primes), 0);
            encoder.set_buffer(4, Some(&resources.fwd_twiddles), 0);
            encoder.set_buffer(5, Some(&resources.inv_twiddles), 0);
            encoder.set_buffer(6, Some(&resources.psi_pows), 0);
            encoder.set_buffer(7, Some(&resources.inverse_scale), 0);
            set_inline_bytes(encoder, 8, &params);
            encoder.dispatch_thread_groups(
                MTLSize::new(params.num_block_groups * params.num_primes, 1, 1),
                MTLSize::new(FP128_RECURSIVE_COMMIT_THREADS as u64, 1, 1),
            );
            encoder.end_encoding();

            let encoder = command.new_compute_command_encoder();
            encoder.set_label("Akita recursive commitment CRT reconstruction");
            encoder.set_compute_pipeline_state(&self.fp128_recursive_commit_reconstruct_pipeline);
            encoder.set_buffer(0, Some(&residues), 0);
            encoder.set_buffer(1, Some(&output), 0);
            encoder.set_buffer(2, Some(&resources.primes), 0);
            encoder.set_buffer(3, Some(&resources.garner_gamma), 0);
            encoder.set_buffer(4, Some(&resources.field_partial_products), 0);
            set_inline_bytes(encoder, 5, &params);
            encoder.dispatch_thread_groups(
                MTLSize::new(
                    params
                        .output_coefficients
                        .div_ceil(FP128_RECURSIVE_COMMIT_RECONSTRUCT_THREADS as u64),
                    1,
                    1,
                ),
                MTLSize::new(FP128_RECURSIVE_COMMIT_RECONSTRUCT_THREADS as u64, 1, 1),
            );
            encoder.end_encoding();
            let (command_wall, gpu) = complete_command(command)?;

            let readback_start = Instant::now();
            // SAFETY: the shared output contains exactly `expected_output`
            // initialized, aligned fp128 limb values.
            let coefficients = unsafe {
                std::slice::from_raw_parts(output.contents().cast::<Fp128Limbs>(), expected_output)
                    .to_vec()
            };
            let readback_copy = readback_start.elapsed();
            let allocation_bytes = residue_bytes
                .checked_add(output_bytes)
                .and_then(|bytes| {
                    bytes.checked_add(if digit_buffer.zero_copy {
                        0
                    } else {
                        size_of_val(digits)
                    })
                })
                .ok_or(MetalCommitError::ShapeOverflow(
                    "recursive commit allocation bytes",
                ))?;
            Ok(RecursiveCommitDispatchOutcome {
                coefficients,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy,
                },
                allocation_bytes,
            })
        })
    }
}
