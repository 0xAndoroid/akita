use akita_error::checked::{product, sum};

use super::*;

/// Keep in sync with `CenteredDigitsParams` in `kernels/ring_switch.metal`.
#[repr(C)]
#[derive(Clone, Copy)]
struct CenteredDigitsParams {
    source_coefficients: u64,
    ring_d: u32,
    digits: u32,
    log_basis: u32,
    padding: u32,
}

const _: [(); 24] = [(); size_of::<CenteredDigitsParams>()];

impl MetalRuntime {
    pub(crate) fn dispatch_centered_relation_rows<const D: usize>(
        &self,
        matrix: &Buffer,
        centered: &[[i32; D]],
        num_rows: usize,
    ) -> Result<DigitRowsDispatchOutcome, MetalCommitError> {
        autoreleasepool(|| {
            const COLUMNS_PER_PARTIAL: usize = 256;
            let columns = centered.len();
            let column_partials = columns.div_ceil(COLUMNS_PER_PARTIAL);
            let output_count = product([num_rows, D])
                .ok_or(MetalCommitError::ShapeOverflow("centered row outputs"))?;
            let groups = product([num_rows, column_partials])
                .ok_or(MetalCommitError::ShapeOverflow("centered row groups"))?;
            let matrix_bytes = product([output_count, columns, size_of::<Fp128Limbs>()])
                .ok_or(MetalCommitError::ShapeOverflow("centered row matrix"))?;
            let partial_bytes = product([output_count, column_partials, size_of::<Fp128Limbs>()])
                .ok_or(MetalCommitError::ShapeOverflow("centered row partials"))?;
            let output_bytes = product([output_count, size_of::<Fp128Limbs>()])
                .ok_or(MetalCommitError::ShapeOverflow("centered row output bytes"))?;
            if !matches!(D, 64 | 128)
                || columns == 0
                || num_rows == 0
                || columns > u32::MAX as usize
                || groups > u32::MAX as usize
                || output_count > u32::MAX as usize
                || matrix.length() < matrix_bytes as u64
                || column_partials.div_ceil(256) > i32::MAX as usize / u16::MAX as usize
                || [
                    matrix_bytes,
                    partial_bytes,
                    output_bytes,
                    size_of_val(centered),
                ]
                .iter()
                .any(|&bytes| bytes as u64 > self.device.max_buffer_length())
                || self
                    .fp128_centered_rows_partials_pipeline
                    .max_total_threads_per_threadgroup()
                    < D as u64
                || self
                    .fp128_relation_rows_reduce_pipeline
                    .max_total_threads_per_threadgroup()
                    < 256
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "centered relation rows exceed device limits".into(),
                ));
            }
            let params = DigitRowsParams {
                num_vectors: 1,
                num_rows: num_rows as u64,
                num_cols: columns as u64,
                ring_d: D as u64,
                output_coefficients: output_count as u64,
                columns_per_partial: COLUMNS_PER_PARTIAL as u64,
                column_partials: column_partials as u64,
                retain_quotients: 0,
                cyclic: 0,
            };
            let buffer_start = Instant::now();
            let source = self.shared_slice_buffer(centered)?;
            let partials = self.private_buffer(partial_bytes)?;
            let output = self.shared_buffer(output_bytes)?;
            let buffer_setup = buffer_start.elapsed();
            let command = self.queue.new_command_buffer();
            command.set_label("Akita centered relation rows");
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.fp128_centered_rows_partials_pipeline);
            encoder.set_buffer(0, Some(matrix), 0);
            encoder.set_buffer(1, Some(&source.buffer), 0);
            encoder.set_buffer(2, Some(&partials), 0);
            set_inline_bytes(encoder, 3, &params);
            encoder.dispatch_thread_groups(
                MTLSize::new(groups as u64, 1, 1),
                MTLSize::new(D as u64, 1, 1),
            );
            encoder.end_encoding();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.fp128_relation_rows_reduce_pipeline);
            encoder.set_buffer(0, Some(&partials), 0);
            encoder.set_buffer(1, Some(&output), 0);
            set_inline_bytes(encoder, 2, &params);
            encoder.dispatch_thread_groups(
                MTLSize::new(output_count as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
            encoder.end_encoding();
            let (command_wall, gpu) = complete_command(command)?;
            let readback_start = Instant::now();
            // SAFETY: the completed reduction initialized all `output_count` shared values.
            let coefficients = unsafe {
                std::slice::from_raw_parts(output.contents().cast::<Fp128Limbs>(), output_count)
                    .to_vec()
            };
            Ok(DigitRowsDispatchOutcome {
                coefficients,
                allocation_bytes: sum([
                    partial_bytes,
                    output_bytes,
                    if source.zero_copy {
                        0
                    } else {
                        size_of_val(centered)
                    },
                ])
                .ok_or(MetalCommitError::ShapeOverflow("centered row allocations"))?,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy: readback_start.elapsed(),
                },
            })
        })
    }

    pub(crate) fn dispatch_centered_digit_planes<const D: usize>(
        &self,
        centered: &[i32],
        num_digits: usize,
        log_basis: u32,
    ) -> Result<(Vec<[i8; D]>, DispatchTimings, usize), MetalCommitError> {
        autoreleasepool(|| {
            if !matches!(D, 64 | 128)
                || centered.is_empty()
                || !centered.len().is_multiple_of(D)
                || centered.len() > u32::MAX as usize
                || !(1..=8).contains(&log_basis)
                || num_digits == 0
                || num_digits > (128 + log_basis) as usize / log_basis as usize
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "invalid centered digit geometry".into(),
                ));
            }
            let output_bytes = product([centered.len(), num_digits])
                .ok_or(MetalCommitError::ShapeOverflow("centered digit output"))?;
            if output_bytes as u64 > self.device.max_buffer_length()
                || size_of_val(centered) as u64 > self.device.max_buffer_length()
                || self
                    .centered_digit_planes_pipeline
                    .max_total_threads_per_threadgroup()
                    < 256
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "centered digits exceed device limits".into(),
                ));
            }
            let buffer_start = Instant::now();
            let source = self.shared_slice_buffer(centered)?;
            let mut planes = vec![[0i8; D]; output_bytes / D];
            let zero_copy = planes
                .as_ptr()
                .addr()
                .is_multiple_of(PACKED_ONEHOT_BUFFER_ALIGNMENT)
                && output_bytes.is_multiple_of(PACKED_ONEHOT_BUFFER_ALIGNMENT);
            let output = if zero_copy {
                self.device.new_buffer_with_bytes_no_copy(
                    planes.as_mut_ptr().cast::<c_void>(),
                    output_bytes as u64,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
            } else {
                self.shared_buffer(output_bytes)?
            };
            let params = CenteredDigitsParams {
                source_coefficients: centered.len() as u64,
                ring_d: D as u32,
                digits: num_digits as u32,
                log_basis,
                padding: 0,
            };
            let buffer_setup = buffer_start.elapsed();
            let command = self.queue.new_command_buffer();
            command.set_label("Akita centered Z digit planes");
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.centered_digit_planes_pipeline);
            encoder.set_buffer(0, Some(&source.buffer), 0);
            encoder.set_buffer(1, Some(&output), 0);
            set_inline_bytes(encoder, 2, &params);
            encoder.dispatch_threads(
                MTLSize::new(centered.len() as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
            encoder.end_encoding();
            let (command_wall, gpu) = complete_command(command)?;
            let readback_start = Instant::now();
            if !zero_copy {
                // SAFETY: completion initialized exactly the checked output allocation.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        output.contents().cast::<i8>(),
                        planes.as_mut_ptr().cast::<i8>(),
                        output_bytes,
                    );
                }
            }
            Ok((
                planes,
                DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy: readback_start.elapsed(),
                },
                sum([
                    output_bytes,
                    if source.zero_copy {
                        0
                    } else {
                        size_of_val(centered)
                    },
                ])
                .ok_or(MetalCommitError::ShapeOverflow(
                    "centered digit allocations",
                ))?,
            ))
        })
    }
}
