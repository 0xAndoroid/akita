use akita_algebra::CyclotomicRing;

use super::*;

/// Keep in sync with `RecursiveDecomposeFoldParams` in `kernels/recursive_fold.metal`.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct RecursiveDecomposeFoldParams {
    pub(crate) source_rings: u64,
    pub(crate) positions: u64,
    pub(crate) blocks: u64,
    pub(crate) digits: u64,
    pub(crate) ring_d: u32,
    pub(crate) log_basis: u32,
    pub(crate) output_coefficients: u64,
    pub(crate) threshold: Fp128Limbs,
}

const _: [(); 64] = [(); size_of::<RecursiveDecomposeFoldParams>()];

impl MetalRuntime {
    pub(crate) fn dispatch_recursive_decompose_fold<const D: usize>(
        &self,
        source: &[CyclotomicRing<F, D>],
        offsets: &[u32],
        positions: &[u32],
        coefficients: &[i8],
        params: RecursiveDecomposeFoldParams,
    ) -> Result<(Vec<[i32; D]>, DispatchTimings, usize), MetalCommitError> {
        autoreleasepool(|| {
            let source_coefficients =
                source
                    .len()
                    .checked_mul(D)
                    .ok_or(MetalCommitError::ShapeOverflow(
                        "recursive decompose source",
                    ))?;
            let digit_bytes = source_coefficients
                .checked_mul(params.digits as usize)
                .and_then(|count| count.checked_mul(size_of::<i16>()))
                .ok_or(MetalCommitError::ShapeOverflow("recursive digit planes"))?;
            let output_count = params.output_coefficients as usize;
            let output_bytes = output_count
                .checked_mul(size_of::<i32>())
                .ok_or(MetalCommitError::ShapeOverflow("recursive fold output"))?;
            let buffer_start = Instant::now();
            let source_buffer = self.shared_slice_buffer(source)?;
            let digit_buffer = self.private_buffer(digit_bytes)?;
            let offsets = self.shared_buffer_from_slice(offsets)?;
            let positions = self.shared_buffer_from_slice(positions)?;
            let coefficients = self.shared_buffer_from_slice(coefficients)?;
            let output = self.shared_buffer(output_bytes)?;
            let buffer_setup = buffer_start.elapsed();
            let command = self.queue.new_command_buffer();
            command.set_label("Akita recursive decompose fold");
            let encoder = command.new_compute_command_encoder();
            encoder.set_label("Akita recursive balanced decomposition");
            encoder.set_compute_pipeline_state(&self.fp128_recursive_decompose_pipeline);
            encoder.set_buffer(0, Some(&source_buffer.buffer), 0);
            encoder.set_buffer(1, Some(&digit_buffer), 0);
            set_inline_bytes(encoder, 2, &params);
            encoder.dispatch_threads(
                MTLSize::new(source_coefficients as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
            encoder.end_encoding();
            let encoder = command.new_compute_command_encoder();
            encoder.set_label("Akita recursive sparse digit fold");
            encoder.set_compute_pipeline_state(&self.recursive_sparse_digit_fold_pipeline);
            encoder.set_buffer(0, Some(&digit_buffer), 0);
            encoder.set_buffer(1, Some(&offsets), 0);
            encoder.set_buffer(2, Some(&positions), 0);
            encoder.set_buffer(3, Some(&coefficients), 0);
            encoder.set_buffer(4, Some(&output), 0);
            set_inline_bytes(encoder, 5, &params);
            encoder.dispatch_threads(
                MTLSize::new(params.output_coefficients, 1, 1),
                MTLSize::new(256, 1, 1),
            );
            encoder.end_encoding();
            let (command_wall, gpu) = complete_command(command)?;
            let readback_start = Instant::now();
            // SAFETY: the completed fold wrote every coefficient in the checked
            // output allocation; [i32; D] has the same alignment as i32.
            let centered = unsafe {
                std::slice::from_raw_parts(output.contents().cast::<[i32; D]>(), output_count / D)
                    .to_vec()
            };
            Ok((
                centered,
                DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy: readback_start.elapsed(),
                },
                digit_bytes + output_bytes,
            ))
        })
    }
}
