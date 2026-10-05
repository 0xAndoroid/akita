use super::*;

impl MetalRuntime {
    pub(crate) fn dispatch_fp128_i8_coefficient_packing(
        &self,
        sources: &[&[i8]],
        combined_weights: &[Fp128Limbs],
        params: I8CoefficientPackingParams,
    ) -> Result<CoefficientPackingDispatchOutcome, MetalCommitError> {
        autoreleasepool(|| {
            let source_count = usize::try_from(params.num_sources)
                .map_err(|_| MetalCommitError::ShapeOverflow("coefficient-packing sources"))?;
            let source_coefficients = usize::try_from(params.source_coefficients)
                .map_err(|_| MetalCommitError::ShapeOverflow("coefficient-packing source width"))?;
            let weight_count = params
                .positions_per_block
                .checked_mul(params.stride)
                .ok_or(MetalCommitError::ShapeOverflow(
                    "coefficient-packing weights",
                ))?;
            let expected_output = params
                .num_sources
                .checked_mul(params.num_blocks)
                .and_then(|count| count.checked_mul(params.subring_dimension))
                .ok_or(MetalCommitError::ShapeOverflow(
                    "coefficient-packing output",
                ))?;
            if source_count == 0
                || source_coefficients == 0
                || sources.len() != source_count
                || sources
                    .iter()
                    .any(|source| source.len() != source_coefficients)
                || params.live_coefficients > params.source_coefficients
                || params.ring_d == 0
                || params.stride == 0
                || params.subring_dimension == 0
                || params
                    .stride
                    .checked_mul(params.subring_dimension)
                    .is_none_or(|dimension| dimension != params.ring_d)
                || params.num_live_positions == 0
                || params.positions_per_block == 0
                || params.num_blocks == 0
                || params.output_coefficients != expected_output
                || params.output_coefficients > u64::from(u32::MAX)
                || u64::try_from(combined_weights.len()).ok() != Some(weight_count)
                || self
                    .fp128_i8_coefficient_packing_pipeline
                    .max_total_threads_per_threadgroup()
                    < FP128_COEFFICIENT_PACKING_THREADS as u64
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "fp128 i8 coefficient-packing shape is unsupported".into(),
                ));
            }

            let buffer_start = Instant::now();
            let source = self.shared_byte_buffer_from_slices(sources)?;
            let weights = self.shared_buffer_from_slice(combined_weights)?;
            let output_count = usize::try_from(params.output_coefficients)
                .map_err(|_| MetalCommitError::ShapeOverflow("coefficient-packing output"))?;
            let output_bytes = output_count.checked_mul(size_of::<Fp128Limbs>()).ok_or(
                MetalCommitError::ShapeOverflow("coefficient-packing output bytes"),
            )?;
            let output = self.shared_buffer(output_bytes)?;
            let buffer_setup = buffer_start.elapsed();

            let command = self.queue.new_command_buffer();
            command.set_label("Akita fp128 i8 coefficient packing");
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.fp128_i8_coefficient_packing_pipeline);
            encoder.set_buffer(0, Some(&source.buffer), 0);
            encoder.set_buffer(1, Some(&weights), 0);
            encoder.set_buffer(2, Some(&output), 0);
            set_inline_bytes(encoder, 3, &params);
            encoder.dispatch_thread_groups(
                MTLSize::new(params.output_coefficients, 1, 1),
                MTLSize::new(FP128_COEFFICIENT_PACKING_THREADS as u64, 1, 1),
            );
            encoder.end_encoding();
            let (command_wall, gpu) = complete_command(command)?;

            let readback_start = Instant::now();
            // SAFETY: `output` is live shared storage for `output_count`
            // aligned canonical limb values written by the completed command.
            let coefficients = unsafe {
                std::slice::from_raw_parts(output.contents().cast::<Fp128Limbs>(), output_count)
                    .to_vec()
            };
            let source_bytes = source_count.checked_mul(source_coefficients).ok_or(
                MetalCommitError::ShapeOverflow("coefficient-packing input bytes"),
            )?;
            let allocation_bytes = output_bytes
                .checked_add(size_of_val(combined_weights))
                .and_then(|bytes| {
                    bytes.checked_add(if source.zero_copy { 0 } else { source_bytes })
                })
                .ok_or(MetalCommitError::ShapeOverflow(
                    "coefficient-packing allocation bytes",
                ))?;
            Ok(CoefficientPackingDispatchOutcome {
                coefficients,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy: readback_start.elapsed(),
                },
                allocation_bytes,
            })
        })
    }

    pub(crate) fn dispatch_fp128_packed_onehot_coefficient_packing(
        &self,
        lanes: &[u8],
        active_zero_rows: &[u64],
        combined_weights: &[Fp128Limbs],
        params: PackedOneHotCoefficientPackingParams,
    ) -> Result<CoefficientPackingDispatchOutcome, MetalCommitError> {
        autoreleasepool(|| {
            let expected_lanes = params.num_rows.checked_mul(params.num_columns).ok_or(
                MetalCommitError::ShapeOverflow("packed coefficient-packing lanes"),
            )?;
            let expected_blocks = params
                .column_capacity
                .checked_mul(params.blocks_per_column)
                .ok_or(MetalCommitError::ShapeOverflow(
                    "packed coefficient-packing blocks",
                ))?;
            let expected_output = params
                .num_blocks
                .checked_mul(params.subring_dimension)
                .ok_or(MetalCommitError::ShapeOverflow(
                    "packed coefficient-packing output",
                ))?;
            let expected_weights = params
                .positions_per_block
                .checked_mul(params.stride)
                .ok_or(MetalCommitError::ShapeOverflow(
                    "packed coefficient-packing weights",
                ))?;
            let expected_row_partials = params
                .rows_per_block
                .div_ceil(params.rows_per_partial.max(1));
            let expected_partials = params
                .num_blocks
                .checked_mul(params.row_partials_per_block)
                .and_then(|count| count.checked_mul(params.subring_dimension))
                .ok_or(MetalCommitError::ShapeOverflow(
                    "packed coefficient-packing partials",
                ))?;
            let live_column_mask = if params.num_columns >= u64::BITS as u64 {
                u64::MAX
            } else {
                (1u64 << params.num_columns) - 1
            };
            let expected_active_zero_words = params.num_rows.div_ceil(u64::BITS as u64);
            if params.num_rows == 0
                || params.num_columns == 0
                || !params.column_capacity.is_power_of_two()
                || params.num_columns > params.column_capacity
                || params.onehot_k == 0
                || !params.ring_d.is_power_of_two()
                || params.positions_per_block == 0
                || params.blocks_per_column == 0
                || params.rows_per_block == 0
                || params.rows_per_partial
                    != FP128_PACKED_COEFFICIENT_PACKING_ROWS_PER_PARTIAL as u64
                || !params.row_partials_per_block.is_power_of_two()
                || params.row_partials_per_block != expected_row_partials
                || params.stride == 0
                || params.subring_dimension == 0
                || params.subring_dimension > FP128_PACKED_COEFFICIENT_PACKING_REDUCE_THREADS as u64
                || params
                    .stride
                    .checked_mul(params.subring_dimension)
                    .is_none_or(|dimension| dimension != params.ring_d)
                || params.num_blocks != expected_blocks
                || params.output_coefficients != expected_output
                || params.partial_coefficients != expected_partials
                || params.output_coefficients > u64::from(u32::MAX)
                || params
                    .num_blocks
                    .checked_mul(params.row_partials_per_block)
                    .is_none_or(|groups| groups > u64::from(u32::MAX))
                || u64::try_from(lanes.len()).ok() != Some(expected_lanes)
                || params.zero_column_mask & !live_column_mask != 0
                || (params.zero_column_mask == 0 && !active_zero_rows.is_empty())
                || (params.zero_column_mask != 0
                    && u64::try_from(active_zero_rows.len()).ok()
                        != Some(expected_active_zero_words))
                || u64::try_from(combined_weights.len()).ok() != Some(expected_weights)
                || self
                    .fp128_packed_onehot_coefficient_packing_partials_pipeline
                    .max_total_threads_per_threadgroup()
                    < FP128_PACKED_COEFFICIENT_PACKING_PARTIAL_THREADS as u64
                || self
                    .fp128_packed_onehot_coefficient_packing_reduce_pipeline
                    .max_total_threads_per_threadgroup()
                    < FP128_PACKED_COEFFICIENT_PACKING_REDUCE_THREADS as u64
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "fp128 packed one-hot coefficient-packing shape is unsupported".into(),
                ));
            }

            let buffer_start = Instant::now();
            let lane_bytes = lanes.len();
            let lanes = self.packed_lane_buffer(lanes)?;
            let no_active_zero_rows = [0u64];
            let active_zero_rows = if active_zero_rows.is_empty() {
                no_active_zero_rows.as_slice()
            } else {
                active_zero_rows
            };
            let active_zero_bytes = size_of_val(active_zero_rows);
            let active_zero_rows = self.shared_buffer_from_slice(active_zero_rows)?;
            let weights = self.shared_buffer_from_slice(combined_weights)?;
            let output_count = usize::try_from(params.output_coefficients)
                .map_err(|_| MetalCommitError::ShapeOverflow("coefficient-packing output"))?;
            let output_bytes = output_count.checked_mul(size_of::<Fp128Limbs>()).ok_or(
                MetalCommitError::ShapeOverflow("coefficient-packing output bytes"),
            )?;
            let output = self.shared_buffer(output_bytes)?;
            let partial_count = usize::try_from(params.partial_coefficients)
                .map_err(|_| MetalCommitError::ShapeOverflow("coefficient-packing partials"))?;
            let partial_bytes = partial_count.checked_mul(size_of::<Fp128Limbs>()).ok_or(
                MetalCommitError::ShapeOverflow("coefficient-packing partial bytes"),
            )?;
            let partials = self.private_buffer(partial_bytes)?;
            let buffer_setup = buffer_start.elapsed();

            let command = self.queue.new_command_buffer();
            command.set_label("Akita fp128 packed one-hot coefficient packing");
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(
                &self.fp128_packed_onehot_coefficient_packing_partials_pipeline,
            );
            encoder.set_buffer(0, Some(&lanes.buffer), 0);
            encoder.set_buffer(1, Some(&weights), 0);
            encoder.set_buffer(2, Some(&partials), 0);
            encoder.set_buffer(3, Some(&active_zero_rows), 0);
            set_inline_bytes(encoder, 4, &params);
            encoder.dispatch_thread_groups(
                MTLSize::new(params.num_blocks * params.row_partials_per_block, 1, 1),
                MTLSize::new(
                    FP128_PACKED_COEFFICIENT_PACKING_PARTIAL_THREADS as u64,
                    1,
                    1,
                ),
            );
            encoder.end_encoding();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(
                &self.fp128_packed_onehot_coefficient_packing_reduce_pipeline,
            );
            encoder.set_buffer(0, Some(&partials), 0);
            encoder.set_buffer(1, Some(&output), 0);
            set_inline_bytes(encoder, 2, &params);
            encoder.dispatch_thread_groups(
                MTLSize::new(params.num_blocks, 1, 1),
                MTLSize::new(FP128_PACKED_COEFFICIENT_PACKING_REDUCE_THREADS as u64, 1, 1),
            );
            encoder.end_encoding();
            let (command_wall, gpu) = complete_command(command)?;

            let readback_start = Instant::now();
            // SAFETY: the completed command initialized the full shared output.
            let coefficients = unsafe {
                std::slice::from_raw_parts(output.contents().cast::<Fp128Limbs>(), output_count)
                    .to_vec()
            };
            let allocation_bytes = output_bytes
                .checked_add(partial_bytes)
                .and_then(|bytes| bytes.checked_add(size_of_val(combined_weights)))
                .and_then(|bytes| bytes.checked_add(active_zero_bytes))
                .and_then(|bytes| bytes.checked_add(if lanes.zero_copy { 0 } else { lane_bytes }))
                .ok_or(MetalCommitError::ShapeOverflow(
                    "coefficient-packing allocation bytes",
                ))?;
            Ok(CoefficientPackingDispatchOutcome {
                coefficients,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy: readback_start.elapsed(),
                },
                allocation_bytes,
            })
        })
    }

    pub(crate) fn dispatch_fp128_indexed_packed_onehot_coefficient_packing(
        &self,
        index: &PackedFp128D512CoefficientPackingIndex,
        combined_weights: &[Fp128Limbs],
        params: PackedOneHotCoefficientPackingParams,
    ) -> Result<CoefficientPackingDispatchOutcome, MetalCommitError> {
        autoreleasepool(|| {
            let index_params = index.params;
            let expected_blocks = params
                .column_capacity
                .checked_mul(params.blocks_per_column)
                .ok_or(MetalCommitError::ShapeOverflow(
                    "indexed coefficient-packing blocks",
                ))?;
            let expected_output = expected_blocks
                .checked_mul(params.subring_dimension)
                .ok_or(MetalCommitError::ShapeOverflow(
                    "indexed coefficient-packing output",
                ))?;
            let expected_weights = params
                .positions_per_block
                .checked_mul(params.stride)
                .ok_or(MetalCommitError::ShapeOverflow(
                    "indexed coefficient-packing weights",
                ))?;
            let tile_chunks = index_params
                .position_tiles
                .div_ceil(FP128_D512_PACKING_TILES_PER_CHUNK as u64);
            let live_streams = params
                .blocks_per_column
                .checked_mul(params.num_columns)
                .and_then(|count| count.checked_mul(2))
                .ok_or(MetalCommitError::ShapeOverflow(
                    "indexed coefficient-packing live streams",
                ))?;
            let partial_groups =
                live_streams
                    .checked_mul(tile_chunks)
                    .ok_or(MetalCommitError::ShapeOverflow(
                        "indexed coefficient-packing partial groups",
                    ))?;
            let partial_count = live_streams
                .checked_mul(32)
                .and_then(|count| count.checked_mul(tile_chunks))
                .ok_or(MetalCommitError::ShapeOverflow(
                    "indexed coefficient-packing partials",
                ))?;
            if params.num_rows == 0
                || params.num_columns == 0
                || params.num_columns > params.column_capacity
                || params.onehot_k != 256
                || params.ring_d != 512
                || params.stride != 8
                || params.subring_dimension != 64
                || params.positions_per_block == 0
                || params.positions_per_block.checked_mul(2) != Some(params.rows_per_block)
                || params.blocks_per_column == 0
                || params.num_blocks != expected_blocks
                || params.output_coefficients != expected_output
                || params.output_coefficients > u64::from(u32::MAX)
                || partial_groups > u64::from(u32::MAX)
                || u64::try_from(combined_weights.len()).ok() != Some(expected_weights)
                || index_params.num_rows != params.num_rows
                || index_params.num_columns != params.num_columns
                || index_params.lane_stride != params.num_columns
                || index_params.num_positions != params.positions_per_block
                || index_params.blocks_per_column != params.blocks_per_column
                || index_params.position_tiles
                    != params
                        .positions_per_block
                        .div_ceil(FP128_D512_PACKING_INDEX_TILE_POSITIONS as u64)
                || self
                    .fp128_d512_indexed_coefficient_packing_pipeline
                    .max_total_threads_per_threadgroup()
                    < FP128_COEFFICIENT_PACKING_THREADS as u64
                || self
                    .fp128_d512_indexed_coefficient_packing_reduce_pipeline
                    .max_total_threads_per_threadgroup()
                    < FP128_COEFFICIENT_PACKING_THREADS as u64
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "indexed fp128 packed coefficient-packing shape is unsupported".into(),
                ));
            }

            let buffer_start = Instant::now();
            let weights = self.shared_buffer_from_slice(combined_weights)?;
            let output_count = usize::try_from(params.output_coefficients).map_err(|_| {
                MetalCommitError::ShapeOverflow("indexed coefficient-packing output")
            })?;
            let output_bytes = output_count.checked_mul(size_of::<Fp128Limbs>()).ok_or(
                MetalCommitError::ShapeOverflow("indexed coefficient-packing output bytes"),
            )?;
            let output = self.shared_buffer(output_bytes)?;
            let partial_count = usize::try_from(partial_count).map_err(|_| {
                MetalCommitError::ShapeOverflow("indexed coefficient-packing partial count")
            })?;
            let partial_bytes = partial_count.checked_mul(size_of::<Fp128Limbs>()).ok_or(
                MetalCommitError::ShapeOverflow("indexed coefficient-packing partial bytes"),
            )?;
            let partials = self.private_buffer(partial_bytes)?;
            let buffer_setup = buffer_start.elapsed();

            let command = self.queue.new_command_buffer();
            command.set_label("Akita fp128 indexed coefficient packing");
            let encoder = command.new_compute_command_encoder();
            encoder
                .set_compute_pipeline_state(&self.fp128_d512_indexed_coefficient_packing_pipeline);
            encoder.set_buffer(0, Some(&index.records), 0);
            encoder.set_buffer(1, Some(&index.offsets), 0);
            encoder.set_buffer(2, Some(&weights), 0);
            encoder.set_buffer(3, Some(&partials), 0);
            set_inline_bytes(encoder, 4, &index_params);
            set_inline_bytes(encoder, 5, &params);
            encoder.dispatch_thread_groups(
                MTLSize::new(partial_groups, 1, 1),
                MTLSize::new(FP128_COEFFICIENT_PACKING_THREADS as u64, 1, 1),
            );
            encoder.end_encoding();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(
                &self.fp128_d512_indexed_coefficient_packing_reduce_pipeline,
            );
            encoder.set_buffer(0, Some(&partials), 0);
            encoder.set_buffer(1, Some(&output), 0);
            set_inline_bytes(encoder, 2, &index_params);
            set_inline_bytes(encoder, 3, &params);
            encoder.dispatch_threads(
                MTLSize::new(params.output_coefficients, 1, 1),
                MTLSize::new(FP128_COEFFICIENT_PACKING_THREADS as u64, 1, 1),
            );
            encoder.end_encoding();
            let (command_wall, gpu) = complete_command(command)?;

            let readback_start = Instant::now();
            // SAFETY: the completed command initialized exactly `output_count`
            // aligned canonical limb values in shared storage.
            let coefficients = unsafe {
                std::slice::from_raw_parts(output.contents().cast::<Fp128Limbs>(), output_count)
                    .to_vec()
            };
            Ok(CoefficientPackingDispatchOutcome {
                coefficients,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy: readback_start.elapsed(),
                },
                allocation_bytes: output_bytes
                    .checked_add(size_of_val(combined_weights))
                    .and_then(|bytes| bytes.checked_add(partial_bytes))
                    .ok_or(MetalCommitError::ShapeOverflow(
                        "indexed coefficient-packing allocation bytes",
                    ))?,
            })
        })
    }
}
