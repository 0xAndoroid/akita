use super::*;

impl MetalRuntime {
    pub(crate) fn dispatch_onehot(
        &self,
        matrix: &Buffer,
        hot_indices: &[u16],
        mut params: OneHotCommitParams,
        kernel: MetalOneHotKernel,
    ) -> Result<DispatchOutcome, MetalCommitError> {
        autoreleasepool(|| {
            let blocks_per_threadgroup = self.dispatch_geometry(params, kernel)?;
            params.blocks_per_threadgroup = blocks_per_threadgroup as u64;
            let buffer_start = Instant::now();
            let indices = self.shared_buffer_from_slice(hot_indices)?;
            let output_count = usize::try_from(params.output_coefficients)
                .map_err(|_| MetalCommitError::ShapeOverflow("output coefficients"))?;
            if kernel == MetalOneHotKernel::DirectGather
                && params.output_coefficients > u64::from(u32::MAX)
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "one-hot output grid exceeds u32::MAX threads".into(),
                ));
            }
            let output_bytes = output_count
                .checked_mul(size_of::<Fp128Limbs>())
                .ok_or(MetalCommitError::ShapeOverflow("output bytes"))?;
            let output = self.shared_buffer(output_bytes)?;
            let buffer_setup = buffer_start.elapsed();

            let command = self.queue.new_command_buffer();
            command.set_label("Akita one-hot root commitment");
            let encoder = command.new_compute_command_encoder();
            encoder.set_label(match kernel {
                MetalOneHotKernel::DirectGather => "Akita one-hot gather",
                MetalOneHotKernel::BlockBatched => "Akita one-hot block batch",
                MetalOneHotKernel::PackedFp128D512Panels
                | MetalOneHotKernel::PackedFp128D128Rank3 => {
                    return Err(MetalCommitError::UnsupportedShape(
                        "packed kernel requires packed parameters".into(),
                    ));
                }
            });
            self.encode_onehot(encoder, matrix, &indices, &output, &params, kernel)?;
            encoder.end_encoding();
            let (command_wall, gpu) = complete_command(command)?;

            let readback_start = Instant::now();
            // SAFETY: `output` is live shared storage for exactly `output_count`
            // aligned `Fp128Limbs` values.
            let coefficients = unsafe {
                std::slice::from_raw_parts(output.contents().cast::<Fp128Limbs>(), output_count)
                    .to_vec()
            };
            Ok(DispatchOutcome {
                coefficients,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy: readback_start.elapsed(),
                },
                panel_gpu_active: gpu,
                panel_gpu_span: gpu,
                reduction_gpu: None,
                command_buffers: 1,
                kernel,
                blocks_per_threadgroup,
                columns_per_threadgroup: 0,
                matrix_block_streams: 0,
                scratch_bytes: 0,
                input_zero_copy: false,
            })
        })
    }

    pub(crate) fn dispatch_packed_onehot(
        &self,
        matrix: &Buffer,
        lanes: &[u8],
        active_zero_rows: &[u64],
        params: PackedOneHotCommitParams,
        streams_per_command: usize,
    ) -> Result<DispatchOutcome, MetalCommitError> {
        let expected_active_words = params.num_rows.div_ceil(u64::BITS as u64);
        let zero_mask_exceeds_columns = params.num_columns < u64::BITS as u64
            && params.zero_column_mask >> params.num_columns != 0;
        if zero_mask_exceeds_columns
            || (params.zero_column_mask == 0 && !active_zero_rows.is_empty())
            || (params.zero_column_mask != 0
                && u64::try_from(active_zero_rows.len()).ok() != Some(expected_active_words))
        {
            return Err(MetalCommitError::UnsupportedShape(
                "packed committed-zero selector geometry is invalid".into(),
            ));
        }
        self.dispatch_packed_onehot_source(
            matrix,
            &ResidentPackedLanes { lanes },
            active_zero_rows,
            params,
            streams_per_command,
        )
    }

    pub(super) fn dispatch_packed_onehot_source<S: PackedLaneSource>(
        &self,
        matrix: &Buffer,
        source: &S,
        active_zero_rows: &[u64],
        mut params: PackedOneHotCommitParams,
        streams_per_command: usize,
    ) -> Result<DispatchOutcome, MetalCommitError> {
        autoreleasepool(|| {
            let lane_count = params
                .num_rows
                .checked_mul(params.lane_stride)
                .ok_or(MetalCommitError::ShapeOverflow("packed lane count"))?;
            let expected_tasks = params
                .num_columns
                .checked_mul(params.full_blocks_per_column)
                .ok_or(MetalCommitError::ShapeOverflow("packed task count"))?;
            let expected_output = params
                .column_capacity
                .checked_mul(params.blocks_per_column)
                .and_then(|count| count.checked_mul(params.n_a))
                .and_then(|count| count.checked_mul(params.ring_d))
                .ok_or(MetalCommitError::ShapeOverflow("packed output"))?;
            let supported_source = matches!(
                (params.onehot_k, params.column_capacity),
                (16, 64) | (256, 32)
            );
            if lane_count != source.lane_count() as u64
                || params.ring_d != 512
                || !supported_source
                || params.num_columns == 0
                || params.num_columns > params.column_capacity
                || params.lane_stride != params.num_columns
                || params.n_a != 1
                || params.num_digits_inner != 1
                || params.position_partials_per_block != FP128_D512_POSITION_PARTIALS as u64
                || !params.positions_per_partial.is_multiple_of(4)
                || !params.blocks_per_column.is_power_of_two()
                || params.blocks_per_column > 512
                || params.full_blocks_per_column > params.blocks_per_column
                || params.boundary_columns != 0
                || params.num_blocks != expected_tasks
                || params.task_offset != 0
                || params.dispatch_tasks != params.num_blocks
                || params.lane_row_offset != 0
                || params.output_coefficients != expected_output
                || !matches!(streams_per_command, 1 | 5)
                || (streams_per_command == 5 && params.positions_per_block > (1 << 16))
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "packed fp128 D512 dispatch geometry is outside the registered schedule".into(),
                ));
            }
            let expected_matrix_bytes = params
                .n_a
                .checked_mul(params.positions_per_block)
                .and_then(|count| count.checked_mul(params.ring_d))
                .and_then(|count| count.checked_mul(size_of::<Fp128Limbs>() as u64))
                .ok_or(MetalCommitError::ShapeOverflow("packed matrix bytes"))?;
            if matrix.length() < expected_matrix_bytes {
                return Err(MetalCommitError::UnsupportedShape(
                    "packed fp128 D512 matrix prefix is shorter than the plan".into(),
                ));
            }
            if !self.supports_packed_fp128_d512_panels() {
                return Err(MetalCommitError::UnsupportedShape(
                    "packed fp128 D512 pipeline does not expose the registered resources".into(),
                ));
            }
            params.columns_per_threadgroup = 1;
            let base_streams = params
                .num_blocks
                .div_ceil(FP128_D512_TASKS_PER_STREAM as u64);
            let matrix_block_streams = base_streams
                .checked_mul(FP128_D512_COEFFICIENT_BANDS as u64)
                .ok_or(MetalCommitError::ShapeOverflow("coefficient-band streams"))?;
            let threadgroups = params
                .n_a
                .checked_mul(params.position_partials_per_block)
                .and_then(|count| count.checked_mul(matrix_block_streams))
                .ok_or(MetalCommitError::ShapeOverflow("packed threadgroups"))?;
            if threadgroups > u64::from(u32::MAX)
                || params.output_coefficients > u64::from(u32::MAX)
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "packed fp128 D512 grid exceeds u32 indexing".into(),
                ));
            }

            let buffer_start = Instant::now();
            let output_count = usize::try_from(params.output_coefficients)
                .map_err(|_| MetalCommitError::ShapeOverflow("output coefficients"))?;
            let output_bytes = output_count
                .checked_mul(size_of::<Fp128Limbs>())
                .ok_or(MetalCommitError::ShapeOverflow("output bytes"))?;
            let output = self.shared_buffer(output_bytes)?;
            let partial_count = output_count
                .checked_mul(params.position_partials_per_block as usize)
                .ok_or(MetalCommitError::ShapeOverflow("partial coefficients"))?;
            let scratch_bytes = partial_count
                .checked_mul(size_of::<Fp128Limbs>())
                .ok_or(MetalCommitError::ShapeOverflow("partial bytes"))?;
            let partials = self.private_buffer(scratch_bytes)?;
            let no_active_zero_rows = [0u64];
            let active_zero_rows = if active_zero_rows.is_empty() {
                no_active_zero_rows.as_slice()
            } else {
                active_zero_rows
            };
            let active_zero_rows = self.shared_buffer_from_slice(active_zero_rows)?;
            let mut buffer_setup = buffer_start.elapsed();

            let command_start = Instant::now();
            let command_count = base_streams.div_ceil(streams_per_command as u64) as usize;
            let mut commands = Vec::with_capacity(command_count);
            let mut lane_buffers = Vec::with_capacity(command_count);
            let mut input_zero_copy = true;
            for first_stream in (0..base_streams).step_by(streams_per_command) {
                let stream_count = (base_streams - first_stream).min(streams_per_command as u64);
                let task_offset = first_stream * FP128_D512_TASKS_PER_STREAM as u64;
                let mut dispatch_params = params;
                dispatch_params.task_offset = task_offset;
                dispatch_params.dispatch_tasks = (stream_count
                    * FP128_D512_TASKS_PER_STREAM as u64)
                    .min(params.num_blocks - task_offset);
                let final_task = task_offset + dispatch_params.dispatch_tasks - 1;
                let first_block = task_offset / params.num_columns;
                let final_block = final_task / params.num_columns;
                let rows_per_position = params.ring_d / params.onehot_k;
                let rows_per_block = params.positions_per_block * rows_per_position;
                let first_row = first_block * rows_per_block;
                let final_row = (final_block + 1) * rows_per_block;
                let first_row = usize::try_from(first_row)
                    .map_err(|_| MetalCommitError::ShapeOverflow("packed first row"))?;
                let final_row = usize::try_from(final_row)
                    .map_err(|_| MetalCommitError::ShapeOverflow("packed final row"))?;
                let lane_stride = usize::try_from(params.lane_stride)
                    .map_err(|_| MetalCommitError::ShapeOverflow("packed lane stride"))?;
                let command_lanes = source.wait_lanes(first_row..final_row, lane_stride)?;
                let lane_buffer_start = Instant::now();
                let lane_buffer = self.packed_lane_buffer(command_lanes)?;
                buffer_setup += lane_buffer_start.elapsed();
                input_zero_copy &= lane_buffer.zero_copy;
                dispatch_params.lane_row_offset = first_row as u64;
                let command_threadgroups = dispatch_params
                    .n_a
                    .checked_mul(dispatch_params.position_partials_per_block)
                    .and_then(|count| {
                        count.checked_mul(stream_count * FP128_D512_COEFFICIENT_BANDS as u64)
                    })
                    .ok_or(MetalCommitError::ShapeOverflow(
                        "packed command threadgroups",
                    ))?;
                let command = self.queue.new_command_buffer();
                command.set_label("Akita packed fp128 D512 root commitment");
                let encoder = command.new_compute_command_encoder();
                encoder.set_label("Akita packed fp128 D512 coefficient bands");
                encoder.set_compute_pipeline_state(&self.packed_fp128_d512_pipeline);
                encoder.set_buffer(0, Some(matrix), 0);
                encoder.set_buffer(1, Some(&lane_buffer.buffer), 0);
                encoder.set_buffer(2, Some(&partials), 0);
                set_inline_bytes(encoder, 3, &dispatch_params);
                encoder.set_buffer(4, Some(&active_zero_rows), 0);
                encoder.dispatch_thread_groups(
                    MTLSize::new(command_threadgroups, 1, 1),
                    MTLSize::new(FP128_D512_THREADS as u64, 1, 1),
                );
                encoder.end_encoding();
                command.commit();
                commands.push(command);
                lane_buffers.push(lane_buffer);
            }

            let reduction_command = self.queue.new_command_buffer();
            reduction_command.set_label("Akita packed fp128 D512 partial reduction");
            let reduction = reduction_command.new_compute_command_encoder();
            reduction.set_label("Akita packed fp128 D512 partial reduction");
            reduction.set_compute_pipeline_state(&self.packed_partial_reduction_pipeline);
            reduction.set_buffer(0, Some(&partials), 0);
            reduction.set_buffer(1, Some(&output), 0);
            set_inline_bytes(reduction, 2, &params);
            let reduction_width = (self
                .packed_partial_reduction_pipeline
                .thread_execution_width()
                * 4)
            .min(
                self.packed_partial_reduction_pipeline
                    .max_total_threads_per_threadgroup(),
            );
            reduction.dispatch_threads(
                MTLSize::new(params.output_coefficients, 1, 1),
                MTLSize::new(reduction_width, 1, 1),
            );
            reduction.end_encoding();
            reduction_command.commit();
            reduction_command.wait_until_completed();
            let command_wall = command_start.elapsed();
            for command in &commands {
                validate_completed_command(command)?;
            }
            validate_completed_command(reduction_command)?;
            let panel_gpu_active = commands.iter().try_fold(Duration::ZERO, |total, command| {
                total.checked_add(completed_command_gpu_time(command)?)
            });
            let panel_gpu_span = commands
                .first()
                .zip(commands.last())
                .and_then(|(first, last)| completed_commands_gpu_span(first, last));
            let reduction_gpu = completed_command_gpu_time(reduction_command);
            let gpu = commands
                .first()
                .and_then(|first| completed_commands_gpu_span(first, reduction_command));

            let readback_start = Instant::now();
            // SAFETY: `output` is live shared storage for exactly `output_count`
            // aligned `Fp128Limbs` values.
            let coefficients = unsafe {
                std::slice::from_raw_parts(output.contents().cast::<Fp128Limbs>(), output_count)
                    .to_vec()
            };
            Ok(DispatchOutcome {
                coefficients,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy: readback_start.elapsed(),
                },
                panel_gpu_active,
                panel_gpu_span,
                reduction_gpu,
                command_buffers: commands.len() + 1,
                kernel: MetalOneHotKernel::PackedFp128D512Panels,
                blocks_per_threadgroup: FP128_D512_TASKS_PER_STREAM,
                columns_per_threadgroup: 1,
                matrix_block_streams: matrix_block_streams as usize,
                scratch_bytes,
                input_zero_copy,
            })
        })
    }

    /// Dispatch the packed fp128 D128 rank-3 commit: threadgroups own one
    /// matrix element and one position partial for 64 (column, block) tasks.
    pub(crate) fn dispatch_packed_onehot_d128_rank3(
        &self,
        matrix: &Buffer,
        lanes: &[u8],
        active_zero_rows: &[u64],
        mut params: PackedOneHotCommitParams,
        streams_per_command: usize,
    ) -> Result<DispatchOutcome, MetalCommitError> {
        autoreleasepool(|| {
            let expected_active_words = params.num_rows.div_ceil(u64::BITS as u64);
            let zero_mask_exceeds_columns = params.num_columns < u64::BITS as u64
                && params.zero_column_mask >> params.num_columns != 0;
            if zero_mask_exceeds_columns
                || (params.zero_column_mask == 0 && !active_zero_rows.is_empty())
                || (params.zero_column_mask != 0
                    && u64::try_from(active_zero_rows.len()).ok() != Some(expected_active_words))
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "packed committed-zero selector geometry is invalid".into(),
                ));
            }
            let lane_count = params
                .num_rows
                .checked_mul(params.lane_stride)
                .ok_or(MetalCommitError::ShapeOverflow("packed lane count"))?;
            let expected_tasks = params
                .num_columns
                .checked_mul(params.full_blocks_per_column)
                .ok_or(MetalCommitError::ShapeOverflow("packed task count"))?;
            let expected_output = params
                .column_capacity
                .checked_mul(params.blocks_per_column)
                .and_then(|count| count.checked_mul(params.n_a))
                .and_then(|count| count.checked_mul(params.ring_d))
                .ok_or(MetalCommitError::ShapeOverflow("packed output"))?;
            let partial_alignment = FP128_D128_RANK3_POSITION_PARTIAL_ALIGNMENT as u64;
            if lane_count != lanes.len() as u64
                || params.ring_d != FP128_D128_RANK3_RING_D
                || params.onehot_k != 256
                || params.column_capacity != 32
                || params.num_columns == 0
                || params.num_columns > params.column_capacity
                || params.lane_stride != params.num_columns
                || params.n_a != FP128_D128_RANK3_INNER_RANK
                || params.num_digits_inner != 1
                || params.position_partials_per_block != FP128_D512_POSITION_PARTIALS as u64
                || !params
                    .positions_per_partial
                    .is_multiple_of(partial_alignment)
                || !params.positions_per_block.is_multiple_of(2)
                || !params.blocks_per_column.is_power_of_two()
                || params.blocks_per_column > (1 << 12)
                || params.full_blocks_per_column > params.blocks_per_column
                || params.boundary_columns != 0
                || params.num_blocks != expected_tasks
                || params.task_offset != 0
                || params.dispatch_tasks != params.num_blocks
                || params.lane_row_offset != 0
                || params.output_coefficients != expected_output
                || streams_per_command == 0
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "packed fp128 D128 rank-3 dispatch geometry is outside the registered schedule"
                        .into(),
                ));
            }
            let expected_matrix_bytes = params
                .n_a
                .checked_mul(params.positions_per_block)
                .and_then(|count| count.checked_mul(params.ring_d))
                .and_then(|count| count.checked_mul(size_of::<Fp128Limbs>() as u64))
                .ok_or(MetalCommitError::ShapeOverflow("packed matrix bytes"))?;
            if matrix.length() < expected_matrix_bytes {
                return Err(MetalCommitError::UnsupportedShape(
                    "packed fp128 D128 rank-3 matrix prefix is shorter than the plan".into(),
                ));
            }
            if !self.supports_packed_fp128_d128_rank3() {
                return Err(MetalCommitError::UnsupportedShape(
                    "packed fp128 D128 rank-3 pipeline does not expose the registered resources"
                        .into(),
                ));
            }
            params.columns_per_threadgroup = 1;
            let tasks_per_stream = FP128_D128_RANK3_TASKS_PER_STREAM as u64;
            let base_streams = params.num_blocks.div_ceil(tasks_per_stream);
            let groups_per_stream = params
                .n_a
                .checked_mul(params.position_partials_per_block)
                .ok_or(MetalCommitError::ShapeOverflow("packed partial groups"))?;
            let threadgroups = base_streams
                .checked_mul(groups_per_stream)
                .ok_or(MetalCommitError::ShapeOverflow("packed threadgroups"))?;
            if threadgroups > u64::from(u32::MAX)
                || params.output_coefficients > u64::from(u32::MAX)
                || params
                    .output_coefficients
                    .checked_mul(params.position_partials_per_block)
                    .is_none_or(|count| count > u64::from(u32::MAX))
            {
                return Err(MetalCommitError::UnsupportedShape(
                    "packed fp128 D128 rank-3 grid exceeds u32 indexing".into(),
                ));
            }

            let buffer_start = Instant::now();
            let output_count = usize::try_from(params.output_coefficients)
                .map_err(|_| MetalCommitError::ShapeOverflow("output coefficients"))?;
            let output_bytes = output_count
                .checked_mul(size_of::<Fp128Limbs>())
                .ok_or(MetalCommitError::ShapeOverflow("output bytes"))?;
            let output = self.shared_buffer(output_bytes)?;
            let partial_count = output_count
                .checked_mul(params.position_partials_per_block as usize)
                .ok_or(MetalCommitError::ShapeOverflow("partial coefficients"))?;
            let scratch_bytes = partial_count
                .checked_mul(size_of::<Fp128Limbs>())
                .ok_or(MetalCommitError::ShapeOverflow("partial bytes"))?;
            let partials = self.private_buffer(scratch_bytes)?;
            let no_active_zero_rows = [0u64];
            let active_zero_rows = if active_zero_rows.is_empty() {
                no_active_zero_rows.as_slice()
            } else {
                active_zero_rows
            };
            let active_zero_rows = self.shared_buffer_from_slice(active_zero_rows)?;
            let mut buffer_setup = buffer_start.elapsed();

            let source = ResidentPackedLanes { lanes };
            let command_start = Instant::now();
            let command_count = base_streams.div_ceil(streams_per_command as u64) as usize;
            let mut commands = Vec::with_capacity(command_count);
            let mut lane_buffers = Vec::with_capacity(command_count);
            let mut input_zero_copy = true;
            // Two ring positions per trace row at K = 256, D = 128.
            let rows_per_block = params.positions_per_block / 2;
            for first_stream in (0..base_streams).step_by(streams_per_command) {
                let stream_count = (base_streams - first_stream).min(streams_per_command as u64);
                let task_offset = first_stream * tasks_per_stream;
                let mut dispatch_params = params;
                dispatch_params.task_offset = task_offset;
                dispatch_params.dispatch_tasks =
                    (stream_count * tasks_per_stream).min(params.num_blocks - task_offset);
                let final_task = task_offset + dispatch_params.dispatch_tasks - 1;
                let first_block = task_offset / params.num_columns;
                let final_block = final_task / params.num_columns;
                let first_row = usize::try_from(first_block * rows_per_block)
                    .map_err(|_| MetalCommitError::ShapeOverflow("packed first row"))?;
                let final_row = usize::try_from((final_block + 1) * rows_per_block)
                    .map_err(|_| MetalCommitError::ShapeOverflow("packed final row"))?;
                let lane_stride = usize::try_from(params.lane_stride)
                    .map_err(|_| MetalCommitError::ShapeOverflow("packed lane stride"))?;
                let command_lanes = source.wait_lanes(first_row..final_row, lane_stride)?;
                let lane_buffer_start = Instant::now();
                let lane_buffer = self.packed_lane_buffer(command_lanes)?;
                buffer_setup += lane_buffer_start.elapsed();
                input_zero_copy &= lane_buffer.zero_copy;
                dispatch_params.lane_row_offset = first_row as u64;
                let command_threadgroups = stream_count.checked_mul(groups_per_stream).ok_or(
                    MetalCommitError::ShapeOverflow("packed command threadgroups"),
                )?;
                let command = self.queue.new_command_buffer();
                command.set_label("Akita packed fp128 D128 rank-3 root commitment");
                let encoder = command.new_compute_command_encoder();
                encoder.set_label("Akita packed fp128 D128 rank-3 element tiles");
                encoder.set_compute_pipeline_state(&self.packed_fp128_d128_rank3_pipeline);
                encoder.set_buffer(0, Some(matrix), 0);
                encoder.set_buffer(1, Some(&lane_buffer.buffer), 0);
                encoder.set_buffer(2, Some(&partials), 0);
                set_inline_bytes(encoder, 3, &dispatch_params);
                encoder.set_buffer(4, Some(&active_zero_rows), 0);
                // The serial encoder orders second-half reads after first-half writes.
                for half in 0..2u32 {
                    set_inline_bytes(encoder, 5, &half);
                    encoder.dispatch_thread_groups(
                        MTLSize::new(command_threadgroups, 1, 1),
                        MTLSize::new(FP128_D512_THREADS as u64, 1, 1),
                    );
                }
                encoder.end_encoding();
                command.commit();
                commands.push(command);
                lane_buffers.push(lane_buffer);
            }

            let reduction_command = self.queue.new_command_buffer();
            reduction_command.set_label("Akita packed fp128 D128 rank-3 partial reduction");
            let reduction = reduction_command.new_compute_command_encoder();
            reduction.set_label("Akita packed fp128 D128 rank-3 partial reduction");
            reduction.set_compute_pipeline_state(&self.packed_partial_reduction_pipeline);
            reduction.set_buffer(0, Some(&partials), 0);
            reduction.set_buffer(1, Some(&output), 0);
            set_inline_bytes(reduction, 2, &params);
            let reduction_width = (self
                .packed_partial_reduction_pipeline
                .thread_execution_width()
                * 4)
            .min(
                self.packed_partial_reduction_pipeline
                    .max_total_threads_per_threadgroup(),
            );
            reduction.dispatch_threads(
                MTLSize::new(params.output_coefficients, 1, 1),
                MTLSize::new(reduction_width, 1, 1),
            );
            reduction.end_encoding();
            reduction_command.commit();
            reduction_command.wait_until_completed();
            let command_wall = command_start.elapsed();
            for command in &commands {
                validate_completed_command(command)?;
            }
            validate_completed_command(reduction_command)?;
            let panel_gpu_active = commands.iter().try_fold(Duration::ZERO, |total, command| {
                total.checked_add(completed_command_gpu_time(command)?)
            });
            let panel_gpu_span = commands
                .first()
                .zip(commands.last())
                .and_then(|(first, last)| completed_commands_gpu_span(first, last));
            let reduction_gpu = completed_command_gpu_time(reduction_command);
            let gpu = commands
                .first()
                .and_then(|first| completed_commands_gpu_span(first, reduction_command));

            let readback_start = Instant::now();
            // SAFETY: `output` is live shared storage for exactly `output_count`
            // aligned `Fp128Limbs` values.
            let coefficients = unsafe {
                std::slice::from_raw_parts(output.contents().cast::<Fp128Limbs>(), output_count)
                    .to_vec()
            };
            Ok(DispatchOutcome {
                coefficients,
                timings: DispatchTimings {
                    buffer_setup,
                    command_wall,
                    gpu,
                    readback_copy: readback_start.elapsed(),
                },
                panel_gpu_active,
                panel_gpu_span,
                reduction_gpu,
                command_buffers: commands.len() + 1,
                kernel: MetalOneHotKernel::PackedFp128D128Rank3,
                blocks_per_threadgroup: FP128_D128_RANK3_TASKS_PER_STREAM,
                columns_per_threadgroup: 1,
                matrix_block_streams: threadgroups as usize,
                scratch_bytes,
                input_zero_copy,
            })
        })
    }

    pub(super) fn dispatch_geometry(
        &self,
        params: OneHotCommitParams,
        kernel: MetalOneHotKernel,
    ) -> Result<usize, MetalCommitError> {
        match kernel {
            MetalOneHotKernel::DirectGather => Ok(1),
            MetalOneHotKernel::BlockBatched => {
                let ring_d = usize::try_from(params.ring_d)
                    .map_err(|_| MetalCommitError::ShapeOverflow("ring dimension"))?;
                let max_threads = self
                    .block_batched_pipeline
                    .max_total_threads_per_threadgroup() as usize;
                let blocks = max_threads / ring_d;
                if blocks == 0 {
                    Err(MetalCommitError::UnsupportedShape(format!(
                        "D={} exceeds the block-batched pipeline's {max_threads}-thread limit",
                        params.ring_d
                    )))
                } else {
                    Ok(blocks)
                }
            }
            MetalOneHotKernel::PackedFp128D512Panels | MetalOneHotKernel::PackedFp128D128Rank3 => {
                Err(MetalCommitError::UnsupportedShape(
                    "packed kernel requires packed parameters".into(),
                ))
            }
        }
    }

    pub(super) fn encode_onehot(
        &self,
        encoder: &ComputeCommandEncoderRef,
        matrix: &Buffer,
        indices: &Buffer,
        output: &Buffer,
        params: &OneHotCommitParams,
        kernel: MetalOneHotKernel,
    ) -> Result<(), MetalCommitError> {
        let pipeline = match kernel {
            MetalOneHotKernel::DirectGather => &self.direct_pipeline,
            MetalOneHotKernel::BlockBatched => &self.block_batched_pipeline,
            MetalOneHotKernel::PackedFp128D512Panels | MetalOneHotKernel::PackedFp128D128Rank3 => {
                return Err(MetalCommitError::UnsupportedShape(
                    "packed kernel requires packed parameters".into(),
                ));
            }
        };
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(matrix), 0);
        encoder.set_buffer(1, Some(indices), 0);
        encoder.set_buffer(2, Some(output), 0);
        set_inline_bytes(encoder, 3, params);
        match kernel {
            MetalOneHotKernel::DirectGather => {
                let width = (pipeline.thread_execution_width() * 4)
                    .min(pipeline.max_total_threads_per_threadgroup());
                encoder.dispatch_threads(
                    MTLSize::new(params.output_coefficients, 1, 1),
                    MTLSize::new(width, 1, 1),
                );
            }
            MetalOneHotKernel::BlockBatched => {
                let block_groups = params.num_blocks.div_ceil(params.blocks_per_threadgroup);
                let threadgroups = params
                    .num_sources
                    .checked_mul(params.n_a)
                    .and_then(|count| count.checked_mul(block_groups))
                    .ok_or(MetalCommitError::ShapeOverflow("block threadgroups"))?;
                encoder.dispatch_thread_groups(
                    MTLSize::new(threadgroups, 1, 1),
                    MTLSize::new(params.blocks_per_threadgroup * params.ring_d, 1, 1),
                );
            }
            MetalOneHotKernel::PackedFp128D512Panels | MetalOneHotKernel::PackedFp128D128Rank3 => {
                unreachable!()
            }
        }
        Ok(())
    }
}
