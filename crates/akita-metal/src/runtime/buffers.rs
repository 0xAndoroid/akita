use super::*;

impl MetalRuntime {
    pub(crate) fn shared_buffer_from_slice<T>(
        &self,
        values: &[T],
    ) -> Result<Buffer, MetalCommitError> {
        let bytes = size_of_val(values);
        self.validate_buffer_length(bytes)?;
        if bytes == 0 {
            return Err(MetalCommitError::UnsupportedShape(
                "zero-length Metal input buffer".into(),
            ));
        }
        Ok(self.device.new_buffer_with_data(
            values.as_ptr().cast::<c_void>(),
            bytes as u64,
            MTLResourceOptions::StorageModeShared,
        ))
    }

    pub(super) fn shared_buffer_from_digit_rows<const D: usize>(
        &self,
        digit_vectors: &[&[[i8; D]]],
    ) -> Result<Buffer, MetalCommitError> {
        let bytes = digit_vectors.iter().try_fold(0usize, |total, digits| {
            total
                .checked_add(size_of_val(*digits))
                .ok_or(MetalCommitError::ShapeOverflow("digit-row input bytes"))
        })?;
        let buffer = self.shared_buffer(bytes)?;
        let mut destination = buffer.contents().cast::<u8>();
        for digits in digit_vectors {
            let len = size_of_val(*digits);
            // SAFETY: `buffer` owns `bytes`, the checked sum of the disjoint
            // source lengths, and both pointers are valid for `len` bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(digits.as_ptr().cast::<u8>(), destination, len);
                destination = destination.add(len);
            }
        }
        Ok(buffer)
    }

    pub(crate) fn shared_slice_buffer<'a, T>(
        &self,
        values: &'a [T],
    ) -> Result<SharedSliceBuffer<'a, T>, MetalCommitError> {
        let bytes = size_of_val(values);
        self.validate_buffer_length(bytes)?;
        if bytes == 0 {
            return Err(MetalCommitError::UnsupportedShape(
                "zero-length Metal input buffer".into(),
            ));
        }
        let zero_copy = values
            .as_ptr()
            .addr()
            .is_multiple_of(PACKED_ONEHOT_BUFFER_ALIGNMENT)
            && bytes.is_multiple_of(PACKED_ONEHOT_BUFFER_ALIGNMENT);
        let buffer = if zero_copy {
            self.device.new_buffer_with_bytes_no_copy(
                values.as_ptr().cast_mut().cast::<c_void>(),
                bytes as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            )
        } else {
            self.shared_buffer_from_slice(values)?
        };
        Ok(SharedSliceBuffer {
            buffer,
            zero_copy,
            marker: PhantomData,
        })
    }

    pub(super) fn shared_byte_buffer_from_slices<'a>(
        &self,
        values: &[&'a [i8]],
    ) -> Result<SharedByteBuffer<'a>, MetalCommitError> {
        let bytes = values.iter().try_fold(0usize, |total, value| {
            total
                .checked_add(size_of_val(*value))
                .ok_or(MetalCommitError::ShapeOverflow("byte input"))
        })?;
        self.validate_buffer_length(bytes)?;
        if bytes == 0 {
            return Err(MetalCommitError::UnsupportedShape(
                "zero-length Metal byte input".into(),
            ));
        }
        let zero_copy = values.len() == 1
            && values[0]
                .as_ptr()
                .addr()
                .is_multiple_of(PACKED_ONEHOT_BUFFER_ALIGNMENT)
            && bytes.is_multiple_of(PACKED_ONEHOT_BUFFER_ALIGNMENT);
        let buffer = if zero_copy {
            self.device.new_buffer_with_bytes_no_copy(
                values[0].as_ptr().cast::<c_void>(),
                bytes as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            )
        } else {
            let buffer = self.shared_buffer(bytes)?;
            let mut destination = buffer.contents().cast::<u8>();
            for value in values {
                let len = size_of_val(*value);
                // SAFETY: `buffer` owns the checked sum of every source length;
                // the destination advances over disjoint initialized ranges.
                unsafe {
                    std::ptr::copy_nonoverlapping(value.as_ptr().cast::<u8>(), destination, len);
                    destination = destination.add(len);
                }
            }
            buffer
        };
        Ok(SharedByteBuffer {
            buffer,
            zero_copy,
            marker: PhantomData,
        })
    }

    pub(crate) fn private_buffer_from_slice<T>(
        &self,
        values: &[T],
    ) -> Result<Buffer, MetalCommitError> {
        let bytes = size_of_val(values);
        let staging = self.shared_buffer_from_slice(values)?;
        let buffer = self.private_buffer(bytes)?;
        let command = self.queue.new_command_buffer();
        command.set_label("Akita immutable setup upload");
        let encoder = command.new_blit_command_encoder();
        // A single 6 GiB blit on M5 Max leaves the bytes past 4 GiB uncopied.
        const COPY_CHUNK_BYTES: usize = 1 << 30;
        for offset in (0..bytes).step_by(COPY_CHUNK_BYTES) {
            let length = (bytes - offset).min(COPY_CHUNK_BYTES);
            encoder.copy_from_buffer(
                &staging,
                offset as u64,
                &buffer,
                offset as u64,
                length as u64,
            );
        }
        encoder.end_encoding();
        let _ = complete_command(command)?;
        Ok(buffer)
    }

    pub(super) fn packed_lane_buffer<'a>(
        &self,
        lanes: &'a [u8],
    ) -> Result<PackedLaneBuffer<'a>, MetalCommitError> {
        let bytes = lanes.len();
        self.validate_buffer_length(bytes)?;
        if bytes == 0 {
            return Err(MetalCommitError::UnsupportedShape(
                "zero-length Metal input buffer".into(),
            ));
        }
        let zero_copy = lanes
            .as_ptr()
            .addr()
            .is_multiple_of(PACKED_ONEHOT_BUFFER_ALIGNMENT)
            && bytes.is_multiple_of(PACKED_ONEHOT_BUFFER_ALIGNMENT);
        let buffer = if zero_copy {
            self.device.new_buffer_with_bytes_no_copy(
                lanes.as_ptr().cast::<c_void>(),
                bytes as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            )
        } else {
            self.shared_buffer_from_slice(lanes)?
        };
        Ok(PackedLaneBuffer {
            buffer,
            zero_copy,
            marker: PhantomData,
        })
    }

    pub(super) fn shared_buffer(&self, bytes: usize) -> Result<Buffer, MetalCommitError> {
        self.validate_buffer_length(bytes)?;
        if bytes == 0 {
            return Err(MetalCommitError::UnsupportedShape(
                "zero-length Metal output buffer".into(),
            ));
        }
        Ok(self
            .device
            .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared))
    }

    pub(super) fn private_buffer(&self, bytes: usize) -> Result<Buffer, MetalCommitError> {
        self.validate_buffer_length(bytes)?;
        if bytes == 0 {
            return Err(MetalCommitError::UnsupportedShape(
                "zero-length Metal scratch buffer".into(),
            ));
        }
        Ok(self
            .device
            .new_buffer(bytes as u64, MTLResourceOptions::StorageModePrivate))
    }

    pub(super) fn validate_buffer_length(&self, bytes: usize) -> Result<(), MetalCommitError> {
        let requested =
            u64::try_from(bytes).map_err(|_| MetalCommitError::ShapeOverflow("buffer bytes"))?;
        let maximum = self.device.max_buffer_length();
        if requested > maximum {
            return Err(MetalCommitError::BufferTooLong { requested, maximum });
        }
        Ok(())
    }
}
