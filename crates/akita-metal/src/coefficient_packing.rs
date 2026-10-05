use akita_error::AkitaError;
use akita_prover::compute::{SubringCoefficientPackingPartials, SubringCoefficientPackingPlan};
use akita_types::PreparedSubringCoefficientPackingPoint;

use crate::backend::MetalBackend;
use crate::field::{Fp128Limbs, MetalField, F};
use crate::packed_onehot::PackedOneHotCommitView;
use crate::runtime::{
    CoefficientPackingDispatchOutcome, PackedOneHotCoefficientPackingParams,
    FP128_PACKED_COEFFICIENT_PACKING_ROWS_PER_PARTIAL,
};
use crate::MetalCommitError;

fn checked_u64(value: usize, label: &'static str) -> Result<u64, AkitaError> {
    u64::try_from(value).map_err(|_| MetalCommitError::ShapeOverflow(label).into_akita())
}

fn combined_weights(
    point: &PreparedSubringCoefficientPackingPoint<F>,
) -> Result<Vec<Fp128Limbs>, AkitaError> {
    let geometry = point.geometry();
    if geometry.extension_degree() != 1
        || point.position_weights().len() != point.num_positions_per_block()
        || point.packing_weights().len() != geometry.subring_embedding_stride()
    {
        return Err(AkitaError::InvalidSetup(
            "fp128 Metal coefficient packing requires base-field weights".into(),
        ));
    }

    let count = point
        .num_positions_per_block()
        .checked_mul(geometry.subring_embedding_stride())
        .ok_or_else(|| AkitaError::InvalidInput("coefficient-packing weight overflow".into()))?;
    let mut weights = Vec::with_capacity(count);
    for &position in point.position_weights() {
        for &packing in point.packing_weights() {
            weights.push((position * packing).into_device());
        }
    }
    Ok(weights)
}

fn decode_outcome(
    backend: &MetalBackend,
    outcome: CoefficientPackingDispatchOutcome,
) -> Result<Vec<F>, AkitaError> {
    let coefficients = outcome
        .coefficients
        .into_iter()
        .enumerate()
        .map(|(index, value)| F::from_device(value, index))
        .collect::<Result<Vec<_>, _>>()
        .map_err(MetalCommitError::into_akita)?;
    backend
        .update_opening_metrics(|metrics| {
            metrics.command_wall_time += outcome.timings.command_wall;
            metrics.gpu_active_time += outcome.timings.gpu.unwrap_or_default();
            metrics.buffer_setup_time += outcome.timings.buffer_setup;
            metrics.readback_time += outcome.timings.readback_copy;
            metrics.allocation_bytes = metrics
                .allocation_bytes
                .saturating_add(outcome.allocation_bytes);
        })
        .map_err(MetalCommitError::into_akita)?;
    Ok(coefficients)
}

impl MetalBackend {
    /// Project a borrowed packed one-hot source into base-field coefficient
    /// packing partials without materializing dense ring coefficients.
    pub fn packed_onehot_coefficient_packing<const D: usize>(
        &self,
        source: PackedOneHotCommitView<'_>,
        point: &PreparedSubringCoefficientPackingPoint<F>,
    ) -> Result<SubringCoefficientPackingPartials<F>, AkitaError> {
        let total_field_elements = source
            .num_rows()
            .checked_mul(source.column_capacity())
            .and_then(|count| count.checked_mul(source.onehot_k()))
            .ok_or_else(|| AkitaError::InvalidInput("packed source length overflow".into()))?;
        if !total_field_elements.is_power_of_two() {
            return Err(AkitaError::InvalidSize {
                expected: total_field_elements.next_power_of_two(),
                actual: total_field_elements,
            });
        }
        SubringCoefficientPackingPlan { point }
            .validate::<D>(total_field_elements.trailing_zeros() as usize)?;

        let block_field_elements = point
            .num_positions_per_block()
            .checked_mul(D)
            .ok_or_else(|| AkitaError::InvalidInput("packing block width overflow".into()))?;
        let segment_field_elements = source
            .num_rows()
            .checked_mul(source.onehot_k())
            .ok_or_else(|| AkitaError::InvalidInput("packing segment width overflow".into()))?;
        if !segment_field_elements.is_multiple_of(block_field_elements)
            || !block_field_elements.is_multiple_of(source.onehot_k())
        {
            return Err(MetalCommitError::UnsupportedShape(
                "packed Metal coefficient packing requires row-aligned blocks".into(),
            )
            .into_akita());
        }

        let blocks_per_column = segment_field_elements / block_field_elements;
        let rows_per_block = block_field_elements / source.onehot_k();
        let num_blocks = source
            .column_capacity()
            .checked_mul(blocks_per_column)
            .ok_or_else(|| AkitaError::InvalidInput("packing block count overflow".into()))?;
        if num_blocks != point.num_live_blocks()
            || point.num_live_positions().checked_mul(D) != Some(total_field_elements)
        {
            return Err(AkitaError::InvalidSetup(
                "packed source disagrees with coefficient-packing point".into(),
            ));
        }

        let runtime = self
            .runtime()
            .ok_or_else(|| MetalCommitError::DeviceUnavailable.into_akita())?;
        let geometry = point.geometry();
        let weights = combined_weights(point)?;
        let rows_per_partial = FP128_PACKED_COEFFICIENT_PACKING_ROWS_PER_PARTIAL;
        let row_partials_per_block = rows_per_block.div_ceil(rows_per_partial);
        let output_coefficients = num_blocks
            .checked_mul(geometry.partial_base_field_width())
            .ok_or_else(|| AkitaError::InvalidInput("packing output overflow".into()))?;
        let partial_coefficients = num_blocks
            .checked_mul(row_partials_per_block)
            .and_then(|count| count.checked_mul(geometry.challenge_subring_dimension()))
            .ok_or_else(|| AkitaError::InvalidInput("packing partial overflow".into()))?;
        let params = PackedOneHotCoefficientPackingParams {
            num_rows: checked_u64(source.num_rows(), "packing rows")?,
            num_columns: checked_u64(source.num_columns(), "packing columns")?,
            column_capacity: checked_u64(source.column_capacity(), "packing column capacity")?,
            onehot_k: checked_u64(source.onehot_k(), "packing one-hot K")?,
            ring_d: checked_u64(D, "packing ring dimension")?,
            positions_per_block: checked_u64(
                point.num_positions_per_block(),
                "packing block positions",
            )?,
            blocks_per_column: checked_u64(blocks_per_column, "packing blocks per column")?,
            rows_per_block: checked_u64(rows_per_block, "packing rows per block")?,
            rows_per_partial: checked_u64(rows_per_partial, "packing rows per partial")?,
            row_partials_per_block: checked_u64(row_partials_per_block, "packing row partials")?,
            num_blocks: checked_u64(num_blocks, "packing blocks")?,
            stride: checked_u64(geometry.subring_embedding_stride(), "packing stride")?,
            subring_dimension: checked_u64(
                geometry.challenge_subring_dimension(),
                "packing subring dimension",
            )?,
            output_coefficients: checked_u64(output_coefficients, "packing output")?,
            partial_coefficients: checked_u64(partial_coefficients, "packing partial output")?,
            zero_column_mask: source.zero_column_mask(),
        };
        let outcome = runtime
            .dispatch_fp128_packed_onehot_coefficient_packing(
                source.lanes(),
                source.active_zero_rows(),
                &weights,
                params,
            )
            .map_err(MetalCommitError::into_akita)?;
        SubringCoefficientPackingPartials::new(geometry, num_blocks, decode_outcome(self, outcome)?)
    }
}

#[cfg(test)]
mod tests {
    use akita_types::{
        BasisMode, PreparedSubringCoefficientPackingPoint, SubringCoefficientPackingGeometry,
    };
    use jolt_field::{Ring, Zero};

    use super::*;
    use crate::{MetalExecutionPolicy, PackedOneHotCommitView};

    #[test]
    fn packed_coefficient_packing_preserves_selected_row_zero() {
        check_packed_coefficient_packing::<128>();
        check_packed_coefficient_packing::<512>();
    }

    fn check_packed_coefficient_packing<const D: usize>() {
        const ROWS: usize = 1 << 16;
        const LIVE_COLUMNS: usize = 2;
        const COLUMN_CAPACITY: usize = 4;
        const K: usize = 256;
        const POSITIONS_PER_BLOCK: usize = 1 << 15;
        const ZERO_COLUMN_MASK: u64 = 0b01;

        let mut lanes = (0..ROWS * LIVE_COLUMNS)
            .map(|index| ((index * 19 + 7) % (K - 1) + 1) as u8)
            .collect::<Vec<_>>();
        let mut active_zero_rows = vec![0u64; ROWS.div_ceil(u64::BITS as usize)];
        for row in (0..ROWS).step_by(11) {
            lanes[row * LIVE_COLUMNS] = 0;
            active_zero_rows[row / u64::BITS as usize] |= 1u64 << (row % u64::BITS as usize);
        }

        let source = PackedOneHotCommitView::new_with_active_zero_rows(
            K,
            COLUMN_CAPACITY,
            LIVE_COLUMNS,
            &lanes,
            &active_zero_rows,
            ZERO_COLUMN_MASK,
        )
        .unwrap();
        let source_num_vars = (ROWS * COLUMN_CAPACITY * K).trailing_zeros() as usize;
        let geometry = SubringCoefficientPackingGeometry::try_new(1, D, 64).unwrap();
        let public_point = (0..source_num_vars)
            .map(|index| F::from_u64((index + 3) as u64))
            .collect::<Vec<_>>();
        let point = PreparedSubringCoefficientPackingPoint::new(
            geometry,
            BasisMode::Lagrange,
            ROWS * COLUMN_CAPACITY * K / D,
            POSITIONS_PER_BLOCK,
            source_num_vars,
            &public_point,
        )
        .unwrap();

        let blocks_per_column = ROWS * K / (POSITIONS_PER_BLOCK * D);
        let mut expected =
            vec![F::zero(); point.num_live_blocks() * geometry.partial_base_field_width()];
        for column in 0..LIVE_COLUMNS {
            for row in 0..ROWS {
                let hot = usize::from(lanes[row * LIVE_COLUMNS + column]);
                let committed_zero = hot == 0
                    && ZERO_COLUMN_MASK & (1u64 << column) != 0
                    && active_zero_rows[row / u64::BITS as usize]
                        & (1u64 << (row % u64::BITS as usize))
                        != 0;
                if hot == 0 && !committed_zero {
                    continue;
                }
                let field_in_column = row * K + hot;
                let block = field_in_column / (POSITIONS_PER_BLOCK * D);
                let within_block = field_in_column % (POSITIONS_PER_BLOCK * D);
                let position = within_block / D;
                let coefficient = within_block % D;
                let bucket = coefficient / geometry.subring_embedding_stride();
                let low = coefficient % geometry.subring_embedding_stride();
                let output_block = column * blocks_per_column + block;
                expected[output_block * geometry.partial_base_field_width() + bucket] +=
                    point.position_weights()[position] * point.packing_weights()[low];
            }
        }

        let backend = MetalBackend::new(MetalExecutionPolicy::RequireMetal).unwrap();
        backend.begin_opening_metrics().unwrap();
        let actual = backend
            .packed_onehot_coefficient_packing::<D>(source, &point)
            .unwrap();
        assert_eq!(actual.coordinates(), expected);
        let metrics = backend.last_opening_metrics().unwrap().unwrap();
        assert_eq!(metrics.cpu_fallback_calls, 0);
        assert!(metrics.command_wall_time > std::time::Duration::ZERO);
    }
}
