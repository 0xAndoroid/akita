use akita_error::AkitaError;
use jolt_field::solinas::parallel::*;

pub(crate) fn balanced_decompose_centered_i32_i8_into<const D: usize>(
    centered: &[i32; D],
    out: &mut [[i8; D]],
    log_basis: u32,
) {
    let levels = out.len();
    assert!(
        log_basis > 0 && log_basis <= 8,
        "log_basis must be in 1..=8 for i8 output"
    );
    assert!(
        (levels as u32).saturating_mul(log_basis) <= 128 + log_basis,
        "levels * log_basis must be <= 128 + log_basis"
    );

    let half_b = 1i32 << (log_basis - 1);
    let mask = (half_b << 1) - 1;
    let mut carries = *centered;
    for plane in out {
        for (digit, value) in plane.iter_mut().zip(&mut carries) {
            let raw = *value & mask;
            let carry = i32::from(raw >= half_b);
            *digit = (raw - (carry << log_basis)) as i8;
            *value = (*value >> log_basis) + carry;
        }
    }
}

/// Decompose centered Z fold responses into `(position, commit_digit, fold_digit)` planes.
pub(crate) fn decompose_z_folded_planes<const D: usize>(
    z_folded_centered: &[i32],
    num_digits_fold: usize,
    log_basis: u32,
) -> Result<Vec<[i8; D]>, AkitaError> {
    if !(1..=8).contains(&log_basis)
        || num_digits_fold == 0
        || num_digits_fold > (128 + log_basis) as usize / log_basis as usize
    {
        return Err(AkitaError::InvalidInput(
            "invalid centered digit decomposition".into(),
        ));
    }
    let (rows, remainder) = z_folded_centered.as_chunks::<D>();
    if !remainder.is_empty() {
        return Err(AkitaError::InvalidSize {
            expected: D,
            actual: z_folded_centered.len(),
        });
    }
    let plane_count = rows
        .len()
        .checked_mul(num_digits_fold)
        .ok_or_else(|| AkitaError::InvalidSetup("Z plane count overflow".to_string()))?;
    let mut all_planes = vec![[0i8; D]; plane_count];
    cfg_iter!(rows)
        .zip(cfg_chunks_mut!(&mut all_planes, num_digits_fold))
        .for_each(|(z_j, planes)| {
            balanced_decompose_centered_i32_i8_into(z_j, planes, log_basis);
        });
    Ok(all_planes)
}
