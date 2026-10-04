use super::*;

fn pow_mod(mut base: i64, mut exponent: i64, modulus: i64) -> i64 {
    let mut result = 1i64;
    base %= modulus;
    while exponent != 0 {
        if exponent & 1 == 1 {
            result = result * base % modulus;
        }
        base = base * base % modulus;
        exponent >>= 1;
    }
    result
}

fn primitive_root_of_unity(modulus: i64, order: usize) -> i64 {
    let half = (modulus - 1) / 2;
    let exponent = (modulus - 1) / order as i64;
    for candidate in 2..modulus {
        if pow_mod(candidate, half, modulus) == modulus - 1 {
            let root = pow_mod(candidate, exponent, modulus);
            if pow_mod(root, order as i64, modulus) == 1
                && pow_mod(root, order as i64 / 2, modulus) == modulus - 1
            {
                return root;
            }
        }
    }
    unreachable!("the fixed NTT prime has no root of the required order")
}

fn constant_buffer<T>(device: &Device, values: &[T]) -> Buffer {
    device.new_buffer_with_data(
        values.as_ptr().cast::<c_void>(),
        size_of_val(values) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

pub(super) fn d512_linear_relation_resources(device: &Device) -> D512LinearRelationResources {
    let primes = FP128_D512_LINEAR_RELATION_RAW_PRIMES.map(NttPrime::<i32>::compute);
    let mut device_primes = Vec::with_capacity(primes.len());
    let mut fwd_twiddles = Vec::with_capacity(primes.len() * FP128_D512_LINEAR_RELATION_NTT_SIZE);
    let mut inv_twiddles = Vec::with_capacity(primes.len() * FP128_D512_LINEAR_RELATION_NTT_SIZE);
    let mut d_inv = Vec::with_capacity(primes.len());
    let mut limb_weights = Vec::with_capacity(primes.len() * 4);
    let mut field_moduli = Vec::with_capacity(primes.len());
    let field_modulus = (-F::one()).to_canonical_u128() + 1;

    for prime in primes {
        device_primes.push(D512LinearNttPrime {
            p: prime.p,
            pinv: prime.pinv,
            mont: prime.mont,
            montsq: prime.montsq,
        });
        let modulus = i64::from(prime.p);
        let root = primitive_root_of_unity(modulus, FP128_D512_LINEAR_RELATION_NTT_SIZE);
        let root_inverse = pow_mod(root, modulus - 2, modulus);
        let one = prime.from_canonical(1);
        let mut prime_fwd = vec![0i32; FP128_D512_LINEAR_RELATION_NTT_SIZE];
        let mut prime_inv = vec![0i32; FP128_D512_LINEAR_RELATION_NTT_SIZE];
        for stage in 0..FP128_D512_LINEAR_RELATION_NTT_SIZE.ilog2() as usize {
            let len = 1usize << stage;
            let exponent = (FP128_D512_LINEAR_RELATION_NTT_SIZE / (2 * len)) as i64;
            let fwd_step = prime.from_canonical(pow_mod(root, exponent, modulus) as i32);
            let inv_step = prime.from_canonical(pow_mod(root_inverse, exponent, modulus) as i32);
            let mut fwd = one;
            let mut inv = one;
            for offset in 0..len {
                prime_fwd[len - 1 + offset] = fwd.raw();
                prime_inv[len - 1 + offset] = inv.raw();
                fwd = prime.mul(fwd, fwd_step);
                inv = prime.mul(inv, inv_step);
            }
        }
        fwd_twiddles.extend(prime_fwd);
        inv_twiddles.extend(prime_inv);
        let inverse = prime.from_canonical(pow_mod(
            FP128_D512_LINEAR_RELATION_NTT_SIZE as i64,
            modulus - 2,
            modulus,
        ) as i32);
        d_inv.push(inverse.raw());

        let radix = (1u128 << 32) % prime.p as u128;
        let mut weight = 1u128;
        for _ in 0..4 {
            limb_weights.push(prime.from_canonical(weight as i32).raw());
            weight = weight * radix % prime.p as u128;
        }
        field_moduli.push(
            prime
                .from_canonical((field_modulus % prime.p as u128) as i32)
                .raw(),
        );
    }

    let garner = GarnerData::compute(&primes);
    let garner_gamma = garner
        .gamma
        .into_iter()
        .flatten()
        .map(|value| value as u32)
        .collect::<Vec<_>>();
    let mut partial_product = F::one();
    let mut field_partial_products = Vec::with_capacity(primes.len());
    for prime in primes {
        field_partial_products.push(Fp128Limbs::from_field(partial_product));
        partial_product *= F::from_u64(prime.p as u64);
    }

    D512LinearRelationResources {
        primes: constant_buffer(device, &device_primes),
        fwd_twiddles: constant_buffer(device, &fwd_twiddles),
        inv_twiddles: constant_buffer(device, &inv_twiddles),
        d_inv: constant_buffer(device, &d_inv),
        limb_weights: constant_buffer(device, &limb_weights),
        field_moduli: constant_buffer(device, &field_moduli),
        garner_gamma: constant_buffer(device, &garner_gamma),
        field_partial_products: constant_buffer(device, &field_partial_products),
    }
}

pub(super) fn recursive_commit_resources(
    device: &Device,
    ring_d: usize,
) -> RecursiveCommitResources {
    let primes = FP128_D512_LINEAR_RELATION_RAW_PRIMES.map(NttPrime::<i32>::compute);
    let mut device_primes = Vec::with_capacity(primes.len());
    let mut fwd_twiddles = Vec::with_capacity(primes.len() * ring_d);
    let mut inv_twiddles = Vec::with_capacity(primes.len() * ring_d);
    let mut psi_pows = Vec::with_capacity(primes.len() * ring_d);
    let mut inverse_scale = Vec::with_capacity(primes.len() * ring_d);
    let mut limb_weights = Vec::with_capacity(primes.len() * 4);
    let mut field_moduli = Vec::with_capacity(primes.len());
    let field_modulus = (-F::one()).to_canonical_u128() + 1;

    for prime in primes {
        device_primes.push(D512LinearNttPrime {
            p: prime.p,
            pinv: prime.pinv,
            mont: prime.mont,
            montsq: prime.montsq,
        });
        let modulus = i64::from(prime.p);
        let psi = primitive_root_of_unity(modulus, 2 * ring_d);
        let psi_inverse = pow_mod(psi, modulus - 2, modulus);
        let omega = psi * psi % modulus;
        let omega_inverse = pow_mod(omega, modulus - 2, modulus);
        let one = prime.from_canonical(1);
        let mut prime_fwd = vec![0i32; ring_d];
        let mut prime_inv = vec![0i32; ring_d];
        for stage in 0..ring_d.ilog2() as usize {
            let len = 1usize << stage;
            let exponent = (ring_d / (2 * len)) as i64;
            let fwd_step = prime.from_canonical(pow_mod(omega, exponent, modulus) as i32);
            let inv_step = prime.from_canonical(pow_mod(omega_inverse, exponent, modulus) as i32);
            let mut fwd = one;
            let mut inv = one;
            for offset in 0..len {
                prime_fwd[len - 1 + offset] = fwd.raw();
                prime_inv[len - 1 + offset] = inv.raw();
                fwd = prime.mul(fwd, fwd_step);
                inv = prime.mul(inv, inv_step);
            }
        }
        fwd_twiddles.extend(prime_fwd);
        inv_twiddles.extend(prime_inv);

        let psi_mont = prime.from_canonical(psi as i32);
        let psi_inverse_mont = prime.from_canonical(psi_inverse as i32);
        let d_inverse = prime.from_canonical(pow_mod(ring_d as i64, modulus - 2, modulus) as i32);
        let mut psi_power = one;
        let mut psi_inverse_power = one;
        for _ in 0..ring_d {
            psi_pows.push(psi_power.raw());
            inverse_scale.push(prime.mul(d_inverse, psi_inverse_power).raw());
            psi_power = prime.mul(psi_power, psi_mont);
            psi_inverse_power = prime.mul(psi_inverse_power, psi_inverse_mont);
        }

        let radix = (1u128 << 32) % prime.p as u128;
        let mut weight = 1u128;
        for _ in 0..4 {
            limb_weights.push(prime.from_canonical(weight as i32).raw());
            weight = weight * radix % prime.p as u128;
        }
        field_moduli.push(
            prime
                .from_canonical((field_modulus % prime.p as u128) as i32)
                .raw(),
        );
    }

    let garner = GarnerData::compute(&primes);
    let garner_gamma = garner
        .gamma
        .into_iter()
        .flatten()
        .map(|value| value as u32)
        .collect::<Vec<_>>();
    let mut partial_product = F::one();
    let mut field_partial_products = Vec::with_capacity(primes.len());
    for prime in primes {
        field_partial_products.push(Fp128Limbs::from_field(partial_product));
        partial_product *= F::from_u64(prime.p as u64);
    }

    RecursiveCommitResources {
        ring_d,
        primes: constant_buffer(device, &device_primes),
        fwd_twiddles: constant_buffer(device, &fwd_twiddles),
        inv_twiddles: constant_buffer(device, &inv_twiddles),
        psi_pows: constant_buffer(device, &psi_pows),
        inverse_scale: constant_buffer(device, &inverse_scale),
        limb_weights: constant_buffer(device, &limb_weights),
        field_moduli: constant_buffer(device, &field_moduli),
        garner_gamma: constant_buffer(device, &garner_gamma),
        field_partial_products: constant_buffer(device, &field_partial_products),
    }
}

impl MetalRuntime {
    pub(crate) fn supports_fp128_d64_digit_rows<const D: usize>(
        &self,
        num_vectors: usize,
        num_rows: usize,
        num_cols: usize,
        retain_quotients: bool,
    ) -> bool {
        let Ok(num_vectors) = u64::try_from(num_vectors) else {
            return false;
        };
        let Ok(num_rows) = u64::try_from(num_rows) else {
            return false;
        };
        let Ok(num_cols) = u64::try_from(num_cols) else {
            return false;
        };
        if D != 64
            || num_vectors == 0
            || num_rows == 0
            || num_cols == 0
            || num_cols > u64::from(u32::MAX)
        {
            return false;
        }
        let column_partials = num_cols.div_ceil(FP128_D64_DIGIT_ROWS_COLUMNS_PER_PARTIAL as u64);
        let output_coefficients = num_vectors
            .checked_mul(num_rows)
            .and_then(|count| count.checked_mul(D as u64));
        let threadgroups = num_vectors
            .checked_mul(num_rows)
            .and_then(|count| count.checked_mul(column_partials));
        let matrix_bytes = num_rows
            .checked_mul(num_cols)
            .and_then(|count| count.checked_mul(D as u64))
            .and_then(|count| count.checked_mul(size_of::<Fp128Limbs>() as u64));
        let digit_bytes = num_vectors
            .checked_mul(num_cols)
            .and_then(|count| count.checked_mul(D as u64));
        let product_count = 1 + u64::from(retain_quotients);
        let partial_bytes = threadgroups
            .and_then(|count| count.checked_mul(D as u64))
            .and_then(|count| count.checked_mul(product_count))
            .and_then(|count| count.checked_mul(size_of::<Fp128Limbs>() as u64));
        let output_bytes = output_coefficients
            .and_then(|count| count.checked_mul(product_count))
            .and_then(|count| count.checked_mul(size_of::<Fp128Limbs>() as u64));
        let maximum = self.device.max_buffer_length();
        output_coefficients
            .and_then(|count| count.checked_mul(product_count))
            .is_some_and(|count| count <= u64::from(u32::MAX))
            && threadgroups.is_some_and(|count| count <= u64::from(u32::MAX))
            && [matrix_bytes, digit_bytes, partial_bytes, output_bytes]
                .into_iter()
                .all(|bytes| bytes.is_some_and(|bytes| bytes != 0 && bytes <= maximum))
            && self
                .fp128_d64_digit_rows_partials_pipeline
                .max_total_threads_per_threadgroup()
                >= FP128_D64_DIGIT_ROWS_PARTIAL_THREADS as u64
            && self
                .fp128_d64_digit_rows_reduce_pipeline
                .max_total_threads_per_threadgroup()
                >= FP128_D64_DIGIT_ROWS_THREADS as u64
    }

    pub(crate) fn supports_fp128_d512_linear_relation(
        &self,
        num_columns: usize,
        rhs_abs_bound: u64,
    ) -> bool {
        if num_columns == 0 || rhs_abs_bound >= FP128_D512_LINEAR_RELATION_RAW_PRIMES[0] as u64 {
            return false;
        }
        let capacity = CrtCapacity::from_prime_moduli(
            FP128_D512_LINEAR_RELATION_RAW_PRIMES.map(|prime| prime as u128),
        );
        let field_modulus = (-F::one()).to_canonical_u128() + 1;
        if !capacity.supports_modulus(num_columns, 512, field_modulus, rhs_abs_bound) {
            return false;
        }
        let num_tiles = num_columns.div_ceil(FP128_D512_LINEAR_RELATION_COLUMNS_PER_TILE);
        let matrix_bytes = num_columns
            .checked_mul(512)
            .and_then(|count| count.checked_mul(size_of::<Fp128Limbs>()));
        let rhs_bytes = num_columns
            .checked_mul(512)
            .and_then(|count| count.checked_mul(size_of::<i32>()));
        let partial_bytes = num_tiles
            .checked_mul(FP128_D512_LINEAR_RELATION_NUM_PRIMES)
            .and_then(|count| count.checked_mul(FP128_D512_LINEAR_RELATION_NTT_SIZE))
            .and_then(|count| count.checked_mul(size_of::<i32>()));
        [matrix_bytes, rhs_bytes, partial_bytes]
            .into_iter()
            .all(|bytes| {
                bytes.is_some_and(|bytes| {
                    bytes != 0 && bytes as u64 <= self.device.max_buffer_length()
                })
            })
            && num_tiles
                .checked_mul(FP128_D512_LINEAR_RELATION_NUM_PRIMES)
                .is_some_and(|groups| groups <= u32::MAX as usize)
            && self
                .fp128_d512_linear_relation_partials_pipeline
                .max_total_threads_per_threadgroup()
                >= FP128_D512_LINEAR_RELATION_THREADS as u64
            && self
                .fp128_d512_linear_relation_reduce_pipeline
                .max_total_threads_per_threadgroup()
                >= FP128_D512_LINEAR_RELATION_THREADS as u64
            && self
                .fp128_d512_linear_relation_reconstruct_pipeline
                .max_total_threads_per_threadgroup()
                >= FP128_D512_LINEAR_RELATION_THREADS as u64
    }

    pub(super) fn recursive_commit_resources(
        &self,
        ring_d: usize,
    ) -> Option<&RecursiveCommitResources> {
        self.fp128_recursive_commit_resources
            .iter()
            .find(|resources| resources.ring_d == ring_d)
    }

    pub(crate) fn supports_fp128_recursive_commit<const D: usize>(
        &self,
        num_blocks: usize,
        num_rows: usize,
        num_cols: usize,
        rhs_abs_bound: u64,
    ) -> bool {
        if !matches!(D, 64 | 128)
            || num_blocks == 0
            || num_rows == 0
            || num_rows > FP128_RECURSIVE_COMMIT_MAX_ROWS
            || num_cols == 0
            || rhs_abs_bound >= FP128_D512_LINEAR_RELATION_RAW_PRIMES[0] as u64
            || self.recursive_commit_resources(D).is_none()
        {
            return false;
        }
        let capacity = CrtCapacity::from_prime_moduli(
            FP128_D512_LINEAR_RELATION_RAW_PRIMES.map(|prime| prime as u128),
        );
        let field_modulus = (-F::one()).to_canonical_u128() + 1;
        if !capacity.supports_modulus(num_cols, D, field_modulus, rhs_abs_bound) {
            return false;
        }
        let matrix_rings = num_rows.checked_mul(num_cols);
        let source_bytes = num_blocks
            .checked_mul(num_cols)
            .and_then(|count| count.checked_mul(D));
        let matrix_ntt_bytes = matrix_rings
            .and_then(|count| count.checked_mul(D))
            .and_then(|count| count.checked_mul(FP128_D512_LINEAR_RELATION_NUM_PRIMES))
            .and_then(|count| count.checked_mul(size_of::<i32>()));
        let residue_bytes = num_blocks
            .checked_mul(num_rows)
            .and_then(|count| count.checked_mul(D))
            .and_then(|count| count.checked_mul(FP128_D512_LINEAR_RELATION_NUM_PRIMES))
            .and_then(|count| count.checked_mul(size_of::<u32>()));
        let output_bytes = num_blocks
            .checked_mul(num_rows)
            .and_then(|count| count.checked_mul(D))
            .and_then(|count| count.checked_mul(size_of::<Fp128Limbs>()));
        let maximum = self.device.max_buffer_length();
        [source_bytes, matrix_ntt_bytes, residue_bytes, output_bytes]
            .into_iter()
            .all(|bytes| bytes.is_some_and(|bytes| bytes != 0 && bytes as u64 <= maximum))
            && matrix_rings
                .and_then(|count| count.checked_mul(FP128_D512_LINEAR_RELATION_NUM_PRIMES))
                .is_some_and(|groups| groups <= u32::MAX as usize)
            && num_blocks
                .div_ceil(FP128_RECURSIVE_COMMIT_BLOCKS_PER_GROUP)
                .checked_mul(FP128_D512_LINEAR_RELATION_NUM_PRIMES)
                .is_some_and(|groups| groups <= u32::MAX as usize)
            && output_bytes
                .map(|bytes| bytes / size_of::<Fp128Limbs>())
                .is_some_and(|count| {
                    count.div_ceil(FP128_RECURSIVE_COMMIT_RECONSTRUCT_THREADS) <= u32::MAX as usize
                })
            && self
                .fp128_recursive_commit_matrix_ntt_pipeline
                .max_total_threads_per_threadgroup()
                >= D as u64
            && self
                .fp128_recursive_commit_matvec_pipeline
                .max_total_threads_per_threadgroup()
                >= FP128_RECURSIVE_COMMIT_THREADS as u64
            && self
                .fp128_recursive_commit_reconstruct_pipeline
                .max_total_threads_per_threadgroup()
                >= FP128_RECURSIVE_COMMIT_RECONSTRUCT_THREADS as u64
    }
}
