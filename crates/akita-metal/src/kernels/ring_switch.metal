kernel void akita_fp128_centered_rows_partials(
    device const AkitaFp128 *matrix [[buffer(0)]],
    device const int *centered [[buffer(1)]],
    device AkitaFp128 *partials [[buffer(2)]],
    constant DigitRowsParams &params [[buffer(3)]],
    uint coefficient [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]])
{
    threadgroup AkitaFp128 matrix_ring[128];
    threadgroup int centered_ring[128];
    uint ring_d = (uint)params.ring_d;
    uint row = group.x / (uint)params.column_partials;
    uint partial = group.x % (uint)params.column_partials;
    uint begin = partial * (uint)params.columns_per_partial;
    uint end = begin + min((uint)params.columns_per_partial, (uint)params.num_cols - begin);
    AkitaFp128 sum = akita_zero();
    for (uint column = begin; column < end; ++column) {
        matrix_ring[coefficient] = matrix[
            ((ulong)row * params.num_cols + column) * ring_d + coefficient];
        centered_ring[coefficient] = centered[(ulong)column * ring_d + coefficient];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Admission bounds |centered| by 1024: 32 * 1024 * 65535 < 2^31.
        for (uint start = coefficient + 1u; start < ring_d; start += 32u) {
            AkitaWideAccumulator accumulator = akita_wide_zero();
            uint stop = min(start + 32u, ring_d);
            for (uint term = start; term < stop; ++term) {
                int value = centered_ring[term];
                akita_wide_accumulate_scaled(
                    accumulator, matrix_ring[coefficient + ring_d - term], value);
            }
            sum = akita_add(sum, akita_reduce_wide(accumulator));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    partials[(ulong)group.x * ring_d + coefficient] = sum;
}

struct CenteredDigitsParams {
    ulong source_coefficients;
    uint ring_d;
    uint digits;
    uint log_basis;
    uint padding;
};

kernel void akita_centered_digit_planes(
    device const int *centered [[buffer(0)]],
    device char *digits [[buffer(1)]],
    constant CenteredDigitsParams &params [[buffer(2)]],
    uint coefficient [[thread_position_in_grid]])
{
    if ((ulong)coefficient >= params.source_coefficients) return;
    uint ring = coefficient / params.ring_d;
    uint lane = coefficient % params.ring_d;
    int value = centered[coefficient];
    int half_basis = 1 << (params.log_basis - 1u);
    int mask = (half_basis << 1) - 1;
    for (uint digit = 0u; digit < params.digits; ++digit) {
        int raw = value & mask;
        int carry = int(raw >= half_basis);
        digits[((ulong)ring * params.digits + digit) * params.ring_d + lane] =
            char(raw - (carry << params.log_basis));
        value = (value >> params.log_basis) + carry;
    }
}
