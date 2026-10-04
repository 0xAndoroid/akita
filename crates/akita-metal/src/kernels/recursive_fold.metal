// Keep in sync with `RecursiveDecomposeFoldParams` in `runtime/recursive_fold.rs`.
// ring_d is 64 or 128 (power-of-two masks below); enforced by `decompose_recursive_rings`.
struct RecursiveDecomposeFoldParams {
    ulong source_rings;
    ulong positions;
    ulong blocks;
    ulong digits;
    uint ring_d;
    uint log_basis;
    ulong output_coefficients;
    AkitaFp128 threshold;
};

constant uint RECURSIVE_FOLD_MAX_DIGITS = 16u;

kernel void akita_fp128_recursive_decompose_fold(
    device const AkitaFp128 *source [[buffer(0)]],
    device const uint *offsets [[buffer(1)]],
    device const uint *challenge_positions [[buffer(2)]],
    device const char *coefficients [[buffer(3)]],
    device int *output [[buffer(4)]],
    constant RecursiveDecomposeFoldParams &params [[buffer(5)]],
    threadgroup short *digits [[threadgroup(0)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]])
{
    int accumulators[RECURSIVE_FOLD_MAX_DIGITS] = {0};
    ulong position = (ulong)group.x;
    for (ulong block = 0ul; block < params.blocks; ++block) {
        ulong ring = block * params.positions + position;
        if (ring >= params.source_rings) break;
        uint4 value = source[ring * params.ring_d + thread_index].limb;
        uint4 threshold = params.threshold.limb;
        bool negative = false;
        for (int limb = 3; limb >= 0; --limb) {
            if (value[limb] != threshold[limb]) {
                negative = value[limb] > threshold[limb];
                break;
            }
        }
        if (negative) {
            uint4 modulus = uint4(0u - AKITA_OFFSET, 0xffffffffu, 0xffffffffu, 0xffffffffu);
            ulong borrow = 0;
            for (uint limb = 0; limb < 4; ++limb) {
                ulong subtrahend = (ulong)modulus[limb] + borrow;
                ulong previous = value[limb];
                value[limb] = (uint)(previous - subtrahend);
                borrow = previous < subtrahend;
            }
        }
        // `negative` is bit 128 of the centered value: below an asymmetric threshold,
        // value - q underflows i128 and bit 127 is clear, so the first shift must fill
        // from `negative`, not from bit 127.
        uint bits = params.log_basis;
        uint mask = (1u << bits) - 1u;
        for (ulong digit = 0; digit < params.digits; ++digit) {
            uint raw = value.x & mask;
            int balanced = (int)raw - (raw >= (1u << (bits - 1u)) ? (1 << bits) : 0);
            digits[digit * params.ring_d + thread_index] = (short)balanced;
            value = uint4(
                (value.x >> bits) | (value.y << (32u - bits)),
                (value.y >> bits) | (value.z << (32u - bits)),
                (value.z >> bits) | (value.w << (32u - bits)),
                (value.w >> bits) | (negative ? (0xffffffffu << (32u - bits)) : 0u));
            ulong carry = balanced < 0;
            for (uint limb = 0; limb < 4; ++limb) {
                ulong next = (ulong)value[limb] + carry;
                value[limb] = (uint)next;
                carry = next >> 32u;
            }
            negative = (value.w >> 31u) != 0u;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint term = offsets[block]; term < offsets[block + 1ul]; ++term) {
            uint shift = challenge_positions[term];
            uint source_coefficient = (thread_index + params.ring_d - shift) & (params.ring_d - 1u);
            int multiplier = (int)coefficients[term];
            if (thread_index < shift) multiplier = -multiplier;
            #pragma unroll
            for (uint digit = 0u; digit < RECURSIVE_FOLD_MAX_DIGITS; ++digit) {
                if (digit < params.digits) {
                    accumulators[digit] += (int)digits[digit * params.ring_d + source_coefficient] * multiplier;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma unroll
    for (uint digit = 0u; digit < RECURSIVE_FOLD_MAX_DIGITS; ++digit) {
        if (digit < params.digits) {
            output[(position * params.digits + digit) * params.ring_d + thread_index] = accumulators[digit];
        }
    }
}
