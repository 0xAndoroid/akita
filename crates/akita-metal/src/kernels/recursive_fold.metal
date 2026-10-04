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

kernel void akita_fp128_recursive_decompose(
    device const AkitaFp128 *source [[buffer(0)]],
    device short *digits [[buffer(1)]],
    constant RecursiveDecomposeFoldParams &params [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if ((ulong)index >= params.source_rings * params.ring_d) return;
    uint4 value = source[index].limb;
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
    ulong ring = (ulong)index / params.ring_d;
    uint coefficient = index % params.ring_d;
    for (ulong digit = 0; digit < params.digits; ++digit) {
        uint raw = value.x & mask;
        int balanced = (int)raw - (raw >= (1u << (bits - 1u)) ? (1 << bits) : 0);
        digits[(ring * params.digits + digit) * params.ring_d + coefficient] = (short)balanced;
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
}

kernel void akita_recursive_sparse_digit_fold(
    device const short *digits [[buffer(0)]],
    device const uint *offsets [[buffer(1)]],
    device const uint *positions [[buffer(2)]],
    device const char *coefficients [[buffer(3)]],
    device int *output [[buffer(4)]],
    constant RecursiveDecomposeFoldParams &params [[buffer(5)]],
    uint index [[thread_position_in_grid]])
{
    if ((ulong)index >= params.output_coefficients) return;
    uint destination = index % params.ring_d;
    ulong digit = ((ulong)index / params.ring_d) % params.digits;
    ulong position = (ulong)index / (params.digits * params.ring_d);
    int accumulator = 0;
    for (ulong block = 0; block < params.blocks; ++block) {
        ulong ring = block * params.positions + position;
        if (ring >= params.source_rings) break;
        ulong source_base = (ring * params.digits + digit) * params.ring_d;
        for (uint term = offsets[block]; term < offsets[block + 1]; ++term) {
            uint shift = positions[term];
            uint source = (destination + params.ring_d - shift) & (params.ring_d - 1u);
            int product = (int)digits[source_base + source] * (int)coefficients[term];
            accumulator += destination < shift ? -product : product;
        }
    }
    output[index] = accumulator;
}
