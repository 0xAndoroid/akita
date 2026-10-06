// Dense signed-byte D128 rank-5 root microbench (Track B feasibility gate).
// Compiled after onehot.metal: reuses D512LinearNttPrime, the d512_ntt_*
// Montgomery helpers and the recursive-commit D128 twiddle layout
// (fwd_twiddles[len - 1 + offset], coefficient = lane + 32 * register).

#define BYTEROOT_RING_D 128u
#define BYTEROOT_RANKS 5u
#define BYTEROOT_SIMDS 16u
// Lazy accumulators fold their high word every 8 columns: a post-fold value
// is below 2^32 + 2^32 * (2^32 mod p) < 2^52 and eight products below p^2 <
// 2^60 add at most 2^63, so the u64 never wraps.
#define BYTEROOT_FOLD_COLUMNS 8u

struct ByteRootParams {
    ulong num_tasks;
    ulong positions;
    ulong positions_per_partial;
    ulong partials;
    ulong task_groups;
    ulong prime_index;
    ulong output_coefficients;
    ulong num_primes;
};

struct ByteRootMontAccumulator {
    int value;

    void clear() { value = 0; }

    void mac(int matrix, int source, D512LinearNttPrime prime) {
        value = d512_ntt_add(value, d512_ntt_mul(matrix, source, prime), prime.p);
    }

    void fold(uint) {}

    int montgomery_residue(D512LinearNttPrime) { return value; }
};

struct ByteRootLazyAccumulator {
    ulong value;

    void clear() { value = 0ul; }

    // Both operands are canonical Montgomery residues, so the plain product
    // sum is R^2 * sum(A * S) mod p.
    void mac(int matrix, int source, D512LinearNttPrime) {
        value += (ulong)(uint)matrix * (ulong)(uint)source;
    }

    void fold(uint two32_mod_p) {
        value = (value & 0xfffffffful) + (value >> 32u) * (ulong)two32_mod_p;
    }

    int montgomery_residue(D512LinearNttPrime prime) {
        uint canonical = (uint)(value % (ulong)(uint)prime.p);
        return d512_ntt_mul((int)canonical, 1, prime);
    }
};

template <uint TASKS, typename Acc>
inline void byteroot_matvec_body(
    device const char *source,
    device const int *matrix_ntt,
    device uint *partial_residues,
    device const D512LinearNttPrime *primes,
    device const int *fwd_twiddles,
    device const int *inv_twiddles,
    device const int *psi_pows,
    device const int *inverse_scale,
    constant ByteRootParams &params,
    uint thread_index,
    uint threadgroup_x)
{
    uint lane = thread_index & 31u;
    uint simd = thread_index >> 5u;
    ulong task_group = (ulong)threadgroup_x % params.task_groups;
    ulong partial = (ulong)threadgroup_x / params.task_groups;
    uint prime_index = (uint)params.prime_index;
    D512LinearNttPrime prime = primes[prime_index];
    ulong table_base = (ulong)prime_index * BYTEROOT_RING_D;

    ulong task[TASKS];
    bool live[TASKS];
    bool any_live = false;
    for (uint t = 0u; t < TASKS; ++t) {
        task[t] = ((task_group * BYTEROOT_SIMDS) + (ulong)simd) * TASKS + t;
        live[t] = task[t] < params.num_tasks;
        any_live = any_live || live[t];
    }
    if (!any_live) {
        return;
    }

    int psi[4];
    for (uint r = 0u; r < 4u; ++r) {
        psi[r] = psi_pows[table_base + (ulong)(lane + (r << 5u))];
    }
    int fwd_64_0 = fwd_twiddles[table_base + 63ul + (ulong)lane];
    int fwd_64_1 = fwd_twiddles[table_base + 95ul + (ulong)lane];
    int fwd_32 = fwd_twiddles[table_base + 31ul + (ulong)lane];
    int fwd_small[5];
    for (uint stage = 0u; stage < 5u; ++stage) {
        uint len = 16u >> stage;
        fwd_small[stage] = fwd_twiddles[table_base + (ulong)(len - 1u + (lane & (len - 1u)))];
    }
    uint two32_mod_p = (uint)((1ul << 32u) % (ulong)(uint)prime.p);

    Acc acc[TASKS][BYTEROOT_RANKS][4];
    for (uint t = 0u; t < TASKS; ++t) {
        for (uint rank = 0u; rank < BYTEROOT_RANKS; ++rank) {
            for (uint r = 0u; r < 4u; ++r) {
                acc[t][rank][r].clear();
            }
        }
    }

    ulong column_begin = partial * params.positions_per_partial;
    ulong column_end = column_begin + params.positions_per_partial;
    for (ulong column = column_begin; column < column_end; ++column) {
        int values[TASKS][4];
        for (uint t = 0u; t < TASKS; ++t) {
            ulong ring_base = (task[t] * params.positions + column) * BYTEROOT_RING_D;
            for (uint r = 0u; r < 4u; ++r) {
                int digit = live[t] ? (int)source[ring_base + (ulong)(lane + (r << 5u))] : 0;
                values[t][r] = d512_ntt_mul(d512_i32_to_mont(digit, prime), psi[r], prime);
            }
        }

        for (uint t = 0u; t < TASKS; ++t) {
            int value_0 = values[t][0];
            int value_1 = values[t][1];
            int value_2 = values[t][2];
            int value_3 = values[t][3];
            values[t][0] = d512_ntt_add(value_0, value_2, prime.p);
            values[t][2] = d512_ntt_mul(d512_ntt_sub(value_0, value_2, prime.p), fwd_64_0, prime);
            values[t][1] = d512_ntt_add(value_1, value_3, prime.p);
            values[t][3] = d512_ntt_mul(d512_ntt_sub(value_1, value_3, prime.p), fwd_64_1, prime);
            for (uint pair = 0u; pair < 4u; pair += 2u) {
                int lhs = values[t][pair];
                int rhs = values[t][pair + 1u];
                values[t][pair] = d512_ntt_add(lhs, rhs, prime.p);
                values[t][pair + 1u] = d512_ntt_mul(d512_ntt_sub(lhs, rhs, prime.p), fwd_32, prime);
            }
        }

        for (uint stage = 0u; stage < 5u; ++stage) {
            uint len = 16u >> stage;
            bool right_lane = (lane & len) != 0u;
            int twiddle = fwd_small[stage];
            for (uint t = 0u; t < TASKS; ++t) {
                for (uint r = 0u; r < 4u; ++r) {
                    int value = values[t][r];
                    int partner = simd_shuffle_xor(value, len);
                    int lhs = right_lane ? partner : value;
                    int rhs = right_lane ? value : partner;
                    values[t][r] = right_lane
                        ? d512_ntt_mul(d512_ntt_sub(lhs, rhs, prime.p), twiddle, prime)
                        : d512_ntt_add(lhs, rhs, prime.p);
                }
            }
        }

        // Matrix residues are laid out [prime][column][rank][coefficient], so
        // one column's five rank rings are one contiguous 2.5 KiB run.
        ulong matrix_base = (((ulong)prime_index * params.positions + column) * BYTEROOT_RANKS)
            * BYTEROOT_RING_D;
        for (uint rank = 0u; rank < BYTEROOT_RANKS; ++rank) {
            for (uint r = 0u; r < 4u; ++r) {
                int matrix = matrix_ntt[matrix_base + (ulong)(rank * BYTEROOT_RING_D + lane + (r << 5u))];
                for (uint t = 0u; t < TASKS; ++t) {
                    acc[t][rank][r].mac(matrix, values[t][r], prime);
                }
            }
        }
        if (((column + 1ul - column_begin) & (ulong)(BYTEROOT_FOLD_COLUMNS - 1u)) == 0ul) {
            for (uint t = 0u; t < TASKS; ++t) {
                for (uint rank = 0u; rank < BYTEROOT_RANKS; ++rank) {
                    for (uint r = 0u; r < 4u; ++r) {
                        acc[t][rank][r].fold(two32_mod_p);
                    }
                }
            }
        }
    }

    int inv_small[5];
    for (uint stage = 0u; stage < 5u; ++stage) {
        uint len = 1u << stage;
        inv_small[stage] = inv_twiddles[table_base + (ulong)(len - 1u + (lane & (len - 1u)))];
    }
    int inv_32 = inv_twiddles[table_base + 31ul + (ulong)lane];
    int inv_64_0 = inv_twiddles[table_base + 63ul + (ulong)lane];
    int inv_64_1 = inv_twiddles[table_base + 95ul + (ulong)lane];
    int scale[4];
    for (uint r = 0u; r < 4u; ++r) {
        scale[r] = inverse_scale[table_base + (ulong)(lane + (r << 5u))];
    }
    ulong residue_base = (partial * params.num_primes + (ulong)prime_index) * params.output_coefficients;
    for (uint t = 0u; t < TASKS; ++t) {
        if (!live[t]) {
            continue;
        }
        for (uint rank = 0u; rank < BYTEROOT_RANKS; ++rank) {
            int values[4];
            for (uint r = 0u; r < 4u; ++r) {
                values[r] = acc[t][rank][r].montgomery_residue(prime);
            }
            for (uint stage = 0u; stage < 5u; ++stage) {
                uint len = 1u << stage;
                bool right_lane = (lane & len) != 0u;
                int twiddle = inv_small[stage];
                for (uint r = 0u; r < 4u; ++r) {
                    int value = values[r];
                    int partner = simd_shuffle_xor(value, len);
                    int lhs = right_lane ? partner : value;
                    int rhs_raw = right_lane ? value : partner;
                    int rhs = d512_ntt_mul(rhs_raw, twiddle, prime);
                    values[r] = right_lane
                        ? d512_ntt_sub(lhs, rhs, prime.p)
                        : d512_ntt_add(lhs, rhs, prime.p);
                }
            }
            for (uint pair = 0u; pair < 4u; pair += 2u) {
                int lhs = values[pair];
                int rhs = d512_ntt_mul(values[pair + 1u], inv_32, prime);
                values[pair] = d512_ntt_add(lhs, rhs, prime.p);
                values[pair + 1u] = d512_ntt_sub(lhs, rhs, prime.p);
            }
            int value_0 = values[0];
            int value_1 = values[1];
            int rhs_2 = d512_ntt_mul(values[2], inv_64_0, prime);
            int rhs_3 = d512_ntt_mul(values[3], inv_64_1, prime);
            values[0] = d512_ntt_add(value_0, rhs_2, prime.p);
            values[2] = d512_ntt_sub(value_0, rhs_2, prime.p);
            values[1] = d512_ntt_add(value_1, rhs_3, prime.p);
            values[3] = d512_ntt_sub(value_1, rhs_3, prime.p);

            ulong output_base = (task[t] * BYTEROOT_RANKS + (ulong)rank) * BYTEROOT_RING_D;
            for (uint r = 0u; r < 4u; ++r) {
                int scaled = d512_ntt_mul(values[r], scale[r], prime);
                int canonical = d512_ntt_reduce((long)d512_ntt_mul_raw(scaled, 1, prime), prime.p);
                partial_residues[residue_base + output_base + (ulong)(lane + (r << 5u))] =
                    (uint)canonical;
            }
        }
    }
}

struct ByteRootShoupConstant {
    uint value;
    uint quotient;
};

inline ByteRootShoupConstant byteroot_shoup_constant(uint value, uint p) {
    ByteRootShoupConstant shoup;
    shoup.value = value;
    shoup.quotient = (uint)(((ulong)value << 32u) / (ulong)p);
    return shoup;
}

// Harvey/Shoup product for p < 2^30: any a < 2^32 and a canonical constant
// give a result in [0, 2p) congruent to a * value.
inline uint byteroot_shoup_mul(uint a, ByteRootShoupConstant shoup, uint p) {
    return a * shoup.value - mulhi(a, shoup.quotient) * p;
}

inline uint byteroot_reduce_2p(uint value, uint two_p) {
    return value >= two_p ? value - two_p : value;
}

// Plain-form forward transform with Harvey lazy butterflies (values in
// [0, 2p), sums below 4p < 2^32). The matrix residues stay in Montgomery form,
// so the u64 MAC sum is already R * sum(A * S) and feeds the Montgomery
// inverse transform unchanged.
template <uint TASKS>
inline void byteroot_matvec_fast_body(
    device const char *source,
    device const int *matrix_ntt,
    device uint *partial_residues,
    device const D512LinearNttPrime *primes,
    device const int *fwd_twiddles,
    device const int *inv_twiddles,
    device const int *psi_pows,
    device const int *inverse_scale,
    constant ByteRootParams &params,
    uint thread_index,
    uint threadgroup_x)
{
    uint lane = thread_index & 31u;
    uint simd = thread_index >> 5u;
    ulong task_group = (ulong)threadgroup_x % params.task_groups;
    ulong partial = (ulong)threadgroup_x / params.task_groups;
    uint prime_index = (uint)params.prime_index;
    D512LinearNttPrime prime = primes[prime_index];
    ulong table_base = (ulong)prime_index * BYTEROOT_RING_D;
    uint p = (uint)prime.p;
    uint two_p = p << 1u;

    ulong task[TASKS];
    bool live[TASKS];
    bool any_live = false;
    for (uint t = 0u; t < TASKS; ++t) {
        task[t] = ((task_group * BYTEROOT_SIMDS) + (ulong)simd) * TASKS + t;
        live[t] = task[t] < params.num_tasks;
        any_live = any_live || live[t];
    }
    if (!any_live) {
        return;
    }

    ByteRootShoupConstant psi[4];
    for (uint r = 0u; r < 4u; ++r) {
        psi[r] = byteroot_shoup_constant(
            (uint)d512_ntt_mul(psi_pows[table_base + (ulong)(lane + (r << 5u))], 1, prime), p);
    }
    ByteRootShoupConstant fwd_64_0 = byteroot_shoup_constant(
        (uint)d512_ntt_mul(fwd_twiddles[table_base + 63ul + (ulong)lane], 1, prime), p);
    ByteRootShoupConstant fwd_64_1 = byteroot_shoup_constant(
        (uint)d512_ntt_mul(fwd_twiddles[table_base + 95ul + (ulong)lane], 1, prime), p);
    ByteRootShoupConstant fwd_32 = byteroot_shoup_constant(
        (uint)d512_ntt_mul(fwd_twiddles[table_base + 31ul + (ulong)lane], 1, prime), p);
    // Left lanes multiply their sum by one so every lane runs one product.
    ByteRootShoupConstant stage_shoup[5];
    for (uint stage = 0u; stage < 5u; ++stage) {
        uint len = 16u >> stage;
        uint twiddle = (lane & len) != 0u
            ? (uint)d512_ntt_mul(
                fwd_twiddles[table_base + (ulong)(len - 1u + (lane & (len - 1u)))], 1, prime)
            : 1u;
        stage_shoup[stage] = byteroot_shoup_constant(twiddle, p);
    }
    uint two32_mod_p = (uint)((1ul << 32u) % (ulong)p);

    ulong acc[TASKS][BYTEROOT_RANKS][4];
    for (uint t = 0u; t < TASKS; ++t) {
        for (uint rank = 0u; rank < BYTEROOT_RANKS; ++rank) {
            for (uint r = 0u; r < 4u; ++r) {
                acc[t][rank][r] = 0ul;
            }
        }
    }

    ulong column_begin = partial * params.positions_per_partial;
    ulong column_end = column_begin + params.positions_per_partial;
    for (ulong column = column_begin; column < column_end; ++column) {
        uint values[TASKS][4];
        for (uint t = 0u; t < TASKS; ++t) {
            ulong ring_base = (task[t] * params.positions + column) * BYTEROOT_RING_D;
            for (uint r = 0u; r < 4u; ++r) {
                int digit = live[t] ? (int)source[ring_base + (ulong)(lane + (r << 5u))] : 0;
                uint lifted = (uint)(digit < 0 ? digit + (int)p : digit);
                values[t][r] = byteroot_shoup_mul(lifted, psi[r], p);
            }
        }

        for (uint t = 0u; t < TASKS; ++t) {
            uint value_0 = values[t][0];
            uint value_1 = values[t][1];
            uint value_2 = values[t][2];
            uint value_3 = values[t][3];
            values[t][0] = byteroot_reduce_2p(value_0 + value_2, two_p);
            values[t][2] = byteroot_shoup_mul(value_0 - value_2 + two_p, fwd_64_0, p);
            values[t][1] = byteroot_reduce_2p(value_1 + value_3, two_p);
            values[t][3] = byteroot_shoup_mul(value_1 - value_3 + two_p, fwd_64_1, p);
            for (uint pair = 0u; pair < 4u; pair += 2u) {
                uint lhs = values[t][pair];
                uint rhs = values[t][pair + 1u];
                values[t][pair] = byteroot_reduce_2p(lhs + rhs, two_p);
                values[t][pair + 1u] = byteroot_shoup_mul(lhs - rhs + two_p, fwd_32, p);
            }
        }

        for (uint stage = 0u; stage < 5u; ++stage) {
            uint len = 16u >> stage;
            bool right_lane = (lane & len) != 0u;
            ByteRootShoupConstant shoup = stage_shoup[stage];
            for (uint t = 0u; t < TASKS; ++t) {
                for (uint r = 0u; r < 4u; ++r) {
                    uint value = values[t][r];
                    uint partner = simd_shuffle_xor(value, len);
                    uint sum = right_lane ? partner - value + two_p : value + partner;
                    values[t][r] = byteroot_shoup_mul(sum, shoup, p);
                }
            }
        }

        for (uint t = 0u; t < TASKS; ++t) {
            for (uint r = 0u; r < 4u; ++r) {
                values[t][r] = values[t][r] >= p ? values[t][r] - p : values[t][r];
            }
        }
        ulong matrix_base = (((ulong)prime_index * params.positions + column) * BYTEROOT_RANKS)
            * BYTEROOT_RING_D;
        for (uint rank = 0u; rank < BYTEROOT_RANKS; ++rank) {
            for (uint r = 0u; r < 4u; ++r) {
                ulong matrix = (ulong)(uint)matrix_ntt[
                    matrix_base + (ulong)(rank * BYTEROOT_RING_D + lane + (r << 5u))];
                for (uint t = 0u; t < TASKS; ++t) {
                    acc[t][rank][r] += matrix * (ulong)values[t][r];
                }
            }
        }
        if (((column + 1ul - column_begin) & (ulong)(BYTEROOT_FOLD_COLUMNS - 1u)) == 0ul) {
            for (uint t = 0u; t < TASKS; ++t) {
                for (uint rank = 0u; rank < BYTEROOT_RANKS; ++rank) {
                    for (uint r = 0u; r < 4u; ++r) {
                        ulong value = acc[t][rank][r];
                        acc[t][rank][r] = (value & 0xfffffffful) + (value >> 32u) * (ulong)two32_mod_p;
                    }
                }
            }
        }
    }

    int inv_small[5];
    for (uint stage = 0u; stage < 5u; ++stage) {
        uint len = 1u << stage;
        inv_small[stage] = inv_twiddles[table_base + (ulong)(len - 1u + (lane & (len - 1u)))];
    }
    int inv_32 = inv_twiddles[table_base + 31ul + (ulong)lane];
    int inv_64_0 = inv_twiddles[table_base + 63ul + (ulong)lane];
    int inv_64_1 = inv_twiddles[table_base + 95ul + (ulong)lane];
    int scale[4];
    for (uint r = 0u; r < 4u; ++r) {
        scale[r] = inverse_scale[table_base + (ulong)(lane + (r << 5u))];
    }
    ulong residue_base = (partial * params.num_primes + (ulong)prime_index) * params.output_coefficients;
    for (uint t = 0u; t < TASKS; ++t) {
        if (!live[t]) {
            continue;
        }
        for (uint rank = 0u; rank < BYTEROOT_RANKS; ++rank) {
            int values[4];
            for (uint r = 0u; r < 4u; ++r) {
                values[r] = (int)(acc[t][rank][r] % (ulong)p);
            }
            for (uint stage = 0u; stage < 5u; ++stage) {
                uint len = 1u << stage;
                bool right_lane = (lane & len) != 0u;
                int twiddle = inv_small[stage];
                for (uint r = 0u; r < 4u; ++r) {
                    int value = values[r];
                    int partner = simd_shuffle_xor(value, len);
                    int lhs = right_lane ? partner : value;
                    int rhs_raw = right_lane ? value : partner;
                    int rhs = d512_ntt_mul(rhs_raw, twiddle, prime);
                    values[r] = right_lane
                        ? d512_ntt_sub(lhs, rhs, prime.p)
                        : d512_ntt_add(lhs, rhs, prime.p);
                }
            }
            for (uint pair = 0u; pair < 4u; pair += 2u) {
                int lhs = values[pair];
                int rhs = d512_ntt_mul(values[pair + 1u], inv_32, prime);
                values[pair] = d512_ntt_add(lhs, rhs, prime.p);
                values[pair + 1u] = d512_ntt_sub(lhs, rhs, prime.p);
            }
            int value_0 = values[0];
            int value_1 = values[1];
            int rhs_2 = d512_ntt_mul(values[2], inv_64_0, prime);
            int rhs_3 = d512_ntt_mul(values[3], inv_64_1, prime);
            values[0] = d512_ntt_add(value_0, rhs_2, prime.p);
            values[2] = d512_ntt_sub(value_0, rhs_2, prime.p);
            values[1] = d512_ntt_add(value_1, rhs_3, prime.p);
            values[3] = d512_ntt_sub(value_1, rhs_3, prime.p);

            ulong output_base = (task[t] * BYTEROOT_RANKS + (ulong)rank) * BYTEROOT_RING_D;
            for (uint r = 0u; r < 4u; ++r) {
                int scaled = d512_ntt_mul(values[r], scale[r], prime);
                int canonical = d512_ntt_reduce((long)d512_ntt_mul_raw(scaled, 1, prime), prime.p);
                partial_residues[residue_base + output_base + (ulong)(lane + (r << 5u))] =
                    (uint)canonical;
            }
        }
    }
}

#define BYTEROOT_MATVEC_KERNEL(NAME, BODY)                                        \
    kernel void NAME(                                                                   \
        device const char *source [[buffer(0)]],                                        \
        device const int *matrix_ntt [[buffer(1)]],                                     \
        device uint *partial_residues [[buffer(2)]],                                    \
        device const D512LinearNttPrime *primes [[buffer(3)]],                          \
        device const int *fwd_twiddles [[buffer(4)]],                                   \
        device const int *inv_twiddles [[buffer(5)]],                                   \
        device const int *psi_pows [[buffer(6)]],                                       \
        device const int *inverse_scale [[buffer(7)]],                                  \
        constant ByteRootParams &params [[buffer(8)]],                                  \
        uint thread_index [[thread_index_in_threadgroup]],                              \
        uint3 threadgroup_index [[threadgroup_position_in_grid]])                       \
    {                                                                                   \
        BODY(source, matrix_ntt, partial_residues, primes,                              \
            fwd_twiddles, inv_twiddles, psi_pows, inverse_scale, params, thread_index,  \
            threadgroup_index.x);                                                       \
    }

BYTEROOT_MATVEC_KERNEL(akita_byteroot_matvec_mont1, (byteroot_matvec_body<1u, ByteRootMontAccumulator>))
BYTEROOT_MATVEC_KERNEL(akita_byteroot_matvec_mont2, (byteroot_matvec_body<2u, ByteRootMontAccumulator>))
BYTEROOT_MATVEC_KERNEL(akita_byteroot_matvec_lazy1, (byteroot_matvec_body<1u, ByteRootLazyAccumulator>))
BYTEROOT_MATVEC_KERNEL(akita_byteroot_matvec_lazy2, (byteroot_matvec_body<2u, ByteRootLazyAccumulator>))
BYTEROOT_MATVEC_KERNEL(akita_byteroot_matvec_fast1, byteroot_matvec_fast_body<1u>)
BYTEROOT_MATVEC_KERNEL(akita_byteroot_matvec_fast2, byteroot_matvec_fast_body<2u>)

// Sums the position partials of one (coefficient, prime) residue into the
// coefficient-major layout read by akita_fp128_recursive_commit_reconstruct.
kernel void akita_byteroot_reduce_partials(
    device const uint *partial_residues [[buffer(0)]],
    device uint *residues [[buffer(1)]],
    device const D512LinearNttPrime *primes [[buffer(2)]],
    constant ByteRootParams &params [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    ulong coefficient = (ulong)index / params.num_primes;
    if (coefficient >= params.output_coefficients) {
        return;
    }
    ulong prime_index = (ulong)index - coefficient * params.num_primes;
    ulong sum = 0ul;
    for (ulong partial = 0ul; partial < params.partials; ++partial) {
        sum += (ulong)partial_residues[
            (partial * params.num_primes + prime_index) * params.output_coefficients + coefficient];
    }
    residues[(ulong)index] = (uint)(sum % (ulong)(uint)primes[prime_index].p);
}
