//! Dense signed-byte D128 rank-5 root microbench (Track B feasibility gate).
//!
//! `t[task, r] = sum_j A[r, j] * s[task, j]` in `F[X]/(X^128 + 1)` with
//! `A: 5 x 2^16` rings and signed-byte source rings laid out
//! `[task][position][coefficient]`, `task = source column * blocks + block`.
//! Exactness comes from the recursive-commit D128 CRT resources: six ~2^30
//! primes, Garner reconstruction in `akita_fp128_recursive_commit_reconstruct`.

use akita_algebra::tables::Q128_RAW_PRIMES;
use jolt_field::{CanonicalEncoding, Zero};
use metal::{CaptureDescriptor, CaptureManager, MTLCaptureDestination};

use super::*;

const SOURCE: &str = concat!(
    include_str!("../kernels/onehot.metal"),
    "\n",
    include_str!("../kernels/byteroot.metal"),
);
const RING_D: usize = 128;
const RANKS: usize = 5;
const PRIMES: usize = FP128_D512_LINEAR_RELATION_NUM_PRIMES;
const PARTIALS: usize = 16;
const SIMDS: usize = 16;
const THREADS: u64 = 512;
const ROOT_POSITIONS: usize = 1 << 16;
const SOURCE_COLUMNS: usize = 30;
const SOURCE_BOUND: u64 = 128;

#[derive(Clone, Copy, Debug)]
enum Variant {
    Mont1,
    Mont2,
    Lazy1,
    Lazy2,
    Fast1,
    Fast2,
}

const VARIANTS: [Variant; 6] = [
    Variant::Mont1,
    Variant::Mont2,
    Variant::Lazy1,
    Variant::Lazy2,
    Variant::Fast1,
    Variant::Fast2,
];

impl Variant {
    fn name(self) -> &'static str {
        match self {
            Self::Mont1 => "mont1",
            Self::Mont2 => "mont2",
            Self::Lazy1 => "lazy1",
            Self::Lazy2 => "lazy2",
            Self::Fast1 => "fast1",
            Self::Fast2 => "fast2",
        }
    }

    fn tasks_per_simd(self) -> usize {
        match self {
            Self::Mont1 | Self::Lazy1 | Self::Fast1 => 1,
            Self::Mont2 | Self::Lazy2 | Self::Fast2 => 2,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct ByteRootParams {
    num_tasks: u64,
    positions: u64,
    positions_per_partial: u64,
    partials: u64,
    task_groups: u64,
    prime_index: u64,
    output_coefficients: u64,
    num_primes: u64,
}

const _: [(); 64] = [(); size_of::<ByteRootParams>()];

#[derive(Clone, Copy, Debug)]
struct RootTimings {
    matvec_span: Duration,
    matvec_active: Duration,
    finish: Duration,
    total_span: Duration,
    wall: Duration,
}

struct ByteRoot<'a> {
    runtime: &'a MetalRuntime,
    matvec: Vec<ComputePipelineState>,
    reduce: ComputePipelineState,
}

impl<'a> ByteRoot<'a> {
    fn new(runtime: &'a MetalRuntime) -> Self {
        let options = CompileOptions::new();
        options.set_fast_math_enabled(false);
        let library = runtime
            .device
            .new_library_with_source(SOURCE, &options)
            .unwrap();
        let pipeline = |name: &str| {
            let function = library.get_function(name, None).unwrap();
            runtime
                .device
                .new_compute_pipeline_state_with_function(&function)
                .unwrap()
        };
        let matvec = VARIANTS
            .iter()
            .map(|variant| pipeline(&format!("akita_byteroot_matvec_{}", variant.name())))
            .collect::<Vec<_>>();
        for (variant, pipeline) in VARIANTS.iter().zip(&matvec) {
            println!(
                "pipeline\t{}\tmax_threads={}\texecution_width={}",
                variant.name(),
                pipeline.max_total_threads_per_threadgroup(),
                pipeline.thread_execution_width()
            );
            assert!(pipeline.max_total_threads_per_threadgroup() >= THREADS);
        }
        Self {
            runtime,
            matvec,
            reduce: pipeline("akita_byteroot_reduce_partials"),
        }
    }

    fn matrix_params(positions: usize) -> RecursiveCommitParams {
        RecursiveCommitParams {
            num_blocks: 1,
            blocks_per_group: FP128_RECURSIVE_COMMIT_BLOCKS_PER_GROUP as u64,
            num_block_groups: 1,
            num_rows: RANKS as u64,
            num_cols: positions as u64,
            ring_d: RING_D as u64,
            num_primes: PRIMES as u64,
            matrix_rings: (RANKS * positions) as u64,
            output_coefficients: (RANKS * RING_D) as u64,
            rhs_abs_bound: SOURCE_BOUND,
        }
    }

    /// `matrix` holds `positions * RANKS` canonical rings in
    /// `[position][rank][coefficient]` order; the transform keeps that ring
    /// order per prime.
    fn prepare_matrix(&self, matrix: &Buffer, positions: usize) -> (Buffer, Option<Duration>) {
        let outcome = self
            .runtime
            .prepare_fp128_recursive_commit_matrix::<RING_D>(matrix, Self::matrix_params(positions))
            .unwrap();
        (outcome.buffer, outcome.timings.gpu)
    }

    fn run(
        &self,
        variant: Variant,
        matrix_ntt: &Buffer,
        source: &Buffer,
        num_tasks: usize,
        positions: usize,
    ) -> (Vec<Fp128Limbs>, RootTimings) {
        assert!(positions.is_multiple_of(PARTIALS));
        assert_eq!(source.length(), (num_tasks * positions * RING_D) as u64);
        assert!(CrtCapacity::from_prime_moduli(
            FP128_D512_LINEAR_RELATION_RAW_PRIMES.map(|prime| prime as u128)
        )
        .supports_modulus(positions, RING_D, field_modulus(), SOURCE_BOUND));
        let runtime = self.runtime;
        let resources = runtime.recursive_commit_resources(RING_D).unwrap();
        let tasks_per_group = SIMDS * variant.tasks_per_simd();
        let task_groups = num_tasks.div_ceil(tasks_per_group);
        let output_coefficients = num_tasks * RANKS * RING_D;
        let params = ByteRootParams {
            num_tasks: num_tasks as u64,
            positions: positions as u64,
            positions_per_partial: (positions / PARTIALS) as u64,
            partials: PARTIALS as u64,
            task_groups: task_groups as u64,
            prime_index: 0,
            output_coefficients: output_coefficients as u64,
            num_primes: PRIMES as u64,
        };
        autoreleasepool(|| {
            let partial_residues = runtime
                .private_buffer(PARTIALS * PRIMES * output_coefficients * size_of::<u32>())
                .unwrap();
            let residues = runtime
                .private_buffer(PRIMES * output_coefficients * size_of::<u32>())
                .unwrap();
            let output = runtime
                .shared_buffer(output_coefficients * size_of::<Fp128Limbs>())
                .unwrap();
            let start = Instant::now();
            let mut commands = Vec::with_capacity(PRIMES);
            for prime_index in 0..PRIMES {
                let command = runtime.queue.new_command_buffer();
                command.set_label("Akita byte-root matvec");
                let encoder = command.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&self.matvec[variant as usize]);
                encoder.set_buffer(0, Some(source), 0);
                encoder.set_buffer(1, Some(matrix_ntt), 0);
                encoder.set_buffer(2, Some(&partial_residues), 0);
                encoder.set_buffer(3, Some(&resources.primes), 0);
                encoder.set_buffer(4, Some(&resources.fwd_twiddles), 0);
                encoder.set_buffer(5, Some(&resources.inv_twiddles), 0);
                encoder.set_buffer(6, Some(&resources.psi_pows), 0);
                encoder.set_buffer(7, Some(&resources.inverse_scale), 0);
                set_inline_bytes(
                    encoder,
                    8,
                    &ByteRootParams {
                        prime_index: prime_index as u64,
                        ..params
                    },
                );
                encoder.dispatch_thread_groups(
                    MTLSize::new((task_groups * PARTIALS) as u64, 1, 1),
                    MTLSize::new(THREADS, 1, 1),
                );
                encoder.end_encoding();
                command.commit();
                commands.push(command);
            }

            let finish = runtime.queue.new_command_buffer();
            finish.set_label("Akita byte-root partial reduction and CRT reconstruction");
            let encoder = finish.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.reduce);
            encoder.set_buffer(0, Some(&partial_residues), 0);
            encoder.set_buffer(1, Some(&residues), 0);
            encoder.set_buffer(2, Some(&resources.primes), 0);
            set_inline_bytes(encoder, 3, &params);
            encoder.dispatch_threads(
                MTLSize::new((output_coefficients * PRIMES) as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
            encoder.end_encoding();
            let encoder = finish.new_compute_command_encoder();
            encoder
                .set_compute_pipeline_state(&runtime.fp128_recursive_commit_reconstruct_pipeline);
            encoder.set_buffer(0, Some(&residues), 0);
            encoder.set_buffer(1, Some(&output), 0);
            encoder.set_buffer(2, Some(&resources.primes), 0);
            encoder.set_buffer(3, Some(&resources.garner_gamma), 0);
            encoder.set_buffer(4, Some(&resources.field_partial_products), 0);
            set_inline_bytes(
                encoder,
                5,
                &RecursiveCommitParams {
                    output_coefficients: output_coefficients as u64,
                    ..Self::matrix_params(positions)
                },
            );
            encoder.dispatch_thread_groups(
                MTLSize::new(
                    output_coefficients.div_ceil(FP128_RECURSIVE_COMMIT_RECONSTRUCT_THREADS) as u64,
                    1,
                    1,
                ),
                MTLSize::new(FP128_RECURSIVE_COMMIT_RECONSTRUCT_THREADS as u64, 1, 1),
            );
            encoder.end_encoding();
            finish.commit();
            finish.wait_until_completed();
            let wall = start.elapsed();
            for command in &commands {
                validate_completed_command(command).unwrap();
            }
            validate_completed_command(finish).unwrap();
            let timings = RootTimings {
                matvec_span: completed_commands_gpu_span(commands[0], commands[PRIMES - 1])
                    .unwrap(),
                matvec_active: commands
                    .iter()
                    .map(|command| completed_command_gpu_time(command).unwrap())
                    .sum(),
                finish: completed_command_gpu_time(finish).unwrap(),
                total_span: completed_commands_gpu_span(commands[0], finish).unwrap(),
                wall,
            };
            // SAFETY: `output` is live shared storage for exactly
            // `output_coefficients` aligned `Fp128Limbs` values.
            let coefficients = unsafe {
                std::slice::from_raw_parts(
                    output.contents().cast::<Fp128Limbs>(),
                    output_coefficients,
                )
                .to_vec()
            };
            (coefficients, timings)
        })
    }
}

fn field_modulus() -> u128 {
    (-F::one()).to_canonical_u128() + 1
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn random_field(state: &mut u64) -> F {
    let value = (u128::from(splitmix64(state)) << 64) | u128::from(splitmix64(state));
    F::from_u128_checked(value % field_modulus()).unwrap()
}

/// A shared buffer of `len` bytes written by `fill` before any command uses it.
fn shared_bytes(runtime: &MetalRuntime, len: usize, fill: impl FnOnce(&mut [u8])) -> Buffer {
    let buffer = runtime.shared_buffer(len).unwrap();
    // SAFETY: fresh shared storage of exactly `len` bytes; no command has
    // been encoded against it.
    fill(unsafe { std::slice::from_raw_parts_mut(buffer.contents().cast::<u8>(), len) });
    buffer
}

fn fill_random_bytes(destination: &mut [u8], seed: u64) {
    let threads = std::thread::available_parallelism().map_or(8, usize::from);
    let part = destination.len().div_ceil(threads).max(8);
    std::thread::scope(|scope| {
        for (index, chunk) in destination.chunks_mut(part).enumerate() {
            scope.spawn(move || {
                let mut state = seed ^ (index as u64 + 1).wrapping_mul(0xd1b5_4a32_d192_ed03);
                for word in chunk.chunks_mut(8) {
                    let bytes = splitmix64(&mut state).to_le_bytes();
                    word.copy_from_slice(&bytes[..word.len()]);
                }
            });
        }
    });
}

fn matrix_buffer(
    runtime: &MetalRuntime,
    positions: usize,
    mut element: impl FnMut(usize) -> F,
) -> (Buffer, Vec<F>) {
    let count = positions * RANKS * RING_D;
    let mut field = Vec::with_capacity(count);
    let buffer = shared_bytes(runtime, count * size_of::<Fp128Limbs>(), |bytes| {
        // SAFETY: shared buffers are page aligned and `bytes` holds exactly
        // `count` limb values.
        let limbs = unsafe {
            std::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast::<Fp128Limbs>(), count)
        };
        for (index, limb) in limbs.iter_mut().enumerate() {
            let value = element(index);
            *limb = Fp128Limbs::from_field(value);
            field.push(value);
        }
    });
    (buffer, field)
}

/// Schoolbook `sum_j A[r, j] * s[task, j]` in `F[X]/(X^128 + 1)` over the
/// listed positions, independent of every NTT/CRT table.
fn cpu_root(
    matrix: &[F],
    source: &[i8],
    num_tasks: usize,
    positions: usize,
    columns: &[usize],
) -> Vec<F> {
    let mut output = vec![F::zero(); num_tasks * RANKS * RING_D];
    std::thread::scope(|scope| {
        for (task, task_output) in output.chunks_mut(RANKS * RING_D).enumerate() {
            scope.spawn(move || {
                for &column in columns {
                    let ring = &source[(task * positions + column) * RING_D..][..RING_D];
                    let ring = ring
                        .iter()
                        .map(|&digit| F::from_i64(i64::from(digit)))
                        .collect::<Vec<_>>();
                    for rank in 0..RANKS {
                        let a = &matrix[(column * RANKS + rank) * RING_D..][..RING_D];
                        let out = &mut task_output[rank * RING_D..][..RING_D];
                        for (k, &a_k) in a.iter().enumerate() {
                            for (m, &s_m) in ring.iter().enumerate() {
                                let product = a_k * s_m;
                                if k + m < RING_D {
                                    out[k + m] += product;
                                } else {
                                    out[k + m - RING_D] -= product;
                                }
                            }
                        }
                    }
                }
            });
        }
    });
    output
}

fn assert_exact(gpu: &[Fp128Limbs], expected: &[F], label: &str) {
    assert_eq!(gpu.len(), expected.len(), "{label}: length");
    let mismatches = gpu
        .iter()
        .zip(expected)
        .enumerate()
        .filter(|(_, (gpu, expected))| **gpu != Fp128Limbs::from_field(**expected))
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    assert!(
        mismatches.is_empty(),
        "{label}: {} of {} coefficients differ; first at {}",
        mismatches.len(),
        gpu.len(),
        mismatches[0]
    );
}

/// Library compile only: no command buffer is created.
#[test]
#[ignore = "Metal compile check for the microbench"]
fn byteroot_library_compiles() {
    let runtime = MetalRuntime::new().unwrap();
    let root = ByteRoot::new(&runtime);
    assert_eq!(root.matvec.len(), VARIANTS.len());
}

#[test]
#[ignore = "GPU microbench; run through gpu-window.sh"]
fn byteroot_matches_cpu_schoolbook_small() {
    const POSITIONS: usize = 256;
    const TASKS: usize = 18;
    let runtime = MetalRuntime::new().unwrap();
    let root = ByteRoot::new(&runtime);
    let half = F::from_u128_checked((field_modulus() - 1) / 2).unwrap();
    let mut state = 0x6279_7465_726f_6f74u64;
    let (matrix, matrix_field) = matrix_buffer(&runtime, POSITIONS, |index| match index % 23 {
        0 => half,
        1 => half + F::one(),
        2 => -F::one(),
        3 => F::zero(),
        _ => random_field(&mut state),
    });
    let mut digits = Vec::new();
    let source = shared_bytes(&runtime, TASKS * POSITIONS * RING_D, |bytes| {
        fill_random_bytes(bytes, 7);
        for (index, byte) in bytes.iter_mut().enumerate() {
            match index % 31 {
                0 => *byte = 0x80,
                1 => *byte = 0x7f,
                _ => {}
            }
        }
        digits = bytes.iter().map(|&byte| byte as i8).collect();
    });
    let columns = (0..POSITIONS).collect::<Vec<_>>();
    let expected = cpu_root(&matrix_field, &digits, TASKS, POSITIONS, &columns);
    let (matrix_ntt, _) = root.prepare_matrix(&matrix, POSITIONS);
    for variant in VARIANTS {
        let (gpu, _) = root.run(variant, &matrix_ntt, &source, TASKS, POSITIONS);
        assert_exact(&gpu, &expected, variant.name());
    }
}

#[test]
#[ignore = "GPU microbench; run through gpu-window.sh"]
fn byteroot_full_width_sparse_matches_cpu() {
    const TASKS: usize = 3;
    let columns = [0, 1, 4095, 4096, 32_767, 40_000, 61_439, 65_535];
    let runtime = MetalRuntime::new().unwrap();
    let root = ByteRoot::new(&runtime);
    let mut state = 0x7370_6172_7365u64;
    let (matrix, matrix_field) =
        matrix_buffer(&runtime, ROOT_POSITIONS, |_| random_field(&mut state));
    let mut digits = Vec::new();
    let source = shared_bytes(&runtime, TASKS * ROOT_POSITIONS * RING_D, |bytes| {
        bytes.fill(0);
        for task in 0..TASKS {
            for &column in &columns {
                let start = (task * ROOT_POSITIONS + column) * RING_D;
                fill_random_bytes(
                    &mut bytes[start..start + RING_D],
                    (task * 65_536 + column) as u64,
                );
            }
        }
        digits = bytes.iter().map(|&byte| byte as i8).collect();
    });
    let expected = cpu_root(&matrix_field, &digits, TASKS, ROOT_POSITIONS, &columns);
    let (matrix_ntt, _) = root.prepare_matrix(&matrix, ROOT_POSITIONS);
    for variant in VARIANTS {
        let (gpu, _) = root.run(variant, &matrix_ntt, &source, TASKS, ROOT_POSITIONS);
        assert_exact(&gpu, &expected, variant.name());
    }
}

/// All `A` coefficients `(q - 1) / 2` and all digits `-128`: coefficient `i`
/// of every rank is `P * a * s * (2i - 126)`, reaching the CRT design bound
/// `P * 128 * floor(q / 2) * 128` at `i = 127`.
#[test]
#[ignore = "GPU microbench; run through gpu-window.sh"]
fn byteroot_full_width_crt_extreme() {
    const TASKS: usize = 2;
    for primes in [FP128_D512_LINEAR_RELATION_RAW_PRIMES, Q128_RAW_PRIMES] {
        assert!(
            CrtCapacity::from_prime_moduli(primes.map(|prime| prime as u128)).supports_modulus(
                ROOT_POSITIONS,
                RING_D,
                field_modulus(),
                SOURCE_BOUND
            )
        );
    }
    let runtime = MetalRuntime::new().unwrap();
    let root = ByteRoot::new(&runtime);
    let half = F::from_u128_checked((field_modulus() - 1) / 2).unwrap();
    let (matrix, _) = matrix_buffer(&runtime, ROOT_POSITIONS, |_| half);
    let source = shared_bytes(&runtime, TASKS * ROOT_POSITIONS * RING_D, |bytes| {
        bytes.fill(0x80);
    });
    let scale = F::from_u64(ROOT_POSITIONS as u64) * half * F::from_i64(-128);
    let expected = (0..TASKS * RANKS * RING_D)
        .map(|index| scale * F::from_i64(2 * (index % RING_D) as i64 - 126))
        .collect::<Vec<_>>();
    let (matrix_ntt, _) = root.prepare_matrix(&matrix, ROOT_POSITIONS);
    for variant in VARIANTS {
        let (gpu, _) = root.run(variant, &matrix_ntt, &source, TASKS, ROOT_POSITIONS);
        assert_exact(&gpu, &expected, variant.name());
    }
}

/// `BYTEROOT_LOG_N` cycles (>= 23) in the 30-column byte geometry, of which
/// `BYTEROOT_COVERED_BLOCKS` 2^23-cycle blocks per column are committed
/// (default all), `BYTEROOT_REPS` timed runs per variant after one warm-up.
/// `BYTEROOT_CAPTURE=<path>.gputrace` instead captures one run of the first
/// selected variant for `gpudebug` replay profiling.
#[test]
#[ignore = "GPU microbench; run through gpu-window.sh"]
fn byteroot_timing() {
    let log_n: usize = std::env::var("BYTEROOT_LOG_N").map_or(26, |value| value.parse().unwrap());
    let reps: usize = std::env::var("BYTEROOT_REPS").map_or(3, |value| value.parse().unwrap());
    let selected = std::env::var("BYTEROOT_VARIANTS").ok();
    let covered_blocks: usize = std::env::var("BYTEROOT_COVERED_BLOCKS")
        .map_or(1 << (log_n - 23), |value| value.parse().unwrap());
    assert!(covered_blocks <= 1 << (log_n - 23));
    let num_tasks = SOURCE_COLUMNS * covered_blocks;
    let runtime = MetalRuntime::new().unwrap();
    let root = ByteRoot::new(&runtime);
    let variants = VARIANTS
        .into_iter()
        .filter(|variant| {
            selected
                .as_deref()
                .is_none_or(|names| names.split(',').any(|name| name == variant.name()))
        })
        .collect::<Vec<_>>();

    let setup = Instant::now();
    let mut state = 0x7469_6d69_6e67u64;
    let (matrix, _) = matrix_buffer(&runtime, ROOT_POSITIONS, |_| random_field(&mut state));
    let source = shared_bytes(&runtime, num_tasks * ROOT_POSITIONS * RING_D, |bytes| {
        fill_random_bytes(bytes, 0x5eed);
    });
    let (matrix_ntt, matrix_gpu) = root.prepare_matrix(&matrix, ROOT_POSITIONS);
    drop(matrix);
    println!(
        "setup\tlog_n={log_n}\tcovered_blocks={covered_blocks}\ttasks={num_tasks}\tsource_bytes={}\tmatrix_ntt_bytes={}\tmatrix_ntt_gpu_s={:.4}\thost_s={:.2}",
        source.length(),
        matrix_ntt.length(),
        matrix_gpu.map_or(f64::NAN, |gpu| gpu.as_secs_f64()),
        setup.elapsed().as_secs_f64()
    );

    if let Ok(path) = std::env::var("BYTEROOT_CAPTURE") {
        let descriptor = CaptureDescriptor::new();
        descriptor.set_capture_device(&runtime.device);
        descriptor.set_destination(MTLCaptureDestination::GpuTraceDocument);
        descriptor.set_output_url(&path);
        let manager = CaptureManager::shared();
        manager.start_capture(&descriptor).unwrap();
        root.run(variants[0], &matrix_ntt, &source, num_tasks, ROOT_POSITIONS);
        manager.stop_capture();
        println!("capture\t{}\t{path}", variants[0].name());
        return;
    }

    println!(
        "run\tvariant\tlog_n\trep\tmatvec_span_s\tmatvec_active_s\tfinish_s\ttotal_span_s\twall_s"
    );
    let mut reference: Option<Vec<Fp128Limbs>> = None;
    for variant in variants {
        let (warm, _) = root.run(variant, &matrix_ntt, &source, num_tasks, ROOT_POSITIONS);
        match &reference {
            Some(reference) => assert!(warm == *reference, "{} differs", variant.name()),
            None => reference = Some(warm),
        }
        let mut spans = Vec::with_capacity(reps);
        for rep in 0..reps {
            let (_, timings) = root.run(variant, &matrix_ntt, &source, num_tasks, ROOT_POSITIONS);
            println!(
                "run\t{}\t{log_n}\t{rep}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}",
                variant.name(),
                timings.matvec_span.as_secs_f64(),
                timings.matvec_active.as_secs_f64(),
                timings.finish.as_secs_f64(),
                timings.total_span.as_secs_f64(),
                timings.wall.as_secs_f64()
            );
            spans.push(timings.total_span.as_secs_f64());
        }
        spans.sort_by(f64::total_cmp);
        println!(
            "median\t{}\tlog_n={log_n}\ttotal_span_s={:.4}",
            variant.name(),
            spans[spans.len() / 2]
        );
    }
}
