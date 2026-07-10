//! MX (microscaling) block-scaled matrix multiply on the ACE tile register file.
//!
//! This is the block-scaled cousin of `int8_gemv`: a full `C = A·B` matrix
//! multiply where the f32 operands are quantized to OCP FP8 (BF8 = E5M2) with a
//! shared power-of-two E8M0 scale per microscaling block, the products are
//! accumulated with the rank-4 MX outer product `TOP4MXBF8PS`, and the f32
//! accumulator tile is read back and checked against the exact f32 reference.
//!
//! It exercises the whole ACE tile + Block Scale Register (BSR) pipeline:
//!
//! * `_tile_loadconfig` acquires the RAII `TileScope` (palette 2: eight fixed
//!   16x16 tiles plus one 1024-bit BSR, initialized to the E8M0 unit scale 0x7F);
//! * `_bsrmovf` writes the per-row A scales and per-column B scales into the BSR;
//! * `_tile_top4mxbf8ps` accumulates the outer product, reading the block scales
//!   implicitly and dequantizing in the precise domain (spec section 14.1);
//! * `_tile_movrow` reads the f32 result rows back out of the accumulator tile;
//! * `_tile_cvtrowps2bf16h` / `_tile_cvtrowps2bf16l` (family C, spec section
//!   12.5) store each accumulator row back out as BF16.
//!
//! # The MX block layout this example uses
//!
//! One ACE tile op is naturally 16x16: element `(i, j)` contracts A row `i` with
//! B "row" `j` over a 4-byte K-group (`src1[4*i..][..4]` · `src2[4*j..][..4]`,
//! spec section 14.1.6). We pick K = 32 — one MX block — and cover it with
//! `CHUNKS = 8` rank-4 accumulations that all share the same block scales. The
//! A-scale is associated with the output ROW and the B-scale with the output
//! COLUMN (not with the contraction index), so a single E8M0 byte per row of A
//! and per column of B scales the entire 32-wide block. That is exactly the OCP
//! MX contract: one shared scale per 32-element block.
//!
//! `src2` therefore holds B transposed: `src2[4*j + b]` is the quantized
//! `B[4*c + b][j]` for chunk `c`, so output column `j` reads down column `j` of B.
//!
//! # Running it
//!
//! ```text
//! cargo run --example mx_matmul
//! ```
//!
//! The `TOP*` / BSR families are ACE-only, and Intel SDE `-future` has no ACE
//! emulation yet (it is palette-1/AMX-only), so those instructions have no native
//! encoding to reach at all. The family-C row converts are different: they are
//! intrinsic-reachable (AMX-AVX512) and DO execute natively under SDE `-future` —
//! that is how the crate's `--features native` test suite validates the
//! `TCVTROWPS2BF16{H,L}` shims against the scalar oracle. The public dispatchers
//! this example calls still model the tile register file in Rust and therefore
//! take the scalar oracle on every target (see `src/tcvtrow.rs`), so this example
//! produces identical, spec-conformant results natively and under SDE:
//!
//! ```text
//! cargo build --example mx_matmul
//! ~/sde/sde64 -future -- target/debug/examples/mx_matmul
//! ```

use ace::{
    _bsrmovf, _tile_cvtrowps2bf16h, _tile_cvtrowps2bf16h_scalar, _tile_cvtrowps2bf16l,
    _tile_cvtrowps2bf16l_scalar, _tile_loadconfig, _tile_movrow, _tile_top4mxbf8ps,
    _tile_top4mxbf8ps_scalar, _tile_zero, ace_scale_a, ace_scale_b, cvtps_bf8, TileConfig,
    BSR_HALF_BYTES, BSR_INIT_BYTE,
};

/// Output rows / columns: an ACE tile is a fixed 16x16 f32 grid (spec section 10.2.1).
const M: usize = 16;
const N: usize = 16;
/// Contraction depth: one MX block of 32 elements shares a single E8M0 scale.
const K: usize = 32;
/// `TOP4MXBF8PS` contracts a 4-byte K-group per call (spec section 14.1.6)...
const KGROUP: usize = 4;
/// ...so 8 rank-4 accumulations cover the 32-wide MX block, all sharing one scale.
const CHUNKS: usize = K / KGROUP;

/// E8M0 encodes `2^(byte - 127)`: 0x7F = `2^0` = 1.0, 0xFF = NaN (spec section 10.2.3).
const E8M0_BIAS: i32 = 127;

/// Deterministic pseudo-random f32 in [-1, 1) from a tiny LCG — `ace` is a
/// zero-dependency crate and its examples stay that way (Numerical Recipes LCG).
fn lcg_f32(state: &mut u32) -> f32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    ((*state >> 8) as f32) / ((1u32 << 23) as f32) - 1.0
}

/// The MX shared-scale exponent for a block: `floor(log2(max_abs))`. Dividing the
/// block by `2^exp` lands its largest magnitude in [1, 2), keeping every element in
/// the normal FP8 range (mantissa precision is fixed at 2 bits for E5M2 regardless).
/// An all-zero block gets exponent 0 (unit scale, all payloads quantize to zero).
fn block_exp(vals: impl Iterator<Item = f32>) -> i32 {
    let max_abs = vals.fold(0.0f32, |m, v| m.max(v.abs()));
    if max_abs == 0.0 {
        0
    } else {
        max_abs.log2().floor() as i32
    }
}

/// Encode a scale exponent as its E8M0 byte, clamped to the finite range [0, 254]
/// (255 is the E8M0 NaN the MX ops treat as "poison this element").
fn e8m0_byte(exp: i32) -> u8 {
    (exp + E8M0_BIAS).clamp(0, 254) as u8
}

/// The FP32 -> BF16 rounding the family-C converts mandate (spec sections 12.5.1 and
/// 16.1), restated independently so the tile op is checked against the rule, not against
/// itself: round-to-nearest-EVEN on the 16 discarded low bits (bias by 0x7FFF plus the
/// least-significant kept bit, then truncate — an exact tie rounds to even), with DAZ=1
/// forced, so an FP32 zero or denormal input becomes a signed BF16 zero. Finite inputs
/// only, which a finite f32 accumulator guarantees (NaN quieting is the oracle's job).
fn bf16_rne(f: f32) -> u16 {
    let bits = f.to_bits();
    if bits & 0x7f80_0000 == 0 {
        return ((bits >> 31) as u16) << 15;
    }
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

fn main() {
    // ---- Deterministic f32 operands and the exact reference product ----------

    let mut state = 0x0ACE_2026u32;
    let a: [[f32; K]; M] = core::array::from_fn(|_| core::array::from_fn(|_| lcg_f32(&mut state)));
    let b: [[f32; N]; K] = core::array::from_fn(|_| core::array::from_fn(|_| lcg_f32(&mut state)));

    // Ground-truth f32 matmul the FP8 path must approximate.
    let mut c_ref = [[0.0f32; N]; M];
    for (i, row) in c_ref.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = (0..K).map(|k| a[i][k] * b[k][j]).sum();
        }
    }

    // ---- MX quantization: shared E8M0 scale per block, BF8 payloads ----------

    // One shared exponent per row of A (the output-row scale) and per column of B
    // (the output-column scale).
    let a_exp: [i32; M] = core::array::from_fn(|i| block_exp(a[i].iter().copied()));
    let b_exp: [i32; N] = core::array::from_fn(|j| block_exp((0..K).map(|k| b[k][j])));

    // Quantize to BF8 (E5M2) after dividing each element by its block's 2^exp.
    // cvtps_bf8 rounds a full 16-lane FP32 vector at once, matching M = N = 16.
    let mut bf8_a = [[0u8; K]; M];
    for k in 0..K {
        let lane: [f32; 16] = core::array::from_fn(|i| a[i][k] / 2.0f32.powi(a_exp[i]));
        let bytes = cvtps_bf8(lane);
        for i in 0..M {
            bf8_a[i][k] = bytes[i];
        }
    }
    let mut bf8_b = [[0u8; N]; K];
    for k in 0..K {
        let lane: [f32; 16] = core::array::from_fn(|j| b[k][j] / 2.0f32.powi(b_exp[j]));
        // Row k of B is quantized as a whole 16-lane vector (N == 16).
        bf8_b[k] = cvtps_bf8(lane);
    }

    // ---- Configure the tile scope and load the block scales into the BSR -----

    let mut scope = _tile_loadconfig(&TileConfig::ace()).expect("palette-2 descriptor is valid");

    // BSR scale-group 0: A_scales[i] at byte 64 + i*4 + 0, B_scales[j] at byte
    // j*4 + 0 (spec section 14.1.6). Every other byte stays at the unit scale 0x7F.
    let mut a_scales = [BSR_INIT_BYTE; BSR_HALF_BYTES];
    let mut b_scales = [BSR_INIT_BYTE; BSR_HALF_BYTES];
    for i in 0..M {
        a_scales[i * 4] = e8m0_byte(a_exp[i]);
    }
    for j in 0..N {
        b_scales[j * 4] = e8m0_byte(b_exp[j]);
    }
    _bsrmovf(&mut scope, a_scales, b_scales);

    // Two zeroed accumulators: tile 0 driven by the public dispatcher, tile 1 by
    // the scalar oracle on identical state — they must agree bit for bit.
    let acc = scope.tile(0).expect("tile 0");
    let acc_oracle = scope.tile(1).expect("tile 1");
    _tile_zero(&mut scope, acc);
    _tile_zero(&mut scope, acc_oracle);

    // ---- Accumulate the outer product over the 32-wide MX block --------------

    // imm8 selects scale-group 0 for both A (imm8[5:4]) and B (imm8[1:0]).
    let imm8 = ace_scale_a(0) | ace_scale_b(0);
    for c in 0..CHUNKS {
        // src1 row i = A[i][4c..4c+4]; src2 "row" j = B[4c..4c+4][j] (B transposed).
        let src1: [u8; 64] = core::array::from_fn(|idx| bf8_a[idx / 4][c * KGROUP + idx % 4]);
        let src2: [u8; 64] = core::array::from_fn(|idx| bf8_b[c * KGROUP + idx % 4][idx / 4]);
        _tile_top4mxbf8ps(&mut scope, acc, src1, src2, imm8);
        _tile_top4mxbf8ps_scalar(&mut scope, acc_oracle, src1, src2, imm8);
    }

    // ---- Dispatcher vs oracle: identical state must produce identical tiles --

    for row in 0..M {
        assert_eq!(
            _tile_movrow(&scope, acc, row as u32),
            _tile_movrow(&scope, acc_oracle, row as u32),
            "TOP4MXBF8PS dispatcher diverged from its scalar oracle at row {row}"
        );
    }

    // ---- Read the f32 accumulator back and compare to the reference ----------

    // Each tile row is 64 bytes = 16 f32 dwords, little-endian (spec section 10.2.1).
    let mut c_got = [[0.0f32; N]; M];
    for (i, dst) in c_got.iter_mut().enumerate() {
        let rowbytes = _tile_movrow(&scope, acc, i as u32);
        for (j, cell) in dst.iter_mut().enumerate() {
            let off = j * 4;
            *cell = f32::from_bits(u32::from_le_bytes([
                rowbytes[off],
                rowbytes[off + 1],
                rowbytes[off + 2],
                rowbytes[off + 3],
            ]));
        }
    }

    // Relative error normalized by the reference's peak magnitude — the standard
    // matmul metric (a plain per-element ratio explodes on cancellation-near-zero
    // elements, which say nothing about the quantization quality).
    let ref_mag = c_ref.iter().flatten().fold(0.0f32, |m, &v| m.max(v.abs()));
    let mut max_rel_err = 0.0f32;
    for i in 0..M {
        for j in 0..N {
            max_rel_err = max_rel_err.max((c_got[i][j] - c_ref[i][j]).abs() / ref_mag);
        }
    }

    // ---- Family C: store the f32 accumulator rows back out as BF16 -----------

    // TCVTROWPS2BF16H and TCVTROWPS2BF16L both convert ALL 16 f32 elements of the
    // addressed row — H/L is a destination-lane choice, not a source split. H puts
    // each BF16 in the HIGH word of its 32-bit destination slot (odd u16 lane
    // 2j + 1) and zeroes the low word; L is the mirror (even lane 2j, high word
    // zeroed) — disjoint half-lanes (spec section 12.5.3, INV-7).
    let mut max_bf16_rel_err = 0.0f32;
    for (i, c_row) in c_got.iter().enumerate() {
        let row = i as u32;
        let h = _tile_cvtrowps2bf16h(&scope, acc, row);
        let l = _tile_cvtrowps2bf16l(&scope, acc, row);
        assert_eq!(
            h,
            _tile_cvtrowps2bf16h_scalar(&scope, acc, row),
            "TCVTROWPS2BF16H dispatcher diverged from its scalar oracle at row {i}"
        );
        assert_eq!(
            l,
            _tile_cvtrowps2bf16l_scalar(&scope, acc, row),
            "TCVTROWPS2BF16L dispatcher diverged from its scalar oracle at row {i}"
        );
        for (j, &f32_val) in c_row.iter().enumerate() {
            assert_eq!(h[2 * j], 0, "H must zero the low word of dword {j}");
            assert_eq!(l[2 * j + 1], 0, "L must zero the high word of dword {j}");
            let bf16 = l[2 * j];
            assert_eq!(
                h[2 * j + 1],
                bf16,
                "H and L must carry the same BF16 payload"
            );
            // The BF16 must be exactly the RNE rounding of the f32 the TILEMOVROW
            // path already read back from this row.
            assert_eq!(bf16, bf16_rne(f32_val), "BF16 != RNE(f32) at ({i}, {j})");
            // BF16 -> FP32 is exact — BF16 is the top 16 bits of FP32 — so the
            // decode is a pure shift and the difference is pure rounding error.
            let back = f32::from_bits(u32::from(bf16) << 16);
            if f32_val != 0.0 {
                max_bf16_rel_err = max_bf16_rel_err.max((back - f32_val).abs() / f32_val.abs());
            }
        }
    }
    // RNE on a 7-bit mantissa is within half an ulp: relative error <= 2^-8.
    assert!(
        max_bf16_rel_err <= 0.00390625,
        "BF16 rounding error {max_bf16_rel_err} exceeds the half-ulp bound 2^-8"
    );

    // ---- Report --------------------------------------------------------------

    // BF8 (E5M2) has 2 mantissa bits; with per-block scaling and a K = 32
    // accumulation the block-normalized error settles in the low single digits.
    const TOLERANCE: f32 = 0.10;
    assert!(
        max_rel_err < TOLERANCE,
        "max relative error {max_rel_err} exceeds tolerance {TOLERANCE}"
    );

    #[cfg(target_arch = "x86_64")]
    let native = "no (dispatchers model tile state in Rust — scalar oracle everywhere; \
                  the family-C TCVTROW shims run natively under SDE via `--features native` tests, \
                  TOP/BSR are ACE-only with no SDE encoding)";
    #[cfg(not(target_arch = "x86_64"))]
    let native = "n/a (not x86_64)";

    println!("MX block-scaled matmul  C[{M}x{N}] = A[{M}x{K}] · B[{K}x{N}]");
    println!("  format:           BF8 (E5M2), rank-4 TOP4MXBF8PS x {CHUNKS} chunks");
    println!("  MX block size:    {K} elements, one shared E8M0 scale per block");
    println!("  ACE native path:  {native}");
    println!("  ||C||_max:        {ref_mag:.5}");
    println!("  max rel error:    {max_rel_err:.5} (tolerance {TOLERANCE})");
    println!("  bf16 store act:   TCVTROWPS2BF16H/L, max rounding error {max_bf16_rel_err:.6} (RNE half-ulp bound 2^-8 = 0.003906)");
    println!("PASS");
}
