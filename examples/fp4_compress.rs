//! FP4 weight compression built on the `AVX10_V2_AUX` OCP format conversions.
//!
//! This example shows the inference-time storage trick FP4 exists for: a block of
//! f32 "weights" is compressed 8x by converting FP32 -> FP8 E4M3 (`cvtps_hf8`) and
//! then FP8 -> FP4 E2M1 nibble-packed (`cvtf8_bf4s_e4m3`), so 64 weights live in
//! 32 bytes. Two read-back paths are then demonstrated:
//!
//! * **compute path** — FP4 -> FP8 (`cvtbf4_hf8`, exact) -> f32 (`cvthf8_ps`),
//!   measuring the reconstruction error the coarse FP4 grid costs;
//! * **inspection path** — `unpackb` with an `imm8` built from `ACE_UNPACKB_SIZE` /
//!   `ACE_UNPACKB_START` (size 4, start 0, zero-extend), recovering each raw FP4
//!   nibble right-aligned in a byte: the read-back complement of the packed layout.
//!
//! Every dispatcher call is also checked against its portable `_scalar` oracle twin.
//! Group 3 currently ships oracle-only (OQ-5: no `-mavx10.2` toolchain exposes the
//! intrinsics yet), so the dispatchers resolve to the oracles on every host:
//!
//! ```text
//! cargo run --example fp4_compress
//! ```
//!
//! Once native paths land, exercise them on a machine without `AVX10_V2_AUX` by
//! running the built example under Intel SDE with a future-CPU model:
//!
//! ```text
//! cargo build --example fp4_compress
//! ~/sde/sde64 -future -- target/debug/examples/fp4_compress
//! ```

use ace::{
    cvtbf4_hf8, cvtbf4_hf8_scalar, cvtf8_bf4s_e4m3, cvtf8_bf4s_e4m3_scalar, cvthf8_ps,
    cvthf8_ps_scalar, cvtps_hf8, cvtps_hf8_scalar, unpackb, unpackb_scalar, ACE_UNPACKB_SIZE,
    ACE_UNPACKB_START,
};

/// Number of weights: exactly one FP4 convert (64 FP8 lanes -> 32 packed bytes).
const N: usize = 64;

/// Deterministic pseudo-random f32 in [-1, 1) from a tiny LCG (no external crates:
/// `ace` is a zero-dependency crate and its examples stay that way). Constants are
/// the classic Numerical Recipes LCG.
fn lcg_f32(state: &mut u32) -> f32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    ((*state >> 8) as f32) / ((1u32 << 23) as f32) - 1.0
}

fn main() {
    // ---- Act 1: deterministic weights sized for FP4's tiny range ------------
    //
    // FP4 E2M1 spans only {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6} (max normal ±6.0),
    // so keep magnitudes in [0.5, 4.0) — inside the grid, away from the flush-to-
    // zero region below 0.25 where relative error is unbounded.
    let mut state = 0xACE1_2026u32;
    let weights: [f32; N] = core::array::from_fn(|_| {
        let u = lcg_f32(&mut state); // [-1, 1)
        let sign = if u < 0.0 { -1.0 } else { 1.0 };
        sign * (0.5 + 3.5 * u.abs()) // ±[0.5, 4.0)
    });

    // ---- Act 2: compress f32 -> FP8 E4M3 -> FP4 E2M1 (nibble-packed) --------

    // FP32 -> FP8 E4M3, 16 lanes per call; dispatcher checked against its oracle.
    let mut fp8 = [0u8; N];
    for (i, chunk) in weights.chunks_exact(16).enumerate() {
        let lanes: [f32; 16] = chunk.try_into().expect("chunks_exact yields 16");
        let out = cvtps_hf8(lanes);
        assert_eq!(
            out,
            cvtps_hf8_scalar(lanes),
            "cvtps_hf8 dispatcher diverged from its scalar oracle"
        );
        fp8[i * 16..(i + 1) * 16].copy_from_slice(&out);
    }

    // FP8 E4M3 -> FP4 E2M1, saturating RTNE, two lanes per output byte.
    let packed: [u8; 32] = cvtf8_bf4s_e4m3(fp8);
    assert_eq!(
        packed,
        cvtf8_bf4s_e4m3_scalar(fp8),
        "cvtf8_bf4s_e4m3 dispatcher diverged from its scalar oracle"
    );

    let f32_bytes = N * core::mem::size_of::<f32>();
    let fp4_bytes = packed.len();
    println!("Footprint: {N} weights");
    println!("  f32: {f32_bytes} bytes");
    println!(
        "  fp4: {fp4_bytes} bytes  ({}x smaller)",
        f32_bytes / fp4_bytes
    );
    assert_eq!(f32_bytes / fp4_bytes, 8, "FP4 packing must be 8x smaller");

    // ---- Act 3: read-back path A (compute) — FP4 -> FP8 -> f32 --------------

    // FP4 -> FP8 E4M3 is EXACT (each nibble maps to one E4M3 byte, no rounding);
    // all the loss happened on the way down.
    let fp8_back: [u8; N] = cvtbf4_hf8(packed);
    assert_eq!(
        fp8_back,
        cvtbf4_hf8_scalar(packed),
        "cvtbf4_hf8 dispatcher diverged from its scalar oracle"
    );

    let mut recon = [0.0f32; N];
    for (i, chunk) in fp8_back.chunks_exact(16).enumerate() {
        let lanes: [u8; 16] = chunk.try_into().expect("chunks_exact yields 16");
        let out = cvthf8_ps(lanes);
        assert_eq!(
            out,
            cvthf8_ps_scalar(lanes),
            "cvthf8_ps dispatcher diverged from its scalar oracle"
        );
        recon[i * 16..(i + 1) * 16].copy_from_slice(&out);
    }

    let mut max_rel_err = 0.0f32;
    let mut sum_rel_err = 0.0f32;
    for (orig, back) in weights.iter().zip(&recon) {
        let rel_err = (back - orig).abs() / orig.abs(); // |orig| >= 0.5, never zero
        sum_rel_err += rel_err;
        max_rel_err = max_rel_err.max(rel_err);
    }
    let mean_rel_err = sum_rel_err / N as f32;
    println!("\nReconstruction vs original (through the FP4 bottleneck):");
    println!("  w[0] = {:+.4} -> {:+.1}", weights[0], recon[0]);
    println!("  w[1] = {:+.4} -> {:+.1}", weights[1], recon[1]);
    println!("  max rel err:  {max_rel_err:.4}");
    println!("  mean rel err: {mean_rel_err:.4}");

    // FP4 E2M1 is coarse: RTNE to the nearest grid point costs up to ~1/3 relative
    // error near 0.75 (midway between 0.5 and 1.0). Generous bound for the grid.
    const TOLERANCE: f32 = 0.34;
    assert!(
        max_rel_err < TOLERANCE,
        "max relative error {max_rel_err} exceeds FP4 tolerance {TOLERANCE}"
    );

    // ---- Act 4: read-back path B (inspection) — unpackb on the raw nibbles ---

    // imm8 = size 4 (imm8[4:2]) | start 0 (imm8[1:0]); no ACE_UNPACKB_SEXT, so each
    // 4-bit field zero-extends into its byte. unpackb reads 64 size-bit fields from
    // bit offset (start*64 + i)*size, i.e. exactly the nibble stream family D wrote.
    let imm8 = ACE_UNPACKB_SIZE(4) | ACE_UNPACKB_START(0);
    let mut buf = [0u8; 64];
    buf[..32].copy_from_slice(&packed);
    let nibbles = unpackb(buf, imm8);
    assert_eq!(
        nibbles,
        unpackb_scalar(buf, imm8),
        "unpackb dispatcher diverged from its scalar oracle"
    );
    for (i, &lane) in nibbles.iter().enumerate() {
        // Hand extraction: lane i is the low (i even) or high (i odd) nibble of
        // packed byte i/2 — nibble-packed LSB-first from bit 0.
        let by_hand = (packed[i / 2] >> (4 * (i % 2))) & 0x0f;
        assert_eq!(
            lane, by_hand,
            "lane {i}: unpackb must read back the packed nibble"
        );
    }
    println!("\nunpackb(size=4, start=0, zext): all 64 nibbles match hand extraction");
    println!(
        "  packed[0] = {:#04x} -> lanes [{:#03x}, {:#03x}]",
        packed[0], nibbles[0], nibbles[1]
    );

    // ---- Report --------------------------------------------------------------

    println!("\ngroup-3 dispatch: oracle-only this toolchain (OQ-5) on every host");
    println!("PASS");
}
