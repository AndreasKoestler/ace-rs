//! Quantized int8 GEMV (matrix-vector product) built on `ace::dpbssd`.
//!
//! This example shows the classic inference-time use of AVX-VNNI-INT8: a small
//! linear layer `y = W·x` where the f32 weights and activations are symmetrically
//! quantized to int8, the integer dot products are computed with `VPDPBSSD`
//! (one `dpbssd` call per output row), and the i32 accumulators are dequantized
//! back to f32 and checked against the exact f32 reference.
//!
//! It also demonstrates the one behavioural fork in the group-1 grid: the wrapping
//! accumulate of `dpbssd` versus the saturating accumulate of `dpbssds`, on an
//! adversarial input where the two visibly diverge.
//!
//! Run it on any x86-64 (or other) host — the crate falls back to the portable
//! scalar path when `avxvnniint8` is absent:
//!
//! ```text
//! cargo run --example int8_gemv
//! ```
//!
//! To exercise the native `VPDPBSSD`/`VPDPBSSDS` instructions on a machine that
//! lacks them, run the built example under Intel SDE with a future-CPU model:
//!
//! ```text
//! cargo build --example int8_gemv
//! ~/sde/sde64 -future -- target/debug/examples/int8_gemv
//! ```

use ace::{dpbssd, dpbssd_scalar, dpbssds};

/// Layer shape: OUT output neurons, each with IN weights. IN = 32 is chosen so a
/// full weight row fits exactly one `[i8; 32]` operand of `dpbssd`.
const OUT: usize = 8;
const IN: usize = 32;

/// Deterministic pseudo-random f32 in [-1, 1) from a tiny LCG (no external crates:
/// `ace` is a zero-dependency crate and its examples stay that way). Constants are
/// the classic Numerical Recipes LCG.
fn lcg_f32(state: &mut u32) -> f32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    // Take the high 24 bits for a well-distributed mantissa, map to [-1, 1).
    ((*state >> 8) as f32) / ((1u32 << 23) as f32) - 1.0
}

/// Symmetric per-tensor quantization: scale = max_abs / 127, q = round(v / scale).
/// Returns the i8 values and the scale needed to dequantize (`v ≈ q * scale`).
fn quantize(values: &[f32]) -> (Vec<i8>, f32) {
    let max_abs = values.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let scale = max_abs / 127.0;
    let q = values
        .iter()
        .map(|v| (v / scale).round().clamp(-127.0, 127.0) as i8)
        .collect();
    (q, scale)
}

fn main() {
    // ---- Act 1: quantized linear layer vs f32 reference -------------------

    // Deterministic weights W (OUT x IN) and input x (IN).
    let mut state = 0xACE1_2026u32;
    let w: Vec<f32> = (0..OUT * IN).map(|_| lcg_f32(&mut state)).collect();
    let x: Vec<f32> = (0..IN).map(|_| lcg_f32(&mut state)).collect();

    // Exact f32 reference GEMV, the ground truth the int8 path must approximate.
    let reference: Vec<f32> = (0..OUT)
        .map(|row| {
            w[row * IN..(row + 1) * IN]
                .iter()
                .zip(&x)
                .map(|(wi, xi)| wi * xi)
                .sum()
        })
        .collect();

    // Per-tensor symmetric quantization of both operands.
    let (wq, w_scale) = quantize(&w);
    let (xq, x_scale) = quantize(&x);
    let xq: [i8; IN] = xq.try_into().expect("x is exactly IN elements");

    // Integer GEMV: one dpbssd call per output row. Each call produces 8 lanes,
    // lane i holding the sum of the 4 adjacent byte products a[4i..4i+4]·b[4i..4i+4],
    // so a horizontal add of the 8 lanes yields the full 32-element dot product.
    // Dequantize with the product of the two per-tensor scales.
    let mut max_rel_err = 0.0f32;
    let ref_mag = reference.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    for (row, want) in reference.iter().enumerate() {
        let wrow: [i8; IN] = wq[row * IN..(row + 1) * IN]
            .try_into()
            .expect("each row is exactly IN elements");

        let lanes = dpbssd([0i32; 8], wrow, xq);
        // Dispatcher vs oracle: whatever path the host took, the portable scalar
        // reference must agree bit for bit.
        assert_eq!(
            lanes,
            dpbssd_scalar([0i32; 8], wrow, xq),
            "dpbssd dispatcher diverged from its scalar oracle"
        );

        let acc: i32 = lanes.iter().sum();
        let got = acc as f32 * w_scale * x_scale;
        let rel_err = (got - want).abs() / ref_mag;
        max_rel_err = max_rel_err.max(rel_err);
        println!("  y[{row}] = {got:+.5}  (f32 reference {want:+.5}, rel err {rel_err:.5})");
    }

    // int8 symmetric quantization of both operands: ~1-2% relative error is normal.
    const TOLERANCE: f32 = 0.02;
    assert!(
        max_rel_err < TOLERANCE,
        "max relative error {max_rel_err} exceeds tolerance {TOLERANCE}"
    );

    // ---- Act 2: wrap (dpbssd) vs saturate (dpbssds) ------------------------

    // Adversarial accumulate: src sits 100 below i32::MAX, and every lane's four
    // byte products contribute +400 (10*10 each). The wrapping form overflows past
    // MAX into negative territory; the saturating form clamps at i32::MAX.
    let src = [i32::MAX - 100; 8];
    let a = [10i8; 32];
    let b = [10i8; 32];
    let wrapped = dpbssd(src, a, b);
    let saturated = dpbssds(src, a, b);
    println!("\nOverflow demo (src = i32::MAX - 100, +400 per lane):");
    println!("  dpbssd  (wrap):     lane 0 = {}", wrapped[0]);
    println!("  dpbssds (saturate): lane 0 = {}", saturated[0]);
    assert!(
        wrapped
            .iter()
            .all(|&v| v == (i32::MAX - 100).wrapping_add(400)),
        "dpbssd must wrap on overflow"
    );
    assert!(
        saturated.iter().all(|&v| v == i32::MAX),
        "dpbssds must clamp to i32::MAX"
    );

    // ---- Report -------------------------------------------------------------

    #[cfg(target_arch = "x86_64")]
    let native = if std::is_x86_feature_detected!("avxvnniint8") {
        "yes"
    } else {
        "no (scalar fallback)"
    };
    #[cfg(not(target_arch = "x86_64"))]
    let native = "n/a (not x86_64)";

    println!("\navxvnniint8 native path: {native}");
    println!("max relative error:      {max_rel_err:.5} (tolerance {TOLERANCE})");
    println!("PASS");
}
