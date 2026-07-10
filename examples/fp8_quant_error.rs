//! FP8 quantization-error explorer: the E5M2-vs-E4M3 tradeoff for LLM weights.
//!
//! A deterministic pseudo-Gaussian weight tensor (~N(0, 0.02), Irwin-Hall style —
//! no external crates) is round-tripped f32 -> FP8 -> f32 through both OCP formats:
//! BF8/E5M2 (`cvtps*_bf8` + `cvtbf8_ps`) and HF8/E4M3 (`cvtps*_hf8` + `cvthf8_ps`).
//! Two acts tell the story:
//!
//! * **precision** — per-element relative error, raw and with the per-tensor absmax
//!   scale real FP8 weight quantization uses. Raw, E5M2 wins: sigma-0.02 weights sit
//!   below E4M3's min normal 2^-6 where its grid degrades to subnormals and flushes.
//!   Scaled into the top binades, E4M3's extra mantissa bit wins (asserted).
//! * **range** — large magnitudes through the non-saturating and saturating (`S`)
//!   encoders: E5M2 holds its max finite 57344 and overflows to a real ±Inf; E4M3
//!   tops out at 448 and overflows to its sole NaN `S.1111.111` (no Inf exists);
//!   the `S` variants clamp to ±max finite instead.
//!
//! Every dispatcher call is checked against its portable `_scalar` oracle twin.
//! Group 3 currently ships oracle-only (OQ-5: no `-mavx10.2` toolchain exposes the
//! FP32<->FP8 intrinsics yet), so this example is pure numerics on every host:
//!
//! ```text
//! cargo run --example fp8_quant_error
//! ```
//!
//! Once native paths land, exercise them on a machine without `AVX10_V2_AUX` by
//! running the built example under Intel SDE with a future-CPU model:
//!
//! ```text
//! cargo build --example fp8_quant_error
//! ~/sde/sde64 -future -- target/debug/examples/fp8_quant_error
//! ```

use ace::{
    cvtbf8_ps, cvtbf8_ps_scalar, cvthf8_ps, cvthf8_ps_scalar, cvtps_bf8, cvtps_bf8_scalar,
    cvtps_hf8, cvtps_hf8_scalar, cvtpss_bf8, cvtpss_bf8_scalar, cvtpss_hf8, cvtpss_hf8_scalar,
};

/// Number of weights (64 full 16-lane converts).
const N: usize = 1024;

/// OCP max finite values: E5M2 `S.11110.11` and E4M3 `S.1111.110`.
const MAX_E5M2: f32 = 57344.0;
const MAX_E4M3: f32 = 448.0;

type Enc = fn([f32; 16]) -> [u8; 16];
type Dec = fn([u8; 16]) -> [f32; 16];

/// Deterministic f32 in [0, 1) from a tiny LCG (no external crates: `ace` is a
/// zero-dependency crate and its examples stay that way). Numerical Recipes constants.
fn lcg_f32(state: &mut u32) -> f32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    ((*state >> 8) as f32) / ((1u32 << 24) as f32)
}

/// Round-trip `weights * scale` through an FP8 format and back, checking each
/// dispatcher against its `_scalar` oracle; returns (max, mean) relative error.
fn quant_error(weights: &[f32; N], scale: f32, enc: (Enc, Enc), dec: (Dec, Dec)) -> (f32, f32) {
    let (mut max, mut sum) = (0.0f32, 0.0f32);
    for chunk in weights.chunks_exact(16) {
        let lanes: [f32; 16] = chunk.try_into().expect("chunks_exact yields 16");
        let scaled: [f32; 16] = core::array::from_fn(|i| lanes[i] * scale);
        let q = enc.0(scaled);
        assert_eq!(q, enc.1(scaled), "encoder diverged from its scalar oracle");
        let back = dec.0(q);
        // Compare bit patterns: NaN lanes must match too, and NaN != NaN as f32.
        assert_eq!(
            back.map(f32::to_bits),
            dec.1(q).map(f32::to_bits),
            "decoder diverged from its scalar oracle"
        );
        for (&x, &y) in lanes.iter().zip(&back) {
            let rel = (y / scale - x).abs() / x.abs();
            sum += rel;
            max = max.max(rel);
        }
    }
    (max, sum / N as f32)
}

fn main() {
    // ---- Act 1: deterministic pseudo-Gaussian weights ------------------------
    //
    // Irwin-Hall: the sum of 12 uniforms minus 6 has mean 0 and variance 1; scale
    // by 0.02 for a typical NN weight distribution ~N(0, 0.02).
    let mut state = 0xACE1_2026u32;
    let weights: [f32; N] = core::array::from_fn(|_| {
        let sum: f32 = (0..12).map(|_| lcg_f32(&mut state)).sum();
        0.02 * (sum - 6.0)
    });
    assert!(
        weights.iter().all(|&x| x != 0.0),
        "relative error needs x != 0"
    );
    let absmax = weights.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    println!("{N} pseudo-Gaussian weights ~N(0, 0.02), absmax {absmax:.4}");

    // ---- Act 2: round-trip precision, raw and per-tensor scaled --------------
    //
    // The saturating encoders quantize (real FP8 weight quant saturates); the
    // per-tensor scale maps absmax onto each format's max finite, the calibration
    // step production FP8 uses. Raw at sigma 0.02, most weights fall below E4M3's
    // min normal 2^-6 = 0.015625 (flush to zero below 2^-10), so its mean error
    // loses to E5M2 despite the extra mantissa bit; the scale reverses that.
    let bf8 = |scale| {
        quant_error(
            &weights,
            scale,
            (cvtpss_bf8, cvtpss_bf8_scalar),
            (cvtbf8_ps, cvtbf8_ps_scalar),
        )
    };
    let hf8 = |scale| {
        quant_error(
            &weights,
            scale,
            (cvtpss_hf8, cvtpss_hf8_scalar),
            (cvthf8_ps, cvthf8_ps_scalar),
        )
    };
    println!("\nRelative error (max / mean) over the round trip:");
    println!("  {:<22} {:>10} {:>10}", "", "max", "mean");
    let mut scaled = (0.0, 0.0);
    for (label, s5, s4) in [
        ("raw", 1.0, 1.0),
        ("per-tensor scaled", MAX_E5M2 / absmax, MAX_E4M3 / absmax),
    ] {
        let (b_max, b_mean) = bf8(s5);
        let (h_max, h_mean) = hf8(s4);
        println!("  E5M2 {label:<17} {b_max:>10.5} {b_mean:>10.5}");
        println!("  E4M3 {label:<17} {h_max:>10.5} {h_mean:>10.5}");
        if s5 == 1.0 {
            // Range act, part 0: raw sigma-0.02 weights sit in E4M3's subnormal
            // band, so the wider-range format wins unscaled.
            assert!(b_mean < h_mean, "raw: E5M2's range should beat E4M3");
        } else {
            scaled = (b_mean, h_mean);
        }
    }
    // The headline tradeoff: once scaled in-range, E4M3's third mantissa bit wins.
    assert!(
        scaled.1 < scaled.0,
        "scaled: E4M3's mantissa should beat E5M2"
    );

    // ---- Act 3: range — overflow vs saturation at the format edges -----------
    let big = [MAX_E4M3, 480.0, MAX_E5M2, 60000.0, 1.0e6];
    let mut lanes = [0.0f32; 16];
    lanes[..big.len()].copy_from_slice(&big);
    let mut decoded = [[0.0f32; 16]; 4];
    // Encoder table: E5M2 pair first, E4M3 pair second; the matching decoder is
    // picked by index (both E5M2 variants decode via cvtbf8_ps, both E4M3 via
    // cvthf8_ps).
    let encoders: [(&str, Enc, Enc); 4] = [
        ("E5M2 ", cvtps_bf8, cvtps_bf8_scalar),
        ("E5M2 S", cvtpss_bf8, cvtpss_bf8_scalar),
        ("E4M3 ", cvtps_hf8, cvtps_hf8_scalar),
        ("E4M3 S", cvtpss_hf8, cvtpss_hf8_scalar),
    ];
    println!("\nLarge magnitudes through non-saturating vs saturating encoders:");
    print!("  {:<8}", "input");
    for (i, (name, enc, enc_scalar)) in encoders.iter().enumerate() {
        let (dec, dec_scalar): (Dec, Dec) = if i < 2 {
            (cvtbf8_ps, cvtbf8_ps_scalar)
        } else {
            (cvthf8_ps, cvthf8_ps_scalar)
        };
        let q = enc(lanes);
        assert_eq!(
            q,
            enc_scalar(lanes),
            "encoder diverged from its scalar oracle"
        );
        decoded[i] = dec(q);
        // Compare bit patterns: E4M3 overflow lanes are NaN, and NaN != NaN as f32.
        assert_eq!(
            decoded[i].map(f32::to_bits),
            dec_scalar(q).map(f32::to_bits),
            "decoder diverged from its scalar oracle"
        );
        print!(" {name:>9}");
    }
    println!();
    for (j, &x) in big.iter().enumerate() {
        print!("  {x:<8}");
        for d in &decoded {
            print!(" {:>9}", d[j]);
        }
        println!();
    }
    let (bf, bfs, hf, hfs) = (decoded[0], decoded[1], decoded[2], decoded[3]);
    // E5M2: 448 and its max finite 57344 are exact; 480 RTNE-ties up to 512 (even
    // mantissa); 60000 rounds DOWN to 57344 (below the 61440 midpoint) even without
    // saturation; 1e6 overflows -> real +Inf non-S, clamp to +57344 with S.
    assert_eq!(
        (bf[0], bf[1], bf[2], bf[3]),
        (448.0, 512.0, MAX_E5M2, MAX_E5M2)
    );
    assert_eq!(bf[4], f32::INFINITY, "E5M2 non-S overflow encodes +Inf");
    assert_eq!(
        (bfs[2], bfs[4]),
        (MAX_E5M2, MAX_E5M2),
        "E5M2 S clamps to max finite"
    );
    // E4M3: 448 is the max finite; everything above overflows -> the sole NaN
    // S.1111.111 non-S (E4M3 has no Inf), clamp to 448 with S.
    assert_eq!(hf[0], MAX_E4M3);
    assert!(
        hf[1..5].iter().all(|v| v.is_nan()),
        "E4M3 non-S overflow encodes NaN"
    );
    assert!(
        hfs[1..5].iter().all(|&v| v == MAX_E4M3),
        "E4M3 S clamps to max finite"
    );

    // ---- Report ---------------------------------------------------------------
    println!("\nE5M2: wider range (max 57344, real Inf) — survives unscaled tensors");
    println!("E4M3: more mantissa (max 448, NaN-only overflow) — wins once scaled");
    println!("group-3 dispatch: oracle-only this toolchain (OQ-5) on every host");
    println!("PASS");
}
