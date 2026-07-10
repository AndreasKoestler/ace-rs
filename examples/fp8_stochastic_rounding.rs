//! Stochastic rounding to FP8 with `VCVTBIASPH2BF8`: why random bias bits fix
//! systematic quantization bias.
//!
//! FP8 E5M2 (BF8) has a 2-bit mantissa; converting an FP16 value that sits between two
//! BF8 grid points must discard 8 bits. Deterministic rounding (RTNE via `cvtph_bf8`)
//! sends every copy of the same value to the same grid point, so the error is
//! *systematic*: accumulate many FP8 gradients and the bias compounds instead of
//! cancelling. Stochastic rounding (SR) rounds up with probability equal to the
//! discarded fraction, making each conversion *unbiased in expectation* — the standard
//! trick for FP8 gradient accumulation and optimizer states in low-precision training.
//!
//! The hardware hook is the bias operand of `VCVTBIASPH2BF8` (`cvtbiasph_bf8`): per the
//! ACE spec section-16.2 SR pseudocode, the 8-bit bias byte (the LOW byte of each `u16`
//! bias lane) is added window-aligned into the 8 bits directly below the BF8 lsb, then
//! the sum is TRUNCATED. So bias 0 truncates (round toward zero — NOT RTNE!), a fixed
//! midpoint 0x80 behaves round-to-nearest-ish, and fresh random bytes give true SR:
//! P(round up) = discarded_fraction. This example measures all three against exact
//! values sitting 1/4 lsb above a grid point (RTNE always rounds them down), then shows
//! the SR trial average converging back to the exact value.
//!
//! Every dispatcher call is checked against its portable `_scalar` oracle twin. Run:
//!
//! ```text
//! cargo run --example fp8_stochastic_rounding
//! ```
//!
//! With the native `AVX10_V1_AUX` shims compiled in, exercise the real EVEX encodings
//! under Intel SDE with a future-CPU model (the oracle asserts become differentials):
//!
//! ```text
//! cargo build --examples --features native
//! ~/sde/sde64 -future -- target/debug/examples/fp8_stochastic_rounding
//! ```

use ace::{
    cvtbf8_ps, cvtbf8_ps_scalar, cvtbiasph_bf8, cvtbiasph_bf8_scalar, cvtph_bf8, cvtph_bf8_scalar,
};

/// SR trials; the per-lane trial average has a standard error of about
/// `sqrt(0.25 * 0.75 / 1000)` ~ 0.014 lsb.
const TRIALS: usize = 1000;

/// Exact f32 -> FP16 bits, restricted to FP16 normals that need no rounding — the
/// asserts guarantee the conversion is bit-exact for every value this example builds.
fn f32_to_fp16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127;
    let mant = b & 0x007f_ffff;
    assert!((-14..=15).contains(&exp), "FP16 normal range only");
    assert_eq!(mant & 0x1fff, 0, "must be exactly representable in FP16");
    sign | (((exp + 15) as u16) << 10) | (mant >> 13) as u16
}

/// Exact FP16 bits -> f32 (finite values only; every FP16 finite is exact in f32).
fn fp16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((bits >> 10) & 0x1f) as i32;
    let mant = (bits & 0x3ff) as f32;
    assert_ne!(exp, 0x1f, "finite values only");
    if exp == 0 {
        sign * mant * 2f32.powi(-24)
    } else {
        sign * (1.0 + mant / 1024.0) * 2f32.powi(exp - 15)
    }
}

/// Deterministic pseudo-random byte from a tiny LCG (no external crates: `ace` is a
/// zero-dependency crate and its examples stay that way). Numerical Recipes constants.
fn lcg_byte(state: &mut u32) -> u8 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    (*state >> 16) as u8
}

/// Decode 32 BF8 bytes to their exact f32 values via `cvtbf8_ps` (16 lanes per call),
/// asserting the dispatcher against its scalar oracle on every call.
fn decode_bf8(q: [u8; 32]) -> [f32; 32] {
    let mut out = [0.0f32; 32];
    for (chunk, dst) in q.chunks_exact(16).zip(out.chunks_exact_mut(16)) {
        let lanes: [u8; 16] = chunk.try_into().expect("chunks_exact yields 16");
        let dec = cvtbf8_ps(lanes);
        assert_eq!(
            dec.map(f32::to_bits),
            cvtbf8_ps_scalar(lanes).map(f32::to_bits),
            "cvtbf8_ps diverged from its scalar oracle"
        );
        dst.copy_from_slice(&dec);
    }
    out
}

/// Mean signed error over the 32 lanes, in units of each lane's BF8 lsb.
fn mean_err_lsb(dec: &[f64; 32], exact: &[f32; 32], lsb: &[f32; 32]) -> f64 {
    let sum: f64 = (0..32)
        .map(|i| (dec[i] - exact[i] as f64) / lsb[i] as f64)
        .sum();
    sum / 32.0
}

/// Informational AVX10.2 probe mirroring `detect::avx10_base` loosely (leaf 0x24 present
/// and converged version >= 2). Dispatch stays internal to the crate; this only reports.
#[cfg(target_arch = "x86_64")]
fn avx10_2_probe() -> &'static str {
    // `__cpuid` / `__cpuid_count` are safe fns (CPUID always exists on x86_64).
    use core::arch::x86_64::{__cpuid, __cpuid_count};
    if __cpuid(0).eax < 0x24 {
        return "no (CPUID leaf 0x24 absent)";
    }
    if (__cpuid_count(7, 1).edx >> 19) & 1 == 0 {
        return "no (AVX10 enumeration bit clear)";
    }
    if __cpuid_count(0x24, 0).ebx & 0xff >= 2 {
        "yes"
    } else {
        "no (AVX10 version < 2)"
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn avx10_2_probe() -> &'static str {
    "n/a (not x86_64)"
}

fn main() {
    // ---- Act 1: values engineered to sit BETWEEN BF8 grid points -------------
    //
    // Lane i lives in binade 2^(e-15) (FP16 exponent field e = 12..=19) at BF8 mantissa
    // k = 0..=3, plus exactly 1/4 of a BF8 lsb: the 8 discarded FP16 bits are 0x40, so
    // RTNE must round DOWN on every lane and SR must round up with probability 1/4.
    let exact: [f32; 32] = core::array::from_fn(|i| {
        let (e, k) = (12 + (i % 8) as i32, (i / 8) as f32);
        (1.0 + k / 4.0 + 1.0 / 16.0) * 2f32.powi(e - 15)
    });
    let lsb: [f32; 32] = core::array::from_fn(|i| 2f32.powi(12 + (i % 8) as i32 - 17));
    let a: [u16; 32] = exact.map(f32_to_fp16);
    for (&bits, &x) in a.iter().zip(&exact) {
        assert_eq!(fp16_to_f32(bits), x, "f32 -> FP16 must be exact");
        assert_eq!(bits & 0xff, 0x40, "discarded fraction must be 1/4 lsb");
    }
    println!("32 FP16 values, each exactly 1/4 BF8 lsb above a grid point");

    // ---- Act 2: deterministic RTNE — systematic bias --------------------------
    let q_rtne = cvtph_bf8(a);
    assert_eq!(
        q_rtne,
        cvtph_bf8_scalar(a),
        "cvtph_bf8 diverged from oracle"
    );
    let dec_rtne = decode_bf8(q_rtne);
    let rtne_bias = mean_err_lsb(&dec_rtne.map(f64::from), &exact, &lsb);
    println!("RTNE  (cvtph_bf8):              mean signed error {rtne_bias:+.4} lsb");
    assert!(
        (rtne_bias + 0.25).abs() < 1e-6,
        "every lane rounds down: bias must be exactly -1/4 lsb"
    );

    // ---- Act 3: what the bias operand really does (fraction 3/4 probe) --------
    //
    // Zero bias TRUNCATES — it is NOT RTNE. At discarded fraction 3/4 (low byte 0xC0)
    // RTNE rounds up, zero bias still rounds down, and the fixed midpoint byte 0x80
    // carries the sum up, agreeing with RTNE away from exact ties.
    let a75: [u16; 32] = a.map(|bits| (bits & !0xff) | 0xc0);
    let (zero, mid) = ([0u16; 32], [0x80u16; 32]);
    let q75_rtne = cvtph_bf8(a75);
    assert_eq!(q75_rtne, cvtph_bf8_scalar(a75), "cvtph_bf8 diverged");
    let q75_zero = cvtbiasph_bf8(a75, zero);
    assert_eq!(
        q75_zero,
        cvtbiasph_bf8_scalar(a75, zero),
        "cvtbiasph_bf8 (zero bias) diverged from oracle"
    );
    let q75_mid = cvtbiasph_bf8(a75, mid);
    assert_eq!(
        q75_mid,
        cvtbiasph_bf8_scalar(a75, mid),
        "cvtbiasph_bf8 (midpoint bias) diverged from oracle"
    );
    assert_eq!(q75_mid, q75_rtne, "midpoint bias 0x80 == RTNE off ties");
    let (d_up, d_dn) = (decode_bf8(q75_rtne), decode_bf8(q75_zero));
    for i in 0..32 {
        assert_eq!(d_up[i] - d_dn[i], lsb[i], "truncation lands one lsb below");
    }
    println!("bias 0x00 truncates, bias 0x80 matches RTNE (fraction-3/4 probe)");

    // ---- Act 4: stochastic rounding — fresh random bias bits per trial --------
    let mut state = 0xACE1_5EEDu32;
    let mut acc = [0f64; 32];
    for _ in 0..TRIALS {
        // Spec section 8.4.5: lane i reads bias byte `src1.byte[2*i]`, the LOW byte of
        // the i-th u16 — so one random byte per lane goes in the low half.
        let bias: [u16; 32] = core::array::from_fn(|_| lcg_byte(&mut state) as u16);
        let q = cvtbiasph_bf8(a, bias);
        assert_eq!(
            q,
            cvtbiasph_bf8_scalar(a, bias),
            "cvtbiasph_bf8 (random bias) diverged from oracle"
        );
        for (s, &d) in acc.iter_mut().zip(&decode_bf8(q)) {
            *s += f64::from(d);
        }
    }
    let sr_avg = acc.map(|s| s / TRIALS as f64);
    let sr_mean = mean_err_lsb(&sr_avg, &exact, &lsb);
    println!("SR    ({TRIALS} random-bias trials): mean signed error {sr_mean:+.4} lsb");
    for i in 0..32 {
        let lane_err = (sr_avg[i] - f64::from(exact[i])) / f64::from(lsb[i]);
        assert!(
            lane_err.abs() < 0.1,
            "lane {i}: SR trial average must converge near the exact value"
        );
    }
    assert!(
        sr_mean.abs() < rtne_bias.abs() / 5.0,
        "SR mean error must beat the RTNE bias by a solid margin"
    );

    // ---- Act 5: EVEX VNNI contrast — same semantics, wider vectors ------------
    let av: [i8; 64] = core::array::from_fn(|i| (i as i8).wrapping_sub(32));
    let bv: [i8; 64] = core::array::from_fn(|i| (i as i8).wrapping_mul(37));
    let evex = ace::vnni::dpbssd([7i32; 16], av, bv);
    assert_eq!(
        evex,
        ace::vnni::dpbssd_scalar([7i32; 16], av, bv),
        "vnni::dpbssd diverged from its scalar oracle"
    );
    let (lo_a, lo_b): ([i8; 32], [i8; 32]) = (
        av[..32].try_into().expect("low half"),
        bv[..32].try_into().expect("low half"),
    );
    assert_eq!(
        evex[..8],
        ace::dpbssd([7i32; 8], lo_a, lo_b),
        "EVEX vnni::dpbssd low dwords == 256-bit VEX dpbssd"
    );
    println!("vnni::dpbssd (EVEX, 16 dwords) matches VEX dpbssd on the low 8");

    // ---- Report ----------------------------------------------------------------
    println!("\nAVX10.2 (native shim eligible): {}", avx10_2_probe());
    println!("RTNE bias {rtne_bias:+.4} lsb vs SR {sr_mean:+.4} lsb — SR is unbiased");
    println!("PASS");
}
