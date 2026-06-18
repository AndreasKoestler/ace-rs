//! `ace-rs` — x86 AI Compute Extensions (ACE) primitives for Rust.
//!
//! See `DESIGN_RATIONALE.md` for the full design. Each primitive follows the same shape:
//!
//! ```text
//! safe dispatch fn  →  native path (core::arch intrinsic)  →  scalar fallback (primary path)
//! ```
//!
//! with a differential test asserting the native path agrees with the scalar oracle.
//!
//! This is **iteration 0** (the tracer bullet, design §6 / D9): one primitive — [`dpbssd`] —
//! wired end to end on stable Rust: build → runtime detect → intrinsic → fallback → test.
//! It is the only ACE primitive already present in stable `core::arch`, so it needs no
//! emulator and runs natively on AVX-VNNI-INT8 hardware.

/// Signed int8 dot-product-accumulate. (ACE group 1: AVX-VNNI-INT8, `VPDPBSSD`.)
///
/// For each of the 8 output lanes `i`:
///
/// ```text
/// out[i] = src[i] + Σ_{k=0..4} a[4i+k] * b[4i+k]
/// ```
///
/// Dispatches to the native intrinsic when the running CPU supports `avxvnniint8`,
/// otherwise uses the portable scalar path. Both produce identical results.
pub fn dpbssd(src: [i32; 8], a: [i8; 32], b: [i8; 32]) -> [i32; 8] {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avxvnniint8") {
            // SAFETY: the `avxvnniint8` feature was confirmed present immediately above.
            return unsafe { dpbssd_hw(src, a, b) };
        }
    }
    dpbssd_scalar(src, a, b)
}

/// Portable reference path — and the oracle the native path is tested against.
pub fn dpbssd_scalar(src: [i32; 8], a: [i8; 32], b: [i8; 32]) -> [i32; 8] {
    let mut out = src;
    for i in 0..8 {
        let mut acc = 0i32;
        for k in 0..4 {
            acc = acc.wrapping_add(a[4 * i + k] as i32 * b[4 * i + k] as i32);
        }
        out[i] = out[i].wrapping_add(acc);
    }
    out
}

/// Native path: `VPDPBSSD` via `core::arch::x86_64::_mm256_dpbssd_epi32`.
///
/// # Safety
/// The CPU must support the `avxvnniint8` feature. Callers go through [`dpbssd`],
/// which checks this at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avxvnniint8")]
unsafe fn dpbssd_hw(src: [i32; 8], a: [i8; 32], b: [i8; 32]) -> [i32; 8] {
    use std::arch::x86_64::*;
    let vsrc = _mm256_loadu_si256(src.as_ptr().cast());
    let va = _mm256_loadu_si256(a.as_ptr().cast());
    let vb = _mm256_loadu_si256(b.as_ptr().cast());
    let vout = _mm256_dpbssd_epi32(vsrc, va, vb);
    let mut out = [0i32; 8];
    _mm256_storeu_si256(out.as_mut_ptr().cast(), vout);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Differential test: native path must match the scalar oracle bit-for-bit.
    /// Runs the comparison only where the native path is actually available.
    #[test]
    fn hw_matches_scalar() {
        let a: [i8; 32] = core::array::from_fn(|i| i as i8 - 16);
        let b: [i8; 32] = core::array::from_fn(|i| (i as i8).wrapping_mul(3).wrapping_sub(7));
        let src: [i32; 8] = core::array::from_fn(|i| i as i32 * 100);

        let want = dpbssd_scalar(src, a, b);

        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avxvnniint8") {
            // SAFETY: feature checked above.
            assert_eq!(
                unsafe { dpbssd_hw(src, a, b) },
                want,
                "native path disagrees with oracle"
            );
        }

        // Public API always works (falls back when the feature is absent).
        assert_eq!(dpbssd(src, a, b), want);
    }

    /// Coverage guard. When `ACE_REQUIRE_NATIVE=1` is set (CI's SDE job), the native
    /// path *must* be the one that runs — otherwise a green suite would only prove the
    /// scalar fallback. Off by default, so local/non-x86 runs are unaffected.
    #[test]
    fn native_runs_when_required() {
        if std::env::var_os("ACE_REQUIRE_NATIVE").is_none() {
            return;
        }
        #[cfg(target_arch = "x86_64")]
        assert!(
            std::is_x86_feature_detected!("avxvnniint8"),
            "ACE_REQUIRE_NATIVE=1 but avxvnniint8 is not detected — the native path was NOT exercised"
        );
        #[cfg(not(target_arch = "x86_64"))]
        panic!("ACE_REQUIRE_NATIVE=1 on a non-x86_64 target — the native path cannot run here");
    }

    /// Hand-computed value, independent of the implementation.
    #[test]
    fn known_value() {
        // lane 0: 1*1 + 2*2 + 3*3 + 4*4 = 30; all other lanes use zero inputs.
        let mut a = [0i8; 32];
        let mut b = [0i8; 32];
        for k in 0..4 {
            a[k] = (k as i8) + 1;
            b[k] = (k as i8) + 1;
        }
        assert_eq!(dpbssd([0; 8], a, b), [30, 0, 0, 0, 0, 0, 0, 0]);
    }
}
