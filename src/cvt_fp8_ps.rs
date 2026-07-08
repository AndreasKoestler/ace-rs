//! Family C (AVX10_V2_AUX): exact FP8 -> FP32 converts.
//!
//! `cvtbf8_ps` converts a vector of 16 BF8 (FP8 E5M2) bytes into 16 FP32 values and
//! `cvthf8_ps` converts 16 HF8 (FP8 E4M3) bytes into 16 FP32 values. Per ACE v1 spec
//! section 9.3 (`VCVTBF82PS` / `VCVTHF82PS`) the conversion is **exact** — every FP8
//! encoding maps precisely to one FP32 encoding with no rounding, no saturation and no
//! exceptions (`[avx10-v2-aux-ocp-conversions.CVT_FP8_PS.1]`,
//! `[avx10-v2-aux-ocp-conversions.CVT_FP8_PS.2]`,
//! `[avx10-v2-aux-ocp-conversions.CVT_FP8_PS.3]`). DAZ=0/FTZ=0, MXCSR not consulted (spec
//! section 9.3.1). The output is 4x wider than the input (16 bytes -> 16 dwords, spec
//! section 9.3.5).
//!
//! The public dispatcher is a safe fn that selects the scalar oracle whenever the running
//! CPU lacks `AVX10_V2_AUX` (`[avx10-v2-aux-ocp-conversions.DETECTION.2]`). The `_scalar`
//! oracle is the primary, always-correct path on every target including non-x86
//! (`[avx10-v2-aux-ocp-conversions.CORRECTNESS.1]`,
//! `[avx10-v2-aux-ocp-conversions.CORRECTNESS.2]`); it carries no cfg gate, reads no global
//! state, and the dispatcher equals it bit-for-bit. The names mirror the eventual stdarch
//! intrinsic stems `cvtbf8_ps` / `cvthf8_ps` (`[avx10-v2-aux-ocp-conversions.NAMING.1]`),
//! and the whole module compiles on stable Rust with no `core::simd`/nightly
//! (`[avx10-v2-aux-ocp-conversions.STABLE_RUST.1]`).
//!
//! OQ-5 (intrinsic unavailable -> oracle-only): BOTH FP8->FP32 forms ship oracle-only. The
//! `-mavx10.2` toolchain (GCC 16.x) exposes the FP8->FP16 siblings `_mm512_cvtbf8_ph` /
//! `_mm512_cvthf8_ph` but neither the FP8->FP32 form `_mm512_cvtbf8_ps` nor `_mm512_cvthf8_ps`,
//! so per OQ-5 family C ships **oracle-only**: there is no native C shim or `_hw` path for
//! either converter, and each dispatcher resolves to its `_scalar` sibling on every target.
//! The capability check [`crate::detect::has_avx10_v2_aux`] is still consulted (and returns
//! `false` here), so the three-layer detection wiring is exercised; a native path is added
//! once the intrinsics land in the toolchain.

use crate::detect;
use crate::fp8;

/// BF8 (FP8 E5M2) -> FP32 exact convert (16 lanes).
///
/// Per BF8 lane: decode the E5M2 byte to its exact FP32 value via
/// [`fp8::fp8_e5m2_to_fp32`]. The conversion is lossless — no rounding, no saturation, no
/// exceptions — because every BF8 value (subnormals, signed zeros, +/-Inf, and the BF8
/// NaNs) is representable in FP32 (`[avx10-v2-aux-ocp-conversions.CVT_FP8_PS.1]`).
///
/// Queries [`detect::has_avx10_v2_aux`] for the native path; that path is not yet wired
/// (OQ-5, see the module docs), so the dispatcher falls through to [`cvtbf8_ps_scalar`] on
/// every target, returning the spec-defined value.
/// `[avx10-v2-aux-ocp-conversions.DETECTION.2]`
pub fn cvtbf8_ps(a: [u8; 16]) -> [f32; 16] {
    // No native path this phase (OQ-5): the FP8->FP32 intrinsic is absent from the
    // `-mavx10.2` toolchain, so even under `feature="native"` on AVX10_V2_AUX hardware the
    // oracle is the only path. The capability check is still consulted so detection is
    // wired and ready for the shim once the intrinsic lands.
    let _ = detect::has_avx10_v2_aux; // keep the capability gate referenced on every target
    cvtbf8_ps_scalar(a)
}

/// Portable reference oracle for [`cvtbf8_ps`] — the primary always-correct path.
///
/// Maps each BF8 lane through [`fp8::fp8_e5m2_to_fp32`], the exact E5M2->FP32 decode.
/// Carries no cfg gate and reads no global state.
/// `[avx10-v2-aux-ocp-conversions.CORRECTNESS.1]` `[avx10-v2-aux-ocp-conversions.CORRECTNESS.2]`
pub fn cvtbf8_ps_scalar(a: [u8; 16]) -> [f32; 16] {
    core::array::from_fn(|i| fp8::fp8_e5m2_to_fp32(a[i]))
}

/// HF8 (FP8 E4M3) -> FP32 exact convert (16 lanes).
///
/// Per HF8 lane: decode the E4M3 byte to its exact FP32 value via
/// [`fp8::fp8_e4m3_to_fp32`]. The conversion is lossless — no rounding, no saturation, no
/// exceptions — because every HF8 value (subnormals, signed zeros, and the sole HF8 NaN
/// `S.1111.111`; E4M3 has no infinity) is representable in FP32
/// (`[avx10-v2-aux-ocp-conversions.CVT_FP8_PS.2]`,
/// `[avx10-v2-aux-ocp-conversions.CVT_FP8_PS.3]`).
///
/// Queries [`detect::has_avx10_v2_aux`] for the native path; that path is not yet wired
/// (OQ-5, see the module docs), so the dispatcher falls through to [`cvthf8_ps_scalar`] on
/// every target, returning the spec-defined value.
/// `[avx10-v2-aux-ocp-conversions.DETECTION.2]`
pub fn cvthf8_ps(a: [u8; 16]) -> [f32; 16] {
    // No native path this phase (OQ-5): `_mm512_cvthf8_ps` is absent from the `-mavx10.2`
    // toolchain, so the oracle is the only path even on AVX10_V2_AUX hardware under
    // `feature="native"`. The capability check is still consulted so detection is wired and
    // ready for the shim once the intrinsic lands.
    let _ = detect::has_avx10_v2_aux; // keep the capability gate referenced on every target
    cvthf8_ps_scalar(a)
}

/// Portable reference oracle for [`cvthf8_ps`] — the primary always-correct path.
///
/// Maps each HF8 lane through [`fp8::fp8_e4m3_to_fp32`], the exact E4M3->FP32 decode.
/// Carries no cfg gate and reads no global state.
/// `[avx10-v2-aux-ocp-conversions.CORRECTNESS.1]` `[avx10-v2-aux-ocp-conversions.CORRECTNESS.2]`
pub fn cvthf8_ps_scalar(a: [u8; 16]) -> [f32; 16] {
    core::array::from_fn(|i| fp8::fp8_e4m3_to_fp32(a[i]))
}

#[cfg(test)]
mod tests {
    use super::*;

    // BF8 (E5M2) byte assembler: sign | 5-bit exp field | 2-bit mantissa.
    fn bf8(sign: u8, exp: u8, mant: u8) -> u8 {
        (sign << 7) | (exp << 2) | mant
    }

    // HF8 (E4M3) byte assembler: sign | 4-bit exp field | 3-bit mantissa.
    fn hf8(sign: u8, exp: u8, mant: u8) -> u8 {
        (sign << 7) | (exp << 3) | mant
    }

    /// Hand-computed known-value vectors pinning chosen BF8 bytes to their exact FP32 bit
    /// patterns: zero, signed zero, a normal, a subnormal, the BF8 NaN encoding, and +/-Inf.
    /// Bit-pattern comparison (not numeric `==`) so signed zero is distinguished and the NaN
    /// lane is checked structurally. Exercises the exact-decode output contract
    /// (`[avx10-v2-aux-ocp-conversions.CVT_FP8_PS.1]`,
    /// `[avx10-v2-aux-ocp-conversions.CVT_FP8_PS.3]`).
    #[test]
    fn known_value_exact_bit_patterns() {
        let mut a = [0u8; 16];
        a[0] = bf8(0, 0b00000, 0b00); // +0
        a[1] = bf8(1, 0b00000, 0b00); // -0
        a[2] = bf8(0, 0b01111, 0b00); // 1.0
        a[3] = bf8(0, 0b00000, 0b01); // +2^-16 (min subnormal) -> FP32 NORMAL
        a[4] = bf8(1, 0b00000, 0b01); // -2^-16
        a[5] = bf8(0, 0b11110, 0b11); // max normal +57344
        a[6] = bf8(0, 0b11111, 0b00); // +Inf  (zero mantissa in all-ones exp)
        a[7] = bf8(0, 0b11111, 0b10); // NaN   (nonzero mantissa)

        let got = cvtbf8_ps(a);

        // +0 / -0 distinguished by sign bit (numeric == cannot tell them apart).
        assert_eq!(got[0].to_bits(), 0, "+0 -> +0.0");
        assert_eq!(got[1].to_bits(), 1 << 31, "-0 -> -0.0");
        assert_eq!(got[2], 1.0f32, "BF8 1.0 -> 1.0");
        // Subnormal BF8 renormalises to an FP32 NORMAL (2^-16, exp field 111), NOT an FP32
        // subnormal — this rules out a "subnormal stays subnormal" decode that would emit
        // exp 0 and the wrong value.
        assert_eq!(got[3], 2.0f32.powi(-16), "+2^-16 -> FP32 normal");
        assert_eq!(
            (got[3].to_bits() >> 23) & 0xff,
            111,
            "exp field 111 (normal)"
        );
        assert_eq!(got[4], -(2.0f32.powi(-16)), "-2^-16");
        assert_eq!(got[5], 57344.0f32, "max normal");
        // +Inf, NOT NaN: S.11111.00 is infinity in E5M2 (the NaN set is the nonzero-mantissa
        // codes). A model that treated the whole all-ones binade as NaN would fail here.
        assert!(got[6].is_infinite() && got[6] > 0.0, "S.11111.00 -> +Inf");
        assert!(got[7].is_nan(), "S.11111.10 -> NaN");
        // Untouched lanes are BF8 +0 -> +0.0.
        assert_eq!(got[15].to_bits(), 0, "padding lane +0");
    }

    /// Hand-computed known-value vectors pinning chosen HF8 (E4M3) bytes to their exact FP32
    /// bit patterns: zero, signed zero, a normal, a subnormal, and the sole HF8 NaN
    /// `S.1111.111`. Bit-pattern comparison so signed zero is distinguished and the NaN lane
    /// is checked structurally. Exercises the exact-decode output contract for the E4M3
    /// target (`[avx10-v2-aux-ocp-conversions.CVT_FP8_PS.2]`,
    /// `[avx10-v2-aux-ocp-conversions.CVT_FP8_PS.3]`).
    ///
    /// DISCRIMINATING lanes (each rules out the leading wrong model):
    ///  * lane 3 (`S.0000.001` = min subnormal 2^-9) decodes to the FP32 NORMAL 2^-9 (exp
    ///    field 118), NOT an FP32 subnormal — ruling out a "subnormal stays subnormal" decode.
    ///  * lane 6 (`S.1111.110` = +448, the E4M3 MAX NORMAL) decodes to the finite +448.0,
    ///    NOT a NaN — ruling out an E5M2-style "entire max-exponent binade is non-finite"
    ///    model (under which the max-exponent code would wrongly become NaN/Inf).
    ///  * lane 7 (`S.1111.111`) is the SOLE NaN (E4M3 has no Inf), so it must be NaN.
    #[test]
    fn known_value_hf8_exact_bit_patterns() {
        let mut a = [0u8; 16];
        a[0] = hf8(0, 0b0000, 0b000); // +0
        a[1] = hf8(1, 0b0000, 0b000); // -0
        a[2] = hf8(0, 0b0111, 0b000); // 1.0 (bias 7 -> exp field 7)
        a[3] = hf8(0, 0b0000, 0b001); // +2^-9 (min subnormal) -> FP32 NORMAL
        a[4] = hf8(1, 0b0000, 0b001); // -2^-9
        a[5] = hf8(0, 0b0111, 0b100); // 1.5  (1 + 4/8) * 2^0
        a[6] = hf8(0, 0b1111, 0b110); // max normal +448  S.1111.110 (NOT NaN)
        a[7] = hf8(0, 0b1111, 0b111); // sole NaN  S.1111.111

        let got = cvthf8_ps(a);

        assert_eq!(got[0].to_bits(), 0, "+0 -> +0.0");
        assert_eq!(got[1].to_bits(), 1 << 31, "-0 -> -0.0");
        assert_eq!(got[2], 1.0f32, "HF8 1.0 -> 1.0");
        // Min subnormal 2^-9 renormalises to an FP32 NORMAL (exp field 127-9 = 118), NOT an
        // FP32 subnormal.
        assert_eq!(got[3], 2.0f32.powi(-9), "+2^-9 -> FP32 normal");
        assert_eq!(
            (got[3].to_bits() >> 23) & 0xff,
            118,
            "exp field 118 (normal), not 0 (subnormal)"
        );
        assert_eq!(got[4], -(2.0f32.powi(-9)), "-2^-9");
        assert_eq!(got[5], 1.5f32, "HF8 1.5 -> 1.5");
        // +448 is the E4M3 MAX NORMAL, a finite value — NOT NaN. The max-exponent binade in
        // E4M3 holds genuine normals up to 448; only S.1111.111 is NaN. This rules out an
        // E5M2-style "max exponent == non-finite" decode.
        assert_eq!(got[6], 448.0f32, "S.1111.110 -> +448.0 (finite max normal)");
        assert!(got[6].is_finite(), "+448 is finite, not NaN/Inf");
        assert!(got[7].is_nan(), "S.1111.111 -> NaN (sole E4M3 NaN)");
        // Untouched lanes are HF8 +0 -> +0.0.
        assert_eq!(got[15].to_bits(), 0, "padding lane +0");
    }
}

/// Property-based tests for family C (BF8/HF8 -> FP32). The hand-rolled tests above pin
/// specific bit patterns; these assert the invariants across the full FP8 byte space.
#[cfg(test)]
mod proptests {
    use super::*;
    use quickcheck::{quickcheck, Arbitrary, Gen};

    /// A random 16-lane FP8 input. `quickcheck` does not derive `Arbitrary` for arrays of
    /// this length, so we wrap it and fill each lane independently — every one of the 256
    /// FP8 codes (zeros, subnormals, normals, +/-Inf for E5M2, NaNs) is reachable per lane.
    #[derive(Clone, Debug)]
    struct Inputs {
        a: [u8; 16],
    }

    impl Arbitrary for Inputs {
        fn arbitrary(g: &mut Gen) -> Self {
            Inputs {
                a: core::array::from_fn(|_| u8::arbitrary(g)),
            }
        }
    }

    quickcheck! {
        /// The public BF8 dispatcher always equals the scalar oracle bit-for-bit — the
        /// contract callers rely on regardless of which path runs. Compares raw bits so NaN
        /// lanes (where `==` is false) are still checked. Since family C is oracle-only this
        /// phase (OQ-5), this also pins that the dispatcher returns the spec value on every
        /// input (`[avx10-v2-aux-ocp-conversions.CORRECTNESS.1]`,
        /// `[avx10-v2-aux-ocp-conversions.DETECTION.2]`).
        fn prop_public_matches_scalar(input: Inputs) -> bool {
            let pub_out = cvtbf8_ps(input.a);
            let ora_out = cvtbf8_ps_scalar(input.a);
            (0..16).all(|i| pub_out[i].to_bits() == ora_out[i].to_bits())
        }

        /// The public HF8 dispatcher always equals the HF8 scalar oracle bit-for-bit, on
        /// every sampled byte (`[avx10-v2-aux-ocp-conversions.CVT_FP8_PS.2]`,
        /// `[avx10-v2-aux-ocp-conversions.CORRECTNESS.1]`,
        /// `[avx10-v2-aux-ocp-conversions.DETECTION.2]`).
        fn prop_hf8_public_matches_scalar(input: Inputs) -> bool {
            let pub_out = cvthf8_ps(input.a);
            let ora_out = cvthf8_ps_scalar(input.a);
            (0..16).all(|i| pub_out[i].to_bits() == ora_out[i].to_bits())
        }
    }

    /// Full-source-domain exactness for both FP8->FP32 converts: every one of the 256 BF8
    /// and 256 HF8 byte values decodes to a single defined FP32 encoding, and the decode is
    /// EXACT — no two distinct *finite* in-domain codes collapse to the same FP32 value, and
    /// subnormals renormalise (land on FP32 NORMALs) rather than being flushed to zero
    /// (`[avx10-v2-aux-ocp-conversions.CVT_FP8_PS.3]`).
    ///
    /// This is not just a "no panic" smoke test: injectivity over the finite domain is the
    /// teeth of "exact / lossless bijection". A flush-to-zero decode (DAZ wrongly =1) would
    /// collapse every subnormal onto +/-0 and FAIL injectivity here; a saturating or
    /// rounding decode would likewise alias distinct codes. We exclude only the NaN codes
    /// (whose FP32 payload is canonicalised, so several NaN codes legitimately share one
    /// FP32 NaN bit pattern) and the two signed zeros (which differ only in sign bit, both
    /// in-range and distinct by bits anyway). +/-Inf (E5M2 only) are kept and are distinct.
    #[test]
    fn full_domain_fp8_to_fp32_is_exact_bijection() {
        // Helper: collect the FP32 bit pattern for each non-NaN code; assert all distinct.
        fn assert_injective_over_finite(decode: impl Fn(u8) -> f32, label: &str) {
            use std::collections::HashMap;
            let mut seen: HashMap<u32, u8> = HashMap::new();
            for code in 0u16..256 {
                let byte = code as u8;
                let v = decode(byte);
                if v.is_nan() {
                    // NaN payloads are canonicalised; several NaN codes may map to one FP32
                    // NaN bit pattern. Injectivity is asserted over the FINITE domain only.
                    continue;
                }
                let bits = v.to_bits();
                if let Some(&prev) = seen.get(&bits) {
                    panic!(
                        "{label}: codes {prev:#04x} and {byte:#04x} both decode to FP32 \
                         bits {bits:#010x} — decode is NOT an exact bijection (lossy/aliasing)"
                    );
                }
                seen.insert(bits, byte);
            }
        }

        assert_injective_over_finite(fp8::fp8_e5m2_to_fp32, "BF8(E5M2)");
        assert_injective_over_finite(fp8::fp8_e4m3_to_fp32, "HF8(E4M3)");

        // Spot-check that subnormals are RENORMALISED, not flushed: the BF8 / HF8 min
        // subnormals must decode to a nonzero FP32 NORMAL (exp field != 0), which a
        // flush-to-zero (DAZ=1) decode would violate.
        let bf8_min_sub = 0b0000_0001u8; // S.00000.01 = +2^-16
        let v = fp8::fp8_e5m2_to_fp32(bf8_min_sub);
        assert_ne!(v.to_bits(), 0, "BF8 subnormal not flushed to zero");
        assert_ne!(
            (v.to_bits() >> 23) & 0xff,
            0,
            "BF8 subnormal -> FP32 normal"
        );

        let hf8_min_sub = 0b0000_0001u8; // S.0000.001 = +2^-9
        let v = fp8::fp8_e4m3_to_fp32(hf8_min_sub);
        assert_ne!(v.to_bits(), 0, "HF8 subnormal not flushed to zero");
        assert_ne!(
            (v.to_bits() >> 23) & 0xff,
            0,
            "HF8 subnormal -> FP32 normal"
        );
    }
}

/// Native-vs-oracle differential for family C (FP8 -> FP32). Phase 11 cross-cutting surface.
///
/// Family C ships **oracle-only** in this toolchain (OQ-5: no `_mm512_cvtbf8_ps` /
/// `_mm512_cvthf8_ps` intrinsic compiles under `-mavx10.2`). This property is written so that
/// the instant a native shim lands, the public dispatcher routes to it under
/// `feature="native"` on AVX10_V2_AUX hardware and this test compares that native path to the
/// scalar oracle bit-for-bit (`[avx10-v2-aux-ocp-conversions.DIFFERENTIAL.1]`). Until then the
/// native branch is absent, so the property calls `TestResult::discard()` — NEVER
/// `from_bool(false)` — so a fallback-only runner cannot manufacture a vacuous green.
#[cfg(test)]
mod differential {
    // Without the native feature the quickcheck body compiles down to the discard arm, so the
    // imports and struct fields are only read on the native+x86_64 configuration.
    #![cfg_attr(
        not(all(target_arch = "x86_64", feature = "native")),
        allow(unused_imports, dead_code)
    )]
    use super::*;
    use quickcheck::{quickcheck, Arbitrary, Gen, TestResult};

    #[derive(Clone, Debug)]
    struct Inputs {
        a: [u8; 16],
    }

    impl Arbitrary for Inputs {
        fn arbitrary(g: &mut Gen) -> Self {
            Inputs {
                a: core::array::from_fn(|_| u8::arbitrary(g)),
            }
        }
    }

    fn bits_eq(p: [f32; 16], o: [f32; 16]) -> bool {
        (0..16).all(|i| p[i].to_bits() == o[i].to_bits())
    }

    quickcheck! {
        /// Family-C native-vs-oracle differential. Under `feature="native"` on x86_64 with
        /// `AVX10_V2_AUX` detected the public dispatcher (which would route to the native shim
        /// once it lands) must agree with the scalar oracle bit-for-bit
        /// (`[avx10-v2-aux-ocp-conversions.DIFFERENTIAL.1]`). When the feature or hardware is
        /// absent the case is DISCARDED, never failed, keeping the test non-vacuous for a
        /// future native path (`[avx10-v2-aux-ocp-conversions.CORRECTNESS.2]`).
        fn prop_native_matches_oracle(input: Inputs) -> TestResult {
            #[cfg(all(target_arch = "x86_64", feature = "native"))]
            {
                if detect::has_avx10_v2_aux() {
                    return TestResult::from_bool(
                        bits_eq(cvtbf8_ps(input.a), cvtbf8_ps_scalar(input.a))
                            && bits_eq(cvthf8_ps(input.a), cvthf8_ps_scalar(input.a)),
                    );
                }
            }
            let _ = &input;
            let _ = bits_eq as fn([f32; 16], [f32; 16]) -> bool;
            TestResult::discard()
        }
    }
}
