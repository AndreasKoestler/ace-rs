//! Family D: HF8 (FP8 E4M3) -> FP16 convert.
//!
//! `cvthf8_ph` converts a vector of 32 HF8 (E4M3) bytes into 32 FP16 values. Per ACE v1
//! spec section 8.5 (`VCVTHF82PH`) the conversion is **exact** — every HF8 value is
//! representable in FP16, so it performs no rounding, no saturation, and raises no
//! exceptions (`[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_HF82PH.1]`).
//!
//! The public dispatcher is a safe fn that selects a native path when the running CPU
//! supports `AVX10_V1_AUX` (via a hand-written C shim behind the opt-in `native` feature —
//! no stable `core::arch` EVEX intrinsic exists yet — per
//! `[avx10-v1-aux-fp16-fp8-evex-vnni.DISPATCH.3]`) and otherwise falls back to its `_scalar`
//! oracle. The `_scalar` oracle is the primary,
//! always-correct path on every target including non-x86
//! (`[avx10-v1-aux-fp16-fp8-evex-vnni.ORACLE.1]`); it carries no cfg gate, reads no
//! global state, and the dispatcher equals it bit-for-bit
//! (`[avx10-v1-aux-fp16-fp8-evex-vnni.ORACLE.2]`). The name mirrors the eventual stdarch
//! intrinsic stem `cvthf8_ph` (`[avx10-v1-aux-fp16-fp8-evex-vnni.NAMING.1]`).

use crate::detect;
use crate::fp8;

/// HF8 (FP8 E4M3) -> FP16 exact convert.
///
/// Per HF8 lane: decode the E4M3 byte to its exact FP16 (E5M10) bit pattern. The
/// conversion is lossless — no rounding, no saturation, no exceptions — because every HF8
/// value (including subnormals, signed zeros, and the sole NaN encoding) is representable
/// in FP16 (`[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_HF82PH.1]`).
///
/// Dispatches to the native path under `AVX10_V1_AUX` (C shim, opt-in `native` feature) and otherwise to
/// [`cvthf8_ph_scalar`]; both return identical FP16 bit patterns.
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.DISPATCH.1]`
pub fn cvthf8_ph(a: [u8; 32]) -> [u16; 32] {
    #[cfg(all(target_arch = "x86_64", feature = "native"))]
    {
        if detect::has_avx10_v1_aux() {
            // SAFETY: `has_avx10_v1_aux()` confirmed full AVX10.2 (the feature set this shim's
            // translation unit is compiled for) plus OS XSAVE state immediately above.
            return unsafe { cvthf8_ph_hw(a) };
        }
    }
    let _ = detect::has_avx10_v1_aux; // keep `detect` referenced on every target
    cvthf8_ph_scalar(a)
}

/// Native path: EVEX `VCVTHF82PH` via the `ace_native_cvthf8_ph` C shim.
///
/// # Safety
/// The CPU must support `AVX10_V1_AUX`; callers go through [`cvthf8_ph`], which checks it.
#[cfg(all(target_arch = "x86_64", feature = "native"))]
unsafe fn cvthf8_ph_hw(a: [u8; 32]) -> [u16; 32] {
    let mut out = [0u16; 32];
    crate::native::ace_native_cvthf8_ph(a.as_ptr(), out.as_mut_ptr());
    out
}

/// Portable reference oracle for [`cvthf8_ph`] — the primary always-correct path.
///
/// Maps each HF8 lane through [`fp8::hf8_to_fp16`], the exact E4M3->FP16 decode. Carries
/// no cfg gate and reads no global state. `[avx10-v1-aux-fp16-fp8-evex-vnni.ORACLE.1]`
pub fn cvthf8_ph_scalar(a: [u8; 32]) -> [u16; 32] {
    core::array::from_fn(|i| fp8::hf8_to_fp16(a[i]))
}

#[cfg(test)]
mod tests {
    use super::*;

    // HF8 (E4M3) byte assembler: sign | 4-bit exp field | 3-bit mantissa.
    fn hf8(sign: u8, exp: u8, mant: u8) -> u8 {
        (sign << 7) | (exp << 3) | mant
    }

    // FP16 bit assembler: sign | 5-bit exp field | 10-bit mantissa.
    fn fp16(sign: u16, exp: u16, mant: u16) -> u16 {
        (sign << 15) | (exp << 10) | mant
    }

    /// Hand-computed known-value vectors covering HF8 zero, signed zero, a normal, a
    /// subnormal (+/-2^-9), and the sole HF8 NaN encoding S.1111.111, asserting the exact
    /// FP16 bit pattern. The conversion is exact and lossless: every HF8 value is
    /// representable in FP16 (`[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_HF82PH.1]`).
    #[test]
    fn known_value_exact_bit_patterns() {
        let mut a = [0u8; 32];
        // lane 0: +0 -> FP16 +0.
        a[0] = hf8(0, 0b0000, 0b000);
        // lane 1: -0 -> FP16 -0.
        a[1] = hf8(1, 0b0000, 0b000);
        // lane 2: HF8 1.0 (S.0111.000) -> FP16 1.0 (exp 15, mant 0).
        a[2] = hf8(0, 0b0111, 0b000);
        // lane 3: HF8 min subnormal +2^-9 (S.0000.001) -> FP16 normal exp field 6, mant 0.
        a[3] = hf8(0, 0b0000, 0b001);
        // lane 4: HF8 min subnormal -2^-9 -> FP16 normal, sign set.
        a[4] = hf8(1, 0b0000, 0b001);
        // lane 5: HF8 max normal +448 (S.1111.110) -> FP16 exp field 23, mant 0b11_0...0.
        a[5] = hf8(0, 0b1111, 0b110);
        // lane 6: HF8 NaN (S.1111.111) -> FP16 NaN.
        a[6] = hf8(0, 0b1111, 0b111);

        let got = cvthf8_ph(a);

        assert_eq!(got[0], fp16(0, 0, 0), "+0");
        assert_eq!(got[1], fp16(1, 0, 0), "-0");
        assert_eq!(got[2], fp16(0, 15, 0), "1.0");
        // 2^-9 renormalises to an FP16 NORMAL (exp field 6), not a subnormal — this rules
        // out a "subnormal stays subnormal" decode that would emit exp 0.
        assert_eq!(got[3], fp16(0, 6, 0), "+2^-9 -> FP16 normal");
        assert_eq!(got[4], fp16(1, 6, 0), "-2^-9 -> FP16 normal");
        assert_eq!(got[5], fp16(0, 23, 0b11_0000_0000), "+448");
        // NaN: all-ones FP16 exponent with a nonzero mantissa.
        assert_eq!((got[6] >> 10) & 0x1f, 0x1f, "NaN exp all ones");
        assert!(got[6] & 0x3ff != 0, "NaN mantissa nonzero");
        // Untouched lanes are HF8 +0 -> FP16 +0.
        assert_eq!(got[31], fp16(0, 0, 0), "padding lane");
    }
}

/// Property-based tests for family D (HF8 -> FP16). The hand-rolled tests above pin
/// specific bit patterns; these assert the invariants across the full HF8 byte space.
#[cfg(test)]
mod proptests {
    use super::*;
    use quickcheck::{quickcheck, Arbitrary, Gen, TestResult};

    /// A random 32-lane HF8 input. `quickcheck` does not derive `Arbitrary` for arrays of
    /// this length, so we wrap it and fill each lane independently — every one of the 256
    /// HF8 codes (zeros, subnormals, normals, NaN) is reachable per lane.
    #[derive(Clone, Debug)]
    struct Inputs {
        a: [u8; 32],
    }

    impl Arbitrary for Inputs {
        fn arbitrary(g: &mut Gen) -> Self {
            Inputs {
                a: core::array::from_fn(|_| u8::arbitrary(g)),
            }
        }
    }

    quickcheck! {
        /// The public dispatcher always equals the scalar oracle — the contract callers
        /// rely on regardless of which path runs
        /// (`[avx10-v1-aux-fp16-fp8-evex-vnni.ORACLE.2]`).
        fn prop_public_matches_scalar(input: Inputs) -> bool {
            cvthf8_ph(input.a) == cvthf8_ph_scalar(input.a)
        }

        /// Exact round-trip: for any HF8 byte, decoding to FP16 and re-encoding with the
        /// family-A HF8 encoder returns the original byte. This is the losslessness
        /// guarantee — FP16 represents every HF8 value, so HF8->FP16->HF8 is the identity
        /// (`[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_HF82PH.1]`,
        /// `[avx10-v1-aux-fp16-fp8-evex-vnni.PROPERTIES.2]`).
        fn prop_hf8_fp16_hf8_roundtrip(input: Inputs) -> bool {
            let fp16 = cvthf8_ph(input.a);
            (0..32).all(|i| crate::fp8::fp16_to_hf8(fp16[i], false) == input.a[i])
        }

        /// Family-D native-vs-oracle differential. Under `feature="native"` on x86_64 with
        /// `AVX10_V1_AUX` detected, the real EVEX `VCVTHF82PH` path must agree with the scalar
        /// oracle bit-for-bit (`[avx10-v1-aux-fp16-fp8-evex-vnni.DIFFERENTIAL.1]`). When the
        /// native feature or detection is absent the case is *discarded* (never
        /// `from_bool(false)`), so a fallback-only runner cannot produce a vacuous green.
        fn prop_native_matches_oracle(input: Inputs) -> TestResult {
            #[cfg(all(target_arch = "x86_64", feature = "native"))]
            {
                use crate::detect;
                if detect::has_avx10_v1_aux() {
                    // Bit-for-bit: the oracle's HF8(NaN) -> FP16(NaN) payload was aligned to
                    // hardware (0x7f80 / 0xff80, verified under SDE), so even NaN lanes match
                    // exactly.
                    return TestResult::from_bool(cvthf8_ph(input.a) == cvthf8_ph_scalar(input.a));
                }
            }
            let _ = &input;
            TestResult::discard()
        }
    }

    /// Hand-value family-D native-vs-oracle differential. Runs only under `feature="native"`
    /// with `AVX10_V1_AUX` detected, otherwise a silent no-op. Sweeps every one of the 256 HF8
    /// codes across the 32 lanes and compares bit-for-bit (the NaN payload is hardware-aligned).
    /// `[avx10-v1-aux-fp16-fp8-evex-vnni.DIFFERENTIAL.1-1]`
    #[cfg(all(target_arch = "x86_64", feature = "native"))]
    #[test]
    fn hand_value_native_matches_oracle() {
        use crate::detect;
        if !detect::has_avx10_v1_aux() {
            return;
        }
        for base in (0u16..256).step_by(32) {
            let a: [u8; 32] = core::array::from_fn(|i| (base as usize + i) as u8);
            assert_eq!(
                cvthf8_ph(a),
                cvthf8_ph_scalar(a),
                "hw cvthf8_ph != oracle (base {base})"
            );
        }
    }
}
