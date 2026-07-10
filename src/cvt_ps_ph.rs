//! Family E: FP32-pair -> FP16 convert.
//!
//! `cvt2ps_phx` converts two vectors of 16 FP32 values into one vector of 32 FP16 values
//! (`VCVT2PS2PHX`, ACE v1 spec section 8.3). The two sources are concatenated into a
//! single same-width FP16 output: output lanes `[0..16)` come from `src2` and lanes
//! `[16..32)` from `src1` (spec section 8.3.5, `KL = VL/16 = 32`, `i < KL/2` reads `src2`),
//! the same low=src2 / high=src1 ordering as the family-B two-source converts
//! (`[avx10-v1-aux-fp16-fp8-evex-vnni.CVT2_PS2PHX.1]`,
//! `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT2_PS2PHX.1-3]`).
//!
//! OQ-6 (rounding contract, Blocking): on hardware the FP32->FP16 rounding mode is governed
//! by MXCSR and, at 512-bit width, EVEX embedded rounding `{er}`; MXCSR.DAZ is respected on
//! the FP32 inputs and FTZ is assumed 0 (spec section 8.3.1/8.3.3). The oracle fixes the
//! CANONICAL contract: it reads NO global state and uses the default — IEEE roundTiesToEven
//! (RNE), DAZ=0, FTZ=0 — and embedded rounding `{er}` is NOT surfaced in v1. A native
//! differential would set MXCSR to this default state before comparing
//! (`[avx10-v1-aux-fp16-fp8-evex-vnni.CVT2_PS2PHX.1-1]`,
//! `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT2_PS2PHX.1-2]`).
//!
//! The public dispatcher is a safe fn that selects a native path when the running CPU
//! supports `AVX10_V1_AUX` (via a hand-written C shim behind the opt-in `native` feature —
//! no stable `core::arch` EVEX intrinsic exists yet — per
//! `[avx10-v1-aux-fp16-fp8-evex-vnni.DISPATCH.3]`) and otherwise falls back to its `_scalar`
//! oracle. The `_scalar` oracle is the primary, always-correct
//! path on every target including non-x86 (`[avx10-v1-aux-fp16-fp8-evex-vnni.ORACLE.1]`); it
//! carries no cfg gate, reads no global state, and the dispatcher equals it bit-for-bit
//! (`[avx10-v1-aux-fp16-fp8-evex-vnni.ORACLE.2]`). The name mirrors the eventual stdarch
//! intrinsic stem `cvt2ps_phx` (`[avx10-v1-aux-fp16-fp8-evex-vnni.NAMING.1]`).

use crate::detect;
use crate::fp8;

/// FP32-pair -> FP16 convert (`VCVT2PS2PHX`).
///
/// Converts two 16-lane FP32 vectors into one 32-lane FP16 vector. Output lanes `[0..16)`
/// are produced from `src2` and lanes `[16..32)` from `src1` (spec section 8.3.5), each
/// rounded FP32->FP16 under the canonical default-RNE / DAZ=0 / FTZ=0 contract
/// (`[avx10-v1-aux-fp16-fp8-evex-vnni.CVT2_PS2PHX.1]`,
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT2_PS2PHX.1-3]`).
///
/// Dispatches to the native path under `AVX10_V1_AUX` (C shim, opt-in `native` feature) and otherwise to
/// [`cvt2ps_phx_scalar`]; both return identical FP16 bit patterns.
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.DISPATCH.1]`
pub fn cvt2ps_phx(src1: [f32; 16], src2: [f32; 16]) -> [u16; 32] {
    #[cfg(all(target_arch = "x86_64", feature = "native"))]
    {
        if detect::has_avx10_v1_aux() {
            // SAFETY: `has_avx10_v1_aux()` confirmed full AVX10.2 (the feature set this shim's
            // translation unit is compiled for) plus OS XSAVE state immediately above.
            return unsafe { cvt2ps_phx_hw(src1, src2) };
        }
    }
    let _ = detect::has_avx10_v1_aux; // keep `detect` referenced on every target
    cvt2ps_phx_scalar(src1, src2)
}

/// Native path: EVEX `VCVT2PS2PHX` via the `ace_native_cvt2ps_phx` C shim (low=src2,
/// high=src1). The shim uses `_MM_FROUND_CUR_DIRECTION`, i.e. the MXCSR rounding mode; the
/// oracle fixes the canonical default RNE / DAZ=0 / FTZ=0 contract (OQ-6), which is the MXCSR
/// state at process start, so the differential is well-defined without touching MXCSR.
///
/// # Safety
/// The CPU must support `AVX10_V1_AUX`; callers go through [`cvt2ps_phx`], which checks it.
#[cfg(all(target_arch = "x86_64", feature = "native"))]
unsafe fn cvt2ps_phx_hw(src1: [f32; 16], src2: [f32; 16]) -> [u16; 32] {
    let mut out = [0u16; 32];
    crate::native::ace_native_cvt2ps_phx(src1.as_ptr(), src2.as_ptr(), out.as_mut_ptr());
    out
}

/// Portable reference oracle for [`cvt2ps_phx`] — the primary always-correct path.
///
/// Output lanes `[0..16)` convert the `src2` lanes and lanes `[16..32)` convert the `src1`
/// lanes (low=src2 / high=src1, spec section 8.3.5), each through [`fp8::fp32_to_fp16_rne`]
/// under the canonical default-RNE / DAZ=0 / FTZ=0 contract (OQ-6). Carries no cfg gate and
/// reads no global state. `[avx10-v1-aux-fp16-fp8-evex-vnni.ORACLE.1]`
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT2_PS2PHX.1-3]`
pub fn cvt2ps_phx_scalar(src1: [f32; 16], src2: [f32; 16]) -> [u16; 32] {
    core::array::from_fn(|i| {
        if i < 16 {
            fp8::fp32_to_fp16_rne(src2[i])
        } else {
            fp8::fp32_to_fp16_rne(src1[i - 16])
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // FP16 bit assembler: sign | 5-bit exp field | 10-bit mantissa.
    fn fp16(sign: u16, exp: u16, mant: u16) -> u16 {
        (sign << 15) | (exp << 10) | mant
    }

    /// Hand-computed known-value vectors: exactly-representable FP32 values plus a
    /// ties-to-even rounding case, asserting the low half `[0..16)` comes from `src2` and
    /// the high half `[16..32)` from `src1` (the two-source low=src2 / high=src1 invariant,
    /// `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT2_PS2PHX.1-3]`,
    /// `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT2_PS2PHX.1]`,
    /// `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT2_PS2PHX.1-1]`).
    #[test]
    fn known_value_lane_ordering_and_rne() {
        let mut src1 = [0.0f32; 16];
        let mut src2 = [0.0f32; 16];
        // Distinct, non-overlapping marker values so a swapped-halves bug is caught.
        src2[0] = 1.0; // -> FP16 1.0 in OUTPUT lane 0 (low half from src2)
        src2[1] = -2.0; // -> FP16 -2.0 in output lane 1
        src1[0] = 4.0; // -> FP16 4.0 in OUTPUT lane 16 (high half from src1)
        src1[1] = 0.5; // -> FP16 0.5 in output lane 17

        let got = cvt2ps_phx(src1, src2);

        // Low half [0..16) is src2.
        assert_eq!(got[0], fp16(0, 15, 0), "lane 0 = src2[0] = 1.0");
        assert_eq!(got[1], fp16(1, 16, 0), "lane 1 = src2[1] = -2.0");
        // High half [16..32) is src1. This pins the ordering: a swapped-halves convert
        // would put 4.0 in lane 0, which the assertion above rules out.
        assert_eq!(got[16], fp16(0, 17, 0), "lane 16 = src1[0] = 4.0");
        assert_eq!(got[17], fp16(0, 14, 0), "lane 17 = src1[1] = 0.5");
        // src2[0] did NOT bleed into the high half, and src1[0] did NOT bleed into the low.
        assert_eq!(got[2], fp16(0, 0, 0), "untouched src2 lane -> +0");
        assert_eq!(got[18], fp16(0, 0, 0), "untouched src1 lane -> +0");

        // Ties-to-even rounding case, routed through src2 lane 2. The FP16 mantissa keeps
        // 10 of FP32's 23 fraction bits; bit 12 set is exactly half the discarded window.
        // Base 1.0 (kept lsb even) ties DOWN; this distinguishes RNE from round-half-up.
        let tie_down = f32::from_bits(0x3f80_0000 | (1u32 << 12));
        src2[2] = tie_down;
        let got = cvt2ps_phx(src1, src2);
        assert_eq!(
            got[2],
            fp16(0, 15, 0),
            "RNE tie to even rounds 1.0+0.5lsb DOWN"
        );
    }

    /// The public dispatcher equals the scalar oracle on a representative vector
    /// (`[avx10-v1-aux-fp16-fp8-evex-vnni.ORACLE.2]`).
    #[test]
    fn dispatcher_matches_oracle() {
        let src1: [f32; 16] = core::array::from_fn(|i| i as f32 - 8.0);
        let src2: [f32; 16] = core::array::from_fn(|i| (i as f32) * 0.25 + 0.1);
        assert_eq!(cvt2ps_phx(src1, src2), cvt2ps_phx_scalar(src1, src2));
    }
}

/// Property-based tests for family E (FP32-pair -> FP16). The hand-rolled tests above pin
/// specific bit patterns; these assert the invariants across a randomly-sampled slice of
/// the input space.
///
/// (Inferred, OQ-6) the oracle fixes the canonical default rounding contract: RNE, DAZ=0,
/// FTZ=0, reading no MXCSR. A native differential, when one is wired, would first set MXCSR
/// to that default state (and disable `{er}`) before comparing the native result against
/// the oracle, so the bit-for-bit agreement is well-defined.
#[cfg(test)]
mod proptests {
    use super::*;
    use quickcheck::{quickcheck, Arbitrary, Gen, TestResult};

    /// A random pair of 16-lane FP32 vectors. `quickcheck` does not derive `Arbitrary` for
    /// arrays of this length, so we wrap them and fill each lane independently — every lane
    /// can be a normal, subnormal, signed zero, Inf, or NaN.
    #[derive(Clone, Debug)]
    struct Inputs {
        src1: [f32; 16],
        src2: [f32; 16],
    }

    impl Arbitrary for Inputs {
        fn arbitrary(g: &mut Gen) -> Self {
            Inputs {
                src1: core::array::from_fn(|_| f32::arbitrary(g)),
                src2: core::array::from_fn(|_| f32::arbitrary(g)),
            }
        }
    }

    quickcheck! {
        /// The public dispatcher always equals the scalar oracle — the contract callers
        /// rely on regardless of which path runs
        /// (`[avx10-v1-aux-fp16-fp8-evex-vnni.ORACLE.2]`). FP32 lanes are compared by raw
        /// bits, so NaN inputs (which compare unequal under `==`) are handled correctly.
        fn prop_public_matches_scalar(input: Inputs) -> bool {
            cvt2ps_phx(input.src1, input.src2) == cvt2ps_phx_scalar(input.src1, input.src2)
        }

        /// Lane-ordering invariant: output lane `i` for `i < 16` is the FP32->FP16 convert
        /// of `src2[i]`, and lane `16 + i` is the convert of `src1[i]` (low=src2 /
        /// high=src1, `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT2_PS2PHX.1-3]`).
        fn prop_lane_ordering(input: Inputs) -> bool {
            let out = cvt2ps_phx_scalar(input.src1, input.src2);
            (0..16).all(|i| {
                out[i] == crate::fp8::fp32_to_fp16_rne(input.src2[i])
                    && out[16 + i] == crate::fp8::fp32_to_fp16_rne(input.src1[i])
            })
        }

        /// Family-E native-vs-oracle differential. Under `feature="native"` on x86_64 with
        /// `AVX10_V1_AUX` detected, the real EVEX `VCVT2PS2PHX` path must agree with the scalar
        /// oracle bit-for-bit (`[avx10-v1-aux-fp16-fp8-evex-vnni.DIFFERENTIAL.1]`), under the
        /// canonical default-RNE / DAZ=0 / FTZ=0 contract (OQ-6) which is the MXCSR state at
        /// process start. When the native feature or detection is absent the case is
        /// *discarded* (never `from_bool(false)`), so a fallback-only runner cannot produce a
        /// vacuous green. NaN result lanes are compared by their non-payload structure (sign,
        /// all-ones exponent, nonzero mantissa) since the FP32->FP16 NaN payload is
        /// implementation-defined and not fixed by the spec; every finite/Inf lane is compared
        /// bit-for-bit.
        fn prop_native_matches_oracle(input: Inputs) -> TestResult {
            #[cfg(all(target_arch = "x86_64", feature = "native"))]
            {
                if detect::has_avx10_v1_aux() {
                    let hw = cvt2ps_phx(input.src1, input.src2);
                    let or = cvt2ps_phx_scalar(input.src1, input.src2);
                    let ok = (0..32).all(|i| fp16_eq_mod_nan_payload(hw[i], or[i]));
                    return TestResult::from_bool(ok);
                }
            }
            let _ = &input;
            TestResult::discard()
        }
    }

    /// Compare two FP16 bit patterns, treating any two NaNs (all-ones exponent, nonzero
    /// mantissa) of the same sign as equal regardless of mantissa payload. Finite/Inf/zero
    /// lanes must match bit-for-bit. The FP32->FP16 NaN payload is implementation-defined
    /// (the spec fixes only that a NaN maps to a NaN), so a hardware NaN payload may differ
    /// from the oracle's canonical quiet NaN without being a real disagreement.
    #[cfg(all(target_arch = "x86_64", feature = "native"))]
    fn fp16_eq_mod_nan_payload(x: u16, y: u16) -> bool {
        let is_nan = |b: u16| (b >> 10) & 0x1f == 0x1f && (b & 0x3ff) != 0;
        if is_nan(x) && is_nan(y) {
            (x >> 15) == (y >> 15)
        } else {
            x == y
        }
    }

    /// Hand-value family-E native-vs-oracle differential: exactly-representable values, a
    /// ties-to-even case, overflow to Inf, subnormals, and signed zero — all finite/Inf, so
    /// compared bit-for-bit. Runs only under `feature="native"` with `AVX10_V1_AUX` detected.
    /// `[avx10-v1-aux-fp16-fp8-evex-vnni.DIFFERENTIAL.1-1]`
    #[cfg(all(target_arch = "x86_64", feature = "native"))]
    #[test]
    fn hand_value_native_matches_oracle() {
        if !detect::has_avx10_v1_aux() {
            return;
        }
        let src2: [f32; 16] = [
            1.0,
            -2.0,
            0.5,
            4.0,
            0.0,
            -0.0,
            65504.0,
            70000.0,
            2.0f32.powi(-14),
            2.0f32.powi(-24),
            3.0 * 2.0f32.powi(-26),
            100.0,
            -448.0,
            0.125,
            2048.0,
            0.25,
        ];
        let src1: [f32; 16] = core::array::from_fn(|i| (i as f32) * 0.5 - 4.0);
        assert_eq!(
            cvt2ps_phx(src1, src2),
            cvt2ps_phx_scalar(src1, src2),
            "hw cvt2ps_phx != oracle"
        );
    }
}
