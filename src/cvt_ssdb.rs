//! Family H (AVX10_V2_AUX): `VPMOVSSDB` — INT32 -> INT8 with **symmetric** signed
//! saturation.
//!
//! `cvtssepi32_epi8` converts a vector of 16 signed `i32` lanes into 16 signed `i8` lanes,
//! clamping each lane to the SYMMETRIC range `[-127, +127]` (ACE v1 sect 9.8.1:
//! `MAX_POSITIVE = 0x7F = +127`, `MAX_NEGATIVE = 0x81 = -127`). This is the KEY distinction
//! from the ordinary `VPMOVSDB` / `_mm_cvtsepi32_epi8` family, which clamps to the
//! ASYMMETRIC two's-complement range `[-128, +127]` (sect 9.8.7 note). The `cvtss` prefix
//! ("convert symmetric-saturate") is exactly what names that distinction
//! (`[avx10-v2-aux-ocp-conversions.CVT_SSDB.1]`,
//! `[avx10-v2-aux-ocp-conversions.CVT_SSDB.1-note]`).
//!
//! Per sect 9.8.5, `KL = VL/32 = 16` lanes, and `dest.byte[i] =
//! saturate_int32_to_symmetric_int8(src.dword[i])`; the INT8 output occupies one QUARTER of
//! the INT32 input width (16 dwords -> 16 bytes, `[avx10-v2-aux-ocp-conversions.CVT_SSDB.3]`).
//! The conversion raises no exceptions (sect 9.8.3) and never faults — it is a total
//! function on every `i32`, saturating rather than trapping on overflow.
//!
//! The symmetric clamp makes the convert an ODD function about zero: for every `i32` `x`,
//! `f(-x) == -f(x)`, and in particular `f(i32::MIN) == -127` (NOT `-128`)
//! (`[avx10-v2-aux-ocp-conversions.CVT_SSDB.2]`). The asymmetric `VPMOVSDB` lacks this
//! property — it would map `i32::MIN` to `-128`.
//!
//! The public dispatcher is a safe fn; with no native group-3 intrinsic available in
//! current toolchains it always takes the scalar oracle (see the OQ-5 note below), with
//! `detect::has_avx10_v2_aux` marking where the native gate goes live
//! (`[avx10-v2-aux-ocp-conversions.DETECTION.2]`). The `_scalar`
//! oracle is the primary, always-correct path on every target including non-x86
//! (`[avx10-v2-aux-ocp-conversions.CORRECTNESS.1]`,
//! `[avx10-v2-aux-ocp-conversions.CORRECTNESS.2]`); it carries no cfg gate, reads no global
//! state, and the dispatcher equals it bit-for-bit. The name mirrors the eventual stdarch
//! intrinsic stem `cvtssepi32_epi8` (`[avx10-v2-aux-ocp-conversions.NAMING.1]`), and the
//! whole module compiles on stable Rust with no `core::simd`/nightly
//! (`[avx10-v2-aux-ocp-conversions.STABLE_RUST.1]`).
//!
//! OQ-5 (intrinsic unavailable -> oracle-only): the symmetric-saturation convert ships
//! **oracle-only**. `VPMOVSSDB` is encoded `EVEX.512.F3.0F38.W0 41 /r` and its intrinsic is
//! `_mm512_cvtssepi32_epi8` (sect 9.8.7), but a compile probe under `-mavx10.2` (GCC 16.1.1)
//! shows that intrinsic is ABSENT — the compiler offers only `_mm512_cvtusepi32_epi8` (the
//! *unsigned*-saturation sibling), confirming the symmetric `cvtss` form does not yet exist
//! in the toolchain. Per OQ-5 there is therefore no native C shim, no `extern "C"`
//! declaration, and no `_hw` path; each dispatcher resolves to its `_scalar` sibling on every
//! target. The capability check
//! `crate::detect::has_avx10_v2_aux` is never consulted — with no native path there is
//! nothing to gate; the dispatcher only references the detector to mark the future gate
//! site. A native path is added once the intrinsic lands in the toolchain. The
//! differential test that would otherwise tie a native path to the oracle DISCARDS (no native
//! path exists), so correctness is grounded against the sect 9.8.5 pseudocode transcribed in
//! `saturate_int32_to_symmetric_int8`.

use crate::detect;

/// The symmetric-signed INT8 saturation bounds (ACE v1 sect 9.8.1).
///
/// `MAX_POSITIVE = 0x7F = +127`, `MAX_NEGATIVE = 0x81 = -127`. The range is symmetric about
/// zero — distinct from the asymmetric `[-128, +127]` (`0x80 = -128`) of ordinary
/// `VPMOVSDB` (sect 9.8.7 note).
const MAX_POSITIVE: i32 = 0x7F; // +127
const MAX_NEGATIVE: i32 = -127; // 0x81 as i8

/// Symmetric signed saturation of one INT32 lane to INT8, per sect 9.8.5
/// (`saturate_int32_to_symmetric_int8`).
///
/// Clamps `x` into the SYMMETRIC range `[-127, +127]` and narrows to `i8`. Because the clamp
/// bounds both fit in `i8`, the `as i8` cast is exact (no truncation): a clamped value is
/// already in `-127..=127`. This is the teeth of the symmetric-vs-ordinary distinction —
/// `x = i32::MIN` clamps to `-127` (`0x81`), NOT `-128` (`0x80`) as the asymmetric
/// `VPMOVSDB` would yield (`[avx10-v2-aux-ocp-conversions.CVT_SSDB.1]`,
/// `[avx10-v2-aux-ocp-conversions.CVT_SSDB.1-note]`).
///
/// The clamp is an odd function: `saturate(-x) == -saturate(x)` for every `i32` `x`, because
/// the bounds are negatives of each other (`[avx10-v2-aux-ocp-conversions.CVT_SSDB.2]`).
fn saturate_int32_to_symmetric_int8(x: i32) -> i8 {
    x.clamp(MAX_NEGATIVE, MAX_POSITIVE) as i8
}

/// INT32 -> INT8 with symmetric signed saturation (16 lanes), the public `VPMOVSSDB`
/// dispatcher.
///
/// Per lane: clamp the signed `i32` to the symmetric `[-127, +127]` and narrow to `i8` via
/// `saturate_int32_to_symmetric_int8`. The output is one quarter of the input width (16
/// dwords -> 16 bytes, `[avx10-v2-aux-ocp-conversions.CVT_SSDB.3]`).
///
/// No native path is wired, so `detect::has_avx10_v2_aux` is never consulted (OQ-5,
/// see the module docs — `_mm512_cvtssepi32_epi8` is absent from the `-mavx10.2` toolchain);
/// the dispatcher resolves to [`cvtssepi32_epi8_scalar`] on every target, returning
/// the spec-defined value (`[avx10-v2-aux-ocp-conversions.DETECTION.2]`).
pub fn cvtssepi32_epi8(a: [i32; 16]) -> [i8; 16] {
    // No native path this phase (OQ-5): the symmetric-saturation intrinsic
    // `_mm512_cvtssepi32_epi8` is absent from the `-mavx10.2` toolchain (the compiler exposes
    // only the unsigned-saturation `_mm512_cvtusepi32_epi8`), so even under `feature="native"`
    // on AVX10_V2_AUX hardware the oracle is the only path. The detector is only referenced
    // (never called), marking the gate site for the shim once the intrinsic lands.
    let _ = detect::has_avx10_v2_aux; // reference (not call) the future gate; see fn docs
    cvtssepi32_epi8_scalar(a)
}

/// Portable reference oracle for [`cvtssepi32_epi8`] — the primary always-correct path.
///
/// Maps each `i32` lane through `saturate_int32_to_symmetric_int8`, the sect 9.8.5
/// symmetric clamp. Carries no cfg gate and reads no global state.
/// `[avx10-v2-aux-ocp-conversions.CORRECTNESS.1]` `[avx10-v2-aux-ocp-conversions.CORRECTNESS.2]`
pub fn cvtssepi32_epi8_scalar(a: [i32; 16]) -> [i8; 16] {
    core::array::from_fn(|i| saturate_int32_to_symmetric_int8(a[i]))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-computed known-value vectors pinning the symmetric saturation, chosen so each
    /// lane DISTINGUISHES the symmetric `[-127, +127]` clamp from the leading wrong model —
    /// the ordinary asymmetric `VPMOVSDB` clamp `[-128, +127]`
    /// (`[avx10-v2-aux-ocp-conversions.CVT_SSDB.1]`,
    /// `[avx10-v2-aux-ocp-conversions.CVT_SSDB.1-note]`,
    /// `[avx10-v2-aux-ocp-conversions.CVT_SSDB.2]`).
    ///
    /// DISCRIMINATING lanes (each rules out the asymmetric `[-128,+127]` model):
    ///  * lane 0 `i32::MIN` -> `-127`: the asymmetric clamp would give `-128`. This is the
    ///    single sharpest discriminator and the `f(i32::MIN) = -127` property
    ///    (`[avx10-v2-aux-ocp-conversions.CVT_SSDB.2]`).
    ///  * lane 5 `-128` -> `-127`: an in-`i8`-range NEGATIVE value that the symmetric clamp
    ///    still pulls up to `-127`, whereas the asymmetric clamp (and a naive `as i8` cast)
    ///    would PASS `-128` through unchanged. This pins the symmetric lower bound at exactly
    ///    `-127`, the most subtle boundary.
    ///  * lane 6 `-127` -> `-127` (unchanged): the largest-magnitude negative that survives.
    ///  * lane 1 `i32::MAX` -> `+127`, lane 3 `+200` -> `+127`, lane 4 `-200` -> `-127`,
    ///    lane 7 `+127` -> `+127`: positive/large clamps (these agree with the asymmetric
    ///    model on the high side, so they pin the clamp magnitude, not the asymmetry).
    ///  * lane 2 `0` -> `0`, lane 8 `+42` -> `+42`, lane 9 `-42` -> `-42`: in-range
    ///    pass-through.
    #[test]
    fn known_value_symmetric_saturation() {
        let mut a = [0i32; 16];
        a[0] = i32::MIN; // -> -127 (NOT -128)
        a[1] = i32::MAX; // -> +127
        a[2] = 0; // -> 0
        a[3] = 200; // -> +127
        a[4] = -200; // -> -127
        a[5] = -128; // -> -127 (NOT -128 — symmetric vs ordinary distinction)
        a[6] = -127; // -> -127 (unchanged)
        a[7] = 127; // -> +127 (unchanged)
        a[8] = 42; // -> +42 (in range)
        a[9] = -42; // -> -42 (in range)

        let got = cvtssepi32_epi8(a);

        assert_eq!(got[0], -127, "i32::MIN -> -127 (symmetric, NOT -128)");
        assert_eq!(got[1], 127, "i32::MAX -> +127");
        assert_eq!(got[2], 0, "0 -> 0");
        assert_eq!(got[3], 127, "+200 -> +127");
        assert_eq!(got[4], -127, "-200 -> -127");
        assert_eq!(
            got[5], -127,
            "-128 -> -127 (symmetric clamp pulls up; ordinary [-128,+127] would keep -128)"
        );
        assert_eq!(got[6], -127, "-127 -> -127 (largest negative, unchanged)");
        assert_eq!(got[7], 127, "+127 -> +127 (unchanged)");
        assert_eq!(got[8], 42, "+42 -> +42 (in range)");
        assert_eq!(got[9], -42, "-42 -> -42 (in range)");
        // Untouched lanes are 0 -> 0.
        assert_eq!(got[15], 0, "padding lane 0 -> 0");

        // Explicit witness that MAX_NEGATIVE is the byte 0x81, not 0x80.
        assert_eq!(got[0] as u8, 0x81, "-127 encodes as 0x81 (MAX_NEGATIVE)");
        assert_eq!(got[1] as u8, 0x7F, "+127 encodes as 0x7F (MAX_POSITIVE)");
    }
}

/// Property-based tests for family H (`VPMOVSSDB`). The hand-rolled tests above pin specific
/// boundary values; these assert the invariants across the full `i32` input space.
#[cfg(test)]
mod proptests {
    use super::*;
    use quickcheck::{quickcheck, Arbitrary, Gen};

    /// A random 16-lane INT32 input. `quickcheck` does not derive `Arbitrary` for arrays of
    /// this length, so we wrap it and fill each lane independently — every `i32` (incl.
    /// `i32::MIN` / `i32::MAX` and in-range values) is reachable per lane.
    #[derive(Clone, Debug)]
    struct Inputs {
        a: [i32; 16],
    }

    impl Arbitrary for Inputs {
        fn arbitrary(g: &mut Gen) -> Self {
            Inputs {
                a: core::array::from_fn(|_| i32::arbitrary(g)),
            }
        }
    }

    quickcheck! {
        /// The public dispatcher always equals the scalar oracle bit-for-bit — the contract
        /// callers rely on regardless of which path runs. Since family H is oracle-only this
        /// phase (OQ-5), this also pins that the dispatcher returns the spec value on every
        /// input (`[avx10-v2-aux-ocp-conversions.CORRECTNESS.1]`,
        /// `[avx10-v2-aux-ocp-conversions.DETECTION.2]`).
        fn prop_public_matches_scalar(input: Inputs) -> bool {
            cvtssepi32_epi8(input.a) == cvtssepi32_epi8_scalar(input.a)
        }

        /// Odd-function symmetry about zero: `f(-x) == -f(x)` for every `i32` `x`
        /// (`[avx10-v2-aux-ocp-conversions.CVT_SSDB.2]`).
        ///
        /// `x == i32::MIN` is handled EXPLICITLY: `-i32::MIN` overflows `i32`, so the property
        /// `f(-x) == -f(x)` cannot be stated by negating the input. Instead we pin the
        /// boundary directly — `f(i32::MIN) == -127` and `-f(i32::MIN) == 127 == f(i32::MAX)`
        /// — which is exactly what the odd-function identity demands at the limit. For all
        /// other lanes the negation is safe and the identity is checked directly. (`f(x)` is
        /// always in `[-127, 127]`, so `-f(x)` never overflows `i8`.)
        fn prop_odd_function(input: Inputs) -> bool {
            let fx = cvtssepi32_epi8(input.a);
            (0..16).all(|i| {
                let x = input.a[i];
                if x == i32::MIN {
                    // -i32::MIN overflows; assert the boundary the identity implies instead.
                    fx[i] == -127 && cvtssepi32_epi8([i32::MAX; 16])[i] == 127
                } else {
                    let fneg = saturate_int32_to_symmetric_int8(-x);
                    fneg == -fx[i]
                }
            })
        }

        /// Clamp-range invariant: every output lane lies within `[-127, +127]`
        /// (`[avx10-v2-aux-ocp-conversions.CVT_SSDB.1]`).
        fn prop_in_range(input: Inputs) -> bool {
            cvtssepi32_epi8(input.a)
                .iter()
                .all(|&b| (-127..=127).contains(&b))
        }

        /// Lane independence: each output lane depends only on the corresponding input lane.
        /// Replacing one lane changes only that output lane.
        /// (`[avx10-v2-aux-ocp-conversions.CORRECTNESS.1]`).
        fn prop_lane_independence(input: Inputs, lane: usize, repl: i32) -> bool {
            let lane = lane % 16;
            let base = cvtssepi32_epi8(input.a);
            let mut perturbed = input.a;
            perturbed[lane] = repl;
            let out = cvtssepi32_epi8(perturbed);
            (0..16).all(|i| {
                if i == lane {
                    out[i] == saturate_int32_to_symmetric_int8(repl)
                } else {
                    out[i] == base[i]
                }
            })
        }
    }
}

/// Native-vs-oracle differential for family H (`VPMOVSSDB`). Phase 11.
///
/// Family H ships **oracle-only** in this toolchain (OQ-5: `_mm512_cvtssepi32_epi8` is absent
/// under `-mavx10.2`). The property compares the public dispatcher to its scalar oracle under
/// `feature="native"` on AVX10_V2_AUX hardware (`[avx10-v2-aux-ocp-conversions.DIFFERENTIAL.1]`),
/// and `TestResult::discard()`s (never `from_bool(false)`) otherwise, so a fallback-only runner
/// cannot go vacuously green and the test becomes live the moment a native path lands.
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
        a: [i32; 16],
    }

    impl Arbitrary for Inputs {
        fn arbitrary(g: &mut Gen) -> Self {
            Inputs {
                a: core::array::from_fn(|_| i32::arbitrary(g)),
            }
        }
    }

    quickcheck! {
        /// Family-H native-vs-oracle differential. Under `feature="native"` on x86_64 with
        /// `AVX10_V2_AUX` detected, the public dispatcher must equal the scalar oracle
        /// bit-for-bit (`[avx10-v2-aux-ocp-conversions.DIFFERENTIAL.1]`). DISCARDED (not failed)
        /// when the feature or hardware is absent (`[avx10-v2-aux-ocp-conversions.CORRECTNESS.2]`),
        /// so a fallback-only runner never produces a vacuous green.
        fn prop_native_matches_oracle(input: Inputs) -> TestResult {
            #[cfg(all(target_arch = "x86_64", feature = "native"))]
            {
                if detect::has_avx10_v2_aux() {
                    return TestResult::from_bool(
                        cvtssepi32_epi8(input.a) == cvtssepi32_epi8_scalar(input.a),
                    );
                }
            }
            let _ = &input;
            TestResult::discard()
        }
    }
}
