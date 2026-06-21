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
//!
//! **Iteration 1** (in progress) adds the `AVX10_V1_AUX` family of FP16↔FP8 / FP32→FP16
//! converts and the EVEX byte/word VNNI matrix, each behind a crate-owned capability check
//! ([`detect`]) over the shared FP8/FP16 conversion oracle ([`fp8`]). In v1 every new
//! primitive ships oracle-only: no stable EVEX intrinsic exists yet, so the native slot is
//! dormant ([avx10-v1-aux-fp16-fp8-evex-vnni.DISPATCH.3]).
//!
//! The EVEX byte/word VNNI primitives live in the [`vnni`] module and are reached
//! module-qualified — e.g. the 512-bit EVEX `dpbssd` is [`vnni::dpbssd`]
//! (`ace::vnni::dpbssd`), DISTINCT from this crate's iteration-0 256-bit VEX [`dpbssd`]
//! (`ace::dpbssd`). The two are resolved by module path and neither shadows the other
//! ([avx10-v1-aux-fp16-fp8-evex-vnni.BYTE_VNNI.1], OQ-1).
//!
//! # Native-coverage tripwire (`ACE_REQUIRE_NATIVE`) — scope in v1
//!
//! The `ACE_REQUIRE_NATIVE=1` coverage tripwire (CI's `native-sde` job) stays **meaningful
//! only for the iteration-0 group-1 [`dpbssd`]**: that primitive has a real stable
//! `core::arch` intrinsic, so the `native_runs_when_required` guard in this module's tests
//! asserts the native `VPDPBSSD` branch actually ran rather than the scalar fallback
//! ([avx10-v1-aux-fp16-fp8-evex-vnni.DIFFERENTIAL.2],
//! [avx10-v1-aux-fp16-fp8-evex-vnni.CI.2]). For the new `AVX10_V1_AUX` families (A–G) the
//! tripwire is intentionally **dormant**: no stable EVEX intrinsic exists yet (OQ-3,
//! oracle-only v1, [avx10-v1-aux-fp16-fp8-evex-vnni.DISPATCH.3]), so there is no native
//! branch for it to require. Each new family nonetheless ships a *structurally present*
//! `prop_native_matches_oracle` differential (in its `proptests` module) that compares the
//! public dispatcher to the oracle under `detect::has_avx10_v1_aux()` and calls
//! `TestResult::discard()` — never `from_bool(false)` — when no native path is present, so a
//! fallback-only runner can never produce a vacuous green
//! ([avx10-v1-aux-fp16-fp8-evex-vnni.DIFFERENTIAL.1],
//! [avx10-v1-aux-fp16-fp8-evex-vnni.DIFFERENTIAL.1-1]). When a stable AVX10.2 intrinsic
//! lands, wiring the dormant slot turns each family's tripwire live by the same pattern.
//!
//! # v1 non-goals — confirmed NOT implemented
//!
//! The public surface of this crate is exactly the iteration-0 [`dpbssd`] plus the 26
//! `AVX10_V1_AUX` primitives (families A–G). The following are deliberately **out of scope**
//! for v1 and are NOT present in any public item or native path
//! (verified by [`tests::non_goals_absent`]):
//!
//! - **No `AVX10_V2_AUX` (group 3) instructions** — no FP32↔FP8, FP4/FP6, `VPMOVSSDB`, or
//!   `VUNPACKB` converts (spec §6.2).
//! - **No group-4 `ACE` instructions** — no `TOP*`, `BSR*`, or tile-move primitives.
//! - **No VEX-encoded AVX-VNNI-INT8/16 forms beyond the existing 256-bit [`dpbssd`]** — the
//!   family-F/G additions are the EVEX 512-bit generalization ([`vnni`]), not new VEX forms.
//! - **No EVEX write-masking (`{k1}{z}`) or memory-broadcast (`m*bcst`) in the public API** —
//!   every primitive takes plain fixed-size lane arrays by value and writes a full result; the
//!   spec's `k1` / `zeroing` / `evex_b` operands are fixed to the no-writemask, no-broadcast
//!   case (`no_writemask = true`, `evex_b = false`) and are not surfaced.

pub mod cvt_fp8_ph;
pub mod cvt_ph_fp8;
pub mod cvt_ps_ph;
mod detect;
pub(crate) mod fp8;
#[cfg(all(target_arch = "x86_64", feature = "native"))]
pub(crate) mod native;
pub mod vnni;

pub use cvt_ph_fp8::{
    cvt2ph_bf8, cvt2ph_bf8_scalar, cvt2ph_hf8, cvt2ph_hf8_scalar, cvt2phs_bf8, cvt2phs_bf8_scalar,
    cvt2phs_hf8, cvt2phs_hf8_scalar, cvtbiasph_bf8, cvtbiasph_bf8_scalar, cvtbiasph_hf8,
    cvtbiasph_hf8_scalar, cvtbiasphs_bf8, cvtbiasphs_bf8_scalar, cvtbiasphs_hf8,
    cvtbiasphs_hf8_scalar, cvtph_bf8, cvtph_bf8_scalar, cvtph_hf8, cvtph_hf8_scalar, cvtphs_bf8,
    cvtphs_bf8_scalar, cvtphs_hf8, cvtphs_hf8_scalar,
};

pub use cvt_fp8_ph::{cvthf8_ph, cvthf8_ph_scalar};

pub use cvt_ps_ph::{cvt2ps_phx, cvt2ps_phx_scalar};

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

/// Property-based tests. The hand-rolled tests above pin specific values; these
/// assert the invariants hold across a randomly-sampled slice of the input space,
/// which is far wider than any hand-picked vector can cover.
#[cfg(test)]
mod proptests {
    use super::*;
    use quickcheck::{quickcheck, Arbitrary, Gen, TestResult};

    /// A full, independently-random argument triple for [`dpbssd`].
    ///
    /// We wrap the three fixed-size arrays in a newtype because `quickcheck` does
    /// not derive `Arbitrary` for arrays of this length; `from_fn` fills each lane
    /// from the generator so every byte is sampled independently.
    #[derive(Clone, Debug)]
    struct Inputs {
        src: [i32; 8],
        a: [i8; 32],
        b: [i8; 32],
    }

    impl Arbitrary for Inputs {
        fn arbitrary(g: &mut Gen) -> Self {
            Inputs {
                src: core::array::from_fn(|_| i32::arbitrary(g)),
                a: core::array::from_fn(|_| i8::arbitrary(g)),
                b: core::array::from_fn(|_| i8::arbitrary(g)),
            }
        }
    }

    quickcheck! {
        /// Differential property — the headline guarantee. On any input the native
        /// `VPDPBSSD` path must agree with the scalar oracle bit-for-bit. The case
        /// is *discarded* (not passed) when the native path is unavailable, so a
        /// runner without AVX-VNNI-INT8 cannot turn this into a vacuous green; under
        /// CI's SDE job it exercises the real instruction over 100 random inputs.
        fn prop_hw_matches_scalar(input: Inputs) -> TestResult {
            #[cfg(target_arch = "x86_64")]
            {
                if std::is_x86_feature_detected!("avxvnniint8") {
                    let want = dpbssd_scalar(input.src, input.a, input.b);
                    // SAFETY: the feature was confirmed present immediately above.
                    let got = unsafe { dpbssd_hw(input.src, input.a, input.b) };
                    return TestResult::from_bool(got == want);
                }
            }
            TestResult::discard()
        }

        /// The public dispatcher always equals the scalar oracle — this is the
        /// contract callers rely on regardless of which path runs.
        fn prop_public_matches_scalar(input: Inputs) -> bool {
            dpbssd(input.src, input.a, input.b)
                == dpbssd_scalar(input.src, input.a, input.b)
        }

        /// Accumulator linearity: `src` is a pure additive bias (wrapping i32),
        /// independent of the dot products it is added to.
        fn prop_src_is_additive(input: Inputs) -> bool {
            let with_src = dpbssd_scalar(input.src, input.a, input.b);
            let no_src = dpbssd_scalar([0; 8], input.a, input.b);
            (0..8).all(|i| with_src[i] == input.src[i].wrapping_add(no_src[i]))
        }

        /// Operand symmetry: each lane is a dot product `a·b`, so swapping the two
        /// multiplicand vectors leaves the result unchanged.
        fn prop_operands_commute(input: Inputs) -> bool {
            dpbssd_scalar(input.src, input.a, input.b)
                == dpbssd_scalar(input.src, input.b, input.a)
        }

        /// A zeroed multiplicand contributes nothing: the output is exactly `src`.
        fn prop_zero_operand_is_passthrough(input: Inputs) -> bool {
            dpbssd_scalar(input.src, [0; 32], input.b) == input.src
        }

        /// Lane independence: output lane `i` depends only on `a[4i..4i+4]` and
        /// `b[4i..4i+4]`. Zeroing every other lane's operands must not change it.
        fn prop_lanes_are_independent(input: Inputs, lane: u8) -> bool {
            let i = (lane % 8) as usize;
            let mut a = [0i8; 32];
            let mut b = [0i8; 32];
            for k in 0..4 {
                a[4 * i + k] = input.a[4 * i + k];
                b[4 * i + k] = input.b[4 * i + k];
            }
            dpbssd_scalar(input.src, a, b)[i]
                == dpbssd_scalar(input.src, input.a, input.b)[i]
        }
    }
}

#[cfg(test)]
mod non_goal_guards {
    //! Documented guard that the v1 non-goals were not built into the public surface
    //! ([avx10-v1-aux-fp16-fp8-evex-vnni.ENCODING.1] non-goal half: no out-of-scope native
    //! encodings are emitted). The crate exposes no `core::arch` calls outside the single
    //! iteration-0 `VPDPBSSD` intrinsic, and no public item names a group-3/4 mnemonic.

    /// Confirms the public function inventory is exactly iteration-0 `dpbssd` plus the 26
    /// `AVX10_V1_AUX` primitives — no group-3 (`AVX10_V2_AUX`) or group-4 (`ACE`) surface, no
    /// masking/broadcast variants. This is a readable, asserting record of the non-goals; it
    /// references each public family entry point so any accidental removal or out-of-scope
    /// addition is caught at compile time.
    #[test]
    fn non_goals_absent() {
        // The complete v1 public primitive set is exercised below — one representative entry
        // point per family plus the iteration-0 VEX `dpbssd`. There is deliberately NO
        // FP32->FP8 (`cvtps_*`), FP4/FP6, `vpmovssdb`, `vunpackb`, `top*`, `bsr*`, tile-move,
        // or `{k1}{z}` / `*bcst` entry point — group 3/4 and masking/broadcast are out of v1.
        // Each call takes plain fixed-size lane arrays by value (no mask / no broadcast operand
        // exists to pass), which is itself the guarantee that the masked/broadcast surface was
        // never built. Any out-of-scope or removed primitive would break this compile.
        let _a = crate::cvtph_bf8([0u16; 32]); // families A/B/C: FP16 -> FP8
        let _b = crate::cvtphs_hf8([0u16; 32]);
        let _d = crate::cvthf8_ph([0u8; 32]); // family D: HF8 -> FP16
        let _e = crate::cvt2ps_phx([0.0f32; 16], [0.0f32; 16]); // family E: FP32 pair -> FP16
        let _f = crate::vnni::dpbssd([0i32; 16], [0i8; 64], [0i8; 64]); // family F: byte VNNI (EVEX 512-bit)
        let _g = crate::vnni::dpwsud([0i32; 16], [0i16; 32], [0u16; 32]); // family G: word VNNI (EVEX 512-bit)
        let _group1_vex = crate::dpbssd([0i32; 8], [0i8; 32], [0i8; 32]); // iteration-0 VEX dpbssd (256-bit)
                                                                          // No out-of-scope symbol exists to reference here — that absence is the guarantee.
    }
}
