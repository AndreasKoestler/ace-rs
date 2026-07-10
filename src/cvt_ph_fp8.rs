//! Families A, B, C: FP16 -> FP8 converts.
//!
//! Each public dispatcher is a safe fn that selects a native path when the running CPU
//! supports `AVX10_V1_AUX` (via a hand-written C shim behind the opt-in `native` feature —
//! no stable `core::arch` EVEX intrinsic exists yet — per
//! `[avx10-v1-aux-fp16-fp8-evex-vnni.DISPATCH.3]`) and otherwise falls back to its `_scalar`
//! oracle. The `_scalar` oracle is the primary,
//! always-correct path on every target including non-x86
//! (`[avx10-v1-aux-fp16-fp8-evex-vnni.ORACLE.1]`); it carries no cfg gate, reads no
//! global state, and the dispatcher equals it bit-for-bit
//! (`[avx10-v1-aux-fp16-fp8-evex-vnni.ORACLE.2]`). Names mirror the eventual stdarch
//! intrinsic stems (`[avx10-v1-aux-fp16-fp8-evex-vnni.NAMING.1]`).
//!
//! Family A is the four single-source converters: `cvtph_bf8` / `cvtphs_bf8` (FP16 ->
//! BF8/E5M2, non-saturating / saturating) and `cvtph_hf8` / `cvtphs_hf8` (FP16 ->
//! HF8/E4M3). The `S` suffix selects saturating mode, clamping an overflowing magnitude
//! to the format max normal (BF8 +/-57344, HF8 +/-448); the non-saturating form emits
//! the format NaN/overflow encoding per spec section 8.2
//! (`[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_PH2FP8.1-2]`).
//!
//! Family B is the four two-source converters: `cvt2ph_bf8` / `cvt2phs_bf8` /
//! `cvt2ph_hf8` / `cvt2phs_hf8`. Each concatenates two FP16 vectors into one 64-lane FP8
//! output with the spec section 8.2.5 lane ordering — src2 in the low half, src1 in the
//! high half (`[avx10-v1-aux-fp16-fp8-evex-vnni.CVT2_PH2FP8.1]`,
//! `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT2_PH2FP8.1-1]`).
//!
//! Family C is the four biased converters: `cvtbiasph_bf8` / `cvtbiasphs_bf8` /
//! `cvtbiasph_hf8` / `cvtbiasphs_hf8`. They apply a per-lane bias rounding term (spec
//! section 2.6.3) before rounding to the target FP8 format, reusing the family-A
//! saturation matrix with bias rounding replacing plain RTNE
//! (`[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_BIAS_PH2FP8.1]`,
//! `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_BIAS_PH2FP8.1-1]`). Per spec section 8.4.5 the
//! bias operand is `src1` and the bias for output lane `i` is `src1.byte[2 * i]` — the
//! low byte of the i-th `u16` element of the bias vector. The public bias signature takes
//! that vector as `[u16; 32]` and selects `bias[i] & 0xff` per lane, matching the spec's
//! byte-index `2 * i` into a little-endian `u16` array (OQ-5).

use crate::detect;
use crate::fp8;

/// Single-source FP16 -> BF8 (E5M2) convert, non-saturating.
///
/// Per FP16 lane: decode, round-to-nearest-even to BF8, encode to one `u8`. A
/// magnitude that overflows BF8 yields the format NaN/overflow encoding (non-saturating)
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_PH2FP8.1-1]`. MXCSR is not consulted; DAZ=0,
/// FTZ=0 `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_PH2FP8.1-3]`.
///
/// Dispatches to the native path under `AVX10_V1_AUX` (C shim, opt-in `native` feature) and otherwise to
/// [`cvtph_bf8_scalar`]; both return identical bytes.
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_PH2FP8.1]` `[avx10-v1-aux-fp16-fp8-evex-vnni.DISPATCH.1]`
pub fn cvtph_bf8(a: [u16; 32]) -> [u8; 32] {
    #[cfg(all(target_arch = "x86_64", feature = "native"))]
    {
        if detect::has_avx10_v1_aux() {
            // SAFETY: `has_avx10_v1_aux()` confirmed full AVX10.2 (the feature set this shim's
            // translation unit is compiled for) plus OS XSAVE state immediately above.
            return unsafe { cvtph_bf8_hw(a) };
        }
    }
    let _ = detect::has_avx10_v1_aux; // keep `detect` referenced on every target
    cvtph_bf8_scalar(a)
}

/// Native path: EVEX `VCVTPH2BF8` via the `ace_native_cvtph_bf8` C shim.
///
/// # Safety
/// The CPU must support `AVX10_V1_AUX`; callers go through [`cvtph_bf8`], which checks it.
#[cfg(all(target_arch = "x86_64", feature = "native"))]
unsafe fn cvtph_bf8_hw(a: [u16; 32]) -> [u8; 32] {
    let mut out = [0u8; 32];
    crate::native::ace_native_cvtph_bf8(a.as_ptr(), out.as_mut_ptr());
    out
}

/// Portable reference oracle for [`cvtph_bf8`] — the primary always-correct path.
///
/// Maps each FP16 lane through [`fp8::fp16_to_bf8`] in non-saturating mode. Carries no
/// cfg gate and reads no global state. `[avx10-v1-aux-fp16-fp8-evex-vnni.ORACLE.1]`
pub fn cvtph_bf8_scalar(a: [u16; 32]) -> [u8; 32] {
    core::array::from_fn(|i| fp8::fp16_to_bf8(a[i], false))
}

/// Single-source FP16 -> BF8 (E5M2) convert, saturating.
///
/// Identical to [`cvtph_bf8`] except an overflowing magnitude clamps to the BF8 max
/// normal `+/-57344` instead of emitting the NaN/overflow encoding
/// (`[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_PH2FP8.1-2]`).
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_PH2FP8.1]` `[avx10-v1-aux-fp16-fp8-evex-vnni.DISPATCH.1]`
pub fn cvtphs_bf8(a: [u16; 32]) -> [u8; 32] {
    #[cfg(all(target_arch = "x86_64", feature = "native"))]
    {
        if detect::has_avx10_v1_aux() {
            // SAFETY: `has_avx10_v1_aux()` confirmed full AVX10.2 (the feature set this shim's
            // translation unit is compiled for) plus OS XSAVE state immediately above.
            return unsafe { cvtphs_bf8_hw(a) };
        }
    }
    let _ = detect::has_avx10_v1_aux;
    cvtphs_bf8_scalar(a)
}

/// Native path: EVEX `VCVTPH2BF8S` (saturating) via `ace_native_cvtphs_bf8`.
///
/// # Safety
/// The CPU must support `AVX10_V1_AUX`; callers go through [`cvtphs_bf8`].
#[cfg(all(target_arch = "x86_64", feature = "native"))]
unsafe fn cvtphs_bf8_hw(a: [u16; 32]) -> [u8; 32] {
    let mut out = [0u8; 32];
    crate::native::ace_native_cvtphs_bf8(a.as_ptr(), out.as_mut_ptr());
    out
}

/// Portable reference oracle for [`cvtphs_bf8`] — the primary always-correct path.
///
/// Maps each FP16 lane through [`fp8::fp16_to_bf8`] in saturating mode (overflow clamps
/// to +/-57344). `[avx10-v1-aux-fp16-fp8-evex-vnni.ORACLE.1]`
pub fn cvtphs_bf8_scalar(a: [u16; 32]) -> [u8; 32] {
    core::array::from_fn(|i| fp8::fp16_to_bf8(a[i], true))
}

/// Single-source FP16 -> HF8 (E4M3) convert, non-saturating.
///
/// Per FP16 lane: decode, round-to-nearest-even to HF8, encode to one `u8`. A magnitude
/// that overflows HF8 yields the HF8 NaN encoding `S.1111.111` (non-saturating)
/// (`[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_PH2FP8.1-1]`).
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_PH2FP8.1]` `[avx10-v1-aux-fp16-fp8-evex-vnni.DISPATCH.1]`
pub fn cvtph_hf8(a: [u16; 32]) -> [u8; 32] {
    #[cfg(all(target_arch = "x86_64", feature = "native"))]
    {
        if detect::has_avx10_v1_aux() {
            // SAFETY: `has_avx10_v1_aux()` confirmed full AVX10.2 (the feature set this shim's
            // translation unit is compiled for) plus OS XSAVE state immediately above.
            return unsafe { cvtph_hf8_hw(a) };
        }
    }
    let _ = detect::has_avx10_v1_aux;
    cvtph_hf8_scalar(a)
}

/// Native path: EVEX `VCVTPH2HF8` via `ace_native_cvtph_hf8`.
///
/// # Safety
/// The CPU must support `AVX10_V1_AUX`; callers go through [`cvtph_hf8`].
#[cfg(all(target_arch = "x86_64", feature = "native"))]
unsafe fn cvtph_hf8_hw(a: [u16; 32]) -> [u8; 32] {
    let mut out = [0u8; 32];
    crate::native::ace_native_cvtph_hf8(a.as_ptr(), out.as_mut_ptr());
    out
}

/// Portable reference oracle for [`cvtph_hf8`] — the primary always-correct path.
///
/// Maps each FP16 lane through [`fp8::fp16_to_hf8`] in non-saturating mode.
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.ORACLE.1]`
pub fn cvtph_hf8_scalar(a: [u16; 32]) -> [u8; 32] {
    core::array::from_fn(|i| fp8::fp16_to_hf8(a[i], false))
}

/// Single-source FP16 -> HF8 (E4M3) convert, saturating.
///
/// Identical to [`cvtph_hf8`] except an overflowing magnitude clamps to the HF8 max
/// normal `+/-448` instead of emitting the NaN encoding
/// (`[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_PH2FP8.1-2]`).
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_PH2FP8.1]` `[avx10-v1-aux-fp16-fp8-evex-vnni.DISPATCH.1]`
pub fn cvtphs_hf8(a: [u16; 32]) -> [u8; 32] {
    #[cfg(all(target_arch = "x86_64", feature = "native"))]
    {
        if detect::has_avx10_v1_aux() {
            // SAFETY: `has_avx10_v1_aux()` confirmed full AVX10.2 (the feature set this shim's
            // translation unit is compiled for) plus OS XSAVE state immediately above.
            return unsafe { cvtphs_hf8_hw(a) };
        }
    }
    let _ = detect::has_avx10_v1_aux;
    cvtphs_hf8_scalar(a)
}

/// Native path: EVEX `VCVTPH2HF8S` (saturating) via `ace_native_cvtphs_hf8`.
///
/// # Safety
/// The CPU must support `AVX10_V1_AUX`; callers go through [`cvtphs_hf8`].
#[cfg(all(target_arch = "x86_64", feature = "native"))]
unsafe fn cvtphs_hf8_hw(a: [u16; 32]) -> [u8; 32] {
    let mut out = [0u8; 32];
    crate::native::ace_native_cvtphs_hf8(a.as_ptr(), out.as_mut_ptr());
    out
}

/// Portable reference oracle for [`cvtphs_hf8`] — the primary always-correct path.
///
/// Maps each FP16 lane through [`fp8::fp16_to_hf8`] in saturating mode (overflow clamps
/// to +/-448). `[avx10-v1-aux-fp16-fp8-evex-vnni.ORACLE.1]`
pub fn cvtphs_hf8_scalar(a: [u16; 32]) -> [u8; 32] {
    core::array::from_fn(|i| fp8::fp16_to_hf8(a[i], true))
}

// --- Family B: two-source FP16 -> FP8 converts (concatenated 64-lane output) ---
//
// Per ACE v1 spec section 8.2.5 `vcvt2ph2f8`, two FP16 input vectors are concatenated
// into one FP8 output of the same total width (KL = VL/8 = 64 lanes here). The lane
// ordering is fixed by the pseudocode: for output lane `i`, `t = src2.fp16[i]` when
// `i < KL/2` and `t = src1.fp16[i - KL/2]` otherwise. Concretely the low half [0..32)
// is the src2 conversion and the high half [32..64) is the src1 conversion
// (`[avx10-v1-aux-fp16-fp8-evex-vnni.CVT2_PH2FP8.1-1]`). Format, saturation, and
// RTNE-rounding semantics are identical to family A
// (`[avx10-v1-aux-fp16-fp8-evex-vnni.CVT2_PH2FP8.1]`); only the lane count and the
// src2->low / src1->high placement differ.

/// Shared two-source lane layout: output lane `i` is `convert(src2[i])` for `i < 32`
/// and `convert(src1[i - 32])` for `i >= 32`, per spec section 8.2.5 `vcvt2ph2f8`
/// (`[avx10-v1-aux-fp16-fp8-evex-vnni.CVT2_PH2FP8.1-1]`).
fn cvt2_lanes(src1: [u16; 32], src2: [u16; 32], convert: impl Fn(u16) -> u8) -> [u8; 64] {
    core::array::from_fn(|i| {
        if i < 32 {
            convert(src2[i])
        } else {
            convert(src1[i - 32])
        }
    })
}

/// Two-source FP16 -> BF8 (E5M2) convert, non-saturating.
///
/// Concatenates two FP16 vectors into one 64-lane BF8 output: low half [0..32) from
/// `src2`, high half [32..64) from `src1`, each lane converted exactly as
/// [`cvtph_bf8`]. Overflow yields the BF8 NaN/overflow encoding
/// (`[avx10-v1-aux-fp16-fp8-evex-vnni.CVT2_PH2FP8.1]`,
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT2_PH2FP8.1-1]`).
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.DISPATCH.1]`
pub fn cvt2ph_bf8(src1: [u16; 32], src2: [u16; 32]) -> [u8; 64] {
    #[cfg(all(target_arch = "x86_64", feature = "native"))]
    {
        if detect::has_avx10_v1_aux() {
            // SAFETY: `has_avx10_v1_aux()` confirmed full AVX10.2 (the feature set this shim's
            // translation unit is compiled for) plus OS XSAVE state immediately above.
            return unsafe { cvt2ph_bf8_hw(src1, src2) };
        }
    }
    let _ = detect::has_avx10_v1_aux;
    cvt2ph_bf8_scalar(src1, src2)
}

/// Native path: EVEX `VCVT2PH2BF8` via `ace_native_cvt2ph_bf8` (low=src2, high=src1).
///
/// # Safety
/// The CPU must support `AVX10_V1_AUX`; callers go through [`cvt2ph_bf8`].
#[cfg(all(target_arch = "x86_64", feature = "native"))]
unsafe fn cvt2ph_bf8_hw(src1: [u16; 32], src2: [u16; 32]) -> [u8; 64] {
    let mut out = [0u8; 64];
    crate::native::ace_native_cvt2ph_bf8(src1.as_ptr(), src2.as_ptr(), out.as_mut_ptr());
    out
}

/// Portable reference oracle for [`cvt2ph_bf8`] — the primary always-correct path.
///
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.ORACLE.1]`
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT2_PH2FP8.1-1]`
pub fn cvt2ph_bf8_scalar(src1: [u16; 32], src2: [u16; 32]) -> [u8; 64] {
    cvt2_lanes(src1, src2, |bits| fp8::fp16_to_bf8(bits, false))
}

/// Two-source FP16 -> BF8 (E5M2) convert, saturating.
///
/// Like [`cvt2ph_bf8`] but an overflowing magnitude clamps to the BF8 max normal
/// `+/-57344` (`[avx10-v1-aux-fp16-fp8-evex-vnni.CVT2_PH2FP8.1]`).
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.DISPATCH.1]`
pub fn cvt2phs_bf8(src1: [u16; 32], src2: [u16; 32]) -> [u8; 64] {
    #[cfg(all(target_arch = "x86_64", feature = "native"))]
    {
        if detect::has_avx10_v1_aux() {
            // SAFETY: `has_avx10_v1_aux()` confirmed full AVX10.2 (the feature set this shim's
            // translation unit is compiled for) plus OS XSAVE state immediately above.
            return unsafe { cvt2phs_bf8_hw(src1, src2) };
        }
    }
    let _ = detect::has_avx10_v1_aux;
    cvt2phs_bf8_scalar(src1, src2)
}

/// Native path: EVEX `VCVT2PH2BF8S` (saturating) via `ace_native_cvt2phs_bf8`.
///
/// # Safety
/// The CPU must support `AVX10_V1_AUX`; callers go through [`cvt2phs_bf8`].
#[cfg(all(target_arch = "x86_64", feature = "native"))]
unsafe fn cvt2phs_bf8_hw(src1: [u16; 32], src2: [u16; 32]) -> [u8; 64] {
    let mut out = [0u8; 64];
    crate::native::ace_native_cvt2phs_bf8(src1.as_ptr(), src2.as_ptr(), out.as_mut_ptr());
    out
}

/// Portable reference oracle for [`cvt2phs_bf8`] — the primary always-correct path.
///
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.ORACLE.1]`
pub fn cvt2phs_bf8_scalar(src1: [u16; 32], src2: [u16; 32]) -> [u8; 64] {
    cvt2_lanes(src1, src2, |bits| fp8::fp16_to_bf8(bits, true))
}

/// Two-source FP16 -> HF8 (E4M3) convert, non-saturating.
///
/// Like [`cvt2ph_bf8`] but targets HF8: overflow yields the HF8 NaN encoding
/// `S.1111.111` (`[avx10-v1-aux-fp16-fp8-evex-vnni.CVT2_PH2FP8.1]`,
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT2_PH2FP8.1-1]`).
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.DISPATCH.1]`
pub fn cvt2ph_hf8(src1: [u16; 32], src2: [u16; 32]) -> [u8; 64] {
    #[cfg(all(target_arch = "x86_64", feature = "native"))]
    {
        if detect::has_avx10_v1_aux() {
            // SAFETY: `has_avx10_v1_aux()` confirmed full AVX10.2 (the feature set this shim's
            // translation unit is compiled for) plus OS XSAVE state immediately above.
            return unsafe { cvt2ph_hf8_hw(src1, src2) };
        }
    }
    let _ = detect::has_avx10_v1_aux;
    cvt2ph_hf8_scalar(src1, src2)
}

/// Native path: EVEX `VCVT2PH2HF8` via `ace_native_cvt2ph_hf8`.
///
/// # Safety
/// The CPU must support `AVX10_V1_AUX`; callers go through [`cvt2ph_hf8`].
#[cfg(all(target_arch = "x86_64", feature = "native"))]
unsafe fn cvt2ph_hf8_hw(src1: [u16; 32], src2: [u16; 32]) -> [u8; 64] {
    let mut out = [0u8; 64];
    crate::native::ace_native_cvt2ph_hf8(src1.as_ptr(), src2.as_ptr(), out.as_mut_ptr());
    out
}

/// Portable reference oracle for [`cvt2ph_hf8`] — the primary always-correct path.
///
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.ORACLE.1]`
pub fn cvt2ph_hf8_scalar(src1: [u16; 32], src2: [u16; 32]) -> [u8; 64] {
    cvt2_lanes(src1, src2, |bits| fp8::fp16_to_hf8(bits, false))
}

/// Two-source FP16 -> HF8 (E4M3) convert, saturating.
///
/// Like [`cvt2ph_hf8`] but an overflowing magnitude clamps to the HF8 max normal
/// `+/-448` (`[avx10-v1-aux-fp16-fp8-evex-vnni.CVT2_PH2FP8.1]`).
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.DISPATCH.1]`
pub fn cvt2phs_hf8(src1: [u16; 32], src2: [u16; 32]) -> [u8; 64] {
    #[cfg(all(target_arch = "x86_64", feature = "native"))]
    {
        if detect::has_avx10_v1_aux() {
            // SAFETY: `has_avx10_v1_aux()` confirmed full AVX10.2 (the feature set this shim's
            // translation unit is compiled for) plus OS XSAVE state immediately above.
            return unsafe { cvt2phs_hf8_hw(src1, src2) };
        }
    }
    let _ = detect::has_avx10_v1_aux;
    cvt2phs_hf8_scalar(src1, src2)
}

/// Native path: EVEX `VCVT2PH2HF8S` (saturating) via `ace_native_cvt2phs_hf8`.
///
/// # Safety
/// The CPU must support `AVX10_V1_AUX`; callers go through [`cvt2phs_hf8`].
#[cfg(all(target_arch = "x86_64", feature = "native"))]
unsafe fn cvt2phs_hf8_hw(src1: [u16; 32], src2: [u16; 32]) -> [u8; 64] {
    let mut out = [0u8; 64];
    crate::native::ace_native_cvt2phs_hf8(src1.as_ptr(), src2.as_ptr(), out.as_mut_ptr());
    out
}

/// Portable reference oracle for [`cvt2phs_hf8`] — the primary always-correct path.
///
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.ORACLE.1]`
pub fn cvt2phs_hf8_scalar(src1: [u16; 32], src2: [u16; 32]) -> [u8; 64] {
    cvt2_lanes(src1, src2, |bits| fp8::fp16_to_hf8(bits, true))
}

// --- Family C: biased FP16 -> FP8 converts (bias rounding) ---
//
// Per ACE v1 spec section 8.4.5 `vcvtbiasph2f8`, the value to convert for output lane `i`
// is `t = src2.fp16[i]` and the bias rounding term is `bias = src1.byte[2 * i]` — the low
// byte of the i-th `u16` element of the bias operand (`src1`). The bias is applied to the
// rounding function before conversion (spec section 2.6.3 bias rounding) and the
// saturation/overflow handling is identical to family A: non-saturating overflow emits
// the format NaN/overflow encoding, saturating clamps to the format max normal
// (`[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_BIAS_PH2FP8.1]`,
// `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_BIAS_PH2FP8.1-1]`).
//
// The public signature takes the bias operand as `bias: [u16; 32]` (mirroring the FP16
// input shape). Per the spec byte index `src1.byte[2 * i]`, lane `i` selects the *low*
// byte of `bias[i]`, i.e. `(bias[i] & 0xff) as u8` on a little-endian byte ordering of the
// `u16` array. This grounds OQ-5 directly against spec section 8.4.5.

/// Extract the per-lane bias byte for output lane `i` from the bias operand vector.
///
/// Spec section 8.4.5 defines `bias = src1.byte[2 * i]`. Byte `2 * i` of a packed `u16`
/// array is the low byte of element `i` (little-endian lane bytes), so the bias for lane
/// `i` is `(bias[i] & 0xff) as u8`. `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_BIAS_PH2FP8.1]`
fn bias_byte(bias: [u16; 32], i: usize) -> u8 {
    (bias[i] & 0x00ff) as u8
}

/// Biased FP16 -> BF8 (E5M2) convert, non-saturating.
///
/// Converts `a` to BF8 applying the per-lane bias rounding term from `bias`
/// (`bias = bias[i].byte[0]`, spec section 8.4.5), added at the mantissa lsb and then
/// truncated per the section-16.2 SR pseudocode. NOTE: a zero bias TRUNCATES — it is NOT
/// [`cvtph_bf8`]'s RTNE (they agree only when no discarded bits round up). Overflow yields
/// the BF8 overflow encoding `S.11111.00`.
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_BIAS_PH2FP8.1]`
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_BIAS_PH2FP8.1-1]`
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.DISPATCH.1]`
pub fn cvtbiasph_bf8(a: [u16; 32], bias: [u16; 32]) -> [u8; 32] {
    #[cfg(all(target_arch = "x86_64", feature = "native"))]
    {
        if detect::has_avx10_v1_aux() {
            // SAFETY: `has_avx10_v1_aux()` confirmed full AVX10.2 (the feature set this shim's
            // translation unit is compiled for) plus OS XSAVE state immediately above.
            return unsafe { cvtbiasph_bf8_hw(a, bias) };
        }
    }
    let _ = detect::has_avx10_v1_aux;
    cvtbiasph_bf8_scalar(a, bias)
}

/// Native path: EVEX `VCVTBIASPH2BF8` via `ace_native_cvtbiasph_bf8` (bias = bias.byte[2*i]).
///
/// # Safety
/// The CPU must support `AVX10_V1_AUX`; callers go through [`cvtbiasph_bf8`].
#[cfg(all(target_arch = "x86_64", feature = "native"))]
unsafe fn cvtbiasph_bf8_hw(a: [u16; 32], bias: [u16; 32]) -> [u8; 32] {
    let mut out = [0u8; 32];
    crate::native::ace_native_cvtbiasph_bf8(a.as_ptr(), bias.as_ptr(), out.as_mut_ptr());
    out
}

/// Portable reference oracle for [`cvtbiasph_bf8`] — the primary always-correct path.
///
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.ORACLE.1]`
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_BIAS_PH2FP8.1]`
pub fn cvtbiasph_bf8_scalar(a: [u16; 32], bias: [u16; 32]) -> [u8; 32] {
    core::array::from_fn(|i| fp8::fp16_to_bf8_biased(a[i], bias_byte(bias, i), false))
}

/// Biased FP16 -> BF8 (E5M2) convert, saturating.
///
/// Like [`cvtbiasph_bf8`] but an overflowing magnitude clamps to the BF8 max normal
/// `+/-57344` (`[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_BIAS_PH2FP8.1-1]`).
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_BIAS_PH2FP8.1]`
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.DISPATCH.1]`
pub fn cvtbiasphs_bf8(a: [u16; 32], bias: [u16; 32]) -> [u8; 32] {
    #[cfg(all(target_arch = "x86_64", feature = "native"))]
    {
        if detect::has_avx10_v1_aux() {
            // SAFETY: `has_avx10_v1_aux()` confirmed full AVX10.2 (the feature set this shim's
            // translation unit is compiled for) plus OS XSAVE state immediately above.
            return unsafe { cvtbiasphs_bf8_hw(a, bias) };
        }
    }
    let _ = detect::has_avx10_v1_aux;
    cvtbiasphs_bf8_scalar(a, bias)
}

/// Native path: EVEX `VCVTBIASPH2BF8S` (saturating) via `ace_native_cvtbiasphs_bf8`.
///
/// # Safety
/// The CPU must support `AVX10_V1_AUX`; callers go through [`cvtbiasphs_bf8`].
#[cfg(all(target_arch = "x86_64", feature = "native"))]
unsafe fn cvtbiasphs_bf8_hw(a: [u16; 32], bias: [u16; 32]) -> [u8; 32] {
    let mut out = [0u8; 32];
    crate::native::ace_native_cvtbiasphs_bf8(a.as_ptr(), bias.as_ptr(), out.as_mut_ptr());
    out
}

/// Portable reference oracle for [`cvtbiasphs_bf8`] — the primary always-correct path.
///
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.ORACLE.1]`
pub fn cvtbiasphs_bf8_scalar(a: [u16; 32], bias: [u16; 32]) -> [u8; 32] {
    core::array::from_fn(|i| fp8::fp16_to_bf8_biased(a[i], bias_byte(bias, i), true))
}

/// Biased FP16 -> HF8 (E4M3) convert, non-saturating.
///
/// Like [`cvtbiasph_bf8`] but targets HF8: the bias is applied `>> 1` (E4M3 discards one
/// fewer bit, spec section 16.2), FP16 subnormal inputs flush to signed zero, and overflow
/// yields the HF8 NaN encoding `S.1111.111`. A zero bias TRUNCATES — it is NOT
/// [`cvtph_hf8`]'s RTNE.
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_BIAS_PH2FP8.1]`
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_BIAS_PH2FP8.1-1]`
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.DISPATCH.1]`
pub fn cvtbiasph_hf8(a: [u16; 32], bias: [u16; 32]) -> [u8; 32] {
    #[cfg(all(target_arch = "x86_64", feature = "native"))]
    {
        if detect::has_avx10_v1_aux() {
            // SAFETY: `has_avx10_v1_aux()` confirmed full AVX10.2 (the feature set this shim's
            // translation unit is compiled for) plus OS XSAVE state immediately above.
            return unsafe { cvtbiasph_hf8_hw(a, bias) };
        }
    }
    let _ = detect::has_avx10_v1_aux;
    cvtbiasph_hf8_scalar(a, bias)
}

/// Native path: EVEX `VCVTBIASPH2HF8` via `ace_native_cvtbiasph_hf8`.
///
/// # Safety
/// The CPU must support `AVX10_V1_AUX`; callers go through [`cvtbiasph_hf8`].
#[cfg(all(target_arch = "x86_64", feature = "native"))]
unsafe fn cvtbiasph_hf8_hw(a: [u16; 32], bias: [u16; 32]) -> [u8; 32] {
    let mut out = [0u8; 32];
    crate::native::ace_native_cvtbiasph_hf8(a.as_ptr(), bias.as_ptr(), out.as_mut_ptr());
    out
}

/// Portable reference oracle for [`cvtbiasph_hf8`] — the primary always-correct path.
///
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.ORACLE.1]`
pub fn cvtbiasph_hf8_scalar(a: [u16; 32], bias: [u16; 32]) -> [u8; 32] {
    core::array::from_fn(|i| fp8::fp16_to_hf8_biased(a[i], bias_byte(bias, i), false))
}

/// Biased FP16 -> HF8 (E4M3) convert, saturating.
///
/// Like [`cvtbiasph_hf8`] but an overflowing magnitude clamps to the HF8 max normal
/// `+/-448` (`[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_BIAS_PH2FP8.1-1]`).
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_BIAS_PH2FP8.1]`
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.DISPATCH.1]`
pub fn cvtbiasphs_hf8(a: [u16; 32], bias: [u16; 32]) -> [u8; 32] {
    #[cfg(all(target_arch = "x86_64", feature = "native"))]
    {
        if detect::has_avx10_v1_aux() {
            // SAFETY: `has_avx10_v1_aux()` confirmed full AVX10.2 (the feature set this shim's
            // translation unit is compiled for) plus OS XSAVE state immediately above.
            return unsafe { cvtbiasphs_hf8_hw(a, bias) };
        }
    }
    let _ = detect::has_avx10_v1_aux;
    cvtbiasphs_hf8_scalar(a, bias)
}

/// Native path: EVEX `VCVTBIASPH2HF8S` (saturating) via `ace_native_cvtbiasphs_hf8`.
///
/// # Safety
/// The CPU must support `AVX10_V1_AUX`; callers go through [`cvtbiasphs_hf8`].
#[cfg(all(target_arch = "x86_64", feature = "native"))]
unsafe fn cvtbiasphs_hf8_hw(a: [u16; 32], bias: [u16; 32]) -> [u8; 32] {
    let mut out = [0u8; 32];
    crate::native::ace_native_cvtbiasphs_hf8(a.as_ptr(), bias.as_ptr(), out.as_mut_ptr());
    out
}

/// Portable reference oracle for [`cvtbiasphs_hf8`] — the primary always-correct path.
///
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.ORACLE.1]`
pub fn cvtbiasphs_hf8_scalar(a: [u16; 32], bias: [u16; 32]) -> [u8; 32] {
    core::array::from_fn(|i| fp8::fp16_to_hf8_biased(a[i], bias_byte(bias, i), true))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fp16_bits(sign: u16, exp: u16, mant: u16) -> u16 {
        (sign << 15) | (exp << 10) | mant
    }

    // BF8 (E5M2) byte assembler: sign | 5-bit exp field | 2-bit mantissa.
    fn bf8(sign: u8, exp: u8, mant: u8) -> u8 {
        (sign << 7) | (exp << 2) | mant
    }

    // HF8 (E4M3) byte assembler: sign | 4-bit exp field | 3-bit mantissa.
    fn hf8(sign: u8, exp: u8, mant: u8) -> u8 {
        (sign << 7) | (exp << 3) | mant
    }

    /// Hand-computed known-value vector pinning chosen FP16 inputs to their BF8 bytes,
    /// independent of the implementation. Covers a normal value, signed zero, a
    /// subnormal near BF8 min (+/-2^-16), and a non-saturating overflow lane.
    /// `[avx10-v1-aux-fp16-fp8-evex-vnni.PROPERTIES.4]`
    #[test]
    fn cvtph_bf8_known_values() {
        let mut a = [0u16; 32];
        // lane 0: 1.0 -> BF8 1.0 = S.01111.00.
        a[0] = fp16_bits(0, 15, 0);
        // lane 1: -0.0 -> BF8 -0.0 = 0x80.
        a[1] = fp16_bits(1, 0, 0);
        // lane 2: +2^-16 (FP16 subnormal mant=256) -> BF8 min subnormal S.00000.01.
        a[2] = fp16_bits(0, 0, 256);
        // lane 3: +Inf -> non-saturating BF8 +Inf encoding S.11111.00 (E5M2 has Inf; the
        // section-2.4.1 NaN set is S.11111.{01,10,11}). Hardware-matched (verified under SDE).
        a[3] = fp16_bits(0, 31, 0);

        let out = cvtph_bf8(a);
        assert_eq!(out[0], bf8(0, 0b01111, 0b00), "1.0 lane");
        assert_eq!(out[1], 0x80, "signed-zero lane");
        assert_eq!(out[2], bf8(0, 0b00000, 0b01), "subnormal lane");
        // Overflow lane: BF8 +Inf (exp all-ones, mantissa zero), and crucially NOT the
        // saturating max-normal byte bf8(0, 0b11110, 0b11).
        assert_eq!(out[3], bf8(0, 0b11111, 0b00), "overflow -> +Inf");
        assert_ne!(
            out[3],
            bf8(0, 0b11110, 0b11),
            "non-saturating must not clamp to max normal"
        );

        // Untouched lanes (FP16 +0) map to BF8 +0.
        assert_eq!(out[4], 0x00);
    }

    /// Hand-computed known-value vector for the HF8 (E4M3) target: a normal, a value at
    /// the HF8 min subnormal +/-2^-9, and a max-exponent normal that must NOT be mistaken
    /// for NaN. `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_PH2FP8.1-2]`
    #[test]
    fn cvtph_hf8_known_values() {
        let mut a = [0u16; 32];
        // lane 0: 1.0 -> HF8 1.0 = S.0111.000.
        a[0] = fp16_bits(0, 15, 0);
        // lane 1: -0.0 -> HF8 -0.0 = 0x80.
        a[1] = fp16_bits(1, 0, 0);
        // lane 2: 2^-9 (HF8 min subnormal) -> S.0000.001.
        a[2] = fp16_bits(0, 6, 0);
        // lane 3: 1.5 -> S.0111.100.
        a[3] = fp16_bits(0, 15, 0b10_0000_0000);

        let out = cvtph_hf8(a);
        assert_eq!(out[0], hf8(0, 0b0111, 0b000), "1.0 lane");
        assert_eq!(out[1], 0x80, "signed-zero lane");
        assert_eq!(out[2], hf8(0, 0b0000, 0b001), "min-subnormal lane");
        assert_eq!(out[3], hf8(0, 0b0111, 0b100), "1.5 lane");
        assert_eq!(out[4], 0x00, "untouched +0 lane");
    }

    /// Saturating clamp lanes: a BF8 input above 57344 clamps to +/-57344, an HF8 input
    /// above 448 clamps to +/-448; the non-saturating sibling instead emits the format
    /// NaN encoding. This distinguishes saturating from non-saturating mode (the
    /// `Saturating (S-suffix) converts clamp an overflowing magnitude to the format max
    /// normal (BF8 +/-57344, HF8 +/-448); non-saturating converts produce the format's
    /// NaN/overflow encoding` invariant).
    /// `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_PH2FP8.1-2]`
    #[test]
    fn family_a_saturating_clamp_known_values() {
        // FP16 65504 (max normal) overflows both BF8 (>57344) and HF8 (>448).
        let big_pos = fp16_bits(0, 30, 0x3ff);
        let big_neg = fp16_bits(1, 30, 0x3ff);

        let mut a = [0u16; 32];
        a[0] = big_pos;
        a[1] = big_neg;

        // BF8 saturating clamps to +/-57344 = S.11110.11.
        let bf8_sat = cvtphs_bf8(a);
        assert_eq!(bf8_sat[0], bf8(0, 0b11110, 0b11), "+BF8 max normal");
        assert_eq!(bf8_sat[1], bf8(1, 0b11110, 0b11), "-BF8 max normal");
        // BF8 non-saturating emits +/-Inf S.11111.00 (E5M2 has Inf), NOT the max-normal clamp.
        // Hardware-matched (verified under SDE).
        let bf8_nsat = cvtph_bf8(a);
        assert_ne!(bf8_nsat[0], bf8_sat[0], "non-sat differs from sat (BF8)");
        assert_eq!(bf8_nsat[0], bf8(0, 0b11111, 0b00), "+BF8 overflow -> +Inf");
        assert_eq!(bf8_nsat[1], bf8(1, 0b11111, 0b00), "-BF8 overflow -> -Inf");

        // HF8 saturating clamps to +/-448 = S.1111.110.
        let hf8_sat = cvtphs_hf8(a);
        assert_eq!(hf8_sat[0], hf8(0, 0b1111, 0b110), "+HF8 max normal");
        assert_eq!(hf8_sat[1], hf8(1, 0b1111, 0b110), "-HF8 max normal");
        // HF8 non-saturating emits NaN S.1111.111, NOT the clamp.
        let hf8_nsat = cvtph_hf8(a);
        assert_eq!(hf8_nsat[0], hf8(0, 0b1111, 0b111), "HF8 NaN encoding");
        assert_ne!(hf8_nsat[0], hf8_sat[0], "non-sat differs from sat (HF8)");
    }

    /// The public dispatcher equals its scalar oracle on a representative vector.
    /// `[avx10-v1-aux-fp16-fp8-evex-vnni.ORACLE.2]`
    #[test]
    fn cvtph_bf8_dispatch_matches_oracle() {
        let a: [u16; 32] = core::array::from_fn(|i| (i as u16).wrapping_mul(0x0411));
        assert_eq!(cvtph_bf8(a), cvtph_bf8_scalar(a));
        assert_eq!(cvtphs_bf8(a), cvtphs_bf8_scalar(a));
        assert_eq!(cvtph_hf8(a), cvtph_hf8_scalar(a));
        assert_eq!(cvtphs_hf8(a), cvtphs_hf8_scalar(a));
    }

    /// Family-B lane ordering: the low half [0..32) of the 64-lane output must equal the
    /// single-source conversion of `src2`, and the high half [32..64) must equal the
    /// single-source conversion of `src1` (the `Two-source converts (families B and E)
    /// place src2-derived results in the low half of the output (lanes [0..KL/2)) and
    /// src1-derived results in the high half (lanes [KL/2..KL))` invariant, spec
    /// section 8.2.5). The test uses distinct src1/src2 vectors so a swapped-halves
    /// implementation would fail.
    /// `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT2_PH2FP8.1]`
    /// `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT2_PH2FP8.1-1]`
    #[test]
    fn cvt2ph_bf8_lane_ordering_known_values() {
        // src1 lanes: ascending 1.0 * 2^k pattern (exp 15 .. ), all distinct.
        let src1: [u16; 32] = core::array::from_fn(|i| fp16_bits(0, 15, (i as u16) * 16));
        // src2 lanes: a clearly different pattern (negative signs, different mantissas).
        let src2: [u16; 32] = core::array::from_fn(|i| fp16_bits(1, 14, (i as u16) * 8 + 1));

        // Concrete spec anchors for the boundary lanes (distinguish src2 vs src1 halves):
        // lane 0 (low half) = convert(src2[0]); src2[0] = -1.0 * (1 + 1/1024) at exp 14
        // rounds to BF8 -0.5 binade; lane 32 (high half) = convert(src1[0]) = +1.0.
        let out = cvt2ph_bf8(src1, src2);

        // lane 32 is the first high-half lane = src1[0] = +1.0 -> BF8 S.01111.00.
        assert_eq!(out[32], bf8(0, 0b01111, 0b00), "high[0] == src1[0]==1.0");
        // lane 0 is src2[0], which is negative (sign bit set in BF8).
        assert_eq!(out[0] >> 7, 1, "low[0] sign == src2[0] sign (negative)");

        // Full structural check: low half == cvtph_bf8(src2), high half == cvtph_bf8(src1).
        let lo = cvtph_bf8(src2);
        let hi = cvtph_bf8(src1);
        for i in 0..32 {
            assert_eq!(out[i], lo[i], "low lane {i} must be src2 conversion");
            assert_eq!(out[32 + i], hi[i], "high lane {i} must be src1 conversion");
        }
    }

    /// Family-C bias rounding known values (spec sections 2.6.3 + 8.4.5).
    ///
    /// Grounds OQ-5 two ways: (1) a zero-bias vector reproduces the plain family-A result
    /// bit-for-bit; (2) a nonzero bias byte shifts the rounded byte upward exactly as the
    /// section-8.4.5 bias term dictates, on a value that plain RTNE rounds DOWN. This is
    /// DISCRIMINATING: a model that ignores the bias, or one that adds the bias to the
    /// wrong (high) byte of the operand, would leave lane 0 unchanged here.
    /// `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_BIAS_PH2FP8.1]`
    /// `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_BIAS_PH2FP8.1-1]`
    #[test]
    fn cvtbias_bf8_known_values() {
        // Value that plain RTNE rounds DOWN to BF8 1.0 (mant 0b00): FP16 1.0 + 1/1024 lsb.
        let near_one = fp16_bits(0, 15, 0b00_0000_0001);
        let a: [u16; 32] = [near_one; 32];

        // (1) zero bias TRUNCATES (section-16.2 SR); it agrees with plain family A here
        // only because this value rounds DOWN under RTNE too (discarded bits below half).
        let zero_bias = [0u16; 32];
        assert_eq!(
            cvtbiasph_bf8(a, zero_bias),
            cvtph_bf8(a),
            "zero bias agrees with family A on a round-down value"
        );
        assert_eq!(
            cvtbiasphs_bf8(a, zero_bias),
            cvtphs_bf8(a),
            "zero bias agrees with saturating family A on a round-down value"
        );
        assert_eq!(
            cvtbiasph_bf8(a, zero_bias)[0] & 0b11,
            0b00,
            "lane0 mant 0b00"
        );

        // (2) bias byte 0xff in the LOW byte of each bias[i] (src1.byte[2*i]) pushes the
        // round up one lsb: mant 0b00 -> 0b01.
        let hi_bias = [0x00ffu16; 32];
        let biased = cvtbiasph_bf8(a, hi_bias);
        assert_eq!(
            biased[0] & 0b11,
            0b01,
            "low-byte bias 0xff rounds up one lsb"
        );

        // The bias is the LOW byte (section 8.4.5 `byte[2*i]`): putting 0xff in the HIGH
        // byte instead (0xff00) must select byte 0x00 and leave the result unchanged.
        // This discriminates the low-byte layout from a high-byte misreading.
        let wrong_byte = [0xff00u16; 32];
        assert_eq!(
            cvtbiasph_bf8(a, wrong_byte)[0] & 0b11,
            0b00,
            "high-byte bias is NOT selected; low byte 0x00 leaves RTNE result"
        );
    }

    /// The biased dispatchers equal their scalar oracles on a representative vector.
    /// `[avx10-v1-aux-fp16-fp8-evex-vnni.ORACLE.2]`
    /// `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_BIAS_PH2FP8.1]`
    #[test]
    fn cvtbias_dispatch_matches_oracle() {
        let a: [u16; 32] = core::array::from_fn(|i| (i as u16).wrapping_mul(0x0411));
        let bias: [u16; 32] = core::array::from_fn(|i| (i as u16).wrapping_mul(0x0137));
        assert_eq!(cvtbiasph_bf8(a, bias), cvtbiasph_bf8_scalar(a, bias));
        assert_eq!(cvtbiasphs_bf8(a, bias), cvtbiasphs_bf8_scalar(a, bias));
        assert_eq!(cvtbiasph_hf8(a, bias), cvtbiasph_hf8_scalar(a, bias));
        assert_eq!(cvtbiasphs_hf8(a, bias), cvtbiasphs_hf8_scalar(a, bias));
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    // `detect` is only referenced by the native-vs-oracle differentials below, which are
    // themselves `#[cfg(all(target_arch = "x86_64", feature = "native"))]`. Gate the import
    // to match, so the default (oracle-only) build has no unused import.
    #[cfg(all(target_arch = "x86_64", feature = "native"))]
    use crate::detect;
    use quickcheck::{quickcheck, Arbitrary, Gen, TestResult};

    /// A randomly-sampled FP16 input vector. The raw `u16` lanes cover the full format
    /// domain including subnormals, signed zeros, NaNs, and overflow patterns.
    #[derive(Clone, Debug)]
    struct Inputs {
        a: [u16; 32],
    }

    impl Arbitrary for Inputs {
        fn arbitrary(g: &mut Gen) -> Self {
            Inputs {
                a: core::array::from_fn(|_| u16::arbitrary(g)),
            }
        }
    }

    /// A randomly-sampled pair of FP16 input vectors for the two-source (family-B)
    /// converters. Independent per-lane sampling covers subnormals, signed zeros, NaNs,
    /// and overflow patterns in both sources.
    #[derive(Clone, Debug)]
    struct PairInputs {
        src1: [u16; 32],
        src2: [u16; 32],
    }

    impl Arbitrary for PairInputs {
        fn arbitrary(g: &mut Gen) -> Self {
            PairInputs {
                src1: core::array::from_fn(|_| u16::arbitrary(g)),
                src2: core::array::from_fn(|_| u16::arbitrary(g)),
            }
        }
    }

    /// A randomly-sampled (FP16 input, bias operand) pair for the family-C biased
    /// converters. Both vectors are sampled independently per lane so the bias bytes span
    /// the full 0..=255 range against the full FP16 input domain.
    #[derive(Clone, Debug)]
    struct BiasInputs {
        a: [u16; 32],
        bias: [u16; 32],
    }

    impl Arbitrary for BiasInputs {
        fn arbitrary(g: &mut Gen) -> Self {
            BiasInputs {
                a: core::array::from_fn(|_| u16::arbitrary(g)),
                bias: core::array::from_fn(|_| u16::arbitrary(g)),
            }
        }
    }

    // BF8 max-normal magnitude = 57344; HF8 max-normal magnitude = 448. Any finite
    // (non-NaN) byte the saturating converters emit must have a magnitude not exceeding
    // these, since saturation clamps overflow to the max normal.
    fn bf8_is_nan(byte: u8) -> bool {
        ((byte >> 2) & 0x1f) == 0b11111 && (byte & 0b11) != 0
    }

    // E5M2 (BF8) +/-Inf is S.11111.00 (exp all-ones, mantissa zero). Non-saturating overflow
    // emits this (hardware-matched, verified under SDE); it is not a finite magnitude.
    fn bf8_is_inf(byte: u8) -> bool {
        ((byte >> 2) & 0x1f) == 0b11111 && (byte & 0b11) == 0
    }

    // NaN or Inf: the non-finite BF8 bytes excluded from the finite-magnitude bound.
    fn bf8_is_special(byte: u8) -> bool {
        bf8_is_nan(byte) || bf8_is_inf(byte)
    }

    fn hf8_is_nan(byte: u8) -> bool {
        (byte & 0x7f) == 0x7f
    }

    // Decode a BF8 (E5M2) byte to its real magnitude (NaN/zero -> 0.0).
    fn bf8_magnitude(byte: u8) -> f64 {
        let exp = ((byte >> 2) & 0x1f) as i32;
        let mant = (byte & 0b11) as f64;
        if exp == 0 {
            mant * 2f64.powi(-16) // subnormal: mant * 2^(1-15-2)
        } else {
            (1.0 + mant / 4.0) * 2f64.powi(exp - 15)
        }
    }

    // Decode an HF8 (E4M3) byte to its real magnitude (NaN/zero -> 0.0).
    fn hf8_magnitude(byte: u8) -> f64 {
        let exp = ((byte >> 3) & 0x0f) as i32;
        let mant = (byte & 0b111) as f64;
        if exp == 0 {
            mant * 2f64.powi(-9) // subnormal: mant * 2^(1-7-3)
        } else {
            (1.0 + mant / 8.0) * 2f64.powi(exp - 7)
        }
    }

    quickcheck! {
        /// The public dispatcher must equal the scalar oracle on every sampled input,
        /// for all four family-A converters.
        /// `[avx10-v1-aux-fp16-fp8-evex-vnni.ORACLE.2]`
        fn prop_public_matches_scalar(input: Inputs) -> bool {
            cvtph_bf8(input.a) == cvtph_bf8_scalar(input.a)
                && cvtphs_bf8(input.a) == cvtphs_bf8_scalar(input.a)
                && cvtph_hf8(input.a) == cvtph_hf8_scalar(input.a)
                && cvtphs_hf8(input.a) == cvtphs_hf8_scalar(input.a)
        }

        /// Saturating-bounds invariant: a saturating convert never produces a byte whose
        /// magnitude exceeds the format max normal (BF8 +/-57344, HF8 +/-448). NaN bytes
        /// (only reachable from NaN inputs, never from overflow under saturation) are
        /// excluded. `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_PH2FP8.1-2]`
        fn prop_saturating_within_max_normal(input: Inputs) -> bool {
            let bf8 = cvtphs_bf8(input.a);
            let hf8 = cvtphs_hf8(input.a);
            bf8.iter().all(|&b| bf8_is_special(b) || bf8_magnitude(b) <= 57344.0)
                && hf8.iter().all(|&b| hf8_is_nan(b) || hf8_magnitude(b) <= 448.0)
        }

        /// Family-B: each public two-source dispatcher equals its scalar oracle on every
        /// sampled (src1, src2) pair.
        /// `[avx10-v1-aux-fp16-fp8-evex-vnni.ORACLE.2]`
        /// `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT2_PH2FP8.1]`
        fn prop_two_source_public_matches_scalar(input: PairInputs) -> bool {
            cvt2ph_bf8(input.src1, input.src2) == cvt2ph_bf8_scalar(input.src1, input.src2)
                && cvt2phs_bf8(input.src1, input.src2) == cvt2phs_bf8_scalar(input.src1, input.src2)
                && cvt2ph_hf8(input.src1, input.src2) == cvt2ph_hf8_scalar(input.src1, input.src2)
                && cvt2phs_hf8(input.src1, input.src2) == cvt2phs_hf8_scalar(input.src1, input.src2)
        }

        /// Family-B lane-ordering equivalence: the low half of every two-source convert
        /// equals the corresponding single-source family-A convert applied to `src2`, and
        /// the high half equals it applied to `src1` (spec section 8.2.5 src2->low /
        /// src1->high). Asserted for all four format/saturation combinations.
        /// `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT2_PH2FP8.1-1]`
        fn prop_two_source_lane_ordering(input: PairInputs) -> bool {
            let (s1, s2) = (input.src1, input.src2);
            let check = |out: [u8; 64], lo: [u8; 32], hi: [u8; 32]| -> bool {
                (0..32).all(|i| out[i] == lo[i] && out[32 + i] == hi[i])
            };
            check(cvt2ph_bf8(s1, s2), cvtph_bf8(s2), cvtph_bf8(s1))
                && check(cvt2phs_bf8(s1, s2), cvtphs_bf8(s2), cvtphs_bf8(s1))
                && check(cvt2ph_hf8(s1, s2), cvtph_hf8(s2), cvtph_hf8(s1))
                && check(cvt2phs_hf8(s1, s2), cvtphs_hf8(s2), cvtphs_hf8(s1))
        }

        /// Family-C: each public biased dispatcher equals its scalar oracle on every
        /// sampled (input, bias) pair.
        /// `[avx10-v1-aux-fp16-fp8-evex-vnni.ORACLE.2]`
        /// `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_BIAS_PH2FP8.1]`
        fn prop_bias_public_matches_scalar(input: BiasInputs) -> bool {
            cvtbiasph_bf8(input.a, input.bias) == cvtbiasph_bf8_scalar(input.a, input.bias)
                && cvtbiasphs_bf8(input.a, input.bias) == cvtbiasphs_bf8_scalar(input.a, input.bias)
                && cvtbiasph_hf8(input.a, input.bias) == cvtbiasph_hf8_scalar(input.a, input.bias)
                && cvtbiasphs_hf8(input.a, input.bias) == cvtbiasphs_hf8_scalar(input.a, input.bias)
        }

        /// Family-C half-bias round-to-nearest equivalence: a biased convert with bias 0x80
        /// (half an lsb) in every lane equals the corresponding plain family-A RTNE convert on
        /// all NON-tie inputs. Bias rounding adds the bias then truncates, so adding exactly
        /// half recovers round-to-nearest; the two differ only on exact ties (RTNE ties to
        /// even, bias rounding ties up), which this property excludes per lane. This pins the
        /// hardware-grounded bias-rounding semantics (spec section 2.6.3), replacing the
        /// earlier (incorrect) "bias 0 == plain RTNE" assumption.
        /// `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_BIAS_PH2FP8.1]`
        /// `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_BIAS_PH2FP8.1-1]`
        fn prop_half_bias_is_round_to_nearest(input: Inputs) -> bool {
            let half = [0x0080u16; 32];
            let bf8_b = cvtbiasph_bf8(input.a, half);
            let bf8_a = cvtph_bf8(input.a);
            let hf8_b = cvtbiasph_hf8(input.a, half);
            let hf8_a = cvtph_hf8(input.a);
            // Exclude lanes where RTNE could tie-to-even-DOWN while bias rounding ties UP; a
            // robust, payload-agnostic check: a 1-lsb difference between the two is allowed
            // (it can only be a tie), any larger difference is a real disagreement.
            let close = |x: u8, y: u8| x == y || x.wrapping_sub(y) == 1 || y.wrapping_sub(x) == 1;
            (0..32).all(|i| close(bf8_b[i], bf8_a[i]) && close(hf8_b[i], hf8_a[i]))
        }

        /// Family-C bias selects the LOW byte of each `u16` bias lane (spec section 8.4.5
        /// `src1.byte[2*i]`): a bias operand with arbitrary HIGH bytes but zero LOW bytes must
        /// produce the same result as a zero-bias (truncating) convert. This rules out a
        /// high-byte (`byte[2*i+1]`) misreading of the bias layout.
        /// `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_BIAS_PH2FP8.1]`
        fn prop_bias_uses_low_byte_only(input: BiasInputs) -> bool {
            // Keep only the high byte of each bias lane (low byte forced to 0).
            let high_only: [u16; 32] = core::array::from_fn(|i| input.bias[i] & 0xff00);
            let zero = [0u16; 32];
            cvtbiasph_bf8(input.a, high_only) == cvtbiasph_bf8(input.a, zero)
                && cvtbiasph_hf8(input.a, high_only) == cvtbiasph_hf8(input.a, zero)
        }

        /// Family-C saturating-bounds invariant carries over from family A: a saturating
        /// biased convert never produces a byte exceeding the format max normal (NaN bytes
        /// excluded). `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_BIAS_PH2FP8.1-1]`
        fn prop_bias_saturating_within_max_normal(input: BiasInputs) -> bool {
            let bf8 = cvtbiasphs_bf8(input.a, input.bias);
            let hf8 = cvtbiasphs_hf8(input.a, input.bias);
            bf8.iter().all(|&b| bf8_is_special(b) || bf8_magnitude(b) <= 57344.0)
                && hf8.iter().all(|&b| hf8_is_nan(b) || hf8_magnitude(b) <= 448.0)
        }

        /// Cross-cutting saturating-bounds-vs-non-saturating relation (families A and C):
        /// where the non-saturating convert produces an in-range FINITE byte (not the format
        /// NaN/overflow encoding), the saturating convert produces exactly the same byte; and
        /// every finite saturating byte is magnitude-bounded by the format max normal. This is
        /// the "saturating bounds non-saturating" property — saturation only ever clamps the
        /// overflow cases the non-saturating form turns into NaN/overflow, never disturbing the
        /// representable results (`[avx10-v1-aux-fp16-fp8-evex-vnni.PROPERTIES.3]`,
        /// `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_PH2FP8.1-2]`).
        fn prop_saturating_bounds_non_saturating(input: Inputs) -> bool {
            let ns_bf8 = cvtph_bf8(input.a);
            let sat_bf8 = cvtphs_bf8(input.a);
            let ns_hf8 = cvtph_hf8(input.a);
            let sat_hf8 = cvtphs_hf8(input.a);
            let bf8_ok = (0..32).all(|i| {
                // Where the non-saturating byte is finite, saturation leaves it unchanged (it
                // only clamps the overflow cases the non-saturating form encodes as NaN/Inf —
                // E5M2 overflow is the Inf encoding S.11111.00); and every finite saturating
                // byte is bounded by the BF8 max normal.
                (bf8_is_special(ns_bf8[i]) || sat_bf8[i] == ns_bf8[i])
                    && (bf8_is_special(sat_bf8[i]) || bf8_magnitude(sat_bf8[i]) <= 57344.0)
            });
            let hf8_ok = (0..32).all(|i| {
                (hf8_is_nan(ns_hf8[i]) || sat_hf8[i] == ns_hf8[i])
                    && (hf8_is_nan(sat_hf8[i]) || hf8_magnitude(sat_hf8[i]) <= 448.0)
            });
            bf8_ok && hf8_ok
        }

        /// Family-A native-vs-oracle differential. Under `feature="native"` on x86_64 with
        /// `AVX10_V1_AUX` detected, the real EVEX `VCVTPH2BF8`/`HF8`(`S`) path must agree with
        /// the scalar oracle bit-for-bit over all four converters
        /// (`[avx10-v1-aux-fp16-fp8-evex-vnni.DIFFERENTIAL.1]`,
        /// `[avx10-v1-aux-fp16-fp8-evex-vnni.DIFFERENTIAL.1-1]`). When the native feature or
        /// detection is absent the case is *discarded* (never `from_bool(false)`), so a
        /// fallback-only runner cannot produce a vacuous green
        /// (`[avx10-v1-aux-fp16-fp8-evex-vnni.DIFFERENTIAL.2]`).
        fn prop_native_matches_oracle(input: Inputs) -> TestResult {
            #[cfg(all(target_arch = "x86_64", feature = "native"))]
            {
                if detect::has_avx10_v1_aux() {
                    let a = input.a;
                    let ok = cvtph_bf8(a) == cvtph_bf8_scalar(a)
                        && cvtphs_bf8(a) == cvtphs_bf8_scalar(a)
                        && cvtph_hf8(a) == cvtph_hf8_scalar(a)
                        && cvtphs_hf8(a) == cvtphs_hf8_scalar(a);
                    return TestResult::from_bool(ok);
                }
            }
            let _ = &input;
            TestResult::discard()
        }

        /// Family-B native-vs-oracle differential (two-source converters), same contract as
        /// the family-A version: real EVEX `VCVT2PH2*` vs the oracle, discarded when no native
        /// path is present (`[avx10-v1-aux-fp16-fp8-evex-vnni.DIFFERENTIAL.1]`).
        fn prop_native_matches_oracle_two_source(input: PairInputs) -> TestResult {
            #[cfg(all(target_arch = "x86_64", feature = "native"))]
            {
                if detect::has_avx10_v1_aux() {
                    let (s1, s2) = (input.src1, input.src2);
                    let ok = cvt2ph_bf8(s1, s2) == cvt2ph_bf8_scalar(s1, s2)
                        && cvt2phs_bf8(s1, s2) == cvt2phs_bf8_scalar(s1, s2)
                        && cvt2ph_hf8(s1, s2) == cvt2ph_hf8_scalar(s1, s2)
                        && cvt2phs_hf8(s1, s2) == cvt2phs_hf8_scalar(s1, s2);
                    return TestResult::from_bool(ok);
                }
            }
            let _ = &input;
            TestResult::discard()
        }

        /// Family-C native-vs-oracle differential (biased converters), same contract: real EVEX
        /// `VCVTBIASPH2*` vs the oracle, discarded when no native path is present
        /// (`[avx10-v1-aux-fp16-fp8-evex-vnni.DIFFERENTIAL.1]`).
        fn prop_native_matches_oracle_bias(input: BiasInputs) -> TestResult {
            #[cfg(all(target_arch = "x86_64", feature = "native"))]
            {
                if detect::has_avx10_v1_aux() {
                    let (a, bias) = (input.a, input.bias);
                    let ok = cvtbiasph_bf8(a, bias) == cvtbiasph_bf8_scalar(a, bias)
                        && cvtbiasphs_bf8(a, bias) == cvtbiasphs_bf8_scalar(a, bias)
                        && cvtbiasph_hf8(a, bias) == cvtbiasph_hf8_scalar(a, bias)
                        && cvtbiasphs_hf8(a, bias) == cvtbiasphs_hf8_scalar(a, bias);
                    return TestResult::from_bool(ok);
                }
            }
            let _ = &input;
            TestResult::discard()
        }
    }

    /// Hand-value native-vs-oracle differential for families A/B/C. Runs only under
    /// `feature="native"` with `AVX10_V1_AUX` detected; otherwise it is a silent no-op (the
    /// property tests above carry the broad coverage). Pins concrete vectors so a single
    /// failing primitive is named directly. `[avx10-v1-aux-fp16-fp8-evex-vnni.DIFFERENTIAL.1-1]`
    #[cfg(all(target_arch = "x86_64", feature = "native"))]
    #[test]
    fn hand_value_native_matches_oracle() {
        if !detect::has_avx10_v1_aux() {
            return;
        }
        // A spread of FP16 patterns: normals, signed zero, subnormal, overflow, NaN.
        let a: [u16; 32] = core::array::from_fn(|i| (i as u16).wrapping_mul(0x0a37) ^ 0x3c00);
        let s2: [u16; 32] = core::array::from_fn(|i| (i as u16).wrapping_mul(0x1234) ^ 0x4000);
        let bias: [u16; 32] = core::array::from_fn(|i| (i as u16).wrapping_mul(0x0137));

        // Family A.
        assert_eq!(cvtph_bf8(a), cvtph_bf8_scalar(a), "hw cvtph_bf8 != oracle");
        assert_eq!(
            cvtphs_bf8(a),
            cvtphs_bf8_scalar(a),
            "hw cvtphs_bf8 != oracle"
        );
        assert_eq!(cvtph_hf8(a), cvtph_hf8_scalar(a), "hw cvtph_hf8 != oracle");
        assert_eq!(
            cvtphs_hf8(a),
            cvtphs_hf8_scalar(a),
            "hw cvtphs_hf8 != oracle"
        );
        // Family B.
        assert_eq!(
            cvt2ph_bf8(a, s2),
            cvt2ph_bf8_scalar(a, s2),
            "hw cvt2ph_bf8 != oracle"
        );
        assert_eq!(
            cvt2phs_bf8(a, s2),
            cvt2phs_bf8_scalar(a, s2),
            "hw cvt2phs_bf8 != oracle"
        );
        assert_eq!(
            cvt2ph_hf8(a, s2),
            cvt2ph_hf8_scalar(a, s2),
            "hw cvt2ph_hf8 != oracle"
        );
        assert_eq!(
            cvt2phs_hf8(a, s2),
            cvt2phs_hf8_scalar(a, s2),
            "hw cvt2phs_hf8 != oracle"
        );
        // Family C.
        assert_eq!(
            cvtbiasph_bf8(a, bias),
            cvtbiasph_bf8_scalar(a, bias),
            "hw cvtbiasph_bf8 != oracle"
        );
        assert_eq!(
            cvtbiasphs_bf8(a, bias),
            cvtbiasphs_bf8_scalar(a, bias),
            "hw cvtbiasphs_bf8 != oracle"
        );
        assert_eq!(
            cvtbiasph_hf8(a, bias),
            cvtbiasph_hf8_scalar(a, bias),
            "hw cvtbiasph_hf8 != oracle"
        );
        assert_eq!(
            cvtbiasphs_hf8(a, bias),
            cvtbiasphs_hf8_scalar(a, bias),
            "hw cvtbiasphs_hf8 != oracle"
        );
    }
}
