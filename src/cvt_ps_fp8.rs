//! Families A & B (AVX10_V2_AUX): single-source FP32 -> FP8 converts.
//!
//! This module wires **family A**, the six single-source FP32 -> FP8 converts of ACE v1 spec
//! section 9.2 (`VCVTPS2BF8` / `VCVTPS2BF8S` / `VCVTPS2HF8` / `VCVTPS2HF8S` /
//! `VCVTROPS2HF8` / `VCVTROPS2HF8S`), and **family B**, the four bias-rounding FP32 -> FP8
//! converts (`VCVTBIASPS2BF8` / `VCVTBIASPS2BF8S` / `VCVTBIASPS2HF8` / `VCVTBIASPS2HF8S`).
//! Each takes a vector of 16 FP32 lanes and produces 16 FP8 bytes, rounding once to the target
//! format per the section-16.1 `fp32_to_fp8_e5m2` / `fp32_to_fp8_e4m3` pseudocode:
//!
//! | public fn        | target     | rounding | saturating |
//! |------------------|------------|----------|------------|
//! | `cvtps_bf8`      | E5M2 (BF8) | RTNE     | no         |
//! | `cvtpss_bf8`     | E5M2 (BF8) | RTNE     | yes        |
//! | `cvtps_hf8`      | E4M3 (HF8) | RTNE     | no         |
//! | `cvtpss_hf8`     | E4M3 (HF8) | RTNE     | yes        |
//! | `cvtrops_hf8`    | E4M3 (HF8) | RTO      | no         |
//! | `cvtropss_hf8`   | E4M3 (HF8) | RTO      | yes        |
//! | `cvtbiasps_bf8`  | E5M2 (BF8) | BIAS     | no         |
//! | `cvtbiaspss_bf8` | E5M2 (BF8) | BIAS     | yes        |
//! | `cvtbiasps_hf8`  | E4M3 (HF8) | BIAS     | no         |
//! | `cvtbiaspss_hf8` | E4M3 (HF8) | BIAS     | yes        |
//!
//! **RTO is E4M3-only** (spec section 9.1 / 9.2.1): there is deliberately no `cvtrops_bf8` —
//! the E5M2 target has no round-to-odd form. RTNE is spec section 2.6.1 (IEEE
//! roundTiesToEven); RTO is spec section 2.6.2 (round-to-odd, which forces the kept mantissa
//! lsb to 1 whenever the discarded bits are nonzero, so it never selects an even target
//! mantissa for an inexact value, avoiding double-rounding in repeated downsizing).
//!
//! **Family B — bias rounding (spec section 2.6.3 + the section-9.2.5 `vcvtbiasps2f8`
//! pseudocode):** a three-operand form whose per-lane bias term is the full `i32` word taken
//! from Operand 2 (`VVVV`) and whose FP32 source is Operand 3. The bias enters the FP32->FP8
//! rounding at the FP32-mantissa-lsb alignment fixed by the section-16.1 BIAS branch:
//! `m_b = m_i + (bias & 0x1FFFFF)` for E5M2 (21-bit mask) / `m_b = m_i + (bias & 0xFFFFF)` for
//! E4M3 (20-bit mask); on a carry out of bit 23 (`m_b & 0xFF800000`) the exponent is
//! incremented and `m_b &= 0x7FFFFF`; the kept mantissa is `m_b >> 21` (E5M2) / `m_b >> 20`
//! (E4M3). This is **add-then-truncate** with NO separate round step, so a **zero bias term
//! reduces to the pseudocode's truncate-toward-zero behaviour**. The same section-9.2.1
//! overflow/NaN/signed-zero table applies, with bias replacing RTNE/RTO. This is implemented by
//! [`crate::fp8::fp32_to_fp8_e5m2_biased`] / [`crate::fp8::fp32_to_fp8_e4m3_biased`] under
//! [`Fp8RoundMode::Bias`].
//! `[avx10-v2-aux-ocp-conversions.CVT_BIAS_PS_FP8.1]`
//! `[avx10-v2-aux-ocp-conversions.CVT_BIAS_PS_FP8.2]`
//! `[avx10-v2-aux-ocp-conversions.CVT_BIAS_PS_FP8.3]`
//! `[avx10-v2-aux-ocp-conversions.CVT_BIAS_PS_FP8.4]`
//!
//! Overflow / NaN / signed-zero behaviour follows the section-9.2.1 table, encoded in the
//! [`crate::fp8`] front-end: E5M2 non-saturating overflow -> the BF8 Inf/overflow-coded
//! `S.11111.00`, saturating -> `±max_E5M2 = ±57344`; E4M3 non-saturating overflow -> the
//! sole HF8 NaN `S.1111.111`, saturating -> `±max_E4M3 = ±448`. A NaN input maps to a NaN
//! FP8 result; a signed zero maps to the same-signed FP8 zero. MXCSR is neither consulted
//! nor updated (DAZ=1, FTZ=0; no FP exceptions).
//! `[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.1]` `[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.2]`
//! `[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.3]`
//! `[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.3-note]`
//! `[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.4]` `[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.5]`
//! `[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.6]` `[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.7]`
//! `[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.8]` `[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.9]`
//! `[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.10]`
//!
//! Each public fn is a safe dispatcher that consults [`detect::has_avx10_v2_aux`] and falls
//! through to its `_scalar` oracle, the primary always-correct path on every target
//! (`[avx10-v2-aux-ocp-conversions.CORRECTNESS.1]`,
//! `[avx10-v2-aux-ocp-conversions.CORRECTNESS.2]`,
//! `[avx10-v2-aux-ocp-conversions.DETECTION.2]`). The names mirror the eventual stdarch
//! intrinsic stems (`[avx10-v2-aux-ocp-conversions.NAMING.1]`); the module is stable Rust,
//! no `core::simd`/nightly (`[avx10-v2-aux-ocp-conversions.STABLE_RUST.1]`).
//!
//! OQ-5 (intrinsic unavailable -> oracle-only): ALL family-A AND family-B forms ship
//! **oracle-only**. The installed GCC 16.x `-mavx10.2` headers expose only the FP16 -> FP8
//! intrinsics (`_mm512_cvtph_bf8`, `_mm512_cvtbiasph_bf8`, ...); there is no FP32 -> FP8 form.
//! `_mm512_cvtps_bf8` / `_mm512_cvts_ps_bf8` / `_mm512_cvtps_hf8` / `_mm512_cvtroundps_hf8` and
//! the bias siblings `_mm512_cvtbiasps_bf8` / `_mm512_cvtbiaspss_bf8` / `_mm512_cvtbiasps_hf8` /
//! `_mm512_cvtbiaspss_hf8` are all absent — confirmed by a compile probe (GCC suggests the
//! FP16-source `_mm512_cvtbiasph_bf8` in their place). Per OQ-5 there is therefore no native C
//! shim and no `_hw` path for any family-A or family-B converter; each dispatcher resolves to
//! its `_scalar` sibling on every target. The differential test that would otherwise tie the
//! native path to the oracle is discarded (no native path exists), so the oracle's correctness
//! is grounded against the spec section-16.1 pseudocode, transcribed bit-for-bit in
//! [`crate::fp8`]. The capability check is never consulted — the dispatchers only
//! reference the detector to mark the gate site for the shim once the intrinsics land.

use crate::detect;
use crate::fp8::{self, Fp8RoundMode};

// ---------------------------------------------------------------------------------------
// Family A oracles (the primary, always-correct path). Each maps every lane through the
// section-16.1 fp8 front-end with the matching target / rounding mode / saturating flag.
// ---------------------------------------------------------------------------------------

/// Oracle for [`cvtps_bf8`] — FP32 -> BF8 (E5M2), RTNE, non-saturating.
/// `[avx10-v2-aux-ocp-conversions.CORRECTNESS.1]`
pub fn cvtps_bf8_scalar(a: [f32; 16]) -> [u8; 16] {
    core::array::from_fn(|i| fp8::fp32_to_fp8_e5m2(a[i], Fp8RoundMode::Rtne, false))
}

/// Oracle for [`cvtpss_bf8`] — FP32 -> BF8 (E5M2), RTNE, saturating.
/// `[avx10-v2-aux-ocp-conversions.CORRECTNESS.1]`
pub fn cvtpss_bf8_scalar(a: [f32; 16]) -> [u8; 16] {
    core::array::from_fn(|i| fp8::fp32_to_fp8_e5m2(a[i], Fp8RoundMode::Rtne, true))
}

/// Oracle for [`cvtps_hf8`] — FP32 -> HF8 (E4M3), RTNE, non-saturating.
/// `[avx10-v2-aux-ocp-conversions.CORRECTNESS.1]`
pub fn cvtps_hf8_scalar(a: [f32; 16]) -> [u8; 16] {
    core::array::from_fn(|i| fp8::fp32_to_fp8_e4m3(a[i], Fp8RoundMode::Rtne, false))
}

/// Oracle for [`cvtpss_hf8`] — FP32 -> HF8 (E4M3), RTNE, saturating.
/// `[avx10-v2-aux-ocp-conversions.CORRECTNESS.1]`
pub fn cvtpss_hf8_scalar(a: [f32; 16]) -> [u8; 16] {
    core::array::from_fn(|i| fp8::fp32_to_fp8_e4m3(a[i], Fp8RoundMode::Rtne, true))
}

/// Oracle for [`cvtrops_hf8`] — FP32 -> HF8 (E4M3), RTO, non-saturating (E4M3-only mode).
/// `[avx10-v2-aux-ocp-conversions.CORRECTNESS.1]`
pub fn cvtrops_hf8_scalar(a: [f32; 16]) -> [u8; 16] {
    core::array::from_fn(|i| fp8::fp32_to_fp8_e4m3(a[i], Fp8RoundMode::Rto, false))
}

/// Oracle for [`cvtropss_hf8`] — FP32 -> HF8 (E4M3), RTO, saturating (E4M3-only mode).
/// `[avx10-v2-aux-ocp-conversions.CORRECTNESS.1]`
pub fn cvtropss_hf8_scalar(a: [f32; 16]) -> [u8; 16] {
    core::array::from_fn(|i| fp8::fp32_to_fp8_e4m3(a[i], Fp8RoundMode::Rto, true))
}

// ---------------------------------------------------------------------------------------
// Family A public dispatchers. OQ-5: no native FP32->FP8 intrinsic exists in the toolchain,
// so each merely references the capability check (marking the three-layer gate site) and
// falls through to its oracle on every target; there is no `if detect::...` arm until the
// intrinsic lands — the unconditional `_scalar` tail is the deliverable value.
// ---------------------------------------------------------------------------------------

/// FP32 -> BF8 (E5M2), RTNE, **non-saturating** (`VCVTPS2BF8`). On post-rounding overflow:
/// the BF8 Inf/overflow-coded value `S.11111.00`; NaN input -> NaN; signed zero preserved.
/// `[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.1]` `[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.5]`
/// `[avx10-v2-aux-ocp-conversions.DETECTION.2]`
pub fn cvtps_bf8(a: [f32; 16]) -> [u8; 16] {
    let _ = detect::has_avx10_v2_aux; // reference (not call) the future gate; see fn docs
    cvtps_bf8_scalar(a)
}

/// FP32 -> BF8 (E5M2), RTNE, **saturating** (`VCVTPS2BF8S`). On post-rounding overflow:
/// clamp to `±max_E5M2 = ±57344` `S.11110.11`; NaN input -> NaN; signed zero preserved.
/// `[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.1]` `[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.5]`
/// `[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.7]` `[avx10-v2-aux-ocp-conversions.DETECTION.2]`
pub fn cvtpss_bf8(a: [f32; 16]) -> [u8; 16] {
    let _ = detect::has_avx10_v2_aux; // reference (not call) the future gate; see fn docs
    cvtpss_bf8_scalar(a)
}

/// FP32 -> HF8 (E4M3), RTNE, **non-saturating** (`VCVTPS2HF8`). On post-rounding overflow:
/// the sole HF8 NaN `S.1111.111`; NaN input -> NaN; signed zero preserved.
/// `[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.2]` `[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.6]`
/// `[avx10-v2-aux-ocp-conversions.DETECTION.2]`
pub fn cvtps_hf8(a: [f32; 16]) -> [u8; 16] {
    let _ = detect::has_avx10_v2_aux; // reference (not call) the future gate; see fn docs
    cvtps_hf8_scalar(a)
}

/// FP32 -> HF8 (E4M3), RTNE, **saturating** (`VCVTPS2HF8S`). On post-rounding overflow:
/// clamp to `±max_E4M3 = ±448` `S.1111.110`; NaN input -> NaN; signed zero preserved.
/// `[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.2]` `[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.6]`
/// `[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.7]` `[avx10-v2-aux-ocp-conversions.DETECTION.2]`
pub fn cvtpss_hf8(a: [f32; 16]) -> [u8; 16] {
    let _ = detect::has_avx10_v2_aux; // reference (not call) the future gate; see fn docs
    cvtpss_hf8_scalar(a)
}

/// FP32 -> HF8 (E4M3), **round-to-odd**, non-saturating (`VCVTROPS2HF8`, E4M3-only). On
/// post-rounding overflow: the HF8 NaN `S.1111.111`; an inexact value yields an odd target
/// mantissa (never even); NaN input -> NaN; signed zero preserved.
/// `[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.3]` `[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.4]`
/// `[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.6]` `[avx10-v2-aux-ocp-conversions.DETECTION.2]`
pub fn cvtrops_hf8(a: [f32; 16]) -> [u8; 16] {
    let _ = detect::has_avx10_v2_aux; // reference (not call) the future gate; see fn docs
    cvtrops_hf8_scalar(a)
}

/// FP32 -> HF8 (E4M3), **round-to-odd**, saturating (`VCVTROPS2HF8S`, E4M3-only). On
/// post-rounding overflow: clamp to `±max_E4M3 = ±448`; inexact values round to odd; NaN
/// input -> NaN; signed zero preserved.
/// `[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.3]` `[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.4]`
/// `[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.6]` `[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.7]`
/// `[avx10-v2-aux-ocp-conversions.DETECTION.2]`
pub fn cvtropss_hf8(a: [f32; 16]) -> [u8; 16] {
    let _ = detect::has_avx10_v2_aux; // reference (not call) the future gate; see fn docs
    cvtropss_hf8_scalar(a)
}

// ---------------------------------------------------------------------------------------
// Family B oracles — FP32 -> FP8 with per-lane bias rounding (spec section 2.6.3 + the
// section-9.2.5 `vcvtbiasps2f8` pseudocode). The per-lane bias term is the full `i32` word
// from Operand 2 (`VVVV`); the FP32 source is Operand 3. Each lane reinterprets `bias[i]` as
// `u32` and feeds it to the fp8 BIAS branch, which masks it to 21 bits (E5M2) / 20 bits
// (E4M3) and applies the add-then-truncate rounding. A zero bias term reduces to the
// pseudocode's truncate-toward-zero behaviour.
// `[avx10-v2-aux-ocp-conversions.CVT_BIAS_PS_FP8.1]`
// `[avx10-v2-aux-ocp-conversions.CVT_BIAS_PS_FP8.2]`
// `[avx10-v2-aux-ocp-conversions.CVT_BIAS_PS_FP8.4]`
// ---------------------------------------------------------------------------------------

/// Oracle for [`cvtbiasps_bf8`] — FP32 -> BF8 (E5M2), bias rounding, non-saturating.
/// `[avx10-v2-aux-ocp-conversions.CORRECTNESS.1]`
pub fn cvtbiasps_bf8_scalar(a: [f32; 16], bias: [i32; 16]) -> [u8; 16] {
    core::array::from_fn(|i| {
        fp8::fp32_to_fp8_e5m2_biased(a[i], Fp8RoundMode::Bias, false, bias[i] as u32)
    })
}

/// Oracle for [`cvtbiaspss_bf8`] — FP32 -> BF8 (E5M2), bias rounding, saturating.
/// `[avx10-v2-aux-ocp-conversions.CORRECTNESS.1]`
pub fn cvtbiaspss_bf8_scalar(a: [f32; 16], bias: [i32; 16]) -> [u8; 16] {
    core::array::from_fn(|i| {
        fp8::fp32_to_fp8_e5m2_biased(a[i], Fp8RoundMode::Bias, true, bias[i] as u32)
    })
}

/// Oracle for [`cvtbiasps_hf8`] — FP32 -> HF8 (E4M3), bias rounding, non-saturating.
/// `[avx10-v2-aux-ocp-conversions.CORRECTNESS.1]`
pub fn cvtbiasps_hf8_scalar(a: [f32; 16], bias: [i32; 16]) -> [u8; 16] {
    core::array::from_fn(|i| {
        fp8::fp32_to_fp8_e4m3_biased(a[i], Fp8RoundMode::Bias, false, bias[i] as u32)
    })
}

/// Oracle for [`cvtbiaspss_hf8`] — FP32 -> HF8 (E4M3), bias rounding, saturating.
/// `[avx10-v2-aux-ocp-conversions.CORRECTNESS.1]`
pub fn cvtbiaspss_hf8_scalar(a: [f32; 16], bias: [i32; 16]) -> [u8; 16] {
    core::array::from_fn(|i| {
        fp8::fp32_to_fp8_e4m3_biased(a[i], Fp8RoundMode::Bias, true, bias[i] as u32)
    })
}

// ---------------------------------------------------------------------------------------
// Family B public dispatchers. OQ-5: no native FP32->FP8 bias intrinsic exists in the
// toolchain (`_mm512_cvtbiasps_bf8` etc. are absent — only the FP16-source siblings exist),
// so each merely references the capability check and falls through to its oracle on every
// target.
// ---------------------------------------------------------------------------------------

/// FP32 -> BF8 (E5M2) with per-lane **bias rounding**, **non-saturating** (`VCVTBIASPS2BF8`).
/// `bias` is the per-lane Operand-2 (`VVVV`) `i32` term applied at the section-9.2.5 alignment
/// (21-bit mask, add-then-truncate); a zero bias reduces to the pseudocode's truncate
/// behaviour. On post-rounding overflow: the BF8 Inf/overflow-coded value `S.11111.00`; NaN
/// input -> NaN; signed zero preserved.
/// `[avx10-v2-aux-ocp-conversions.CVT_BIAS_PS_FP8.1]`
/// `[avx10-v2-aux-ocp-conversions.CVT_BIAS_PS_FP8.3]`
/// `[avx10-v2-aux-ocp-conversions.CVT_BIAS_PS_FP8.4]`
/// `[avx10-v2-aux-ocp-conversions.DETECTION.2]`
pub fn cvtbiasps_bf8(a: [f32; 16], bias: [i32; 16]) -> [u8; 16] {
    let _ = detect::has_avx10_v2_aux; // reference (not call) the future gate; see fn docs
    cvtbiasps_bf8_scalar(a, bias)
}

/// FP32 -> BF8 (E5M2) with per-lane **bias rounding**, **saturating** (`VCVTBIASPS2BF8S`).
/// On post-rounding overflow: clamp to `±max_E5M2 = ±57344`; NaN input -> NaN; signed zero
/// preserved.
/// `[avx10-v2-aux-ocp-conversions.CVT_BIAS_PS_FP8.1]`
/// `[avx10-v2-aux-ocp-conversions.CVT_BIAS_PS_FP8.3]`
/// `[avx10-v2-aux-ocp-conversions.CVT_BIAS_PS_FP8.4]`
/// `[avx10-v2-aux-ocp-conversions.DETECTION.2]`
pub fn cvtbiaspss_bf8(a: [f32; 16], bias: [i32; 16]) -> [u8; 16] {
    let _ = detect::has_avx10_v2_aux; // reference (not call) the future gate; see fn docs
    cvtbiaspss_bf8_scalar(a, bias)
}

/// FP32 -> HF8 (E4M3) with per-lane **bias rounding**, **non-saturating** (`VCVTBIASPS2HF8`).
/// `bias` is the per-lane Operand-2 (`VVVV`) `i32` term applied at the section-9.2.5 alignment
/// (20-bit mask, add-then-truncate); a zero bias reduces to the pseudocode's truncate
/// behaviour. On post-rounding overflow: the sole HF8 NaN `S.1111.111`; NaN input -> NaN;
/// signed zero preserved.
/// `[avx10-v2-aux-ocp-conversions.CVT_BIAS_PS_FP8.2]`
/// `[avx10-v2-aux-ocp-conversions.CVT_BIAS_PS_FP8.3]`
/// `[avx10-v2-aux-ocp-conversions.CVT_BIAS_PS_FP8.4]`
/// `[avx10-v2-aux-ocp-conversions.DETECTION.2]`
pub fn cvtbiasps_hf8(a: [f32; 16], bias: [i32; 16]) -> [u8; 16] {
    let _ = detect::has_avx10_v2_aux; // reference (not call) the future gate; see fn docs
    cvtbiasps_hf8_scalar(a, bias)
}

/// FP32 -> HF8 (E4M3) with per-lane **bias rounding**, **saturating** (`VCVTBIASPS2HF8S`).
/// On post-rounding overflow: clamp to `±max_E4M3 = ±448`; NaN input -> NaN; signed zero
/// preserved.
/// `[avx10-v2-aux-ocp-conversions.CVT_BIAS_PS_FP8.2]`
/// `[avx10-v2-aux-ocp-conversions.CVT_BIAS_PS_FP8.3]`
/// `[avx10-v2-aux-ocp-conversions.CVT_BIAS_PS_FP8.4]`
/// `[avx10-v2-aux-ocp-conversions.DETECTION.2]`
pub fn cvtbiaspss_hf8(a: [f32; 16], bias: [i32; 16]) -> [u8; 16] {
    let _ = detect::has_avx10_v2_aux; // reference (not call) the future gate; see fn docs
    cvtbiaspss_hf8_scalar(a, bias)
}

#[cfg(test)]
mod tests {
    use super::*;

    // E5M2 byte assembler: sign | 5-bit exp field | 2-bit mantissa.
    fn bf8(sign: u8, exp: u8, mant: u8) -> u8 {
        (sign << 7) | (exp << 2) | mant
    }
    // E4M3 byte assembler: sign | 4-bit exp field | 3-bit mantissa.
    fn hf8(sign: u8, exp: u8, mant: u8) -> u8 {
        (sign << 7) | (exp << 3) | mant
    }
    // FP32 just above 1.0 by one ULP (mantissa lsb set): inexact in E4M3 (which keeps 3
    // mantissa bits) — the canonical RTNE-vs-RTO discriminator.
    fn one_plus_ulp() -> f32 {
        f32::from_bits(0x3F80_0001)
    }

    /// RTNE rounding pins (`[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.1]`,
    /// `[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.2]`). Exact-representable values map to their
    /// canonical FP8 byte: 1.0 -> E5M2 `S.01111.00` (0x3C) and E4M3 `S.0111.000` (0x38).
    #[test]
    fn known_value_rtne_exact() {
        let mut a = [0.0f32; 16];
        a[0] = 1.0;
        a[1] = -1.0;
        a[2] = 1.5;
        assert_eq!(cvtps_bf8(a)[0], bf8(0, 0b01111, 0b00), "E5M2 1.0");
        assert_eq!(cvtps_bf8(a)[1], bf8(1, 0b01111, 0b00), "E5M2 -1.0");
        assert_eq!(cvtps_hf8(a)[0], hf8(0, 0b0111, 0b000), "E4M3 1.0");
        // 1.5 = 1.10b: E5M2 mantissa 10 -> S.01111.10 (0x3E); E4M3 mantissa 100 -> 0x3C.
        assert_eq!(cvtps_bf8(a)[2], bf8(0, 0b01111, 0b10), "E5M2 1.5");
        assert_eq!(cvtps_hf8(a)[2], hf8(0, 0b0111, 0b100), "E4M3 1.5");
    }

    /// RTO odd-on-inexact discriminator (`[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.3]`,
    /// `[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.4]`). `1.0 + 2^-23` is exact in FP32 but
    /// inexact in E4M3. Under **RTNE** the discarded bits round to the nearest even mantissa
    /// (0) -> `S.0111.000` (0x38). Under **RTO** the nonzero discarded bits force the kept lsb
    /// to 1 -> `S.0111.001` (0x39), an **odd** mantissa. The two bytes DIFFER, so this rules
    /// out an implementation that models RTO as RTNE (which would also give 0x38).
    #[test]
    fn known_value_rto_is_odd_on_inexact() {
        let mut a = [0.0f32; 16];
        a[0] = one_plus_ulp();
        let rtne = cvtps_hf8(a)[0];
        let rto = cvtrops_hf8(a)[0];
        assert_eq!(
            rtne,
            hf8(0, 0b0111, 0b000),
            "RTNE rounds to even mantissa 0"
        );
        assert_eq!(rto, hf8(0, 0b0111, 0b001), "RTO forces odd mantissa 1");
        assert_ne!(rtne, rto, "RTO must differ from RTNE on an inexact value");
        assert_eq!(rto & 0x1, 1, "RTO target mantissa is odd");
    }

    /// BF8 (E5M2) overflow: non-saturating vs saturating
    /// (`[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.5]`,
    /// `[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.7]`). A magnitude far above max_E5M2 maps,
    /// non-saturating, to the BF8 Inf/overflow code `S.11111.00` (0x7C) — NOT a NaN
    /// `S.11111.1x` (this rules out an "E5M2 overflow == NaN" model). Saturating clamps to
    /// `±max_E5M2 = ±57344` `S.11110.11` (0x7B / 0xFB).
    #[test]
    fn known_value_bf8_overflow() {
        let mut a = [0.0f32; 16];
        a[0] = 1e30;
        a[1] = -1e30;
        assert_eq!(
            cvtps_bf8(a)[0],
            bf8(0, 0b11111, 0b00),
            "nsat overflow -> Inf-coded"
        );
        assert_eq!(
            cvtps_bf8(a)[1],
            bf8(1, 0b11111, 0b00),
            "nsat -overflow -> -Inf-coded"
        );
        assert_eq!(cvtpss_bf8(a)[0], bf8(0, 0b11110, 0b11), "sat -> +57344");
        assert_eq!(cvtpss_bf8(a)[1], bf8(1, 0b11110, 0b11), "sat -> -57344");
        // The saturating byte is finite-coded (exp field 0b11110 != all-ones), the
        // non-saturating one is the all-ones (Inf) code — saturating magnitude <= non-sat.
        assert_ne!(cvtps_bf8(a)[0], cvtpss_bf8(a)[0]);
    }

    /// HF8 (E4M3) overflow (`[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.6]`,
    /// `[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.7]`). E4M3 has no Inf, so non-saturating
    /// overflow maps to the sole NaN `S.1111.111` (0x7F); saturating clamps to
    /// `±max_E4M3 = ±448` `S.1111.110` (0x7E / 0xFE). This rules out an E5M2-style Inf model
    /// for E4M3 overflow.
    #[test]
    fn known_value_hf8_overflow() {
        let mut a = [0.0f32; 16];
        a[0] = 1e30;
        a[1] = -1e30;
        assert_eq!(
            cvtps_hf8(a)[0],
            hf8(0, 0b1111, 0b111),
            "nsat overflow -> NaN"
        );
        assert_eq!(cvtpss_hf8(a)[0], hf8(0, 0b1111, 0b110), "sat -> +448");
        assert_eq!(cvtpss_hf8(a)[1], hf8(1, 0b1111, 0b110), "sat -> -448");
        // RTO overflow behaves identically (NaN nsat / clamp sat).
        assert_eq!(
            cvtrops_hf8(a)[0],
            hf8(0, 0b1111, 0b111),
            "RTO nsat overflow -> NaN"
        );
        assert_eq!(cvtropss_hf8(a)[0], hf8(0, 0b1111, 0b110), "RTO sat -> +448");
    }

    /// NaN input -> NaN; signed zero preserved (`[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.8]`),
    /// for both targets and both modes. The E5M2 NaN is `S.11111.1x` (mantissa nonzero), which
    /// is DISTINCT from the E5M2 Inf code `S.11111.00` — so a NaN input does not collapse to
    /// the Inf encoding. The E4M3 NaN is `S.1111.111`. Signed zeros map to `0x00` / `0x80`.
    #[test]
    fn known_value_nan_and_signed_zero() {
        let mut a = [0.0f32; 16];
        a[0] = f32::NAN;
        a[1] = -0.0;
        a[2] = 0.0;

        // E5M2 NaN: all-ones exponent with a NONZERO mantissa (the quiet-NaN code), not Inf.
        let bf = cvtps_bf8(a);
        assert_eq!((bf[0] >> 2) & 0x1f, 0x1f, "E5M2 NaN exp all-ones");
        assert_ne!(
            bf[0] & 0x3,
            0,
            "E5M2 NaN mantissa nonzero (distinct from Inf 0x7C)"
        );
        // E4M3 NaN: the sole S.1111.111 code, both RTNE and RTO.
        assert_eq!(cvtps_hf8(a)[0], hf8(0, 0b1111, 0b111), "E4M3 RTNE NaN");
        assert_eq!(cvtrops_hf8(a)[0], hf8(0, 0b1111, 0b111), "E4M3 RTO NaN");

        // Signed zero preserved across every form.
        for f in [
            cvtps_bf8,
            cvtpss_bf8,
            cvtps_hf8,
            cvtpss_hf8,
            cvtrops_hf8,
            cvtropss_hf8,
        ] {
            assert_eq!(f(a)[1], 0x80, "-0.0 -> S.0 zero");
            assert_eq!(f(a)[2], 0x00, "+0.0 -> +0 zero");
        }
    }

    /// `cvtrops_bf8` does not exist on the public surface: RTO is E4M3-only
    /// (`[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.3-note]`). This compile-time fact is pinned
    /// by the re-export set in `lib.rs` (there is no `cvtrops_bf8` symbol); here we document
    /// that the BF8 oracles only ever use `Rtne`, never `Rto`.
    #[test]
    fn rto_is_e4m3_only() {
        // Both BF8 oracles are RTNE — feeding an inexact value yields the RTNE (even) result,
        // never an RTO (odd) one. (There is intentionally no RTO BF8 entry point.)
        let mut a = [0.0f32; 16];
        // 1.0 + 2^-21 is inexact in E5M2 (2 mantissa bits): RTNE rounds to nearest-even.
        a[0] = f32::from_bits(0x3F80_0004);
        let nsat = cvtps_bf8(a)[0];
        let sat = cvtpss_bf8(a)[0];
        // RTNE on this value: mantissa 00 stays even (round down); never the odd-forced 01.
        assert_eq!(nsat, bf8(0, 0b01111, 0b00), "E5M2 RTNE round-to-even");
        assert_eq!(sat, nsat, "in-range value: sat == nsat");
    }

    // -----------------------------------------------------------------------------------
    // Family B — bias-rounding known-value tests.
    // -----------------------------------------------------------------------------------

    /// Zero bias reduces to the section-9.2.5 BIAS-branch truncate-toward-zero behaviour
    /// (`[avx10-v2-aux-ocp-conversions.CVT_BIAS_PS_FP8.4]`), which is NOT RTNE. Take
    /// `1.0 + 2^-23` (mantissa lsb set, inexact in E4M3): RTNE rounds the discarded bits to the
    /// nearest even mantissa (0) -> `S.0111.000` (0x38), but **truncate** discards them
    /// downward, also giving mantissa 0 here, so we need a value where truncate and RTNE
    /// DIFFER to discriminate. Use `1.0 + 5*2^-23 = 1.0000005...`: in E4M3 the kept mantissa is
    /// 3 bits, the discarded 20-bit window is `0x500000 / 2^23` which is just ABOVE half an
    /// E4M3 lsb (`0x400000`), so RTNE rounds UP to mantissa 1 (`S.0111.001` = 0x39) while
    /// zero-bias TRUNCATE keeps mantissa 0 (`S.0111.000` = 0x38). The bytes differ — this rules
    /// out a "bias==0 behaves as RTNE" model. For BF8 (E5M2, 2 mantissa bits) the analogous
    /// `1.0 + 0x180000/2^23` is above half an E5M2 lsb so RTNE -> mantissa 01 but truncate ->
    /// mantissa 00.
    #[test]
    fn known_value_bias_zero_equals_truncate() {
        let zero = [0i32; 16];

        // E4M3 discriminator: `1.0 + m_i` with `m_i` ABOVE the E4M3 half-lsb. E4M3 keeps the
        // top 3 mantissa bits (`m_i >> 20`); the round position is bit 19 (half-lsb 0x80000).
        // Pick `m_i = 0xC0000` (bits 18..19 set): kept = `0xC0000 >> 20 = 0`, discarded
        // `0xC0000 > 0x80000` half. Under **RTNE** the discarded bits round UP -> mantissa 1
        // (`S.0111.001` = 0x39). Under zero-bias **TRUNCATE** they drop -> mantissa 0
        // (`S.0111.000` = 0x38). The bytes DIFFER, ruling out a "bias==0 behaves as RTNE" model.
        let mut a = [0.0f32; 16];
        a[0] = f32::from_bits(0x3F80_0000 | 0x000C_0000); // 1.0, m_i = 0xC0000
        let rtne = cvtps_hf8(a)[0];
        let bias0 = cvtbiasps_hf8(a, zero)[0];
        assert_eq!(rtne, hf8(0, 0b0111, 0b001), "E4M3 RTNE rounds up to mant 1");
        assert_eq!(
            bias0,
            hf8(0, 0b0111, 0b000),
            "E4M3 zero-bias truncates down to mant 0"
        );
        assert_ne!(bias0, rtne, "zero-bias TRUNCATE must differ from RTNE");

        // E5M2 discriminator: E5M2 keeps the top 2 mantissa bits (`m_i >> 21`); the round
        // position is bit 20 (half-lsb 0x100000). Pick `m_i = 0x180000` (bits 19..20 set):
        // kept = `0x180000 >> 21 = 0`, discarded `0x180000 > 0x100000` half. RTNE rounds UP ->
        // mantissa 1 (`S.01111.01`); zero-bias TRUNCATE -> mantissa 0 (`S.01111.00`).
        let mut b = [0.0f32; 16];
        b[0] = f32::from_bits(0x3F80_0000 | 0x0018_0000); // 1.0, m_i = 0x180000
        let rtne_bf = cvtps_bf8(b)[0];
        let bias0_bf = cvtbiasps_bf8(b, zero)[0];
        assert_eq!(
            rtne_bf,
            bf8(0, 0b01111, 0b01),
            "E5M2 RTNE rounds up to mant 1"
        );
        assert_eq!(
            bias0_bf,
            bf8(0, 0b01111, 0b00),
            "E5M2 zero-bias truncates down to mant 0"
        );
        assert_ne!(
            bias0_bf, rtne_bf,
            "zero-bias TRUNCATE must differ from RTNE"
        );
    }

    /// A non-zero bias shifts the truncated byte upward at the section-9.2.5 alignment
    /// (`[avx10-v2-aux-ocp-conversions.CVT_BIAS_PS_FP8.1]`,
    /// `[avx10-v2-aux-ocp-conversions.CVT_BIAS_PS_FP8.2]`). Take `t = 1.0` exactly
    /// (`m_i == 0`). For E4M3 the kept mantissa is `m_b >> 20`; a bias of `1 << 20 = 0x100000`
    /// makes `m_b = 0x100000`, so `m_b >> 20 == 1` -> `S.0111.001` (0x38 + 1 = 0x39), one ULP
    /// above the zero-bias `S.0111.000` (0x38). This is a DISCRIMINATING case: a bias that
    /// entered at the wrong alignment (e.g. byte-aligned like the FP16 family, or a 21-bit
    /// E5M2 mask) would not produce exactly mantissa 1 here. For E5M2 the kept mantissa is
    /// `m_b >> 21`; a bias of `1 << 21 = 0x200000` makes `m_b >> 21 == 1` -> `S.01111.01`
    /// (0x3C + 1 = 0x3D), one ULP above the zero-bias `S.01111.00` (0x3C).
    #[test]
    fn known_value_bias_nonzero_shifts() {
        // The bias term occupies the SAME bit window as the discarded FP32 fraction (the
        // section-9.2.5 alignment: masked to 20 bits for E4M3, 21 bits for E5M2), so it nudges
        // within that window and rounds the kept mantissa up only when the add carries past the
        // target lsb. Pick `t = 1.0 + half-lsb`: with `bias == 0` it truncates DOWN (kept
        // mantissa 0), and a bias that pushes the windowed sum to the target lsb carries it UP
        // by exactly one ULP. This is a DISCRIMINATING alignment check: a byte-aligned bias
        // (the FP16 family) or a wrong mask width would not carry here.

        // E4M3: half-lsb is bit 19 (0x80000). t = 1.0 with m_i = 0x80000; bias = 0x80000 makes
        // m_b = 0x100000 -> kept mantissa `0x100000 >> 20 = 1`.
        let mut a_hf = [0.0f32; 16];
        a_hf[0] = f32::from_bits(0x3F80_0000 | 0x0008_0000); // 1.0, m_i = 0x80000
        let mut bias_hf = [0i32; 16];
        bias_hf[0] = 0x0008_0000;
        let z = cvtbiasps_hf8(a_hf, [0i32; 16])[0];
        let s = cvtbiasps_hf8(a_hf, bias_hf)[0];
        assert_eq!(
            z,
            hf8(0, 0b0111, 0b000),
            "E4M3 zero-bias truncates -> mant 0"
        );
        assert_eq!(
            s,
            hf8(0, 0b0111, 0b001),
            "E4M3 bias carries -> mant 1 (one ULP up)"
        );
        assert_eq!(s, z + 1, "non-zero bias shifts up by exactly one E4M3 ULP");

        // E5M2: half-lsb is bit 20 (0x100000). t = 1.0 with m_i = 0x100000; bias = 0x100000
        // makes m_b = 0x200000 -> kept mantissa `0x200000 >> 21 = 1`.
        let mut a_bf = [0.0f32; 16];
        a_bf[0] = f32::from_bits(0x3F80_0000 | 0x0010_0000); // 1.0, m_i = 0x100000
        let mut bias_bf = [0i32; 16];
        bias_bf[0] = 0x0010_0000;
        let zb = cvtbiasps_bf8(a_bf, [0i32; 16])[0];
        let sb = cvtbiasps_bf8(a_bf, bias_bf)[0];
        assert_eq!(
            zb,
            bf8(0, 0b01111, 0b00),
            "E5M2 zero-bias truncates -> mant 0"
        );
        assert_eq!(
            sb,
            bf8(0, 0b01111, 0b01),
            "E5M2 bias carries -> mant 1 (one ULP up)"
        );
        assert_eq!(
            sb,
            zb + 1,
            "non-zero bias shifts up by exactly one E5M2 ULP"
        );
    }

    /// A bias large enough to carry out of bit 23 increments the exponent
    /// (`[avx10-v2-aux-ocp-conversions.CVT_BIAS_PS_FP8.1]`,
    /// `[avx10-v2-aux-ocp-conversions.CVT_BIAS_PS_FP8.4]`). Take `t = 2.0 - eps` just below the
    /// 2.0 binade so its mantissa is near all-ones, then add a bias that overflows bit 23,
    /// rolling the value into the next exponent. Use `t = f32 with e_i for [1,2), m_i = 0x7FFFFF`
    /// (the largest mantissa in the 1.x binade); a bias of 1 makes `m_b = 0x800000`, which sets
    /// bit 23 (`m_b & 0xFF800000 != 0`), so the exponent increments by 1 and `m_b &= 0x7FFFFF`
    /// becomes 0 -> the result is exactly `2.0` in the target. For E4M3 that is
    /// `S.1000.000` (exp field 8, mantissa 0).
    #[test]
    fn known_value_bias_carry_out_of_bit23() {
        let mut a = [0.0f32; 16];
        // exponent field 127 (unbiased 0 -> binade [1,2)), mantissa all-ones.
        a[0] = f32::from_bits((127u32 << 23) | 0x7F_FFFF);
        let mut bias = [0i32; 16];
        bias[0] = 1; // carries m_b past bit 23 -> exponent++ and mantissa wraps to 0.

        // E4M3: result should be exactly 2.0 = S.1000.000 (exp field 0b1000, mant 0).
        let hf = cvtbiasps_hf8(a, bias)[0];
        assert_eq!(
            hf,
            hf8(0, 0b1000, 0b000),
            "carry out of bit 23 rolls 1.111.. into 2.0 (exp++)"
        );
        // E5M2: result should be exactly 2.0 = S.10000.00 (exp field 0b10000, mant 0).
        let bf = cvtbiasps_bf8(a, bias)[0];
        assert_eq!(
            bf,
            bf8(0, 0b10000, 0b00),
            "carry out of bit 23 rolls 1.111.. into 2.0 (exp++)"
        );
    }

    /// Family-B saturating clamp lane (`[avx10-v2-aux-ocp-conversions.CVT_BIAS_PS_FP8.3]`). A
    /// huge FP32 magnitude overflows for both targets: non-saturating yields the target's
    /// overflow encoding (BF8 Inf-coded `S.11111.00`, HF8 NaN `S.1111.111`); saturating clamps
    /// to `±max` (BF8 `±57344`, HF8 `±448`). The bias does not rescue an already-overflowing
    /// exponent, so the overflow split matches family A.
    #[test]
    fn known_value_bias_saturating_clamp() {
        let mut a = [0.0f32; 16];
        a[0] = 1e30;
        a[1] = -1e30;
        let bias = [0i32; 16];

        // BF8.
        assert_eq!(
            cvtbiasps_bf8(a, bias)[0],
            bf8(0, 0b11111, 0b00),
            "BF8 nsat bias overflow -> Inf-coded"
        );
        assert_eq!(
            cvtbiaspss_bf8(a, bias)[0],
            bf8(0, 0b11110, 0b11),
            "BF8 sat -> +57344"
        );
        assert_eq!(
            cvtbiaspss_bf8(a, bias)[1],
            bf8(1, 0b11110, 0b11),
            "BF8 sat -> -57344"
        );
        // HF8.
        assert_eq!(
            cvtbiasps_hf8(a, bias)[0],
            hf8(0, 0b1111, 0b111),
            "HF8 nsat bias overflow -> NaN"
        );
        assert_eq!(
            cvtbiaspss_hf8(a, bias)[0],
            hf8(0, 0b1111, 0b110),
            "HF8 sat -> +448"
        );
        assert_eq!(
            cvtbiaspss_hf8(a, bias)[1],
            hf8(1, 0b1111, 0b110),
            "HF8 sat -> -448"
        );
    }

    /// A zero-bias family-B lane reproduces the family-A behaviour ONLY where family A also
    /// truncates — but family A rounds RTNE, so they agree only on exact / round-down inputs.
    /// Here we pin that an EXACT-representable value gives the same byte through both family A
    /// (RTNE) and family B (bias=0 truncate), since there are no discarded bits to round
    /// (`[avx10-v2-aux-ocp-conversions.CVT_BIAS_PS_FP8.4]`).
    #[test]
    fn known_value_bias_zero_matches_family_a_on_exact() {
        let mut a = [0.0f32; 16];
        a[0] = 1.0;
        a[1] = -1.5;
        a[2] = 2.0;
        let zero = [0i32; 16];
        // Exact values: RTNE and zero-bias-truncate agree byte-for-byte.
        assert_eq!(
            cvtbiasps_bf8(a, zero),
            cvtps_bf8(a),
            "BF8 exact: bias0 == RTNE"
        );
        assert_eq!(
            cvtbiasps_hf8(a, zero),
            cvtps_hf8(a),
            "HF8 exact: bias0 == RTNE"
        );
    }

    /// Regression: the section-16.1 E4M3 Bias branch FLUSHES underflow (`newexp <= 0`) to
    /// signed zero — it has NO subnormal-truncate block, unlike the E5M2 Bias branch and
    /// the E4M3 RTNE/RTO branches. The spec is deliberately asymmetric here: even a value
    /// exactly representable as an E4M3 subnormal (e.g. `2^-7 = 4 * 2^-9` = `S.0000.100`)
    /// flushes to zero under Bias rounding.
    #[test]
    fn known_value_bias_hf8_underflow_flushes_to_zero() {
        let mut a = [0.0f32; 16];
        a[0] = f32::from_bits(120u32 << 23); // +2^-7, exact E4M3 subnormal — still flushes
        a[1] = -a[0];
        a[2] = f32::from_bits((119u32 << 23) | 0x40_0000); // +3*2^-9, exact — still flushes
        a[3] = f32::from_bits(117u32 << 23); // +2^-10, far below the window
        let zero = [0i32; 16];

        let out = cvtbiasps_hf8(a, zero);
        assert_eq!(out[0], hf8(0, 0b0000, 0b000), "+2^-7 -> +0 (no subnormal block)");
        assert_eq!(out[1], hf8(1, 0b0000, 0b000), "-2^-7 -> -0 (sign kept)");
        assert_eq!(out[2], hf8(0, 0b0000, 0b000), "+3*2^-9 -> +0");
        assert_eq!(out[3], hf8(0, 0b0000, 0b000), "+2^-10 -> +0");
        // Saturating variant shares the underflow path.
        assert_eq!(
            cvtbiaspss_hf8(a, zero)[0],
            hf8(0, 0b0000, 0b000),
            "saturating variant flushes to zero too"
        );
    }

    /// Regression: the section-16.1 E4M3 Bias branch has NO saturating NaN-slot clamp —
    /// that clamp exists only in the RTNE/RTO branches. A finite input whose biased
    /// mantissa truncates into `e_o = 0xF, m_o = 0x7` (inputs in `[480, 512)`) yields the
    /// NaN encoding `S.1111.111` in BOTH the saturating and non-saturating Bias variants.
    #[test]
    fn known_value_bias_hf8_nan_slot_is_not_clamped() {
        let mut a = [0.0f32; 16];
        a[0] = 480.0;
        a[1] = -480.0;
        a[2] = 500.0; // still in [480, 512): truncates to the same slot
        let zero = [0i32; 16];

        let sat = cvtbiaspss_hf8(a, zero);
        assert_eq!(sat[0], hf8(0, 0b1111, 0b111), "sat +480 -> NaN-coded (no clamp)");
        assert_eq!(sat[1], hf8(1, 0b1111, 0b111), "sat -480 -> NaN-coded (no clamp)");
        assert_eq!(sat[2], hf8(0, 0b1111, 0b111), "sat +500 -> NaN-coded (no clamp)");

        let nsat = cvtbiasps_hf8(a, zero);
        assert_eq!(nsat[0], hf8(0, 0b1111, 0b111), "nsat +480 -> NaN-coded");
        assert_eq!(nsat[1], hf8(1, 0b1111, 0b111), "nsat -480 -> NaN-coded");
    }
}

/// Property tests for families A and B. The known-value tests above pin specific bytes; these
/// assert the cross-cutting invariants over random FP32 (and, for family B, random `i32` bias)
/// inputs (every bit pattern, including subnormals / signed zeros / NaNs / Inf / overflow).
#[cfg(test)]
mod proptests {
    use super::*;
    use quickcheck::{quickcheck, Arbitrary, Gen};

    /// 16 random FP32 lanes. We mix fully-random bit patterns (reaching NaN/Inf/subnormal/
    /// overflow magnitudes) with the occasional "nice" finite value so in-range rounding is
    /// well exercised too.
    #[derive(Clone, Debug)]
    struct Inputs {
        a: [f32; 16],
    }

    impl Arbitrary for Inputs {
        fn arbitrary(g: &mut Gen) -> Self {
            Inputs {
                a: core::array::from_fn(|_| arb_f32(g)),
            }
        }
    }

    /// 16 random FP32 lanes plus 16 random `i32` bias words, for the family-B properties.
    #[derive(Clone, Debug)]
    struct BiasInputs {
        a: [f32; 16],
        bias: [i32; 16],
    }

    impl Arbitrary for BiasInputs {
        fn arbitrary(g: &mut Gen) -> Self {
            BiasInputs {
                a: core::array::from_fn(|_| arb_f32(g)),
                // Bias words span the full i32 range (the front-end masks to 21/20 bits), with
                // small values mixed in so the carry-out-of-bit-23 edge is exercised.
                bias: core::array::from_fn(|_| {
                    if bool::arbitrary(g) {
                        i32::arbitrary(g)
                    } else {
                        (u8::arbitrary(g) as i32) << 16
                    }
                }),
            }
        }
    }

    /// A single random FP32 lane: half fully-random bit patterns (NaN/Inf/subnormal/overflow),
    /// half a finite value scaled across the interesting magnitude range.
    fn arb_f32(g: &mut Gen) -> f32 {
        if bool::arbitrary(g) {
            f32::from_bits(u32::arbitrary(g))
        } else {
            let m = i16::arbitrary(g) as f32 / 17.0;
            let e = (i8::arbitrary(g) % 40) as i32;
            m * 2.0f32.powi(e)
        }
    }

    // The six family-A public forms paired with their oracles, for the public==oracle and
    // differential-discard properties.
    type Conv = (fn([f32; 16]) -> [u8; 16], fn([f32; 16]) -> [u8; 16]);
    fn all_convs() -> [Conv; 6] {
        [
            (cvtps_bf8, cvtps_bf8_scalar),
            (cvtpss_bf8, cvtpss_bf8_scalar),
            (cvtps_hf8, cvtps_hf8_scalar),
            (cvtpss_hf8, cvtpss_hf8_scalar),
            (cvtrops_hf8, cvtrops_hf8_scalar),
            (cvtropss_hf8, cvtropss_hf8_scalar),
        ]
    }

    // The four family-B public forms paired with their oracles.
    type BiasConv = (
        fn([f32; 16], [i32; 16]) -> [u8; 16],
        fn([f32; 16], [i32; 16]) -> [u8; 16],
    );
    fn all_bias_convs() -> [BiasConv; 4] {
        [
            (cvtbiasps_bf8, cvtbiasps_bf8_scalar),
            (cvtbiaspss_bf8, cvtbiaspss_bf8_scalar),
            (cvtbiasps_hf8, cvtbiasps_hf8_scalar),
            (cvtbiaspss_hf8, cvtbiaspss_hf8_scalar),
        ]
    }

    quickcheck! {
        /// The public dispatcher equals its scalar oracle on every input, for all six
        /// family-A converters. OQ-5: no native path exists, so the differential test that
        /// would tie a native path to the oracle DISCARDS (there is nothing to differentiate
        /// against); this property is the public-vs-oracle contract that survives
        /// (`[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.1]` ..
        /// `[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.6]`,
        /// `[avx10-v2-aux-ocp-conversions.CORRECTNESS.1]`,
        /// `[avx10-v2-aux-ocp-conversions.DETECTION.2]`).
        fn prop_public_matches_scalar(input: Inputs) -> bool {
            all_convs().iter().all(|(pub_fn, ora_fn)| pub_fn(input.a) == ora_fn(input.a))
        }

        /// The public dispatcher equals its scalar oracle for all four **family-B** bias
        /// converters, over random FP32 sources AND random `i32` bias words. OQ-5: the
        /// native bias path is absent (`_mm512_cvtbiasps_bf8` etc. do not exist), so the
        /// differential DISCARDS and this public-vs-oracle property is the surviving contract
        /// (`[avx10-v2-aux-ocp-conversions.CVT_BIAS_PS_FP8.1]`,
        /// `[avx10-v2-aux-ocp-conversions.CVT_BIAS_PS_FP8.2]`,
        /// `[avx10-v2-aux-ocp-conversions.CORRECTNESS.1]`,
        /// `[avx10-v2-aux-ocp-conversions.DETECTION.2]`).
        fn prop_bias_public_matches_scalar(input: BiasInputs) -> bool {
            all_bias_convs()
                .iter()
                .all(|(pub_fn, ora_fn)| pub_fn(input.a, input.bias) == ora_fn(input.a, input.bias))
        }

        /// A **zero bias** family-B convert through the public dispatchers equals the shared
        /// `crate::fp8` Bias-branch oracle called directly with `bias == 0`
        /// (`[avx10-v2-aux-ocp-conversions.CVT_BIAS_PS_FP8.4]`). NOTE: this pins the
        /// dispatcher wiring (lane mapping, bias-word extraction, saturation flag routing)
        /// against the oracle — it is NOT an independent re-derivation of the section-16.1
        /// truncate rule; the known-value tests above pin that rule with literal bytes.
        fn prop_zero_bias_equals_truncate(input: Inputs) -> bool {
            let zero = [0i32; 16];
            (0..16).all(|i| {
                let want_bf = crate::fp8::fp32_to_fp8_e5m2_biased(
                    input.a[i], crate::fp8::Fp8RoundMode::Bias, false, 0);
                let want_bf_s = crate::fp8::fp32_to_fp8_e5m2_biased(
                    input.a[i], crate::fp8::Fp8RoundMode::Bias, true, 0);
                let want_hf = crate::fp8::fp32_to_fp8_e4m3_biased(
                    input.a[i], crate::fp8::Fp8RoundMode::Bias, false, 0);
                let want_hf_s = crate::fp8::fp32_to_fp8_e4m3_biased(
                    input.a[i], crate::fp8::Fp8RoundMode::Bias, true, 0);
                cvtbiasps_bf8(input.a, zero)[i] == want_bf
                    && cvtbiaspss_bf8(input.a, zero)[i] == want_bf_s
                    && cvtbiasps_hf8(input.a, zero)[i] == want_hf
                    && cvtbiaspss_hf8(input.a, zero)[i] == want_hf_s
            })
        }

        /// Across families A AND B, the saturating variant's output magnitude is <= the
        /// non-saturating variant's for the same FP32 input (and, for family B, the same bias)
        /// and target (`[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.7]`,
        /// `[avx10-v2-aux-ocp-conversions.CVT_BIAS_PS_FP8.3]`). Magnitude is the decoded
        /// |value|; a non-saturating Inf/NaN code has magnitude +inf (>= any finite), so the
        /// bound holds. NaN inputs map to NaN in both, which we skip.
        fn prop_saturating_le_nonsaturating(input: BiasInputs) -> bool {
            let a = input.a;
            let b = input.bias;
            // Family A.
            let nsat_bf = cvtps_bf8(a);
            let sat_bf = cvtpss_bf8(a);
            let nsat_hf = cvtps_hf8(a);
            let sat_hf = cvtpss_hf8(a);
            let nsat_ro = cvtrops_hf8(a);
            let sat_ro = cvtropss_hf8(a);
            // Family B.
            let nsat_bbf = cvtbiasps_bf8(a, b);
            let sat_bbf = cvtbiaspss_bf8(a, b);
            let nsat_bhf = cvtbiasps_hf8(a, b);
            let sat_bhf = cvtbiaspss_hf8(a, b);
            (0..16).all(|i| {
                let bf_ok = mag_le(crate::fp8::fp8_e5m2_to_fp32(sat_bf[i]),
                                   crate::fp8::fp8_e5m2_to_fp32(nsat_bf[i]));
                let hf_ok = mag_le(crate::fp8::fp8_e4m3_to_fp32(sat_hf[i]),
                                   crate::fp8::fp8_e4m3_to_fp32(nsat_hf[i]));
                let ro_ok = mag_le(crate::fp8::fp8_e4m3_to_fp32(sat_ro[i]),
                                   crate::fp8::fp8_e4m3_to_fp32(nsat_ro[i]));
                let bbf_ok = mag_le(crate::fp8::fp8_e5m2_to_fp32(sat_bbf[i]),
                                    crate::fp8::fp8_e5m2_to_fp32(nsat_bbf[i]));
                let bhf_ok = mag_le(crate::fp8::fp8_e4m3_to_fp32(sat_bhf[i]),
                                    crate::fp8::fp8_e4m3_to_fp32(nsat_bhf[i]));
                bf_ok && hf_ok && ro_ok && bbf_ok && bhf_ok
            })
        }

        /// Lane independence (`[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.10]`): replacing one
        /// input lane changes only the corresponding output lane. We rebuild the input with
        /// lane `idx` perturbed and confirm every OTHER output byte is unchanged, for both
        /// families (family B also perturbs only its bias lane `idx`).
        fn prop_lane_independence(input: BiasInputs, idx: u8) -> bool {
            let i = (idx as usize) % 16;
            // Family A: perturb just lane i of the source.
            let mut a2 = input.a;
            a2[i] = if a2[i].to_bits() == 0 { 1.0 } else { 0.0 };
            let fam_a_ok = all_convs().iter().all(|(pub_fn, _)| {
                let oa = pub_fn(input.a);
                let ob = pub_fn(a2);
                (0..16).all(|j| j == i || oa[j] == ob[j])
            });
            // Family B: perturb just the bias lane i (source held fixed).
            let mut bias2 = input.bias;
            bias2[i] = bias2[i].wrapping_add(0x10_0000);
            let fam_b_ok = all_bias_convs().iter().all(|(pub_fn, _)| {
                let oa = pub_fn(input.a, input.bias);
                let ob = pub_fn(input.a, bias2);
                (0..16).all(|j| j == i || oa[j] == ob[j])
            });
            fam_a_ok && fam_b_ok
        }

        /// RTO never selects an even target mantissa when the FP32 value is inexact in E4M3
        /// (`[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.4]`, spec section 2.6.2). For each finite
        /// non-overflowing lane, if the RTNE result re-decoded to FP32 differs from the input
        /// (i.e. the value is inexact in E4M3), the RTO byte's mantissa lsb must be 1.
        fn prop_rto_odd_on_inexact(input: Inputs) -> bool {
            let rto = cvtrops_hf8(input.a);
            (0..16).all(|i| {
                let x = input.a[i];
                if !x.is_finite() { return true; }
                let byte = rto[i];
                let exp = (byte >> 3) & 0xF;
                // Skip overflow (NaN-coded S.1111.111) and zero/subnormal-flushed lanes:
                // RTO oddness applies to representable inexact magnitudes.
                if exp == 0xF && (byte & 0x7) == 0x7 { return true; }
                if (byte & 0x7F) == 0 { return true; } // ±0 (DAZ-flushed or true zero)
                // Inexact iff the value does not equal its E4M3 re-decode under RTNE.
                let rtne_val = crate::fp8::fp8_e4m3_to_fp32(
                    crate::fp8::fp32_to_fp8_e4m3(x, crate::fp8::Fp8RoundMode::Rtne, false));
                let inexact = rtne_val.to_bits() != x.to_bits() && {
                    // exact-representable check: x equals SOME E4M3 value's decode.
                    let direct = crate::fp8::fp8_e4m3_to_fp32(byte);
                    direct.to_bits() != x.to_bits()
                };
                if inexact {
                    (byte & 0x1) == 1
                } else {
                    true
                }
            })
        }
    }

    /// |a| <= |b| treating NaN as unordered-skip and +/-Inf as the largest magnitude.
    fn mag_le(a: f32, b: f32) -> bool {
        if a.is_nan() || b.is_nan() {
            return true; // NaN has no ordered magnitude; the bound is asserted elsewhere.
        }
        a.abs() <= b.abs()
    }
}

/// Native-vs-oracle differential for families A and B (FP32 -> FP8). Phase 11 cross-cutting.
///
/// Families A/B ship **oracle-only** in this toolchain (OQ-5: none of the `_mm512_cvtps_bf8`
/// / `_mm512_cvtbiasps_bf8` family of FP32->FP8 intrinsics compile under `-mavx10.2`). This
/// property compares the public dispatcher to its scalar oracle bit-for-bit under
/// `feature="native"` on AVX10_V2_AUX hardware — the live differential the instant a shim
/// lands (`[avx10-v2-aux-ocp-conversions.DIFFERENTIAL.1]`) — and `TestResult::discard()`s
/// (never `from_bool(false)`) when the feature/hardware is absent, so a fallback-only runner
/// cannot produce a vacuous green.
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
        a: [f32; 16],
        bias: [i32; 16],
    }

    impl Arbitrary for Inputs {
        fn arbitrary(g: &mut Gen) -> Self {
            Inputs {
                a: core::array::from_fn(|_| f32::from_bits(u32::arbitrary(g))),
                bias: core::array::from_fn(|_| i32::arbitrary(g)),
            }
        }
    }

    quickcheck! {
        /// Families-A/B native-vs-oracle differential. Under `feature="native"` on x86_64 with
        /// `AVX10_V2_AUX` detected, every family-A single-source and family-B bias dispatcher
        /// must equal its scalar oracle bit-for-bit
        /// (`[avx10-v2-aux-ocp-conversions.DIFFERENTIAL.1]`). DISCARDED (not failed) when the
        /// feature or hardware is absent (`[avx10-v2-aux-ocp-conversions.CORRECTNESS.2]`).
        fn prop_native_matches_oracle(input: Inputs) -> TestResult {
            #[cfg(all(target_arch = "x86_64", feature = "native"))]
            {
                if detect::has_avx10_v2_aux() {
                    let a = input.a;
                    let b = input.bias;
                    let fam_a = cvtps_bf8(a) == cvtps_bf8_scalar(a)
                        && cvtpss_bf8(a) == cvtpss_bf8_scalar(a)
                        && cvtps_hf8(a) == cvtps_hf8_scalar(a)
                        && cvtpss_hf8(a) == cvtpss_hf8_scalar(a)
                        && cvtrops_hf8(a) == cvtrops_hf8_scalar(a)
                        && cvtropss_hf8(a) == cvtropss_hf8_scalar(a);
                    let fam_b = cvtbiasps_bf8(a, b) == cvtbiasps_bf8_scalar(a, b)
                        && cvtbiaspss_bf8(a, b) == cvtbiaspss_bf8_scalar(a, b)
                        && cvtbiasps_hf8(a, b) == cvtbiasps_hf8_scalar(a, b)
                        && cvtbiaspss_hf8(a, b) == cvtbiaspss_hf8_scalar(a, b);
                    return TestResult::from_bool(fam_a && fam_b);
                }
            }
            let _ = &input;
            TestResult::discard()
        }
    }
}
