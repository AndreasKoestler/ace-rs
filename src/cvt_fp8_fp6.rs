//! Families F and G (AVX10_V2_AUX): FP8 <-> FP6 converts.
//!
//! Family F (saturating-RTNE FP8 -> FP6, 6-bit packed): `cvtf8_bf6s` converts 64 BF8 (FP8
//! E5M2) bytes to 64 FP6 E3M2 (BF6) lanes and `cvtf8_hf6s` converts 64 HF8 (FP8 E4M3) bytes
//! to 64 FP6 E2M3 (HF6) lanes. Per ACE v1 spec section 9.6 (`VCVTBF82BF6S` / `VCVTHF82HF6S`)
//! the conversion rounds **RTNE** and is **always saturating**: an FP8 input whose magnitude
//! exceeds the target FP6 max normal — FP6 E3M2 `+/-28.0`, FP6 E2M3 `+/-7.5` (spec section
//! 2.4.2) — clamps to the same-signed FP6 max normal, including BF8 +/-Inf/NaN and the HF8
//! NaN/max-exponent binade, which FP6 cannot represent
//! (`[avx10-v2-aux-ocp-conversions.CVT_FP8_FP6.1]`,
//! `[avx10-v2-aux-ocp-conversions.CVT_FP8_FP6.2]`). DAZ=1/FTZ=0, MXCSR not consulted, no
//! floating-point exceptions (spec section 9.6.1).
//!
//! The supported pairs were chosen to **match source/target mantissa width** — E5M2 (2
//! mantissa bits) -> E3M2 (2), E4M3 (3) -> E2M3 (3) — so **no mantissa precision is lost**;
//! only exponent-range narrowing can round, and every FP8 subnormal lies far below the
//! smallest FP6 subnormal midpoint and converts to the same-signed FP6 zero (spec section
//! 9.6.1 note, `[avx10-v2-aux-ocp-conversions.CVT_FP8_FP6.3]`).
//!
//! The family-F output is **6-bit-packed** at `VL*6/8` bytes: 64 source bytes (`[u8; 64]`)
//! produce 48 packed bytes (`[u8; 48]`), FP6 lane `i` at bit offset `6 * i` straddling byte
//! boundaries from bit 0 (spec section 9.6.5), every lane written, no masking/zeroing
//! (`[avx10-v2-aux-ocp-conversions.CVT_FP8_FP6.4]`).
//!
//! Family G (exact FP6 -> FP8 E4M3, 6-bit-unpacked): `cvtf6_hf8_e3m2` converts 64
//! 6-bit-packed FP6 E3M2 (BF6) lanes (`[u8; 48]`) to 64 FP8 E4M3 bytes (`[u8; 64]`) and
//! `cvtf6_hf8_e2m3` converts 64 FP6 E2M3 (HF6) lanes likewise. Per ACE v1 spec section 9.7
//! (`VCVTBF62HF8` / `VCVTHF62HF8`) the conversion is **exact** — every one of the 64 FP6
//! encodings (per format) maps to exactly one FP8 E4M3 encoding with no rounding, saturation
//! or approximation (`[avx10-v2-aux-ocp-conversions.CVT_FP6_FP8.1]`,
//! `[avx10-v2-aux-ocp-conversions.CVT_FP6_FP8.2]`). DAZ=0/FTZ=0, MXCSR not consulted, no
//! floating-point exceptions (spec section 9.7.1) — FP6 subnormals are renormalised into the
//! wider format, not flushed. FP6 E3M2 max `+/-28.0` and FP6 E2M3 max `+/-7.5` are both well
//! within the E4M3 range (`+/-448`), so every input is representable. The output is **twice**
//! the input width and the input is read 6-bit-packed from bit 0:
//! `dest.byte[i] = fp6_to_fp8_e4m3(src[6*i+5 : 6*i])` (spec section 9.7.5, KL = VL/8 = 64
//! lanes, 48 input bytes = 64*6/8, `[avx10-v2-aux-ocp-conversions.CVT_FP6_FP8.1]`).
//!
//! Each public dispatcher is a safe fn; with no native group-3 intrinsic available in
//! current toolchains it always takes the scalar oracle (see the OQ-5 note below), with
//! `detect::has_avx10_v2_aux` marking where the native gate goes live
//! (`[avx10-v2-aux-ocp-conversions.DETECTION.2]`). The `_scalar`
//! oracle is the primary, always-correct path on every target including non-x86
//! (`[avx10-v2-aux-ocp-conversions.CORRECTNESS.1]`,
//! `[avx10-v2-aux-ocp-conversions.CORRECTNESS.2]`); it carries no cfg gate, reads no global
//! state, and the dispatcher equals it bit-for-bit. The two family-F converts have distinct
//! public names (`cvtf8_bf6s` / `cvtf8_hf6s`) and the two family-G converts carry a
//! source-format suffix (`_e3m2` / `_e2m3`) because both target FP8 E4M3 (OQ-3). The whole
//! module compiles on stable Rust with no `core::simd`/nightly
//! (`[avx10-v2-aux-ocp-conversions.STABLE_RUST.1]`).
//!
//! OQ-5 (intrinsic unavailable -> oracle-only): ALL FOUR converts (`cvtf8_bf6s`,
//! `cvtf8_hf6s`, `cvtf6_hf8_e3m2`, `cvtf6_hf8_e2m3`) ship oracle-only. The
//! `_mm512_cvtf8_bf6s` (`VCVTBF82BF6S`) / `_mm512_cvtf8_hf6s` (`VCVTHF82HF6S`) and the
//! `_mm512_cvtf6_hf8` family (`VCVTBF62HF8` / `VCVTHF62HF8`) intrinsics are ABSENT from the
//! installed GCC 16.1.1 `-mavx10.2` headers (verified by compile probes — every naming
//! variant `_mm512_cvtf6_hf8` / `_mm512_cvtf62hf8` / `_mm512_cvtbf62hf8` / `_mm512_cvthf62hf8`
//! / `_mm512_cvtbf6_hf8` / `_mm512_cvthf6_hf8` is rejected with `error: implicit declaration
//! of function ... did you mean '_mm512_cvtph_hf8'?`). Per OQ-5 there is therefore no native C
//! shim and no `extern "C"` declaration for any of them, and each dispatcher resolves to its
//! `_scalar` sibling on every target. The capability check
//! `crate::detect::has_avx10_v2_aux` is never consulted — with no native path there is
//! nothing to gate; each dispatcher only references the detector to mark the future gate
//! site. A native path is added once the intrinsics land in the toolchain. The differential
//! test that would otherwise tie a native path to the
//! oracle DISCARDS (no native path exists), so correctness is grounded against the
//! section-9.6.5 / section-9.7.5 / section-16.3 (FP8->FP6) / section-16.4 (FP6->FP8)
//! pseudocode transcribed in `crate::fp6`.

use crate::detect;
use crate::fp6;

/// BF8 (FP8 E5M2) -> FP6 E3M2 saturating-RTNE convert, 6-bit packed (64 lanes -> 48 bytes).
///
/// Per BF8 lane: convert `a[i]` to its FP6 E3M2 code via `fp6::fp8_e5m2_to_fp6_e3m2`
/// (always-saturating, RTNE, matched mantissa width), then 6-bit-pack into `[u8; 48]` at bit
/// offset `6 * i` (spec section 9.6.5). Always saturating: magnitudes above `+/-28.0`, and BF8
/// +/-Inf/NaN, clamp to the same-signed FP6 E3M2 max normal `S.111.11`
/// (`[avx10-v2-aux-ocp-conversions.CVT_FP8_FP6.2]`). Every BF8 subnormal -> FP6 same-signed
/// zero (`[avx10-v2-aux-ocp-conversions.CVT_FP8_FP6.3]`).
///
/// No native path is wired yet (OQ-5, see the module docs), so
/// `detect::has_avx10_v2_aux` is never consulted; the dispatcher resolves to [`cvtf8_bf6s_scalar`]
/// on every target, returning the spec-defined value.
/// `[avx10-v2-aux-ocp-conversions.DETECTION.2]`
pub fn cvtf8_bf6s(a: [u8; 64]) -> [u8; 48] {
    // No native path this phase (OQ-5): the FP8->FP6 intrinsic is absent from the `-mavx10.2`
    // toolchain, so even under `feature="native"` on AVX10_V2_AUX hardware the oracle is the
    // only path. The detector is only referenced
    // (never called), marking the gate site for the shim once the intrinsic lands.
    let _ = detect::has_avx10_v2_aux; // reference (not call) the future gate; see fn docs
    cvtf8_bf6s_scalar(a)
}

/// Portable reference oracle for [`cvtf8_bf6s`] — the primary always-correct path.
///
/// Maps each BF8 lane through `fp6::fp8_e5m2_to_fp6_e3m2` (the saturating-RTNE section-16.3
/// helper), then 6-bit-packs all 64 results into 48 bytes via `fp6::pack` (every lane
/// written, no masking). Carries no cfg gate and reads no global state.
/// `[avx10-v2-aux-ocp-conversions.CORRECTNESS.1]` `[avx10-v2-aux-ocp-conversions.CORRECTNESS.2]`
/// `[avx10-v2-aux-ocp-conversions.CVT_FP8_FP6.4]`
pub fn cvtf8_bf6s_scalar(a: [u8; 64]) -> [u8; 48] {
    let lanes: [u8; 64] = core::array::from_fn(|i| fp6::fp8_e5m2_to_fp6_e3m2(a[i]));
    let mut out = [0u8; 48];
    fp6::pack(&lanes, &mut out);
    out
}

/// HF8 (FP8 E4M3) -> FP6 E2M3 saturating-RTNE convert, 6-bit packed (64 lanes -> 48 bytes).
///
/// Per HF8 lane: convert `a[i]` to its FP6 E2M3 code via `fp6::fp8_e4m3_to_fp6_e2m3`
/// (always-saturating, RTNE, matched mantissa width), then 6-bit-pack into `[u8; 48]` at bit
/// offset `6 * i` (spec section 9.6.5). Always saturating: magnitudes above `+/-7.5`, and the
/// HF8 NaN/max-exponent binade, clamp to the same-signed FP6 E2M3 max normal `S.11.111`
/// (`[avx10-v2-aux-ocp-conversions.CVT_FP8_FP6.2]`). Every HF8 subnormal -> FP6 same-signed
/// zero (`[avx10-v2-aux-ocp-conversions.CVT_FP8_FP6.3]`).
///
/// No native path is wired yet (OQ-5, see the module docs), so
/// `detect::has_avx10_v2_aux` is never consulted; the dispatcher resolves to [`cvtf8_hf6s_scalar`]
/// on every target, returning the spec-defined value.
/// `[avx10-v2-aux-ocp-conversions.DETECTION.2]`
pub fn cvtf8_hf6s(a: [u8; 64]) -> [u8; 48] {
    // No native path this phase (OQ-5); see the module docs and `cvtf8_bf6s`.
    let _ = detect::has_avx10_v2_aux; // reference (not call) the future gate; see fn docs
    cvtf8_hf6s_scalar(a)
}

/// Portable reference oracle for [`cvtf8_hf6s`] — the primary always-correct path.
///
/// Maps each HF8 lane through `fp6::fp8_e4m3_to_fp6_e2m3` (the saturating-RTNE section-16.3
/// helper), then 6-bit-packs all 64 results into 48 bytes via `fp6::pack` (every lane
/// written, no masking). Carries no cfg gate and reads no global state.
/// `[avx10-v2-aux-ocp-conversions.CORRECTNESS.1]` `[avx10-v2-aux-ocp-conversions.CORRECTNESS.2]`
/// `[avx10-v2-aux-ocp-conversions.CVT_FP8_FP6.4]`
pub fn cvtf8_hf6s_scalar(a: [u8; 64]) -> [u8; 48] {
    let lanes: [u8; 64] = core::array::from_fn(|i| fp6::fp8_e4m3_to_fp6_e2m3(a[i]));
    let mut out = [0u8; 48];
    fp6::pack(&lanes, &mut out);
    out
}

/// FP6 E3M2 (BF6) -> FP8 E4M3 (HF8) exact convert, 6-bit-unpacked (64 lanes -> 64 bytes).
///
/// Per FP6 lane: read the 6-bit slice at bit offset `6 * i` from the 6-bit-packed input and
/// map it to its exact FP8 E4M3 byte via `fp6::fp6_e3m2_to_fp8_e4m3` (the section-9.7.5
/// rebias/mantissa-shift decode). The conversion is **exact** — every one of the 64 FP6 E3M2
/// encodings maps to exactly one FP8 E4M3 encoding, no rounding/approximation
/// (`[avx10-v2-aux-ocp-conversions.CVT_FP6_FP8.1]`,
/// `[avx10-v2-aux-ocp-conversions.CVT_FP6_FP8.2]`). FP6 E3M2 max `+/-28.0` is well within the
/// E4M3 range. The output (`[u8; 64]`) is twice the 6-bit-packed input width (`[u8; 48]`), the
/// input read from bit 0 (spec section 9.7.5).
///
/// No native path is wired yet (OQ-5, see the module docs), so
/// `detect::has_avx10_v2_aux` is never consulted; the dispatcher resolves to [`cvtf6_hf8_e3m2_scalar`]
/// on every target, returning the spec-defined value.
/// `[avx10-v2-aux-ocp-conversions.DETECTION.2]`
pub fn cvtf6_hf8_e3m2(a: [u8; 48]) -> [u8; 64] {
    // No native path this phase (OQ-5): `_mm512_cvtf6_hf8` (`VCVTBF62HF8`) is absent from the
    // `-mavx10.2` toolchain (GCC 16.1.1), so even under `feature="native"` on AVX10_V2_AUX
    // hardware the oracle is the only path. The detector is only referenced
    // (never called), marking the gate site for the shim once the intrinsic lands.
    let _ = detect::has_avx10_v2_aux; // reference (not call) the future gate; see fn docs
    cvtf6_hf8_e3m2_scalar(a)
}

/// Portable reference oracle for [`cvtf6_hf8_e3m2`] — the primary always-correct path.
///
/// Unpacks each 6-bit FP6 E3M2 lane via `fp6::unpack` (right-aligning the lane's 6 bits) and
/// maps it through the exact section-9.7.5 decode `fp6::fp6_e3m2_to_fp8_e4m3`, widening 48
/// packed input bytes into 64 FP8 E4M3 output bytes. Carries no cfg gate and reads no global
/// state. `[avx10-v2-aux-ocp-conversions.CORRECTNESS.1]`
/// `[avx10-v2-aux-ocp-conversions.CORRECTNESS.2]` `[avx10-v2-aux-ocp-conversions.CVT_FP6_FP8.1]`
pub fn cvtf6_hf8_e3m2_scalar(a: [u8; 48]) -> [u8; 64] {
    let mut codes = [0u8; 64];
    fp6::unpack(&a, &mut codes);
    core::array::from_fn(|i| fp6::fp6_e3m2_to_fp8_e4m3(codes[i]))
}

/// FP6 E2M3 (HF6) -> FP8 E4M3 (HF8) exact convert, 6-bit-unpacked (64 lanes -> 64 bytes).
///
/// Per FP6 lane: read the 6-bit slice at bit offset `6 * i` from the 6-bit-packed input and
/// map it to its exact FP8 E4M3 byte via `fp6::fp6_e2m3_to_fp8_e4m3` (the section-9.7.5
/// rebias/mantissa-shift decode). The conversion is **exact** — every one of the 64 FP6 E2M3
/// encodings maps to exactly one FP8 E4M3 encoding, no rounding/approximation
/// (`[avx10-v2-aux-ocp-conversions.CVT_FP6_FP8.1]`,
/// `[avx10-v2-aux-ocp-conversions.CVT_FP6_FP8.2]`). FP6 E2M3 max `+/-7.5` is well within the
/// E4M3 range. The output (`[u8; 64]`) is twice the 6-bit-packed input width (`[u8; 48]`), the
/// input read from bit 0 (spec section 9.7.5).
///
/// No native path is wired yet (OQ-5, see the module docs), so
/// `detect::has_avx10_v2_aux` is never consulted; the dispatcher resolves to [`cvtf6_hf8_e2m3_scalar`]
/// on every target, returning the spec-defined value.
/// `[avx10-v2-aux-ocp-conversions.DETECTION.2]`
pub fn cvtf6_hf8_e2m3(a: [u8; 48]) -> [u8; 64] {
    // No native path this phase (OQ-5): `_mm512_cvtf6_hf8` (`VCVTHF62HF8`) is absent from the
    // `-mavx10.2` toolchain (GCC 16.1.1); see the module docs and `cvtf6_hf8_e3m2`.
    let _ = detect::has_avx10_v2_aux; // reference (not call) the future gate; see fn docs
    cvtf6_hf8_e2m3_scalar(a)
}

/// Portable reference oracle for [`cvtf6_hf8_e2m3`] — the primary always-correct path.
///
/// Unpacks each 6-bit FP6 E2M3 lane via `fp6::unpack` (right-aligning the lane's 6 bits) and
/// maps it through the exact section-9.7.5 decode `fp6::fp6_e2m3_to_fp8_e4m3`, widening 48
/// packed input bytes into 64 FP8 E4M3 output bytes. Carries no cfg gate and reads no global
/// state. `[avx10-v2-aux-ocp-conversions.CORRECTNESS.1]`
/// `[avx10-v2-aux-ocp-conversions.CORRECTNESS.2]` `[avx10-v2-aux-ocp-conversions.CVT_FP6_FP8.1]`
pub fn cvtf6_hf8_e2m3_scalar(a: [u8; 48]) -> [u8; 64] {
    let mut codes = [0u8; 64];
    fp6::unpack(&a, &mut codes);
    core::array::from_fn(|i| fp6::fp6_e2m3_to_fp8_e4m3(codes[i]))
}

#[cfg(test)]
mod tests {
    use super::*;

    // BF8 (E5M2) byte assembler: sign | 5-bit exp | 2-bit mantissa.
    fn bf8(sign: u8, exp: u8, mant: u8) -> u8 {
        (sign << 7) | (exp << 2) | mant
    }
    // HF8 (E4M3) byte assembler: sign | 4-bit exp | 3-bit mantissa.
    fn hf8(sign: u8, exp: u8, mant: u8) -> u8 {
        (sign << 7) | (exp << 3) | mant
    }
    // FP6 E3M2 code assembler: sign | 3-bit exp | 2-bit mantissa.
    fn bf6(sign: u8, exp: u8, mant: u8) -> u8 {
        (sign << 5) | (exp << 2) | mant
    }
    // FP6 E2M3 code assembler: sign | 2-bit exp | 3-bit mantissa.
    fn hf6(sign: u8, exp: u8, mant: u8) -> u8 {
        (sign << 5) | (exp << 3) | mant
    }
    // Read FP6 lane `i` (6 bits) back out of a 6-bit-packed `[u8; 48]` output.
    fn lane(out: &[u8; 48], i: usize) -> u8 {
        crate::fp4::extract_field(out, 6 * i, 6)
    }
    // Pack FP6 codes (right-aligned in `[5:0]`) into a `[u8; 48]` family-G input: lane `i` at
    // bit offset `6*i`.
    fn pack6(codes: &[u8; 64]) -> [u8; 48] {
        let mut out = [0u8; 48];
        crate::fp6::pack(codes, &mut out);
        out
    }

    /// Hand-computed family-F E5M2->E3M2 known values pinning the saturation, exact-max-normal,
    /// Inf/NaN-clamp, subnormal->+/-0, and mantissa-preserved cases, plus the 6-bit-packed
    /// cross-byte layout (`[avx10-v2-aux-ocp-conversions.CVT_FP8_FP6.1]`, `.2`, `.3`, `.4`).
    /// The differential native check discards in this environment (no `_mm512_cvtf8_bf6s`
    /// intrinsic, OQ-5), so these spec-grounded vectors are the correctness evidence.
    ///
    /// DISCRIMINATING lanes (each rules out a plausible-but-wrong model):
    ///  * lane 0 = BF8 +28.0 (S.10011.11) -> E3M2 S.111.11 (+28.0) *exactly* — NOT saturation:
    ///    a model that clamped every large value would still land here, but a model that
    ///    overflowed the exponent or dropped the mantissa would not. Distinguishes exact-rebias.
    ///  * lane 1 = BF8 +32.0 (S.10100.00, 2^5 > 28.0) -> S.111.11 (saturation): a non-saturating
    ///    model would overflow the 3-bit FP6 exponent rather than clamp.
    ///  * lane 2 = BF8 +Inf (S.11111.00) -> S.111.11 (+28.0): FP6 has no Inf.
    ///  * lane 3 = BF8 min subnormal +2^-16 (S.00000.01) -> +0 (section 9.6.1 note), NOT an FP6
    ///    subnormal: rules out a DAZ=0 decode that would try to represent it.
    ///  * lane 4 = BF8 +1.25 (S.01111.01) -> E3M2 S.011.01: the 2-bit mantissa survives the
    ///    rebias unchanged (no mantissa loss), ruling out a truncate-mantissa model.
    #[test]
    fn known_value_e5m2_six_bit_packed() {
        let mut a = [0u8; 64];
        a[0] = bf8(0, 0b10011, 0b11); // +28.0 -> S.111.11 (exact max normal)
        a[1] = bf8(0, 0b10100, 0b00); // +32.0 -> S.111.11 (saturation)
        a[2] = bf8(0, 0b11111, 0b00); // +Inf  -> S.111.11 (clamp)
        a[3] = bf8(0, 0b00000, 0b01); // +2^-16 subnormal -> +0
        a[4] = bf8(0, 0b01111, 0b01); // +1.25 -> S.011.01 (mantissa preserved)
        a[5] = bf8(1, 0b11110, 0b11); // -57344 (BF8 max normal) -> -28.0 (saturation)

        let out = cvtf8_bf6s(a);

        assert_eq!(
            lane(&out, 0),
            bf6(0, 0b111, 0b11),
            "+28.0 maps exactly to E3M2 max normal +28.0 (not a clamp)"
        );
        assert_eq!(
            lane(&out, 1),
            bf6(0, 0b111, 0b11),
            "+32.0 saturates to E3M2 max normal +28.0"
        );
        assert_eq!(
            lane(&out, 2),
            bf6(0, 0b111, 0b11),
            "+Inf clamps to E3M2 max normal +28.0"
        );
        assert_eq!(
            lane(&out, 3),
            bf6(0, 0b000, 0b00),
            "BF8 subnormal +2^-16 -> FP6 +0"
        );
        assert_eq!(
            lane(&out, 4),
            bf6(0, 0b011, 0b01),
            "+1.25 -> E3M2 S.011.01 (mantissa preserved)"
        );
        assert_eq!(
            lane(&out, 5),
            bf6(1, 0b111, 0b11),
            "-57344 saturates to E3M2 max normal -28.0"
        );

        // 6-BIT CROSS-BYTE LAYOUT: lane 0 = S.111.11 = 0b011111 = 0x1F occupies bits [5:0] of
        // byte 0. lane 1 = 0x1F occupies bits [11:6]: its low 2 bits land in byte 0 [7:6] and
        // its high 4 bits in byte 1 [3:0]. So byte 0 = 0x1F | (0x1F << 6 & 0xFF) =
        // 0x1F | 0xC0 = 0xDF, and byte 1 carries 0x1F >> 2 = 0x07 in its low nibble plus the
        // low 2 bits of lane 2 (also 0x1F) at bits [5:4] -> 0x07 | (0x1F << 4 & 0xFF)... A
        // straddle-ignoring packer (one lane per byte) would fail byte 0. We assert byte 0
        // directly and re-read lanes 1,2 via extract_field above (which exercises the straddle).
        assert_eq!(
            out[0], 0xDF,
            "lane 0 [5:0] + low 2 bits of lane 1 [7:6] pack into byte 0 (6-bit straddle)"
        );
    }

    /// Hand-computed family-F E4M3->E2M3 known values, pinning the same exact-max-normal /
    /// saturation / NaN-clamp / subnormal-flush / mantissa-preserved cases for the E2M3 target.
    ///
    /// DISCRIMINATING lanes:
    ///  * lane 0 = HF8 +7.5 (S.1001.111) -> E2M3 S.11.111 (+7.5) *exactly* (not saturation).
    ///  * lane 1 = HF8 +8.0 (S.1010.000, 2^3 > 7.5) -> S.11.111 (saturation).
    ///  * lane 2 = HF8 NaN (S.1111.111) -> S.11.111 (+7.5): FP6 has no NaN.
    ///  * lane 3 = HF8 min subnormal +2^-9 (S.0000.001) -> +0 (section 9.6.1 note).
    ///  * lane 4 = HF8 +1.625 (S.0111.101) -> E2M3 S.01.101: 3-bit mantissa preserved.
    #[test]
    fn known_value_e4m3_six_bit_packed() {
        let mut a = [0u8; 64];
        a[0] = hf8(0, 0b1001, 0b111); // +7.5 -> S.11.111 (exact max normal)
        a[1] = hf8(0, 0b1010, 0b000); // +8.0 -> S.11.111 (saturation)
        a[2] = hf8(0, 0b1111, 0b111); // NaN  -> S.11.111 (clamp)
        a[3] = hf8(0, 0b0000, 0b001); // +2^-9 subnormal -> +0
        a[4] = hf8(0, 0b0111, 0b101); // +1.625 -> S.01.101 (mantissa preserved)
        a[5] = hf8(1, 0b1111, 0b110); // -448 (HF8 max normal) -> -7.5 (saturation)

        let out = cvtf8_hf6s(a);

        assert_eq!(
            lane(&out, 0),
            hf6(0, 0b11, 0b111),
            "+7.5 maps exactly to E2M3 max normal +7.5 (not a clamp)"
        );
        assert_eq!(
            lane(&out, 1),
            hf6(0, 0b11, 0b111),
            "+8.0 saturates to E2M3 max normal +7.5"
        );
        assert_eq!(
            lane(&out, 2),
            hf6(0, 0b11, 0b111),
            "NaN clamps to E2M3 max normal +7.5"
        );
        assert_eq!(
            lane(&out, 3),
            hf6(0, 0b00, 0b000),
            "HF8 subnormal +2^-9 -> FP6 +0"
        );
        assert_eq!(
            lane(&out, 4),
            hf6(0, 0b01, 0b101),
            "+1.625 -> E2M3 S.01.101 (mantissa preserved)"
        );
        assert_eq!(
            lane(&out, 5),
            hf6(1, 0b11, 0b111),
            "-448 saturates to E2M3 max normal -7.5"
        );
    }

    /// 6-bit-packed width and no-masking: the result is exactly 48 bytes (VL*6/8) for the
    /// 64-byte input (`[avx10-v2-aux-ocp-conversions.CVT_FP8_FP6.4]`), and EVERY 6-bit lane is
    /// written — a non-default input must yield a non-default-everywhere output, never a
    /// partially masked one.
    #[test]
    fn output_is_six_eighths_width_no_masking() {
        // Every source lane = HF8 +1.0 -> E2M3 S.01.000 = 0b001000 = 0x08 on every lane.
        let a = [hf8(0, 0b0111, 0b000); 64];
        let out = cvtf8_hf6s(a);
        assert_eq!(
            out.len(),
            48,
            "output is VL*6/8 = 48 bytes for the 64-byte input"
        );
        // Re-read every lane: all must be the spec conversion (0x08), no lane left unwritten.
        for i in 0..64 {
            assert_eq!(
                lane(&out, i),
                hf6(0, 0b01, 0b000),
                "lane {i} written (no masking)"
            );
        }
    }

    /// EXACTNESS-style round-trip via family F: an FP6-representable HF8 value, converted HF8
    /// -> FP6 E2M3 (`cvtf8_hf6s`) and read back, recovers the FP6 code that family G (FP6 ->
    /// FP8, exact) would map back to the same HF8. Here we check the headline in-range mantissa
    /// is preserved end-to-end through the public packer (no mantissa loss, matched width).
    #[test]
    fn mantissa_preserved_through_public_packer() {
        // HF8 values exactly representable in E2M3 (|x| <= 7.5, exponent in range): their 3-bit
        // mantissa must survive unchanged through the 64-lane convert + 6-bit pack/unpack.
        let samples = [
            (hf8(0, 0b0111, 0b000), hf6(0, 0b01, 0b000)), // 1.0
            (hf8(0, 0b0111, 0b011), hf6(0, 0b01, 0b011)), // 1.375
            (hf8(0, 0b1000, 0b101), hf6(0, 0b10, 0b101)), // 2.625
            (hf8(0, 0b1001, 0b110), hf6(0, 0b11, 0b110)), // 7.0
        ];
        let mut a = [0u8; 64];
        for (i, (src, _)) in samples.iter().enumerate() {
            a[i] = *src;
        }
        let out = cvtf8_hf6s(a);
        for (i, (_, want)) in samples.iter().enumerate() {
            assert_eq!(
                lane(&out, i),
                *want,
                "lane {i}: in-range HF8 preserves its mantissa into E2M3 (no loss)"
            );
        }
    }

    // ===================== Family G: exact FP6 -> FP8 E4M3, 6-bit-unpacked =====================

    // Numeric reference decoders (independent of the production codec) used by the family-G
    // value-exactness assertions. A defect in the codec cannot hide behind a shared bug because
    // these turn an encoding straight into its real number from the section-2.4.1/2.4.2 layout.
    fn e3m2_value(code: u8) -> f64 {
        let s = (code >> 5) & 1;
        let e = (code >> 2) & 0x7;
        let m = code & 0x3;
        let sign = if s == 1 { -1.0 } else { 1.0 };
        let mag = if e == 0 {
            (m as f64 / 4.0) * 2f64.powi(1 - 3)
        } else {
            (1.0 + m as f64 / 4.0) * 2f64.powi(e as i32 - 3)
        };
        sign * mag
    }
    fn e2m3_value(code: u8) -> f64 {
        let s = (code >> 5) & 1;
        let e = (code >> 3) & 0x3;
        let m = code & 0x7;
        let sign = if s == 1 { -1.0 } else { 1.0 };
        let mag = if e == 0 {
            (m as f64 / 8.0) * 2f64.powi(0) // exp = 1 - bias, bias = 1
        } else {
            (1.0 + m as f64 / 8.0) * 2f64.powi(e as i32 - 1)
        };
        sign * mag
    }
    fn e4m3_value(byte: u8) -> f64 {
        let s = (byte >> 7) & 1;
        let e = (byte >> 3) & 0xF;
        let m = byte & 0x7;
        let sign = if s == 1 { -1.0 } else { 1.0 };
        let mag = if e == 0 {
            (m as f64 / 8.0) * 2f64.powi(1 - 7)
        } else {
            (1.0 + m as f64 / 8.0) * 2f64.powi(e as i32 - 7)
        };
        sign * mag
    }

    /// Family-G E3M2 KNOWN VALUE, full source domain: every one of the 64 FP6 E3M2 codes,
    /// placed at lane 0 of the 6-bit-packed input, maps through the public `cvtf6_hf8_e3m2`
    /// dispatcher to the single exact E4M3 byte that the per-code helper produces, and that
    /// byte's real number equals the FP6 code's real number (value-exact widening). The named
    /// headline lane is E3M2 `S.111.11 -> +28.0` (`[avx10-v2-aux-ocp-conversions.CVT_FP6_FP8.1]`,
    /// `[avx10-v2-aux-ocp-conversions.CVT_FP6_FP8.2]`).
    ///
    /// DISCRIMINATING: each expected E4M3 byte is independently recomputed from the section-9.7.5
    /// pseudocode via `fp6::fp6_e3m2_to_fp8_e4m3`, AND the value is independently checked against
    /// the section-2.4.2 decoders `e3m2_value`/`e4m3_value`. A wrong rebias (e.g. forgetting the
    /// 3->7 bias shift) or a wrong subnormal split would change the decoded value and fail. The
    /// differential native tiebreaker is unavailable here (`_mm512_cvtf6_hf8` absent from GCC
    /// 16.1.1 `-mavx10.2`, OQ-5), so this spec-grounded value-preservation check is the
    /// correctness evidence.
    #[test]
    fn known_value_e3m2_to_hf8_full_domain() {
        // The named headline lane first: E3M2 +28.0 (S.111.11) -> E4M3 0x5E (=1.75*2^4=28.0).
        let mut a = [0u8; 64];
        a[0] = bf6(0, 0b111, 0b11);
        let out = cvtf6_hf8_e3m2(pack6(&a));
        assert_eq!(
            out[0],
            hf8(0, 0b1011, 0b110),
            "E3M2 +28.0 (S.111.11) -> E4M3 0x5E (=1.75*2^4=28.0)"
        );
        assert_eq!(
            e4m3_value(out[0]),
            28.0,
            "headline lane value is exactly 28.0"
        );

        // Full source domain: each of the 64 codes at lane 0 -> exact E4M3 byte (helper) AND
        // value-preserving (independent decoders).
        for code in 0u8..64 {
            let mut a = [0u8; 64];
            a[0] = code;
            let out = cvtf6_hf8_e3m2(pack6(&a));
            assert_eq!(
                out[0],
                fp6::fp6_e3m2_to_fp8_e4m3(code),
                "E3M2 code {code:#04x} -> single exact E4M3 byte (per-code helper)"
            );
            assert_eq!(
                e4m3_value(out[0]),
                e3m2_value(code),
                "E3M2 code {code:#04x} (={}) widens value-exactly to E4M3 {:#04x} (={})",
                e3m2_value(code),
                out[0],
                e4m3_value(out[0])
            );
            assert_eq!(
                out[0] >> 7,
                code >> 5,
                "sign preserved for code {code:#04x}"
            );
        }
    }

    /// Family-G E2M3 KNOWN VALUE, full source domain: every one of the 64 FP6 E2M3 codes maps
    /// through the public `cvtf6_hf8_e2m3` dispatcher to the single exact E4M3 byte, value-exact.
    /// Same value-preservation argument as the E3M2 case (`[avx10-v2-aux-ocp-conversions.CVT_FP6_FP8.1]`,
    /// `[avx10-v2-aux-ocp-conversions.CVT_FP6_FP8.2]`).
    #[test]
    fn known_value_e2m3_to_hf8_full_domain() {
        // Headline lane: E2M3 +7.5 (S.11.111) -> E4M3 0x4F (=1.875*2^2=7.5).
        let mut a = [0u8; 64];
        a[0] = hf6(0, 0b11, 0b111);
        let out = cvtf6_hf8_e2m3(pack6(&a));
        assert_eq!(
            out[0],
            hf8(0, 0b1001, 0b111),
            "E2M3 +7.5 (S.11.111) -> E4M3 0x4F (=1.875*2^2=7.5)"
        );
        assert_eq!(
            e4m3_value(out[0]),
            7.5,
            "headline lane value is exactly 7.5"
        );

        for code in 0u8..64 {
            let mut a = [0u8; 64];
            a[0] = code;
            let out = cvtf6_hf8_e2m3(pack6(&a));
            assert_eq!(
                out[0],
                fp6::fp6_e2m3_to_fp8_e4m3(code),
                "E2M3 code {code:#04x} -> single exact E4M3 byte (per-code helper)"
            );
            assert_eq!(
                e4m3_value(out[0]),
                e2m3_value(code),
                "E2M3 code {code:#04x} (={}) widens value-exactly to E4M3 {:#04x} (={})",
                e2m3_value(code),
                out[0],
                e4m3_value(out[0])
            );
            assert_eq!(
                out[0] >> 7,
                code >> 5,
                "sign preserved for code {code:#04x}"
            );
        }
    }

    /// Family-G width and no-masking: 48 packed input bytes widen to exactly 64 E4M3 bytes
    /// (2x), and EVERY output byte is the spec conversion of its 6-bit lane — no byte left at a
    /// default/unwritten value. A swept input exercises a wide spread of FP6 codes per lane.
    #[test]
    fn family_g_output_is_double_width_no_masking() {
        // lane i carries E3M2 code (i % 64); sweep covers all 64 codes.
        let codes: [u8; 64] = core::array::from_fn(|i| (i % 64) as u8);
        let packed = pack6(&codes);
        let out_e3m2 = cvtf6_hf8_e3m2(packed);
        let out_e2m3 = cvtf6_hf8_e2m3(packed);
        assert_eq!(
            out_e3m2.len(),
            64,
            "output is 2x the 48-byte 6-bit-packed input"
        );
        for i in 0..64 {
            assert_eq!(
                out_e3m2[i],
                fp6::fp6_e3m2_to_fp8_e4m3(codes[i]),
                "E3M2 lane {i} byte must equal the per-lane helper (no masking)"
            );
            assert_eq!(
                out_e2m3[i],
                fp6::fp6_e2m3_to_fp8_e4m3(codes[i]),
                "E2M3 lane {i} byte must equal the per-lane helper (no masking)"
            );
        }
    }

    /// EXACTNESS round-trip via family F (`[avx10-v2-aux-ocp-conversions.EXACTNESS.1]`-style):
    /// for an FP6 E2M3 lane that is exactly representable, FP6 -> FP8 E4M3 (`cvtf6_hf8_e2m3`,
    /// exact) -> FP6 E2M3 (`cvtf8_hf6s`, saturating-RTNE) recovers the original FP6 code. Every
    /// FP6 E2M3 value `|x| <= 7.5` is exactly representable, its E4M3 image is exact, and the
    /// reverse RTNE convert is lossless (matched mantissa width) — so the round-trip is the
    /// identity over the full E2M3 domain. (Signed-zero note: FP6 -0 (`0x20`) -> E4M3 -0 -> FP6
    /// -0, sign survives.)
    #[test]
    fn exactness_round_trip_fp6_hf8_fp6_e2m3() {
        // Sweep all 64 E2M3 codes across the 64 lanes.
        let codes: [u8; 64] = core::array::from_fn(|i| (i % 64) as u8);
        let packed = pack6(&codes);
        let hf8_bytes = cvtf6_hf8_e2m3(packed); // [u8; 64], exact FP6 E2M3 -> E4M3
        let back = cvtf8_hf6s(hf8_bytes); // [u8; 48], E4M3 -> FP6 E2M3 (saturating-RTNE)
        for (i, &code) in codes.iter().enumerate() {
            assert_eq!(
                lane(&back, i),
                code,
                "lane {i}: FP6 E2M3 {code:#04x} -> E4M3 -> FP6 must recover the original code"
            );
        }
    }
}

/// Property-based tests for families F (FP8 -> FP6) and G (FP6 -> FP8). The hand-rolled tests
/// above pin specific spec vectors; these assert the invariants across the full input space.
#[cfg(test)]
mod proptests {
    use super::*;
    use quickcheck::{quickcheck, Arbitrary, Gen};

    /// A random 64-lane FP8 input. `quickcheck` does not derive `Arbitrary` for arrays of this
    /// length, so we wrap it and fill each lane independently — every one of the 256 FP8 codes
    /// (zeros, subnormals, normals, +/-Inf for E5M2, NaNs) is reachable per lane.
    #[derive(Clone, Debug)]
    struct Inputs {
        a: [u8; 64],
    }

    impl Arbitrary for Inputs {
        fn arbitrary(g: &mut Gen) -> Self {
            Inputs {
                a: core::array::from_fn(|_| u8::arbitrary(g)),
            }
        }
    }

    /// A random 48-byte 6-bit-packed FP6 input for family G (every one of the 64 FP6 codes is
    /// reachable per 6-bit lane).
    #[derive(Clone, Debug)]
    struct Fp6Inputs {
        a: [u8; 48],
    }

    impl Arbitrary for Fp6Inputs {
        fn arbitrary(g: &mut Gen) -> Self {
            Fp6Inputs {
                a: core::array::from_fn(|_| u8::arbitrary(g)),
            }
        }
    }

    // Read FP6 lane `i` (6 bits) out of a 6-bit-packed `[u8; 48]`.
    fn lane_at(out: &[u8; 48], i: usize) -> u8 {
        crate::fp4::extract_field(out, 6 * i, 6)
    }

    // Decode an FP6 E3M2 code's numeric magnitude (spec section 2.4.2): bias 3, 2-bit mantissa.
    fn e3m2_magnitude(code: u8) -> f64 {
        let e = (code >> 2) & 0x7;
        let m = code & 0x3;
        if e == 0 {
            // Subnormal: m * 2^(1 - bias) where the implicit bit is 0 -> m * 2^-2 * 2^(1-3)...
            // value = (m / 4) * 2^(1 - 3) = (m/4) * 2^-2 = m * 2^-4 = m * 0.0625.
            m as f64 * 0.0625
        } else {
            (1.0 + m as f64 / 4.0) * 2f64.powi(e as i32 - 3)
        }
    }

    // Decode an FP6 E2M3 code's numeric magnitude (spec section 2.4.2): bias 1, 3-bit mantissa.
    fn e2m3_magnitude(code: u8) -> f64 {
        let e = (code >> 3) & 0x3;
        let m = code & 0x7;
        if e == 0 {
            // Subnormal: value = (m / 8) * 2^(1 - 1) = m / 8 = m * 0.125.
            m as f64 * 0.125
        } else {
            (1.0 + m as f64 / 8.0) * 2f64.powi(e as i32 - 1)
        }
    }

    // Decode an FP8 E4M3 byte's numeric value (spec section 2.4.1): bias 7, 3-bit mantissa, no
    // Inf; only the S.1111.111 slot is NaN, never produced by the exact FP6 widenings.
    fn e4m3_value(byte: u8) -> f64 {
        let s = (byte >> 7) & 1;
        let e = (byte >> 3) & 0xF;
        let m = byte & 0x7;
        let sign = if s == 1 { -1.0 } else { 1.0 };
        let mag = if e == 0 {
            (m as f64 / 8.0) * 2f64.powi(1 - 7)
        } else {
            (1.0 + m as f64 / 8.0) * 2f64.powi(e as i32 - 7)
        };
        sign * mag
    }

    // Numeric value of a full FP6 E3M2 / E2M3 code (with sign), section 2.4.2.
    fn e3m2_value(code: u8) -> f64 {
        let sign = if (code >> 5) & 1 == 1 { -1.0 } else { 1.0 };
        sign * e3m2_magnitude(code)
    }
    fn e2m3_value(code: u8) -> f64 {
        let sign = if (code >> 5) & 1 == 1 { -1.0 } else { 1.0 };
        sign * e2m3_magnitude(code)
    }

    // Is a BF8 (E5M2) byte a subnormal (or zero)? e_i == 0.
    fn bf8_is_subnormal_nonzero(byte: u8) -> bool {
        let e_i = (byte >> 2) & 0x1F;
        let m_i = byte & 0x03;
        e_i == 0 && m_i != 0
    }
    // Is an HF8 (E4M3) byte a subnormal (nonzero)? e_i == 0, m_i != 0.
    fn hf8_is_subnormal_nonzero(byte: u8) -> bool {
        let e_i = (byte >> 3) & 0x0F;
        let m_i = byte & 0x07;
        e_i == 0 && m_i != 0
    }
    // Strip the sign bit of an FP6 code (bit 5) -> magnitude code.
    fn fp6_mag_code(code: u8) -> u8 {
        code & 0x1F
    }

    quickcheck! {
        /// The public `cvtf8_bf6s` dispatcher always equals the scalar oracle byte-for-byte —
        /// the contract callers rely on regardless of which path runs. OQ-5: no native path
        /// exists, so the differential test that would otherwise tie a native path to the
        /// oracle discards; this pins that the dispatcher returns the spec value on every input
        /// (`[avx10-v2-aux-ocp-conversions.CVT_FP8_FP6.1]`,
        /// `[avx10-v2-aux-ocp-conversions.CORRECTNESS.1]`,
        /// `[avx10-v2-aux-ocp-conversions.DETECTION.2]`).
        fn prop_bf6s_public_matches_scalar(input: Inputs) -> bool {
            cvtf8_bf6s(input.a) == cvtf8_bf6s_scalar(input.a)
        }

        /// The public `cvtf8_hf6s` dispatcher always equals the scalar oracle byte-for-byte, on
        /// every sampled byte (`[avx10-v2-aux-ocp-conversions.CVT_FP8_FP6.1]`,
        /// `[avx10-v2-aux-ocp-conversions.CORRECTNESS.1]`,
        /// `[avx10-v2-aux-ocp-conversions.DETECTION.2]`).
        fn prop_hf6s_public_matches_scalar(input: Inputs) -> bool {
            cvtf8_hf6s(input.a) == cvtf8_hf6s_scalar(input.a)
        }

        /// ALWAYS SATURATING: no output FP6 lane ever exceeds the target max normal magnitude —
        /// E3M2 `+/-28.0`, E2M3 `+/-7.5` — for every sampled input, including the lanes holding
        /// FP8 +/-Inf/NaN and overflow magnitudes
        /// (`[avx10-v2-aux-ocp-conversions.CVT_FP8_FP6.2]`).
        fn prop_always_saturating(input: Inputs) -> bool {
            let bf6 = cvtf8_bf6s(input.a);
            let hf6 = cvtf8_hf6s(input.a);
            (0..64).all(|i| e3m2_magnitude(lane_at(&bf6, i)) <= 28.0)
                && (0..64).all(|i| e2m3_magnitude(lane_at(&hf6, i)) <= 7.5)
        }

        /// SUBNORMAL -> +/-0: every FP8 subnormal source lane converts to a same-signed FP6
        /// zero (magnitude code 0), for both targets (spec section 9.6.1 note,
        /// `[avx10-v2-aux-ocp-conversions.CVT_FP8_FP6.3]`).
        fn prop_subnormal_to_zero(input: Inputs) -> bool {
            let bf6 = cvtf8_bf6s(input.a);
            let hf6 = cvtf8_hf6s(input.a);
            (0..64).all(|i| {
                let bf6_ok = !bf8_is_subnormal_nonzero(input.a[i])
                    || fp6_mag_code(lane_at(&bf6, i)) == 0;
                let hf6_ok = !hf8_is_subnormal_nonzero(input.a[i])
                    || fp6_mag_code(lane_at(&hf6, i)) == 0;
                bf6_ok && hf6_ok
            })
        }

        /// LANE INDEPENDENCE: each output lane depends only on the corresponding input lane.
        /// Mutating one source byte changes only that lane's 6-bit output and leaves every
        /// other lane's converted value untouched, for both source formats. (6-bit lanes
        /// straddle bytes, so we compare the unpacked lane values rather than raw bytes.)
        fn prop_lane_independence(input: Inputs, idx: u8) -> bool {
            let i = (idx as usize) % 64;
            for cvt in [cvtf8_bf6s as fn([u8; 64]) -> [u8; 48], cvtf8_hf6s] {
                let base = cvt(input.a);
                let mut a2 = input.a;
                a2[i] = a2[i].wrapping_add(1); // perturb a single source lane
                let pert = cvt(a2);
                for j in 0..64 {
                    if j != i && lane_at(&base, j) != lane_at(&pert, j) {
                        return false;
                    }
                }
            }
            true
        }

        /// Family G: the public `cvtf6_hf8_e3m2` dispatcher always equals the scalar oracle
        /// byte-for-byte. OQ-5: no native path exists, so the differential test discards; this
        /// pins that the dispatcher returns the spec value on every 6-bit-packed input
        /// (`[avx10-v2-aux-ocp-conversions.CVT_FP6_FP8.1]`,
        /// `[avx10-v2-aux-ocp-conversions.CORRECTNESS.1]`,
        /// `[avx10-v2-aux-ocp-conversions.DETECTION.2]`).
        fn prop_cvtf6_hf8_e3m2_public_matches_scalar(input: Fp6Inputs) -> bool {
            cvtf6_hf8_e3m2(input.a) == cvtf6_hf8_e3m2_scalar(input.a)
        }

        /// Family G: the public `cvtf6_hf8_e2m3` dispatcher always equals the scalar oracle
        /// byte-for-byte on every 6-bit-packed input
        /// (`[avx10-v2-aux-ocp-conversions.CVT_FP6_FP8.1]`,
        /// `[avx10-v2-aux-ocp-conversions.CORRECTNESS.1]`,
        /// `[avx10-v2-aux-ocp-conversions.DETECTION.2]`).
        fn prop_cvtf6_hf8_e2m3_public_matches_scalar(input: Fp6Inputs) -> bool {
            cvtf6_hf8_e2m3(input.a) == cvtf6_hf8_e2m3_scalar(input.a)
        }

        /// Family G EXACTNESS: every output E4M3 byte's real number equals the source FP6
        /// lane's real number — the widening is value-exact for both source formats and every
        /// sampled input. The differential native check discards (no `_mm512_cvtf6_hf8`
        /// intrinsic, OQ-5), so this value-preservation property against the independent
        /// section-2.4.2 decoders is the correctness evidence
        /// (`[avx10-v2-aux-ocp-conversions.CVT_FP6_FP8.2]`).
        fn prop_family_g_value_exact(input: Fp6Inputs) -> bool {
            let out_e3m2 = cvtf6_hf8_e3m2(input.a);
            let out_e2m3 = cvtf6_hf8_e2m3(input.a);
            (0..64).all(|i| {
                let code = lane_at(&input.a, i);
                e4m3_value(out_e3m2[i]) == e3m2_value(code)
                    && e4m3_value(out_e2m3[i]) == e2m3_value(code)
                    // Sign preserved bit-for-bit (incl. -0).
                    && (out_e3m2[i] >> 7) == (code >> 5)
                    && (out_e2m3[i] >> 7) == (code >> 5)
            })
        }

        /// Family G LANE INDEPENDENCE: each output byte depends only on its 6-bit source lane.
        /// Because 6-bit lanes straddle byte boundaries, perturbing one packed input byte can
        /// touch at most the (up to two) lanes overlapping it; every other OUTPUT byte (whose
        /// lane does not overlap the perturbed byte) is unchanged
        /// (`[avx10-v2-aux-ocp-conversions.CVT_FP6_FP8.1]`).
        fn prop_family_g_lane_independence(input: Fp6Inputs, idx: u8) -> bool {
            let b = (idx as usize) % 48; // perturb packed input byte b
            // Lanes overlapping byte b: a lane i covers bits [6i, 6i+6); it overlaps byte b's
            // bits [8b, 8b+8) iff 6i < 8b+8 and 6i+6 > 8b.
            let overlaps = |i: usize| 6 * i < 8 * b + 8 && 6 * i + 6 > 8 * b;
            for cvt in [cvtf6_hf8_e3m2 as fn([u8; 48]) -> [u8; 64], cvtf6_hf8_e2m3] {
                let base = cvt(input.a);
                let mut a2 = input.a;
                a2[b] = a2[b].wrapping_add(1);
                let pert = cvt(a2);
                for i in 0..64 {
                    if !overlaps(i) && base[i] != pert[i] {
                        return false;
                    }
                }
            }
            true
        }
    }

    /// NO MASKING (full coverage): for an arbitrary swept input every one of the 64 output
    /// lanes is the spec conversion of its source lane — there is no lane the converter leaves
    /// at a default/unwritten value (`[avx10-v2-aux-ocp-conversions.CVT_FP8_FP6.4]`). Checked
    /// against the per-lane helper for both source formats.
    #[test]
    fn no_masking_every_lane_written() {
        // Sweep: source lane i = byte value i (covers a wide spread of codes incl. NaN/Inf).
        let a: [u8; 64] = core::array::from_fn(|i| i as u8);

        let out_bf6 = cvtf8_bf6s(a);
        let out_hf6 = cvtf8_hf6s(a);
        for (i, &byte) in a.iter().enumerate() {
            assert_eq!(
                lane_at(&out_bf6, i),
                crate::fp6::fp8_e5m2_to_fp6_e3m2(byte),
                "E5M2->E3M2 lane {i} must equal the per-lane helper (no masking)"
            );
            assert_eq!(
                lane_at(&out_hf6, i),
                crate::fp6::fp8_e4m3_to_fp6_e2m3(byte),
                "E4M3->E2M3 lane {i} must equal the per-lane helper (no masking)"
            );
        }
    }

    /// FULL-SOURCE-DOMAIN EXACTNESS for family G: every one of the 64 FP6 codes (per format)
    /// maps to a single fixed E4M3 byte, and that byte equals the exact per-lane decode helper,
    /// on EVERY lane position of a swept input. The entire FP6 domain is exhausted (64 codes),
    /// so the property is a total proof, not a sample
    /// (`[avx10-v2-aux-ocp-conversions.CVT_FP6_FP8.2]`).
    #[test]
    fn full_domain_fp6_to_e4m3_is_exact() {
        // Sweep every lane through all 64 FP6 codes: lane i carries code (i % 64).
        let codes: [u8; 64] = core::array::from_fn(|i| (i % 64) as u8);
        let mut packed = [0u8; 48];
        crate::fp6::pack(&codes, &mut packed);
        let out_e3m2 = cvtf6_hf8_e3m2(packed);
        let out_e2m3 = cvtf6_hf8_e2m3(packed);

        for code in 0u8..64 {
            let want_e3m2 = crate::fp6::fp6_e3m2_to_fp8_e4m3(code);
            let want_e2m3 = crate::fp6::fp6_e2m3_to_fp8_e4m3(code);
            for i in 0..64 {
                if codes[i] == code {
                    assert_eq!(
                        out_e3m2[i], want_e3m2,
                        "E3M2 code {code:#04x} at lane {i} -> single fixed E4M3 byte {want_e3m2:#04x}"
                    );
                    assert_eq!(
                        out_e2m3[i], want_e2m3,
                        "E2M3 code {code:#04x} at lane {i} -> single fixed E4M3 byte {want_e2m3:#04x}"
                    );
                }
            }
        }
    }
}

/// Native-vs-oracle differential for families F (FP8 -> FP6) and G (FP6 -> FP8). Phase 11.
///
/// Both ship **oracle-only** in this toolchain (OQ-5: `_mm512_cvtf8_bf6s` /
/// `_mm512_cvtf8_hf6s` / `_mm512_cvtf6_hf8` are absent under `-mavx10.2`). The property
/// compares each public dispatcher to its scalar oracle under `feature="native"` on
/// AVX10_V2_AUX hardware (`[avx10-v2-aux-ocp-conversions.DIFFERENTIAL.1]`), and
/// `TestResult::discard()`s (never `from_bool(false)`) otherwise, so a fallback-only runner
/// cannot go vacuously green.
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
        fp8: [u8; 64],
        fp6: [u8; 48],
    }

    impl Arbitrary for Inputs {
        fn arbitrary(g: &mut Gen) -> Self {
            Inputs {
                fp8: core::array::from_fn(|_| u8::arbitrary(g)),
                fp6: core::array::from_fn(|_| u8::arbitrary(g)),
            }
        }
    }

    quickcheck! {
        /// Families-F/G native-vs-oracle differential. Under `feature="native"` on x86_64 with
        /// `AVX10_V2_AUX` detected, the two FP8->FP6 dispatchers and the two FP6->FP8
        /// dispatchers must each equal their scalar oracle bit-for-bit
        /// (`[avx10-v2-aux-ocp-conversions.DIFFERENTIAL.1]`). DISCARDED (not failed) when the
        /// feature or hardware is absent (`[avx10-v2-aux-ocp-conversions.CORRECTNESS.2]`).
        fn prop_native_matches_oracle(input: Inputs) -> TestResult {
            #[cfg(all(target_arch = "x86_64", feature = "native"))]
            {
                if detect::has_avx10_v2_aux() {
                    let f = cvtf8_bf6s(input.fp8) == cvtf8_bf6s_scalar(input.fp8)
                        && cvtf8_hf6s(input.fp8) == cvtf8_hf6s_scalar(input.fp8);
                    let g = cvtf6_hf8_e3m2(input.fp6) == cvtf6_hf8_e3m2_scalar(input.fp6)
                        && cvtf6_hf8_e2m3(input.fp6) == cvtf6_hf8_e2m3_scalar(input.fp6);
                    return TestResult::from_bool(f && g);
                }
            }
            let _ = &input;
            TestResult::discard()
        }
    }
}
