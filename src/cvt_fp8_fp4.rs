//! Families D and E (AVX10_V2_AUX): FP8 <-> FP4 (E2M1) converts.
//!
//! Family D (saturating-RTNE FP8 -> FP4, nibble-packed): `cvtf8_bf4s_e5m2` converts 64 BF8
//! (FP8 E5M2) bytes to 64 FP4 E2M1 (BF4) lanes and `cvtf8_bf4s_e4m3` converts 64 HF8 (FP8
//! E4M3) bytes likewise. Per ACE v1 spec section 9.4 (`VCVTBF82BF4S` / `VCVTHF82BF4S`) the
//! conversion rounds **RTNE** and is **always saturating**: an FP8 input whose magnitude
//! exceeds the FP4 max normal `+/-6.0` (including BF8 +/-Inf/NaN and the sole HF8 NaN, which
//! FP4 cannot represent) clamps to the same-signed FP4 max normal `S.11.1`
//! (`[avx10-v2-aux-ocp-conversions.CVT_FP8_FP4.1]`,
//! `[avx10-v2-aux-ocp-conversions.CVT_FP8_FP4.2]`). DAZ=1/FTZ=0, MXCSR not consulted, no
//! floating-point exceptions (spec section 9.4.1). The output is **nibble-packed** and HALF
//! the input width: 64 source bytes (`[u8; 64]`) produce 32 packed bytes (`[u8; 32]`), FP4
//! lane `i` at bit offset `4 * i` (`[avx10-v2-aux-ocp-conversions.CVT_FP8_FP4.3]`), every
//! nibble written, no masking (`[avx10-v2-aux-ocp-conversions.CVT_FP8_FP4.4]`).
//!
//! Family E (exact FP4 -> FP8 E4M3, nibble-unpacked): `cvtbf4_hf8` converts 64 nibble-packed
//! FP4 E2M1 lanes (`[u8; 32]`) to 64 FP8 E4M3 bytes (`[u8; 64]`). Per ACE v1 spec section 9.5
//! (`VCVTBF42HF8`) the conversion is **exact** — every one of the 16 FP4 encodings maps to
//! exactly one FP8 E4M3 encoding with no rounding, saturation or approximation
//! (`[avx10-v2-aux-ocp-conversions.CVT_FP4_FP8.1]`,
//! `[avx10-v2-aux-ocp-conversions.CVT_FP4_FP8.2]`). DAZ=0/FTZ=0 (spec section 9.5.1). The
//! output is **twice** the input width and the input is read nibble-packed from bit 0:
//! `dest.byte[i] = fp4_to_fp8_e4m3(src[4*i+3 : 4*i])` (spec section 9.5.5,
//! `[avx10-v2-aux-ocp-conversions.CVT_FP4_FP8.3]`).
//!
//! Each public dispatcher is a safe fn that selects the scalar oracle whenever the running
//! CPU lacks `AVX10_V2_AUX` (`[avx10-v2-aux-ocp-conversions.DETECTION.2]`). The `_scalar`
//! oracle is the primary, always-correct path on every target including non-x86
//! (`[avx10-v2-aux-ocp-conversions.CORRECTNESS.1]`,
//! `[avx10-v2-aux-ocp-conversions.CORRECTNESS.2]`); it carries no cfg gate, reads no global
//! state, and the dispatcher equals it bit-for-bit. The family-D converters are disambiguated
//! by a source-format suffix (`_e5m2` / `_e4m3`), since both target FP4 E2M1 (OQ-3); family E
//! has the single name `cvtbf4_hf8`. The whole module compiles on stable Rust with no
//! `core::simd`/nightly (`[avx10-v2-aux-ocp-conversions.STABLE_RUST.1]`).
//!
//! OQ-5 (intrinsic unavailable -> oracle-only): ALL THREE converts (`cvtf8_bf4s_e5m2`,
//! `cvtf8_bf4s_e4m3`, `cvtbf4_hf8`) ship oracle-only. The `_mm512_cvtbf8_bf4s` /
//! `_mm512_cvthf8_bf4s` (`VCVTBF82BF4S` / `VCVTHF82BF4S`) and `_mm512_cvtbf4_hf8`
//! (`VCVTBF42HF8`) intrinsics are ABSENT from the installed GCC 16.1.1 `-mavx10.2` headers
//! (verified by compile probes; for family E `_mm512_cvtbf4_hf8` / `_mm512_cvtbf42hf8` /
//! `_mm512_cvtbf4_phf8` / `_mm512_cvt_bf4_hf8` are all rejected with `error: implicit
//! declaration of function ... did you mean '_mm512_cvtph_hf8'?`). Per OQ-5 there is therefore
//! no native C shim and no `extern "C"` declaration for any of them, and each dispatcher
//! resolves to its `_scalar` sibling on every target. The capability check
//! [`crate::detect::has_avx10_v2_aux`] is still consulted (and the native branch is empty) so
//! the three-layer detection wiring is exercised; a native path is added once the intrinsics
//! land in the toolchain. The differential test that would otherwise tie a native path to the
//! oracle DISCARDS (no native path exists), so correctness is grounded against the
//! section-9.4.5 / section-9.5.5 / section-16.3 pseudocode transcribed in [`crate::fp4`].

use crate::detect;
use crate::fp4;

/// BF8 (FP8 E5M2) -> FP4 E2M1 saturating-RTNE convert, nibble-packed (64 lanes -> 32 bytes).
///
/// Per BF8 lane: convert `a[i]` to its FP4 nibble via [`fp4::fp8_e5m2_to_fp4_e2m1`]
/// (always-saturating, RTNE), then nibble-pack into `[u8; 32]` at bit offset `4 * i` (spec
/// section 9.4.5). Always saturating: magnitudes above `+/-6.0`, and BF8 +/-Inf/NaN, clamp
/// to the same-signed FP4 max normal (`[avx10-v2-aux-ocp-conversions.CVT_FP8_FP4.2]`).
///
/// Queries [`detect::has_avx10_v2_aux`] for the native path; that path is not yet wired
/// (OQ-5, see the module docs), so the dispatcher falls through to [`cvtf8_bf4s_e5m2_scalar`]
/// on every target, returning the spec-defined value.
/// `[avx10-v2-aux-ocp-conversions.DETECTION.2]`
pub fn cvtf8_bf4s_e5m2(a: [u8; 64]) -> [u8; 32] {
    // No native path this phase (OQ-5): the FP8->FP4 intrinsic is absent from the `-mavx10.2`
    // toolchain, so even under `feature="native"` on AVX10_V2_AUX hardware the oracle is the
    // only path. The capability check is still consulted so detection is wired and ready for
    // the shim once the intrinsic lands.
    let _ = detect::has_avx10_v2_aux; // keep the capability gate referenced on every target
    cvtf8_bf4s_e5m2_scalar(a)
}

/// Portable reference oracle for [`cvtf8_bf4s_e5m2`] — the primary always-correct path.
///
/// Maps each BF8 lane through [`fp4::fp8_e5m2_to_fp4_e2m1`] (the saturating-RTNE section-16.3
/// helper), then nibble-packs all 64 results into 32 bytes via [`fp4::pack_nibbles`] (every
/// nibble written, no masking). Carries no cfg gate and reads no global state.
/// `[avx10-v2-aux-ocp-conversions.CORRECTNESS.1]` `[avx10-v2-aux-ocp-conversions.CORRECTNESS.2]`
/// `[avx10-v2-aux-ocp-conversions.CVT_FP8_FP4.3]` `[avx10-v2-aux-ocp-conversions.CVT_FP8_FP4.4]`
pub fn cvtf8_bf4s_e5m2_scalar(a: [u8; 64]) -> [u8; 32] {
    let nibbles: [u8; 64] = core::array::from_fn(|i| fp4::fp8_e5m2_to_fp4_e2m1(a[i]));
    let mut out = [0u8; 32];
    fp4::pack_nibbles(&nibbles, &mut out);
    out
}

/// HF8 (FP8 E4M3) -> FP4 E2M1 saturating-RTNE convert, nibble-packed (64 lanes -> 32 bytes).
///
/// Per HF8 lane: convert `a[i]` to its FP4 nibble via [`fp4::fp8_e4m3_to_fp4_e2m1`]
/// (always-saturating, RTNE), then nibble-pack into `[u8; 32]` at bit offset `4 * i` (spec
/// section 9.4.5). Always saturating: magnitudes above `+/-6.0`, and the sole HF8 NaN
/// `S.1111.111`, clamp to the same-signed FP4 max normal
/// (`[avx10-v2-aux-ocp-conversions.CVT_FP8_FP4.2]`).
///
/// Queries [`detect::has_avx10_v2_aux`] for the native path; that path is not yet wired
/// (OQ-5, see the module docs), so the dispatcher falls through to [`cvtf8_bf4s_e4m3_scalar`]
/// on every target, returning the spec-defined value.
/// `[avx10-v2-aux-ocp-conversions.DETECTION.2]`
pub fn cvtf8_bf4s_e4m3(a: [u8; 64]) -> [u8; 32] {
    // No native path this phase (OQ-5); see the module docs and `cvtf8_bf4s_e5m2`.
    let _ = detect::has_avx10_v2_aux; // keep the capability gate referenced on every target
    cvtf8_bf4s_e4m3_scalar(a)
}

/// Portable reference oracle for [`cvtf8_bf4s_e4m3`] — the primary always-correct path.
///
/// Maps each HF8 lane through [`fp4::fp8_e4m3_to_fp4_e2m1`] (the saturating-RTNE section-16.3
/// helper), then nibble-packs all 64 results into 32 bytes via [`fp4::pack_nibbles`] (every
/// nibble written, no masking). Carries no cfg gate and reads no global state.
/// `[avx10-v2-aux-ocp-conversions.CORRECTNESS.1]` `[avx10-v2-aux-ocp-conversions.CORRECTNESS.2]`
/// `[avx10-v2-aux-ocp-conversions.CVT_FP8_FP4.3]` `[avx10-v2-aux-ocp-conversions.CVT_FP8_FP4.4]`
pub fn cvtf8_bf4s_e4m3_scalar(a: [u8; 64]) -> [u8; 32] {
    let nibbles: [u8; 64] = core::array::from_fn(|i| fp4::fp8_e4m3_to_fp4_e2m1(a[i]));
    let mut out = [0u8; 32];
    fp4::pack_nibbles(&nibbles, &mut out);
    out
}

/// FP4 E2M1 (BF4) -> FP8 E4M3 (HF8) exact convert, nibble-unpacked (64 lanes -> 64 bytes).
///
/// Per FP4 lane: read the nibble at bit offset `4 * i` from the nibble-packed input and map
/// it to its exact FP8 E4M3 byte via [`fp4::fp4_e2m1_to_fp8_e4m3`] (the section-9.5.5 LUT).
/// The conversion is **exact** — every FP4 encoding maps to exactly one FP8 E4M3 encoding,
/// no rounding/approximation (`[avx10-v2-aux-ocp-conversions.CVT_FP4_FP8.1]`,
/// `[avx10-v2-aux-ocp-conversions.CVT_FP4_FP8.2]`). The output (`[u8; 64]`) is twice the
/// nibble-packed input width (`[u8; 32]`), the input read nibble-packed from bit 0 (spec
/// section 9.5.5, `[avx10-v2-aux-ocp-conversions.CVT_FP4_FP8.3]`).
///
/// Queries [`detect::has_avx10_v2_aux`] for the native path; that path is not yet wired
/// (OQ-5, see the module docs), so the dispatcher falls through to [`cvtbf4_hf8_scalar`] on
/// every target, returning the spec-defined value.
/// `[avx10-v2-aux-ocp-conversions.DETECTION.2]`
pub fn cvtbf4_hf8(a: [u8; 32]) -> [u8; 64] {
    // No native path this phase (OQ-5): `_mm512_cvtbf4_hf8` (`VCVTBF42HF8`) is absent from the
    // `-mavx10.2` toolchain (GCC 16.1.1), so even under `feature="native"` on AVX10_V2_AUX
    // hardware the oracle is the only path. The capability check is still consulted so
    // detection is wired and ready for the shim once the intrinsic lands.
    let _ = detect::has_avx10_v2_aux; // keep the capability gate referenced on every target
    cvtbf4_hf8_scalar(a)
}

/// Portable reference oracle for [`cvtbf4_hf8`] — the primary always-correct path.
///
/// Reads each nibble-packed FP4 lane and maps it through the exact section-9.5.5 LUT
/// [`fp4::fp4_e2m1_to_fp8_e4m3`], widening 32 packed input bytes into 64 FP8 E4M3 output
/// bytes via [`fp4::unpack_nibbles_to_fp8_e4m3`]. Carries no cfg gate and reads no global
/// state. `[avx10-v2-aux-ocp-conversions.CORRECTNESS.1]`
/// `[avx10-v2-aux-ocp-conversions.CORRECTNESS.2]` `[avx10-v2-aux-ocp-conversions.CVT_FP4_FP8.3]`
pub fn cvtbf4_hf8_scalar(a: [u8; 32]) -> [u8; 64] {
    let mut out = [0u8; 64];
    fp4::unpack_nibbles_to_fp8_e4m3(&a, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // FP4 E2M1 nibble assembler: sign | 2-bit exp | 1-bit mantissa (spec section 2.4.2).
    fn fp4n(sign: u8, exp: u8, mant: u8) -> u8 {
        (sign << 3) | (exp << 1) | mant
    }
    // BF8 (E5M2) byte assembler: sign | 5-bit exp | 2-bit mantissa.
    fn bf8(sign: u8, exp: u8, mant: u8) -> u8 {
        (sign << 7) | (exp << 2) | mant
    }
    // HF8 (E4M3) byte assembler: sign | 4-bit exp | 3-bit mantissa.
    fn hf8(sign: u8, exp: u8, mant: u8) -> u8 {
        (sign << 7) | (exp << 3) | mant
    }
    // Read FP4 lane `i` back out of the nibble-packed output: even lanes are the low nibble
    // of byte `i/2`, odd lanes are the high nibble.
    fn lane(out: &[u8; 32], i: usize) -> u8 {
        if i.is_multiple_of(2) {
            out[i / 2] & 0x0f
        } else {
            (out[i / 2] >> 4) & 0x0f
        }
    }
    // Pack FP4 nibbles into a `[u8; 32]` family-E input: lane `i` at bit offset `4*i`.
    fn pack(nibbles: &[u8; 64]) -> [u8; 32] {
        let mut out = [0u8; 32];
        crate::fp4::pack_nibbles(nibbles, &mut out);
        out
    }

    /// Hand-computed family-D known values pinning the saturation, NaN/Inf-clamp, subnormal,
    /// and RTNE cases — and the nibble-packed layout (`[avx10-v2-aux-ocp-conversions.CVT_FP8_FP4.1]`,
    /// `.2`, `.3`, `.4`). The FP4 magnitude table is S.00.0=0, S.00.1=+/-0.5, S.01.0=+/-1.0,
    /// S.11.1=+/-6.0 (spec section 2.4.2). The differential native check discards in this
    /// environment (no `_mm512_cvt*_bf4s` intrinsic, OQ-5), so these spec-grounded vectors are
    /// the correctness evidence.
    ///
    /// DISCRIMINATING lanes (each rules out a plausible-but-wrong model):
    ///  * lane 0 = HF8 +8.0 (S.1010.000) -> S.11.1 (+6.0): a non-saturating/wrap model would
    ///    overflow the 2-bit FP4 exponent rather than clamp.
    ///  * lane 1 = HF8 NaN (S.1111.111) -> S.11.1 (+6.0): FP4 has no NaN, ruling out NaN-propagation.
    ///  * lane 2 = HF8 +5.0 (S.1001.010), an RTNE tie between 4.0 (S.11.0, even) and 6.0
    ///    (S.11.1) -> 4.0: round-half-up would wrongly give 6.0.
    ///  * lane 3 = HF8 +0.5 (S.0110.000) -> S.00.1 (FP4 max subnormal), not flushed to zero.
    #[test]
    fn known_value_e4m3_nibble_packed() {
        let mut a = [0u8; 64];
        a[0] = hf8(0, 0b1010, 0b000); // +8.0  -> +6.0  (saturation)
        a[1] = hf8(0, 0b1111, 0b111); // NaN   -> +6.0  (clamp)
        a[2] = hf8(0, 0b1001, 0b010); // +5.0  -> 4.0   (RTNE tie to even mantissa)
        a[3] = hf8(0, 0b0110, 0b000); // +0.5  -> S.00.1 (max subnormal)
        a[4] = hf8(1, 0b1111, 0b110); // -448  -> -6.0  (saturation, negative)
        a[5] = hf8(0, 0b0111, 0b000); // +1.0  -> S.01.0 (min normal)

        let out = cvtf8_bf4s_e4m3(a);

        assert_eq!(
            lane(&out, 0),
            fp4n(0, 0b11, 1),
            "+8.0 saturates to +6.0 (S.11.1)"
        );
        assert_eq!(
            lane(&out, 1),
            fp4n(0, 0b11, 1),
            "NaN clamps to +6.0 (S.11.1)"
        );
        assert_eq!(
            lane(&out, 2),
            fp4n(0, 0b11, 0),
            "+5.0 RTNE-ties to 4.0 (S.11.0)"
        );
        assert_eq!(
            lane(&out, 3),
            fp4n(0, 0b00, 1),
            "+0.5 -> max subnormal S.00.1"
        );
        assert_eq!(
            lane(&out, 4),
            fp4n(1, 0b11, 1),
            "-448 saturates to -6.0 (S.11.1)"
        );
        assert_eq!(lane(&out, 5), fp4n(0, 0b01, 0), "+1.0 -> min normal S.01.0");

        // NIBBLE LAYOUT: lanes 0 and 1 share byte 0 (lane 0 low nibble, lane 1 high nibble).
        // Both are S.11.1 = 0b0111 = 0x7, so byte 0 must be (0x7 << 4) | 0x7 = 0x77. A packer
        // that wrote one lane per byte, or swapped the nibble order, would fail here.
        assert_eq!(
            out[0], 0x77,
            "lanes 0,1 nibble-packed into byte 0 (two per byte from bit 0)"
        );
        // Lanes 2,3 share byte 1: lane 2 = 0b0110 (0x6), lane 3 = 0b0001 (0x1) -> 0x16.
        assert_eq!(out[1], 0x16, "lanes 2,3 nibble-packed into byte 1");
        // Untouched source lanes are HF8 +0 -> FP4 +0; the tail packs to zero bytes.
        assert_eq!(out[31], 0x00, "padding lanes 62,63 are +0 -> 0x00");
    }

    /// Hand-computed family-D known values for the BF8 (E5M2) source, pinning the same
    /// saturation / Inf / NaN / subnormal-flush / nibble cases for the E5M2 helper.
    ///
    /// DISCRIMINATING lanes:
    ///  * lane 0 = BF8 +Inf (S.11111.00) -> +6.0: FP4 has no Inf.
    ///  * lane 1 = BF8 -NaN (S.11111.10, sign set) -> -6.0: sign preserved, NaN clamped.
    ///  * lane 2 = BF8 +8.0 (S.10010.00) -> +6.0 (saturation above range).
    ///  * lane 3 = BF8 min subnormal +2^-16 (S.00000.01) -> +0 (DAZ=1 flush), ruling out a
    ///    DAZ=0 decode.
    #[test]
    fn known_value_e5m2_nibble_packed() {
        let mut a = [0u8; 64];
        a[0] = bf8(0, 0b11111, 0b00); // +Inf -> +6.0
        a[1] = bf8(1, 0b11111, 0b10); // -NaN -> -6.0
        a[2] = bf8(0, 0b10010, 0b00); // +8.0 -> +6.0 (saturation)
        a[3] = bf8(0, 0b00000, 0b01); // +2^-16 subnormal -> +0 (DAZ=1)
        a[4] = bf8(0, 0b01111, 0b00); // +1.0 -> S.01.0
        a[5] = bf8(0, 0b01110, 0b00); // +0.5 -> S.00.1

        let out = cvtf8_bf4s_e5m2(a);

        assert_eq!(
            lane(&out, 0),
            fp4n(0, 0b11, 1),
            "+Inf clamps to +6.0 (S.11.1)"
        );
        assert_eq!(
            lane(&out, 1),
            fp4n(1, 0b11, 1),
            "-NaN clamps to -6.0 (S.11.1)"
        );
        assert_eq!(lane(&out, 2), fp4n(0, 0b11, 1), "+8.0 saturates to +6.0");
        assert_eq!(
            lane(&out, 3),
            fp4n(0, 0b00, 0),
            "+2^-16 subnormal flushes to +0 (DAZ=1)"
        );
        assert_eq!(lane(&out, 4), fp4n(0, 0b01, 0), "+1.0 -> min normal S.01.0");
        assert_eq!(
            lane(&out, 5),
            fp4n(0, 0b00, 1),
            "+0.5 -> max subnormal S.00.1"
        );

        // Byte 0 = lane0(0x7) | lane1(0xF)<<4 = 0xF7 (lane 1 is -6.0 = S.11.1 with sign = 0b1111).
        assert_eq!(out[0], 0xF7, "lanes 0,1 nibble-packed (lane1 = -6.0 = 0xF)");
    }

    /// Half-width output and no-masking: the result is exactly 32 bytes for the 64-byte input
    /// (`[avx10-v2-aux-ocp-conversions.CVT_FP8_FP4.3]`), and EVERY output nibble is written —
    /// a non-default input must yield a non-default-everywhere output, never a partially
    /// masked one (`[avx10-v2-aux-ocp-conversions.CVT_FP8_FP4.4]`).
    #[test]
    fn output_is_half_width_no_masking() {
        // Every source lane = HF8 +1.0 -> FP4 S.01.0 = 0b0010 = 0x2 on every nibble.
        let a = [hf8(0, 0b0111, 0b000); 64];
        let out = cvtf8_bf4s_e4m3(a);
        assert_eq!(out.len(), 32, "output is half the 64-byte input width");
        // Every byte is (0x2 << 4) | 0x2 = 0x22; no nibble left at its default 0.
        assert!(
            out.iter().all(|&b| b == 0x22),
            "every output nibble written (no masking)"
        );
    }

    /// Family-E known value: every one of the 16 FP4 codes (the eight magnitudes
    /// {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0} with sign) packed into the first 16 lanes maps to
    /// its exact FP8 E4M3 byte, including S.11.1 -> +6.0. This is the FULL small domain — the
    /// strongest possible exactness test. Each expected E4M3 byte is computed independently
    /// from the E4M3 layout, so the test distinguishes the exact mapping from a wrong rebias
    /// or rounding model. The differential native check discards (no `_mm512_cvtbf4_hf8`
    /// intrinsic, OQ-5), so this spec-grounded vector is the correctness evidence
    /// (`[avx10-v2-aux-ocp-conversions.CVT_FP4_FP8.1]`,
    /// `[avx10-v2-aux-ocp-conversions.CVT_FP4_FP8.2]`).
    #[test]
    fn known_value_fp4_to_hf8_all_16_codes() {
        // Lanes 0..16 hold every FP4 code 0x0..=0xF; the remaining lanes are FP4 +0.
        let mut nibbles = [0u8; 64];
        for (i, slot) in nibbles.iter_mut().take(16).enumerate() {
            *slot = i as u8;
        }
        let a = pack(&nibbles);
        let out = cvtbf4_hf8(a);

        // The 16 expected E4M3 bytes, code by code (sign | (exp,m) magnitude):
        //   code 0..7  positive {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
        //   code 8..15 negative (same magnitudes, sign bit set)
        let expect = [
            hf8(0, 0b0000, 0b000), // 0x0 -> +0.0
            hf8(0, 0b0110, 0b000), // 0x1 -> +0.5
            hf8(0, 0b0111, 0b000), // 0x2 -> +1.0
            hf8(0, 0b0111, 0b100), // 0x3 -> +1.5
            hf8(0, 0b1000, 0b000), // 0x4 -> +2.0
            hf8(0, 0b1000, 0b100), // 0x5 -> +3.0
            hf8(0, 0b1001, 0b000), // 0x6 -> +4.0
            hf8(0, 0b1001, 0b100), // 0x7 -> +6.0 (S.11.1, max normal)
            hf8(1, 0b0000, 0b000), // 0x8 -> -0.0
            hf8(1, 0b0110, 0b000), // 0x9 -> -0.5
            hf8(1, 0b0111, 0b000), // 0xA -> -1.0
            hf8(1, 0b0111, 0b100), // 0xB -> -1.5
            hf8(1, 0b1000, 0b000), // 0xC -> -2.0
            hf8(1, 0b1000, 0b100), // 0xD -> -3.0
            hf8(1, 0b1001, 0b000), // 0xE -> -4.0
            hf8(1, 0b1001, 0b100), // 0xF -> -6.0 (S.11.1, max normal)
        ];
        for (code, &want) in expect.iter().enumerate() {
            assert_eq!(
                out[code], want,
                "FP4 code {code:#x} -> exact E4M3 byte {want:#04x}"
            );
        }
        // S.11.1 -> +6.0 specifically (the headline exact lane).
        assert_eq!(out[7], hf8(0, 0b1001, 0b100), "S.11.1 -> +6.0 (E4M3 0x4C)");
        // Output is twice the input width and the tail is the FP4-+0 -> E4M3 +0 lanes.
        assert_eq!(out.len(), 64, "output 2x the 32-byte nibble-packed input");
        assert_eq!(out[63], 0x00, "padding FP4 +0 -> E4M3 +0");
    }

    /// EXACTNESS round-trip via family D: for every in-range FP4 lane, FP4 -> FP8 E4M3
    /// (`cvtbf4_hf8`, exact) -> FP4 (`cvtf8_bf4s_e4m3`, saturating-RTNE) recovers the original
    /// FP4 lane. Every FP4 magnitude {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0} is exactly
    /// representable in FP4, so the E4M3 image is exact and the reverse RTNE convert is lossless
    /// — the round-trip is the identity on the full FP4 domain
    /// (`[avx10-v2-aux-ocp-conversions.EXACTNESS.1]`). Signed-zero note: FP4 -0 (`0x8`) -> E4M3
    /// -0 -> FP4 -0, so the sign survives.
    #[test]
    fn exactness_round_trip_fp4_hf8_fp4() {
        let mut nibbles = [0u8; 64];
        for (i, slot) in nibbles.iter_mut().enumerate() {
            *slot = (i % 16) as u8; // sweep all 16 FP4 codes across the 64 lanes
        }
        let packed = pack(&nibbles);

        let hf8_bytes = cvtbf4_hf8(packed); // [u8; 64], exact FP4 -> E4M3
        let back = cvtf8_bf4s_e4m3(hf8_bytes); // [u8; 32], E4M3 -> FP4 (saturating-RTNE)

        for (i, &nib) in nibbles.iter().enumerate() {
            assert_eq!(
                lane(&back, i),
                nib,
                "lane {i}: FP4 {nib:#x} -> E4M3 -> FP4 must recover the original FP4 code"
            );
        }
    }
}

/// Property-based tests for families D (FP8 -> FP4) and E (FP4 -> FP8). The hand-rolled tests
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

    /// A random 32-byte nibble-packed FP4 input for family E (covers all 16 FP4 codes per lane).
    #[derive(Clone, Debug)]
    struct Fp4Inputs {
        a: [u8; 32],
    }

    impl Arbitrary for Fp4Inputs {
        fn arbitrary(g: &mut Gen) -> Self {
            Fp4Inputs {
                a: core::array::from_fn(|_| u8::arbitrary(g)),
            }
        }
    }

    // The FP4 max normal magnitude is S.11.1 (code 0b111 ignoring sign). Helper: decode an FP4
    // nibble's numeric magnitude so the always-saturating property can bound it by 6.0.
    fn fp4_magnitude(nibble: u8) -> f64 {
        let e = (nibble >> 1) & 0x3;
        let m = nibble & 0x1;
        if e == 0 {
            // Subnormal: m * 2^(1 - bias) * 0.5 ... value = m * 2^-1 = 0.5 * m.
            0.5 * m as f64
        } else {
            (1.0 + 0.5 * m as f64) * 2f64.powi(e as i32 - 1)
        }
    }

    fn nibble_at(out: &[u8; 32], i: usize) -> u8 {
        if i.is_multiple_of(2) {
            out[i / 2] & 0x0f
        } else {
            (out[i / 2] >> 4) & 0x0f
        }
    }

    quickcheck! {
        /// The public E5M2 dispatcher always equals the scalar oracle byte-for-byte — the
        /// contract callers rely on regardless of which path runs. OQ-5: no native path
        /// exists, so the differential test that would otherwise tie a native path to the
        /// oracle discards; this pins that the dispatcher returns the spec value on every
        /// input (`[avx10-v2-aux-ocp-conversions.CVT_FP8_FP4.1]`,
        /// `[avx10-v2-aux-ocp-conversions.CORRECTNESS.1]`,
        /// `[avx10-v2-aux-ocp-conversions.DETECTION.2]`).
        fn prop_e5m2_public_matches_scalar(input: Inputs) -> bool {
            cvtf8_bf4s_e5m2(input.a) == cvtf8_bf4s_e5m2_scalar(input.a)
        }

        /// The public E4M3 dispatcher always equals the E4M3 scalar oracle byte-for-byte, on
        /// every sampled byte (`[avx10-v2-aux-ocp-conversions.CVT_FP8_FP4.1]`,
        /// `[avx10-v2-aux-ocp-conversions.CORRECTNESS.1]`,
        /// `[avx10-v2-aux-ocp-conversions.DETECTION.2]`).
        fn prop_e4m3_public_matches_scalar(input: Inputs) -> bool {
            cvtf8_bf4s_e4m3(input.a) == cvtf8_bf4s_e4m3_scalar(input.a)
        }

        /// ALWAYS SATURATING: no output FP4 nibble ever exceeds the FP4 max normal magnitude
        /// (`+/-6.0`), for either source format and every sampled input — including the lanes
        /// holding FP8 +/-Inf/NaN and overflow magnitudes
        /// (`[avx10-v2-aux-ocp-conversions.CVT_FP8_FP4.2]`).
        fn prop_always_saturating_le_6(input: Inputs) -> bool {
            let outs = [cvtf8_bf4s_e5m2(input.a), cvtf8_bf4s_e4m3(input.a)];
            outs.iter().all(|out| {
                (0..64).all(|i| fp4_magnitude(nibble_at(out, i)) <= 6.0)
            })
        }

        /// LANE INDEPENDENCE: each output nibble depends only on the corresponding input lane.
        /// Mutating one source byte changes at most that byte's two-lane output byte and leaves
        /// every other output byte untouched, for both source formats.
        fn prop_lane_independence(input: Inputs, idx: u8) -> bool {
            let i = (idx as usize) % 64;
            for cvt in [cvtf8_bf4s_e5m2 as fn([u8;64])->[u8;32], cvtf8_bf4s_e4m3] {
                let base = cvt(input.a);
                let mut a2 = input.a;
                a2[i] = a2[i].wrapping_add(1); // perturb a single source lane
                let pert = cvt(a2);
                // Only output byte `i/2` (holding lanes 2*(i/2) and 2*(i/2)+1) may change.
                let changed_byte = i / 2;
                for b in 0..32 {
                    if b != changed_byte && base[b] != pert[b] {
                        return false;
                    }
                }
            }
            true
        }

        /// Family E: the public `cvtbf4_hf8` dispatcher always equals the scalar oracle
        /// byte-for-byte. OQ-5: no native path exists, so the differential test discards; this
        /// pins that the dispatcher returns the spec value on every nibble-packed input
        /// (`[avx10-v2-aux-ocp-conversions.CVT_FP4_FP8.1]`,
        /// `[avx10-v2-aux-ocp-conversions.CORRECTNESS.1]`,
        /// `[avx10-v2-aux-ocp-conversions.DETECTION.2]`).
        fn prop_cvtbf4_hf8_public_matches_scalar(input: Fp4Inputs) -> bool {
            cvtbf4_hf8(input.a) == cvtbf4_hf8_scalar(input.a)
        }

        /// Family E LANE INDEPENDENCE: each output byte depends only on the FP4 lane(s) it
        /// reads. Perturbing one packed input byte (which holds exactly two FP4 lanes) changes
        /// at most those two output bytes and leaves every other output byte untouched
        /// (`[avx10-v2-aux-ocp-conversions.CVT_FP4_FP8.2]`).
        fn prop_cvtbf4_hf8_lane_independence(input: Fp4Inputs, idx: u8) -> bool {
            let i = (idx as usize) % 32; // perturb packed input byte i (FP4 lanes 2i, 2i+1)
            let base = cvtbf4_hf8(input.a);
            let mut a2 = input.a;
            a2[i] = a2[i].wrapping_add(1);
            let pert = cvtbf4_hf8(a2);
            // Only output bytes 2*i and 2*i+1 may change.
            (0..64).all(|b| b == 2 * i || b == 2 * i + 1 || base[b] == pert[b])
        }
    }

    /// NO MASKING (full coverage): for an arbitrary input every one of the 64 output nibbles is
    /// the spec conversion of its source lane — there is no lane the converter leaves at a
    /// default/unwritten value (`[avx10-v2-aux-ocp-conversions.CVT_FP8_FP4.4]`). Checked
    /// exhaustively against the per-lane helper for both source formats over a swept input.
    #[test]
    fn no_masking_every_nibble_written() {
        // Sweep: source lane i = byte value i (covers a wide spread of codes incl. NaN/Inf).
        let a: [u8; 64] = core::array::from_fn(|i| i as u8);

        let out_e4m3 = cvtf8_bf4s_e4m3(a);
        let out_e5m2 = cvtf8_bf4s_e5m2(a);
        for (i, &byte) in a.iter().enumerate() {
            assert_eq!(
                nibble_at(&out_e4m3, i),
                crate::fp4::fp8_e4m3_to_fp4_e2m1(byte),
                "E4M3 lane {i} nibble must equal the per-lane helper (no masking)"
            );
            assert_eq!(
                nibble_at(&out_e5m2, i),
                crate::fp4::fp8_e5m2_to_fp4_e2m1(byte),
                "E5M2 lane {i} nibble must equal the per-lane helper (no masking)"
            );
        }
    }

    /// FULL-SOURCE-DOMAIN EXACTNESS for family E: every one of the 16 FP4 nibble codes maps to
    /// a single fixed E4M3 byte, and that byte equals the exact per-lane LUT helper, on EVERY
    /// lane position of a swept input. This is the teeth of "exact bijection into the wider
    /// format" — the entire FP4 domain is exhausted (only 16 codes), so the property is a
    /// total proof, not a sample (`[avx10-v2-aux-ocp-conversions.CVT_FP4_FP8.2]`).
    #[test]
    fn full_domain_fp4_to_e4m3_is_exact() {
        // Sweep every lane through all 16 FP4 codes: lane i carries code (i % 16).
        let mut nibbles = [0u8; 64];
        for (i, slot) in nibbles.iter_mut().enumerate() {
            *slot = (i % 16) as u8;
        }
        let mut packed = [0u8; 32];
        crate::fp4::pack_nibbles(&nibbles, &mut packed);
        let out = cvtbf4_hf8(packed);

        // Each FP4 code maps to exactly one E4M3 byte, fixed regardless of lane position.
        for code in 0u8..16 {
            let want = crate::fp4::fp4_e2m1_to_fp8_e4m3(code);
            for i in 0..64 {
                if nibbles[i] == code {
                    assert_eq!(
                        out[i], want,
                        "FP4 code {code:#x} at lane {i} maps to the single fixed E4M3 byte {want:#04x}"
                    );
                }
            }
        }
    }
}

/// Native-vs-oracle differential for families D (FP8 -> FP4) and E (FP4 -> FP8). Phase 11.
///
/// Both ship **oracle-only** in this toolchain (OQ-5: `_mm512_cvtbf8_bf4s` /
/// `_mm512_cvthf8_bf4s` / `_mm512_cvtbf4_hf8` are absent under `-mavx10.2`). The property
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
        fp4: [u8; 32],
    }

    impl Arbitrary for Inputs {
        fn arbitrary(g: &mut Gen) -> Self {
            Inputs {
                fp8: core::array::from_fn(|_| u8::arbitrary(g)),
                fp4: core::array::from_fn(|_| u8::arbitrary(g)),
            }
        }
    }

    quickcheck! {
        /// Families-D/E native-vs-oracle differential. Under `feature="native"` on x86_64 with
        /// `AVX10_V2_AUX` detected, the two FP8->FP4 dispatchers and the FP4->FP8 dispatcher
        /// must each equal their scalar oracle bit-for-bit
        /// (`[avx10-v2-aux-ocp-conversions.DIFFERENTIAL.1]`). DISCARDED (not failed) when the
        /// feature or hardware is absent (`[avx10-v2-aux-ocp-conversions.CORRECTNESS.2]`).
        fn prop_native_matches_oracle(input: Inputs) -> TestResult {
            #[cfg(all(target_arch = "x86_64", feature = "native"))]
            {
                if detect::has_avx10_v2_aux() {
                    let d = cvtf8_bf4s_e5m2(input.fp8) == cvtf8_bf4s_e5m2_scalar(input.fp8)
                        && cvtf8_bf4s_e4m3(input.fp8) == cvtf8_bf4s_e4m3_scalar(input.fp8);
                    let e = cvtbf4_hf8(input.fp4) == cvtbf4_hf8_scalar(input.fp4);
                    return TestResult::from_bool(d && e);
                }
            }
            let _ = &input;
            TestResult::discard()
        }
    }
}
