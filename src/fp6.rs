//! FP6 (E3M2 / E2M3) micro-format codec and 6-bit lane packing.
//!
//! MX FP6 has two layouts (spec section 2.4.2), both 6-bit, sign-magnitude, no infinities
//! and no NaN:
//!
//! * **E3M2** (BF6): 3-bit exponent (bias 3), 2-bit mantissa. Max normal `S.111.11 =
//!   +/-28.0`, min normal `S.001.00 = +/-0.25`.
//! * **E2M3** (HF6): 2-bit exponent (bias 1), 3-bit mantissa. Max normal `S.11.111 =
//!   +/-7.5`, min normal `S.01.000 = +/-1.0`.
//!
//! FP6 lanes are **6-bit-packed** — 6 bits per lane, packed contiguously from bit 0, so a
//! length-`N` lane vector occupies `N * 6 / 8` bytes and lanes straddle byte boundaries
//! (spec section 9.6.5 / 9.7.5). The packer/unpacker here reuse `crate::fp4::extract_field`
//! for the size-6 read.
//!
//! This module also owns the FP8->FP6 saturating-RTNE conversion helpers
//! ([`fp8_e5m2_to_fp6_e3m2`] / [`fp8_e4m3_to_fp6_e2m3`], spec section 16.3, consumed by
//! family F) and the exact FP6->FP8 E4M3 decode helpers
//! ([`fp6_e3m2_to_fp8_e4m3`] / [`fp6_e2m3_to_fp8_e4m3`], spec section 16.4, consumed by
//! family G). The FP6->FP8 decodes are **exact**: every one of the 64 FP6 codes (per format)
//! maps to exactly one FP8 E4M3 byte with no rounding/approximation (spec section 9.7.1).
//!
//! # Iteration-2 open-question resolutions
//!
//! * **OQ-3 — two-instruction-family naming.** Both the FP8→FP6 family (F) and the FP6→FP8
//!   family (G) map a single ticket stem to TWO spec instructions. RESOLVED: family F carries
//!   distinct intrinsic stems (`cvtf8_bf6s` = E5M2→E3M2, `cvtf8_hf6s` = E4M3→E2M3); family G is
//!   disambiguated by a source-format suffix (`cvtf6_hf8_e3m2` / `cvtf6_hf8_e2m3`) since both
//!   target FP8 E4M3 — reconciled against the final stdarch names at upstream time.
//! * **OQ-4 — per-family DAZ.** The forward saturating helpers
//!   ([`fp8_e5m2_to_fp6_e3m2`] / [`fp8_e4m3_to_fp6_e2m3`]) assume DAZ=1 (every FP8 subnormal →
//!   same-signed FP6 zero); the exact reverse decodes
//!   ([`fp6_e3m2_to_fp8_e4m3`] / [`fp6_e2m3_to_fp8_e4m3`]) assume DAZ=0 — encoded per helper.
//! * **OQ-5 — native-path reachability.** Families F/G ship **oracle-only** in this toolchain:
//!   the `_mm512_cvtf8_bf6s` / `_mm512_cvtf6_hf8` intrinsics are absent under `-mavx10.2`, so
//!   there is no native C shim; the differential discards rather than failing (still fully
//!   correct via the scalar oracle, `[avx10-v2-aux-ocp-conversions.CORRECTNESS.2]`).

use crate::fp4::extract_field;

/// Pack a slice of 6-bit FP6 values into a 6-bit-packed byte buffer.
///
/// Lane `i` (low 6 bits of `values[i]`) is written at bit offset `6 * i`, contiguously from
/// bit 0, straddling byte boundaries as needed (spec section 9.6.5). `values.len() * 6` must
/// be a multiple of 8; the output is `values.len() * 6 / 8` bytes. Every lane is written
/// (no masking/zeroing), the inverse of [`unpack`]. Thin `size = 6` wrapper over
/// `crate::fp4::pack_fields`.
pub(crate) fn pack(values: &[u8], out: &mut [u8]) {
    assert_eq!((values.len() * 6) % 8, 0);
    assert_eq!(out.len(), values.len() * 6 / 8);
    crate::fp4::pack_fields(values, 6, out);
}

/// Unpack a 6-bit-packed byte buffer into one right-aligned 6-bit value per lane.
///
/// Reads lane `i` from bit offset `6 * i` via `crate::fp4::extract_field` with `size = 6`,
/// the inverse of [`pack`]: `out[i]` holds the lane's 6 bits in `[5:0]`. `out.len() * 6`
/// must equal `buf.len() * 8`.
pub(crate) fn unpack(buf: &[u8], out: &mut [u8]) {
    assert_eq!(out.len() * 6, buf.len() * 8);
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = extract_field(buf, 6 * i, 6);
    }
}

/// Convert one FP8 E5M2 (BF8) byte to its FP6 E3M2 (BF6) code, RTNE and always saturating.
///
/// Transcribes the ACE v1 section-16.3 `fp8_e5m2_to_fp6_e3m2` helper verbatim (spec section
/// 9.6 `VCVTBF82BF6S`). FP6 E3M2 is sign / 3-bit exponent (bias 3) / 2-bit mantissa with NO
/// NaN and NO Inf (spec section 2.4.2); max normal `S.111.11 = +/-28.0` is `e_o=0x7,
/// m_o=0x3`. The returned `u8` holds the 6-bit code right-aligned in bits `[5:0]`.
///
/// Because the source/target mantissa widths match (E5M2 has 2 mantissa bits, E3M2 has 2),
/// **no mantissa precision is lost** — only exponent-range narrowing can round (spec section
/// 9.6.1 note, `[avx10-v2-aux-ocp-conversions.CVT_FP8_FP6.3]`). Always saturating (spec
/// section 9.6.1): every BF8 +/-Inf (`S.11111.00`) / NaN (`S.11111.{01,10,11}`), and any BF8
/// whose magnitude exceeds the FP6 max normal `+/-28.0`, clamp to the same-signed max normal
/// (`[avx10-v2-aux-ocp-conversions.CVT_FP8_FP6.2]`). DAZ=1: every BF8 zero/subnormal maps to
/// FP6 same-signed zero (the subnormal-output branch's RTNE is a no-op here because every BF8
/// subnormal lies far below the smallest FP6 subnormal midpoint, so it rounds to +/-0 —
/// `[avx10-v2-aux-ocp-conversions.CVT_FP8_FP6.3]`). The normal branch is a direct rebias with
/// the mantissa copied unchanged (`[avx10-v2-aux-ocp-conversions.CVT_FP8_FP6.1]`).
pub(crate) fn fp8_e5m2_to_fp6_e3m2(byte: u8) -> u8 {
    let i = byte as u32;
    let s_i = (i & 0x80) >> 7;
    let e_i = (i & 0x7C) >> 2; // 5-bit biased exponent (bias 15)
    let m_i = i & 0x03; // 2-bit mantissa
    let exp_rebias: i32 = 15 - 3; // FP6 E3M2 bias = 3; FP8 E5M2 bias = 15
    let new_exp: i32 = e_i as i32 - exp_rebias;

    let mut e_o: u32 = 0;
    let mut m_o: u32 = 0;

    if e_i == 0x1F {
        // NaN or Inf (any mantissa) -> clamp to FP6 E3M2 max normal (FP6 has no NaN/Inf).
        e_o = 0x7;
        m_o = 0x3;
    } else if (e_i as i32 > exp_rebias + 7) || (e_i as i32 == exp_rebias + 7 && m_i > 0x3) {
        // Overflow -> clamp to FP6 E3M2 max normal +/-28.0. (The `m_i > 0x3` arm is
        // unreachable since m_i is 2 bits, but transcribed verbatim from the spec — same
        // as the E2M3 helper's `m_i > 0x7` arm.)
        e_o = 0x7;
        m_o = 0x3;
    } else if e_i == 0x00 {
        // Zero or denorm (DAZ=1) -> FP6 zero. (m_i == 0 is exact zero; m_i != 0 is a BF8
        // subnormal, which lies far below the smallest FP6 subnormal midpoint and rounds to
        // signed zero — spec section 9.6.1 note.)
        e_o = 0;
        m_o = 0;
    } else if new_exp <= 0 {
        // Underflow -> FP6 subnormal or zero, RTNE (hidden-bit insertion + shift).
        if (1 - new_exp) <= 3 {
            let mant = m_i | 0x4; // restore hidden bit (E5M2 has 2 mantissa bits)
            let shift = (1 - new_exp) as u32;
            m_o = mant >> shift;
            let lowmant = mant & crate::fp8::mask(shift as i32);
            let halfway = 1u32 << (shift - 1);
            if lowmant > halfway || (lowmant == halfway && (m_o & 0x1) != 0) {
                m_o += 1;
                if (m_o & 0x3) == 0 {
                    e_o += 1;
                }
            }
        }
    } else {
        // Normal: direct rebias, mantissa copied unchanged (matched mantissa width).
        e_o = (e_i as i32 - exp_rebias) as u32;
        m_o = m_i;
    }

    (((s_i & 0x1) << 5) | ((e_o & 0x7) << 2) | (m_o & 0x3)) as u8
}

/// Convert one FP8 E4M3 (HF8) byte to its FP6 E2M3 (HF6) code, RTNE and always saturating.
///
/// Transcribes the ACE v1 section-16.3 `fp8_e4m3_to_fp6_e2m3` helper verbatim (spec section
/// 9.6 `VCVTHF82HF6S`). FP6 E2M3 is sign / 2-bit exponent (bias 1) / 3-bit mantissa with NO
/// NaN and NO Inf (spec section 2.4.2); max normal `S.11.111 = +/-7.5` is `e_o=0x3,
/// m_o=0x7`. The returned `u8` holds the 6-bit code right-aligned in bits `[5:0]`.
///
/// Source/target mantissa widths match (E4M3 has 3 mantissa bits, E2M3 has 3), so **no
/// mantissa precision is lost** — only exponent-range narrowing can round (spec section
/// 9.6.1 note, `[avx10-v2-aux-ocp-conversions.CVT_FP8_FP6.3]`). Always saturating (spec
/// section 9.6.1): the sole HF8 NaN `S.1111.111`, the HF8 max-exponent binade, and any HF8
/// whose magnitude exceeds the FP6 max normal `+/-7.5`, clamp to the same-signed max normal
/// (`[avx10-v2-aux-ocp-conversions.CVT_FP8_FP6.2]`). DAZ=1: every HF8 zero/subnormal maps to
/// FP6 same-signed zero (`[avx10-v2-aux-ocp-conversions.CVT_FP8_FP6.3]`). The normal branch
/// is a direct rebias with the mantissa copied unchanged
/// (`[avx10-v2-aux-ocp-conversions.CVT_FP8_FP6.1]`).
pub(crate) fn fp8_e4m3_to_fp6_e2m3(byte: u8) -> u8 {
    let i = byte as u32;
    let s_i = (i & 0x80) >> 7;
    let e_i = (i & 0x78) >> 3; // 4-bit biased exponent (bias 7)
    let m_i = i & 0x07; // 3-bit mantissa
    let exp_rebias: i32 = 7 - 1; // FP6 E2M3 bias = 1; FP8 E4M3 bias = 7
    let new_exp: i32 = e_i as i32 - exp_rebias;

    let mut e_o: u32 = 0;
    let mut m_o: u32 = 0;

    if e_i == 0xF {
        // NaN (S.1111.111) or max-exponent binade -> clamp to FP6 E2M3 max normal. (The spec
        // pseudocode clamps the entire e_i==0xF binade, including the E4M3 max normal 448 and
        // the NaN slot, since all exceed the FP6 E2M3 range.)
        e_o = 0x3;
        m_o = 0x7;
    } else if (e_i as i32 > exp_rebias + 3) || (e_i as i32 == exp_rebias + 3 && m_i > 0x7) {
        // Overflow -> clamp to FP6 E2M3 max normal +/-7.5. (The `m_i > 0x7` arm is
        // unreachable since m_i is 3 bits, but transcribed verbatim from the spec.)
        e_o = 0x3;
        m_o = 0x7;
    } else if e_i == 0x00 {
        // Zero or denorm (DAZ=1) -> FP6 zero (HF8 subnormals round to signed zero, section
        // 9.6.1 note).
        e_o = 0;
        m_o = 0;
    } else if new_exp <= 0 {
        // Underflow -> FP6 subnormal or zero, RTNE (hidden-bit insertion + shift).
        if (1 - new_exp) <= 4 {
            let mant = m_i | 0x8; // restore hidden bit (E4M3 has 3 mantissa bits)
            let shift = (1 - new_exp) as u32;
            m_o = mant >> shift;
            let lowmant = mant & crate::fp8::mask(shift as i32);
            let halfway = 1u32 << (shift - 1);
            if lowmant > halfway || (lowmant == halfway && (m_o & 0x1) != 0) {
                m_o += 1;
                if (m_o & 0x7) == 0 {
                    e_o += 1;
                }
            }
        }
    } else {
        // Normal: direct rebias, mantissa copied unchanged (matched mantissa width).
        e_o = (e_i as i32 - exp_rebias) as u32;
        m_o = m_i;
    }

    (((s_i & 0x1) << 5) | ((e_o & 0x3) << 3) | (m_o & 0x7)) as u8
}

/// Convert one FP6 E3M2 (BF6) 6-bit code to its single exact FP8 E4M3 (HF8) byte.
///
/// Transcribes the ACE v1 section-16.4 `fp6_e3m2_to_fp8_e4m3` helper VERBATIM (spec section
/// 9.7 `VCVTBF62HF8`, pseudocode reproduced below). The conversion is **lossless widening** —
/// every one of the 64 FP6 E3M2 codes is representable in FP8 E4M3, so the map is **exact**:
/// no rounding, no saturation, no approximation (spec section 9.7.1, `DAZ=0` — subnormals are
/// renormalised, not flushed; `[avx10-v2-aux-ocp-conversions.CVT_FP6_FP8.1]`,
/// `[avx10-v2-aux-ocp-conversions.CVT_FP6_FP8.2]`). The input `code` is read from bits `[5:0]`.
///
/// FP6 E3M2: `sign[5]`, `exp[4:2]` (bias 3), `frac[1:0]`. FP8 E4M3: `sign[7]`, `exp[6:3]`
/// (bias 7), `frac[2:0]`. Spec section-16.4 pseudocode (load-bearing, transcribed line by
/// line below):
///
/// ```text
/// s_i = (i & 0x20) >> 5;  e_i = (i & 0x1C) >> 2;  m_i = (i & 0x03)
/// IF e_i == 0x0:                       // FP6 subnormal
///     IF   m_i == 0: e_o = m_o = 0     // zero
///     ELIF m_i == 1: e_o = 3; m_o = 0  // 2^-4 -> FP8 normal e=3, m=0
///     ELSE:          e_o = 4; m_o = (m_i & 0x1) << 2   // low frac bit -> FP8 mantissa[2]
/// ELSE:                                // FP6 normal: rebias 3->7, shift mantissa left 1
///     e_o = e_i + (7 - 3);  m_o = m_i << 1
/// RETURN ((s_i & 0x1) << 7) | ((e_o & 0xF) << 3) | (m_o & 0x7)
/// ```
pub(crate) fn fp6_e3m2_to_fp8_e4m3(code: u8) -> u8 {
    let i = code as u32;
    let s_i = (i & 0x20) >> 5;
    let e_i = (i & 0x1C) >> 2;
    let m_i = i & 0x03;

    let e_o: u32;
    let m_o: u32;
    if e_i == 0x0 {
        // FP6 subnormal: exp=0, frac in [0..3].
        if m_i == 0 {
            // zero
            e_o = 0;
            m_o = 0;
        } else if m_i == 1 {
            // 0.01 * 2^-2 = 2^-4 -> FP8 normal e=3, m=0
            e_o = 3;
            m_o = 0;
        } else {
            // m_i=2: 0.10*2^-2 = 2^-3; m_i=3: 0.11*2^-2. Leading 1 at bit 1 -> 2^-3 (biased 4);
            // the low frac bit becomes FP8 mantissa[2].
            e_o = 4;
            m_o = (m_i & 0x1) << 2;
        }
    } else {
        // FP6 normal: rebias 3->7, shift the 2-bit frac left 1 into bits [2:1] of the 3-bit
        // FP8 frac (the new low frac bit is 0).
        e_o = e_i + (7 - 3);
        m_o = m_i << 1;
    }

    (((s_i & 0x1) << 7) | ((e_o & 0xF) << 3) | (m_o & 0x7)) as u8
}

/// Convert one FP6 E2M3 (HF6) 6-bit code to its single exact FP8 E4M3 (HF8) byte.
///
/// Transcribes the ACE v1 section-16.4 `fp6_e2m3_to_fp8_e4m3` helper VERBATIM (spec section
/// 9.7 `VCVTHF62HF8`, pseudocode reproduced below). Lossless widening — every one of the 64
/// FP6 E2M3 codes is representable in FP8 E4M3, so the map is **exact**: no rounding, no
/// saturation, no approximation (spec section 9.7.1, `DAZ=0`;
/// `[avx10-v2-aux-ocp-conversions.CVT_FP6_FP8.1]`,
/// `[avx10-v2-aux-ocp-conversions.CVT_FP6_FP8.2]`). The input `code` is read from bits `[5:0]`.
///
/// FP6 E2M3: `sign[5]`, `exp[4:3]` (bias 1), `frac[2:0]`. FP8 E4M3: `sign[7]`, `exp[6:3]`
/// (bias 7), `frac[2:0]`. Spec section-16.4 pseudocode (load-bearing, transcribed line by
/// line below):
///
/// ```text
/// s_i = (i & 0x20) >> 5;  e_i = (i & 0x18) >> 3;  m_i = (i & 0x07)
/// IF e_i == 0x0:                          // FP6 subnormal: 0.mmm * 2^0 = m_i * 2^-3
///     IF   m_i == 0:  e_o = m_o = 0        // zero
///     ELIF m_i == 1:  e_o = 4; m_o = 0     // 2^-3 -> FP8 normal e=4 (=7-3), m=0
///     ELIF m_i <= 3:  e_o = 5; m_o = (m_i & 0x1) << 2   // leading 1 at bit1 -> 2^-2
///     ELSE:           e_o = 6; m_o = (m_i & 0x3) << 1   // leading 1 at bit2 -> 2^-1
/// ELSE:                                   // FP6 normal: rebias 1->7, mantissa unchanged
///     e_o = e_i + (7 - 1);  m_o = m_i
/// RETURN ((s_i & 0x1) << 7) | ((e_o & 0xF) << 3) | (m_o & 0x7)
/// ```
pub(crate) fn fp6_e2m3_to_fp8_e4m3(code: u8) -> u8 {
    let i = code as u32;
    let s_i = (i & 0x20) >> 5;
    let e_i = (i & 0x18) >> 3;
    let m_i = i & 0x07;

    let e_o: u32;
    let m_o: u32;
    if e_i == 0x0 {
        // FP6 subnormal: 0.mmm * 2^0 = m_i * 2^-3.
        if m_i == 0 {
            // zero
            e_o = 0;
            m_o = 0;
        } else if m_i == 1 {
            // 2^-3 -> FP8 normal e=4 (=7-3), m=0
            e_o = 4;
            m_o = 0;
        } else if m_i <= 3 {
            // m_i*2^-3 with leading 1 at bit 1 -> 2^-2 range (unbiased -2 -> biased 5); the
            // remaining frac bit becomes FP8 frac[2].
            e_o = 5;
            m_o = (m_i & 0x1) << 2;
        } else {
            // m_i*2^-3 with leading 1 at bit 2 -> 2^-1 range (unbiased -1 -> biased 6); the
            // remaining two frac bits become FP8 frac[2:1].
            e_o = 6;
            m_o = (m_i & 0x3) << 1;
        }
    } else {
        // FP6 normal: rebias 1->7, the 3-bit frac fits the 3-bit FP8 frac directly.
        e_o = e_i + (7 - 1);
        m_o = m_i;
    }

    (((s_i & 0x1) << 7) | ((e_o & 0xF) << 3) | (m_o & 0x7)) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn six_bit_pack_unpack_round_trips() {
        // Four 6-bit lanes = 24 bits = 3 packed bytes. Lanes chosen so several straddle a
        // byte boundary (lane 1 spans bits 6..12, lane 2 spans bits 12..18).
        let lanes = [0b101101u8, 0b011010, 0b111000, 0b000111];
        let mut packed = [0u8; 3];
        pack(&lanes, &mut packed);
        let mut out = [0u8; 4];
        unpack(&packed, &mut out);
        assert_eq!(
            out, lanes,
            "6-bit pack/unpack must round-trip across boundaries"
        );
    }

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

    // ---- Numeric decoders used by the FP6->FP8 EXACTNESS tests (spec section 2.4.1/2.4.2).
    // These are independent reference decoders (NOT the production codec), so a defect in the
    // codec cannot hide behind a shared bug. They turn an encoding into its real number, letting
    // the test assert that value is *preserved* across the widening.

    // Decode an FP6 E3M2 code to its signed magnitude (bias 3, 2-bit frac).
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
    // Decode an FP6 E2M3 code to its signed magnitude (bias 1, 3-bit frac).
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
    // Decode an FP8 E4M3 byte to its signed magnitude (bias 7, 3-bit frac, no Inf; only the
    // S.1111.111 slot is NaN, never produced by these exact widenings).
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

    /// Hand-computed FP8 E5M2 -> FP6 E3M2 conversions pinning each branch of the section-16.3
    /// helper. FP6 E3M2 magnitude table (spec section 2.4.2): S.000.00=0, S.001.00=+/-0.25
    /// (min normal), S.111.11=+/-28.0 (max normal), S.000.01=+/-0.0625 (min subnormal).
    ///
    /// DISCRIMINATING lanes (each rules out a plausible-but-wrong model):
    ///  * BF8 +1.0 (S.01111.00) -> E3M2 +1.0 (S.011.00): normal rebias 15->3 copies the
    ///    mantissa unchanged. A model that lost the mantissa or mis-rebiased lands elsewhere.
    ///  * BF8 +28.0 (S.10011.11 = 1.75*2^4) -> E3M2 max normal S.111.11. Exactly representable
    ///    (mantissa width matched), so NOT a saturation — distinguishes exact-rebias from clamp.
    ///  * BF8 +32.0 (S.10100.00 = 2^5 > 28.0) -> saturates to S.111.11 (+28.0): a non-saturating
    ///    model would overflow the 3-bit FP6 exponent.
    ///  * BF8 +Inf / NaN -> +28.0 (FP6 has no Inf/NaN), ruling out Inf/NaN propagation.
    ///  * BF8 subnormal +2^-16 (S.00000.01) -> +0: every FP8 subnormal rounds to FP6 +/-0
    ///    (section 9.6.1 note), ruling out an attempt to represent it as an FP6 subnormal.
    #[test]
    fn known_value_e5m2_to_fp6_e3m2() {
        // +0 / -0.
        assert_eq!(
            fp8_e5m2_to_fp6_e3m2(bf8(0, 0b00000, 0b00)),
            bf6(0, 0b000, 0b00)
        );
        assert_eq!(
            fp8_e5m2_to_fp6_e3m2(bf8(1, 0b00000, 0b00)),
            bf6(1, 0b000, 0b00)
        );
        // +1.0 (S.01111.00, bias 15 -> exp field 15) -> E3M2 1.0 (S.011.00, exp field 3).
        assert_eq!(
            fp8_e5m2_to_fp6_e3m2(bf8(0, 0b01111, 0b00)),
            bf6(0, 0b011, 0b00),
            "+1.0 -> E3M2 S.011.00 (mantissa preserved, direct rebias)"
        );
        // MANTISSA PRESERVED: +1.25 (S.01111.01) -> E3M2 S.011.01 (the .01 mantissa survives).
        assert_eq!(
            fp8_e5m2_to_fp6_e3m2(bf8(0, 0b01111, 0b01)),
            bf6(0, 0b011, 0b01),
            "+1.25 -> E3M2 S.011.01 (no mantissa loss, matched width)"
        );
        // EXACT MAX NORMAL: +28.0 = 1.75*2^4 (S.10011.11, exp field 19) -> E3M2 S.111.11
        // (exp 7, mantissa 0b11). Representable exactly; this is NOT saturation.
        assert_eq!(
            fp8_e5m2_to_fp6_e3m2(bf8(0, 0b10011, 0b11)),
            bf6(0, 0b111, 0b11),
            "+28.0 maps exactly to E3M2 max normal S.111.11 (not a clamp)"
        );
        // SATURATION: +32.0 = 2^5 (S.10100.00, exp field 20 > 19) exceeds 28.0 -> S.111.11.
        assert_eq!(
            fp8_e5m2_to_fp6_e3m2(bf8(0, 0b10100, 0b00)),
            bf6(0, 0b111, 0b11),
            "+32.0 saturates to E3M2 max normal +28.0"
        );
        // SATURATION negative: -57344 (BF8 max normal S.11110.11) -> -28.0 (S.111.11).
        assert_eq!(
            fp8_e5m2_to_fp6_e3m2(bf8(1, 0b11110, 0b11)),
            bf6(1, 0b111, 0b11),
            "-57344 saturates to E3M2 max normal -28.0"
        );
        // +Inf (S.11111.00) -> +28.0 (FP6 has no Inf).
        assert_eq!(
            fp8_e5m2_to_fp6_e3m2(bf8(0, 0b11111, 0b00)),
            bf6(0, 0b111, 0b11),
            "+Inf clamps to E3M2 max normal +28.0"
        );
        // -NaN (S.11111.10) -> -28.0 (sign preserved, FP6 has no NaN).
        assert_eq!(
            fp8_e5m2_to_fp6_e3m2(bf8(1, 0b11111, 0b10)),
            bf6(1, 0b111, 0b11),
            "-NaN clamps to E3M2 max normal -28.0"
        );
        // SUBNORMAL -> +/-0: BF8 min subnormal +2^-16 (S.00000.01) -> +0.
        assert_eq!(
            fp8_e5m2_to_fp6_e3m2(bf8(0, 0b00000, 0b01)),
            bf6(0, 0b000, 0b00),
            "BF8 subnormal +2^-16 -> FP6 +0 (section 9.6.1 note)"
        );
        // SUBNORMAL -> -0: signed subnormal carries the sign.
        assert_eq!(
            fp8_e5m2_to_fp6_e3m2(bf8(1, 0b00000, 0b11)),
            bf6(1, 0b000, 0b00),
            "BF8 negative subnormal -> FP6 -0"
        );
    }

    /// Hand-computed FP8 E4M3 -> FP6 E2M3 conversions pinning each branch of the section-16.3
    /// helper. FP6 E2M3 magnitude table (spec section 2.4.2): S.00.000=0, S.01.000=+/-1.0
    /// (min normal), S.11.111=+/-7.5 (max normal), S.00.001=+/-0.125 (min subnormal).
    ///
    /// DISCRIMINATING lanes:
    ///  * HF8 +1.0 (S.0111.000) -> E2M3 +1.0 (S.01.000): direct rebias 7->1, mantissa preserved.
    ///  * HF8 +7.5 (S.1001.111 = 1.875*2^2) -> E2M3 max normal S.11.111. Exactly representable
    ///    (matched mantissa width), so NOT a clamp — distinguishes exact-rebias from saturation.
    ///  * HF8 +8.0 (S.1010.000 = 2^3 > 7.5) -> saturates to S.11.111 (+7.5).
    ///  * HF8 NaN (S.1111.111) and HF8 max normal 448 (S.1111.110) -> +7.5 (clamp).
    ///  * HF8 subnormal +2^-9 (S.0000.001) -> +0 (section 9.6.1 note).
    #[test]
    fn known_value_e4m3_to_fp6_e2m3() {
        // +0 / -0.
        assert_eq!(
            fp8_e4m3_to_fp6_e2m3(hf8(0, 0b0000, 0b000)),
            hf6(0, 0b00, 0b000)
        );
        assert_eq!(
            fp8_e4m3_to_fp6_e2m3(hf8(1, 0b0000, 0b000)),
            hf6(1, 0b00, 0b000)
        );
        // +1.0 (S.0111.000, bias 7 -> exp field 7) -> E2M3 1.0 (S.01.000, exp field 1).
        assert_eq!(
            fp8_e4m3_to_fp6_e2m3(hf8(0, 0b0111, 0b000)),
            hf6(0, 0b01, 0b000),
            "+1.0 -> E2M3 S.01.000 (direct rebias)"
        );
        // MANTISSA PRESERVED: +1.625 = 1.101b*2^0 (S.0111.101) -> E2M3 S.01.101 (3-bit
        // mantissa 0b101 survives unchanged).
        assert_eq!(
            fp8_e4m3_to_fp6_e2m3(hf8(0, 0b0111, 0b101)),
            hf6(0, 0b01, 0b101),
            "+1.625 -> E2M3 S.01.101 (no mantissa loss, matched 3-bit width)"
        );
        // EXACT MAX NORMAL: +7.5 = 1.875*2^2 (S.1001.111, exp field 9) -> E2M3 S.11.111
        // (exp 3, mantissa 0b111). Representable exactly; this is NOT saturation.
        assert_eq!(
            fp8_e4m3_to_fp6_e2m3(hf8(0, 0b1001, 0b111)),
            hf6(0, 0b11, 0b111),
            "+7.5 maps exactly to E2M3 max normal S.11.111 (not a clamp)"
        );
        // SATURATION: +8.0 = 2^3 (S.1010.000, exp field 10 > 9) exceeds 7.5 -> S.11.111.
        assert_eq!(
            fp8_e4m3_to_fp6_e2m3(hf8(0, 0b1010, 0b000)),
            hf6(0, 0b11, 0b111),
            "+8.0 saturates to E2M3 max normal +7.5"
        );
        // SATURATION: HF8 max normal +448 (S.1111.110) -> +7.5.
        assert_eq!(
            fp8_e4m3_to_fp6_e2m3(hf8(0, 0b1111, 0b110)),
            hf6(0, 0b11, 0b111),
            "+448 saturates to E2M3 max normal +7.5"
        );
        // NaN (S.1111.111) -> +7.5 (FP6 has no NaN); negative sign preserved.
        assert_eq!(
            fp8_e4m3_to_fp6_e2m3(hf8(1, 0b1111, 0b111)),
            hf6(1, 0b11, 0b111),
            "-NaN clamps to E2M3 max normal -7.5"
        );
        // SUBNORMAL -> +/-0: HF8 min subnormal +2^-9 (S.0000.001) -> +0.
        assert_eq!(
            fp8_e4m3_to_fp6_e2m3(hf8(0, 0b0000, 0b001)),
            hf6(0, 0b00, 0b000),
            "HF8 subnormal +2^-9 -> FP6 +0 (section 9.6.1 note)"
        );
        // SUBNORMAL -> -0: HF8 max subnormal -0.875*2^-6 (S.0000.111) -> -0.
        assert_eq!(
            fp8_e4m3_to_fp6_e2m3(hf8(1, 0b0000, 0b111)),
            hf6(1, 0b00, 0b000),
            "HF8 negative max subnormal -> FP6 -0"
        );
    }

    /// EXACT FP6 E3M2 -> FP8 E4M3 (family G): full source domain. For all 64 FP6 E3M2 codes
    /// the decode is value-exact — the FP8 byte's real number equals the FP6 code's real
    /// number (`[avx10-v2-aux-ocp-conversions.CVT_FP6_FP8.1]`,
    /// `[avx10-v2-aux-ocp-conversions.CVT_FP6_FP8.2]`). The differential native tiebreaker is
    /// unavailable in this environment (`_mm512_cvtf6_hf8` absent from GCC 16.1.1 `-mavx10.2`,
    /// OQ-5), so this value-preservation check against the independent section-2.4.2 decoders
    /// is the correctness evidence: a wrong rebias/shift would change the decoded value.
    #[test]
    fn fp6_e3m2_to_fp8_e4m3_value_exact_full_domain() {
        for code in 0u8..64 {
            let byte = fp6_e3m2_to_fp8_e4m3(code);
            assert_eq!(
                e4m3_value(byte),
                e3m2_value(code),
                "E3M2 code {code:#04x} (={}) must widen exactly to E4M3 byte {byte:#04x} (={})",
                e3m2_value(code),
                e4m3_value(byte)
            );
            // Sign is preserved bit-for-bit (incl. -0).
            assert_eq!(byte >> 7, code >> 5, "sign preserved for code {code:#04x}");
        }
    }

    /// EXACT FP6 E2M3 -> FP8 E4M3 (family G): full source domain, same value-preservation
    /// argument as the E3M2 case above
    /// (`[avx10-v2-aux-ocp-conversions.CVT_FP6_FP8.1]`,
    /// `[avx10-v2-aux-ocp-conversions.CVT_FP6_FP8.2]`).
    #[test]
    fn fp6_e2m3_to_fp8_e4m3_value_exact_full_domain() {
        for code in 0u8..64 {
            let byte = fp6_e2m3_to_fp8_e4m3(code);
            assert_eq!(
                e4m3_value(byte),
                e2m3_value(code),
                "E2M3 code {code:#04x} (={}) must widen exactly to E4M3 byte {byte:#04x} (={})",
                e2m3_value(code),
                e4m3_value(byte)
            );
            assert_eq!(byte >> 7, code >> 5, "sign preserved for code {code:#04x}");
        }
    }

    /// Headline known values pinning the specific E4M3 bytes the spec pseudocode produces, so
    /// a regression that silently changed the encoding (even while preserving the value via a
    /// different normal/subnormal split) would be caught. The E3M2 `S.111.11 -> +28.0` and
    /// E2M3 `S.11.111 -> +7.5` max-normal lanes are the plan's named vectors.
    #[test]
    fn fp6_to_fp8_e4m3_headline_bytes() {
        // E3M2 S.111.11 (0x1F, +28.0) -> E4M3 0x5E (exp 0xB=11 -> 2^4, mantissa 0b110 -> 1.75).
        assert_eq!(
            fp6_e3m2_to_fp8_e4m3(bf6(0, 0b111, 0b11)),
            hf8(0, 0b1011, 0b110),
            "E3M2 +28.0 -> E4M3 0x5E (=1.75*2^4=28.0)"
        );
        // E3M2 subnormal m_i=1 (0x01) -> e_o=3, m_o=0 = E4M3 S.0011.000 = 2^-4 = 0.0625.
        assert_eq!(
            fp6_e3m2_to_fp8_e4m3(bf6(0, 0b000, 0b01)),
            hf8(0, 0b0011, 0b000),
            "E3M2 subnormal 0.0625 -> E4M3 normal 2^-4"
        );
        // E3M2 subnormal m_i=3 (0x03, 0.1875) -> e_o=4, m_o=(3&1)<<2=0b100 = E4M3 S.0100.100.
        assert_eq!(
            fp6_e3m2_to_fp8_e4m3(bf6(0, 0b000, 0b11)),
            hf8(0, 0b0100, 0b100),
            "E3M2 subnormal 0.1875 -> E4M3 normal (e=4,m=0b100)"
        );
        // E2M3 S.11.111 (0x1F, +7.5) -> E4M3 0x4F (exp 0x9=9 -> 2^2, mantissa 0b111 -> 1.875).
        assert_eq!(
            fp6_e2m3_to_fp8_e4m3(hf6(0, 0b11, 0b111)),
            hf8(0, 0b1001, 0b111),
            "E2M3 +7.5 -> E4M3 0x4F (=1.875*2^2=7.5)"
        );
        // E2M3 subnormal m_i=1 (0x01, 0.125) -> e_o=4, m_o=0 = E4M3 S.0100.000 = 2^-3.
        assert_eq!(
            fp6_e2m3_to_fp8_e4m3(hf6(0, 0b00, 0b001)),
            hf8(0, 0b0100, 0b000),
            "E2M3 subnormal 0.125 -> E4M3 normal 2^-3"
        );
        // E2M3 subnormal m_i=7 (0x07, 0.875) -> e_o=6, m_o=(7&3)<<1=0b110 = E4M3 S.0110.110.
        assert_eq!(
            fp6_e2m3_to_fp8_e4m3(hf6(0, 0b00, 0b111)),
            hf8(0, 0b0110, 0b110),
            "E2M3 subnormal 0.875 -> E4M3 (e=6,m=0b110)"
        );
        // -0 maps to E4M3 -0 (sign bit preserved).
        assert_eq!(fp6_e3m2_to_fp8_e4m3(bf6(1, 0b000, 0b00)), 0x80);
        assert_eq!(fp6_e2m3_to_fp8_e4m3(hf6(1, 0b00, 0b000)), 0x80);
    }

    // ============================================================================================
    // FINDING #7b — exhaustive forward-rounding coverage for the two saturating-RTNE FP8->FP6
    // helpers. Each helper takes an 8-bit FP8 byte, so the source domain is only 256 values; we
    // test ALL 256 for each against an INDEPENDENT oracle (below). The oracle never calls the
    // production codec: it decodes the FP8 byte to an exact f64, then rounds that number onto the
    // FP6 target grid by nearest-neighbour search over the format's own representable value set,
    // ties-to-EVEN, with production's saturation and DAZ policies applied explicitly. A shared
    // bug therefore cannot hide.
    //
    // GROUND-TRUTH NOTE: production is the SDE-verified ground truth. Any 256-way disagreement
    // would mean the oracle here is wrong, not the codec.
    // ============================================================================================

    // ---- FP6 target grids (magnitude only; sign is applied separately). Each grid is the full
    // set of the format's non-negative representable magnitudes, indexed by its 5-bit magnitude
    // code (exp<<mbits | mant), which is monotonically ascending in magnitude (sign-magnitude
    // layout). Reuses the independent section-2.4.2 decoders `e3m2_value` / `e2m3_value` above.

    // (magnitude_code, magnitude_value) for every E3M2 magnitude: code 0..32, value ascending,
    // code 0 = +0, code 31 (0b11111 = e=7,m=3) = max normal 28.0.
    fn e3m2_grid() -> Vec<(u8, f64)> {
        (0u8..32).map(|c| (c, e3m2_value(c))).collect()
    }
    // Same for E2M3: code 0 = +0, code 31 (0b11111 = e=3,m=7) = max normal 7.5.
    fn e2m3_grid() -> Vec<(u8, f64)> {
        (0u8..32).map(|c| (c, e2m3_value(c))).collect()
    }

    // Round a non-negative magnitude `v` onto `grid` (ascending) with round-to-nearest,
    // ties-to-EVEN, returning the winning 5-bit magnitude code. Values at or above the grid's
    // top clamp to the max-normal code (the grid has no Inf, so the nearest representable to any
    // larger finite magnitude IS the max normal — this is exactly the saturating policy). The
    // tie rule picks the neighbour whose mantissa LSB (= code & 1) is 0.
    fn round_to_fp6_grid(v: f64, grid: &[(u8, f64)]) -> u8 {
        let last = grid.len() - 1;
        if v >= grid[last].1 {
            return grid[last].0; // saturate to max normal
        }
        if v <= 0.0 {
            return grid[0].0; // +0
        }
        for i in 0..last {
            let (lo_code, lo) = grid[i];
            let (hi_code, hi) = grid[i + 1];
            if v >= lo && v <= hi {
                let dlo = v - lo;
                let dhi = hi - v;
                if dlo < dhi {
                    return lo_code;
                }
                if dhi < dlo {
                    return hi_code;
                }
                // Exact halfway -> ties to even mantissa LSB.
                return if lo_code & 1 == 0 { lo_code } else { hi_code };
            }
        }
        grid[last].0
    }

    // Independent oracle: FP8 E5M2 (BF8) byte -> expected FP6 E3M2 code. Decodes the byte per
    // spec section 2.4.1 (sign, 5-bit exp bias 15, 2-bit mantissa), then applies production
    // policy: NaN/Inf (e==0x1F) clamp to same-signed max normal; DAZ=1 flushes every
    // zero/subnormal (e==0) to same-signed FP6 zero; every finite normal is rounded onto the
    // E3M2 grid (which naturally reaches the subnormal-output region and saturation).
    fn oracle_e5m2_to_e3m2(byte: u8) -> u8 {
        let s = ((byte >> 7) & 1) as u8;
        let e = (byte >> 2) & 0x1F;
        let m = byte & 0x03;
        let grid = e3m2_grid();
        let mag = if e == 0x1F {
            31 // NaN/Inf -> max normal 28.0
        } else if e == 0 {
            0 // DAZ: zero/subnormal -> +/-0
        } else {
            let v = (1.0 + m as f64 / 4.0) * 2f64.powi(e as i32 - 15);
            round_to_fp6_grid(v, &grid)
        };
        (s << 5) | mag
    }

    // Independent oracle: FP8 E4M3 (HF8) byte -> expected FP6 E2M3 code. Decodes per spec
    // section 2.4.1 (sign, 4-bit exp bias 7, 3-bit mantissa). Production clamps the ENTIRE
    // e==0xF binade (E4M3 max normal 448 AND the NaN slot) to the same-signed FP6 max normal,
    // DAZ-flushes e==0, and rounds every other finite normal onto the E2M3 grid.
    fn oracle_e4m3_to_e2m3(byte: u8) -> u8 {
        let s = ((byte >> 7) & 1) as u8;
        let e = (byte >> 3) & 0x0F;
        let m = byte & 0x07;
        let grid = e2m3_grid();
        let mag = if e == 0x0F {
            31 // whole top binade (incl. 448 and NaN) -> max normal 7.5
        } else if e == 0 {
            0 // DAZ: zero/subnormal -> +/-0
        } else {
            let v = (1.0 + m as f64 / 8.0) * 2f64.powi(e as i32 - 7);
            round_to_fp6_grid(v, &grid)
        };
        (s << 5) | mag
    }

    /// EXHAUSTIVE forward coverage of `fp8_e5m2_to_fp6_e3m2` over ALL 256 FP8 E5M2 bytes,
    /// checked against the independent nearest-ties-to-even oracle. This sweep necessarily hits
    /// the SUBNORMAL-OUTPUT RTNE branch (fp6.rs ~110-124) — normal BF8 inputs whose value
    /// underflows into a nonzero FP6 E3M2 subnormal — as well as the NaN/Inf clamp, DAZ flush,
    /// saturation, exact-max-normal, and signed-zero paths.
    #[test]
    fn exhaustive_e5m2_to_fp6_e3m2_all_256() {
        for byte in 0u8..=255 {
            let got = fp8_e5m2_to_fp6_e3m2(byte);
            let want = oracle_e5m2_to_e3m2(byte);
            assert_eq!(
                got,
                want,
                "BF8 byte {byte:#04x} (decoded={:?}): production={got:#04x}, oracle={want:#04x}",
                {
                    let e = (byte >> 2) & 0x1F;
                    let m = byte & 0x03;
                    if e == 0x1F {
                        f64::NAN
                    } else if e == 0 {
                        (m as f64 / 4.0) * 2f64.powi(1 - 15)
                    } else {
                        (1.0 + m as f64 / 4.0) * 2f64.powi(e as i32 - 15)
                    }
                }
            );
        }
    }

    /// EXHAUSTIVE forward coverage of `fp8_e4m3_to_fp6_e2m3` over ALL 256 FP8 E4M3 bytes,
    /// checked against the independent oracle. Necessarily hits the SUBNORMAL-OUTPUT RTNE branch
    /// (fp6.rs ~177-191) plus the whole-e==0xF clamp, DAZ flush, saturation, exact-max-normal,
    /// and signed-zero paths.
    #[test]
    fn exhaustive_e4m3_to_fp6_e2m3_all_256() {
        for byte in 0u8..=255 {
            let got = fp8_e4m3_to_fp6_e2m3(byte);
            let want = oracle_e4m3_to_e2m3(byte);
            assert_eq!(
                got,
                want,
                "HF8 byte {byte:#04x} (decoded={:?}): production={got:#04x}, oracle={want:#04x}",
                {
                    let e = (byte >> 3) & 0x0F;
                    let m = byte & 0x07;
                    if e == 0x0F {
                        f64::NAN
                    } else if e == 0 {
                        (m as f64 / 8.0) * 2f64.powi(1 - 7)
                    } else {
                        (1.0 + m as f64 / 8.0) * 2f64.powi(e as i32 - 7)
                    }
                }
            );
        }
    }

    /// EXPLICIT hand assertions for the flagged, likely-unexercised SUBNORMAL-OUTPUT branch:
    /// NORMAL FP8 inputs that underflow into a NONZERO FP6 subnormal. Expected FP6 codes are
    /// computed here independently from the real numbers, NOT from the production helper.
    #[test]
    fn subnormal_output_branch_hand_computed() {
        // ---- E5M2 -> E3M2. E3M2 subnormals (e=0): m/4 * 2^-2, i.e. 0.0625, 0.125, 0.1875.
        //
        // BF8 +2^-4 = S.01011.00 (e_i=11, normal, value exactly 1.0*2^-4 = 0.0625). This is a
        // *normal* FP8 whose value equals the E3M2 min subnormal 0.0625 = S.000.01. So it takes
        // the underflow branch and produces a NONZERO subnormal (not the DAZ flush).
        assert_eq!(
            fp8_e5m2_to_fp6_e3m2(bf8(0, 0b01011, 0b00)),
            bf6(0, 0b000, 0b01),
            "normal BF8 2^-4 underflows to NONZERO E3M2 subnormal 0.0625 (S.000.01)"
        );
        // BF8 +1.5*2^-3 = S.01100.10 (e_i=12, normal, value = 1.5 * 0.125 = 0.1875). Equals the
        // E3M2 subnormal 0.1875 = 3/4 * 2^-2 = S.000.11 exactly (no rounding needed).
        assert_eq!(
            fp8_e5m2_to_fp6_e3m2(bf8(0, 0b01100, 0b10)),
            bf6(0, 0b000, 0b11),
            "normal BF8 0.1875 underflows to NONZERO E3M2 subnormal 0.1875 (S.000.11)"
        );
        // BF8 +1.75*2^-4 = S.01011.11 (e_i=11, value = 1.75 * 0.0625 = 0.109375). Between E3M2
        // 0.0625 and 0.125; nearest is 0.125 (dist 0.0156 < 0.0469) -> S.000.10. Sign carried.
        assert_eq!(
            fp8_e5m2_to_fp6_e3m2(bf8(1, 0b01011, 0b11)),
            bf6(1, 0b000, 0b10),
            "negative normal BF8 -0.109375 rounds to NONZERO E3M2 subnormal -0.125 (S.000.10)"
        );

        // ---- E4M3 -> E2M3. E2M3 subnormals (e=0): m/8, i.e. 0.125, 0.25, ..., 0.875.
        //
        // HF8 +2^-1 = S.0110.000 (e_i=6, normal, value = 1.0*2^-1 = 0.5). Equals E2M3 subnormal
        // 0.5 = 4/8 = S.00.100 exactly. Normal FP8 -> NONZERO FP6 subnormal.
        assert_eq!(
            fp8_e4m3_to_fp6_e2m3(hf8(0, 0b0110, 0b000)),
            hf6(0, 0b00, 0b100),
            "normal HF8 0.5 underflows to NONZERO E2M3 subnormal 0.5 (S.00.100)"
        );
        // HF8 +2^-2 = S.0101.000 (e_i=5, normal, value = 0.25). Equals E2M3 subnormal 0.25 =
        // 2/8 = S.00.010 exactly.
        assert_eq!(
            fp8_e4m3_to_fp6_e2m3(hf8(0, 0b0101, 0b000)),
            hf6(0, 0b00, 0b010),
            "normal HF8 0.25 underflows to NONZERO E2M3 subnormal 0.25 (S.00.010)"
        );
        // HF8 +1.75*2^-1 = S.0110.110 (e_i=6, value = 1.75*0.5 = 0.875). Equals E2M3 subnormal
        // 0.875 = 7/8 = S.00.111 exactly (max subnormal). Sign carried on the negative twin.
        assert_eq!(
            fp8_e4m3_to_fp6_e2m3(hf8(1, 0b0110, 0b110)),
            hf6(1, 0b00, 0b111),
            "negative normal HF8 -0.875 underflows to NONZERO E2M3 subnormal -0.875 (S.00.111)"
        );
    }

    // ---- Oracle-free structural properties over all 256 inputs (no reference decoder needed).

    /// Every production output is a valid 6-bit FP6 code: bits [7:6] are zero.
    #[test]
    fn prop_output_is_valid_6bit_code() {
        for byte in 0u8..=255 {
            assert_eq!(
                fp8_e5m2_to_fp6_e3m2(byte) & 0xC0,
                0,
                "E5M2->E3M2 byte {byte:#04x} produced non-6-bit code"
            );
            assert_eq!(
                fp8_e4m3_to_fp6_e2m3(byte) & 0xC0,
                0,
                "E4M3->E2M3 byte {byte:#04x} produced non-6-bit code"
            );
        }
    }

    /// Sign is always preserved (including the NaN/Inf clamp and the signed-zero flush): the
    /// FP6 sign bit [5] equals the FP8 sign bit [7].
    #[test]
    fn prop_sign_preserved() {
        for byte in 0u8..=255 {
            let s = byte >> 7;
            assert_eq!(
                (fp8_e5m2_to_fp6_e3m2(byte) >> 5) & 1,
                s,
                "E5M2->E3M2 byte {byte:#04x} did not preserve sign"
            );
            assert_eq!(
                (fp8_e4m3_to_fp6_e2m3(byte) >> 5) & 1,
                s,
                "E4M3->E2M3 byte {byte:#04x} did not preserve sign"
            );
        }
    }

    /// Monotonicity: over the finite, non-clamped FP8 inputs of a fixed sign, taken in ascending
    /// magnitude order (ascending magnitude code, since FP8 is monotonic in its code), the FP6
    /// output MAGNITUDE is non-decreasing. Excludes the FP8 top binade (E5M2 e==0x1F Inf/NaN;
    /// E4M3 e==0xF NaN slot) which carries no ordered value. DAZ-flushed small inputs produce a
    /// leading run of zeros, which is still non-decreasing before the values ramp up.
    #[test]
    fn prop_magnitude_monotonic() {
        // E5M2: finite non-clamped magnitude codes are e in 0..=0x1E (0x1F is Inf/NaN).
        let mut prev = 0u8;
        for e in 0u8..=0x1E {
            for m in 0u8..=0x03 {
                let byte = (e << 2) | m; // sign 0
                let out_mag = fp8_e5m2_to_fp6_e3m2(byte) & 0x1F;
                assert!(
                    out_mag >= prev,
                    "E5M2->E3M2 not monotonic at byte {byte:#04x}: {out_mag} < {prev}"
                );
                prev = out_mag;
            }
        }
        // E4M3: finite non-clamped magnitude codes are e in 0..=0x0E (0x0F is the clamp binade).
        let mut prev = 0u8;
        for e in 0u8..=0x0E {
            for m in 0u8..=0x07 {
                let byte = (e << 3) | m; // sign 0
                let out_mag = fp8_e4m3_to_fp6_e2m3(byte) & 0x1F;
                assert!(
                    out_mag >= prev,
                    "E4M3->E2M3 not monotonic at byte {byte:#04x}: {out_mag} < {prev}"
                );
                prev = out_mag;
            }
        }
    }

    /// `prop_subnormal_to_zero` and its complement. FORWARD: every FP8 zero/subnormal input
    /// (exp field == 0, the DAZ=1 domain) maps to the same-signed FP6 zero (magnitude bits 0).
    /// COMPLEMENT: there exist NORMAL FP8 inputs (exp field != 0) that produce a NONZERO FP6
    /// output, so the codec is not degenerately flushing everything — this is the very
    /// subnormal-output/normal ramp the exhaustive sweep exercises.
    #[test]
    fn prop_subnormal_to_zero_and_complement() {
        let mut e5m2_nonzero_from_normal = 0usize;
        let mut e4m3_nonzero_from_normal = 0usize;
        for byte in 0u8..=255 {
            let s = byte >> 7;

            let e5 = (byte >> 2) & 0x1F;
            let out5 = fp8_e5m2_to_fp6_e3m2(byte);
            if e5 == 0 {
                assert_eq!(
                    out5,
                    s << 5,
                    "E5M2 subnormal/zero byte {byte:#04x} must flush to same-signed FP6 zero"
                );
            } else if out5 & 0x1F != 0 {
                e5m2_nonzero_from_normal += 1;
            }

            let e4 = (byte >> 3) & 0x0F;
            let out4 = fp8_e4m3_to_fp6_e2m3(byte);
            if e4 == 0 {
                assert_eq!(
                    out4,
                    s << 5,
                    "E4M3 subnormal/zero byte {byte:#04x} must flush to same-signed FP6 zero"
                );
            } else if out4 & 0x1F != 0 {
                e4m3_nonzero_from_normal += 1;
            }
        }
        assert!(
            e5m2_nonzero_from_normal > 0,
            "complement: expected some normal BF8 inputs to yield nonzero FP6"
        );
        assert!(
            e4m3_nonzero_from_normal > 0,
            "complement: expected some normal HF8 inputs to yield nonzero FP6"
        );
    }
}
