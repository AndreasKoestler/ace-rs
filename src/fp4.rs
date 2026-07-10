//! FP4 (E2M1) micro-format codec and the shared sub-byte field-extraction helper.
//!
//! MX FP4 E2M1 (spec section 2.4.2) is a 4-bit format: sign / 2-bit exponent (bias 1) /
//! 1-bit mantissa, with no infinities and no NaN. Magnitudes: zero `S.00.0`, max subnormal
//! `S.00.1 = +/-0.5`, min normal `S.01.0 = +/-1.0`, max normal `S.11.1 = +/-6.0`. FP4
//! lanes are **nibble-packed** — 4 bits per lane, packed contiguously from bit 0, two lanes
//! per byte (lane `2j` in the low nibble of byte `j`, lane `2j+1` in the high nibble), so a
//! length-`N` lane vector occupies `N/2` bytes (spec section 9.4.5 / 9.5.5).
//!
//! This module owns [`extract_field`], the size-parameterized LSB-from-bit-0 sub-byte
//! reader reused by every packed micro-format (FP4 size 4, FP6 size 6, and the `VUNPACKB`
//! family). It also owns the FP8->FP4 saturating-RTNE conversion helpers
//! ([`fp8_e5m2_to_fp4_e2m1`] / [`fp8_e4m3_to_fp4_e2m1`], spec section 16.3, consumed by
//! family D), the exact FP4->FP8 E4M3 decode ([`fp4_e2m1_to_fp8_e4m3`], spec section 9.5.5,
//! consumed by family E) and the nibble pack/unpack primitives the converters build on.
//!
//! # Iteration-2 open-question resolutions
//!
//! * **OQ-3 — two-instruction-family naming.** The FP8→FP4 stem `cvtf8_bf4s` maps to TWO spec
//!   instructions (one per source FP8 format) sharing the intrinsic stem but differing in
//!   source format. RESOLVED: the two public converts are disambiguated by a source-format
//!   suffix — `cvtf8_bf4s_e5m2` / `cvtf8_bf4s_e4m3` (family D) — to be reconciled against the
//!   final stdarch names at upstream time.
//! * **OQ-4 — per-family DAZ.** The forward saturating helpers
//!   ([`fp8_e5m2_to_fp4_e2m1`] / [`fp8_e4m3_to_fp4_e2m1`]) assume DAZ=1; the exact reverse
//!   decode ([`fp4_e2m1_to_fp8_e4m3`]) assumes DAZ=0 — encoded per helper, not globally.
//! * **OQ-5 — native-path reachability.** Families D/E ship **oracle-only** in this toolchain:
//!   the `_mm512_cvtf8_bf4s` / `_mm512_cvtbf4_hf8` intrinsics are absent under `-mavx10.2`, so
//!   there is no native C shim; the differential discards rather than failing (still fully
//!   correct via the scalar oracle, `[avx10-v2-aux-ocp-conversions.CORRECTNESS.2]`).

/// Extract the `size`-bit little-endian field that starts at `bit_offset` from a packed
/// byte buffer, returning it right-aligned in a `u8` (bits `[size-1:0]`, higher bits zero).
///
/// Bits are numbered LSB-from-bit-0 within each byte and contiguously across bytes: bit `b`
/// of the buffer is bit `b & 7` of `buf[b >> 3]`. A field may straddle a byte boundary
/// (e.g. a size-6 field at bit offset 4 spans the top 4 bits of byte 0 and the low 2 bits
/// of byte 1). `size` must be in `1..=8`; the caller guarantees `bit_offset + size` fits in
/// the buffer. This is the inverse of the packers in this module and in `crate::fp6`, and
/// the field-read primitive the section-9.9.4 `vunpackb` decode is defined in terms of.
pub(crate) fn extract_field(buf: &[u8], bit_offset: usize, size: usize) -> u8 {
    assert!((1..=8).contains(&size));
    let mut acc: u16 = 0;
    let mut got = 0;
    let mut pos = bit_offset;
    // Read bit-by-byte: take as many of the wanted bits as live in the current byte, then
    // advance. At most two byte reads are needed for size <= 8, but the loop is general.
    while got < size {
        let byte_idx = pos >> 3;
        let bit_in_byte = pos & 7;
        let avail = 8 - bit_in_byte; // bits remaining in this byte
        let take = avail.min(size - got);
        let mask = (1u16 << take) - 1;
        let chunk = ((buf[byte_idx] >> bit_in_byte) as u16) & mask;
        acc |= chunk << got;
        got += take;
        pos += take;
    }
    (acc & ((1u16 << size) - 1)) as u8
}

/// Convert one FP8 E4M3 (HF8) byte to its FP4 E2M1 (BF4) nibble, RTNE and always saturating.
///
/// Transcribes the ACE v1 section-16.3 `fp8_e4m3_to_fp4_e2m1` helper verbatim (spec section
/// 9.4 `VCVTHF82BF4S`). FP4 E2M1 is sign / 2-bit exponent (bias 1) / 1-bit mantissa with NO
/// NaN and NO Inf (spec section 2.4.2); magnitudes are `S.00.0 = 0`, `S.00.1 = +/-0.5`
/// (max subnormal), `S.01.0 = +/-1.0` (min normal), `S.11.1 = +/-6.0` (max normal). The
/// returned `u8` holds the 4-bit code right-aligned in bits `[3:0]`.
///
/// Always saturating (spec section 9.4.1): the sole HF8 NaN `S.1111.111` and any HF8 whose
/// magnitude exceeds the FP4 max normal `+/-6.0` clamp to the same-signed max normal
/// `e_o=0x3, m_o=0x1` (`[avx10-v2-aux-ocp-conversions.CVT_FP8_FP4.2]`). DAZ=1: every HF8
/// zero/subnormal maps to FP4 zero. The subnormal-output branch rounds RTNE (round half to
/// even); the normal branch rounds via the spec's `rnex = i + 0x01 + fixup` round-then-rebias
/// (`[avx10-v2-aux-ocp-conversions.CVT_FP8_FP4.1]`).
pub(crate) fn fp8_e4m3_to_fp4_e2m1(byte: u8) -> u8 {
    let i = byte as u32;
    let s_i = (i & 0x80) >> 7;
    let e_i = (i & 0x78) >> 3; // 4-bit biased exponent (bias 7)
    let m_i = i & 0x07; // 3-bit mantissa
    let exp_rebias: i32 = 7 - 1; // FP4 E2M1 bias = 1; FP8 E4M3 bias = 7
    let new_exp: i32 = e_i as i32 - exp_rebias;

    let mut e_o: u32;
    let mut m_o: u32;

    if e_i == 0xF && m_i == 0x7 {
        // NaN -> clamp to FP4 max normal (FP4 has no NaN).
        e_o = 0x3;
        m_o = 0x1;
    } else if (e_i as i32 > exp_rebias + 3) || (e_i as i32 == exp_rebias + 3 && m_i > 0x4) {
        // Overflow -> clamp to FP4 max normal +/-6.0.
        e_o = 0x3;
        m_o = 0x1;
    } else if e_i == 0x00 {
        // Zero or denorm (DAZ=1) -> FP4 zero.
        e_o = 0;
        m_o = 0;
    } else if new_exp <= 0 {
        // Underflow -> FP4 subnormal or zero, RTNE.
        e_o = 0;
        m_o = 0;
        if (3 - new_exp) <= 4 {
            let mant = m_i | 0x8; // restore hidden bit (E4M3 has 3 mantissa bits)
            let shift = (3 - new_exp) as u32;
            m_o = mant >> shift;
            let lowmant = mant & crate::fp8::mask(shift as i32);
            let halfway = 1u32 << (shift - 1);
            if lowmant > halfway || (lowmant == halfway && (m_o & 0x1) != 0) {
                m_o += 1;
                if (m_o & 0x1) == 0 {
                    e_o += 1;
                }
            }
        }
    } else {
        // Normal: round-then-rebias. `fixup = m_i[2]` (top mantissa bit) is the RTNE
        // tie-to-even adjustment; `rnex = i + 0x01 + fixup` rounds the byte directly.
        let fixup = (m_i >> 2) & 0x1;
        let rnex = i + 0x01 + fixup;
        e_o = (((rnex & 0x78) >> 3) as i32 - exp_rebias) as u32;
        m_o = (rnex & 0x07) >> 2;
    }

    (((s_i & 0x1) << 3) | ((e_o & 0x3) << 1) | (m_o & 0x1)) as u8
}

/// Convert one FP8 E5M2 (BF8) byte to its FP4 E2M1 (BF4) nibble, RTNE and always saturating.
///
/// Transcribes the ACE v1 section-16.3 `fp8_e5m2_to_fp4_e2m1` helper verbatim (spec section
/// 9.4 `VCVTBF82BF4S`). Returns the 4-bit code right-aligned in bits `[3:0]`. Always
/// saturating (spec section 9.4.1): every BF8 +/-Inf (`S.11111.00`) and NaN
/// (`S.11111.{01,10,11}`), and any BF8 whose magnitude exceeds the FP4 max normal `+/-6.0`,
/// clamp to the same-signed max normal `e_o=0x3, m_o=0x1`
/// (`[avx10-v2-aux-ocp-conversions.CVT_FP8_FP4.2]`). DAZ=1: every BF8 zero/subnormal maps to
/// FP4 zero. The subnormal-output branch rounds RTNE; the normal branch rounds via the
/// spec's `rnex = i + fixup` round-then-rebias (`[avx10-v2-aux-ocp-conversions.CVT_FP8_FP4.1]`).
pub(crate) fn fp8_e5m2_to_fp4_e2m1(byte: u8) -> u8 {
    let i = byte as u32;
    let s_i = (i & 0x80) >> 7;
    let e_i = (i & 0x7C) >> 2; // 5-bit biased exponent (bias 15)
    let m_i = i & 0x03; // 2-bit mantissa
    let exp_rebias: i32 = 15 - 1; // FP4 E2M1 bias = 1; FP8 E5M2 bias = 15
    let new_exp: i32 = e_i as i32 - exp_rebias;

    let mut e_o: u32;
    let mut m_o: u32;

    if e_i == 0x1F {
        // NaN or Inf (any mantissa) -> clamp to FP4 max normal (FP4 has no NaN/Inf).
        e_o = 0x3;
        m_o = 0x1;
    } else if (e_i as i32 > exp_rebias + 3) || (e_i as i32 == exp_rebias + 3 && m_i > 0x2) {
        // Overflow -> clamp to FP4 max normal +/-6.0.
        e_o = 0x3;
        m_o = 0x1;
    } else if e_i == 0x00 {
        // Zero or denorm (DAZ=1) -> FP4 zero. (m_i == 0 is exact zero; m_i != 0 is a BF8
        // subnormal flushed to signed zero under DAZ=1.)
        e_o = 0;
        m_o = 0;
    } else if new_exp <= 0 {
        // Underflow -> FP4 subnormal or zero, RTNE (J-bit insertion).
        e_o = 0;
        m_o = 0;
        if (2 - new_exp) <= 3 {
            let mant = m_i | 0x4; // restore hidden bit (E5M2 has 2 mantissa bits)
            let shift = (2 - new_exp) as u32;
            m_o = mant >> shift;
            let lowmant = mant & crate::fp8::mask(shift as i32);
            let halfway = 1u32 << (shift - 1);
            if lowmant > halfway || (lowmant == halfway && (m_o & 0x1) != 0) {
                m_o += 1;
                if (m_o & 0x1) == 0 {
                    // FP4 mantissa is 1-bit, so a carry-out bumps the exponent.
                    e_o += 1;
                }
            }
        }
    } else {
        // Normal: direct rebias + RTNE truncate. `fixup = m_i[1]` (top mantissa bit).
        let fixup = (m_i >> 1) & 0x1;
        let rnex = i + fixup;
        e_o = (((rnex & 0x7C) >> 2) as i32 - exp_rebias) as u32;
        m_o = (rnex & 0x03) >> 1;
    }

    (((s_i & 0x1) << 3) | ((e_o & 0x3) << 1) | (m_o & 0x1)) as u8
}

/// Convert one FP4 E2M1 (BF4) nibble to its exact FP8 E4M3 (HF8) byte.
///
/// Transcribes the ACE v1 section-9.5.5 `fp4_to_fp8_e4m3` mapping: the conversion is
/// **exact** — every one of the 16 FP4 encodings maps to exactly one FP8 E4M3 encoding, with
/// no rounding, no saturation and no approximation (spec section 9.5.1, DAZ=0/FTZ=0,
/// `[avx10-v2-aux-ocp-conversions.CVT_FP4_FP8.1]`,
/// `[avx10-v2-aux-ocp-conversions.CVT_FP4_FP8.2]`). FP4 E2M1 is sign(bit 3) / 2-bit exponent
/// (bits `[2:1]`, bias 1) / 1-bit mantissa (bit 0), with no Inf and no NaN (spec section
/// 2.4.2); its eight magnitudes are `S.00.0=0`, `S.00.1=+/-0.5` (max subnormal),
/// `S.01.0=+/-1.0` (min normal), `S.01.1=+/-1.5`, `S.10.0=+/-2.0`, `S.10.1=+/-3.0`,
/// `S.11.0=+/-4.0`, `S.11.1=+/-6.0` (max normal). Every one of those eight magnitudes is
/// exactly representable in FP8 E4M3 (sign / 4-bit exp bias 7 / 3-bit mantissa), so the map
/// is a magnitude LUT plus the carried sign.
///
/// The `nibble` argument carries the FP4 code right-aligned in bits `[3:0]` (higher bits
/// ignored); the returned `u8` is the full FP8 E4M3 byte.
///
/// The eight E4M3 magnitude bytes, derived directly from the FP8 E4M3 layout
/// `value = (1 + m/8) * 2^(e-7)` for normals:
///   `0.0 -> 0x00` (`S.0000.000`), `0.5 = 2^-1 -> 0x30` (`S.0110.000`),
///   `1.0 = 2^0 -> 0x38` (`S.0111.000`), `1.5 = (1+4/8)*2^0 -> 0x3C` (`S.0111.100`),
///   `2.0 = 2^1 -> 0x40` (`S.1000.000`), `3.0 = (1+4/8)*2^1 -> 0x44` (`S.1000.100`),
///   `4.0 = 2^2 -> 0x48` (`S.1001.000`), `6.0 = (1+4/8)*2^2 -> 0x4C` (`S.1001.100`).
pub(crate) fn fp4_e2m1_to_fp8_e4m3(nibble: u8) -> u8 {
    // Magnitude LUT indexed by the 3-bit FP4 code (exp<<1 | mantissa), i.e. the FP4 nibble
    // with its sign bit cleared. Each entry is the exact E4M3 byte (sign clear) for that
    // FP4 magnitude (spec section 2.4.2 / 9.5.5).
    const LUT: [u8; 8] = [
        0x00, // S.00.0 = 0.0   -> S.0000.000
        0x30, // S.00.1 = 0.5   -> S.0110.000
        0x38, // S.01.0 = 1.0   -> S.0111.000
        0x3C, // S.01.1 = 1.5   -> S.0111.100
        0x40, // S.10.0 = 2.0   -> S.1000.000
        0x44, // S.10.1 = 3.0   -> S.1000.100
        0x48, // S.11.0 = 4.0   -> S.1001.000
        0x4C, // S.11.1 = 6.0   -> S.1001.100
    ];
    let sign = (nibble >> 3) & 0x1;
    let mag = LUT[(nibble & 0x7) as usize];
    (sign << 7) | mag
}

/// Pack a slice of sub-byte values into a little-endian bit-packed byte buffer — the write
/// side of [`extract_field`].
///
/// Lane `i` (low `size` bits of `values[i]`) is written at bit offset `size * i`,
/// contiguously from bit 0, straddling a byte boundary when needed. This is the single
/// generic packer behind the FP4 nibble pack ([`pack_nibbles`], `size = 4`), the FP6 6-bit
/// pack (`crate::fp6::pack`, `size = 6`), and the `unpackb` test inputs (sizes 2–7).
/// `out` must hold at least `values.len() * size` bits; it is zeroed first, so bits past the
/// last lane stay zero.
pub(crate) fn pack_fields(values: &[u8], size: usize, out: &mut [u8]) {
    assert!((1..=8).contains(&size));
    assert!(values.len() * size <= out.len() * 8);
    for byte in out.iter_mut() {
        *byte = 0;
    }
    for (i, &v) in values.iter().enumerate() {
        let field = (v as u16) & ((1u16 << size) - 1);
        let bit_offset = size * i;
        // A field of size <= 8 spans at most two output bytes.
        let lo_byte = bit_offset >> 3;
        let lo_shift = bit_offset & 7;
        out[lo_byte] |= ((field << lo_shift) & 0xff) as u8;
        let written = 8 - lo_shift; // bits placed in lo_byte
        if written < size {
            out[lo_byte + 1] |= (field >> written) as u8;
        }
    }
}

/// Pack a slice of 4-bit FP4 nibbles into a nibble-packed byte buffer.
///
/// Lane `i` (low 4 bits of `nibbles[i]`) is written at bit offset `4 * i`: even lanes in the
/// low nibble of their byte, odd lanes in the high nibble, two lanes per output byte from
/// bit 0 (spec section 9.4.5). `nibbles.len()` must be even; the output is
/// `nibbles.len() / 2` bytes. Every nibble is written (no masking/zeroing), the inverse of
/// [`unpack_nibbles`]. Thin `size = 4` wrapper over [`pack_fields`].
pub(crate) fn pack_nibbles(nibbles: &[u8], out: &mut [u8]) {
    assert_eq!(nibbles.len() % 2, 0);
    assert_eq!(out.len(), nibbles.len() / 2);
    pack_fields(nibbles, 4, out);
}

/// Unpack a nibble-packed byte buffer into one right-aligned 4-bit value per lane.
///
/// Reads lane `i` from bit offset `4 * i` via [`extract_field`] with `size = 4`, the inverse
/// of [`pack_nibbles`]: `out[i]` holds the lane's 4 bits in `[3:0]` with higher bits zero.
/// `out.len()` must be `2 * buf.len()`. Test-only: the shipping family-E read-and-widen path
/// uses [`unpack_nibbles_to_fp8_e4m3`]; this generic form is exercised only by the
/// pack/unpack round-trip test, so it is gated `#[cfg(test)]` to stay dead-code-clean.
#[cfg(test)]
pub(crate) fn unpack_nibbles(buf: &[u8], out: &mut [u8]) {
    assert_eq!(out.len(), 2 * buf.len());
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = extract_field(buf, 4 * i, 4);
    }
}

/// Unpack a nibble-packed FP4 buffer into one exact FP8 E4M3 byte per FP4 lane.
///
/// Reads FP4 lane `i` from bit offset `4 * i` via [`extract_field`] (`size = 4`) and maps it
/// through the exact [`fp4_e2m1_to_fp8_e4m3`] LUT, writing the FP8 E4M3 byte to `out[i]`. The
/// output is twice the packed input width: `out.len()` must be `2 * buf.len()` (spec section
/// 9.5.5, `[avx10-v2-aux-ocp-conversions.CVT_FP4_FP8.3]`). The inverse pack step (FP8->FP4
/// nibble pack) is [`pack_nibbles`]; this is the family-E read-and-widen primitive.
pub(crate) fn unpack_nibbles_to_fp8_e4m3(buf: &[u8], out: &mut [u8]) {
    assert_eq!(out.len(), 2 * buf.len());
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = fp4_e2m1_to_fp8_e4m3(extract_field(buf, 4 * i, 4));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_field_size6_straddles_byte_boundary() {
        // Two adjacent size-6 fields packed from bit 0:
        //   field0 = 0b101101 (bits 0..6), field1 = 0b011010 (bits 6..12).
        // Packed little-endian (LSB-from-bit-0): the 12 bits are
        //   bit0..5  = field0 = 101101 (read low-to-high: 1,0,1,1,0,1)
        //   bit6..11 = field1 = 011010
        // Byte 0 = bits 0..7  = field0(6 bits) | low 2 bits of field1.
        // Byte 1 = bits 8..15 = high 4 bits of field1.
        let field0 = 0b101101u8; // 45
        let field1 = 0b011010u8; // 26
        let packed0 = field0 | (field1 << 6); // bits 0..7
        let packed1 = field1 >> 2; // bits 8..11
        let buf = [packed0, packed1];

        // field1 is the one that straddles the byte-0/byte-1 boundary (bits 6..11): this is
        // the load-bearing case. A reader that ignored the carry into byte 1 would return a
        // truncated value.
        assert_eq!(extract_field(&buf, 0, 6), field0, "field0 within byte 0");
        assert_eq!(
            extract_field(&buf, 6, 6),
            field1,
            "field1 straddles the byte-0/byte-1 boundary"
        );
    }

    #[test]
    fn nibble_pack_unpack_round_trips() {
        // Four FP4 nibbles -> 2 packed bytes -> back. Lane 0 low nibble of byte 0, lane 1
        // high nibble of byte 0, lane 2 low nibble of byte 1, lane 3 high nibble of byte 1.
        let lanes = [0x3u8, 0xc, 0x5, 0xa];
        let mut packed = [0u8; 2];
        pack_nibbles(&lanes, &mut packed);
        // Byte 0 = (0xc << 4) | 0x3 = 0xc3; byte 1 = (0xa << 4) | 0x5 = 0xa5.
        assert_eq!(packed, [0xc3, 0xa5]);
        let mut out = [0u8; 4];
        unpack_nibbles(&packed, &mut out);
        assert_eq!(out, lanes);
    }

    // FP4 E2M1 nibble assembler: sign | 2-bit exp | 1-bit mantissa (spec section 2.4.2).
    fn fp4(sign: u8, exp: u8, mant: u8) -> u8 {
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

    /// Hand-computed FP8 E4M3 -> FP4 E2M1 conversions pinning each branch of the section-16.3
    /// helper. The FP4 codes use the magnitude table S.00.0=0, S.00.1=+/-0.5, S.01.0=+/-1.0,
    /// S.11.1=+/-6.0 (spec section 2.4.2).
    ///
    /// DISCRIMINATING lanes (each rules out a plausible-but-wrong model):
    ///  * HF8 +8.0 (S.1010.000 = 1.0*2^3, value 8.0 > 6.0) -> S.11.1 (+6.0). A wrap/non-
    ///    saturating model would overflow the 2-bit exponent; saturation pins +6.0.
    ///  * HF8 NaN S.1111.111 -> +6.0 (NOT a NaN code — FP4 has none), ruling out NaN-propagation.
    ///  * HF8 +5.0 (S.1001.010 = 1.25*2^2 = 5.0) is a tie between FP4 4.0 (S.11.0, e_o=3
    ///    m_o=0) and 6.0 (S.11.1, m_o=1); RTNE picks the EVEN mantissa (m_o=0) -> 4.0 (S.11.0),
    ///    ruling out round-half-up (which would give 6.0).
    #[test]
    fn known_value_e4m3_to_fp4() {
        // +0 and -0.
        assert_eq!(fp8_e4m3_to_fp4_e2m1(hf8(0, 0b0000, 0b000)), fp4(0, 0b00, 0));
        assert_eq!(fp8_e4m3_to_fp4_e2m1(hf8(1, 0b0000, 0b000)), fp4(1, 0b00, 0));
        // +1.0 (S.0111.000, bias 7 -> exp field 7) -> min normal S.01.0.
        assert_eq!(fp8_e4m3_to_fp4_e2m1(hf8(0, 0b0111, 0b000)), fp4(0, 0b01, 0));
        // +6.0 (S.1001.100 = 1.5*2^2 = 6.0) -> max normal S.11.1.
        assert_eq!(fp8_e4m3_to_fp4_e2m1(hf8(0, 0b1001, 0b100)), fp4(0, 0b11, 1));
        // SATURATION: +8.0 (S.1010.000 = 1.0*2^3, biased exp 10) exceeds 6.0 -> S.11.1 (+6.0).
        assert_eq!(
            fp8_e4m3_to_fp4_e2m1(hf8(0, 0b1010, 0b000)),
            fp4(0, 0b11, 1),
            "+8.0 saturates to FP4 max normal +6.0"
        );
        // SATURATION: -448 (S.1111.110, E4M3 max normal) -> clamp to S.11.1 (-6.0).
        assert_eq!(
            fp8_e4m3_to_fp4_e2m1(hf8(1, 0b1111, 0b110)),
            fp4(1, 0b11, 1),
            "-448 saturates to FP4 max normal -6.0"
        );
        // NaN S.1111.111 -> clamp to +6.0 (FP4 has no NaN).
        assert_eq!(
            fp8_e4m3_to_fp4_e2m1(hf8(0, 0b1111, 0b111)),
            fp4(0, 0b11, 1),
            "E4M3 NaN clamps to FP4 max normal +6.0"
        );
        // RTNE TIE: +5.0 (S.1001.010 = 1.25*2^2) is halfway between FP4 4.0 (S.11.0, e_o=3
        // m_o=0, value (1+0)*2^2) and 6.0 (S.11.1, m_o=1); round-half-to-even picks the even
        // mantissa (m_o=0) -> 4.0 (S.11.0). Round-half-up would wrongly give 6.0 (S.11.1).
        assert_eq!(
            fp8_e4m3_to_fp4_e2m1(hf8(0, 0b1001, 0b010)),
            fp4(0, 0b11, 0),
            "+5.0 RTNE-ties down to even mantissa -> 4.0 (S.11.0)"
        );
        // SUBNORMAL OUTPUT: +0.5 (S.0110.000 = 2^-1) is the FP4 max subnormal S.00.1.
        assert_eq!(
            fp8_e4m3_to_fp4_e2m1(hf8(0, 0b0110, 0b000)),
            fp4(0, 0b00, 1),
            "+0.5 -> FP4 max subnormal S.00.1"
        );
    }

    /// Hand-computed FP8 E5M2 -> FP4 E2M1 conversions pinning each branch of the section-16.3
    /// helper.
    ///
    /// DISCRIMINATING lanes:
    ///  * BF8 +Inf (S.11111.00) and BF8 NaN (S.11111.10) both -> +6.0 (FP4 has no Inf/NaN),
    ///    ruling out Inf/NaN propagation.
    ///  * BF8 +8.0 (S.10010.00 = 1.0*2^3) > 6.0 -> saturates to S.11.1 (+6.0).
    ///  * BF8 SUBNORMAL +2^-16 (S.00000.01) flushes to FP4 +0 under DAZ=1, ruling out a
    ///    DAZ=0 decode that would attempt to represent it.
    #[test]
    fn known_value_e5m2_to_fp4() {
        // +0 / -0.
        assert_eq!(fp8_e5m2_to_fp4_e2m1(bf8(0, 0b00000, 0b00)), fp4(0, 0b00, 0));
        assert_eq!(fp8_e5m2_to_fp4_e2m1(bf8(1, 0b00000, 0b00)), fp4(1, 0b00, 0));
        // +1.0 (S.01111.00, bias 15 -> exp field 15) -> min normal S.01.0.
        assert_eq!(fp8_e5m2_to_fp4_e2m1(bf8(0, 0b01111, 0b00)), fp4(0, 0b01, 0));
        // +6.0 (S.10001.10 = 1.5*2^2) -> max normal S.11.1.
        assert_eq!(fp8_e5m2_to_fp4_e2m1(bf8(0, 0b10001, 0b10)), fp4(0, 0b11, 1));
        // SATURATION: +8.0 (S.10010.00 = 1.0*2^3) exceeds 6.0 -> S.11.1 (+6.0).
        assert_eq!(
            fp8_e5m2_to_fp4_e2m1(bf8(0, 0b10010, 0b00)),
            fp4(0, 0b11, 1),
            "+8.0 saturates to FP4 max normal +6.0"
        );
        // +Inf (S.11111.00) -> clamp to +6.0.
        assert_eq!(
            fp8_e5m2_to_fp4_e2m1(bf8(0, 0b11111, 0b00)),
            fp4(0, 0b11, 1),
            "+Inf clamps to FP4 max normal +6.0"
        );
        // NaN (S.11111.10) -> clamp to +6.0 (sign preserved).
        assert_eq!(
            fp8_e5m2_to_fp4_e2m1(bf8(1, 0b11111, 0b10)),
            fp4(1, 0b11, 1),
            "-NaN clamps to FP4 max normal -6.0"
        );
        // SUBNORMAL INPUT: +2^-16 (S.00000.01, BF8 min subnormal) -> FP4 +0 (DAZ=1 flush).
        assert_eq!(
            fp8_e5m2_to_fp4_e2m1(bf8(0, 0b00000, 0b01)),
            fp4(0, 0b00, 0),
            "BF8 subnormal flushes to FP4 +0 under DAZ=1"
        );
        // +0.5 (S.01110.00 = 2^-1) -> FP4 max subnormal S.00.1.
        assert_eq!(
            fp8_e5m2_to_fp4_e2m1(bf8(0, 0b01110, 0b00)),
            fp4(0, 0b00, 1),
            "+0.5 -> FP4 max subnormal S.00.1"
        );
    }

    /// Exact FP4 E2M1 -> FP8 E4M3 LUT decode (spec section 9.5.5 / 2.4.2), pinning each of the
    /// eight FP4 magnitudes to its hand-derived E4M3 byte and confirming the sign bit is
    /// carried into E4M3 bit 7. Each expected byte is computed independently from the E4M3
    /// layout `value = (1 + m/8) * 2^(e-7)` (e = biased exponent), so this distinguishes the
    /// exact mapping from a wrong rebias (e.g. forgetting the +6 exponent rebias would land on
    /// the wrong E4M3 binade). `[avx10-v2-aux-ocp-conversions.CVT_FP4_FP8.1]`
    /// `[avx10-v2-aux-ocp-conversions.CVT_FP4_FP8.2]`
    #[test]
    fn known_value_fp4_to_e4m3_lut() {
        // (FP4 (exp,m) magnitude, expected E4M3 byte sign-clear).
        // 0.5 = 2^-1 -> E4M3 e=6 -> S.0110.000 = 0x30.
        assert_eq!(
            fp4_e2m1_to_fp8_e4m3(fp4(0, 0b00, 0)),
            hf8(0, 0b0000, 0b000),
            "0.0"
        );
        assert_eq!(
            fp4_e2m1_to_fp8_e4m3(fp4(0, 0b00, 1)),
            hf8(0, 0b0110, 0b000),
            "0.5 = 2^-1"
        );
        assert_eq!(
            fp4_e2m1_to_fp8_e4m3(fp4(0, 0b01, 0)),
            hf8(0, 0b0111, 0b000),
            "1.0 = 2^0"
        );
        assert_eq!(
            fp4_e2m1_to_fp8_e4m3(fp4(0, 0b01, 1)),
            hf8(0, 0b0111, 0b100),
            "1.5"
        );
        assert_eq!(
            fp4_e2m1_to_fp8_e4m3(fp4(0, 0b10, 0)),
            hf8(0, 0b1000, 0b000),
            "2.0 = 2^1"
        );
        assert_eq!(
            fp4_e2m1_to_fp8_e4m3(fp4(0, 0b10, 1)),
            hf8(0, 0b1000, 0b100),
            "3.0"
        );
        assert_eq!(
            fp4_e2m1_to_fp8_e4m3(fp4(0, 0b11, 0)),
            hf8(0, 0b1001, 0b000),
            "4.0 = 2^2"
        );
        // S.11.1 = +6.0 (the FP4 max normal) -> E4M3 S.1001.100 = 0x4C.
        assert_eq!(
            fp4_e2m1_to_fp8_e4m3(fp4(0, 0b11, 1)),
            hf8(0, 0b1001, 0b100),
            "6.0 (S.11.1)"
        );

        // SIGN: every negative FP4 lane carries its sign into E4M3 bit 7, magnitude unchanged.
        for code in 0u8..8 {
            let pos = fp4_e2m1_to_fp8_e4m3(code);
            let neg = fp4_e2m1_to_fp8_e4m3(code | 0x8);
            assert_eq!(
                neg,
                pos | 0x80,
                "negative FP4 code {code:#x} sets E4M3 sign bit"
            );
        }
        // -6.0 specifically -> S.1001.100 with sign = 0xCC.
        assert_eq!(
            fp4_e2m1_to_fp8_e4m3(fp4(1, 0b11, 1)),
            hf8(1, 0b1001, 0b100),
            "-6.0"
        );
    }
}
