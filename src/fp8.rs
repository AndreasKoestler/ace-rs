//! Shared FP8/FP16 conversion oracle.
//!
//! Bit-exact decode/encode building blocks reused by the FP16->FP8 convert families.
//! BF8 is FP8 E5M2 (5 exponent bits, 2 mantissa bits, bias 15, max normal +/-57344,
//! NaN `S.11111.{01,10,11}`, min subnormal +/-2^-16); HF8 is FP8 E4M3 (4 exponent bits,
//! 3 mantissa bits, bias 7, max normal +/-448 `S.1111.110`, NaN `S.1111.111`, min
//! subnormal +/-2^-9 `S.0000.001`). Conversions round-to-nearest-even (RTNE), consult
//! no MXCSR, raise no FP exceptions, and assume DAZ=0 / FTZ=0, per ACE v1 spec section
//! 8.2.1 (`[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_PH2FP8.1-3]`).
//!
//! OQ-4 (FP8 non-saturating overflow encoding): the oracle is grounded against AVX10.2
//! hardware (verified under Intel SDE) and the section-2.4.1 format table. Non-saturating
//! overflow of a finite/Inf magnitude maps to the format's OVERFLOW encoding, which differs
//! by format: **E5M2 (BF8) has an IEEE infinity** `S.11111.00` (the section-2.4.1 NaN set is
//! `S.11111.{01,10,11}`, so `S.11111.00` is Inf, not NaN), while **E4M3 (HF8) has no infinity**
//! so its overflow is the sole all-ones NaN `S.1111.111`. Saturating clamps to the format max
//! normal. An FP16 NaN *input* always propagates to a NaN regardless of mode. (An earlier
//! oracle emitted a nonzero-mantissa NaN for E5M2 overflow; that disagreed with hardware and
//! is corrected to the Inf encoding here.) (`[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_PH2FP8.1-1]`,
//! `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_PH2FP8.1-2]`).
//!
//! OQ-5 (family-C bias-source layout + rounding): per spec section 8.4.5 the per-lane bias
//! term for output lane `i` is `bias = src1.byte[2 * i]` — the low byte of the i-th `u16`
//! element of the bias operand. [`fp16_to_bf8_biased`] / [`fp16_to_hf8_biased`] take that
//! already-extracted `u8` bias (the caller selects `bias_lane & 0xff`). The bias-rounding rule
//! is grounded against hardware (verified under SDE): the bias is added into the discarded-
//! fraction window and the result is then **truncated** (round toward zero). This is NOT
//! "add-bias-then-RTNE": with `bias == 0` it truncates (so it differs from plain RTNE on
//! above-half inputs), and `bias == 0x80` recovers round-to-nearest. (An earlier oracle modelled
//! the bias as add-then-RTNE; that disagreed with hardware and is corrected here.)
//! (`[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_BIAS_PH2FP8.1]`,
//! `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_BIAS_PH2FP8.1-1]`).

/// Per-format parameters for an OCP MX FP8 target (no infinities).
struct Fp8Format {
    /// Number of mantissa (fraction) bits.
    mant_bits: u32,
    /// Number of exponent bits.
    exp_bits: u32,
    /// Exponent bias.
    bias: i32,
}

/// BF8 = FP8 E5M2: 5 exponent bits, 2 mantissa bits, bias 15 (spec section 2.4.1).
const BF8: Fp8Format = Fp8Format {
    mant_bits: 2,
    exp_bits: 5,
    bias: 15,
};

/// HF8 = FP8 E4M3: 4 exponent bits, 3 mantissa bits, bias 7 (spec section 2.4.1).
/// Max normal +/-448 (`S.1111.110`), NaN `S.1111.111`, min subnormal +/-2^-9.
const HF8: Fp8Format = Fp8Format {
    mant_bits: 3,
    exp_bits: 4,
    bias: 7,
};

/// Decode an FP16 bit pattern into `(sign, class)`.
///
/// FP16 is S/E5/M10, bias 15. Returns the sign bit (0/1) and a classification carrying
/// the exact value for finite inputs as `mantissa * 2^exp2` where `mantissa` is the
/// integer significand (implicit bit made explicit) and `exp2` is the power-of-two
/// scale of the least-significant mantissa bit.
fn decode_fp16(bits: u16) -> (u32, Fp16Class) {
    let sign = (bits >> 15) as u32 & 1;
    let exp = (bits >> 10) & 0x1f;
    let mant = (bits & 0x3ff) as u64;
    let class = if exp == 0x1f {
        if mant == 0 {
            Fp16Class::Inf
        } else {
            Fp16Class::NaN
        }
    } else if exp == 0 {
        if mant == 0 {
            Fp16Class::Zero
        } else {
            // Subnormal: value = mant * 2^(1-15-10) = mant * 2^-24.
            Fp16Class::Finite {
                mantissa: mant,
                exp2: 1 - 15 - 10,
            }
        }
    } else {
        // Normal: value = (1024 + mant) * 2^(exp-15-10).
        Fp16Class::Finite {
            mantissa: 1024 + mant,
            exp2: exp as i32 - 15 - 10,
        }
    };
    (sign, class)
}

enum Fp16Class {
    Zero,
    Inf,
    NaN,
    /// value = mantissa * 2^exp2 (mantissa > 0).
    Finite {
        mantissa: u64,
        exp2: i32,
    },
}

/// Round a strictly-positive finite value `mantissa * 2^exp2` to the target FP8 format and
/// return `(exp_field, mant_field, overflowed)`. `overflowed` is set when the rounded
/// magnitude exceeds the format's max normal.
///
/// Two rounding modes, both grounded against AVX10.2 hardware (verified under Intel SDE):
///
/// * **`bias_mode == false` (families A/B/E plain converts)**: round-to-nearest, ties-to-even
///   (RTNE) on the discarded fraction. `bias` is ignored.
/// * **`bias_mode == true` (family C `VCVTBIASPH2*`, spec section 8.4.5 + 2.6.3)**: add the
///   8-bit `bias` into the discarded-fraction window aligned so the bias byte's MSB sits
///   immediately below the target lsb, then **truncate** (round toward zero) — i.e. the only
///   way to round up is a carry out of the discarded window (`discarded + bias >= 2^shift`).
///   This is NOT RTNE: with `bias == 0` it truncates (e.g. an FP16 value just above half an
///   lsb rounds DOWN), and `bias == 0x80` recovers round-to-nearest-up. Hardware
///   `VCVTBIASPH2BF8`/`HF8` implement exactly this add-bias-then-truncate rule. (The earlier
///   oracle modelled bias as "add bias then RTNE", which disagreed with hardware and is
///   corrected here.)
fn round_finite_to_fp8(
    fmt: &Fp8Format,
    mantissa: u64,
    exp2: i32,
    bias: u32,
    bias_mode: bool,
) -> (u32, u32, bool) {
    let max_exp_field: u32 = (1u32 << fmt.exp_bits) - 1;
    // Unbiased exponent of the most-significant set bit, treating the value as
    // `1.fffff * 2^e`.
    let msb = 63 - mantissa.leading_zeros() as i32; // position of top set bit
    let mut e = exp2 + msb; // unbiased exponent of the leading 1

    // Smallest representable unbiased exponent of a *normal* number is `1 - bias`.
    // Subnormals share that scale with a reduced implicit bit.
    let min_normal_exp = 1 - fmt.bias;

    // We want the target significand as a (mant_bits+1)-bit integer `1.m` for normals,
    // i.e. shift the value so the leading bit sits at position `mant_bits`.
    // For subnormals the leading bit lands below `mant_bits`.
    let target_lsb_exp2 = if e >= min_normal_exp {
        // Normal: lsb of the stored mantissa has scale 2^(e - mant_bits).
        e - fmt.mant_bits as i32
    } else {
        // Subnormal: fixed lsb scale 2^(min_normal_exp - mant_bits).
        min_normal_exp - fmt.mant_bits as i32
    };

    // Shift `mantissa * 2^exp2` to integer units of `2^target_lsb_exp2`.
    let shift = target_lsb_exp2 - exp2;

    let rounded: u64 = if shift <= 0 {
        // No fractional bits discarded; exact left shift (bias/rounding cannot apply).
        mantissa << (-shift) as u32
    } else if shift >= 64 {
        // Everything is below the rounding position.
        0
    } else {
        let s = shift as u32;
        let kept = mantissa >> s;
        let rem = mantissa & ((1u64 << s) - 1); // discarded fraction
        if bias_mode {
            // Add-bias-then-truncate (spec 2.6.3): align the 8-bit bias byte so its MSB sits
            // just below the target lsb (top of the discarded window), then a round-up
            // happens iff the addition carries out of the window. Truncation otherwise.
            let bias_scaled = if s >= 8 {
                (bias as u64) << (s - 8)
            } else {
                (bias as u64) >> (8 - s)
            };
            kept + ((rem + bias_scaled) >> s)
        } else {
            // Plain RTNE.
            let half = 1u64 << (s - 1);
            if rem > half || (rem == half && (kept & 1) == 1) {
                kept + 1
            } else {
                kept
            }
        }
    };

    if rounded == 0 {
        // Underflowed to zero.
        return (0, 0, false);
    }

    // Re-derive exponent/mantissa fields from the rounded integer significand, which
    // may have carried into a new binade.
    let rounded_msb = 63 - rounded.leading_zeros() as i32;
    // Scale of the rounded integer's lsb is target_lsb_exp2; its value is
    // rounded * 2^target_lsb_exp2, so the leading bit has unbiased exponent
    // target_lsb_exp2 + rounded_msb.
    e = target_lsb_exp2 + rounded_msb;

    if e < min_normal_exp {
        // Subnormal result: exponent field 0, mantissa is the low mant_bits of the
        // value expressed in units of 2^(min_normal_exp - mant_bits).
        let mant_field = (rounded as u32) & ((1u32 << fmt.mant_bits) - 1);
        return (0, mant_field, false);
    }

    let exp_field = (e + fmt.bias) as u32;
    if exp_field > max_exp_field {
        return (0, 0, true);
    }
    if exp_field == max_exp_field {
        // All-ones exponent. For E5M2 the entire max-exponent binade encodes NaN, so
        // any value there overflows. For E4M3 only `S.1111.111` is NaN, so the max
        // exponent holds genuine normals up to `S.1111.110` (=448); a mantissa that
        // rounds to all-ones overflows into the NaN slot.
        let mant_field = (rounded as u32) & ((1u32 << fmt.mant_bits) - 1);
        let max_mant = (1u32 << fmt.mant_bits) - 1;
        if fmt.exp_bits == 5 || mant_field == max_mant {
            return (0, 0, true);
        }
        return (exp_field, mant_field, false);
    }
    // Normal: strip the implicit leading 1, keep the low mant_bits.
    let mant_field = (rounded as u32) & ((1u32 << fmt.mant_bits) - 1);
    (exp_field, mant_field, false)
}

/// Encode the FP8 result for the **non-saturating overflow** of a finite (or infinite) FP16
/// magnitude, grounded against AVX10.2 hardware (verified under Intel SDE) and the section
/// 2.4.1 format table:
///
/// * **E5M2 (BF8)** has the IEEE infinity encoding `S.11111.00` (the section-2.4.1 NaN set is
///   `S.11111.{01,10,11}`, so `S.11111.00` is *not* a NaN — it is infinity). Hardware
///   `VCVTPH2BF8` emits exactly this for a finite-magnitude overflow and for an `+/-Inf`
///   input in non-saturating mode. (Originally the oracle emitted a nonzero-mantissa NaN
///   here; that disagreed with hardware and is corrected to the Inf encoding.)
/// * **E4M3 (HF8)** has no infinity; its sole all-ones slot `S.1111.111` is NaN, which is
///   what hardware emits for an HF8 overflow / `+/-Inf` input. So overflow maps to that NaN.
fn fp8_overflow(fmt: &Fp8Format, sign: u32) -> u8 {
    let max_exp_field = (1u32 << fmt.exp_bits) - 1;
    let mant = if fmt.exp_bits == 5 {
        0 // E5M2: S.11111.00 = +/-Inf
    } else {
        (1u32 << fmt.mant_bits) - 1 // E4M3: S.1111.111 = NaN
    };
    ((sign << 7) | (max_exp_field << fmt.mant_bits) | mant) as u8
}

/// Encode the FP8 result for **propagating an FP16 NaN input** for the given format, matching
/// AVX10.2 hardware (verified under SDE):
///
/// * **E5M2 (BF8)**: the result mantissa is the top two FP16 mantissa bits with the quiet bit
///   forced on (`((fp16_mant >> 8) & 0b11) | 0b10`), giving a quiet NaN in `S.11111.{10,11}`.
/// * **E4M3 (HF8)**: the sole NaN encoding `S.1111.111`.
///
/// `fp16_mant` is the raw 10-bit FP16 mantissa of the NaN input.
fn fp8_nan_from_input(fmt: &Fp8Format, sign: u32, fp16_mant: u32) -> u8 {
    let max_exp_field = (1u32 << fmt.exp_bits) - 1;
    let mant = if fmt.exp_bits == 5 {
        // E5M2: top two FP16 mantissa bits, quieted (top bit set). Matches hardware.
        ((fp16_mant >> 8) & 0b11) | 0b10
    } else {
        (1u32 << fmt.mant_bits) - 1 // E4M3: sole NaN slot
    };
    ((sign << 7) | (max_exp_field << fmt.mant_bits) | mant) as u8
}

/// FP8 max-normal magnitude byte for the given sign and format.
///
/// E5M2: `S.11110.11` = +/-57344. E4M3: `S.1111.110` = +/-448 (the max-exponent binade
/// minus the all-ones-mantissa NaN slot).
fn fp8_max_normal(fmt: &Fp8Format, sign: u32) -> u8 {
    let max_exp_field = (1u32 << fmt.exp_bits) - 1;
    let max_mant = (1u32 << fmt.mant_bits) - 1;
    let (exp_field, mant_field) = if fmt.exp_bits == 5 {
        // E5M2: max-exponent binade is NaN; max normal sits one exponent below.
        (max_exp_field - 1, max_mant)
    } else {
        // E4M3: max normal is the max exponent with mantissa just below all-ones.
        (max_exp_field, max_mant - 1)
    };
    ((sign << 7) | (exp_field << fmt.mant_bits) | mant_field) as u8
}

/// Assemble an FP8 byte from sign and the rounded exp/mant fields.
fn fp8_assemble(fmt: &Fp8Format, sign: u32, exp_field: u32, mant_field: u32) -> u8 {
    ((sign << 7) | (exp_field << fmt.mant_bits) | mant_field) as u8
}

/// Convert one FP16 lane (raw bits) to one FP8 byte in the given target format.
///
/// `bias_mode` selects the rounding contract (see [`round_finite_to_fp8`]): `false` is plain
/// RTNE (families A/B), `true` is family C's add-bias-then-truncate using the 8-bit `bias`
/// term (spec 8.4.5 + 2.6.3). Decodes the FP16 pattern to an exact wide intermediate, rounds
/// once to the target FP8 (no double-rounding), and encodes to `u8`. Subnormals, signed
/// zeros, and NaNs are handled bit-exactly with no FTZ/DAZ. On magnitude overflow: when
/// `saturating`, clamp to the format max normal; otherwise emit the format overflow encoding
/// (E5M2 Inf, E4M3 NaN).
fn fp16_to_fp8_biased(
    fmt: &Fp8Format,
    bits: u16,
    saturating: bool,
    bias: u32,
    bias_mode: bool,
) -> u8 {
    let (sign, class) = decode_fp16(bits);
    match class {
        Fp16Class::Zero => (sign << 7) as u8, // S.0...0.0...0
        Fp16Class::NaN => {
            // Propagate as an FP8 NaN (always a NaN encoding, even saturating). Bias does
            // not apply to a non-finite input. The full 10-bit FP16 mantissa is passed so
            // the E5M2 payload mapping can read its top bits (hardware-matched).
            let fp16_mant = (bits & 0x3ff) as u32;
            fp8_nan_from_input(fmt, sign, fp16_mant)
        }
        Fp16Class::Inf => {
            if saturating {
                fp8_max_normal(fmt, sign)
            } else {
                // E5M2 -> Inf encoding; E4M3 (no Inf) -> its NaN slot. (Hardware-matched.)
                fp8_overflow(fmt, sign)
            }
        }
        Fp16Class::Finite { mantissa, exp2 } => {
            let (exp_field, mant_field, overflowed) =
                round_finite_to_fp8(fmt, mantissa, exp2, bias, bias_mode);
            if overflowed {
                if saturating {
                    fp8_max_normal(fmt, sign)
                } else {
                    // Non-saturating finite overflow: E5M2 -> Inf, E4M3 -> NaN.
                    fp8_overflow(fmt, sign)
                }
            } else {
                fp8_assemble(fmt, sign, exp_field, mant_field)
            }
        }
    }
}

/// Convert one FP16 lane (raw bits) to one FP8 byte in the given target format under plain
/// RTNE (no bias rounding).
fn fp16_to_fp8(fmt: &Fp8Format, bits: u16, saturating: bool) -> u8 {
    fp16_to_fp8_biased(fmt, bits, saturating, 0, false)
}

/// Convert one FP16 lane (raw bits) to one BF8 (E5M2) byte.
///
/// On magnitude overflow: when `saturating`, clamp to the BF8 max normal `+/-57344`;
/// otherwise emit the BF8 NaN/overflow encoding `S.11111.{01,10,11}`.
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_PH2FP8.1]`
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_PH2FP8.1-1]`
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_PH2FP8.1-2]`
pub(crate) fn fp16_to_bf8(bits: u16, saturating: bool) -> u8 {
    fp16_to_fp8(&BF8, bits, saturating)
}

/// Convert one FP16 lane (raw bits) to one HF8 (E4M3) byte.
///
/// HF8 is E4M3 (bias 7, max normal +/-448 `S.1111.110`, min subnormal +/-2^-9
/// `S.0000.001`, NaN `S.1111.111`). Mirrors [`fp16_to_bf8`]'s round-once / overflow /
/// saturation structure: on magnitude overflow, `saturating` clamps to +/-448, otherwise
/// emits the HF8 NaN encoding `S.1111.111`.
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_PH2FP8.1]`
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_PH2FP8.1-1]`
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_PH2FP8.1-2]`
pub(crate) fn fp16_to_hf8(bits: u16, saturating: bool) -> u8 {
    fp16_to_fp8(&HF8, bits, saturating)
}

/// Convert one FP16 lane (raw bits) to one BF8 (E5M2) byte using bias rounding.
///
/// `bias` is the 8-bit bias rounding term for this lane — per spec section 8.4.5 the byte
/// `src1.byte[2 * i]` (the low byte of the i-th `u16` of the bias operand). It is applied
/// to the rounding function before the round (spec section 2.6.3) so `bias == 0` is exactly
/// [`fp16_to_bf8`] and a nonzero bias nudges the rounded byte upward. Overflow handling
/// (NaN encoding when non-saturating, clamp to +/-57344 when saturating) is identical to
/// [`fp16_to_bf8`].
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_BIAS_PH2FP8.1]`
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_BIAS_PH2FP8.1-1]`
pub(crate) fn fp16_to_bf8_biased(bits: u16, bias: u8, saturating: bool) -> u8 {
    fp16_to_fp8_biased(&BF8, bits, saturating, bias as u32, true)
}

/// Convert one FP16 lane (raw bits) to one HF8 (E4M3) byte using bias rounding.
///
/// `bias` is the 8-bit bias rounding term for this lane — per spec section 8.4.5 the byte
/// `src1.byte[2 * i]`. Applied before the round (spec section 2.6.3) so `bias == 0` is
/// exactly [`fp16_to_hf8`]. Overflow handling (NaN encoding when non-saturating, clamp to
/// +/-448 when saturating) is identical to [`fp16_to_hf8`].
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_BIAS_PH2FP8.1]`
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_BIAS_PH2FP8.1-1]`
pub(crate) fn fp16_to_hf8_biased(bits: u16, bias: u8, saturating: bool) -> u8 {
    fp16_to_fp8_biased(&HF8, bits, saturating, bias as u32, true)
}

/// Exact lossless decode of one HF8 (E4M3) byte to the equivalent FP16 (E5M10) bit
/// pattern.
///
/// Per ACE v1 spec section 8.5, `VCVTHF82PH` is **exact** — every HF8 value is
/// representable in FP16, so the conversion performs no rounding, no saturation, and
/// raises no exceptions (`[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_HF82PH.1]`). HF8 is E4M3
/// (sign, 4-bit exponent biased 7, 3-bit mantissa; section 2.4.1):
///
/// * `S.0000.000` -> FP16 `S` zero (exp 0, mant 0).
/// * `S.0000.mmm` (mmm != 0) is the subnormal `mmm * 2^-9`. Every such value
///   (`2^-9 .. 7 * 2^-9`) lands in the FP16 *normal* range (FP16 normals reach down to
///   `2^-14`), so it is renormalised: the leading set bit of the 3-bit mantissa becomes
///   FP16's implicit bit and the exponent is set accordingly. Exact — no bits are lost.
/// * `S.1111.111` is the sole HF8 NaN; it maps to an FP16 NaN (all-ones exponent, top
///   mantissa bit set as a canonical quiet NaN).
/// * Any other code is a normal `(1 + mmm/8) * 2^(e-7)`. FP16 shares the same implicit-1
///   normal form, so the unbiased exponent `e - 7` rebiases to `e - 7 + 15 = e + 8` and
///   the 3 mantissa bits sit in the top of FP16's 10-bit field (`mmm << 7`). Exact.
pub(crate) fn hf8_to_fp16(bits: u8) -> u16 {
    let sign = (bits >> 7) as u16 & 1;
    let exp = ((bits >> 3) & 0x0f) as u16; // 4-bit biased exponent (bias 7)
    let mant = (bits & 0x07) as u16; // 3-bit mantissa
    let fp16_sign = sign << 15;

    if exp == 0x0f && mant == 0x07 {
        // Sole HF8 NaN encoding S.1111.111 -> FP16 quiet NaN. Hardware `VCVTHF82PH` places
        // the three HF8 mantissa bits (all set) in the top of the FP16 mantissa field,
        // yielding 0x7f80 / 0xff80 (verified under SDE), which this matches bit-for-bit.
        return fp16_sign | (0x1f << 10) | (0x07 << 7);
    }

    if exp == 0 {
        if mant == 0 {
            // +/-0.
            return fp16_sign;
        }
        // Subnormal mmm * 2^-9, mmm in 1..=7. Renormalise into an FP16 normal: with `k`
        // the index of the leading set bit of mmm (0..=2), the value is
        // 1.<remaining bits> * 2^(k-9), so the unbiased FP16 exponent is (k - 9) and the
        // bits below the leading 1 shift up into the top of FP16's 10-bit field.
        let k = 15 - mant.leading_zeros() as u16; // position of top set bit in mmm (0..=2)
        let unbiased = k as i32 - 9; // value = 1.f * 2^unbiased
        let fp16_exp = (unbiased + 15) as u16; // FP16 bias 15; always >= 1 (normal)
        let frac = mant & ((1 << k) - 1); // bits below the leading 1 (k bits)
        let fp16_mant = frac << (10 - k);
        return fp16_sign | (fp16_exp << 10) | fp16_mant;
    }

    // Normal: value = (1 + mant/8) * 2^(exp - 7). FP16 shares the implicit-1 form, so the
    // unbiased exponent (exp - 7) rebiases to (exp - 7 + 15) = exp + 8, and the 3 mantissa
    // bits occupy the top of FP16's 10-bit mantissa field.
    let fp16_exp = exp + 8;
    let fp16_mant = mant << 7;
    fp16_sign | (fp16_exp << 10) | fp16_mant
}

/// Convert one FP32 value to FP16 (E5M10) under the canonical default RNE contract.
///
/// Family E (`VCVT2PS2PHX`, spec section 8.3) consults MXCSR (and EVEX embedded rounding
/// `{er}`) for the rounding mode on hardware. OQ-6 fixes the oracle's CANONICAL contract:
/// the oracle reads no global state and uses the default — IEEE roundTiesToEven (RNE, spec
/// section 2.6.1) — with DAZ=0 and FTZ=0; embedded rounding `{er}` is NOT surfaced in v1
/// (`[avx10-v1-aux-fp16-fp8-evex-vnni.CVT2_PS2PHX.1-1]`,
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT2_PS2PHX.1-2]`).
///
/// Correctness per IEEE-754 binary32 -> binary16:
///
/// * NaN -> a quiet FP16 NaN (all-ones exponent, top mantissa bit set), sign preserved.
/// * +/-Inf -> FP16 +/-Inf (all-ones exponent, zero mantissa).
/// * Magnitude overflow (rounded result reaches FP16's all-ones exponent) -> +/-Inf, since
///   FP16 *has* infinities (unlike the FP8 targets) and RNE rounds the largest finites
///   toward infinity. With DAZ=0/FTZ=0, subnormal FP32 inputs are honoured and FP16
///   subnormal results are produced rather than flushed.
/// * Signed zero is preserved.
///
/// The mantissa is rounded ONCE (round-to-nearest, ties-to-even) directly from the FP32
/// significand to the FP16 fraction (with the subnormal shift folded into the same round),
/// so there is no double-rounding.
pub(crate) fn fp32_to_fp16_rne(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xff) as i32; // 8-bit biased exponent, bias 127
    let mant = bits & 0x007f_ffff; // 23-bit fraction
    let fp16_sign = sign << 15;

    if exp == 0xff {
        // Inf or NaN.
        if mant == 0 {
            // +/-Inf -> FP16 +/-Inf.
            return fp16_sign | (0x1f << 10);
        }
        // NaN -> quiet FP16 NaN (top mantissa bit set), sign preserved.
        return fp16_sign | (0x1f << 10) | 0x200;
    }

    // Build the exact significand as an integer (implicit bit made explicit) and track the
    // power-of-two scale of its least-significant bit, so the value is
    // `signif * 2^value_lsb_exp2`.
    let (signif, value_lsb_exp2) = if exp == 0 {
        if mant == 0 {
            // +/-0.
            return fp16_sign;
        }
        // FP32 subnormal: value = mant * 2^(1 - 127 - 23) = mant * 2^-149.
        (mant as u64, 1 - 127 - 23)
    } else {
        // Normal: value = (2^23 + mant) * 2^(exp - 127 - 23).
        ((0x0080_0000 | mant) as u64, exp - 127 - 23)
    };

    // Unbiased exponent of the leading set bit of the true value.
    let msb = 63 - signif.leading_zeros() as i32;
    let e = value_lsb_exp2 + msb;

    // FP16: bias 15, 10 mantissa bits. Smallest normal unbiased exponent is 1 - 15 = -14.
    const FP16_MIN_NORMAL_EXP: i32 = -14;
    const FP16_MANT_BITS: i32 = 10;

    // Scale of the FP16 result's least-significant stored mantissa bit. Normal: lsb scale
    // 2^(e - 10). Subnormal: fixed lsb scale 2^(-14 - 10) = 2^-24.
    let target_lsb_exp2 = if e >= FP16_MIN_NORMAL_EXP {
        e - FP16_MANT_BITS
    } else {
        FP16_MIN_NORMAL_EXP - FP16_MANT_BITS
    };

    // Shift the value into integer units of the target lsb, rounding RTNE once.
    let shift = target_lsb_exp2 - value_lsb_exp2;

    let rounded: u64 = if shift <= 0 {
        // No fractional bits discarded; exact left shift.
        signif << (-shift) as u32
    } else if shift >= 64 {
        0
    } else {
        let s = shift as u32;
        let kept = signif >> s;
        let rem = signif & ((1u64 << s) - 1);
        let half = 1u64 << (s - 1);
        if rem > half || (rem == half && (kept & 1) == 1) {
            kept + 1
        } else {
            kept
        }
    };

    if rounded == 0 {
        return fp16_sign;
    }

    // Re-derive the exponent of the rounded significand (the round may have carried into a
    // new binade, e.g. mantissa 0x3ff -> 0x400).
    let rounded_msb = 63 - rounded.leading_zeros() as i32;
    let final_e = target_lsb_exp2 + rounded_msb;

    if final_e < FP16_MIN_NORMAL_EXP {
        // Subnormal FP16 result: exponent field 0, mantissa is the rounded integer (fits in
        // 10 bits while subnormal).
        let mant_field = (rounded as u16) & 0x3ff;
        return fp16_sign | mant_field;
    }

    let exp_field = final_e + 15;
    if exp_field >= 0x1f {
        // Overflow: FP16 has infinities, so RNE pushes the magnitude to +/-Inf.
        return fp16_sign | (0x1f << 10);
    }

    // Normal: strip the implicit leading 1 (keep the low 10 bits of the significand).
    let mant_field = (rounded as u16) & 0x3ff;
    fp16_sign | ((exp_field as u16) << 10) | mant_field
}

#[cfg(test)]
mod tests {
    use super::*;

    // FP16 bit helpers for readable test vectors.
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

    #[test]
    fn fp16_to_bf8_zero_and_signed_zero() {
        // +0 (S.00000.0000000000) -> BF8 +0 (0x00).
        assert_eq!(fp16_to_bf8(fp16_bits(0, 0, 0), false), 0x00);
        // -0 -> BF8 -0 (0x80).
        assert_eq!(fp16_to_bf8(fp16_bits(1, 0, 0), false), 0x80);
    }

    #[test]
    fn fp16_to_bf8_one_is_exact() {
        // 1.0 = FP16 S.01111.0000000000 (exp=15). Shares bias/exp with BF8, mantissa
        // bits all zero -> BF8 1.0 = S.01111.00 = exp 15 << 2 = 0x3c.
        let one = fp16_bits(0, 15, 0);
        assert_eq!(fp16_to_bf8(one, false), bf8(0, 0b01111, 0b00));
    }

    #[test]
    fn fp16_to_bf8_rtne_ties_to_even() {
        // Mantissa 0b00_1000_0000 (=0x080): low 8 bits dropped form exactly half an
        // lsb; kept mantissa bits = 0b00 (even) -> rounds down to 0b00.
        let bits = fp16_bits(0, 15, 0b00_1000_0000);
        assert_eq!(fp16_to_bf8(bits, false) & 0b11, 0b00);
        // Mantissa 0b01_1000_0000: kept = 0b01 (odd), tie -> rounds up to 0b10.
        let bits = fp16_bits(0, 15, 0b01_1000_0000);
        assert_eq!(fp16_to_bf8(bits, false) & 0b11, 0b10);
        // Mantissa 0b00_1000_0001: above half -> rounds up to 0b01.
        let bits = fp16_bits(0, 15, 0b00_1000_0001);
        assert_eq!(fp16_to_bf8(bits, false) & 0b11, 0b01);
    }

    #[test]
    fn fp16_to_bf8_min_subnormal_roundtrip() {
        // BF8 min subnormal is +/-2^-16 = S.00000.01. FP16 2^-16 = exp 0 won't reach;
        // 2^-16 as FP16 is subnormal: 2^-16 = mant * 2^-24 -> mant = 2^8 = 256.
        let bits = fp16_bits(0, 0, 256);
        assert_eq!(fp16_to_bf8(bits, false), bf8(0, 0b00000, 0b01));
        // Negative.
        let bits = fp16_bits(1, 0, 256);
        assert_eq!(fp16_to_bf8(bits, false), bf8(1, 0b00000, 0b01));
    }

    #[test]
    fn fp16_to_bf8_overflow_nonsaturating_is_inf() {
        // E5M2 (BF8) HAS an IEEE infinity encoding S.11111.00 (the section-2.4.1 NaN set is
        // S.11111.{01,10,11}, so S.11111.00 is Inf, not NaN). AVX10.2 hardware `VCVTPH2BF8`
        // emits exactly this for a non-saturating finite overflow and for an +Inf input
        // (verified under Intel SDE). [This corrects an earlier oracle that emitted a
        // nonzero-mantissa NaN here and disagreed with hardware.]
        let inf = fp16_bits(0, 31, 0);
        assert_eq!(
            fp16_to_bf8(inf, false),
            bf8(0, 0b11111, 0b00),
            "+Inf input -> BF8 +Inf"
        );
        // A finite FP16 above 57344 (65504 = FP16 max normal) likewise overflows to +Inf.
        let big = fp16_bits(0, 30, 0x3ff);
        assert_eq!(
            fp16_to_bf8(big, false),
            bf8(0, 0b11111, 0b00),
            "finite overflow -> BF8 +Inf"
        );
        // Negative overflow -> -Inf.
        assert_eq!(
            fp16_to_bf8(fp16_bits(1, 31, 0), false),
            bf8(1, 0b11111, 0b00),
            "-Inf"
        );
    }

    #[test]
    fn fp16_to_bf8_overflow_saturating_clamps() {
        // FP16 +Inf saturating -> BF8 max normal +57344 = S.11110.11.
        let inf = fp16_bits(0, 31, 0);
        assert_eq!(fp16_to_bf8(inf, true), bf8(0, 0b11110, 0b11));
        // A finite FP16 above 57344 (e.g. 65504 = FP16 max normal) saturating -> +57344.
        let fp16_max = fp16_bits(0, 30, 0x3ff);
        assert_eq!(fp16_to_bf8(fp16_max, true), bf8(0, 0b11110, 0b11));
    }

    #[test]
    fn fp16_to_bf8_nan_propagates() {
        // FP16 NaN -> BF8 NaN encoding (exp all ones, mantissa nonzero), both modes.
        let nan = fp16_bits(0, 31, 0x200);
        let got = fp16_to_bf8(nan, false);
        assert_eq!((got >> 2) & 0x1f, 0b11111);
        assert!(got & 0b11 != 0);
        // Even in saturating mode, a NaN input stays a NaN (not clamped to max normal).
        let got_sat = fp16_to_bf8(nan, true);
        assert_eq!((got_sat >> 2) & 0x1f, 0b11111);
        assert!(got_sat & 0b11 != 0);
    }

    #[test]
    fn fp16_to_bf8_max_normal_exact() {
        // BF8 max normal 57344 = 1.75 * 2^15 = 1.11b * 2^15. FP16 representation:
        // exp field = 15+15 = 30, mantissa top two bits set 0b11_0000_0000.
        // Encodes exactly to S.11110.11.
        let bits = fp16_bits(0, 30, 0b11_0000_0000);
        assert_eq!(fp16_to_bf8(bits, false), bf8(0, 0b11110, 0b11));
    }

    // --- HF8 (E4M3) round-trip / known-value unit tests ---

    #[test]
    fn fp16_to_hf8_zero_and_signed_zero() {
        assert_eq!(fp16_to_hf8(fp16_bits(0, 0, 0), false), 0x00);
        assert_eq!(fp16_to_hf8(fp16_bits(1, 0, 0), false), 0x80);
    }

    #[test]
    fn fp16_to_hf8_one_is_exact() {
        // 1.0 = FP16 exp=15. HF8 bias 7 -> exp field 7, mantissa 0 -> S.0111.000.
        let one = fp16_bits(0, 15, 0);
        assert_eq!(fp16_to_hf8(one, false), hf8(0, 0b0111, 0b000));
    }

    #[test]
    fn fp16_to_hf8_min_subnormal() {
        // HF8 min subnormal = +/-2^-9 = S.0000.001. FP16 2^-9 = subnormal:
        // 2^-9 = mant * 2^-24 -> mant = 2^15 = 32768; but 32768 > 0x3ff so 2^-9 is
        // actually a normal FP16: 2^-9 = 1.0 * 2^-9 -> exp field = -9+15 = 6, mant 0.
        let bits = fp16_bits(0, 6, 0);
        assert_eq!(fp16_to_hf8(bits, false), hf8(0, 0b0000, 0b001));
        // Negative.
        let bits = fp16_bits(1, 6, 0);
        assert_eq!(fp16_to_hf8(bits, false), hf8(1, 0b0000, 0b001));
    }

    #[test]
    fn fp16_to_hf8_max_normal_exact() {
        // HF8 max normal 448 = 1.75 * 2^8 = 1.110b * 2^8. FP16: exp field = 8+15 = 23,
        // mantissa top three bits 0b110_0000_0000 -> exact HF8 S.1111.110.
        let bits = fp16_bits(0, 23, 0b11_0000_0000);
        assert_eq!(fp16_to_hf8(bits, false), hf8(0, 0b1111, 0b110));
    }

    #[test]
    fn fp16_to_hf8_overflow_nonsaturating_is_nan() {
        // A value above 448 (e.g. 512 = 2^9 -> exp field 24) non-saturating -> NaN
        // S.1111.111.
        let bits = fp16_bits(0, 24, 0);
        let got = fp16_to_hf8(bits, false);
        assert_eq!(got, hf8(0, 0b1111, 0b111));
        // +Inf likewise -> NaN.
        assert_eq!(
            fp16_to_hf8(fp16_bits(0, 31, 0), false),
            hf8(0, 0b1111, 0b111)
        );
    }

    #[test]
    fn fp16_to_hf8_overflow_saturating_clamps() {
        // 512 saturating -> HF8 max normal +448 = S.1111.110.
        let bits = fp16_bits(0, 24, 0);
        assert_eq!(fp16_to_hf8(bits, true), hf8(0, 0b1111, 0b110));
        // +Inf saturating -> +448.
        assert_eq!(
            fp16_to_hf8(fp16_bits(0, 31, 0), true),
            hf8(0, 0b1111, 0b110)
        );
        // FP16 max normal 65504 saturating -> +448.
        assert_eq!(
            fp16_to_hf8(fp16_bits(0, 30, 0x3ff), true),
            hf8(0, 0b1111, 0b110)
        );
    }

    #[test]
    fn fp16_to_hf8_nan_propagates() {
        // FP16 NaN -> HF8 NaN S.1111.111, both modes.
        let nan = fp16_bits(0, 31, 0x200);
        assert_eq!(fp16_to_hf8(nan, false), hf8(0, 0b1111, 0b111));
        assert_eq!(fp16_to_hf8(nan, true), hf8(0, 0b1111, 0b111));
    }

    #[test]
    fn fp16_to_hf8_normal_with_mantissa_exact() {
        // 1.5 = 1.100b * 2^0 -> HF8 exp 7, mantissa 0b100 -> S.0111.100.
        // FP16 1.5 = exp 15, mantissa 0b10_0000_0000.
        let bits = fp16_bits(0, 15, 0b10_0000_0000);
        assert_eq!(fp16_to_hf8(bits, false), hf8(0, 0b0111, 0b100));
    }

    #[test]
    fn fp16_to_hf8_does_not_overflow_in_max_exponent_normals() {
        // 384 = 1.5 * 2^8 = 1.100b * 2^8 sits in the HF8 max-exponent binade as a genuine
        // normal (S.1111.100), NOT a NaN. This distinguishes E4M3 (max exponent holds
        // normals up to 448) from a naive E5M2-style "max exponent == NaN" model, which
        // would wrongly emit NaN here.
        let bits = fp16_bits(0, 23, 0b10_0000_0000);
        assert_eq!(fp16_to_hf8(bits, false), hf8(0, 0b1111, 0b100));
    }

    // --- Family-C bias-rounding helper unit tests (spec sections 2.6.3 + 8.4.5) ---

    #[test]
    fn bias_rounding_is_add_bias_then_truncate() {
        // OQ-5, hardware-grounded (verified under SDE): family-C bias rounding is NOT
        // "add-bias-then-RTNE". It adds the 8-bit bias into the discarded-fraction window and
        // then TRUNCATES toward zero. Two consequences this test pins:
        //
        //  (1) bias == 0 is TRUNCATION, not RTNE. A value just above half an lsb (which RTNE
        //      rounds up) is rounded DOWN by the bias=0 converter.
        //  (2) bias == 0x80 (half) recovers round-to-nearest (ties up): adding half then
        //      truncating == rounding to nearest.
        let above_half = fp16_bits(0, 15, 0x081); // 1.0 + slightly over half a BF8 lsb
        assert_eq!(
            fp16_to_bf8(above_half, false) & 0b11,
            0b01,
            "plain family-A RTNE rounds the above-half value UP"
        );
        assert_eq!(
            fp16_to_bf8_biased(above_half, 0, false) & 0b11,
            0b00,
            "bias=0 TRUNCATES the above-half value DOWN (not RTNE)"
        );
        assert_eq!(
            fp16_to_bf8_biased(above_half, 0x80, false) & 0b11,
            0b01,
            "bias=0x80 recovers round-to-nearest (rounds the above-half value UP)"
        );
        // Exactly half an lsb: RTNE ties to even (down to 0b00), but bias=0x80 (add half then
        // truncate) carries it UP — the round-to-nearest-ties-UP behaviour bias rounding gives.
        let exactly_half = fp16_bits(0, 15, 0x080);
        assert_eq!(
            fp16_to_bf8(exactly_half, false) & 0b11,
            0b00,
            "RTNE ties to even"
        );
        assert_eq!(
            fp16_to_bf8_biased(exactly_half, 0x80, false) & 0b11,
            0b01,
            "bias=0x80 rounds the exact-half value up (ties up)"
        );
    }

    #[test]
    fn bias_pushes_an_exact_down_value_up_one_lsb() {
        // Pick an FP16 value that rounds DOWN under plain RTNE because its discarded
        // fraction is below half an lsb, then show a large enough bias nudges it up one
        // BF8 lsb. This DISTINGUISHES bias rounding from plain RTNE (which would not move
        // here) and from a model that ignores the bias.
        //
        // BF8 keeps 2 mantissa bits of FP16's 10, so it discards the low 8 bits (shift=8).
        // FP16 mantissa 0b00_0000_0001 (= 1) sits 1/256 of an lsb above 1.0: far below the
        // half-lsb (=128) threshold, so plain RTNE -> BF8 1.0 (mant 0b00).
        let near_one = fp16_bits(0, 15, 0b00_0000_0001);
        assert_eq!(
            fp16_to_bf8(near_one, false) & 0b11,
            0b00,
            "plain RTNE keeps mant 0b00"
        );
        // A bias byte of 0xff aligns its MSB at the half position (bit 7 of the 8 discarded
        // bits): 1 + 0xff = 0x100 >= half (0x80), so the round goes up to mant 0b01.
        assert_eq!(
            fp16_to_bf8_biased(near_one, 0xff, false) & 0b11,
            0b01,
            "max bias rounds up one lsb"
        );
        // A tiny bias (1) leaves it below half (1 + 1 = 2 < 0x80) -> still mant 0b00.
        assert_eq!(
            fp16_to_bf8_biased(near_one, 1, false) & 0b11,
            0b00,
            "tiny bias does not reach half"
        );
    }

    #[test]
    fn bias_overflow_matches_plain_overflow_handling() {
        // Bias on an already-overflowing magnitude still produces the format NaN/overflow
        // (non-saturating) or clamps to max normal (saturating) — bias does not change the
        // overflow handling (spec section 8.4.5 reuses the family-A saturation path).
        let big = fp16_bits(0, 30, 0x3ff); // 65504, overflows both BF8 and HF8
        assert_eq!(
            (fp16_to_bf8_biased(big, 0x40, false) >> 2) & 0x1f,
            0b11111,
            "bf8 nsat overflow stays NaN"
        );
        assert_eq!(
            fp16_to_bf8_biased(big, 0x40, true),
            bf8(0, 0b11110, 0b11),
            "bf8 sat overflow clamps"
        );
        assert_eq!(
            fp16_to_hf8_biased(big, 0x40, true),
            hf8(0, 0b1111, 0b110),
            "hf8 sat overflow clamps"
        );
    }

    // --- HF8 (E4M3) -> FP16 exact decode unit tests (spec section 8.5) ---

    // FP16 field accessors for readable assertions.
    fn fp16_parts(bits: u16) -> (u16, u16, u16) {
        ((bits >> 15) & 1, (bits >> 10) & 0x1f, bits & 0x3ff)
    }

    #[test]
    fn hf8_to_fp16_zero_and_signed_zero() {
        // S.0000.000 -> FP16 +/-0.
        assert_eq!(hf8_to_fp16(hf8(0, 0b0000, 0b000)), fp16_bits(0, 0, 0));
        assert_eq!(hf8_to_fp16(hf8(1, 0b0000, 0b000)), fp16_bits(1, 0, 0));
    }

    #[test]
    fn hf8_to_fp16_one_is_exact() {
        // HF8 1.0 = S.0111.000 (exp field 7 = bias). FP16 1.0 = exp 15, mant 0.
        assert_eq!(hf8_to_fp16(hf8(0, 0b0111, 0b000)), fp16_bits(0, 15, 0));
    }

    #[test]
    fn hf8_to_fp16_normal_with_mantissa() {
        // HF8 1.5 = S.0111.100 -> FP16 1.5 = exp 15, mant 0b10_0000_0000.
        assert_eq!(
            hf8_to_fp16(hf8(0, 0b0111, 0b100)),
            fp16_bits(0, 15, 0b10_0000_0000)
        );
        // HF8 max normal 448 = S.1111.110 = 1.110b * 2^8 -> FP16 exp 8+15=23,
        // mant top three bits 0b110_0000_0000.
        assert_eq!(
            hf8_to_fp16(hf8(0, 0b1111, 0b110)),
            fp16_bits(0, 23, 0b11_0000_0000)
        );
    }

    #[test]
    fn hf8_to_fp16_subnormals_renormalise_exactly() {
        // HF8 min subnormal S.0000.001 = 2^-9. FP16 2^-9 is a NORMAL: exp field
        // -9+15 = 6, mant 0. This DISTINGUISHES the exact renormalising decode from a
        // naive "subnormal stays subnormal" model, which would emit an FP16 subnormal
        // (exp 0) and the wrong magnitude.
        assert_eq!(hf8_to_fp16(hf8(0, 0b0000, 0b001)), fp16_bits(0, 6, 0));
        // S.0000.010 = 2 * 2^-9 = 2^-8 -> FP16 exp field 7, mant 0.
        assert_eq!(hf8_to_fp16(hf8(0, 0b0000, 0b010)), fp16_bits(0, 7, 0));
        // S.0000.011 = 3 * 2^-9 = 1.1b * 2^-8 -> FP16 exp field 7, mant 0b10_0000_0000.
        assert_eq!(
            hf8_to_fp16(hf8(0, 0b0000, 0b011)),
            fp16_bits(0, 7, 0b10_0000_0000)
        );
        // S.0000.100 = 4 * 2^-9 = 2^-7 -> FP16 exp field 8, mant 0.
        assert_eq!(hf8_to_fp16(hf8(0, 0b0000, 0b100)), fp16_bits(0, 8, 0));
        // S.0000.111 = 7 * 2^-9 = 1.11b * 2^-7 -> FP16 exp field 8,
        // mant 0b11_0000_0000. Negative sign carried through.
        assert_eq!(
            hf8_to_fp16(hf8(1, 0b0000, 0b111)),
            fp16_bits(1, 8, 0b11_0000_0000)
        );
    }

    #[test]
    fn hf8_to_fp16_nan_maps_to_fp16_nan() {
        // Sole HF8 NaN S.1111.111 -> FP16 NaN: exp all ones, mantissa nonzero.
        let got = hf8_to_fp16(hf8(0, 0b1111, 0b111));
        let (s, e, m) = fp16_parts(got);
        assert_eq!(s, 0);
        assert_eq!(e, 0x1f);
        assert!(m != 0, "FP16 NaN must have a nonzero mantissa");
        // Sign preserved on the negative NaN encoding.
        let got_neg = hf8_to_fp16(hf8(1, 0b1111, 0b111));
        assert_eq!(fp16_parts(got_neg).0, 1);
    }

    #[test]
    fn hf8_to_fp16_then_back_is_identity_for_all_bytes() {
        // Exactness round-trip across all 256 HF8 codes: decoding to FP16 and re-encoding
        // with the family-A HF8 encoder must return the original byte. NaN codes
        // (S.1111.111) map to an FP16 NaN that re-encodes to the same HF8 NaN.
        for raw in 0u8..=u8::MAX {
            let back = fp16_to_hf8(hf8_to_fp16(raw), false);
            assert_eq!(back, raw, "HF8->FP16->HF8 not identity for raw={raw:#04x}");
        }
    }

    // --- FP32 -> FP16 RNE unit tests (spec section 8.3; OQ-6 canonical RNE/DAZ=0/FTZ=0) ---

    #[test]
    fn fp32_to_fp16_exact_and_signed_zero() {
        // +/-0 preserved.
        assert_eq!(fp32_to_fp16_rne(0.0f32), fp16_bits(0, 0, 0));
        assert_eq!(fp32_to_fp16_rne(-0.0f32), fp16_bits(1, 0, 0));
        // 1.0 -> FP16 exp 15, mant 0.
        assert_eq!(fp32_to_fp16_rne(1.0f32), fp16_bits(0, 15, 0));
        // -2.0 -> FP16 exp 16, mant 0.
        assert_eq!(fp32_to_fp16_rne(-2.0f32), fp16_bits(1, 16, 0));
        // 1.5 = 1.1b -> exp 15, mant top bit set.
        assert_eq!(fp32_to_fp16_rne(1.5f32), fp16_bits(0, 15, 0b10_0000_0000));
    }

    #[test]
    fn fp32_to_fp16_ties_to_even() {
        // The FP16 mantissa keeps 10 of FP32's 23 fraction bits; the low 13 bits are
        // discarded. Construct a value whose discarded part is EXACTLY half an lsb so the
        // tie-break is observable, distinguishing RNE from round-half-up.
        let half = 1u32 << 12; // bit 12 set => exactly half of the 13 discarded bits
                               // Base 1.0 with kept-mantissa lsb = 0 (even): exact tie rounds DOWN to 0.
        let down = f32::from_bits(0x3f80_0000 | half); // 1.0 + 0.5 lsb
        assert_eq!(
            fp32_to_fp16_rne(down),
            fp16_bits(0, 15, 0),
            "tie with even keeps down"
        );
        // Base with kept-mantissa lsb = 1 (odd): exact tie rounds UP to even.
        let odd_base = 0x3f80_0000 | (1u32 << 13); // 1.0 + 1 lsb (kept lsb = 1)
        let up = f32::from_bits(odd_base | half);
        assert_eq!(
            fp32_to_fp16_rne(up),
            fp16_bits(0, 15, 0b00_0000_0010),
            "tie with odd rounds up"
        );
        // Just above half rounds up regardless of parity.
        let above = f32::from_bits(0x3f80_0000 | (half + 1));
        assert_eq!(
            fp32_to_fp16_rne(above),
            fp16_bits(0, 15, 0b00_0000_0001),
            "above half rounds up"
        );
    }

    #[test]
    fn fp32_to_fp16_overflow_to_inf() {
        // 65504 is FP16 max normal; 70000 overflows FP16 -> +Inf (FP16 has infinities,
        // unlike the FP8 targets which saturate/NaN). This rules out an FP8-style clamp.
        let got = fp32_to_fp16_rne(70000.0f32);
        assert_eq!((got >> 10) & 0x1f, 0x1f, "exp all ones");
        assert_eq!(got & 0x3ff, 0, "Inf mantissa zero");
        assert_eq!(got >> 15, 0, "positive");
        // Negative overflow -> -Inf.
        assert_eq!(fp32_to_fp16_rne(-70000.0f32), fp16_bits(1, 0x1f, 0));
        // FP32 +Inf -> FP16 +Inf.
        assert_eq!(fp32_to_fp16_rne(f32::INFINITY), fp16_bits(0, 0x1f, 0));
        // 65504.0 (exactly FP16 max normal) stays finite: exp 30, mant all ones.
        assert_eq!(fp32_to_fp16_rne(65504.0f32), fp16_bits(0, 30, 0x3ff));
    }

    #[test]
    fn fp32_to_fp16_subnormal_results() {
        // 2^-14 = FP16 min normal (exp field 1, mant 0).
        assert_eq!(fp32_to_fp16_rne(2.0f32.powi(-14)), fp16_bits(0, 1, 0));
        // 2^-24 = FP16 min positive subnormal (exp 0, mant 1). DAZ=0/FTZ=0 means it is NOT
        // flushed to zero — this distinguishes the honoured-subnormal model from FTZ.
        assert_eq!(fp32_to_fp16_rne(2.0f32.powi(-24)), fp16_bits(0, 0, 1));
        // 2^-15 = subnormal exp 0, mant 0b10_0000_0000 (= 512).
        assert_eq!(
            fp32_to_fp16_rne(2.0f32.powi(-15)),
            fp16_bits(0, 0, 0b10_0000_0000)
        );
        // 2^-25 is exactly half of the min subnormal lsb (2^-24): RNE ties to even (mant 0)
        // -> rounds to +0, NOT up. Rules out round-half-up.
        assert_eq!(fp32_to_fp16_rne(2.0f32.powi(-25)), fp16_bits(0, 0, 0));
        // Just above half (2^-25 * 1.5 = 3 * 2^-26) rounds up to the min subnormal.
        assert_eq!(fp32_to_fp16_rne(3.0 * 2.0f32.powi(-26)), fp16_bits(0, 0, 1));
        // Below half (2^-26) rounds to zero.
        assert_eq!(fp32_to_fp16_rne(2.0f32.powi(-26)), fp16_bits(0, 0, 0));
    }

    #[test]
    fn fp32_to_fp16_nan_propagates() {
        let got = fp32_to_fp16_rne(f32::NAN);
        assert_eq!((got >> 10) & 0x1f, 0x1f, "NaN exp all ones");
        assert!(got & 0x3ff != 0, "NaN mantissa nonzero");
        // Negative NaN keeps its sign and stays a NaN.
        let neg = fp32_to_fp16_rne(-f32::NAN);
        assert_eq!(neg >> 15, 1, "negative NaN sign preserved");
        assert!(neg & 0x3ff != 0, "NaN mantissa nonzero");
    }

    #[test]
    fn fp32_to_fp16_roundtrips_via_f32_as_f16_when_representable() {
        // For values that are exactly representable in FP16, the RNE convert must be exact.
        // Cross-check against decoding the FP16 back to f32 and comparing magnitudes.
        for &v in &[
            0.0f32, 1.0, -1.0, 0.5, 0.25, 3.0, 100.0, 0.125, 2048.0, -448.0,
        ] {
            let h = fp32_to_fp16_rne(v);
            // Decode FP16 normal back to f32 for the comparison.
            let s = if h >> 15 == 1 { -1.0f32 } else { 1.0 };
            let e = ((h >> 10) & 0x1f) as i32;
            let m = (h & 0x3ff) as f32;
            let decoded = if e == 0 {
                s * m * 2.0f32.powi(-24)
            } else {
                s * (1.0 + m / 1024.0) * 2.0f32.powi(e - 15)
            };
            assert_eq!(decoded, v, "exact value {v} not preserved (h={h:#06x})");
        }
    }
}
