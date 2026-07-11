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
//! FP8 non-saturating overflow encoding (per the section-16.1/16.2 pseudocode and the
//! section-9.2.1 behavior table): non-saturating overflow of a finite/Inf magnitude maps to
//! the format's OVERFLOW encoding, which differs by format: **E5M2 (BF8)** uses `S.11111.00`
//! (the section-2.4.1 OCP table lists infinities as N/A for FP8, but this encoding functions
//! as the overflow code and section 16.3 decodes it to FP32 +/-Inf; the NaN set is
//! `S.11111.{01,10,11}`), while **E4M3 (HF8) has no infinity** so its overflow is the sole
//! all-ones NaN `S.1111.111`. Saturating clamps to the format max normal. An FP16 NaN *input*
//! always propagates to a NaN regardless of mode.
//! (`[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_PH2FP8.1-1]`,
//! `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_PH2FP8.1-2]`).
//!
//! Family-C bias ("SR") rounding: per spec section 8.4.5 the per-lane bias term for output
//! lane `i` is `bias = src1.byte[2 * i]` — the low byte of the i-th `u16` element of the bias
//! operand. [`fp16_to_bf8_biased`] / [`fp16_to_hf8_biased`] take that already-extracted `u8`
//! bias and transcribe the section-16.2 SR branches directly: the bias is added at a FIXED
//! position — the lsb of the (normalized) 10-bit FP16 mantissa, unshifted for the E5M2 target
//! and `>> 1` for the E4M3 target — with carry into the exponent, and the biased mantissa is
//! then **truncated** (no further rounding in the SR path). For E4M3 every FP16 subnormal
//! input flushes to signed zero in SR mode too (all are below the E4M3 minimum, spec section
//! 16.2). With `bias == 0` the SR path truncates, so it differs from plain RTNE on above-half
//! inputs; `bias == 0x80` recovers round-to-nearest (ties up) for normal-range results.
//! (`[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_BIAS_PH2FP8.1]`,
//! `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_BIAS_PH2FP8.1-1]`).
//!
//! # Iteration-2 (`AVX10_V2_AUX`) open-question resolutions
//!
//! * **OQ-1 — shared vs FP32-only rounding-mode enum.** RESOLVED: the iteration-2 FP32→FP8
//!   family gets a DEDICATED [`Fp8RoundMode`] `{ Rtne, Rto, Bias }` enum (declared below),
//!   modeled directly on the §9.2.5 pseudocode. The iteration-1 FP16 path keeps its
//!   `bias_mode: bool` contract byte-identical — the new enum is NOT retrofitted onto it.
//! * **OQ-4 — per-family DAZ.** RESOLVED: DAZ is encoded per §16 helper, never as a global
//!   pre-pass. The forward FP32→FP8 rounders ([`fp32_to_fp8_e5m2`] / [`fp32_to_fp8_e4m3`])
//!   assume DAZ=1 (flush input subnormals to signed zero); the exact reverse FP8→FP32 decoders
//!   ([`fp8_e5m2_to_fp32`] / [`fp8_e4m3_to_fp32`]) assume DAZ=0 (renormalise subnormals).

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

/// Round a strictly-positive finite value `mantissa * 2^exp2` to the target FP8 format under
/// round-to-nearest, ties-to-even (RTNE) and return `(exp_field, mant_field, overflowed)`.
/// `overflowed` is set when the rounded magnitude exceeds the format's max normal.
///
/// Used by the family A/B/E plain converts; verified bit-exact against the section-16.2
/// RTNE pseudocode over the full FP16 input domain. The family-C bias ("SR") rounding is a
/// separate direct transcription of the section-16.2 SR branches (see
/// [`fp16_to_bf8_biased`] / [`fp16_to_hf8_biased`]), not a mode of this rounder.
fn round_finite_to_fp8(fmt: &Fp8Format, mantissa: u64, exp2: i32) -> (u32, u32, bool) {
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
        // No fractional bits discarded; exact left shift (rounding cannot apply).
        mantissa << (-shift) as u32
    } else if shift >= 64 {
        // Everything is below the rounding position.
        0
    } else {
        let s = shift as u32;
        let kept = mantissa >> s;
        let rem = mantissa & ((1u64 << s) - 1); // discarded fraction
        let half = 1u64 << (s - 1);
        if rem > half || (rem == half && (kept & 1) == 1) {
            kept + 1
        } else {
            kept
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

/// Convert one FP16 lane (raw bits) to one FP8 byte in the given target format under plain
/// RTNE (families A/B).
///
/// Decodes the FP16 pattern to an exact wide intermediate, rounds once to the target FP8
/// (no double-rounding), and encodes to `u8`. Subnormals, signed zeros, and NaNs are handled
/// bit-exactly with no FTZ/DAZ. On magnitude overflow: when `saturating`, clamp to the
/// format max normal; otherwise emit the format overflow encoding (E5M2 `S.11111.00`,
/// E4M3 NaN).
fn fp16_to_fp8(fmt: &Fp8Format, bits: u16, saturating: bool) -> u8 {
    let (sign, class) = decode_fp16(bits);
    match class {
        Fp16Class::Zero => (sign << 7) as u8, // S.0...0.0...0
        Fp16Class::NaN => {
            // Propagate as an FP8 NaN (always a NaN encoding, even saturating). The full
            // 10-bit FP16 mantissa is passed so the E5M2 payload mapping can read its
            // top bits (spec section 16.2: `m_o = 0x2 | m_i[8]`).
            let fp16_mant = (bits & 0x3ff) as u32;
            fp8_nan_from_input(fmt, sign, fp16_mant)
        }
        Fp16Class::Inf => {
            if saturating {
                fp8_max_normal(fmt, sign)
            } else {
                // E5M2 -> the S.11111.00 overflow encoding; E4M3 (no Inf) -> its NaN slot.
                fp8_overflow(fmt, sign)
            }
        }
        Fp16Class::Finite { mantissa, exp2 } => {
            let (exp_field, mant_field, overflowed) = round_finite_to_fp8(fmt, mantissa, exp2);
            if overflowed {
                if saturating {
                    fp8_max_normal(fmt, sign)
                } else {
                    // Non-saturating finite overflow: E5M2 -> S.11111.00, E4M3 -> NaN.
                    fp8_overflow(fmt, sign)
                }
            } else {
                fp8_assemble(fmt, sign, exp_field, mant_field)
            }
        }
    }
}

/// Convert one FP16 lane (raw bits) to one BF8 (E5M2) byte.
///
/// On magnitude overflow: when `saturating`, clamp to the BF8 max normal `+/-57344`;
/// otherwise emit the BF8 Inf encoding `S.11111.00` (E5M2 has an Inf; see the module
/// header's OQ-4 correction — E4M3 overflows to NaN instead because it has none).
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

/// Convert one FP16 lane (raw bits) to one BF8 (E5M2) byte using bias ("SR") rounding —
/// the spec section 16.2 `fp16_to_fp8_e5m2` SR path, with the bias applied
/// **window-aligned**: added into the 8 bits directly below the destination lsb after the
/// value is aligned to the output format's grid, then TRUNCATED (no rounding in the SR
/// path). Verified bit-exact against `VCVTBIASPH2BF8` under Intel SDE (the previous
/// fixed-position reading — bias at the lsb of the *normalized* FP16 mantissa — diverged
/// from hardware one lsb on subnormal-result and carry-boundary lanes).
///
/// `bias` is the 8-bit bias rounding term for this lane — per spec section 8.4.5 the byte
/// `src1.byte[2 * i]` (the low byte of the i-th `u16` of the bias operand).
///
/// E5M2 shares FP16's exponent geometry (same width, same bias), so the whole finite-case
/// conversion collapses to one integer addition on the FP16 magnitude bit pattern: the top
/// 8 magnitude bits of `bits + bias` ARE the E5M2 encoding. That single add is exactly the
/// window-aligned semantics — the low 8 FP16 mantissa bits sit directly below the E5M2 lsb
/// in every binade (subnormals included, the encoding being a monotone integer map), and a
/// mantissa carry rolls into the exponent field natively. `bias == 0` therefore truncates
/// (it is NOT [`fp16_to_bf8`]'s RTNE). Overflow — the sum reaching or passing the exponent
/// field's all-ones encoding — gives `S.11111.00` when non-saturating and clamps to
/// +/-57344 when saturating.
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_BIAS_PH2FP8.1]`
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_BIAS_PH2FP8.1-1]`
pub(crate) fn fp16_to_bf8_biased(bits: u16, bias: u8, saturating: bool) -> u8 {
    let s_i = ((bits >> 15) & 1) as u32;
    let e_i = ((bits >> 10) & 0x1f) as i32;
    let m_i = (bits & 0x3ff) as u32;

    if e_i == 0x1f {
        // Inf or NaN (same for all rounding modes; the bias is not applied).
        let enc: u32 = if m_i == 0 {
            if saturating {
                (0x1e << 2) | 0x3 // Inf -> clamp to max_normal
            } else {
                0x1f << 2 // Inf -> overflow encoding S.11111.00
            }
        } else {
            (0x1f << 2) | 0x2 | ((m_i >> 8) & 0x1) // NaN -> preserve payload bit
        };
        return ((s_i << 7) | enc) as u8;
    }

    // Finite: one integer add on the magnitude bits, truncate to the top 8 (see above).
    let t = (bits as u32 & 0x7fff) + bias as u32;
    let mut enc = t >> 8;
    if enc >= 0x7c {
        // Sum reached the all-ones exponent field: overflow encoding / saturate.
        enc = if saturating { 0x7b } else { 0x7c };
    }
    ((s_i << 7) | enc) as u8
}

/// Convert one FP16 lane (raw bits) to one HF8 (E4M3) byte using bias ("SR") rounding —
/// the spec section 16.2 `fp16_to_fp8_e4m3` SR path, with the bias applied
/// **window-aligned** exactly as in [`fp16_to_bf8_biased`]: the value is first aligned to
/// the output format's grid (the binade the unbiased value falls in, or the E4M3 subnormal
/// grid), the 8-bit bias is added into the 8 bits directly below the destination lsb, and
/// the sum is TRUNCATED (no rounding in the SR path). Verified bit-exact against
/// `VCVTBIASPH2HF8` under Intel SDE (the previous fixed-position reading — `bias >> 1` at
/// the lsb of the FP16 mantissa — diverged from hardware one lsb on subnormal-result
/// lanes, and wrongly flushed deep-subnormal results that a large bias rounds up to the
/// E4M3 minimum subnormal).
///
/// `bias` is the 8-bit bias rounding term for this lane — per spec section 8.4.5 the byte
/// `src1.byte[2 * i]`.
///
/// Unlike E5M2, E4M3's exponent geometry differs from FP16's, so the alignment is
/// explicit. In a normal output binade FP16 supplies 7 fraction bits below the E4M3 lsb;
/// the bias is added one bit above their bottom (`frac8 = low7 << 1`) so its full 8-bit
/// range spans exactly one output lsb. In the subnormal window the significand is shifted
/// onto the fixed `2^-9` grid (keeping 8 fraction bits, truncating below) before the add —
/// so FP16 subnormal inputs do NOT unconditionally flush: the bias can carry them up to
/// the E4M3 minimum subnormal. A carry across a binade boundary is handled natively by the
/// encoding being a monotone integer map. `bias == 0` therefore truncates (it is NOT
/// [`fp16_to_hf8`]'s RTNE). Overflow — the sum reaching or passing the NaN slot
/// `S.1111.111` — gives NaN when non-saturating, clamp to +/-448 when saturating.
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_BIAS_PH2FP8.1]`
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.CVT_BIAS_PH2FP8.1-1]`
pub(crate) fn fp16_to_hf8_biased(bits: u16, bias: u8, saturating: bool) -> u8 {
    let s_i = ((bits >> 15) & 1) as u32;
    let e_i = ((bits >> 10) & 0x1f) as i32;
    let m_i = (bits & 0x3ff) as u32;

    if e_i == 0x1f {
        // Inf or NaN -> NaN (E4M3 has no Inf); saturating Inf clamps to max_normal.
        let enc: u32 = if saturating && m_i == 0 { 0x7e } else { 0x7f };
        return ((s_i << 7) | enc) as u8;
    }

    // Align the magnitude to the E4M3 grid keeping 8 fraction bits below the output lsb.
    let (mant_full, e_unb): (u64, i32) = if e_i == 0 {
        (m_i as u64, -14) // FP16 subnormal: no implicit bit, fixed 2^-14 binade
    } else {
        ((m_i | 0x400) as u64, e_i - 15)
    };
    // Output exponent field of the unbiased value (E4M3 bias 7); <= 0 is the subnormal
    // window (grid lsb 2^-9), >= 1 selects a normal binade.
    let e_o0 = e_unb + 7;
    let t: u64 = if e_o0 >= 1 {
        // enc0 = the truncated encoding, frac8 = the 7 discarded FP16 bits, top-aligned.
        ((((e_o0 as u64) << 3) | (m_i as u64 >> 7)) << 8) | ((m_i as u64 & 0x7f) << 1)
    } else {
        // value / 2^-9 with 8 fraction bits = mant_full * 2^(e_unb - 10 + 9 + 8); the
        // shift is e_o0 <= 0 places right (truncating), at most 21 for the smallest input.
        mant_full >> (-e_o0)
    };
    let t = t + bias as u64;
    let mut enc = (t >> 8) as u32;
    if enc >= 0x7f {
        // Sum reached or passed the NaN slot S.1111.111: overflow encoding / saturate.
        enc = if saturating { 0x7e } else { 0x7f };
    }
    ((s_i << 7) | enc) as u8
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

/// Convert one FP32 value to BF16 (bfloat16, S1.E8.M7) under round-to-nearest-ties-to-even.
///
/// R6 / group-4 net-new codec: no BF16 codec existed in the crate before the ACE group-4
/// tile instructions (`TOP2BF16PS`, `TCVTROWPS2BF16{H,L}`). BF16 is the **high 16 bits of
/// FP32** — it shares FP32's sign bit, 8-bit exponent and bias 127, keeping only the top 7
/// of FP32's 23 mantissa bits — so the value is obtained by rounding the discarded low 16
/// mantissa bits into the retained 7, with no exponent rebiasing.
///
/// Rounding is IEEE round-to-nearest, ties-to-even (RNE) applied ONCE to the 16 discarded
/// bits. Using the standard branch-free rounder: add `0x7FFF + lsb` (where `lsb` is bit 16
/// of the FP32 pattern, the least-significant *kept* BF16 mantissa bit) then truncate the
/// low 16 bits. The `+lsb` term makes an exact half round to the even neighbour; a carry out
/// of the mantissa correctly propagates into the exponent (and, at the top of the finite
/// range, to +/-Inf), because the whole 32-bit pattern is incremented before truncation.
///
/// Special-case handling per the section-16.1 `fp32_to_bfloat16` pseudocode:
/// * Zero and FP32 denormal inputs flush to signed zero (DAZ=1 implied by the pseudocode's
///   leading `IF e == 0: RETURN (s << 15)`).
/// * +/-Inf -> BF16 +/-Inf (top 16 bits: exp all ones, mantissa 0).
/// * NaN -> a quiet BF16 NaN with the sign preserved. The raw top-16 bits of an FP32 NaN can
///   have a zero 7-bit mantissa (when the NaN's set payload bits live only in the discarded
///   low 16), which would alias to Inf, so the quiet bit (mantissa bit 6, `0x0040`) is
///   forced on to guarantee the result stays a NaN rather than becoming Inf.
pub(crate) fn fp32_to_bf16_rne(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 31) as u16) << 15;
    if bits & 0x7f80_0000 == 0 {
        // Zero or FP32 denormal -> signed zero (spec section 16.1: DAZ=1 implied).
        return sign;
    }
    if f.is_nan() {
        // Quiet NaN, sign preserved: top 16 bits with the quiet mantissa bit forced on so a
        // NaN can never collapse to the Inf encoding.
        return ((bits >> 16) as u16) | 0x0040;
    }
    // Round-to-nearest-even on the 16 discarded bits (branch-free): bias by 0x7FFF plus the
    // least-significant kept bit, then truncate. A tie (low 16 == 0x8000) rounds to even.
    let lsb = (bits >> 16) & 1;
    let rounding_bias = 0x0000_7fff + lsb;
    ((bits + rounding_bias) >> 16) as u16
}

/// Exact lossless decode of one BF16 (S1.E8.M7) value to the FP32 (S1.E8.M23) it represents.
///
/// BF16 is exactly the high 16 bits of FP32 with a zero low half, so the decode is a pure
/// left shift — no rounding, no saturation, no exceptions. Every BF16 encoding (zeros,
/// subnormals, normals, +/-Inf, NaN) maps to the single FP32 value whose top 16 bits are
/// `bits` and whose low 16 bits are zero. This is the exact inverse of `fp32_to_bf16_rne`
/// on any FP32 already representable in BF16 (low 16 mantissa bits zero).
pub(crate) fn bf16_to_fp32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// FP32-family rounding mode for the AVX10_V2_AUX FP32->FP8 converts (spec section 9.1).
///
/// Distinct from the iteration-1 FP16 `bias_mode: bool` path (which stays byte-identical,
/// OQ-1): the FP32 source supports three rounding contracts per the section-9.2.5
/// `vcvtps2f8` pseudocode:
///
/// * `Rtne` — IEEE round-to-nearest-ties-to-even (spec section 2.6.1); used by the
///   `VCVTPS2BF8`/`VCVTPS2HF8` forms.
/// * `Rto` — round-to-odd (spec section 2.6.2); used by `VCVTROPS2HF8` (E4M3 only).
/// * `Bias` — bias rounding (spec section 2.6.3); used by the `VCVTBIASPS2*` forms.
///
/// Consumed by the FP32 front-end [`fp32_to_fp8_e5m2`] / [`fp32_to_fp8_e4m3`]. `Rtne` and
/// `Rto` are wired by the family-A converts (phase 3); `Bias` is wired by the family-B bias
/// converts (phase 4). All three branches of the section-16.1 pseudocode are transcribed in
/// the front-end so every variant is handled bit-exactly.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Fp8RoundMode {
    Rtne,
    Rto,
    Bias,
}

/// Exact decode of one BF8 (FP8 E5M2) byte to the single FP32 (E8M23) value it maps to.
///
/// Per ACE v1 spec section 9.3 (`VCVTBF82PS`) the conversion is **exact** — every BF8
/// encoding maps precisely to one FP32 encoding with no rounding, no saturation and no
/// exceptions (`[avx10-v2-aux-ocp-conversions.CVT_FP8_PS.1]`,
/// `[avx10-v2-aux-ocp-conversions.CVT_FP8_PS.3]`). DAZ=0/FTZ=0, MXCSR not consulted (spec
/// section 9.3.1). E5M2 is sign / 5-bit exponent (bias 15) / 2-bit mantissa (spec section
/// 2.4.1):
///
/// * `S.00000.00` -> FP32 `+/-0`.
/// * `S.00000.mm` (mm != 0) is the subnormal `mm * 2^-16`. Both subnormal magnitudes
///   (`2^-16`, `2^-15`, `3*2^-16`) are normal in FP32 (FP32 normals reach `2^-126`), so
///   they renormalise exactly: the leading set mantissa bit becomes FP32's implicit bit.
/// * `S.11111.00` is BF8 +/-Inf -> FP32 +/-Inf (the section-2.4.1 NaN set is
///   `S.11111.{01,10,11}`, so a zero mantissa in the all-ones exponent is infinity).
/// * `S.11111.mm` (mm != 0) is a BF8 NaN -> a quiet FP32 NaN (sign preserved).
/// * Any other code is a normal `(1 + mm/4) * 2^(e-15)`. FP32 shares the implicit-1 normal
///   form, so the unbiased exponent `e - 15` rebiases to `e - 15 + 127 = e + 112` and the
///   2 mantissa bits sit at the top of FP32's 23-bit field (`mm << 21`). Exact.
pub(crate) fn fp8_e5m2_to_fp32(byte: u8) -> f32 {
    let sign = (byte >> 7) as u32 & 1;
    let exp = ((byte >> 2) & 0x1f) as u32; // 5-bit biased exponent (bias 15)
    let mant = (byte & 0x03) as u32; // 2-bit mantissa
    let fp32_sign = sign << 31;

    if exp == 0x1f {
        if mant == 0 {
            // S.11111.00 = BF8 +/-Inf -> FP32 +/-Inf.
            return f32::from_bits(fp32_sign | (0xff << 23));
        }
        // BF8 NaN S.11111.{01,10,11} -> quiet FP32 NaN, payload preserved: the section-16.3
        // pseudocode is `m_o = (m_i | 0x2) << 21` — the qBit (FP32 mantissa bit 22) is
        // forced on and the low NaN payload bit lands in FP32 bit 21.
        return f32::from_bits(fp32_sign | (0xff << 23) | ((mant | 0x2) << 21));
    }

    if exp == 0 {
        if mant == 0 {
            // +/-0.
            return f32::from_bits(fp32_sign);
        }
        // Subnormal mm * 2^-16, mm in 1..=3. Renormalise into an FP32 normal: with `k` the
        // index of the leading set bit of mm (0..=1), the value is 1.<rest> * 2^(k-16), so
        // the unbiased exponent is (k - 16) and the bits below the leading 1 shift up into
        // the top of FP32's 23-bit field.
        let k = 31 - mant.leading_zeros() as i32; // top set bit of mm (0..=1)
        let unbiased = k - 16; // value = 1.f * 2^unbiased
        let fp32_exp = (unbiased + 127) as u32; // FP32 bias 127; always normal
        let frac = mant & ((1 << k) - 1); // bits below the leading 1 (k bits)
        let fp32_mant = frac << (23 - k);
        return f32::from_bits(fp32_sign | (fp32_exp << 23) | fp32_mant);
    }

    // Normal: value = (1 + mm/4) * 2^(exp - 15). FP32 shares the implicit-1 form, so the
    // unbiased exponent (exp - 15) rebiases to (exp - 15 + 127) = exp + 112, and the 2
    // mantissa bits occupy the top of FP32's 23-bit mantissa field.
    let fp32_exp = exp + 112;
    let fp32_mant = mant << 21;
    f32::from_bits(fp32_sign | (fp32_exp << 23) | fp32_mant)
}

/// Exact decode of one HF8 (FP8 E4M3) byte to the single FP32 (E8M23) value it maps to.
///
/// Per ACE v1 spec section 9.3 (`VCVTHF82PS`) the conversion is **exact** — every HF8
/// encoding maps precisely to one FP32 encoding with no rounding, no saturation and no
/// exceptions (`[avx10-v2-aux-ocp-conversions.CVT_FP8_PS.2]`,
/// `[avx10-v2-aux-ocp-conversions.CVT_FP8_PS.3]`). DAZ=0/FTZ=0 (spec section 9.3.1). E4M3
/// is sign / 4-bit exponent (bias 7) / 3-bit mantissa (spec section 2.4.1):
///
/// * `S.0000.000` -> FP32 `+/-0`.
/// * `S.0000.mmm` (mmm != 0) is the subnormal `mmm * 2^-9`; every such value is normal in
///   FP32, so it renormalises exactly.
/// * `S.1111.111` is the sole HF8 NaN (E4M3 has no infinity) -> a quiet FP32 NaN.
/// * Any other code is a normal `(1 + mmm/8) * 2^(e-7)` -> FP32 exponent `e - 7 + 127 =
///   e + 120`, mantissa `mmm << 20`. Exact.
pub(crate) fn fp8_e4m3_to_fp32(byte: u8) -> f32 {
    let sign = (byte >> 7) as u32 & 1;
    let exp = ((byte >> 3) & 0x0f) as u32; // 4-bit biased exponent (bias 7)
    let mant = (byte & 0x07) as u32; // 3-bit mantissa
    let fp32_sign = sign << 31;

    if exp == 0x0f && mant == 0x07 {
        // Sole HF8 NaN S.1111.111 -> quiet FP32 NaN, sign kept. The section-16.3 pseudocode
        // is `m_o = (m_i | 0x4) << 20` — with m_i == 0x7 that sets FP32 mantissa bits
        // 22:20 (0x700000). (E4M3 has no infinity, so this is the only non-finite encoding.)
        return f32::from_bits(fp32_sign | (0xff << 23) | ((mant | 0x4) << 20));
    }

    if exp == 0 {
        if mant == 0 {
            // +/-0.
            return f32::from_bits(fp32_sign);
        }
        // Subnormal mmm * 2^-9, mmm in 1..=7. Renormalise into an FP32 normal.
        let k = 31 - mant.leading_zeros() as i32; // top set bit of mmm (0..=2)
        let unbiased = k - 9; // value = 1.f * 2^unbiased
        let fp32_exp = (unbiased + 127) as u32;
        let frac = mant & ((1 << k) - 1);
        let fp32_mant = frac << (23 - k);
        return f32::from_bits(fp32_sign | (fp32_exp << 23) | fp32_mant);
    }

    // Normal: value = (1 + mmm/8) * 2^(exp - 7). FP32 unbiased exponent (exp - 7) rebiases
    // to (exp - 7 + 127) = exp + 120, and the 3 mantissa bits occupy the top of FP32's
    // 23-bit mantissa field.
    let fp32_exp = exp + 120;
    let fp32_mant = mant << 20;
    f32::from_bits(fp32_sign | (fp32_exp << 23) | fp32_mant)
}

/// `mask(n)` from the section-16.1 pseudocode: the low-`n`-bit mask, with `mask(0) == 0`.
/// Shared with the FP4/FP6 rounders in `crate::fp4` / `crate::fp6`.
#[inline]
pub(crate) fn mask(n: i32) -> u32 {
    if n <= 0 {
        0
    } else if n >= 32 {
        u32::MAX
    } else {
        (1u32 << n) - 1
    }
}

/// FP32 (S1.E8.M23) -> FP8 E5M2 (BF8), transcribing the ACE v1 spec section 16.1
/// `fp32_to_fp8_e5m2(i, saturating, rounding, bias)` helper bit-for-bit. The FP32 source is
/// decoded to its exact `(sign, e_i, m_i)` fields and rounded **once** to the E5M2 target
/// per `mode`. Per spec section 9.2.1 MXCSR is neither consulted nor updated, DAZ is assumed
/// 1 / FTZ assumed 0, and no FP exceptions are raised.
///
/// Overflow / NaN handling follows the section-9.2.1 table for E5M2:
/// * NaN input -> the BF8 NaN-coded value `S.11111.1x` (both modes).
/// * post-rounding magnitude `> max_E5M2` -> the BF8 Inf/overflow-coded `S.11111.00`
///   (non-saturating) or `±max_E5M2 = ±57344` `S.11110.11` (saturating).
/// * signed zero -> same-signed E5M2 zero.
///
/// E5M2 has **no RTO form** (spec section 9.1 / 9.2.1): there is no `cvtrops_bf8`, so the
/// family-A wiring never passes `Fp8RoundMode::Rto` here; `Bias` is supplied only by the
/// family-B wiring (phase 4) via [`fp32_to_fp8_e5m2_biased`], so the plain converts pass
/// `bias == 0`.
/// `[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.1]` `[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.5]`
/// `[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.8]` `[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.9]`
pub(crate) fn fp32_to_fp8_e5m2(f: f32, mode: Fp8RoundMode, saturating: bool) -> u8 {
    fp32_to_fp8_e5m2_biased(f, mode, saturating, 0)
}

/// E5M2 Inf/NaN handling (`e_i == 0xFF`). Extracted from [`fp32_to_fp8_e5m2_biased`].
fn e5m2_special(saturating: bool, m_i: u32) -> (i32, u32) {
    if saturating {
        if m_i == 0 {
            (0x1E, 0x3) // Inf -> clamp to max_normal
        } else {
            (0x1F, 0x2 | ((m_i >> 21) & 0x1)) // NaN -> coded NaN (kept even when sat)
        }
    } else if m_i != 0 {
        (0x1F, 0x2 | ((m_i >> 21) & 0x1)) // NaN
    } else {
        (0x1F, 0x0) // Inf -> Inf/overflow-coded
    }
}

/// E5M2 BIAS branch (spec section 16.1). With `bias == 0` this is add-then-truncate.
/// Extracted from [`fp32_to_fp8_e5m2_biased`].
fn e5m2_bias(e_i: i32, m_i: u32, bias: u32, saturating: bool) -> (i32, u32) {
    let mut e_b = e_i;
    let mut m_b = m_i + (bias & 0x1FFFFF);
    if m_b & 0xFF800000 != 0 {
        e_b += 1;
    }
    m_b &= 0x7FFFFF;
    let newexp = e_b - 127 + 15;
    if newexp >= 31 {
        if saturating {
            (0x1E, 0x3)
        } else {
            (0x1F, 0x0)
        }
    } else if newexp <= 0 {
        let mut m_o = 0u32;
        if (22 - newexp) <= 24 {
            let mant = m_b | 0x800000;
            let shift = 22 - newexp;
            m_o = mant >> shift;
        }
        (0, m_o)
    } else {
        (newexp, m_b >> 21)
    }
}

/// E5M2 RTNE underflow (`newexp <= 0`) -> subnormal or zero. Extracted from
/// [`fp32_to_fp8_e5m2_biased`].
fn e5m2_rtne_subnormal(m_i: u32, newexp: i32) -> (i32, u32) {
    let mut e_o = 0i32;
    let mut m_o = 0u32;
    if (22 - newexp) <= 24 {
        let mant = m_i | 0x800000;
        let shift = 22 - newexp;
        m_o = mant >> shift;
        let low = mant & mask(shift);
        let half = 1u32 << (shift - 1);
        if low > half || (low == half && (m_o & 0x1) == 1) {
            m_o += 1;
            if (m_o & 0x3) == 0 {
                e_o += 1;
            }
        }
    }
    (e_o, m_o)
}

/// E5M2 RTNE normal (`0 < newexp < 31`). Extracted from [`fp32_to_fp8_e5m2_biased`].
fn e5m2_rtne_normal(newexp: i32, m_i: u32, saturating: bool) -> (i32, u32) {
    let mut e_o = newexp;
    let mut m_o = m_i >> 21;
    if m_i & 0x100000 != 0
        && ((m_i & 0x1FFFFF) > 0x100000 || (m_o & 0x1) == 1)
        && !(saturating && e_o == 0x1E && m_o == 0x3)
    {
        m_o += 1;
        if (m_o & 0x3) == 0 {
            e_o += 1;
        }
    }
    (e_o, m_o)
}

/// E5M2 RTNE branch (RTO is not an E5M2 form; folded into RTNE). Extracted from
/// [`fp32_to_fp8_e5m2_biased`].
fn e5m2_rtne(e_i: i32, m_i: u32, saturating: bool) -> (i32, u32) {
    let newexp = e_i - 127 + 15;
    if newexp >= 31 {
        if saturating {
            (0x1E, 0x3)
        } else {
            (0x1F, 0x0)
        }
    } else if newexp <= 0 {
        e5m2_rtne_subnormal(m_i, newexp)
    } else {
        e5m2_rtne_normal(newexp, m_i, saturating)
    }
}

/// E5M2 front-end with an explicit 21-bit `bias` term for the `Fp8RoundMode::Bias` branch
/// (spec section 16.1). `Rtne`/`Rto` ignore `bias`; RTO is not an E5M2 form and is folded
/// into RTNE.
pub(crate) fn fp32_to_fp8_e5m2_biased(
    f: f32,
    mode: Fp8RoundMode,
    saturating: bool,
    bias: u32,
) -> u8 {
    let i = f.to_bits();
    let s_i = (i >> 31) & 0x1;
    let e_i = ((i >> 23) & 0xFF) as i32;
    let m_i = i & 0x7FFFFF;

    let (e_o, m_o): (i32, u32) = if e_i == 0xFF {
        e5m2_special(saturating, m_i)
    } else if e_i == 0x00 {
        // Zero or denorm (DAZ=1) -> same-signed zero.
        (0, 0)
    } else if mode == Fp8RoundMode::Bias {
        e5m2_bias(e_i, m_i, bias, saturating)
    } else {
        e5m2_rtne(e_i, m_i, saturating)
    };

    ((s_i & 0x1) << 7 | ((e_o as u32) & 0x1F) << 2 | (m_o & 0x3)) as u8
}

/// FP32 -> FP8 E4M3 (HF8), transcribing the ACE v1 spec section 16.1
/// `fp32_to_fp8_e4m3(i, saturating, rounding, bias)` helper bit-for-bit. Decode the FP32
/// source to exact `(sign, e_i, m_i)` and round **once** to the E4M3 target per `mode`.
/// MXCSR not consulted/updated; DAZ=1, FTZ=0; no FP exceptions (spec section 9.2.1).
///
/// Overflow / NaN handling per the section-9.2.1 table for E4M3:
/// * NaN input -> the sole HF8 NaN `S.1111.111` (both modes).
/// * post-rounding magnitude `> max_E4M3` -> NaN `S.1111.111` (non-saturating) or
///   `±max_E4M3 = ±448` `S.1111.110` (saturating).
/// * signed zero -> same-signed E4M3 zero.
///
/// `Rto` (spec section 2.6.2 round-to-odd) is the E4M3-only mode used by `VCVTROPS2HF8`: on
/// an inexact normal/subnormal it ORs a sticky bit into the kept mantissa lsb, so the result
/// mantissa is **odd** whenever the FP32 value is inexact in E4M3 — round-to-odd never selects
/// an even target mantissa for an inexact value (spec section 16.1 E4M3 RTO branch).
/// `[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.2]` `[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.3]`
/// `[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.4]` `[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.6]`
/// `[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.8]` `[avx10-v2-aux-ocp-conversions.CVT_PS_FP8.9]`
pub(crate) fn fp32_to_fp8_e4m3(f: f32, mode: Fp8RoundMode, saturating: bool) -> u8 {
    fp32_to_fp8_e4m3_biased(f, mode, saturating, 0)
}

/// E4M3 Inf/NaN handling (`e_i == 0xFF`) -> NaN-coded; saturating Inf clamps to
/// max_normal. Extracted from [`fp32_to_fp8_e4m3_biased`].
fn e4m3_special(saturating: bool, m_i: u32) -> (i32, u32) {
    if saturating && m_i == 0 {
        (0xF, 0x6) // Inf -> clamp to max_normal
    } else {
        (0xF, 0x7)
    }
}

/// E4M3 RTO (round-to-odd) branch, E4M3 only. Extracted from
/// [`fp32_to_fp8_e4m3_biased`].
fn e4m3_rto(e_i: i32, m_i: u32, saturating: bool) -> (i32, u32) {
    let newexp = e_i - 127 + 7;
    if newexp >= 16 {
        (0xF, if saturating { 0x6 } else { 0x7 })
    } else if newexp <= 0 {
        if (21 - newexp) <= 24 {
            let mant = m_i | 0x800000;
            let shift = 21 - newexp;
            let mut m_o = mant >> shift;
            let sticky = if mant & mask(shift) != 0 { 1 } else { 0 };
            m_o |= sticky;
            (0, m_o)
        } else {
            // magnitude < 2^-10: J-bit below the subnormal lsb -> odd smallest subnormal.
            (0, 1)
        }
    } else {
        let e_o = newexp;
        let mut m_o = m_i >> 20;
        let sticky = if m_i & 0xFFFFF != 0 { 1 } else { 0 };
        m_o |= sticky;
        if saturating && e_o == 0xF && m_o == 0x7 {
            (e_o, 0x6) // clamp NaN -> max_normal
        } else {
            (e_o, m_o)
        }
    }
}

/// E4M3 BIAS branch (spec section 16.1). With `bias == 0` this is add-then-truncate.
/// Extracted from [`fp32_to_fp8_e4m3_biased`].
fn e4m3_bias(e_i: i32, m_i: u32, bias: u32, saturating: bool) -> (i32, u32) {
    let mut e_b = e_i;
    let mut m_b = m_i + (bias & 0xFFFFF);
    if m_b & 0xFF800000 != 0 {
        e_b += 1;
    }
    m_b &= 0x7FFFFF;
    let newexp = e_b - 127 + 7;
    if newexp >= 16 {
        (0xF, if saturating { 0x6 } else { 0x7 })
    } else if newexp <= 0 {
        // Underflow: hard flush to signed zero. Unlike the E5M2 Bias branch, the
        // section-16.1 E4M3 Bias pseudocode has NO subnormal-truncate block — the
        // spec is deliberately asymmetric here.
        (0, 0)
    } else {
        // Normal: truncate the biased mantissa. The section-16.1 E4M3 Bias branch has
        // no saturating NaN-slot clamp (that clamp exists only in the RTNE/RTO
        // branches); a truncation into `e_o == 0xF, m_o == 0x7` yields the NaN
        // encoding even when saturating.
        (newexp, m_b >> 20)
    }
}

/// E4M3 RTNE underflow (`newexp <= 0`) -> subnormal or zero. Extracted from
/// [`fp32_to_fp8_e4m3_biased`].
fn e4m3_rtne_subnormal(m_i: u32, newexp: i32) -> (i32, u32) {
    let mut e_o = 0i32;
    let mut m_o = 0u32;
    if (21 - newexp) <= 24 {
        let mant = m_i | 0x800000;
        let shift = 21 - newexp;
        m_o = mant >> shift;
        let low = mant & mask(shift);
        let half = 1u32 << (shift - 1);
        if low > half || (low == half && (m_o & 0x1) == 1) {
            m_o += 1;
            if (m_o & 0x7) == 0 {
                e_o += 1;
            }
        }
    }
    (e_o, m_o)
}

/// E4M3 RTNE normal (`0 < newexp < 16`). Extracted from [`fp32_to_fp8_e4m3_biased`].
fn e4m3_rtne_normal(newexp: i32, m_i: u32, saturating: bool) -> (i32, u32) {
    let mut e_o = newexp;
    let mut m_o = m_i >> 20;
    if saturating && e_o == 0xF && m_o == 0x7 {
        m_o = 0x6;
    }
    if m_i & 0x80000 != 0 && ((m_i & 0xFFFFF) > 0x80000 || (m_o & 0x1) == 1) {
        let clamp_sat = saturating && e_o == 0xF && m_o == 0x6;
        let clamp_nan = !saturating && e_o == 0xF && m_o == 0x7;
        if !(clamp_sat || clamp_nan) {
            m_o += 1;
            if (m_o & 0x7) == 0 {
                e_o += 1;
            }
        }
    }
    (e_o, m_o)
}

/// E4M3 RTNE branch. Extracted from [`fp32_to_fp8_e4m3_biased`].
fn e4m3_rtne(e_i: i32, m_i: u32, saturating: bool) -> (i32, u32) {
    let newexp = e_i - 127 + 7;
    if newexp >= 16 {
        (0xF, if saturating { 0x6 } else { 0x7 })
    } else if newexp <= 0 {
        e4m3_rtne_subnormal(m_i, newexp)
    } else {
        e4m3_rtne_normal(newexp, m_i, saturating)
    }
}

/// E4M3 front-end with an explicit 20-bit `bias` term for the `Fp8RoundMode::Bias` branch
/// (spec section 16.1). `Rtne`/`Rto` ignore `bias`.
pub(crate) fn fp32_to_fp8_e4m3_biased(
    f: f32,
    mode: Fp8RoundMode,
    saturating: bool,
    bias: u32,
) -> u8 {
    let i = f.to_bits();
    let s_i = (i >> 31) & 0x1;
    let e_i = ((i >> 23) & 0xFF) as i32;
    let m_i = i & 0x7FFFFF;

    let (e_o, m_o): (i32, u32) = if e_i == 0xFF {
        e4m3_special(saturating, m_i)
    } else if e_i == 0x00 {
        // Zero or denorm (DAZ=1) -> same-signed zero.
        (0, 0)
    } else if mode == Fp8RoundMode::Rto {
        e4m3_rto(e_i, m_i, saturating)
    } else if mode == Fp8RoundMode::Bias {
        e4m3_bias(e_i, m_i, bias, saturating)
    } else {
        e4m3_rtne(e_i, m_i, saturating)
    };

    ((s_i & 0x1) << 7 | ((e_o as u32) & 0xF) << 3 | (m_o & 0x7)) as u8
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
        // Per the section-16.2 SR pseudocode, family-C bias rounding is NOT
        // "add-bias-then-RTNE". It adds the 8-bit bias into the 8 bits below the
        // destination lsb (for a BF8 normal result that is the lsb of the 10-bit FP16
        // mantissa) and then TRUNCATES toward zero. Two consequences this test pins:
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
        // Bias 0xff added at the mantissa lsb (spec section 16.2, E5M2 target: no shift):
        // 1 + 0xff = 0x100, which truncates (>> 8) to mant 0b01 — one lsb up.
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
    fn bias_sr_subnormal_window_alignment_matches_hardware() {
        // Section-16.2 SR pins in the subnormal regime — these distinguish the
        // window-aligned bias add (what `VCVTBIASPH2[B|H]F8` actually do, verified under
        // Intel SDE) from a fixed-position add at the normalized-FP16-mantissa lsb (an
        // earlier misreading of the pseudocode, which underweights the bias for subnormal
        // results). The two divergences that CI caught, pinned as regressions here:
        //
        //  * dispatch lane a=0x2088, bias=0xB8: aligned T = 0x488, +0xB8 = 0x540 -> 5.
        //    The fixed-position model loses the carry and returns 4.
        assert_eq!(fp16_to_hf8_biased(0x2088, 0xb8, false), 0x05);
        //  * dispatch lane a=0x1044, bias=0xDC: aligned T = 0x444 >> 4 = 0x44, +0xDC =
        //    0x120 -> 1. The fixed-position model truncates the bias away entirely -> 0.
        assert_eq!(fp16_to_hf8_biased(0x1044, 0xdc, false), 0x01);
        //
        // E5M2, min FP16 subnormal 0x0001 (= 2^-24), bias 0xFF: E5M2 shares FP16's
        // exponent geometry, so the add is the integer add on the bit pattern:
        // (0x0001 + 0xFF) >> 8 = 0x01 — the bias rounds the minimum subnormal UP.
        assert_eq!(fp16_to_bf8_biased(0x0001, 0xff, false), 0x01);
        // E5M2, subnormal 0x0201, bias 0xFF: (0x0201 + 0xFF) >> 8 = 0x03.
        assert_eq!(fp16_to_bf8_biased(0x0201, 0xff, false), 0x03);
        // E4M3, FP16 subnormal inputs do NOT unconditionally flush: on the 2^-9 grid
        // 0x0080 aligns to T = 0x80 >> 7 = 1, +0xFF = 0x100 -> the minimum subnormal, sign
        // preserved. (With bias 0 they truncate to zero as before.)
        assert_eq!(fp16_to_hf8_biased(0x0080, 0xff, false), 0x01);
        assert_eq!(fp16_to_hf8_biased(0x8080, 0xff, false), 0x81);
        assert_eq!(fp16_to_hf8_biased(0x0080, 0x00, false), 0x00);
        assert_eq!(fp16_to_hf8_biased(0x8080, 0x00, false), 0x80);
        // E4M3, normal source with subnormal result, bias 0x80: 0x1400 = 2^-10 aligns to
        // T = 0x400 >> 3 = 0x80, +0x80 = 0x100 -> 0x01 (exactly half an output lsb, so the
        // half-bias carries it up).
        assert_eq!(fp16_to_hf8_biased(0x1400, 0x80, false), 0x01);
        // E4M3, 0x1CFF with bias 0xFF: T = 0x4FF >> 1 = 0x27F, +0xFF = 0x37E -> 0x03.
        assert_eq!(fp16_to_hf8_biased(0x1cff, 0xff, false), 0x03);
    }

    #[test]
    fn bias_overflow_matches_plain_overflow_handling() {
        // Bias on an already-overflowing magnitude still produces the format overflow
        // encoding (Inf `S.11111.00` for BF8, NaN for HF8, non-saturating) or clamps to max
        // normal (saturating) — bias does not change the overflow handling (spec section
        // 8.4.5 reuses the family-A saturation path).
        let big = fp16_bits(0, 30, 0x3ff); // 65504, overflows both BF8 and HF8
        assert_eq!(
            fp16_to_bf8_biased(big, 0x40, false),
            bf8(0, 0b11111, 0b00),
            "bf8 nsat overflow is the Inf encoding S.11111.00"
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

    // --- FP8 -> FP32 exact decode unit tests (spec section 9.3.5; CVT_FP8_PS.1-3) ---

    // FP32 field accessors for readable assertions.
    fn fp32_parts(f: f32) -> (u32, u32, u32) {
        let b = f.to_bits();
        ((b >> 31) & 1, (b >> 23) & 0xff, b & 0x007f_ffff)
    }

    #[test]
    fn fp8_e5m2_to_fp32_zero_signed_zero_and_normals() {
        // S.00000.00 -> +/-0 (sign-bit preserving, not numeric 0.0 == -0.0).
        assert_eq!(fp8_e5m2_to_fp32(bf8(0, 0b00000, 0b00)).to_bits(), 0);
        assert_eq!(fp8_e5m2_to_fp32(bf8(1, 0b00000, 0b00)).to_bits(), 1 << 31);
        // BF8 1.0 = S.01111.00 (exp field 15 = bias) -> FP32 1.0.
        assert_eq!(fp8_e5m2_to_fp32(bf8(0, 0b01111, 0b00)), 1.0f32);
        // BF8 1.5 = S.01111.10 (1.10b) -> FP32 1.5. mm=0b10 lands at FP32 mant bit 22.
        assert_eq!(fp8_e5m2_to_fp32(bf8(0, 0b01111, 0b10)), 1.5f32);
        // BF8 -2.0 = S.10000.00 -> -2.0.
        assert_eq!(fp8_e5m2_to_fp32(bf8(1, 0b10000, 0b00)), -2.0f32);
        // BF8 max normal 57344 = S.11110.11 = 1.11b * 2^15.
        assert_eq!(fp8_e5m2_to_fp32(bf8(0, 0b11110, 0b11)), 57344.0f32);
    }

    #[test]
    fn fp8_e5m2_to_fp32_subnormals_renormalise_exactly() {
        // BF8 min subnormal S.00000.01 = 2^-16 -> FP32 normal exp field (-16+127)=111, mant 0.
        // This DISTINGUISHES the exact renormalising decode (FP32 normal) from a naive
        // "subnormal stays subnormal" model, which would produce exp 0 and the wrong value.
        let v = fp8_e5m2_to_fp32(bf8(0, 0b00000, 0b01));
        assert_eq!(v, 2.0f32.powi(-16));
        assert_eq!(fp32_parts(v), (0, 111, 0));
        // S.00000.10 = 2*2^-16 = 2^-15 -> exp field 112, mant 0.
        assert_eq!(fp8_e5m2_to_fp32(bf8(0, 0b00000, 0b10)), 2.0f32.powi(-15));
        // S.00000.11 = 3*2^-16 = 1.1b * 2^-15 -> exp field 112, mant top bit set (bit 22).
        let v3 = fp8_e5m2_to_fp32(bf8(0, 0b00000, 0b11));
        assert_eq!(v3, 3.0 * 2.0f32.powi(-16));
        assert_eq!(fp32_parts(v3), (0, 112, 1 << 22));
        // Negative subnormal keeps sign.
        assert_eq!(fp8_e5m2_to_fp32(bf8(1, 0b00000, 0b01)), -(2.0f32.powi(-16)));
    }

    #[test]
    fn fp8_e5m2_to_fp32_inf_and_nan() {
        // S.11111.00 = BF8 +Inf -> FP32 +Inf (zero mantissa in all-ones exp is Inf, NOT NaN
        // — this rules out a model that treats the whole max-exponent binade as NaN).
        assert_eq!(fp8_e5m2_to_fp32(bf8(0, 0b11111, 0b00)), f32::INFINITY);
        assert_eq!(fp8_e5m2_to_fp32(bf8(1, 0b11111, 0b00)), f32::NEG_INFINITY);
        // S.11111.01 / .10 / .11 are BF8 NaN -> FP32 NaN with the section-16.3 payload
        // mapping `m_o = (m_i | 0x2) << 21`: the qBit (FP32 bit 22) is forced on and the
        // low NaN payload bit lands in FP32 bit 21.
        for m in [0b01u8, 0b10, 0b11] {
            let v = fp8_e5m2_to_fp32(bf8(0, 0b11111, m));
            assert!(v.is_nan(), "mm={m:#04b} should be NaN");
            let (s, e, frac) = fp32_parts(v);
            assert_eq!((s, e), (0, 0xff));
            assert_eq!(
                frac,
                ((m as u32) | 0x2) << 21,
                "payload-preserving NaN mantissa per section 16.3"
            );
        }
        assert!(fp8_e5m2_to_fp32(bf8(1, 0b11111, 0b10)).is_nan());
        assert_eq!(fp32_parts(fp8_e5m2_to_fp32(bf8(1, 0b11111, 0b10))).0, 1);
    }

    #[test]
    fn fp8_e4m3_to_fp32_zero_normals_and_max() {
        assert_eq!(fp8_e4m3_to_fp32(hf8(0, 0b0000, 0b000)).to_bits(), 0);
        assert_eq!(fp8_e4m3_to_fp32(hf8(1, 0b0000, 0b000)).to_bits(), 1 << 31);
        // HF8 1.0 = S.0111.000 (exp field 7 = bias) -> 1.0.
        assert_eq!(fp8_e4m3_to_fp32(hf8(0, 0b0111, 0b000)), 1.0f32);
        // HF8 1.5 = S.0111.100 (1.100b) -> 1.5.
        assert_eq!(fp8_e4m3_to_fp32(hf8(0, 0b0111, 0b100)), 1.5f32);
        // HF8 max normal 448 = S.1111.110 = 1.110b * 2^8. This sits in the max-exponent
        // binade as a genuine normal (NOT NaN), distinguishing E4M3 from an E5M2-style
        // "max exponent == NaN" model.
        assert_eq!(fp8_e4m3_to_fp32(hf8(0, 0b1111, 0b110)), 448.0f32);
        // HF8 384 = S.1111.100 = 1.100b * 2^8, also a max-exponent normal.
        assert_eq!(fp8_e4m3_to_fp32(hf8(0, 0b1111, 0b100)), 384.0f32);
    }

    #[test]
    fn fp8_e4m3_to_fp32_subnormals_and_nan() {
        // HF8 min subnormal S.0000.001 = 2^-9 -> FP32 normal exp field (-9+127)=118, mant 0.
        let v = fp8_e4m3_to_fp32(hf8(0, 0b0000, 0b001));
        assert_eq!(v, 2.0f32.powi(-9));
        assert_eq!(fp32_parts(v), (0, 118, 0));
        // S.0000.111 = 7*2^-9 = 1.11b * 2^-7 -> exp field 120, mant top two bits set.
        let v7 = fp8_e4m3_to_fp32(hf8(1, 0b0000, 0b111));
        assert_eq!(v7, -(7.0 * 2.0f32.powi(-9)));
        assert_eq!(fp32_parts(v7), (1, 120, 0b11 << 21));
        // Sole HF8 NaN S.1111.111 -> FP32 NaN, sign preserved. Every OTHER all-ones-exp
        // code is a finite normal, so this is the only NaN — rules out an E5M2-style
        // "any all-ones exponent is non-finite" decode.
        let nan = fp8_e4m3_to_fp32(hf8(0, 0b1111, 0b111));
        assert!(nan.is_nan());
        assert_eq!(fp32_parts(nan).1, 0xff);
        // Section-16.3 payload mapping `m_o = (m_i | 0x4) << 20` with m_i = 0x7: FP32
        // mantissa bits 22:20 all set.
        assert_eq!(fp32_parts(nan).2, 0x70_0000);
        assert!(fp8_e4m3_to_fp32(hf8(1, 0b1111, 0b111)).is_nan());
        assert_eq!(fp32_parts(fp8_e4m3_to_fp32(hf8(1, 0b1111, 0b111))).0, 1);
    }

    /// True when a BF16 (S1.E8.M7) bit pattern is a NaN: exponent all ones, mantissa nonzero.
    fn bf16_is_nan(b: u16) -> bool {
        (b >> 7) & 0xff == 0xff && (b & 0x7f) != 0
    }

    /// Discriminating known-value pins for the FP32->BF16 round-to-nearest-ties-to-even
    /// rounder. Each case is chosen so its expected output DIFFERS from the two leading wrong
    /// models — plain truncation (drop the low 16 bits) and round-half-away-from-zero — so a
    /// passing test uniquely identifies RNE (differential tiebreaker unavailable here;
    /// grounded against the IEEE-754 roundTiesToEven rule for the 16 discarded bits).
    #[test]
    fn fp32_to_bf16_rne_known_values() {
        // Exact (low 16 bits already zero): no rounding, top 16 bits verbatim.
        assert_eq!(fp32_to_bf16_rne(1.0f32), 0x3f80, "1.0 -> 0x3F80 exact");
        assert_eq!(fp32_to_bf16_rne(-1.0f32), 0xbf80, "-1.0 -> 0xBF80 exact");
        assert_eq!(fp32_to_bf16_rne(0.0f32), 0x0000, "+0 -> +0");
        assert_eq!(fp32_to_bf16_rne(-0.0f32), 0x8000, "-0 -> -0 (sign kept)");

        // Exact half (low 16 == 0x8000), kept-lsb EVEN (0x3F80, bit16 = 0): ties-to-even
        // rounds DOWN to 0x3F80. Truncation also gives 0x3F80, but round-half-away gives
        // 0x3F81 — so this case alone rules out round-half-away.
        assert_eq!(
            fp32_to_bf16_rne(f32::from_bits(0x3f80_8000)),
            0x3f80,
            "half, even kept-lsb -> round to even (down)"
        );
        // Exact half, kept-lsb ODD (0x3F81, bit16 = 1): ties-to-even rounds UP to 0x3F82.
        // Truncation would give 0x3F81 — so this case rules out truncation; together the two
        // half cases uniquely pin RNE.
        assert_eq!(
            fp32_to_bf16_rne(f32::from_bits(0x3f81_8000)),
            0x3f82,
            "half, odd kept-lsb -> round to even (up)"
        );
        // Just above half -> always rounds up (all models agree, guards the carry path).
        assert_eq!(
            fp32_to_bf16_rne(f32::from_bits(0x3f80_8001)),
            0x3f81,
            "just above half -> up"
        );
        // Just below half -> always rounds down.
        assert_eq!(
            fp32_to_bf16_rne(f32::from_bits(0x3f80_7fff)),
            0x3f80,
            "just below half -> down"
        );

        // Mantissa carry propagates into the exponent: 0x3FFF_8000 rounds up from mantissa
        // all-ones into the next binade (0x4000), not a wrapped mantissa.
        assert_eq!(
            fp32_to_bf16_rne(f32::from_bits(0x3fff_8000)),
            0x4000,
            "mantissa carry into exponent"
        );

        // Non-finite: +/-Inf exact; NaN stays a NaN (quiet bit forced, sign kept).
        assert_eq!(fp32_to_bf16_rne(f32::INFINITY), 0x7f80, "+Inf");
        assert_eq!(fp32_to_bf16_rne(f32::NEG_INFINITY), 0xff80, "-Inf");
        assert!(bf16_is_nan(fp32_to_bf16_rne(f32::NAN)), "NaN stays NaN");
        // An FP32 NaN whose payload lives only in the discarded low 16 bits must NOT alias to
        // Inf (the raw top-16 would be 0x7F80): the quiet-bit force keeps it a NaN.
        let sneaky_nan = f32::from_bits(0x7f80_0001);
        assert!(sneaky_nan.is_nan());
        assert!(
            bf16_is_nan(fp32_to_bf16_rne(sneaky_nan)),
            "low-payload NaN does not collapse to Inf"
        );
    }

    /// `bf16_to_fp32` is the exact left-shift decode, and the RNE rounder inverts it on
    /// every BF16 pattern except the subnormals, which the section-16.1 `fp32_to_bfloat16`
    /// DAZ flush maps to signed zero. Exhaustive over all 65536 encodings (fast,
    /// deterministic).
    /// `fp8::fp32_bf16_round_trip`
    #[test]
    fn fp32_bf16_round_trip() {
        for b in 0u32..=0xffff {
            let bits = b as u16;
            let f = bf16_to_fp32(bits);
            // Exact decode: low 16 bits are zero, top 16 are the BF16 pattern.
            assert_eq!(
                f.to_bits(),
                (b) << 16,
                "bf16_to_fp32 is an exact left shift"
            );
            let is_subnormal = (bits >> 7) & 0xff == 0 && (bits & 0x7f) != 0;
            if f.is_nan() {
                assert!(
                    bf16_is_nan(bits),
                    "decoded NaN came from a BF16 NaN pattern"
                );
                assert!(
                    bf16_is_nan(fp32_to_bf16_rne(f)),
                    "NaN re-encodes to a BF16 NaN"
                );
            } else if is_subnormal {
                // BF16 subnormal decodes to an FP32 denormal, which the spec's DAZ flush
                // (section 16.1) re-encodes as the same-signed zero, NOT the original bits.
                assert_eq!(
                    fp32_to_bf16_rne(f),
                    bits & 0x8000,
                    "subnormal flushes to signed zero on re-encode (DAZ=1)"
                );
            } else {
                assert_eq!(
                    fp32_to_bf16_rne(f),
                    bits,
                    "non-NaN, non-subnormal BF16 round-trips exactly"
                );
            }
        }
    }
}
