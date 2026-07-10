//! ACE group-4 outer-product accumulators (`TOP*`) — spec section 14.
//!
//! This module hosts the tile outer-product-accumulate instructions:
//!
//! * **Family E — MX FP8 rank 4** (spec section 14.1): [`_tile_top4mxbf8ps`] (BF8xBF8),
//!   [`_tile_top4mxbhf8ps`] (BF8xHF8), [`_tile_top4mxhbf8ps`] (HF8xBF8),
//!   [`_tile_top4mxhf8ps`] (HF8xHF8) — OCP MX block scaling via the implicit Block Scale
//!   register, `imm8`-selected scale groups, FP32 accumulate.
//! * **Family E' — MX INT8 rank 4** (spec section 14.2): [`_tile_top4mxbssps`] — signed
//!   INT8 with the combined `2^-12` MX INT8 implicit product bias, BSR scaled, FP32
//!   accumulate.
//! * **Family F — BF16 rank 2** (spec section 14.3): [`_tile_top2bf16ps`] — no block
//!   scaling, FP32 accumulate.
//! * **Family G — INT8 rank 4** (spec section 14.4): [`_tile_top4bssd`] /
//!   [`_tile_top4bsud`] / [`_tile_top4busd`] / [`_tile_top4buud`] — pure INT32
//!   accumulate, no scaling, no saturation.
//!
//! # Accumulation and rounding model (spec sections 14.1.6, 14.2.6, 14.3.5, 16.5)
//!
//! The FP32-accumulating forms follow the spec pseudocode EXACTLY — this is NOT a
//! per-product FP32 fold:
//!
//! 1. **MX FP8 / MX INT8 (rank 4).** The four sub-element products for one output element
//!    are accumulated EXACTLY in wide integer fixpoint (`convert_fp8_to_fixpoint64` aligns
//!    BF8 at `2^16 x value` and HF8 at `2^9 x value`; MX INT8 products are exact in 32
//!    bits). The combined exponent adjustment — the fixpoint correction plus BOTH E8M0
//!    block scales (`2^(s-127)` each), plus the `-12` MX INT8 implicit product bias for
//!    TOP4MXBSSPS — is applied to the exact sum in the PRECISE domain, with ONE conversion
//!    to FP32 (`convert_fixpoint128_scaled_to_fp32_ftz_rne`: RNE, FTZ=1). The result is
//!    then accumulated onto the prior tile element with a SINGLE FP32 add (`float32_add`
//!    with DAZ=1 on the accumulator, FTZ=1 on the output). The prior tile element enters
//!    LAST, not first.
//! 2. **Scale association.** One E8M0 A-scale per output ROW `i` (`BSR.byte[64 + i*4 +
//!    a_group]`) and one B-scale per output COLUMN `j` (`BSR.byte[j*4 + b_group]`); the two
//!    groups are selected independently by `imm8[5:4]` / `imm8[1:0]` (spec section
//!    14.1.4). An E8M0 NaN (`0xFF`) in EITHER selected scale makes THAT output element
//!    QNaN_Indefinite (early return; the products are bypassed).
//! 3. **BF16 rank 2.** Both BF16 pairs widen exactly to FP32 with DAZ=1
//!    (`bf16_to_fp32_daz`), the two products are added with one FP32 add (FTZ=1 on the
//!    sum), and the result accumulates onto the prior element with a single FP32 add
//!    (DAZ=1 accumulator, FTZ=1 output): `C = float32_add(C, RNE(a0*b0 + a1*b1))`.
//! 4. **Family G is pure integer** — INT8 operands widen to `i32` (signed sign-extend,
//!    unsigned zero-extend per mnemonic), the four products accumulate exactly, and the
//!    tile add is two's-complement wraparound. No saturating `...DS` forms exist in the
//!    spec, correctly.
//!
//! `MXCSR` is neither consulted nor updated; no FP exceptions are raised
//! (`[ace-tile-instructions.TOP.*]`).
//!
//! # Dispatch
//!
//! Each op is a safe public dispatcher plus a cfg-free `_scalar` oracle (the primary path,
//! correct on every target). All `TOP*` forms are ACE-only and gate on full `ACE`
//! (`detect::has_ace`, `[ace-tile-instructions.DETECT.1-3]`,
//! `[ace-tile-instructions.DISPATCH.1]`).

use crate::detect;
use crate::tile::{TileId, TileScope, TILE_COLSB, TILE_ROWS};

/// x86 QNaN Indefinite: the canonical quiet NaN the ACE operations produce for
/// exceptional values (E8M0 NaN scales, NaN/Inf-degenerate sums).
const QNAN_INDEFINITE: u32 = 0xFFC0_0000;

/// Compose the `imm8` scale-group selector's A field: `imm8[5:4]` (spec section 14.1.12).
pub fn ace_scale_a(group: u8) -> u8 {
    (group & 0x3) << 4
}

/// Compose the `imm8` scale-group selector's B field: `imm8[1:0]` (spec section 14.1.12).
pub fn ace_scale_b(group: u8) -> u8 {
    group & 0x3
}

// ---------------------------------------------------------------------------------------------
// Tile element accessors (FP32 / INT32 dword grid, 16x16)
// ---------------------------------------------------------------------------------------------

fn tile_dword(scope: &TileScope, id: TileId, row: usize, col: usize) -> u32 {
    let bytes = scope.tile_bytes_ref(id);
    let off = row * TILE_COLSB + col * 4;
    u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
}

fn set_tile_dword(scope: &mut TileScope, id: TileId, row: usize, col: usize, value: u32) {
    let bytes = scope.tile_bytes_mut(id);
    let off = row * TILE_COLSB + col * 4;
    bytes[off..off + 4].copy_from_slice(&value.to_le_bytes());
}

// ---------------------------------------------------------------------------------------------
// Section-16.5 pseudocode helpers
// ---------------------------------------------------------------------------------------------

/// FP8 source format of one MX outer-product operand (spec section 14.1.1).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Fp8Fmt {
    /// BF8 = FP8 E5M2.
    Bf8,
    /// HF8 = FP8 E4M3.
    Hf8,
}

/// `convert_bf8_to_fixpoint64` (spec section 16.5): BF8 (E5M2) to a 64-bit fixed-point
/// integer, result = `2^16 x float_value`. Every bit pattern is decoded by the formula —
/// the pseudocode has no NaN/Inf special case here (exceptional handling is confined to
/// the E8M0 scales).
fn convert_bf8_to_fixpoint64(fp8_byte: u8) -> i64 {
    let sign = (fp8_byte & 0x80) >> 7;
    let exp = ((fp8_byte & 0x7C) >> 2) as u32;
    let frac = (fp8_byte & 0x03) as i64;
    let mant = if exp == 0 { frac } else { frac | 0x4 };
    let e_count = if exp == 0 { 0 } else { exp - 1 };
    let magnitude = mant << e_count;
    if sign == 1 {
        -magnitude
    } else {
        magnitude
    }
}

/// `convert_hf8_to_fixpoint64` (spec section 16.5): HF8 (E4M3) to a 64-bit fixed-point
/// integer, result = `2^9 x float_value`.
fn convert_hf8_to_fixpoint64(fp8_byte: u8) -> i64 {
    let sign = (fp8_byte & 0x80) >> 7;
    let exp = ((fp8_byte & 0x78) >> 3) as u32;
    let frac = (fp8_byte & 0x07) as i64;
    let mant = if exp == 0 { frac } else { frac | 0x8 };
    let e_count = if exp == 0 { 0 } else { exp - 1 };
    let magnitude = mant << e_count;
    if sign == 1 {
        -magnitude
    } else {
        magnitude
    }
}

/// `convert_fp8_to_fixpoint64` (spec section 16.5).
fn convert_fp8_to_fixpoint64(fp8_byte: u8, fmt: Fp8Fmt) -> i64 {
    match fmt {
        Fp8Fmt::Bf8 => convert_bf8_to_fixpoint64(fp8_byte),
        Fp8Fmt::Hf8 => convert_hf8_to_fixpoint64(fp8_byte),
    }
}

/// `convert_fixpoint128_scaled_to_fp32_ftz_rne` (spec section 16.5): convert a signed
/// 128-bit fixpoint sum of products to FP32 with the combined exponent adjustment applied
/// in the precise domain — left-normalize so the J-bit reaches bit 126, RNE off the fixed
/// guard/sticky positions, fold the rounding carry into the exponent, then a single
/// FTZ/overflow boundary check (`biased <= 0` -> signed zero, `biased > 254` -> signed
/// Inf).
fn convert_fixpoint128_scaled_to_fp32_ftz_rne(x: i128, adjust: i32) -> u32 {
    if x == 0 {
        return 0; // +0.0
    }
    let sign: u32 = if x < 0 { 1 } else { 0 };
    let mut magnitude = x.unsigned_abs();

    // Left-normalize: shift until the J-bit reaches bit 126; Jbit_position tracks the
    // original leading-bit index (decremented each shift).
    let mut jbit_position: i32 = 126;
    while magnitude & (1u128 << 126) == 0 {
        jbit_position -= 1;
        magnitude <<= 1;
    }

    // Fixed extraction positions: L at 103, G at 102, sticky = OR(101:0).
    let sticky: u128 = if magnitude & ((1u128 << 102) - 1) != 0 {
        1
    } else {
        0
    };
    let gbit = (magnitude >> 102) & 1;
    let lbit = (magnitude >> 103) & 1;
    let rnd_add = gbit & (lbit | sticky); // RNE: round up iff G=1 and (L|S)=1

    let mantissa = (magnitude >> 103) as u32; // 24 bits: J at [23], fraction at [22:0]
    let rnd_mantissa = mantissa + rnd_add as u32;
    let ovf = (rnd_mantissa >> 24) & 1; // rounding carry into the exponent

    let biased = 127 + jbit_position + adjust + ovf as i32;

    if biased > 254 {
        // overflow -> +/-Inf
        return (sign << 31) | 0x7F80_0000;
    }
    if biased <= 0 {
        // subnormal or below -> FTZ=1 -> +/-0
        return sign << 31;
    }
    let frac = rnd_mantissa & 0x7F_FFFF;
    (sign << 31) | ((biased as u32) << 23) | frac
}

/// Flush a subnormal FP32 to its signed zero (the DAZ/FTZ primitive).
fn flush_subnormal(x: f32) -> f32 {
    if x != 0.0 && x.abs() < f32::MIN_POSITIVE {
        if x.is_sign_negative() {
            -0.0
        } else {
            0.0
        }
    } else {
        x
    }
}

/// `float32_add(a, b, daz_a, daz_b, ftz)` (spec section 16.5 usage): IEEE FP32 addition
/// under RNE with optional DAZ on each operand and FTZ on the result. Exceptional values
/// collapse to QNaN_Indefinite: `sum(NaN, any)` and `sum(+Inf, -Inf)` (spec section
/// 14.1.6 exceptional-value handling).
fn float32_add(a: f32, b: f32, daz_a: bool, daz_b: bool, ftz: bool) -> f32 {
    let a = if daz_a { flush_subnormal(a) } else { a };
    let b = if daz_b { flush_subnormal(b) } else { b };
    if a.is_nan() || b.is_nan() {
        return f32::from_bits(QNAN_INDEFINITE);
    }
    if a.is_infinite() && b.is_infinite() && a.is_sign_positive() != b.is_sign_positive() {
        return f32::from_bits(QNAN_INDEFINITE);
    }
    let sum = a + b; // IEEE binary32 RNE
    if ftz {
        flush_subnormal(sum)
    } else {
        sum
    }
}

/// `bf16_to_fp32_daz` (spec section 14.3.5): widen BF16 exactly to FP32, flushing BF16
/// denormals (`exp == 0, frac != 0`) to signed zero.
fn bf16_to_fp32_daz(bits: u16) -> f32 {
    if (bits >> 7) & 0xFF == 0 && bits & 0x7F != 0 {
        return f32::from_bits(((bits as u32) & 0x8000) << 16); // signed zero
    }
    crate::fp8::bf16_to_fp32(bits)
}

/// FP32 multiply with the spec's exceptional-value collapse: `mult(NaN, any)` and
/// `mult(Inf, zero)` produce QNaN_Indefinite (spec section 14.3.5 step-2 comments); the
/// IEEE product already yields a NaN for both, so canonicalizing NaN products suffices.
fn float32_mul(a: f32, b: f32) -> f32 {
    let p = a * b;
    if p.is_nan() {
        f32::from_bits(QNAN_INDEFINITE)
    } else {
        p
    }
}

// ---------------------------------------------------------------------------------------------
// Family E: MX FP8 rank 4 (TOP4MX[B|H][B|H]F8PS), spec section 14.1
// ---------------------------------------------------------------------------------------------

/// `op4mxf8_subtile` (spec section 14.1.6): one output element. E8M0 NaN in either
/// selected scale propagates QNaN_Indefinite; otherwise the 4 FP8 products accumulate
/// exactly in 128-bit fixpoint, the combined exponent adjustment (fixpoint correction +
/// both E8M0 scales) converts once to FP32 (RNE, FTZ=1), and one FP32 add accumulates onto
/// the prior element (DAZ=1 accumulator, FTZ=1 output).
fn op4mxf8_subtile(
    srcdest: f32,
    src1_quad: [u8; 4],
    fmt_a: Fp8Fmt,
    src1_scale: u8,
    src2_quad: [u8; 4],
    fmt_b: Fp8Fmt,
    src2_scale: u8,
) -> f32 {
    if src1_scale == 0xFF || src2_scale == 0xFF {
        return f32::from_bits(QNAN_INDEFINITE); // E8M0 NaN -> propagate QNaN
    }

    // Accumulate FP8 products in 128-bit integer fixpoint (DAZ=0 for FP8 inputs).
    let mut sop: i128 = 0;
    for i in 0..4 {
        let s1 = convert_fp8_to_fixpoint64(src1_quad[i], fmt_a) as i128;
        let s2 = convert_fp8_to_fixpoint64(src2_quad[i], fmt_b) as i128;
        sop += s1 * s2; // exact 128-bit integer accumulation
    }

    // Combined exponent adjustment: fp8 fixpoint correction + E8M0 scale shifts.
    //   -factor                     undoes the fp8 fixpoint scaling (BF8=2^16, HF8=2^9)
    //   + src1_scale + src2_scale - 254   both E8M0 scales as 2^(s-127) each
    let factor = match (fmt_a, fmt_b) {
        (Fp8Fmt::Bf8, Fp8Fmt::Bf8) => 32, // BF8 fixpoint scale = 2^16 per operand
        (Fp8Fmt::Hf8, Fp8Fmt::Hf8) => 18, // HF8 fixpoint scale = 2^9 per operand
        _ => 25,                          // mixed BF8/HF8
    };
    let exp_adjust = -factor + src1_scale as i32 + src2_scale as i32 - 254;

    // Convert the scaled fixpoint sum to FP32 (FTZ=1, RNE), then accumulate: DAZ=1 on the
    // FP32 tile element; daz_b=False because sop_fp32 already applied FTZ.
    let sop_fp32 = f32::from_bits(convert_fixpoint128_scaled_to_fp32_ftz_rne(sop, exp_adjust));
    float32_add(srcdest, sop_fp32, true, false, true)
}

/// `top4mxf8ps` (spec section 14.1.6): the shared rank-4 MX FP8 outer-product accumulate.
fn top4mxf8ps_scalar_impl(
    scope: &mut TileScope,
    dst: TileId,
    src1: [u8; 64],
    src2: [u8; 64],
    imm8: u8,
    fmt_a: Fp8Fmt,
    fmt_b: Fp8Fmt,
) {
    let a_group = ((imm8 >> 4) & 0x3) as usize; // imm8[5:4]: A_SCALE
    let b_group = (imm8 & 0x3) as usize; // imm8[1:0]: B_SCALE

    // src1_scales[s] = BSR.byte[64 + s*4 + a_group]; src2_scales[s] = BSR.byte[s*4 + b_group].
    let src1_scales: [u8; 16] = core::array::from_fn(|s| scope.bsr().a_scale(s, a_group));
    let src2_scales: [u8; 16] = core::array::from_fn(|s| scope.bsr().b_scale(s, b_group));

    for i in 0..TILE_ROWS {
        let src1_quad: [u8; 4] = core::array::from_fn(|b| src1[4 * i + b]); // A row i
        let src1_scale = src1_scales[i]; // E8M0 byte for row i
        for j in 0..16 {
            let src2_quad: [u8; 4] = core::array::from_fn(|b| src2[4 * j + b]); // B col j
            let src2_scale = src2_scales[j]; // E8M0 byte for col j
            let prior = f32::from_bits(tile_dword(scope, dst, i, j));
            let out = op4mxf8_subtile(
                prior, src1_quad, fmt_a, src1_scale, src2_quad, fmt_b, src2_scale,
            );
            set_tile_dword(scope, dst, i, j, out.to_bits());
        }
    }
}

macro_rules! define_top4mxf8 {
    ($(#[$doc:meta])* $name:ident, $scalar:ident, $fmt_a:expr, $fmt_b:expr) => {
        $(#[$doc])*
        pub fn $name(
            scope: &mut TileScope,
            dst: TileId,
            src1: [u8; 64],
            src2: [u8; 64],
            imm8: u8,
        ) {
            let _ = detect::has_ace; // ACE-only [DETECT.1-3]
            $scalar(scope, dst, src1, src2, imm8);
        }

        /// Portable oracle — the section-14.1.6 pseudocode (see the module header for the
        /// precise-domain accumulation model).
        pub fn $scalar(
            scope: &mut TileScope,
            dst: TileId,
            src1: [u8; 64],
            src2: [u8; 64],
            imm8: u8,
        ) {
            top4mxf8ps_scalar_impl(scope, dst, src1, src2, imm8, $fmt_a, $fmt_b);
        }
    };
}

define_top4mxf8!(
    /// `TOP4MXBF8PS` (spec section 14.1): rank-4 MX FP8 outer product, A = FP8 E5M2, B =
    /// FP8 E5M2, OCP MX block scaling via the implicit BSR (`imm8` selects the scale
    /// groups), FP32 accumulate (`[ace-tile-instructions.MX_TOP.1]`).
    _tile_top4mxbf8ps, _tile_top4mxbf8ps_scalar, Fp8Fmt::Bf8, Fp8Fmt::Bf8
);
define_top4mxf8!(
    /// `TOP4MXBHF8PS` (spec section 14.1): A = FP8 E5M2 (BF8), B = FP8 E4M3 (HF8)
    /// (`[ace-tile-instructions.MX_TOP.2]`).
    _tile_top4mxbhf8ps, _tile_top4mxbhf8ps_scalar, Fp8Fmt::Bf8, Fp8Fmt::Hf8
);
define_top4mxf8!(
    /// `TOP4MXHBF8PS` (spec section 14.1): A = FP8 E4M3 (HF8), B = FP8 E5M2 (BF8)
    /// (`[ace-tile-instructions.MX_TOP.3]`).
    _tile_top4mxhbf8ps, _tile_top4mxhbf8ps_scalar, Fp8Fmt::Hf8, Fp8Fmt::Bf8
);
define_top4mxf8!(
    /// `TOP4MXHF8PS` (spec section 14.1): A = FP8 E4M3, B = FP8 E4M3
    /// (`[ace-tile-instructions.MX_TOP.4]`).
    _tile_top4mxhf8ps, _tile_top4mxhf8ps_scalar, Fp8Fmt::Hf8, Fp8Fmt::Hf8
);

// ---------------------------------------------------------------------------------------------
// Family E': MX INT8 rank 4 (TOP4MXBSSPS), spec section 14.2
// ---------------------------------------------------------------------------------------------

/// `op4mxb_subtile` (spec section 14.2.6): one MX INT8 output element. The four signed
/// byte products accumulate exactly in 32-bit integer; `exp_adjust = -12 + src1_scale +
/// src2_scale - 254` carries the combined `2^-12` MX INT8 implicit product bias (each MX
/// INT8 term has an implicit scale of `2^-6`) plus both E8M0 block scales.
fn op4mxb_subtile(
    srcdest: f32,
    src1_quad: [u8; 4],
    src1_scale: u8,
    src2_quad: [u8; 4],
    src2_scale: u8,
) -> f32 {
    if src1_scale == 0xFF || src2_scale == 0xFF {
        return f32::from_bits(QNAN_INDEFINITE); // E8M0 NaN -> propagate QNaN
    }

    // Accumulate signed byte products in 32-bit integer (exact).
    let mut sop: i32 = 0;
    for i in 0..4 {
        sop += (src1_quad[i] as i8 as i32) * (src2_quad[i] as i8 as i32);
    }

    // -12 product implicit bias + both E8M0 block scales as 2^(s-127) each.
    let exp_adjust = -12 + src1_scale as i32 + src2_scale as i32 - 254;

    let sop_fp32 = f32::from_bits(convert_fixpoint128_scaled_to_fp32_ftz_rne(
        sop as i128,
        exp_adjust,
    ));
    float32_add(srcdest, sop_fp32, true, false, true)
}

/// `TOP4MXBSSPS` (spec section 14.2): rank-4 MX INT8 outer product, A = MX INT8 signed,
/// B = MX INT8 signed, OCP MX block scaling via the implicit BSR (`imm8` selects the scale
/// groups exactly as the MX FP8 forms do), FP32 accumulate
/// (`[ace-tile-instructions.MX_TOP.5]`).
pub fn _tile_top4mxbssps(
    scope: &mut TileScope,
    dst: TileId,
    src1: [u8; 64],
    src2: [u8; 64],
    imm8: u8,
) {
    let _ = detect::has_ace; // ACE-only [DETECT.1-3]
    _tile_top4mxbssps_scalar(scope, dst, src1, src2, imm8);
}

/// Portable `TOP4MXBSSPS` oracle — the section-14.2.6 pseudocode.
pub fn _tile_top4mxbssps_scalar(
    scope: &mut TileScope,
    dst: TileId,
    src1: [u8; 64],
    src2: [u8; 64],
    imm8: u8,
) {
    let a_group = ((imm8 >> 4) & 0x3) as usize;
    let b_group = (imm8 & 0x3) as usize;
    let src1_scales: [u8; 16] = core::array::from_fn(|s| scope.bsr().a_scale(s, a_group));
    let src2_scales: [u8; 16] = core::array::from_fn(|s| scope.bsr().b_scale(s, b_group));

    for i in 0..TILE_ROWS {
        let src1_quad: [u8; 4] = core::array::from_fn(|b| src1[4 * i + b]);
        let src1_scale = src1_scales[i];
        for j in 0..16 {
            let src2_quad: [u8; 4] = core::array::from_fn(|b| src2[4 * j + b]);
            let src2_scale = src2_scales[j];
            let prior = f32::from_bits(tile_dword(scope, dst, i, j));
            let out = op4mxb_subtile(prior, src1_quad, src1_scale, src2_quad, src2_scale);
            set_tile_dword(scope, dst, i, j, out.to_bits());
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Family F: BF16 rank 2 (TOP2BF16PS), spec section 14.3
// ---------------------------------------------------------------------------------------------

/// `op2bf16_subtile` (spec section 14.3.5): two BF16 pairs -> FP32 SoP -> FP32 accumulate.
/// Step 1 widens with DAZ=1 (BF16 denormals flush to signed zero); step 2 forms the two
/// exact FP32 products and adds them with FTZ=1 on the sum; step 3 accumulates onto the
/// prior element with DAZ=1 on the accumulator and FTZ=1 on the output.
fn op2bf16_subtile(srcdest: f32, op1: u32, op2: u32) -> f32 {
    let a0 = bf16_to_fp32_daz((op1 & 0xFFFF) as u16); // low  BF16 of src1 lane
    let a1 = bf16_to_fp32_daz((op1 >> 16) as u16); // high BF16 of src1 lane
    let b0 = bf16_to_fp32_daz((op2 & 0xFFFF) as u16);
    let b1 = bf16_to_fp32_daz((op2 >> 16) as u16);

    let sop = float32_add(float32_mul(a0, b0), float32_mul(a1, b1), false, false, true);
    float32_add(srcdest, sop, true, false, true)
}

/// `TOP2BF16PS` (spec section 14.3): rank-2 BF16 outer product, no block scaling, FP32
/// accumulate (`[ace-tile-instructions.BF16_TOP.1]`).
pub fn _tile_top2bf16ps(scope: &mut TileScope, dst: TileId, src1: [u16; 32], src2: [u16; 32]) {
    let _ = detect::has_ace; // ACE-only [DETECT.1-3]
    _tile_top2bf16ps_scalar(scope, dst, src1, src2);
}

/// Portable `TOP2BF16PS` oracle — the section-14.3.5 pseudocode.
pub fn _tile_top2bf16ps_scalar(
    scope: &mut TileScope,
    dst: TileId,
    src1: [u16; 32],
    src2: [u16; 32],
) {
    for i in 0..TILE_ROWS {
        let op1 = (src1[2 * i] as u32) | ((src1[2 * i + 1] as u32) << 16); // 2 BF16 for row i
        for j in 0..16 {
            let op2 = (src2[2 * j] as u32) | ((src2[2 * j + 1] as u32) << 16); // 2 BF16 for col j
            let prior = f32::from_bits(tile_dword(scope, dst, i, j));
            let out = op2bf16_subtile(prior, op1, op2);
            set_tile_dword(scope, dst, i, j, out.to_bits());
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Family G: INT8 rank 4 (TOP4B[U|S][U|S]D), spec section 14.4
// ---------------------------------------------------------------------------------------------

macro_rules! define_top4b {
    ($(#[$doc:meta])* $name:ident, $scalar:ident, $a_signed:expr, $b_signed:expr) => {
        $(#[$doc])*
        pub fn $name(scope: &mut TileScope, dst: TileId, src1: [u8; 64], src2: [u8; 64]) {
            let _ = detect::has_ace; // ACE-only [DETECT.1-3]
            $scalar(scope, dst, src1, src2);
        }

        /// Portable oracle — the section-14.4.5 pseudocode: widen per the mnemonic's
        /// signedness, four exact INT32 products per element, wraparound tile accumulate.
        /// No DAZ/FTZ: pure integer; no FP exceptions raised or denoted.
        pub fn $scalar(scope: &mut TileScope, dst: TileId, src1: [u8; 64], src2: [u8; 64]) {
            for row in 0..TILE_ROWS {
                for col in 0..16 {
                    let mut acc: i32 = 0;
                    for k in 0..4 {
                        let a: i32 = if $a_signed {
                            src1[4 * row + k] as i8 as i32 // signed
                        } else {
                            src1[4 * row + k] as i32 // unsigned
                        };
                        let b: i32 = if $b_signed {
                            src2[4 * col + k] as i8 as i32
                        } else {
                            src2[4 * col + k] as i32
                        };
                        acc = acc.wrapping_add(a.wrapping_mul(b));
                    }
                    let prior = tile_dword(scope, dst, row, col) as i32;
                    set_tile_dword(scope, dst, row, col, prior.wrapping_add(acc) as u32);
                }
            }
        }
    };
}

define_top4b!(
    /// `TOP4BSSD` (spec section 14.4): rank-4 byte outer product, A = INT8 signed, B =
    /// INT8 signed, INT32 accumulate, no block scaling
    /// (`[ace-tile-instructions.INT8_TOP.1]`).
    _tile_top4bssd, _tile_top4bssd_scalar, true, true
);
define_top4b!(
    /// `TOP4BSUD` (spec section 14.4): A = INT8 signed, B = INT8 unsigned
    /// (`[ace-tile-instructions.INT8_TOP.2]`).
    _tile_top4bsud, _tile_top4bsud_scalar, true, false
);
define_top4b!(
    /// `TOP4BUSD` (spec section 14.4): A = INT8 unsigned, B = INT8 signed
    /// (`[ace-tile-instructions.INT8_TOP.3]`).
    _tile_top4busd, _tile_top4busd_scalar, false, true
);
define_top4b!(
    /// `TOP4BUUD` (spec section 14.4): A = INT8 unsigned, B = INT8 unsigned
    /// (`[ace-tile-instructions.INT8_TOP.4]`).
    _tile_top4buud, _tile_top4buud_scalar, false, false
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bsr::{_bsrinit, _bsrmovf};
    use crate::tile::{_tile_loadconfig, TileConfig};

    fn scope() -> (TileScope, TileId) {
        let s = _tile_loadconfig(&TileConfig::ace()).unwrap();
        let id = s.tile(0).unwrap();
        (s, id)
    }

    fn read_f32(scope: &TileScope, id: TileId, i: usize, j: usize) -> f32 {
        f32::from_bits(tile_dword(scope, id, i, j))
    }

    // ---- section-16.5 helper pins ----

    /// Fixpoint decoders: BF8 result = 2^16 x value, HF8 result = 2^9 x value.
    /// `top::fixpoint_decoders`
    #[test]
    fn fixpoint_decoders() {
        // BF8 1.0 = S.01111.00 = 0x3C: 2^16 * 1.0 = 65536.
        assert_eq!(convert_bf8_to_fixpoint64(0x3C), 65536);
        // BF8 -1.5 = S.01111.10 = 0xBE: mant 0b110, e_count 14 -> -(6 << 14) = -1.5 * 2^16.
        assert_eq!(convert_bf8_to_fixpoint64(0xBE), -98304);
        // BF8 min subnormal S.00000.01 = 2^-16: fixpoint 1.
        assert_eq!(convert_bf8_to_fixpoint64(0x01), 1);
        // HF8 1.0 = S.0111.000 = 0x38: 2^9 * 1.0 = 512.
        assert_eq!(convert_hf8_to_fixpoint64(0x38), 512);
        // HF8 min subnormal S.0000.001 = 2^-9: fixpoint 1.
        assert_eq!(convert_hf8_to_fixpoint64(0x01), 1);
        // Sign.
        assert_eq!(convert_hf8_to_fixpoint64(0xB8), -512);
    }

    /// The fixpoint->FP32 converter: exact values, RNE at the guard/sticky boundary,
    /// FTZ and overflow boundaries.
    /// `top::fixpoint_to_fp32`
    #[test]
    fn fixpoint_to_fp32() {
        // 3 * 2^0: adjust 0 -> 3.0.
        assert_eq!(
            f32::from_bits(convert_fixpoint128_scaled_to_fp32_ftz_rne(3, 0)),
            3.0
        );
        // Negative.
        assert_eq!(
            f32::from_bits(convert_fixpoint128_scaled_to_fp32_ftz_rne(-3, 0)),
            -3.0
        );
        // 2^24 + 1 needs rounding: RNE to even -> 16777216.0.
        assert_eq!(
            f32::from_bits(convert_fixpoint128_scaled_to_fp32_ftz_rne((1 << 24) + 1, 0)),
            16_777_216.0
        );
        // 2^24 + 3 rounds up -> 16777220.0.
        assert_eq!(
            f32::from_bits(convert_fixpoint128_scaled_to_fp32_ftz_rne((1 << 24) + 3, 0)),
            16_777_220.0
        );
        // Subnormal boundary: 1 * 2^-127 -> biased = 0 -> FTZ -> +0.
        assert_eq!(convert_fixpoint128_scaled_to_fp32_ftz_rne(1, -127), 0);
        // Min normal: 1 * 2^-126 -> biased = 1.
        assert_eq!(
            f32::from_bits(convert_fixpoint128_scaled_to_fp32_ftz_rne(1, -126)),
            f32::MIN_POSITIVE
        );
        // Overflow: 1 * 2^128 -> +Inf.
        assert_eq!(
            f32::from_bits(convert_fixpoint128_scaled_to_fp32_ftz_rne(1, 128)),
            f32::INFINITY
        );
    }

    // ---- family G (integer) ----

    /// Signedness matrix: byte 0x80 is -128 signed / 128 unsigned; each mnemonic widens
    /// its operand per the section-14.4.1 table.
    /// `top::int8_signedness_matrix`
    #[test]
    fn int8_signedness_matrix() {
        let mut src1 = [0u8; 64];
        let mut src2 = [0u8; 64];
        src1[0] = 0x80; // row 0, k = 0
        src2[0] = 2; // col 0, k = 0

        let (mut s, id) = scope();
        _tile_top4bssd(&mut s, id, src1, src2);
        assert_eq!(tile_dword(&s, id, 0, 0) as i32, -256, "signed A: -128 * 2");

        let (mut s, id) = scope();
        _tile_top4busd(&mut s, id, src1, src2);
        assert_eq!(tile_dword(&s, id, 0, 0) as i32, 256, "unsigned A: 128 * 2");
    }

    /// Accumulation is wraparound (no saturation) and adds onto the PRIOR tile value.
    /// `top::int8_wraparound_accumulate`
    #[test]
    fn int8_wraparound_accumulate() {
        let (mut s, id) = scope();
        set_tile_dword(&mut s, id, 0, 0, i32::MAX as u32);
        let mut src1 = [0u8; 64];
        let mut src2 = [0u8; 64];
        src1[0] = 1;
        src2[0] = 1;
        _tile_top4bssd(&mut s, id, src1, src2);
        assert_eq!(
            tile_dword(&s, id, 0, 0) as i32,
            i32::MIN,
            "i32::MAX + 1 wraps (no saturating ...DS forms exist)"
        );
    }

    /// The outer-product shape: element (i, j) contracts A row i with B column j.
    /// `top::outer_product_shape`
    #[test]
    fn outer_product_shape() {
        let mut src1 = [0u8; 64];
        let mut src2 = [0u8; 64];
        // A row 2 = [1, 2, 3, 4]; B col 5 = [5, 6, 7, 8].
        for (k, v) in [1u8, 2, 3, 4].iter().enumerate() {
            src1[4 * 2 + k] = *v;
        }
        for (k, v) in [5u8, 6, 7, 8].iter().enumerate() {
            src2[4 * 5 + k] = *v;
        }
        let (mut s, id) = scope();
        _tile_top4buud(&mut s, id, src1, src2);
        // 1*5 + 2*6 + 3*7 + 4*8 = 70.
        assert_eq!(tile_dword(&s, id, 2, 5), 70, "C[2][5] = A[2]·B[5]");
        assert_eq!(
            tile_dword(&s, id, 5, 2),
            0,
            "C[5][2] untouched (A row 5 = 0)"
        );
    }

    // ---- family F (BF16) ----

    /// TOP2BF16PS: C = C + RNE(a0*b0 + a1*b1) — the SoP is rounded ONCE before the single
    /// accumulate add (spec section 14.3.5), not folded product-by-product.
    /// `top::bf16_single_sop_rounding`
    #[test]
    fn bf16_single_sop_rounding() {
        let one = 0x3F80u16; // BF16 1.0
        let two = 0x4000u16; // BF16 2.0
        let mut src1 = [0u16; 32];
        let mut src2 = [0u16; 32];
        src1[0] = one; // row 0 pair = (1.0, 2.0)
        src1[1] = two;
        src2[0] = two; // col 0 pair = (2.0, 1.0)
        src2[1] = one;
        let (mut s, id) = scope();
        set_tile_dword(&mut s, id, 0, 0, 10.0f32.to_bits());
        _tile_top2bf16ps(&mut s, id, src1, src2);
        assert_eq!(
            read_f32(&s, id, 0, 0),
            10.0 + (1.0 * 2.0 + 2.0 * 1.0),
            "C + RNE(p0 + p1)"
        );
    }

    /// BF16 DAZ=1 on inputs: a BF16 denormal operand contributes exactly zero; NaN and
    /// Inf x 0 products collapse to QNaN_Indefinite.
    /// `top::bf16_daz_and_exceptional`
    #[test]
    fn bf16_daz_and_exceptional() {
        // Denormal BF16 (exp 0, frac != 0) flushes to zero on input.
        let mut src1 = [0u16; 32];
        let mut src2 = [0u16; 32];
        src1[0] = 0x0001; // BF16 min denormal
        src2[0] = 0x3F80; // 1.0
        let (mut s, id) = scope();
        _tile_top2bf16ps(&mut s, id, src1, src2);
        assert_eq!(
            read_f32(&s, id, 0, 0),
            0.0,
            "denormal input flushed (DAZ=1)"
        );

        // Inf x 0 -> QNaN_Indefinite poisons only the affected element.
        let mut src1 = [0u16; 32];
        let mut src2 = [0u16; 32];
        src1[0] = 0x7F80; // BF16 +Inf
        src2[0] = 0x0000; // 0.0
        src1[2] = 0x3F80; // row 1 pair = (1.0, 0)
        src2[2] = 0x3F80; // col 1 pair = (1.0, 0)
        let (mut s, id) = scope();
        _tile_top2bf16ps(&mut s, id, src1, src2);
        assert_eq!(
            read_f32(&s, id, 0, 0).to_bits(),
            QNAN_INDEFINITE,
            "Inf x 0 -> QNaN_Indefinite"
        );
        assert_eq!(read_f32(&s, id, 1, 1), 1.0, "other elements unaffected");
    }

    // ---- family E (MX FP8) ----

    /// With unit scales (0x7F selected in both groups), the MX FP8 product reduces to the
    /// plain OCP FP8 outer product — the section-14.1.3 note — and the four products are
    /// summed EXACTLY before the single RNE conversion.
    /// `top::mxfp8_unit_scale_plain_product`
    #[test]
    fn mxfp8_unit_scale_plain_product() {
        let (mut s, id) = scope();
        // LDTILECFG left the BSR at INIT (0x7F everywhere) — unit scales in every group.
        let mut src1 = [0u8; 64];
        let mut src2 = [0u8; 64];
        // A row 0 = [1.0, 1.5, 0, 0] BF8; B col 0 = [2.0, 2.0, 0, 0] BF8.
        src1[0] = 0x3C; // 1.0
        src1[1] = 0x3D; // 1.25? No: S.01111.01 = 1.25. Use as-is and pin the exact value.
        src2[0] = 0x40; // 2.0
        src2[1] = 0x40; // 2.0
        _tile_top4mxbf8ps(&mut s, id, src1, src2, 0);
        // 1.0*2.0 + 1.25*2.0 = 4.5, exact.
        assert_eq!(read_f32(&s, id, 0, 0), 4.5);
    }

    /// imm8 selects the scale groups: A from imm8[5:4], B from imm8[1:0]. A scale of
    /// 0x80 = 2^1 in the selected A group doubles row 0's result relative to group 0.
    /// `top::mxfp8_imm8_group_selection`
    #[test]
    fn mxfp8_imm8_group_selection() {
        let (mut s, id) = scope();
        // A scales: element 0, group 1 = 0x80 (2^1); everything else 0x7F (2^0).
        let mut a_scales = [0x7Fu8; 64];
        a_scales[1] = 0x80; // A_scales[0], group 1 (byte 0*4 + 1)
        _bsrmovf(&mut s, a_scales, [0x7F; 64]);

        let mut src1 = [0u8; 64];
        let mut src2 = [0u8; 64];
        src1[0] = 0x3C; // A[0][0] = 1.0 BF8
        src2[0] = 0x3C; // B[0][0] = 1.0 BF8

        // Group (a=0, b=0): unit scale -> 1.0.
        _tile_top4mxbf8ps(&mut s, id, src1, src2, ace_scale_a(0) | ace_scale_b(0));
        assert_eq!(read_f32(&s, id, 0, 0), 1.0);

        // Group (a=1, b=0): A scale 2^1 -> adds 2.0 (accumulates onto the 1.0).
        _tile_top4mxbf8ps(&mut s, id, src1, src2, ace_scale_a(1) | ace_scale_b(0));
        assert_eq!(
            read_f32(&s, id, 0, 0),
            3.0,
            "scale 0x80 = 2^1 doubles the product"
        );
    }

    /// The A-scale is associated with the output ROW, the B-scale with the output COLUMN —
    /// not with the contraction index.
    /// `top::mxfp8_scale_row_col_association`
    #[test]
    fn mxfp8_scale_row_col_association() {
        let (mut s, id) = scope();
        let mut a_scales = [0x7Fu8; 64];
        a_scales[2 * 4] = 0x80; // A_scales[2] group 0 = 2^1: affects output ROW 2 only
        _bsrmovf(&mut s, a_scales, [0x7F; 64]);

        let mut src1 = [0u8; 64];
        let mut src2 = [0u8; 64];
        src1[0] = 0x3C; // A row 0 (bytes 4*0..) = 1.0
        src1[4 * 2] = 0x3C; // A row 2 = 1.0 (kept)
        src2[0] = 0x3C; // B col 0 = 1.0
        _tile_top4mxbf8ps(&mut s, id, src1, src2, 0);
        assert_eq!(read_f32(&s, id, 0, 0), 1.0, "row 0 uses A_scales[0] = 2^0");
        assert_eq!(read_f32(&s, id, 2, 0), 2.0, "row 2 uses A_scales[2] = 2^1");
    }

    /// An E8M0 NaN (0xFF) scale makes exactly the affected output elements
    /// QNaN_Indefinite — the products are bypassed, other elements are untouched.
    /// `top::mxfp8_e8m0_nan_per_element`
    #[test]
    fn mxfp8_e8m0_nan_per_element() {
        let (mut s, id) = scope();
        let mut a_scales = [0x7Fu8; 64];
        a_scales[4] = 0xFF; // A_scales[1] group 0 (byte 1*4 + 0) = NaN: poisons output row 1 only
        _bsrmovf(&mut s, a_scales, [0x7F; 64]);

        let mut src1 = [0u8; 64];
        let mut src2 = [0u8; 64];
        src1[0] = 0x3C; // A row 0
        src1[4] = 0x3C; // A row 1
        src2[0] = 0x3C;
        _tile_top4mxbf8ps(&mut s, id, src1, src2, 0);
        assert_eq!(read_f32(&s, id, 0, 0), 1.0, "row 0 unaffected");
        assert_eq!(
            read_f32(&s, id, 1, 0).to_bits(),
            QNAN_INDEFINITE,
            "NaN A-scale for row 1 -> QNaN_Indefinite"
        );
        assert_eq!(read_f32(&s, id, 1, 1).to_bits(), QNAN_INDEFINITE);
        assert_eq!(read_f32(&s, id, 2, 0), 0.0, "row 2 untouched");
    }

    /// Scaling is applied ONCE to the sum in the precise domain — not per operand. With
    /// A-scale 2^1, the result is 2 x (sum of products), not 4 x (an s^2-per-product
    /// model's answer for scale applied to both operands).
    /// `top::mxfp8_scale_applied_once_to_sum`
    #[test]
    fn mxfp8_scale_applied_once_to_sum() {
        let (mut s, id) = scope();
        let mut a_scales = [0x7Fu8; 64];
        a_scales[0] = 0x80; // A_scales[0] group 0 = 2^1
        _bsrmovf(&mut s, a_scales, [0x7F; 64]);

        let mut src1 = [0u8; 64];
        let mut src2 = [0u8; 64];
        src1[0] = 0x3C; // 1.0
        src2[0] = 0x3C; // 1.0
        _tile_top4mxbf8ps(&mut s, id, src1, src2, 0);
        assert_eq!(
            read_f32(&s, id, 0, 0),
            2.0,
            "combined scale 2^(s_a + s_b - 254) = 2^1 applied once to the sum"
        );
    }

    /// Mixed-format forms decode operand A and B with their own formats (BF8 vs HF8
    /// fixpoint alignment, `factor = 25`).
    /// `top::mxfp8_mixed_formats`
    #[test]
    fn mxfp8_mixed_formats() {
        let (mut s, id) = scope();
        let mut src1 = [0u8; 64];
        let mut src2 = [0u8; 64];
        src1[0] = 0x3C; // BF8 1.0 (as A of TOP4MXBHF8PS)
        src2[0] = 0x30; // HF8 0.5 = S.0110.000
        _tile_top4mxbhf8ps(&mut s, id, src1, src2, 0);
        assert_eq!(read_f32(&s, id, 0, 0), 0.5, "BF8 1.0 x HF8 0.5");
    }

    // ---- family E' (MX INT8) ----

    /// TOP4MXBSSPS carries the combined 2^-12 MX INT8 implicit product bias: at unit E8M0
    /// scales, 3 x (-2) accumulates -6 * 2^-12, NOT -6 (the v1.14 revision pin).
    /// `top::mxint8_implicit_bias`
    #[test]
    fn mxint8_implicit_bias() {
        let (mut s, id) = scope();
        let mut src1 = [0u8; 64];
        let mut src2 = [0u8; 64];
        src1[0] = 3i8 as u8;
        src2[0] = (-2i8) as u8;
        _tile_top4mxbssps(&mut s, id, src1, src2, 0);
        assert_eq!(
            read_f32(&s, id, 0, 0),
            -6.0 * (2.0f32).powi(-12),
            "each MX INT8 term has implicit scale 2^-6; the product carries 2^-12"
        );
    }

    /// The E8M0 NaN scale rule and imm8 group selection apply to the INT8 form exactly as
    /// to the FP8 forms.
    /// `top::mxint8_scales`
    #[test]
    fn mxint8_scales() {
        let (mut s, id) = scope();
        let mut b_scales = [0x7Fu8; 64];
        b_scales[2] = 0xFF; // B_scales[0] group 2 (byte 0*4 + 2) = NaN
        _bsrmovf(&mut s, [0x7F; 64], b_scales);

        let mut src1 = [0u8; 64];
        let mut src2 = [0u8; 64];
        src1[0] = 1i8 as u8;
        src2[0] = 1i8 as u8;
        // Group b=0 is clean -> finite result.
        _tile_top4mxbssps(&mut s, id, src1, src2, ace_scale_b(0));
        assert!(read_f32(&s, id, 0, 0).is_finite());
        // Group b=2 hits the NaN scale -> QNaN_Indefinite for column 0.
        _tile_top4mxbssps(&mut s, id, src1, src2, ace_scale_b(2));
        assert_eq!(read_f32(&s, id, 0, 0).to_bits(), QNAN_INDEFINITE);
    }

    // ---- accumulate-add semantics shared by E/E'/F ----

    /// The prior tile element enters LAST via a single FP32 add with DAZ=1 on the
    /// accumulator: a subnormal accumulator is flushed to signed zero before the add.
    /// `top::accumulator_daz`
    #[test]
    fn accumulator_daz() {
        let (mut s, id) = scope();
        // Seed C[0][0] with a subnormal.
        set_tile_dword(&mut s, id, 0, 0, 0x0000_0001);
        let mut src1 = [0u16; 32];
        let mut src2 = [0u16; 32];
        src1[0] = 0x3F80; // 1.0
        src2[0] = 0x3F80; // 1.0
        _tile_top2bf16ps(&mut s, id, src1, src2);
        assert_eq!(
            read_f32(&s, id, 0, 0),
            1.0,
            "subnormal accumulator flushed (DAZ=1) before the single add"
        );

        // BSRINIT after arbitrary scale writes restores unit scaling for the MX forms.
        let mut sc = _tile_loadconfig(&TileConfig::ace()).unwrap();
        let idc = sc.tile(0).unwrap();
        _bsrmovf(&mut sc, [0x00; 64], [0x00; 64]);
        _bsrinit(&mut sc);
        let mut a = [0u8; 64];
        let mut b = [0u8; 64];
        a[0] = 0x3C;
        b[0] = 0x3C;
        _tile_top4mxbf8ps(&mut sc, idc, a, b, 0);
        assert_eq!(f32::from_bits(tile_dword(&sc, idc, 0, 0)), 1.0);
    }
}
