//! ACE group-4 family C: tile-row converts (spec sections 12.4-12.6).
//!
//! This module models the five family-C convert instructions. Each reads one addressed row
//! of a source tile, converts it element-wise to a narrower target format, and writes the
//! result into a destination ZMM vector; the source tile is left unchanged:
//!
//! * `TCVTROWD2PS` — [`_tile_cvtrowd2ps`] converts a tile row of `INT32` elements to `FP32`
//!   (RNE) (spec section 12.4) (`[ace-tile-instructions.TCVTROW.1]`).
//! * `TCVTROWPS2BF16H` / `TCVTROWPS2BF16L` — [`_tile_cvtrowps2bf16h`] /
//!   [`_tile_cvtrowps2bf16l`] convert a tile row of `FP32` elements to `BF16` (via
//!   [`fp32_to_bf16_rne`], the section-16.1 `fp32_to_bfloat16` helper) into the high / low
//!   word of each dword (spec section 12.5) (`[ace-tile-instructions.TCVTROW.2]`,
//!   `[ace-tile-instructions.TCVTROW.3]`).
//! * `TCVTROWPS2PHH` / `TCVTROWPS2PHL` — [`_tile_cvtrowps2phh`] / [`_tile_cvtrowps2phl`]
//!   convert a tile row of `FP32` elements to `FP16` (via [`fp32_to_fp16_rne`]) into the
//!   high / low word of each dword (spec section 12.6)
//!   (`[ace-tile-instructions.TCVTROW.4]`, `[ace-tile-instructions.TCVTROW.5]`).
//!
//! # Index semantics (spec section 12.1.1)
//!
//! `row = specifier & 0xF`: only bits [3:0] of the immediate/GPR row specifier are
//! relevant, an out-of-range specifier raises NO fault, and every specifier addresses a
//! valid row of the fixed 16-row tile — these converts are total.
//!
//! # Disjoint half-lanes (INV-7)
//!
//! Each narrow result occupies one 16-bit word of a 32-bit destination slot. Per the
//! section-12.5.3/12.6.3 pseudocode (`pos`/`zeropos`), the `H` forms write the *high* word
//! of each dword (odd `u16` lane `2*col + 1`) and zero the low word; the `L` forms write
//! the *low* word (even lane `2*col`) and zero the high word. So `H` and `L` touch
//! DISJOINT sets of destination word lanes — odd vs even.
//!
//! # Rounding and DAZ/FTZ (spec sections 12.4.1, 12.5.1, 12.6.1)
//!
//! All three converts round RTNE as if `MXCSR.RC=RNE`; MXCSR is neither consulted nor
//! updated and no FP exceptions are generated. For the BF16 forms "DAZ is not obeyed and
//! is always assumed DAZ=1; FTZ is not obeyed and is always assumed FTZ=1" — FP32 denormal
//! inputs produce BF16 zero, which [`fp32_to_bf16_rne`]'s leading denormal flush
//! implements. For the FP16 forms, input FP32 denormals result in FP16 zero output (an
//! FP32 denormal is below 2^-126, far under FP16's 2^-24 minimum subnormal, so RNE rounds
//! it to signed zero without an explicit flush) while FP16 denormal OUTPUTS are permitted.
//! `INT32 -> FP32` is the exact `i32 as f32` conversion (RNE for magnitudes beyond 2^24).
//!
//! # Dispatch
//!
//! Each convert is a safe public dispatcher plus a cfg-free `_scalar` oracle (the primary
//! path, correct on every target). Per the section-15.3 feature enumeration the family-C
//! converts gate on `AMX-AVX512 || ACE_VSN >= 1` ([`detect::has_amx_avx512`],
//! `[ace-tile-instructions.DETECT.1-2]`, `[ace-tile-instructions.DISPATCH.1]`). The
//! register model lives in Rust, so the dispatchers reference the detector to mark the
//! gate site and take the scalar oracle.

use crate::detect;
use crate::fp8::{fp32_to_bf16_rne, fp32_to_fp16_rne};
use crate::tile::{TileId, TileScope, TILE_COLSB};

/// Number of four-byte (`INT32`/`FP32`) elements in one tile row (`64 / 4 == 16`); also
/// the `f32` lane count of a `D2PS` destination ZMM.
pub const ROW_FP32_LANES: usize = 16;

/// Number of 16-bit (`BF16`/`FP16`) lanes in a 512-bit destination ZMM: each of the
/// [`ROW_FP32_LANES`] source elements maps to one 32-bit destination slot = two `u16`
/// lanes.
pub const ZMM_WORD_LANES: usize = 2 * ROW_FP32_LANES;

/// Read row `specifier & 0xF` of tile `src` as 16 little-endian dwords (spec section
/// 12.1.1: the masked index never faults, so this is total).
fn read_row_dwords(scope: &TileScope, src: TileId, row: u32) -> [u32; ROW_FP32_LANES] {
    let row = (row & 0xF) as usize;
    let bytes = scope.tile_bytes_ref(src);
    let start = row * TILE_COLSB;
    core::array::from_fn(|lane| {
        let off = start + lane * 4;
        u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
    })
}

/// Pack 16 narrow results into the section-12.5.3/12.6.3 `pos`/`zeropos` layout: result
/// `col` goes to `u16` lane `2*col + pos` and lane `2*col + zeropos` is written zero,
/// where `pos = 1` for the `H` variant and `0` for `L` (INV-7).
fn pack_half_lanes(narrow: [u16; ROW_FP32_LANES], high: bool) -> [u16; ZMM_WORD_LANES] {
    let mut out = [0u16; ZMM_WORD_LANES];
    for (i, &w) in narrow.iter().enumerate() {
        out[2 * i + usize::from(high)] = w;
    }
    out
}

// ---------------------------------------------------------------------------------------------
// TCVTROWD2PS — tile row INT32 -> FP32
// ---------------------------------------------------------------------------------------------

/// `TCVTROWD2PS` (spec section 12.4): convert row `row & 0xF` of tile `src` from `INT32`
/// to `FP32` (RNE) (`[ace-tile-instructions.TCVTROW.1]`).
///
/// Gates on `AMX-AVX512 || ACE_VSN >= 1` (`[ace-tile-instructions.DETECT.1-2]`,
/// `[ace-tile-instructions.DISPATCH.1]`).
pub fn _tile_cvtrowd2ps(scope: &TileScope, src: TileId, row: u32) -> [f32; ROW_FP32_LANES] {
    let _ = detect::has_amx_avx512; // family-C gate [DETECT.1-2]
    _tile_cvtrowd2ps_scalar(scope, src, row)
}

/// Portable `TCVTROWD2PS` oracle — the section-12.4.4 pseudocode: each dword converted
/// `INT32 -> FP32` under RNE (`i32 as f32` is exactly that conversion).
pub fn _tile_cvtrowd2ps_scalar(
    scope: &TileScope,
    src: TileId,
    row: u32,
) -> [f32; ROW_FP32_LANES] {
    let dwords = read_row_dwords(scope, src, row);
    core::array::from_fn(|i| dwords[i] as i32 as f32)
}

// ---------------------------------------------------------------------------------------------
// TCVTROWPS2BF16{H,L} — tile row FP32 -> BF16, high / low word of each dword
// ---------------------------------------------------------------------------------------------

/// `TCVTROWPS2BF16H` (spec section 12.5): convert row `row & 0xF` of tile `src` from
/// `FP32` to `BF16` (RNE, DAZ=1/FTZ=1 via [`fp32_to_bf16_rne`]) into the HIGH word of each
/// dword; the low words are zeroed (`[ace-tile-instructions.TCVTROW.2]`). Gates as
/// [`_tile_cvtrowd2ps`].
pub fn _tile_cvtrowps2bf16h(scope: &TileScope, src: TileId, row: u32) -> [u16; ZMM_WORD_LANES] {
    let _ = detect::has_amx_avx512; // family-C gate [DETECT.1-2]
    _tile_cvtrowps2bf16h_scalar(scope, src, row)
}

/// Portable `TCVTROWPS2BF16H` oracle — section-12.5.3 with `variant == "H"`.
pub fn _tile_cvtrowps2bf16h_scalar(
    scope: &TileScope,
    src: TileId,
    row: u32,
) -> [u16; ZMM_WORD_LANES] {
    let dwords = read_row_dwords(scope, src, row);
    let narrow = core::array::from_fn(|i| fp32_to_bf16_rne(f32::from_bits(dwords[i])));
    pack_half_lanes(narrow, true)
}

/// `TCVTROWPS2BF16L` (spec section 12.5): as [`_tile_cvtrowps2bf16h`] but into the LOW
/// word of each dword; the high words are zeroed (`[ace-tile-instructions.TCVTROW.3]`).
pub fn _tile_cvtrowps2bf16l(scope: &TileScope, src: TileId, row: u32) -> [u16; ZMM_WORD_LANES] {
    let _ = detect::has_amx_avx512; // family-C gate [DETECT.1-2]
    _tile_cvtrowps2bf16l_scalar(scope, src, row)
}

/// Portable `TCVTROWPS2BF16L` oracle — section-12.5.3 with `variant == "L"`.
pub fn _tile_cvtrowps2bf16l_scalar(
    scope: &TileScope,
    src: TileId,
    row: u32,
) -> [u16; ZMM_WORD_LANES] {
    let dwords = read_row_dwords(scope, src, row);
    let narrow = core::array::from_fn(|i| fp32_to_bf16_rne(f32::from_bits(dwords[i])));
    pack_half_lanes(narrow, false)
}

// ---------------------------------------------------------------------------------------------
// TCVTROWPS2PH{H,L} — tile row FP32 -> FP16, high / low word of each dword
// ---------------------------------------------------------------------------------------------

/// `TCVTROWPS2PHH` (spec section 12.6): convert row `row & 0xF` of tile `src` from `FP32`
/// to `FP16` (RNE via [`fp32_to_fp16_rne`]; input FP32 denormals produce FP16 zero, FP16
/// denormal outputs are permitted) into the HIGH word of each dword; the low words are
/// zeroed (`[ace-tile-instructions.TCVTROW.4]`). Gates as [`_tile_cvtrowd2ps`].
pub fn _tile_cvtrowps2phh(scope: &TileScope, src: TileId, row: u32) -> [u16; ZMM_WORD_LANES] {
    let _ = detect::has_amx_avx512; // family-C gate [DETECT.1-2]
    _tile_cvtrowps2phh_scalar(scope, src, row)
}

/// Portable `TCVTROWPS2PHH` oracle — section-12.6.3 with `variant == "H"`.
pub fn _tile_cvtrowps2phh_scalar(
    scope: &TileScope,
    src: TileId,
    row: u32,
) -> [u16; ZMM_WORD_LANES] {
    let dwords = read_row_dwords(scope, src, row);
    let narrow = core::array::from_fn(|i| fp32_to_fp16_rne(f32::from_bits(dwords[i])));
    pack_half_lanes(narrow, true)
}

/// `TCVTROWPS2PHL` (spec section 12.6): as [`_tile_cvtrowps2phh`] but into the LOW word of
/// each dword; the high words are zeroed (`[ace-tile-instructions.TCVTROW.5]`).
pub fn _tile_cvtrowps2phl(scope: &TileScope, src: TileId, row: u32) -> [u16; ZMM_WORD_LANES] {
    let _ = detect::has_amx_avx512; // family-C gate [DETECT.1-2]
    _tile_cvtrowps2phl_scalar(scope, src, row)
}

/// Portable `TCVTROWPS2PHL` oracle — section-12.6.3 with `variant == "L"`.
pub fn _tile_cvtrowps2phl_scalar(
    scope: &TileScope,
    src: TileId,
    row: u32,
) -> [u16; ZMM_WORD_LANES] {
    let dwords = read_row_dwords(scope, src, row);
    let narrow = core::array::from_fn(|i| fp32_to_fp16_rne(f32::from_bits(dwords[i])));
    pack_half_lanes(narrow, false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile::{TileConfig, _tile_loadconfig};

    /// Build a scope whose tile 0 row 2 holds the given 16 dwords.
    fn scope_with_row2(dwords: [u32; 16]) -> (TileScope, TileId) {
        let mut scope = _tile_loadconfig(&TileConfig::ace()).unwrap();
        let id = scope.tile(0).unwrap();
        let bytes = scope.tile_bytes_mut(id);
        for (i, d) in dwords.iter().enumerate() {
            bytes[2 * TILE_COLSB + 4 * i..2 * TILE_COLSB + 4 * i + 4]
                .copy_from_slice(&d.to_le_bytes());
        }
        (scope, id)
    }

    /// D2PS converts each INT32 exactly (small values) and under RNE beyond 2^24; the row
    /// index masks to 4 bits and never faults (spec sections 12.4.4 and 12.1.1).
    /// `tcvtrow::d2ps_rne_and_index_mask`
    #[test]
    fn d2ps_rne_and_index_mask() {
        let mut dwords = [0u32; 16];
        dwords[0] = 7i32 as u32;
        dwords[1] = (-3i32) as u32;
        dwords[2] = 0x0100_0001; // 16777217 = 2^24 + 1: RNE rounds to 2^24 (16777216.0)
        let (scope, id) = scope_with_row2(dwords);

        let out = _tile_cvtrowd2ps(&scope, id, 2);
        assert_eq!(out[0], 7.0);
        assert_eq!(out[1], -3.0);
        assert_eq!(out[2], 16_777_216.0, "2^24+1 rounds to even under RNE");
        assert_eq!(out[3], 0.0);

        // Specifier 18 & 0xF = 2: same row, no fault (spec section 12.1.1).
        assert_eq!(_tile_cvtrowd2ps(&scope, id, 18), out);
    }

    /// The H/L variants write disjoint word lanes with the complementary word zeroed
    /// (section-12.5.3/12.6.3 `pos`/`zeropos`), and BF16 obeys the mandated DAZ=1: an FP32
    /// denormal input yields BF16 zero (spec section 12.5.1).
    /// `tcvtrow::half_lane_layout_and_bf16_daz`
    #[test]
    fn half_lane_layout_and_bf16_daz() {
        let mut dwords = [0u32; 16];
        dwords[0] = 1.0f32.to_bits();
        dwords[1] = (-2.5f32).to_bits();
        dwords[2] = 0x0000_0001; // FP32 min denormal: DAZ=1 -> BF16 +0
        dwords[3] = 0x8000_0001; // negative denormal -> BF16 -0
        let (scope, id) = scope_with_row2(dwords);

        let h = _tile_cvtrowps2bf16h(&scope, id, 2);
        let l = _tile_cvtrowps2bf16l(&scope, id, 2);
        // H: word 2i+1 carries the BF16, word 2i is zero; L is the mirror.
        assert_eq!(h[1], 0x3F80, "1.0 -> BF16 0x3F80 in the high word");
        assert_eq!(h[0], 0, "H zeroes the low word");
        assert_eq!(l[0], 0x3F80, "1.0 -> BF16 0x3F80 in the low word");
        assert_eq!(l[1], 0, "L zeroes the high word");
        assert_eq!(h[3], 0xC020, "-2.5 -> BF16 0xC020");
        // DAZ=1 (spec section 12.5.1): denormal inputs flush to signed zero.
        assert_eq!(h[5], 0x0000, "+denormal -> +0 (DAZ=1)");
        assert_eq!(h[7], 0x8000, "-denormal -> -0 (DAZ=1, sign kept)");

        // PH forms share the layout.
        let ph = _tile_cvtrowps2phh(&scope, id, 2);
        let pl = _tile_cvtrowps2phl(&scope, id, 2);
        assert_eq!(ph[1], 0x3C00, "1.0 -> FP16 0x3C00 in the high word");
        assert_eq!(ph[0], 0);
        assert_eq!(pl[0], 0x3C00);
        assert_eq!(pl[3], 0, "L zeroes the high word of every dword");
        // Input FP32 denormal -> FP16 zero (spec section 12.6.1).
        assert_eq!(ph[5], 0x0000);
        assert_eq!(ph[7], 0x8000);
    }
}
