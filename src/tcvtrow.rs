//! ACE group-4 family C: tile-row converts.
//!
//! This module models the five family-C convert instructions. Each reads one addressed row
//! of a source tile, converts it element-wise to a narrower target format, and writes the
//! result into a destination ZMM vector; the source tile is left unchanged:
//!
//! * `TCVTROWD2PS` — [`_tile_tcvtrowd2ps`] converts a tile row of `INT32` elements to `FP32`
//!   (`[ace-tile-instructions.TCVTROW.1]`).
//! * `TCVTROWPS2BF16H` / `TCVTROWPS2BF16L` — [`_tile_tcvtrowps2bf16h`] /
//!   [`_tile_tcvtrowps2bf16l`] convert a tile row of `FP32` elements to `BF16` (via
//!   [`fp32_to_bf16_rne`]) into the high / low half-lanes respectively
//!   (`[ace-tile-instructions.TCVTROW.2]`, `[ace-tile-instructions.TCVTROW.3]`).
//! * `TCVTROWPS2PHH` / `TCVTROWPS2PHL` — [`_tile_tcvtrowps2phh`] / [`_tile_tcvtrowps2phl`]
//!   convert a tile row of `FP32` elements to `FP16` (via [`fp32_to_fp16_rne`]) into the high
//!   / low half-lanes respectively (`[ace-tile-instructions.TCVTROW.4]`,
//!   `[ace-tile-instructions.TCVTROW.5]`).
//!
//! # Disjoint half-lanes (INV-7)
//!
//! Each narrow result occupies one 16-bit word of a 32-bit destination slot. The `H` forms
//! write the *high* word of each slot (odd `u16` lane `2*i + 1`) and zero the low word; the
//! `L` forms write the *low* word (even `u16` lane `2*i`) and zero the high word. So `H` and
//! `L` touch DISJOINT sets of destination word lanes — odd vs even — and `H | L` over the same
//! source row tiles the full destination with no overlap
//! (`[ace-tile-instructions.TCVTROW.5-note]`).
//!
//! # Rounding oracle
//!
//! NaN / overflow follow the codec format rules of the reused rounders: [`fp32_to_bf16_rne`]
//! and [`fp32_to_fp16_rne`] both round IEEE-754 roundTiesToEven (RNE), and `INT32 -> FP32` is
//! the exact `i32 as f32` conversion (RNE on the 24-bit-plus values). The oracle is total for
//! in-domain rows — there is no runtime error path for a NaN / overflowing element.
//!
//! # Dispatch
//!
//! Each convert is a safe public dispatcher plus a cfg-free `_scalar` oracle (the primary
//! path, correct on every target). Per `[ace-tile-instructions.DISPATCH.1]` /
//! `[ace-tile-instructions.DETECT.1-2]` the family-C converts gate on AMX-AVX512 or
//! `ACE_VSN >= 1` ([`detect::has_amx_avx512`]). No native tile shim exists yet — the native
//! path is layer-3-blocked until Intel SDE gains ACE emulation (OQ-6, wired in phase 8) — so,
//! exactly as the oracle-only group-3 modules do, the dispatchers reference the detector to
//! mark the gate site and take the scalar oracle on every target.
//!
//! # OQ-8 (canonical width) surfaced here
//!
//! One canonical, largest palette-2 form per convert: a source row is up to `colsb <= 64`
//! bytes, i.e. up to [`ROW_FP32_LANES`] (`= 16`) four-byte `INT32`/`FP32` elements, and the
//! destination is one full 512-bit ZMM — [`ROW_FP32_LANES`] `f32` lanes for `D2PS`, or
//! [`ZMM_WORD_LANES`] (`= 32`) `u16` lanes for the narrowing converts. Rows shorter than the
//! canonical width zero-extend into the unused lanes, matching family B's fixed-width vector
//! convention.

use crate::detect;
use crate::fp8::{fp32_to_bf16_rne, fp32_to_fp16_rne};
use crate::tile::{TileId, TileScope};

/// Number of four-byte (`INT32`/`FP32`) elements in one canonical palette-2 tile row
/// (`colsb == 64` -> `64 / 4 == 16`); also the `f32` lane count of a `D2PS` destination ZMM.
pub const ROW_FP32_LANES: usize = 16;

/// Number of 16-bit (`BF16`/`FP16`) lanes in a 512-bit destination ZMM. Each of the
/// [`ROW_FP32_LANES`] source elements maps to one 32-bit destination slot = two `u16` lanes,
/// so `ZMM_WORD_LANES == 2 * ROW_FP32_LANES == 32`.
pub const ZMM_WORD_LANES: usize = 2 * ROW_FP32_LANES;

/// Read the addressed `row` of tile `src` as up to [`ROW_FP32_LANES`] little-endian four-byte
/// words, or [`None`] if `row` is outside the tile's configured `rows`. A row shorter than the
/// canonical width (`colsb < 64`) fills only its `colsb / 4` low lanes; the rest are zero.
fn read_row_dwords(scope: &TileScope, src: TileId, row: usize) -> Option<[u32; ROW_FP32_LANES]> {
    let (rows, colsb) = scope.tile_shape(src);
    // Typed row-index bound: an index outside the configured rows addresses no slot.
    if row >= rows as usize {
        return None;
    }
    let colsb = colsb as usize;
    let bytes = scope.tile_bytes_ref(src);
    let start = row * colsb;
    let lanes = (colsb / 4).min(ROW_FP32_LANES);
    let mut out = [0u32; ROW_FP32_LANES];
    for (lane, slot) in out.iter_mut().enumerate().take(lanes) {
        let off = start + lane * 4;
        *slot = u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
    }
    Some(out)
}

/// Pack [`ROW_FP32_LANES`] narrow results into a 512-bit ZMM half-lane layout. Result `i` is
/// written to `u16` lane `2 * i + 1` (the high word of slot `i`) when `high`, else to lane
/// `2 * i` (the low word); the complementary word of each slot is left zero, so the `H` and
/// `L` packings occupy DISJOINT lanes (INV-7, `[ace-tile-instructions.TCVTROW.5-note]`).
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

/// `TCVTROWD2PS` (`[ace-tile-instructions.TCVTROW.1]`): convert the addressed `row` of tile
/// `src` from `INT32` to `FP32`, or [`None`] if `row` is outside the configured `rows`.
///
/// Gates on AMX-AVX512 or `ACE_VSN >= 1` (`[ace-tile-instructions.DETECT.1-2]`,
/// `[ace-tile-instructions.DISPATCH.1]`); with no native shim yet (OQ-6) the detector marks
/// the gate site and the scalar oracle runs on every target.
pub fn _tile_tcvtrowd2ps(
    scope: &TileScope,
    src: TileId,
    row: usize,
) -> Option<[f32; ROW_FP32_LANES]> {
    let _ = detect::has_amx_avx512; // family-C gate: AMX-AVX512 or ACE_VSN>=1 [DETECT.1-2]
    _tile_tcvtrowd2ps_scalar(scope, src, row)
}

/// Portable `TCVTROWD2PS` oracle — the primary, always-correct path. Reinterprets each row
/// element as `i32` and converts to `f32` (exact, IEEE-754 roundTiesToEven for magnitudes
/// beyond 2^24); unused lanes are `0.0`.
pub fn _tile_tcvtrowd2ps_scalar(
    scope: &TileScope,
    src: TileId,
    row: usize,
) -> Option<[f32; ROW_FP32_LANES]> {
    let dwords = read_row_dwords(scope, src, row)?;
    Some(core::array::from_fn(|i| dwords[i] as i32 as f32))
}

// ---------------------------------------------------------------------------------------------
// TCVTROWPS2BF16{H,L} — tile row FP32 -> BF16, high / low half-lanes
// ---------------------------------------------------------------------------------------------

/// `TCVTROWPS2BF16H` (`[ace-tile-instructions.TCVTROW.2]`): convert the addressed `row` of
/// tile `src` from `FP32` to `BF16` (RNE, [`fp32_to_bf16_rne`]) into the HIGH half-lanes, or
/// [`None`] if `row` is out of range. Gates as [`_tile_tcvtrowd2ps`].
pub fn _tile_tcvtrowps2bf16h(
    scope: &TileScope,
    src: TileId,
    row: usize,
) -> Option<[u16; ZMM_WORD_LANES]> {
    let _ = detect::has_amx_avx512; // family-C gate: AMX-AVX512 or ACE_VSN>=1 [DETECT.1-2]
    _tile_tcvtrowps2bf16h_scalar(scope, src, row)
}

/// Portable `TCVTROWPS2BF16H` oracle — converts each row element to `BF16` (RNE) and packs it
/// into the high word of its slot; low words stay zero.
pub fn _tile_tcvtrowps2bf16h_scalar(
    scope: &TileScope,
    src: TileId,
    row: usize,
) -> Option<[u16; ZMM_WORD_LANES]> {
    let dwords = read_row_dwords(scope, src, row)?;
    let narrow = core::array::from_fn(|i| fp32_to_bf16_rne(f32::from_bits(dwords[i])));
    Some(pack_half_lanes(narrow, true))
}

/// `TCVTROWPS2BF16L` (`[ace-tile-instructions.TCVTROW.3]`): as [`_tile_tcvtrowps2bf16h`] but
/// into the LOW half-lanes (disjoint from the high form, INV-7).
pub fn _tile_tcvtrowps2bf16l(
    scope: &TileScope,
    src: TileId,
    row: usize,
) -> Option<[u16; ZMM_WORD_LANES]> {
    let _ = detect::has_amx_avx512; // family-C gate: AMX-AVX512 or ACE_VSN>=1 [DETECT.1-2]
    _tile_tcvtrowps2bf16l_scalar(scope, src, row)
}

/// Portable `TCVTROWPS2BF16L` oracle — converts each row element to `BF16` (RNE) and packs it
/// into the low word of its slot; high words stay zero.
pub fn _tile_tcvtrowps2bf16l_scalar(
    scope: &TileScope,
    src: TileId,
    row: usize,
) -> Option<[u16; ZMM_WORD_LANES]> {
    let dwords = read_row_dwords(scope, src, row)?;
    let narrow = core::array::from_fn(|i| fp32_to_bf16_rne(f32::from_bits(dwords[i])));
    Some(pack_half_lanes(narrow, false))
}

// ---------------------------------------------------------------------------------------------
// TCVTROWPS2PH{H,L} — tile row FP32 -> FP16, high / low half-lanes
// ---------------------------------------------------------------------------------------------

/// `TCVTROWPS2PHH` (`[ace-tile-instructions.TCVTROW.4]`): convert the addressed `row` of tile
/// `src` from `FP32` to `FP16` (RNE, [`fp32_to_fp16_rne`]) into the HIGH half-lanes, or
/// [`None`] if `row` is out of range. NaN / overflow follow the FP16 codec format rules.
/// Gates as [`_tile_tcvtrowd2ps`].
pub fn _tile_tcvtrowps2phh(
    scope: &TileScope,
    src: TileId,
    row: usize,
) -> Option<[u16; ZMM_WORD_LANES]> {
    let _ = detect::has_amx_avx512; // family-C gate: AMX-AVX512 or ACE_VSN>=1 [DETECT.1-2]
    _tile_tcvtrowps2phh_scalar(scope, src, row)
}

/// Portable `TCVTROWPS2PHH` oracle — converts each row element to `FP16` (RNE) and packs it
/// into the high word of its slot; low words stay zero.
pub fn _tile_tcvtrowps2phh_scalar(
    scope: &TileScope,
    src: TileId,
    row: usize,
) -> Option<[u16; ZMM_WORD_LANES]> {
    let dwords = read_row_dwords(scope, src, row)?;
    let narrow = core::array::from_fn(|i| fp32_to_fp16_rne(f32::from_bits(dwords[i])));
    Some(pack_half_lanes(narrow, true))
}

/// `TCVTROWPS2PHL` (`[ace-tile-instructions.TCVTROW.5]`): as [`_tile_tcvtrowps2phh`] but into
/// the LOW half-lanes (disjoint from the high form, INV-7,
/// `[ace-tile-instructions.TCVTROW.5-note]`).
pub fn _tile_tcvtrowps2phl(
    scope: &TileScope,
    src: TileId,
    row: usize,
) -> Option<[u16; ZMM_WORD_LANES]> {
    let _ = detect::has_amx_avx512; // family-C gate: AMX-AVX512 or ACE_VSN>=1 [DETECT.1-2]
    _tile_tcvtrowps2phl_scalar(scope, src, row)
}

/// Portable `TCVTROWPS2PHL` oracle — converts each row element to `FP16` (RNE) and packs it
/// into the low word of its slot; high words stay zero.
pub fn _tile_tcvtrowps2phl_scalar(
    scope: &TileScope,
    src: TileId,
    row: usize,
) -> Option<[u16; ZMM_WORD_LANES]> {
    let dwords = read_row_dwords(scope, src, row)?;
    let narrow = core::array::from_fn(|i| fp32_to_fp16_rne(f32::from_bits(dwords[i])));
    Some(pack_half_lanes(narrow, false))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile::{_tile_loadconfig, TileConfig};

    /// A palette-2 scope with tile 0 = 1 row x 64 colsb (16 four-byte elements), seeded with
    /// `dwords` (little-endian) in its low lanes and zero elsewhere.
    fn seeded_row(dwords: &[u32]) -> TileScope {
        let cfg = TileConfig {
            palette_id: 2,
            rows: [1, 0, 0, 0, 0, 0, 0, 0],
            colsb: [64, 0, 0, 0, 0, 0, 0, 0],
        };
        let mut scope = _tile_loadconfig(&cfg).unwrap();
        let t0 = scope.tile(0).unwrap();
        let bytes = scope.tile_bytes_mut(t0);
        for (i, &d) in dwords.iter().enumerate() {
            bytes[i * 4..i * 4 + 4].copy_from_slice(&d.to_le_bytes());
        }
        scope
    }

    /// `TCVTROWD2PS` converts a row of INT32 to FP32, honouring sign and IEEE-754
    /// roundTiesToEven for magnitudes beyond 2^24; an out-of-range row is rejected.
    /// `tcvtrow::d2ps_int32_to_fp32`
    #[test]
    fn d2ps_int32_to_fp32() {
        let scope = seeded_row(&[
            0,                    // -> 0.0
            1i32 as u32,          // -> 1.0
            (-3i32) as u32,       // signed: -> -3.0 (an unsigned model gives ~4.29e9)
            16_777_217i32 as u32, // 2^24 + 1: not exactly representable -> 16_777_216.0 (RNE)
        ]);
        let t0 = scope.tile(0).unwrap();

        let out = _tile_tcvtrowd2ps(&scope, t0, 0).expect("row 0 is configured");
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 1.0);
        assert_eq!(out[2], -3.0, "row element is INT32 (signed), not UINT32");
        assert_eq!(
            out[3], 16_777_216.0,
            "2^24+1 rounds to even (16_777_216.0) under IEEE-754 RNE"
        );
        assert!(
            out[4..].iter().all(|&f| f == 0.0),
            "unused lanes zero-extend"
        );

        // Typed row-index bound: tile 0 has a single row, so row 1 addresses no slot.
        assert_eq!(_tile_tcvtrowd2ps(&scope, t0, 1), None);
    }

    /// `TCVTROWPS2BF16H` / `TCVTROWPS2BF16L` convert FP32 -> BF16 (RNE) into the high / low
    /// word of each 32-bit destination slot. The expected bytes are hand-derived, and the H
    /// and L forms place the SAME converted value in different lanes.
    /// `tcvtrow::ps2bf16_high_low_halves`
    #[test]
    fn ps2bf16_high_low_halves() {
        // v0 = 1.0 -> BF16 0x3F80 (exact). v1 has low-16 mantissa 0x8001 > half, so RNE
        // rounds the kept BF16 mantissa UP to 0x404A (truncation would keep 0x4049).
        let v0 = 0x3F80_0000u32;
        let v1 = 0x4049_8001u32;
        let scope = seeded_row(&[v0, v1]);
        let t0 = scope.tile(0).unwrap();

        let h = _tile_tcvtrowps2bf16h(&scope, t0, 0).unwrap();
        let l = _tile_tcvtrowps2bf16l(&scope, t0, 0).unwrap();

        // HIGH form: converted value in the high word (odd lane), low word zero.
        assert_eq!(h[1], 0x3F80);
        assert_eq!(h[0], 0);
        assert_eq!(h[3], 0x404A, "RNE rounds up (truncation would give 0x4049)");
        assert_eq!(h[2], 0);

        // LOW form: converted value in the low word (even lane), high word zero.
        assert_eq!(l[0], 0x3F80);
        assert_eq!(l[1], 0);
        assert_eq!(l[2], 0x404A);
        assert_eq!(l[3], 0);
    }

    /// `TCVTROWPS2PHH` / `TCVTROWPS2PHL` convert FP32 -> FP16 (RNE) into the high / low word of
    /// each slot. Hand-derived FP16 encodings; H and L place the same value in different lanes.
    /// `tcvtrow::ps2ph_high_low_halves`
    #[test]
    fn ps2ph_high_low_halves() {
        // 1.0 -> FP16 0x3C00; -2.0 -> FP16 0xC000 (exact encodings).
        let v0 = 1.0f32.to_bits();
        let v1 = (-2.0f32).to_bits();
        let scope = seeded_row(&[v0, v1]);
        let t0 = scope.tile(0).unwrap();

        let h = _tile_tcvtrowps2phh(&scope, t0, 0).unwrap();
        let l = _tile_tcvtrowps2phl(&scope, t0, 0).unwrap();

        // HIGH form: high word carries the value, low word zero.
        assert_eq!(h[1], 0x3C00);
        assert_eq!(h[0], 0);
        assert_eq!(h[3], 0xC000);
        assert_eq!(h[2], 0);

        // LOW form: low word carries the value, high word zero.
        assert_eq!(l[0], 0x3C00);
        assert_eq!(l[1], 0);
        assert_eq!(l[2], 0xC000);
        assert_eq!(l[3], 0);
    }

    /// INV-7: the H and L converts write DISJOINT destination halves
    /// (`[ace-tile-instructions.TCVTROW.5-note]`, `[ace-tile-instructions.TESTING.3]`). With a
    /// full row of distinct nonzero values, the H form touches only odd word lanes and the L
    /// form only even word lanes, so no lane is written by both and `H | L` tiles the whole
    /// 512-bit destination.
    /// `tcvtrow::h_l_disjoint_halves`
    #[test]
    fn h_l_disjoint_halves() {
        // 16 distinct nonzero FP32 values (1.0..=16.0) — each converts to a nonzero narrow
        // result, so a zero lane can only come from the disjoint-half zeroing, not the value.
        let dwords: [u32; ROW_FP32_LANES] = core::array::from_fn(|i| (i as f32 + 1.0).to_bits());
        let scope = seeded_row(&dwords);
        let t0 = scope.tile(0).unwrap();

        for (h, l) in [
            (
                _tile_tcvtrowps2bf16h(&scope, t0, 0).unwrap(),
                _tile_tcvtrowps2bf16l(&scope, t0, 0).unwrap(),
            ),
            (
                _tile_tcvtrowps2phh(&scope, t0, 0).unwrap(),
                _tile_tcvtrowps2phl(&scope, t0, 0).unwrap(),
            ),
        ] {
            for lane in 0..ZMM_WORD_LANES {
                // Disjoint: no lane is written by both H and L.
                assert!(
                    !(h[lane] != 0 && l[lane] != 0),
                    "lane {lane} written by both H and L"
                );
                // Union tiles the full destination: every lane is nonzero in exactly one form.
                assert_ne!(
                    h[lane] | l[lane],
                    0,
                    "lane {lane} left unwritten by H and L"
                );
            }
            // H occupies only the odd (high) word lanes; L only the even (low) word lanes.
            for even in (0..ZMM_WORD_LANES).step_by(2) {
                assert_eq!(h[even], 0, "H must not write even lane {even}");
                assert_eq!(l[even + 1], 0, "L must not write odd lane {}", even + 1);
            }
        }
    }

    /// Hand-computed known-value pins per convert, independent of the implementation
    /// (`[ace-tile-instructions.TESTING.4]`). The narrowing rounders implement IEEE-754
    /// roundTiesToEven; the differential tiebreaker is unavailable here, so each pin is
    /// grounded against roundTiesToEven and chosen so its expected result DIFFERS under a
    /// wrong model (truncation, or round-half-away). Placement pins distinguish the H half
    /// from the L half.
    /// `tcvtrow::convert_known_value_pins`
    #[test]
    fn convert_known_value_pins() {
        // --- D2PS: signed INT32 -> FP32. ---
        let d2ps = seeded_row(&[(-3i32) as u32, 7i32 as u32]);
        let d0 = d2ps.tile(0).unwrap();
        let ps = _tile_tcvtrowd2ps(&d2ps, d0, 0).unwrap();
        assert_eq!(ps[0], -3.0);
        assert_eq!(ps[1], 7.0);

        // --- BF16 RNE discriminators (kept mantissa = top 16 bits of the FP32 pattern). ---
        // 0x4049_8001: discarded low-16 = 0x8001 > half -> round UP to 0x404A (truncation
        //   gives 0x4049).
        // 0x3F80_8000: exact half, kept mantissa (0x3F80) is EVEN -> ties-to-even rounds DOWN
        //   to 0x3F80 (round-half-away would give 0x3F81).
        // 0x3F81_8000: exact half, kept mantissa (0x3F81) is ODD -> ties-to-even rounds UP to
        //   0x3F82.
        let bf = seeded_row(&[0x4049_8001, 0x3F80_8000, 0x3F81_8000]);
        let b0 = bf.tile(0).unwrap();
        let bh = _tile_tcvtrowps2bf16h(&bf, b0, 0).unwrap();
        let bl = _tile_tcvtrowps2bf16l(&bf, b0, 0).unwrap();
        assert_eq!(bl[0], 0x404A, "RNE rounds above-half UP, not truncation");
        assert_eq!(
            bl[2], 0x3F80,
            "tie to even (even kept) rounds down, not half-away"
        );
        assert_eq!(bl[4], 0x3F82, "tie to even (odd kept) rounds up");
        // Same values, high half: identical encodings shifted into the odd lanes.
        assert_eq!(bh[1], 0x404A);
        assert_eq!(bh[3], 0x3F80);
        assert_eq!(bh[5], 0x3F82);
        // Placement discriminator: the L result is in the even lane, NOT the odd lane.
        assert_eq!(bl[1], 0, "L writes the low (even) word only");
        assert_eq!(bh[0], 0, "H writes the high (odd) word only");

        // --- FP16 RNE discriminator. ---
        // 0x3F80_1000 = 1.0 + 2^-11: the 11th mantissa bit is an exact half and the kept FP16
        // mantissa is EVEN, so ties-to-even rounds DOWN to 0x3C00 (round-half-away -> 0x3C01).
        let ph = seeded_row(&[0x3F80_1000, (-2.0f32).to_bits()]);
        let p0 = ph.tile(0).unwrap();
        let phh = _tile_tcvtrowps2phh(&ph, p0, 0).unwrap();
        let phl = _tile_tcvtrowps2phl(&ph, p0, 0).unwrap();
        assert_eq!(
            phl[0], 0x3C00,
            "FP16 tie to even rounds down, not half-away (0x3C01)"
        );
        assert_eq!(phl[2], 0xC000, "-2.0 -> FP16 0xC000");
        assert_eq!(
            phh[1], 0x3C00,
            "high half carries the same encoding in the odd lane"
        );
        assert_eq!(phh[3], 0xC000);
        assert_eq!(phh[0], 0, "H leaves the even (low) word zero");
        assert_eq!(phl[1], 0, "L leaves the odd (high) word zero");
    }

    /// The source tile is unchanged by a convert, and an out-of-range row is rejected across
    /// all five converts (persistenceState + errorPath).
    #[test]
    fn source_tile_unchanged_and_row_bound() {
        let scope = seeded_row(&[1.0f32.to_bits(), 2.0f32.to_bits()]);
        let t0 = scope.tile(0).unwrap();
        let before = scope.tile_bytes_ref(t0).to_vec();

        let _ = _tile_tcvtrowd2ps(&scope, t0, 0).unwrap();
        let _ = _tile_tcvtrowps2bf16h(&scope, t0, 0).unwrap();
        let _ = _tile_tcvtrowps2phl(&scope, t0, 0).unwrap();
        assert_eq!(
            scope.tile_bytes_ref(t0),
            &before[..],
            "converts do not mutate the tile"
        );

        // rows == 1: row 1 is the first out-of-range index for every convert.
        assert_eq!(_tile_tcvtrowd2ps(&scope, t0, 1), None);
        assert_eq!(_tile_tcvtrowps2bf16h(&scope, t0, 1), None);
        assert_eq!(_tile_tcvtrowps2bf16l(&scope, t0, 1), None);
        assert_eq!(_tile_tcvtrowps2phh(&scope, t0, 1), None);
        assert_eq!(_tile_tcvtrowps2phl(&scope, t0, 1), None);
    }

    /// System-as-a-whole wiring check: seed a tile row, convert it through the public
    /// dispatchers, and confirm the H/L halves land in disjoint lanes end to end. Prints the
    /// observable low lanes and the gate helper the family reads.
    #[test]
    fn end_to_end_convert_and_gate() {
        // D2PS reads the row as INT32; the FP32-narrowing converts read the SAME row bytes as
        // FP32. Seed the row with the bit patterns of 1.0/2.0/-4.0 so the FP32 converts see
        // those floats, and pin D2PS against the row's INT32 reinterpretation.
        let src_f32 = [1.0f32, 2.0, -4.0];
        let scope = seeded_row(&src_f32.map(f32::to_bits));
        let t0 = scope.tile(0).unwrap();

        let ps = _tile_tcvtrowd2ps(&scope, t0, 0).unwrap();
        let h = _tile_tcvtrowps2bf16h(&scope, t0, 0).unwrap();
        let l = _tile_tcvtrowps2bf16l(&scope, t0, 0).unwrap();

        // H writes odd lanes, L writes even lanes: each picks the value from the correct half
        // for the first source element (1.0 -> BF16 0x3F80).
        let h_slot0 = h[1];
        let l_slot0 = l[0];
        println!(
            "E2E d2ps[0..3]={:?} bf16_high_slot0=0x{h_slot0:04X} bf16_low_slot0=0x{l_slot0:04X}",
            &ps[..3]
        );
        println!(
            "E2E detect has_amx_avx512={} has_ace={}",
            crate::detect::has_amx_avx512(),
            crate::detect::has_ace(),
        );
        // D2PS sees the row as INT32 (the raw bit patterns), FP32-narrowing sees it as FP32.
        let expected_d2ps: [f32; 3] = src_f32.map(|f| f.to_bits() as i32 as f32);
        assert_eq!(ps[..3], expected_d2ps);
        assert_eq!(h_slot0, 0x3F80, "1.0 -> BF16 0x3F80 in the HIGH word");
        assert_eq!(l_slot0, 0x3F80, "1.0 -> BF16 0x3F80 in the LOW word");
        assert_eq!(h[0], 0, "H leaves the low word of slot 0 zero");
        assert_eq!(l[1], 0, "L leaves the high word of slot 0 zero");
    }
}

/// Layer-4 differential (family C). The tile-row converts are intrinsic-reachable, so under
/// `feature="native"` on x86_64 with the family-C gate detected the native `TCVTROWD2PS` shim
/// must equal the scalar oracle bit-for-bit (`[ace-tile-instructions.TESTING.1]`); it LIGHTS UP
/// under Intel SDE. Returns [`quickcheck::TestResult::discard`] — never `from_bool(false)` —
/// when the native path is unavailable. `TCVTROWD2PS` is the representative convert; the four
/// BF16/FP16 H/L forms share the identical posture and native shims.
#[cfg(test)]
mod differential {
    #![cfg_attr(
        not(all(target_arch = "x86_64", feature = "native")),
        allow(unused_imports, dead_code)
    )]
    use super::*;
    use crate::tile::{_tile_loadconfig, TileConfig};
    use quickcheck::{quickcheck, Arbitrary, Gen, TestResult};

    /// A single 64-byte INT32 tile row (`rows=1`, `colsb=64` = 16 dwords).
    #[derive(Clone, Debug)]
    struct Row {
        data: [u8; 64],
    }

    impl Arbitrary for Row {
        fn arbitrary(g: &mut Gen) -> Self {
            Row {
                data: core::array::from_fn(|_| u8::arbitrary(g)),
            }
        }
    }

    quickcheck! {
        /// All five family-C converts (INT32->FP32, FP32->BF16 H/L, FP32->FP16 H/L), native vs
        /// oracle bit-for-bit over one 64-byte tile row. Discards off-tile; lights up under SDE.
        fn prop_native_matches_oracle(row: Row) -> TestResult {
            #[cfg(all(target_arch = "x86_64", feature = "native"))]
            {
                if detect::has_amx_avx512() {
                    use crate::native;
                    let config = TileConfig {
                        palette_id: 2,
                        rows: [1, 0, 0, 0, 0, 0, 0, 0],
                        colsb: [64, 0, 0, 0, 0, 0, 0, 0],
                    };
                    let mut scope = _tile_loadconfig(&config).expect("valid descriptor");
                    let src = scope.tile(0).unwrap();
                    scope.tile_bytes_mut(src).copy_from_slice(&row.data);
                    let cfg = native::encode_tilecfg(2, &config.rows, &config.colsb);
                    // SAFETY: has_amx_avx512() confirmed the family-C gate + tile XSAVE state.
                    let d2ps = unsafe { native::tcvtrowd2ps_hw(&cfg, &row.data, 0) }
                        == _tile_tcvtrowd2ps_scalar(&scope, src, 0).unwrap();
                    let bf16h = unsafe { native::tcvtrowps2bf16h_hw(&cfg, &row.data, 0) }
                        == _tile_tcvtrowps2bf16h_scalar(&scope, src, 0).unwrap();
                    let bf16l = unsafe { native::tcvtrowps2bf16l_hw(&cfg, &row.data, 0) }
                        == _tile_tcvtrowps2bf16l_scalar(&scope, src, 0).unwrap();
                    let phh = unsafe { native::tcvtrowps2phh_hw(&cfg, &row.data, 0) }
                        == _tile_tcvtrowps2phh_scalar(&scope, src, 0).unwrap();
                    let phl = unsafe { native::tcvtrowps2phl_hw(&cfg, &row.data, 0) }
                        == _tile_tcvtrowps2phl_scalar(&scope, src, 0).unwrap();
                    return TestResult::from_bool(d2ps && bf16h && bf16l && phh && phl);
                }
            }
            let _ = &row;
            TestResult::discard()
        }
    }
}
