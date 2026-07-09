//! ACE group-4 family B: tile <-> vector row/column moves.
//!
//! This module models the two family-B move instructions, each with a *read* form (tile ->
//! ZMM vector) and a *write* form (ZMM vector -> tile):
//!
//! * `TILEMOVROW` — [`_tile_movrow`] extracts one addressed row of a tile into a ZMM vector
//!   (`[ace-tile-instructions.TILE_MOVE.1]`); [`_tile_movrow_write`] inserts a ZMM vector
//!   back into one addressed row.
//! * `TILEMOVCOL` — [`_tile_movcol`] extracts one addressed column of a tile into a ZMM
//!   vector (`[ace-tile-instructions.TILE_MOVE.2]`); [`_tile_movcol_write`] inserts a ZMM
//!   vector back into one addressed column.
//!
//! Writing a row then reading it back returns the original row (INV-6,
//! `[ace-tile-instructions.TILE_MOVE.1-1]`), and a move touches exactly the addressed
//! row/column, leaving the rest of the tile unchanged.
//!
//! # Row / column index bound
//!
//! The row/column index is bounded at the typed boundary by the tile's configured
//! [`TileScope::tile_shape`]: a row index must be `< rows`, a column index must be `<
//! colsb`. An out-of-range index addresses no slot — the read forms return [`None`] and the
//! write forms return [`None`] having mutated no tile state.
//!
//! # Dispatch
//!
//! Each move is a safe public dispatcher plus a cfg-free `_scalar` oracle (the primary path,
//! correct on every target). Per `[ace-tile-instructions.DISPATCH.1]` /
//! `[ace-tile-instructions.DETECT.1-2]` the *read* forms gate on AMX-AVX512 or `ACE_VSN >= 1`
//! ([`detect::has_amx_avx512`]); the *write* forms are `ACE`-only and gate on full `ACE`
//! ([`detect::has_ace`], `[ace-tile-instructions.DETECT.1-3]`). No native tile shim exists yet
//! — the write forms are layer-3-blocked until Intel SDE gains ACE emulation (OQ-6, wired in
//! phase 8) — so, exactly as the oracle-only group-3 modules do, the dispatchers reference the
//! detector to mark the gate site and take the scalar oracle on every target.
//!
//! # OQ-8 (canonical width) surfaced here
//!
//! One canonical, largest palette-2 form per move: the vector is a full 512-bit ZMM modelled
//! as [`ZMM_BYTES`] (`= 64`) raw bytes ([`u8; 64`]). A row occupies its first `colsb` (`<=
//! 64`) bytes; a column occupies its first `rows` (`<= 16`) bytes; the unused lanes are zero.
//! The `[_; 64]` / `[_; 32]` widths named in the design are placeholders: the element-typed
//! views (e.g. `[u16; 32]`) the later outer-product families use are the same 512 bits
//! reinterpreted, pending OQ-8's final tile-shape / vector-width decision.

use crate::detect;
use crate::tile::{TileId, TileScope};

/// A ZMM vector modelled as raw bytes: one canonical 512-bit lane group (OQ-8). A row uses
/// the first `colsb` bytes; a column uses the first `rows` bytes; the rest are zero.
pub const ZMM_BYTES: usize = 64;

// ---------------------------------------------------------------------------------------------
// Read forms (tile -> ZMM vector)
// ---------------------------------------------------------------------------------------------

/// `TILEMOVROW` (read form): extract the addressed `row` of tile `src` into a ZMM vector, or
/// [`None`] if `row` is outside the tile's configured `rows`
/// (`[ace-tile-instructions.TILE_MOVE.1]`).
///
/// The read form gates on AMX-AVX512 or `ACE_VSN >= 1` (`[ace-tile-instructions.DETECT.1-2]`,
/// `[ace-tile-instructions.DISPATCH.1]`); with no native shim yet (OQ-6) the detector marks
/// the gate site and the scalar oracle runs on every target.
pub fn _tile_movrow(scope: &TileScope, src: TileId, row: usize) -> Option<[u8; ZMM_BYTES]> {
    let _ = detect::has_amx_avx512; // read-form gate: AMX-AVX512 or ACE_VSN>=1 [DETECT.1-2]
    _tile_movrow_scalar(scope, src, row)
}

/// Portable `TILEMOVROW` (read form) oracle — the primary, always-correct path. Copies the
/// `colsb` bytes of row `row` into the low lanes of the returned vector, zero-extended.
pub fn _tile_movrow_scalar(scope: &TileScope, src: TileId, row: usize) -> Option<[u8; ZMM_BYTES]> {
    let (rows, colsb) = scope.tile_shape(src);
    // Typed row-index bound: an index outside the configured rows addresses no slot.
    if row >= rows as usize {
        return None;
    }
    let colsb = colsb as usize;
    let bytes = scope.tile_bytes_ref(src);
    let start = row * colsb;
    let mut out = [0u8; ZMM_BYTES];
    out[..colsb].copy_from_slice(&bytes[start..start + colsb]);
    Some(out)
}

/// `TILEMOVCOL` (read form): extract the addressed `col` of tile `src` into a ZMM vector (one
/// byte per configured row), or [`None`] if `col` is outside the tile's configured `colsb`
/// (`[ace-tile-instructions.TILE_MOVE.2]`).
///
/// Gates identically to [`_tile_movrow`] (`[ace-tile-instructions.DETECT.1-2]`,
/// `[ace-tile-instructions.DISPATCH.1]`).
pub fn _tile_movcol(scope: &TileScope, src: TileId, col: usize) -> Option<[u8; ZMM_BYTES]> {
    let _ = detect::has_amx_avx512; // read-form gate: AMX-AVX512 or ACE_VSN>=1 [DETECT.1-2]
    _tile_movcol_scalar(scope, src, col)
}

/// Portable `TILEMOVCOL` (read form) oracle — copies the byte at column `col` of each of the
/// `rows` configured rows into the low lanes of the returned vector, zero-extended.
pub fn _tile_movcol_scalar(scope: &TileScope, src: TileId, col: usize) -> Option<[u8; ZMM_BYTES]> {
    let (rows, colsb) = scope.tile_shape(src);
    let colsb = colsb as usize;
    // Typed column-index bound: an index outside the configured colsb addresses no slot.
    if col >= colsb {
        return None;
    }
    let bytes = scope.tile_bytes_ref(src);
    let mut out = [0u8; ZMM_BYTES];
    for (r, slot) in out.iter_mut().enumerate().take(rows as usize) {
        *slot = bytes[r * colsb + col];
    }
    Some(out)
}

// ---------------------------------------------------------------------------------------------
// Write forms (ZMM vector -> tile)
// ---------------------------------------------------------------------------------------------

/// `TILEMOVROW` (write form): insert the low `colsb` lanes of `val` into the addressed `row`
/// of tile `dst`, leaving every other row unchanged. Returns [`None`] with no tile state
/// changed if `row` is outside the configured `rows` (`[ace-tile-instructions.TILE_MOVE.1]`).
///
/// The write form is `ACE`-only and gates on full `ACE` (`[ace-tile-instructions.DETECT.1-3]`,
/// `[ace-tile-instructions.DISPATCH.1]`). It is layer-3-blocked until SDE ACE lands (OQ-6),
/// so the detector marks the gate site and the scalar oracle runs on every target.
pub fn _tile_movrow_write(
    scope: &mut TileScope,
    dst: TileId,
    row: usize,
    val: [u8; ZMM_BYTES],
) -> Option<()> {
    let _ = detect::has_ace; // write-form gate: full ACE [DETECT.1-3]
    _tile_movrow_write_scalar(scope, dst, row, val)
}

/// Portable `TILEMOVROW` (write form) oracle — writes the low `colsb` lanes of `val` into the
/// addressed row after the typed row-index bound check.
pub fn _tile_movrow_write_scalar(
    scope: &mut TileScope,
    dst: TileId,
    row: usize,
    val: [u8; ZMM_BYTES],
) -> Option<()> {
    let (rows, colsb) = scope.tile_shape(dst);
    if row >= rows as usize {
        return None;
    }
    let colsb = colsb as usize;
    let start = row * colsb;
    scope.tile_bytes_mut(dst)[start..start + colsb].copy_from_slice(&val[..colsb]);
    Some(())
}

/// `TILEMOVCOL` (write form): insert the low `rows` lanes of `val` into the addressed `col`
/// of tile `dst`, leaving every other column unchanged. Returns [`None`] with no tile state
/// changed if `col` is outside the configured `colsb` (`[ace-tile-instructions.TILE_MOVE.2]`).
///
/// Gates identically to [`_tile_movrow_write`] (`[ace-tile-instructions.DETECT.1-3]`,
/// `[ace-tile-instructions.DISPATCH.1]`).
pub fn _tile_movcol_write(
    scope: &mut TileScope,
    dst: TileId,
    col: usize,
    val: [u8; ZMM_BYTES],
) -> Option<()> {
    let _ = detect::has_ace; // write-form gate: full ACE [DETECT.1-3]
    _tile_movcol_write_scalar(scope, dst, col, val)
}

/// Portable `TILEMOVCOL` (write form) oracle — writes the low `rows` lanes of `val` into the
/// addressed column after the typed column-index bound check.
pub fn _tile_movcol_write_scalar(
    scope: &mut TileScope,
    dst: TileId,
    col: usize,
    val: [u8; ZMM_BYTES],
) -> Option<()> {
    let (rows, colsb) = scope.tile_shape(dst);
    let colsb = colsb as usize;
    if col >= colsb {
        return None;
    }
    let bytes = scope.tile_bytes_mut(dst);
    for (r, &lane) in val.iter().enumerate().take(rows as usize) {
        bytes[r * colsb + col] = lane;
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile::{_tile_loadconfig, TileConfig};

    /// A palette-2 descriptor with tile 0 = 3 rows x 4 colsb (rows != colsb, so a
    /// row/column transposition is observable), rest empty.
    fn config_3x4() -> TileConfig {
        TileConfig {
            palette_id: 2,
            rows: [3, 0, 0, 0, 0, 0, 0, 0],
            colsb: [4, 0, 0, 0, 0, 0, 0, 0],
        }
    }

    /// Seed tile 0 of a fresh 3x4 scope with a recognisable row-major pattern where every
    /// byte is distinct and encodes its (row, col): byte = 0x10 * (row + 1) + col.
    ///
    /// ```text
    ///   row0: 10 11 12 13
    ///   row1: 20 21 22 23
    ///   row2: 30 31 32 33
    /// ```
    fn seeded_3x4() -> TileScope {
        let mut scope = _tile_loadconfig(&config_3x4()).unwrap();
        let t0 = scope.tile(0).unwrap();
        let (rows, colsb) = scope.tile_shape(t0);
        let bytes = scope.tile_bytes_mut(t0);
        for r in 0..rows as usize {
            for c in 0..colsb as usize {
                bytes[r * colsb as usize + c] = 0x10 * (r as u8 + 1) + c as u8;
            }
        }
        scope
    }

    /// TILEMOVROW extracts the correct single row and nothing else; a transposed model
    /// (reading a column) would return different bytes.
    /// `tile_move::movrow_moves_correct_row`
    #[test]
    fn movrow_moves_correct_row() {
        let scope = seeded_3x4();
        let t0 = scope.tile(0).unwrap();

        let mut expected = [0u8; ZMM_BYTES];
        expected[..4].copy_from_slice(&[0x20, 0x21, 0x22, 0x23]); // row 1, zero-extended
        assert_eq!(_tile_movrow(&scope, t0, 1), Some(expected));

        // Discriminator: the correct row-1 read is NOT the column-1 read (a transposed
        // implementation returns [0x11, 0x21, 0x31, ...] here).
        let mut col1 = [0u8; ZMM_BYTES];
        col1[..3].copy_from_slice(&[0x11, 0x21, 0x31]);
        assert_ne!(_tile_movrow(&scope, t0, 1), Some(col1));
    }

    /// TILEMOVCOL extracts the correct single column (one byte per row), zero-extended, and
    /// differs from the same-index row read.
    /// `tile_move::movcol_moves_correct_column`
    #[test]
    fn movcol_moves_correct_column() {
        let scope = seeded_3x4();
        let t0 = scope.tile(0).unwrap();

        let mut expected = [0u8; ZMM_BYTES];
        expected[..3].copy_from_slice(&[0x12, 0x22, 0x32]); // column 2, one byte per row
        assert_eq!(_tile_movcol(&scope, t0, 2), Some(expected));

        // Discriminator: column 2 differs from row 2 ([0x30, 0x31, 0x32, 0x33]).
        let mut row2 = [0u8; ZMM_BYTES];
        row2[..4].copy_from_slice(&[0x30, 0x31, 0x32, 0x33]);
        assert_ne!(_tile_movcol(&scope, t0, 2), Some(row2));
    }

    /// INV-6: TILEMOVROW (write) then read-back returns the original row, and the write
    /// leaves every other row unchanged (`[ace-tile-instructions.TILE_MOVE.1-1]`). The
    /// original row is distinct from any column, so a transposed/wrong-index write would
    /// fail the round-trip.
    /// `tile_move::movrow_read_back_round_trip`
    #[test]
    fn movrow_read_back_round_trip() {
        let mut scope = seeded_3x4();
        let t0 = scope.tile(0).unwrap();

        // Capture row 1 and its neighbours before any mutation.
        let original_row1 = _tile_movrow(&scope, t0, 1).unwrap();
        let original_row0 = _tile_movrow(&scope, t0, 0).unwrap();
        let original_row2 = _tile_movrow(&scope, t0, 2).unwrap();

        // Overwrite row 1 with a fresh pattern, then restore it via the write form.
        let mut scratch = [0u8; ZMM_BYTES];
        scratch[..4].copy_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD]);
        _tile_movrow_write(&mut scope, t0, 1, scratch).unwrap();
        assert_eq!(_tile_movrow(&scope, t0, 1), Some(scratch));

        _tile_movrow_write(&mut scope, t0, 1, original_row1).unwrap();

        // Round-trip: reading row 1 back returns exactly the original row.
        assert_eq!(_tile_movrow(&scope, t0, 1), Some(original_row1));
        // The write addressed only row 1: the neighbouring rows are unchanged.
        assert_eq!(_tile_movrow(&scope, t0, 0), Some(original_row0));
        assert_eq!(_tile_movrow(&scope, t0, 2), Some(original_row2));
    }

    /// Hand-computed known-value pins for both moves and both directions, independent of the
    /// implementation (`[ace-tile-instructions.TESTING.4]`). Each expected byte is derived
    /// directly from the seeded pattern `0x10 * (row + 1) + col`.
    /// `tile_move::move_known_value_pins`
    #[test]
    fn move_known_value_pins() {
        let mut scope = seeded_3x4();
        let t0 = scope.tile(0).unwrap();

        // Read pins: row 0 and column 0.
        let mut row0 = [0u8; ZMM_BYTES];
        row0[..4].copy_from_slice(&[0x10, 0x11, 0x12, 0x13]);
        assert_eq!(_tile_movrow(&scope, t0, 0), Some(row0));

        let mut col0 = [0u8; ZMM_BYTES];
        col0[..3].copy_from_slice(&[0x10, 0x20, 0x30]);
        assert_eq!(_tile_movcol(&scope, t0, 0), Some(col0));

        // Column write pin: set column 3 to [0x01, 0x02, 0x03]; the byte at (row, 3) becomes
        // the row-th lane and no other column changes.
        let mut newcol = [0u8; ZMM_BYTES];
        newcol[..3].copy_from_slice(&[0x01, 0x02, 0x03]);
        _tile_movcol_write(&mut scope, t0, 3, newcol).unwrap();
        assert_eq!(_tile_movcol(&scope, t0, 3), Some(newcol));
        // Column 2 is untouched by writing column 3.
        let mut col2 = [0u8; ZMM_BYTES];
        col2[..3].copy_from_slice(&[0x12, 0x22, 0x32]);
        assert_eq!(_tile_movcol(&scope, t0, 2), Some(col2));
    }

    /// An out-of-range row/column index is rejected at the typed bound: the read forms return
    /// `None`, and a failed write mutates no tile state.
    #[test]
    fn out_of_range_index_rejected_no_state_change() {
        let mut scope = seeded_3x4();
        let t0 = scope.tile(0).unwrap();

        // rows == 3, colsb == 4: indices 3 (row) and 4 (col) are the first out-of-range.
        assert_eq!(_tile_movrow(&scope, t0, 3), None);
        assert_eq!(_tile_movcol(&scope, t0, 4), None);

        // Snapshot every configured row, attempt out-of-range writes, and confirm nothing
        // changed.
        let before: Vec<_> = (0..3)
            .map(|r| _tile_movrow(&scope, t0, r).unwrap())
            .collect();
        assert_eq!(
            _tile_movrow_write(&mut scope, t0, 3, [0xFF; ZMM_BYTES]),
            None
        );
        assert_eq!(
            _tile_movcol_write(&mut scope, t0, 4, [0xFF; ZMM_BYTES]),
            None
        );
        let after: Vec<_> = (0..3)
            .map(|r| _tile_movrow(&scope, t0, r).unwrap())
            .collect();
        assert_eq!(before, after, "rejected writes leave tile state unchanged");
    }

    /// System-as-a-whole wiring check: seed a tile, move a column out, move it into a
    /// different column, and confirm the round-trip and gate helpers compose end to end.
    #[test]
    fn end_to_end_move_and_gates() {
        let mut scope = seeded_3x4();
        let t0 = scope.tile(0).unwrap();

        let col0 = _tile_movcol(&scope, t0, 0).unwrap();
        // Move column 0's contents into column 3.
        _tile_movcol_write(&mut scope, t0, 3, col0).unwrap();
        let col3 = _tile_movcol(&scope, t0, 3).unwrap();
        let round_trips = col0 == col3;

        println!(
            "E2E movcol_read={:?} movcol_write_round_trip={round_trips}",
            &col0[..3],
        );
        println!(
            "E2E detect has_amx_avx512={} has_ace={}",
            crate::detect::has_amx_avx512(),
            crate::detect::has_ace(),
        );
        assert!(
            round_trips,
            "column written then read back matches the source"
        );
    }
}

/// Layer-4 differential (family B). The tile->ZMM READ move is intrinsic-reachable, so under
/// `feature="native"` on x86_64 with the family-B read gate detected it compares the native
/// `TILEMOVROW` read shim to the oracle bit-for-bit (`[ace-tile-instructions.TESTING.1]`); the
/// ZMM->tile WRITE form is `ACE`-only (`.byte`, layer-3-blocked, OQ-6). Returns
/// [`quickcheck::TestResult::discard`] — never `from_bool(false)` — when the native path is
/// unavailable, so a non-tile runner never goes vacuously green.
#[cfg(test)]
mod differential {
    #![cfg_attr(
        not(all(target_arch = "x86_64", feature = "native")),
        allow(unused_imports, dead_code)
    )]
    use super::*;
    use crate::tile::{_tile_loadconfig, TileConfig};
    use quickcheck::{quickcheck, Arbitrary, Gen, TestResult};

    /// A single 64-byte tile row (`rows=1`, `colsb=64`), so the whole tile marshals through one
    /// 64-byte buffer.
    #[derive(Clone, Debug)]
    struct Row {
        data: [u8; ZMM_BYTES],
    }

    impl Arbitrary for Row {
        fn arbitrary(g: &mut Gen) -> Self {
            Row {
                data: core::array::from_fn(|_| u8::arbitrary(g)),
            }
        }
    }

    quickcheck! {
        /// Family-B moves native vs oracle over a 1x64 tile: the intrinsic `TILEMOVROW` READ
        /// form (lights up under SDE) and the `ACE`-only `.byte` WRITE forms (discard until SDE
        /// ACE). All discard off-tile.
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
                    let cfg = native::encode_tilecfg(2, &config.rows, &config.colsb);

                    // Read form (intrinsic).
                    let mut scope = _tile_loadconfig(&config).expect("valid descriptor");
                    let src = scope.tile(0).unwrap();
                    scope.tile_bytes_mut(src).copy_from_slice(&row.data);
                    let read_oracle = _tile_movrow_scalar(&scope, src, 0).expect("row 0 in range");
                    // SAFETY: has_amx_avx512() confirmed the read-form gate + tile XSAVE state.
                    let read_ok =
                        unsafe { native::tile_movrow_read_hw(&cfg, &row.data, 0) } == read_oracle;

                    // Write forms (.byte). Full-ACE-only: only exercise the native path once ACE
                    // is present; the write-form oracle writes `row.data` into a zeroed tile.
                    let write_ok = if detect::has_ace() {
                        let mut ws = _tile_loadconfig(&config).expect("valid descriptor");
                        let wd = ws.tile(0).unwrap();
                        _tile_movrow_write_scalar(&mut ws, wd, 0, row.data);
                        let row_want = ws.tile_bytes_ref(wd).to_vec();
                        let got_row = unsafe { native::tile_movrow_write_hw(&cfg, &row.data) };

                        let mut cs = _tile_loadconfig(&config).expect("valid descriptor");
                        let cd = cs.tile(0).unwrap();
                        _tile_movcol_write_scalar(&mut cs, cd, 0, row.data);
                        let col_want = cs.tile_bytes_ref(cd).to_vec();
                        let got_col = unsafe { native::tile_movcol_write_hw(&cfg, &row.data) };

                        got_row.as_slice() == row_want.as_slice()
                            && got_col.as_slice() == col_want.as_slice()
                    } else {
                        true // write forms discard until full ACE (SDE ACE) is present
                    };

                    return TestResult::from_bool(read_ok && write_ok);
                }
            }
            let _ = &row;
            TestResult::discard()
        }
    }
}
