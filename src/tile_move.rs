//! ACE group-4 family B: tile data movement (spec section 12).
//!
//! Models the two plain move operation classes between ACE tile registers and ZMM-sized
//! buffers:
//!
//! * `TILEMOVROW` — extended under ACE to allow BOTH read and write of a tile register row
//!   (spec section 12.2): [`_tile_movrow`] (read form, `zmm1, tmm2, r32/imm8`) and
//!   [`_tile_setrow`] (write form, `tmm1, zmm2, r32/imm8`)
//!   (`[ace-tile-instructions.TILE_MOVE.1]`).
//! * `TILEMOVCOL` — a new, WRITE-ONLY operation class (spec section 12.3): tile move column
//!   operations transfer data from AVX registers to ACE tile registers; there is no read
//!   form. [`_tile_setcol`] (`tmm1, zmm2, r32/imm8`)
//!   (`[ace-tile-instructions.TILE_MOVE.2]`).
//!
//! # Index semantics (spec section 12.1.1)
//!
//! Only bits `[3:0]` of the immediate or general-purpose row/column specifier are relevant;
//! other bits are RESERVED/SBZ, and an out-of-range specifier raises NO fault — the
//! interpretation is modified so the index is simply masked (`& 0xF`). These operations are
//! therefore total: every specifier value addresses a valid row/column of the fixed
//! 16-row tile.
//!
//! Row moves always transfer the full 64-byte (512-bit) row (`FOR col = 0 TO 63`, spec
//! section 12.2.4). The read form's architectural `dst[MAXVL-1:VL] = 0` is modeled
//! trivially by the 64-byte return. The column write follows the section-12.3.4 pseudocode
//! `for row in range(dst.rows): dst.byte[row][col] = src.byte[row]` — one byte per row from
//! the low 16 source bytes into byte-column `col`.
//!
//! # Dispatch and gates
//!
//! Each op is a safe public dispatcher plus a cfg-free `_scalar` oracle (the primary path,
//! correct on every target). Per the section-15.3 feature enumeration, the `TILEMOVROW`
//! read form gates on `AMX-AVX512 || ACE_VSN >= 1` (`detect::has_amx_avx512`,
//! `[ace-tile-instructions.DETECT.1-2]`) and the write forms (`TILEMOVROW` write,
//! `TILEMOVCOL`) are ACE-only (`detect::has_ace`,
//! `[ace-tile-instructions.DETECT.1-3]`). The register model lives in Rust, so the
//! dispatchers reference the detectors to mark the gate sites and take the scalar oracle
//! (`[ace-tile-instructions.DISPATCH.1]`).

use crate::detect;
use crate::tile::{TileId, TileScope, TILE_COLSB, TILE_ROWS};

/// Byte width of one ZMM register / one full tile row (512 bits).
pub const ZMM_BYTES: usize = 64;

/// Mask a row/column specifier to its architecturally relevant bits `[3:0]`
/// (spec section 12.1.1): out-of-range specifiers never fault, they wrap.
#[inline]
fn mask_index(specifier: u32) -> usize {
    (specifier & 0xF) as usize
}

/// `TILEMOVROW` read form: move row `row & 0xF` of tile `src` into a ZMM-sized buffer
/// (spec section 12.2.4 `tilemovrow_read_*`). Total — every specifier addresses a valid
/// row of the fixed 16-row tile (`[ace-tile-instructions.TILE_MOVE.1]`).
pub fn _tile_movrow(scope: &TileScope, src: TileId, row: u32) -> [u8; ZMM_BYTES] {
    let _ = detect::has_amx_avx512; // read-form gate: AMX-AVX512 || ACE_VSN >= 1 [DETECT.1-2]
    _tile_movrow_scalar(scope, src, row)
}

/// Portable `TILEMOVROW` (read) oracle — `row = specifier & 0xF`, then the full 64-byte
/// row (`FOR col = 0 TO 63`).
pub fn _tile_movrow_scalar(scope: &TileScope, src: TileId, row: u32) -> [u8; ZMM_BYTES] {
    let row = mask_index(row);
    let bytes = scope.tile_bytes_ref(src);
    let mut out = [0u8; ZMM_BYTES];
    out.copy_from_slice(&bytes[row * TILE_COLSB..(row + 1) * TILE_COLSB]);
    out
}

/// `TILEMOVROW` write form: move a ZMM-sized buffer into row `row & 0xF` of tile `dst`
/// (spec section 12.2.4 `tilemovrow_write_*`). ACE-only
/// (`[ace-tile-instructions.TILE_MOVE.1]`).
pub fn _tile_setrow(scope: &mut TileScope, dst: TileId, row: u32, src: [u8; ZMM_BYTES]) {
    let _ = detect::has_ace; // write-form gate: ACE [DETECT.1-3]
    _tile_setrow_scalar(scope, dst, row, src);
}

/// Portable `TILEMOVROW` (write) oracle — `row = specifier & 0xF`, then the full 64-byte
/// row is replaced.
pub fn _tile_setrow_scalar(scope: &mut TileScope, dst: TileId, row: u32, src: [u8; ZMM_BYTES]) {
    let row = mask_index(row);
    scope.tile_bytes_mut(dst)[row * TILE_COLSB..(row + 1) * TILE_COLSB].copy_from_slice(&src);
}

/// `TILEMOVCOL` (write-only): write byte-column `col & 0xF` of tile `dst` from the low 16
/// bytes of a ZMM-sized buffer (spec section 12.3.4:
/// `for row in range(dst.rows): dst.byte[row][col] = src.byte[row]`). There is NO read
/// form — column moves transfer data from AVX registers to ACE tile registers only (spec
/// section 12.3.1) (`[ace-tile-instructions.TILE_MOVE.2]`).
pub fn _tile_setcol(scope: &mut TileScope, dst: TileId, col: u32, src: [u8; ZMM_BYTES]) {
    let _ = detect::has_ace; // ACE-only operation class [DETECT.1-3]
    _tile_setcol_scalar(scope, dst, col, src);
}

/// Portable `TILEMOVCOL` oracle — the section-12.3.4 pseudocode, transcribed byte for
/// byte: `col = specifier & 0xF`, one source byte per row.
pub fn _tile_setcol_scalar(scope: &mut TileScope, dst: TileId, col: u32, src: [u8; ZMM_BYTES]) {
    let col = mask_index(col);
    let bytes = scope.tile_bytes_mut(dst);
    for row in 0..TILE_ROWS {
        bytes[row * TILE_COLSB + col] = src[row];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile::{_tile_loadconfig, TileConfig};

    fn scope_with_pattern() -> (TileScope, TileId) {
        let mut scope = _tile_loadconfig(&TileConfig::ace()).unwrap();
        let id = scope.tile(0).unwrap();
        // Row-major pattern: byte at (row, col) = (row*64+col) & 0xFF.
        let bytes = scope.tile_bytes_mut(id);
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = (i & 0xFF) as u8;
        }
        (scope, id)
    }

    /// Row read returns the full 64-byte row (`FOR col = 0 TO 63`, spec section 12.2.4)
    /// and row write replaces exactly that row (write/read round-trip, INV-6).
    /// `tile_move::row_moves_full_64_bytes`
    #[test]
    fn row_moves_full_64_bytes() {
        let (mut scope, id) = scope_with_pattern();
        let row3 = _tile_movrow(&scope, id, 3);
        let expect: [u8; 64] = core::array::from_fn(|c| ((3 * 64 + c) & 0xFF) as u8);
        assert_eq!(row3, expect, "read returns the full 64-byte row 3");

        let fresh: [u8; 64] = core::array::from_fn(|c| 0xE0 ^ c as u8);
        _tile_setrow(&mut scope, id, 3, fresh);
        assert_eq!(_tile_movrow(&scope, id, 3), fresh, "write replaced row 3");
        // Neighboring rows untouched.
        let row2 = _tile_movrow(&scope, id, 2);
        assert_eq!(row2[0], ((2 * 64) & 0xFF) as u8, "row 2 untouched");
    }

    /// Out-of-range specifiers never fault: only bits [3:0] are relevant, so specifier 16
    /// addresses row 0 and specifier 0xFFFF_FFF5 addresses row 5 (spec section 12.1.1).
    /// `tile_move::index_masks_to_4_bits`
    #[test]
    fn index_masks_to_4_bits() {
        let (mut scope, id) = scope_with_pattern();
        assert_eq!(
            _tile_movrow(&scope, id, 16),
            _tile_movrow(&scope, id, 0),
            "specifier 16 wraps to row 0"
        );
        assert_eq!(
            _tile_movrow(&scope, id, 0xFFFF_FFF5),
            _tile_movrow(&scope, id, 5),
            "only bits [3:0] of the specifier are relevant"
        );
        // Same wrap on the write forms.
        let marker = [0x99u8; 64];
        _tile_setrow(&mut scope, id, 21, marker); // 21 & 0xF = 5
        assert_eq!(_tile_movrow(&scope, id, 5), marker);
        _tile_setcol(&mut scope, id, 19, [0x42; 64]); // 19 & 0xF = 3
        assert_eq!(
            scope.tile_bytes_ref(id)[3],
            0x42,
            "row 0, byte-column 3 written"
        );
    }

    /// TILEMOVCOL writes one byte per row from the low 16 source bytes into byte-column
    /// `col` (spec section 12.3.4: `dst.byte[row][col] = src.byte[row]`), leaving every
    /// other byte untouched.
    /// `tile_move::setcol_writes_byte_column`
    #[test]
    fn setcol_writes_byte_column() {
        let (mut scope, id) = scope_with_pattern();
        let before = *scope.tile_bytes_ref(id);
        let src: [u8; 64] = core::array::from_fn(|i| (0xC0 + i) as u8);

        _tile_setcol(&mut scope, id, 7, src);

        let after = scope.tile_bytes_ref(id);
        for row in 0..TILE_ROWS {
            for col in 0..TILE_COLSB {
                if col == 7 {
                    assert_eq!(
                        after[row * TILE_COLSB + col],
                        src[row],
                        "byte-column 7, row {row} takes src.byte[{row}]"
                    );
                } else {
                    assert_eq!(
                        after[row * TILE_COLSB + col],
                        before[row * TILE_COLSB + col],
                        "byte ({row},{col}) untouched"
                    );
                }
            }
        }
    }
}
