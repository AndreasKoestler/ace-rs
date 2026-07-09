//! ACE group-4 family A: tile configuration lifecycle, wrapped in the crate's first RAII
//! guard.
//!
//! This module models the palette-2 tile register file and the four family-A lifecycle
//! instructions:
//!
//! * `LDTILECFG` — [`_tile_loadconfig`] validates a palette-2 [`TileConfig`], allocates the
//!   zeroed tile backing buffers it describes, and returns the [`TileScope`] guard
//!   (`[ace-tile-instructions.TILE_LIFECYCLE.1]`).
//! * `STTILECFG` — [`_tile_storeconfig`] returns a [`TileConfig`] equal to the one the guard
//!   loaded (INV-3, `[ace-tile-instructions.TILE_LIFECYCLE.2]`).
//! * `TILEZERO` — [`_tile_zero`] clears the addressed accumulator tile to zero and nothing
//!   else (`[ace-tile-instructions.TILE_LIFECYCLE.3]`).
//! * `TILERELEASE` — run by `impl Drop for TileScope`, invalidating all tile register state
//!   exactly once, including on panic unwind (`[ace-tile-instructions.TILE_LIFECYCLE.4]`,
//!   `[ace-tile-instructions.TILE_LIFECYCLE.5]`). There is no free-standing `_tile_release`
//!   call: release is the guard's `Drop`, so configuration can never leak (INV-1).
//!
//! # Lifecycle state model
//!
//! ```text
//!   Uninitialized --Acquire (_tile_loadconfig)--> Configured
//!   Configured    --Zero / Store config (self-loops)--> Configured
//!   Configured    --Release (Drop, incl. panic unwind)--> Released
//! ```
//!
//! A [`TileScope`] owns its register model outright and holds no global mutable state, so
//! independent guards — nested or sequential — never interfere and never leak configuration
//! (`[ace-tile-instructions.TILE_LIFECYCLE.7]`). The guard's tile bytes and descriptor are
//! reached through the guard's own accessors and the guard-borrowed [`TileId`] handle; that
//! is the only storage-shaped interface (there is no persistence layer).
//!
//! # Dispatch
//!
//! Each lifecycle op is a safe public dispatcher plus a cfg-free `_scalar` oracle (the
//! primary path, correct on every target including non-x86). Family A's native gate is
//! AMX-TILE (`detect::has_amx_tile`, `[ace-tile-instructions.DETECT.1-1]`). No native tile
//! shim exists yet — the `.byte` / C-intrinsic backend and its layer-3 execution land with
//! the native path (OQ-6) — so, exactly as the oracle-only group-3 modules do, the
//! dispatchers reference the detector to mark the gate site and take the scalar oracle on
//! every target (`[ace-tile-instructions.DISPATCH.1]`).
//!
//! # Open questions surfaced here
//!
//! * **OQ-2 (guard shape).** The default is realised: [`TileConfig`] is a validated palette-2
//!   descriptor, [`Tile`] is an opaque newtype over a descriptor-sized byte backing buffer,
//!   and [`TileId`] is a lightweight handle minted only by the guard (no public constructor),
//!   so it cannot be forged and is useless once the guard has dropped (every op needs the
//!   scope). The byte buffers are transmute-shaped for the eventual `__tile1024i`/`__tile`
//!   intrinsic types.
//! * **OQ-7 (module split).** This is the `src/tile.rs` file of the five-file split
//!   (`tile` / `tile_move` / `tcvtrow` / `bsr` / `top`); the block-scale register (`BSR`)
//!   file the guard will also own is added with family D.

use crate::bsr::{BsrId, BsrReg, NUM_BSR};
use crate::detect;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Number of addressable tiles in a palette-2 configuration (`TMM0..=TMM7`).
pub const MAX_TILES: usize = 8;

// Forward-reference the rest of the group-4 detection + codec surface this feature delivers
// in phase 1 but whose consumers arrive in later phases (crate idiom for
// delivered-but-not-yet-consumed items, cf. the gate markers in `src/cvt_fp8_ps.rs`). A
// `const _` binding keeps them "used" without a runtime effect and without lint-muting:
//   * `detect::has_amx_avx512` — family-C converts + `TILEMOVROW` read form gate
//     (`[ace-tile-instructions.DETECT.1-2]`); family A itself gates on AMX-TILE below.
//   * `detect::has_ace` — families D/E/F/G + write-form moves gate
//     (`[ace-tile-instructions.DETECT.1-3]`).
//   * `fp8::fp32_to_bf16_rne` / `fp8::bf16_to_fp32` — the net-new BF16 codec (R6), decoded /
//     encoded by the family-C `TCVTROWPS2BF16*` converts (phase 3) and the family-F
//     `TOP2BF16PS` outer product (phase 6).
const _: () = {
    let _ = detect::has_amx_avx512;
    let _ = detect::has_ace;
    let _ = crate::fp8::fp32_to_bf16_rne;
    let _ = crate::fp8::bf16_to_fp32;
};

/// The only palette this family supports.
const PALETTE_2: u8 = 2;
/// Palette-2 per-tile row limit.
const MAX_ROWS: u8 = 16;
/// Palette-2 per-tile bytes-per-row (`colsb`) limit.
const MAX_COLSB: u16 = 64;

/// A palette-2 tile configuration descriptor.
///
/// Mirrors the `LDTILECFG` descriptor: a palette id (which must be `2` for this family) and,
/// per tile, a row count and a bytes-per-row (`colsb`) count. The descriptor drives the tile
/// backing-buffer sizing on Acquire. Validation ([`TileConfig::validate`]) runs in
/// [`_tile_loadconfig`] before any tile state is established
/// (`[ace-tile-instructions.TILE_LIFECYCLE.1]`).
///
/// Field names are Inferred (OQ-2): the descriptor has no ticket-named layout, so they are
/// derived directly from the palette-2 descriptor limits (palette id `== 2`, `rows <= 16`,
/// `colsb <= 64`).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TileConfig {
    /// Palette identifier; must equal `2`.
    pub palette_id: u8,
    /// Per-tile row counts; each must be `<= 16`. A `0` entry is an unconfigured tile.
    pub rows: [u8; MAX_TILES],
    /// Per-tile bytes-per-row; each must be `<= 64`. A `0` entry is an unconfigured tile.
    pub colsb: [u16; MAX_TILES],
}

/// Rejection reasons for an out-of-limits palette-2 descriptor
/// (`[ace-tile-instructions.TILE_LIFECYCLE.1]`).
///
/// Variant names are Inferred (OQ-2): the descriptor validation has no ticket-named error
/// type, so the variants are derived from the three palette-2 descriptor limits.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TileConfigError {
    /// The palette id was not `2`.
    InvalidPalette,
    /// Some tile's `rows` entry exceeded `16`.
    RowsOutOfRange,
    /// Some tile's `colsb` entry exceeded `64`.
    ColsbOutOfRange,
}

impl TileConfig {
    /// Validate the palette-2 descriptor limits, checking palette id first, then `rows`, then
    /// `colsb`. Returns the first violation; a palette error takes priority over a
    /// row/column error so an all-bad descriptor reports [`TileConfigError::InvalidPalette`].
    fn validate(&self) -> Result<(), TileConfigError> {
        if self.palette_id != PALETTE_2 {
            return Err(TileConfigError::InvalidPalette);
        }
        if self.rows.iter().any(|&r| r > MAX_ROWS) {
            return Err(TileConfigError::RowsOutOfRange);
        }
        if self.colsb.iter().any(|&c| c > MAX_COLSB) {
            return Err(TileConfigError::ColsbOutOfRange);
        }
        Ok(())
    }
}

/// One 2-D tile: an opaque newtype over a descriptor-sized byte backing buffer (OQ-2).
///
/// The buffer is `rows * colsb` bytes, row-major, and is reinterpreted as `[i32]`/`[f32]`
/// for accumulator tiles by the outer-product families; family A only needs to allocate,
/// zero, and clear it. Only reachable through the owning [`TileScope`].
#[derive(Clone, Debug)]
struct Tile {
    rows: u8,
    colsb: u16,
    bytes: Vec<u8>,
}

impl Tile {
    /// Allocate a zeroed tile sized from its descriptor entry.
    fn new(rows: u8, colsb: u16) -> Self {
        Tile {
            rows,
            colsb,
            bytes: vec![0u8; rows as usize * colsb as usize],
        }
    }
}

/// A handle addressing one configured tile of a [`TileScope`].
///
/// A `TileId` is minted only by [`TileScope::tile`] — it has no public constructor, so it
/// cannot be forged to bypass the guard (OQ-2). Because every tile operation takes the
/// scope, a handle is useless once the guard has released its configuration: dropping the
/// scope and then using the handle does not compile (INV-2,
/// `[ace-tile-instructions.TILE_LIFECYCLE.6]`):
///
/// ```compile_fail
/// use ace::{_tile_loadconfig, _tile_zero, TileConfig};
/// let cfg = TileConfig {
///     palette_id: 2,
///     rows: [4, 0, 0, 0, 0, 0, 0, 0],
///     colsb: [16, 0, 0, 0, 0, 0, 0, 0],
/// };
/// let mut scope = _tile_loadconfig(&cfg).unwrap();
/// let id = scope.tile(0).unwrap();
/// drop(scope); // TILERELEASE: the guard releases its configuration here
/// _tile_zero(&mut scope, id); // ERROR: use of moved value `scope` — the handle is unusable
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TileId {
    index: usize,
}

/// The RAII tile lifecycle guard (`[ace-tile-instructions.TILE_LIFECYCLE.5]`).
///
/// Constructed by [`_tile_loadconfig`] (Acquire), it owns the configured tile backing
/// buffers and the loaded descriptor. Its `Drop` runs `TILERELEASE`, invalidating all tile
/// register state exactly once — including on panic unwind — so configuration never leaks
/// (INV-1). The guard holds no global mutable state, so nested or sequential guards are
/// independent (`[ace-tile-instructions.TILE_LIFECYCLE.7]`).
///
/// The block-scale register (`BSR`) file the guard will also own is added with family D; a
/// palette-2 guard needs only the tile file for family A.
///
/// The block-scale register (`BSR`) file (family D) is now also owned here, fulfilling the
/// phase-1 deferral: it shares the tile file's RAII lifecycle (zeroed on Acquire, invalidated
/// on Release) so the MX-FP8 outer products read the block scale back through the same guard
/// the `BSR` ops wrote it into (INV-5).
#[derive(Debug)]
pub struct TileScope {
    config: TileConfig,
    tiles: [Tile; MAX_TILES],
    /// The ACE block-scale register file (`BSR0..=BSR7`), addressed by [`BsrId`]. Owned by the
    /// guard so `BSR` writes and the MX products' reads share one register model (INV-5).
    bsr: [BsrReg; NUM_BSR],
    /// Set once by [`TileScope::release`] so the release logic is idempotent.
    released: bool,
    /// Release ledger: when attached, `Drop` increments it exactly once, letting a caller
    /// confirm the RAII `TILERELEASE` fired — including across a panic unwind, where the
    /// released guard's own state is gone but the ledger survives. `None` in ordinary use.
    release_ledger: Option<Arc<AtomicUsize>>,
}

impl TileScope {
    /// Mint a handle to tile `index` (`0..MAX_TILES`), or `None` if out of range. The only
    /// way to obtain a [`TileId`]; the returned handle borrows nothing, so it can be passed
    /// to the `&mut self`-style lifecycle ops.
    pub fn tile(&self, index: usize) -> Option<TileId> {
        if index < MAX_TILES {
            Some(TileId { index })
        } else {
            None
        }
    }

    /// Mint a handle to block-scale register `index` (`0..NUM_BSR`), or `None` if out of range.
    /// The only way to obtain a [`BsrId`]; mirrors [`TileScope::tile`] so the guard owns the
    /// whole tile + `BSR` register model.
    pub fn bsr(&self, index: usize) -> Option<BsrId> {
        if index < NUM_BSR {
            Some(BsrId::new(index))
        } else {
            None
        }
    }

    /// Read-back accessor: the full 8-bit scale exponent of `block` in the addressed
    /// block-scale register, or `None` if `block` is out of range. This is the register-model
    /// storage interface the family-D tests and the MX outer products (phase 7) read the block
    /// scale back through (INV-5).
    pub(crate) fn bsr_scale(&self, id: BsrId, block: usize) -> Option<u8> {
        self.bsr[id.index()].scale(block)
    }

    /// Shared read accessor for the whole addressed block-scale register, used by the MX-FP8
    /// outer products (phase 7) to read every per-block scale the `BSR` ops wrote (INV-5).
    pub(crate) fn bsr_reg(&self, id: BsrId) -> &BsrReg {
        &self.bsr[id.index()]
    }

    /// Mutable register-model accessor for the addressed block-scale register, used by the
    /// family-D `BSRINIT` / `BSRMOV{F,H,L}` oracles to seed / move its per-block exponents.
    pub(crate) fn bsr_reg_mut(&mut self, id: BsrId) -> &mut BsrReg {
        &mut self.bsr[id.index()]
    }

    /// Read-back accessor for the addressed tile's raw backing bytes (the register-model
    /// storage interface). Test-only in phase 1 — the real convert/move families that read
    /// tiles land in later phases; family A's own tests use it to inspect register state.
    #[cfg(test)]
    fn tile_bytes(&self, id: TileId) -> &[u8] {
        &self.tiles[id.index].bytes
    }

    /// The configured `(rows, colsb)` shape of the addressed tile — the register-model
    /// accessor family B uses to bound a row/column move to the tile's configured extent
    /// (`[ace-tile-instructions.TILE_MOVE.1-1]`). A `(0, 0)` shape denotes an unconfigured
    /// tile, for which every index is out of range.
    pub(crate) fn tile_shape(&self, id: TileId) -> (u8, u16) {
        let tile = &self.tiles[id.index];
        (tile.rows, tile.colsb)
    }

    /// Shared read-back accessor for the addressed tile's raw backing bytes (the register-model
    /// storage interface), used by the family-B move oracles to extract a row/column.
    pub(crate) fn tile_bytes_ref(&self, id: TileId) -> &[u8] {
        &self.tiles[id.index].bytes
    }

    /// Mutable register-model accessor for the addressed tile's raw backing bytes, used by the
    /// family-B write-form move oracles to insert a row/column.
    pub(crate) fn tile_bytes_mut(&mut self, id: TileId) -> &mut [u8] {
        &mut self.tiles[id.index].bytes
    }

    /// `TILERELEASE` (Drop-only): invalidate all tile register state, idempotently. Zeroes
    /// and unconfigures every tile so a subsequent read observes released state, and bumps
    /// the release ledger if one is attached.
    fn release(&mut self) {
        if self.released {
            return;
        }
        self.released = true;
        for tile in &mut self.tiles {
            tile.bytes.iter_mut().for_each(|b| *b = 0);
            tile.rows = 0;
            tile.colsb = 0;
        }
        // TILERELEASE also invalidates the block-scale file: a released guard observes a clean
        // BSR file, and the next Acquire starts from cleared block scales (INV-1).
        for reg in &mut self.bsr {
            *reg = BsrReg::zeroed();
        }
        if let Some(ledger) = &self.release_ledger {
            ledger.fetch_add(1, Ordering::SeqCst);
        }
    }
}

impl Drop for TileScope {
    /// `TILERELEASE` on scope exit — normal return, early return, or panic unwind — so tile
    /// configuration can never leak (INV-1, `[ace-tile-instructions.TILE_LIFECYCLE.4]`,
    /// `[ace-tile-instructions.TILE_LIFECYCLE.5]`). Runs exactly once (Rust ownership +
    /// [`TileScope::release`]'s idempotence guard).
    fn drop(&mut self) {
        self.release();
    }
}

/// `LDTILECFG`: validate a palette-2 descriptor and return the configured, zeroed
/// [`TileScope`] guard, or `Err` with no tile state established
/// (`[ace-tile-instructions.TILE_LIFECYCLE.1]`).
///
/// Family A's native gate is AMX-TILE (`[ace-tile-instructions.DETECT.1-1]`). The sibling
/// per-family gates are part of the same detection surface the tile family builds on and are
/// wired now for its later members: [`detect::has_amx_avx512`] for the family-C converts and
/// the `TILEMOVROW` read form (`[ace-tile-instructions.DETECT.1-2]`), and [`detect::has_ace`]
/// for families D/E/F/G and the write-form moves (`[ace-tile-instructions.DETECT.1-3]`). No
/// native tile shim exists yet (OQ-6), so — as in the oracle-only group-3 modules — the
/// detectors are referenced (not called) to mark the gate sites and the scalar oracle is the
/// path taken on every target (`[ace-tile-instructions.DISPATCH.1]`).
pub fn _tile_loadconfig(config: &TileConfig) -> Result<TileScope, TileConfigError> {
    let _ = detect::has_amx_tile; // family A native gate: AMX-TILE [DETECT.1-1]
    _tile_loadconfig_scalar(config)
}

/// Portable `LDTILECFG` oracle — the primary, always-correct path. Validates before
/// establishing any state, then allocates the zeroed tile backing buffers the descriptor
/// sizes and records the configured state.
pub fn _tile_loadconfig_scalar(config: &TileConfig) -> Result<TileScope, TileConfigError> {
    config.validate()?;
    let tiles = core::array::from_fn(|i| Tile::new(config.rows[i], config.colsb[i]));
    let bsr = core::array::from_fn(|_| BsrReg::zeroed());
    Ok(TileScope {
        config: config.clone(),
        tiles,
        bsr,
        released: false,
        release_ledger: None,
    })
}

/// `STTILECFG`: return a [`TileConfig`] equal to the one Acquire loaded (INV-3,
/// `[ace-tile-instructions.TILE_LIFECYCLE.2]`). Pure read; identical behavior off-x86.
pub fn _tile_storeconfig(scope: &TileScope) -> TileConfig {
    let _ = detect::has_amx_tile; // family A gate site [DETECT.1-1]
    _tile_storeconfig_scalar(scope)
}

/// Portable `STTILECFG` oracle — returns the guard's stored descriptor verbatim.
pub fn _tile_storeconfig_scalar(scope: &TileScope) -> TileConfig {
    scope.config.clone()
}

/// `TILEZERO`: clear only the addressed accumulator tile to all-zero
/// (`[ace-tile-instructions.TILE_LIFECYCLE.3]`).
pub fn _tile_zero(scope: &mut TileScope, dst: TileId) {
    let _ = detect::has_amx_tile; // family A gate site [DETECT.1-1]
    _tile_zero_scalar(scope, dst);
}

/// Portable `TILEZERO` oracle — zeroes the addressed tile's bytes and leaves every other
/// tile untouched.
pub fn _tile_zero_scalar(scope: &mut TileScope, dst: TileId) {
    scope.tiles[dst.index].bytes.iter_mut().for_each(|b| *b = 0);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A palette-2 descriptor with tile 0 = `rows x colsb` and tile 1 = 2x8, rest empty.
    fn two_tile_config() -> TileConfig {
        TileConfig {
            palette_id: 2,
            rows: [4, 2, 0, 0, 0, 0, 0, 0],
            colsb: [16, 8, 0, 0, 0, 0, 0, 0],
        }
    }

    /// STTILECFG round-trips the descriptor LDTILECFG loaded, and Acquire zeroes the tiles
    /// (INV-3, `[ace-tile-instructions.TILE_LIFECYCLE.1]`,
    /// `[ace-tile-instructions.TILE_LIFECYCLE.2]`).
    /// `tile::loadconfig_storeconfig_round_trip`
    #[test]
    fn loadconfig_storeconfig_round_trip() {
        let cfg = two_tile_config();
        let scope = _tile_loadconfig(&cfg).expect("valid palette-2 descriptor");
        // Round-trip: stored descriptor equals the loaded one, field for field.
        assert_eq!(_tile_storeconfig(&scope), cfg);
        // Hand-value pin: tile 0 is 4x16 = 64 bytes, tile 1 is 2x8 = 16 bytes, both zeroed.
        assert_eq!(scope.tile_bytes(scope.tile(0).unwrap()).len(), 64);
        assert_eq!(scope.tile_bytes(scope.tile(1).unwrap()).len(), 16);
        assert!(
            scope
                .tile_bytes(scope.tile(0).unwrap())
                .iter()
                .all(|&b| b == 0),
            "Acquire zeroes the tile"
        );
    }

    /// TILEZERO clears only the addressed tile; every other tile is unchanged
    /// (`[ace-tile-instructions.TILE_LIFECYCLE.3]`).
    /// `tile::zero_clears_only_addressed_tile`
    #[test]
    fn zero_clears_only_addressed_tile() {
        let mut scope = _tile_loadconfig(&two_tile_config()).unwrap();
        // Seed both tiles with a recognisable nonzero pattern (no move op exists in family A,
        // so write the backing bytes directly through the module-private field).
        for byte in scope.tiles[0].bytes.iter_mut() {
            *byte = 0xAB;
        }
        for byte in scope.tiles[1].bytes.iter_mut() {
            *byte = 0xCD;
        }
        let t0 = scope.tile(0).unwrap();
        let t1 = scope.tile(1).unwrap();

        _tile_zero(&mut scope, t0);

        assert!(
            scope.tile_bytes(t0).iter().all(|&b| b == 0),
            "addressed tile 0 is now all-zero"
        );
        assert!(
            scope.tile_bytes(t1).iter().all(|&b| b == 0xCD),
            "tile 1 is untouched by zeroing tile 0"
        );
    }

    /// The RAII guard releases on panic unwind, exactly once
    /// (`[ace-tile-instructions.TILE_LIFECYCLE.4]`, `[ace-tile-instructions.TILE_LIFECYCLE.5]`).
    /// Observed through the release ledger, which survives the guard's drop.
    /// `tile::drop_on_panic_releases`
    #[test]
    fn drop_on_panic_releases() {
        use std::panic::{catch_unwind, AssertUnwindSafe};

        let ledger = Arc::new(AtomicUsize::new(0));
        let ledger_for_guard = ledger.clone();

        let result = catch_unwind(AssertUnwindSafe(|| {
            let mut scope = _tile_loadconfig(&two_tile_config()).unwrap();
            scope.release_ledger = Some(ledger_for_guard);
            // Panic between Acquire and the normal end of scope; Drop must still run.
            panic!("boom mid-lifecycle");
        }));

        assert!(result.is_err(), "the closure panicked");
        assert_eq!(
            ledger.load(Ordering::SeqCst),
            1,
            "TILERELEASE ran exactly once on panic unwind"
        );
    }

    /// Nested and sequential guards do not leak configuration: each owns an independent
    /// register model, and each Acquire starts from clean, zeroed tiles
    /// (`[ace-tile-instructions.TILE_LIFECYCLE.7]`).
    /// `tile::nested_sequential_guards_do_not_leak`
    #[test]
    fn nested_sequential_guards_do_not_leak() {
        let ledger = Arc::new(AtomicUsize::new(0));

        // Sequential: a guard's release fires when it drops, and the next Acquire is clean.
        {
            let mut first = _tile_loadconfig(&two_tile_config()).unwrap();
            first.release_ledger = Some(ledger.clone());
            for byte in first.tiles[0].bytes.iter_mut() {
                *byte = 0xFF;
            }
        } // first drops here -> release
        assert_eq!(
            ledger.load(Ordering::SeqCst),
            1,
            "first guard released on drop"
        );

        let second = _tile_loadconfig(&two_tile_config()).unwrap();
        assert!(
            second
                .tile_bytes(second.tile(0).unwrap())
                .iter()
                .all(|&b| b == 0),
            "a fresh guard does not inherit the previous guard's tile bytes"
        );

        // Nested: two live guards are independent — mutating one leaves the other untouched.
        let mut outer = _tile_loadconfig(&two_tile_config()).unwrap();
        for byte in outer.tiles[0].bytes.iter_mut() {
            *byte = 0x11;
        }
        {
            let mut inner = _tile_loadconfig(&two_tile_config()).unwrap();
            let inner_t0 = inner.tile(0).unwrap();
            _tile_zero(&mut inner, inner_t0);
            for byte in inner.tiles[1].bytes.iter_mut() {
                *byte = 0x22;
            }
        }
        assert!(
            outer
                .tile_bytes(outer.tile(0).unwrap())
                .iter()
                .all(|&b| b == 0x11),
            "the outer guard is unaffected by the inner guard's lifetime"
        );
    }

    /// Hand-computed family-A known-value pins, independent of the implementation
    /// (`[ace-tile-instructions.TESTING.4]`). Palette-2 descriptor limits are the load-bearing
    /// semantics with subtle inclusive/exclusive boundaries and an error-priority order, so
    /// each case's expected result DIFFERS under a leading wrong model (exclusive bounds, or
    /// checking rows/colsb before palette); differential tiebreaker unavailable here, so these
    /// are grounded against the palette-2 descriptor limits (palette `== 2`, `rows <= 16`,
    /// `colsb <= 64`).
    /// `tile::family_a_known_value_pins`
    #[test]
    fn family_a_known_value_pins() {
        // Boundary values are INCLUSIVE: rows == 16 and colsb == 64 are valid (an exclusive
        // `< 16` / `< 64` model would wrongly reject these).
        let at_limit = TileConfig {
            palette_id: 2,
            rows: [16, 0, 0, 0, 0, 0, 0, 0],
            colsb: [64, 0, 0, 0, 0, 0, 0, 0],
        };
        let scope = _tile_loadconfig(&at_limit).expect("rows==16, colsb==64 are valid");
        assert_eq!(
            scope.tile_bytes(scope.tile(0).unwrap()).len(),
            16 * 64,
            "tile 0 backing buffer is rows*colsb = 1024 bytes"
        );

        // Just over each limit is rejected with the matching variant.
        let rows_over = TileConfig {
            palette_id: 2,
            rows: [17, 0, 0, 0, 0, 0, 0, 0],
            colsb: [16, 0, 0, 0, 0, 0, 0, 0],
        };
        assert_eq!(
            _tile_loadconfig(&rows_over).unwrap_err(),
            TileConfigError::RowsOutOfRange
        );

        let colsb_over = TileConfig {
            palette_id: 2,
            rows: [1, 0, 0, 0, 0, 0, 0, 0],
            colsb: [65, 0, 0, 0, 0, 0, 0, 0],
        };
        assert_eq!(
            _tile_loadconfig(&colsb_over).unwrap_err(),
            TileConfigError::ColsbOutOfRange
        );

        // Wrong palette is rejected regardless of the other fields.
        let bad_palette = TileConfig {
            palette_id: 1,
            rows: [1, 0, 0, 0, 0, 0, 0, 0],
            colsb: [16, 0, 0, 0, 0, 0, 0, 0],
        };
        assert_eq!(
            _tile_loadconfig(&bad_palette).unwrap_err(),
            TileConfigError::InvalidPalette
        );

        // Error PRIORITY: palette is checked before rows/colsb, so an all-bad descriptor
        // reports InvalidPalette (a model that validated rows/colsb first would return
        // RowsOutOfRange or ColsbOutOfRange here).
        let all_bad = TileConfig {
            palette_id: 0,
            rows: [17, 0, 0, 0, 0, 0, 0, 0],
            colsb: [65, 0, 0, 0, 0, 0, 0, 0],
        };
        assert_eq!(
            _tile_loadconfig(&all_bad).unwrap_err(),
            TileConfigError::InvalidPalette,
            "palette error takes priority over rows/colsb errors"
        );
    }

    /// The error path establishes no tile state: an invalid descriptor yields `Err` and no
    /// `TileScope` value at all (`[ace-tile-instructions.TILE_LIFECYCLE.1]`). Also confirms
    /// the oracle and dispatcher agree.
    #[test]
    fn invalid_descriptor_establishes_no_state() {
        let bad = TileConfig {
            palette_id: 3,
            rows: [1; MAX_TILES],
            colsb: [1; MAX_TILES],
        };
        let dispatched = _tile_loadconfig(&bad);
        let oracle = _tile_loadconfig_scalar(&bad);
        assert!(dispatched.is_err() && oracle.is_err());
        assert_eq!(dispatched.unwrap_err(), TileConfigError::InvalidPalette);
        // No guard exists on the error path -> nothing to release, nothing to leak.
    }

    /// System-as-a-whole wiring check: the full family-A lifecycle end to end plus the
    /// per-family detection helpers, printing observable state. Confirms the module is wired
    /// through the crate root and the pieces compose (Acquire -> Store round-trip -> Zero ->
    /// Drop/Release; detect helpers compile and return `bool`).
    #[test]
    fn end_to_end_lifecycle_and_detection() {
        let ledger = Arc::new(AtomicUsize::new(0));
        let cfg = two_tile_config();
        {
            let mut scope = _tile_loadconfig(&cfg).unwrap();
            scope.release_ledger = Some(ledger.clone());
            let round_trip = _tile_storeconfig(&scope) == cfg;
            let t0 = scope.tile(0).unwrap();
            for byte in scope.tiles[0].bytes.iter_mut() {
                *byte = 0x5A;
            }
            _tile_zero(&mut scope, t0);
            let zeroed = scope.tile_bytes(t0).iter().all(|&b| b == 0);
            println!(
                "E2E acquire_ok=true store_round_trip={round_trip} zero_ok={zeroed} release_before_drop={}",
                ledger.load(Ordering::SeqCst)
            );
        } // Drop -> Release here.
        println!(
            "E2E release_count_after_drop={}",
            ledger.load(Ordering::SeqCst)
        );
        println!(
            "E2E detect has_amx_tile={} has_amx_avx512={} has_ace={}",
            crate::detect::has_amx_tile(),
            crate::detect::has_amx_avx512(),
            crate::detect::has_ace(),
        );
        assert_eq!(
            ledger.load(Ordering::SeqCst),
            1,
            "released exactly once on drop"
        );
    }
}

/// Layer-4 differential (family A). Bit-for-bit native-vs-oracle agreement of the STTILECFG
/// round-trip: under `feature="native"` on x86_64 with AMX-TILE + the tile XSAVE state detected,
/// the native `LDTILECFG`/`STTILECFG` shim must round-trip the palette-2 descriptor to the same
/// `(rows, colsb)` the oracle stores (INV-3, `[ace-tile-instructions.TESTING.1]`). Family A is
/// intrinsic-reachable, so this LIGHTS UP under Intel SDE; on any host without AMX-TILE it
/// returns [`quickcheck::TestResult::discard`] — never `from_bool(false)` — so a non-tile runner
/// can never go vacuously green (the crate-wide non-vacuous convention, OQ-6).
#[cfg(test)]
mod differential {
    // Under the default (no-`native`) build the quickcheck body compiles down to the discard
    // arm, so the `super` imports are only read on the native + x86_64 configuration.
    #![cfg_attr(
        not(all(target_arch = "x86_64", feature = "native")),
        allow(unused_imports, dead_code)
    )]
    use super::*;
    use quickcheck::{quickcheck, Arbitrary, Gen, TestResult};

    /// A random, always-valid palette-2 descriptor (`rows <= 16`, `colsb <= 64`).
    #[derive(Clone, Debug)]
    struct Cfg {
        rows: [u8; MAX_TILES],
        colsb: [u16; MAX_TILES],
    }

    impl Arbitrary for Cfg {
        fn arbitrary(g: &mut Gen) -> Self {
            Cfg {
                rows: core::array::from_fn(|_| u8::arbitrary(g) % (MAX_ROWS + 1)),
                colsb: core::array::from_fn(|_| u16::arbitrary(g) % (MAX_COLSB + 1)),
            }
        }
    }

    quickcheck! {
        fn prop_native_matches_oracle(cfg: Cfg) -> TestResult {
            #[cfg(all(target_arch = "x86_64", feature = "native"))]
            {
                if detect::has_amx_tile() {
                    let config = TileConfig {
                        palette_id: PALETTE_2,
                        rows: cfg.rows,
                        colsb: cfg.colsb,
                    };
                    let scope = _tile_loadconfig(&config).expect("valid palette-2 descriptor");
                    let oracle = _tile_storeconfig(&scope);
                    let desc = crate::native::encode_tilecfg(PALETTE_2, &cfg.rows, &cfg.colsb);
                    // SAFETY: has_amx_tile() confirmed AMX-TILE + the tile XSAVE state.
                    let got = unsafe { crate::native::tile_cfg_roundtrip_hw(&desc) };
                    let rt_rows: [u8; MAX_TILES] = core::array::from_fn(|t| got[48 + t]);
                    let rt_colsb: [u16; MAX_TILES] =
                        core::array::from_fn(|t| u16::from_le_bytes([got[16 + 2 * t], got[16 + 2 * t + 1]]));
                    let cfg_ok = rt_rows == oracle.rows && rt_colsb == oracle.colsb;

                    // TILEZERO differential: a 1x64 tile seeded with the descriptor bytes, zeroed
                    // natively, must match the oracle's zeroed tile.
                    let zcfg = TileConfig {
                        palette_id: PALETTE_2,
                        rows: [1, 0, 0, 0, 0, 0, 0, 0],
                        colsb: [64, 0, 0, 0, 0, 0, 0, 0],
                    };
                    let mut zscope = _tile_loadconfig(&zcfg).expect("valid descriptor");
                    let zt = zscope.tile(0).unwrap();
                    zscope.tile_bytes_mut(zt).copy_from_slice(&desc);
                    _tile_zero(&mut zscope, zt);
                    let zdesc = crate::native::encode_tilecfg(PALETTE_2, &zcfg.rows, &zcfg.colsb);
                    // SAFETY: capability confirmed above.
                    let zgot = unsafe { crate::native::tile_zero_hw(&zdesc, &desc) };
                    let zero_ok = zgot.as_slice() == zscope.tile_bytes_ref(zt);

                    return TestResult::from_bool(cfg_ok && zero_ok);
                }
            }
            let _ = &cfg;
            TestResult::discard()
        }
    }
}
