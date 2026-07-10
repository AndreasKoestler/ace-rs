//! ACE group-4 family A: tile configuration lifecycle, wrapped in the crate's RAII guard.
//!
//! This module models the palette-2 (ACE) tile register file and the four family-A
//! lifecycle instructions:
//!
//! * `LDTILECFG` — [`_tile_loadconfig`] validates a [`TileConfig`] descriptor, zeroes the
//!   tile data, sets the Block Scale register to its INIT (`0x7F`) state, and returns the
//!   [`TileScope`] guard (spec section 11.2)
//!   (`[ace-tile-instructions.TILE_LIFECYCLE.1]`).
//! * `STTILECFG` — [`_tile_storeconfig`] stores the current configuration: byte 0 = the
//!   palette id, bytes 1-63 = 0 for palette 2; all 64 bytes zero when unconfigured (spec
//!   section 11.3.4) (`[ace-tile-instructions.TILE_LIFECYCLE.2]`).
//! * `TILEZERO` — [`_tile_zero`] clears the addressed tile to zero and nothing else (spec
//!   section 11.1) (`[ace-tile-instructions.TILE_LIFECYCLE.3]`).
//! * `TILERELEASE` — run by `impl Drop for TileScope`: zeroes all tile data, returns the
//!   Block Scale register to INIT (`0x7F` — NOT zero, spec section 11.4.3), selects
//!   palette 0, exactly once, including on panic unwind (spec section 11.4)
//!   (`[ace-tile-instructions.TILE_LIFECYCLE.4]`,
//!   `[ace-tile-instructions.TILE_LIFECYCLE.5]`). There is no free-standing
//!   `_tile_release` call: release is the guard's `Drop`, so configuration can never leak
//!   (INV-1).
//!
//! # Architecture (spec sections 10.2 and 15.2.2.3)
//!
//! ACE tiles have FIXED dimensions: eight tile registers, each 512 bits x 16 rows
//! (16 rows x 64 bytes = 16x16 32-bit elements). The palette-2 descriptor carries NO
//! per-tile fields — byte 0 selects the palette and bytes 1-63 are reserved, must be zero.
//! `LDTILECFG` raises `#GP` on an unsupported palette id or non-zero reserved bytes (spec
//! sections 11.2.5 and 15.2.2.4), modeled by [`TileConfigError`].
//!
//! # Lifecycle state model
//!
//! ```text
//!   Uninitialized --Acquire (_tile_loadconfig)--> Configured
//!   Configured    --Zero / Store config (self-loops)--> Configured
//!   Configured    --Release (Drop, incl. panic unwind)--> Released
//! ```
//!
//! A [`TileScope`] owns its register model outright (the eight tiles plus the single Block
//! Scale register) and holds no global mutable state, so independent guards — nested or
//! sequential — never interfere and never leak configuration
//! (`[ace-tile-instructions.TILE_LIFECYCLE.7]`).
//!
//! # Dispatch
//!
//! Each lifecycle op is a safe public dispatcher plus a cfg-free `_scalar` oracle (the
//! primary path, correct on every target including non-x86). Family A's native gate is
//! AMX-TILE (`detect::has_amx_tile`, `[ace-tile-instructions.DETECT.1-1]`). The native
//! C-intrinsic shims live in `src/native/ace_tile.c` and are exercised by the per-module
//! differential tests; the public dispatchers themselves always take the scalar oracle (the
//! register model lives in Rust, not in live CPU tile state)
//! (`[ace-tile-instructions.DISPATCH.1]`).

use crate::bsr::BsrReg;
use crate::detect;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Number of architected tile registers (`TMM0..=TMM7`, spec section 10.2.1).
pub const MAX_TILES: usize = 8;

/// Fixed row count of every ACE tile (spec section 10.2.1).
pub const TILE_ROWS: usize = 16;

/// Fixed bytes-per-row of every ACE tile: 512 bits (spec section 10.2.1).
pub const TILE_COLSB: usize = 64;

/// Fixed byte size of one tile's data (16 rows x 64 bytes).
pub const TILE_BYTES: usize = TILE_ROWS * TILE_COLSB;

/// Monotonic source of per-scope tokens: every [`TileScope`] gets a distinct token, and the
/// handles it mints ([`TileId`]) carry it, so a handle used against a *different* live scope
/// is caught at the accessor (a programmer error the lifetime system alone cannot express —
/// two scopes in one block share a region).
static NEXT_SCOPE_TOKEN: AtomicUsize = AtomicUsize::new(0);

/// The INIT palette (spec section 15.2.1).
const PALETTE_INIT: u8 = 0;
/// The ACE palette (spec section 15.2.1).
const PALETTE_ACE: u8 = 2;

/// A 64-byte `LDTILECFG` tile configuration descriptor (spec section 11.2.3).
///
/// For palette 2 (ACE) the descriptor is byte 0 = palette id and bytes 1-63 reserved, must
/// be zero — ACE tiles have fixed dimensions and NO per-tile fields (spec section
/// 15.2.2.3). The palette-0 (INIT) descriptor is 64 bytes of zero. This model supports
/// palettes 0 and 2; palette 1 (AMX TMUL) is a different register-file shape outside this
/// crate's scope and is reported as unsupported, matching an ACE-only implementation
/// (palette support above 0 is implementation-defined, spec section 15.2.2.4).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TileConfig {
    /// Palette identifier (descriptor byte 0).
    pub palette_id: u8,
    /// Descriptor bytes 1-63: reserved, must be zero.
    pub reserved: [u8; 63],
}

impl TileConfig {
    /// The palette-2 (ACE) descriptor: byte 0 = 2, bytes 1-63 = 0.
    pub fn ace() -> Self {
        TileConfig {
            palette_id: PALETTE_ACE,
            reserved: [0; 63],
        }
    }

    /// The palette-0 (INIT) descriptor: 64 bytes of zero (spec section 15.2.2.1).
    pub fn init() -> Self {
        TileConfig {
            palette_id: PALETTE_INIT,
            reserved: [0; 63],
        }
    }

    /// The raw 64-byte descriptor this configuration encodes.
    pub fn to_bytes(&self) -> [u8; 64] {
        let mut buf = [0u8; 64];
        buf[0] = self.palette_id;
        buf[1..].copy_from_slice(&self.reserved);
        buf
    }

    /// Parse a raw 64-byte descriptor.
    pub fn from_bytes(bytes: &[u8; 64]) -> Self {
        let mut reserved = [0u8; 63];
        reserved.copy_from_slice(&bytes[1..]);
        TileConfig {
            palette_id: bytes[0],
            reserved,
        }
    }

    /// Validate the descriptor the way `LDTILECFG` does (spec sections 11.2.5 and
    /// 15.2.2.4): the palette must be supported (0 or 2 in this ACE-only model) and every
    /// reserved byte must be zero. Palette is checked first, so an all-bad descriptor
    /// reports [`TileConfigError::UnsupportedPalette`].
    fn validate(&self) -> Result<(), TileConfigError> {
        if self.palette_id != PALETTE_INIT && self.palette_id != PALETTE_ACE {
            return Err(TileConfigError::UnsupportedPalette);
        }
        if self.reserved.iter().any(|&b| b != 0) {
            return Err(TileConfigError::NonZeroReserved);
        }
        Ok(())
    }
}

/// Rejection reasons — the `#GP` conditions of `LDTILECFG` (spec sections 11.2.1, 11.2.5
/// and 15.2.2.4) (`[ace-tile-instructions.TILE_LIFECYCLE.1]`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TileConfigError {
    /// The palette id is not supported by the implementation (this ACE-only model supports
    /// palettes 0 and 2).
    UnsupportedPalette,
    /// A reserved descriptor byte (1-63) was non-zero.
    NonZeroReserved,
}

/// One fixed-size 16x64-byte tile, row-major. Only reachable through the owning
/// [`TileScope`].
#[derive(Clone, Debug)]
struct Tile {
    bytes: [u8; TILE_BYTES],
}

impl Tile {
    fn zeroed() -> Self {
        Tile {
            bytes: [0u8; TILE_BYTES],
        }
    }
}

/// A handle addressing one tile of a [`TileScope`].
///
/// A `TileId` is minted only by [`TileScope::tile`] — it has no public constructor, so it
/// cannot be forged to bypass the guard. It is bound to the scope that minted it: it
/// carries that scope's token, and every accessor panics if the handle is presented to a
/// *different* scope (a programmer error; two live scopes in one block defeat a purely
/// lifetime-based check). Because every tile operation takes the scope, a handle is also
/// useless once the guard has released its configuration: dropping the scope and then using
/// the handle does not compile (INV-2, `[ace-tile-instructions.TILE_LIFECYCLE.6]`):
///
/// ```compile_fail
/// use ace::{_tile_loadconfig, _tile_zero, TileConfig};
/// let mut scope = _tile_loadconfig(&TileConfig::ace()).unwrap();
/// let id = scope.tile(0).unwrap();
/// drop(scope); // TILERELEASE: the guard releases its configuration here
/// _tile_zero(&mut scope, id); // ERROR: use of moved value `scope` — the handle is unusable
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TileId {
    index: usize,
    /// Token of the [`TileScope`] that minted this handle (see [`NEXT_SCOPE_TOKEN`]).
    scope_token: usize,
}

/// The RAII tile lifecycle guard (`[ace-tile-instructions.TILE_LIFECYCLE.5]`).
///
/// Constructed by [`_tile_loadconfig`] (Acquire), it owns the eight fixed 16x64 tiles and
/// the single Block Scale register. Its `Drop` runs `TILERELEASE` (spec section 11.4):
/// tile data zeroed, BSR returned to INIT (`0x7F`), palette 0 selected — exactly once,
/// including on panic unwind, so configuration never leaks (INV-1). The guard holds no
/// global mutable state, so nested or sequential guards are independent
/// (`[ace-tile-instructions.TILE_LIFECYCLE.7]`).
#[derive(Debug)]
pub struct TileScope {
    /// The configured palette (0 = INIT, 2 = ACE).
    palette_id: u8,
    /// `TILES_CONFIGURED` (spec section 11.2.5): set when a non-INIT palette is loaded.
    configured: bool,
    tiles: [Tile; MAX_TILES],
    /// The single ACE Block Scale register (SCALEDATA), owned by the guard so BSR writes
    /// and the MX products' reads share one register model (INV-5).
    bsr: BsrReg,
    /// Set once by `TileScope::release` so the release logic is idempotent.
    released: bool,
    /// Release ledger: when attached, `Drop` increments it exactly once, letting a caller
    /// confirm the RAII `TILERELEASE` fired — including across a panic unwind, where the
    /// released guard's own state is gone but the ledger survives. `None` in ordinary use.
    release_ledger: Option<Arc<AtomicUsize>>,
    /// This scope's distinct token; stamped into every handle it mints (see
    /// [`NEXT_SCOPE_TOKEN`]).
    token: usize,
}

impl TileScope {
    /// Mint a handle to tile `index` (`0..MAX_TILES`), or `None` if out of range. The only
    /// way to obtain a [`TileId`]; the returned handle borrows nothing, so it can be passed
    /// to the `&mut self`-style lifecycle ops.
    pub fn tile(&self, index: usize) -> Option<TileId> {
        if index < MAX_TILES {
            Some(TileId {
                index,
                scope_token: self.token,
            })
        } else {
            None
        }
    }

    /// Panic unless `id` was minted by this scope — the cross-scope guard every tile
    /// accessor runs (see [`TileId`]).
    fn check_tile(&self, id: TileId) {
        assert_eq!(
            id.scope_token, self.token,
            "TileId used with a TileScope other than the one that minted it"
        );
    }

    /// Shared read accessor for the Block Scale register.
    pub fn bsr(&self) -> &BsrReg {
        &self.bsr
    }

    /// Mutable accessor for the Block Scale register, used by the family-D `BSRINIT` /
    /// `BSRMOV{F,H,L}` oracles.
    pub(crate) fn bsr_mut(&mut self) -> &mut BsrReg {
        &mut self.bsr
    }

    /// Shared read-back accessor for the addressed tile's raw backing bytes (row-major,
    /// 16 rows x 64 bytes), used by the move/convert oracles to extract rows.
    pub(crate) fn tile_bytes_ref(&self, id: TileId) -> &[u8; TILE_BYTES] {
        self.check_tile(id);
        &self.tiles[id.index].bytes
    }

    /// Mutable register-model accessor for the addressed tile's raw backing bytes, used by
    /// the write-form move and outer-product oracles.
    pub(crate) fn tile_bytes_mut(&mut self, id: TileId) -> &mut [u8; TILE_BYTES] {
        self.check_tile(id);
        &mut self.tiles[id.index].bytes
    }

    /// `TILERELEASE` (Drop-only, spec section 11.4.3): zero all tile data, return the BSR
    /// to INIT (`0x7F` — explicitly NOT zeroed), select palette 0, idempotently; bumps the
    /// release ledger if one is attached.
    fn release(&mut self) {
        if self.released {
            return;
        }
        self.released = true;
        for tile in &mut self.tiles {
            tile.bytes = [0u8; TILE_BYTES];
        }
        // SCALEDATA -> 0x7F (E8M0 = 2^0 = 1.0), not zeroed (spec section 11.4.3).
        self.bsr = BsrReg::init();
        self.palette_id = PALETTE_INIT;
        self.configured = false;
        if let Some(ledger) = &self.release_ledger {
            ledger.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// Attach a release ledger (test observability for the RAII `TILERELEASE`).
    #[cfg(test)]
    fn set_release_ledger(&mut self, ledger: Arc<AtomicUsize>) {
        self.release_ledger = Some(ledger);
    }
}

impl Drop for TileScope {
    /// `TILERELEASE` on scope exit — normal return, early return, or panic unwind — so tile
    /// configuration can never leak (INV-1, `[ace-tile-instructions.TILE_LIFECYCLE.4]`,
    /// `[ace-tile-instructions.TILE_LIFECYCLE.5]`). Runs exactly once (Rust ownership +
    /// `TileScope::release`'s idempotence guard).
    fn drop(&mut self) {
        self.release();
    }
}

/// `LDTILECFG`: validate a descriptor and return the configured [`TileScope`] guard, or
/// `Err` (the `#GP` model) with no tile state established (spec section 11.2)
/// (`[ace-tile-instructions.TILE_LIFECYCLE.1]`).
///
/// On success (spec section 11.2.1 note): TILEDATA bytes are set to 0 and the SCALEDATA
/// (Block Scale Register) bytes are set to the INIT (`0x7F`) state.
pub fn _tile_loadconfig(config: &TileConfig) -> Result<TileScope, TileConfigError> {
    let _ = detect::has_amx_tile; // family A native gate: AMX-TILE [DETECT.1-1]
    _tile_loadconfig_scalar(config)
}

/// Portable `LDTILECFG` oracle — the section-11.2.5 pseudocode: validate before
/// establishing any state (unsupported palette or non-zero reserved bytes -> `#GP`), then
/// zero all tile data, set the BSR to INIT (`0x7F`), and set `TILES_CONFIGURED` per the
/// loaded palette (0 for the INIT palette, 1 for palette 2).
pub fn _tile_loadconfig_scalar(config: &TileConfig) -> Result<TileScope, TileConfigError> {
    config.validate()?;
    Ok(TileScope {
        palette_id: config.palette_id,
        configured: config.palette_id != PALETTE_INIT,
        tiles: core::array::from_fn(|_| Tile::zeroed()),
        bsr: BsrReg::init(),
        released: false,
        release_ledger: None,
        token: NEXT_SCOPE_TOKEN.fetch_add(1, Ordering::Relaxed),
    })
}

/// `STTILECFG`: store the current tile configuration (spec section 11.3.4). For palette 2
/// the stored descriptor is byte 0 = 2 and bytes 1-63 = 0 (per-tile fields are written only
/// for palette 1); when `TILES_CONFIGURED == 0` all 64 bytes are zero
/// (`[ace-tile-instructions.TILE_LIFECYCLE.2]`).
pub fn _tile_storeconfig(scope: &TileScope) -> TileConfig {
    let _ = detect::has_amx_tile; // family A gate site [DETECT.1-1]
    _tile_storeconfig_scalar(scope)
}

/// Portable `STTILECFG` oracle — the section-11.3.4 pseudocode.
pub fn _tile_storeconfig_scalar(scope: &TileScope) -> TileConfig {
    if !scope.configured {
        return TileConfig::init();
    }
    TileConfig {
        palette_id: scope.palette_id,
        reserved: [0; 63], // ACE palette: bytes 1-63 = 0
    }
}

/// `TILEZERO`: clear only the addressed tile to all-zero (spec section 11.1.4)
/// (`[ace-tile-instructions.TILE_LIFECYCLE.3]`).
pub fn _tile_zero(scope: &mut TileScope, dst: TileId) {
    let _ = detect::has_amx_tile; // family A gate site [DETECT.1-1]
    _tile_zero_scalar(scope, dst);
}

/// Portable `TILEZERO` oracle — zeroes the addressed tile's bytes and leaves every other
/// tile untouched.
pub fn _tile_zero_scalar(scope: &mut TileScope, dst: TileId) {
    scope.tile_bytes_mut(dst).fill(0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bsr::BSR_INIT_BYTE;

    /// LDTILECFG `#GP` model (spec sections 11.2.5 / 15.2.2.4): non-zero reserved bytes and
    /// unsupported palettes are rejected; palette priority puts UnsupportedPalette first.
    /// `tile::descriptor_gp_conditions`
    #[test]
    fn descriptor_gp_conditions() {
        // The ACE descriptor and the INIT descriptor are both valid.
        assert!(_tile_loadconfig(&TileConfig::ace()).is_ok());
        assert!(_tile_loadconfig(&TileConfig::init()).is_ok());

        // Non-zero reserved byte -> #GP (NonZeroReserved), anywhere in 1-63.
        let mut bad = TileConfig::ace();
        bad.reserved[0] = 1; // descriptor byte 1
        assert_eq!(
            _tile_loadconfig(&bad).unwrap_err(),
            TileConfigError::NonZeroReserved
        );
        let mut bad_last = TileConfig::ace();
        bad_last.reserved[62] = 0xFF; // descriptor byte 63
        assert_eq!(
            _tile_loadconfig(&bad_last).unwrap_err(),
            TileConfigError::NonZeroReserved
        );

        // Unsupported palette -> #GP; checked before the reserved bytes.
        let mut bad_palette = TileConfig::ace();
        bad_palette.palette_id = 3;
        bad_palette.reserved[5] = 7;
        assert_eq!(
            _tile_loadconfig(&bad_palette).unwrap_err(),
            TileConfigError::UnsupportedPalette,
            "palette error takes priority over reserved-byte errors"
        );
        // Palette 1 (TMUL) is unsupported in this ACE-only model.
        let mut tmul = TileConfig::ace();
        tmul.palette_id = 1;
        assert_eq!(
            _tile_loadconfig(&tmul).unwrap_err(),
            TileConfigError::UnsupportedPalette
        );
    }

    /// Acquire establishes the architected init state (spec section 11.2.1): all tile data
    /// zero, BSR bytes at INIT `0x7F`, fixed 16x64 tiles.
    /// `tile::acquire_initializes_state`
    #[test]
    fn acquire_initializes_state() {
        let scope = _tile_loadconfig(&TileConfig::ace()).unwrap();
        for t in 0..MAX_TILES {
            let id = scope.tile(t).unwrap();
            assert_eq!(scope.tile_bytes_ref(id).len(), TILE_BYTES);
            assert!(
                scope.tile_bytes_ref(id).iter().all(|&b| b == 0),
                "tile {t} data zeroed on Acquire"
            );
        }
        assert!(
            scope.bsr().bytes().iter().all(|&b| b == BSR_INIT_BYTE),
            "SCALEDATA set to INIT (0x7F) on Acquire"
        );
    }

    /// STTILECFG stores byte 0 = palette and a zero body for palette 2 (spec section
    /// 11.3.4) — NOT per-tile fields — and 64 zero bytes when unconfigured.
    /// `tile::storeconfig_palette2_layout`
    #[test]
    fn storeconfig_palette2_layout() {
        let scope = _tile_loadconfig(&TileConfig::ace()).unwrap();
        let stored = _tile_storeconfig(&scope);
        assert_eq!(stored, TileConfig::ace());
        let raw = stored.to_bytes();
        assert_eq!(raw[0], 2);
        assert!(
            raw[1..].iter().all(|&b| b == 0),
            "bytes 1-63 stored as zero"
        );

        // INIT palette: TILES_CONFIGURED == 0 -> all-zero descriptor.
        let init_scope = _tile_loadconfig(&TileConfig::init()).unwrap();
        assert_eq!(_tile_storeconfig(&init_scope).to_bytes(), [0u8; 64]);
    }

    /// TILEZERO clears only the addressed tile; every other tile is unchanged
    /// (`[ace-tile-instructions.TILE_LIFECYCLE.3]`).
    /// `tile::zero_clears_only_addressed_tile`
    #[test]
    fn zero_clears_only_addressed_tile() {
        let mut scope = _tile_loadconfig(&TileConfig::ace()).unwrap();
        scope.tiles[0].bytes.fill(0xAB);
        scope.tiles[1].bytes.fill(0xCD);
        let t0 = scope.tile(0).unwrap();
        let t1 = scope.tile(1).unwrap();

        _tile_zero(&mut scope, t0);

        assert!(
            scope.tile_bytes_ref(t0).iter().all(|&b| b == 0),
            "addressed tile 0 is now all-zero"
        );
        assert!(
            scope.tile_bytes_ref(t1).iter().all(|&b| b == 0xCD),
            "tile 1 is untouched by zeroing tile 0"
        );
    }

    /// TILERELEASE returns the BSR to INIT `0x7F` — explicitly NOT zero (spec section
    /// 11.4.3) — and zeroes tile data, exactly once, including on panic unwind.
    /// `tile::release_semantics`
    #[test]
    fn release_semantics() {
        use std::panic::{catch_unwind, AssertUnwindSafe};

        // Direct release observation.
        let mut scope = _tile_loadconfig(&TileConfig::ace()).unwrap();
        scope.tiles[0].bytes.fill(0x77);
        *scope.bsr_mut() = crate::bsr::BsrReg::init();
        scope.bsr_mut().bytes_mut().fill(0x11);
        scope.release();
        assert!(scope.tiles[0].bytes.iter().all(|&b| b == 0));
        assert!(
            scope.bsr.bytes().iter().all(|&b| b == BSR_INIT_BYTE),
            "release sets SCALEDATA to 0x7F, not zero"
        );
        assert_eq!(scope.palette_id, PALETTE_INIT, "palette 0 selected");

        // Panic-unwind release, observed through the ledger.
        let ledger = Arc::new(AtomicUsize::new(0));
        let ledger_for_guard = ledger.clone();
        let result = catch_unwind(AssertUnwindSafe(|| {
            let mut s = _tile_loadconfig(&TileConfig::ace()).unwrap();
            s.set_release_ledger(ledger_for_guard);
            panic!("boom mid-lifecycle");
        }));
        assert!(result.is_err(), "the closure panicked");
        assert_eq!(
            ledger.load(Ordering::SeqCst),
            1,
            "TILERELEASE ran exactly once on panic unwind"
        );
    }

    /// A `TileId` presented to a scope other than the one that minted it is a programmer
    /// error and panics at the accessor instead of silently addressing the wrong register
    /// file.
    /// `tile::cross_scope_handle_panics`
    #[test]
    #[should_panic(expected = "TileId used with a TileScope other than the one that minted it")]
    fn cross_scope_handle_panics() {
        let scope_a = _tile_loadconfig(&TileConfig::ace()).unwrap();
        let mut scope_b = _tile_loadconfig(&TileConfig::ace()).unwrap();
        let id_from_a = scope_a.tile(0).unwrap();
        _tile_zero(&mut scope_b, id_from_a);
    }

    /// Nested and sequential guards do not leak configuration: each owns an independent
    /// register model, and each Acquire starts from clean, zeroed tiles
    /// (`[ace-tile-instructions.TILE_LIFECYCLE.7]`).
    /// `tile::nested_sequential_guards_do_not_leak`
    #[test]
    fn nested_sequential_guards_do_not_leak() {
        {
            let mut first = _tile_loadconfig(&TileConfig::ace()).unwrap();
            first.tiles[0].bytes.fill(0xFF);
        } // first drops here -> release
        let second = _tile_loadconfig(&TileConfig::ace()).unwrap();
        assert!(
            second
                .tile_bytes_ref(second.tile(0).unwrap())
                .iter()
                .all(|&b| b == 0),
            "a fresh guard does not inherit the previous guard's tile bytes"
        );

        // Nested: two live guards are independent — mutating one leaves the other untouched.
        let mut outer = _tile_loadconfig(&TileConfig::ace()).unwrap();
        outer.tiles[0].bytes.fill(0x11);
        {
            let mut inner = _tile_loadconfig(&TileConfig::ace()).unwrap();
            let inner_t0 = inner.tile(0).unwrap();
            _tile_zero(&mut inner, inner_t0);
            inner.tiles[1].bytes.fill(0x22);
        }
        assert!(
            outer
                .tile_bytes_ref(outer.tile(0).unwrap())
                .iter()
                .all(|&b| b == 0x11),
            "the outer guard is unaffected by the inner guard's lifetime"
        );
    }
}
