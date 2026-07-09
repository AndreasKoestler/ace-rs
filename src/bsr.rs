//! ACE group-4 family D: block-scale (`BSR`) registers.
//!
//! This module models the ACE-new block-scale register file and the four family-D
//! instructions that seed and move the per-block microscaling exponents the MX-FP8 outer
//! products (family E, phase 7) consume:
//!
//! * `BSRINIT` — [`_tile_bsrinit`] seeds an addressed [`BsrReg`] with a full set of per-block
//!   scale exponents (`[ace-tile-instructions.BSR.1]`).
//! * `BSRMOVF` — [`_tile_bsrmovf`] moves the FULL block-scale factor of one addressed block
//!   (`[ace-tile-instructions.BSR.2]`).
//! * `BSRMOVH` — [`_tile_bsrmovh`] moves the HIGH portion of one addressed block's factor
//!   (`[ace-tile-instructions.BSR.3]`).
//! * `BSRMOVL` — [`_tile_bsrmovl`] moves the LOW portion of one addressed block's factor
//!   (`[ace-tile-instructions.BSR.4]`).
//!
//! Only the addressed [`BsrReg`] changes; the rest of the block-scale file is untouched. The
//! file lives inside the [`TileScope`] guard, so a later MX outer product reads back exactly
//! the block scale these ops wrote (INV-5, `[ace-tile-instructions.BSR.4-1]`) through the
//! shared register-model accessor [`TileScope::bsr_scale`].
//!
//! # BSR scale-field layout (OQ-3 assumption, grounded against ACE v1 §12)
//!
//! The exact `BSR` scale-field bit layout is not confirmed against the ACE v1 rev-1.15 PDF
//! (OQ-1/OQ-3). Grounded against ACE v1 §12 (block-scale registers), a [`BsrReg`] holds
//! [`BSR_SCALE_BLOCKS`] per-block microscaling **scale exponents**, one `u8` each — the MX
//! (microscaling) E8M0 shared-exponent-per-block convention §2.4/§12 describes. A block-scale
//! *factor* is therefore one 8-bit exponent byte, and the three `BSRMOV` forms address it at
//! byte / nibble granularity:
//!
//! * **F (full)** — the complete 8-bit exponent of one block.
//! * **H (high)** — the high nibble (bits `7:4`) of one block's exponent, low nibble preserved.
//! * **L (low)** — the low nibble (bits `3:0`) of one block's exponent, high nibble preserved.
//!
//! This is the single place to correct once the field layout is pinned against the spec; the
//! oracle's write/read behaviour (which block / which nibble a factor lands in) is what the
//! hand-value pins and the write/read-consistency property lock down.
//!
//! # Dispatch
//!
//! Each op is a safe public dispatcher plus a cfg-free `_scalar` oracle (the primary path,
//! correct on every target). Family D is `ACE`-only and gates on full `ACE`
//! ([`detect::has_ace`], `[ace-tile-instructions.DETECT.1-3]`,
//! `[ace-tile-instructions.DISPATCH.1]`). No native tile shim exists yet — the native path is
//! layer-3-blocked until Intel SDE gains ACE emulation (OQ-6, wired in phase 8) — so, exactly
//! as the oracle-only group-3 modules and the sibling tile families do, the dispatchers
//! reference the detector to mark the gate site and take the scalar oracle on every target.
//!
//! # OQ-3 (public surface) surfaced here
//!
//! The default is realised: first-class `_tile_bsr*` primitives plus a [`BsrId`] handle,
//! minted only by [`TileScope::bsr`], that the MX products (phase 7) accept to read the block
//! scale back. The block-scale file is owned by the guard (added to [`TileScope`] here,
//! fulfilling the phase-1 deferral), so it shares the tile file's RAII lifecycle.

use crate::detect;
use crate::tile::TileScope;

/// Number of addressable block-scale registers in the file (`BSR0..=BSR7`), mirroring the
/// eight tiles of a palette-2 configuration.
pub const NUM_BSR: usize = 8;

/// Number of per-block scale exponents one [`BsrReg`] holds (canonical width, OQ-3/OQ-8). Each
/// entry is one `u8` E8M0-style microscaling exponent for one block of an MX operand row.
pub const BSR_SCALE_BLOCKS: usize = 4;

/// One block-scale register: [`BSR_SCALE_BLOCKS`] per-block microscaling scale exponents
/// (`u8` each), the values the MX-FP8 outer products apply per block (INV-5).
///
/// The layout is the OQ-3 assumption grounded against ACE v1 §12 (see the module docs): a
/// block-scale factor is one 8-bit exponent, addressed full / high-nibble / low-nibble by the
/// `BSRMOV{F,H,L}` forms. Only reachable through the owning [`TileScope`].
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct BsrReg {
    scales: [u8; BSR_SCALE_BLOCKS],
}

/// A handle addressing one block-scale register of a [`TileScope`].
///
/// A `BsrId` is minted only by [`TileScope::bsr`] — it has no public constructor, so it cannot
/// be forged to bypass the guard (OQ-3), mirroring [`crate::tile::TileId`]. It is the handle
/// the MX-FP8 outer products (phase 7) accept to read the block scale these ops wrote (INV-5).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BsrId {
    index: usize,
}

impl BsrId {
    /// Mint a handle for register `index`; called only by [`TileScope::bsr`], which bounds
    /// `index` to `0..NUM_BSR` (no public constructor).
    pub(crate) fn new(index: usize) -> Self {
        BsrId { index }
    }

    /// The addressed register's index into the guard-owned file.
    pub(crate) fn index(&self) -> usize {
        self.index
    }
}

impl BsrReg {
    /// A cleared block-scale register (every block exponent `0`). Used on Acquire and reset on
    /// Release so a released or freshly-configured guard observes a clean file.
    pub(crate) fn zeroed() -> Self {
        BsrReg {
            scales: [0u8; BSR_SCALE_BLOCKS],
        }
    }

    /// The full 8-bit scale exponent of `block`, or [`None`] if `block` is out of range.
    pub(crate) fn scale(&self, block: usize) -> Option<u8> {
        self.scales.get(block).copied()
    }

    /// All per-block scale exponents, for the MX outer products (phase 7) to read the block
    /// scale back verbatim (INV-5).
    pub(crate) fn scales(&self) -> &[u8; BSR_SCALE_BLOCKS] {
        &self.scales
    }
}

// Forward-reference the read side of the BSR register model that phase 7's MX-FP8 outer
// products consume to read back the block scale these ops wrote (INV-5,
// `[ace-tile-instructions.BSR.4-1]`). These guard-owned read accessors have no non-test caller
// yet, so — exactly as `src/tile.rs` marks its delivered-but-not-yet-consumed detector/codec
// items — a `const _` binding keeps them "used" without a runtime effect and without
// lint-muting (`#[allow]` is forbidden in production):
//   * `TileScope::bsr_reg` / `BsrReg::scales` — the whole-register read the MX products apply
//     per block.
//   * `TileScope::bsr_scale` / `BsrReg::scale` — the single-block read the family-D tests use.
const _: () = {
    let _ = TileScope::bsr_reg;
    let _ = TileScope::bsr_scale;
    let _ = BsrReg::scale;
    let _ = BsrReg::scales;
};

// ---------------------------------------------------------------------------------------------
// BSRINIT — seed a block-scale register with per-block scale exponents
// ---------------------------------------------------------------------------------------------

/// `BSRINIT` (`[ace-tile-instructions.BSR.1]`): seed the addressed [`BsrReg`] with a full set
/// of per-block scale exponents. Only the addressed register changes.
///
/// Family D is `ACE`-only and gates on full `ACE` (`[ace-tile-instructions.DETECT.1-3]`,
/// `[ace-tile-instructions.DISPATCH.1]`); with no native shim yet (OQ-6) the detector marks
/// the gate site and the scalar oracle runs on every target.
pub fn _tile_bsrinit(scope: &mut TileScope, dst: BsrId, exponents: [u8; BSR_SCALE_BLOCKS]) {
    let _ = detect::has_ace; // family D gate: full ACE [DETECT.1-3]
    _tile_bsrinit_scalar(scope, dst, exponents);
}

/// Portable `BSRINIT` oracle — the primary, always-correct path. Overwrites the addressed
/// register's per-block exponents; `BSRINIT` with the same seed is idempotent.
pub fn _tile_bsrinit_scalar(scope: &mut TileScope, dst: BsrId, exponents: [u8; BSR_SCALE_BLOCKS]) {
    scope.bsr_reg_mut(dst).scales = exponents;
}

// ---------------------------------------------------------------------------------------------
// BSRMOVF — move the full block-scale factor of one addressed block
// ---------------------------------------------------------------------------------------------

/// `BSRMOVF` (`[ace-tile-instructions.BSR.2]`): move the FULL 8-bit block-scale factor into
/// `block` of the addressed [`BsrReg`], or [`None`] (mutating nothing) if `block` is outside
/// [`BSR_SCALE_BLOCKS`]. Gates as [`_tile_bsrinit`].
pub fn _tile_bsrmovf(scope: &mut TileScope, dst: BsrId, block: usize, factor: u8) -> Option<()> {
    let _ = detect::has_ace; // family D gate: full ACE [DETECT.1-3]
    _tile_bsrmovf_scalar(scope, dst, block, factor)
}

/// Portable `BSRMOVF` oracle — writes the full exponent byte of the addressed block.
pub fn _tile_bsrmovf_scalar(
    scope: &mut TileScope,
    dst: BsrId,
    block: usize,
    factor: u8,
) -> Option<()> {
    let reg = scope.bsr_reg_mut(dst);
    *reg.scales.get_mut(block)? = factor;
    Some(())
}

// ---------------------------------------------------------------------------------------------
// BSRMOVH — move the high portion (high nibble) of one addressed block's factor
// ---------------------------------------------------------------------------------------------

/// `BSRMOVH` (`[ace-tile-instructions.BSR.3]`): move the HIGH portion of the block-scale
/// factor — the high nibble, bits `7:4` — into `block`, preserving its low nibble. The low 4
/// bits of `high` are the payload. Returns [`None`] (mutating nothing) if `block` is out of
/// range. Gates as [`_tile_bsrinit`].
pub fn _tile_bsrmovh(scope: &mut TileScope, dst: BsrId, block: usize, high: u8) -> Option<()> {
    let _ = detect::has_ace; // family D gate: full ACE [DETECT.1-3]
    _tile_bsrmovh_scalar(scope, dst, block, high)
}

/// Portable `BSRMOVH` oracle — replaces bits `7:4` of the addressed block's exponent with the
/// low nibble of `high`; the low nibble (bits `3:0`) is left unchanged.
pub fn _tile_bsrmovh_scalar(
    scope: &mut TileScope,
    dst: BsrId,
    block: usize,
    high: u8,
) -> Option<()> {
    let reg = scope.bsr_reg_mut(dst);
    let slot = reg.scales.get_mut(block)?;
    *slot = (*slot & 0x0F) | ((high & 0x0F) << 4);
    Some(())
}

// ---------------------------------------------------------------------------------------------
// BSRMOVL — move the low portion (low nibble) of one addressed block's factor
// ---------------------------------------------------------------------------------------------

/// `BSRMOVL` (`[ace-tile-instructions.BSR.4]`): move the LOW portion of the block-scale factor
/// — the low nibble, bits `3:0` — into `block`, preserving its high nibble. The low 4 bits of
/// `low` are the payload. Returns [`None`] (mutating nothing) if `block` is out of range.
/// Gates as [`_tile_bsrinit`].
pub fn _tile_bsrmovl(scope: &mut TileScope, dst: BsrId, block: usize, low: u8) -> Option<()> {
    let _ = detect::has_ace; // family D gate: full ACE [DETECT.1-3]
    _tile_bsrmovl_scalar(scope, dst, block, low)
}

/// Portable `BSRMOVL` oracle — replaces bits `3:0` of the addressed block's exponent with the
/// low nibble of `low`; the high nibble (bits `7:4`) is left unchanged.
pub fn _tile_bsrmovl_scalar(
    scope: &mut TileScope,
    dst: BsrId,
    block: usize,
    low: u8,
) -> Option<()> {
    let reg = scope.bsr_reg_mut(dst);
    let slot = reg.scales.get_mut(block)?;
    *slot = (*slot & 0xF0) | (low & 0x0F);
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile::{_tile_loadconfig, TileConfig};

    /// A live palette-2 guard whose BSR file the family-D ops address (a minimal single-tile
    /// descriptor; family D touches only the block-scale file, not the tiles).
    fn scope() -> TileScope {
        let cfg = TileConfig {
            palette_id: 2,
            rows: [1, 0, 0, 0, 0, 0, 0, 0],
            colsb: [4, 0, 0, 0, 0, 0, 0, 0],
        };
        _tile_loadconfig(&cfg).unwrap()
    }

    /// `BSRINIT` seeds the addressed register with exactly the given per-block exponents, read
    /// back through the shared model; a fresh register is all-zero before the seed
    /// (`[ace-tile-instructions.BSR.1]`).
    /// `bsr::bsrinit_seeds_exponents`
    #[test]
    fn bsrinit_seeds_exponents() {
        let mut scope = scope();
        let b0 = scope.bsr(0).unwrap();

        // Fresh register: every block exponent is zero before any seed.
        for block in 0..BSR_SCALE_BLOCKS {
            assert_eq!(scope.bsr_scale(b0, block), Some(0));
        }

        let exps: [u8; BSR_SCALE_BLOCKS] = [127, 128, 1, 254];
        _tile_bsrinit(&mut scope, b0, exps);
        for (block, &want) in exps.iter().enumerate() {
            assert_eq!(
                scope.bsr_scale(b0, block),
                Some(want),
                "seeded block {block} exponent reads back"
            );
        }

        // Idempotent: re-seeding with the same exponents leaves the register unchanged.
        _tile_bsrinit(&mut scope, b0, exps);
        for (block, &want) in exps.iter().enumerate() {
            assert_eq!(scope.bsr_scale(b0, block), Some(want));
        }
    }

    /// `BSRMOVF/H/L` round-trip: a full write reads back verbatim; a high-nibble write changes
    /// only bits 7:4 (low nibble preserved); a low-nibble write changes only bits 3:0 (high
    /// nibble preserved). Written via the ops, read back through the shared model
    /// (`[ace-tile-instructions.BSR.2]`, `[ace-tile-instructions.BSR.3]`,
    /// `[ace-tile-instructions.BSR.4]`).
    /// `bsr::bsrmov_full_high_low_round_trip`
    #[test]
    fn bsrmov_full_high_low_round_trip() {
        let mut scope = scope();
        let b0 = scope.bsr(0).unwrap();

        // F (full): the whole exponent byte round-trips.
        assert_eq!(_tile_bsrmovf(&mut scope, b0, 0, 0xA5), Some(()));
        assert_eq!(scope.bsr_scale(b0, 0), Some(0xA5));

        // H (high): only bits 7:4 change; the low nibble (0x5) is preserved.
        assert_eq!(_tile_bsrmovh(&mut scope, b0, 0, 0x0C), Some(()));
        assert_eq!(
            scope.bsr_scale(b0, 0),
            Some(0xC5),
            "high nibble -> 0xC, low nibble 0x5 preserved"
        );

        // L (low): only bits 3:0 change; the high nibble (0xC) is preserved.
        assert_eq!(_tile_bsrmovl(&mut scope, b0, 0, 0x03), Some(()));
        assert_eq!(
            scope.bsr_scale(b0, 0),
            Some(0xC3),
            "low nibble -> 0x3, high nibble 0xC preserved"
        );

        // Only the low 4 bits of the H/L payload are used (upper bits ignored, not shifted in).
        assert_eq!(_tile_bsrmovh(&mut scope, b0, 0, 0xF1), Some(())); // payload nibble = 0x1
        assert_eq!(
            scope.bsr_scale(b0, 0),
            Some(0x13),
            "H payload masked to 0x1"
        );

        // Out-of-range block: no mutation, None returned.
        let before: Vec<_> = (0..BSR_SCALE_BLOCKS)
            .map(|b| scope.bsr_scale(b0, b))
            .collect();
        assert_eq!(_tile_bsrmovf(&mut scope, b0, BSR_SCALE_BLOCKS, 0xFF), None);
        assert_eq!(_tile_bsrmovh(&mut scope, b0, BSR_SCALE_BLOCKS, 0xF), None);
        assert_eq!(_tile_bsrmovl(&mut scope, b0, BSR_SCALE_BLOCKS, 0xF), None);
        let after: Vec<_> = (0..BSR_SCALE_BLOCKS)
            .map(|b| scope.bsr_scale(b0, b))
            .collect();
        assert_eq!(before, after, "out-of-range move mutates nothing");
    }

    /// Only the addressed `BsrReg` (and, within it, the addressed block) changes: seeding /
    /// moving one register leaves every other register untouched, and a per-block move leaves
    /// the register's other blocks untouched.
    /// `bsr::only_addressed_bsr_changes`
    #[test]
    fn only_addressed_bsr_changes() {
        let mut scope = scope();
        let b0 = scope.bsr(0).unwrap();
        let b1 = scope.bsr(1).unwrap();

        // Seed both with recognisable, DISTINCT patterns.
        _tile_bsrinit(&mut scope, b0, [0x11; BSR_SCALE_BLOCKS]);
        _tile_bsrinit(&mut scope, b1, [0x22; BSR_SCALE_BLOCKS]);

        // Move into one block of b0; b1 must be entirely unchanged, and b0's OTHER blocks too.
        _tile_bsrmovf(&mut scope, b0, 1, 0x99).unwrap();
        assert_eq!(
            scope.bsr_scale(b0, 1),
            Some(0x99),
            "addressed block changed"
        );
        for block in [0, 2, 3] {
            assert_eq!(
                scope.bsr_scale(b0, block),
                Some(0x11),
                "b0 block {block} untouched by a move to block 1"
            );
        }
        for block in 0..BSR_SCALE_BLOCKS {
            assert_eq!(
                scope.bsr_scale(b1, block),
                Some(0x22),
                "b1 block {block} untouched by writes to b0"
            );
        }
    }

    /// Write/read-consistency property (`[ace-tile-instructions.BSR.4-1]`,
    /// `[ace-tile-instructions.TESTING.3]`): every value written to a register/block through
    /// the BSR ops reads back identically through the shared `TileScope` model, and a write to
    /// one register/block is NOT observed at any other — a DISCRIMINATING check that fails if
    /// the model conflated registers or blocks (write one scale, read a different one back must
    /// not pass).
    /// `bsr::write_read_consistency_property`
    #[test]
    fn write_read_consistency_property() {
        let mut scope = scope();

        // A deterministic spread of (register, block, value) triples over the whole file.
        for reg_ix in 0..NUM_BSR {
            let id = scope.bsr(reg_ix).unwrap();
            for block in 0..BSR_SCALE_BLOCKS {
                // Distinct per (reg, block); +1 keeps every value nonzero so a stale zero from a
                // conflated/never-written slot is detectable.
                let value = ((reg_ix * BSR_SCALE_BLOCKS + block) as u8)
                    .wrapping_mul(7)
                    .wrapping_add(1);
                _tile_bsrmovf(&mut scope, id, block, value).unwrap();
                // Read-back consistency: what we wrote is what we read.
                assert_eq!(scope.bsr_scale(id, block), Some(value));
            }
        }

        // After all writes, every slot still holds ITS OWN value — no cross-register or
        // cross-block aliasing. This is the negative/discriminating half: if the model shared
        // storage across registers, later writes would have clobbered earlier reads.
        for reg_ix in 0..NUM_BSR {
            let id = scope.bsr(reg_ix).unwrap();
            for block in 0..BSR_SCALE_BLOCKS {
                let expected = ((reg_ix * BSR_SCALE_BLOCKS + block) as u8)
                    .wrapping_mul(7)
                    .wrapping_add(1);
                let got = scope.bsr_scale(id, block).unwrap();
                assert_eq!(
                    got, expected,
                    "reg {reg_ix} block {block} kept its own scale"
                );
                // Discriminating: reading a value that was written to a DIFFERENT slot must not
                // pass. Compare against a neighbouring slot's distinct value.
                let neighbour = ((reg_ix * BSR_SCALE_BLOCKS + block + 1) as u8)
                    .wrapping_mul(7)
                    .wrapping_add(1);
                if neighbour != expected {
                    assert_ne!(
                        got, neighbour,
                        "reg {reg_ix} block {block} must not read a neighbour's scale"
                    );
                }
            }
        }
    }

    /// Hand-computed known-value pins, independent of the implementation
    /// (`[ace-tile-instructions.TESTING.4]`). The load-bearing semantics are (a) which block a
    /// factor lands in and (b) the full / high-nibble / low-nibble field split (OQ-3, grounded
    /// against ACE v1 §12); each case's expected result DIFFERS under a wrong model (a MOVH
    /// that wrote the full byte, or that shifted the whole payload in), so no differential
    /// tiebreaker is needed here.
    /// `bsr::known_value_pins`
    #[test]
    fn known_value_pins() {
        let mut scope = scope();
        let b0 = scope.bsr(0).unwrap();

        // BSRINIT pins the exact per-block exponents (E8M0: 127 is the unit-scale exponent).
        _tile_bsrinit(&mut scope, b0, [127, 0, 255, 64]);
        assert_eq!(scope.bsr_scale(b0, 0), Some(127));
        assert_eq!(scope.bsr_scale(b0, 1), Some(0));
        assert_eq!(scope.bsr_scale(b0, 2), Some(255));
        assert_eq!(scope.bsr_scale(b0, 3), Some(64));

        // MOVH on block 1 (currently 0x00): high nibble := 0xA -> 0xA0 (a full-byte write model
        // would give 0x0A; a whole-payload shift would give 0xA0 only for a masked 0x0A input).
        _tile_bsrmovh(&mut scope, b0, 1, 0x0A).unwrap();
        assert_eq!(
            scope.bsr_scale(b0, 1),
            Some(0xA0),
            "MOVH sets bits 7:4 only (0xA0), not the whole byte (0x0A)"
        );

        // MOVL on block 2 (currently 0xFF): low nibble := 0x3 -> 0xF3 (high nibble 0xF kept).
        _tile_bsrmovl(&mut scope, b0, 2, 0x3).unwrap();
        assert_eq!(
            scope.bsr_scale(b0, 2),
            Some(0xF3),
            "MOVL sets bits 3:0 only (0xF3), high nibble 0xF preserved"
        );
    }

    /// System-as-a-whole wiring check: seed a BSR via `BSRINIT`, move factors in via
    /// `BSRMOV{F,H,L}`, and read the same scale back through the shared `TileScope` model end
    /// to end (INV-5). Prints the observable file state and the gate helper family D reads.
    #[test]
    fn end_to_end_bsr_seed_move_read() {
        let mut scope = scope();
        let b0 = scope.bsr(0).unwrap();

        _tile_bsrinit(&mut scope, b0, [10, 20, 30, 40]);
        _tile_bsrmovf(&mut scope, b0, 0, 0x88).unwrap();
        _tile_bsrmovh(&mut scope, b0, 1, 0x0C).unwrap(); // 20 = 0x14 -> 0xC4
        _tile_bsrmovl(&mut scope, b0, 2, 0x0F).unwrap(); // 30 = 0x1E -> 0x1F

        let readback: [u8; BSR_SCALE_BLOCKS] =
            core::array::from_fn(|b| scope.bsr_scale(b0, b).unwrap());
        println!("E2E bsr file readback = {readback:?}");
        println!("E2E detect has_ace={}", crate::detect::has_ace());

        // The MX products read this same file (INV-5); pin the composed result.
        assert_eq!(readback, [0x88, 0xC4, 0x1F, 40]);
        // Cross-check the read-all accessor the MX products use agrees with per-block reads.
        assert_eq!(scope.bsr_reg(b0).scales(), &readback);
    }
}

/// Layer-4 differential (family D). `BSRINIT`/`BSRMOV*` are `ACE`-only, so their native arm is a
/// `.byte` raw encoding that is layer-3-blocked until Intel SDE gains ACE emulation (OQ-6, R2).
/// This asserts bit-for-bit native-vs-oracle agreement of the seeded block-scale exponents
/// (`[ace-tile-instructions.TESTING.1]`) inside the `feature="native"` + full-`ACE` branch, and
/// returns [`quickcheck::TestResult::discard`] — never `from_bool(false)` — on every current
/// host, discarding until SDE ACE lands rather than passing vacuously or failing.
#[cfg(test)]
mod differential {
    #![cfg_attr(
        not(all(target_arch = "x86_64", feature = "native")),
        allow(unused_imports, dead_code)
    )]
    use super::*;
    use crate::tile::{_tile_loadconfig, TileConfig, TileScope};
    use quickcheck::{quickcheck, Arbitrary, Gen, TestResult};

    #[derive(Clone, Debug)]
    struct Exps {
        exponents: [u8; BSR_SCALE_BLOCKS],
    }

    impl Arbitrary for Exps {
        fn arbitrary(g: &mut Gen) -> Self {
            Exps {
                exponents: core::array::from_fn(|_| u8::arbitrary(g)),
            }
        }
    }

    quickcheck! {
        /// Family-D `.byte` differential: `BSRINIT` and the full/high/low `BSRMOV*` moves, native
        /// vs the guard-model oracle. `ACE`-only, so this discards on every current host (OQ-6)
        /// and lights up bit-for-bit once an ACE-capable SDE lands.
        fn prop_native_matches_oracle(exps: Exps) -> TestResult {
            #[cfg(all(target_arch = "x86_64", feature = "native"))]
            {
                if detect::has_ace() {
                    use crate::native;
                    let config = TileConfig {
                        palette_id: 2,
                        rows: [1, 0, 0, 0, 0, 0, 0, 0],
                        colsb: [64, 0, 0, 0, 0, 0, 0, 0],
                    };
                    let cfg = native::encode_tilecfg(2, &config.rows, &config.colsb);
                    let mut data = [0u8; native::TILE_BYTES];
                    data[..BSR_SCALE_BLOCKS].copy_from_slice(&exps.exponents);

                    // BSRINIT: seed the register and compare the per-block exponents.
                    let mut scope = _tile_loadconfig(&config).expect("valid descriptor");
                    let b0 = scope.bsr(0).unwrap();
                    _tile_bsrinit(&mut scope, b0, exps.exponents);
                    let seeded: [u8; BSR_SCALE_BLOCKS] =
                        core::array::from_fn(|k| scope.bsr_scale(b0, k).unwrap());
                    // SAFETY: has_ace() confirmed full ACE + the tile/BSR XSAVE state.
                    let init_ok = unsafe { native::bsrinit_hw(&cfg, &data) }[..BSR_SCALE_BLOCKS]
                        == seeded;

                    // BSRMOVF/H/L: apply block-0 moves to the seeded register and compare each
                    // native move shim's output to the oracle read-back of block 0.
                    let readback_after = |mv: fn(&mut TileScope, BsrId, usize, u8) -> Option<()>,
                                          arg: u8|
                     -> u8 {
                        let mut s = _tile_loadconfig(&config).expect("valid descriptor");
                        let r = s.bsr(0).unwrap();
                        _tile_bsrinit(&mut s, r, exps.exponents);
                        mv(&mut s, r, 0, arg);
                        s.bsr_scale(r, 0).unwrap()
                    };
                    let movf_ref = readback_after(_tile_bsrmovf, exps.exponents[0]);
                    let movh_ref = readback_after(_tile_bsrmovh, exps.exponents[0] >> 4);
                    let movl_ref = readback_after(_tile_bsrmovl, exps.exponents[0] & 0x0F);
                    let movf_ok = unsafe { native::bsrmovf_hw(&cfg, &data) }[0] == movf_ref;
                    let movh_ok = unsafe { native::bsrmovh_hw(&cfg, &data) }[0] == movh_ref;
                    let movl_ok = unsafe { native::bsrmovl_hw(&cfg, &data) }[0] == movl_ref;

                    return TestResult::from_bool(init_ok && movf_ok && movh_ok && movl_ok);
                }
            }
            let _ = &exps;
            TestResult::discard()
        }
    }
}
