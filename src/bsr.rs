//! ACE group-4 family D: the Block Scale register (`bsr0` / SCALEDATA) and its operations.
//!
//! This module models the single ACE Block Scale register and the family-D instructions
//! that initialize and move the E8M0 microscaling exponents the MX outer products
//! (families E/F) consume:
//!
//! * `BSRINIT` — [`_bsrinit`] sets all 128 BSR bytes to `0x7F` (E8M0 encoding of `2^0`,
//!   scale = 1.0). It takes no data operand (spec section 13.1)
//!   (`[ace-tile-instructions.BSR.1]`).
//! * `BSRMOVF` — [`_bsrmovf`] moves 1024 bits from two ZMM-sized sources into the full BSR:
//!   the first source provides the upper 512 bits (A scales), the second the lower 512 bits
//!   (B scales) (spec section 13.2) (`[ace-tile-instructions.BSR.2]`).
//! * `BSRMOVH` — [`_bsrmovh`] (write) / [`_bsrmovh_read`] (read) move the upper (A-scale)
//!   512-bit half of the BSR to or from a ZMM-sized buffer (spec section 13.3)
//!   (`[ace-tile-instructions.BSR.3]`).
//! * `BSRMOVL` — [`_bsrmovl`] (write) / [`_bsrmovl_read`] (read) move the lower (B-scale)
//!   512-bit half (spec section 13.3) (`[ace-tile-instructions.BSR.4]`).
//!
//! The register lives inside the [`TileScope`] guard, so a later MX outer product reads back
//! exactly the block scales these ops wrote (INV-5, `[ace-tile-instructions.BSR.4-1]`).
//!
//! # Register layout (spec sections 10.2.2 and 10.5)
//!
//! One 1024-bit (128-byte) Block Scale register is architected:
//!
//! * `BSR[1023:512]` (bytes 64..128) — **A scales**: `A_scales[0..15][0..3]`.
//! * `BSR[511:0]` (bytes 0..64) — **B scales**: `B_scales[0..15][0..3]`.
//!
//! Each 512-bit segment holds four groups of 16 E8M0 scale bytes, organized by element
//! index: all four group bytes for element `s` are adjacent, so `A_scales[s]` group `g` is
//! BSR byte `64 + s*4 + g` and `B_scales[s]` group `g` is BSR byte `s*4 + g` — exactly the
//! `BSR.byte[64 + s*4 + a_group]` / `BSR.byte[s*4 + b_group]` indexing of the section-14
//! outer-product pseudocode. At reset, and after `BSRINIT` / `LDTILECFG` / `TILERELEASE`,
//! every byte reads `0x7F` (spec sections 10.2.3, 11.2.1, 11.4.1, 13.1).
//!
//! # Dispatch
//!
//! Each op is a safe public dispatcher plus a cfg-free `_scalar` oracle (the primary path,
//! correct on every target). Family D is `ACE`-only and gates on full `ACE`
//! (`detect::has_ace`, `[ace-tile-instructions.DETECT.1-3]`,
//! `[ace-tile-instructions.DISPATCH.1]`). The dispatchers reference the detector to mark the
//! gate site and take the scalar oracle on every target.

use crate::detect;
use crate::tile::TileScope;

/// Size of the Block Scale register in bytes (1024 bits, spec section 10.2.2).
pub const BSR_BYTES: usize = 128;

/// Size of one BSR half (A scales or B scales) in bytes (512 bits).
pub const BSR_HALF_BYTES: usize = 64;

/// The INIT value of every BSR byte: E8M0 encoding of `2^0` = 1.0 (spec section 10.2.3).
pub const BSR_INIT_BYTE: u8 = 0x7F;

/// The single architected Block Scale register (`bsr0`, SCALEDATA): 128 bytes, A scales in
/// the upper half (bytes 64..128), B scales in the lower half (bytes 0..64).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct BsrReg {
    bytes: [u8; BSR_BYTES],
}

impl BsrReg {
    /// A BSR in its INIT state: every byte `0x7F` (scale = 1.0, spec section 10.2.3).
    pub(crate) fn init() -> Self {
        BsrReg {
            bytes: [BSR_INIT_BYTE; BSR_BYTES],
        }
    }

    /// Raw register bytes (byte 0 = BSR bits 7:0).
    pub(crate) fn bytes(&self) -> &[u8; BSR_BYTES] {
        &self.bytes
    }

    /// Mutable raw register bytes.
    pub(crate) fn bytes_mut(&mut self) -> &mut [u8; BSR_BYTES] {
        &mut self.bytes
    }

    /// The A-scale byte for output-row element `s` (0..16) in group `group` (0..4):
    /// `BSR.byte[64 + s*4 + group]` (spec section 14.1.6).
    pub(crate) fn a_scale(&self, s: usize, group: usize) -> u8 {
        self.bytes[BSR_HALF_BYTES + s * 4 + group]
    }

    /// The B-scale byte for output-column element `s` (0..16) in group `group` (0..4):
    /// `BSR.byte[s*4 + group]` (spec section 14.1.6).
    pub(crate) fn b_scale(&self, s: usize, group: usize) -> u8 {
        self.bytes[s * 4 + group]
    }
}

/// `BSRINIT`: set all 128 BSR bytes to `0x7F` (E8M0 scale = 1.0). No data operand
/// (spec section 13.1) (`[ace-tile-instructions.BSR.1]`).
pub fn _bsrinit(scope: &mut TileScope) {
    let _ = detect::has_ace; // family D gate site [DETECT.1-3]
    _bsrinit_scalar(scope);
}

/// Portable `BSRINIT` oracle — the section-13.1.4 pseudocode: `FOR i = 0 TO 127:
/// BSR.byte[i] = 0x7F`.
pub fn _bsrinit_scalar(scope: &mut TileScope) {
    *scope.bsr_mut() = BsrReg::init();
}

/// `BSRMOVF`: write the full BSR from two 512-bit sources — `a_scales` (the first source
/// operand, zmm1) becomes `BSR[1023:512]`, `b_scales` (the second source operand,
/// zmm2/m512) becomes `BSR[511:0]` (spec section 13.2) (`[ace-tile-instructions.BSR.2]`).
pub fn _bsrmovf(
    scope: &mut TileScope,
    a_scales: [u8; BSR_HALF_BYTES],
    b_scales: [u8; BSR_HALF_BYTES],
) {
    let _ = detect::has_ace; // family D gate site [DETECT.1-3]
    _bsrmovf_scalar(scope, a_scales, b_scales);
}

/// Portable `BSRMOVF` oracle — the section-13.2.4 pseudocode:
/// `BSR[511:0] = src2[511:0]; BSR[1023:512] = src1[511:0]`.
pub fn _bsrmovf_scalar(
    scope: &mut TileScope,
    a_scales: [u8; BSR_HALF_BYTES],
    b_scales: [u8; BSR_HALF_BYTES],
) {
    let bytes = scope.bsr_mut().bytes_mut();
    bytes[..BSR_HALF_BYTES].copy_from_slice(&b_scales);
    bytes[BSR_HALF_BYTES..].copy_from_slice(&a_scales);
}

/// `BSRMOVH` write form: load a 512-bit source into the BSR upper (A-scale) half —
/// `BSR[1023:512] = src[511:0]` (spec section 13.3) (`[ace-tile-instructions.BSR.3]`).
pub fn _bsrmovh(scope: &mut TileScope, a_scales: [u8; BSR_HALF_BYTES]) {
    let _ = detect::has_ace; // family D gate site [DETECT.1-3]
    _bsrmovh_scalar(scope, a_scales);
}

/// Portable `BSRMOVH` (write) oracle — section-13.3.4 `bsrmovh(src, w1=1)`.
pub fn _bsrmovh_scalar(scope: &mut TileScope, a_scales: [u8; BSR_HALF_BYTES]) {
    scope.bsr_mut().bytes_mut()[BSR_HALF_BYTES..].copy_from_slice(&a_scales);
}

/// `BSRMOVH` read form: store the BSR upper (A-scale) half — `dst[511:0] = BSR[1023:512]`
/// (spec section 13.3). The architectural form also zeroes `dst[MAXVL-1:VL]`, which the
/// 64-byte return models trivially.
pub fn _bsrmovh_read(scope: &TileScope) -> [u8; BSR_HALF_BYTES] {
    let _ = detect::has_ace; // family D gate site [DETECT.1-3]
    _bsrmovh_read_scalar(scope)
}

/// Portable `BSRMOVH` (read) oracle — section-13.3.4 `bsrmovh(src, w1=0)`.
pub fn _bsrmovh_read_scalar(scope: &TileScope) -> [u8; BSR_HALF_BYTES] {
    let mut out = [0u8; BSR_HALF_BYTES];
    out.copy_from_slice(&scope.bsr().bytes()[BSR_HALF_BYTES..]);
    out
}

/// `BSRMOVL` write form: load a 512-bit source into the BSR lower (B-scale) half —
/// `BSR[511:0] = src[511:0]` (spec section 13.3) (`[ace-tile-instructions.BSR.4]`).
pub fn _bsrmovl(scope: &mut TileScope, b_scales: [u8; BSR_HALF_BYTES]) {
    let _ = detect::has_ace; // family D gate site [DETECT.1-3]
    _bsrmovl_scalar(scope, b_scales);
}

/// Portable `BSRMOVL` (write) oracle — section-13.3.4 `bsrmovl(src, w1=1)`.
pub fn _bsrmovl_scalar(scope: &mut TileScope, b_scales: [u8; BSR_HALF_BYTES]) {
    scope.bsr_mut().bytes_mut()[..BSR_HALF_BYTES].copy_from_slice(&b_scales);
}

/// `BSRMOVL` read form: store the BSR lower (B-scale) half — `dst[511:0] = BSR[511:0]`
/// (spec section 13.3).
pub fn _bsrmovl_read(scope: &TileScope) -> [u8; BSR_HALF_BYTES] {
    let _ = detect::has_ace; // family D gate site [DETECT.1-3]
    _bsrmovl_read_scalar(scope)
}

/// Portable `BSRMOVL` (read) oracle — section-13.3.4 `bsrmovl(src, w1=0)`.
pub fn _bsrmovl_read_scalar(scope: &TileScope) -> [u8; BSR_HALF_BYTES] {
    let mut out = [0u8; BSR_HALF_BYTES];
    out.copy_from_slice(&scope.bsr().bytes()[..BSR_HALF_BYTES]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile::{_tile_loadconfig, TileConfig};

    fn scope() -> TileScope {
        _tile_loadconfig(&TileConfig::ace()).expect("valid ACE descriptor")
    }

    /// LDTILECFG leaves the BSR in INIT state (`0x7F` everywhere, spec section 11.2.1), and
    /// BSRINIT restores it after arbitrary writes (spec section 13.1).
    /// `bsr::init_state_is_0x7f`
    #[test]
    fn init_state_is_0x7f() {
        let mut s = scope();
        assert!(
            s.bsr().bytes().iter().all(|&b| b == BSR_INIT_BYTE),
            "Acquire leaves the BSR in INIT (0x7F) state"
        );
        _bsrmovf(&mut s, [0x11; 64], [0x22; 64]);
        assert!(s.bsr().bytes().iter().any(|&b| b != BSR_INIT_BYTE));
        _bsrinit(&mut s);
        assert!(
            s.bsr().bytes().iter().all(|&b| b == BSR_INIT_BYTE),
            "BSRINIT resets every byte to 0x7F"
        );
    }

    /// BSRMOVF places the FIRST source in the UPPER (A-scale) half and the SECOND source in
    /// the LOWER (B-scale) half — the section-13.2.4 operand order. A swapped model fails
    /// both assertions.
    /// `bsr::movf_operand_order`
    #[test]
    fn movf_operand_order() {
        let mut s = scope();
        let a: [u8; 64] = core::array::from_fn(|i| 0x80 + i as u8);
        let b: [u8; 64] = core::array::from_fn(|i| i as u8);
        _bsrmovf(&mut s, a, b);
        assert_eq!(
            &s.bsr().bytes()[64..],
            &a,
            "src1 -> BSR[1023:512] (A scales)"
        );
        assert_eq!(&s.bsr().bytes()[..64], &b, "src2 -> BSR[511:0] (B scales)");
    }

    /// BSRMOVH/BSRMOVL write exactly one half and leave the other untouched; the read forms
    /// return what the write forms stored (write/read consistency).
    /// `bsr::half_moves_write_read_consistent`
    #[test]
    fn half_moves_write_read_consistent() {
        let mut s = scope();
        let a: [u8; 64] = core::array::from_fn(|i| 0xA0 ^ i as u8);
        _bsrmovh(&mut s, a);
        assert_eq!(_bsrmovh_read(&s), a, "H write/read round-trips");
        assert!(
            s.bsr().bytes()[..64].iter().all(|&b| b == BSR_INIT_BYTE),
            "H write leaves the B half in INIT state"
        );

        let b: [u8; 64] = core::array::from_fn(|i| 0x50 ^ i as u8);
        _bsrmovl(&mut s, b);
        assert_eq!(_bsrmovl_read(&s), b, "L write/read round-trips");
        assert_eq!(_bsrmovh_read(&s), a, "L write leaves the A half untouched");
    }

    /// The scale-byte accessors implement the section-14.1.6 indexing:
    /// `A_scales[s]` group `g` = byte `64 + s*4 + g`, `B_scales[s]` group `g` = byte
    /// `s*4 + g` (element-major grouping, spec section 10.2.2).
    /// `bsr::scale_indexing_matches_spec`
    #[test]
    fn scale_indexing_matches_spec() {
        let mut s = scope();
        let a: [u8; 64] = core::array::from_fn(|i| i as u8);
        let b: [u8; 64] = core::array::from_fn(|i| 0x40 + i as u8);
        _bsrmovf(&mut s, a, b);
        // Element 3, group 2: A byte = 64 + 3*4 + 2 = byte 78 = a[14]; B byte = b[14].
        assert_eq!(s.bsr().a_scale(3, 2), a[3 * 4 + 2]);
        assert_eq!(s.bsr().b_scale(3, 2), b[3 * 4 + 2]);
        // Element 15, group 3 is the last byte of each half.
        assert_eq!(s.bsr().a_scale(15, 3), a[63]);
        assert_eq!(s.bsr().b_scale(15, 3), b[63]);
        // Element 0, group 0 is the first byte of each half.
        assert_eq!(s.bsr().a_scale(0, 0), a[0]);
        assert_eq!(s.bsr().b_scale(0, 0), b[0]);
    }
}
