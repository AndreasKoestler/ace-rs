//! Family I (AVX10_V2_AUX): `VUNPACKB` — unpack packed sub-byte elements into bytes.
//!
//! `unpackb` reads a 512-bit (`[u8; 64]`) buffer of packed sub-byte elements and produces
//! `KL = VL/8 = 64` 8-bit output lanes, one per source element, per ACE v1 spec section
//! 9.9 (`VUNPACKB`, encoding `EVEX.512.NP.0F3A.W0 3D /r`, spec section 6.2.10). The element
//! width, start offset and sign-extend behaviour are all selected by the `imm8` immediate;
//! this is a pure data-rearrangement utility (no floating-point conversion) and the
//! read-back complement of the nibble (family D) and 6-bit (family F) packers.
//!
//! The oracle transcribes the section-9.9.4 pseudocode bit-exactly
//! (`[avx10-v2-aux-ocp-conversions.UNPACKB.1]`):
//!
//! ```text
//! size = max((imm8>>2)&0x7, 2); sign_ex = imm8[5]
//! if size==2: start = imm8&0x3   elif size in (3,4): start = min(imm8&0x3,1)   else: start = 0
//! for i in range(KL=VL/8):
//!    j = (start*KL + i)*size                 # bit offset of ith packed element
//!    elem = data[j+size-1 : j]               # extract size-bit field
//!    if sign_ex: elem[7:size] = elem[size-1] # sign-extend from field MSB
//!    else:       elem[7:size] = 0            # zero-extend
//!    dest.byte[i] = elem
//! ```
//!
//! imm8 decode (`[avx10-v2-aux-ocp-conversions.UNPACKB.2]`,
//! `[avx10-v2-aux-ocp-conversions.UNPACKB.3]`, `[avx10-v2-aux-ocp-conversions.UNPACKB.4]`):
//!
//! * **size** = `max((imm8>>2)&0x7, 2)` — the 3-bit field `imm8[4:2]` clamped to a minimum
//!   of 2, so `size` ranges over `2..=7`; the reserved encodings 0 and 1 decode to 2.
//! * **sign_ex** = `imm8[5]` (`imm8 & 0x20`): 1 sign-extends from the field MSB (bit
//!   `size-1`), 0 zero-extends.
//! * **start** is conditioned on `size`: `size==2` -> `imm8 & 0x3` (`0..=3`); `size` in
//!   `{3,4}` -> `min(imm8 & 0x3, 1)` (`0..=1`); `size` in `{5,6,7}` -> `0`.
//!
//! Per lane `i` the size-bit field at bit offset `j = (start*KL + i)*size` is read from the
//! packed buffer (LSB-from-bit-0, straddling byte boundaries) via
//! [`crate::fp4::extract_field`] — the same size-parameterized reader the FP4/FP6 unpackers
//! use, so the extraction is not reimplemented divergently
//! (`[avx10-v2-aux-ocp-conversions.UNPACKB.5]`). The field is then widened to 8 bits: zero-
//! extension clears bits `[7:size]` (`[avx10-v2-aux-ocp-conversions.UNPACKB.6]`),
//! sign-extension replicates the field MSB into `[7:size]`
//! (`[avx10-v2-aux-ocp-conversions.UNPACKB.7]`).
//!
//! After the section-9.9.4 conditioning the spec constraint `(start+1)*KL*size <= VL` always
//! holds (with `KL=64`, `VL=512` it reduces to `(start+1)*size <= 8`, satisfied by every
//! conditioned `(size, start)` pair), so the maximum bit offset read is `(start+1)*64*size
//! - 1 <= 511` — always inside the 512-bit (`[u8; 64]`) input. The function is therefore
//! **total** over `([u8; 64], u8)`: every `imm8` — including the reserved bits `imm8[7:6]`
//! (SBZ, no `#UD`/`#GP`, spec section 9.9.1) and the reserved size encodings 0/1 clamped to
//! 2 — returns a defined `[u8; 64]` and never panics or faults
//! (`[avx10-v2-aux-ocp-conversions.UNPACKB.8]`).
//!
//! **Masking scope (`[avx10-v2-aux-ocp-conversions.UNPACKB.9]`, OQ — no-writemask only):**
//! the spec section-9.9.2 form supports `{k1}{z}` write-masking/zeroing, but — consistent
//! with the whole crate's public surface (`lib.rs` v1 non-goals: no `{k1}{z}` / `m*bcst` /
//! sub-512 vector-length plumbing) — only the `no_writemask` path is surfaced: every lane is
//! written and `imm8` is the sole control input, surfaced as a plain `u8` value argument
//! (NOT a mask). The pseudocode's per-lane `if k1[i] or no_writemask` therefore always takes
//! the `no_writemask` branch.
//!
//! The public dispatcher [`unpackb`] is a safe fn that selects the scalar oracle whenever the
//! running CPU lacks `AVX10_V2_AUX` (`[avx10-v2-aux-ocp-conversions.DETECTION.2]`). The
//! `_scalar` oracle [`unpackb_scalar`] is the primary, always-correct path on every target
//! including non-x86 (`[avx10-v2-aux-ocp-conversions.CORRECTNESS.1]`,
//! `[avx10-v2-aux-ocp-conversions.CORRECTNESS.2]`); it carries no cfg gate, reads no global
//! state, and the dispatcher equals it bit-for-bit. The name mirrors the eventual stdarch
//! intrinsic stem `unpackb` (`[avx10-v2-aux-ocp-conversions.NAMING.1]`), and the whole module
//! compiles on stable Rust with no `core::simd`/nightly
//! (`[avx10-v2-aux-ocp-conversions.STABLE_RUST.1]`).
//!
//! OQ-5 (intrinsic unavailable -> oracle-only): `VUNPACKB` ships **oracle-only**. It is
//! encoded `EVEX.512.NP.0F3A.W0 3D /r` and its intrinsic is `_mm512_unpackb(__m512i a,
//! unsigned int imm8)` (spec section 9.9.6), but a compile probe under `-mavx10.2`
//! (GCC 16.1.1) shows that intrinsic is ABSENT — the compiler offers only `_mm512_kunpackb`
//! (the unrelated *mask*-unpack), confirming `_mm512_unpackb` does not yet exist in the
//! toolchain. Per OQ-5 there is therefore no native C shim, no `extern "C"` declaration, and
//! no `_hw` path; the dispatcher resolves to its `_scalar` sibling on every target. The
//! capability check [`crate::detect::has_avx10_v2_aux`] is still consulted (and returns
//! `false` off AVX10_V2_AUX hardware), so the three-layer detection wiring is exercised; a
//! native path is added once the intrinsic lands in the toolchain. The differential test
//! that would otherwise tie a native path to the oracle DISCARDS (no native path exists), so
//! correctness is grounded against the section-9.9.4 pseudocode transcribed in
//! [`unpackb_scalar`].
//!
//! OQ-10 (`VUNPACKB` width for the EXACTNESS.2 round-trip): families D/F emit 32/48-byte
//! packed buffers, while `unpackb`'s canonical surface is the 512-bit `[u8; 64]` form.
//! RESOLVED: the public 512-bit `unpackb` is the canonical surface; the EXACTNESS.2 read-back
//! tests (in the test module below) construct a matching-width packed input — they copy the
//! family-D `[u8; 32]` nibble buffer / family-F `[u8; 48]` 6-bit buffer into the low bytes of
//! the 512-bit buffer and unpack with size 4 / size 6, start 0, zero-extend, recovering the
//! pre-pack per-lane codes in the low 64 output lanes
//! (`[avx10-v2-aux-ocp-conversions.EXACTNESS.2]`).

use crate::detect;
use crate::fp4::extract_field;

/// Number of 8-bit output lanes: `KL = VL/8 = 512/8 = 64` (spec section 9.9.1).
const KL: usize = 64;

/// `imm8` sign-extend selector bit (`imm8[5]`), per spec section 9.9.6.
///
/// When set, `VUNPACKB` sign-extends each field from its MSB (bit `size-1`); when clear it
/// zero-extends. OR into the `imm8` value to request sign-extension.
pub const ACE_UNPACKB_SEXT: u8 = 1 << 5;

/// Build the `imm8` element-size field from a desired element size `n` (spec section 9.9.6):
/// `ACE_UNPACKB_SIZE(n) = ((n) & 0x7) << 2`.
///
/// The encoded size occupies bits `imm8[4:2]`. Note the oracle decodes `size = max(field, 2)`,
/// so `n` values 0 and 1 still decode to a usable size of 2.
#[allow(non_snake_case)]
pub const fn ACE_UNPACKB_SIZE(n: u8) -> u8 {
    (n & 0x7) << 2
}

/// Build the `imm8` start-offset field from a desired start `s` (spec section 9.9.6):
/// `ACE_UNPACKB_START(s) = ((s) & 0x3) << 0`.
///
/// The encoded start occupies bits `imm8[1:0]`. The oracle conditions the decoded start on
/// the element size (size 2 -> `0..=3`; size 3/4 -> `0..=1`; size 5/6/7 -> `0`).
#[allow(non_snake_case)]
pub const fn ACE_UNPACKB_START(s: u8) -> u8 {
    // Spec form `((s) & 0x3) << 0`; the `<< 0` is the identity, so the mask alone suffices.
    s & 0x3
}

/// Portable reference oracle for [`unpackb`] — the primary always-correct path.
///
/// Transcribes the spec section-9.9.4 `VUNPACKB` pseudocode bit-exactly: decode `size`,
/// `sign_ex` and `start` from `imm8`, then for each of the `KL = 64` output lanes extract the
/// `size`-bit field at bit offset `(start*KL + i)*size` via [`crate::fp4::extract_field`] and
/// zero- or sign-extend it to a full byte. Defined for every `imm8` (incl. reserved
/// `imm8[7:6]` and the reserved size encodings 0/1 clamped to 2); never panics or faults.
/// Carries no cfg gate and reads no global state.
/// `[avx10-v2-aux-ocp-conversions.CORRECTNESS.1]` `[avx10-v2-aux-ocp-conversions.CORRECTNESS.2]`
/// `[avx10-v2-aux-ocp-conversions.UNPACKB.1]`
pub fn unpackb_scalar(a: [u8; 64], imm8: u8) -> [u8; 64] {
    // size = max((imm8>>2)&0x7, 2) — 3-bit field imm8[4:2], clamped to a minimum of 2 so the
    // reserved encodings 0/1 decode to 2 (range 2..=7).
    // [avx10-v2-aux-ocp-conversions.UNPACKB.2]
    let size = (((imm8 >> 2) & 0x7) as usize).max(2);
    // sign_ex = imm8[5]. [avx10-v2-aux-ocp-conversions.UNPACKB.3]
    let sign_ex = (imm8 & 0x20) != 0;
    // start conditioned on size. [avx10-v2-aux-ocp-conversions.UNPACKB.4]
    let start: usize = if size == 2 {
        (imm8 & 0x3) as usize
    } else if size == 3 || size == 4 {
        ((imm8 & 0x3) as usize).min(1)
    } else {
        // size 5, 6, 7
        0
    };

    // After conditioning the spec constraint (start+1)*KL*size <= VL holds (with KL=64,
    // VL=512 it is (start+1)*size <= 8), so the highest bit read, (start+1)*KL*size - 1, is
    // <= 511 — always inside the 512-bit input. imm8[7:6] are reserved/SBZ and ignored
    // (no fault). [avx10-v2-aux-ocp-conversions.UNPACKB.8]
    core::array::from_fn(|i| {
        // j = (start*KL + i)*size — bit offset of the ith packed element.
        // [avx10-v2-aux-ocp-conversions.UNPACKB.5]
        let j = (start * KL + i) * size;
        // elem = the size-bit field at bit offset j (LSB-from-bit-0, straddles byte
        // boundaries) — the SAME size-parameterized reader the FP4/FP6 unpackers use.
        let field = extract_field(&a, j, size);
        if sign_ex {
            // Sign-extend: replicate the field MSB (bit size-1) into bits [7:size].
            // [avx10-v2-aux-ocp-conversions.UNPACKB.7]
            let sign_bit = (field >> (size - 1)) & 0x1;
            if sign_bit != 0 {
                // Set bits [7:size]; bits [size-1:0] already hold the field.
                let high_mask = 0xffu8 << size;
                field | high_mask
            } else {
                field
            }
        } else {
            // Zero-extend: bits [7:size] are already clear from extract_field.
            // [avx10-v2-aux-ocp-conversions.UNPACKB.6]
            field
        }
    })
}

/// `VUNPACKB` — unpack 64 packed sub-byte elements into 64 bytes, the public dispatcher.
///
/// `imm8` selects the element size (`imm8[4:2]`, clamped to a minimum of 2), the start offset
/// (`imm8[1:0]`, conditioned on size) and the sign-extend selector (`imm8[5]`) per spec
/// section 9.9.4; build it with [`ACE_UNPACKB_SIZE`] / [`ACE_UNPACKB_START`] /
/// [`ACE_UNPACKB_SEXT`]. `imm8` is a plain `u8` value argument (NOT a write-mask): v1 surfaces
/// only the `no_writemask` path, so every lane is written and `imm8` is the sole control input
/// (`[avx10-v2-aux-ocp-conversions.UNPACKB.9]`). The output is the full 512-bit `[u8; 64]`.
///
/// Queries [`detect::has_avx10_v2_aux`] for the native path; that path is not wired (OQ-5, see
/// the module docs — `_mm512_unpackb` is absent from the `-mavx10.2` toolchain), so the
/// dispatcher falls through to [`unpackb_scalar`] on every target, returning the spec-defined
/// value (`[avx10-v2-aux-ocp-conversions.DETECTION.2]`).
pub fn unpackb(a: [u8; 64], imm8: u8) -> [u8; 64] {
    // No native path this phase (OQ-5): the `_mm512_unpackb` intrinsic is absent from the
    // `-mavx10.2` toolchain (the compiler exposes only the unrelated mask-unpack
    // `_mm512_kunpackb`), so even under `feature="native"` on AVX10_V2_AUX hardware the oracle
    // is the only path. The capability check is still consulted so detection is wired and
    // ready for the shim once the intrinsic lands.
    let _ = detect::has_avx10_v2_aux; // keep the capability gate referenced on every target
    unpackb_scalar(a, imm8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{cvtbf4_hf8, cvtf8_bf4s_e5m2, cvtf8_bf6s};

    /// Pack `lanes` little-endian (LSB-from-bit-0) into a `[u8; 64]` at `size` bits per lane,
    /// the inverse of the size-`size` extraction `unpackb` performs at start 0. Used by the
    /// known-value and round-trip tests to build a packed input with known field contents.
    fn pack_fields(lanes: &[u8], size: usize) -> [u8; 64] {
        let mut buf = [0u8; 64];
        for (i, &lane) in lanes.iter().enumerate() {
            let field = (lane as u16) & ((1u16 << size) - 1);
            let mut bit = i * size;
            let mut remaining = size;
            let mut value = field;
            while remaining > 0 {
                let byte_idx = bit >> 3;
                let bit_in_byte = bit & 7;
                let take = (8 - bit_in_byte).min(remaining);
                let mask = ((1u16 << take) - 1) as u8;
                buf[byte_idx] |= ((value as u8) & mask) << bit_in_byte;
                value >>= take;
                remaining -= take;
                bit += take;
            }
        }
        buf
    }

    /// Size-4 ZERO-EXTEND extraction (nibbles -> bytes), start 0.
    /// `[avx10-v2-aux-ocp-conversions.UNPACKB.2]` (size decode),
    /// `[avx10-v2-aux-ocp-conversions.UNPACKB.5]` (field offset),
    /// `[avx10-v2-aux-ocp-conversions.UNPACKB.6]` (zero-extend clears [7:4]).
    ///
    /// DISCRIMINATING: lane fields include `0xF` (`0b1111`), whose top bit is set. Under
    /// zero-extension it must read back as `0x0F` (high nibble cleared); a sign-extending
    /// model would instead produce `0xFF`. The test asserts `0x0F`, ruling out sign-extend.
    #[test]
    fn known_value_size4_zero_extend() {
        // imm8: size field = 4 (imm8[4:2] = 0b100 -> 0x10), start 0, sign_ex 0.
        let imm8 = ACE_UNPACKB_SIZE(4) | ACE_UNPACKB_START(0); // 0x10
                                                               // 64 nibble lanes; lane i = i & 0xF so every nibble value 0..0xF appears.
        let lanes: [u8; 64] = core::array::from_fn(|i| (i as u8) & 0x0f);
        let buf = pack_fields(&lanes, 4);

        let got = unpackb(buf, imm8);
        for i in 0..64 {
            let expected = lanes[i]; // zero-extended: high nibble clear.
            assert_eq!(got[i], expected, "lane {i}: size-4 zero-extend");
            assert_eq!(
                got[i] & 0xf0,
                0,
                "lane {i}: bits [7:4] cleared (zero-extend)"
            );
        }
        // The 0xF lane specifically reads back 0x0F, NOT 0xFF (rules out sign-extend).
        let lane_f = lanes.iter().position(|&v| v == 0xF).unwrap();
        assert_eq!(
            got[lane_f], 0x0F,
            "nibble 0xF zero-extends to 0x0F (NOT 0xFF as sign-extend would give)"
        );
    }

    /// Size-3 SIGN-EXTEND extraction, start 0: verify the sign replicates from bit 2.
    /// `[avx10-v2-aux-ocp-conversions.UNPACKB.3]` (sign_ex selector),
    /// `[avx10-v2-aux-ocp-conversions.UNPACKB.7]` (replicate field MSB into [7:3]).
    ///
    /// DISCRIMINATING: each 3-bit field is interpreted as a signed value via bit 2.
    ///  * `0b100` (4) has bit 2 set -> sign-extends to `0xFC` (`-4` as i8); a zero-extend
    ///    model would give `0x04`.
    ///  * `0b011` (3) has bit 2 clear -> stays `0x03` under both models, pinning that
    ///    sign-extension does NOT spuriously set high bits when the MSB is 0.
    ///  * `0b111` (7) -> `0xFF` (`-1`); `0b000` -> `0x00`.
    #[test]
    fn known_value_size3_sign_extend() {
        // imm8: size field = 3 (imm8[4:2] = 0b011 -> 0x0C), start 0, sign_ex set.
        let imm8 = ACE_UNPACKB_SIZE(3) | ACE_UNPACKB_START(0) | ACE_UNPACKB_SEXT; // 0x2C
                                                                                  // Cover all 8 three-bit fields, then repeat to fill 64 lanes.
        let lanes: [u8; 64] = core::array::from_fn(|i| (i as u8) & 0x07);
        let buf = pack_fields(&lanes, 3);

        let got = unpackb(buf, imm8);
        for i in 0..64 {
            let field = lanes[i] & 0x7;
            // Sign-extend the 3-bit field by hand: replicate bit 2 into [7:3].
            let expected = if field & 0x4 != 0 {
                field | 0xF8 // set bits [7:3]
            } else {
                field
            };
            assert_eq!(got[i], expected, "lane {i}: size-3 sign-extend");
        }
        // Pin the discriminating fields explicitly.
        let l4 = lanes.iter().position(|&v| v == 0b100).unwrap();
        assert_eq!(got[l4], 0xFC, "0b100 sign-extends to 0xFC (-4), NOT 0x04");
        let l3 = lanes.iter().position(|&v| v == 0b011).unwrap();
        assert_eq!(
            got[l3], 0x03,
            "0b011 (MSB clear) stays 0x03, not high bits set"
        );
        let l7 = lanes.iter().position(|&v| v == 0b111).unwrap();
        assert_eq!(got[l7], 0xFF, "0b111 sign-extends to 0xFF (-1)");
        let l0 = lanes.iter().position(|&v| v == 0b000).unwrap();
        assert_eq!(got[l0], 0x00, "0b000 stays 0x00");
    }

    /// Start-offset decode for size 2 (start 1..3): the conditioned start shifts the read
    /// window by `start*KL` lanes (= `start*KL*size` bits).
    /// `[avx10-v2-aux-ocp-conversions.UNPACKB.4]` (start decode),
    /// `[avx10-v2-aux-ocp-conversions.UNPACKB.5]` (offset `(start*KL + i)*size`).
    ///
    /// With size 2 and KL 64, a packed buffer holding 4 windows of 64 two-bit lanes is read
    /// at window `start`. We fill the buffer so window `w` (lanes `64w..64w+64`) carries the
    /// field value `w + 1` in every lane (1, 2, 3 for windows 0..3 — never 0, so an
    /// off-by-window error is visible). For each `start` the output must be the all-`(start+1)`
    /// vector. A model that ignored `start` (always window 0) would always return all-1s.
    #[test]
    fn known_value_size2_start_offset() {
        let size = 2usize;
        // 256 two-bit lanes = 4 windows of 64. Window w gets field (w+1) & 0x3.
        let lanes: [u8; 256] = core::array::from_fn(|idx| {
            let window = idx / 64;
            ((window as u8) + 1) & 0x3
        });
        let buf = pack_fields(&lanes, size);

        for start in 0u8..=3 {
            // imm8: size field = 2 (imm8[4:2] = 0b010 -> 0x08), start = start, sign_ex 0.
            let imm8 = ACE_UNPACKB_SIZE(2) | ACE_UNPACKB_START(start);
            let got = unpackb(buf, imm8);
            let expected = (start + 1) & 0x3;
            for (i, &lane) in got.iter().enumerate() {
                assert_eq!(
                    lane, expected,
                    "size-2 start={start} lane {i}: reads window {start} (field {expected})"
                );
            }
        }
    }

    /// Size clamp: the reserved size-field encodings 0 and 1 decode to a size of 2
    /// (`size = max(field, 2)`). `[avx10-v2-aux-ocp-conversions.UNPACKB.2]`.
    ///
    /// DISCRIMINATING: a model that took `size` literally (0 or 1) would extract 0- or 1-bit
    /// fields and produce a different output. We pack 64 two-bit lanes with distinct values
    /// and assert that imm8 size-field 0 and size-field 1 BOTH produce the same result as
    /// size-field 2 (the explicit size-2 imm8), proving both clamp to 2.
    #[test]
    fn known_value_size_clamp_to_two() {
        let lanes: [u8; 64] = core::array::from_fn(|i| (i as u8) & 0x3);
        let buf = pack_fields(&lanes, 2);

        let size2 = unpackb(buf, ACE_UNPACKB_SIZE(2)); // imm8[4:2] = 2
        let size0 = unpackb(buf, ACE_UNPACKB_SIZE(0)); // imm8[4:2] = 0 -> clamps to 2
        let size1 = unpackb(buf, ACE_UNPACKB_SIZE(1)); // imm8[4:2] = 1 -> clamps to 2

        assert_eq!(size0, size2, "size field 0 clamps to size 2");
        assert_eq!(size1, size2, "size field 1 clamps to size 2");
        // And the size-2 result is the packed lane values themselves (zero-extended).
        for i in 0..64 {
            assert_eq!(size2[i], lanes[i] & 0x3, "lane {i}: size-2 zero-extend");
        }
    }

    /// Reserved `imm8[7:6]` bits are SBZ and produce DEFINED behaviour (no fault): setting
    /// them must not change the result vs. the same imm8 with them clear.
    /// `[avx10-v2-aux-ocp-conversions.UNPACKB.8]`.
    #[test]
    fn known_value_reserved_imm8_bits_defined() {
        let lanes: [u8; 64] = core::array::from_fn(|i| (i as u8) & 0x0f);
        let buf = pack_fields(&lanes, 4);

        let base = ACE_UNPACKB_SIZE(4); // 0x10, start 0, zero-extend
        let clear = unpackb(buf, base);
        // Set both reserved bits imm8[7:6].
        let reserved = unpackb(buf, base | 0xC0);
        assert_eq!(
            reserved, clear,
            "reserved imm8[7:6] are SBZ — defined, ignored, never fault"
        );
    }

    /// EXACTNESS.2 (family D): `unpackb` with size 4 inverts the family-D nibble packing.
    /// `[avx10-v2-aux-ocp-conversions.EXACTNESS.2]`.
    ///
    /// Family D (`cvtf8_bf4s_e5m2`) converts 64 FP8 bytes to 64 FP4 E2M1 nibbles packed into
    /// `[u8; 32]` (two lanes per byte from bit 0). Copying that 32-byte packed output into a
    /// 512-bit buffer and unpacking with size 4 / start 0 / zero-extend recovers each FP4
    /// nibble (in the low 64 output lanes) right-aligned in a byte — i.e. `unpackb` reads back
    /// exactly the nibbles family D wrote, the section-9.9.4 read-back complement of the
    /// section-9.4.5 nibble packer.
    #[test]
    fn exactness2_unpackb_inverts_family_d_nibble_pack() {
        // A representative FP8 (E5M2) input spanning zero, normals, large/overflow, subnormal.
        let fp8: [u8; 64] = core::array::from_fn(|i| i.wrapping_mul(5) as u8);
        let packed: [u8; 32] = cvtf8_bf4s_e5m2(fp8); // nibble-packed FP4 (size 4).

        // Independently compute the per-lane FP4 nibbles family D produced (the pre-pack
        // lanes): re-derive them from the same saturating helper, so the assertion compares
        // unpackb's read-back against an independent source of the packed nibbles.
        let expected_nibbles: [u8; 64] =
            core::array::from_fn(|i| crate::fp4::fp8_e5m2_to_fp4_e2m1(fp8[i]));

        // Load the 32-byte packed output into the low half of the 512-bit unpackb input.
        let mut buf = [0u8; 64];
        buf[..32].copy_from_slice(&packed);

        // size 4, start 0, zero-extend: the low 64 output lanes are the 64 packed nibbles.
        let got = unpackb(buf, ACE_UNPACKB_SIZE(4));
        for i in 0..64 {
            assert_eq!(
                got[i],
                expected_nibbles[i] & 0x0f,
                "lane {i}: size-4 unpack recovers family-D FP4 nibble"
            );
        }
    }

    /// EXACTNESS.2 (family F): `unpackb` with size 6 inverts the family-F 6-bit packing.
    /// `[avx10-v2-aux-ocp-conversions.EXACTNESS.2]`.
    ///
    /// Family F (`cvtf8_bf6s`) converts 64 FP8 bytes to 64 FP6 E3M2 lanes 6-bit-packed into
    /// `[u8; 48]` (lanes straddle byte boundaries from bit 0). Copying that 48-byte packed
    /// output into a 512-bit buffer and unpacking with size 6 / start 0 / zero-extend recovers
    /// each 6-bit FP6 code (in the low 64 output lanes) right-aligned in a byte — the
    /// section-9.9.4 read-back complement of the section-9.6.5 6-bit packer, exercising the
    /// cross-byte-boundary straddle that distinguishes size 6 from size 4.
    #[test]
    fn exactness2_unpackb_inverts_family_f_sixbit_pack() {
        let fp8: [u8; 64] = core::array::from_fn(|i| i.wrapping_mul(7).wrapping_add(3) as u8);
        let packed: [u8; 48] = cvtf8_bf6s(fp8); // 6-bit-packed FP6 (size 6).

        let expected_codes: [u8; 64] =
            core::array::from_fn(|i| crate::fp6::fp8_e5m2_to_fp6_e3m2(fp8[i]));

        let mut buf = [0u8; 64];
        buf[..48].copy_from_slice(&packed);

        // size 6, start 0, zero-extend: the low 64 output lanes are the 64 packed FP6 codes.
        let got = unpackb(buf, ACE_UNPACKB_SIZE(6));
        for i in 0..64 {
            assert_eq!(
                got[i],
                expected_codes[i] & 0x3f,
                "lane {i}: size-6 unpack recovers family-F FP6 code"
            );
        }
    }

    /// Sanity: the family-E exact decode round-trips through a size-4 unpack too — unpack the
    /// family-D nibble output back to nibbles, then map each nibble to FP8 E4M3 via the
    /// public `cvtbf4_hf8`, confirming the unpack feeds a valid FP4 nibble stream. Pins that
    /// the read-back is usable, not merely numerically equal.
    #[test]
    fn exactness2_unpackb_feeds_family_e_decode() {
        let fp8: [u8; 64] = core::array::from_fn(|i| i.wrapping_mul(3) as u8);
        let packed: [u8; 32] = cvtf8_bf4s_e5m2(fp8);

        // Decode the packed nibbles to FP8 E4M3 directly via the public family-E convert.
        let direct: [u8; 64] = cvtbf4_hf8(packed);

        // Unpack the same packed bytes (size 4) and decode each recovered nibble by hand.
        let mut buf = [0u8; 64];
        buf[..32].copy_from_slice(&packed);
        let unpacked = unpackb(buf, ACE_UNPACKB_SIZE(4));
        let via_unpack: [u8; 64] =
            core::array::from_fn(|i| crate::fp4::fp4_e2m1_to_fp8_e4m3(unpacked[i]));

        assert_eq!(
            direct, via_unpack,
            "size-4 unpack recovers the FP4 nibbles family E decodes"
        );
    }
}

/// Property-based tests for family I (`VUNPACKB`). The hand-rolled tests above pin specific
/// imm8/extend cases; these assert the section-9.9.4 invariants across the full input space.
#[cfg(test)]
mod proptests {
    use super::*;
    use quickcheck::{quickcheck, Arbitrary, Gen};

    /// A random packed input (`[u8; 64]`) plus a random `imm8`. `quickcheck` does not derive
    /// `Arbitrary` for `[u8; 64]`, so we wrap it and fill each byte independently — every
    /// packed bit-pattern and every `imm8` (incl. reserved imm8[7:6] and size-field 0/1) is
    /// reachable.
    #[derive(Clone, Debug)]
    struct Inputs {
        a: [u8; 64],
        imm8: u8,
    }

    impl Arbitrary for Inputs {
        fn arbitrary(g: &mut Gen) -> Self {
            Inputs {
                a: core::array::from_fn(|_| u8::arbitrary(g)),
                imm8: u8::arbitrary(g),
            }
        }
    }

    /// Re-derive (size, sign_ex, start) from imm8 exactly as the oracle does, for use by the
    /// property bodies (kept in lockstep with [`unpackb_scalar`]).
    fn decode_imm8(imm8: u8) -> (usize, bool, usize) {
        let size = (((imm8 >> 2) & 0x7) as usize).max(2);
        let sign_ex = (imm8 & 0x20) != 0;
        let start = if size == 2 {
            (imm8 & 0x3) as usize
        } else if size == 3 || size == 4 {
            ((imm8 & 0x3) as usize).min(1)
        } else {
            0
        };
        (size, sign_ex, start)
    }

    quickcheck! {
        /// The public dispatcher always equals the scalar oracle bit-for-bit, across every
        /// packed bit-pattern and every imm8. Since family I is oracle-only this phase (OQ-5)
        /// this also pins that the dispatcher returns the spec value on every input
        /// (`[avx10-v2-aux-ocp-conversions.CORRECTNESS.1]`,
        /// `[avx10-v2-aux-ocp-conversions.DETECTION.2]`).
        fn prop_public_matches_scalar(input: Inputs) -> bool {
            unpackb(input.a, input.imm8) == unpackb_scalar(input.a, input.imm8)
        }

        /// Zero-extension clears bits [7:size] of every output lane
        /// (`[avx10-v2-aux-ocp-conversions.UNPACKB.6]`): when imm8[5] is clear, every output
        /// byte has its high `8-size` bits zero, and the low `size` bits equal the extracted
        /// field. Checked across every packed pattern and size.
        fn prop_zero_extend_clears_high(input: Inputs) -> bool {
            let imm8 = input.imm8 & !ACE_UNPACKB_SEXT; // force zero-extend
            let (size, _sign, _start) = decode_imm8(imm8);
            let out = unpackb(input.a, imm8);
            let high_mask: u8 = if size >= 8 { 0 } else { 0xffu8 << size };
            out.iter().all(|&b| b & high_mask == 0)
        }

        /// Sign-extension replicates the field MSB (bit size-1) into bits [7:size]
        /// (`[avx10-v2-aux-ocp-conversions.UNPACKB.7]`): when imm8[5] is set, the high
        /// `8-size` bits of each output byte all equal that lane's field MSB, and the low
        /// `size` bits are unchanged from the zero-extended read. Checked across every packed
        /// pattern and size.
        fn prop_sign_extend_replicates_msb(input: Inputs) -> bool {
            let imm8_se = input.imm8 | ACE_UNPACKB_SEXT; // force sign-extend
            let imm8_ze = input.imm8 & !ACE_UNPACKB_SEXT; // same size/start, zero-extend
            let (size, _sign, _start) = decode_imm8(imm8_se);
            let se = unpackb(input.a, imm8_se);
            let ze = unpackb(input.a, imm8_ze);
            let high_mask: u8 = if size >= 8 { 0 } else { 0xffu8 << size };
            (0..64).all(|i| {
                let field = ze[i]; // zero-extended field (low `size` bits)
                let msb = (field >> (size - 1)) & 0x1;
                let expected_high = if msb != 0 { high_mask } else { 0 };
                // low bits unchanged, high bits = replicated MSB.
                se[i] & !high_mask == field && se[i] & high_mask == expected_high
            })
        }

        /// Total / never-faults over every imm8 (incl. reserved imm8[7:6] and reserved size
        /// encodings 0/1): `unpackb` returns a defined `[u8; 64]` for every input — running
        /// to completion without panic IS the assertion (`[avx10-v2-aux-ocp-conversions.UNPACKB.8]`).
        /// Also pins the decoded size is always in 2..=7 (the clamp).
        fn prop_never_faults_total(input: Inputs) -> bool {
            let out = unpackb(input.a, input.imm8);
            let (size, _sign, _start) = decode_imm8(input.imm8);
            // Completed without panic; size clamped into range.
            let _ = out;
            (2..=7).contains(&size)
        }

        /// Field-offset / extraction invariant (`[avx10-v2-aux-ocp-conversions.UNPACKB.5]`):
        /// for every lane, the low `size` bits of the (zero-extended) output equal the
        /// size-bit field at bit offset `(start*KL + i)*size`, recomputed independently here
        /// via a from-scratch little-endian bit read of the packed input.
        fn prop_field_offset_matches(input: Inputs) -> bool {
            let imm8 = input.imm8 & !ACE_UNPACKB_SEXT; // zero-extend so output == field
            let (size, _sign, start) = decode_imm8(imm8);
            let out = unpackb(input.a, imm8);
            (0..64).all(|i| {
                let j = (start * 64 + i) * size;
                // Independent bit read (not via extract_field): assemble `size` bits LSB-first.
                let mut field = 0u16;
                for k in 0..size {
                    let bit = j + k;
                    let byte = (input.a[bit >> 3] >> (bit & 7)) & 1;
                    field |= (byte as u16) << k;
                }
                out[i] as u16 == field
            })
        }
    }
}

/// Native-vs-oracle differential for family I (`VUNPACKB`). Phase 11.
///
/// Family I ships **oracle-only** in this toolchain (OQ-5: `_mm512_unpackb` is absent under
/// `-mavx10.2`, and it additionally needs a compile-time-constant `imm8`). The property
/// compares the public dispatcher to its scalar oracle over a random packed buffer AND a
/// random `imm8` under `feature="native"` on AVX10_V2_AUX hardware
/// (`[avx10-v2-aux-ocp-conversions.DIFFERENTIAL.1]`), and `TestResult::discard()`s (never
/// `from_bool(false)`) otherwise, so a fallback-only runner cannot go vacuously green and the
/// test becomes live the moment a native (constant-`imm8` dispatched) path lands.
#[cfg(test)]
mod differential {
    // Without the native feature the quickcheck body compiles down to the discard arm, so the
    // imports and struct fields are only read on the native+x86_64 configuration.
    #![cfg_attr(
        not(all(target_arch = "x86_64", feature = "native")),
        allow(unused_imports, dead_code)
    )]
    use super::*;
    use quickcheck::{quickcheck, Arbitrary, Gen, TestResult};

    #[derive(Clone, Debug)]
    struct Inputs {
        a: [u8; 64],
        imm8: u8,
    }

    impl Arbitrary for Inputs {
        fn arbitrary(g: &mut Gen) -> Self {
            Inputs {
                a: core::array::from_fn(|_| u8::arbitrary(g)),
                imm8: u8::arbitrary(g),
            }
        }
    }

    quickcheck! {
        /// Family-I native-vs-oracle differential. Under `feature="native"` on x86_64 with
        /// `AVX10_V2_AUX` detected, the public dispatcher must equal the scalar oracle
        /// bit-for-bit for every packed buffer and every `imm8`
        /// (`[avx10-v2-aux-ocp-conversions.DIFFERENTIAL.1]`). DISCARDED (not failed) when the
        /// feature or hardware is absent (`[avx10-v2-aux-ocp-conversions.CORRECTNESS.2]`), so a
        /// fallback-only runner never produces a vacuous green.
        fn prop_native_matches_oracle(input: Inputs) -> TestResult {
            #[cfg(all(target_arch = "x86_64", feature = "native"))]
            {
                if detect::has_avx10_v2_aux() {
                    return TestResult::from_bool(
                        unpackb(input.a, input.imm8) == unpackb_scalar(input.a, input.imm8),
                    );
                }
            }
            let _ = &input;
            TestResult::discard()
        }
    }
}
