//! Families F and G: EVEX byte/word VNNI (multiply-and-add).
//!
//! This module hosts the 512-bit EVEX VNNI primitives. Phase 7 lands family F — the six
//! **byte** VNNI primitives (`VPDPBSSD`/`SSDS`/`SUD`/`SUDS`/`UUD`/`UUDS`, ACE v1 spec
//! section 8.6). Each computes, per `i32` dword lane, the sum of four byte-by-byte products
//! under the `SS`/`SU`/`UU` sign matrix and accumulates it into the existing destination
//! dword: `result = dst + sum-of-products`, with `dst` treated as a by-value **input** that
//! is never mutated in place (`[avx10-v1-aux-fp16-fp8-evex-vnni.BYTE_VNNI.1]`,
//! `[avx10-v1-aux-fp16-fp8-evex-vnni.BYTE_VNNI.1-1]`).
//!
//! Phase 8 lands family G — the six **word** VNNI primitives (`VPDPWSUD`/`SUDS`/`USD`/`USDS`/
//! `UUD`/`UUDS`, ACE v1 spec section 8.7). Each computes, per `i32` dword lane, the sum of
//! **two** word-by-word products and accumulates it into the destination dword
//! (`result = dst + sum-of-products`), reusing family F's RMW + saturation rule. NOTE: the
//! word group has NO signed×signed (`SS`) form — only the `SU`/`US`/`UU` sign matrix
//! (`[avx10-v1-aux-fp16-fp8-evex-vnni.WORD_VNNI.1]`,
//! `[avx10-v1-aux-fp16-fp8-evex-vnni.WORD_VNNI.1-1]`).
//!
//! Saturation matrix (spec sections 8.6.5 / 8.7.5, identical for both groups): the
//! non-saturating (no-suffix) forms wrap the INT32 accumulation modulo 2^32 — exactly the
//! wrapping behavior of iteration-0's `dpbssd_scalar`
//! (`[avx10-v1-aux-fp16-fp8-evex-vnni.BYTE_VNNI.1-2]`,
//! `[avx10-v1-aux-fp16-fp8-evex-vnni.WORD_VNNI.1-2]`). The saturating (`S`-suffix) forms
//! unsigned-saturate the accumulation to `[0, 2^32-1]` **iff both operands are unsigned** (the
//! `UU` form — `dpbuuds` / `dpwuuds`), and otherwise signed-saturate to `[-2^31, 2^31-1]`
//! (`[avx10-v1-aux-fp16-fp8-evex-vnni.BYTE_VNNI.1-3]`,
//! `[avx10-v1-aux-fp16-fp8-evex-vnni.WORD_VNNI.1-3]`). The accumulation is computed in a wider
//! type (`i64`) before clamping, so the intermediate sum cannot overflow ahead of the
//! saturation/wrap decision.
//!
//! OQ-1 (`dpbssd` name path, resolved): the 512-bit EVEX `dpbssd` lives at
//! [`crate::vnni::dpbssd`] — a DISTINCT primitive from iteration-0's 256-bit VEX
//! [`crate::dpbssd`] (`ace::dpbssd`). The two are resolved by module path; the iteration-0
//! form is left untouched and is not shadowed by this module (the family-F/G items are reached
//! module-qualified as `ace::vnni::*`).
//!
//! Every primitive is a safe public dispatcher that selects a native path when the running
//! CPU supports `AVX10_V1_AUX` (via a hand-written C shim behind the opt-in `native` feature
//! — no stable `core::arch` EVEX intrinsic exists yet — per
//! `[avx10-v1-aux-fp16-fp8-evex-vnni.DISPATCH.3]`) and otherwise falls back to its `_scalar`
//! oracle. The `_scalar` oracle is the primary,
//! always-correct path on every target including non-x86
//! (`[avx10-v1-aux-fp16-fp8-evex-vnni.ORACLE.1]`); the dispatcher equals it bit-for-bit
//! (`[avx10-v1-aux-fp16-fp8-evex-vnni.ORACLE.2]`). The names mirror the eventual stdarch
//! intrinsic stems (`[avx10-v1-aux-fp16-fp8-evex-vnni.NAMING.1]`).

use crate::detect;

/// Signedness of a VNNI byte operand: whether each byte is sign-extended (`Signed`) or
/// zero-extended (`Unsigned`) to the product width before multiplication
/// (spec section 8.6.5 `extend = sign_extend8 if signed else zero_extend8`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Sign {
    Signed,
    Unsigned,
}

/// How one accumulated dword total is written back (spec sections 8.6.5 / 8.7.5): the
/// non-saturating forms wrap, the `...DS` forms saturate — signed unless BOTH operands are
/// unsigned (`UU`), where the destination is the spec's unsigned accumulator.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Finalize {
    /// INT32 truncation == wrap modulo 2^32.
    Wrap,
    /// Signed-saturate to `[-2^31, 2^31-1]`.
    SatSigned,
    /// Unsigned-saturate to `[0, 2^32-1]`, stored as the `i32` bit pattern.
    SatUnsigned,
}

impl Finalize {
    /// Pick the write-back rule from a variant's saturation flag and operand sign pair.
    fn from_variant(saturating: bool, both_unsigned: bool) -> Self {
        match (saturating, both_unsigned) {
            (false, _) => Finalize::Wrap,
            (true, false) => Finalize::SatSigned,
            (true, true) => Finalize::SatUnsigned,
        }
    }
}

/// Widen one raw operand byte to its `i32` value per its signedness (spec section 8.6.5,
/// "8-bit -> 16-bit sign/zero extension before multiplication"; widened to `i32` here since
/// the product of two extended bytes fits comfortably in `i32`).
#[inline]
fn widen_byte(raw: u8, sign: Sign) -> i32 {
    match sign {
        Sign::Signed => (raw as i8) as i32,
        Sign::Unsigned => raw as i32,
    }
}

/// Widen one raw operand word to its `i64` value per its signedness (spec section 8.7.5,
/// "16-bit -> 32-bit extension before multiplication"). Widened to `i64` here so the `i32`
/// dword product `extend1(word) * extend2(word)` cannot overflow the intermediate before it
/// is summed into the wider accumulator.
#[inline]
fn widen_word(raw: u16, sign: Sign) -> i64 {
    match sign {
        Sign::Signed => (raw as i16) as i64,
        Sign::Unsigned => raw as i64,
    }
}

/// Shared byte-VNNI oracle parameterized over the sign matrix and saturation mode.
///
/// Per `i32` dword lane `i`: copy `dst[i]` (read-modify-write — `dst` is an input, never
/// mutated in place), widen the four byte pairs `a[4i+k]`/`b[4i+k]` per `(a_sign, b_sign)`,
/// and form `total = dst[i] + p1 + p2 + p3 + p4` in `i64` (wide enough that the sum cannot
/// overflow before the saturation/wrap decision, spec section 8.6.5). Then:
///
/// * non-saturating: keep the low 32 bits (`total & 0xFFFFFFFF`), i.e. wrap modulo 2^32;
/// * saturating + both operands unsigned (`UU`): clamp to `[0, u32::MAX]`;
/// * saturating otherwise: clamp to `[i32::MIN, i32::MAX]`.
///
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.BYTE_VNNI.1]`
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.BYTE_VNNI.1-1]`
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.BYTE_VNNI.1-2]`
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.BYTE_VNNI.1-3]`
#[inline]
fn dpb_oracle(
    dst: [i32; 16],
    a: [u8; 64],
    b: [u8; 64],
    a_sign: Sign,
    b_sign: Sign,
    saturating: bool,
) -> [i32; 16] {
    let both_unsigned = a_sign == Sign::Unsigned && b_sign == Sign::Unsigned;
    let finalize = Finalize::from_variant(saturating, both_unsigned);
    core::array::from_fn(|i| {
        // total accumulates in i64 so dst + four byte-products never overflows the
        // intermediate before the saturate/wrap decision (spec section 8.6.5). For the
        // all-unsigned (UU) form the destination dword is the *unsigned* accumulator the
        // spec's `unsigned_dword_saturate` operates on, so it is read as a u32 (bit
        // pattern 0xffffffff -> 2^32-1); every signed form reads it as a signed i32.
        let mut total: i64 = if both_unsigned {
            dst[i] as u32 as i64
        } else {
            dst[i] as i64
        };
        for k in 0..4 {
            let p = widen_byte(a[4 * i + k], a_sign) * widen_byte(b[4 * i + k], b_sign);
            total += p as i64;
        }
        finalize_dword(total, finalize)
    })
}

/// Shared word-VNNI oracle parameterized over the sign matrix and saturation mode (spec
/// section 8.7.5, `vpdpw_d`). Structurally identical to [`dpb_oracle`] but with **two** word
/// products per dword lane instead of four byte products.
///
/// Per `i32` dword lane `i`: copy `dst[i]` (read-modify-write — `dst` is an input, never
/// mutated in place), widen the two word pairs `a[2i+k]`/`b[2i+k]` per `(a_sign, b_sign)` to
/// `i64`, form each `i32`-range product, and accumulate `total = dst[i] + p1 + p2` in `i64`.
/// Then the saturation/wrap decision is identical to the byte group (delegated to
/// [`finalize_dword`]):
///
/// * non-saturating: keep the low 32 bits, i.e. wrap modulo 2^32;
/// * saturating + both operands unsigned (`UU`): clamp to `[0, u32::MAX]`;
/// * saturating otherwise: clamp to `[i32::MIN, i32::MAX]`.
///
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.WORD_VNNI.1]`
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.WORD_VNNI.1-1]`
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.WORD_VNNI.1-2]`
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.WORD_VNNI.1-3]`
#[inline]
fn dpw_oracle(
    dst: [i32; 16],
    a: [u16; 32],
    b: [u16; 32],
    a_sign: Sign,
    b_sign: Sign,
    saturating: bool,
) -> [i32; 16] {
    let both_unsigned = a_sign == Sign::Unsigned && b_sign == Sign::Unsigned;
    let finalize = Finalize::from_variant(saturating, both_unsigned);
    core::array::from_fn(|i| {
        // Same widening discipline as the byte oracle: the UU form reads the destination
        // dword as the unsigned accumulator (spec section 8.7.5 `unsigned_dword_saturate`).
        let mut total: i64 = if both_unsigned {
            dst[i] as u32 as i64
        } else {
            dst[i] as i64
        };
        for k in 0..2 {
            // Products MUST form in i64 (`widen_word` returns i64): the UU product
            // 65535*65535 = 4294836225 exceeds i32::MAX, so an i32 intermediate would
            // overflow. Accumulated in i64 alongside dst.
            let p = widen_word(a[2 * i + k], a_sign) * widen_word(b[2 * i + k], b_sign);
            total += p;
        }
        finalize_dword(total, finalize)
    })
}

/// Apply the shared VNNI saturation/wrap rule to one accumulated dword (spec sections 8.6.5 /
/// 8.7.5, identical for byte and word groups). `total` is the wider-type (`i64`) accumulation
/// `dst + sum-of-products`; the result is the `i32` bit pattern the hardware writes back per
/// the [`Finalize`] rule (e.g. unsigned-saturated `2^32-1 -> -1`).
#[inline]
fn finalize_dword(total: i64, finalize: Finalize) -> i32 {
    match finalize {
        // INT32 truncation == wrap modulo 2^32, matching dpbssd_scalar.
        Finalize::Wrap => total as i32,
        Finalize::SatSigned => total.clamp(i32::MIN as i64, i32::MAX as i64) as i32,
        Finalize::SatUnsigned => total.clamp(0, u32::MAX as i64) as u32 as i32,
    }
}

/// Standard dispatcher body shared by every family-F/G primitive: under `feature="native"`
/// on x86_64 with `AVX10_V1_AUX` detected it calls the `_hw` slot (real EVEX VNNI via the C
/// shim), otherwise the `_scalar` oracle. `detect` is consulted so the dispatch shape matches
/// `DISPATCH.1`/`DISPATCH.2`. The original operands are forwarded unchanged; both the `_hw`
/// shim and the oracle apply the correct sign-extension themselves.
macro_rules! dispatch {
    ($scalar:ident, $hw:ident, $dst:ident, $a:ident, $b:ident) => {{
        #[cfg(all(target_arch = "x86_64", feature = "native"))]
        {
            if detect::has_avx10_v1_aux() {
                // SAFETY: `has_avx10_v1_aux()` confirmed full AVX10.2 (the feature set this shim's
                // translation unit is compiled for) plus OS XSAVE state immediately above.
                return unsafe { $hw($dst, $a, $b) };
            }
        }
        let _ = detect::has_avx10_v1_aux; // keep `detect` referenced on every target
        $scalar($dst, $a, $b)
    }};
}

/// Define the `_hw` native slot for one family-F/G primitive: marshal the typed lane arrays
/// into the matching `extern "C"` shim and read the 16-lane `i32` result back.
///
/// # Safety
/// Every generated `_hw` fn is `unsafe`: the CPU must support `AVX10_V1_AUX` (the EVEX form
/// would otherwise #UD). Callers reach them only through the public dispatchers, which check
/// `detect::has_avx10_v1_aux()` first.
#[cfg(all(target_arch = "x86_64", feature = "native"))]
macro_rules! define_hw {
    ($hw:ident, $shim:ident, $at:ty, $bt:ty, $alen:expr, $blen:expr) => {
        pub(crate) unsafe fn $hw(dst: [i32; 16], a: [$at; $alen], b: [$bt; $blen]) -> [i32; 16] {
            let mut out = [0i32; 16];
            $shim(dst.as_ptr(), a.as_ptr(), b.as_ptr(), out.as_mut_ptr());
            out
        }
    };
}

// Native `_hw` slots, one per family-F/G primitive, defined directly in this module so the
// dispatchers can reach them. Pointer types match the C shim exactly.
#[cfg(all(target_arch = "x86_64", feature = "native"))]
mod hw_slots {
    use crate::native::{
        ace_native_dpbssd, ace_native_dpbssds, ace_native_dpbsud, ace_native_dpbsuds,
        ace_native_dpbuud, ace_native_dpbuuds, ace_native_dpwsud, ace_native_dpwsuds,
        ace_native_dpwusd, ace_native_dpwusds, ace_native_dpwuud, ace_native_dpwuuds,
    };
    define_hw!(dpbssd_hw, ace_native_dpbssd, i8, i8, 64, 64);
    define_hw!(dpbssds_hw, ace_native_dpbssds, i8, i8, 64, 64);
    define_hw!(dpbsud_hw, ace_native_dpbsud, i8, u8, 64, 64);
    define_hw!(dpbsuds_hw, ace_native_dpbsuds, i8, u8, 64, 64);
    define_hw!(dpbuud_hw, ace_native_dpbuud, u8, u8, 64, 64);
    define_hw!(dpbuuds_hw, ace_native_dpbuuds, u8, u8, 64, 64);
    define_hw!(dpwsud_hw, ace_native_dpwsud, i16, u16, 32, 32);
    define_hw!(dpwsuds_hw, ace_native_dpwsuds, i16, u16, 32, 32);
    define_hw!(dpwusd_hw, ace_native_dpwusd, u16, i16, 32, 32);
    define_hw!(dpwusds_hw, ace_native_dpwusds, u16, i16, 32, 32);
    define_hw!(dpwuud_hw, ace_native_dpwuud, u16, u16, 32, 32);
    define_hw!(dpwuuds_hw, ace_native_dpwuuds, u16, u16, 32, 32);
}
#[cfg(all(target_arch = "x86_64", feature = "native"))]
use hw_slots::*;

// ---- SS: signed x signed -------------------------------------------------------------

/// Byte VNNI, signed x signed, non-saturating (`VPDPBSSD`, EVEX 512-bit).
///
/// Per dword lane: `result = dst + Σ_{k=0..4} a·b` with both operands sign-extended, wrapping
/// modulo 2^32 (`[avx10-v1-aux-fp16-fp8-evex-vnni.BYTE_VNNI.1]`,
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.BYTE_VNNI.1-2]`). DISTINCT from the 256-bit
/// [`crate::dpbssd`] (OQ-1). `[avx10-v1-aux-fp16-fp8-evex-vnni.DISPATCH.1]`
pub fn dpbssd(dst: [i32; 16], a: [i8; 64], b: [i8; 64]) -> [i32; 16] {
    dispatch!(dpbssd_scalar, dpbssd_hw, dst, a, b)
}

/// Portable reference oracle for [`dpbssd`] (SS, non-saturating).
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.ORACLE.1]`
pub fn dpbssd_scalar(dst: [i32; 16], a: [i8; 64], b: [i8; 64]) -> [i32; 16] {
    dpb_oracle(
        dst,
        bytes_i8(a),
        bytes_i8(b),
        Sign::Signed,
        Sign::Signed,
        false,
    )
}

/// Byte VNNI, signed x signed, signed-saturating (`VPDPBSSDS`, EVEX 512-bit).
///
/// As [`dpbssd`] but the accumulation signed-saturates to `[-2^31, 2^31-1]`
/// (`[avx10-v1-aux-fp16-fp8-evex-vnni.BYTE_VNNI.1-3]`).
pub fn dpbssds(dst: [i32; 16], a: [i8; 64], b: [i8; 64]) -> [i32; 16] {
    dispatch!(dpbssds_scalar, dpbssds_hw, dst, a, b)
}

/// Portable reference oracle for [`dpbssds`] (SS, signed-saturate).
pub fn dpbssds_scalar(dst: [i32; 16], a: [i8; 64], b: [i8; 64]) -> [i32; 16] {
    dpb_oracle(
        dst,
        bytes_i8(a),
        bytes_i8(b),
        Sign::Signed,
        Sign::Signed,
        true,
    )
}

// ---- SU: signed x unsigned -----------------------------------------------------------

/// Byte VNNI, signed x unsigned, non-saturating (`VPDPBSUD`, EVEX 512-bit).
///
/// `a` is sign-extended, `b` is zero-extended; wraps modulo 2^32
/// (`[avx10-v1-aux-fp16-fp8-evex-vnni.BYTE_VNNI.1]`,
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.BYTE_VNNI.1-2]`).
pub fn dpbsud(dst: [i32; 16], a: [i8; 64], b: [u8; 64]) -> [i32; 16] {
    dispatch!(dpbsud_scalar, dpbsud_hw, dst, a, b)
}

/// Portable reference oracle for [`dpbsud`] (SU, non-saturating).
pub fn dpbsud_scalar(dst: [i32; 16], a: [i8; 64], b: [u8; 64]) -> [i32; 16] {
    dpb_oracle(dst, bytes_i8(a), b, Sign::Signed, Sign::Unsigned, false)
}

/// Byte VNNI, signed x unsigned, signed-saturating (`VPDPBSUDS`, EVEX 512-bit).
///
/// As [`dpbsud`] but signed-saturates to `[-2^31, 2^31-1]` (not unsigned: only `UU`
/// saturates unsigned, `[avx10-v1-aux-fp16-fp8-evex-vnni.BYTE_VNNI.1-3]`).
pub fn dpbsuds(dst: [i32; 16], a: [i8; 64], b: [u8; 64]) -> [i32; 16] {
    dispatch!(dpbsuds_scalar, dpbsuds_hw, dst, a, b)
}

/// Portable reference oracle for [`dpbsuds`] (SU, signed-saturate).
pub fn dpbsuds_scalar(dst: [i32; 16], a: [i8; 64], b: [u8; 64]) -> [i32; 16] {
    dpb_oracle(dst, bytes_i8(a), b, Sign::Signed, Sign::Unsigned, true)
}

// ---- UU: unsigned x unsigned ---------------------------------------------------------

/// Byte VNNI, unsigned x unsigned, non-saturating (`VPDPBUUD`, EVEX 512-bit).
///
/// Both operands zero-extended; wraps modulo 2^32
/// (`[avx10-v1-aux-fp16-fp8-evex-vnni.BYTE_VNNI.1]`,
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.BYTE_VNNI.1-2]`).
pub fn dpbuud(dst: [i32; 16], a: [u8; 64], b: [u8; 64]) -> [i32; 16] {
    dispatch!(dpbuud_scalar, dpbuud_hw, dst, a, b)
}

/// Portable reference oracle for [`dpbuud`] (UU, non-saturating).
pub fn dpbuud_scalar(dst: [i32; 16], a: [u8; 64], b: [u8; 64]) -> [i32; 16] {
    dpb_oracle(dst, a, b, Sign::Unsigned, Sign::Unsigned, false)
}

/// Byte VNNI, unsigned x unsigned, unsigned-saturating (`VPDPBUUDS`, EVEX 512-bit).
///
/// The only family-F form that unsigned-saturates: the accumulation clamps to `[0, 2^32-1]`
/// because both operands are unsigned (`[avx10-v1-aux-fp16-fp8-evex-vnni.BYTE_VNNI.1-3]`).
pub fn dpbuuds(dst: [i32; 16], a: [u8; 64], b: [u8; 64]) -> [i32; 16] {
    dispatch!(dpbuuds_scalar, dpbuuds_hw, dst, a, b)
}

/// Portable reference oracle for [`dpbuuds`] (UU, unsigned-saturate to `[0, 2^32-1]`).
pub fn dpbuuds_scalar(dst: [i32; 16], a: [u8; 64], b: [u8; 64]) -> [i32; 16] {
    dpb_oracle(dst, a, b, Sign::Unsigned, Sign::Unsigned, true)
}

// ======================================================================================
// Family G — EVEX word VNNI (spec section 8.7). Two word products per dword lane. The word
// group has NO signed×signed form; the sign matrix is SU / US / UU only
// (`[avx10-v1-aux-fp16-fp8-evex-vnni.WORD_VNNI.1]`).
// ======================================================================================

// ---- SU: signed x unsigned -----------------------------------------------------------

/// Word VNNI, signed x unsigned, non-saturating (`VPDPWSUD`, EVEX 512-bit).
///
/// Per dword lane: `result = dst + Σ_{k=0..2} a·b` with `a` sign-extended and `b`
/// zero-extended, wrapping modulo 2^32 (`[avx10-v1-aux-fp16-fp8-evex-vnni.WORD_VNNI.1]`,
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.WORD_VNNI.1-2]`). `[avx10-v1-aux-fp16-fp8-evex-vnni.DISPATCH.1]`
pub fn dpwsud(dst: [i32; 16], a: [i16; 32], b: [u16; 32]) -> [i32; 16] {
    dispatch!(dpwsud_scalar, dpwsud_hw, dst, a, b)
}

/// Portable reference oracle for [`dpwsud`] (SU, non-saturating).
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.ORACLE.1]`
pub fn dpwsud_scalar(dst: [i32; 16], a: [i16; 32], b: [u16; 32]) -> [i32; 16] {
    dpw_oracle(dst, words_i16(a), b, Sign::Signed, Sign::Unsigned, false)
}

/// Word VNNI, signed x unsigned, signed-saturating (`VPDPWSUDS`, EVEX 512-bit).
///
/// As [`dpwsud`] but signed-saturates to `[-2^31, 2^31-1]` (not unsigned: only `UU`
/// saturates unsigned, `[avx10-v1-aux-fp16-fp8-evex-vnni.WORD_VNNI.1-3]`).
pub fn dpwsuds(dst: [i32; 16], a: [i16; 32], b: [u16; 32]) -> [i32; 16] {
    dispatch!(dpwsuds_scalar, dpwsuds_hw, dst, a, b)
}

/// Portable reference oracle for [`dpwsuds`] (SU, signed-saturate).
pub fn dpwsuds_scalar(dst: [i32; 16], a: [i16; 32], b: [u16; 32]) -> [i32; 16] {
    dpw_oracle(dst, words_i16(a), b, Sign::Signed, Sign::Unsigned, true)
}

// ---- US: unsigned x signed -----------------------------------------------------------

/// Word VNNI, unsigned x signed, non-saturating (`VPDPWUSD`, EVEX 512-bit).
///
/// `a` is zero-extended, `b` is sign-extended; wraps modulo 2^32
/// (`[avx10-v1-aux-fp16-fp8-evex-vnni.WORD_VNNI.1]`,
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.WORD_VNNI.1-2]`).
pub fn dpwusd(dst: [i32; 16], a: [u16; 32], b: [i16; 32]) -> [i32; 16] {
    dispatch!(dpwusd_scalar, dpwusd_hw, dst, a, b)
}

/// Portable reference oracle for [`dpwusd`] (US, non-saturating).
pub fn dpwusd_scalar(dst: [i32; 16], a: [u16; 32], b: [i16; 32]) -> [i32; 16] {
    dpw_oracle(dst, a, words_i16(b), Sign::Unsigned, Sign::Signed, false)
}

/// Word VNNI, unsigned x signed, signed-saturating (`VPDPWUSDS`, EVEX 512-bit).
///
/// As [`dpwusd`] but signed-saturates to `[-2^31, 2^31-1]`
/// (`[avx10-v1-aux-fp16-fp8-evex-vnni.WORD_VNNI.1-3]`).
pub fn dpwusds(dst: [i32; 16], a: [u16; 32], b: [i16; 32]) -> [i32; 16] {
    dispatch!(dpwusds_scalar, dpwusds_hw, dst, a, b)
}

/// Portable reference oracle for [`dpwusds`] (US, signed-saturate).
pub fn dpwusds_scalar(dst: [i32; 16], a: [u16; 32], b: [i16; 32]) -> [i32; 16] {
    dpw_oracle(dst, a, words_i16(b), Sign::Unsigned, Sign::Signed, true)
}

// ---- UU: unsigned x unsigned ---------------------------------------------------------

/// Word VNNI, unsigned x unsigned, non-saturating (`VPDPWUUD`, EVEX 512-bit).
///
/// Both operands zero-extended; wraps modulo 2^32
/// (`[avx10-v1-aux-fp16-fp8-evex-vnni.WORD_VNNI.1]`,
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.WORD_VNNI.1-2]`).
pub fn dpwuud(dst: [i32; 16], a: [u16; 32], b: [u16; 32]) -> [i32; 16] {
    dispatch!(dpwuud_scalar, dpwuud_hw, dst, a, b)
}

/// Portable reference oracle for [`dpwuud`] (UU, non-saturating).
pub fn dpwuud_scalar(dst: [i32; 16], a: [u16; 32], b: [u16; 32]) -> [i32; 16] {
    dpw_oracle(dst, a, b, Sign::Unsigned, Sign::Unsigned, false)
}

/// Word VNNI, unsigned x unsigned, unsigned-saturating (`VPDPWUUDS`, EVEX 512-bit).
///
/// The only family-G form that unsigned-saturates: the accumulation clamps to `[0, 2^32-1]`
/// because both operands are unsigned (`[avx10-v1-aux-fp16-fp8-evex-vnni.WORD_VNNI.1-3]`).
pub fn dpwuuds(dst: [i32; 16], a: [u16; 32], b: [u16; 32]) -> [i32; 16] {
    dispatch!(dpwuuds_scalar, dpwuuds_hw, dst, a, b)
}

/// Portable reference oracle for [`dpwuuds`] (UU, unsigned-saturate to `[0, 2^32-1]`).
pub fn dpwuuds_scalar(dst: [i32; 16], a: [u16; 32], b: [u16; 32]) -> [i32; 16] {
    dpw_oracle(dst, a, b, Sign::Unsigned, Sign::Unsigned, true)
}

/// Reinterpret an `[i8; 64]` operand as its raw `[u8; 64]` bytes (the oracle re-applies the
/// correct sign-extension per [`Sign`]). A pure bit-cast; no value changes.
#[inline]
fn bytes_i8(v: [i8; 64]) -> [u8; 64] {
    core::array::from_fn(|i| v[i] as u8)
}

/// Reinterpret an `[i16; 32]` operand as its raw `[u16; 32]` words (the oracle re-applies the
/// correct sign-extension per [`Sign`]). A pure bit-cast; no value changes.
#[inline]
fn words_i16(v: [i16; 32]) -> [u16; 32] {
    core::array::from_fn(|i| v[i] as u16)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Non-saturating SS dot product reproduces the spec dot-product-plus-dst, and matches
    /// the structure of iteration-0's `dpbssd_scalar` per lane
    /// (`[avx10-v1-aux-fp16-fp8-evex-vnni.BYTE_VNNI.1]`,
    /// `[avx10-v1-aux-fp16-fp8-evex-vnni.BYTE_VNNI.1-1]`).
    #[test]
    fn known_value_ss_dot_product() {
        let mut a = [0i8; 64];
        let mut b = [0i8; 64];
        // lane 0: 1*1 + 2*2 + 3*3 + 4*4 = 30; lane 1: (-1)*5 + (-2)*6 + 3*7 + 4*8 = 36.
        for k in 0..4 {
            a[k] = (k as i8) + 1;
            b[k] = (k as i8) + 1;
        }
        a[4] = -1;
        a[5] = -2;
        a[6] = 3;
        a[7] = 4;
        b[4] = 5;
        b[5] = 6;
        b[6] = 7;
        b[7] = 8;
        // dst is a read-modify-write input: lane 0 carries +100, lane 1 carries -10.
        let mut dst = [0i32; 16];
        dst[0] = 100;
        dst[1] = -10;
        let out = dpbssd(dst, a, b);
        assert_eq!(out[0], 100 + 30, "dst[0] + (1+4+9+16)");
        assert_eq!(out[1], -10 + (-5 - 12 + 21 + 32), "dst[1] + 36");
        assert_eq!(out[2], 0, "untouched lane stays at dst");
    }

    /// Non-saturating accumulation wraps modulo 2^32 like `dpbssd_scalar`
    /// (`[avx10-v1-aux-fp16-fp8-evex-vnni.BYTE_VNNI.1-2]`). Discriminates wrap from saturate:
    /// a saturating model would clamp to i32::MAX here, not wrap to a negative.
    #[test]
    fn known_value_wrap_mod_2_32() {
        let mut a = [0i8; 64];
        let mut b = [0i8; 64];
        // lane 0: 127*127 = 16129, summed four times = 64516.
        for k in 0..4 {
            a[k] = 127;
            b[k] = 127;
        }
        let mut dst = [0i32; 16];
        dst[0] = i32::MAX; // 2147483647 + 64516 overflows i32 -> wraps negative
        let out = dpbssd(dst, a, b);
        let want = (i32::MAX as i64 + 64516) as i32; // wrapping (two's complement) result
        assert_eq!(out[0], want, "non-saturating wraps mod 2^32");
        assert!(
            out[0] < 0,
            "wrap produced a negative dword, NOT a clamp to i32::MAX"
        );
    }

    /// Signed-saturate clamps a positive overflow to i32::MAX (`dpbssds`,
    /// `[avx10-v1-aux-fp16-fp8-evex-vnni.BYTE_VNNI.1-3]`). DISCRIMINATING vs a wrapping
    /// model: dst=i32::MAX with a positive product sum saturates to i32::MAX (a wrapping
    /// model would yield a negative dword, ruled out by the assertion), and dst=i32::MIN
    /// with a negative product sum saturates to i32::MIN.
    #[test]
    fn known_value_signed_saturate() {
        let mut a = [0i8; 64];
        let mut b = [0i8; 64];
        for k in 0..4 {
            a[k] = 127;
            b[k] = 127;
        }
        let mut dst = [0i32; 16];
        dst[0] = i32::MAX;
        dst[1] = i32::MIN;
        // lane 1: negative product sum on top of i32::MIN -> saturate to i32::MIN.
        let mut a2 = a;
        let mut b2 = b;
        for k in 0..4 {
            a2[4 + k] = -128;
            b2[4 + k] = 127;
        }
        let out = dpbssds(dst, a2, b2);
        assert_eq!(out[0], i32::MAX, "positive overflow saturates to i32::MAX");
        assert_eq!(out[1], i32::MIN, "negative overflow saturates to i32::MIN");
    }

    /// Unsigned-saturate clamps to 2^32-1 (stored as -1) for the UU `dpbuuds` form
    /// (`[avx10-v1-aux-fp16-fp8-evex-vnni.BYTE_VNNI.1-3]`). DISCRIMINATING: a signed-saturate
    /// model would clamp to i32::MAX (0x7fffffff); unsigned-saturate yields 0xffffffff (-1).
    #[test]
    fn known_value_unsigned_saturate() {
        let mut a = [0u8; 64];
        let mut b = [0u8; 64];
        for k in 0..4 {
            a[k] = 255;
            b[k] = 255;
        }
        let mut dst = [0i32; 16];
        // dst = u32::MAX bit pattern (-1) + 4*255*255 = 4294967295 + 260100 -> clamps to
        // u32::MAX. Signed interpretation of dst is -1, but UU treats the accumulation as
        // unsigned and saturates to 2^32-1.
        dst[0] = -1; // 0xffffffff == u32::MAX
        let out = dpbuuds(dst, a, b);
        assert_eq!(
            out[0], -1i32,
            "unsigned-saturate clamps to 2^32-1 (bit pattern 0xffffffff)"
        );
        assert_ne!(
            out[0],
            i32::MAX,
            "NOT a signed-saturate clamp to 0x7fffffff"
        );
    }

    /// The public dispatcher equals the scalar oracle for every family-F form
    /// (`[avx10-v1-aux-fp16-fp8-evex-vnni.ORACLE.2]`).
    #[test]
    fn dispatcher_matches_oracle() {
        let dst: [i32; 16] = core::array::from_fn(|i| i as i32 * 1000 - 5000);
        let ai: [i8; 64] = core::array::from_fn(|i| i as i8 - 32);
        let bi: [i8; 64] = core::array::from_fn(|i| (i as i8).wrapping_mul(3));
        let au: [u8; 64] = core::array::from_fn(|i| i as u8);
        let bu: [u8; 64] = core::array::from_fn(|i| (i as u8).wrapping_mul(5));
        assert_eq!(dpbssd(dst, ai, bi), dpbssd_scalar(dst, ai, bi));
        assert_eq!(dpbssds(dst, ai, bi), dpbssds_scalar(dst, ai, bi));
        assert_eq!(dpbsud(dst, ai, bu), dpbsud_scalar(dst, ai, bu));
        assert_eq!(dpbsuds(dst, ai, bu), dpbsuds_scalar(dst, ai, bu));
        assert_eq!(dpbuud(dst, au, bu), dpbuud_scalar(dst, au, bu));
        assert_eq!(dpbuuds(dst, au, bu), dpbuuds_scalar(dst, au, bu));
    }

    // ---- Family G (word VNNI) known-value tests ------------------------------------

    /// Non-saturating word dot product reproduces `dst + Σ a·b` over the SU/US/UU sign matrix
    /// (`[avx10-v1-aux-fp16-fp8-evex-vnni.WORD_VNNI.1]`,
    /// `[avx10-v1-aux-fp16-fp8-evex-vnni.WORD_VNNI.1-1]`). Each lane consumes exactly TWO word
    /// products. DISCRIMINATING on signedness: lane 0 uses a negative `a`/`b` word against an
    /// unsigned operand, so SU and US disagree with UU (which zero-extends both) — a model
    /// that ignored sign-extension would produce a different, larger lane value.
    #[test]
    fn known_value_word_dot_product() {
        // lane 0: a = [-1, 2], with the other operand = [3, 4].
        //   SU (a signed, b unsigned): (-1)*3 + 2*4 = -3 + 8 = 5
        //   US (a unsigned, b signed): a here is the unsigned operand. Using a=[65535, 2]
        //     interpreted unsigned = [65535, 2] times signed b=[3,4]? We test US separately.
        // To keep the two operands' roles clear we pin SU explicitly.
        let mut a = [0i16; 32];
        let mut b = [0u16; 32];
        a[0] = -1;
        a[1] = 2;
        b[0] = 3;
        b[1] = 4;
        // lane 1: 10*100 + 20*200 = 1000 + 4000 = 5000.
        a[2] = 10;
        a[3] = 20;
        b[2] = 100;
        b[3] = 200;
        let mut dst = [0i32; 16];
        dst[0] = 1000; // RMW: lane 0 carries +1000
        dst[1] = -50; // lane 1 carries -50
        let out = dpwsud(dst, a, b);
        assert_eq!(out[0], 1000 + 5, "SU lane0: dst + (-1*3 + 2*4)");
        assert_eq!(out[1], -50 + 5000, "SU lane1: dst + (10*100 + 20*200)");
        assert_eq!(out[2], 0, "untouched lane stays at dst");

        // US form: a is the *unsigned* operand, b is signed. Reuse mirrored values.
        let mut ua = [0u16; 32];
        let mut sb = [0i16; 32];
        ua[0] = 3;
        ua[1] = 4;
        sb[0] = -1;
        sb[1] = 2;
        let out_us = dpwusd([0i32; 16], ua, sb);
        // US: a unsigned, b signed -> 3*(-1) + 4*2 = -3 + 8 = 5.
        assert_eq!(out_us[0], 5, "US lane0: 3*(-1) + 4*2 = 5");

        // UU form: both zero-extended. 65535*1 + 0 = 65535 (NOT -1 if a were sign-extended).
        let mut uua = [0u16; 32];
        let mut uub = [0u16; 32];
        uua[0] = 65535;
        uub[0] = 1;
        let out_uu = dpwuud([0i32; 16], uua, uub);
        assert_eq!(
            out_uu[0], 65535,
            "UU zero-extends: 65535*1 = 65535, NOT (-1)*1 = -1"
        );
    }

    /// Word VNNI non-saturating accumulation wraps modulo 2^32
    /// (`[avx10-v1-aux-fp16-fp8-evex-vnni.WORD_VNNI.1-2]`). DISCRIMINATING wrap vs saturate:
    /// dst=i32::MAX plus a positive product sum wraps to a negative dword; a saturating model
    /// would clamp to i32::MAX instead.
    #[test]
    fn known_value_word_wrap_mod_2_32() {
        let mut a = [0i16; 32];
        let mut b = [0u16; 32];
        // lane 0 product sum: 1*2 + 1*2 = 4. dst = i32::MAX (2^31 - 1), so
        // total = 2147483647 + 4 = 2147483651, which exceeds i32::MAX. Non-saturating
        // truncation to 32 bits yields a NEGATIVE dword (2147483651 - 2^32 region: as i32
        // it is i32::MIN + 3 = -2147483645), whereas a saturating model would clamp to
        // i32::MAX. The modest product sum keeps the total in [2^31, 2^32) so wrap lands
        // negative rather than circling all the way back to a positive value.
        a[0] = 1;
        a[1] = 1;
        b[0] = 2;
        b[1] = 2;
        let mut dst = [0i32; 16];
        dst[0] = i32::MAX;
        let out = dpwsud(dst, a, b);
        let total = i32::MAX as i64 + 4;
        let want = total as i32; // wrapping truncation to 32 bits
        assert_eq!(out[0], want, "word VNNI non-saturating wraps mod 2^32");
        assert!(
            out[0] < 0,
            "wrap produced a negative dword, NOT a clamp to i32::MAX"
        );
    }

    /// Word VNNI signed-saturate clamps to the i32 range for the non-UU forms
    /// (`[avx10-v1-aux-fp16-fp8-evex-vnni.WORD_VNNI.1-3]`). DISCRIMINATING: dst=i32::MAX with a
    /// positive product sum saturates to i32::MAX (a wrapping model yields a negative dword),
    /// and dst=i32::MIN with a negative product sum (US form, signed b negative) saturates to
    /// i32::MIN.
    #[test]
    fn known_value_word_signed_saturate() {
        // SU positive overflow on lane 0.
        let mut a = [0i16; 32];
        let mut b = [0u16; 32];
        a[0] = i16::MAX;
        a[1] = i16::MAX;
        b[0] = u16::MAX;
        b[1] = u16::MAX;
        let mut dst = [0i32; 16];
        dst[0] = i32::MAX;
        let out = dpwsuds(dst, a, b);
        assert_eq!(
            out[0],
            i32::MAX,
            "SU positive overflow saturates to i32::MAX"
        );

        // US negative overflow on lane 0: a (unsigned) large, b (signed) negative.
        let mut ua = [0u16; 32];
        let mut sb = [0i16; 32];
        ua[0] = u16::MAX;
        ua[1] = u16::MAX;
        sb[0] = i16::MIN; // -32768
        sb[1] = i16::MIN;
        let mut dst2 = [0i32; 16];
        dst2[0] = i32::MIN;
        let out_us = dpwusds(dst2, ua, sb);
        assert_eq!(
            out_us[0],
            i32::MIN,
            "US negative overflow saturates to i32::MIN"
        );
    }

    /// Word VNNI unsigned-saturate clamps to 2^32-1 (bit pattern 0xffffffff == -1) for the UU
    /// `dpwuuds` form (`[avx10-v1-aux-fp16-fp8-evex-vnni.WORD_VNNI.1-3]`). DISCRIMINATING: a
    /// signed-saturate model would clamp to i32::MAX (0x7fffffff); unsigned-saturate yields
    /// 0xffffffff.
    #[test]
    fn known_value_word_unsigned_saturate() {
        let mut a = [0u16; 32];
        let mut b = [0u16; 32];
        // lane 0: 65535*65535 + 65535*65535 = 2 * 4294836225 = 8589672450 -> clamps to
        // u32::MAX (4294967295) even before adding dst.
        a[0] = u16::MAX;
        a[1] = u16::MAX;
        b[0] = u16::MAX;
        b[1] = u16::MAX;
        let mut dst = [0i32; 16];
        dst[0] = -1; // 0xffffffff == u32::MAX accumulator input
        let out = dpwuuds(dst, a, b);
        assert_eq!(
            out[0], -1i32,
            "UU unsigned-saturate clamps to 2^32-1 (0xffffffff)"
        );
        assert_ne!(
            out[0],
            i32::MAX,
            "NOT a signed-saturate clamp to 0x7fffffff"
        );
    }

    /// The public dispatcher equals the scalar oracle for every family-G form
    /// (`[avx10-v1-aux-fp16-fp8-evex-vnni.ORACLE.2]`).
    #[test]
    fn word_dispatcher_matches_oracle() {
        let dst: [i32; 16] = core::array::from_fn(|i| i as i32 * 1000 - 5000);
        let ai: [i16; 32] = core::array::from_fn(|i| (i as i16).wrapping_mul(257) - 4096);
        let bi: [i16; 32] = core::array::from_fn(|i| (i as i16).wrapping_mul(131) - 2048);
        let au: [u16; 32] = core::array::from_fn(|i| (i as u16).wrapping_mul(521));
        let bu: [u16; 32] = core::array::from_fn(|i| (i as u16).wrapping_mul(331).wrapping_add(7));
        assert_eq!(dpwsud(dst, ai, bu), dpwsud_scalar(dst, ai, bu));
        assert_eq!(dpwsuds(dst, ai, bu), dpwsuds_scalar(dst, ai, bu));
        assert_eq!(dpwusd(dst, au, bi), dpwusd_scalar(dst, au, bi));
        assert_eq!(dpwusds(dst, au, bi), dpwusds_scalar(dst, au, bi));
        assert_eq!(dpwuud(dst, au, bu), dpwuud_scalar(dst, au, bu));
        assert_eq!(dpwuuds(dst, au, bu), dpwuuds_scalar(dst, au, bu));
    }
}

/// Property-based tests for families F (byte VNNI) and G (word VNNI). The hand-rolled tests
/// above pin specific values; these assert the invariants across a randomly-sampled slice of
/// the input space.
#[cfg(test)]
mod proptests {
    use super::*;
    use quickcheck::{quickcheck, Arbitrary, Gen, TestResult};

    /// Random `(dst, a, b)` triple with signed byte operands. `quickcheck` does not derive
    /// `Arbitrary` for arrays of this length, so each lane is filled independently.
    #[derive(Clone, Debug)]
    struct InputsSS {
        dst: [i32; 16],
        a: [i8; 64],
        b: [i8; 64],
    }
    impl Arbitrary for InputsSS {
        fn arbitrary(g: &mut Gen) -> Self {
            InputsSS {
                dst: core::array::from_fn(|_| i32::arbitrary(g)),
                a: core::array::from_fn(|_| i8::arbitrary(g)),
                b: core::array::from_fn(|_| i8::arbitrary(g)),
            }
        }
    }

    /// Random `(dst, a:[i8], b:[u8])` triple for the SU forms.
    #[derive(Clone, Debug)]
    struct InputsSU {
        dst: [i32; 16],
        a: [i8; 64],
        b: [u8; 64],
    }
    impl Arbitrary for InputsSU {
        fn arbitrary(g: &mut Gen) -> Self {
            InputsSU {
                dst: core::array::from_fn(|_| i32::arbitrary(g)),
                a: core::array::from_fn(|_| i8::arbitrary(g)),
                b: core::array::from_fn(|_| u8::arbitrary(g)),
            }
        }
    }

    /// Random `(dst, a:[u8], b:[u8])` triple for the UU forms.
    #[derive(Clone, Debug)]
    struct InputsUU {
        dst: [i32; 16],
        a: [u8; 64],
        b: [u8; 64],
    }
    impl Arbitrary for InputsUU {
        fn arbitrary(g: &mut Gen) -> Self {
            InputsUU {
                dst: core::array::from_fn(|_| i32::arbitrary(g)),
                a: core::array::from_fn(|_| u8::arbitrary(g)),
                b: core::array::from_fn(|_| u8::arbitrary(g)),
            }
        }
    }

    /// Random word-VNNI triple, `a` signed / `b` unsigned (the SU shape).
    #[derive(Clone, Debug)]
    struct InputsWordSU {
        dst: [i32; 16],
        a: [i16; 32],
        b: [u16; 32],
    }
    impl Arbitrary for InputsWordSU {
        fn arbitrary(g: &mut Gen) -> Self {
            InputsWordSU {
                dst: core::array::from_fn(|_| i32::arbitrary(g)),
                a: core::array::from_fn(|_| i16::arbitrary(g)),
                b: core::array::from_fn(|_| u16::arbitrary(g)),
            }
        }
    }

    /// Random word-VNNI triple, `a` unsigned / `b` signed (the US shape).
    #[derive(Clone, Debug)]
    struct InputsWordUS {
        dst: [i32; 16],
        a: [u16; 32],
        b: [i16; 32],
    }
    impl Arbitrary for InputsWordUS {
        fn arbitrary(g: &mut Gen) -> Self {
            InputsWordUS {
                dst: core::array::from_fn(|_| i32::arbitrary(g)),
                a: core::array::from_fn(|_| u16::arbitrary(g)),
                b: core::array::from_fn(|_| i16::arbitrary(g)),
            }
        }
    }

    /// Random word-VNNI triple, both operands unsigned (the UU shape).
    #[derive(Clone, Debug)]
    struct InputsWordUU {
        dst: [i32; 16],
        a: [u16; 32],
        b: [u16; 32],
    }
    impl Arbitrary for InputsWordUU {
        fn arbitrary(g: &mut Gen) -> Self {
            InputsWordUU {
                dst: core::array::from_fn(|_| i32::arbitrary(g)),
                a: core::array::from_fn(|_| u16::arbitrary(g)),
                b: core::array::from_fn(|_| u16::arbitrary(g)),
            }
        }
    }

    // Independent recomputation of the byte-VNNI result, used to anchor the RMW and
    // saturating-bounds properties without reusing the implementation's helper.
    fn expected(
        dst: [i32; 16],
        a: [i64; 64],
        b: [i64; 64],
        saturating: bool,
        uu: bool,
    ) -> [i32; 16] {
        core::array::from_fn(|i| {
            // Mirror the oracle: UU reads the destination dword as unsigned.
            let mut total = if uu {
                dst[i] as u32 as i64
            } else {
                dst[i] as i64
            };
            for k in 0..4 {
                total += a[4 * i + k] * b[4 * i + k];
            }
            if saturating {
                if uu {
                    total.clamp(0, u32::MAX as i64) as u32 as i32
                } else {
                    total.clamp(i32::MIN as i64, i32::MAX as i64) as i32
                }
            } else {
                total as i32
            }
        })
    }

    // Independent recomputation of the word-VNNI result (two products per lane), used to
    // anchor the family-G RMW and saturating-bounds properties without reusing the helper.
    fn expected_word(
        dst: [i32; 16],
        a: [i64; 32],
        b: [i64; 32],
        saturating: bool,
        uu: bool,
    ) -> [i32; 16] {
        core::array::from_fn(|i| {
            let mut total = if uu {
                dst[i] as u32 as i64
            } else {
                dst[i] as i64
            };
            for k in 0..2 {
                total += a[2 * i + k] * b[2 * i + k];
            }
            if saturating {
                if uu {
                    total.clamp(0, u32::MAX as i64) as u32 as i32
                } else {
                    total.clamp(i32::MIN as i64, i32::MAX as i64) as i32
                }
            } else {
                total as i32
            }
        })
    }

    quickcheck! {
        /// Public dispatcher == scalar oracle for every family-F form
        /// (`[avx10-v1-aux-fp16-fp8-evex-vnni.ORACLE.2]`).
        fn prop_public_matches_scalar_ss(x: InputsSS) -> bool {
            dpbssd(x.dst, x.a, x.b) == dpbssd_scalar(x.dst, x.a, x.b)
                && dpbssds(x.dst, x.a, x.b) == dpbssds_scalar(x.dst, x.a, x.b)
        }
        fn prop_public_matches_scalar_su(x: InputsSU) -> bool {
            dpbsud(x.dst, x.a, x.b) == dpbsud_scalar(x.dst, x.a, x.b)
                && dpbsuds(x.dst, x.a, x.b) == dpbsuds_scalar(x.dst, x.a, x.b)
        }
        fn prop_public_matches_scalar_uu(x: InputsUU) -> bool {
            dpbuud(x.dst, x.a, x.b) == dpbuud_scalar(x.dst, x.a, x.b)
                && dpbuuds(x.dst, x.a, x.b) == dpbuuds_scalar(x.dst, x.a, x.b)
        }

        /// RMW additivity: the non-saturating result equals `dst + sum-of-products`
        /// (wrapping), i.e. `dst` enters as a pure additive accumulator and is never mutated
        /// in place (`[avx10-v1-aux-fp16-fp8-evex-vnni.BYTE_VNNI.1-1]`). Verified against an
        /// independent recomputation, and confirmed that passing a zeroed `dst` then adding
        /// it back reproduces the full result.
        fn prop_rmw_additivity(x: InputsSS) -> bool {
            let a64: [i64; 64] = core::array::from_fn(|i| x.a[i] as i64);
            let b64: [i64; 64] = core::array::from_fn(|i| x.b[i] as i64);
            let want = expected(x.dst, a64, b64, false, false);
            let got = dpbssd_scalar(x.dst, x.a, x.b);
            // dst untouched: recompute with zero dst and add dst back (wrapping).
            let zero = dpbssd_scalar([0; 16], x.a, x.b);
            got == want && (0..16).all(|i| got[i] == x.dst[i].wrapping_add(zero[i]))
        }

        /// Saturating-bounds: the SS signed-saturate result equals the wider-type
        /// accumulation clamped to the i32 range (`[avx10-v1-aux-fp16-fp8-evex-vnni.BYTE_VNNI.1-3]`).
        /// Anchored to an independent wider-type recomputation.
        fn prop_saturating_bounds_ss(x: InputsSS) -> bool {
            let a64: [i64; 64] = core::array::from_fn(|i| x.a[i] as i64);
            let b64: [i64; 64] = core::array::from_fn(|i| x.b[i] as i64);
            dpbssds_scalar(x.dst, x.a, x.b) == expected(x.dst, a64, b64, true, false)
        }

        /// The UU unsigned-saturate result equals the wider-type accumulation clamped to
        /// `[0, 2^32-1]` (`[avx10-v1-aux-fp16-fp8-evex-vnni.BYTE_VNNI.1-3]`).
        fn prop_saturating_bounds_uu(x: InputsUU) -> bool {
            let a64: [i64; 64] = core::array::from_fn(|i| x.a[i] as i64);
            let b64: [i64; 64] = core::array::from_fn(|i| x.b[i] as i64);
            dpbuuds_scalar(x.dst, x.a, x.b) == expected(x.dst, a64, b64, true, true)
        }

        // ---- Family G (word VNNI) properties ---------------------------------------

        /// Public dispatcher == scalar oracle for every family-G form
        /// (`[avx10-v1-aux-fp16-fp8-evex-vnni.ORACLE.2]`,
        /// `[avx10-v1-aux-fp16-fp8-evex-vnni.WORD_VNNI.1]`).
        fn prop_word_public_matches_scalar_su(x: InputsWordSU) -> bool {
            dpwsud(x.dst, x.a, x.b) == dpwsud_scalar(x.dst, x.a, x.b)
                && dpwsuds(x.dst, x.a, x.b) == dpwsuds_scalar(x.dst, x.a, x.b)
        }
        fn prop_word_public_matches_scalar_us(x: InputsWordUS) -> bool {
            dpwusd(x.dst, x.a, x.b) == dpwusd_scalar(x.dst, x.a, x.b)
                && dpwusds(x.dst, x.a, x.b) == dpwusds_scalar(x.dst, x.a, x.b)
        }
        fn prop_word_public_matches_scalar_uu(x: InputsWordUU) -> bool {
            dpwuud(x.dst, x.a, x.b) == dpwuud_scalar(x.dst, x.a, x.b)
                && dpwuuds(x.dst, x.a, x.b) == dpwuuds_scalar(x.dst, x.a, x.b)
        }

        /// Word-VNNI RMW additivity: the non-saturating result equals `dst + sum-of-products`
        /// (wrapping); `dst` is a pure additive accumulator, never mutated in place
        /// (`[avx10-v1-aux-fp16-fp8-evex-vnni.WORD_VNNI.1-1]`). Verified against an independent
        /// recomputation and the zero-dst-plus-dst reconstruction.
        fn prop_word_rmw_additivity(x: InputsWordSU) -> bool {
            let a64: [i64; 32] = core::array::from_fn(|i| x.a[i] as i64);
            let b64: [i64; 32] = core::array::from_fn(|i| x.b[i] as i64);
            let want = expected_word(x.dst, a64, b64, false, false);
            let got = dpwsud_scalar(x.dst, x.a, x.b);
            let zero = dpwsud_scalar([0; 16], x.a, x.b);
            got == want && (0..16).all(|i| got[i] == x.dst[i].wrapping_add(zero[i]))
        }

        /// Word-VNNI signed-saturating bounds (SU form): the result equals the wider-type
        /// accumulation clamped to the i32 range (`[avx10-v1-aux-fp16-fp8-evex-vnni.WORD_VNNI.1-3]`).
        fn prop_word_saturating_bounds_su(x: InputsWordSU) -> bool {
            let a64: [i64; 32] = core::array::from_fn(|i| x.a[i] as i64);
            let b64: [i64; 32] = core::array::from_fn(|i| x.b[i] as i64);
            dpwsuds_scalar(x.dst, x.a, x.b) == expected_word(x.dst, a64, b64, true, false)
        }

        /// Word-VNNI signed-saturating bounds (US form): same i32-range clamp, confirming the
        /// US form signed-saturates (NOT unsigned) (`[avx10-v1-aux-fp16-fp8-evex-vnni.WORD_VNNI.1-3]`).
        fn prop_word_saturating_bounds_us(x: InputsWordUS) -> bool {
            let a64: [i64; 32] = core::array::from_fn(|i| x.a[i] as i64);
            let b64: [i64; 32] = core::array::from_fn(|i| x.b[i] as i64);
            dpwusds_scalar(x.dst, x.a, x.b) == expected_word(x.dst, a64, b64, true, false)
        }

        /// Word-VNNI unsigned-saturating bounds (UU form): the result equals the wider-type
        /// accumulation clamped to `[0, 2^32-1]` (`[avx10-v1-aux-fp16-fp8-evex-vnni.WORD_VNNI.1-3]`).
        fn prop_word_saturating_bounds_uu(x: InputsWordUU) -> bool {
            let a64: [i64; 32] = core::array::from_fn(|i| x.a[i] as i64);
            let b64: [i64; 32] = core::array::from_fn(|i| x.b[i] as i64);
            dpwuuds_scalar(x.dst, x.a, x.b) == expected_word(x.dst, a64, b64, true, true)
        }

        // ---- Cross-family algebraic properties (Phase 9) ---------------------------

        /// VNNI lane independence (byte VNNI): output dword lane `i` depends only on `dst[i]`
        /// and the four byte pairs `a[4i..4i+4]` / `b[4i..4i+4]`. Zeroing every OTHER lane's
        /// operands must not change lane `i`'s result
        /// (`[avx10-v1-aux-fp16-fp8-evex-vnni.PROPERTIES.1]`).
        fn prop_lane_independence(x: InputsSS, lane: u8) -> bool {
            let i = (lane % 16) as usize;
            let mut a = [0i8; 64];
            let mut b = [0i8; 64];
            let mut dst = [0i32; 16];
            dst[i] = x.dst[i];
            for k in 0..4 {
                a[4 * i + k] = x.a[4 * i + k];
                b[4 * i + k] = x.b[4 * i + k];
            }
            dpbssd_scalar(dst, a, b)[i] == dpbssd_scalar(x.dst, x.a, x.b)[i]
        }

        /// VNNI sign-combination consistency: when every operand byte is NON-NEGATIVE
        /// (`0..=127`), the SS, SU and UU sign matrices all sign/zero-extend to the SAME
        /// magnitude, so the three byte-VNNI forms must produce identical results on such
        /// inputs (`[avx10-v1-aux-fp16-fp8-evex-vnni.PROPERTIES.1]`).
        fn prop_sign_combination_consistency(x: InputsSS) -> bool {
            // Force the sign bit clear so signed and unsigned extension coincide.
            let ai: [i8; 64] = core::array::from_fn(|i| (x.a[i] as u8 & 0x7f) as i8);
            let bi: [i8; 64] = core::array::from_fn(|i| (x.b[i] as u8 & 0x7f) as i8);
            let au: [u8; 64] = core::array::from_fn(|i| ai[i] as u8);
            let bu: [u8; 64] = core::array::from_fn(|i| bi[i] as u8);
            let ss = dpbssd_scalar(x.dst, ai, bi);
            let su = dpbsud_scalar(x.dst, ai, bu);
            let uu = dpbuud_scalar(x.dst, au, bu);
            ss == su && su == uu
        }

        /// Saturating-bounds-vs-non-saturating relation (byte VNNI, families F): for every lane
        /// the signed-saturating SS result equals the non-saturating SS result clamped to the
        /// i32 range — saturation bounds the wrapped accumulation rather than recomputing it
        /// (`[avx10-v1-aux-fp16-fp8-evex-vnni.PROPERTIES.3]`,
        /// `[avx10-v1-aux-fp16-fp8-evex-vnni.BYTE_VNNI.1-3]`).
        fn prop_saturating_bounds_non_saturating_byte(x: InputsSS) -> bool {
            let a64: [i64; 64] = core::array::from_fn(|i| x.a[i] as i64);
            let b64: [i64; 64] = core::array::from_fn(|i| x.b[i] as i64);
            let sat = dpbssds_scalar(x.dst, x.a, x.b);
            (0..16).all(|i| {
                let mut total = x.dst[i] as i64;
                for k in 0..4 {
                    total += a64[4 * i + k] * b64[4 * i + k];
                }
                sat[i] == total.clamp(i32::MIN as i64, i32::MAX as i64) as i32
            })
        }

        /// Saturating-bounds-vs-non-saturating relation (word VNNI, families G): the
        /// signed-saturating SU result equals the wider-type accumulation clamped to the i32
        /// range (`[avx10-v1-aux-fp16-fp8-evex-vnni.PROPERTIES.3]`,
        /// `[avx10-v1-aux-fp16-fp8-evex-vnni.WORD_VNNI.1-3]`).
        fn prop_saturating_bounds_non_saturating_word(x: InputsWordSU) -> bool {
            let a64: [i64; 32] = core::array::from_fn(|i| x.a[i] as i64);
            let b64: [i64; 32] = core::array::from_fn(|i| x.b[i] as i64);
            let sat = dpwsuds_scalar(x.dst, x.a, x.b);
            (0..16).all(|i| {
                let mut total = x.dst[i] as i64;
                for k in 0..2 {
                    total += a64[2 * i + k] * b64[2 * i + k];
                }
                sat[i] == total.clamp(i32::MIN as i64, i32::MAX as i64) as i32
            })
        }

        /// Family-F/G native-vs-oracle differential. Under `feature="native"` on x86_64 with
        /// `AVX10_V1_AUX` detected, every real EVEX `VPDPB*`/`VPDPW*` path must agree with its
        /// scalar oracle bit-for-bit (`[avx10-v1-aux-fp16-fp8-evex-vnni.DIFFERENTIAL.1]`). The
        /// random `InputsSS` triple is reinterpreted across all six byte forms and all six word
        /// forms by raw bit-cast so every sign matrix and saturation mode is exercised. When the
        /// native feature or detection is absent the case is *discarded* (never
        /// `from_bool(false)`), so a fallback-only runner cannot produce a vacuous green.
        fn prop_native_matches_oracle(x: InputsSS) -> TestResult {
            #[cfg(all(target_arch = "x86_64", feature = "native"))]
            {
                if detect::has_avx10_v1_aux() {
                    return TestResult::from_bool(native_matches_oracle_all(x.dst, x.a, x.b));
                }
            }
            let _ = &x;
            TestResult::discard()
        }
    }

    /// Run the native-vs-oracle differential for every family-F and family-G primitive on one
    /// `(dst, a, b)` triple (the byte operands reinterpreted as words for family G, and
    /// signed/unsigned operands reinterpreted by raw bit-cast). Returns `true` iff every
    /// dispatcher equals its oracle. Used by both the property and hand-value differentials.
    #[cfg(all(target_arch = "x86_64", feature = "native"))]
    fn native_matches_oracle_all(dst: [i32; 16], a: [i8; 64], b: [i8; 64]) -> bool {
        let au: [u8; 64] = core::array::from_fn(|i| a[i] as u8);
        let bu: [u8; 64] = core::array::from_fn(|i| b[i] as u8);
        // Reinterpret the 64 bytes as 32 little-endian words for the word forms.
        let aw_i: [i16; 32] =
            core::array::from_fn(|i| i16::from_le_bytes([a[2 * i] as u8, a[2 * i + 1] as u8]));
        let bw_i: [i16; 32] =
            core::array::from_fn(|i| i16::from_le_bytes([b[2 * i] as u8, b[2 * i + 1] as u8]));
        let aw_u: [u16; 32] = core::array::from_fn(|i| aw_i[i] as u16);
        let bw_u: [u16; 32] = core::array::from_fn(|i| bw_i[i] as u16);

        // Family F (byte VNNI).
        dpbssd(dst, a, b) == dpbssd_scalar(dst, a, b)
            && dpbssds(dst, a, b) == dpbssds_scalar(dst, a, b)
            && dpbsud(dst, a, bu) == dpbsud_scalar(dst, a, bu)
            && dpbsuds(dst, a, bu) == dpbsuds_scalar(dst, a, bu)
            && dpbuud(dst, au, bu) == dpbuud_scalar(dst, au, bu)
            && dpbuuds(dst, au, bu) == dpbuuds_scalar(dst, au, bu)
            // Family G (word VNNI).
            && dpwsud(dst, aw_i, bw_u) == dpwsud_scalar(dst, aw_i, bw_u)
            && dpwsuds(dst, aw_i, bw_u) == dpwsuds_scalar(dst, aw_i, bw_u)
            && dpwusd(dst, aw_u, bw_i) == dpwusd_scalar(dst, aw_u, bw_i)
            && dpwusds(dst, aw_u, bw_i) == dpwusds_scalar(dst, aw_u, bw_i)
            && dpwuud(dst, aw_u, bw_u) == dpwuud_scalar(dst, aw_u, bw_u)
            && dpwuuds(dst, aw_u, bw_u) == dpwuuds_scalar(dst, aw_u, bw_u)
    }

    /// Hand-value family-F/G native-vs-oracle differential. Runs only under `feature="native"`
    /// with `AVX10_V1_AUX` detected; pins a vector that drives both wrapping and saturation
    /// (large operands on top of `i32::MAX`/`i32::MIN` accumulators).
    /// `[avx10-v1-aux-fp16-fp8-evex-vnni.DIFFERENTIAL.1-1]`
    #[cfg(all(target_arch = "x86_64", feature = "native"))]
    #[test]
    fn hand_value_native_matches_oracle() {
        if !detect::has_avx10_v1_aux() {
            return;
        }
        let mut dst = [0i32; 16];
        dst[0] = i32::MAX;
        dst[1] = i32::MIN;
        dst[2] = 1000;
        dst[3] = -1; // u32::MAX accumulator for the UU forms
        let a: [i8; 64] = core::array::from_fn(|i| (i as i8).wrapping_mul(7).wrapping_sub(64));
        let b: [i8; 64] = core::array::from_fn(|i| (i as i8).wrapping_mul(-5).wrapping_add(13));
        assert!(
            native_matches_oracle_all(dst, a, b),
            "a family-F/G hardware path disagrees with its oracle"
        );
    }
}
