//! `extern "C"` declarations for the opt-in native AVX10_V1_AUX backend (design decision
//! D7).
//!
//! These resolve to the shims in `src/native/avx10_v1_aux.c`, compiled with `-mavx10.2` by
//! `build.rs` only when the `native` feature is enabled on an x86_64 target. The whole
//! module is gated on `#[cfg(all(target_arch = "x86_64", feature = "native"))]` (see
//! `lib.rs`), so the default build never references it.
//!
//! # No AVX10_V2_AUX (group-3) shims — OQ-5
//!
//! Every group-3 OCP-convert intrinsic is ABSENT from the current GCC/Clang `-mavx10.2`
//! headers (verified by compile probes against GCC 16.1.1; each convert module's docs record
//! its probe): `_mm512_cvtps_bf8`/`_mm512_cvts_ps_bf8`/`_mm512_cvtps_hf8`/
//! `_mm512_cvtroundps_hf8` (family A), `_mm512_cvtbiasps_bf8` and siblings (family B — only
//! the FP16-source `_mm512_cvtbiasph_*` forms exist), `_mm512_cvtbf8_ps`/`_mm512_cvthf8_ps`
//! (family C — only the FP8->FP16 siblings exist), `_mm512_cvtbf8_bf4s`/`_mm512_cvthf8_bf4s`
//! (family D), `_mm512_cvtbf4_hf8` (family E), `_mm512_cvtf8_bf6s`/`_mm512_cvtf8_hf6s`
//! (family F), the `_mm512_cvtf6_hf8` family (family G), `_mm512_cvtssepi32_epi8` (family H
//! — only the ordinary asymmetric `_mm512_cvtsepi32_epi8` exists), and `_mm512_unpackb`
//! (family I, which would additionally need a compile-time-constant `imm8` dispatch).
//!
//! Per OQ-5 every group-3 family therefore ships **oracle-only**: no C TU, no `extern "C"`
//! declaration, no `_hw` path — the always-correct scalar oracle is the sole path, and each
//! family's `prop_native_matches_oracle` differential discards (never passes vacuously)
//! until a toolchain supplies the intrinsic. When one lands, add
//! `src/native/avx10_v2_aux.c`, wire it in `build.rs`, and declare its shims here.
//! Re-probe on every toolchain bump.
//!
//! Each shim takes plain pointers; the per-family `_hw` wrappers in the convert / VNNI
//! modules marshal the fixed-size lane arrays into and out of these calls. Every `_hw`
//! wrapper is `unsafe` and may only be called once the matching capability check
//! (`detect::has_avx10_v1_aux()` / `detect::has_avx10_v2_aux()`) has confirmed the running
//! CPU supports the EVEX forms — otherwise the EVEX-encoded instruction would fault (#UD).

extern "C" {
    // Family A: single-source FP16 -> FP8 (32 u16 in -> 32 u8 out).
    pub(crate) fn ace_native_cvtph_bf8(a: *const u16, out: *mut u8);
    pub(crate) fn ace_native_cvtphs_bf8(a: *const u16, out: *mut u8);
    pub(crate) fn ace_native_cvtph_hf8(a: *const u16, out: *mut u8);
    pub(crate) fn ace_native_cvtphs_hf8(a: *const u16, out: *mut u8);

    // Family B: two-source FP16 -> FP8 (src1, src2 of 32 u16 -> 64 u8; low=src2, high=src1).
    pub(crate) fn ace_native_cvt2ph_bf8(src1: *const u16, src2: *const u16, out: *mut u8);
    pub(crate) fn ace_native_cvt2phs_bf8(src1: *const u16, src2: *const u16, out: *mut u8);
    pub(crate) fn ace_native_cvt2ph_hf8(src1: *const u16, src2: *const u16, out: *mut u8);
    pub(crate) fn ace_native_cvt2phs_hf8(src1: *const u16, src2: *const u16, out: *mut u8);

    // Family C: biased FP16 -> FP8 (a, bias of 32 u16 -> 32 u8; bias = bias.byte[2*i]).
    pub(crate) fn ace_native_cvtbiasph_bf8(a: *const u16, bias: *const u16, out: *mut u8);
    pub(crate) fn ace_native_cvtbiasphs_bf8(a: *const u16, bias: *const u16, out: *mut u8);
    pub(crate) fn ace_native_cvtbiasph_hf8(a: *const u16, bias: *const u16, out: *mut u8);
    pub(crate) fn ace_native_cvtbiasphs_hf8(a: *const u16, bias: *const u16, out: *mut u8);

    // Family D: HF8 (E4M3) -> FP16 (32 u8 in -> 32 u16 out).
    pub(crate) fn ace_native_cvthf8_ph(a: *const u8, out: *mut u16);

    // Family E: FP32 pair -> FP16 (src1, src2 of 16 f32 -> 32 u16; low=src2, high=src1).
    pub(crate) fn ace_native_cvt2ps_phx(src1: *const f32, src2: *const f32, out: *mut u16);

    // Family F: byte VNNI (dst of 16 i32 + two 64-byte operands -> 16 i32 out).
    pub(crate) fn ace_native_dpbssd(dst: *const i32, a: *const i8, b: *const i8, out: *mut i32);
    pub(crate) fn ace_native_dpbssds(dst: *const i32, a: *const i8, b: *const i8, out: *mut i32);
    pub(crate) fn ace_native_dpbsud(dst: *const i32, a: *const i8, b: *const u8, out: *mut i32);
    pub(crate) fn ace_native_dpbsuds(dst: *const i32, a: *const i8, b: *const u8, out: *mut i32);
    pub(crate) fn ace_native_dpbuud(dst: *const i32, a: *const u8, b: *const u8, out: *mut i32);
    pub(crate) fn ace_native_dpbuuds(dst: *const i32, a: *const u8, b: *const u8, out: *mut i32);

    // Family G: word VNNI (dst of 16 i32 + two 32-word operands -> 16 i32 out).
    pub(crate) fn ace_native_dpwsud(dst: *const i32, a: *const i16, b: *const u16, out: *mut i32);
    pub(crate) fn ace_native_dpwsuds(dst: *const i32, a: *const i16, b: *const u16, out: *mut i32);
    pub(crate) fn ace_native_dpwusd(dst: *const i32, a: *const u16, b: *const i16, out: *mut i32);
    pub(crate) fn ace_native_dpwusds(dst: *const i32, a: *const u16, b: *const i16, out: *mut i32);
    pub(crate) fn ace_native_dpwuud(dst: *const i32, a: *const u16, b: *const u16, out: *mut i32);
    pub(crate) fn ace_native_dpwuuds(dst: *const i32, a: *const u16, b: *const u16, out: *mut i32);
}
