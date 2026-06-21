//! `extern "C"` declarations for the opt-in native AVX10_V1_AUX backend (design decision D7).
//!
//! These resolve to the shims in `src/native/avx10_v1_aux.c`, compiled with `-mavx10.2` by
//! `build.rs` only when the `native` feature is enabled on an x86_64 target. The whole module
//! is gated on `#[cfg(all(target_arch = "x86_64", feature = "native"))]` (see `lib.rs`), so
//! the default build never references it.
//!
//! Each shim takes plain pointers; the per-family `_hw` wrappers in the convert / VNNI modules
//! marshal the fixed-size lane arrays into and out of these calls. Every `_hw` wrapper is
//! `unsafe` and may only be called once `detect::has_avx10_v1_aux()` has confirmed the running
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
