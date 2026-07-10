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

// ============================ ACE group-4 tile instructions ============================
//
// `extern "C"` declarations for `src/native/ace_tile.c` (design decision D7, OQ-6). The tile
// operands are marshalled through memory: each shim takes a 64-byte palette-2 `LDTILECFG`
// descriptor (`cfg`) plus row-major tile / ZMM buffers, and writes the destination tile / ZMM
// vector back through the `out` pointer. Families A / B-read / C resolve to real AMX-TILE +
// AMX-AVX512 intrinsics and execute under Intel SDE; the `ACE`-only families (B write, D, E, F,
// G) resolve to `.byte` raw-encoding shims, present but not executed until SDE gains ACE
// emulation (see the C TU header and `tests/encoding.rs`).
//
// OQ-6 (native path per family, D7) — default assumption realised here: C-INTRINSIC shims for
// families A / B-read / C (assembler/intrinsic-reachable in current GCC/Clang AMX-TILE +
// AMX-AVX512), `.byte` raw-encoding shims for the `ACE`-only rest (family B write, D, E, F, G).
// Layer-3 EXECUTION of the `.byte` families is gated behind a confirmed SDE ACE probe
// (`sde64 -help | grep -i ace`, or a one-instruction `#UD` probe): they are built-not-executed
// until that probe passes, at which point the per-family `prop_native_matches_oracle`
// differentials stop discarding and light up automatically. Families A/C already execute under
// SDE today.
extern "C" {
    // Family A (intrinsic): tile config lifecycle.
    pub(crate) fn ace_tile_cfg_roundtrip(cfg: *const u8, out: *mut u8);
    pub(crate) fn ace_tile_zero(cfg: *const u8, data: *const u8, out: *mut u8);

    // Family B read (intrinsic): TILEMOVROW read form (tile -> ZMM).
    pub(crate) fn ace_tile_movrow_read(cfg: *const u8, data: *const u8, row: u32, out: *mut u8);

    // Family C (intrinsic): tile-row converts (tile row -> ZMM).
    pub(crate) fn ace_tile_tcvtrowd2ps(cfg: *const u8, data: *const u8, row: u32, out: *mut f32);
    pub(crate) fn ace_tile_tcvtrowps2bf16h(
        cfg: *const u8,
        data: *const u8,
        row: u32,
        out: *mut u16,
    );
    pub(crate) fn ace_tile_tcvtrowps2bf16l(
        cfg: *const u8,
        data: *const u8,
        row: u32,
        out: *mut u16,
    );
    pub(crate) fn ace_tile_tcvtrowps2phh(cfg: *const u8, data: *const u8, row: u32, out: *mut u16);
    pub(crate) fn ace_tile_tcvtrowps2phl(cfg: *const u8, data: *const u8, row: u32, out: *mut u16);

    // Family B write (.byte, §6.3.3): ZMM -> tile row / byte-column (row/col fixed 0).
    pub(crate) fn ace_tile_movrow_write(cfg: *const u8, data: *const u8, out: *mut u8);
    pub(crate) fn ace_tile_movcol_write(cfg: *const u8, data: *const u8, out: *mut u8);

    // Family D (.byte, §6.3.5): Block Scale register ops over the implicit bsr0. Each shim
    // reads the halves back through the BSRMOVH/BSRMOVL read forms so the register state is
    // observable.
    pub(crate) fn ace_bsrinit_read(cfg: *const u8, out_a: *mut u8, out_b: *mut u8);
    pub(crate) fn ace_bsrmovf_read(
        a: *const u8,
        b: *const u8,
        cfg: *const u8,
        out_a: *mut u8,
        out_b: *mut u8,
    );
    pub(crate) fn ace_bsrmovh_roundtrip(a: *const u8, cfg: *const u8, out_a: *mut u8);
    pub(crate) fn ace_bsrmovl_roundtrip(b: *const u8, cfg: *const u8, out_b: *mut u8);

    // Families G/F (.byte, §6.3.8/§6.3.9): outer products, C(tmm1) += A(zmm0) (x) B(zmm2).
    pub(crate) fn ace_tile_top4bssd(
        cfg: *const u8,
        c: *const u8,
        a: *const u8,
        b: *const u8,
        out: *mut u8,
    );
    pub(crate) fn ace_tile_top4bsud(
        cfg: *const u8,
        c: *const u8,
        a: *const u8,
        b: *const u8,
        out: *mut u8,
    );
    pub(crate) fn ace_tile_top4busd(
        cfg: *const u8,
        c: *const u8,
        a: *const u8,
        b: *const u8,
        out: *mut u8,
    );
    pub(crate) fn ace_tile_top4buud(
        cfg: *const u8,
        c: *const u8,
        a: *const u8,
        b: *const u8,
        out: *mut u8,
    );
    pub(crate) fn ace_tile_top2bf16ps(
        cfg: *const u8,
        c: *const u8,
        a: *const u8,
        b: *const u8,
        out: *mut u8,
    );

    // Family E (.byte, §6.3.6/§6.3.7): MX rank-4 outer products. The BSR is implicit; the
    // shim seeds it with BSRMOVF from the caller's A/B scale buffers; imm8 fixed 0.
    pub(crate) fn ace_tile_top4mxbf8ps(
        cfg: *const u8,
        c: *const u8,
        a: *const u8,
        b: *const u8,
        a_scales: *const u8,
        b_scales: *const u8,
        out: *mut u8,
    );
    pub(crate) fn ace_tile_top4mxbhf8ps(
        cfg: *const u8,
        c: *const u8,
        a: *const u8,
        b: *const u8,
        a_scales: *const u8,
        b_scales: *const u8,
        out: *mut u8,
    );
    pub(crate) fn ace_tile_top4mxhbf8ps(
        cfg: *const u8,
        c: *const u8,
        a: *const u8,
        b: *const u8,
        a_scales: *const u8,
        b_scales: *const u8,
        out: *mut u8,
    );
    pub(crate) fn ace_tile_top4mxhf8ps(
        cfg: *const u8,
        c: *const u8,
        a: *const u8,
        b: *const u8,
        a_scales: *const u8,
        b_scales: *const u8,
        out: *mut u8,
    );
    pub(crate) fn ace_tile_top4mxbssps(
        cfg: *const u8,
        c: *const u8,
        a: *const u8,
        b: *const u8,
        a_scales: *const u8,
        b_scales: *const u8,
        out: *mut u8,
    );
}

/// A 64-byte marshalling buffer: one ZMM / one tile row / one BSR half.
pub(crate) const ROW_BYTES: usize = 64;

/// Serialize a [`crate::tile::TileConfig`]-shaped descriptor into the 64-byte `LDTILECFG`
/// layout: byte 0 = palette id, bytes 1-63 = 0 (the palette-2 descriptor has no per-tile
/// fields — spec section 11.2.3).
pub(crate) fn encode_tilecfg(palette: u8) -> [u8; 64] {
    let mut cfg = [0u8; 64];
    cfg[0] = palette;
    cfg
}

// `_hw` wrappers: marshal fixed-size Rust buffers into the C shims. Every wrapper is `unsafe`
// and may be called only once the matching capability check has confirmed the running CPU
// supports the form (`detect::has_amx_tile` / `has_amx_avx512` / `has_ace`) with the required
// XSAVE state enabled — otherwise the tile / EVEX instruction would fault (`#UD`). Callers go
// through the differential properties, which gate on those probes.

/// Family A: STTILECFG round-trip of a 64-byte descriptor.
///
/// # Safety
/// The CPU must support AMX-TILE with the tile XSAVE state enabled
/// ([`crate::detect::has_amx_tile`]), otherwise the tile instructions fault (`#UD`).
pub(crate) unsafe fn tile_cfg_roundtrip_hw(cfg: &[u8; 64]) -> [u8; 64] {
    let mut out = [0u8; 64];
    // SAFETY (FFI): fixed-size borrows match the shim's documented 64-byte buffers; the
    // caller upholds the capability precondition above.
    ace_tile_cfg_roundtrip(cfg.as_ptr(), out.as_mut_ptr());
    out
}

/// Family A: TILEZERO of tile 0 (one 64-byte row marshalled).
///
/// # Safety
/// The CPU must support AMX-TILE with the tile XSAVE state enabled
/// ([`crate::detect::has_amx_tile`]), otherwise the tile instructions fault (`#UD`).
pub(crate) unsafe fn tile_zero_hw(cfg: &[u8; 64], data: &[u8; ROW_BYTES]) -> [u8; ROW_BYTES] {
    let mut out = [0u8; ROW_BYTES];
    // SAFETY (FFI): fixed-size borrows match the shim's documented 64-byte buffers.
    ace_tile_zero(cfg.as_ptr(), data.as_ptr(), out.as_mut_ptr());
    out
}

/// Family B read: TILEMOVROW read form (tile row -> ZMM).
///
/// # Safety
/// The CPU must support AMX-AVX512 (or full ACE) with the tile XSAVE state enabled
/// ([`crate::detect::has_amx_avx512`]), otherwise the instruction faults (`#UD`).
pub(crate) unsafe fn tile_movrow_read_hw(
    cfg: &[u8; 64],
    data: &[u8; ROW_BYTES],
    row: u32,
) -> [u8; ROW_BYTES] {
    let mut out = [0u8; ROW_BYTES];
    // SAFETY (FFI): fixed-size borrows match the shim's documented 64-byte buffers.
    ace_tile_movrow_read(cfg.as_ptr(), data.as_ptr(), row, out.as_mut_ptr());
    out
}

/// Family C: TCVTROWD2PS (tile row INT32 -> FP32 ZMM).
///
/// # Safety
/// The CPU must support AMX-AVX512 (or full ACE) with the tile XSAVE state enabled
/// ([`crate::detect::has_amx_avx512`]), otherwise the instruction faults (`#UD`).
pub(crate) unsafe fn tcvtrowd2ps_hw(cfg: &[u8; 64], data: &[u8; ROW_BYTES], row: u32) -> [f32; 16] {
    let mut out = [0f32; 16];
    // SAFETY (FFI): the 64-byte inputs and 16-lane f32 output match the shim's contract.
    ace_tile_tcvtrowd2ps(cfg.as_ptr(), data.as_ptr(), row, out.as_mut_ptr());
    out
}

macro_rules! tcvtrow_word_hw {
    ($name:ident, $shim:ident) => {
        /// Family C tile-row FP32 -> narrow (BF16 / FP16) convert (ZMM word lanes out).
        ///
        /// # Safety
        /// The CPU must support AMX-AVX512 (or full ACE) with the tile XSAVE state enabled
        /// ([`crate::detect::has_amx_avx512`]), otherwise the instruction faults (`#UD`).
        pub(crate) unsafe fn $name(cfg: &[u8; 64], data: &[u8; ROW_BYTES], row: u32) -> [u16; 32] {
            let mut out = [0u16; 32];
            // SAFETY (FFI): fixed-size borrows match the shim's documented buffer lengths.
            $shim(cfg.as_ptr(), data.as_ptr(), row, out.as_mut_ptr());
            out
        }
    };
}
tcvtrow_word_hw!(tcvtrowps2bf16h_hw, ace_tile_tcvtrowps2bf16h);
tcvtrow_word_hw!(tcvtrowps2bf16l_hw, ace_tile_tcvtrowps2bf16l);
tcvtrow_word_hw!(tcvtrowps2phh_hw, ace_tile_tcvtrowps2phh);
tcvtrow_word_hw!(tcvtrowps2phl_hw, ace_tile_tcvtrowps2phl);

macro_rules! move_write_hw {
    ($name:ident, $shim:ident) => {
        /// Family B write `.byte` shim (ZMM -> tile row/column 0, tile 1 stored back).
        ///
        /// # Safety
        /// The CPU must support full ACE with the tile + SCALEDATA XSAVE state enabled
        /// ([`crate::detect::has_ace`]), otherwise the encoded instruction faults (`#UD`).
        pub(crate) unsafe fn $name(cfg: &[u8; 64], data: &[u8; ROW_BYTES]) -> [u8; ROW_BYTES] {
            let mut out = [0u8; ROW_BYTES];
            // SAFETY (FFI): fixed-size borrows match the shim's documented 64-byte buffers.
            $shim(cfg.as_ptr(), data.as_ptr(), out.as_mut_ptr());
            out
        }
    };
}
move_write_hw!(tile_movrow_write_hw, ace_tile_movrow_write);
move_write_hw!(tile_movcol_write_hw, ace_tile_movcol_write);

/// Family D: BSRINIT then read both BSR halves back (A half, B half).
///
/// # Safety
/// The CPU must support full ACE with the tile + SCALEDATA XSAVE state enabled
/// ([`crate::detect::has_ace`]), otherwise the encoded instruction faults (`#UD`).
pub(crate) unsafe fn bsrinit_hw(cfg: &[u8; 64]) -> ([u8; ROW_BYTES], [u8; ROW_BYTES]) {
    let mut a = [0u8; ROW_BYTES];
    let mut b = [0u8; ROW_BYTES];
    // SAFETY (FFI): fixed-size borrows match the shim's documented 64-byte buffers.
    ace_bsrinit_read(cfg.as_ptr(), a.as_mut_ptr(), b.as_mut_ptr());
    (a, b)
}

/// Family D: BSRMOVF (a -> A scales, b -> B scales), both halves read back.
///
/// # Safety
/// The CPU must support full ACE with the tile + SCALEDATA XSAVE state enabled
/// ([`crate::detect::has_ace`]), otherwise the encoded instruction faults (`#UD`).
pub(crate) unsafe fn bsrmovf_hw(
    cfg: &[u8; 64],
    a: &[u8; ROW_BYTES],
    b: &[u8; ROW_BYTES],
) -> ([u8; ROW_BYTES], [u8; ROW_BYTES]) {
    let mut oa = [0u8; ROW_BYTES];
    let mut ob = [0u8; ROW_BYTES];
    // SAFETY (FFI): fixed-size borrows match the shim's documented 64-byte buffers.
    ace_bsrmovf_read(a.as_ptr(), b.as_ptr(), cfg.as_ptr(), oa.as_mut_ptr(), ob.as_mut_ptr());
    (oa, ob)
}

/// Family D: BSRMOVH write-then-read round-trip of the A half.
///
/// # Safety
/// The CPU must support full ACE with the tile + SCALEDATA XSAVE state enabled
/// ([`crate::detect::has_ace`]), otherwise the encoded instruction faults (`#UD`).
pub(crate) unsafe fn bsrmovh_hw(cfg: &[u8; 64], a: &[u8; ROW_BYTES]) -> [u8; ROW_BYTES] {
    let mut out = [0u8; ROW_BYTES];
    // SAFETY (FFI): fixed-size borrows match the shim's documented 64-byte buffers.
    ace_bsrmovh_roundtrip(a.as_ptr(), cfg.as_ptr(), out.as_mut_ptr());
    out
}

/// Family D: BSRMOVL write-then-read round-trip of the B half.
///
/// # Safety
/// The CPU must support full ACE with the tile + SCALEDATA XSAVE state enabled
/// ([`crate::detect::has_ace`]), otherwise the encoded instruction faults (`#UD`).
pub(crate) unsafe fn bsrmovl_hw(cfg: &[u8; 64], b: &[u8; ROW_BYTES]) -> [u8; ROW_BYTES] {
    let mut out = [0u8; ROW_BYTES];
    // SAFETY (FFI): fixed-size borrows match the shim's documented 64-byte buffers.
    ace_bsrmovl_roundtrip(b.as_ptr(), cfg.as_ptr(), out.as_mut_ptr());
    out
}

macro_rules! top_hw {
    ($name:ident, $shim:ident) => {
        /// Families G/F outer-product `.byte` shim: `C += A (x) B`, dst tile stored back
        /// (one 64-byte row marshalled).
        ///
        /// # Safety
        /// The CPU must support full ACE with the tile + SCALEDATA XSAVE state enabled
        /// ([`crate::detect::has_ace`]), otherwise the encoded instruction faults (`#UD`).
        pub(crate) unsafe fn $name(
            cfg: &[u8; 64],
            c: &[u8; ROW_BYTES],
            a: &[u8; ROW_BYTES],
            b: &[u8; ROW_BYTES],
        ) -> [u8; ROW_BYTES] {
            let mut out = [0u8; ROW_BYTES];
            // SAFETY (FFI): fixed-size borrows match the shim's documented 64-byte buffers.
            $shim(
                cfg.as_ptr(),
                c.as_ptr(),
                a.as_ptr(),
                b.as_ptr(),
                out.as_mut_ptr(),
            );
            out
        }
    };
}
top_hw!(top4bssd_hw, ace_tile_top4bssd);
top_hw!(top4bsud_hw, ace_tile_top4bsud);
top_hw!(top4busd_hw, ace_tile_top4busd);
top_hw!(top4buud_hw, ace_tile_top4buud);
top_hw!(top2bf16ps_hw, ace_tile_top2bf16ps);

macro_rules! mx_top_hw {
    ($name:ident, $shim:ident) => {
        /// Family E MX rank-4 outer-product `.byte` shim: the implicit BSR is seeded with
        /// BSRMOVF from the A/B scale buffers, imm8 fixed 0 (scale groups 0/0).
        ///
        /// # Safety
        /// The CPU must support full ACE with the tile + SCALEDATA XSAVE state enabled
        /// ([`crate::detect::has_ace`]), otherwise the encoded instruction faults (`#UD`).
        pub(crate) unsafe fn $name(
            cfg: &[u8; 64],
            c: &[u8; ROW_BYTES],
            a: &[u8; ROW_BYTES],
            b: &[u8; ROW_BYTES],
            a_scales: &[u8; ROW_BYTES],
            b_scales: &[u8; ROW_BYTES],
        ) -> [u8; ROW_BYTES] {
            let mut out = [0u8; ROW_BYTES];
            // SAFETY (FFI): fixed-size borrows match the shim's documented 64-byte buffers.
            $shim(
                cfg.as_ptr(),
                c.as_ptr(),
                a.as_ptr(),
                b.as_ptr(),
                a_scales.as_ptr(),
                b_scales.as_ptr(),
                out.as_mut_ptr(),
            );
            out
        }
    };
}
mx_top_hw!(top4mxbf8ps_hw, ace_tile_top4mxbf8ps);
mx_top_hw!(top4mxbhf8ps_hw, ace_tile_top4mxbhf8ps);
mx_top_hw!(top4mxhbf8ps_hw, ace_tile_top4mxhbf8ps);
mx_top_hw!(top4mxhf8ps_hw, ace_tile_top4mxhf8ps);
mx_top_hw!(top4mxbssps_hw, ace_tile_top4mxbssps);

// ================================ Differential layer (group 4) ================================
//
// Native-vs-oracle differentials for the group-4 shims: each property gates on the matching
// capability probe and DISCARDS (never passes vacuously) when the host lacks it. Families
// A / B-read / C execute under Intel SDE today; the `.byte` families light up automatically
// once an ACE-capable SDE/hardware host runs them.
#[cfg(test)]
mod differential {
    use super::*;
    use crate::detect;
    use crate::tile::{TileConfig, _tile_loadconfig};
    use quickcheck::{quickcheck, TestResult};

    quickcheck! {
        /// Family A: the native STTILECFG round-trip of the palette-2 descriptor equals the
        /// oracle's stored descriptor (byte 0 = 2, bytes 1-63 = 0, spec section 11.3.4), and
        /// native TILEZERO equals the oracle over one marshalled row.
        fn prop_family_a_matches_oracle(seed: Vec<u8>) -> TestResult {
            if !detect::has_amx_tile() {
                return TestResult::discard();
            }
            let cfg = encode_tilecfg(2);
            // SAFETY: has_amx_tile() confirmed AMX-TILE + XCR0[18:17] + XFD permission.
            let got = unsafe { tile_cfg_roundtrip_hw(&cfg) };
            let scope = _tile_loadconfig(&TileConfig::ace()).unwrap();
            let want = crate::tile::_tile_storeconfig(&scope).to_bytes();
            if got != want {
                return TestResult::from_bool(false);
            }

            let mut row = [0u8; ROW_BYTES];
            for (i, b) in seed.iter().take(ROW_BYTES).enumerate() {
                row[i] = *b;
            }
            // SAFETY: capability confirmed above.
            let zeroed = unsafe { tile_zero_hw(&cfg, &row) };
            TestResult::from_bool(zeroed == [0u8; ROW_BYTES])
        }

        /// Families B-read / C: native TILEMOVROW / TCVTROW* over a one-row tile equal the
        /// oracle for the same row data.
        fn prop_row_converts_match_oracle(seed: Vec<u8>, row_sel: u8) -> TestResult {
            if !detect::has_amx_avx512() {
                return TestResult::discard();
            }
            let cfg = encode_tilecfg(2);
            let mut row = [0u8; ROW_BYTES];
            for (i, b) in seed.iter().take(ROW_BYTES).enumerate() {
                row[i] = *b;
            }
            // Oracle scope with the row in tile 0 row 0.
            let mut scope = _tile_loadconfig(&TileConfig::ace()).unwrap();
            let id = scope.tile(0).unwrap();
            crate::tile_move::_tile_setrow_scalar(&mut scope, id, 0, row);
            let row_sel = (row_sel & 0xF) as u32;
            // The shim loads only one 64-byte row into the tile, so compare row 0 reads.
            if row_sel != 0 {
                return TestResult::discard();
            }
            // SAFETY: has_amx_avx512() confirmed the family-C capability set.
            unsafe {
                let mv = tile_movrow_read_hw(&cfg, &row, 0);
                let d2 = tcvtrowd2ps_hw(&cfg, &row, 0);
                let bh = tcvtrowps2bf16h_hw(&cfg, &row, 0);
                let bl = tcvtrowps2bf16l_hw(&cfg, &row, 0);
                let ph = tcvtrowps2phh_hw(&cfg, &row, 0);
                let pl = tcvtrowps2phl_hw(&cfg, &row, 0);
                let ok = mv == crate::tile_move::_tile_movrow_scalar(&scope, id, 0)
                    && d2.iter().map(|f| f.to_bits()).collect::<Vec<_>>()
                        == crate::tcvtrow::_tile_cvtrowd2ps_scalar(&scope, id, 0)
                            .iter()
                            .map(|f| f.to_bits())
                            .collect::<Vec<_>>()
                    && bh == crate::tcvtrow::_tile_cvtrowps2bf16h_scalar(&scope, id, 0)
                    && bl == crate::tcvtrow::_tile_cvtrowps2bf16l_scalar(&scope, id, 0)
                    && ph == crate::tcvtrow::_tile_cvtrowps2phh_scalar(&scope, id, 0)
                    && pl == crate::tcvtrow::_tile_cvtrowps2phl_scalar(&scope, id, 0);
                TestResult::from_bool(ok)
            }
        }

        /// ACE-only `.byte` families (B-write, D, E, F, G): native equals oracle. Discards
        /// until an ACE-capable host runs the suite (has_ace() is false everywhere today).
        fn prop_ace_byte_families_match_oracle(seed: Vec<u8>) -> TestResult {
            if !detect::has_ace() {
                return TestResult::discard();
            }
            let cfg = encode_tilecfg(2);
            let mut a = [0u8; ROW_BYTES];
            let mut b = [0u8; ROW_BYTES];
            for i in 0..ROW_BYTES {
                a[i] = seed.get(i).copied().unwrap_or(0x11);
                b[i] = seed.get(ROW_BYTES + i).copied().unwrap_or(0x22);
            }
            let c = [0u8; ROW_BYTES];

            // BSR: BSRINIT reads back 0x7F halves; BSRMOVF/H/L round-trip the written halves.
            // SAFETY: has_ace() confirmed the full §3.2 capability set.
            unsafe {
                let (ia, ib) = bsrinit_hw(&cfg);
                if ia != [0x7F; 64] || ib != [0x7F; 64] {
                    return TestResult::from_bool(false);
                }
                let (oa, ob) = bsrmovf_hw(&cfg, &a, &b);
                if oa != a || ob != b {
                    return TestResult::from_bool(false);
                }
                if bsrmovh_hw(&cfg, &a) != a || bsrmovl_hw(&cfg, &b) != b {
                    return TestResult::from_bool(false);
                }

                // TILEMOVROW/TILEMOVCOL write (row/col 0): compare tile row 0 against oracle.
                let mut scope = _tile_loadconfig(&TileConfig::ace()).unwrap();
                let id = scope.tile(0).unwrap();
                crate::tile_move::_tile_setrow_scalar(&mut scope, id, 0, a);
                let want_row0 = crate::tile_move::_tile_movrow_scalar(&scope, id, 0);
                if tile_movrow_write_hw(&cfg, &a) != want_row0 {
                    return TestResult::from_bool(false);
                }
                let mut scope = _tile_loadconfig(&TileConfig::ace()).unwrap();
                let id = scope.tile(0).unwrap();
                crate::tile_move::_tile_setcol_scalar(&mut scope, id, 0, a);
                let want_row0 = crate::tile_move::_tile_movrow_scalar(&scope, id, 0);
                if tile_movcol_write_hw(&cfg, &a) != want_row0 {
                    return TestResult::from_bool(false);
                }

                // TOP families over a single marshalled row (row 0 of C).
                let mut scope = _tile_loadconfig(&TileConfig::ace()).unwrap();
                let id = scope.tile(0).unwrap();
                crate::top::_tile_top4bssd_scalar(&mut scope, id, a, b);
                let want = crate::tile_move::_tile_movrow_scalar(&scope, id, 0);
                if top4bssd_hw(&cfg, &c, &a, &b) != want {
                    return TestResult::from_bool(false);
                }
                let scales = [0x7Fu8; 64];
                let mut scope = _tile_loadconfig(&TileConfig::ace()).unwrap();
                let id = scope.tile(0).unwrap();
                crate::top::_tile_top4mxbf8ps_scalar(&mut scope, id, a, b, 0);
                let want = crate::tile_move::_tile_movrow_scalar(&scope, id, 0);
                if top4mxbf8ps_hw(&cfg, &c, &a, &b, &scales, &scales) != want {
                    return TestResult::from_bool(false);
                }
            }
            TestResult::passed()
        }
    }
}
