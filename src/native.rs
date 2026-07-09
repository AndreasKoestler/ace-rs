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

    // Family B write (.byte): ZMM -> tile row / column.
    pub(crate) fn ace_tile_movrow_write(cfg: *const u8, data: *const u8, out: *mut u8);
    pub(crate) fn ace_tile_movcol_write(cfg: *const u8, data: *const u8, out: *mut u8);

    // Family D (.byte): block-scale registers.
    pub(crate) fn ace_tile_bsrinit(cfg: *const u8, data: *const u8, out: *mut u8);
    pub(crate) fn ace_tile_bsrmovf(cfg: *const u8, data: *const u8, out: *mut u8);
    pub(crate) fn ace_tile_bsrmovh(cfg: *const u8, data: *const u8, out: *mut u8);
    pub(crate) fn ace_tile_bsrmovl(cfg: *const u8, data: *const u8, out: *mut u8);

    // Family G (.byte): INT8 rank-4 outer products (C += A (x) B).
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

    // Family F (.byte): BF16 rank-2 outer product, no block scale.
    pub(crate) fn ace_tile_top2bf16ps(
        cfg: *const u8,
        c: *const u8,
        a: *const u8,
        b: *const u8,
        out: *mut u8,
    );

    // Family E (.byte): MX-FP8 rank-4 outer products with a per-block BSR scale.
    pub(crate) fn ace_tile_top4mxbf8ps(
        cfg: *const u8,
        c: *const u8,
        a: *const u8,
        b: *const u8,
        bsr: *const u8,
        out: *mut u8,
    );
    pub(crate) fn ace_tile_top4mxbhf8ps(
        cfg: *const u8,
        c: *const u8,
        a: *const u8,
        b: *const u8,
        bsr: *const u8,
        out: *mut u8,
    );
    pub(crate) fn ace_tile_top4mxhbf8ps(
        cfg: *const u8,
        c: *const u8,
        a: *const u8,
        b: *const u8,
        bsr: *const u8,
        out: *mut u8,
    );
    pub(crate) fn ace_tile_top4mxhf8ps(
        cfg: *const u8,
        c: *const u8,
        a: *const u8,
        b: *const u8,
        bsr: *const u8,
        out: *mut u8,
    );
    pub(crate) fn ace_tile_top4mxbssps(
        cfg: *const u8,
        c: *const u8,
        a: *const u8,
        b: *const u8,
        bsr: *const u8,
        out: *mut u8,
    );
}

/// A 64-byte row-major tile / ZMM marshalling buffer (palette-2 tiles are at most
/// `16 rows * 64 colsb` but every native shim moves data one 64-byte row / ZMM at a time).
pub(crate) const TILE_BYTES: usize = 64;

/// Serialize a palette-2 tile descriptor into the 64-byte `LDTILECFG` layout the AMX intrinsics
/// consume: byte 0 = palette id, byte 1 = start row (0), `colsb[t]` as little-endian `u16` at
/// offset `16 + 2*t`, `rows[t]` at offset `48 + t`; every other byte reserved (0). This is the
/// memory-marshalling counterpart the family-A/C shims load with `_tile_loadconfig`.
pub(crate) fn encode_tilecfg(palette: u8, rows: &[u8; 8], colsb: &[u16; 8]) -> [u8; 64] {
    let mut cfg = [0u8; 64];
    cfg[0] = palette;
    for t in 0..8 {
        let c = colsb[t].to_le_bytes();
        cfg[16 + 2 * t] = c[0];
        cfg[16 + 2 * t + 1] = c[1];
        cfg[48 + t] = rows[t];
    }
    cfg
}

/// `_hw` wrappers: marshal fixed-size Rust buffers into the C shims. Every wrapper is `unsafe`
/// and may be called only once the matching capability check has confirmed the running CPU
/// supports the form (`detect::has_amx_tile` / `has_amx_avx512` / `has_ace`) with the tile +
/// BSR XSAVE state enabled — otherwise the tile / EVEX instruction would fault (`#UD`). Callers
/// go through the differential properties, which gate on those probes.
///
/// # Safety
/// The CPU must support the relevant tile capability and OS tile-state enablement; the input
/// buffers must be the documented lengths.

/// Family A: STTILECFG round-trip of a 64-byte descriptor (INV-3).
pub(crate) unsafe fn tile_cfg_roundtrip_hw(cfg: &[u8; 64]) -> [u8; 64] {
    let mut out = [0u8; 64];
    ace_tile_cfg_roundtrip(cfg.as_ptr(), out.as_mut_ptr());
    out
}

/// Family A: TILEZERO of tile 0.
pub(crate) unsafe fn tile_zero_hw(cfg: &[u8; 64], data: &[u8; TILE_BYTES]) -> [u8; TILE_BYTES] {
    let mut out = [0u8; TILE_BYTES];
    ace_tile_zero(cfg.as_ptr(), data.as_ptr(), out.as_mut_ptr());
    out
}

/// Family B read: TILEMOVROW read form (tile row -> ZMM).
pub(crate) unsafe fn tile_movrow_read_hw(
    cfg: &[u8; 64],
    data: &[u8; TILE_BYTES],
    row: u32,
) -> [u8; TILE_BYTES] {
    let mut out = [0u8; TILE_BYTES];
    ace_tile_movrow_read(cfg.as_ptr(), data.as_ptr(), row, out.as_mut_ptr());
    out
}

/// Family C: TCVTROWD2PS (tile row INT32 -> FP32 ZMM).
pub(crate) unsafe fn tcvtrowd2ps_hw(
    cfg: &[u8; 64],
    data: &[u8; TILE_BYTES],
    row: u32,
) -> [f32; 16] {
    let mut out = [0f32; 16];
    ace_tile_tcvtrowd2ps(cfg.as_ptr(), data.as_ptr(), row, out.as_mut_ptr());
    out
}

macro_rules! tcvtrow_word_hw {
    ($name:ident, $shim:ident) => {
        /// Family C tile-row FP32 -> narrow (BF16 / FP16) convert (ZMM word lanes out).
        pub(crate) unsafe fn $name(cfg: &[u8; 64], data: &[u8; TILE_BYTES], row: u32) -> [u16; 32] {
            let mut out = [0u16; 32];
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
        /// Family B write / family D single-tile `.byte` shim (ZMM/data -> tile, stored back).
        pub(crate) unsafe fn $name(cfg: &[u8; 64], data: &[u8; TILE_BYTES]) -> [u8; TILE_BYTES] {
            let mut out = [0u8; TILE_BYTES];
            $shim(cfg.as_ptr(), data.as_ptr(), out.as_mut_ptr());
            out
        }
    };
}
move_write_hw!(tile_movrow_write_hw, ace_tile_movrow_write);
move_write_hw!(tile_movcol_write_hw, ace_tile_movcol_write);
move_write_hw!(bsrinit_hw, ace_tile_bsrinit);
move_write_hw!(bsrmovf_hw, ace_tile_bsrmovf);
move_write_hw!(bsrmovh_hw, ace_tile_bsrmovh);
move_write_hw!(bsrmovl_hw, ace_tile_bsrmovl);

macro_rules! top_hw {
    ($name:ident, $shim:ident) => {
        /// Family G / F rank-N outer-product `.byte` shim: `C += A (x) B`, dst tile stored back.
        pub(crate) unsafe fn $name(
            cfg: &[u8; 64],
            c: &[u8; TILE_BYTES],
            a: &[u8; TILE_BYTES],
            b: &[u8; TILE_BYTES],
        ) -> [u8; TILE_BYTES] {
            let mut out = [0u8; TILE_BYTES];
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
        /// Family E MX-FP8 rank-4 outer-product `.byte` shim with a per-block BSR scale.
        pub(crate) unsafe fn $name(
            cfg: &[u8; 64],
            c: &[u8; TILE_BYTES],
            a: &[u8; TILE_BYTES],
            b: &[u8; TILE_BYTES],
            bsr: &[u8; TILE_BYTES],
        ) -> [u8; TILE_BYTES] {
            let mut out = [0u8; TILE_BYTES];
            $shim(
                cfg.as_ptr(),
                c.as_ptr(),
                a.as_ptr(),
                b.as_ptr(),
                bsr.as_ptr(),
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
