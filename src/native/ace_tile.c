/*
 * Native ACE group-4 tile-instruction shims (design decision D7, OQ-6).
 *
 * Two kinds of shim live here:
 *
 *   1. INTRINSIC shims (families A, B-read, C) — the tile config lifecycle, the TILEMOVROW
 *      read form, and the five tile-row converts are assembler/intrinsic-reachable in
 *      current GCC/Clang (AMX-TILE + AMX-AVX512): `_tile_loadconfig`/`_tile_storeconfig`/
 *      `_tile_zero`/`_tile_release`, `_tile_loadd`/`_tile_stored`, `_tile_movrow`,
 *      `_tile_cvtrowd2ps`/`_tile_cvtrowps2bf16{h,l}`/`_tile_cvtrowps2ph{h,l}`. Compiled with
 *      `-mamx-tile -mamx-avx512 -mavx10.2` (see build.rs). These execute under Intel SDE.
 *
 *   2. RAW-ENCODING shims (family B write, D, E, F, G) — the `ACE`-only outer products,
 *      Block Scale register ops, and ZMM->tile write moves. No assembler knows the ACE
 *      mnemonics yet (binutils 2.46 / SDE 10.8 predate ACE v1.15), so each is emitted as a
 *      localized `.byte` sequence encoding the EXACT ACE v1 rev-1.15 section-6.3 form
 *      (map / pp / W / opcode / operand fields transcribed from the PDF tables on pages
 *      26-27). ZMM operands are marshalled into the fixed registers the encodings name
 *      (vmovdqu64), tile operands through `_tile_loadd`/`_tile_stored`, and the BSR is the
 *      single implicit `bsr0`. They are BUILT and asserted byte-for-byte by
 *      `tests/encoding.rs`; execution lights up automatically once an SDE/hardware ACE
 *      probe passes (R2).
 *
 * The `.byte` constants are the single source of truth shared with the golden table in
 * `tests/encoding.rs` — keep the two in lockstep. Section-6.3 encoding summary used here:
 *
 *   TILEMOVROW (write, imm8)  EVEX.512.66.0F3A.W1 07 /ib   -> 62 F3 FD 48 07 C8 ib
 *   TILEMOVCOL (write, imm8)  EVEX.512.66.0F3A.W1 2F /ib   -> 62 F3 FD 48 2F C8 ib
 *   BSRINIT                   VEX.128.F2.0F38.W1 49 C0     -> C4 E2 FB 49 C0
 *   BSRMOVF                   EVEX.512.NP.MAP6.W1 95       -> 62 F6 F4 48 95 C2 (vvvv=zmm1, rm=zmm2)
 *   BSRMOVH (write / read)    EVEX.512.F2.MAP6.W1/W0 95    -> 62 F6 FF/7F 48 95 C1 (rm=zmm1)
 *   BSRMOVL (write / read)    EVEX.512.F3.MAP6.W1/W0 95    -> 62 F6 FE/7E 48 95 C1 (rm=zmm1)
 *   TOP4B{SS,SU,US,UU}D       EVEX.512.{F2,F3,66,NP}.0F38.W0 5E -> 62 F2 {6F,6E,6D,6C} 48 5E C8
 *   TOP2BF16PS                EVEX.512.F3.0F38.W0 5C       -> 62 F2 6E 48 5C C8
 *   TOP4MX{B,BH,HB,H}F8PS     EVEX.512.{NP,F2,F3,66}.0F3A.W0 8D /ib -> 62 F3 {6C,6F,6E,6D} 48 8D C8 ib
 *   TOP4MXBSSPS               EVEX.512.F2.0F3A.W0 8F /ib   -> 62 F3 6F 48 8F C8 ib
 *
 * Operand register assignment in the ModRM/vvvv fields (register-direct forms):
 *   TOP*:   ModRM 0xC8 = reg tmm1 (dst), rm zmm0 (src1/A); EVEX.vvvv = zmm2 (src2/B).
 *   Moves:  ModRM 0xC8 = reg tmm1 (dst), rm zmm0 (source vector); imm8 = row/col 0.
 *   BSR:    ModRM.reg fixed 000 (bsr0 is implicit); BSRMOVF vvvv = zmm1 (A), rm = zmm2 (B);
 *           BSRMOVH/L rm = zmm1. The MX forms' imm8 (A/B scale-group selector) is fixed 0.
 */
#include <immintrin.h>
#include <stdint.h>

/* Tile row stride: ACE tiles are fixed 16 rows x 64 bytes (spec section 10.2.1). */
#define ACE_TILE_STRIDE 64

/* ------------------------------------------------------------------------------------------- */
/* Family A (intrinsic): tile config lifecycle — LDTILECFG / STTILECFG / TILEZERO / TILERELEASE */
/* ------------------------------------------------------------------------------------------- */

/* STTILECFG round-trip: load the 64-byte descriptor with LDTILECFG, read it back with
 * STTILECFG. For palette 2 the stored descriptor is byte 0 = 2 and bytes 1-63 = 0 (spec
 * section 11.3.4). */
__attribute__((target("amx-tile")))
void ace_tile_cfg_roundtrip(const uint8_t *cfg, uint8_t *out) {
    _tile_loadconfig(cfg);
    _tile_storeconfig(out);
    _tile_release();
}

/* TILEZERO: configure, load tile 0 from `data` (stride 64), zero it, store it back. */
__attribute__((target("amx-tile")))
void ace_tile_zero(const uint8_t *cfg, const uint8_t *data, uint8_t *out) {
    _tile_loadconfig(cfg);
    _tile_loadd(0, data, ACE_TILE_STRIDE);
    _tile_zero(0);
    _tile_stored(0, out, ACE_TILE_STRIDE);
    _tile_release();
}

/* ------------------------------------------------------------------------------------------- */
/* Family B read (intrinsic): TILEMOVROW, tile -> ZMM.                                          */
/* ------------------------------------------------------------------------------------------- */

/* TILEMOVROW (read): configure, load tile 0, extract `row` into a ZMM, store 64 bytes out. */
__attribute__((target("amx-avx512")))
void ace_tile_movrow_read(const uint8_t *cfg, const uint8_t *data, uint32_t row, uint8_t *out) {
    _tile_loadconfig(cfg);
    _tile_loadd(0, data, ACE_TILE_STRIDE);
    __m512i r = (__m512i)_tile_movrow(0, row);
    _mm512_storeu_si512((void *)out, r);
    _tile_release();
}

/* ------------------------------------------------------------------------------------------- */
/* Family C (intrinsic): tile-row converts, tile row -> ZMM.                                    */
/* ------------------------------------------------------------------------------------------- */

__attribute__((target("amx-avx512")))
void ace_tile_tcvtrowd2ps(const uint8_t *cfg, const uint8_t *data, uint32_t row, float *out) {
    _tile_loadconfig(cfg);
    _tile_loadd(0, data, ACE_TILE_STRIDE);
    _mm512_storeu_ps(out, _tile_cvtrowd2ps(0, row));
    _tile_release();
}

__attribute__((target("amx-avx512")))
void ace_tile_tcvtrowps2bf16h(const uint8_t *cfg, const uint8_t *data, uint32_t row, uint16_t *out) {
    _tile_loadconfig(cfg);
    _tile_loadd(0, data, ACE_TILE_STRIDE);
    _mm512_storeu_si512((void *)out, (__m512i)_tile_cvtrowps2bf16h(0, row));
    _tile_release();
}

__attribute__((target("amx-avx512")))
void ace_tile_tcvtrowps2bf16l(const uint8_t *cfg, const uint8_t *data, uint32_t row, uint16_t *out) {
    _tile_loadconfig(cfg);
    _tile_loadd(0, data, ACE_TILE_STRIDE);
    _mm512_storeu_si512((void *)out, (__m512i)_tile_cvtrowps2bf16l(0, row));
    _tile_release();
}

__attribute__((target("amx-avx512")))
void ace_tile_tcvtrowps2phh(const uint8_t *cfg, const uint8_t *data, uint32_t row, uint16_t *out) {
    _tile_loadconfig(cfg);
    _tile_loadd(0, data, ACE_TILE_STRIDE);
    _mm512_storeu_si512((void *)out, (__m512i)_tile_cvtrowps2phh(0, row));
    _tile_release();
}

__attribute__((target("amx-avx512")))
void ace_tile_tcvtrowps2phl(const uint8_t *cfg, const uint8_t *data, uint32_t row, uint16_t *out) {
    _tile_loadconfig(cfg);
    _tile_loadd(0, data, ACE_TILE_STRIDE);
    _mm512_storeu_si512((void *)out, (__m512i)_tile_cvtrowps2phl(0, row));
    _tile_release();
}

/* ------------------------------------------------------------------------------------------- */
/* RAW-ENCODING shims (ACE-only: family B write, D, E, F, G). Encodings per ACE v1 §6.3.       */
/* ------------------------------------------------------------------------------------------- */

/* Family B write — TILEMOVROW write imm8 form (EVEX.512.66.0F3A.W1 07 /ib): tmm1 row imm8
 * <- zmm0. Loads `data` into zmm0, executes with row = 0, stores tile 1. */
__attribute__((target("amx-tile,avx512f")))
void ace_tile_movrow_write(const uint8_t *cfg, const uint8_t *data, uint8_t *out) {
    _tile_loadconfig(cfg);
    _tile_zero(1);
    __asm__ volatile(
        "vmovdqu64 (%[src]), %%zmm0\n\t"
        ".byte 0x62,0xf3,0xfd,0x48,0x07,0xc8,0x00" /* TILEMOVROW tmm1, zmm0, 0 */
        :: [src] "r"(data) : "zmm0", "memory");
    _tile_stored(1, out, ACE_TILE_STRIDE);
    _tile_release();
}

/* Family B write — TILEMOVCOL imm8 form (EVEX.512.66.0F3A.W1 2F /ib): tmm1 byte-column
 * imm8 <- low 16 bytes of zmm0 (spec section 12.3.4). Column fixed at 0. */
__attribute__((target("amx-tile,avx512f")))
void ace_tile_movcol_write(const uint8_t *cfg, const uint8_t *data, uint8_t *out) {
    _tile_loadconfig(cfg);
    _tile_zero(1);
    __asm__ volatile(
        "vmovdqu64 (%[src]), %%zmm0\n\t"
        ".byte 0x62,0xf3,0xfd,0x48,0x2f,0xc8,0x00" /* TILEMOVCOL tmm1, zmm0, 0 */
        :: [src] "r"(data) : "zmm0", "memory");
    _tile_stored(1, out, ACE_TILE_STRIDE);
    _tile_release();
}

/* Family D — BSRINIT (VEX.128.F2.0F38.W1 49 11:000:000, spec section 6.3.5) then read both
 * halves back with the BSRMOVH/BSRMOVL read forms (EVEX.512.F2/F3.MAP6.W0 95). */
__attribute__((target("amx-tile,avx512f")))
void ace_bsrinit_read(const uint8_t *cfg, uint8_t *out_a, uint8_t *out_b) {
    _tile_loadconfig(cfg);
    __asm__ volatile(
        ".byte 0xc4,0xe2,0xfb,0x49,0xc0\n\t"      /* BSRINIT bsr0 */
        ".byte 0x62,0xf6,0x7f,0x48,0x95,0xc1\n\t" /* BSRMOVH zmm1, bsr0 (read) */
        "vmovdqu64 %%zmm1, (%[oa])\n\t"
        ".byte 0x62,0xf6,0x7e,0x48,0x95,0xc1\n\t" /* BSRMOVL zmm1, bsr0 (read) */
        "vmovdqu64 %%zmm1, (%[ob])"
        :: [oa] "r"(out_a), [ob] "r"(out_b) : "zmm1", "memory");
    _tile_release();
}

/* Family D — BSRMOVF (EVEX.512.NP.MAP6.W1 95: zmm1 -> A scales, zmm2 -> B scales, spec
 * section 13.2.4) then read both halves back. */
__attribute__((target("amx-tile,avx512f")))
void ace_bsrmovf_read(const uint8_t *a, const uint8_t *b, const uint8_t *cfg,
                      uint8_t *out_a, uint8_t *out_b) {
    _tile_loadconfig(cfg);
    __asm__ volatile(
        "vmovdqu64 (%[ia]), %%zmm1\n\t"
        "vmovdqu64 (%[ib]), %%zmm2\n\t"
        ".byte 0x62,0xf6,0xf4,0x48,0x95,0xc2\n\t" /* BSRMOVF bsr0, zmm1, zmm2 */
        ".byte 0x62,0xf6,0x7f,0x48,0x95,0xc1\n\t" /* BSRMOVH zmm1, bsr0 (read) */
        "vmovdqu64 %%zmm1, (%[oa])\n\t"
        ".byte 0x62,0xf6,0x7e,0x48,0x95,0xc1\n\t" /* BSRMOVL zmm1, bsr0 (read) */
        "vmovdqu64 %%zmm1, (%[ob])"
        :: [ia] "r"(a), [ib] "r"(b), [oa] "r"(out_a), [ob] "r"(out_b)
        : "zmm1", "zmm2", "memory");
    _tile_release();
}

/* Family D — BSRMOVH/BSRMOVL write forms (EVEX.512.F2/F3.MAP6.W1 95: zmm1 -> half), each
 * read back through its own read form. */
__attribute__((target("amx-tile,avx512f")))
void ace_bsrmovh_roundtrip(const uint8_t *a, const uint8_t *cfg, uint8_t *out_a) {
    _tile_loadconfig(cfg);
    __asm__ volatile(
        "vmovdqu64 (%[ia]), %%zmm1\n\t"
        ".byte 0x62,0xf6,0xff,0x48,0x95,0xc1\n\t" /* BSRMOVH bsr0, zmm1 (write) */
        "vpxord %%zmm1, %%zmm1, %%zmm1\n\t"
        ".byte 0x62,0xf6,0x7f,0x48,0x95,0xc1\n\t" /* BSRMOVH zmm1, bsr0 (read) */
        "vmovdqu64 %%zmm1, (%[oa])"
        :: [ia] "r"(a), [oa] "r"(out_a) : "zmm1", "memory");
    _tile_release();
}

__attribute__((target("amx-tile,avx512f")))
void ace_bsrmovl_roundtrip(const uint8_t *b, const uint8_t *cfg, uint8_t *out_b) {
    _tile_loadconfig(cfg);
    __asm__ volatile(
        "vmovdqu64 (%[ib]), %%zmm1\n\t"
        ".byte 0x62,0xf6,0xfe,0x48,0x95,0xc1\n\t" /* BSRMOVL bsr0, zmm1 (write) */
        "vpxord %%zmm1, %%zmm1, %%zmm1\n\t"
        ".byte 0x62,0xf6,0x7e,0x48,0x95,0xc1\n\t" /* BSRMOVL zmm1, bsr0 (read) */
        "vmovdqu64 %%zmm1, (%[ob])"
        :: [ib] "r"(b), [ob] "r"(out_b) : "zmm1", "memory");
    _tile_release();
}

/* Families G/F — TOP4B{SS,SU,US,UU}D (EVEX.512.pp.0F38.W0 5E) and TOP2BF16PS
 * (EVEX.512.F3.0F38.W0 5C): tmm1 (dst, ModRM.reg) accumulates src1 = zmm0 (ModRM.rm) x
 * src2 = zmm2 (EVEX.vvvv). The accumulator tile is loaded from `c` and stored back. */
#define ACE_TOP_SHIM(fn, bytes)                                                                 \
    __attribute__((target("amx-tile,avx512f")))                                                 \
    void fn(const uint8_t *cfg, const uint8_t *c, const uint8_t *a, const uint8_t *b,           \
            uint8_t *out) {                                                                     \
        _tile_loadconfig(cfg);                                                                  \
        _tile_loadd(1, c, ACE_TILE_STRIDE);                                                     \
        __asm__ volatile(                                                                       \
            "vmovdqu64 (%[ia]), %%zmm0\n\t"                                                     \
            "vmovdqu64 (%[ib]), %%zmm2\n\t"                                                     \
            ".byte " bytes                                                                      \
            :: [ia] "r"(a), [ib] "r"(b) : "zmm0", "zmm2", "memory");                            \
        _tile_stored(1, out, ACE_TILE_STRIDE);                                                  \
        _tile_release();                                                                        \
    }

ACE_TOP_SHIM(ace_tile_top4bssd,   "0x62,0xf2,0x6f,0x48,0x5e,0xc8") /* F2 */
ACE_TOP_SHIM(ace_tile_top4bsud,   "0x62,0xf2,0x6e,0x48,0x5e,0xc8") /* F3 */
ACE_TOP_SHIM(ace_tile_top4busd,   "0x62,0xf2,0x6d,0x48,0x5e,0xc8") /* 66 */
ACE_TOP_SHIM(ace_tile_top4buud,   "0x62,0xf2,0x6c,0x48,0x5e,0xc8") /* NP */
ACE_TOP_SHIM(ace_tile_top2bf16ps, "0x62,0xf2,0x6e,0x48,0x5c,0xc8") /* F3, op 5C */

/* Family E — MX rank-4 outer products (EVEX.512.pp.0F3A.W0 8D/8F /ib). The BSR is implicit
 * (bsr0), seeded with BSRMOVF from the caller's A/B scale buffers; imm8 (the A/B
 * scale-group selector, spec section 14.1.4) is fixed at 0 (groups 0/0). */
#define ACE_MX_SHIM(fn, bytes)                                                                  \
    __attribute__((target("amx-tile,avx512f")))                                                 \
    void fn(const uint8_t *cfg, const uint8_t *c, const uint8_t *a, const uint8_t *b,           \
            const uint8_t *a_scales, const uint8_t *b_scales, uint8_t *out) {                   \
        _tile_loadconfig(cfg);                                                                  \
        _tile_loadd(1, c, ACE_TILE_STRIDE);                                                     \
        __asm__ volatile(                                                                       \
            "vmovdqu64 (%[sa]), %%zmm1\n\t"                                                     \
            "vmovdqu64 (%[sb]), %%zmm2\n\t"                                                     \
            ".byte 0x62,0xf6,0xf4,0x48,0x95,0xc2\n\t" /* BSRMOVF bsr0, zmm1, zmm2 */            \
            "vmovdqu64 (%[ia]), %%zmm0\n\t"                                                     \
            "vmovdqu64 (%[ib]), %%zmm2\n\t"                                                     \
            ".byte " bytes                                                                      \
            :: [sa] "r"(a_scales), [sb] "r"(b_scales), [ia] "r"(a), [ib] "r"(b)                 \
            : "zmm0", "zmm1", "zmm2", "memory");                                                \
        _tile_stored(1, out, ACE_TILE_STRIDE);                                                  \
        _tile_release();                                                                        \
    }

ACE_MX_SHIM(ace_tile_top4mxbf8ps,  "0x62,0xf3,0x6c,0x48,0x8d,0xc8,0x00") /* NP */
ACE_MX_SHIM(ace_tile_top4mxbhf8ps, "0x62,0xf3,0x6f,0x48,0x8d,0xc8,0x00") /* F2 */
ACE_MX_SHIM(ace_tile_top4mxhbf8ps, "0x62,0xf3,0x6e,0x48,0x8d,0xc8,0x00") /* F3 */
ACE_MX_SHIM(ace_tile_top4mxhf8ps,  "0x62,0xf3,0x6d,0x48,0x8d,0xc8,0x00") /* 66 */
ACE_MX_SHIM(ace_tile_top4mxbssps,  "0x62,0xf3,0x6f,0x48,0x8f,0xc8,0x00") /* F2, op 8F */
