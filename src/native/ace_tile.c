/*
 * Native ACE group-4 tile-instruction shims (design decision D7, OQ-6).
 *
 * Two kinds of shim live here, exactly as design.md §9/§11.6 prescribes:
 *
 *   1. INTRINSIC shims (families A, B-read, C) — the tile config lifecycle, the tile->ZMM
 *      row/column READ moves, and the five tile-row converts are assembler/intrinsic-reachable
 *      in current GCC/Clang (AMX-TILE + AMX-AVX512): `_tile_loadconfig`/`_tile_storeconfig`/
 *      `_tile_zero`/`_tile_release`, `_tile_loadd`/`_tile_stored`, `_tile_movrow`,
 *      `_tile_cvtrowd2ps`/`_tile_cvtrowps2bf16{h,l}`/`_tile_cvtrowps2ph{h,l}`. Compiled with
 *      `-mamx-tile -mamx-avx512 -mavx10.2` (see build.rs). These execute under Intel SDE.
 *
 *   2. `.byte` RAW-ENCODING shims (family B WRITE forms, D, E, F, G) — the `ACE`-only outer
 *      products, block-scale registers, and ZMM->tile write moves. No assembler knows the ACE
 *      mnemonics yet (binutils 2.46 / SDE 10.8 predate ACE v1.15), so each is emitted as a
 *      localized `.byte` sequence hand-encoded per ACE v1 §6 (the EVEX tile-instruction format).
 *      Operands are marshalled through memory: the tiles are loaded from the caller's buffers
 *      with the assembler-known `_tile_loadd`, the ACE instruction executes over the fixed tmm
 *      register operands its `.byte` encoding names, and the destination tile is stored back
 *      with `_tile_stored`. They are BUILT, not executed: SDE has no ACE emulation yet (R2), so
 *      the differential layer discards for these families until a `sde64` ACE probe passes; the
 *      golden `.byte` constants are asserted by `tests/encoding.rs` with no external tool.
 *
 * The exact `.byte` constants are the single source of truth shared with the golden table in
 * `tests/encoding.rs::golden` — keep the two in lockstep. Each encoding is a 6-byte EVEX form
 * `62 P0 P1 P2 op modrm`, register-register (ModRM.mod=11), 512-bit (EVEX.L'L), no write-mask;
 * the per-mnemonic (map, W, pp, opcode) assignment is documented in the encoding test.
 *
 * ASSEMBLER/SDE ACE EMULATION UNAVAILABLE; ENCODINGS GROUNDED AGAINST ACE v1 §6.
 */
#include <immintrin.h>
#include <stdint.h>

/* The palette-2 tile config the shims load. The caller passes a 64-byte LDTILECFG descriptor. */
#define ACE_TILE_STRIDE 64

/* ------------------------------------------------------------------------------------------- */
/* Family A (intrinsic): tile config lifecycle — LDTILECFG / STTILECFG / TILEZERO / TILERELEASE */
/* ------------------------------------------------------------------------------------------- */

/* STTILECFG round-trip: load the 64-byte palette-2 descriptor with LDTILECFG, read it back with
 * STTILECFG. The differential asserts the stored descriptor equals the loaded one (INV-3). */
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
/* Family B read (intrinsic): TILEMOVROW / TILEMOVCOL, tile -> ZMM.                            */
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
/* Family C (intrinsic): tile-row converts, tile row -> ZMM.                                   */
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
/* `.byte` RAW-ENCODING shims (ACE-only: family B write, D, E, F, G).                          */
/*                                                                                             */
/* Each marshals operands through memory: `_tile_loadconfig` sets the palette-2 shape,          */
/* `_tile_loadd` loads the operand tiles into fixed tmm registers (tmm0 = A / value / dst-in,    */
/* tmm2 = B), the ACE instruction runs via its `.byte` encoding over those tmm operands, and     */
/* the destination tile 1 is stored back with `_tile_stored`. Built, not executed, until SDE     */
/* ACE emulation lands (OQ-6, R2). The `.byte` constants match tests/encoding.rs::golden.        */
/* ------------------------------------------------------------------------------------------- */

/* Load config + the (up to) three tile operands used by the outer-product shims. tmm0=A(dst-in
 * for moves), tmm1=C(dst accumulator), tmm2=B. */
__attribute__((target("amx-tile"))) static inline
void ace_tile_load3(const uint8_t *cfg, const uint8_t *c, const uint8_t *a, const uint8_t *b) {
    _tile_loadconfig(cfg);
    if (c) _tile_loadd(1, c, ACE_TILE_STRIDE);
    if (a) _tile_loadd(0, a, ACE_TILE_STRIDE);
    if (b) _tile_loadd(2, b, ACE_TILE_STRIDE);
}

/* Family B write — TILEMOVROW (write form): ZMM -> tile row. .byte: 62 F5 7E 48 6C C8 */
__attribute__((target("amx-tile")))
void ace_tile_movrow_write(const uint8_t *cfg, const uint8_t *data, uint8_t *out) {
    ace_tile_load3(cfg, data, NULL, NULL);
    __asm__ volatile(".byte 0x62,0xf5,0x7e,0x48,0x6c,0xc8" ::: "memory");
    _tile_stored(1, out, ACE_TILE_STRIDE);
    _tile_release();
}

/* Family B write — TILEMOVCOL (write form): ZMM -> tile column. .byte: 62 F5 7E 48 6D C8 */
__attribute__((target("amx-tile")))
void ace_tile_movcol_write(const uint8_t *cfg, const uint8_t *data, uint8_t *out) {
    ace_tile_load3(cfg, data, NULL, NULL);
    __asm__ volatile(".byte 0x62,0xf5,0x7e,0x48,0x6d,0xc8" ::: "memory");
    _tile_stored(1, out, ACE_TILE_STRIDE);
    _tile_release();
}

/* Family D — BSRINIT: seed a block-scale register. .byte: 62 F5 FC 48 50 C8 */
__attribute__((target("amx-tile")))
void ace_tile_bsrinit(const uint8_t *cfg, const uint8_t *data, uint8_t *out) {
    ace_tile_load3(cfg, data, NULL, NULL);
    __asm__ volatile(".byte 0x62,0xf5,0xfc,0x48,0x50,0xc8" ::: "memory");
    _tile_stored(1, out, ACE_TILE_STRIDE);
    _tile_release();
}

/* Family D — BSRMOVF / BSRMOVH / BSRMOVL: move the full / high / low block-scale factor.
 * .byte: 62 F5 FC 48 {51,52,53} C8 */
__attribute__((target("amx-tile")))
void ace_tile_bsrmovf(const uint8_t *cfg, const uint8_t *data, uint8_t *out) {
    ace_tile_load3(cfg, data, NULL, NULL);
    __asm__ volatile(".byte 0x62,0xf5,0xfc,0x48,0x51,0xc8" ::: "memory");
    _tile_stored(1, out, ACE_TILE_STRIDE);
    _tile_release();
}

__attribute__((target("amx-tile")))
void ace_tile_bsrmovh(const uint8_t *cfg, const uint8_t *data, uint8_t *out) {
    ace_tile_load3(cfg, data, NULL, NULL);
    __asm__ volatile(".byte 0x62,0xf5,0xfc,0x48,0x52,0xc8" ::: "memory");
    _tile_stored(1, out, ACE_TILE_STRIDE);
    _tile_release();
}

__attribute__((target("amx-tile")))
void ace_tile_bsrmovl(const uint8_t *cfg, const uint8_t *data, uint8_t *out) {
    ace_tile_load3(cfg, data, NULL, NULL);
    __asm__ volatile(".byte 0x62,0xf5,0xfc,0x48,0x53,0xc8" ::: "memory");
    _tile_stored(1, out, ACE_TILE_STRIDE);
    _tile_release();
}

/* Family G — INT8 rank-4 outer products TOP4B{SS,SU,US,UU}D: C(tmm1) += A(tmm0) (x) B(tmm2).
 * .byte: 62 F6 {7F,7E,7D,7C} 48 60 C8 (pp encodes the signedness pair). */
#define ACE_TOP_SHIM(fn, p1)                                                                    \
    __attribute__((target("amx-tile")))                                                         \
    void fn(const uint8_t *cfg, const uint8_t *c, const uint8_t *a, const uint8_t *b,           \
            uint8_t *out) {                                                                     \
        ace_tile_load3(cfg, c, a, b);                                                           \
        __asm__ volatile(".byte 0x62,0xf6," #p1 ",0x48,0x60,0xc8" ::: "memory");                \
        _tile_stored(1, out, ACE_TILE_STRIDE);                                                  \
        _tile_release();                                                                        \
    }
ACE_TOP_SHIM(ace_tile_top4bssd, 0x7f)
ACE_TOP_SHIM(ace_tile_top4bsud, 0x7e)
ACE_TOP_SHIM(ace_tile_top4busd, 0x7d)
ACE_TOP_SHIM(ace_tile_top4buud, 0x7c)

/* Family F — TOP2BF16PS: BF16 rank-2 outer product into FP32, no block scale.
 * .byte: 62 F6 7E 48 61 C8 */
__attribute__((target("amx-tile")))
void ace_tile_top2bf16ps(const uint8_t *cfg, const uint8_t *c, const uint8_t *a, const uint8_t *b,
                         uint8_t *out) {
    ace_tile_load3(cfg, c, a, b);
    __asm__ volatile(".byte 0x62,0xf6,0x7e,0x48,0x61,0xc8" ::: "memory");
    _tile_stored(1, out, ACE_TILE_STRIDE);
    _tile_release();
}

/* Family E — MX-FP8 rank-4 outer products with per-block BSR scale (BSR in tmm3).
 * .byte: 62 F6 {FC,FD,FE,FF} 48 70 C8 for the four mixed-format forms; 71 for TOP4MXBSSPS. */
#define ACE_MX_SHIM(fn, p1, op)                                                                 \
    __attribute__((target("amx-tile")))                                                         \
    void fn(const uint8_t *cfg, const uint8_t *c, const uint8_t *a, const uint8_t *b,           \
            const uint8_t *bsr, uint8_t *out) {                                                 \
        ace_tile_load3(cfg, c, a, b);                                                           \
        if (bsr) _tile_loadd(3, bsr, ACE_TILE_STRIDE);                                          \
        __asm__ volatile(".byte 0x62,0xf6," #p1 ",0x48," #op ",0xc8" ::: "memory");             \
        _tile_stored(1, out, ACE_TILE_STRIDE);                                                  \
        _tile_release();                                                                        \
    }
ACE_MX_SHIM(ace_tile_top4mxbf8ps,  0xfc, 0x70)
ACE_MX_SHIM(ace_tile_top4mxbhf8ps, 0xfd, 0x70)
ACE_MX_SHIM(ace_tile_top4mxhbf8ps, 0xfe, 0x70)
ACE_MX_SHIM(ace_tile_top4mxhf8ps,  0xff, 0x70)
ACE_MX_SHIM(ace_tile_top4mxbssps,  0xfc, 0x71)
