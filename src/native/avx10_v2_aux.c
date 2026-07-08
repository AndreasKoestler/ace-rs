/*
 * Native AVX10_V2_AUX shims (design decision D7; OCP format-conversion family).
 *
 * One `extern "C"` shim per AVX10_V2_AUX primitive, compiled with `-mavx10.2` (see
 * build.rs) and tagged __attribute__((target("avx10.2"))) so the EVEX MAP5/0F38 forms are
 * emitted. Each takes plain pointers, loads with the matching _mm{128,256,512}_loadu_*,
 * calls the GCC/Clang AVX10.2 intrinsic, and stores with _storeu_.
 *
 * This is the iteration-2 sibling of `avx10_v1_aux.c`; both are compiled into the crate's
 * native object under the opt-in `native` feature on x86_64.
 *
 * OQ-5 (a family whose intrinsic does not compile under the available -mavx10.2 toolchain
 * ships oracle-only): the FP8->FP32 (family C), FP32->FP8 (families A and B), FP8->FP4
 * (family D), FP4->FP8 (family E), FP8->FP6 (family F), FP6->FP8 (family G), VPMOVSSDB
 * (family H) and VUNPACKB (family I) primitives all ship oracle-only.
 *
 *  - Family C: `VCVTBF82PS` has no `_mm512_cvtbf8_ps` intrinsic and `VCVTHF82PS` has no
 *    `_mm512_cvthf8_ps` intrinsic in the installed GCC 16.x `-mavx10.2` headers (the FP8->FP16
 *    siblings `_mm512_cvtbf8_ph` / `_mm512_cvthf8_ph` exist, but not the FP8->FP32 forms).
 *  - Family A: no `_mm512_cvtps_bf8` / `_mm512_cvts_ps_bf8` / `_mm512_cvtps_hf8` /
 *    `_mm512_cvtroundps_hf8` FP32->FP8 intrinsics exist.
 *  - Family B (FP32->FP8 bias-rounding): no `_mm512_cvtbiasps_bf8` / `_mm512_cvtbiaspss_bf8` /
 *    `_mm512_cvtbiasps_hf8` / `_mm512_cvtbiaspss_hf8` intrinsics exist in GCC 16.1.1 — only the
 *    FP16-source siblings (`_mm512_cvtbiasph_bf8`, ...) are present, confirmed by a compile
 *    probe (`error: implicit declaration of function '_mm512_cvtbiasps_bf8'; did you mean
 *    '_mm512_cvtbiasph_bf8'?`).
 *  - Family D (FP8->FP4 E2M1, saturating, nibble-packed): no `_mm512_cvtbf8_bf4s` /
 *    `_mm512_cvthf8_bf4s` (`VCVTBF82BF4S` / `VCVTHF82BF4S`) intrinsics exist in GCC 16.1.1,
 *    confirmed by a compile probe (`error: implicit declaration of function
 *    '_mm512_cvtbf8_bf4s'; did you mean '_mm512_cvtbf8_ph'?`); every naming variant
 *    (`_mm512_cvts_bf8_bf4`, `_mm512_cvtbf8s_bf4`, `_mm512_cvts_hf8_bf4`, `_mm512_cvtbf82bf4s`)
 *    is equally absent.
 *  - Family E (FP4 E2M1 -> FP8 E4M3, exact, nibble-unpacked): no `_mm512_cvtbf4_hf8`
 *    (`VCVTBF42HF8`) intrinsic exists in GCC 16.1.1, confirmed by a compile probe (`error:
 *    implicit declaration of function '_mm512_cvtbf4_hf8'; did you mean '_mm512_cvtph_hf8'?`);
 *    every naming variant (`_mm512_cvtbf42hf8`, `_mm512_cvtbf4_phf8`, `_mm512_cvt_bf4_hf8`) is
 *    equally absent.
 *  - Family F (FP8->FP6, saturating-RTNE, 6-bit packed): no `_mm512_cvtf8_bf6s`
 *    (`VCVTBF82BF6S`, FP8 E5M2 -> FP6 E3M2) and no `_mm512_cvtf8_hf6s` (`VCVTHF82HF6S`, FP8
 *    E4M3 -> FP6 E2M3) intrinsic exists in GCC 16.1.1, confirmed by a compile probe (`error:
 *    implicit declaration of function '_mm512_cvtf8_bf6s'; did you mean '_mm512_cvtph_bf8'?`
 *    and `... '_mm512_cvtf8_hf6s'; did you mean '_mm512_cvtph_hf8'?`); every naming variant
 *    (`_mm512_cvtbf8_bf6s`, `_mm512_cvthf8_hf6s`, `_mm512_cvtbf82bf6s`, `_mm512_cvts_bf8_bf6`,
 *    `_mm512_cvtbf8s_bf6`, `_mm512_cvtf8_bf6`) is equally absent.
 *  - Family G (FP6 -> FP8 E4M3, exact, 6-bit unpacked): no `_mm512_cvtf6_hf8` /
 *    `_mm512_cvtbf6_hf8` / `_mm512_cvthf6_hf8` (`VCVTBF62HF8` / `VCVTHF62HF8`) intrinsics
 *    exist in GCC 16.1.1.
 *  - Family H (`VPMOVSSDB`, INT32 -> INT8 symmetric saturation): no `_mm512_cvtssepi32_epi8`
 *    intrinsic exists (only the ordinary asymmetric `_mm512_cvtsepi32_epi8` of `VPMOVSDB`).
 *  - Family I (`VUNPACKB`): no `_mm512_unpackb` intrinsic exists (and the instruction's
 *    `imm8` would additionally need a compile-time-constant dispatch).
 *
 * Per OQ-5 every group-3 family (A through I) therefore ships
 * **oracle-only**: there is no native shim and no `extern "C"` declaration for any of them, so
 * the always-correct scalar oracle is the only path until the intrinsics land in the
 * toolchain. The differential test that would otherwise tie a native path to the oracle
 * DISCARDS (no native path exists), so correctness is grounded against the spec section-16.1 /
 * section-16.3 pseudocode transcribed in `src/fp8.rs`, `src/fp4.rs` and `src/fp6.rs`. This TU
 * is intentionally otherwise empty (later AVX10_V2_AUX families will add their shims here as
 * their intrinsics become available); the translation unit is still compiled by build.rs so
 * wiring is in place.
 */
#include <immintrin.h>
#include <stdint.h>

/* No AVX10_V2_AUX shims are emitted yet (OQ-5; see the file header). A single dummy symbol
 * keeps the object non-empty and well-formed for the linker. It is never referenced from
 * Rust. */
int ace_native_avx10_v2_aux_placeholder(void) {
    return 0;
}
