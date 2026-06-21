/*
 * Native AVX10_V1_AUX shims (design decision D7).
 *
 * One `extern "C"` shim per AVX10_V1_AUX primitive (26 total). Each is compiled with
 * `-mavx10.2` (see build.rs) and tagged __attribute__((target("avx10.2"))) so the EVEX
 * forms are emitted. Each takes plain pointers, loads with the matching _mm512_loadu_* /
 * _mm256_loadu_si256, calls the GCC AVX10.2 intrinsic, and stores with _storeu_.
 *
 * Lane-ordering note (families B and E, spec sec 8.2.5 / 8.3.5): the output low half comes
 * from src2 and the high half from src1. Empirically verified under SDE that the GCC
 * two-source intrinsics map their (A, B) arguments as A -> high half, B -> low half, i.e.
 * intrinsic(A=src1, B=src2). So we pass (load(src1), load(src2)) directly.
 *
 * Family C bias (spec sec 8.4.5): bias = src1.byte[2*i]. The Rust side hands us the bias
 * operand as a 32-lane u16 array (64 bytes); the hardware selects byte[2*i] itself, which
 * is the low byte of the i-th u16 -- matching the oracle's `bias[i] & 0xff`.
 *
 * VNNI (families F/G, spec sec 8.6.5 / 8.7.5): _mm512_dp*_epi32(W, A, B) computes
 * W + dot(A, B); W is the by-value accumulator (dst), A is the first (left) operand, B the
 * second. For the mixed-sign forms A is the operand named first in the mnemonic (e.g. SU:
 * A signed, B unsigned), matching the oracle's (a, b) operand order.
 */
#include <immintrin.h>
#include <stdint.h>

/* ---- Family A: single-source FP16 -> FP8 (32 lanes in, 32 bytes out) ---- */

__attribute__((target("avx10.2")))
void ace_native_cvtph_bf8(const uint16_t *a, uint8_t *out) {
    __m512h va = _mm512_loadu_ph(a);
    __m256i r = _mm512_cvtph_bf8(va);
    _mm256_storeu_si256((__m256i *)out, r);
}

__attribute__((target("avx10.2")))
void ace_native_cvtphs_bf8(const uint16_t *a, uint8_t *out) {
    __m512h va = _mm512_loadu_ph(a);
    __m256i r = _mm512_cvts_ph_bf8(va);
    _mm256_storeu_si256((__m256i *)out, r);
}

__attribute__((target("avx10.2")))
void ace_native_cvtph_hf8(const uint16_t *a, uint8_t *out) {
    __m512h va = _mm512_loadu_ph(a);
    __m256i r = _mm512_cvtph_hf8(va);
    _mm256_storeu_si256((__m256i *)out, r);
}

__attribute__((target("avx10.2")))
void ace_native_cvtphs_hf8(const uint16_t *a, uint8_t *out) {
    __m512h va = _mm512_loadu_ph(a);
    __m256i r = _mm512_cvts_ph_hf8(va);
    _mm256_storeu_si256((__m256i *)out, r);
}

/* ---- Family B: two-source FP16 -> FP8 (64 lanes out; low=src2, high=src1) ---- */

__attribute__((target("avx10.2")))
void ace_native_cvt2ph_bf8(const uint16_t *src1, const uint16_t *src2, uint8_t *out) {
    __m512h a = _mm512_loadu_ph(src1); /* -> high half */
    __m512h b = _mm512_loadu_ph(src2); /* -> low half  */
    __m512i r = _mm512_cvt2ph_bf8(a, b);
    _mm512_storeu_si512((void *)out, r);
}

__attribute__((target("avx10.2")))
void ace_native_cvt2phs_bf8(const uint16_t *src1, const uint16_t *src2, uint8_t *out) {
    __m512h a = _mm512_loadu_ph(src1);
    __m512h b = _mm512_loadu_ph(src2);
    __m512i r = _mm512_cvts_2ph_bf8(a, b);
    _mm512_storeu_si512((void *)out, r);
}

__attribute__((target("avx10.2")))
void ace_native_cvt2ph_hf8(const uint16_t *src1, const uint16_t *src2, uint8_t *out) {
    __m512h a = _mm512_loadu_ph(src1);
    __m512h b = _mm512_loadu_ph(src2);
    __m512i r = _mm512_cvt2ph_hf8(a, b);
    _mm512_storeu_si512((void *)out, r);
}

__attribute__((target("avx10.2")))
void ace_native_cvt2phs_hf8(const uint16_t *src1, const uint16_t *src2, uint8_t *out) {
    __m512h a = _mm512_loadu_ph(src1);
    __m512h b = _mm512_loadu_ph(src2);
    __m512i r = _mm512_cvts_2ph_hf8(a, b);
    _mm512_storeu_si512((void *)out, r);
}

/* ---- Family C: biased FP16 -> FP8 (bias = src1.byte[2*i]) ---- */

__attribute__((target("avx10.2")))
void ace_native_cvtbiasph_bf8(const uint16_t *a, const uint16_t *bias, uint8_t *out) {
    __m512i vbias = _mm512_loadu_si512((const void *)bias); /* 64 bias bytes */
    __m512h va = _mm512_loadu_ph(a);
    __m256i r = _mm512_cvtbiasph_bf8(vbias, va);
    _mm256_storeu_si256((__m256i *)out, r);
}

__attribute__((target("avx10.2")))
void ace_native_cvtbiasphs_bf8(const uint16_t *a, const uint16_t *bias, uint8_t *out) {
    __m512i vbias = _mm512_loadu_si512((const void *)bias);
    __m512h va = _mm512_loadu_ph(a);
    __m256i r = _mm512_cvts_biasph_bf8(vbias, va);
    _mm256_storeu_si256((__m256i *)out, r);
}

__attribute__((target("avx10.2")))
void ace_native_cvtbiasph_hf8(const uint16_t *a, const uint16_t *bias, uint8_t *out) {
    __m512i vbias = _mm512_loadu_si512((const void *)bias);
    __m512h va = _mm512_loadu_ph(a);
    __m256i r = _mm512_cvtbiasph_hf8(vbias, va);
    _mm256_storeu_si256((__m256i *)out, r);
}

__attribute__((target("avx10.2")))
void ace_native_cvtbiasphs_hf8(const uint16_t *a, const uint16_t *bias, uint8_t *out) {
    __m512i vbias = _mm512_loadu_si512((const void *)bias);
    __m512h va = _mm512_loadu_ph(a);
    __m256i r = _mm512_cvts_biasph_hf8(vbias, va);
    _mm256_storeu_si256((__m256i *)out, r);
}

/* ---- Family D: HF8 (E4M3) -> FP16 (32 bytes in, 32 lanes out) ---- */

__attribute__((target("avx10.2")))
void ace_native_cvthf8_ph(const uint8_t *a, uint16_t *out) {
    __m256i va = _mm256_loadu_si256((const __m256i *)a);
    __m512h r = _mm512_cvthf8_ph(va);
    _mm512_storeu_ph(out, r);
}

/* ---- Family E: FP32 pair -> FP16 (low=src2, high=src1) ---- */

__attribute__((target("avx10.2")))
void ace_native_cvt2ps_phx(const float *src1, const float *src2, uint16_t *out) {
    __m512 a = _mm512_loadu_ps(src1); /* -> high half */
    __m512 b = _mm512_loadu_ps(src2); /* -> low half  */
    __m512h r = _mm512_cvtx2ps_ph(a, b);
    _mm512_storeu_ph(out, r);
}

/* ---- Family F: byte VNNI (W + dot(A,B), 16 i32 lanes) ---- */

__attribute__((target("avx10.2")))
void ace_native_dpbssd(const int32_t *dst, const int8_t *a, const int8_t *b, int32_t *out) {
    __m512i w = _mm512_loadu_si512((const void *)dst);
    __m512i va = _mm512_loadu_si512((const void *)a);
    __m512i vb = _mm512_loadu_si512((const void *)b);
    _mm512_storeu_si512((void *)out, _mm512_dpbssd_epi32(w, va, vb));
}

__attribute__((target("avx10.2")))
void ace_native_dpbssds(const int32_t *dst, const int8_t *a, const int8_t *b, int32_t *out) {
    __m512i w = _mm512_loadu_si512((const void *)dst);
    __m512i va = _mm512_loadu_si512((const void *)a);
    __m512i vb = _mm512_loadu_si512((const void *)b);
    _mm512_storeu_si512((void *)out, _mm512_dpbssds_epi32(w, va, vb));
}

__attribute__((target("avx10.2")))
void ace_native_dpbsud(const int32_t *dst, const int8_t *a, const uint8_t *b, int32_t *out) {
    __m512i w = _mm512_loadu_si512((const void *)dst);
    __m512i va = _mm512_loadu_si512((const void *)a);
    __m512i vb = _mm512_loadu_si512((const void *)b);
    _mm512_storeu_si512((void *)out, _mm512_dpbsud_epi32(w, va, vb));
}

__attribute__((target("avx10.2")))
void ace_native_dpbsuds(const int32_t *dst, const int8_t *a, const uint8_t *b, int32_t *out) {
    __m512i w = _mm512_loadu_si512((const void *)dst);
    __m512i va = _mm512_loadu_si512((const void *)a);
    __m512i vb = _mm512_loadu_si512((const void *)b);
    _mm512_storeu_si512((void *)out, _mm512_dpbsuds_epi32(w, va, vb));
}

__attribute__((target("avx10.2")))
void ace_native_dpbuud(const int32_t *dst, const uint8_t *a, const uint8_t *b, int32_t *out) {
    __m512i w = _mm512_loadu_si512((const void *)dst);
    __m512i va = _mm512_loadu_si512((const void *)a);
    __m512i vb = _mm512_loadu_si512((const void *)b);
    _mm512_storeu_si512((void *)out, _mm512_dpbuud_epi32(w, va, vb));
}

__attribute__((target("avx10.2")))
void ace_native_dpbuuds(const int32_t *dst, const uint8_t *a, const uint8_t *b, int32_t *out) {
    __m512i w = _mm512_loadu_si512((const void *)dst);
    __m512i va = _mm512_loadu_si512((const void *)a);
    __m512i vb = _mm512_loadu_si512((const void *)b);
    _mm512_storeu_si512((void *)out, _mm512_dpbuuds_epi32(w, va, vb));
}

/* ---- Family G: word VNNI (W + dot(A,B), 16 i32 lanes) ---- */

__attribute__((target("avx10.2")))
void ace_native_dpwsud(const int32_t *dst, const int16_t *a, const uint16_t *b, int32_t *out) {
    __m512i w = _mm512_loadu_si512((const void *)dst);
    __m512i va = _mm512_loadu_si512((const void *)a);
    __m512i vb = _mm512_loadu_si512((const void *)b);
    _mm512_storeu_si512((void *)out, _mm512_dpwsud_epi32(w, va, vb));
}

__attribute__((target("avx10.2")))
void ace_native_dpwsuds(const int32_t *dst, const int16_t *a, const uint16_t *b, int32_t *out) {
    __m512i w = _mm512_loadu_si512((const void *)dst);
    __m512i va = _mm512_loadu_si512((const void *)a);
    __m512i vb = _mm512_loadu_si512((const void *)b);
    _mm512_storeu_si512((void *)out, _mm512_dpwsuds_epi32(w, va, vb));
}

__attribute__((target("avx10.2")))
void ace_native_dpwusd(const int32_t *dst, const uint16_t *a, const int16_t *b, int32_t *out) {
    __m512i w = _mm512_loadu_si512((const void *)dst);
    __m512i va = _mm512_loadu_si512((const void *)a);
    __m512i vb = _mm512_loadu_si512((const void *)b);
    _mm512_storeu_si512((void *)out, _mm512_dpwusd_epi32(w, va, vb));
}

__attribute__((target("avx10.2")))
void ace_native_dpwusds(const int32_t *dst, const uint16_t *a, const int16_t *b, int32_t *out) {
    __m512i w = _mm512_loadu_si512((const void *)dst);
    __m512i va = _mm512_loadu_si512((const void *)a);
    __m512i vb = _mm512_loadu_si512((const void *)b);
    _mm512_storeu_si512((void *)out, _mm512_dpwusds_epi32(w, va, vb));
}

__attribute__((target("avx10.2")))
void ace_native_dpwuud(const int32_t *dst, const uint16_t *a, const uint16_t *b, int32_t *out) {
    __m512i w = _mm512_loadu_si512((const void *)dst);
    __m512i va = _mm512_loadu_si512((const void *)a);
    __m512i vb = _mm512_loadu_si512((const void *)b);
    _mm512_storeu_si512((void *)out, _mm512_dpwuud_epi32(w, va, vb));
}

__attribute__((target("avx10.2")))
void ace_native_dpwuuds(const int32_t *dst, const uint16_t *a, const uint16_t *b, int32_t *out) {
    __m512i w = _mm512_loadu_si512((const void *)dst);
    __m512i va = _mm512_loadu_si512((const void *)a);
    __m512i vb = _mm512_loadu_si512((const void *)b);
    _mm512_storeu_si512((void *)out, _mm512_dpwuuds_epi32(w, va, vb));
}
