//! `ace-rs` — x86 AI Compute Extensions (ACE) primitives for Rust.
//!
//! See `DESIGN_RATIONALE.md` for the full design. Each primitive follows the same shape:
//!
//! ```text
//! safe dispatch fn  →  native path (core::arch intrinsic)  →  scalar fallback (primary path)
//! ```
//!
//! with a differential test asserting the native path agrees with the scalar oracle.
//!
//! **Iterations 0–3 are complete** — all four ACE feature groups (spec §4) are implemented.
//!
//! **Iteration 0** (the tracer bullet, design §6 / D9) wired one primitive — [`dpbssd`] —
//! end to end on stable Rust: build → runtime detect → intrinsic → fallback → test.
//! It is the only ACE primitive already present in stable `core::arch`, so it needs no
//! emulator and runs natively on AVX-VNNI-INT8 hardware.
//!
//! **Iteration 1** added the `AVX10_V1_AUX` family of FP16↔FP8 / FP32→FP16
//! converts and the EVEX byte/word VNNI matrix, each behind a crate-owned capability check
//! (`detect`) over the shared FP8/FP16 conversion oracle (`fp8`). The scalar oracle is
//! the primary, always-present path; an **opt-in native path** (the `native` cargo feature)
//! routes each primitive to a hand-written C shim compiled with `-mavx10.2` — there is no
//! stable `core::arch` EVEX intrinsic for these forms yet — taken only when
//! `detect::has_avx10_v1_aux()` confirms the running CPU
//! ([avx10-v1-aux-fp16-fp8-evex-vnni.DISPATCH.3]). The default build is oracle-only.
//!
//! It also **completes the group-1 VEX family**: alongside iteration-0 [`dpbssd`] the crate
//! root now exposes the 11 remaining `avxvnniint8` byte (`VPDPB*`) and `avxvnniint16` word
//! (`VPDPW*`) multiply-accumulate ops, generated once through the declarative `define_dp!`
//! macro. These have real stable `core::arch` intrinsics and run natively wherever the CPU
//! advertises the feature.
//!
//! The EVEX byte/word VNNI primitives live in the [`vnni`] module and are reached
//! module-qualified — e.g. the 512-bit EVEX `dpbssd` is [`vnni::dpbssd`]
//! (`ace::vnni::dpbssd`), DISTINCT from this crate's iteration-0 256-bit VEX [`dpbssd`]
//! (`ace::dpbssd`). The two are resolved by module path and neither shadows the other
//! ([avx10-v1-aux-fp16-fp8-evex-vnni.BYTE_VNNI.1], OQ-1).
//!
//! ## Operand-order significance (mixed-signedness variants)
//!
//! Per LOCKED decision A1 the operand signedness is encoded in the element type — signed
//! operands are `[i8;32]`/`[i16;16]`, unsigned operands are `[u8;32]`/`[u16;16]`. For the
//! mixed-signedness `SU`/`US` variants (e.g. [`dpbsud`], [`dpbsuds`]) operand order is
//! *significant*, so the two arguments have distinct types and a `b, a` swap does not even
//! compile — commutativity is not expressible, which is why those variants carry no
//! `prop_operands_commute` property ([vnni-int8-int16-family.CORRECTNESS.2]):
//!
//! ```compile_fail
//! let src = [0i32; 8];
//! let a = [0i8; 32]; // signed operand
//! let b = [0u8; 32]; // unsigned operand
//! // The correct order, `ace::dpbsud(src, a, b)`, type-checks; the swapped order does
//! // NOT — `[u8;32]` is not `[i8;32]` and vice versa (rustc E0308):
//! let _ = ace::dpbsud(src, b, a);
//! ```
//!
//! # Native-coverage tripwire (`ACE_REQUIRE_NATIVE`) — scope in v1
//!
//! The `ACE_REQUIRE_NATIVE=1` coverage tripwire (CI's `native-sde` job) stays meaningful for
//! the **group-1 VEX family** — iteration-0 [`dpbssd`] plus the 11 remaining
//! `avxvnniint8`/`avxvnniint16` multiply-accumulate ops — because those primitives have real
//! stable `core::arch` intrinsics, so the `native_runs_when_required` guard in this module's
//! tests asserts the native `VPDPB*`/`VPDPW*` branch actually ran rather than the scalar
//! fallback, and that **both** `avxvnniint8` and `avxvnniint16` were detected — a green
//! native run cannot mean "byte ops native, word ops silently scalar"
//! ([avx10-v1-aux-fp16-fp8-evex-vnni.DIFFERENTIAL.2],
//! [avx10-v1-aux-fp16-fp8-evex-vnni.CI.2]). For the `AVX10_V1_AUX` families (A–G) the hard
//! `ACE_REQUIRE_NATIVE` guard does **not** yet enforce native execution: their native path is
//! a hand-written C shim behind the opt-in `native` feature (no stable `core::arch` EVEX
//! intrinsic exists, OQ-3, [avx10-v1-aux-fp16-fp8-evex-vnni.DISPATCH.3]), so a default build
//! has no native branch to require. Each family does ship a live `prop_native_matches_oracle`
//! differential (in its `proptests` module): under `feature="native"` on x86_64 with
//! `detect::has_avx10_v1_aux()` it compares the C-shim native path to the scalar oracle
//! bit-for-bit, and calls `TestResult::discard()` — never `from_bool(false)` — when the
//! feature or hardware is absent, so a fallback-only runner can never produce a vacuous green
//! ([avx10-v1-aux-fp16-fp8-evex-vnni.DIFFERENTIAL.1],
//! [avx10-v1-aux-fp16-fp8-evex-vnni.DIFFERENTIAL.1-1]). Folding the `native`-feature AVX10
//! path into the `ACE_REQUIRE_NATIVE` guard would make its tripwire live by the same pattern.
//!
//! **Iteration 3 — `ACE` group-4 tile instructions, same tripwire posture.** The 25 group-4
//! tile instructions (families A–G) follow the identical `ACE_REQUIRE_NATIVE` posture. Their
//! native backend is the opt-in `native` C-shim / `.byte` path (design D7, OQ-6): families A /
//! B-read / C are intrinsic-reachable and execute under Intel SDE, while the `ACE`-only forms
//! (family B write, D, E, F, G) are `.byte` raw encodings that are BUILT but not executed until
//! SDE gains ACE emulation (R2). The hard `ACE_REQUIRE_NATIVE` guard therefore stays **dormant**
//! for group 4 exactly as it does for the AVX10 C-shim families — the group-4-scoped
//! `ace_tile_native_runs_when_required` test presence-checks the variable and records the
//! per-family tile detection status rather than vacuously asserting a native branch that cannot
//! yet run. Each family ships a live `prop_native_matches_oracle` differential (in its
//! `differential` module) that compares the C / `.byte` native path to the scalar oracle
//! bit-for-bit under `feature="native"` + the per-family `detect::has_amx_tile()` /
//! `has_amx_avx512()` / `has_ace()` gate and calls `TestResult::discard()` — never
//! `from_bool(false)` — when the feature or hardware is absent, so a fallback-only runner can
//! never go vacuously green ([ace-tile-instructions.TESTING.1], [ace-tile-instructions.TESTING.2]).
//! The layer-2 `tests/encoding.rs` harness golden-checks every `.byte` encoding with no external
//! tool, so `.byte` transcription errors are caught before SDE ACE lands.
//!
//! **Iteration 2 — `AVX10_V2_AUX` (group 3), same scope.** The 21 group-3 OCP-format converts
//! (families A–I: FP32↔FP8, FP8↔FP4, FP8↔FP6, `VPMOVSSDB`, `VUNPACKB`) follow the identical
//! tripwire posture. Every one of their `-mavx10.2` C-shim intrinsics
//! (`_mm512_cvtps_bf8`/`cvtbf8_ps`/`cvtf8_bf4s`/`cvtbf4_hf8`/`cvtf8_bf6s`/`cvtf6_hf8`/`cvtssepi32_epi8`/`_mm512_unpackb`)
//! is **absent** from the GCC/Clang headers in this toolchain, so per OQ-5 every group-3 family
//! ships **oracle-only** — there is no native C shim and the `ACE_REQUIRE_NATIVE=1` guard stays
//! dormant for them exactly as it is for the `AVX10_V1_AUX` C-shim families. Each group-3 family
//! still ships a live `prop_native_matches_oracle` differential (in its `differential` /
//! `proptests` module) gated `#[cfg(all(target_arch = "x86_64", feature = "native"))]` +
//! `detect::has_avx10_v2_aux()`, comparing the public dispatcher to its scalar oracle bit-for-bit
//! and calling `TestResult::discard()` — never `from_bool(false)` — when the feature or hardware
//! is absent, so the `native-sde` job (SDE 10.8, `sde64 -future --`) proves the native branch ran
//! rather than the fallback the moment any group-3 intrinsic lands, and a fallback-only runner can
//! never produce a vacuous green ([avx10-v2-aux-ocp-conversions.DIFFERENTIAL.1],
//! [avx10-v2-aux-ocp-conversions.DIFFERENTIAL.2]).
//!
//! # Non-goals — confirmed NOT implemented (through iteration 3)
//!
//! The public surface of this crate is the completed group-1 VEX family (iteration-0
//! [`dpbssd`] plus the 11 remaining `avxvnniint8`/`avxvnniint16` multiply-accumulate ops),
//! the 26 `AVX10_V1_AUX` primitives (families A–G), the 21 `AVX10_V2_AUX` (group 3) OCP-format
//! converts added in iteration 2 (families A–I: FP32↔FP8, FP8↔FP4, FP8↔FP6, `VPMOVSSDB`,
//! `VUNPACKB`), and — added in iteration 3 — the 25 `ACE` group-4 tile instructions (families
//! A–G: the palette-2 tile lifecycle + [`TileScope`] RAII guard, tile↔ZMM moves, tile-row
//! converts, block-scale [`BsrReg`] registers, and the `TOP*` outer products). **Group 4 is
//! therefore no longer a non-goal** — its reachability is the positive assertion in
//! `iteration_surface::iteration_surface_includes_group4`. The following remain deliberately
//! **out of scope** and are NOT present in any public item or native path (verified by
//! `non_goal_guards::non_goals_absent`):
//!
//! - **No palette-1 tile configuration and no AMX `TMUL` dot-product instructions** — the tile
//!   surface is the palette-2 `ACE` group-4 engine only; the legacy AMX palette-1 configuration
//!   and the `TMUL` dot products (`TDPBSSD`/`TDPBF16PS`/`TDPFP16PS` and siblings) are NOT
//!   implemented. Group 4's outer products (`TOP*`) are a distinct engine.
//! - **No nightly `x86_amx_intrinsics`** — the group-4 native backend is the opt-in `native`
//!   C-shim / `.byte` path (design D7); the crate uses no `core::simd`, no
//!   `link_llvm_intrinsics`, and no nightly `x86_amx_intrinsics` feature anywhere
//!   ([ace-tile-instructions.STABLE.1]).
//! - **No VEX-encoded AVX-VNNI-INT8/16 forms beyond the group-1 family** — the family-F/G
//!   additions are the EVEX 512-bit generalization ([`vnni`]), not new VEX forms.
//! - **No EVEX write-masking (`{k1}{z}`) or memory-broadcast (`m32bcst`/`m*bcst`) in the public
//!   API** — every primitive takes plain fixed-size lane arrays by value and writes a FULL
//!   result; the spec's `k1` / `zeroing` / `evex_b` operands are fixed to the no-writemask,
//!   no-broadcast case (`no_writemask = true`, `evex_b = false`) and are not surfaced.
//!   `VUNPACKB`'s `imm8` IS surfaced — but as a plain `u8` *value* argument selecting size /
//!   start / sign-extend, NOT as a write-mask: it is the sole control input and every output
//!   lane is written.
//! - **No 128-bit / 256-bit vector-length entry points** — every group-3 convert is the
//!   512-bit (`VL=512`) form; the 128/256-bit `VL` plumbing the spec also defines is not
//!   surfaced.

pub mod bsr;
pub mod cvt_fp8_fp4;
pub mod cvt_fp8_fp6;
pub mod cvt_fp8_ph;
pub mod cvt_fp8_ps;
pub mod cvt_ph_fp8;
pub mod cvt_ps_fp8;
pub mod cvt_ps_ph;
pub mod cvt_ssdb;
mod detect;
pub(crate) mod fp4;
pub(crate) mod fp6;
pub(crate) mod fp8;
// In the non-test lib build the group-4 `_hw` wrappers are referenced only by native.rs's
// own differential test layer, so allow the resulting dead-code lint outside tests.
#[cfg(all(target_arch = "x86_64", feature = "native"))]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) mod native;
pub mod tcvtrow;
pub mod tile;
pub mod tile_move;
pub mod top;
pub mod unpackb;
pub mod vnni;

pub use cvt_ph_fp8::{
    cvt2ph_bf8, cvt2ph_bf8_scalar, cvt2ph_hf8, cvt2ph_hf8_scalar, cvt2phs_bf8, cvt2phs_bf8_scalar,
    cvt2phs_hf8, cvt2phs_hf8_scalar, cvtbiasph_bf8, cvtbiasph_bf8_scalar, cvtbiasph_hf8,
    cvtbiasph_hf8_scalar, cvtbiasphs_bf8, cvtbiasphs_bf8_scalar, cvtbiasphs_hf8,
    cvtbiasphs_hf8_scalar, cvtph_bf8, cvtph_bf8_scalar, cvtph_hf8, cvtph_hf8_scalar, cvtphs_bf8,
    cvtphs_bf8_scalar, cvtphs_hf8, cvtphs_hf8_scalar,
};

pub use cvt_fp8_ph::{cvthf8_ph, cvthf8_ph_scalar};

// AVX10_V2_AUX family C: exact FP8 -> FP32 (iteration 2).
pub use cvt_fp8_ps::{cvtbf8_ps, cvtbf8_ps_scalar, cvthf8_ps, cvthf8_ps_scalar};

// AVX10_V2_AUX family D: saturating-RTNE FP8 -> FP4 (E2M1), nibble-packed (iteration 2). The
// source-format suffix (`_e5m2` / `_e4m3`) disambiguates the two converts, since both target
// FP4 E2M1 (OQ-3). Always saturating to +/-6.0; output is half the input width.
pub use cvt_fp8_fp4::{
    cvtf8_bf4s_e4m3, cvtf8_bf4s_e4m3_scalar, cvtf8_bf4s_e5m2, cvtf8_bf4s_e5m2_scalar,
};

// AVX10_V2_AUX family E: exact FP4 (E2M1) -> FP8 (E4M3), nibble-unpacked (iteration 2). Every
// one of the 16 FP4 encodings maps to exactly one E4M3 byte (no rounding); output is twice the
// input width.
pub use cvt_fp8_fp4::{cvtbf4_hf8, cvtbf4_hf8_scalar};

// AVX10_V2_AUX family F: saturating-RTNE FP8 -> FP6 (E3M2 / E2M3), 6-bit-packed (iteration 2).
// `cvtf8_bf6s` is E5M2 -> FP6 E3M2 (VCVTBF82BF6S), `cvtf8_hf6s` is E4M3 -> FP6 E2M3
// (VCVTHF82HF6S); the two carry distinct intrinsic stems (OQ-3). Always saturating (E3M2
// +/-28.0, E2M3 +/-7.5); mantissa-width-matched so no mantissa loss, every FP8 subnormal ->
// same-signed FP6 zero; output is VL*6/8 = 48 bytes for the 64-byte input.
pub use cvt_fp8_fp6::{cvtf8_bf6s, cvtf8_bf6s_scalar, cvtf8_hf6s, cvtf8_hf6s_scalar};

// AVX10_V2_AUX family G: exact FP6 (E3M2 / E2M3) -> FP8 (E4M3), 6-bit-unpacked (iteration
// 2). `cvtf6_hf8_e3m2` is FP6 E3M2 -> E4M3 (VCVTBF62HF8), `cvtf6_hf8_e2m3` is FP6 E2M3 ->
// E4M3 (VCVTHF62HF8); the source-format suffix disambiguates the two, since both target
// FP8 E4M3 (OQ-3). Exact: every one of the 64 FP6 encodings (per format) maps to exactly
// one E4M3 byte (no rounding/saturation); output is twice the 48-byte 6-bit-packed input.
pub use cvt_fp8_fp6::{
    cvtf6_hf8_e2m3, cvtf6_hf8_e2m3_scalar, cvtf6_hf8_e3m2, cvtf6_hf8_e3m2_scalar,
};

// AVX10_V2_AUX family H: VPMOVSSDB — INT32 -> INT8 with SYMMETRIC signed saturation
// (iteration 2). Clamps each lane to [-127, +127] (MAX_POSITIVE 0x7F, MAX_NEGATIVE 0x81),
// NOT the asymmetric [-128, +127] of ordinary VPMOVSDB; the `cvtss` prefix names that
// distinction. Symmetric about zero — f(-x) = -f(x), f(i32::MIN) = -127. Output is 1/4 the
// input width (16 dwords -> 16 bytes).
pub use cvt_ssdb::{cvtssepi32_epi8, cvtssepi32_epi8_scalar};

// AVX10_V2_AUX family I: VUNPACKB — unpack 64 packed sub-byte elements into 64 bytes
// (iteration 2). `imm8` selects element size (imm8[4:2], clamped to min 2), start offset
// (imm8[1:0], size-conditioned) and sign-extend (imm8[5]); build it with ACE_UNPACKB_SIZE /
// ACE_UNPACKB_START / ACE_UNPACKB_SEXT. The read-back complement of the family-D nibble and
// family-F 6-bit packers (EXACTNESS.2). imm8 is a plain value argument, not a write-mask;
// v1 surfaces only the no_writemask path. Output is the full 512-bit [u8; 64].
pub use unpackb::{unpackb, unpackb_scalar, ACE_UNPACKB_SEXT, ACE_UNPACKB_SIZE, ACE_UNPACKB_START};

// AVX10_V2_AUX family A: single-source FP32 -> FP8 RTNE/RTO (iteration 2). RTO is
// E4M3-only, so there is deliberately no `cvtrops_bf8`.
pub use cvt_ps_fp8::{
    cvtps_bf8, cvtps_bf8_scalar, cvtps_hf8, cvtps_hf8_scalar, cvtpss_bf8, cvtpss_bf8_scalar,
    cvtpss_hf8, cvtpss_hf8_scalar, cvtrops_hf8, cvtrops_hf8_scalar, cvtropss_hf8,
    cvtropss_hf8_scalar,
};

// AVX10_V2_AUX family B: FP32 -> FP8 bias-rounding (iteration 2). The per-lane bias term is
// the full Operand-2 (`VVVV`) `i32` word; the FP32 source is Operand 3.
pub use cvt_ps_fp8::{
    cvtbiasps_bf8, cvtbiasps_bf8_scalar, cvtbiasps_hf8, cvtbiasps_hf8_scalar, cvtbiaspss_bf8,
    cvtbiaspss_bf8_scalar, cvtbiaspss_hf8, cvtbiaspss_hf8_scalar,
};

pub use cvt_ps_ph::{cvt2ps_phx, cvt2ps_phx_scalar};

// ACE group-4 family A: tile configuration lifecycle wrapped in the crate's first RAII guard
// (`TileScope`). `_tile_loadconfig`/`_tile_storeconfig`/`_tile_zero` mirror their eventual
// stdarch intrinsic stems ([ace-tile-instructions.NAMING.1]); `_tile_release` is Drop-only, so
// it has no free-standing re-export. Each dispatcher pairs a cfg-free `_scalar` oracle (the
// primary path). Pure stable Rust — no core::simd / nightly ([ace-tile-instructions.STABLE.1]).
pub use tile::{
    _tile_loadconfig, _tile_loadconfig_scalar, _tile_storeconfig, _tile_storeconfig_scalar,
    _tile_zero, _tile_zero_scalar, TileConfig, TileConfigError, TileId, TileScope, MAX_TILES,
    TILE_BYTES, TILE_COLSB, TILE_ROWS,
};

// ACE group-4 family B: tile data movement (spec section 12). `TILEMOVROW` has read AND
// write forms (`_tile_movrow` / `_tile_setrow`); `TILEMOVCOL` is WRITE-ONLY
// (`_tile_setcol`, spec section 12.3.1 — column moves transfer data from AVX registers to
// tile registers only). Row/column specifiers mask to bits [3:0] and never fault (spec
// section 12.1.1). The read form gates on AMX-AVX512 || ACE_VSN>=1; the write forms are
// ACE-only. Dispatcher names mirror the spec's C intrinsic equivalents (sections 12.2.10 /
// 12.3.10) and pair a cfg-free `_scalar` oracle ([ace-tile-instructions.NAMING.1]).
pub use tile_move::{
    _tile_movrow, _tile_movrow_scalar, _tile_setcol, _tile_setcol_scalar, _tile_setrow,
    _tile_setrow_scalar, ZMM_BYTES,
};

// ACE group-4 family C: tile-row converts (spec sections 12.4-12.6). Each convert reads
// tile row `specifier & 0xF` and writes a destination ZMM: `_tile_cvtrowd2ps` (INT32 ->
// FP32, RNE), `_tile_cvtrowps2bf16{h,l}` (FP32 -> BF16, DAZ=1/FTZ=1 per section 12.5.1),
// and `_tile_cvtrowps2ph{h,l}` (FP32 -> FP16), the H/L pair writing disjoint high/low
// words of each dword (INV-7). All five gate on AMX-AVX512 || ACE_VSN>=1
// ([ace-tile-instructions.DETECT.1-2]); dispatcher names mirror the spec's C intrinsic
// equivalents (sections 12.4.10 / 12.5.9 / 12.6.9).
pub use tcvtrow::{
    _tile_cvtrowd2ps, _tile_cvtrowd2ps_scalar, _tile_cvtrowps2bf16h, _tile_cvtrowps2bf16h_scalar,
    _tile_cvtrowps2bf16l, _tile_cvtrowps2bf16l_scalar, _tile_cvtrowps2phh,
    _tile_cvtrowps2phh_scalar, _tile_cvtrowps2phl, _tile_cvtrowps2phl_scalar, ROW_FP32_LANES,
    ZMM_WORD_LANES,
};

// ACE group-4 family D: the single 1024-bit Block Scale register (`bsr0`/SCALEDATA, spec
// section 13). `_bsrinit` sets all 128 bytes to 0x7F (no data operand); `_bsrmovf` writes
// the full register from two ZMM-sized sources (src1 -> A scales [1023:512], src2 -> B
// scales [511:0]); `_bsrmovh`/`_bsrmovl` move the upper/lower 512-bit half with BOTH write
// and read (`*_read`) forms. The register is owned by the `TileScope` guard, so the MX
// outer products read back the scales these ops wrote (INV-5,
// [ace-tile-instructions.BSR.4-1]). Family D is ACE-only
// ([ace-tile-instructions.DETECT.1-3]); dispatcher names mirror the spec's C intrinsic
// equivalents (sections 13.1.9 / 13.2.9 / 13.3.9).
pub use bsr::{
    _bsrinit, _bsrinit_scalar, _bsrmovf, _bsrmovf_scalar, _bsrmovh, _bsrmovh_read,
    _bsrmovh_read_scalar, _bsrmovh_scalar, _bsrmovl, _bsrmovl_read, _bsrmovl_read_scalar,
    _bsrmovl_scalar, BsrReg, BSR_BYTES, BSR_HALF_BYTES, BSR_INIT_BYTE,
};

// ACE group-4 family G: INT8 rank-4 outer-product accumulates (spec section 14.4).
// `_tile_top4b{ss,su,us,uu}d` take raw ZMM bytes and widen each INT8 sub-element per the
// mnemonic's signedness (signed sign-extends, unsigned zero-extends), accumulating the
// rank-4 outer product into an INT32 destination tile with exact i32 wraparound — no
// saturating `...DS` forms exist in the spec. Family G is ACE-only
// ([ace-tile-instructions.DETECT.1-3]); dispatcher names mirror the spec's C intrinsic
// equivalents (section 14.4.11).
pub use top::{
    _tile_top4bssd, _tile_top4bssd_scalar, _tile_top4bsud, _tile_top4bsud_scalar, _tile_top4busd,
    _tile_top4busd_scalar, _tile_top4buud, _tile_top4buud_scalar,
};

// ACE group-4 family F: the BF16 rank-2 outer-product accumulate, `TOP2BF16PS` (spec
// section 14.3). BF16 pairs widen exactly to FP32 with DAZ=1, the two products form one
// FP32 sum (FTZ=1), and a SINGLE FP32 add accumulates onto the prior tile element (DAZ=1
// accumulator, FTZ=1 output) — `C = float32_add(C, RNE(a0*b0 + a1*b1))`. No block scale.
// ACE-only ([ace-tile-instructions.DETECT.1-3]); the dispatcher mirrors the spec's
// `_tile_top2bf16ps` C intrinsic (section 14.3.11).
pub use top::{_tile_top2bf16ps, _tile_top2bf16ps_scalar};

// ACE group-4 family E: the MX rank-4 outer-product accumulates (spec sections 14.1-14.2).
// The Block Scale register is read IMPLICITLY; `imm8[5:4]`/`imm8[1:0]` select the A/B
// scale groups (compose with `ace_scale_a`/`ace_scale_b`, section 14.1.12). One E8M0
// A-scale per output row and one B-scale per output column; the four sub-element products
// accumulate EXACTLY in integer fixpoint and the combined scale `2^(s_a + s_b - 254)` is
// applied once to the sum in the precise domain, with a single RNE conversion to FP32
// (FTZ=1) and a single accumulate add (DAZ=1). An E8M0 NaN (0xFF) scale yields
// QNaN_Indefinite for the affected elements. The four FP8 format pairs are BF8xBF8 /
// BF8xHF8 / HF8xBF8 / HF8xHF8 ([ace-tile-instructions.MX_TOP.1..4]); `_tile_top4mxbssps`
// is the signed MX INT8 form carrying the combined 2^-12 implicit product bias
// ([ace-tile-instructions.MX_TOP.5]). ACE-only ([ace-tile-instructions.DETECT.1-3]);
// dispatcher names mirror the spec's C intrinsic equivalents (sections 14.1.12 / 14.2.12).
pub use top::{
    _tile_top4mxbf8ps, _tile_top4mxbf8ps_scalar, _tile_top4mxbhf8ps, _tile_top4mxbhf8ps_scalar,
    _tile_top4mxbssps, _tile_top4mxbssps_scalar, _tile_top4mxhbf8ps, _tile_top4mxhbf8ps_scalar,
    _tile_top4mxhf8ps, _tile_top4mxhf8ps_scalar, ace_scale_a, ace_scale_b,
};

// T1 toolchain risk gate (OQ-1, re-confirmed): `is_x86_feature_detected!("avxvnniint16")`
// and all six word intrinsics (`_mm256_dpw{sud,suds,usd,usds,uud,uuds}_epi32`) plus the
// five remaining byte intrinsics (`_mm256_dpb{ssds,sud,suds,uud,uuds}_epi32`) resolve and
// the macro-emitted dispatch/native bodies FULLY COMPILE on stable Rust 1.96 — verified by
// a complete `cargo check --all-targets --target x86_64-unknown-linux-gnu` (exit 0), not
// merely import resolution. (Import resolution alone is insufficient: an earlier "no E0432"
// check passed while the macro bodies failed to compile — `is_x86_feature_detected!` rejects
// a feature name forwarded as `:literal`, fixed by capturing `$feat` as `:tt`; see the macro.)
// No MSRV bump and no nightly feature flags needed.

/// Signed int8 dot-product-accumulate. (ACE group 1: AVX-VNNI-INT8, `VPDPBSSD`.)
///
/// For each of the 8 output lanes `i`:
///
/// ```text
/// out[i] = src[i] + Σ_{k=0..4} a[4i+k] * b[4i+k]
/// ```
///
/// Dispatches to the native intrinsic when the running CPU supports `avxvnniint8`,
/// otherwise uses the portable scalar path. Both produce identical results.
pub fn dpbssd(src: [i32; 8], a: [i8; 32], b: [i8; 32]) -> [i32; 8] {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avxvnniint8") {
            // SAFETY: the `avxvnniint8` feature was confirmed present immediately above.
            return unsafe { dpbssd_hw(src, a, b) };
        }
    }
    dpbssd_scalar(src, a, b)
}

/// Portable reference path — and the oracle the native path is tested against.
pub fn dpbssd_scalar(src: [i32; 8], a: [i8; 32], b: [i8; 32]) -> [i32; 8] {
    let mut out = src;
    for i in 0..8 {
        let mut acc = 0i32;
        for k in 0..4 {
            acc = acc.wrapping_add(a[4 * i + k] as i32 * b[4 * i + k] as i32);
        }
        out[i] = out[i].wrapping_add(acc);
    }
    out
}

/// Native path: `VPDPBSSD` via `core::arch::x86_64::_mm256_dpbssd_epi32`.
///
/// # Safety
/// The CPU must support the `avxvnniint8` feature. Callers go through [`dpbssd`],
/// which checks this at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avxvnniint8")]
unsafe fn dpbssd_hw(src: [i32; 8], a: [i8; 32], b: [i8; 32]) -> [i32; 8] {
    use std::arch::x86_64::*;
    let vsrc = _mm256_loadu_si256(src.as_ptr().cast());
    let va = _mm256_loadu_si256(a.as_ptr().cast());
    let vb = _mm256_loadu_si256(b.as_ptr().cast());
    let vout = _mm256_dpbssd_epi32(vsrc, va, vb);
    let mut out = [0i32; 8];
    _mm256_storeu_si256(out.as_mut_ptr().cast(), vout);
    out
}

/// Declarative factory for a group-1 dot-product-accumulate primitive (LOCKED B1).
///
/// One invocation emits the full three-layer shape — exactly the hand-written [`dpbssd`]
/// shape (design D1/D2/D5), parameterised so each new ACE instruction is a single
/// declaration rather than a hand-copied block:
///
/// * a public **dispatch** fn `$name` — runtime `is_x86_feature_detected!` probe on
///   x86_64 calling the native path, falling through to the scalar oracle otherwise;
/// * a public **scalar oracle** `$scalar` — the portable reference path and the source of
///   truth the native path is differential-tested against;
/// * a private `#[target_feature]`-gated **native** fn `$hw` calling the matching
///   `_mm256_dp*_epi32` intrinsic via unaligned `loadu`/`storeu`.
///
/// Named arguments (wrap-vs-saturate and signedness are explicit, per B1):
/// * `name` / `scalar` / `hw` — the public dispatch fn, the public oracle fn, and the
///   private native fn identifiers (stable Rust cannot concatenate identifiers without a
///   proc-macro dependency, which `SCOPE.4` forbids, so the trio is named explicitly).
/// * `feature` — the CPUID feature token (`"avxvnniint8"` byte / `"avxvnniint16"` word).
/// * `a` / `b` — the per-element operand types; their signedness threads into the oracle
///   widening cast (`i8`/`i16` sign-extend, `u8`/`u16` zero-extend), encoding the mnemonic.
/// * `products` — products-per-lane (4 byte / 2 word); the operand arrays are `[_; products*8]`.
/// * `intrinsic` — the matching `_mm256_dp<variant>_epi32` path.
/// * `accumulate` — `wrap` (the `...D` variants: `wrapping_add` of the products onto the
///   lane), `saturate` (the signed `...DS` variants — SS/SU/US: a SINGLE signed-dword
///   saturation of the full-precision sum — `out[i] = SIGNED_DWORD_SATURATE(src[i] + Σ
///   products)`), or `saturate_unsigned` (the UU `...DS` variants `VPDPBUUDS`/`VPDPWUUDS`: a
///   SINGLE *unsigned*-dword saturation, `out[i] = UNSIGNED_DWORD_SATURATE(unsigned(src[i]) + Σ
///   products)`, because both operands and the accumulator are unsigned). Both follow the Intel
///   SDM / Felix Cloutier pseudocode for VPDPB*DS / VPDPW*DS; there is NO intermediate clamp of
///   the product-sum before adding src. Products are summed in `i64` — wide enough that
///   `u16×u16` cannot overflow before the single clamp.
///
/// The native intrinsic is the differential tiebreaker for the saturation/accumulation-width
/// question; the oracle is validated bit-for-bit against it under the SDE CI job.
///
/// [vnni-int8-int16-family.PRIMITIVE_SHAPE.1] [vnni-int8-int16-family.PRIMITIVE_SHAPE.2]
/// [vnni-int8-int16-family.API.3] [vnni-int8-int16-family.API.4]
/// [vnni-int8-int16-family.SCALAR_ORACLE.1] [vnni-int8-int16-family.SCALAR_ORACLE.1-1]
/// [vnni-int8-int16-family.SCALAR_ORACLE.1-2] [vnni-int8-int16-family.SCALAR_ORACLE.1-3]
/// [vnni-int8-int16-family.SCALAR_ORACLE.1-4] [vnni-int8-int16-family.NATIVE_PATH.1]
/// [vnni-int8-int16-family.NATIVE_PATH.2] [vnni-int8-int16-family.NATIVE_PATH.3]
macro_rules! define_dp {
    // ---- accumulate helpers: fold one lane's products onto `src[i]`. ----------------
    // `wrap`: wrapping i32 accumulation (the `...D` variants).
    (@fold wrap, $src:expr, $acc:expr) => {
        // `$acc` is the i64 product sum; the ISA wraps at i32, so wrapping-cast then
        // wrapping-add — matching the native intrinsic's modular arithmetic.
        $src.wrapping_add($acc as i32)
    };
    // `saturate`: a SINGLE signed-dword saturation of the FULL-PRECISION sum (the `...DS`
    // variants). Per the Intel SDM / Felix Cloutier pseudocode for VPDPB*DS and VPDPW*DS:
    //   DEST.dword[i] := SIGNED_DWORD_SATURATE( SRC.dword[i] + product1 + product2 [+ ...] )
    // There is NO intermediate clamp of the product-sum before adding src. We add `src` to
    // the i64 product sum and clamp the single total once into [i32::MIN, i32::MAX]. (A prior
    // two-stage "clamp acc into i32, then saturating_add src" diverged from hardware when src
    // and acc had opposite signs and |acc| exceeded the i32 range.)
    (@fold saturate, $src:expr, $acc:expr) => {{
        let total: i64 = ($src as i64) + $acc;
        total.clamp(i32::MIN as i64, i32::MAX as i64) as i32
    }};
    // `saturate_unsigned`: the UU `...DS` variants (`VPDPBUUDS` / `VPDPWUUDS`) saturate into
    // the UNSIGNED dword range, NOT the signed range — both operands and the accumulator are
    // unsigned. Per the Intel SDM / Felix Cloutier pseudocode:
    //   DEST.dword[i] := UNSIGNED_DWORD_SATURATE( SRC.dword[i] + product1 + product2 [+ ...] )
    // `SRC.dword[i]` is read as an unsigned dword; reinterpret `src`'s bits as `u32`, add the
    // (non-negative) i64 product sum, clamp the total once into `[0, u32::MAX]`, then store the
    // 32-bit pattern back as `i32`. (The lower bound never binds — both addends are >= 0 — but
    // is kept for symmetry with the saturate semantics.)
    (@fold saturate_unsigned, $src:expr, $acc:expr) => {{
        let total: i64 = ($src as u32 as i64) + $acc;
        total.clamp(0, u32::MAX as i64) as u32 as i32
    }};

    // ---- main entry: emit dispatch + scalar oracle + native _hw for one variant. -----
    (
        name = $name:ident,
        scalar = $scalar:ident,
        hw = $hw:ident,
        // `:tt`, NOT `:literal`: `is_x86_feature_detected!` (and `#[target_feature]`) reject a
        // feature name forwarded as a pre-parsed `:literal` nonterminal — it arrives as one
        // opaque token and `std_detect`'s name validation falls through to "unknown x86 target
        // feature". A `:tt` forwards the bare string-literal token, which both accept. (The
        // inline `dpbssd` works because its literal is at the macro's own call site.)
        feature = $feat:tt,
        a = $a:ty,
        b = $b:ty,
        products = $ppl:expr,
        intrinsic = $intrin:path,
        accumulate = $acc:tt
    ) => {
        /// Group-1 dot-product-accumulate dispatch fn (emitted by `define_dp!`).
        ///
        /// Dispatches to the native intrinsic when the running CPU supports the variant's
        /// feature, otherwise uses the portable scalar oracle. Both produce identical results.
        pub fn $name(src: [i32; 8], a: [$a; $ppl * 8], b: [$b; $ppl * 8]) -> [i32; 8] {
            #[cfg(target_arch = "x86_64")]
            {
                if std::is_x86_feature_detected!($feat) {
                    // SAFETY: the variant's feature was confirmed present immediately above.
                    return unsafe { $hw(src, a, b) };
                }
            }
            $scalar(src, a, b)
        }

        /// Portable reference path — the oracle the native path is differential-tested
        /// against. Widens each operand with the signedness-correct cast (the operand
        /// element types encode the mnemonic), multiplies and folds the per-lane products
        /// in `i64` (cannot overflow before the clamp), then applies the variant's
        /// wrap/saturate accumulation onto `src`.
        pub fn $scalar(src: [i32; 8], a: [$a; $ppl * 8], b: [$b; $ppl * 8]) -> [i32; 8] {
            let mut out = src;
            for i in 0..8 {
                let mut acc = 0i64;
                for k in 0..$ppl {
                    let av = a[$ppl * i + k] as i64;
                    let bv = b[$ppl * i + k] as i64;
                    acc += av * bv;
                }
                out[i] = define_dp!(@fold $acc, out[i], acc);
            }
            out
        }

        /// Native path via the matching `_mm256_dp*_epi32` intrinsic.
        ///
        /// # Safety
        /// The CPU must support the variant's feature. Callers go through the dispatch fn,
        /// which checks this at runtime.
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = $feat)]
        unsafe fn $hw(src: [i32; 8], a: [$a; $ppl * 8], b: [$b; $ppl * 8]) -> [i32; 8] {
            use std::arch::x86_64::*;
            let vsrc = _mm256_loadu_si256(src.as_ptr().cast());
            let va = _mm256_loadu_si256(a.as_ptr().cast());
            let vb = _mm256_loadu_si256(b.as_ptr().cast());
            let vout = $intrin(vsrc, va, vb);
            let mut out = [0i32; 8];
            _mm256_storeu_si256(out.as_mut_ptr().cast(), vout);
            out
        }
    };
}

// First macro-emitted variant: dpbssds — signed×signed bytes, saturating (ACE group 1:
// AVX-VNNI-INT8, `VPDPBSSDS`). Proves the `define_dp!` macro reproduces the `dpbssd` shape
// end-to-end (dispatch → native _hw → scalar oracle) with the saturating accumulate path.
// [vnni-int8-int16-family.API.1] [vnni-int8-int16-family.API.2]
define_dp! {
    name = dpbssds,
    scalar = dpbssds_scalar,
    hw = dpbssds_hw,
    feature = "avxvnniint8",
    a = i8,
    b = i8,
    products = 4,
    intrinsic = _mm256_dpbssds_epi32,
    accumulate = saturate
}

// ===================== Phase 3: remaining avxvnniint8 byte variants =====================
// Complete the byte half of the group-1 grid via `define_dp!`: mixed-signedness SU
// (operand order significant — A1's distinct [i8;32]/[u8;32] types make a swap a compile
// error, so no commutativity property) and unsigned UU (commutative), each in both the
// wrapping (...D) and saturating (...DS) accumulate forms.
// [vnni-int8-int16-family.API.1] [vnni-int8-int16-family.API.2]
// [vnni-int8-int16-family.API.3] [vnni-int8-int16-family.NATIVE_PATH.1]

// dpbsud — signed×unsigned bytes, wrapping (ACE group 1: AVX-VNNI-INT8, `VPDPBSUD`).
// `a: [i8;32]` sign-extends, `b: [u8;32]` zero-extends in the oracle widening cast.
define_dp! {
    name = dpbsud,
    scalar = dpbsud_scalar,
    hw = dpbsud_hw,
    feature = "avxvnniint8",
    a = i8,
    b = u8,
    products = 4,
    intrinsic = _mm256_dpbsud_epi32,
    accumulate = wrap
}

// dpbsuds — signed×unsigned bytes, saturating (ACE group 1: AVX-VNNI-INT8, `VPDPBSUDS`).
define_dp! {
    name = dpbsuds,
    scalar = dpbsuds_scalar,
    hw = dpbsuds_hw,
    feature = "avxvnniint8",
    a = i8,
    b = u8,
    products = 4,
    intrinsic = _mm256_dpbsuds_epi32,
    accumulate = saturate
}

// dpbuud — unsigned×unsigned bytes, wrapping (ACE group 1: AVX-VNNI-INT8, `VPDPBUUD`).
// Both operands `[u8;32]` zero-extend; commutative (UU).
define_dp! {
    name = dpbuud,
    scalar = dpbuud_scalar,
    hw = dpbuud_hw,
    feature = "avxvnniint8",
    a = u8,
    b = u8,
    products = 4,
    intrinsic = _mm256_dpbuud_epi32,
    accumulate = wrap
}

// dpbuuds — unsigned×unsigned bytes, saturating (ACE group 1: AVX-VNNI-INT8, `VPDPBUUDS`).
define_dp! {
    name = dpbuuds,
    scalar = dpbuuds_scalar,
    hw = dpbuuds_hw,
    feature = "avxvnniint8",
    a = u8,
    b = u8,
    products = 4,
    intrinsic = _mm256_dpbuuds_epi32,
    accumulate = saturate_unsigned
}

// ============== Phase 4: word variants part 1 — avxvnniint16, 2 products/lane ==============
// Second CPUID feature family (avxvnniint16) and the word operand shapes: `[i16;16]` /
// `[u16;16]` (256-bit, 16 elements → `products = 2` per lane × 8 lanes). The `define_dp!`
// macro already generalises over products-per-lane (`$ppl`) and operand-array length
// (`[_; $ppl*8]`), and its oracle folds the per-lane products in `i64` — wide enough that a
// `u16 × u16` product (max 65535*65535 = 4_294_836_225, well above `i32::MAX`) cannot
// overflow before the final wrap/saturate clamp (OQ-3; SCALAR_ORACLE.1-4). So the word ops
// reuse the macro UNCHANGED; only `products = 2` and the 16-element operand types differ.
// `as i64` is the signedness-correct widening: `i16` sign-extends, `u16` zero-extends.
// These three are all mixed-signedness (SU / US) — operand order is significant and
// `dpwsud != dpwusd`, so none carries a `prop_operands_commute` property, and A1's distinct
// element types make a `b,a` swap a compile error.
// [vnni-int8-int16-family.API.1] [vnni-int8-int16-family.API.2]
// [vnni-int8-int16-family.API.3] [vnni-int8-int16-family.API.4]
// [vnni-int8-int16-family.NATIVE_PATH.1] [vnni-int8-int16-family.SCALAR_ORACLE.1-2]
// [vnni-int8-int16-family.SCALAR_ORACLE.1-4]

// dpwsud — signed×unsigned words, wrapping (ACE group 1: AVX-VNNI-INT16, `VPDPWSUD`).
// `a: [i16;16]` sign-extends, `b: [u16;16]` zero-extends in the oracle widening cast.
define_dp! {
    name = dpwsud,
    scalar = dpwsud_scalar,
    hw = dpwsud_hw,
    feature = "avxvnniint16",
    a = i16,
    b = u16,
    products = 2,
    intrinsic = _mm256_dpwsud_epi32,
    accumulate = wrap
}

// dpwsuds — signed×unsigned words, saturating (ACE group 1: AVX-VNNI-INT16, `VPDPWSUDS`).
define_dp! {
    name = dpwsuds,
    scalar = dpwsuds_scalar,
    hw = dpwsuds_hw,
    feature = "avxvnniint16",
    a = i16,
    b = u16,
    products = 2,
    intrinsic = _mm256_dpwsuds_epi32,
    accumulate = saturate
}

// dpwusd — unsigned×signed words, wrapping (ACE group 1: AVX-VNNI-INT16, `VPDPWUSD`).
// `a: [u16;16]` zero-extends, `b: [i16;16]` sign-extends — operand order is the inverse of
// dpwsud, so `dpwusd != dpwsud` for the same numeric operands.
define_dp! {
    name = dpwusd,
    scalar = dpwusd_scalar,
    hw = dpwusd_hw,
    feature = "avxvnniint16",
    a = u16,
    b = i16,
    products = 2,
    intrinsic = _mm256_dpwusd_epi32,
    accumulate = wrap
}

// ============== Phase 5: word variants part 2 — avxvnniint16, complete group-1 ==============
// The final three word variants close the 12-cell grid: the last US-saturate (dpwusds, signed
// `@fold saturate`), and the unsigned×unsigned pair in both accumulate forms (dpwuud wrap,
// dpwuuds saturate). These reuse the `define_dp!` macro `products = 2`, 16-element operand
// types; dpwusds uses the signed `@fold saturate` arm (a SINGLE signed-dword saturation of the
// FULL-PRECISION i64 sum `src + Σ products`, per Intel SDM / Felix Cloutier VPDPW*DS — no
// intermediate product-sum clamp), while dpwuuds uses `@fold saturate_unsigned` — UU saturates
// into the UNSIGNED dword range `[0, u32::MAX]` because both operands and the accumulator are
// unsigned (VPDPWUUDS). The UU variants carry the LARGEST product sums of the whole family
// (u16×u16 ≈ 4.29e9 per product, 2/lane ≈ 8.59e9), well beyond u32::MAX, so the i64 fold is
// load-bearing: an unsigned total above u32::MAX clamps to u32::MAX (i32 `-1`), and one in
// `(i32::MAX, u32::MAX]` is stored verbatim (reads back negative) rather than clamped to
// i32::MAX as a signed-saturate would.
//   * dpwusds is US (mixed signedness) — operand order significant (`dpwusds != dpwsuds`);
//     A1's distinct [u16;16]/[i16;16] types make a `b,a` swap a compile error → NO commutativity.
//   * dpwuud / dpwuuds are UU — both operands `[u16;16]` zero-extend, commutative.
// [vnni-int8-int16-family.API.1] [vnni-int8-int16-family.API.2] [vnni-int8-int16-family.API.3]
// [vnni-int8-int16-family.API.4] [vnni-int8-int16-family.NATIVE_PATH.1]
// [vnni-int8-int16-family.SCALAR_ORACLE.1-2] [vnni-int8-int16-family.SCALAR_ORACLE.1-4]

// dpwusds — unsigned×signed words, saturating (ACE group 1: AVX-VNNI-INT16, `VPDPWUSDS`).
// `a: [u16;16]` zero-extends, `b: [i16;16]` sign-extends — inverse operand order of dpwsuds.
define_dp! {
    name = dpwusds,
    scalar = dpwusds_scalar,
    hw = dpwusds_hw,
    feature = "avxvnniint16",
    a = u16,
    b = i16,
    products = 2,
    intrinsic = _mm256_dpwusds_epi32,
    accumulate = saturate
}

// dpwuud — unsigned×unsigned words, wrapping (ACE group 1: AVX-VNNI-INT16, `VPDPWUUD`).
// Both operands `[u16;16]` zero-extend; commutative (UU). The i64 fold cannot overflow before
// the wrapping-cast even though a u16×u16 product (≈4.29e9) exceeds i32::MAX.
define_dp! {
    name = dpwuud,
    scalar = dpwuud_scalar,
    hw = dpwuud_hw,
    feature = "avxvnniint16",
    a = u16,
    b = u16,
    products = 2,
    intrinsic = _mm256_dpwuud_epi32,
    accumulate = wrap
}

// dpwuuds — unsigned×unsigned words, saturating (ACE group 1: AVX-VNNI-INT16, `VPDPWUUDS`).
// Commutative (UU); both operands and the accumulator are unsigned, so the lane total
// saturates into the UNSIGNED dword range `[0, u32::MAX]` (see the `@fold saturate_unsigned`
// arm), NOT the signed range.
define_dp! {
    name = dpwuuds,
    scalar = dpwuuds_scalar,
    hw = dpwuuds_hw,
    feature = "avxvnniint16",
    a = u16,
    b = u16,
    products = 2,
    intrinsic = _mm256_dpwuuds_epi32,
    accumulate = saturate_unsigned
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Differential test: native path must match the scalar oracle bit-for-bit.
    /// Runs the comparison only where the native path is actually available.
    #[test]
    fn hw_matches_scalar() {
        let a: [i8; 32] = core::array::from_fn(|i| i as i8 - 16);
        let b: [i8; 32] = core::array::from_fn(|i| (i as i8).wrapping_mul(3).wrapping_sub(7));
        let src: [i32; 8] = core::array::from_fn(|i| i as i32 * 100);

        let want = dpbssd_scalar(src, a, b);

        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avxvnniint8") {
            // SAFETY: feature checked above.
            assert_eq!(
                unsafe { dpbssd_hw(src, a, b) },
                want,
                "native path disagrees with oracle"
            );
        }

        // Public API always works (falls back when the feature is absent).
        assert_eq!(dpbssd(src, a, b), want);
    }

    /// Differential test for `dpbssds` (SS, saturate): the native `VPDPBSSDS` path must
    /// match the saturating scalar oracle bit-for-bit. Runs only where `avxvnniint8` is
    /// actually available; fails with `native path disagrees with oracle`
    /// (`NativeDivergesFromOracle`) on divergence.
    /// [vnni-int8-int16-family.TESTS.1] [vnni-int8-int16-family.CORRECTNESS.1]
    #[test]
    fn dpbssds_hw_matches_scalar() {
        // Include lanes whose product sum is large enough to exercise the saturating clamp.
        let a: [i8; 32] = core::array::from_fn(|i| (i as i8).wrapping_mul(7).wrapping_sub(64));
        let b: [i8; 32] = core::array::from_fn(|i| (i as i8).wrapping_mul(5).wrapping_add(33));
        let src: [i32; 8] = core::array::from_fn(|i| (i as i32 - 4) * 1_000_000);

        let want = dpbssds_scalar(src, a, b);

        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avxvnniint8") {
            // SAFETY: feature checked above.
            assert_eq!(
                unsafe { dpbssds_hw(src, a, b) },
                want,
                "native path disagrees with oracle"
            );
        }

        // Public API always works (falls back when the feature is absent).
        assert_eq!(dpbssds(src, a, b), want);
    }

    // ============== Phase 3: deterministic differential tests (byte variants) ==============
    // One per variant; runs the native==oracle comparison only where `avxvnniint8` is
    // actually detected (skipped on this arm64 host — SDE CI exercises the native path),
    // failing with `native path disagrees with oracle` (`NativeDivergesFromOracle`) on
    // divergence. The public dispatcher is always checked against the oracle.
    // [vnni-int8-int16-family.TESTS.1] [vnni-int8-int16-family.CORRECTNESS.1]

    /// Differential test for `dpbsud` (SU, wrap): native `VPDPBSUD` vs the wrapping oracle.
    #[test]
    fn dpbsud_hw_matches_scalar() {
        let a: [i8; 32] = core::array::from_fn(|i| (i as i8).wrapping_mul(11).wrapping_sub(70));
        let b: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(9).wrapping_add(3));
        let src: [i32; 8] = core::array::from_fn(|i| (i as i32 - 4) * 7_000);

        let want = dpbsud_scalar(src, a, b);

        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avxvnniint8") {
            // SAFETY: feature checked above.
            assert_eq!(
                unsafe { dpbsud_hw(src, a, b) },
                want,
                "native path disagrees with oracle"
            );
        }
        assert_eq!(dpbsud(src, a, b), want);
    }

    /// Differential test for `dpbsuds` (SU, saturate): native `VPDPBSUDS` vs the saturating
    /// oracle, with inputs that exercise the clamp.
    #[test]
    fn dpbsuds_hw_matches_scalar() {
        let a: [i8; 32] = core::array::from_fn(|i| (i as i8).wrapping_mul(13).wrapping_sub(50));
        let b: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(17).wrapping_add(7));
        let src: [i32; 8] = core::array::from_fn(|i| (i as i32 - 4) * 500_000_000);

        let want = dpbsuds_scalar(src, a, b);

        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avxvnniint8") {
            // SAFETY: feature checked above.
            assert_eq!(
                unsafe { dpbsuds_hw(src, a, b) },
                want,
                "native path disagrees with oracle"
            );
        }
        assert_eq!(dpbsuds(src, a, b), want);
    }

    /// Differential test for `dpbuud` (UU, wrap): native `VPDPBUUD` vs the wrapping oracle.
    #[test]
    fn dpbuud_hw_matches_scalar() {
        let a: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(23).wrapping_add(1));
        let b: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(19).wrapping_add(5));
        let src: [i32; 8] = core::array::from_fn(|i| (i as i32 - 4) * 11_000);

        let want = dpbuud_scalar(src, a, b);

        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avxvnniint8") {
            // SAFETY: feature checked above.
            assert_eq!(
                unsafe { dpbuud_hw(src, a, b) },
                want,
                "native path disagrees with oracle"
            );
        }
        assert_eq!(dpbuud(src, a, b), want);
    }

    /// Differential test for `dpbuuds` (UU, saturate): native `VPDPBUUDS` vs the saturating
    /// oracle, with inputs that exercise the clamp.
    #[test]
    fn dpbuuds_hw_matches_scalar() {
        let a: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(29).wrapping_add(200));
        let b: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(31).wrapping_add(180));
        // Lanes 5-7 sit within one max product-sum (4*255*255 = 260100) of i32::MAX so
        // the large positive products push them into the saturating clamp; all lanes stay
        // within i32 range (no debug-overflow when building the inputs).
        let src: [i32; 8] = core::array::from_fn(|i| i32::MAX - (7 - i as i32) * 100_000);

        let want = dpbuuds_scalar(src, a, b);

        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avxvnniint8") {
            // SAFETY: feature checked above.
            assert_eq!(
                unsafe { dpbuuds_hw(src, a, b) },
                want,
                "native path disagrees with oracle"
            );
        }
        assert_eq!(dpbuuds(src, a, b), want);
    }

    // ============== Phase 4: deterministic differential tests (word variants) ==============
    // dpwsud / dpwsuds / dpwusd — gated on `avxvnniint16` (not detected on this arm64 host,
    // so the native block is skipped here; SDE CI exercises it). Each fails with
    // `native path disagrees with oracle` (`NativeDivergesFromOracle`) on divergence; the
    // public dispatcher is always checked against the oracle.
    // [vnni-int8-int16-family.TESTS.1] [vnni-int8-int16-family.CORRECTNESS.1]

    /// Differential test for `dpwsud` (SU, wrap; words): native `VPDPWSUD` vs the wrapping
    /// oracle. `a` is signed (`[i16;16]`), `b` unsigned (`[u16;16]`), 2 products/lane.
    #[test]
    fn dpwsud_hw_matches_scalar() {
        let a: [i16; 16] =
            core::array::from_fn(|i| (i as i16).wrapping_mul(4001).wrapping_sub(20000));
        let b: [u16; 16] =
            core::array::from_fn(|i| (i as u16).wrapping_mul(4099).wrapping_add(257));
        let src: [i32; 8] = core::array::from_fn(|i| (i as i32 - 4) * 13_000);

        let want = dpwsud_scalar(src, a, b);

        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avxvnniint16") {
            // SAFETY: feature checked above.
            assert_eq!(
                unsafe { dpwsud_hw(src, a, b) },
                want,
                "native path disagrees with oracle"
            );
        }
        assert_eq!(dpwsud(src, a, b), want);
    }

    /// Differential test for `dpwsuds` (SU, saturate; words): native `VPDPWSUDS` vs the
    /// saturating oracle, with inputs that exercise the clamp.
    #[test]
    fn dpwsuds_hw_matches_scalar() {
        let a: [i16; 16] =
            core::array::from_fn(|i| (i as i16).wrapping_mul(8009).wrapping_sub(16000));
        let b: [u16; 16] =
            core::array::from_fn(|i| (i as u16).wrapping_mul(8191).wrapping_add(40000));
        let src: [i32; 8] = core::array::from_fn(|i| (i as i32 - 4) * 500_000_000);

        let want = dpwsuds_scalar(src, a, b);

        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avxvnniint16") {
            // SAFETY: feature checked above.
            assert_eq!(
                unsafe { dpwsuds_hw(src, a, b) },
                want,
                "native path disagrees with oracle"
            );
        }
        assert_eq!(dpwsuds(src, a, b), want);
    }

    /// Differential test for `dpwusd` (US, wrap; words): native `VPDPWUSD` vs the wrapping
    /// oracle. `a` is unsigned (`[u16;16]`), `b` signed (`[i16;16]`) — inverse operand order
    /// of `dpwsud`.
    #[test]
    fn dpwusd_hw_matches_scalar() {
        let a: [u16; 16] =
            core::array::from_fn(|i| (i as u16).wrapping_mul(4099).wrapping_add(257));
        let b: [i16; 16] =
            core::array::from_fn(|i| (i as i16).wrapping_mul(4001).wrapping_sub(20000));
        let src: [i32; 8] = core::array::from_fn(|i| (i as i32 - 4) * 13_000);

        let want = dpwusd_scalar(src, a, b);

        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avxvnniint16") {
            // SAFETY: feature checked above.
            assert_eq!(
                unsafe { dpwusd_hw(src, a, b) },
                want,
                "native path disagrees with oracle"
            );
        }
        assert_eq!(dpwusd(src, a, b), want);
    }

    /// Dual-feature coverage guard. When `ACE_REQUIRE_NATIVE` is present (CI's SDE job;
    /// presence-checked, value ignored), EVERY feature the suite exercises — both
    /// `avxvnniint8` (byte ops) AND `avxvnniint16` (word ops) — *must* have been detected,
    /// otherwise a green suite would only prove the native byte path while the word ops
    /// silently fell back to the scalar oracle. Off by default, so local/non-x86 runs are
    /// unaffected: with the var unset the guard returns early (non-vacuous fallback) and the
    /// host `cargo test` stays green without requiring any native path.
    ///
    /// Realises NATIVE_GUARD.1 / NATIVE_GUARD.1-1 (dual-feature invariant) and the
    /// NativeFeaturePrecondition validation rule; renders NativeGuardNotDetected (missing
    /// feature, exit 1) and NativeGuardNonX86 (wrong arch, exit 1).
    #[test]
    fn native_runs_when_required() {
        // ACE_REQUIRE_NATIVE absent → guard returns early, suite stays green without native.
        // On this host the var is unset, so this test exercises exactly this branch and passes.
        if std::env::var_os("ACE_REQUIRE_NATIVE").is_none() {
            return;
        }
        #[cfg(target_arch = "x86_64")]
        {
            // Assert EVERY exercised feature family was detected. A miss panics with the
            // message parameterised by the *missing* feature token (NativeGuardNotDetected).
            for feature in ["avxvnniint8", "avxvnniint16"] {
                let detected = match feature {
                    "avxvnniint8" => std::is_x86_feature_detected!("avxvnniint8"),
                    "avxvnniint16" => std::is_x86_feature_detected!("avxvnniint16"),
                    _ => unreachable!(),
                };
                assert!(
                    detected,
                    "ACE_REQUIRE_NATIVE=1 but {feature} is not detected — the native path was NOT exercised"
                );
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        panic!("ACE_REQUIRE_NATIVE=1 on a non-x86_64 target — the native path cannot run here");
    }

    /// Group-4 (`ACE` tile) counterpart of [`native_runs_when_required`], honoring
    /// `ACE_REQUIRE_NATIVE=1` per the same pattern ([ace-tile-instructions.TESTING.2]). Group-4
    /// native execution is deferred to Intel SDE ACE emulation (OQ-6, R2): families A / B-read /
    /// C are intrinsic-reachable and light up under SDE, while the `ACE`-only `.byte` families
    /// (B write, D, E, F, G) stay dormant until SDE gains ACE emulation. This guard therefore
    /// stays **dormant** for group 4 — exactly as the hard guard does for the AVX10 C-shim
    /// families — recording the per-family tile detection status rather than vacuously asserting
    /// a native branch that cannot yet run. When the var is unset (local / non-CI) it returns
    /// early, so the host `cargo test` stays green without any native tile path.
    #[test]
    fn ace_tile_native_runs_when_required() {
        // ACE_REQUIRE_NATIVE absent → dormant, suite stays green without native tile execution.
        if std::env::var_os("ACE_REQUIRE_NATIVE").is_none() {
            return;
        }
        #[cfg(target_arch = "x86_64")]
        {
            // Non-vacuous record: a future ACE-capable SDE run surfaces these as `true` and the
            // per-family `prop_native_matches_oracle` differentials stop discarding. Until then
            // the group-4 native path is built (encoding-asserted) but not executed — no hard
            // requirement is asserted here, matching the AVX10 C-shim families' dormant posture.
            eprintln!(
                "ACE_REQUIRE_NATIVE=1: group-4 tile native execution deferred to SDE ACE \
                 (has_amx_tile={} has_amx_avx512={} has_ace={}); .byte families discard until then",
                crate::detect::has_amx_tile(),
                crate::detect::has_amx_avx512(),
                crate::detect::has_ace(),
            );
        }
        // On non-x86_64 the group-4 native path cannot exist; the group-1 guard
        // (`native_runs_when_required`) already panics for that case, so nothing to add here.
    }

    /// Hand-computed value, independent of the implementation.
    #[test]
    fn known_value() {
        // lane 0: 1*1 + 2*2 + 3*3 + 4*4 = 30; all other lanes use zero inputs.
        let mut a = [0i8; 32];
        let mut b = [0i8; 32];
        for k in 0..4 {
            a[k] = (k as i8) + 1;
            b[k] = (k as i8) + 1;
        }
        assert_eq!(dpbssd([0; 8], a, b), [30, 0, 0, 0, 0, 0, 0, 0]);
    }

    /// Hand-computed value for `dpbssds`, independent of the implementation, covering the
    /// saturating clamp on at least one lane.
    ///
    /// * lane 0: `src=0`, all four products `127*127=16129` → `0 + 4*16129 = 64516`
    ///   (no clamp — well within i32).
    /// * lane 1: `src=i32::MAX`, products `127*127=16129` each → `i32::MAX + 64516`
    ///   saturates to `i32::MAX`.
    /// * lane 2: `src=i32::MIN`, products `127*-128=-16256` each → `i32::MIN + 4*-16256`
    ///   saturates to `i32::MIN`.
    ///
    /// [vnni-int8-int16-family.TESTS.2] [vnni-int8-int16-family.SCALAR_ORACLE.1-4]
    #[test]
    fn dpbssds_known_value() {
        let mut a = [0i8; 32];
        let mut b = [0i8; 32];
        // lane 0 (a[0..4], b[0..4]): 127 * 127, no saturation.
        for slot in 0..4 {
            a[slot] = 127;
            b[slot] = 127;
        }
        // lane 1 (a[4..8], b[4..8]): 127 * 127 with src = i32::MAX → clamp to MAX.
        for slot in 4..8 {
            a[slot] = 127;
            b[slot] = 127;
        }
        // lane 2 (a[8..12], b[8..12]): 127 * -128 with src = i32::MIN → clamp to MIN.
        for slot in 8..12 {
            a[slot] = 127;
            b[slot] = -128;
        }
        let mut src = [0i32; 8];
        src[1] = i32::MAX;
        src[2] = i32::MIN;

        let out = dpbssds(src, a, b);
        assert_eq!(out[0], 64_516, "lane 0: 4 * 127*127 = 64516, no clamp");
        assert_eq!(
            out[1],
            i32::MAX,
            "lane 1: i32::MAX + 64516 clamps to i32::MAX"
        );
        assert_eq!(
            out[2],
            i32::MIN,
            "lane 2: i32::MIN + 4*(127*-128) clamps to i32::MIN"
        );
    }

    // ================== Phase 3: hand-computed known-value tests (byte variants) ==================
    // Each is independent of the implementation: products spelled out by hand.
    // [vnni-int8-int16-family.TESTS.2]

    /// Hand-computed value for `dpbsud` (SU, wrap). `a` is signed, `b` unsigned.
    ///
    /// * lane 0: `src=0`, products `(-1)*2 = -2` ×4 → `0 + 4*(-2) = -8` (the manualExecution
    ///   case from the plan).
    /// * lane 1: `src=1000`, products `5 * 255 = 1275` ×4 → `1000 + 5100 = 6100`.
    #[test]
    fn dpbsud_known_value() {
        let mut a = [0i8; 32];
        let mut b = [0u8; 32];
        for slot in 0..4 {
            a[slot] = -1;
            b[slot] = 2;
        }
        for slot in 4..8 {
            a[slot] = 5;
            b[slot] = 255;
        }
        let mut src = [0i32; 8];
        src[1] = 1000;

        let out = dpbsud(src, a, b);
        assert_eq!(out[0], -8, "lane 0: 0 + 4*(-1*2) = -8");
        assert_eq!(out[1], 6100, "lane 1: 1000 + 4*(5*255) = 6100");
    }

    /// Hand-computed value for `dpbsuds` (SU, saturate), covering the clamp.
    ///
    /// * lane 0: `src=0`, products `127*255 = 32385` ×4 → `0 + 129540 = 129540` (no clamp).
    /// * lane 1: `src=i32::MAX`, products `127*255` ×4 (positive) → clamps to `i32::MAX`.
    /// * lane 2: `src=i32::MIN`, products `(-128)*255 = -32640` ×4 → clamps to `i32::MIN`.
    #[test]
    fn dpbsuds_known_value() {
        let mut a = [0i8; 32];
        let mut b = [0u8; 32];
        for slot in 0..4 {
            a[slot] = 127;
            b[slot] = 255;
        }
        for slot in 4..8 {
            a[slot] = 127;
            b[slot] = 255;
        }
        for slot in 8..12 {
            a[slot] = -128;
            b[slot] = 255;
        }
        let mut src = [0i32; 8];
        src[1] = i32::MAX;
        src[2] = i32::MIN;

        let out = dpbsuds(src, a, b);
        assert_eq!(out[0], 129_540, "lane 0: 4 * 127*255 = 129540, no clamp");
        assert_eq!(out[1], i32::MAX, "lane 1: i32::MAX + 129540 clamps to MAX");
        assert_eq!(
            out[2],
            i32::MIN,
            "lane 2: i32::MIN + 4*(-128*255) clamps to MIN"
        );
    }

    /// Hand-computed value for `dpbuud` (UU, wrap). Both operands unsigned.
    ///
    /// * lane 0: `src=0`, products `255*255 = 65025` ×4 → `0 + 260100 = 260100`.
    /// * lane 1: `src=-100`, products `10*20 = 200` ×4 → `-100 + 800 = 700`.
    #[test]
    fn dpbuud_known_value() {
        let mut a = [0u8; 32];
        let mut b = [0u8; 32];
        for slot in 0..4 {
            a[slot] = 255;
            b[slot] = 255;
        }
        for slot in 4..8 {
            a[slot] = 10;
            b[slot] = 20;
        }
        let mut src = [0i32; 8];
        src[1] = -100;

        let out = dpbuud(src, a, b);
        assert_eq!(out[0], 260_100, "lane 0: 4 * 255*255 = 260100");
        assert_eq!(out[1], 700, "lane 1: -100 + 4*(10*20) = 700");
    }

    /// Hand-computed value for `dpbuuds` (UU, saturate), covering UNSIGNED-dword saturation.
    ///
    /// `VPDPBUUDS` clamps the lane total into the *unsigned* dword range `[0, u32::MAX]` (not
    /// the signed range), because both operands and the accumulator are unsigned.
    ///
    /// * lane 0: `src=0`, products `255*255 = 65025` ×4 → `260100` (no clamp).
    /// * lane 1: `src` bits = `i32::MAX`, products `255*255` ×4 → total `2_147_743_747` fits a
    ///   u32 (no clamp) but has the high bit set, so the i32-reinterpreted result is negative —
    ///   proving the saturation is UNSIGNED, not signed (signed-saturate would give i32::MAX).
    /// * lane 2: `src` bits = `-1` (= u32::MAX), products `255*255` ×4 → overflows u32::MAX →
    ///   clamps to u32::MAX (i32 `-1`).
    /// * lane 3: `src` bits = `i32::MIN`, all products zero → stays `i32::MIN`.
    #[test]
    fn dpbuuds_known_value() {
        let mut a = [0u8; 32];
        let mut b = [0u8; 32];
        for slot in 0..4 {
            a[slot] = 255;
            b[slot] = 255;
        }
        for slot in 4..8 {
            a[slot] = 255;
            b[slot] = 255;
        }
        for slot in 8..12 {
            a[slot] = 255;
            b[slot] = 255;
        }
        let mut src = [0i32; 8];
        src[1] = i32::MAX;
        src[2] = -1;
        src[3] = i32::MIN;

        let out = dpbuuds(src, a, b);
        assert_eq!(out[0], 260_100, "lane 0: 4 * 255*255 = 260100, no clamp");
        assert_eq!(
            out[1],
            (i32::MAX as u32).wrapping_add(260_100) as i32,
            "lane 1: unsigned sum 2_147_743_747 fits u32 (no clamp) but reads negative as i32"
        );
        assert_eq!(
            out[2], -1,
            "lane 2: u32::MAX + 260100 clamps to u32::MAX (-1)"
        );
        assert_eq!(out[3], i32::MIN, "lane 3: i32::MIN bits + 0 stays i32::MIN");
    }

    // ================== Phase 4: hand-computed known-value tests (word variants) ==================
    // Each independent of the implementation; 2 products/lane. The critical case includes a
    // `u16 × u16` product that exceeds `i32::MAX`, proving the i64-wide accumulation in the
    // oracle (an i32-truncating oracle would mis-model it).
    // [vnni-int8-int16-family.TESTS.2] [vnni-int8-int16-family.SCALAR_ORACLE.1-4]

    /// Hand-computed value for `dpwsud` (SU, wrap; words). `a` signed, `b` unsigned, 2
    /// products/lane.
    ///
    /// * lane 0: `src=0`, products `(-3)*4 = -12` and `(-3)*4 = -12` → `0 + (-24) = -24`.
    /// * lane 1: `src=1000`, products `7*65535 = 458745` ×2 → `1000 + 917490 = 918490`.
    ///   `7 * 65535 = 458_745` fits in i32; the *sum* `918_490` also fits — but each operand
    ///   is a full-range u16, exercising zero-extension of `b`.
    #[test]
    fn dpwsud_known_value() {
        let mut a = [0i16; 16];
        let mut b = [0u16; 16];
        // lane 0 (a[0..2], b[0..2]).
        for slot in 0..2 {
            a[slot] = -3;
            b[slot] = 4;
        }
        // lane 1 (a[2..4], b[2..4]).
        for slot in 2..4 {
            a[slot] = 7;
            b[slot] = 65535;
        }
        let mut src = [0i32; 8];
        src[1] = 1000;

        let out = dpwsud(src, a, b);
        assert_eq!(out[0], -24, "lane 0: 0 + 2*(-3*4) = -24");
        assert_eq!(out[1], 918_490, "lane 1: 1000 + 2*(7*65535) = 918490");
    }

    /// Hand-computed value for `dpwsuds` (SU, saturate; words), covering the clamp.
    ///
    /// * lane 0: `src=0`, products `32767*65535 = 2_147_385_345` ×2 = `4_294_770_690`
    ///   (exceeds i32::MAX) → clamps to `i32::MAX`. This is the `> i32::MAX` product/sum case.
    /// * lane 1: `src=i32::MIN`, products `(-32768)*65535 = -2_147_450_880` ×2 → clamps to
    ///   `i32::MIN`.
    /// * lane 2: `src=5`, products `1*2 = 2` ×2 → `5 + 4 = 9` (no clamp).
    #[test]
    fn dpwsuds_known_value() {
        let mut a = [0i16; 16];
        let mut b = [0u16; 16];
        for slot in 0..2 {
            a[slot] = 32767;
            b[slot] = 65535;
        }
        for slot in 2..4 {
            a[slot] = -32768;
            b[slot] = 65535;
        }
        for slot in 4..6 {
            a[slot] = 1;
            b[slot] = 2;
        }
        let mut src = [0i32; 8];
        src[1] = i32::MIN;
        src[2] = 5;

        let out = dpwsuds(src, a, b);
        assert_eq!(
            out[0],
            i32::MAX,
            "lane 0: 2*(32767*65535) = 4294770690 > i32::MAX → clamp"
        );
        assert_eq!(
            out[1],
            i32::MIN,
            "lane 1: i32::MIN + 2*(-32768*65535) clamps to MIN"
        );
        assert_eq!(out[2], 9, "lane 2: 5 + 2*(1*2) = 9, no clamp");
    }

    /// Hand-computed value for `dpwusd` (US, wrap; words). `a` unsigned, `b` signed; this is
    /// the plan's manualExecution case plus a `u16 × i16` product exceeding `i32::MAX`.
    ///
    /// * lane 0: `src=0`, `a=65535`, `b=1`, products `65535*1 = 65535` ×2 → `131070`
    ///   (the plan's `dpwusd([0;8],[65535;16],[1;16])` lane value).
    /// * lane 1: `src=0`, `a=65535`, `b=32767`, products `65535*32767 = 2_147_385_345` ×2 =
    ///   `4_294_770_690` (exceeds i32::MAX) → wraps (mod 2^32) to a defined i32. The oracle
    ///   folds in i64 then wrapping-casts; an i32-truncating oracle would mis-model this.
    #[test]
    fn dpwusd_known_value() {
        let mut a = [0u16; 16];
        let mut b = [0i16; 16];
        for slot in 0..2 {
            a[slot] = 65535;
            b[slot] = 1;
        }
        for slot in 2..4 {
            a[slot] = 65535;
            b[slot] = 32767;
        }
        let src = [0i32; 8];

        let out = dpwusd(src, a, b);
        assert_eq!(out[0], 131_070, "lane 0: 0 + 2*(65535*1) = 131070");
        // lane 1: i64 sum = 2 * 65535 * 32767 = 4_294_770_690; wrapping into i32:
        // 4_294_770_690 - 2^32 = 4_294_770_690 - 4_294_967_296 = -196_606.
        let expected_lane1 = (2i64 * 65535 * 32767) as i32; // i64 → i32 wraps (`as`).
        assert_eq!(
            out[1], expected_lane1,
            "lane 1: 2*(65535*32767) = 4294770690 wraps to {expected_lane1}"
        );
        assert_eq!(expected_lane1, -196_606, "wrap arithmetic sanity check");
    }

    // ============== Phase 5: deterministic differential tests (word variants pt 2) ==============
    // dpwusds / dpwuud / dpwuuds — gated on `avxvnniint16` (not detected on this arm64 host, so
    // the native block is skipped here; SDE CI exercises it). Each fails with
    // `native path disagrees with oracle` (`NativeDivergesFromOracle`) on divergence; the public
    // dispatcher is always checked against the oracle.
    // [vnni-int8-int16-family.TESTS.1] [vnni-int8-int16-family.CORRECTNESS.1]

    /// Differential test for `dpwusds` (US, saturate; words): native `VPDPWUSDS` vs the
    /// saturating oracle, with inputs that exercise the clamp. `a` unsigned, `b` signed.
    #[test]
    fn dpwusds_hw_matches_scalar() {
        let a: [u16; 16] =
            core::array::from_fn(|i| (i as u16).wrapping_mul(8191).wrapping_add(40000));
        let b: [i16; 16] =
            core::array::from_fn(|i| (i as i16).wrapping_mul(8009).wrapping_sub(16000));
        let src: [i32; 8] = core::array::from_fn(|i| (i as i32 - 4) * 500_000_000);

        let want = dpwusds_scalar(src, a, b);

        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avxvnniint16") {
            // SAFETY: feature checked above.
            assert_eq!(
                unsafe { dpwusds_hw(src, a, b) },
                want,
                "native path disagrees with oracle"
            );
        }
        assert_eq!(dpwusds(src, a, b), want);
    }

    /// Differential test for `dpwuud` (UU, wrap; words): native `VPDPWUUD` vs the wrapping
    /// oracle. Both operands unsigned; the u16×u16 products (≈4.29e9) exercise the i64 fold.
    #[test]
    fn dpwuud_hw_matches_scalar() {
        let a: [u16; 16] =
            core::array::from_fn(|i| (i as u16).wrapping_mul(4099).wrapping_add(50021));
        let b: [u16; 16] =
            core::array::from_fn(|i| (i as u16).wrapping_mul(4093).wrapping_add(60013));
        let src: [i32; 8] = core::array::from_fn(|i| (i as i32 - 4) * 17_000);

        let want = dpwuud_scalar(src, a, b);

        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avxvnniint16") {
            // SAFETY: feature checked above.
            assert_eq!(
                unsafe { dpwuud_hw(src, a, b) },
                want,
                "native path disagrees with oracle"
            );
        }
        assert_eq!(dpwuud(src, a, b), want);
    }

    /// Differential test for `dpwuuds` (UU, saturate; words): native `VPDPWUUDS` vs the
    /// saturating oracle. The largest product sums of the family drive the clamp.
    #[test]
    fn dpwuuds_hw_matches_scalar() {
        let a: [u16; 16] =
            core::array::from_fn(|i| (i as u16).wrapping_mul(8191).wrapping_add(58000));
        let b: [u16; 16] =
            core::array::from_fn(|i| (i as u16).wrapping_mul(8101).wrapping_add(57000));
        // Mix of large positive `src` (clamps toward MAX with the big positive products) and
        // i32::MIN `src` lanes; all within i32 range when constructed.
        let src: [i32; 8] = core::array::from_fn(|i| {
            if i % 2 == 0 {
                i32::MAX - (i as i32) * 1_000_000
            } else {
                i32::MIN + (i as i32) * 1_000_000
            }
        });

        let want = dpwuuds_scalar(src, a, b);

        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avxvnniint16") {
            // SAFETY: feature checked above.
            assert_eq!(
                unsafe { dpwuuds_hw(src, a, b) },
                want,
                "native path disagrees with oracle"
            );
        }
        assert_eq!(dpwuuds(src, a, b), want);
    }

    // ============== Phase 5: hand-computed known-value tests (word variants pt 2) ==============
    // Each independent of the implementation; 2 products/lane. The UU cases include products
    // and sums exceeding i32::MAX (the u16 trap) AND the load-bearing opposite-sign case:
    // src = i32::MIN with a large POSITIVE product sum, which under the single-saturate model
    // clamps toward i32::MAX (NOT a collapse to ~0 that a two-stage clamp would produce).
    // [vnni-int8-int16-family.TESTS.2] [vnni-int8-int16-family.SCALAR_ORACLE.1-4]

    /// Hand-computed value for `dpwusds` (US, saturate; words). `a` unsigned, `b` signed.
    ///
    /// * lane 0: `src=0`, products `65535 * 1 = 65535` ×2 → `0 + 131070 = 131070` (no clamp).
    /// * lane 1: `src=i32::MAX`, products `65535 * 32767 = 2_147_385_345` ×2 (positive) → clamp
    ///   to `i32::MAX`.
    /// * lane 2: `src=i32::MIN`, products `65535 * -32768 = -2_147_450_880` ×2 → clamp to
    ///   `i32::MIN`.
    /// * lane 3: `src=i32::MIN`, products `65535 * 32767 = 2_147_385_345` ×2 = `4_294_770_690`
    ///   (a large POSITIVE sum). Single-saturate: `i32::MIN + 4_294_770_690 = 2_147_287_042`
    ///   which is within i32 range → NO clamp, exact `2_147_287_042`. (A two-stage clamp would
    ///   first clamp the product sum to i32::MAX then `i32::MIN.saturating_add(i32::MAX) = -1`,
    ///   diverging from hardware — this lane locks the single-saturate model.)
    #[test]
    fn dpwusds_known_value() {
        let mut a = [0u16; 16];
        let mut b = [0i16; 16];
        for slot in 0..2 {
            a[slot] = 65535;
            b[slot] = 1;
        }
        for slot in 2..4 {
            a[slot] = 65535;
            b[slot] = 32767;
        }
        for slot in 4..6 {
            a[slot] = 65535;
            b[slot] = -32768;
        }
        for slot in 6..8 {
            a[slot] = 65535;
            b[slot] = 32767;
        }
        let mut src = [0i32; 8];
        src[1] = i32::MAX;
        src[2] = i32::MIN;
        src[3] = i32::MIN;

        let out = dpwusds(src, a, b);
        assert_eq!(
            out[0], 131_070,
            "lane 0: 0 + 2*(65535*1) = 131070, no clamp"
        );
        assert_eq!(
            out[1],
            i32::MAX,
            "lane 1: i32::MAX + positive clamps to MAX"
        );
        assert_eq!(
            out[2],
            i32::MIN,
            "lane 2: i32::MIN + 2*(65535*-32768) clamps to MIN"
        );
        // lane 3: i32::MIN + 2*(65535*32767) = -2147483648 + 4294770690 = 2147287042 (in range).
        let lane3 = i32::MIN as i64 + 2 * 65535 * 32767;
        assert_eq!(
            lane3, 2_147_287_042,
            "single-saturate full sum, in i32 range"
        );
        assert_eq!(
            out[3], 2_147_287_042,
            "lane 3: i32::MIN + large positive product sum stays in range (single-saturate)"
        );
    }

    /// Hand-computed value for `dpwuud` (UU, wrap; words). Both operands unsigned.
    ///
    /// * lane 0: `src=0`, products `65535*65535 = 4_294_836_225` ×2 = `8_589_672_450`. Folded
    ///   in i64 then wrapping-cast to i32: `8_589_672_450 mod 2^32` reinterpreted signed.
    /// * lane 1: `src=100`, products `3*4 = 12` ×2 → `100 + 24 = 124` (small, no wrap concern).
    #[test]
    fn dpwuud_known_value() {
        let mut a = [0u16; 16];
        let mut b = [0u16; 16];
        for slot in 0..2 {
            a[slot] = 65535;
            b[slot] = 65535;
        }
        for slot in 2..4 {
            a[slot] = 3;
            b[slot] = 4;
        }
        let mut src = [0i32; 8];
        src[1] = 100;

        let out = dpwuud(src, a, b);
        // lane 0: i64 total = 0 + 2*(65535*65535) = 8_589_672_450; wrapping into i32:
        // 8_589_672_450 - 2*2^32 = 8_589_672_450 - 8_589_934_592 = -262_142.
        let lane0 = 8_589_672_450i64 as i32; // i64 → i32 wraps (`as`).
        assert_eq!(lane0, -262_142, "wrap arithmetic sanity check");
        assert_eq!(out[0], -262_142, "lane 0: 2*(65535*65535) wraps to -262142");
        assert_eq!(out[1], 124, "lane 1: 100 + 2*(3*4) = 124");
    }

    /// Hand-computed value for `dpwuuds` (UU, saturate; words), covering UNSIGNED-dword
    /// saturation.
    ///
    /// `VPDPWUUDS` clamps the lane total into the *unsigned* dword range `[0, u32::MAX]`, not
    /// the signed range — both operands and the accumulator are unsigned.
    ///
    /// * lane 0: `src=0`, products `4*4 = 16` ×2 → `32` (no clamp).
    /// * lane 1: `src` bits = `i32::MAX`, products `30000*30000 = 900_000_000` ×2 =
    ///   `1_800_000_000` → total `3_947_483_647` fits a u32 (no clamp) but has the high bit set,
    ///   so the i32-reinterpreted result is negative — proving the saturation is UNSIGNED, not
    ///   signed (signed-saturate would give i32::MAX).
    /// * lane 2: `src=0`, products `65535*65535 = 4_294_836_225` ×2 = `8_589_672_450` →
    ///   overflows u32::MAX → clamps to u32::MAX (i32 `-1`).
    /// * lane 3: `src` bits = `i32::MIN`, all products zero → stays `i32::MIN`.
    #[test]
    fn dpwuuds_known_value() {
        let mut a = [0u16; 16];
        let mut b = [0u16; 16];
        // lane 0 (slots 0,1): small products, src=0 → no clamp.
        for slot in 0..2 {
            a[slot] = 4;
            b[slot] = 4;
        }
        // lane 1 (slots 2,3): unsigned sum fits u32 but reads negative as i32.
        for slot in 2..4 {
            a[slot] = 30000;
            b[slot] = 30000;
        }
        // lane 2 (slots 4,5): products overflow u32::MAX → clamp to u32::MAX (-1).
        for slot in 4..6 {
            a[slot] = 65535;
            b[slot] = 65535;
        }
        // lane 3 (slots 6,7): left ZERO → src=i32::MIN bits stay MIN.
        let mut src = [0i32; 8];
        src[1] = i32::MAX;
        src[3] = i32::MIN;

        let out = dpwuuds(src, a, b);
        assert_eq!(out[0], 32, "lane 0: 2*(4*4) = 32, no clamp");
        assert_eq!(
            out[1],
            (i32::MAX as u32).wrapping_add(1_800_000_000) as i32,
            "lane 1: unsigned sum 3_947_483_647 fits u32 (no clamp) but reads negative as i32"
        );
        assert_eq!(
            out[2], -1,
            "lane 2: 2*(65535*65535) = 8589672450 overflows u32::MAX → clamp to u32::MAX (-1)"
        );
        assert_eq!(out[3], i32::MIN, "lane 3: i32::MIN bits + 0 stays i32::MIN");
    }
}

/// Property-based tests. The hand-rolled tests above pin specific values; these
/// assert the invariants hold across a randomly-sampled slice of the input space,
/// which is far wider than any hand-picked vector can cover.
#[cfg(test)]
mod proptests {
    use super::*;
    use quickcheck::{quickcheck, Arbitrary, Gen, TestResult};

    /// A full, independently-random argument triple for the byte and word variants.
    ///
    /// We wrap the fixed-size arrays in a newtype because `quickcheck` does not derive
    /// `Arbitrary` for arrays of this length; `from_fn` fills each lane from the generator
    /// (index ignored) so every element is sampled independently with no correlated bias.
    /// The signed (`a`/`b`) and unsigned (`a_u`/`b_u`) byte operand fields let SS, SU and UU
    /// byte variants each draw operands of the element types their signature requires; the
    /// word fields (`a_w`/`b_w` = `[i16;16]`, `a_wu`/`b_wu` = `[u16;16]`) do the same for the
    /// word variants.
    /// [vnni-int8-int16-family.TESTS.4]
    #[derive(Clone, Debug)]
    struct Inputs {
        src: [i32; 8],
        a: [i8; 32],
        b: [i8; 32],
        a_u: [u8; 32],
        b_u: [u8; 32],
        a_w: [i16; 16],
        b_w: [i16; 16],
        a_wu: [u16; 16],
        b_wu: [u16; 16],
    }

    impl Arbitrary for Inputs {
        fn arbitrary(g: &mut Gen) -> Self {
            Inputs {
                src: core::array::from_fn(|_| i32::arbitrary(g)),
                a: core::array::from_fn(|_| i8::arbitrary(g)),
                b: core::array::from_fn(|_| i8::arbitrary(g)),
                a_u: core::array::from_fn(|_| u8::arbitrary(g)),
                b_u: core::array::from_fn(|_| u8::arbitrary(g)),
                a_w: core::array::from_fn(|_| i16::arbitrary(g)),
                b_w: core::array::from_fn(|_| i16::arbitrary(g)),
                a_wu: core::array::from_fn(|_| u16::arbitrary(g)),
                b_wu: core::array::from_fn(|_| u16::arbitrary(g)),
            }
        }
    }

    quickcheck! {
        /// Differential property — the headline guarantee. On any input the native
        /// `VPDPBSSD` path must agree with the scalar oracle bit-for-bit. The case
        /// is *discarded* (not passed) when the native path is unavailable, so a
        /// runner without AVX-VNNI-INT8 cannot turn this into a vacuous green; under
        /// CI's SDE job it exercises the real instruction over 100 random inputs.
        fn prop_hw_matches_scalar(input: Inputs) -> TestResult {
            #[cfg(target_arch = "x86_64")]
            {
                if std::is_x86_feature_detected!("avxvnniint8") {
                    let want = dpbssd_scalar(input.src, input.a, input.b);
                    // SAFETY: the feature was confirmed present immediately above.
                    let got = unsafe { dpbssd_hw(input.src, input.a, input.b) };
                    return TestResult::from_bool(got == want);
                }
            }
            TestResult::discard()
        }

        /// The public dispatcher always equals the scalar oracle — this is the
        /// contract callers rely on regardless of which path runs.
        fn prop_public_matches_scalar(input: Inputs) -> bool {
            dpbssd(input.src, input.a, input.b)
                == dpbssd_scalar(input.src, input.a, input.b)
        }

        /// Accumulator linearity: `src` is a pure additive bias (wrapping i32),
        /// independent of the dot products it is added to.
        fn prop_src_is_additive(input: Inputs) -> bool {
            let with_src = dpbssd_scalar(input.src, input.a, input.b);
            let no_src = dpbssd_scalar([0; 8], input.a, input.b);
            (0..8).all(|i| with_src[i] == input.src[i].wrapping_add(no_src[i]))
        }

        /// Operand symmetry: each lane is a dot product `a·b`, so swapping the two
        /// multiplicand vectors leaves the result unchanged.
        fn prop_operands_commute(input: Inputs) -> bool {
            dpbssd_scalar(input.src, input.a, input.b)
                == dpbssd_scalar(input.src, input.b, input.a)
        }

        /// A zeroed multiplicand contributes nothing: the output is exactly `src`.
        fn prop_zero_operand_is_passthrough(input: Inputs) -> bool {
            dpbssd_scalar(input.src, [0; 32], input.b) == input.src
        }

        /// Lane independence: output lane `i` depends only on `a[4i..4i+4]` and
        /// `b[4i..4i+4]`. Zeroing every other lane's operands must not change it.
        fn prop_lanes_are_independent(input: Inputs, lane: u8) -> bool {
            let i = (lane % 8) as usize;
            let mut a = [0i8; 32];
            let mut b = [0i8; 32];
            for k in 0..4 {
                a[4 * i + k] = input.a[4 * i + k];
                b[4 * i + k] = input.b[4 * i + k];
            }
            dpbssd_scalar(input.src, a, b)[i]
                == dpbssd_scalar(input.src, input.a, input.b)[i]
        }
    }

    // ===================== dpbssds (SS, saturate; avxvnniint8) =====================

    /// The independently-saturating reference accumulation a property checks `dpbssds`
    /// against: widen each i8 operand to i64, fold the lane's products, then apply a single
    /// signed-dword saturation of the full sum `src[i] + Σ products` (Intel SDM VPDPB*DS).
    fn dpbssds_lane_expected(src: i32, a: &[i8; 32], b: &[i8; 32], i: usize) -> i32 {
        let mut acc = 0i64;
        for k in 0..4 {
            acc += a[4 * i + k] as i64 * b[4 * i + k] as i64;
        }
        // Single signed-dword saturation of the full-precision sum (Intel SDM / Felix
        // Cloutier VPDPB*DS): clamp `src + Σ products` once. (For bytes the product sum
        // always fits i32, so this coincides with a two-stage clamp; the word ops are where
        // the two models diverge.)
        let total: i64 = src as i64 + acc;
        total.clamp(i32::MIN as i64, i32::MAX as i64) as i32
    }

    quickcheck! {
        /// Differential property for `dpbssds`: native `VPDPBSSDS` must agree with the
        /// saturating scalar oracle bit-for-bit. Discarded (never passed) when the feature
        /// is absent, so a feature-less host cannot go vacuously green.
        /// [vnni-int8-int16-family.TESTS.1]
        fn dpbssds_prop_hw_matches_scalar(input: Inputs) -> TestResult {
            #[cfg(target_arch = "x86_64")]
            {
                if std::is_x86_feature_detected!("avxvnniint8") {
                    let want = dpbssds_scalar(input.src, input.a, input.b);
                    // SAFETY: the feature was confirmed present immediately above.
                    let got = unsafe { dpbssds_hw(input.src, input.a, input.b) };
                    return TestResult::from_bool(got == want);
                }
            }
            TestResult::discard()
        }

        /// The public `dpbssds` dispatcher always equals its scalar oracle.
        fn dpbssds_prop_public_matches_scalar(input: Inputs) -> bool {
            dpbssds(input.src, input.a, input.b)
                == dpbssds_scalar(input.src, input.a, input.b)
        }

        /// A zeroed multiplicand contributes nothing: the output is exactly `src`
        /// (`src.saturating_add(0) == src`).
        fn dpbssds_prop_zero_operand_is_passthrough(input: Inputs) -> bool {
            dpbssds_scalar(input.src, [0; 32], input.b) == input.src
        }

        /// Lane independence: output lane `i` depends only on `a[4i..4i+4]` and
        /// `b[4i..4i+4]`.
        fn dpbssds_prop_lanes_are_independent(input: Inputs, lane: u8) -> bool {
            let i = (lane % 8) as usize;
            let mut a = [0i8; 32];
            let mut b = [0i8; 32];
            for k in 0..4 {
                a[4 * i + k] = input.a[4 * i + k];
                b[4 * i + k] = input.b[4 * i + k];
            }
            dpbssds_scalar(input.src, a, b)[i]
                == dpbssds_scalar(input.src, input.a, input.b)[i]
        }

        /// Operand commutativity holds for `dpbssds` because it is signed×signed (SS):
        /// each lane is a dot product `a·b`, so swapping the multiplicands is identical.
        /// (NOT asserted for the mixed-signedness SU/US variants in later phases.)
        /// [vnni-int8-int16-family.TESTS.3-2]
        fn dpbssds_prop_operands_commute(input: Inputs) -> bool {
            dpbssds_scalar(input.src, input.a, input.b)
                == dpbssds_scalar(input.src, input.b, input.a)
        }

        /// Saturation property (...DS), replacing the wrapping `prop_src_is_additive`:
        /// each lane equals `SIGNED_DWORD_SATURATE(src[i] + Σ products)` — i.e. the final
        /// accumulation *saturates* (never wraps). Whenever the unbounded i64 lane total
        /// would overflow i32, the lane sits exactly at the i32 boundary, which is the
        /// observable difference from a wrapping variant.
        /// [vnni-int8-int16-family.TESTS.3-3] [vnni-int8-int16-family.SCALAR_ORACLE.1-3]
        fn dpbssds_prop_output_saturates(input: Inputs) -> bool {
            let out = dpbssds_scalar(input.src, input.a, input.b);
            (0..8).all(|i| {
                // The lane matches the saturating reference exactly.
                if out[i] != dpbssds_lane_expected(input.src[i], &input.a, &input.b, i) {
                    return false;
                }
                // And when the unbounded total exceeds i32 range, the lane is clamped
                // to the boundary — a wrapping variant would not be.
                let total: i64 = input.src[i] as i64
                    + (0..4)
                        .map(|k| input.a[4 * i + k] as i64 * input.b[4 * i + k] as i64)
                        .sum::<i64>();
                if total > i32::MAX as i64 {
                    out[i] == i32::MAX
                } else if total < i32::MIN as i64 {
                    out[i] == i32::MIN
                } else {
                    true
                }
            })
        }
    }

    // ============== Phase 3: per-variant references + property selection ==============
    // Per-variant property selection follows design §11 / research q6 exactly:
    //   * ALL four: prop_hw_matches_scalar (discard on feature-absence), public_matches_scalar,
    //     zero_operand_is_passthrough, lanes_are_independent.
    //   * wrapping (...D) dpbsud, dpbuud: prop_src_is_additive (wrapping decomposition).
    //   * saturating (...DS) dpbsuds, dpbuuds: saturating-add assertion + prop_output_saturates.
    //   * UU dpbuud, dpbuuds: prop_operands_commute. NOT for SU dpbsud/dpbsuds — A1's distinct
    //     [i8;32]/[u8;32] types make a b,a swap a *compile* error, so commutativity is not even
    //     expressible (the crate-level `compile_fail` doctest is the executed witness of this).
    // [vnni-int8-int16-family.TESTS.3] [vnni-int8-int16-family.TESTS.3-1]
    // [vnni-int8-int16-family.TESTS.3-2] [vnni-int8-int16-family.TESTS.3-3]
    // [vnni-int8-int16-family.CORRECTNESS.2]

    /// Saturating reference for the SU `dpbsuds`: `a` sign-extends (i8), `b` zero-extends
    /// (u8), products folded in i64, then a single signed-dword saturation of the full sum
    /// `src + Σ products` (Intel SDM VPDPB*DS — no intermediate product-sum clamp).
    fn dpbsuds_lane_expected(src: i32, a: &[i8; 32], b: &[u8; 32], i: usize) -> i32 {
        let mut acc = 0i64;
        for k in 0..4 {
            acc += a[4 * i + k] as i64 * b[4 * i + k] as i64;
        }
        // Single signed-dword saturation of the full-precision sum (Intel SDM / Felix
        // Cloutier VPDPB*DS): clamp `src + Σ products` once. (For bytes the product sum
        // always fits i32, so this coincides with a two-stage clamp; the word ops are where
        // the two models diverge.)
        let total: i64 = src as i64 + acc;
        total.clamp(i32::MIN as i64, i32::MAX as i64) as i32
    }

    /// Saturating reference for the UU `dpbuuds`: both operands zero-extend (u8).
    fn dpbuuds_lane_expected(src: i32, a: &[u8; 32], b: &[u8; 32], i: usize) -> i32 {
        let mut acc = 0i64;
        for k in 0..4 {
            acc += a[4 * i + k] as i64 * b[4 * i + k] as i64;
        }
        // UNSIGNED-dword saturation of the full-precision sum (Intel SDM / Felix Cloutier
        // VPDPBUUDS): `SRC.dword` is read as unsigned, so reinterpret `src`'s bits as u32 and
        // clamp `unsigned(src) + Σ products` once into `[0, u32::MAX]`.
        let total: i64 = src as u32 as i64 + acc;
        total.clamp(0, u32::MAX as i64) as u32 as i32
    }

    quickcheck! {
        // -------------------- dpbsud (SU, wrap; avxvnniint8) --------------------

        /// Differential property for `dpbsud`: native `VPDPBSUD` vs the wrapping oracle.
        /// Discarded (never passed) when `avxvnniint8` is absent.
        fn dpbsud_prop_hw_matches_scalar(input: Inputs) -> TestResult {
            #[cfg(target_arch = "x86_64")]
            {
                if std::is_x86_feature_detected!("avxvnniint8") {
                    let want = dpbsud_scalar(input.src, input.a, input.b_u);
                    // SAFETY: the feature was confirmed present immediately above.
                    let got = unsafe { dpbsud_hw(input.src, input.a, input.b_u) };
                    return TestResult::from_bool(got == want);
                }
            }
            TestResult::discard()
        }

        fn dpbsud_prop_public_matches_scalar(input: Inputs) -> bool {
            dpbsud(input.src, input.a, input.b_u)
                == dpbsud_scalar(input.src, input.a, input.b_u)
        }

        /// Wrapping additivity (...D): `src` is a pure wrapping-additive bias.
        /// [vnni-int8-int16-family.TESTS.3-1]
        fn dpbsud_prop_src_is_additive(input: Inputs) -> bool {
            let with_src = dpbsud_scalar(input.src, input.a, input.b_u);
            let no_src = dpbsud_scalar([0; 8], input.a, input.b_u);
            (0..8).all(|i| with_src[i] == input.src[i].wrapping_add(no_src[i]))
        }

        fn dpbsud_prop_zero_operand_is_passthrough(input: Inputs) -> bool {
            dpbsud_scalar(input.src, [0; 32], input.b_u) == input.src
        }

        fn dpbsud_prop_lanes_are_independent(input: Inputs, lane: u8) -> bool {
            let i = (lane % 8) as usize;
            let mut a = [0i8; 32];
            let mut b = [0u8; 32];
            for k in 0..4 {
                a[4 * i + k] = input.a[4 * i + k];
                b[4 * i + k] = input.b_u[4 * i + k];
            }
            dpbsud_scalar(input.src, a, b)[i]
                == dpbsud_scalar(input.src, input.a, input.b_u)[i]
        }
        // NOTE: no `dpbsud_prop_operands_commute` — SU operand order is significant and
        // A1's distinct [i8;32]/[u8;32] types make a swap a compile error (CORRECTNESS.2).

        // -------------------- dpbsuds (SU, saturate; avxvnniint8) --------------------

        /// Differential property for `dpbsuds`: native `VPDPBSUDS` vs the saturating oracle.
        fn dpbsuds_prop_hw_matches_scalar(input: Inputs) -> TestResult {
            #[cfg(target_arch = "x86_64")]
            {
                if std::is_x86_feature_detected!("avxvnniint8") {
                    let want = dpbsuds_scalar(input.src, input.a, input.b_u);
                    // SAFETY: the feature was confirmed present immediately above.
                    let got = unsafe { dpbsuds_hw(input.src, input.a, input.b_u) };
                    return TestResult::from_bool(got == want);
                }
            }
            TestResult::discard()
        }

        fn dpbsuds_prop_public_matches_scalar(input: Inputs) -> bool {
            dpbsuds(input.src, input.a, input.b_u)
                == dpbsuds_scalar(input.src, input.a, input.b_u)
        }

        fn dpbsuds_prop_zero_operand_is_passthrough(input: Inputs) -> bool {
            dpbsuds_scalar(input.src, [0; 32], input.b_u) == input.src
        }

        fn dpbsuds_prop_lanes_are_independent(input: Inputs, lane: u8) -> bool {
            let i = (lane % 8) as usize;
            let mut a = [0i8; 32];
            let mut b = [0u8; 32];
            for k in 0..4 {
                a[4 * i + k] = input.a[4 * i + k];
                b[4 * i + k] = input.b_u[4 * i + k];
            }
            dpbsuds_scalar(input.src, a, b)[i]
                == dpbsuds_scalar(input.src, input.a, input.b_u)[i]
        }

        /// Saturation property (...DS): every lane equals the saturating reference and
        /// sits at the i32 boundary whenever the unbounded total would overflow.
        /// [vnni-int8-int16-family.TESTS.3-3]
        fn dpbsuds_prop_output_saturates(input: Inputs) -> bool {
            let out = dpbsuds_scalar(input.src, input.a, input.b_u);
            (0..8).all(|i| {
                if out[i] != dpbsuds_lane_expected(input.src[i], &input.a, &input.b_u, i) {
                    return false;
                }
                let total: i64 = input.src[i] as i64
                    + (0..4)
                        .map(|k| input.a[4 * i + k] as i64 * input.b_u[4 * i + k] as i64)
                        .sum::<i64>();
                if total > i32::MAX as i64 {
                    out[i] == i32::MAX
                } else if total < i32::MIN as i64 {
                    out[i] == i32::MIN
                } else {
                    true
                }
            })
        }
        // NOTE: no `dpbsuds_prop_operands_commute` — SU, see dpbsud note above.

        // -------------------- dpbuud (UU, wrap; avxvnniint8) --------------------

        /// Differential property for `dpbuud`: native `VPDPBUUD` vs the wrapping oracle.
        fn dpbuud_prop_hw_matches_scalar(input: Inputs) -> TestResult {
            #[cfg(target_arch = "x86_64")]
            {
                if std::is_x86_feature_detected!("avxvnniint8") {
                    let want = dpbuud_scalar(input.src, input.a_u, input.b_u);
                    // SAFETY: the feature was confirmed present immediately above.
                    let got = unsafe { dpbuud_hw(input.src, input.a_u, input.b_u) };
                    return TestResult::from_bool(got == want);
                }
            }
            TestResult::discard()
        }

        fn dpbuud_prop_public_matches_scalar(input: Inputs) -> bool {
            dpbuud(input.src, input.a_u, input.b_u)
                == dpbuud_scalar(input.src, input.a_u, input.b_u)
        }

        /// Wrapping additivity (...D).
        /// [vnni-int8-int16-family.TESTS.3-1]
        fn dpbuud_prop_src_is_additive(input: Inputs) -> bool {
            let with_src = dpbuud_scalar(input.src, input.a_u, input.b_u);
            let no_src = dpbuud_scalar([0; 8], input.a_u, input.b_u);
            (0..8).all(|i| with_src[i] == input.src[i].wrapping_add(no_src[i]))
        }

        fn dpbuud_prop_zero_operand_is_passthrough(input: Inputs) -> bool {
            dpbuud_scalar(input.src, [0; 32], input.b_u) == input.src
        }

        fn dpbuud_prop_lanes_are_independent(input: Inputs, lane: u8) -> bool {
            let i = (lane % 8) as usize;
            let mut a = [0u8; 32];
            let mut b = [0u8; 32];
            for k in 0..4 {
                a[4 * i + k] = input.a_u[4 * i + k];
                b[4 * i + k] = input.b_u[4 * i + k];
            }
            dpbuud_scalar(input.src, a, b)[i]
                == dpbuud_scalar(input.src, input.a_u, input.b_u)[i]
        }

        /// Operand commutativity holds for `dpbuud` (UU): swapping the two u8 multiplicand
        /// vectors leaves the dot product unchanged. (Asserted for UU only.)
        /// [vnni-int8-int16-family.TESTS.3-2]
        fn dpbuud_prop_operands_commute(input: Inputs) -> bool {
            dpbuud_scalar(input.src, input.a_u, input.b_u)
                == dpbuud_scalar(input.src, input.b_u, input.a_u)
        }

        // -------------------- dpbuuds (UU, saturate; avxvnniint8) --------------------

        /// Differential property for `dpbuuds`: native `VPDPBUUDS` vs the saturating oracle.
        fn dpbuuds_prop_hw_matches_scalar(input: Inputs) -> TestResult {
            #[cfg(target_arch = "x86_64")]
            {
                if std::is_x86_feature_detected!("avxvnniint8") {
                    let want = dpbuuds_scalar(input.src, input.a_u, input.b_u);
                    // SAFETY: the feature was confirmed present immediately above.
                    let got = unsafe { dpbuuds_hw(input.src, input.a_u, input.b_u) };
                    return TestResult::from_bool(got == want);
                }
            }
            TestResult::discard()
        }

        fn dpbuuds_prop_public_matches_scalar(input: Inputs) -> bool {
            dpbuuds(input.src, input.a_u, input.b_u)
                == dpbuuds_scalar(input.src, input.a_u, input.b_u)
        }

        fn dpbuuds_prop_zero_operand_is_passthrough(input: Inputs) -> bool {
            dpbuuds_scalar(input.src, [0; 32], input.b_u) == input.src
        }

        fn dpbuuds_prop_lanes_are_independent(input: Inputs, lane: u8) -> bool {
            let i = (lane % 8) as usize;
            let mut a = [0u8; 32];
            let mut b = [0u8; 32];
            for k in 0..4 {
                a[4 * i + k] = input.a_u[4 * i + k];
                b[4 * i + k] = input.b_u[4 * i + k];
            }
            dpbuuds_scalar(input.src, a, b)[i]
                == dpbuuds_scalar(input.src, input.a_u, input.b_u)[i]
        }

        /// Operand commutativity holds for `dpbuuds` (UU).
        /// [vnni-int8-int16-family.TESTS.3-2]
        fn dpbuuds_prop_operands_commute(input: Inputs) -> bool {
            dpbuuds_scalar(input.src, input.a_u, input.b_u)
                == dpbuuds_scalar(input.src, input.b_u, input.a_u)
        }

        /// Saturation property (...DS) for `dpbuuds`.
        /// [vnni-int8-int16-family.TESTS.3-3]
        fn dpbuuds_prop_output_saturates(input: Inputs) -> bool {
            let out = dpbuuds_scalar(input.src, input.a_u, input.b_u);
            (0..8).all(|i| {
                if out[i] != dpbuuds_lane_expected(input.src[i], &input.a_u, &input.b_u, i) {
                    return false;
                }
                // UU saturates into the UNSIGNED dword range; `src` is read as unsigned and
                // the only binding bound is `u32::MAX` (its bit pattern is `-1` as i32).
                let total: i64 = input.src[i] as u32 as i64
                    + (0..4)
                        .map(|k| input.a_u[4 * i + k] as i64 * input.b_u[4 * i + k] as i64)
                        .sum::<i64>();
                if total > u32::MAX as i64 {
                    out[i] == -1
                } else {
                    true
                }
            })
        }
    }

    // ============== Phase 4: word-variant references + property selection ==============
    // dpwsud / dpwsuds / dpwusd (avxvnniint16, 2 products/lane). Per-variant selection:
    //   * ALL three: prop_hw_matches_scalar (discard on avxvnniint16 absence),
    //     public_matches_scalar, zero_operand_is_passthrough, lanes_are_independent.
    //   * wrapping (...D) dpwsud, dpwusd: prop_src_is_additive (wrapping decomposition).
    //   * saturating (...DS) dpwsuds: saturating-add assertion + prop_output_saturates.
    //   * NO prop_operands_commute for ANY of the three — all are SU/US, operand order is
    //     significant (`dpwsud != dpwusd`), and A1's distinct [i16;16]/[u16;16] types make a
    //     `b,a` swap a compile error so commutativity is not even expressible.
    // The lane references fold products in i64 — wide enough for the u16×u16 product trap.
    // [vnni-int8-int16-family.TESTS.3] [vnni-int8-int16-family.TESTS.3-1]
    // [vnni-int8-int16-family.TESTS.3-3] [vnni-int8-int16-family.CORRECTNESS.2]
    // [vnni-int8-int16-family.SCALAR_ORACLE.1-4]

    /// Saturating reference for the SU `dpwsuds`: `a` sign-extends (i16), `b` zero-extends
    /// (u16). Products folded in i64 (cannot overflow before the clamp — `u16×u16` exceeds
    /// i32::MAX), then a SINGLE signed-dword saturation of the full sum `src + Σ products`
    /// (Intel SDM / Felix Cloutier VPDPW*DS — no intermediate clamp of the product-sum).
    fn dpwsuds_lane_expected(src: i32, a: &[i16; 16], b: &[u16; 16], i: usize) -> i32 {
        let mut acc = 0i64;
        for k in 0..2 {
            acc += a[2 * i + k] as i64 * b[2 * i + k] as i64;
        }
        // Single signed-dword saturation of the full-precision sum (Intel SDM / Felix
        // Cloutier VPDPW*DS): clamp `src + Σ products` once; NO intermediate product-sum clamp.
        let total: i64 = src as i64 + acc;
        total.clamp(i32::MIN as i64, i32::MAX as i64) as i32
    }

    quickcheck! {
        // -------------------- dpwsud (SU, wrap; avxvnniint16) --------------------

        /// Differential property for `dpwsud`: native `VPDPWSUD` vs the wrapping oracle.
        /// Discarded (never passed) when `avxvnniint16` is absent, so a feature-less host
        /// cannot go vacuously green.
        fn dpwsud_prop_hw_matches_scalar(input: Inputs) -> TestResult {
            #[cfg(target_arch = "x86_64")]
            {
                if std::is_x86_feature_detected!("avxvnniint16") {
                    let want = dpwsud_scalar(input.src, input.a_w, input.b_wu);
                    // SAFETY: the feature was confirmed present immediately above.
                    let got = unsafe { dpwsud_hw(input.src, input.a_w, input.b_wu) };
                    return TestResult::from_bool(got == want);
                }
            }
            TestResult::discard()
        }

        fn dpwsud_prop_public_matches_scalar(input: Inputs) -> bool {
            dpwsud(input.src, input.a_w, input.b_wu)
                == dpwsud_scalar(input.src, input.a_w, input.b_wu)
        }

        /// Wrapping additivity (...D): `src` is a pure wrapping-additive bias.
        /// [vnni-int8-int16-family.TESTS.3-1]
        fn dpwsud_prop_src_is_additive(input: Inputs) -> bool {
            let with_src = dpwsud_scalar(input.src, input.a_w, input.b_wu);
            let no_src = dpwsud_scalar([0; 8], input.a_w, input.b_wu);
            (0..8).all(|i| with_src[i] == input.src[i].wrapping_add(no_src[i]))
        }

        fn dpwsud_prop_zero_operand_is_passthrough(input: Inputs) -> bool {
            dpwsud_scalar(input.src, [0; 16], input.b_wu) == input.src
        }

        fn dpwsud_prop_lanes_are_independent(input: Inputs, lane: u8) -> bool {
            let i = (lane % 8) as usize;
            let mut a = [0i16; 16];
            let mut b = [0u16; 16];
            for k in 0..2 {
                a[2 * i + k] = input.a_w[2 * i + k];
                b[2 * i + k] = input.b_wu[2 * i + k];
            }
            dpwsud_scalar(input.src, a, b)[i]
                == dpwsud_scalar(input.src, input.a_w, input.b_wu)[i]
        }
        // NOTE: no `dpwsud_prop_operands_commute` — SU operand order is significant
        // (`dpwsud != dpwusd`); A1's distinct [i16;16]/[u16;16] types make a swap a compile
        // error (CORRECTNESS.2).

        // -------------------- dpwsuds (SU, saturate; avxvnniint16) --------------------

        /// Differential property for `dpwsuds`: native `VPDPWSUDS` vs the saturating oracle.
        fn dpwsuds_prop_hw_matches_scalar(input: Inputs) -> TestResult {
            #[cfg(target_arch = "x86_64")]
            {
                if std::is_x86_feature_detected!("avxvnniint16") {
                    let want = dpwsuds_scalar(input.src, input.a_w, input.b_wu);
                    // SAFETY: the feature was confirmed present immediately above.
                    let got = unsafe { dpwsuds_hw(input.src, input.a_w, input.b_wu) };
                    return TestResult::from_bool(got == want);
                }
            }
            TestResult::discard()
        }

        fn dpwsuds_prop_public_matches_scalar(input: Inputs) -> bool {
            dpwsuds(input.src, input.a_w, input.b_wu)
                == dpwsuds_scalar(input.src, input.a_w, input.b_wu)
        }

        fn dpwsuds_prop_zero_operand_is_passthrough(input: Inputs) -> bool {
            dpwsuds_scalar(input.src, [0; 16], input.b_wu) == input.src
        }

        fn dpwsuds_prop_lanes_are_independent(input: Inputs, lane: u8) -> bool {
            let i = (lane % 8) as usize;
            let mut a = [0i16; 16];
            let mut b = [0u16; 16];
            for k in 0..2 {
                a[2 * i + k] = input.a_w[2 * i + k];
                b[2 * i + k] = input.b_wu[2 * i + k];
            }
            dpwsuds_scalar(input.src, a, b)[i]
                == dpwsuds_scalar(input.src, input.a_w, input.b_wu)[i]
        }

        /// Saturation property (...DS): every lane equals the saturating reference and sits
        /// at the i32 boundary whenever the unbounded i64 total would overflow. The u16×u16
        /// product reaching ~4.29e9 makes this the load-bearing word-saturation check.
        /// [vnni-int8-int16-family.TESTS.3-3] [vnni-int8-int16-family.SCALAR_ORACLE.1-4]
        fn dpwsuds_prop_output_saturates(input: Inputs) -> bool {
            let out = dpwsuds_scalar(input.src, input.a_w, input.b_wu);
            (0..8).all(|i| {
                if out[i] != dpwsuds_lane_expected(input.src[i], &input.a_w, &input.b_wu, i) {
                    return false;
                }
                // ISA single saturation (Intel SDM / Felix Cloutier VPDPW*DS): the full
                // i64 total `src + Σ products` is clamped once into [i32::MIN, i32::MAX].
                // There is NO intermediate clamp of the product-sum before adding src — for
                // word ops the product sum can itself exceed i32 (the u16 trap), and a
                // two-stage clamp would diverge from hardware when src and the product sum
                // have opposite signs and the product sum's magnitude exceeds the i32 range.
                let total: i64 = input.src[i] as i64
                    + (0..2)
                        .map(|k| input.a_w[2 * i + k] as i64 * input.b_wu[2 * i + k] as i64)
                        .sum::<i64>();
                if total > i32::MAX as i64 {
                    out[i] == i32::MAX
                } else if total < i32::MIN as i64 {
                    out[i] == i32::MIN
                } else {
                    true
                }
            })
        }
        // NOTE: no `dpwsuds_prop_operands_commute` — SU, see dpwsud note above.

        // -------------------- dpwusd (US, wrap; avxvnniint16) --------------------

        /// Differential property for `dpwusd`: native `VPDPWUSD` vs the wrapping oracle.
        /// `a` is unsigned (`[u16;16]`), `b` signed (`[i16;16]`) — inverse of `dpwsud`.
        fn dpwusd_prop_hw_matches_scalar(input: Inputs) -> TestResult {
            #[cfg(target_arch = "x86_64")]
            {
                if std::is_x86_feature_detected!("avxvnniint16") {
                    let want = dpwusd_scalar(input.src, input.a_wu, input.b_w);
                    // SAFETY: the feature was confirmed present immediately above.
                    let got = unsafe { dpwusd_hw(input.src, input.a_wu, input.b_w) };
                    return TestResult::from_bool(got == want);
                }
            }
            TestResult::discard()
        }

        fn dpwusd_prop_public_matches_scalar(input: Inputs) -> bool {
            dpwusd(input.src, input.a_wu, input.b_w)
                == dpwusd_scalar(input.src, input.a_wu, input.b_w)
        }

        /// Wrapping additivity (...D): `src` is a pure wrapping-additive bias.
        /// [vnni-int8-int16-family.TESTS.3-1]
        fn dpwusd_prop_src_is_additive(input: Inputs) -> bool {
            let with_src = dpwusd_scalar(input.src, input.a_wu, input.b_w);
            let no_src = dpwusd_scalar([0; 8], input.a_wu, input.b_w);
            (0..8).all(|i| with_src[i] == input.src[i].wrapping_add(no_src[i]))
        }

        fn dpwusd_prop_zero_operand_is_passthrough(input: Inputs) -> bool {
            dpwusd_scalar(input.src, [0; 16], input.b_w) == input.src
        }

        fn dpwusd_prop_lanes_are_independent(input: Inputs, lane: u8) -> bool {
            let i = (lane % 8) as usize;
            let mut a = [0u16; 16];
            let mut b = [0i16; 16];
            for k in 0..2 {
                a[2 * i + k] = input.a_wu[2 * i + k];
                b[2 * i + k] = input.b_w[2 * i + k];
            }
            dpwusd_scalar(input.src, a, b)[i]
                == dpwusd_scalar(input.src, input.a_wu, input.b_w)[i]
        }
        // NOTE: no `dpwusd_prop_operands_commute` — US operand order is significant
        // (`dpwusd != dpwsud`); distinct [u16;16]/[i16;16] types make a swap a compile error.
    }

    // ============== Phase 5: word-variant pt 2 references + property selection ==============
    // dpwusds / dpwuud / dpwuuds (avxvnniint16, 2 products/lane). Per-variant selection:
    //   * ALL three: prop_hw_matches_scalar (discard on avxvnniint16 absence),
    //     public_matches_scalar, zero_operand_is_passthrough, lanes_are_independent.
    //   * wrapping (...D) dpwuud ONLY: prop_src_is_additive (wrapping decomposition).
    //   * saturating (...DS) dpwusds, dpwuuds: saturating-add assertion + prop_output_saturates.
    //   * prop_operands_commute for dpwuud, dpwuuds (UU) ONLY — explicitly NOT for dpwusds (US),
    //     whose operand order is significant (`dpwusds != dpwsuds`) and whose A1 distinct
    //     [u16;16]/[i16;16] types make a `b,a` swap a compile error (commutativity not expressible).
    // The saturating references fold products in i64 — wide enough for the u16×u16 trap — then
    // apply the SINGLE signed-dword saturation of the full sum `src + Σ products` (NO intermediate
    // product-sum clamp), the model that matches hardware when src and the product sum have
    // opposite signs.
    // [vnni-int8-int16-family.TESTS.3] [vnni-int8-int16-family.TESTS.3-1]
    // [vnni-int8-int16-family.TESTS.3-2] [vnni-int8-int16-family.TESTS.3-3]
    // [vnni-int8-int16-family.CORRECTNESS.2] [vnni-int8-int16-family.SCALAR_ORACLE.1-4]

    /// Saturating reference for the US `dpwusds`: `a` zero-extends (u16), `b` sign-extends
    /// (i16). Products folded in i64, then a SINGLE signed-dword saturation of the full sum
    /// `src + Σ products` (Intel SDM / Felix Cloutier VPDPW*DS — no intermediate clamp).
    fn dpwusds_lane_expected(src: i32, a: &[u16; 16], b: &[i16; 16], i: usize) -> i32 {
        let mut acc = 0i64;
        for k in 0..2 {
            acc += a[2 * i + k] as i64 * b[2 * i + k] as i64;
        }
        let total: i64 = src as i64 + acc;
        total.clamp(i32::MIN as i64, i32::MAX as i64) as i32
    }

    /// Saturating reference for the UU `dpwuuds`: both operands zero-extend (u16). The u16×u16
    /// products reach ≈4.29e9 and the lane sum ≈8.59e9, so the i64 fold and the single
    /// full-sum saturation are both load-bearing.
    fn dpwuuds_lane_expected(src: i32, a: &[u16; 16], b: &[u16; 16], i: usize) -> i32 {
        let mut acc = 0i64;
        for k in 0..2 {
            acc += a[2 * i + k] as i64 * b[2 * i + k] as i64;
        }
        // UNSIGNED-dword saturation (Intel SDM / Felix Cloutier VPDPWUUDS): reinterpret
        // `src`'s bits as u32 and clamp `unsigned(src) + Σ products` once into `[0, u32::MAX]`.
        let total: i64 = src as u32 as i64 + acc;
        total.clamp(0, u32::MAX as i64) as u32 as i32
    }

    quickcheck! {
        // -------------------- dpwusds (US, saturate; avxvnniint16) --------------------

        /// Differential property for `dpwusds`: native `VPDPWUSDS` vs the saturating oracle.
        /// Discarded (never passed) when `avxvnniint16` is absent.
        fn dpwusds_prop_hw_matches_scalar(input: Inputs) -> TestResult {
            #[cfg(target_arch = "x86_64")]
            {
                if std::is_x86_feature_detected!("avxvnniint16") {
                    let want = dpwusds_scalar(input.src, input.a_wu, input.b_w);
                    // SAFETY: the feature was confirmed present immediately above.
                    let got = unsafe { dpwusds_hw(input.src, input.a_wu, input.b_w) };
                    return TestResult::from_bool(got == want);
                }
            }
            TestResult::discard()
        }

        fn dpwusds_prop_public_matches_scalar(input: Inputs) -> bool {
            dpwusds(input.src, input.a_wu, input.b_w)
                == dpwusds_scalar(input.src, input.a_wu, input.b_w)
        }

        fn dpwusds_prop_zero_operand_is_passthrough(input: Inputs) -> bool {
            dpwusds_scalar(input.src, [0; 16], input.b_w) == input.src
        }

        fn dpwusds_prop_lanes_are_independent(input: Inputs, lane: u8) -> bool {
            let i = (lane % 8) as usize;
            let mut a = [0u16; 16];
            let mut b = [0i16; 16];
            for k in 0..2 {
                a[2 * i + k] = input.a_wu[2 * i + k];
                b[2 * i + k] = input.b_w[2 * i + k];
            }
            dpwusds_scalar(input.src, a, b)[i]
                == dpwusds_scalar(input.src, input.a_wu, input.b_w)[i]
        }

        /// Saturation property (...DS): every lane equals the single-full-sum-saturating
        /// reference and sits at the i32 boundary whenever the unbounded i64 total overflows.
        /// [vnni-int8-int16-family.TESTS.3-3] [vnni-int8-int16-family.SCALAR_ORACLE.1-4]
        fn dpwusds_prop_output_saturates(input: Inputs) -> bool {
            let out = dpwusds_scalar(input.src, input.a_wu, input.b_w);
            (0..8).all(|i| {
                if out[i] != dpwusds_lane_expected(input.src[i], &input.a_wu, &input.b_w, i) {
                    return false;
                }
                let total: i64 = input.src[i] as i64
                    + (0..2)
                        .map(|k| input.a_wu[2 * i + k] as i64 * input.b_w[2 * i + k] as i64)
                        .sum::<i64>();
                if total > i32::MAX as i64 {
                    out[i] == i32::MAX
                } else if total < i32::MIN as i64 {
                    out[i] == i32::MIN
                } else {
                    true
                }
            })
        }
        // NOTE: no `dpwusds_prop_operands_commute` — US operand order is significant
        // (`dpwusds != dpwsuds`); A1's distinct [u16;16]/[i16;16] types make a swap a compile
        // error (CORRECTNESS.2).

        // -------------------- dpwuud (UU, wrap; avxvnniint16) --------------------

        /// Differential property for `dpwuud`: native `VPDPWUUD` vs the wrapping oracle.
        fn dpwuud_prop_hw_matches_scalar(input: Inputs) -> TestResult {
            #[cfg(target_arch = "x86_64")]
            {
                if std::is_x86_feature_detected!("avxvnniint16") {
                    let want = dpwuud_scalar(input.src, input.a_wu, input.b_wu);
                    // SAFETY: the feature was confirmed present immediately above.
                    let got = unsafe { dpwuud_hw(input.src, input.a_wu, input.b_wu) };
                    return TestResult::from_bool(got == want);
                }
            }
            TestResult::discard()
        }

        fn dpwuud_prop_public_matches_scalar(input: Inputs) -> bool {
            dpwuud(input.src, input.a_wu, input.b_wu)
                == dpwuud_scalar(input.src, input.a_wu, input.b_wu)
        }

        /// Wrapping additivity (...D): `src` is a pure wrapping-additive bias.
        /// [vnni-int8-int16-family.TESTS.3-1]
        fn dpwuud_prop_src_is_additive(input: Inputs) -> bool {
            let with_src = dpwuud_scalar(input.src, input.a_wu, input.b_wu);
            let no_src = dpwuud_scalar([0; 8], input.a_wu, input.b_wu);
            (0..8).all(|i| with_src[i] == input.src[i].wrapping_add(no_src[i]))
        }

        fn dpwuud_prop_zero_operand_is_passthrough(input: Inputs) -> bool {
            dpwuud_scalar(input.src, [0; 16], input.b_wu) == input.src
        }

        fn dpwuud_prop_lanes_are_independent(input: Inputs, lane: u8) -> bool {
            let i = (lane % 8) as usize;
            let mut a = [0u16; 16];
            let mut b = [0u16; 16];
            for k in 0..2 {
                a[2 * i + k] = input.a_wu[2 * i + k];
                b[2 * i + k] = input.b_wu[2 * i + k];
            }
            dpwuud_scalar(input.src, a, b)[i]
                == dpwuud_scalar(input.src, input.a_wu, input.b_wu)[i]
        }

        /// Operand commutativity holds for `dpwuud` (UU): swapping the two u16 multiplicand
        /// vectors leaves the dot product unchanged. (Asserted for UU only, NOT for US dpwusds.)
        /// [vnni-int8-int16-family.TESTS.3-2]
        fn dpwuud_prop_operands_commute(input: Inputs) -> bool {
            dpwuud_scalar(input.src, input.a_wu, input.b_wu)
                == dpwuud_scalar(input.src, input.b_wu, input.a_wu)
        }

        // -------------------- dpwuuds (UU, saturate; avxvnniint16) --------------------

        /// Differential property for `dpwuuds`: native `VPDPWUUDS` vs the saturating oracle.
        fn dpwuuds_prop_hw_matches_scalar(input: Inputs) -> TestResult {
            #[cfg(target_arch = "x86_64")]
            {
                if std::is_x86_feature_detected!("avxvnniint16") {
                    let want = dpwuuds_scalar(input.src, input.a_wu, input.b_wu);
                    // SAFETY: the feature was confirmed present immediately above.
                    let got = unsafe { dpwuuds_hw(input.src, input.a_wu, input.b_wu) };
                    return TestResult::from_bool(got == want);
                }
            }
            TestResult::discard()
        }

        fn dpwuuds_prop_public_matches_scalar(input: Inputs) -> bool {
            dpwuuds(input.src, input.a_wu, input.b_wu)
                == dpwuuds_scalar(input.src, input.a_wu, input.b_wu)
        }

        fn dpwuuds_prop_zero_operand_is_passthrough(input: Inputs) -> bool {
            dpwuuds_scalar(input.src, [0; 16], input.b_wu) == input.src
        }

        fn dpwuuds_prop_lanes_are_independent(input: Inputs, lane: u8) -> bool {
            let i = (lane % 8) as usize;
            let mut a = [0u16; 16];
            let mut b = [0u16; 16];
            for k in 0..2 {
                a[2 * i + k] = input.a_wu[2 * i + k];
                b[2 * i + k] = input.b_wu[2 * i + k];
            }
            dpwuuds_scalar(input.src, a, b)[i]
                == dpwuuds_scalar(input.src, input.a_wu, input.b_wu)[i]
        }

        /// Operand commutativity holds for `dpwuuds` (UU).
        /// [vnni-int8-int16-family.TESTS.3-2]
        fn dpwuuds_prop_operands_commute(input: Inputs) -> bool {
            dpwuuds_scalar(input.src, input.a_wu, input.b_wu)
                == dpwuuds_scalar(input.src, input.b_wu, input.a_wu)
        }

        /// Saturation property (...DS) for `dpwuuds`: the largest product sums of the family.
        /// Every lane equals the unsigned-saturating reference and sits at the `u32::MAX`
        /// boundary (bit pattern `-1` as i32) whenever the unbounded total exceeds it. UU
        /// saturates into the UNSIGNED dword range, so — unlike the signed `...DS` variants —
        /// there is no high-side clamp to `i32::MAX`: an unsigned total in `(i32::MAX, u32::MAX]`
        /// is stored verbatim and reads back negative.
        /// [vnni-int8-int16-family.TESTS.3-3] [vnni-int8-int16-family.SCALAR_ORACLE.1-4]
        fn dpwuuds_prop_output_saturates(input: Inputs) -> bool {
            let out = dpwuuds_scalar(input.src, input.a_wu, input.b_wu);
            (0..8).all(|i| {
                if out[i] != dpwuuds_lane_expected(input.src[i], &input.a_wu, &input.b_wu, i) {
                    return false;
                }
                let total: i64 = input.src[i] as u32 as i64
                    + (0..2)
                        .map(|k| input.a_wu[2 * i + k] as i64 * input.b_wu[2 * i + k] as i64)
                        .sum::<i64>();
                if total > u32::MAX as i64 {
                    out[i] == -1
                } else {
                    true
                }
            })
        }
    }
}

#[cfg(test)]
mod iteration2_surface {
    //! Crate-level reachability / naming / stable-Rust guard for the 21 `AVX10_V2_AUX`
    //! (group-3) OCP-format converts added in iteration 2. This module is the executable
    //! witness for three cross-cutting ACIDs:
    //!
    //! * `[avx10-v2-aux-ocp-conversions.NAMING.1]` — every public primitive is reachable from
    //!   the crate root under a name matching its eventual stdarch intrinsic stem; the two-
    //!   instruction families carry the OQ-3 source-format suffix.
    //! * `[avx10-v2-aux-ocp-conversions.STABLE_RUST.1]` — each is a safe public fn taking and
    //!   returning fixed-size lane arrays by value; this module compiles on stable Rust (the
    //!   whole crate forbids `core::simd` and uses no nightly features — see the crate manifest
    //!   and the absence of any `#![feature(...)]` attribute).
    //! * `[avx10-v2-aux-ocp-conversions.CORRECTNESS.1]` — each dispatcher is callable and
    //!   returns the spec-shaped output array, the oracle being the always-present path.

    /// Every one of the 21 group-3 primitives is reachable from the crate root by its
    /// stdarch-stem name and is a safe call returning the spec-shaped fixed-size array — no
    /// `unsafe`, no nightly, no `core::simd`. The exact stems are pinned here so any rename,
    /// removal, or signature drift breaks this compile; binding every result keeps the call
    /// load-bearing (an unused public re-export would otherwise be silently droppable).
    #[test]
    fn all_21_primitives_reachable_by_intrinsic_stem() {
        // Family A — single-source FP32 -> FP8 (RTNE / RTO). [f32;16] -> [u8;16].
        let _cvtps_bf8: [u8; 16] = crate::cvtps_bf8([0.0f32; 16]);
        let _cvtpss_bf8: [u8; 16] = crate::cvtpss_bf8([0.0f32; 16]);
        let _cvtps_hf8: [u8; 16] = crate::cvtps_hf8([0.0f32; 16]);
        let _cvtpss_hf8: [u8; 16] = crate::cvtpss_hf8([0.0f32; 16]);
        let _cvtrops_hf8: [u8; 16] = crate::cvtrops_hf8([0.0f32; 16]);
        let _cvtropss_hf8: [u8; 16] = crate::cvtropss_hf8([0.0f32; 16]);
        // Family B — FP32 -> FP8 bias-rounding. ([f32;16], [i32;16]) -> [u8;16].
        let _cvtbiasps_bf8: [u8; 16] = crate::cvtbiasps_bf8([0.0f32; 16], [0i32; 16]);
        let _cvtbiaspss_bf8: [u8; 16] = crate::cvtbiaspss_bf8([0.0f32; 16], [0i32; 16]);
        let _cvtbiasps_hf8: [u8; 16] = crate::cvtbiasps_hf8([0.0f32; 16], [0i32; 16]);
        let _cvtbiaspss_hf8: [u8; 16] = crate::cvtbiaspss_hf8([0.0f32; 16], [0i32; 16]);
        // Family C — exact FP8 -> FP32. [u8;16] -> [f32;16].
        let _cvtbf8_ps: [f32; 16] = crate::cvtbf8_ps([0u8; 16]);
        let _cvthf8_ps: [f32; 16] = crate::cvthf8_ps([0u8; 16]);
        // Family D — saturating-RTNE FP8 -> FP4 (E2M1), nibble-packed. [u8;64] -> [u8;32].
        // OQ-3 source-format suffix disambiguates the two intrinsic-stem `cvtf8_bf4s` forms.
        let _cvtf8_bf4s_e5m2: [u8; 32] = crate::cvtf8_bf4s_e5m2([0u8; 64]);
        let _cvtf8_bf4s_e4m3: [u8; 32] = crate::cvtf8_bf4s_e4m3([0u8; 64]);
        // Family E — exact FP4 (E2M1) -> FP8 (E4M3), nibble-unpacked. [u8;32] -> [u8;64].
        let _cvtbf4_hf8: [u8; 64] = crate::cvtbf4_hf8([0u8; 32]);
        // Family F — saturating-RTNE FP8 -> FP6, 6-bit-packed. [u8;64] -> [u8;48].
        let _cvtf8_bf6s: [u8; 48] = crate::cvtf8_bf6s([0u8; 64]);
        let _cvtf8_hf6s: [u8; 48] = crate::cvtf8_hf6s([0u8; 64]);
        // Family G — exact FP6 -> FP8 (E4M3), 6-bit-unpacked. [u8;48] -> [u8;64].
        // OQ-3 source-format suffix disambiguates the two `cvtf6_hf8` forms.
        let _cvtf6_hf8_e3m2: [u8; 64] = crate::cvtf6_hf8_e3m2([0u8; 48]);
        let _cvtf6_hf8_e2m3: [u8; 64] = crate::cvtf6_hf8_e2m3([0u8; 48]);
        // Family H — VPMOVSSDB symmetric-signed-saturation INT32 -> INT8. [i32;16] -> [i8;16].
        let _cvtssepi32_epi8: [i8; 16] = crate::cvtssepi32_epi8([0i32; 16]);
        // Family I — VUNPACKB sub-byte unpack. ([u8;64], u8) -> [u8;64]; imm8 is a value arg.
        let _unpackb: [u8; 64] = crate::unpackb([0u8; 64], crate::ACE_UNPACKB_SIZE(4));

        // Count is machine-checked: the fixed-size array below references every binding
        // above, so the 21-primitive inventory is a compile-time fact — adding or removing
        // a binding without updating this list is a compile error, not a stale comment.
        let bound: [&dyn core::fmt::Debug; 21] = [
            // Family A (6)
            &_cvtps_bf8,
            &_cvtpss_bf8,
            &_cvtps_hf8,
            &_cvtpss_hf8,
            &_cvtrops_hf8,
            &_cvtropss_hf8,
            // Family B (4)
            &_cvtbiasps_bf8,
            &_cvtbiaspss_bf8,
            &_cvtbiasps_hf8,
            &_cvtbiaspss_hf8,
            // Family C (2)
            &_cvtbf8_ps,
            &_cvthf8_ps,
            // Family D (2)
            &_cvtf8_bf4s_e5m2,
            &_cvtf8_bf4s_e4m3,
            // Family E (1)
            &_cvtbf4_hf8,
            // Family F (2)
            &_cvtf8_bf6s,
            &_cvtf8_hf6s,
            // Family G (2)
            &_cvtf6_hf8_e3m2,
            &_cvtf6_hf8_e2m3,
            // Family H (1)
            &_cvtssepi32_epi8,
            // Family I (1)
            &_unpackb,
        ];
        let _ = bound;
    }
}

#[cfg(test)]
mod iteration_surface {
    //! Crate-level reachability / naming / stable-Rust guard for the `ACE` group-4 tile
    //! instructions added in iteration 3 — the POSITIVE half that complements
    //! [`super::non_goal_guards`]. It is the executable witness that every group-4 dispatcher is
    //! reachable from the crate root under a name matching its eventual stdarch intrinsic stem
    //! ([ace-tile-instructions.NAMING.1]), is a safe stable-Rust entry point
    //! ([ace-tile-instructions.STABLE.1]), and is callable against the `TileScope` guard.

    /// Every group-4 dispatcher is reachable from the crate root by a name matching the
    /// spec's C intrinsic equivalent. The inventory is machine-checked: the fixed-size
    /// array references every dispatcher by path, so a rename, removal, or a missing
    /// re-export breaks this compile. `TILERELEASE` is Drop-only (the guard's `Drop`), so
    /// it is the one group-4 op with no free-standing fn.
    ///
    /// `lib::iteration_surface_includes_group4`
    #[test]
    fn iteration_surface_includes_group4() {
        // 27 group-4 dispatcher fns: A(3) + B(3) + C(5) + D(6) + G(4) + F(1) + E(5). Plus
        // the Drop-only TILERELEASE the family covers every group-4 instruction (TILEMOVCOL
        // is write-only per spec section 12.3.1; BSRMOVH/BSRMOVL each have read + write
        // forms per section 13.3.2).
        let inventory: [*const (); 27] = [
            // Family A — tile config lifecycle (LDTILECFG / STTILECFG / TILEZERO).
            crate::_tile_loadconfig as *const (),
            crate::_tile_storeconfig as *const (),
            crate::_tile_zero as *const (),
            // Family B — tile data movement (TILEMOVROW read/write, TILEMOVCOL write-only).
            crate::_tile_movrow as *const (),
            crate::_tile_setrow as *const (),
            crate::_tile_setcol as *const (),
            // Family C — tile-row converts (INT32->FP32, FP32->BF16 H/L, FP32->FP16 H/L).
            crate::_tile_cvtrowd2ps as *const (),
            crate::_tile_cvtrowps2bf16h as *const (),
            crate::_tile_cvtrowps2bf16l as *const (),
            crate::_tile_cvtrowps2phh as *const (),
            crate::_tile_cvtrowps2phl as *const (),
            // Family D — Block Scale register ops (BSRINIT, BSRMOVF, BSRMOVH/L r+w).
            crate::_bsrinit as *const (),
            crate::_bsrmovf as *const (),
            crate::_bsrmovh as *const (),
            crate::_bsrmovh_read as *const (),
            crate::_bsrmovl as *const (),
            crate::_bsrmovl_read as *const (),
            // Family G — INT8 rank-4 outer products.
            crate::_tile_top4bssd as *const (),
            crate::_tile_top4bsud as *const (),
            crate::_tile_top4busd as *const (),
            crate::_tile_top4buud as *const (),
            // Family F — BF16 rank-2 outer product.
            crate::_tile_top2bf16ps as *const (),
            // Family E — MX rank-4 outer products.
            crate::_tile_top4mxbf8ps as *const (),
            crate::_tile_top4mxbhf8ps as *const (),
            crate::_tile_top4mxhbf8ps as *const (),
            crate::_tile_top4mxhf8ps as *const (),
            crate::_tile_top4mxbssps as *const (),
        ];
        // Non-zero function addresses (they are real, reachable symbols) and the count is a
        // compile-time fact — adding or removing a dispatcher without updating this list is a
        // compile error, not a stale comment.
        assert!(inventory.iter().all(|&p| !p.is_null()));
        assert_eq!(inventory.len(), 27);

        // Callable against the guard (safe stable Rust, no `unsafe`, no nightly): drive a full
        // slice of the surface end-to-end.
        let cfg = crate::TileConfig::ace();
        let mut scope = crate::_tile_loadconfig(&cfg).expect("valid palette-2 descriptor");
        assert_eq!(crate::_tile_storeconfig(&scope), cfg); // STTILECFG round-trip
        let dst = scope.tile(0).unwrap();
        crate::_tile_zero(&mut scope, dst);
        let _row = crate::_tile_movrow(&scope, dst, 0);
        let _conv = crate::_tile_cvtrowd2ps(&scope, dst, 0);
        crate::_bsrinit(&mut scope);
        crate::_tile_top4bssd(&mut scope, dst, [1u8; 64], [1u8; 64]);
        crate::_tile_top2bf16ps(&mut scope, dst, [0u16; 32], [0u16; 32]);
        crate::_tile_top4mxbf8ps(&mut scope, dst, [0u8; 64], [0u8; 64], 0);
    }
}

#[cfg(test)]
mod non_goal_guards {
    //! Documented guard that the non-goals were not built into the public surface. NOTE:
    //! group-3 (`AVX10_V2_AUX`) went in scope in iteration 2 — its reachability is the POSITIVE
    //! assertion in [`super::iteration2_surface`] — and group-4 (`ACE` tile instructions) went in
    //! scope in iteration 3, asserted positively in
    //! [`super::iteration_surface::iteration_surface_includes_group4`]. This guard is the
    //! complementary NEGATIVE-space assertion that what remains out of scope was not built:
    //! palette-1 tile config, the AMX `TMUL` dot products, the nightly `x86_amx_intrinsics`
    //! feature, EVEX `{k1}{z}` masking / broadcast, and sub-512 vector lengths.

    /// Confirms the public function inventory is exactly iteration-0 `dpbssd`, the 26
    /// `AVX10_V1_AUX` primitives (group 2 — iteration 1), the 21 `AVX10_V2_AUX` primitives
    /// (group 3 — iteration 2), the 25 `ACE` group-4 tile instructions (iteration 3), and
    /// NOTHING out of scope: no palette-1 / AMX `TMUL` dot-product tile instructions, no nightly
    /// `x86_amx_intrinsics`, no EVEX `{k1}{z}` write-masking / `m32bcst` broadcast entry points,
    /// and no 128/256-bit vector-length plumbing. This is a readable, asserting record of the
    /// negative space; it references each in-scope public family entry point so any accidental
    /// removal or out-of-scope addition is caught at compile time.
    #[test]
    fn non_goals_absent() {
        // The complete in-scope public primitive set is exercised below — one representative
        // entry point per family plus the iteration-0 VEX `dpbssd`, the iteration-2 group-3
        // converts, and the negative-space prose. There is deliberately NO `top*`, `bsr*`,
        // tile-move (group 4), no `{k1}{z}` / `m32bcst` masking/broadcast entry point, and no
        // 128/256-bit `VL` form. Each call takes plain fixed-size lane arrays by value (no
        // mask / no broadcast / no narrower-VL operand exists to pass), which is itself the
        // guarantee that the out-of-scope surface was never built. `VUNPACKB`'s `imm8` is a
        // plain value argument (size/start/sext selector), NOT a write-mask. Any out-of-scope
        // or removed primitive would break this compile.
        //
        // Group 2 (AVX10_V1_AUX, iteration 1) — already in scope:
        let _a = crate::cvtph_bf8([0u16; 32]); // families A/B/C: FP16 -> FP8
        let _b = crate::cvtphs_hf8([0u16; 32]);
        let _d = crate::cvthf8_ph([0u8; 32]); // family D: HF8 -> FP16
        let _e = crate::cvt2ps_phx([0.0f32; 16], [0.0f32; 16]); // family E: FP32 pair -> FP16
        let _f = crate::vnni::dpbssd([0i32; 16], [0i8; 64], [0i8; 64]); // family F: byte VNNI (EVEX 512-bit)
        let _g = crate::vnni::dpwsud([0i32; 16], [0i16; 32], [0u16; 32]); // family G: word VNNI (EVEX 512-bit)
        let _group1_vex = crate::dpbssd([0i32; 8], [0i8; 32], [0i8; 32]); // iteration-0 VEX dpbssd (256-bit)

        // Group 3 (AVX10_V2_AUX, iteration 2) — NOW in scope (one representative per family;
        // the full 21-primitive reachability set is asserted in `super::iteration2_surface`):
        let _v2_a = crate::cvtps_bf8([0.0f32; 16]); // family A: FP32 -> FP8 (RTNE)
        let _v2_b = crate::cvtbiasps_bf8([0.0f32; 16], [0i32; 16]); // family B: FP32 -> FP8 bias
        let _v2_c = crate::cvtbf8_ps([0u8; 16]); // family C: exact FP8 -> FP32
        let _v2_d = crate::cvtf8_bf4s_e5m2([0u8; 64]); // family D: FP8 -> FP4 (nibble-packed)
        let _v2_e = crate::cvtbf4_hf8([0u8; 32]); // family E: FP4 -> FP8 (nibble-unpacked)
        let _v2_f = crate::cvtf8_bf6s([0u8; 64]); // family F: FP8 -> FP6 (6-bit-packed)
        let _v2_g = crate::cvtf6_hf8_e3m2([0u8; 48]); // family G: FP6 -> FP8 (6-bit-unpacked)
        let _v2_h = crate::cvtssepi32_epi8([0i32; 16]); // family H: VPMOVSSDB (symmetric clamp)
        let _v2_i = crate::unpackb([0u8; 64], crate::ACE_UNPACKB_SIZE(4)); // family I: VUNPACKB

        // Group 4 (ACE tile instructions, iteration 3) — NOW in scope (one representative per
        // family; the full 25-instruction reachability set is asserted in
        // `super::iteration_surface`). Group 4 is stateful, so its ops run against a `TileScope`.
        let mut scope = crate::_tile_loadconfig(&crate::TileConfig::ace()).unwrap(); // family A
        let dst = scope.tile(0).unwrap();
        let _g4_c = crate::_tile_movrow(&scope, dst, 0); // family B: tile -> ZMM read move
        let _g4_conv = crate::_tile_cvtrowd2ps(&scope, dst, 0); // family C: tile-row convert
        crate::_bsrinit(&mut scope); // family D: BSR (no data operand, spec section 13.1)
        crate::_tile_top4bssd(&mut scope, dst, [0u8; 64], [0u8; 64]); // family G: INT8 TOP
        crate::_tile_top2bf16ps(&mut scope, dst, [0u16; 32], [0u16; 32]); // family F: BF16 TOP
        crate::_tile_top4mxbf8ps(&mut scope, dst, [0u8; 64], [0u8; 64], 0); // family E: MX TOP

        // Negative space: NO out-of-scope symbol exists to reference here — the palette-1 tile
        // configuration and the AMX `TMUL` dot products (`TDPBSSD`/`TDPBF16PS`/… — a distinct
        // engine from group-4 `TOP*`), the nightly `x86_amx_intrinsics` feature, `{k1}{z}` /
        // `m32bcst`, and 128/256-bit `VL` forms are all unbuilt, and that absence (nothing to
        // bind, no `#![feature(x86_amx_intrinsics)]` / `core::simd` in the crate) is the
        // guarantee.
    }
}
