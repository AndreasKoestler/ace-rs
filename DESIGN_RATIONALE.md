# `ace-rs` — Design Rationale

Status: draft. Scope: an out-of-tree Rust crate exposing the x86 **AI Compute Extensions (ACE)** as usable primitives *before* hardware ships and *before* the intrinsics land in the standard library.

Spec: *ACE v1 Specification*, rev 1.15 (May 2026), joint Intel/AMD, x86 Ecosystem Advisory Group. Referenced below as **[spec §x]**. Companion: *ACE Whitepaper v1* (Apr 2026).

Every decision states how the standard library handles the same concern, under **core:: reference**. Code paths are given relative to the `stdarch` tree (`rust-lang/stdarch`, vendored in `rust-lang/rust` at `library/stdarch/`); browse via `doc.rust-lang.org/src/core/stdarch/...`.

---

## 1. What ACE is

ACE adds matrix-multiplication primitives and reduced-precision ML data formats on an **AVX10.1 baseline**, reusing the **AMX tile register file** [spec §2.1–2.2]. It is split into four independently-enumerated feature groups [spec §4]:

| Group | Feature flag (CPUID) | Encoding | Content | core:: status today |
|-------|----------------------|----------|---------|---------------------|
| 1. AVX-VNNI-INT8/16 | `AVX-VNNI-INT8` Fn7/1 EDX[4]; `-INT16` EDX[10] | VEX | `VPDPB*`, `VPDPW*` | **In `core::arch`, stable** (`avxvnniint8`) |
| 2. AVX10.2 subset | `AVX10_V1_AUX` Fn24/1 ECX[2] | EVEX | FP16→FP8 convert, EVEX VNNI | Partially (AVX10.2 enablement in progress) |
| 3. OCP conversions | `AVX10_V2_AUX` Fn24/1 ECX[3] | EVEX | FP32↔FP8, FP8↔FP4/FP6, `VPMOVSSDB`, `VUNPACKB` | Not yet |
| 4. ACE tile | `ACE` Fn7/1 ECX[11] + `ACE_VSN≥1` Fn1D/2 + `AMX-TILE` | EVEX/tile | Outer products (`TOP*`), tile moves, `BSR*`, palette 2 | Not yet |

Full detection requires the layered check in [spec §3.2]: `(AVX10.1 ∧ AVX10_V1_AUX) ∨ AVX10.2`, plus `AVX10_V2_AUX`, `ACE`, `ACE_VSN≥1`, and the XSAVE/`XCR0`/`CR4.OSXSAVE` state bits.

**core:: reference:** the equivalent feature flags are surfaced by the detection macro in `std_detect` (`crates/std_detect/src/detect/`), e.g. `is_x86_feature_detected!("avxvnniint8")`. `std_detect` does not yet define `ace`/`avx10.2`-tile tokens; until it does, `ace-rs` runs the CPUID checks itself.

---

## 2. Constraints

1. The crate is **publishable on crates.io and compiles on stable Rust**, with a stable public API.
2. It is **deprecated once ACE is upstreamed** into `core::arch`.

ACE groups 2–4 are not natively reachable from a stable out-of-tree crate today — not in shipping silicon, not in stable `core::arch`, and AVX-512 ZMM / AMX TMM operands have no stable `asm!` register class. So the **scalar fallback is the primary path**: it produces correct results everywhere now. Native execution is layered on as a gradient (fallback → opt-in accelerated → real hardware) as toolchains and silicon catch up.

---

## 3. Architecture

Three layers per primitive, plus a sunset hook. The shape deliberately mirrors the standard library's own split.

```
public safe fn  cvtps_hf8(a) -> r      // names mirror the future stdarch intrinsic
  ├─ (1) cfg(ace_in_stdarch)  -> core::arch::x86_64::_mm512_cvtps_hf8   // auto when upstream lands
  ├─ (2) feature="native" + detected -> backend::…                     // opt-in, emits the real insn
  └─ (3) fallback::…                                                    // universal, stable, correct
```

---

## 4. Design decisions

### D1 — Mirror `core::arch`'s contract: gated raw intrinsic, no built-in fallback

The lowest layer is an `unsafe`, `#[target_feature(enable=…)]` function that is *only* the instruction. It assumes the feature is present; calling it otherwise is UB.

**Rationale:** matches the layer ACE will eventually occupy, so upstreaming is a forwarding swap, not a rewrite. Keeps the hot path branch-free.

**core:: reference:** this is exactly `core::arch`'s policy. `crates/core_arch/src/x86/avx.rs` carries **187 `#[target_feature]` gates and zero `is_x86_feature_detected!` calls in its implementations**; every fn is the bare op, e.g.

```rust
#[target_feature(enable = "avx")]
pub const fn _mm256_div_ps(a: __m256, b: __m256) -> __m256 { unsafe { simd_div(a, b) } }
```

The `core::arch` module docs state calling an intrinsic on an unsupporting CPU is **undefined behavior** and that "this module is intended to be a low-level implementation detail for higher-level APIs." `ace-rs`'s raw layer adopts the identical contract.

### D2 — `ace-rs` owns the detection + dispatch + fallback layer

The safe public fn does `is_x86_feature_detected!` → native, else scalar fallback.

**Rationale:** `core::arch` supplies *only* the gated intrinsic and explicitly pushes detection onto the caller. The dispatch+fallback is the value a higher-level crate adds — not duplication. This value **survives upstreaming**: `core::arch` will *never* add a fallback, so consumers wanting graceful degradation always need this layer.

**core:: reference:** the `core::arch` docs' own recommended pattern names the fallback as the caller's:
```rust
if is_x86_feature_detected!("avx2") { return unsafe { foo_avx2() }; }  // core::arch supplies this
foo_fallback()                                                          // caller supplies this (== ace-rs)
```
Detection macro: `std::arch::is_x86_feature_detected!` (impl in `crates/std_detect/`). Prior art for the dispatch idiom: `multiversion`, `pulp`.

### D3 — Public signatures use `core::arch` vector types, not `core::simd`

Inputs/outputs are `__m512`, `__m256i`, `__tile1024i`, etc.

**Rationale:** these types are **stable** and are the eventual upstream signature, so the deprecation swap is type-identical. `core::simd` is nightly-gated and would forfeit stable compilation.

**core:: reference:** `__mNNN` are defined in `core::arch` (`crates/core_arch/src/x86/mod.rs`), stable since 1.27. `core::simd::Simd<T,N>` (`crates/core_simd/`) is gated behind `feature(portable_simd)` (tracking #86656) — nightly only.

### D4 — Do not depend on `core::simd`; bridge optionally via `transmute`

No dependency on the portable SIMD library. If a consumer wants `Simd<T,N>` ergonomics, expose layout-compatible conversions.

**Rationale:** ACE intrinsics do not require `core::simd`, and neither does the standard library's own intrinsic layer. Keeping the dependency out preserves stable-Rust compilation.

**core:: reference:** `crates/core_arch/src/x86/avx.rs` imports `crate::core_arch::simd` (its *internal* vector module, `crates/core_arch/src/simd.rs`) and `crate::intrinsics::simd::*` (the rustc-internal generic ops `simd_add`, `simd_shuffle`, …) — **never** `core::simd`. Grep of `avx.rs` for `core::simd`/`core_simd`: zero hits. `core_arch` and `core_simd` are *siblings* over the same `intrinsics::simd` foundation, not a dependency chain. The `transmute` bridge is sound because both `__m512` and `Simd<f32,16>` are `#[repr(simd)]` with identical layout. (Note: `core_arch::simd` and `intrinsics::simd` are crate-private to stdarch and unavailable out-of-tree — another reason `ace-rs` builds on the public `__mNNN` types only.)

### D5 — Name every public item after its eventual stdarch intrinsic

`cvtps_hf8` mirrors `_mm512_cvtps_hf8`; `tile_top4mxhf8ps` mirrors `_tile_top4mxhf8ps`.

**Rationale:** makes deprecation mechanical — consumers migrate by changing an import.

**core:: reference:** target names are the intrinsic equivalents the spec already publishes [spec §8.x, §9.x, §14.x "C/C++ Compiler Intrinsic Equivalent"], which are the names stdarch will use in `core::arch::x86_64` (cf. existing `core::arch::x86_64::_mm512_*` and the AMX `_tile_*` in `crates/core_arch/src/x86_64/amx.rs`).

### D6 — Sunset hook: `build.rs` autocfg probe → forward to `core::arch`

A build probe compile-tests whether `core::arch::x86_64::_mm512_cvtps_hf8` (etc.) exists and sets `cfg(ace_in_stdarch)`. When set, the safe fn forwards to the real intrinsic.

**Rationale:** the day stdarch ships ACE, the crate forwards automatically with no consumer change; the release that adds `#[deprecated(note="use core::arch::x86_64::…")]` keeps the dispatch/fallback (D2) intact.

**core:: reference:** forwards to `core::arch::x86_64::_*`. Probe mirrors the `autocfg`/`cfg(accessible(...))`-style detection pattern; the deprecated wrappers continue to wrap `is_x86_feature_detected!` because `core::arch` itself never gates.

### D7 — Native backend: C/asm stub via `cc`, with a `.byte` escape hatch; not stable `asm!`

The opt-in `native` feature compiles a C/assembly translation unit (using `<immintrin.h>` intrinsics from [spec §8/§9/§14], or mnemonics) and links it via FFI. Instructions no assembler knows yet use localized `.byte` raw encodings.

**Rationale:** as of mid-2026, **GNU binutils 2.44** (gas) supports AVX10.2 and "Diamond Rapids instructions" (the first ACE part), and **Intel SDE 10.8** emulates AVX10.2 — so real mnemonics/intrinsics are usable now and the C compiler allocates ZMM/TMM registers for us. The Rust side stays 100% stable (plain FFI). Pure-Rust `asm!` is rejected as the default because there is **no stable ZMM/TMM operand register class**, forcing memory-operand workarounds and hand-encoding.

**core:: reference:** `core::arch` does **not** use inline `asm!` for this — it binds LLVM intrinsics directly. Verified counts: `amx.rs` has **0** `asm!` and **60** `#[link_name="llvm.x86.*"]` bindings; `avx.rs` has **3** `asm!` vs **35** LLVM bindings and **528** generic `simd_*` calls. Every tile op (`_tile_loadconfig`, `_tile_dpbf8ps`, …) lowers via an `extern` block with `#[link_name="llvm.x86.*"]`. That mechanism is **in-tree/nightly-only** — `link_llvm_intrinsics` and `intrinsics::simd::*` are perma-unstable, unavailable to a stable out-of-tree crate. The **C stub (D7) is the stable out-of-tree proxy for it**: the C compiler emits the same LLVM intrinsic/instruction, reached via FFI instead of `link_name`. Stable `core::arch::asm!`/`global_asm!` are the last-resort fallback only — they expose `reg`/`xmm_reg`/`ymm_reg` but not `zmm_reg`/`tmm_reg`, so they can't carry wide/tile operands without memory-operand workarounds and hand-encoding.

### D8 — Stateful tile ops: RAII guard around the AMX-derived lifecycle

`TILEZERO`/`LDTILECFG`/`TILERELEASE` and the `TOP*`/`BSR*` ops operate on tile + block-scale register state [spec §10–§14]. Model the config/release lifecycle with an RAII guard; outer products take an opaque tile handle and `__m512i` ZMM inputs.

**Rationale:** the tile group consumes SIMD vectors but accumulates into a separate register file with required init/release; an RAII guard prevents leaking tile configuration. Palette 2 (ACE) descriptor per [spec §11.2.3].

**core:: reference:** the primitives already exist for AMX in `crates/core_arch/src/x86_64/amx.rs` — `_tile_loadconfig`, `_tile_release`, `_tile_zero` — behind `#[target_feature(enable="amx-tile")]` and the unstable `x86_amx_intrinsics` feature (tracking #126622). `core::arch` provides them raw and **un-guarded**; `ace-rs` adds the RAII wrapper (consistent with D2: safety ergonomics are the crate's job, not `core::arch`'s).

### D9 — First primitive (tracer bullet): `dpbssd` (AVX-VNNI-INT8)

Iteration 0 implements one primitive end to end: signed int8 dot-product-accumulate.

**Rationale:** it is the *only* ACE primitive already in stable `core::arch`, integer (exact oracle, no rounding), and the 256-bit VEX form runs on shipping silicon — so the full vertical (build → detect → intrinsic → fallback → differential test) is provable today on stable, no emulator required.

**core:: reference:** `core::arch::x86_64::_mm256_dpbssd_epi32` (feature `avxvnniint8`), defined in `crates/core_arch/src/x86/avxvnniint8.rs`.

### D10 (A1) — Operand signedness is encoded in the element type, on raw fixed-size arrays (no newtype)

For the full group-1 family, each variant's `a`/`b` operands are typed *per signedness*: signed operands take `[i8; 32]`/`[i16; 16]`, unsigned operands take `[u8; 32]`/`[u16; 16]`, and `src`/result stay raw `[i32; 8]`. The arrays are exposed **directly**, not behind a newtype wrapper. (Captures requirement API.3 / API.3-1.)

**Rationale:** the signedness is the single most error-prone axis of this family (`dpwsud != dpwusd`), and the mixed-signedness `SU`/`US` variants treat operand order as *significant*. Encoding signedness in the element type pushes that distinction into the type system: a wrong-signedness call, or a `b, a` swap on a mixed-signedness variant, is a **compile error** (rustc E0308) — never a silent wrong answer at runtime. This is why `prop_operands_commute` is asserted only for the genuinely commutative `SS`/`UU` variants and is *not even expressible* for `SU`/`US` (the swapped call does not type-check; the crate carries a `compile_fail` doctest as the executed witness). A newtype was rejected: it would add no safety over the distinct primitive element types (which already make swaps ill-typed) while obscuring the obvious "it's just an array of bytes/words" mental model and the eventual `core::arch` `__m256i` mapping. Raw arrays also keep the public surface a thin, mechanical wrapper over the future intrinsic signature (consistent with D3/D5).

**core:: reference:** the eventual `core::arch::x86_64::_mm256_dp*_epi32` intrinsics take untyped `__m256i` for every operand, with signedness implied by the mnemonic alone — there is no compile-time signedness check at that layer (D1: the raw layer is the bare op). `ace-rs` adds the typed-operand safety at its dispatch layer (D2: ergonomics/safety are the crate's job), and the `[i8;32]`/`[u8;32]`/`[i16;16]`/`[u16;16]` arrays transmute losslessly to `__m256i` via the unaligned `_mm256_loadu_si256` marshalling each `_hw` path already uses.

### D11 (B1) — One declarative `define_dp!` macro emits the dispatch + `_hw` + `_scalar` trio per variant

The 11 new primitives are not hand-copied. A single `macro_rules!` macro, `define_dp!`, is parameterised over `(name, scalar, hw, feature, a_elem_ty, b_elem_ty, products_per_lane, intrinsic_path, accumulate=wrap|saturate)` and emits, for one variant, the public dispatch fn, the public scalar oracle, and the private `#[target_feature]`-gated native fn — reproducing the hand-written `dpbssd` shape (D1/D2/D5) exactly. The hand-written `dpbssd` is retained verbatim as the reference the macro's expansion is compared against. (Captures requirements PRIMITIVE_SHAPE.1/.2.)

**Rationale:** the 12-cell grid is near-identical across cells; the *only* per-cell differences are the feature token, the two operand element types, the products-per-lane (4 byte / 2 word), the intrinsic path, and wrap-vs-saturate. Copy-pasting 11 ~40-line blocks would invite exactly the signedness/saturation/products-per-lane transcription bugs this family is most prone to, and would make a fix have to be applied 12 times. A *declarative* (`macro_rules!`) macro — rather than a procedural macro — keeps the factoring with **zero new dependencies** (SCOPE: a proc-macro would need `syn`/`quote`). Wrap-vs-saturate and the signed/unsigned element types are kept as **explicit named arguments** so each variant's invocation reads as a one-line spec and the per-variant differences stay reviewable at the call site rather than being buried in the macro body — satisfying the constraint that the shared factoring must not obscure the per-variant signedness/saturation differences (PRIMITIVE_SHAPE.2). The macro cannot concatenate identifiers without a proc-macro, so the `name`/`scalar`/`hw` trio is named explicitly per invocation; this is a small, deliberate verbosity in exchange for staying dependency-free on stable.

The `...DS` saturation is implemented once, in the macro's `@fold saturate` arm, as a **single** `SIGNED_DWORD_SATURATE` of the full-precision `src + Σ products` (products folded in `i64`), matching the Intel SDM / Felix Cloutier pseudocode — see OQ-3 in §9. Doing it in one place means the (subtle, opposite-sign) saturation semantics are got right once and inherited by all six saturating variants.

**core:: reference:** stdarch itself generates much of `core::arch` from declarative/codegen sources rather than hand-writing each intrinsic (e.g. the spec-driven generation under `crates/stdarch-gen-*`). `ace-rs`'s `define_dp!` is the same instinct at out-of-tree scale: one parameterised shape, expanded per primitive, with the bare per-variant intrinsic (`_mm256_dp*_epi32`) threaded in as a macro argument.

---

## 5. Testing strategy (before hardware)

Four layers; only one needs an emulator.

| Layer | Emulator? | Proves | core:: reference |
|-------|-----------|--------|------------------|
| 1. Differential vs scalar oracle | No | Correctness. Oracle implements spec rounding (RNE / round-to-odd / bias [spec §2.6]); assert bit-exact. The oracle *is* the fallback (D2). | — (crate-owned) |
| 2. Encoding verification | No | Emitted bytes/stub disassemble to the spec mnemonic+operands. Catches `.byte`/backend bugs without executing. | analogous to stdarch's `assert_instr` test attribute (`stdarch_test`), used throughout `avx.rs` |
| 3. Native execution under Intel SDE | Yes (SDE) | Native path matches oracle. Confirmed for AVX10.2 converts on SDE 10.8. | runs the D1 raw layer / D7 backend |
| 4. CI matrix | Mixed | `(stable, no emulator)` → fallback for everyone; `(stable + SDE)` → native per supported group. | gates on `is_x86_feature_detected!` (`std_detect`) |

Capability-detect SDE coverage and **skip-with-warning**, do not fail: a one-instruction smoke test that catches `SIGILL`/`#UD` tells you whether SDE emulates a given group; unsupported groups fall back to layers 1–2.

For the group-1 family the differential property (`prop_hw_matches_scalar`) is the headline guarantee and is *non-vacuous*: when the variant's feature is absent it `TestResult::discard()`s rather than passing, and the `native_runs_when_required` guard (under `ACE_REQUIRE_NATIVE=1`) asserts **both** `avxvnniint8` and `avxvnniint16` were detected — so a green native CI run cannot mean "byte ops native, word ops silently scalar". Per-variant property selection: additivity + lane-independence + public-matches-oracle for all; `prop_operands_commute` for `SS`/`UU` only (never `SU`/`US` — see D10); a `prop_output_saturates` boundary check for every `...DS`.

---

## 6. Roadmap

| Bullet | Primitive | New axis introduced |
|--------|-----------|---------------------|
| 0 ✅ | **group 1 complete** — `dpbssd` + the 11 remaining VEX integer multiply-accumulate ops (group 1: `avxvnniint8` byte `VPDPB*` + `avxvnniint16` word `VPDPW*`) | dispatch + oracle + differential-test skeleton (D9), generalised once via the declarative `define_dp!` macro (D11/B1) over typed-per-signedness operands (D10/A1); second CPUID feature family (`avxvnniint16`); dual-feature native-coverage guard |
| 1 | AVX10.2 Subset / `AVX10_V1_AUX` (group 2): FP16↔FP8 converts + EVEX VNNI | `AVX10_V1_AUX` gating, FP8 format + RTNE/bias rounding oracle, first SDE-only tests; EVEX-generalizes the group-1 dot product. See `ticket.md`. |
| 2 | `VCVTPS2HF8` FP32→FP8 (group 3) | FP32→FP8 convert + round-to-odd oracle (reuses the FP8 format oracle from bullet 1) |
| 3 | `TOP2BF16PS` BF16 rank-2 outer product (group 4) | stateful RAII tile guard (D8), palette 2, the real engine — **`ACE`-gated; not in binutils 2.44 / likely not SDE 10.8 (§7): `.byte` encoding + layers 1–2 only until tooling lands** |

---

## 7. Tooling coverage (verified)

Verified by cross-mapping binutils `gas/NEWS` against the ACE spec's own feature-flag table [spec §15.3]. The decisive insight: ACE instructions are gated by *different* CPUID flags, and toolchains support them per-flag, not as one "ACE" unit.

binutils 2.44 `gas/NEWS` ("Changes in 2.44") lists Diamond Rapids support under **Intel AMX names** — `AMX-AVX512, AMX-FP8, AMX-MOVRS, AMX-TF32, AMX-TRANSPOSE` — plus `AVX10.2`. Zero occurrences of `ACE`/`TOP4`/`TOP2`/`BSR`/"outer product"/"block scale". Spec §15.3 resolves which ACE instruction sits behind which flag:

| ACE instruction(s) | §15.3 gate | binutils 2.44 | SDE 10.8 |
|--------------------|-----------|---------------|----------|
| AVX10.2 converts — `VCVTPS2HF8`, FP4/FP6, bias forms (groups 2–3) | `AVX10_V*_AUX` | ✅ yes | ✅ yes |
| `LDTILECFG/STTILECFG/TILEZERO/TILERELEASE` | `AMX-TILE` | ✅ yes | ✅ yes |
| `TILEMOVROW(R)`, `TCVTROWD2PS`, `TCVTROWPS2BF16[H,L]`, `TCVTROWPS2PH[H,L]` | `AMX-AVX512` \|\| `ACE_VSN≥1` | ✅ yes (`AMX-AVX512`) | ✅ likely (`AMX-AVX512`) |
| **`TOP4MX*F8PS`, `TOP4MXB*PS`, `TOP2BF16PS`, `TOP4B*D`** | **`ACE` only** | ❌ no | ❓ unverified — likely no |
| **`BSRINIT`, `BSRMOVF`, `BSRMOV[H,L]`** | **`ACE` only** | ❌ no | ❓ unverified — likely no |
| `TILEMOVROW(W)`, `TILEMOVCOL(W)` | **`ACE` only** | ❌ no | ❓ unverified — likely no |

**Consequences:**

1. The ACE value-add — the outer-product engine (`TOP*`) and block-scale registers (`BSR*`) — is gated by the **`ACE` flag alone**, which binutils 2.44 does not know. Intel's `AMX-FP8` in 2.44 is the *dot-product* AMX under palette 1 (TMUL) — a **different instruction family** from ACE's *outer-product* `TOP4MX*F8PS` under palette 2 (ACE). Group 4's compute therefore needs `.byte` raw encoding (D7) or a later binutils.
2. **SDE 10.8 (15 Mar 2026) predates the ACE v1.15 spec (May 2026)** by two months, so it almost certainly does not emulate the `ACE`-gated `TOP*`/`BSR*`. Treat ACE-exclusive emulation as unavailable until proven on a machine: `sde64 -help | grep -i ace` (knob present?) or run a one-instruction `TOP4MX` test and check for `#UD`. (intel.com blocks scripted access to the release notes, so this could not be confirmed remotely.)

**Net:** bullets 0–1 (groups 1–3) are fully buildable + testable today. **Bullet 2 (group-4 `TOP*`/`BSR*`) is blocked on layer-3 testing** — implement via `.byte` (D7), verify via layers 1–2 (oracle + encoding, §5), and add SDE execution only once a build proves ACE emulation exists. Re-verify when binutils >2.44 and SDE >10.8 ship.

---

## 8. References

**Spec / background**
- ACE v1 Specification rev 1.15 (May 2026) — formats §2; CPUID §3; instruction groups §4; exception classes §5; conversions §8–9; tile/outer-product §10–14; intrinsic equivalents in each `*.x` subsection.
- ACE Whitepaper v1 (Apr 2026).

**Standard library code**
- `core::arch::x86` / `x86_64` — raw intrinsics + `__mNNN` types. `crates/core_arch/src/x86{,_64}/` (`avx.rs`, `avx512*.rs`, `avxvnniint8.rs`, `amx.rs`). Docs: `doc.rust-lang.org/core/arch/x86_64/`.
- `core::arch::x86_64::_mm256_dpbssd_epi32` — `avxvnniint8.rs` (D9).
- `core::arch::x86_64::{_tile_loadconfig,_tile_release,_tile_zero}` — `amx.rs`, unstable `x86_amx_intrinsics` (#126622) (D8).
- `core::simd` / `std::simd` — `crates/core_simd/`, `feature(portable_simd)` (#86656) (D3, D4).
- `std::arch::is_x86_feature_detected!` — `crates/std_detect/` (D2).
- internal `core_arch::simd` (`crates/core_arch/src/simd.rs`) and `intrinsics::simd::*` — crate-private to stdarch (D4).
- `core::arch::asm!` / `global_asm!` — inline asm; stable register classes lack `zmm_reg`/`tmm_reg` (D7).

**Related crates**: `multiversion`, `pulp` (dispatch); `wide`, `safe_arch` (safe wrappers); `cc` (native stub build, D7).

**Tooling**: Intel SDE 10.8 (15 Mar 2026); GNU binutils 2.44 (gas: AVX10.2 + Diamond Rapids); LLVM (AVX10.2).

---

## 9. Open questions (group-1 family) — resolved / assumed

These were carried from the design/structure stages for the group-1 extension; their resolution is recorded here (and surfaced for callers in `README.md`):

- **OQ-1 — toolchain (resolved):** `is_x86_feature_detected!("avxvnniint16")` and the six `_mm256_dpw*_epi32` word intrinsics (plus the five new byte intrinsics) compile on **stable Rust 1.96** — no MSRV bump, no nightly feature flags. Confirmed with a **full** `cargo check --all-targets --target x86_64-unknown-linux-gnu` (exit 0), i.e. the macro-emitted dispatch/native bodies actually compile, not merely that the imports resolve (TOOLCHAIN.1 / .1-1). *Note:* import resolution alone is **not** sufficient evidence — an early "no E0432" check passed while the macro bodies still failed to compile, because `is_x86_feature_detected!` rejects a feature name forwarded through a `macro_rules!` as a `:literal` fragment ("unknown x86 target feature"). The fix is to capture the feature token as `:tt` in `define_dp!` (the bare string literal is forwarded to the macro's own expansion site, which `is_x86_feature_detected!`/`#[target_feature]` both accept). Always verify with a complete x86_64 compile, never just `cargo check` on the arm64 dev host (which `#[cfg(target_arch = "x86_64")]`-excludes every native path and masks the breakage).
- **OQ-2 — SDE arch flag (assumed; CI-verified):** the `native-sde` job uses `sde64 -future --` to enable runtime detection of *both* feature families. Whether `-future` makes `is_x86_feature_detected!("avxvnniint16")` return `true` inside the emulated process can only be confirmed by an actual SDE run; the dual-feature guard turns a wrong flag into a **loud red** (the `avxvnniint16` assert fails) rather than a silently-vacuous green. Documented fallbacks: `-gnr` (Granite Rapids) / `-srf` (Sierra Forest). Must be confirmed in CI before declaring INT16 coverage done.
- **OQ-3 — saturation / accumulation width (resolved):** the `...DS` variants apply a **single** `SIGNED_DWORD_SATURATE` to the full-precision lane total `src[i] + Σ products`, with the products folded in `i64` (wide enough that a `u16×u16` product cannot overflow before the clamp). This matches the Intel SDM / Felix Cloutier pseudocode for `VPDPB*DS` / `VPDPW*DS` — it is **not** a two-stage "clamp the product sum, then saturating-add `src`" (which diverges from hardware when `src` and the product sum have opposite signs and the product sum exceeds the i32 range). The native intrinsic is the differential tiebreaker, and the per-variant `prop_output_saturates` + opposite-sign known-value lanes lock the model (SCALAR_ORACLE.1-4).
- **OQ-4 — test summary wording (resolved):** the suite asserts on the stable substring `passed; 0 failed` plus exit code 0, and on the exact panic/assert message strings (`native path disagrees with oracle`, the guard messages), rather than a verbatim toolchain-formatted summary line or a hardcoded per-variant test count.
