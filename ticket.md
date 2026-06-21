# ACE-RS Iteration 1 — AVX10.2 Subset (AVX10_V1_AUX): FP16↔FP8 converts + EVEX VNNI

**Type:** Feature / PRD
**Component:** `ace` crate (primitives)
**Depends on:** Iteration 0 (`dpbssd` tracer bullet) — established the dispatch → native → scalar-oracle → differential-test skeleton.
**Spec reference:** *ACE v1 Specification*, rev 1.15 (May 2026), §2 (formats/rounding), §3 (CPUID), §4.2, §6.1, §8.2–§8.7. Cited below as **[§x]**.
**Design reference:** `DESIGN_RATIONALE.md` decisions D1–D7, testing strategy §5.

---

## 1. Background & motivation

`ace-rs` exposes x86 AI Compute Extensions as stable-Rust primitives ahead of hardware and ahead of upstreaming into `core::arch`. Iteration 0 wired exactly one primitive (`dpbssd`, ACE group 1, AVX-VNNI-INT8) end to end: safe dispatch → native intrinsic when detected → portable scalar oracle as the primary path → differential test proving the two agree.

This ticket implements **ACE group 2 — the AVX10.2 Subset, enumerated under `AVX10_V1_AUX`** [§4.2, §6.1]. It is the natural next increment: the EVEX VNNI forms are a direct generalization of the `dpbssd` integer dot-product already in the crate (exact integer oracle, no rounding), and the FP16↔FP8 converts introduce the first **reduced-precision floating-point formats and rounding oracle** the crate needs for every later group.

This re-sequences the `DESIGN_RATIONALE.md` roadmap (which named the group-3 FP32→FP8 convert as bullet 1): group 2 is enumerated first in the spec, reuses the existing integer-oracle machinery for half its surface, and unblocks the FP8 format/rounding oracle that group 3 will also depend on.

## 2. Goal

Add safe, portable, spec-accurate Rust primitives for **all 26 instructions enumerated under `AVX10_V1_AUX`** [§6.1], each following the crate's established layering (D1–D7): a scalar oracle that is the primary, always-correct path, a dispatch layer, and — where a stable native path and emulator coverage exist — a native execution path tested differentially against the oracle.

The headline guarantee is the same as iteration 0: **on every input the result equals the spec-defined value**, and where the native path runs it agrees with the scalar oracle bit-for-bit.

## 3. Scope

### 3.1 In scope — the 26 `AVX10_V1_AUX` instructions [§6.1]

Grouped by the operation each performs (the *what*). All are EVEX-encoded on hardware; the scalar oracle is encoding-independent.

**A. FP16 → FP8 convert, single source** [§8.2] — 4 instructions
`VCVTPH2BF8`, `VCVTPH2BF8S`, `VCVTPH2HF8`, `VCVTPH2HF8S`
Convert one vector of FP16 to one vector of FP8. `BF8` = FP8 E5M2, `HF8` = FP8 E4M3. The `S` suffix = **saturating**; no suffix = **non-saturating**. Rounding is always **RTNE** (round-to-nearest-even). MXCSR is not consulted; DAZ=0, FTZ=0 assumed; no FP exceptions raised.

**B. FP16 → FP8 convert, two sources** [§8.2] — 4 instructions
`VCVT2PH2BF8`, `VCVT2PH2BF8S`, `VCVT2PH2HF8`, `VCVT2PH2HF8S`
Convert **two** FP16 input vectors into **one** FP8 output vector of the same total width (output lanes `[0..KL/2)` come from src2, `[KL/2..KL)` from src1). Same format/saturation/rounding semantics as family A.

**C. FP16 → FP8 convert with bias rounding** [§8.4] — 4 instructions
`VCVTBIASPH2BF8`, `VCVTBIASPH2BF8S`, `VCVTBIASPH2HF8`, `VCVTBIASPH2HF8S`
Convert one FP16 vector to FP8, applying a per-lane **bias rounding term** [§2.6.3] taken from a second (bias) source operand before rounding to the target format. Bias rounding facilitates stochastic rounding. Same format/saturation matrix as family A; bias replaces plain RTNE.

**D. FP8 → FP16 convert** [§8.5] — 1 instruction
`VCVTHF82PH`
Convert FP8 **E4M3 (HF8)** to FP16. The conversion is **exact** — no rounding, no saturation, no exceptions. (Note: only the HF8/E4M3 source is in this group; BF8→FP16 is not part of `AVX10_V1_AUX`.)

**E. FP32-pair → FP16 convert** [§8.3] — 1 instruction
`VCVT2PS2PHX`
Convert **two** FP32 input vectors into **one** FP16 output vector. Unlike families A–D, this convert **does** consult rounding state: the rounding mode is governed by MXCSR (and EVEX embedded rounding `{er}` at 512-bit width); MXCSR.DAZ is respected on FP32 inputs; FTZ=0 assumed. May set MXCSR status flags `[DE,IE,OE,PE,UE]`.

**F. Byte VNNI, EVEX** [§8.6] — 6 instructions
`VPDPBSSD`, `VPDPBSSDS`, `VPDPBSUD`, `VPDPBSUDS`, `VPDPBUUD`, `VPDPBUUDS`
Per dword lane: 4 byte×byte products summed and **accumulated into the existing dword** (read-modify-write of the destination). Sign matrix is `SS` (signed×signed), `SU` (signed×unsigned), `UU` (unsigned×unsigned). `S` suffix = saturating (unsigned saturate iff both operands unsigned, else signed saturate); no suffix = INT32 wrapping truncation. This is the EVEX generalization of iteration 0's `dpbssd` (which is exactly the VEX `BSS`, non-saturating case).

**G. Word VNNI, EVEX** [§8.7] — 6 instructions
`VPDPWSUD`, `VPDPWSUDS`, `VPDPWUSD`, `VPDPWUSDS`, `VPDPWUUD`, `VPDPWUUDS`
Per dword lane: 2 word×word products summed and accumulated into the existing dword. Sign matrix is `SU`, `US`, `UU` (note: no signed×signed word form in this group). Saturation rule identical to family F.

### 3.2 Out of scope

- Any instruction enumerated under `AVX10_V2_AUX` (group 3: FP32↔FP8, FP4/FP6, `VPMOVSSDB`, `VUNPACKB`) [§6.2] or under `ACE` (group 4: `TOP*`, `BSR*`, tile moves).
- The VEX-encoded AVX-VNNI-INT8/16 forms (group 1) beyond the existing `dpbssd`.
- Hardware-only EVEX encoding concerns that have no bearing on the computed value, **unless** a decision in §7 pulls them in (write-masking `{k1}{z}`, broadcast `m*bcst`, vector-length plumbing). The functional contract is the *value*, not the encoding.

## 4. Data formats (oracle definitions) [§2.3, §2.4.1]

The scalar oracle must implement these bit-exact. Values are OCP MX FP8 (no infinities in either FP8 format).

| Format | Layout | Exp bias | Max normal | NaN encoding | Notes |
|--------|--------|----------|------------|--------------|-------|
| FP32 | S E8 M23 | 127 | — | IEEE-754 | input to family E |
| FP16 | S E5 M10 | 15 | — | IEEE-754 | |
| **BF8** = FP8 E5M2 | S E5 M2 | 15 | ±57344 | `S.11111.{01,10,11}` | "Brain Float", no infinities |
| **HF8** = FP8 E4M3 | S E4 M3 | 7 | ±448 | `S.1111.111` | "Half Float", no infinities |

Saturation behavior on FP→FP8 overflow: **saturating** clamps to the format max normal; **non-saturating** produces the format's NaN/overflow encoding per spec. Subnormal handling per §2.4.1 (e.g. HF8 min subnormal ±2⁻⁹, BF8 min subnormal ±2⁻¹⁶).

## 5. Functional requirements

- **FR-1 — Correctness.** Each of the 26 primitives returns exactly the value defined by its `[§8.x]` operation pseudocode, for every input in the format's domain, including subnormals, signed zeros, NaNs, and overflow/underflow.
- **FR-2 — Rounding.** Families A/B/C round to FP8; A/B use RTNE [§2.6.1], C uses the supplied bias term [§2.6.3]. Family E uses RNE by default (MXCSR/embedded-rounding-governed). Family D (`VCVTHF82PH`) is exact. The oracle is the authority on rounding and is asserted bit-exact.
- **FR-3 — Saturation matrix.** Families A/B/C honor the saturating vs non-saturating distinction (`S` suffix). Families F/G honor it for INT32 accumulation: unsigned saturate iff both operands unsigned, else signed saturate; non-saturating wraps modulo 2³² (matches existing `dpbssd_scalar`).
- **FR-4 — VNNI accumulation.** Families F/G **read-modify-write** the destination accumulator (`total = dest + Σ products`), per §8.6.5 / §8.7.5 — the destination is an input, as in `dpbssd`'s `src`.
- **FR-5 — Two-source lane ordering.** Families B and E place src2-derived results in the low half of the output and src1-derived results in the high half, per §8.2.5 / §8.3.5.
- **FR-6 — Dispatch & fallback (D2).** Each primitive is a safe public fn that selects the native path when the running CPU supports `AVX10_V1_AUX`, else the scalar oracle. The scalar oracle is the **primary** path and is correct on every target (including non-x86).
- **FR-7 — Detection (D2, §3.2).** Native dispatch must gate on the `AVX10_V1_AUX` capability: `CPUID.(EAX=24H,ECX=1):ECX[2]`, subject to the layered check `(AVX10.1 ∧ AVX10_V1_AUX) ∨ AVX10.2` and the XCR0/`CR4.OSXSAVE` state bits [§3.2]. (See §7 — `std_detect` may not expose a stable token for this; if so the crate performs its own CPUID check.)
- **FR-8 — Naming (D5).** Every public item is named after its eventual stdarch intrinsic equivalent [§8.x "C/C++ Compiler Intrinsic Equivalent"], e.g. `cvtph_bf8`, `cvtphs_hf8`, `cvt2ph_bf8`, `cvtbiasph_hf8`, `cvthf8_ph`, `cvt2ps_phx`, `dpwsud`, so future upstreaming is an import swap.
- **FR-9 — Stable Rust (D3).** The crate continues to compile on stable Rust with a stable public API; no `core::simd`, no nightly features.

## 6. Acceptance criteria

Mirrors iteration 0's proven approach (`DESIGN_RATIONALE.md` §5; `src/lib.rs` tests):

- **AC-1.** Every primitive has a scalar oracle implementing its `[§8.x]` semantics, and a public dispatcher that equals the oracle on all inputs.
- **AC-2 (differential).** Where a native path exists and is exercised, it agrees with the scalar oracle **bit-for-bit**, asserted by a differential test, including a property-based (`quickcheck`) differential over randomly sampled inputs (per `AGENT.md`: favour property tests for cross-input invariants).
- **AC-3 (no false green).** The `ACE_REQUIRE_NATIVE=1` guard pattern is honored: a green suite under the native/SDE job must prove the native branch actually ran, not just the fallback.
- **AC-4 (properties).** Algebraic invariants are asserted as properties where they hold: VNNI additivity/accumulator-bias, lane independence, sign-combination consistency; convert round-trips that are lossless (e.g. HF8→FP16 exact, and FP16→HF8→FP16 for in-range representable values); saturating ≥ non-saturating bounds.
- **AC-5 (known values).** A handful of hand-computed vectors per family pin specific results and serve as readable documentation.
- **AC-6 (CI).** `cargo fmt --check`, `cargo clippy -D warnings`, `cargo build`, `cargo test` all green on the `test` gate; the `native-sde` job executes the available native paths under Intel SDE.
- **AC-7 (encoding, where applicable).** Where a native path emits real instructions, emitted bytes/stub disassemble to the spec mnemonic + operands (testing layer 2).

## 7. Open decisions for the design stage

These are genuine design choices the ticket deliberately leaves open (focus is *what*, not *how*); the qrspi questions/requirements/design stages should resolve them against the codebase and spec:

1. **Value representation.** Iteration 0 used plain fixed-size primitive arrays (`[i8;32]`, `[i32;8]`). FP8/FP16/FP32 lanes have no native Rust scalar type — decide the public representation (e.g. raw bit-pattern `[u8;N]`/`[u16;N]`/`[f32;N]` lanes vs. `core::arch` `__mNNN` vector types per D3). Must stay stable-Rust and upstreaming-compatible.
2. **Vector-width coverage.** The instructions exist at 128/256/512-bit. Decide which width(s) the v1 API exposes (one canonical width per primitive, as `dpbssd` did with the 256-bit form, vs. all three).
3. **Masking / broadcast exposure.** Whether the public API surfaces EVEX write-masking `{k1}{z}` and broadcast (`m*bcst`), or omits them in v1 (the scalar value is well-defined without them). Recommendation: omit in v1 unless required, matching the maskless `dpbssd` API.
4. **Native-path reachability.** Confirm during research which of these EVEX intrinsics are reachable on stable today (the design table marks AVX10.2 enablement "partially / in progress") and which are SDE-emulable (§7 of the rationale marks groups 2–3 as SDE 10.8-supported). Primitives without a stable native path ship oracle-only (still fully correct) until tooling lands — explicitly, per the gradient in D-rationale §2.
5. **Detection token.** Confirm whether `std_detect`'s `is_x86_feature_detected!` exposes an `AVX10_V1_AUX`/`avx10.2` token; if not, scope the crate-owned CPUID check (FR-7).
6. **Module/API surface granularity.** 26 instructions is a large surface; decide grouping (e.g. by family) and how much is generated vs. hand-written, and whether the FP8/FP16 conversion oracle is a shared internal building block.

## 8. References

- **Spec:** ACE v1 rev 1.15 — §2.3/§2.4.1 formats, §2.6 rounding, §3.2 detection, §4.2 group listing, §6.1 encodings + CPUID (`EAX=24H,ECX=1:ECX[2]`), §8.2 FP16→FP8, §8.3 FP32-pair→FP16, §8.4 biased FP16→FP8, §8.5 HF8→FP16, §8.6 byte VNNI, §8.7 word VNNI; §5.1 exception classes (E2 / E4 / E4NF).
- **Crate design:** `DESIGN_RATIONALE.md` D1 (raw gated intrinsic), D2 (dispatch+fallback layer), D3/D4 (types, no `core::simd`), D5 (intrinsic-mirroring names), D6 (sunset autocfg), D7 (native backend strategy), §5 (4-layer testing), §7 (tooling coverage: groups 2–3 buildable + SDE-testable today).
- **Prior art in repo:** `src/lib.rs` (`dpbssd` / `dpbssd_scalar` / `dpbssd_hw` + differential, known-value, and property tests), `.github/workflows/ci.yml` (`test` gate + `native-sde` job with `ACE_REQUIRE_NATIVE=1`), `AGENT.md` (branch discipline; favour property tests).
- **Intrinsic equivalents (target public names, D5):** `_mm512_cvtph_bf8`/`_hf8` (+ saturating `cvtphs_*`), `_mm512_cvt2ph_bf8`/`_hf8`, `_mm512_cvtbiasph_bf8`/`_hf8`, `_mm512_cvthf8_ph`, `_mm512_cvt2ps_phx`, `_mm512_dpwsud`/`dpwusd`/`dpwuud` (+ saturating) and the EVEX byte-VNNI `dpbss`/`dpbsu`/`dpbuu` forms.
