# ace-rs

x86 **AI Compute Extensions (ACE)** primitives for Rust.

ACE is the joint Intel/AMD matrix-acceleration ISA (outer products, reduced-precision
FP8/FP6/FP4 formats, tile + block-scale registers). This crate exposes those primitives
on stable Rust *before* the intrinsics are upstreamed into `core::arch`, with a portable
scalar fallback so calls are correct everywhere and accelerated where supported.

See [`DESIGN_RATIONALE.md`](./DESIGN_RATIONALE.md) for the full design (layering, the
`core::arch` mapping, tooling-coverage matrix, and roadmap).

## Status

Tracking the feature groups of the ACE v1 instruction summary (§4):

| § | Feature group | Status |
|---|---------------|--------|
| **4.1** | **AVX-VNNI-INT8 and AVX-VNNI-INT16** — VEX-encoded integer multiply-accumulate | ✅ implemented |
| **4.2** | **AVX10.2 Subset (`AVX10_V1_AUX`)** — FP16↔FP8 conversions and EVEX VNNI forms | ✅ implemented |
| **4.3** | **OCP Format Conversions (`AVX10_V2_AUX`)** — FP32↔FP8, FP8↔FP4/FP6, utility ops | ✅ implemented |
| **4.4** | **ACE Tile Instructions (ACE v1)** — tile management, data movement, outer products | ✅ implemented |

### 4.1 — AVX-VNNI-INT8 and AVX-VNNI-INT16 ✅

All 12 group-4.1 VEX-encoded integer multiply-accumulate primitives are wired end to end
(build → runtime detect → intrinsic → portable scalar fallback → differential test): the
original tracer bullet `dpbssd` plus the 11 remaining variants. Each is the 256-bit form,
named after its eventual `core::arch::x86_64` intrinsic with the `_mm256_` prefix and
`_epi32` suffix stripped, and dispatches to that intrinsic when the CPU reports the
variant's feature, otherwise to the scalar path — both paths return identical results.

```rust
use ace::dpbssd;

// out[i] = src[i] + Σ a[k]*b[k] over the 4 byte products in lane i
let out = dpbssd([0i32; 8], a /* [i8; 32] */, b /* [i8; 32] */);
```

### The group-1 grid

The group is the 12-cell grid of `{int8 byte ops, int16 word ops} × {SS, SU, US, UU} ×
{wrap, saturate}`. The mnemonic encodes everything you need to pick the right function:

| Fn | Feature | `a` type | `b` type | products/lane | accumulate |
|----|---------|----------|----------|:-------------:|------------|
| `dpbssd`   | `avxvnniint8`  | `[i8; 32]`  | `[i8; 32]`  | 4 | wrap |
| `dpbssds`  | `avxvnniint8`  | `[i8; 32]`  | `[i8; 32]`  | 4 | **saturate** |
| `dpbsud`   | `avxvnniint8`  | `[i8; 32]`  | `[u8; 32]`  | 4 | wrap |
| `dpbsuds`  | `avxvnniint8`  | `[i8; 32]`  | `[u8; 32]`  | 4 | **saturate** |
| `dpbuud`   | `avxvnniint8`  | `[u8; 32]`  | `[u8; 32]`  | 4 | wrap |
| `dpbuuds`  | `avxvnniint8`  | `[u8; 32]`  | `[u8; 32]`  | 4 | **saturate** |
| `dpwsud`   | `avxvnniint16` | `[i16; 16]` | `[u16; 16]` | 2 | wrap |
| `dpwsuds`  | `avxvnniint16` | `[i16; 16]` | `[u16; 16]` | 2 | **saturate** |
| `dpwusd`   | `avxvnniint16` | `[u16; 16]` | `[i16; 16]` | 2 | wrap |
| `dpwusds`  | `avxvnniint16` | `[u16; 16]` | `[i16; 16]` | 2 | **saturate** |
| `dpwuud`   | `avxvnniint16` | `[u16; 16]` | `[u16; 16]` | 2 | wrap |
| `dpwuuds`  | `avxvnniint16` | `[u16; 16]` | `[u16; 16]` | 2 | **saturate** |

Every variant takes `src: [i32; 8]` and returns `[i32; 8]`. The element *types* of `a`
and `b` are the signedness — there is no untyped "raw bytes" entry point.

### Picking a variant — signedness and operand order

The two letters after `dpb`/`dpw` are the signedness of the **`a` then `b`** operand, in
order: `s` = signed, `u` = unsigned. So `dpwsud` is signed `a` × unsigned `b`, and
`dpwusd` is unsigned `a` × signed `b`.

**Operand order is significant for the mixed-signedness (SU / US) variants** —
`dpwsud != dpwusd` (and likewise `dpbsud`/`dpbsuds` are not their own mirror). Because the
operand signedness is carried in the Rust *type* (`[i8;32]`/`[i16;16]` for signed,
`[u8;32]`/`[u16;16]` for unsigned), you cannot accidentally swap `a` and `b` on these
variants — the two arguments have *different types*, so a swap is a **compile error**, not
a silent wrong answer. The SS and UU variants are genuinely commutative (`a·b == b·a`).

### Wrap vs. saturate (`...D` vs. `...DS`)

The wrapping `...D` variants (`dpbssd`, `dpbsud`, `dpbuud`, `dpwsud`, `dpwusd`, `dpwuud`)
accumulate with wrapping `i32` arithmetic — a lane that overflows wraps modulo 2³².

The saturating `...DS` variants (`dpbssds`, `dpbsuds`, `dpbuuds`, `dpwsuds`, `dpwusds`,
`dpwuuds`) **clamp each lane to `[i32::MIN, i32::MAX]`** instead of wrapping. The clamp is
a *single* signed-dword saturation of the full-precision lane total — `out[i] =
SIGNED_DWORD_SATURATE(src[i] + Σ products)`, matching the Intel SDM / Felix Cloutier
pseudocode for `VPDPB*DS` / `VPDPW*DS`. There is **no** intermediate clamp of the product
sum before `src` is added: for the word ops a `u16 × u16` product (≈ 4.29 × 10⁹) already
exceeds `i32::MAX`, so the products are folded in `i64` and the single clamp is applied to
`src + Σ products` once. (A two-stage "clamp the product sum, then saturating-add `src`"
gives a different answer than hardware when `src` and the product sum have opposite signs.)

### 4.2 — AVX10.2 Subset (`AVX10_V1_AUX`) ✅

The 26 `AVX10_V1_AUX` primitives are implemented: the FP16↔FP8 / FP32→FP16 conversions
and the EVEX VNNI forms. The scalar oracle is the always-present primary path; an opt-in
`native` cargo feature routes each primitive to a hand-written C shim compiled with
`-mavx10.2` (there is no stable `core::arch` EVEX intrinsic for these forms yet), taken
only when `detect::has_avx10_v1_aux()` confirms the running CPU, and differentially tested
against the oracle under Intel SDE. (The same `native` feature also compiles the group-4
tile shims, `src/native/ace_tile.c`, which additionally need `-mamx-tile -mamx-avx512` —
`build.rs` fails fast with a clear message if the toolchain rejects any of those flags.)

- **FP16→FP8 converts:** `VCVTPH2BF8`, `VCVTPH2BF8S`, `VCVTPH2HF8`, `VCVTPH2HF8S`
- **Two-source FP16→FP8 converts:** `VCVT2PH2BF8`, `VCVT2PH2BF8S`, `VCVT2PH2HF8`, `VCVT2PH2HF8S`
- **FP32→FP16 convert:** `VCVT2PS2PHX`
- **Bias FP16→FP8 converts:** `VCVTBIASPH2BF8`, `VCVTBIASPH2BF8S`, `VCVTBIASPH2HF8`, `VCVTBIASPH2HF8S`
- **FP8→FP16 convert:** `VCVTHF82PH`
- **EVEX byte VNNI:** `VPDPBSSD`, `VPDPBSSDS`, `VPDPBSUD`, `VPDPBSUDS`, `VPDPBUUD`, `VPDPBUUDS`
- **EVEX word VNNI:** `VPDPWSUD`, `VPDPWSUDS`, `VPDPWUSD`, `VPDPWUSDS`, `VPDPWUUD`, `VPDPWUUDS`

The EVEX byte/word VNNI ops live in the `vnni` module (e.g. `ace::vnni::dpbssd`), distinct
from the 256-bit VEX `ace::dpbssd` of group 4.1.

### 4.3 — OCP Format Conversions (`AVX10_V2_AUX`) ✅

The 21 group-3 OCP-format converts (families A–I) are implemented: FP32↔FP8, FP8↔FP4,
FP8↔FP6, and the two utility ops. All are the 512-bit (`VL=512`) forms, dispatch-gated on
`AVX10_V2_AUX` detection, and each ships a scalar oracle plus a native-vs-oracle
differential property that discards (never passes vacuously) when the native path is
absent.

**Group 3 currently ships oracle-only (OQ-5):** none of the group-3 intrinsics
(`_mm512_cvtps_bf8`, `_mm512_cvtbf8_ps`, `_mm512_cvtf8_bf4s`, `_mm512_cvtbf4_hf8`,
`_mm512_cvtf8_bf6s`, `_mm512_cvtf6_hf8`, `_mm512_cvtssepi32_epi8`, `_mm512_unpackb`, …)
compile under `-mavx10.2` in current GCC/Clang headers, so there is no native C shim yet.
The differential tests are wired to go live the moment an intrinsic lands.

| Mnemonic | Rust function(s) |
|----------|------------------|
| `VCVTPS2BF8` / `VCVTPS2BF8S` | `cvtps_bf8` / `cvtpss_bf8` |
| `VCVTPS2HF8` / `VCVTPS2HF8S` | `cvtps_hf8` / `cvtpss_hf8` |
| `VCVTROPS2HF8` / `VCVTROPS2HF8S` | `cvtrops_hf8` / `cvtropss_hf8` (RTO is E4M3-only — no `cvtrops_bf8`) |
| `VCVTBIASPS2BF8` / `VCVTBIASPS2BF8S` | `cvtbiasps_bf8` / `cvtbiaspss_bf8` |
| `VCVTBIASPS2HF8` / `VCVTBIASPS2HF8S` | `cvtbiasps_hf8` / `cvtbiaspss_hf8` |
| `VCVTBF82PS` / `VCVTHF82PS` | `cvtbf8_ps` / `cvthf8_ps` |
| `VCVTBF82BF4S` / `VCVTHF82BF4S` | `cvtf8_bf4s_e5m2` / `cvtf8_bf4s_e4m3` |
| `VCVTBF42HF8` | `cvtbf4_hf8` |
| `VCVTBF82BF6S` / `VCVTHF82HF6S` | `cvtf8_bf6s` / `cvtf8_hf6s` |
| `VCVTBF62HF8` / `VCVTHF62HF8` | `cvtf6_hf8_e3m2` / `cvtf6_hf8_e2m3` |
| `VPMOVSSDB` | `cvtssepi32_epi8` (symmetric saturation to `[-127, +127]`) |
| `VUNPACKB` | `unpackb` (build `imm8` with `ACE_UNPACKB_SIZE` / `ACE_UNPACKB_START` / `ACE_UNPACKB_SEXT`) |

Where two converts share a target format, the Rust name carries a source-format suffix
(`_e5m2` / `_e4m3` / `_e3m2` / `_e2m3`) to disambiguate. Every dispatcher has a public
`*_scalar` oracle twin (e.g. `cvtps_bf8_scalar`). FP4 results are nibble-packed
(`[u8; 32]` for 64 lanes) and FP6 results 6-bit-packed (`[u8; 48]`); `unpackb` is the
read-back complement of those packed layouts.

### 4.4 — ACE Tile Instructions (ACE v1) ✅

All 25 group-4 tile instructions are implemented with the spec-conformant model: a fixed
register file of **8 tiles × 16 rows × 64 bytes** (the palette-2 descriptor is byte 0 =
palette plus 63 reserved-zero bytes), a **single 1024-bit Block Scale register**
(`bsr0`/SCALEDATA, initialized to `0x7F`), row/column specifiers that mask to bits `[3:0]`
and never fault, and `TILEMOVCOL` as a **write-only** instruction. Tile + BSR state lives
in an RAII `TileScope` guard returned by `_tile_loadconfig`, which issues `TILERELEASE` on
`Drop`. Every instruction ships a scalar oracle (the always-present primary path), a native
shim behind the opt-in `native` feature (`src/native/ace_tile.c`), and — for the
`ACE`-only families no assembler knows yet — `.byte` raw encodings golden-checked against
the spec §6.3 tables by `tests/encoding.rs`.

- **Tile management:** `_tile_loadconfig`, `_tile_storeconfig`, `_tile_zero` (plus
  `TileConfig`, `TileScope`, `TileId`; `TILERELEASE` is `Drop`-only)
- **Tile data movement:** `_tile_movrow` (read), `_tile_setrow`, `_tile_setcol`
  (`TILEMOVCOL` is write-only — there is no column read)
- **Tile row converts:** `_tile_cvtrowd2ps`, `_tile_cvtrowps2bf16h`/`_tile_cvtrowps2bf16l`,
  `_tile_cvtrowps2phh`/`_tile_cvtrowps2phl`
- **Block-scale register ops:** `_bsrinit`, `_bsrmovf`, `_bsrmovh`/`_bsrmovh_read`,
  `_bsrmovl`/`_bsrmovl_read`
- **MX outer products:** `_tile_top4mxbf8ps`, `_tile_top4mxbhf8ps`, `_tile_top4mxhbf8ps`,
  `_tile_top4mxhf8ps`, `_tile_top4mxbssps` — each takes an `imm8` scale-group selector
  (compose with `ace_scale_a`/`ace_scale_b`) and applies the combined E8M0 scale once in
  the precise domain
- **BF16 outer product:** `_tile_top2bf16ps`
- **Byte outer products:** `_tile_top4bssd`, `_tile_top4bsud`, `_tile_top4busd`,
  `_tile_top4buud`

The public API takes fixed-size arrays (`[u8; 64]`, `[u16; 32]`, …), not `__m512`/`__tile`
types. Families A / B-read / C (tile config, `TILEMOVROW` read, row converts) are
intrinsic-reachable and execute under Intel SDE; the `ACE`-only `.byte` families are built
and encoding-verified, with their SDE differentials wired to go live when SDE gains ACE
emulation.

## Examples

Six self-contained demos in [`examples/`](./examples), each built from the public API,
each asserting its dispatcher outputs against the `*_scalar` oracle twins, and each
runnable everywhere (scalar path) or under Intel SDE (native path where SDE emulates the
feature):

| Example | Group | Shows |
|---------|-------|-------|
| `int8_gemv` | 4.1 | Quantized linear layer via `dpbssd`, plus wrap-vs-saturate (`dpbssds`) |
| `int8_image_filter` | 4.1 | 4-tap image blur/edge-detect via `dpbuud`/`dpbsud`, ASCII render |
| `fp8_stochastic_rounding` | 4.2 | Stochastic rounding via the `VCVTBIASPH2BF8` bias operand — RTNE's systematic −0.25 lsb bias vs SR's near-zero trial average; EVEX-vs-VEX VNNI contrast |
| `fp8_quant_error` | 4.3 | E4M3 vs E5M2 round-trip error, range/overflow, why scaling decides the winner |
| `fp4_compress` | 4.3 | 8× weight compression to nibble-packed FP4, `unpackb` read-back |
| `mx_matmul` | 4.4 | MX block-scaled matmul: `TileScope`, BSR scales, `_tile_top4mxbf8ps`, bf16 row store via `TCVTROWPS2BF16H/L` |

```sh
cargo run --example int8_gemv
# native paths under SDE (group 4.1 examples report "native path: yes"; the group-4.2
# example needs --features native for its EVEX shims):
cargo build --examples --features native
sde64 -future -- target/debug/examples/fp8_stochastic_rounding
```

The group-4.3 examples are oracle-only everywhere today (OQ-5); `mx_matmul` runs the
Rust-modeled scalar tile file everywhere (the family-C TCVTROW shims execute natively
under SDE only via the crate's internal `--features native` differential tests).

## Test

```sh
cargo test
```

The differential test compares the native path against the scalar oracle wherever the
feature is present. On hardware/toolchains without it, the fallback path is exercised and
the differential **property** tests *discard* (they never pass vacuously) so a feature-less
runner cannot report a false green for the native path.

To **execute and verify the native paths** without AVX-VNNI-INT8/INT16 hardware, run the
test binaries under Intel SDE (x86_64 host only — SDE has no arm64 build). Setting
`ACE_REQUIRE_NATIVE=1` makes the suite fail unless the native branch actually ran — and the
guard now requires **both** the `avxvnniint8` (byte ops) and `avxvnniint16` (word ops)
families to have been detected, so a green result can't silently mean "byte ops native,
word ops fell back to scalar":

```sh
ACE_REQUIRE_NATIVE=1 \
CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER="sde64 -future --" \
cargo test --target x86_64-unknown-linux-gnu
```

## CI

[`.github/workflows/ci.yml`](./.github/workflows/ci.yml):

- **`test`** (x86_64 Linux) — `fmt --check`, `clippy -D warnings`, `build`, `test`. Always runs; gates merges. Scalar-only on the runner; native execution is proven by `native-sde`.
- **`native-sde`** — executes the real group-1 instructions (both the `VPDPB*` byte ops and the `VPDPW*` word ops) under Intel SDE with `ACE_REQUIRE_NATIVE=1`, so both feature families must fire natively or the job goes red. It also builds with `--features native`, which compiles the `AVX10_V1_AUX` C shims (`-mavx10.2`) **and** the ACE tile shims (`src/native/ace_tile.c`, `-mavx10.2 -mamx-tile -mamx-avx512` — `build.rs` fails fast if the toolchain rejects those flags) and exercises the group-2 (families A–G) native-vs-oracle differentials under SDE. Group-4 families A / B-read / C (tile config, `TILEMOVROW` read, row converts) execute natively under SDE too; the `ACE`-only `.byte` families build and are encoding-verified but their execution differentials discard until SDE gains ACE emulation. The group-3 differentials discard for now because group 3 is oracle-only (OQ-5, no C shims exist). Runs on push-to-main and `workflow_dispatch` (skipped on PRs). Skipped until the repo variable / `SDE_URL` (the SDE Linux tarball URL) is set, since SDE's download is version-rotated and license-gated; see the workflow comments.

### Resolved open questions

- **OQ-1 (toolchain):** `is_x86_feature_detected!("avxvnniint16")` and all six `_mm256_dpw*_epi32` word intrinsics (plus the five new byte intrinsics) compile on **stable Rust 1.96** — no MSRV bump and no nightly feature flags. Confirmed with a full `cargo check --all-targets --target x86_64-unknown-linux-gnu` — the macro-emitted dispatch/native bodies compile, not merely that the imports resolve. (The arm64 dev host `#[cfg(target_arch = "x86_64")]`-excludes every native path, so a green `cargo test` there proves nothing about the native build — always verify against an x86_64 target.)
- **OQ-2 (SDE arch flag):** the `native-sde` job uses `sde64 -future --` as the default arch flag to enable runtime detection of both feature families. *This must be confirmed by an actual CI run.* If a run shows `avxvnniint16` undetected, the dual-feature guard fails **loudly** (red), not silently green — switch the flag to `-gnr` (Granite Rapids) or `-srf` (Sierra Forest), whichever an empirical run confirms enables both features. See the workflow comments.
- **OQ-3 (saturation boundary):** the `...DS` clamp is a **single** `SIGNED_DWORD_SATURATE` of the full-precision `src + Σ products`, with products folded in `i64` — verified against the Intel SDM / Felix Cloutier pseudocode, *not* a two-stage clamp. The native intrinsic is the differential tiebreaker.
- **OQ-4 (test summary):** the suite asserts on the stable substring `passed; 0 failed` plus exit code 0 and on the exact panic/assert message strings, not a verbatim toolchain-formatted summary line.
- **OQ-5 (group-3 native availability):** a group-3 family whose `-mavx10.2` intrinsic does not compile in the current GCC/Clang headers ships **oracle-only** — scalar oracle as the sole path, with the native differential wired to go live (never vacuously green) the moment the intrinsic lands. Today that is *every* group-3 family.

## Roadmap

| Bullet | Primitive | Group |
|--------|-----------|-------|
| 0 ✅ | `dpbssd` + `dpbssds`/`dpbsud`/`dpbsuds`/`dpbuud`/`dpbuuds` (AVX-VNNI-INT8) and `dpwsud`/`dpwsuds`/`dpwusd`/`dpwusds`/`dpwuud`/`dpwuuds` (AVX-VNNI-INT16) | 4.1 (AVX-VNNI-INT8/16) |
| 1 ✅ | FP16↔FP8 converts + EVEX VNNI (26 `AVX10_V1_AUX` primitives) | 4.2 (AVX10.2 subset, `AVX10_V1_AUX`) |
| 2 ✅ | OCP format converts (21 `AVX10_V2_AUX` primitives, oracle-only per OQ-5) | 4.3 (OCP conversions, `AVX10_V2_AUX`) |
| 3 ✅ | ACE tile instructions — the whole group 4 (25 primitives): tile lifecycle + `TileScope` RAII guard, tile moves, row converts, Block Scale register ops, and all `TOP*` outer products, with §6.3-verified `.byte` encodings | 4.4 (ACE tile) |

## Contributing

Contributions are welcome — see [`CONTRIBUTING.md`](./CONTRIBUTING.md) for local
setup, the test/lint gates, and how to wire a new primitive end to end. Please
also read the [Code of Conduct](./CODE_OF_CONDUCT.md). To report a security
issue privately, see [`SECURITY.md`](./SECURITY.md).

## License

Licensed under either of

- Apache License, Version 2.0 ([`LICENSE-APACHE`](./LICENSE-APACHE) or
  <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([`LICENSE-MIT`](./LICENSE-MIT) or
  <http://opensource.org/licenses/MIT>)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
