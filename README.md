# ace-rs

x86 **AI Compute Extensions (ACE)** primitives for Rust.

ACE is the joint Intel/AMD matrix-acceleration ISA (outer products, reduced-precision
FP8/FP6/FP4 formats, tile + block-scale registers). This crate exposes those primitives
on stable Rust *before* the intrinsics are upstreamed into `core::arch`, with a portable
scalar fallback so calls are correct everywhere and accelerated where supported.

See [`DESIGN_RATIONALE.md`](./DESIGN_RATIONALE.md) for the full design (layering, the
`core::arch` mapping, tooling-coverage matrix, and roadmap).

## Status

**ACE group 1 complete.** All 12 group-1 VEX-encoded integer multiply-accumulate
primitives are wired end to end (build → runtime detect → intrinsic → portable scalar
fallback → differential test): the original tracer bullet `dpbssd` plus the 11 remaining
variants. Each is the 256-bit form, named after its eventual `core::arch::x86_64`
intrinsic with the `_mm256_` prefix and `_epi32` suffix stripped, and dispatches to that
intrinsic when the CPU reports the variant's feature, otherwise to the scalar path — both
paths return identical results.

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
- **`native-sde`** — executes the real group-1 instructions (both the `VPDPB*` byte ops and the `VPDPW*` word ops) under Intel SDE with `ACE_REQUIRE_NATIVE=1`, so both feature families must fire natively or the job goes red. Runs on push-to-main and `workflow_dispatch` (skipped on PRs). Skipped until the repo variable / `SDE_URL` (the SDE Linux tarball URL) is set, since SDE's download is version-rotated and license-gated; see the workflow comments.

### Resolved open questions

- **OQ-1 (toolchain):** `is_x86_feature_detected!("avxvnniint16")` and all six `_mm256_dpw*_epi32` word intrinsics (plus the five new byte intrinsics) compile on **stable Rust 1.96** — no MSRV bump and no nightly feature flags. Confirmed with a full `cargo check --all-targets --target x86_64-unknown-linux-gnu` — the macro-emitted dispatch/native bodies compile, not merely that the imports resolve. (The arm64 dev host `#[cfg(target_arch = "x86_64")]`-excludes every native path, so a green `cargo test` there proves nothing about the native build — always verify against an x86_64 target.)
- **OQ-2 (SDE arch flag):** the `native-sde` job uses `sde64 -future --` as the default arch flag to enable runtime detection of both feature families. *This must be confirmed by an actual CI run.* If a run shows `avxvnniint16` undetected, the dual-feature guard fails **loudly** (red), not silently green — switch the flag to `-gnr` (Granite Rapids) or `-srf` (Sierra Forest), whichever an empirical run confirms enables both features. See the workflow comments.
- **OQ-3 (saturation boundary):** the `...DS` clamp is a **single** `SIGNED_DWORD_SATURATE` of the full-precision `src + Σ products`, with products folded in `i64` — verified against the Intel SDM / Felix Cloutier pseudocode, *not* a two-stage clamp. The native intrinsic is the differential tiebreaker.
- **OQ-4 (test summary):** the suite asserts on the stable substring `passed; 0 failed` plus exit code 0 and on the exact panic/assert message strings, not a verbatim toolchain-formatted summary line.

## Roadmap

| Bullet | Primitive | Group |
|--------|-----------|-------|
| 0 ✅ | group 1 complete — `dpbssd` + `dpbssds`/`dpbsud`/`dpbsuds`/`dpbuud`/`dpbuuds` (AVX-VNNI-INT8) and `dpwsud`/`dpwsuds`/`dpwusd`/`dpwusds`/`dpwuud`/`dpwuuds` (AVX-VNNI-INT16) | 1 (AVX-VNNI-INT8/16) |
| 1 | FP16↔FP8 converts + EVEX VNNI (see `ticket.md`) | 2 (AVX10.2 subset, `AVX10_V1_AUX`) |
| 2 | `VCVTPS2HF8` (FP32→FP8) | 3 (OCP conversions) |
| 3 | `TOP2BF16PS` (BF16 rank-2 outer product) | 4 (ACE tile) — gated, see design §7 |
