# ace-rs

x86 **AI Compute Extensions (ACE)** primitives for Rust.

ACE is the joint Intel/AMD matrix-acceleration ISA (outer products, reduced-precision
FP8/FP6/FP4 formats, tile + block-scale registers). This crate exposes those primitives
on stable Rust *before* the intrinsics are upstreamed into `core::arch`, with a portable
scalar fallback so calls are correct everywhere and accelerated where supported.

See [`DESIGN_RATIONALE.md`](./DESIGN_RATIONALE.md) for the full design (layering, the
`core::arch` mapping, tooling-coverage matrix, and roadmap).

## Status

Iteration 0 (tracer bullet): one primitive — `dpbssd` (AVX-VNNI-INT8 signed int8
dot-product-accumulate) — wired end to end:

```rust
use ace::dpbssd;

let out = dpbssd([0; 8], a /* [i8; 32] */, b /* [i8; 32] */);
```

It dispatches to `core::arch::x86_64::_mm256_dpbssd_epi32` when the CPU reports
`avxvnniint8`, otherwise to a portable scalar path. Both return identical results.

## Test

```sh
cargo test
```

The differential test compares the native path against the scalar oracle wherever the
feature is present. On hardware/toolchains without it, the fallback path is exercised.

To **execute and verify the native path** without AVX-VNNI-INT8 hardware, run the test
binaries under Intel SDE (x86_64 host only — SDE has no arm64 build). Setting
`ACE_REQUIRE_NATIVE=1` makes the suite fail unless the native branch actually ran, so a
green result can't silently mean "fallback only":

```sh
ACE_REQUIRE_NATIVE=1 \
CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER="sde64 -future --" \
cargo test --target x86_64-unknown-linux-gnu
```

## CI

[`.github/workflows/ci.yml`](./.github/workflows/ci.yml):

- **`test`** (x86_64 Linux) — `fmt --check`, `clippy -D warnings`, `build`, `test`. Always runs; gates merges.
- **`native-sde`** — executes the real `VPDPBSSD` under Intel SDE with `ACE_REQUIRE_NATIVE=1`. Skipped until the repo variable `SDE_URL` (the SDE Linux tarball URL) is set, since SDE's download is version-rotated and license-gated; see the workflow comments.

## Roadmap

| Bullet | Primitive | Group |
|--------|-----------|-------|
| 0 ✅ | `dpbssd` | 1 (AVX-VNNI-INT8) |
| 1 | FP16↔FP8 converts + EVEX VNNI (see `ticket.md`) | 2 (AVX10.2 subset, `AVX10_V1_AUX`) |
| 2 | `VCVTPS2HF8` (FP32→FP8) | 3 (OCP conversions) |
| 3 | `TOP2BF16PS` (BF16 rank-2 outer product) | 4 (ACE tile) — gated, see design §7 |
