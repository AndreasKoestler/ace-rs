# Contributing to ace-rs

Thanks for your interest in contributing! `ace-rs` exposes x86 AI Compute
Extensions (ACE) as stable-Rust primitives ahead of their upstreaming into
`core::arch`. Contributions of new primitives, tests, documentation, and
tooling are all welcome.

Please read [`DESIGN_RATIONALE.md`](./DESIGN_RATIONALE.md) before adding a new
primitive — it documents the layering, the `core::arch` mapping, the testing
strategy, and the design decisions (D1–D11) every primitive is expected to
follow.

## Local development setup

You need a stable Rust toolchain (MSRV **1.96**):

```sh
rustup toolchain install stable
git clone <your-fork-url>
cd ace
cargo build
cargo test
```

The default build is pure stable Rust with no compiled C and no native code —
it compiles and is correct on every target.

### Exercising the native paths

Native (EVEX/VEX) execution can be verified without AVX-VNNI hardware by running
the test binaries under [Intel SDE](https://www.intel.com/content/www/us/en/developer/articles/tool/software-development-emulator.html)
(x86_64 host only):

```sh
ACE_REQUIRE_NATIVE=1 \
CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER="sde64 -future --" \
cargo test --target x86_64-unknown-linux-gnu
```

`ACE_REQUIRE_NATIVE=1` makes the suite fail unless the native branch actually
ran, so a feature-less runner cannot report a false green.

The opt-in `native` cargo feature additionally compiles the `AVX10_V1_AUX` C
shims (`src/native/avx10_v1_aux.c`) with `-mavx10.2`, which requires GCC >= 15
or Clang >= 20:

```sh
cargo test --features native --target x86_64-unknown-linux-gnu
```

Under SDE this also exercises the group-2 native-vs-oracle differentials. The
group-3 (`AVX10_V2_AUX`) families are currently oracle-only — their intrinsics
are absent from the compiler headers (OQ-5 in `DESIGN_RATIONALE.md`) — so their
differentials discard until a toolchain supplies the intrinsics.

## Tests and lint

Every change must pass the same gates CI enforces:

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo build
cargo test
```

Favour **property-based tests** (`quickcheck`) where a primitive has invariants
that hold across the whole input space — a native path matching its scalar
oracle, algebraic properties (additivity, lane independence), identity elements.
Keep a few hand-rolled tests for hand-computed known values and as readable
documentation, but let properties cover the breadth.

## Branching and pull requests

1. Create a feature branch off `main` (e.g. `feat/dpwssd` or `fix/saturation-boundary`).
2. Keep each PR focused on one primitive or one logical change.
3. Make sure all the gates above pass locally.
4. Open a PR against `main` with a clear description of the primitive/behaviour
   and a reference to the relevant spec section or design decision.
5. New primitives should be wired end to end — build → runtime detect →
   intrinsic → portable scalar fallback → differential test — matching the
   pattern established by the existing group-1 functions.

## Reporting bugs and security issues

Functional bugs and feature requests: please open a GitHub issue. For security
vulnerabilities, **do not** open a public issue — see [`SECURITY.md`](./SECURITY.md).

By contributing, you agree that your contributions will be dual-licensed under
the MIT and Apache-2.0 licenses, matching the project license.
