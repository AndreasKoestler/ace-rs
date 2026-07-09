//! Build script for the opt-in `native` backend (design decision D7).
//!
//! It compiles the AVX10_V1_AUX C shims (`src/native/avx10_v1_aux.c`) and the ACE group-4
//! tile shims (`src/native/ace_tile.c`) ONLY when both conditions hold:
//!
//!   * the `native` feature is enabled (`CARGO_FEATURE_NATIVE` set by Cargo), and
//!   * the target architecture is `x86_64` (`CARGO_CFG_TARGET_ARCH == "x86_64"`).
//!
//! In every other configuration the build script is a no-op: the default build pulls in no
//! `cc` dependency (it is an optional build-dependency, gated behind `native`) and compiles
//! no C, preserving the pure-stable-Rust default that is correct on non-x86 targets.
//!
//! There is deliberately NO AVX10_V2_AUX translation unit: every group-3 intrinsic is
//! absent from the current GCC/Clang `-mavx10.2` headers (OQ-5, see the module docs of
//! `src/native.rs`), so a group-3 TU would contain no shims. Add
//! `src/native/avx10_v2_aux.c` here (plus its `rerun-if-changed` line) when the first
//! group-3 intrinsic lands in a toolchain.
fn main() {
    // Always re-run if either source TU changes (cheap, and avoids stale objects).
    println!("cargo:rerun-if-changed=src/native/avx10_v1_aux.c");
    println!("cargo:rerun-if-changed=src/native/ace_tile.c");

    // Cargo passes `--cfg feature="native"` when it compiles this build script, and the
    // optional `cc` build-dependency is only present (linkable) under that feature. So the
    // `cc`-using path is itself feature-gated: in the default build it is not compiled at
    // all, the `cc` crate is never pulled in, and the build script is a no-op.
    #[cfg(feature = "native")]
    compile_native();
}

/// Compile the AVX10_V1_AUX C shims with `-mavx10.2`, but only on an x86_64 target. On any
/// other architecture the native feature still produces no compiled C (the EVEX forms only
/// exist on x86_64), preserving correctness on non-x86 targets.
#[cfg(feature = "native")]
fn compile_native() {
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if arch != "x86_64" {
        return;
    }
    cc::Build::new()
        .file("src/native/avx10_v1_aux.c")
        .flag("-mavx10.2")
        .opt_level(2)
        .compile("ace_native_avx10_v1_aux");

    // ACE group-4 tile instructions (design decision D7, OQ-6). Families A/B-read/C use real
    // GCC/Clang AMX-TILE + AMX-AVX512 intrinsics (assembler/intrinsic-reachable); the `ACE`-only
    // forms (family B write, D, E, F, G) are hand-encoded `.byte` shims so the TU compiles even
    // though no assembler knows the ACE mnemonics yet. The tile config/data + AVX-512 forms need
    // `-mamx-tile -mamx-avx512` alongside `-mavx10.2`; the `.byte` shims need no ISA flag at all.
    cc::Build::new()
        .file("src/native/ace_tile.c")
        .flag("-mavx10.2")
        .flag("-mamx-tile")
        .flag("-mamx-avx512")
        .opt_level(2)
        .compile("ace_native_ace_tile");
}
