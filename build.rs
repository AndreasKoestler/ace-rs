//! Build script for the opt-in `native` backend (design decision D7).
//!
//! It compiles the AVX10_V1_AUX C shims (`src/native/avx10_v1_aux.c`) and the AVX10_V2_AUX
//! C shims (`src/native/avx10_v2_aux.c`) with `-mavx10.2` ONLY when both conditions hold:
//!
//!   * the `native` feature is enabled (`CARGO_FEATURE_NATIVE` set by Cargo), and
//!   * the target architecture is `x86_64` (`CARGO_CFG_TARGET_ARCH == "x86_64"`).
//!
//! In every other configuration the build script is a no-op: the default build pulls in no
//! `cc` dependency (it is an optional build-dependency, gated behind `native`) and compiles
//! no C, preserving the pure-stable-Rust default that is correct on non-x86 targets.
fn main() {
    // Always re-run if either source TU changes (cheap, and avoids stale objects).
    println!("cargo:rerun-if-changed=src/native/avx10_v1_aux.c");
    println!("cargo:rerun-if-changed=src/native/avx10_v2_aux.c");

    // Cargo passes `--cfg feature="native"` when it compiles this build script, and the
    // optional `cc` build-dependency is only present (linkable) under that feature. So the
    // `cc`-using path is itself feature-gated: in the default build it is not compiled at
    // all, the `cc` crate is never pulled in, and the build script is a no-op.
    #[cfg(feature = "native")]
    compile_native();
}

/// Compile the AVX10_V1_AUX and AVX10_V2_AUX C shims with `-mavx10.2`, but only on an
/// x86_64 target. On any other architecture the native feature still produces no compiled C
/// (the EVEX forms only exist on x86_64), preserving correctness on non-x86 targets.
#[cfg(feature = "native")]
fn compile_native() {
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if arch != "x86_64" {
        return;
    }
    // Both TUs link into ONE static archive. Its name is the iteration-1 archive renamed in
    // place from `ace_native_avx10_v1_aux` to the umbrella `ace_native_avx10_aux`, because the
    // single archive now carries both the V1_AUX and V2_AUX shims — a `v1`-specific name would
    // be inaccurate. The name is only the archive filename (`libace_native_avx10_aux.a`); `cc`
    // emits the matching `cargo:rustc-link-lib` directive automatically, so nothing references
    // the old name and there is no stale artifact.
    cc::Build::new()
        .file("src/native/avx10_v1_aux.c")
        .file("src/native/avx10_v2_aux.c")
        .flag("-mavx10.2")
        .opt_level(2)
        .compile("ace_native_avx10_aux");
}
