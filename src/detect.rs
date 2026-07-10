//! Crate-owned `AVX10_V1_AUX` / `AVX10_V2_AUX` capability checks.
//!
//! `std_detect` exposes no stable token for the AVX10.2-subset features this crate
//! targets, so detection is a hand-rolled CPUID probe. The capability gate is
//! `AVX10.2` (plus the `AVX10_V2_AUX` bit for the group-3 converts) together with the
//! `XCR0`/`CR4.OSXSAVE` operating-system enablement state bits
//! (`[avx10-v1-aux-fp16-fp8-evex-vnni.DISPATCH.2]`,
//! `[avx10-v2-aux-ocp-conversions.DETECTION.1]`).
//!
//! The spec's layered check `(AVX10.1 AND AVX10_V1_AUX) OR AVX10.2` describes when the
//! *instructions* exist, but these gates guard the crate's native shims, which are
//! compiled as whole translation units with `-mavx10.2` / `target("avx10.2")` — the
//! compiler may emit any AVX10.2 instruction inside them, not only the AUX subset. A CPU
//! satisfying the `(AVX10.1 AND AVX10_V1_AUX)` arm without AVX10.2 could therefore still
//! `#UD` inside a shim, so the gates require the full `AVX10.2` the shims are built for.
//! Every result is cached (`OnceLock`) — CPUID feature bits are immutable for the life of
//! the process, and the raw probe costs several serializing CPUIDs per call.
//!
//! On non-x86_64 targets the checks return `false` so the public dispatchers always
//! select the scalar oracle (`[avx10-v1-aux-fp16-fp8-evex-vnni.DISPATCH.3]`,
//! `[avx10-v2-aux-ocp-conversions.DETECTION.2]`).
//!
//! Both checks share one CPUID/XCR0 probe ([`avx10_base`]); each then applies the one
//! feature bit and layered guard that distinguishes it. They differ only in which
//! sub-leaf-1 `ECX` bit they read (V1_AUX = bit 2, V2_AUX = bit 3) and whether the
//! `AVX10_V2_AUX` bit is additionally required.

/// XCR0 XSAVE-state bits required by the EVEX-encoded AVX10.2 *vector* converts:
/// SSE (1), AVX/`YMM_Hi128` (2), opmask (5), `ZMM_Hi256` (6), `Hi16_ZMM` (7).
///
/// These are the only state bits the `AVX10_V1_AUX` / `AVX10_V2_AUX` OCP format converts
/// touch — they read and write XMM/YMM/ZMM (and opmask) registers only. The ACE v1
/// spec §3.2 *full-ACE-v1* detection algorithm additionally lists `XCR0[20,18:17]=0b111`
/// (the AMX-tile + BSR XSAVE state), but those bits belong to the **group-4 tile
/// instructions** (`TOP*`/`BSR*`/tile moves), which are out of scope here and use a
/// different register file. Requiring them to gate the group-3 vector converts would
/// wrongly reject a CPU that supports the converts but whose OS has not enabled the
/// AMX/BSR state, so they are deliberately NOT part of this gate.
#[cfg(target_arch = "x86_64")]
const XCR0_VECTOR_STATE: u64 = (1 << 1) | (1 << 2) | (1 << 5) | (1 << 6) | (1 << 7);

/// Shared OS-enablement + AVX10-base probe common to every AVX10.2-subset capability
/// check. Returns `Some((AVX10_VSN, sub-leaf-1 ECX))` once the AVX10 leaf, `CR4.OSXSAVE`,
/// the `XCR0` vector state, the AVX10-supported bit, and 512-bit vector support are all
/// confirmed; `None` if any precondition fails (in which case the caller reports no
/// capability). `AVX10_VSN >= 1` means AVX10.1, `>= 2` means AVX10.2 (spec section 3.1).
#[cfg(target_arch = "x86_64")]
fn avx10_base() -> Option<(u32, u32)> {
    use core::arch::x86_64::{__cpuid, __cpuid_count, _xgetbv};

    // CPUID leaf 0 reports the maximum standard leaf. AVX10 lives at leaf 0x24, so a CPU
    // that does not even advertise that leaf cannot support any AVX10.2-subset feature.
    if __cpuid(0).eax < 0x24 {
        return None;
    }

    // OS enablement (spec §3.2 steps 6-7): CR4.OSXSAVE (CPUID.1:ECX[27]) gates XGETBV, and
    // XCR0 must have the AVX-512 vector state enabled, otherwise an EVEX-encoded native
    // path would fault.
    if (__cpuid(1).ecx >> 27) & 1 == 0 {
        return None;
    }
    // SAFETY: CR4.OSXSAVE (CPUID.1:ECX[27]) was confirmed set immediately above, so
    // XGETBV with ECX=0 is a defined, non-faulting read of XCR0.
    let xcr0 = unsafe { _xgetbv(0) };
    if xcr0 & XCR0_VECTOR_STATE != XCR0_VECTOR_STATE {
        return None;
    }

    // AVX10 enumeration proper: CPUID.(EAX=07H,ECX=1):EDX[19]. This — not any bit in leaf
    // 0x24 — is the architectural "AVX10 supported" bit (ACE v1 spec section 3.1).
    if (__cpuid_count(7, 1).edx >> 19) & 1 == 0 {
        return None;
    }

    // AVX10 converged-ISA leaf: CPUID.(EAX=24H,ECX=0):EBX.
    //   bits 7:0  = AVX10 converged version (>= 1 means AVX10.1, >= 2 means AVX10.2)
    //   bits 18:16 = RESERVED, architecturally reading "111" (ACE v1 spec section 3.1 /
    //   15.5.1). Bit 18 was the 512-bit vector-length bit in AVX10 spec rev 1.0; checking
    //   it accepts both the rev-1.0 reading and the fixed "111" reading, and every native
    //   shim in this crate is 512-bit (ZMM), so a part reporting it 0 must stay on the
    //   scalar oracle.
    let avx10 = __cpuid_count(0x24, 0);
    if (avx10.ebx >> 18) & 1 == 0 {
        return None;
    }
    let version = avx10.ebx & 0xff;

    // The AUX feature bits live in CPUID.(EAX=24H,ECX=1):ECX (V1_AUX at bit 2, V2_AUX at
    // bit 3, spec section 3.1); hand the raw ECX back so each caller reads what it needs.
    let aux_ecx = __cpuid_count(0x24, 1).ecx;
    Some((version, aux_ecx))
}

/// The spec section-3.2 step-1 base arm: `(AVX10.1 AND AVX10_V1_AUX) OR AVX10.2`,
/// evaluated on top of [`avx10_base`]'s OS-state and enumeration checks.
#[cfg(target_arch = "x86_64")]
fn avx10_step1_base_arm() -> bool {
    match avx10_base() {
        Some((version, aux_ecx)) => {
            let v1_aux = (aux_ecx >> 2) & 1 != 0;
            (version >= 1 && v1_aux) || version >= 2
        }
        None => false,
    }
}

/// Returns `true` when the running CPU supports the `AVX10_V1_AUX` native shims with OS
/// state enabled.
///
/// Requires full `AVX10.2` (leaf 0x24 version >= 2), plus the [`XCR0_VECTOR_STATE`] and
/// `CR4.OSXSAVE` bits surfaced through CPUID. The spec's weaker
/// `(AVX10.1 AND AVX10_V1_AUX)` arm is deliberately NOT accepted: it proves the AUX
/// *instructions* exist, but the shims this gate guards are compiled with
/// `target("avx10.2")`, so only full AVX10.2 makes calling them sound (see the module
/// docs). Cached after the first probe. `[avx10-v1-aux-fp16-fp8-evex-vnni.DISPATCH.2]`
#[cfg(target_arch = "x86_64")]
pub(crate) fn has_avx10_v1_aux() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        let Some((version, _aux)) = avx10_base() else {
            return false;
        };
        // Shim-soundness guard: full AVX10.2 only (module docs explain why the spec's
        // (AVX10.1 AND AVX10_V1_AUX) arm is insufficient for `-mavx10.2` shims).
        version >= 2
    })
}

/// Non-x86_64 stub: no AVX10 capability exists, so the dispatcher always selects the
/// scalar oracle. `[avx10-v1-aux-fp16-fp8-evex-vnni.DISPATCH.3]`
#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn has_avx10_v1_aux() -> bool {
    false
}

/// Returns `true` when the running CPU supports `AVX10_V2_AUX` with OS state enabled.
///
/// Gates on `CPUID.(EAX=24H,ECX=1):ECX[3]` (the `AVX10_V2_AUX` token — FP32->FP8
/// converts, the FP4/FP6 converts, `VPMOVSSDB`, `VUNPACKB`) AND full `AVX10.2`, together
/// with the [`XCR0_VECTOR_STATE`] (AVX-512 vector/opmask XSAVE state) and `CR4.OSXSAVE`
/// operating-system enablement bits, otherwise an EVEX-encoded native path would fault
/// (`[avx10-v2-aux-ocp-conversions.DETECTION.1]`). As with [`has_avx10_v1_aux`], the ACE
/// v1 spec §3.2 layered base check `(AVX10.1 AND AVX10_V1_AUX) OR AVX10.2` is tightened
/// to full `AVX10.2` because any future V2_AUX shim will be compiled `-mavx10.2` (module
/// docs). Cached after the first probe.
///
/// NOTE: §3.2 also lists the tile + BSR XSAVE state (`XCR0[20,18:17]`) for *full ACE v1*
/// support, but those belong to the out-of-scope group-4 tile instructions and are not
/// required to issue the group-3 vector converts — see [`XCR0_VECTOR_STATE`].
#[cfg(target_arch = "x86_64")]
pub(crate) fn has_avx10_v2_aux() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        let Some((version, aux)) = avx10_base() else {
            return false;
        };
        let avx10_v2_aux = (aux >> 3) & 1 != 0;
        // Shim-soundness guard: full AVX10.2 AND the AVX10_V2_AUX feature itself.
        version >= 2 && avx10_v2_aux
    })
}

/// Non-x86_64 stub: no AVX10 capability exists, so the dispatcher always selects the
/// scalar oracle. `[avx10-v2-aux-ocp-conversions.DETECTION.2]`
#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn has_avx10_v2_aux() -> bool {
    false
}

// ===================== ACE group-4 tile-instruction detection =====================
//
// The group-4 tile instructions (families A-G: tile lifecycle, moves, tile-row converts,
// BSR register, and the `TOP*` outer products) use a stateful register file — the AMX-tile
// data/config registers plus the ACE block-scale register (SCALEDATA) — that the group-3
// vector converts above never touch. They therefore need SEPARATE XSAVE-state masks and a
// per-family capability gate rather than the single `XCR0_VECTOR_STATE` / AVX10 check.
//
// `std_detect` still exposes no stable token for AMX-TILE/AMX-AVX512/ACE, so — exactly as
// [`avx10_base`] does for AVX10 — these are hand-rolled CPUID probes. All positions are
// confirmed against the rev-1.15 PDF (sections 3.1, 15.5, Appendix A): AMX-TILE = leaf 07H
// EDX[24]; AMX-AVX512 = leaf 1EH sub-leaf 1 EAX[7]; ACE = leaf 07H sub-leaf 1 ECX[11];
// ACE_VSN = leaf 1DH sub-leaf 2 EAX[7:0] (the spec redefines that sub-leaf: EAX[7:0] =
// ACE_VSN, EAX[31:8] reserved, populated with zeros when ACE is absent — section 15.5.1).
// Per the spec formula `ACEv1 = (ACE AND ACE_VSN >= 1)`, `ACE_VSN` is never consulted
// without the ACE feature bit. Non-x86_64 targets stub every helper to `false` so the
// dispatchers always take the scalar oracle.
//
// XSAVE granularity (spec section 15.4): the existing AMX framework instructions
// (LDTILECFG/STTILECFG/TILEZERO/TILERELEASE) remain sensitive to only `XCR0[18:17]`
// (section 15.4.1) — NOT to the ACE-only `XCR0[20]` (SCALEDATA), which cannot even be set
// on non-ACE parts. Only the ACE instruction families require `XCR0[20,18:17] = 0b111`
// (sections 3.2 and 15.4.6). Gating the plain AMX families on bit 20 would permanently
// disable them on every existing AMX CPU.

/// XSAVE state for the plain AMX framework instructions (family A):
/// `XCR0[18:17]` — bit 17 = TILECFG, bit 18 = TILEDATA (spec section 15.4.1).
#[cfg(target_arch = "x86_64")]
const XCR0_AMX_FRAMEWORK_STATE: u64 = (1 << 18) | (1 << 17);

/// XSAVE state required before an ACE-family native path may run: the AVX-512 vector state
/// PLUS the AMX-tile + SCALEDATA state bits `XCR0[20,18:17]=0b111` (spec sections 3.2 and
/// 15.4.6) (`[ace-tile-instructions.DETECT.2]`).
///
/// This is a SEPARATE constant, deliberately NOT a widening of [`XCR0_VECTOR_STATE`]: the
/// group-3 vector converts must keep gating on the vector state alone (see that constant's
/// docs). Bit 17 = tile config, bit 18 = tile data, bit 20 = the ACE block-scale
/// (SCALEDATA) state.
#[cfg(target_arch = "x86_64")]
const XCR0_TILE_STATE: u64 = XCR0_VECTOR_STATE | (1 << 20) | (1 << 18) | (1 << 17);

/// `true` when `CR4.OSXSAVE` is set and `XCR0` has all the bits in `mask` enabled.
#[cfg(target_arch = "x86_64")]
fn xcr0_state_enabled(mask: u64) -> bool {
    use core::arch::x86_64::{__cpuid, _xgetbv};
    // CR4.OSXSAVE (CPUID.1:ECX[27]) gates XGETBV.
    if (__cpuid(1).ecx >> 27) & 1 == 0 {
        return false;
    }
    // SAFETY: CR4.OSXSAVE (CPUID.1:ECX[27]) was confirmed set immediately above, so
    // XGETBV with ECX=0 is a defined, non-faulting read of XCR0.
    let xcr0 = unsafe { _xgetbv(0) };
    xcr0 & mask == mask
}

/// Per-process tile-state permission (`IA32_XFD[18]` handling, spec sections 15.4.3 and
/// 15.4.6): every tile-data instruction raises `#NM` while `IA32_XFD[18] = 1`, and on
/// Linux the kernel keeps `XFD[18]` set for each process until it requests the dynamic
/// XSTATE component via `arch_prctl(ARCH_REQ_XCOMP_PERM, <feature>)` — even though
/// `XGETBV` already reports `XCR0[18] = 1`. So a CPUID/XCR0-only probe would report tile
/// support while the very first native tile shim faults.
///
/// `feature` is the highest dynamic XSTATE component number needed (18 = TILEDATA for the
/// plain AMX families, 20 = SCALEDATA for the ACE families). Returns `true` when the
/// kernel grants (or has already granted) the permission. On x86_64 non-Linux targets
/// this returns `true`: no equivalent userspace request exists there and the OSes that
/// enable the XCR0 bits do so without a per-process opt-in.
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
fn request_xcomp_perm(feature: u64) -> bool {
    // arch_prctl(2) codes from the kernel ABI (asm/prctl.h).
    const ARCH_GET_XCOMP_PERM: u64 = 0x1022;
    const ARCH_REQ_XCOMP_PERM: u64 = 0x1023;
    const SYS_ARCH_PRCTL: u64 = 158;

    unsafe fn arch_prctl(code: u64, addr: u64) -> i64 {
        let ret: i64;
        // SAFETY: raw `arch_prctl` syscall; both commands only read/update the calling
        // process's XSTATE permission bitmap and write nothing through `addr` for
        // ARCH_REQ_XCOMP_PERM (addr is the feature number). For ARCH_GET_XCOMP_PERM the
        // kernel writes a u64 bitmap through the pointer in `addr`.
        unsafe {
            core::arch::asm!(
                "syscall",
                in("rax") SYS_ARCH_PRCTL,
                in("rdi") code,
                in("rsi") addr,
                out("rcx") _,
                out("r11") _,
                lateout("rax") ret,
                options(nostack),
            );
        }
        ret
    }

    // Fast path: already permitted?
    let mut bitmap: u64 = 0;
    // SAFETY: `bitmap` is a valid writable u64 for ARCH_GET_XCOMP_PERM.
    if unsafe { arch_prctl(ARCH_GET_XCOMP_PERM, &mut bitmap as *mut u64 as u64) } == 0
        && bitmap & (1 << feature) != 0
    {
        return true;
    }
    // SAFETY: ARCH_REQ_XCOMP_PERM takes the feature number by value.
    unsafe { arch_prctl(ARCH_REQ_XCOMP_PERM, feature) == 0 }
}

/// Non-Linux x86_64 stub (see the Linux variant's docs).
#[cfg(all(target_arch = "x86_64", not(target_os = "linux")))]
fn request_xcomp_perm(_feature: u64) -> bool {
    true
}

/// `ACE_VSN`, the ACE version, read from `CPUID.(EAX=1DH,ECX=2):EAX[7:0]` (spec sections
/// 3.1 and 15.5.1); `0` when the leaf is absent. `>= 1` denotes ACE v1. The spec defines
/// this sub-leaf as EAX[7:0] = ACE_VSN with EAX[31:8] reserved, populated with zeros when
/// ACE is not implemented; per the spec formula `ACEv1 = (ACE AND ACE_VSN >= 1)` callers
/// consult it only together with the ACE feature bit ([`ace_v1_present`]).
#[cfg(target_arch = "x86_64")]
fn ace_vsn() -> u32 {
    use core::arch::x86_64::{__cpuid, __cpuid_count};
    // Guard the leaf: a CPU whose max standard leaf is below 0x1D cannot report ACE_VSN.
    if __cpuid(0).eax < 0x1d {
        return 0;
    }
    __cpuid_count(0x1d, 2).eax & 0xff
}

/// `true` when the ACE feature bit `CPUID.(EAX=07H,ECX=1):ECX[11]` is set AND
/// `ACE_VSN >= 1` — the spec's `ACEv1 = (ACE AND ACE_VSN >= 1)` formula (sections 3.1,
/// 15.5.1, Appendix A; §3.2 steps 3-4).
#[cfg(target_arch = "x86_64")]
fn ace_v1_present() -> bool {
    use core::arch::x86_64::__cpuid_count;
    let ace_bit = (__cpuid_count(7, 1).ecx >> 11) & 1 != 0; // ACE = Fn07H/1 ECX[11] (§3.1)
    ace_bit && ace_vsn() >= 1
}

/// `true` when the tile file enumerates the palette-2 (ACE) descriptor.
///
/// `LDTILECFG` raises `#GP` on an unsupported palette id (spec section 15.2.2.3), and
/// palette support above 0 is implementation-defined (section 15.2.2.4): the ACE tile file
/// is enumerated solely through the ACE feature bit + `ACE_VSN` ([`ace_v1_present`],
/// sections 3.1/15.5.1). A plain-AMX host — including Intel SDE's `-future` model — has
/// AMX-TILE / AMX-AVX512 but only palette 1, so its `LDTILECFG` `#GP`s on the palette-2
/// descriptor even though the family-A/B-read/C *instructions* themselves execute fine.
/// Every native path that loads the palette-2 config must therefore gate on this IN
/// ADDITION to its per-family capability probe (`has_amx_tile` / `has_amx_avx512`); this
/// helper deliberately repeats none of the XSAVE/permission checks those probes do.
///
/// Only the `native`-feature differential layer loads the palette-2 config natively today,
/// so the helper is compiled (and dead-code-allowed outside tests) exactly like
/// `mod native` itself.
#[cfg(all(target_arch = "x86_64", feature = "native"))]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn has_palette2() -> bool {
    ace_v1_present()
}

/// Returns `true` when the running CPU supports the AMX-TILE capability with the AMX
/// framework XSAVE state enabled and per-process tile permission granted — the native gate
/// for family A (tile config lifecycle)
/// (`[ace-tile-instructions.DETECT.1]`, `[ace-tile-instructions.DETECT.1-1]`).
///
/// Gates on `CPUID.(EAX=07H,ECX=0):EDX[24]` (AMX-TILE, spec section 3.1) plus
/// `XCR0[18:17]` ([`XCR0_AMX_FRAMEWORK_STATE`] — the plain AMX framework instructions are
/// sensitive to only these two bits, spec section 15.4.1, NOT the ACE-only `XCR0[20]`)
/// plus the Linux `arch_prctl` TILEDATA permission ([`request_xcomp_perm`]).
#[cfg(target_arch = "x86_64")]
pub(crate) fn has_amx_tile() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        use core::arch::x86_64::__cpuid_count;
        if !xcr0_state_enabled(XCR0_AMX_FRAMEWORK_STATE) {
            return false;
        }
        // AMX-TILE = CPUID.(EAX=07H,ECX=0):EDX[24].
        if (__cpuid_count(7, 0).edx >> 24) & 1 == 0 {
            return false;
        }
        // IA32_XFD[18] must be 0 for tile-data access (spec section 15.4.6).
        request_xcomp_perm(18)
    })
}

/// Returns `true` when the running CPU supports the tile-row read/convert path — the native
/// gate for family C (tile-row converts) and the `TILEMOVROW` read form
/// (`[ace-tile-instructions.DETECT.1]`, `[ace-tile-instructions.DETECT.1-2]`).
///
/// The spec section-15.3 feature enumeration for these instructions is
/// `AMX-AVX512 || ACE_VSN >= 1`, on top of AMX-TILE. AMX-AVX512 =
/// `CPUID.(EAX=1EH,ECX=1):EAX[7]` (leaf 1EH sub-leaf 1 EAX enumerates the AMX feature
/// flags; the spec populates the leaf with zeros on ACE-only parts, which the
/// `ace_v1_present()` arm covers — section 15.5.2). These instructions also write ZMM
/// state, so the AVX-512 vector XSAVE state is required (section 15.4.6); [`avx10_base`]
/// additionally requires the AVX10 enumeration because the crate's family-C shims are
/// compiled `-mavx10.2` (a shim-soundness tightening, see the module docs).
#[cfg(target_arch = "x86_64")]
pub(crate) fn has_amx_avx512() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        use core::arch::x86_64::{__cpuid, __cpuid_count};
        if !has_amx_tile() {
            return false;
        }
        // ZMM writes need the AVX-512 vector XSAVE state; avx10_base also proves the
        // AVX10 enumeration the `-mavx10.2`-compiled shim needs.
        if avx10_base().is_none() {
            return false;
        }
        // AMX-AVX512 = CPUID.(EAX=1EH,ECX=1):EAX[7]; guard the leaf first.
        let amx_avx512 = __cpuid(0).eax >= 0x1e && (__cpuid_count(0x1e, 1).eax >> 7) & 1 != 0;
        amx_avx512 || ace_v1_present()
    })
}

/// Returns `true` when the running CPU supports the full ACE v1 capability — the native gate
/// for families D/E/F/G and the write-form tile moves
/// (`[ace-tile-instructions.DETECT.1]`, `[ace-tile-instructions.DETECT.1-3]`).
///
/// Implements ALL seven steps of the spec section-3.2 detection algorithm:
/// 1. `(AVX10.1 AND AVX10_V1_AUX) OR AVX10.2` ([`avx10_step1_base_arm`])
/// 2. `AVX10_V2_AUX` (`CPUID.(EAX=24H,ECX=1):ECX[3]`)
/// 3. `ACE` (`CPUID.(EAX=07H,ECX=1):ECX[11]`)
/// 4. `ACE_VSN >= 1` (`CPUID.(EAX=1DH,ECX=2):EAX[7:0]`)
/// 5. `XCR0[20,18:17] = 0b111` (tile + SCALEDATA XSAVE state)
/// 6. `XCR0[7:5] = 0b111` (AVX-512 state; [`avx10_base`] also checks `XCR0[2:1]`,
///    required by section 15.4.6 for ACE instructions that access AVX state)
/// 7. `CR4.OSXSAVE = 1`
///
/// plus the per-process SCALEDATA permission (`IA32_XFD`, spec sections 15.4.3/15.4.6,
/// [`request_xcomp_perm`]) and AMX-TILE (spec section 3.1 lists it as a required AMX
/// component; also keeps `has_ace() => has_amx_tile()`).
#[cfg(target_arch = "x86_64")]
pub(crate) fn has_ace() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        // Steps 1-2 (+ steps 6-7 via avx10_base's CR4.OSXSAVE/XCR0 vector checks).
        if !avx10_step1_base_arm() {
            return false;
        }
        let Some((_, aux)) = avx10_base() else {
            return false;
        };
        if (aux >> 3) & 1 == 0 {
            return false; // step 2: AVX10_V2_AUX
        }
        // Steps 3-4.
        if !ace_v1_present() {
            return false;
        }
        // Step 5 (plus vector state again; XCR0_TILE_STATE is a superset of step 6).
        if !xcr0_state_enabled(XCR0_TILE_STATE) {
            return false;
        }
        // AMX-TILE component + TILEDATA permission.
        if !has_amx_tile() {
            return false;
        }
        // SCALEDATA (dynamic XSTATE component 20) permission.
        request_xcomp_perm(20)
    })
}

/// Non-x86_64 stubs: no tile capability exists, so every tile dispatcher takes the scalar
/// oracle (`[ace-tile-instructions.DETECT.1]`).
#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn has_amx_tile() -> bool {
    false
}

/// Non-x86_64 stub (see [`has_amx_tile`]).
#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn has_amx_avx512() -> bool {
    false
}

/// Non-x86_64 stub (see [`has_amx_tile`]).
#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn has_ace() -> bool {
    false
}

#[cfg(test)]
mod tests {
    /// `XCR0_TILE_STATE` is the vector state PLUS exactly the AMX-tile + BSR bits
    /// `XCR0[20,18:17]`, and it does NOT widen [`super::XCR0_VECTOR_STATE`] (the group-3
    /// vector gate) (`[ace-tile-instructions.DETECT.2]`). Checked only on x86_64, where the
    /// masks are defined.
    /// `detect::xcr0_tile_state_mask_bits`
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn xcr0_tile_state_mask_bits() {
        let vector = super::XCR0_VECTOR_STATE;
        let tile = super::XCR0_TILE_STATE;
        // The three tile+BSR bits are present in the tile mask...
        for bit in [17u64, 18, 20] {
            assert_eq!(tile & (1 << bit), 1 << bit, "tile mask includes bit {bit}");
            // ...and absent from the (unchanged) vector mask.
            assert_eq!(
                vector & (1 << bit),
                0,
                "vector mask must NOT include tile bit {bit}"
            );
        }
        // Tile mask == vector mask plus exactly those three bits (no other bits added or lost).
        assert_eq!(
            tile,
            vector | (1 << 20) | (1 << 18) | (1 << 17),
            "tile mask is the vector mask plus XCR0[20,18:17] only"
        );
        assert_eq!(
            tile & vector,
            vector,
            "tile mask is a strict superset of the vector mask"
        );
    }

    /// Per-family gate helpers are callable, return a `bool`, and compose consistently: full
    /// ACE implies AMX-TILE (has_ace requires has_amx_tile) and implies the family-C gate.
    /// On non-x86 every helper is a `false` stub (`[ace-tile-instructions.DETECT.1]`,
    /// `[ace-tile-instructions.DETECT.1-1]`).
    /// `detect::per_family_gate_helpers`
    #[test]
    fn per_family_gate_helpers() {
        let amx_tile = super::has_amx_tile();
        let amx_avx512 = super::has_amx_avx512();
        let ace = super::has_ace();

        #[cfg(not(target_arch = "x86_64"))]
        {
            // Stubs: the tile dispatchers always fall back to the scalar oracle off-x86.
            assert!(
                !amx_tile && !amx_avx512 && !ace,
                "non-x86 helpers stub to false"
            );
        }

        #[cfg(target_arch = "x86_64")]
        {
            // Composition invariants hold regardless of what this host reports (§3.2):
            // full ACE is a strict refinement of AMX-TILE and of the family-C gate.
            if ace {
                assert!(
                    amx_tile,
                    "has_ace() implies has_amx_tile() (AMX-TILE ∧ ...)"
                );
                assert!(
                    amx_avx512,
                    "has_ace() implies the family-C gate (ACE_VSN >= 1)"
                );
            }
            // Idempotent / side-effect-free: a second read agrees with the first.
            assert_eq!(amx_tile, super::has_amx_tile());
            assert_eq!(amx_avx512, super::has_amx_avx512());
            assert_eq!(ace, super::has_ace());
        }
    }
}
