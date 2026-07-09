//! Crate-owned `AVX10_V1_AUX` / `AVX10_V2_AUX` capability checks.
//!
//! `std_detect` exposes no stable token for the AVX10.2-subset features this crate
//! targets, so detection is a hand-rolled CPUID probe. The capability gate is the
//! layered check `(AVX10.1 AND AVX10_V1_AUX) OR AVX10.2` (plus the `AVX10_V2_AUX` bit for
//! the group-3 converts) together with the `XCR0`/`CR4.OSXSAVE` operating-system
//! enablement state bits (`[avx10-v1-aux-fp16-fp8-evex-vnni.DISPATCH.2]`,
//! `[avx10-v2-aux-ocp-conversions.DETECTION.1]`).
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
/// check. Returns `Some((avx10.1, avx10.2, sub-leaf-1 ECX))` once the AVX10 leaf,
/// `CR4.OSXSAVE`, the `XCR0` vector state, and the AVX10-supported bit are all confirmed;
/// `None` if any precondition fails (in which case the caller reports no capability).
#[cfg(target_arch = "x86_64")]
fn avx10_base() -> Option<(bool, bool, u32)> {
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
    let xcr0 = unsafe { _xgetbv(0) };
    if xcr0 & XCR0_VECTOR_STATE != XCR0_VECTOR_STATE {
        return None;
    }

    // AVX10 converged-ISA leaf: CPUID.(EAX=24H,ECX=0):EBX.
    //   bit 16    = AVX10 supported at all
    //   bits 7:0  = AVX10 converged version (>= 1 means AVX10.1, >= 2 means AVX10.2)
    let avx10 = __cpuid_count(0x24, 0);
    if (avx10.ebx >> 16) & 1 == 0 {
        return None;
    }
    let version = avx10.ebx & 0xff;

    // The AUX feature bits live in CPUID.(EAX=24H,ECX=1):ECX (V1_AUX at bit 2, V2_AUX at
    // bit 3); hand the raw ECX back so each caller reads the bit it needs.
    let aux_ecx = __cpuid_count(0x24, 1).ecx;
    Some((version >= 1, version >= 2, aux_ecx))
}

/// Returns `true` when the running CPU supports `AVX10_V1_AUX` with OS state enabled.
///
/// Gates on `CPUID.(EAX=24H,ECX=1):ECX[2]` under the layered check
/// `(AVX10.1 AND AVX10_V1_AUX) OR AVX10.2`, plus the [`XCR0_VECTOR_STATE`] and
/// `CR4.OSXSAVE` bits surfaced through CPUID. `[avx10-v1-aux-fp16-fp8-evex-vnni.DISPATCH.2]`
#[cfg(target_arch = "x86_64")]
pub(crate) fn has_avx10_v1_aux() -> bool {
    let Some((avx10_1, avx10_2, aux)) = avx10_base() else {
        return false;
    };
    let avx10_v1_aux = (aux >> 2) & 1 != 0;
    // Layered guard: (AVX10.1 AND AVX10_V1_AUX) OR AVX10.2.
    (avx10_1 && avx10_v1_aux) || avx10_2
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
/// converts, the FP4/FP6 converts, `VPMOVSSDB`, `VUNPACKB`) under the ACE v1 spec §3.2
/// layered detection:
///
///   1. `(AVX10.1 AND AVX10_V1_AUX) OR AVX10.2`
///   2. `AVX10_V2_AUX`
///
/// together with the [`XCR0_VECTOR_STATE`] (AVX-512 vector/opmask XSAVE state) and
/// `CR4.OSXSAVE` operating-system enablement bits, otherwise an EVEX-encoded native path
/// would fault (`[avx10-v2-aux-ocp-conversions.DETECTION.1]`).
///
/// NOTE: §3.2 also lists the tile + BSR XSAVE state (`XCR0[20,18:17]`) for *full ACE v1*
/// support, but those belong to the out-of-scope group-4 tile instructions and are not
/// required to issue the group-3 vector converts — see [`XCR0_VECTOR_STATE`]. The
/// iteration-1 [`has_avx10_v1_aux`] gate (which reads `ECX[2]`) shares the same base probe
/// and is behaviourally unchanged.
#[cfg(target_arch = "x86_64")]
pub(crate) fn has_avx10_v2_aux() -> bool {
    let Some((avx10_1, avx10_2, aux)) = avx10_base() else {
        return false;
    };
    let avx10_v1_aux = (aux >> 2) & 1 != 0;
    let avx10_v2_aux = (aux >> 3) & 1 != 0;
    // Layered guard (§3.2 steps 1-2): the base ISA must be present
    // ((AVX10.1 AND AVX10_V1_AUX) OR AVX10.2) AND the AVX10_V2_AUX feature itself.
    ((avx10_1 && avx10_v1_aux) || avx10_2) && avx10_v2_aux
}

/// Non-x86_64 stub: no AVX10 capability exists, so the dispatcher always selects the
/// scalar oracle. `[avx10-v2-aux-ocp-conversions.DETECTION.2]`
#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn has_avx10_v2_aux() -> bool {
    false
}
