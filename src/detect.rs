//! Crate-owned `AVX10_V1_AUX` capability check.
//!
//! `std_detect` exposes no stable token for the AVX10.2-subset features this crate
//! targets, so detection is a hand-rolled CPUID probe. The capability gate is the
//! layered check `(AVX10.1 AND AVX10_V1_AUX) OR AVX10.2` together with the
//! `XCR0`/`CR4.OSXSAVE` operating-system enablement state bits
//! (`[avx10-v1-aux-fp16-fp8-evex-vnni.DISPATCH.2]`).
//!
//! On non-x86_64 targets the check returns `false` so the public dispatchers always
//! select the scalar oracle (`[avx10-v1-aux-fp16-fp8-evex-vnni.DISPATCH.3]`).

/// Returns `true` when the running CPU supports `AVX10_V1_AUX` with OS state enabled.
///
/// Gates on `CPUID.(EAX=24H,ECX=1):ECX[2]` under the layered check
/// `(AVX10.1 AND AVX10_V1_AUX) OR AVX10.2`, plus the `XCR0`
/// (AVX/opmask/ZMM state) and `CR4.OSXSAVE` bits surfaced through CPUID.
/// `[avx10-v1-aux-fp16-fp8-evex-vnni.DISPATCH.2]`
#[cfg(target_arch = "x86_64")]
pub(crate) fn has_avx10_v1_aux() -> bool {
    use core::arch::x86_64::{__cpuid, __cpuid_count, _xgetbv};

    // CPUID leaf 0 reports the maximum standard leaf. AVX10 lives at leaf 0x24, so
    // a CPU that does not even advertise that leaf cannot support AVX10_V1_AUX.
    let max_leaf = __cpuid(0).eax;
    if max_leaf < 0x24 {
        return false;
    }

    // OS enablement: CR4.OSXSAVE (CPUID.1:ECX[27]) gates XGETBV, and XCR0 must have
    // the SSE (bit 1), AVX (bit 2), and AVX-512 opmask/ZMM-hi/hi16-ZMM (bits 5,6,7)
    // state-save bits set, otherwise an EVEX-encoded native path would fault.
    let leaf1 = __cpuid(1);
    let osxsave = (leaf1.ecx >> 27) & 1 != 0;
    if !osxsave {
        return false;
    }
    let xcr0 = unsafe { _xgetbv(0) };
    const XCR0_REQUIRED: u64 = (1 << 1) | (1 << 2) | (1 << 5) | (1 << 6) | (1 << 7);
    if xcr0 & XCR0_REQUIRED != XCR0_REQUIRED {
        return false;
    }

    // AVX10 converged-ISA leaf: CPUID.(EAX=24H,ECX=0):EBX.
    //   bit 16 = AVX10 supported at all
    //   bits 7:0 = AVX10 converged version number (>= 1 means AVX10.1, >= 2 AVX10.2)
    let avx10 = __cpuid_count(0x24, 0);
    let avx10_supported = (avx10.ebx >> 16) & 1 != 0;
    if !avx10_supported {
        return false;
    }
    let version = avx10.ebx & 0xff;
    let avx10_1 = version >= 1;
    let avx10_2 = version >= 2;

    // AVX10_V1_AUX bit: CPUID.(EAX=24H,ECX=1):ECX[2].
    let aux = __cpuid_count(0x24, 1);
    let avx10_v1_aux = (aux.ecx >> 2) & 1 != 0;

    // Layered guard: (AVX10.1 AND AVX10_V1_AUX) OR AVX10.2.
    (avx10_1 && avx10_v1_aux) || avx10_2
}

/// Non-x86_64 stub: no AVX10 capability exists, so the dispatcher always selects the
/// scalar oracle. `[avx10-v1-aux-fp16-fp8-evex-vnni.DISPATCH.3]`
#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn has_avx10_v1_aux() -> bool {
    false
}
