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

// ===================== ACE group-4 tile-instruction detection =====================
//
// The group-4 tile instructions (families A-G: tile lifecycle, moves, tile-row converts,
// BSR registers, and the `TOP*` outer products) use a stateful register file — the AMX-tile
// data/config registers plus the ACE block-scale registers (`BSR*`) — that the group-3
// vector converts above never touch. They therefore need a SEPARATE XSAVE-state mask and a
// per-family capability gate rather than the single `XCR0_VECTOR_STATE` / AVX10 check.
//
// OQ-5 (hand-rolled CPUID composition): `std_detect` still exposes no stable token for
// `has_ace`/AMX-TILE/AMX-AVX512, so — exactly as [`avx10_base`] does for AVX10 — these are
// hand-rolled CPUID probes composed on the shared base probe. The exact ACE feature-bit and
// `ACE_VSN` leaf/sub-leaf positions transcribe the ACE v1 spec section 3.2 *full-ACE-v1*
// detection algorithm; they are Inferred (OQ-5) pending confirmation against the rev-1.15
// PDF and are the single place to correct once the tokens are pinned or `std_detect` gains
// them. Non-x86_64 targets stub every helper to `false` so the dispatchers always take the
// scalar oracle.

/// XSAVE state required before ANY native tile path may run: the AVX-512 vector state PLUS
/// the AMX-tile + BSR state bits `XCR0[20,18:17]=0b111` (spec section 3.2)
/// (`[ace-tile-instructions.DETECT.2]`).
///
/// This is a SEPARATE constant, deliberately NOT a widening of [`XCR0_VECTOR_STATE`]: the
/// group-3 vector converts must keep gating on the vector state alone (see that constant's
/// docs), so requiring the AMX/BSR bits there would wrongly reject a CPU that supports the
/// converts but whose OS has not enabled the tile state. Bit 17 = tile config, bit 18 = tile
/// data, bit 20 = the ACE block-scale (`BSR`) state.
#[cfg(target_arch = "x86_64")]
const XCR0_TILE_STATE: u64 = XCR0_VECTOR_STATE | (1 << 20) | (1 << 18) | (1 << 17);

/// `true` when `CR4.OSXSAVE` is set and `XCR0` has the full tile + BSR state
/// ([`XCR0_TILE_STATE`]) enabled — the OS-enablement precondition for every native tile
/// path (`[ace-tile-instructions.DETECT.2]`).
#[cfg(target_arch = "x86_64")]
fn xcr0_tile_state_enabled() -> bool {
    use core::arch::x86_64::{__cpuid, _xgetbv};
    // CR4.OSXSAVE (CPUID.1:ECX[27]) gates XGETBV.
    if (__cpuid(1).ecx >> 27) & 1 == 0 {
        return false;
    }
    let xcr0 = unsafe { _xgetbv(0) };
    xcr0 & XCR0_TILE_STATE == XCR0_TILE_STATE
}

/// `ACE_VSN`, the ACE version, read from `CPUID.(EAX=1DH,ECX=2):EAX[7:0]`; `0` when the leaf
/// is absent. Inferred leaf/sub-leaf/field (OQ-5). `>= 1` denotes ACE v1.
#[cfg(target_arch = "x86_64")]
fn ace_vsn() -> u32 {
    use core::arch::x86_64::{__cpuid, __cpuid_count};
    // Guard the leaf: a CPU whose max standard leaf is below 0x1D cannot report ACE_VSN.
    if __cpuid(0).eax < 0x1d {
        return 0;
    }
    __cpuid_count(0x1d, 2).eax & 0xff
}

/// Returns `true` when the running CPU supports the AMX-TILE capability with the tile + BSR
/// XSAVE state enabled — the native gate for family A (tile config lifecycle)
/// (`[ace-tile-instructions.DETECT.1]`, `[ace-tile-instructions.DETECT.1-1]`).
///
/// Gates on `CPUID.(EAX=07H,ECX=0):EDX[24]` (AMX-TILE) plus [`XCR0_TILE_STATE`].
#[cfg(target_arch = "x86_64")]
pub(crate) fn has_amx_tile() -> bool {
    use core::arch::x86_64::__cpuid_count;
    if !xcr0_tile_state_enabled() {
        return false;
    }
    // AMX-TILE = CPUID.(EAX=07H,ECX=0):EDX[24].
    (__cpuid_count(7, 0).edx >> 24) & 1 != 0
}

/// Returns `true` when the running CPU supports the AMX-AVX512 tile path OR ACE v1 — the
/// native gate for family C (tile-row converts) and the `TILEMOVROW` read form
/// (`[ace-tile-instructions.DETECT.1]`, `[ace-tile-instructions.DETECT.1-2]`).
///
/// Composes on [`avx10_base`] (AVX10 base + vector OS state) and requires the tile XSAVE
/// state, then the AMX-AVX512 feature bit `CPUID.(EAX=07H,ECX=1):EDX[21]` OR `ACE_VSN >= 1`.
/// The AMX-AVX512 bit position is Inferred (OQ-5).
#[cfg(target_arch = "x86_64")]
pub(crate) fn has_amx_avx512() -> bool {
    use core::arch::x86_64::__cpuid_count;
    if avx10_base().is_none() || !xcr0_tile_state_enabled() {
        return false;
    }
    let amx_avx512 = (__cpuid_count(7, 1).edx >> 21) & 1 != 0; // Inferred bit (OQ-5)
    amx_avx512 || ace_vsn() >= 1
}

/// Returns `true` when the running CPU supports the full ACE v1 capability — the native gate
/// for families D/E/F/G and the write-form tile moves
/// (`[ace-tile-instructions.DETECT.1]`, `[ace-tile-instructions.DETECT.1-3]`).
///
/// Full ACE v1 (spec section 3.2) = ACE `CPUID.(EAX=07H,ECX=1):ECX[11]` AND `ACE_VSN >= 1`
/// AND AMX-TILE, with the tile + BSR XSAVE state enabled (the last two via
/// [`has_amx_tile`]). The ACE feature-bit position and `ACE_VSN` leaf are Inferred (OQ-5).
#[cfg(target_arch = "x86_64")]
pub(crate) fn has_ace() -> bool {
    use core::arch::x86_64::__cpuid_count;
    if !has_amx_tile() {
        return false;
    }
    let ace_bit = (__cpuid_count(7, 1).ecx >> 11) & 1 != 0; // ACE Fn7/1 ECX[11] (Inferred, OQ-5)
    ace_bit && ace_vsn() >= 1
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
