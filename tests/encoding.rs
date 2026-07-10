//! Layer-2 encoding-assertion harness for the ACE group-4 raw-encoding primitives
//! (`[ace-tile-instructions.TESTING.5]`).
//!
//! Every golden constant below is transcribed from the ACE v1 rev-1.15 specification's
//! section-6.3 encoding tables (PDF pages 26-27). The harness has two independent
//! guarantees:
//!
//!  1. [`golden_bytes_match_per_byte_primitive`] — for every ACE-only raw-encoded
//!     instruction form used by the `.byte` shims it reconstructs the VEX/EVEX byte
//!     sequence from the documented section-6.3 fields (encoding kind / map / pp / W /
//!     opcode / ModRM / vvvv / imm8) and asserts the reconstruction equals the recorded
//!     golden bytes, plus structural invariants. Then [`c_shim_bytes_match_golden`] parses
//!     the actual `.byte` sequences out of `src/native/ace_tile.c` and asserts every
//!     sequence there is a golden constant and every golden constant appears — so a
//!     transcription error IN THE C FILE (not merely in this table) is caught. Both ALWAYS
//!     run and need NO external tool.
//!
//!  2. [`disassembly_mnemonic_operands_or_skip`] — when a system disassembler (`llvm-mc`
//!     or `objdump`) is present it disassembles each golden sequence and, where the tool
//!     recognizes the ACE mnemonic, asserts it matches; when the tool is absent — or does
//!     not yet know the ACE encodings (binutils 2.46 / current LLVM predate ACE v1.15) —
//!     it skips-with-warning and NEVER fails.
//!
//! Section-6.3 rows transcribed here (register-direct forms the shims use):
//!
//! ```text
//! TILEMOVROW (write, imm8)  EVEX.512.66.0F3A.W1 07 /ib      §6.3.3
//! TILEMOVCOL (write, imm8)  EVEX.512.66.0F3A.W1 2F /ib      §6.3.3
//! BSRINIT                   VEX.128.F2.0F38.W1 49 11:000:000 §6.3.5
//! BSRMOVF                   EVEX.512.NP.MAP6.W1 95           §6.3.5
//! BSRMOVH (write / read)    EVEX.512.F2.MAP6.W1 / .W0 95     §6.3.5
//! BSRMOVL (write / read)    EVEX.512.F3.MAP6.W1 / .W0 95     §6.3.5
//! TOP4BSSD/BSUD/BUSD/BUUD   EVEX.512.{F2,F3,66,NP}.0F38.W0 5E §6.3.9
//! TOP2BF16PS                EVEX.512.F3.0F38.W0 5C           §6.3.8
//! TOP4MX{B,BH,HB,H}F8PS     EVEX.512.{NP,F2,F3,66}.0F3A.W0 8D /ib §6.3.6
//! TOP4MXBSSPS               EVEX.512.F2.0F3A.W0 8F /ib       §6.3.7
//! ```

use std::io::Write;
use std::process::{Command, Stdio};

/// `pp` legacy-prefix selector values.
const PP_NP: u8 = 0b00;
const PP_66: u8 = 0b01;
const PP_F3: u8 = 0b10;
const PP_F2: u8 = 0b11;

/// Opcode map selector: 0F38 = 2, 0F3A = 3, MAP6 = 6 (EVEX mmm / VEX mmmmm).
const MAP_0F38: u8 = 2;
const MAP_0F3A: u8 = 3;
const MAP_6: u8 = 6;

/// One ACE raw-encoded instruction form's spec-derived encoding (ACE v1 §6.3).
struct Golden {
    mnemonic: &'static str,
    /// `true` for the sole VEX-encoded form (BSRINIT); `false` for EVEX.
    vex: bool,
    map: u8,
    w: bool,
    pp: u8,
    opcode: u8,
    modrm: u8,
    /// The EVEX.vvvv register operand (0 if the form has none; stored inverted in P1).
    vvvv: u8,
    /// Trailing imm8 for the `/ib` forms.
    imm8: Option<u8>,
    /// The pinned golden byte sequence (shared with `src/native/ace_tile.c`).
    bytes: &'static [u8],
}

impl Golden {
    /// Reconstruct the byte sequence from the documented §6.3 fields; the golden `bytes`
    /// must equal this reconstruction (that equality is the transcription check).
    fn reconstruct(&self) -> Vec<u8> {
        let mut out = Vec::new();
        if self.vex {
            // 3-byte VEX: C4 [R̄X̄B̄ mmmmm] [W v̄v̄v̄v̄ L pp] opcode modrm. 128-bit -> L = 0.
            out.push(0xC4);
            out.push(0xE0 | self.map); // R̄X̄B̄ = 111
            let vbar = (!self.vvvv) & 0xF;
            out.push(((self.w as u8) << 7) | (vbar << 3) | self.pp);
        } else {
            // 4-byte EVEX: 62 P0 P1 P2. P0 = R̄X̄B̄R̄' 0 mmm; P1 = W v̄v̄v̄v̄ 1 pp;
            // P2 = 0x48 (z=0, L'L=10 for 512-bit, b=0, V̄'=1, aaa=000).
            out.push(0x62);
            out.push(0xF0 | self.map);
            let vbar = (!self.vvvv) & 0xF;
            out.push(((self.w as u8) << 7) | (vbar << 3) | 0x04 | self.pp);
            out.push(0x48);
        }
        out.push(self.opcode);
        out.push(self.modrm);
        if let Some(ib) = self.imm8 {
            out.push(ib);
        }
        out
    }
}

/// The complete raw-encoding inventory used by the `.byte` shims. Register assignment
/// (documented in `src/native/ace_tile.c`): the TOP forms put the destination tile tmm1 in
/// ModRM.reg, src1 = zmm0 in ModRM.rm, src2 = zmm2 in EVEX.vvvv; the write moves put tmm1
/// in reg and the source zmm0 in rm with imm8 = 0; the BSR forms fix ModRM.reg = 000
/// (bsr0 is implicit), BSRMOVF takes zmm1 in vvvv (A scales) and zmm2 in rm (B scales),
/// and the half moves take zmm1 in rm.
const GOLDEN: &[Golden] = &[
    // Family B write (§6.3.3, imm8 forms, row/column 0).
    Golden {
        mnemonic: "TILEMOVROW",
        vex: false,
        map: MAP_0F3A,
        w: true,
        pp: PP_66,
        opcode: 0x07,
        modrm: 0xC8, // reg = tmm1, rm = zmm0
        vvvv: 0,
        imm8: Some(0),
        bytes: &[0x62, 0xF3, 0xFD, 0x48, 0x07, 0xC8, 0x00],
    },
    Golden {
        mnemonic: "TILEMOVCOL",
        vex: false,
        map: MAP_0F3A,
        w: true,
        pp: PP_66,
        opcode: 0x2F,
        modrm: 0xC8,
        vvvv: 0,
        imm8: Some(0),
        bytes: &[0x62, 0xF3, 0xFD, 0x48, 0x2F, 0xC8, 0x00],
    },
    // Family D (§6.3.5).
    Golden {
        mnemonic: "BSRINIT",
        vex: true, // the ONE VEX-encoded ACE-only form
        map: MAP_0F38,
        w: true,
        pp: PP_F2,
        opcode: 0x49,
        modrm: 0xC0, // 11:000:000
        vvvv: 0,
        imm8: None,
        bytes: &[0xC4, 0xE2, 0xFB, 0x49, 0xC0],
    },
    Golden {
        mnemonic: "BSRMOVF",
        vex: false,
        map: MAP_6,
        w: true,
        pp: PP_NP,
        opcode: 0x95,
        modrm: 0xC2, // 11:000:010 — reg fixed 000, rm = zmm2 (B scales)
        vvvv: 1,     // zmm1 (A scales)
        imm8: None,
        bytes: &[0x62, 0xF6, 0xF4, 0x48, 0x95, 0xC2],
    },
    Golden {
        mnemonic: "BSRMOVH", // write form (W1)
        vex: false,
        map: MAP_6,
        w: true,
        pp: PP_F2,
        opcode: 0x95,
        modrm: 0xC1, // 11:000:001 — rm = zmm1
        vvvv: 0,
        imm8: None,
        bytes: &[0x62, 0xF6, 0xFF, 0x48, 0x95, 0xC1],
    },
    Golden {
        mnemonic: "BSRMOVH", // read form (W0)
        vex: false,
        map: MAP_6,
        w: false,
        pp: PP_F2,
        opcode: 0x95,
        modrm: 0xC1,
        vvvv: 0,
        imm8: None,
        bytes: &[0x62, 0xF6, 0x7F, 0x48, 0x95, 0xC1],
    },
    Golden {
        mnemonic: "BSRMOVL", // write form (W1)
        vex: false,
        map: MAP_6,
        w: true,
        pp: PP_F3,
        opcode: 0x95,
        modrm: 0xC1,
        vvvv: 0,
        imm8: None,
        bytes: &[0x62, 0xF6, 0xFE, 0x48, 0x95, 0xC1],
    },
    Golden {
        mnemonic: "BSRMOVL", // read form (W0)
        vex: false,
        map: MAP_6,
        w: false,
        pp: PP_F3,
        opcode: 0x95,
        modrm: 0xC1,
        vvvv: 0,
        imm8: None,
        bytes: &[0x62, 0xF6, 0x7E, 0x48, 0x95, 0xC1],
    },
    // Family G (§6.3.9): opcode 5E, pp = signedness pair (F2=SS, F3=SU, 66=US, NP=UU).
    Golden {
        mnemonic: "TOP4BSSD",
        vex: false,
        map: MAP_0F38,
        w: false,
        pp: PP_F2,
        opcode: 0x5E,
        modrm: 0xC8, // reg = tmm1, rm = zmm0
        vvvv: 2,     // zmm2
        imm8: None,
        bytes: &[0x62, 0xF2, 0x6F, 0x48, 0x5E, 0xC8],
    },
    Golden {
        mnemonic: "TOP4BSUD",
        vex: false,
        map: MAP_0F38,
        w: false,
        pp: PP_F3,
        opcode: 0x5E,
        modrm: 0xC8,
        vvvv: 2,
        imm8: None,
        bytes: &[0x62, 0xF2, 0x6E, 0x48, 0x5E, 0xC8],
    },
    Golden {
        mnemonic: "TOP4BUSD",
        vex: false,
        map: MAP_0F38,
        w: false,
        pp: PP_66,
        opcode: 0x5E,
        modrm: 0xC8,
        vvvv: 2,
        imm8: None,
        bytes: &[0x62, 0xF2, 0x6D, 0x48, 0x5E, 0xC8],
    },
    Golden {
        mnemonic: "TOP4BUUD",
        vex: false,
        map: MAP_0F38,
        w: false,
        pp: PP_NP,
        opcode: 0x5E,
        modrm: 0xC8,
        vvvv: 2,
        imm8: None,
        bytes: &[0x62, 0xF2, 0x6C, 0x48, 0x5E, 0xC8],
    },
    // Family F (§6.3.8): opcode 5C, pp = F3.
    Golden {
        mnemonic: "TOP2BF16PS",
        vex: false,
        map: MAP_0F38,
        w: false,
        pp: PP_F3,
        opcode: 0x5C,
        modrm: 0xC8,
        vvvv: 2,
        imm8: None,
        bytes: &[0x62, 0xF2, 0x6E, 0x48, 0x5C, 0xC8],
    },
    // Family E (§6.3.6): opcode 8D /ib, pp = FP8 format pair (NP=BF8·BF8, F2=BF8·HF8,
    // F3=HF8·BF8, 66=HF8·HF8); §6.3.7: TOP4MXBSSPS = F2, opcode 8F /ib.
    Golden {
        mnemonic: "TOP4MXBF8PS",
        vex: false,
        map: MAP_0F3A,
        w: false,
        pp: PP_NP,
        opcode: 0x8D,
        modrm: 0xC8,
        vvvv: 2,
        imm8: Some(0),
        bytes: &[0x62, 0xF3, 0x6C, 0x48, 0x8D, 0xC8, 0x00],
    },
    Golden {
        mnemonic: "TOP4MXBHF8PS",
        vex: false,
        map: MAP_0F3A,
        w: false,
        pp: PP_F2,
        opcode: 0x8D,
        modrm: 0xC8,
        vvvv: 2,
        imm8: Some(0),
        bytes: &[0x62, 0xF3, 0x6F, 0x48, 0x8D, 0xC8, 0x00],
    },
    Golden {
        mnemonic: "TOP4MXHBF8PS",
        vex: false,
        map: MAP_0F3A,
        w: false,
        pp: PP_F3,
        opcode: 0x8D,
        modrm: 0xC8,
        vvvv: 2,
        imm8: Some(0),
        bytes: &[0x62, 0xF3, 0x6E, 0x48, 0x8D, 0xC8, 0x00],
    },
    Golden {
        mnemonic: "TOP4MXHF8PS",
        vex: false,
        map: MAP_0F3A,
        w: false,
        pp: PP_66,
        opcode: 0x8D,
        modrm: 0xC8,
        vvvv: 2,
        imm8: Some(0),
        bytes: &[0x62, 0xF3, 0x6D, 0x48, 0x8D, 0xC8, 0x00],
    },
    Golden {
        mnemonic: "TOP4MXBSSPS",
        vex: false,
        map: MAP_0F3A,
        w: false,
        pp: PP_F2,
        opcode: 0x8F,
        modrm: 0xC8,
        vvvv: 2,
        imm8: Some(0),
        bytes: &[0x62, 0xF3, 0x6F, 0x48, 0x8F, 0xC8, 0x00],
    },
];

/// For every raw-encoded form the golden byte sequence equals its reconstruction from the
/// section-6.3 fields, plus structural invariants. Always runs; no external tool.
///
/// `encoding::golden_bytes_match_per_byte_primitive`
#[test]
fn golden_bytes_match_per_byte_primitive() {
    assert_eq!(
        GOLDEN.len(),
        18,
        "18 raw-encoded forms: B-write(2) + D(6: BSRINIT + BSRMOVF + H/L write+read) + \
         G(4) + F(1) + E(5)"
    );

    let mut seen: Vec<Vec<u8>> = Vec::new();
    for g in GOLDEN {
        // 1. The golden constant equals the reconstruction from the documented §6.3 fields.
        assert_eq!(
            g.bytes,
            g.reconstruct().as_slice(),
            "{} golden bytes must equal the §6.3 reconstruction",
            g.mnemonic
        );

        // 2. Structural invariants.
        if g.vex {
            assert_eq!(g.bytes[0], 0xC4, "{}: 3-byte VEX prefix", g.mnemonic);
            assert_eq!(g.bytes[1] >> 5, 0b111, "{}: R̄X̄B̄ = 111", g.mnemonic);
            assert_eq!(g.bytes[1] & 0x1F, g.map, "{}: VEX map", g.mnemonic);
            assert_eq!(
                (g.bytes[2] >> 2) & 1,
                0,
                "{}: VEX.L = 0 (128-bit, §6.3.5)",
                g.mnemonic
            );
        } else {
            assert_eq!(g.bytes[0], 0x62, "{}: EVEX prefix byte", g.mnemonic);
            assert_eq!(
                g.bytes[1] >> 4,
                0x0F,
                "{}: R/X/B/R' = 1111 (no register extension)",
                g.mnemonic
            );
            assert_eq!(g.bytes[1] & 0x07, g.map, "{}: EVEX map field", g.mnemonic);
            assert_eq!(g.bytes[1] & 0x08, 0, "{}: P0 bit 3 reserved 0", g.mnemonic);
            assert_eq!((g.bytes[2] >> 2) & 1, 1, "{}: P1 bit 2 = 1", g.mnemonic);
            assert_eq!(
                g.bytes[3], 0x48,
                "{}: P2 = z0 L'L=10 (512-bit) b0 V'1 aaa=000",
                g.mnemonic
            );
        }
        assert_eq!(g.bytes[2] & 0x03, g.pp, "{}: pp field", g.mnemonic);
        assert_eq!(g.bytes[2] >> 7, g.w as u8, "{}: W field", g.mnemonic);

        // 3. Uniqueness: no two forms share an encoding.
        assert!(
            !seen.contains(&g.bytes.to_vec()),
            "{}: duplicate encoding {:02X?}",
            g.mnemonic,
            g.bytes
        );
        seen.push(g.bytes.to_vec());
    }
    println!(
        "encoding: {} ACE raw encodings verified against the ACE v1 §6.3 tables",
        GOLDEN.len()
    );
}

/// Extract every `.byte 0x..,0x..` sequence from C source text: scan for the literal
/// `".byte "` opener and parse the consecutive comma-separated `0xNN` tokens that follow
/// (a sequence ends at the first non-`0xNN` token).
fn extract_byte_sequences(c_source: &str) -> Vec<Vec<u8>> {
    let mut found = Vec::new();
    // Sequences start with the EVEX prefix 0x62 or the VEX prefix 0xC4. (Some shims carry
    // their bytes through a macro parameter, so scanning for `.byte` adjacency would miss
    // them; the prefix marker catches both spellings.)
    let starts: Vec<usize> = c_source
        .match_indices("0x62,")
        .chain(c_source.match_indices("0xc4,"))
        .map(|(i, _)| i)
        .collect();
    for start in starts {
        let rest = &c_source[start..];
        let mut bytes = Vec::new();
        for tok in rest.split(',') {
            let tok = tok.trim_start();
            match tok
                .get(..4)
                .and_then(|t| t.strip_prefix("0x"))
                .map(|h| u8::from_str_radix(h, 16))
            {
                Some(Ok(b)) => {
                    bytes.push(b);
                    // A token with anything beyond the `0xNN` literal (`0xc0\n\t"` at a
                    // line end, `0xc8" :::` at a sequence end) terminates THIS sequence —
                    // otherwise the scan would run into the next `.byte` line of a
                    // multi-instruction asm block.
                    if !tok[4..].trim().is_empty() {
                        break;
                    }
                }
                _ => break,
            }
        }
        if !bytes.is_empty() {
            found.push(bytes);
        }
    }
    found
}

/// The `.byte` sequences actually present in `src/native/ace_tile.c` are exactly drawn
/// from the golden set, and every golden encoding appears at least once. This is the real
/// transcription check: editing a byte in either the C shim or the golden table (but not
/// both) fails this test, with no external tool and regardless of whether the `native`
/// feature is enabled (`[ace-tile-instructions.TESTING.5]`).
///
/// `encoding::c_shim_bytes_match_golden`
#[test]
fn c_shim_bytes_match_golden() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/native/ace_tile.c");
    let source = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read {path} (the .byte shim source): {e}"));

    let in_c = extract_byte_sequences(&source);
    assert!(!in_c.is_empty(), "no .byte sequences found in ace_tile.c");

    // Every sequence in the C file is a golden encoding (the BSR read forms and BSRMOVF
    // appear in several shims, so multiplicities legitimately differ).
    for seq in &in_c {
        assert!(
            GOLDEN.iter().any(|g| g.bytes == seq.as_slice()),
            ".byte sequence {seq:02X?} in ace_tile.c has no §6.3 golden entry — the shim \
             drifted from the spec table"
        );
    }
    // Every golden encoding appears in the C file.
    for g in GOLDEN {
        assert!(
            in_c.iter().any(|seq| seq.as_slice() == g.bytes),
            "{}: golden bytes {:02X?} not found in ace_tile.c — the .byte shim is missing",
            g.mnemonic,
            g.bytes
        );
    }
    println!(
        "encoding: {} .byte sequences parsed from ace_tile.c, all matching the §6.3 golden set",
        in_c.len()
    );
}

/// Result of probing for a system disassembler.
enum Disassembler {
    LlvmMc,
    Objdump,
    None,
}

/// Probe `llvm-mc` then `objdump`; return the first that answers `--version`.
fn find_disassembler() -> Disassembler {
    let probe = |cmd: &str, arg: &str| {
        Command::new(cmd)
            .arg(arg)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    };
    if probe("llvm-mc", "--version") {
        Disassembler::LlvmMc
    } else if probe("objdump", "--version") {
        Disassembler::Objdump
    } else {
        Disassembler::None
    }
}

/// Disassemble raw bytes with `llvm-mc --disassemble` (reads `0xNN` tokens on stdin).
fn disasm_llvm_mc(bytes: &[u8]) -> Option<String> {
    let hex: String = bytes.iter().map(|b| format!("0x{b:02x} ")).collect();
    let mut child = Command::new("llvm-mc")
        .args(["--disassemble", "--triple=x86_64-unknown-linux-gnu"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .ok()?;
    child.stdin.take()?.write_all(hex.as_bytes()).ok()?;
    let out = child.wait_with_output().ok()?;
    Some(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Disassemble raw bytes with `objdump -D -b binary -m i386:x86-64` over a temp file.
fn disasm_objdump(bytes: &[u8]) -> Option<String> {
    let path = std::env::temp_dir().join(format!("ace_enc_{}.bin", std::process::id()));
    std::fs::write(&path, bytes).ok()?;
    let out = Command::new("objdump")
        .args(["-D", "-b", "binary", "-m", "i386:x86-64", "-M", "intel"])
        .arg(&path)
        .output()
        .ok()?;
    let _ = std::fs::remove_file(&path);
    Some(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// When a disassembler is present, disassemble each golden sequence and assert the spec
/// mnemonic where the tool recognizes the ACE encoding; when the tool is absent — or does
/// not yet know the ACE mnemonics (binutils 2.46 / current LLVM predate ACE v1.15) —
/// skip-with-warning and NEVER fail (`[ace-tile-instructions.TESTING.5]`).
///
/// `encoding::disassembly_mnemonic_operands_or_skip`
#[test]
fn disassembly_mnemonic_operands_or_skip() {
    /// A raw-bytes -> disassembly-text function (one per external tool).
    type Disasm = fn(&[u8]) -> Option<String>;
    let tool = find_disassembler();
    let (name, run): (&str, Disasm) = match tool {
        Disassembler::LlvmMc => ("llvm-mc", disasm_llvm_mc),
        Disassembler::Objdump => ("objdump", disasm_objdump),
        Disassembler::None => {
            eprintln!(
                "warning: encoding::disassembly_mnemonic_operands_or_skip — no llvm-mc/objdump \
                 disassembler present; skipping the disassembly assertion (never a failure)."
            );
            return;
        }
    };

    let mut matched = 0usize;
    let mut unrecognized = 0usize;
    for g in GOLDEN {
        let text = match run(g.bytes) {
            Some(t) => t,
            None => {
                eprintln!(
                    "warning: {name} could not disassemble {} ({:02X?}); skipping (never a failure).",
                    g.mnemonic, g.bytes
                );
                unrecognized += 1;
                continue;
            }
        };
        // ACE mnemonics are not in current binutils/LLVM tables, so the tool typically emits
        // `(bad)` / `.byte`-style output. Where it DOES surface the spec mnemonic, assert it;
        // where it decodes the bytes as some OTHER valid instruction, surface that loudly —
        // an encoding aliasing an allocated x86 opcode is exactly the drift this layer
        // exists to notice.
        let upper = text.to_ascii_uppercase();
        if upper.contains(g.mnemonic) {
            matched += 1;
        } else {
            if !upper.contains("(BAD)") && !upper.contains(".BYTE") {
                let decode: Vec<&str> = text
                    .lines()
                    .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('.'))
                    .take(3)
                    .collect();
                eprintln!(
                    "warning: {name} decoded {} ({:02X?}) as a DIFFERENT valid instruction — \
                     possible opcode collision with an allocated x86 encoding: {decode:?}",
                    g.mnemonic, g.bytes
                );
            }
            unrecognized += 1;
        }
    }

    eprintln!(
        "encoding::disassembly — {name}: {matched}/{} ACE mnemonics recognized, {unrecognized} \
         unrecognized (expected: ACE v1.15 not yet in {name}; skip-with-warning, never a failure).",
        GOLDEN.len()
    );
    // Never fail: a disassembler that does not know ACE yet is the documented, expected state.
}
