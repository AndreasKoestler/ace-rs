//! Int8 image filtering built on `ace::dpbuud` and `ace::dpbsud` — AVX-VNNI-INT8
//! outside the usual ML context.
//!
//! A 4-tap horizontal filter is exactly one VNNI lane: each lane of `VPDPBUUD` /
//! `VPDPBSUD` sums 4 adjacent byte products, so a `[1,1,1,1]` box blur and a
//! `[-1,-1,+1,+1]` edge detector both map one output pixel onto one lane. This
//! example generates a small grayscale image (gradient + bright rectangle),
//! blurs it with `dpbuud` (u8 pixels x u8 kernel), finds vertical edges with
//! `dpbsud` (i8 kernel x u8 pixels — the signed operand is `a`, first), checks
//! every call bit-for-bit against its scalar oracle AND a plain nested-loop
//! reference filter, and renders before/after as ASCII art.
//!
//! Run it on any host — the crate falls back to the portable scalar path when
//! `avxvnniint8` is absent:
//!
//! ```text
//! cargo run --example int8_image_filter
//! ```
//!
//! To exercise the native `VPDPBUUD`/`VPDPBSUD` instructions on a machine that
//! lacks them, run the built example under Intel SDE with a future-CPU model:
//!
//! ```text
//! cargo build --example int8_image_filter
//! ~/sde/sde64 -future -- target/debug/examples/int8_image_filter
//! ```

use ace::{dpbsud, dpbsud_scalar, dpbuud, dpbuud_scalar};

/// Image dimensions. W = 32 means each row filters in exactly 4 dpbuud/dpbsud
/// calls (8 lanes per call, one output pixel per lane).
const W: usize = 32;
const H: usize = 32;
/// Filter taps per output pixel — matching the 4 byte products per VNNI lane.
const TAPS: usize = 4;

/// Procedural grayscale test image: a horizontal gradient background with a
/// bright rectangle in the middle. No external crates — `ace` is zero-dependency
/// and its examples stay that way.
fn make_image() -> Vec<u8> {
    let mut img = vec![0u8; W * H];
    for y in 0..H {
        for x in 0..W {
            let in_rect = (10..22).contains(&x) && (10..22).contains(&y);
            img[y * W + x] = if in_rect { 230 } else { (x * 4) as u8 };
        }
    }
    img
}

/// Row `y` padded by replicating its last pixel TAPS-1 times, so every output
/// column 0..W has a full 4-pixel window and filtered images keep the 32-pixel
/// width (windows are anchored left: output x reads pixels x..x+4).
fn padded_row(img: &[u8], y: usize) -> [u8; W + TAPS - 1] {
    let mut row = [img[y * W + W - 1]; W + TAPS - 1];
    row[..W].copy_from_slice(&img[y * W..y * W + W]);
    row
}

/// Horizontal 4-tap box blur of one row via `dpbuud`: pixels in `a`, the all-ones
/// kernel in `b` (both u8). Lane i of each call holds the sum of one 4-pixel
/// window; divide by 4 for the blurred pixel. Every call is checked against the
/// scalar oracle.
fn blur_row(row: &[u8; W + TAPS - 1]) -> [u8; W] {
    let ones = [1u8; 32];
    let mut out = [0u8; W];
    for group in 0..W / 8 {
        // Lane i's 4 products are a[4i..4i+4]·b[4i..4i+4]: stage the 8 overlapping
        // windows (stride 1) of this group into consecutive 4-byte lane slots.
        let mut a = [0u8; 32];
        for lane in 0..8 {
            let x = group * 8 + lane;
            a[TAPS * lane..TAPS * (lane + 1)].copy_from_slice(&row[x..x + TAPS]);
        }
        let lanes = dpbuud([0i32; 8], a, ones);
        assert_eq!(
            lanes,
            dpbuud_scalar([0i32; 8], a, ones),
            "dpbuud dispatcher diverged from its scalar oracle"
        );
        for (lane, &sum) in lanes.iter().enumerate() {
            out[group * 8 + lane] = (sum / TAPS as i32) as u8;
        }
    }
    out
}

/// Horizontal edge detect of one row via `dpbsud`: the signed [-1,-1,+1,+1]
/// kernel tiled across `a` (i8 — dpbsud's FIRST operand is the signed one), the
/// pixel windows in `b` (u8). Lane i is the signed edge response; its absolute
/// value, clamped to 255, is the output intensity.
fn edge_row(row: &[u8; W + TAPS - 1]) -> [u8; W] {
    let mut kernel = [0i8; 32];
    for lane in 0..8 {
        kernel[TAPS * lane..TAPS * (lane + 1)].copy_from_slice(&[-1, -1, 1, 1]);
    }
    let mut out = [0u8; W];
    for group in 0..W / 8 {
        let mut b = [0u8; 32];
        for lane in 0..8 {
            let x = group * 8 + lane;
            b[TAPS * lane..TAPS * (lane + 1)].copy_from_slice(&row[x..x + TAPS]);
        }
        let lanes = dpbsud([0i32; 8], kernel, b);
        assert_eq!(
            lanes,
            dpbsud_scalar([0i32; 8], kernel, b),
            "dpbsud dispatcher diverged from its scalar oracle"
        );
        for (lane, &response) in lanes.iter().enumerate() {
            out[group * 8 + lane] = response.unsigned_abs().min(255) as u8;
        }
    }
    out
}

/// Plain nested-loop reference for both filters — the "obvious" implementation
/// the VNNI path must reproduce exactly. `kernel` taps are i32 so one function
/// covers the unsigned blur and the signed edge detector.
fn reference_filter(img: &[u8], kernel: [i32; TAPS], divide: i32) -> Vec<u8> {
    let mut out = vec![0u8; W * H];
    for y in 0..H {
        let row = padded_row(img, y);
        for x in 0..W {
            let mut acc = 0i32;
            for (k, tap) in kernel.iter().enumerate() {
                acc += tap * row[x + k] as i32;
            }
            out[y * W + x] = (acc.abs() / divide).min(255) as u8;
        }
    }
    out
}

/// Render an image as ASCII art, mapping intensity onto a 10-step ramp. Each
/// pixel prints twice so terminal cell aspect ratio keeps the image square-ish.
fn render(title: &str, img: &[u8]) {
    const RAMP: &[u8] = b" .:-=+*#%@";
    println!("\n{title}");
    for y in 0..H {
        let mut line = String::with_capacity(2 * W);
        for x in 0..W {
            let c = RAMP[img[y * W + x] as usize * RAMP.len() / 256] as char;
            line.push(c);
            line.push(c);
        }
        println!("  {line}");
    }
}

fn main() {
    let img = make_image();

    // VNNI path: filter every row through dpbuud (blur) and dpbsud (edges).
    let mut blurred = vec![0u8; W * H];
    let mut edges = vec![0u8; W * H];
    for y in 0..H {
        let row = padded_row(&img, y);
        blurred[y * W..(y + 1) * W].copy_from_slice(&blur_row(&row));
        edges[y * W..(y + 1) * W].copy_from_slice(&edge_row(&row));
    }

    // The VNNI results must match the nested-loop reference filters exactly —
    // integer arithmetic, so bit-for-bit, not approximately.
    assert_eq!(
        blurred,
        reference_filter(&img, [1, 1, 1, 1], TAPS as i32),
        "dpbuud blur diverged from the nested-loop reference"
    );
    assert_eq!(
        edges,
        reference_filter(&img, [-1, -1, 1, 1], 1),
        "dpbsud edge detect diverged from the nested-loop reference"
    );

    render("Original (gradient + bright rectangle):", &img);
    render("Box blur [1,1,1,1]/4 via dpbuud (edges soften):", &blurred);
    render(
        "Edge detect [-1,-1,+1,+1] via dpbsud (vertical borders light up):",
        &edges,
    );

    #[cfg(target_arch = "x86_64")]
    let native = if std::is_x86_feature_detected!("avxvnniint8") {
        "yes"
    } else {
        "no (scalar fallback)"
    };
    #[cfg(not(target_arch = "x86_64"))]
    let native = "n/a (not x86_64)";

    println!("\navxvnniint8 native path: {native}");
    println!("PASS");
}
