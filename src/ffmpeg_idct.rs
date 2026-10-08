// FFmpeg's integer inverse DCT ("simple IDCT"), 8-bit instantiation.
//
// Ported from FFmpeg libavcodec/simple_idct_template.c (commit 2da55bf),
// Copyright (c) 2001 Michael Niedermayer. This file is licensed under the
// GNU Lesser General Public License version 2.1 or later
// (LGPL-2.1-or-later), as the file it is ported from; the rest of this
// crate is MIT.

//! `ff_simple_idct_put_int16_8bit` / `ff_simple_idct_add_int16_8bit`
//! before their final store: FFmpeg's C IDCT, which `-idct simple` pins
//! (its default on arm64 is NEON assembly that rounds differently). Each
//! output is the `(int)(a ± b) >> COL_SHIFT` the C clips to a pixel
//! (put) or adds to the prediction and clips (add).
//!
//! The C does its sums in `unsigned` (wrapping) arithmetic and narrows the
//! row pass to `int16_t`; this port does the same, so any input gives the
//! C's result.

const W1: u32 = 22725;
const W2: u32 = 21407;
const W3: u32 = 19266;
const W4: u32 = 16383;
const W5: u32 = 12873;
const W6: u32 = 8867;
const W7: u32 = 4520;
const ROW_SHIFT: u32 = 11;
const COL_SHIFT: u32 = 20;
const DC_SHIFT: u32 = 3;

/// `(SUINT)w * x`: the C's unsigned multiply of a weight and a promoted
/// `int16_t`.
#[inline]
fn mul(w: u32, x: i16) -> u32 {
    w.wrapping_mul(x as i32 as u32)
}

/// `idctRowCondDC` (8-bit, `HAVE_FAST_64BIT`, `extra_shift` 0), in place.
fn idct_row_cond_dc(row: &mut [i16]) {
    if row[1..8].iter().all(|&c| c == 0) {
        // `(row[0] * (1 << DC_SHIFT)) & 0xffff`, replicated.
        let v = (i32::from(row[0]) << DC_SHIFT) as i16;
        row[..8].fill(v);
        return;
    }
    let mut a0 = mul(W4, row[0]).wrapping_add(1 << (ROW_SHIFT - 1));
    let mut a1 = a0;
    let mut a2 = a0;
    let mut a3 = a0;
    a0 = a0.wrapping_add(mul(W2, row[2]));
    a1 = a1.wrapping_add(mul(W6, row[2]));
    a2 = a2.wrapping_sub(mul(W6, row[2]));
    a3 = a3.wrapping_sub(mul(W2, row[2]));

    let mut b0 = mul(W1, row[1]).wrapping_add(mul(W3, row[3]));
    let mut b1 = mul(W3, row[1]).wrapping_sub(mul(W7, row[3]));
    let mut b2 = mul(W5, row[1]).wrapping_sub(mul(W1, row[3]));
    let mut b3 = mul(W7, row[1]).wrapping_sub(mul(W5, row[3]));

    if row[4..8].iter().any(|&c| c != 0) {
        a0 = a0.wrapping_add(mul(W4, row[4]).wrapping_add(mul(W6, row[6])));
        a1 = a1.wrapping_add(mul(W4, row[4]).wrapping_neg().wrapping_sub(mul(W2, row[6])));
        a2 = a2.wrapping_add(mul(W4, row[4]).wrapping_neg().wrapping_add(mul(W2, row[6])));
        a3 = a3.wrapping_add(mul(W4, row[4]).wrapping_sub(mul(W6, row[6])));

        b0 = b0.wrapping_add(mul(W5, row[5])).wrapping_add(mul(W7, row[7]));
        b1 = b1.wrapping_sub(mul(W1, row[5])).wrapping_sub(mul(W5, row[7]));
        b2 = b2.wrapping_add(mul(W7, row[5])).wrapping_add(mul(W3, row[7]));
        b3 = b3.wrapping_add(mul(W3, row[5])).wrapping_sub(mul(W1, row[7]));
    }

    // `(int)(a ± b) >> ROW_SHIFT`, stored to `int16_t`.
    let out = |v: u32| ((v as i32) >> ROW_SHIFT) as i16;
    row[0] = out(a0.wrapping_add(b0));
    row[7] = out(a0.wrapping_sub(b0));
    row[1] = out(a1.wrapping_add(b1));
    row[6] = out(a1.wrapping_sub(b1));
    row[2] = out(a2.wrapping_add(b2));
    row[5] = out(a2.wrapping_sub(b2));
    row[3] = out(a3.wrapping_add(b3));
    row[4] = out(a3.wrapping_sub(b3));
}

/// `IDCT_COLS` and the final `(int)(a ± b) >> COL_SHIFT` of column `c`,
/// written to `out[8 * r + c]`.
fn idct_col(block: &[i16; 64], c: usize, out: &mut [i32; 64]) {
    let col = |n: usize| block[8 * n + c];
    let mut a0 = W4.wrapping_mul((i32::from(col(0)) + ((1 << (COL_SHIFT - 1)) / W4 as i32)) as u32);
    let mut a1 = a0;
    let mut a2 = a0;
    let mut a3 = a0;
    a0 = a0.wrapping_add(mul(W2, col(2)));
    a1 = a1.wrapping_add(mul(W6, col(2)));
    a2 = a2.wrapping_sub(mul(W6, col(2)));
    a3 = a3.wrapping_sub(mul(W2, col(2)));

    let mut b0 = mul(W1, col(1)).wrapping_add(mul(W3, col(3)));
    let mut b1 = mul(W3, col(1)).wrapping_sub(mul(W7, col(3)));
    let mut b2 = mul(W5, col(1)).wrapping_sub(mul(W1, col(3)));
    let mut b3 = mul(W7, col(1)).wrapping_sub(mul(W5, col(3)));

    if col(4) != 0 {
        a0 = a0.wrapping_add(mul(W4, col(4)));
        a1 = a1.wrapping_sub(mul(W4, col(4)));
        a2 = a2.wrapping_sub(mul(W4, col(4)));
        a3 = a3.wrapping_add(mul(W4, col(4)));
    }
    if col(5) != 0 {
        b0 = b0.wrapping_add(mul(W5, col(5)));
        b1 = b1.wrapping_sub(mul(W1, col(5)));
        b2 = b2.wrapping_add(mul(W7, col(5)));
        b3 = b3.wrapping_add(mul(W3, col(5)));
    }
    if col(6) != 0 {
        a0 = a0.wrapping_add(mul(W6, col(6)));
        a1 = a1.wrapping_sub(mul(W2, col(6)));
        a2 = a2.wrapping_add(mul(W2, col(6)));
        a3 = a3.wrapping_sub(mul(W6, col(6)));
    }
    if col(7) != 0 {
        b0 = b0.wrapping_add(mul(W7, col(7)));
        b1 = b1.wrapping_sub(mul(W5, col(7)));
        b2 = b2.wrapping_add(mul(W3, col(7)));
        b3 = b3.wrapping_sub(mul(W1, col(7)));
    }

    let px = |v: u32| (v as i32) >> COL_SHIFT;
    out[c] = px(a0.wrapping_add(b0));
    out[8 + c] = px(a1.wrapping_add(b1));
    out[16 + c] = px(a2.wrapping_add(b2));
    out[24 + c] = px(a3.wrapping_add(b3));
    out[32 + c] = px(a3.wrapping_sub(b3));
    out[40 + c] = px(a2.wrapping_sub(b2));
    out[48 + c] = px(a1.wrapping_sub(b1));
    out[56 + c] = px(a0.wrapping_sub(b0));
}

/// The simple IDCT of `block` (row-major: `block[8 * v + u]` holds the
/// coefficient of vertical frequency `v`, horizontal `u`, FFmpeg's
/// unpermuted layout), row-major, before the put clip or the add.
pub fn simple_idct(block: &[i16; 64]) -> [i32; 64] {
    let mut rows = *block;
    for r in 0..8 {
        idct_row_cond_dc(&mut rows[8 * r..8 * r + 8]);
    }
    let mut out = [0i32; 64];
    for c in 0..8 {
        idct_col(&rows, c, &mut out);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A row whose AC is zero takes the DC shortcut, which wraps
    /// `row[0] * 8` to 16 bits as the C's `& 0xffff` does: 4096 * 8
    /// becomes -32768, so every output is that DC's IDCT
    /// (-32768 * 16383 rounded, >> 20 = -512).
    #[test]
    fn the_dc_shortcut_wraps_to_16_bits() {
        let mut block = [0i16; 64];
        block[0] = 4096;
        assert!(simple_idct(&block).iter().all(|&v| v == -512));
    }
}
