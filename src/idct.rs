//! H.263 inverse discrete cosine transform (§6.2.4) and intra-block
//! sample reconstruction (§6.3.2 clip).
//!
//! §6.2.4 defines the inverse transform by its mathematical equation
//! and leaves "the arithmetic procedures for computing the inverse
//! transform" to the implementer, within the Annex A accuracy budget.
//! Decoders therefore differ by ±1 on some samples, and the difference
//! compounds along a P-picture chain. This crate computes FFmpeg's
//! integer "simple IDCT" ([`crate::ffmpeg_idct`], its C path), which
//! meets Annex A, so its pictures equal FFmpeg's (`-idct simple`) sample
//! for sample.
//!
//! ## Output range and clipping
//!
//! §6.2.4 says "The output from the inverse transform ranges from
//! –256 to +255 after clipping to be represented with 9 bits." We
//! apply this clip on the IDCT's integer output before returning. It
//! never changes a reconstructed sample: the §6.3.2 clip to `[0, 255]`
//! follows, after adding a prediction in `[0, 255]` for INTER blocks.
//!
//! ## Intra-block sample reconstruction
//!
//! §6.3.1: for INTRA blocks "the reconstruction is equal to the
//! result of the inverse transformation". §6.3.2 then clips to
//! `[0, 255]` for display. We expose [`reconstruct_intra_samples`]
//! that runs the IDCT on an 8×8 dequantised coefficient block
//! (post-zigzag scatter) and emits an 8×8 `u8` sample block ready for
//! the picture buffer.

use crate::block::COEFFS_PER_BLOCK;

/// §6.2.4 IDCT output range (9-bit signed).
pub const IDCT_OUT_MIN: i16 = -256;
/// §6.2.4 IDCT output range (9-bit signed).
pub const IDCT_OUT_MAX: i16 = 255;

/// Side length of the H.263 transform block.
pub const BLOCK_DIM: usize = 8;

/// §6.2.4 inverse DCT.
///
/// Takes an 8×8 dequantised coefficient block in row-major order
/// (`coefs[row * 8 + col] == F(u=col, v=row)`) and returns the 8×8
/// inverse-transformed sample block, also row-major, clipped to
/// `[-256, +255]` per §6.2.4.
///
/// **Convention.** This crate uses the convention `(u, v) = (col, row)`
/// throughout — i.e. block position `(row, col)` in storage carries
/// the frequency-domain coefficient `F(col, row)`, FFmpeg's layout:
/// the integer transform rounds its row and column passes differently,
/// so the orientation matters.
pub fn idct_8x8(coefs: &[i16; COEFFS_PER_BLOCK]) -> [i16; COEFFS_PER_BLOCK] {
    let pixels = crate::ffmpeg_idct::simple_idct(coefs);
    let mut out = [0i16; COEFFS_PER_BLOCK];
    for (slot, &v) in out.iter_mut().zip(&pixels) {
        *slot = v.clamp(IDCT_OUT_MIN as i32, IDCT_OUT_MAX as i32) as i16;
    }
    out
}

/// §6.3.1 + §6.3.2 reconstruction for INTRA blocks.
///
/// Takes a dequantised 8×8 block (post-zigzag scatter), runs the
/// IDCT, and clips to `[0, 255]` per §6.3.2. INTRA blocks have no
/// motion-compensation prediction, so the IDCT output **is** the
/// reconstructed sample (modulo the §6.3.2 clip). The §6.2.4 nominal
/// output range is `[-256, +255]`; the §6.3.2 clip narrows that to
/// the 8-bit picture range `[0, 255]`.
pub fn reconstruct_intra_samples(coefs: &[i16; COEFFS_PER_BLOCK]) -> [u8; COEFFS_PER_BLOCK] {
    let idct = idct_8x8(coefs);
    let mut out = [0u8; COEFFS_PER_BLOCK];
    for (i, &v) in idct.iter().enumerate() {
        out[i] = v.clamp(0, 255) as u8;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// §A.8: "All zeros in shall produce all zeros out."
    #[test]
    fn all_zero_input_yields_all_zero_output() {
        let zero = [0i16; COEFFS_PER_BLOCK];
        let out = idct_8x8(&zero);
        assert!(out.iter().all(|&p| p == 0));
    }

    /// DC-only input: F(0, 0) = K, rest zero.
    ///
    /// f(x, y) = (1/4) · C(0) · C(0) · K · cos(0) · cos(0)
    ///        = (1/4) · (1/√2) · (1/√2) · K
    ///        = K / 8
    ///
    /// For K = 8 → every pixel is 1. For K = 800 → every pixel is 100.
    #[test]
    fn dc_only_block_gives_uniform_field() {
        for &dc in &[8i16, 16, 80, 800, 8 * 100] {
            let mut coefs = [0i16; COEFFS_PER_BLOCK];
            coefs[0] = dc;
            let out = idct_8x8(&coefs);
            let expected = dc / 8;
            for (i, &p) in out.iter().enumerate() {
                assert_eq!(
                    p, expected,
                    "dc={}, position {}: expected {}, got {}",
                    dc, i, expected, p
                );
            }
        }
    }

    /// DC-only with negative coefficient.
    #[test]
    fn dc_only_negative() {
        let mut coefs = [0i16; COEFFS_PER_BLOCK];
        coefs[0] = -16;
        let out = idct_8x8(&coefs);
        assert!(out.iter().all(|&p| p == -2));
    }

    /// Sample reconstruction: an INTRA DC = 800 reconstructs to a
    /// uniform field of value 100 (§6.3.2 clip is a no-op since 100
    /// is already in `[0, 255]`).
    #[test]
    fn intra_dc_only_reconstruct() {
        let mut coefs = [0i16; COEFFS_PER_BLOCK];
        coefs[0] = 800;
        let samples = reconstruct_intra_samples(&coefs);
        assert!(samples.iter().all(|&p| p == 100));
    }

    /// §6.3.2 clip: an INTRA DC = 2048 (overshoots 255) saturates to
    /// the IDCT's own -256/+255 range first (giving 255 after rounding
    /// from 256.0 → 256, then clipped to 255), then §6.3.2 keeps 255.
    #[test]
    fn intra_dc_clipped_at_top() {
        let mut coefs = [0i16; COEFFS_PER_BLOCK];
        // DC = 2048 → pixel = 2048/8 = 256 → §6.2.4 clip → 255 → §6.3.2 no-op.
        coefs[0] = 2048;
        let samples = reconstruct_intra_samples(&coefs);
        assert!(samples.iter().all(|&p| p == 255));
    }

    /// §6.3.2 clip: a strongly negative DC saturates to 0.
    #[test]
    fn intra_dc_clipped_at_zero() {
        let mut coefs = [0i16; COEFFS_PER_BLOCK];
        coefs[0] = -1000;
        let samples = reconstruct_intra_samples(&coefs);
        assert!(samples.iter().all(|&p| p == 0));
    }

    /// §A.7 peak-error budget self-check: feed a known cosine-pattern
    /// block (forward DCT of a simple ramp) and verify the IDCT
    /// reproduces the ramp within ±1.
    ///
    /// We construct the forward DCT of a 1-D ramp `f(x) = x` along
    /// rows (zero along columns) by inverting our IDCT against a
    /// well-known orthonormal-DCT pair. Rather than encode the
    /// forward DCT separately, we use a single-AC-coefficient test:
    /// F(u=1, v=0) = K gives f(x, y) = (K/4) · cos(π(2x+1)/16) when
    /// y has no v-dependence (and is independent of y).
    #[test]
    fn single_ac_coefficient_horizontal_basis() {
        // F(u=1, v=0) = 64. Expected:
        // f(x, y) = (1/4) · 1 · (1/√2) · 64 · cos(π(2x+1)/16) · cos(0)
        //        = 64 · (1/(4·√2)) · cos(π(2x+1)/16)
        //        ≈ 11.31 · cos(π(2x+1)/16)
        // Independent of y.
        let mut coefs = [0i16; COEFFS_PER_BLOCK];
        coefs[1] = 64; // F(u=1, v=0)
        let out = idct_8x8(&coefs);

        // Compute expected values in f64 and compare with peak error ≤ 1.
        for y in 0..BLOCK_DIM {
            for x in 0..BLOCK_DIM {
                let cos_term = (core::f64::consts::PI * ((2 * x + 1) as f64) / 16.0).cos();
                let expected = 64.0 * (1.0 / (4.0 * core::f64::consts::SQRT_2)) * cos_term;
                let got = out[y * BLOCK_DIM + x] as f64;
                assert!(
                    (got - expected).abs() <= 1.0,
                    "(x={}, y={}): expected ≈ {}, got {}",
                    x,
                    y,
                    expected,
                    got
                );
            }
        }
    }

    /// Symmetry self-check: an IDCT of a block with only F(0,0) and
    /// F(7,7) populated produces a result symmetric across the
    /// diagonal (because the kernel cos terms factorise).
    #[test]
    fn block_symmetry_diagonal_basis() {
        let mut coefs = [0i16; COEFFS_PER_BLOCK];
        coefs[0] = 64; // DC
        coefs[7 * BLOCK_DIM + 7] = 64; // F(u=7, v=7)
        let out = idct_8x8(&coefs);
        // Symmetric under (x, y) → (y, x).
        for y in 0..BLOCK_DIM {
            for x in 0..BLOCK_DIM {
                assert_eq!(
                    out[y * BLOCK_DIM + x],
                    out[x * BLOCK_DIM + y],
                    "asymmetric at (x={}, y={})",
                    x,
                    y
                );
            }
        }
    }

    /// IDCT of a block with all values in [-2048, 2047] never overflows
    /// the i32 accumulator — sanity smoke test for a max-magnitude
    /// input.
    #[test]
    fn no_overflow_on_extreme_input() {
        let mut coefs = [0i16; COEFFS_PER_BLOCK];
        for v in coefs.iter_mut() {
            *v = 2047;
        }
        let out = idct_8x8(&coefs);
        // The exact pixel values aren't the point — we just need this
        // to terminate without panicking.
        assert_eq!(out.len(), COEFFS_PER_BLOCK);
    }
}
