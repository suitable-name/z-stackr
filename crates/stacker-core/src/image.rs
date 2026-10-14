//! [`PlanarImage`], the three-plane image every stage of the pipeline passes around.

use num_traits::Float;

use crate::error::StackerError;

/// Three equally sized planes of one image.
///
/// # Invariant
///
/// `luma`, `chroma_a` and `chroma_b` each hold exactly `width * height`
/// samples in row-major order with stride `width`. [`Self::new`],
/// [`Self::from_planes`] and [`Self::with_luma`] establish it; code that builds
/// the struct by literal (or resizes a plane afterwards) must keep it.
///
/// # Meaning of the planes
///
/// For a frame that comes from a photograph the planes are the pipeline's
/// working model: **linear-light** BT.601 `Y` (`luma`, `[0, 1]`) and `Cb`/`Cr`
/// (`chroma_a`/`chroma_b`, `[-0.5, 0.5]`) computed from linear RGB; the
/// matrices live in [`crate::color`]. The struct is also used as a generic
/// three-plane carrier (focus maps, masks, signed Laplacian bands); for those
/// the planes mean whatever the producer documents and only the invariant above
/// holds.
#[derive(Clone, Debug)]
pub struct PlanarImage<T: Float + Send + Sync> {
    /// Width in pixels (the row stride of every plane).
    pub width: usize,
    /// Height in pixels (the number of rows of every plane).
    pub height: usize,
    /// First plane: linear-light `Y` for a photograph, a generic value plane
    /// (focus map, mask, Laplacian band) otherwise.
    pub luma: Vec<T>,
    /// Second plane: `Cb` for a photograph, zero or a producer-defined value
    /// otherwise.
    pub chroma_a: Vec<T>,
    /// Third plane: `Cr` for a photograph, zero or a producer-defined value
    /// otherwise.
    pub chroma_b: Vec<T>,
}

impl<T: Float + Send + Sync> PlanarImage<T> {
    /// An all-zero image of `w × h` pixels.
    #[must_use]
    pub fn new(w: usize, h: usize) -> Self {
        Self {
            width: w,
            height: h,
            luma: vec![T::zero(); w * h],
            chroma_a: vec![T::zero(); w * h],
            chroma_b: vec![T::zero(); w * h],
        }
    }

    /// An image from three ready-made planes.
    ///
    /// # Errors
    /// Returns [`StackerError::MathError`] when a plane does not hold exactly
    /// `width * height` samples (or the pixel count overflows).
    pub fn from_planes(
        width: usize,
        height: usize,
        luma: Vec<T>,
        chroma_a: Vec<T>,
        chroma_b: Vec<T>,
    ) -> Result<Self, StackerError> {
        let expected = width.checked_mul(height).ok_or_else(|| {
            StackerError::MathError(format!("image size {width}x{height} overflows"))
        })?;
        for (name, plane) in [
            ("luma", luma.len()),
            ("chroma_a", chroma_a.len()),
            ("chroma_b", chroma_b.len()),
        ] {
            if plane != expected {
                return Err(StackerError::MathError(format!(
                    "{name} plane has {plane} samples but a {width}x{height} image needs {expected}"
                )));
            }
        }
        Ok(Self {
            width,
            height,
            luma,
            chroma_a,
            chroma_b,
        })
    }

    /// An image whose `luma` plane is given and whose chroma planes are zero
    /// (focus maps and masks).
    ///
    /// # Errors
    /// Returns [`StackerError::MathError`] when `luma` does not hold exactly
    /// `width * height` samples.
    pub fn with_luma(width: usize, height: usize, luma: Vec<T>) -> Result<Self, StackerError> {
        let len = luma.len();
        Self::from_planes(
            width,
            height,
            luma,
            vec![T::zero(); len],
            vec![T::zero(); len],
        )
    }

    /// `true` when all three planes hold exactly `width * height` samples.
    #[must_use]
    pub fn is_consistent(&self) -> bool {
        let expected = self.width.checked_mul(self.height);
        expected.is_some_and(|n| {
            self.luma.len() == n && self.chroma_a.len() == n && self.chroma_b.len() == n
        })
    }
}
