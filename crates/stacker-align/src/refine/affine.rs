//! The six-degree-of-freedom parameter model.
//!
//! [`RegistrationParams`] (shift, uniform scale, rotation) can only express a
//! similarity.  [`AffineParams`] adds the two remaining degrees of freedom of a
//! full affine map: an **aspect** ratio (independent Y scale) and a **shear**.
//! The model is the one of the GitHub z-stackr `Affine` mode:
//!
//! ```text
//! M = T(tx*W, ty*H) · T(c) · R(rotate) · Shear(shear) · S(scale, scale*aspect) · T(-c)
//! ```
//!
//! with `c = (W/2, H/2)`, `R(r) = [[cos r, -sin r], [sin r, cos r]]`,
//! `Shear(k) = [[1, k], [0, 1]]` and `S(sx, sy) = diag(sx, sy)`.
//!
//! ## Closed form
//!
//! With `s = scale`, `sy = scale * aspect` and `k = shear` the linear part
//! `A = R · Shear · S` is
//!
//! ```text
//! a00 = s  · cos r            a01 = sy · (k · cos r - sin r)
//! a10 = s  · sin r            a11 = sy · (k · sin r + cos r)
//! ```
//!
//! and the translation column keeps the centre fixed before the shift:
//!
//! ```text
//! a02 = cx · (1 - a00) - a01 · cy + tx · W
//! a12 = cy · (1 - a11) - a10 · cx + ty · H
//! ```
//!
//! ## Bit identity with the four-parameter model
//!
//! For `aspect == 1` and `shear == 0` the matrix is built by
//! [`params_to_matrix`] itself, not by the closed form above (the two agree
//! mathematically but not in the last floating-point bit, because the
//! translation column is associated differently).  Every four-parameter search
//! therefore runs through exactly the arithmetic it always did, which keeps the
//! recorded goldens and the ask/tell bit-identity tests valid.
//!
//! The decomposition [`matrix_to_affine_params`] is the exact (QR-style)
//! inverse, unlike [`super::matrix_to_params`], which reduces to the nearest
//! similarity.

#![allow(
    clippy::cast_precision_loss,
    reason = "image dimensions are far below 2^52, so the usize to f64 conversion is exact"
)]
#![allow(
    clippy::cast_possible_truncation,
    reason = "the f64 parameters are narrowed to the f32 warp matrix on purpose"
)]
#![allow(
    clippy::suboptimal_flops,
    reason = "plain multiply-add keeps the arithmetic identical on targets with and without FMA"
)]
#![allow(
    clippy::many_single_char_names,
    reason = "w/h, cx/cy, s/k and the matrix entries are the textbook names"
)]

use nalgebra::Matrix3;

use super::{RegistrationParams, params_to_matrix};

/// Six-degree-of-freedom registration parameters.
///
/// - `tx` / `ty` are **fractional** shifts (pixels / image dimension).
/// - `scale` is the X scale factor (1.0 = unchanged); the Y scale is
///   `scale * aspect`.
/// - `rotate` is the rotation about the image centre in radians.
/// - `aspect` is the ratio of the Y scale to the X scale (1.0 = isotropic).
/// - `shear` is the X shear factor (0.0 = none).
///
/// Like [`RegistrationParams`] the matrix built from it maps **reference
/// coordinates to source coordinates**.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AffineParams {
    /// Horizontal shift as a fraction of the image width.
    pub tx: f64,
    /// Vertical shift as a fraction of the image height.
    pub ty: f64,
    /// X scale about the image centre (1.0 = unchanged).
    pub scale: f64,
    /// Rotation about the image centre in radians (0.0 = unchanged).
    pub rotate: f64,
    /// Y scale relative to the X scale (1.0 = isotropic).
    pub aspect: f64,
    /// X shear factor (0.0 = none).
    pub shear: f64,
}

impl AffineParams {
    /// Identity parameters: no shift, unit scale, no rotation, isotropic, no
    /// shear.
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            tx: 0.0,
            ty: 0.0,
            scale: 1.0,
            rotate: 0.0,
            aspect: 1.0,
            shear: 0.0,
        }
    }

    /// `true` if all six fields are finite.
    #[must_use]
    pub const fn is_finite(self) -> bool {
        self.tx.is_finite()
            && self.ty.is_finite()
            && self.scale.is_finite()
            && self.rotate.is_finite()
            && self.aspect.is_finite()
            && self.shear.is_finite()
    }

    /// Pack into `[tx, ty, scale, rotate, aspect, shear]`.
    #[must_use]
    pub const fn to_vec(self) -> [f64; 6] {
        [
            self.tx,
            self.ty,
            self.scale,
            self.rotate,
            self.aspect,
            self.shear,
        ]
    }

    /// Unpack from `[tx, ty, scale, rotate, aspect, shear]`.
    #[must_use]
    pub const fn from_vec(v: &[f64; 6]) -> Self {
        Self {
            tx: v[0],
            ty: v[1],
            scale: v[2],
            rotate: v[3],
            aspect: v[4],
            shear: v[5],
        }
    }

    /// The four similarity parameters; `aspect` and `shear` are dropped.
    #[must_use]
    pub const fn similarity(self) -> RegistrationParams {
        RegistrationParams {
            tx: self.tx,
            ty: self.ty,
            scale: self.scale,
            rotate: self.rotate,
        }
    }

    /// `true` when `aspect == 1` and `shear == 0`, i.e. the parameters are a
    /// plain similarity and the four-parameter arithmetic applies.
    fn is_similarity(&self) -> bool {
        self.aspect.to_bits() == 1.0_f64.to_bits() && self.shear == 0.0
    }

    /// The forward matrix for an image of `width x height` pixels, or `None`
    /// when a parameter or a matrix entry is not finite.
    ///
    /// For `aspect == 1`, `shear == 0` this is [`params_to_matrix`] of
    /// [`Self::similarity`], bit for bit.
    #[must_use]
    pub fn to_matrix(&self, width: usize, height: usize) -> Option<Matrix3<f32>> {
        if !self.is_finite() {
            return None;
        }
        if self.is_similarity() {
            return params_to_matrix(&self.similarity(), width, height);
        }
        let e = self.entries(width, height);
        if !e.iter().all(|v| v.is_finite()) {
            return None;
        }
        let mut m = Matrix3::<f32>::identity();
        m[(0, 0)] = e[0] as f32;
        m[(0, 1)] = e[1] as f32;
        m[(0, 2)] = e[2] as f32;
        m[(1, 0)] = e[3] as f32;
        m[(1, 1)] = e[4] as f32;
        m[(1, 2)] = e[5] as f32;
        Some(m)
    }

    /// The six free matrix entries `[a00, a01, a02, a10, a11, a12]` of the closed
    /// form in `f64`, always from the general six-parameter formula (also for a
    /// similarity), so that finite differences over any parameter see one smooth
    /// function.  No finiteness check.
    #[must_use]
    pub(crate) fn entries(&self, width: usize, height: usize) -> [f64; 6] {
        let w = width as f64;
        let h = height as f64;
        let cx = w * 0.5;
        let cy = h * 0.5;
        let (sin_r, cos_r) = self.rotate.sin_cos();
        let s = self.scale;
        let sy = self.scale * self.aspect;
        let k = self.shear;
        let a00 = s * cos_r;
        let a10 = s * sin_r;
        let a01 = sy * (cos_r * k - sin_r);
        let a11 = sy * (sin_r * k + cos_r);
        let a02 = cx * (1.0 - a00) - a01 * cy + self.tx * w;
        let a12 = cy * (1.0 - a11) - a10 * cx + self.ty * h;
        [a00, a01, a02, a10, a11, a12]
    }
}

impl From<RegistrationParams> for AffineParams {
    /// The similarity as six parameters (`aspect = 1`, `shear = 0`).
    fn from(p: RegistrationParams) -> Self {
        Self {
            tx: p.tx,
            ty: p.ty,
            scale: p.scale,
            rotate: p.rotate,
            aspect: 1.0,
            shear: 0.0,
        }
    }
}

/// Build the forward matrix of `p` for an image of `width x height` pixels;
/// [`AffineParams::to_matrix`] as a free function.  `None` if any parameter or
/// matrix entry is not finite.
#[must_use]
pub fn affine_params_to_matrix(
    p: &AffineParams,
    width: usize,
    height: usize,
) -> Option<Matrix3<f32>> {
    p.to_matrix(width, height)
}

/// Decompose a 3x3 affine matrix into the six parameters, exactly.
///
/// For a matrix built by [`affine_params_to_matrix`] the parameters come back
/// (up to the f32 rounding of the matrix).  Unlike [`super::matrix_to_params`]
/// nothing is reduced to a similarity.
///
/// With the linear part `A = [[a00, a01], [a10, a11]]`:
///
/// ```text
/// scale_x    = hypot(a00, a10)          rotate = atan2(a10, a00)
/// shear_term = (a00·a01 + a10·a11) / scale_x²
/// scale_y    = det(A) / scale_x         aspect = scale_y / scale_x
/// shear      = shear_term · scale_x / scale_y
/// ```
///
/// and the shift is the displacement of the image centre taken out of the
/// translation column.  The identity is returned for a non-finite or
/// (near-)singular matrix so a search always has a safe start.
#[must_use]
pub fn matrix_to_affine_params(m: &Matrix3<f32>, width: usize, height: usize) -> AffineParams {
    if !m.iter().all(|v| v.is_finite()) {
        return AffineParams::identity();
    }
    let w = width as f64;
    let h = height as f64;
    let cx = w * 0.5;
    let cy = h * 0.5;

    let a00 = f64::from(m[(0, 0)]);
    let a01 = f64::from(m[(0, 1)]);
    let a10 = f64::from(m[(1, 0)]);
    let a11 = f64::from(m[(1, 1)]);
    let a02 = f64::from(m[(0, 2)]);
    let a12 = f64::from(m[(1, 2)]);

    let scale_x = a00.hypot(a10);
    if scale_x < 1.0e-10 {
        return AffineParams::identity();
    }
    let rotate = a10.atan2(a00);
    let det = a00 * a11 - a01 * a10;
    let scale_y = det / scale_x;
    if scale_y.abs() < 1.0e-10 {
        return AffineParams::identity();
    }
    let scale_x_sq = scale_x * scale_x;
    let shear_term = (a00 * a01 + a10 * a11) / scale_x_sq;
    let aspect = scale_y / scale_x;
    let shear = shear_term * (scale_x / scale_y);

    // a02 = cx (1 - a00) - a01 cy + tx W  =>  tx W = a02 - cx (1 - a00) + a01 cy.
    let tx = (a02 - cx * (1.0 - a00) + a01 * cy) / w;
    let ty = (a12 - cy * (1.0 - a11) + a10 * cx) / h;

    let p = AffineParams {
        tx,
        ty,
        scale: scale_x,
        rotate,
        aspect,
        shear,
    };
    if p.is_finite() {
        p
    } else {
        AffineParams::identity()
    }
}
