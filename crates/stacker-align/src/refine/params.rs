use nalgebra::Matrix3;

/// Six-DOF registration parameter model.
///
/// - `tx` / `ty` are **fractional** offsets (pixels / image dimension).
/// - `scale` is the X-axis scale factor (1.0 = no change).
/// - `rotate` is rotation in radians (0.0 = no rotation).
/// - `aspect` is the ratio of Y-scale to X-scale (1.0 = isotropic, i.e. the
///   effective Y-scale is `scale * aspect`).
/// - `shear` is the X-shear factor (0.0 = no shear).
///
/// Not every [`crate::pipeline::align_frame`] mode solves all six DOFs —
/// `Registration` (the default) and `Translation` pin `aspect = 1.0` and
/// `shear = 0.0`, reducing to a plain 4-DOF similarity model. Only `Affine`
/// enables the full 6-DOF search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegistrationParams {
    pub tx: f64,
    pub ty: f64,
    pub scale: f64,
    pub rotate: f64,
    pub aspect: f64,
    pub shear: f64,
}

impl RegistrationParams {
    /// Identity parameters: no translation, no scale change, no rotation,
    /// isotropic scale, no shear.
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

    /// Returns `true` if all fields are finite.
    pub const fn is_finite(self) -> bool {
        self.tx.is_finite()
            && self.ty.is_finite()
            && self.scale.is_finite()
            && self.rotate.is_finite()
            && self.aspect.is_finite()
            && self.shear.is_finite()
    }

    /// Pack into a `[f64; 6]` for the optimizer.
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

    /// Unpack from a `[f64; 6]`.
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
}

/// Build a 3×3 homogeneous forward transform from `RegistrationParams`.
///
/// ```text
/// M = T(tx*W, ty*H) · T(cx, cy) · R(rotate) · Shear(shear) · S(scale, scale*aspect) · T(-cx, -cy)
/// ```
///
/// where `cx = W/2`, `cy = H/2`, `R(rotate) = [[cos,-sin],[sin,cos]]`,
/// `Shear(k) = [[1,k],[0,1]]`, and `S(sx,sy) = diag(sx,sy)`.  The image
/// centre is the pivot for rotation/shear/scale; fractional translation is
/// applied after.
///
/// ## Closed form
///
/// Let `A = R(rotate) · Shear(shear) · S(scale, scale*aspect)`. Expanding:
///
/// ```text
/// a00 = scale * cos_r
/// a10 = scale * sin_r
/// a01 = scale * aspect * (cos_r * shear - sin_r)
/// a11 = scale * aspect * (sin_r * shear + cos_r)
/// ```
///
/// The translation column follows a centre-anchored convention
/// (`p -> A*(p - c) + c + t`), generalised to the full 2×2 block `A`:
///
/// ```text
/// a02 = cx * (1 - a00) - a01 * cy + tx
/// a12 = cy * (1 - a11) - a10 * cx + ty
/// ```
///
/// This reduces exactly to the plain similarity-transform formula when
/// `a01 = -a10` and `a11 = a00` (i.e. `aspect = 1`, `shear = 0`): `-a01*cy`
/// becomes `+a10*cy`, matching the `cy * a10` term that formula uses.
///
/// Returns `None` if any parameter is non-finite.
pub fn params_to_matrix(
    p: &RegistrationParams,
    width: usize,
    height: usize,
) -> Option<Matrix3<f32>> {
    if !p.is_finite() {
        return None;
    }
    let w = width as f64;
    let h = height as f64;
    let cx = w * 0.5;
    let cy = h * 0.5;

    let cos_r = p.rotate.cos();
    let sin_r = p.rotate.sin();
    let s = p.scale;
    let sy = p.scale * p.aspect;
    let k = p.shear;
    let tx = p.tx * w;
    let ty = p.ty * h;

    // A = R(rotate) · Shear(shear) · S(scale, scale*aspect):
    //   Shear·S = [[s, k*sy], [0, sy]]
    //   R·(Shear·S) = [[s*cos_r, sy*(cos_r*k - sin_r)],
    //                  [s*sin_r, sy*(sin_r*k + cos_r)]]
    let a00 = s * cos_r;
    let a10 = s * sin_r;
    let a01 = sy * cos_r.mul_add(k, -sin_r);
    let a11 = sy * sin_r.mul_add(k, cos_r);
    let a02 = cx.mul_add(1.0 - a00, -(a01 * cy)) + tx;
    let a12 = cy.mul_add(1.0 - a11, -(a10 * cx)) + ty;

    for v in [a00, a01, a02, a10, a11, a12] {
        if !v.is_finite() {
            return None;
        }
    }

    let mut m = Matrix3::<f32>::identity();
    m[(0, 0)] = a00 as f32;
    m[(0, 1)] = a01 as f32;
    m[(0, 2)] = a02 as f32;
    m[(1, 0)] = a10 as f32;
    m[(1, 1)] = a11 as f32;
    m[(1, 2)] = a12 as f32;
    Some(m)
}

/// Decompose a 3×3 affine+translation matrix into `RegistrationParams`.
///
/// Uses the image centre as the rotation/scale/shear pivot (same convention
/// as [`params_to_matrix`]). This is the exact algebraic inverse of
/// [`params_to_matrix`]'s composition for any non-degenerate 2×2 block —
/// critical because `pipeline::registration_rms_at_dims` decomposes a
/// full-resolution matrix and rebuilds it at a downsampled resolution; a
/// lossy decomposition would break the post-refinement identity gate.
///
/// ## Decomposition (QR-style)
///
/// Given the 2×2 block `A = [[a00,a01],[a10,a11]]`:
///
/// ```text
/// scale_x    = hypot(a00, a10)
/// rotate     = atan2(a10, a00)
/// shear_term = (a00*a01 + a10*a11) / scale_x²
/// scale_y    = det(A) / scale_x
/// aspect     = scale_y / scale_x
/// shear      = shear_term * (scale_x / scale_y)
/// ```
///
/// Substituting `A`'s closed form from [`params_to_matrix`] shows this
/// recovers the original parameters exactly: `shear_term` works out to
/// `aspect * shear`, so `shear_term * (scale_x/scale_y) = aspect * shear /
/// aspect = shear`, and `scale_y/scale_x = aspect` directly.
///
/// Returns `RegistrationParams::identity()` for a non-finite or degenerate
/// (near-singular) matrix so the optimiser always has a safe starting point.
pub fn matrix_to_params(m: &Matrix3<f32>, width: usize, height: usize) -> RegistrationParams {
    if !m.iter().all(|v| v.is_finite()) {
        return RegistrationParams::identity();
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

    // scale_x = sqrt(a00² + a10²)
    let scale_x = a00.hypot(a10);
    if scale_x < 1.0e-10 {
        return RegistrationParams::identity();
    }

    let rotate = a10.atan2(a00);

    let det = a00 * a11 - a01 * a10;
    let scale_y = det / scale_x;
    if scale_y.abs() < 1.0e-10 {
        return RegistrationParams::identity();
    }

    #[allow(clippy::suspicious_operation_groupings)]
    let shear_term = (a00 * a01 + a10 * a11) / (scale_x * scale_x);
    let aspect = scale_y / scale_x;
    let shear = shear_term * (scale_x / scale_y);

    // Recover absolute tx/ty from the translation column (general 2×2
    // block, generalising the old similarity-only formula):
    //   a02 = cx*(1 - a00) - a01*cy + tx_abs  =>  tx_abs = a02 - cx*(1-a00) + a01*cy
    //   a12 = cy*(1 - a11) - a10*cx + ty_abs  =>  ty_abs = a12 - cy*(1-a11) + a10*cx

    let (tx, ty) = {
        let abs_x = a02 - cx * (1.0 - a00) + a01 * cy;
        let abs_y = a12 - cy * (1.0 - a11) + a10 * cx;
        (abs_x / w, abs_y / h)
    };

    let scale = scale_x;
    if !tx.is_finite()
        || !ty.is_finite()
        || !scale.is_finite()
        || !rotate.is_finite()
        || !aspect.is_finite()
        || !shear.is_finite()
    {
        return RegistrationParams::identity();
    }

    RegistrationParams {
        tx,
        ty,
        scale,
        rotate,
        aspect,
        shear,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Matrix3;

    /// `params_to_matrix` → `matrix_to_params` → `params_to_matrix` must
    /// round-trip exactly (to floating-point tolerance) for a batch of
    /// non-degenerate 6-DOF parameter sets, including negative shear,
    /// anisotropic aspect (0.9..1.1), and rotations up to ±10°. This is the
    /// regression guard for the QR-style decomposition documented on
    /// [`matrix_to_params`]: a lossy round-trip would silently break the
    /// `pipeline::registration_rms_at_dims` full-res → downsampled rebuild
    /// path and the post-refinement identity gate that depends on it.
    #[test]
    fn params_matrix_round_trip_full_affine() {
        let w = 200_usize;
        let h = 150_usize;

        // Deterministic LCG so the test is reproducible without extra deps.
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 * f64::from_bits(0x3CA0_0000_0000_0000_u64)
        };

        for case in 0..200 {
            let tx = (next() - 0.5) * 0.2; // ±10% of width
            let ty = (next() - 0.5) * 0.2; // ±10% of height
            let scale = 0.8 + next() * 0.4; // 0.8..1.2
            let rotate_deg = (next() - 0.5) * 20.0; // ±10 deg
            let rotate = rotate_deg * std::f64::consts::PI / 180.0;
            let aspect = 0.9 + next() * 0.2; // 0.9..1.1
            let shear = (next() - 0.5) * 0.4; // includes negative shear

            let orig = RegistrationParams {
                tx,
                ty,
                scale,
                rotate,
                aspect,
                shear,
            };

            let m1 = params_to_matrix(&orig, w, h).unwrap_or_else(|| {
                panic!("case {case}: params_to_matrix must succeed for {orig:?}")
            });
            let recovered = matrix_to_params(&m1, w, h);
            let m2 = params_to_matrix(&recovered, w, h).unwrap_or_else(|| {
                panic!("case {case}: params_to_matrix (2nd pass) must succeed for {recovered:?}")
            });

            let diff = (m1 - m2).norm();
            assert!(
                diff < 1.0e-4,
                "case {case}: round-trip matrix element error {diff:.8} too large; \
                 orig={orig:?}, recovered={recovered:?}"
            );
        }
    }

    /// Degenerate (near-singular) matrices must decompose to identity
    /// params rather than producing NaN/garbage — the safe-fallback
    /// contract documented on [`matrix_to_params`].
    #[test]
    fn matrix_to_params_degenerate_returns_identity() {
        // All-zero 2×2 block: scale_x == 0.
        let mut m = Matrix3::<f32>::identity();
        m[(0, 0)] = 0.0;
        m[(1, 0)] = 0.0;
        m[(0, 1)] = 0.0;
        m[(1, 1)] = 0.0;
        let p = matrix_to_params(&m, 100, 100);
        assert_eq!(p, RegistrationParams::identity());

        // Rank-deficient (both rows parallel): det == 0, scale_y == 0.
        let mut m2 = Matrix3::<f32>::identity();
        m2[(0, 0)] = 1.0;
        m2[(1, 0)] = 1.0;
        m2[(0, 1)] = 1.0;
        m2[(1, 1)] = 1.0;
        let p2 = matrix_to_params(&m2, 100, 100);
        assert_eq!(p2, RegistrationParams::identity());
    }
}
