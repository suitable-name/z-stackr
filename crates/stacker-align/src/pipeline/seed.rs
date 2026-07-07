use crate::Matrix3;

/// Return `true` if `m` is a plausible coarse alignment seed.
///
/// A sane seed is all-finite, with a similarity scale within a sane band and a
/// translation that stays within 25% of the respective frame dimension.
/// Garbage feature-match estimates (common on glossy / featureless subjects,
/// or on defocused macro frames where most AKAZE matches are wrong) are
/// rejected so the intensity optimiser falls back to identity instead of
/// chasing a wild transform.
///
/// The translation bound is deliberately tight: `refine_alignment_registration`
/// searches a bounded window centred on the seed (+/-20% of the dimension by
/// default), so a seed anywhere near a full-frame-width translation would put
/// the true alignment outside the reachable search interval and the
/// optimiser would converge on a confidently wrong answer instead of failing
/// loudly. A focus-stack frame that has genuinely shifted by more than a
/// quarter frame relative to the reference is beyond what this pipeline can
/// rescue anyway.
#[must_use]
pub fn is_sane_seed(m: &Matrix3<f32>, width: usize, height: usize) -> bool {
    if !m.iter().all(|v| v.is_finite()) {
        return false;
    }
    let sx = m[(0, 0)].hypot(m[(1, 0)]);
    let sy = m[(0, 1)].hypot(m[(1, 1)]);
    if !(0.5..=2.0).contains(&sx) || !(0.5..=2.0).contains(&sy) {
        return false;
    }
    m[(0, 2)].abs() <= width as f32 * 0.25 && m[(1, 2)].abs() <= height as f32 * 0.25
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_sane_seed_rejects_translation_beyond_quarter_dimension() {
        let width = 200_usize;
        let height = 100_usize;

        let mut m = Matrix3::<f32>::identity();
        // Exactly at the 25% boundary: sane.
        m[(0, 2)] = width as f32 * 0.25;
        m[(1, 2)] = height as f32 * 0.25;
        assert!(is_sane_seed(&m, width, height));

        // Just beyond the 25% boundary: not sane.
        let mut m2 = Matrix3::<f32>::identity();
        m2[(0, 2)] = (width as f32).mul_add(0.25, 1.0);
        assert!(!is_sane_seed(&m2, width, height));

        // A full-frame-width translation must be rejected.
        let mut m3 = Matrix3::<f32>::identity();
        m3[(0, 2)] = width as f32 * 0.9;
        assert!(!is_sane_seed(&m3, width, height));
    }
}
