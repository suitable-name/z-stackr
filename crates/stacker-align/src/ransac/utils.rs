use crate::akaze_match::Match;
use akaze::KeyPoint;
use nalgebra::Matrix3;

/// Extract `(sx, sy, dx, dy)` tuples from akaze matches + `KeyPoint` lists.
pub(crate) fn extract_pairs(
    matches: &[Match],
    kps0: &[KeyPoint],
    kps1: &[KeyPoint],
) -> Vec<(f32, f32, f32, f32)> {
    matches
        .iter()
        .filter_map(|m| {
            let s = kps0.get(m.index_0)?;
            let d = kps1.get(m.index_1)?;

            // Use dot notation (.0, .1) for tuple access instead of indexing [0], [1]
            Some((s.point.0, s.point.1, d.point.0, d.point.1))
        })
        .collect()
}

/// Apply a 3×3 homogeneous matrix to a 2-D point.
/// Returns `None` when the homogeneous divisor is near-zero (degenerate).
pub fn apply_h(m: &Matrix3<f32>, px: f32, py: f32) -> Option<(f32, f32)> {
    let w = m[(2, 1)].mul_add(py, m[(2, 0)] * px) + m[(2, 2)];
    if w.abs() < 1e-8 {
        return None;
    }
    let x = (m[(0, 1)].mul_add(py, m[(0, 0)] * px) + m[(0, 2)]) / w;
    let y = (m[(1, 1)].mul_add(py, m[(1, 0)] * px) + m[(1, 2)]) / w;
    Some((x, y))
}

/// Squared reprojection error for one correspondence.
pub(crate) fn reproj_err_sq(m: &Matrix3<f32>, sx: f32, sy: f32, dx: f32, dy: f32) -> f32 {
    match apply_h(m, sx, sy) {
        Some((px, py)) => (py - dy).mul_add(py - dy, (px - dx) * (px - dx)),
        None => f32::MAX,
    }
}
