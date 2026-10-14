//! FLOW-3 / CORE-8: the luma/chroma matrices are defined once, in
//! `stacker_core::color`, and the two directions are inverses of each other.

use stacker_core::color::{
    LUMA_CHROMA_FORWARD, LUMA_CHROMA_INVERSE, linear_rgb_to_planar, planar_to_linear_rgb,
};

/// Linear RGB → planar → linear RGB returns the input on a regular grid. The
/// published six-digit constants are not exact inverses of each other, so the
/// bound is the 1e-6 level, not float equality.
#[test]
fn forward_then_inverse_round_trips_on_a_grid() {
    let steps = [0.0_f32, 0.001, 0.25, 0.5, 0.75, 0.999, 1.0];
    for &r in &steps {
        for &g in &steps {
            for &b in &steps {
                let back = planar_to_linear_rgb(linear_rgb_to_planar([r, g, b]));
                for (got, want) in back.iter().zip([r, g, b]) {
                    assert!(
                        (got - want).abs() < 5e-6,
                        "[{r}, {g}, {b}] came back as {back:?}"
                    );
                }
            }
        }
    }
}

/// The documented ranges: grey has no chroma, white has `Y = 1`, the chroma of
/// pure primaries stays within ±0.5.
#[test]
fn model_properties() {
    for v in [0.0_f32, 0.2, 1.0] {
        let [y, cb, cr] = linear_rgb_to_planar([v, v, v]);
        assert!((y - v).abs() < 1e-6);
        assert!(cb.abs() < 1e-6 && cr.abs() < 1e-6, "grey {v}: {cb} {cr}");
    }
    for rgb in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] {
        let [_, cb, cr] = linear_rgb_to_planar(rgb);
        assert!((-0.5..=0.5).contains(&cb) && (-0.5..=0.5).contains(&cr));
    }
}

/// The inverse matrix has the structure the code relies on (unit luma column,
/// the two structural zeros) and the constants are the BT.601 / JFIF values.
#[test]
fn constants_are_bt601() {
    assert_eq!(LUMA_CHROMA_FORWARD[0], [0.299, 0.587, 0.114]);
    assert_eq!(LUMA_CHROMA_FORWARD[1], [-0.168_736, -0.331_264, 0.5]);
    assert_eq!(LUMA_CHROMA_FORWARD[2], [0.5, -0.418_688, -0.081_312]);
    assert_eq!(LUMA_CHROMA_INVERSE[0], [1.0, 0.0, 1.402]);
    assert_eq!(LUMA_CHROMA_INVERSE[1], [1.0, -0.344_136, -0.714_136]);
    assert_eq!(LUMA_CHROMA_INVERSE[2], [1.0, 1.772, 0.0]);
}
