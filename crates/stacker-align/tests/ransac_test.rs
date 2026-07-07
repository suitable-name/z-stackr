#![cfg(feature = "akaze")]

use stacker_align::{Matrix3, ransac::apply_h};

#[test]
fn test_apply_h_identity() {
    let m = Matrix3::identity();
    let res = apply_h(&m, 5.0, 10.0);
    assert_eq!(res, Some((5.0, 10.0)));
}

#[test]
fn test_apply_h_translation() {
    let mut m = Matrix3::identity();
    m[(0, 2)] = 10.0;
    m[(1, 2)] = -5.0;
    let res = apply_h(&m, 5.0, 10.0);
    assert_eq!(res, Some((15.0, 5.0)));
}
