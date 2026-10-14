use stacker_core::image::PlanarImage;

#[test]
fn test_planar_image_new() {
    let img = PlanarImage::<f32>::new(10, 20);
    assert_eq!(img.width, 10);
    assert_eq!(img.height, 20);
    assert_eq!(img.luma.len(), 200);
    assert_eq!(img.chroma_a.len(), 200);
    assert_eq!(img.chroma_b.len(), 200);
}

/// CORE-8: `from_planes` accepts exactly `width * height` samples per plane.
#[test]
fn from_planes_checks_every_plane_length() {
    let ok = PlanarImage::from_planes(3, 2, vec![0.0_f32; 6], vec![1.0; 6], vec![2.0; 6]).unwrap();
    assert_eq!((ok.width, ok.height), (3, 2));
    assert!(ok.is_consistent());
    assert!((ok.chroma_b[5] - 2.0).abs() < f32::EPSILON);

    for (l, a, b) in [(5, 6, 6), (6, 7, 6), (6, 6, 0)] {
        let err = PlanarImage::from_planes(3, 2, vec![0.0_f32; l], vec![0.0; a], vec![0.0; b])
            .unwrap_err();
        assert!(err.to_string().contains("needs 6"), "{err}");
    }
}

/// `with_luma` keeps the plane and zeroes the chroma; a wrong length is an error
/// only when it disagrees with the size.
#[test]
fn with_luma_zeroes_the_chroma_planes() {
    let img = PlanarImage::with_luma(2, 2, vec![0.25_f32, 0.5, 0.75, 1.0]).unwrap();
    assert_eq!(img.luma, vec![0.25, 0.5, 0.75, 1.0]);
    assert_eq!(img.chroma_a, vec![0.0; 4]);
    assert_eq!(img.chroma_b, vec![0.0; 4]);
    assert!(PlanarImage::<f32>::with_luma(2, 2, vec![0.0; 3]).is_err());
}

/// A literal-built image that breaks the invariant is detected.
#[test]
fn is_consistent_detects_a_broken_literal() {
    let mut img = PlanarImage::<f32>::new(4, 4);
    assert!(img.is_consistent());
    img.chroma_a.pop();
    assert!(!img.is_consistent());
}
