//! Tests for the alignment side of the planar bridge: the pixel / normalised /
//! thumbnail matrix conventions, thumbnailing, and the thumbnail-based alignment
//! entry points.
//!
//! The conventions are the ones documented in `stacker_nn::bridge`: matrices map
//! reference coordinates to frame coordinates; a thumbnail pixel `i` sits at full
//! resolution position `(i + 1/2)·W/tw - 1/2`; ground truth for a crop is
//! `T(-o)·M·T(o)` normalised by the crop's own size.
#![cfg(feature = "ndarray")]
#![allow(
    clippy::suboptimal_flops,
    clippy::many_single_char_names,
    reason = "plain fixture arithmetic and conventional matrix/size names"
)]

use burn::{
    backend::NdArray,
    prelude::{Backend, Tensor},
    tensor::TensorData,
};
use stacker_core::image::PlanarImage;
use stacker_nn::{
    bridge::{
        ALIGN_DOWNSCALE_LONG_SIDE, BridgeError, FrameThumbs, PixelMatrix, align_planar,
        align_planar_pairwise, align_thumbs, crop_pixel_matrix, make_thumbnail,
        normalized_matrix_to_pixel, pixel_matrix_to_normalized, thumbnail_size,
    },
    model::{BatchAlignNetConfig, FusionAlignNetConfig, ModelSize},
    warp::warp_affine,
};

type B = NdArray;

const IDENTITY: PixelMatrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

fn float(v: i32) -> f32 {
    f32::from(i16::try_from(v).unwrap())
}

fn apply(m: &PixelMatrix, x: f32, y: f32) -> (f32, f32) {
    (
        m[0][0] * x + m[0][1] * y + m[0][2],
        m[1][0] * x + m[1][1] * y + m[1][2],
    )
}

fn assert_matrix(got: &PixelMatrix, want: &PixelMatrix, tol: f32, what: &str) {
    for (i, (g, w)) in got.iter().flatten().zip(want.iter().flatten()).enumerate() {
        assert!(
            (g - w).abs() <= tol,
            "{what}: element {i} is {g}, want {w} (matrix {got:?})"
        );
    }
}

/// A smooth analytic texture, defined for any integer position.
fn texture(x: i32, y: i32) -> f32 {
    let (x, y) = (float(x), float(y));
    (0.07 * x + 0.03 * y).sin() + 0.5 * (0.04 * x - 0.09 * y).cos()
}

/// A planar image whose luma is `texture(x - dx, y - dy)`; chroma is zero.
fn textured(w: usize, h: usize, dx: i32, dy: i32) -> PlanarImage<f32> {
    let mut img = PlanarImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            img.luma[y * w + x] = texture(
                i32::try_from(x).unwrap() - dx,
                i32::try_from(y).unwrap() - dy,
            );
        }
    }
    img
}

// ---------------------------------------------------------------------------
// Matrix conventions
// ---------------------------------------------------------------------------

#[test]
fn pixel_and_normalised_matrices_round_trip_and_a_translation_scales() {
    let (w, h) = (101, 51);
    let m: PixelMatrix = [[1.02, -0.03, 4.0], [0.03, 0.98, -2.5], [0.0, 0.0, 1.0]];
    let n = pixel_matrix_to_normalized(m, w, h);
    assert_matrix(&normalized_matrix_to_pixel(n, w, h), &m, 1e-5, "round trip");

    // A pure translation of (dx, dy) pixels is (2dx/(w-1), 2dy/(h-1)) normalised.
    let t = pixel_matrix_to_normalized([[1.0, 0.0, 10.0], [0.0, 1.0, 5.0], [0.0, 0.0, 1.0]], w, h);
    assert_matrix(
        &t,
        &[[1.0, 0.0, 0.2], [0.0, 1.0, 0.2], [0.0, 0.0, 1.0]],
        1e-6,
        "translation",
    );

    // The identity stays the identity.
    assert_matrix(
        &pixel_matrix_to_normalized(IDENTITY, w, h),
        &IDENTITY,
        1e-6,
        "identity",
    );
}

#[test]
fn a_crop_matrix_describes_the_same_motion_in_crop_coordinates() {
    let m: PixelMatrix = [[1.01, 0.02, 3.0], [-0.02, 0.99, -1.0], [0.0, 0.0, 1.0]];
    let (ox, oy) = (40.0_f32, 25.0_f32);
    let m_crop = crop_pixel_matrix(m, ox, oy);
    for (qx, qy) in [(0.0_f32, 0.0_f32), (10.0, 7.0), (63.0, 31.0)] {
        // Reference crop pixel q = full pixel q + o; its frame position is M·(q + o);
        // in frame-crop coordinates that is M·(q + o) - o.
        let (fx, fy) = apply(&m, qx + ox, qy + oy);
        let (want_x, want_y) = (fx - ox, fy - oy);
        let (got_x, got_y) = apply(&m_crop, qx, qy);
        assert!((got_x - want_x).abs() < 1e-3 && (got_y - want_y).abs() < 1e-3);
    }
    // A pure translation is unaffected by the crop origin.
    let shift: PixelMatrix = [[1.0, 0.0, 6.0], [0.0, 1.0, -3.0], [0.0, 0.0, 1.0]];
    assert_matrix(
        &crop_pixel_matrix(shift, ox, oy),
        &shift,
        1e-4,
        "translation",
    );
}

/// Normalising a crop's ground truth by the *full* frame (the original defect)
/// gives a different matrix than normalising by the crop: the two must differ, and
/// only the crop-relative one reproduces the motion of the crop's own pixels.
#[test]
fn crop_ground_truth_must_be_normalised_by_the_crop_not_the_frame() {
    let (full_w, full_h, crop) = (512_usize, 384_usize, 256_usize);
    let shift: PixelMatrix = [[1.0, 0.0, 8.0], [0.0, 1.0, 4.0], [0.0, 0.0, 1.0]];
    let by_crop = pixel_matrix_to_normalized(crop_pixel_matrix(shift, 100.0, 60.0), crop, crop);
    let by_frame = pixel_matrix_to_normalized(shift, full_w, full_h);
    // 8 px of a 256-px crop is 2·8/255 normalised; of a 512-px frame 2·8/511.
    assert!((by_crop[0][2] - 16.0 / 255.0).abs() < 1e-6);
    assert!((by_frame[0][2] - 16.0 / 511.0).abs() < 1e-6);
    assert!((by_crop[0][2] - by_frame[0][2]).abs() > 0.01);
}

// ---------------------------------------------------------------------------
// Thumbnails
// ---------------------------------------------------------------------------

#[test]
fn thumbnail_sizes_follow_the_long_side_rule() {
    assert_eq!(ALIGN_DOWNSCALE_LONG_SIDE, 512);
    assert_eq!(
        thumbnail_size(400, 300),
        (400, 300),
        "within the limit: unchanged"
    );
    assert_eq!(thumbnail_size(512, 512), (512, 512));
    assert_eq!(thumbnail_size(1024, 768), (512, 384));
    assert_eq!(thumbnail_size(768, 1024), (384, 512));
    assert_eq!(thumbnail_size(6000, 4000), (512, 341));
    assert_eq!(
        thumbnail_size(5000, 1),
        (512, 2),
        "each side stays at least 2"
    );
}

#[test]
fn thumbnailing_is_an_area_average_in_linear_light() {
    // Constant planes stay constant, through the fractional-footprint path too.
    let mut flat = PlanarImage::<f32>::new(777, 555);
    flat.luma.fill(0.3);
    flat.chroma_a.fill(-0.1);
    flat.chroma_b.fill(0.05);
    let thumb = make_thumbnail(&flat);
    assert_eq!((thumb.width, thumb.height), thumbnail_size(777, 555));
    assert!(thumb.luma.iter().all(|v| (v - 0.3).abs() < 1e-5));
    assert!(thumb.chroma_a.iter().all(|v| (v + 0.1).abs() < 1e-5));
    assert!(thumb.chroma_b.iter().all(|v| (v - 0.05).abs() < 1e-5));

    // An exact 2x reduction is the mean of every 2x2 block.
    let mut img = PlanarImage::<f32>::new(1024, 8);
    for y in 0..8 {
        for x in 0..1024 {
            img.luma[y * 1024 + x] = if (x + y) % 2 == 0 { 1.0 } else { 0.0 };
        }
    }
    let thumb = make_thumbnail(&img);
    assert_eq!((thumb.width, thumb.height), (512, 4));
    assert!(
        thumb.luma.iter().all(|v| (v - 0.5).abs() < 1e-6),
        "checkerboard averages to 1/2"
    );

    // A small frame is returned as is.
    let small = textured(40, 30, 0, 0);
    let same = make_thumbnail(&small);
    assert_eq!(same.luma, small.luma);
}

#[test]
fn frame_thumbs_validate_their_input() {
    assert!(matches!(
        FrameThumbs::new(1, 100),
        Err(BridgeError::TooSmall {
            width: 1,
            height: 100
        })
    ));
    assert!(matches!(
        FrameThumbs::from_frames(&Vec::<PlanarImage<f32>>::new()),
        Err(BridgeError::EmptyStack)
    ));
    let frames = vec![textured(40, 30, 0, 0), textured(40, 31, 0, 0)];
    assert!(matches!(
        FrameThumbs::from_frames(&frames),
        Err(BridgeError::ShapeMismatch {
            index: 1,
            exp_w: 40,
            exp_h: 30,
            got_w: 40,
            got_h: 31
        })
    ));

    let ok = FrameThumbs::from_frames(&[textured(40, 30, 0, 0), textured(40, 30, 1, 0)]).unwrap();
    assert_eq!((ok.len(), ok.is_empty()), (2, false));
    assert_eq!(ok.full_size(), (40, 30));
    assert_eq!(ok.thumb_size(), (40, 30));
}

#[test]
fn thumbnail_geometry_maps_normalised_matrices_to_full_resolution_pixels() {
    // 1000 x 700 -> 512 x 358 (non-integer scale).
    let thumbs = FrameThumbs::new(1000, 700).unwrap();
    assert_eq!(thumbs.thumb_size(), (512, 358));

    // Round trip of an arbitrary matrix.
    let m: PixelMatrix = [[1.01, -0.02, 12.0], [0.02, 0.99, -7.0], [0.0, 0.0, 1.0]];
    let n = thumbs.full_to_normalized(m);
    assert_matrix(&thumbs.normalized_to_full(n), &m, 1e-3, "round trip");

    // The identity is the identity at any scale.
    assert_matrix(
        &thumbs.normalized_to_full(IDENTITY),
        &IDENTITY,
        1e-6,
        "identity",
    );

    // A scale about the thumbnail centre is the same scale about the full-resolution
    // centre ((W-1)/2, (H-1)/2): both grids are symmetric about the image centre.
    let s = 1.01_f32;
    let (cx, cy) = (1000.0_f32 - 1.0, 700.0_f32 - 1.0);
    let about_centre: PixelMatrix = [
        [s, 0.0, (1.0 - s) * cx / 2.0],
        [0.0, s, (1.0 - s) * cy / 2.0],
        [0.0, 0.0, 1.0],
    ];
    let scale_n: PixelMatrix = [[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, 1.0]];
    assert_matrix(
        &thumbs.normalized_to_full(scale_n),
        &about_centre,
        1e-3,
        "scale about the centre",
    );
}

#[test]
fn a_two_times_thumbnail_halves_a_translation() {
    let thumbs = FrameThumbs::new(1024, 768).unwrap();
    assert_eq!(thumbs.thumb_size(), (512, 384));
    // 4 x 2 thumbnail pixels = 8 x 4 full-resolution pixels.
    let n = pixel_matrix_to_normalized(
        [[1.0, 0.0, 4.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]],
        512,
        384,
    );
    assert_matrix(
        &thumbs.normalized_to_full(n),
        &[[1.0, 0.0, 8.0], [0.0, 1.0, 4.0], [0.0, 0.0, 1.0]],
        1e-3,
        "translation",
    );
    assert_matrix(
        &thumbs.full_to_normalized([[1.0, 0.0, 8.0], [0.0, 1.0, 4.0], [0.0, 0.0, 1.0]]),
        &n,
        1e-5,
        "normalised",
    );
}

/// End to end: a stack shifted by a known full-resolution amount, thumbnailed,
/// converted with the geometry and warped, reproduces the reference thumbnail.
#[test]
fn a_known_shift_survives_thumbnailing_conversion_and_warping() {
    let (w, h) = (1024_usize, 64_usize);
    let (dx, dy) = (8, 4); // full-resolution pixels; 4 x 2 in the 512 x 32 thumbnails
    let reference = textured(w, h, 0, 0);
    let source = textured(w, h, dx, dy); // source(q) = reference(q - d)

    let mut thumbs = FrameThumbs::new(w, h).unwrap();
    thumbs.push(&reference).unwrap();
    thumbs.push(&source).unwrap();
    assert_eq!(thumbs.thumb_size(), (512, 32));

    // The registering matrix at full resolution, mapped to the network's space.
    let full: PixelMatrix = [[1.0, 0.0, 8.0], [0.0, 1.0, 4.0], [0.0, 0.0, 1.0]];
    let m_n = thumbs.full_to_normalized(full);
    let flat: Vec<f32> = m_n.iter().flatten().copied().collect();

    let dev = <B as Backend>::Device::default();
    let plane = |img: &PlanarImage<f32>| {
        Tensor::<B, 4>::from_data(TensorData::new(img.luma.clone(), [1, 1, 32, 512]), &dev)
    };
    let (ref_t, src_t) = (&thumbs.frames()[0], &thumbs.frames()[1]);
    let matrix = Tensor::<B, 3>::from_data(TensorData::new(flat, [1, 3, 3]), &dev);
    let out = warp_affine(plane(src_t), matrix);
    let (warped, valid) = (
        out.warped.into_data().to_vec::<f32>().unwrap(),
        out.valid.into_data().to_vec::<f32>().unwrap(),
    );
    let want = plane(ref_t).into_data().to_vec::<f32>().unwrap();

    let mut checked = 0;
    for (i, ((got, expected), ok)) in warped.iter().zip(&want).zip(&valid).enumerate() {
        if *ok > 0.5 {
            checked += 1;
            assert!(
                (got - expected).abs() < 1e-3,
                "thumbnail pixel {i}: warped {got}, reference {expected}"
            );
        }
    }
    // Everything but the 4 right-most columns and 2 bottom rows samples inside.
    assert_eq!(checked, (512 - 4) * (32 - 2));
}

// ---------------------------------------------------------------------------
// Alignment entry points
// ---------------------------------------------------------------------------

#[test]
fn untrained_networks_return_the_identity_at_full_resolution() {
    let dev = <B as Backend>::Device::default();

    // 600 x 40 is above the 512-px limit, so this goes through the thumbnail path.
    let frames = vec![
        textured(600, 40, 0, 0),
        textured(600, 40, 5, 1),
        textured(600, 40, -3, 2),
    ];
    let batch = BatchAlignNetConfig::from_size(ModelSize::Xs).init::<B>(&dev);
    let matrices = align_planar(&batch, &frames, &dev).unwrap();
    assert_eq!(matrices.len(), 3);
    for m in &matrices {
        assert_matrix(m, &IDENTITY, 1e-4, "batch");
    }

    let pair = FusionAlignNetConfig::from_size(ModelSize::Xs).init::<B>(&dev);
    let matrices = align_planar_pairwise(&pair, &frames, &dev).unwrap();
    assert_eq!(matrices.len(), 3);
    for m in &matrices {
        assert_matrix(m, &IDENTITY, 1e-4, "pair");
    }
}

#[test]
fn alignment_reports_bad_stacks() {
    let dev = <B as Backend>::Device::default();
    let batch = BatchAlignNetConfig::from_size(ModelSize::Xs).init::<B>(&dev);
    assert!(matches!(
        align_planar(&batch, &[], &dev),
        Err(BridgeError::EmptyStack)
    ));
    let empty = FrameThumbs::new(64, 48).unwrap();
    assert!(matches!(
        align_thumbs(&batch, &empty, &dev),
        Err(BridgeError::EmptyStack)
    ));
    let mismatched = vec![textured(64, 48, 0, 0), textured(48, 64, 0, 0)];
    assert!(matches!(
        align_planar(&batch, &mismatched, &dev),
        Err(BridgeError::ShapeMismatch { index: 1, .. })
    ));
}
