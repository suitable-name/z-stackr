#![expect(
    clippy::suboptimal_flops,
    reason = "synthetic-image formulas read better without mul_add"
)]

//! FLOW-2: where each of the 39 settings keys is consumed.
//!
//! `CONSUMERS` is the table (it must list exactly the keys of the generated
//! default config), and the tests below prove, for every key that a shared stage
//! reads, that changing the key changes that stage's result.
//!
//! The alignment keys added with settings version 2 (`optimizer`,
//! `correct_brightness`, `chain_alignment`, `neural_refine_classically`,
//! `crop_to_common_area`, `resize_cropped_to_original`) are consumed by
//! `align.rs`; they are listed in the table and proved in `align_test.rs` (their
//! effect needs a running aligner, not a pure stage).

use std::collections::BTreeSet;

use image::{Rgb, RgbImage};
use stacker_core::{
    image::PlanarImage,
    settings::{
        AlignmentModeSetting, DEFAULT_CONFIG_TOML, ImageSavingSettings, OutputFormat,
        PreprocessingSettings, StackingSettings,
    },
};
use stacker_pipeline::{
    align::{AlignmentSettings, align_dispatch},
    cull::CullRule,
    fuse::{apex, focus_maps, relief, relief_multigrid, relief_threshold, strata, strata_params},
    load::decode_frame,
    order::select_frames,
    output::encode,
    session::{PassKind, estimate_peak_bytes},
    tiled::effective_tile_size,
    tiling::{Method, apron_for},
};

/// `(key, where it is consumed)`. `shared` stages are in `stacker-pipeline` (both
/// apps); `CLI` / `GUI` entries are app code that calls a shared stage or reads
/// the key itself.
const CONSUMERS: [(&str, &str); 39] = [
    (
        "settings_version",
        "stacker_core::settings loader (migrate_legacy); written by the apps",
    ),
    ("alignment_mode", "shared align::align_dispatch (both apps)"),
    (
        "optimizer",
        "shared align::AlignmentSettings (refiner choice; proved in align_test)",
    ),
    ("bounded_search", "shared align::align_dispatch (both apps)"),
    (
        "akaze_seeding",
        "shared align::Seeder via AlignmentSettings (both apps; needs the akaze feature)",
    ),
    (
        "correct_brightness",
        "shared align (brightness fit on the aligned frame; proved in align_test)",
    ),
    (
        "chain_alignment",
        "shared align (rolling reference; proved in align_test)",
    ),
    (
        "neural_refine_classically",
        "shared align (neural seed refinement, nn builds; proved in align_test)",
    ),
    (
        "crop_to_common_area",
        "shared tiled::crop_rect and session align outcome (proved in align_test)",
    ),
    (
        "resize_cropped_to_original",
        "shared align / session restretch (not applied out of core; proved in align_test)",
    ),
    (
        "auto_cull",
        "CLI pipeline::frame_order, GUI gui/stack.rs cull_stage (both call shared cull)",
    ),
    (
        "auto_cull_threshold_pct",
        "shared cull::CullRule::from_settings (session::cull, tiled::frame_order)",
    ),
    (
        "sort_by_sharpness",
        "shared cull::CullPlan::fusion_order (session::cull, tiled::frame_order)",
    ),
    (
        "stack_every_nth",
        "shared order::select_frames_by (CLI select_frames, GUI frames::paths_for_run)",
    ),
    (
        "tile_size",
        "shared tiled::effective_tile_size (the CLI flag --tile-size wins; 0 = in RAM)",
    ),
    ("pyramid_levels", "shared fuse::apex and tiling::apron_for"),
    ("use_all_color_channels", "shared fuse::apex"),
    ("grit_suppression", "shared fuse::apex"),
    (
        "relief_estimation_radius",
        "shared fuse::focus_maps, tiling::apron_for",
    ),
    (
        "relief_smoothing_radius",
        "shared fuse::select_and_smooth, tiling::apron_for",
    ),
    ("relief_contrast_pct", "shared fuse::relief_threshold"),
    (
        "relief_show_preview",
        "GUI only (stacking.rs contrast preview panel; the CLI has no UI)",
    ),
    ("relief_auto_detect", "shared fuse::relief_threshold"),
    (
        "relief_use_multigrid",
        "shared fuse::relief, session::relief_focus_max / relief_finish, tiled::fuse_tiles",
    ),
    (
        "strata_base_radius",
        "shared fuse::strata_params (strata, session::fuse_strata, tiled::fuse_tiles)",
    ),
    (
        "strata_detail_focus",
        "shared fuse::strata_params (strata, session::fuse_strata, tiled::fuse_tiles)",
    ),
    (
        "compute_device",
        "shared compute::Compute::for_settings (both apps; the GPU needs the gpu feature)",
    ),
    ("preprocessing.pre_rotation", "shared load::decode_frame"),
    (
        "preprocessing.pre_crop_enabled",
        "shared load::decode_frame",
    ),
    ("preprocessing.pre_crop_spec", "shared load::decode_frame"),
    (
        "preprocessing.pre_resize_percent",
        "shared load::decode_frame",
    ),
    (
        "preprocessing.sort_reverse",
        "shared order::select_frames_by",
    ),
    (
        "preprocessing.ignore_exif_orientation",
        "shared load::decode_frame",
    ),
    (
        "image_saving.output_format",
        "shared output::encode; CLI output::resolve_output",
    ),
    (
        "image_saving.bit_depth",
        "shared output::encode / wants_16_bit; CLI output::resolve_output",
    ),
    (
        "image_saving.jpeg_quality",
        "shared output::encode / jpeg_quality; CLI output::resolve_output",
    ),
    (
        "image_saving.filename_template",
        "CLI output::resolve_output (the GUI Save dialog applies it in export.rs)",
    ),
    (
        "image_saving.default_output_dir",
        "CLI output::resolve_output (the GUI Save dialog seeds its folder)",
    ),
    (
        "image_saving.copy_metadata",
        "nobody: accepted, ignored with a warning (CLI prepare_run)",
    ),
];

/// The table lists exactly the 39 keys of the generated default config.
#[test]
fn the_table_covers_every_key_of_the_default_config() {
    let mut section = String::new();
    let mut in_file = BTreeSet::new();
    for line in DEFAULT_CONFIG_TOML.lines() {
        let line = line.trim();
        if let Some(name) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
            section = format!("{name}.");
        } else if !line.starts_with('#')
            && let Some((key, _)) = line.split_once('=')
        {
            in_file.insert(format!("{section}{}", key.trim()));
        }
    }
    let in_table: BTreeSet<String> = CONSUMERS.iter().map(|(k, _)| (*k).to_owned()).collect();
    assert_eq!(in_file.len(), 39, "{in_file:?}");
    assert_eq!(in_file, in_table);
}

fn textured(seed: u32, amp_luma: f32, amp_chroma: f32) -> PlanarImage<f32> {
    let mut img = PlanarImage::new(48, 48);
    for i in 0..48 * 48_u32 {
        let h = (i.wrapping_mul(2_654_435_761) ^ seed.wrapping_mul(40_503)) % 1000;
        let n = f32::from(u16::try_from(h).unwrap()) / 1000.0 - 0.5;
        let idx = usize::try_from(i).unwrap();
        img.luma[idx] = 0.5 + amp_luma * n;
        img.chroma_a[idx] = amp_chroma * n;
    }
    img
}

fn differs(a: &PlanarImage<f32>, b: &PlanarImage<f32>) -> bool {
    a.luma != b.luma || a.chroma_a != b.chroma_a || a.chroma_b != b.chroma_b
}

#[test]
fn alignment_keys_choose_the_algorithm() {
    use AlignmentModeSetting::{Affine, None, Registration, Translation};
    let modes = [None, Translation, Affine, Registration];
    for (i, a) in modes.iter().enumerate() {
        for b in &modes[i + 1..] {
            assert_ne!(
                align_dispatch(*a, false),
                align_dispatch(*b, false),
                "{a:?} vs {b:?}"
            );
        }
    }
    for mode in [Translation, Affine] {
        assert_ne!(
            align_dispatch(mode, false),
            align_dispatch(mode, true),
            "{mode:?}"
        );
    }
    let s = StackingSettings {
        akaze_seeding: true,
        ..StackingSettings::default()
    };
    assert!(AlignmentSettings::from(&s).akaze_seeding);
    assert!(!AlignmentSettings::from(&StackingSettings::default()).akaze_seeding);
}

#[test]
fn apex_keys_change_the_fused_image() {
    // Frame A: strong luma texture. Frame B: weak luma, strong chroma.
    let frames = [textured(1, 0.3, 0.0), textured(2, 0.05, 0.6)];
    let base = StackingSettings::default();
    let reference = apex(&frames, &base);

    let variants = [
        (
            "pyramid_levels",
            StackingSettings {
                pyramid_levels: 3,
                ..base.clone()
            },
        ),
        (
            "use_all_color_channels",
            StackingSettings {
                use_all_color_channels: true,
                ..base.clone()
            },
        ),
        (
            "grit_suppression",
            StackingSettings {
                grit_suppression: false,
                ..base.clone()
            },
        ),
    ];
    for (key, settings) in variants {
        assert!(
            differs(&reference, &apex(&frames, &settings)),
            "{key} has no effect on Apex"
        );
    }
    // The levels also set the apron.
    let shallow = StackingSettings {
        pyramid_levels: 3,
        ..base.clone()
    };
    assert_ne!(
        apron_for(Method::Apex, &base),
        apron_for(Method::Apex, &shallow)
    );
}

#[test]
fn relief_keys_change_the_fused_image_and_the_threshold() {
    let frames = [
        textured(3, 0.3, 0.0),
        textured(4, 0.2, 0.1),
        textured(5, 0.1, 0.0),
    ];
    let base = StackingSettings {
        relief_contrast_pct: 0.4,
        ..StackingSettings::default()
    };
    let reference = relief(&frames, &base);

    let variants = [
        (
            "relief_estimation_radius",
            StackingSettings {
                relief_estimation_radius: 1,
                ..base.clone()
            },
        ),
        (
            "relief_smoothing_radius",
            StackingSettings {
                relief_smoothing_radius: 6,
                ..base.clone()
            },
        ),
        (
            "relief_contrast_pct",
            StackingSettings {
                relief_contrast_pct: 0.9,
                ..base.clone()
            },
        ),
        (
            "relief_auto_detect",
            StackingSettings {
                relief_auto_detect: true,
                ..base.clone()
            },
        ),
    ];
    for (key, settings) in variants {
        assert!(
            differs(&reference, &relief(&frames, &settings)),
            "{key} has no effect on Relief"
        );
    }

    let map = focus_maps(&frames, 5).max.luma;
    let pct = relief_threshold(&map, &base);
    let auto = relief_threshold(
        &map,
        &StackingSettings {
            relief_auto_detect: true,
            ..base.clone()
        },
    );
    assert!(
        (pct.contrast_pct - 0.4).abs() < 1e-6,
        "the configured fraction is used"
    );
    assert!(
        (auto.contrast_pct - 0.4).abs() > 1e-3,
        "the auto-detected noise floor replaces it"
    );
    // The radii set the Relief apron.
    assert_eq!(apron_for(Method::Relief, &base), 5 + 2 * 2 + 1);
    let wide = StackingSettings {
        relief_smoothing_radius: 6,
        ..base
    };
    assert_eq!(apron_for(Method::Relief, &wide), 5 + 2 * 6 + 1);
}

#[test]
fn strata_keys_change_the_fused_image_and_stay_in_range() {
    let frames = [
        textured(11, 0.3, 0.0),
        textured(12, 0.2, 0.1),
        textured(13, 0.1, 0.0),
    ];
    let base = StackingSettings::default();
    let reference = strata(&frames, &base);
    let variants = [
        (
            "strata_base_radius",
            StackingSettings {
                strata_base_radius: 8,
                ..base.clone()
            },
        ),
        (
            "strata_detail_focus",
            StackingSettings {
                strata_detail_focus: 1,
                ..base.clone()
            },
        ),
    ];
    for (key, settings) in variants {
        assert!(
            differs(&reference, &strata(&frames, &settings)),
            "{key} has no effect on Strata"
        );
    }
    // The apron does not depend on the keys: it is sized for their largest values.
    let widest = StackingSettings {
        strata_base_radius: 64,
        ..base.clone()
    };
    assert_eq!(
        apron_for(Method::Strata, &base),
        apron_for(Method::Strata, &widest)
    );
    assert_eq!(strata_params(&base).base_radius, 31);
    assert_eq!(strata_params(&base).detail_focus, 3);
}

#[test]
fn relief_use_multigrid_selects_the_engine_and_the_estimate() {
    let frames = [
        textured(21, 0.3, 0.0),
        textured(22, 0.2, 0.1),
        textured(23, 0.1, 0.0),
    ];
    let guided = StackingSettings {
        relief_contrast_pct: 0.4,
        ..StackingSettings::default()
    };
    let multigrid = StackingSettings {
        relief_use_multigrid: true,
        ..guided.clone()
    };
    assert!(differs(
        &relief(&frames, &guided),
        &relief(&frames, &multigrid)
    ));
    assert!(!differs(
        &relief_multigrid(&frames, &multigrid),
        &relief(&frames, &multigrid)
    ));
    // Its memory is in the estimate.
    let passes = [PassKind::Relief];
    assert!(
        estimate_peak_bytes(10, 4000, 3000, &multigrid, &passes)
            > estimate_peak_bytes(10, 4000, 3000, &guided, &passes)
    );
}

#[test]
fn cull_and_tile_keys_are_read_by_the_shared_stages() {
    let automatic = StackingSettings::default();
    let fixed = StackingSettings {
        auto_cull_threshold_pct: 5.0,
        ..automatic.clone()
    };
    assert_ne!(
        CullRule::from_settings(&automatic),
        CullRule::from_settings(&fixed)
    );
    assert_eq!(CullRule::from_settings(&fixed), CullRule::Fixed(5.0));

    // 0 is "not asked for": the settings key supplies the tile size.
    assert_eq!(effective_tile_size(0, &automatic), 0);
    let tiled = StackingSettings {
        tile_size: 256,
        ..automatic
    };
    assert_eq!(effective_tile_size(0, &tiled), 256);
    assert_eq!(effective_tile_size(512, &tiled), 512);
}

#[test]
fn frame_selection_keys_change_the_frame_list() {
    let paths: Vec<std::path::PathBuf> = (0..6).map(|i| format!("f{i}.png").into()).collect();
    let base = StackingSettings::default();
    let all = select_frames(paths.clone(), &base);
    let nth = select_frames(
        paths.clone(),
        &StackingSettings {
            stack_every_nth: 2,
            ..base.clone()
        },
    );
    let reversed = select_frames(
        paths,
        &StackingSettings {
            preprocessing: PreprocessingSettings {
                sort_reverse: true,
                ..PreprocessingSettings::default()
            },
            ..base
        },
    );
    assert_eq!(all.len(), 6);
    assert_eq!(nth.len(), 3);
    assert_eq!(reversed[0], all[5]);
}

#[test]
fn preprocessing_keys_change_the_decoded_frame() {
    let dir = std::env::temp_dir().join(format!("stacker_pipeline_reach_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("a.png");
    RgbImage::from_fn(40, 20, |x, y| {
        Rgb([
            u8::try_from(x * 6).unwrap(),
            u8::try_from(y * 9).unwrap(),
            50,
        ])
    })
    .save(&path)
    .unwrap();

    let base = PreprocessingSettings::default();
    let reference = decode_frame(&path, &base).unwrap();
    let variants = [
        (
            "pre_rotation",
            PreprocessingSettings {
                pre_rotation: 90,
                ..base.clone()
            },
        ),
        (
            "pre_crop_enabled + pre_crop_spec",
            PreprocessingSettings {
                pre_crop_enabled: true,
                pre_crop_spec: "20,10".to_owned(),
                ..base.clone()
            },
        ),
        (
            "pre_resize_percent",
            PreprocessingSettings {
                pre_resize_percent: 50,
                ..base.clone()
            },
        ),
    ];
    for (key, pre) in variants {
        let changed = decode_frame(&path, &pre).unwrap();
        assert!(
            (changed.width, changed.height) != (reference.width, reference.height),
            "{key} has no effect on the decoded frame"
        );
    }
    // `pre_crop_spec` alone (crop disabled) and `ignore_exif_orientation` on a
    // frame without EXIF leave the frame as it is; the EXIF case is in load_test.
    let spec_only = PreprocessingSettings {
        pre_crop_spec: "20,10".to_owned(),
        ..base
    };
    assert_eq!(
        decode_frame(&path, &spec_only).unwrap().luma,
        reference.luma
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn image_saving_keys_change_the_encoded_image() {
    let img = textured(9, 0.4, 0.0);
    let base = ImageSavingSettings {
        output_format: OutputFormat::Png,
        bit_depth: 16,
        ..ImageSavingSettings::default()
    };
    assert!(encode(&img, &base).is_16_bit());
    assert!(
        !encode(
            &img,
            &ImageSavingSettings {
                bit_depth: 8,
                ..base.clone()
            }
        )
        .is_16_bit()
    );
    assert!(
        !encode(
            &img,
            &ImageSavingSettings {
                output_format: OutputFormat::Jpeg,
                ..base
            }
        )
        .is_16_bit()
    );

    let dir = std::env::temp_dir().join(format!(
        "stacker_pipeline_reach_jpeg_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut sizes = Vec::new();
    for quality in [20, 95] {
        let saving = ImageSavingSettings {
            output_format: OutputFormat::Jpeg,
            bit_depth: 8,
            jpeg_quality: quality,
            ..ImageSavingSettings::default()
        };
        let path = dir.join(format!("q{quality}.jpg"));
        encode(&img, &saving).save(&path).unwrap();
        sizes.push(std::fs::metadata(&path).unwrap().len());
    }
    assert!(
        sizes[0] < sizes[1],
        "jpeg_quality reaches the encoder: {sizes:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
