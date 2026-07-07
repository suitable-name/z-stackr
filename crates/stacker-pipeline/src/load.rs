/// Image loading and directory scanning utilities.
///
/// Converts raw [`DynamicImage`] values to the normalised `PlanarImage<f32>`
/// representation used throughout the pipeline, and collects sorted lists of
/// image paths from an input directory.
use image::{ColorType, DynamicImage, GenericImageView};
use stacker_core::{error::StackerError, image::PlanarImage};
use std::path::Path;

/// Convert a [`DynamicImage`] to a normalised `PlanarImage<f32>`.
///
/// # Colour model
///
/// The pipeline operates entirely in **gamma/sRGB space** — no transfer
/// function is applied on load or save.  The steps are:
///
/// 1. Normalise pixel samples to `[0, 1]` using the raw gamma-encoded values:
///    - 8-bit sources: divide by 255.
///    - 16-bit sources: divide by 65 535.
/// 2. Derive a YCbCr-style planar representation directly from the
///    gamma-encoded RGB (same Rec.601 coefficients as the reference):
///    - `luma`     = weighted sum of gamma R, G, B.
///    - `chroma_a` = Cb analogue in `[−0.5, 0.5]`.
///    - `chroma_b` = Cr analogue in `[−0.5, 0.5]`.
///
/// On output, `planar_to_gamma_rgb` inverts the YCbCr matrix to recover
/// gamma RGB, which is then quantised directly to u8 or u16 with no
/// additional transfer function.
pub fn dynamic_to_planar(img: &DynamicImage) -> PlanarImage<f32> {
    let (img_width, img_height) = img.dimensions();
    let mut planar = PlanarImage::new(img_width as usize, img_height as usize);

    // Detect whether the source image carries 16-bit depth so we can
    // normalise correctly and avoid discarding sub-8-bit precision.
    let is_16bit = matches!(
        img.color(),
        ColorType::L16 | ColorType::La16 | ColorType::Rgb16 | ColorType::Rgba16
    );

    if is_16bit {
        let rgb16 = img.to_rgb16();
        for (i, p) in rgb16.pixels().enumerate() {
            let val_r = f32::from(p[0]) / 65_535.0;
            let val_g = f32::from(p[1]) / 65_535.0;
            let val_b = f32::from(p[2]) / 65_535.0;
            planar.luma[i] = 0.299 * val_r + 0.587 * val_g + 0.114 * val_b;
            planar.chroma_a[i] = -0.168_74 * val_r - 0.331_26 * val_g + 0.5 * val_b;
            planar.chroma_b[i] = 0.5 * val_r - 0.418_688 * val_g - 0.081_312 * val_b;
        }
    } else {
        let rgb8 = img.to_rgb8();
        for (i, p) in rgb8.pixels().enumerate() {
            let val_r = f32::from(p[0]) / 255.0;
            let val_g = f32::from(p[1]) / 255.0;
            let val_b = f32::from(p[2]) / 255.0;
            planar.luma[i] = 0.299 * val_r + 0.587 * val_g + 0.114 * val_b;
            planar.chroma_a[i] = -0.168_74 * val_r - 0.331_26 * val_g + 0.5 * val_b;
            planar.chroma_b[i] = 0.5 * val_r - 0.418_688 * val_g - 0.081_312 * val_b;
        }
    }

    planar
}

/// Collect all image paths from `dir` (non-recursive, common extensions).
///
/// Standard formats (`png`/`jpg`/`jpeg`/`tif`/`tiff`) are always included.
/// RAW extensions (see [`stacker_core::io::RAW_EXTENSIONS`]) are only
/// included when this crate is built with the `raw` feature — without it,
/// RAW files in the input directory are silently skipped here rather than
/// collected and then failing later at decode time, matching how a
/// non-`raw` build has no way to do anything useful with them anyway.
///
/// # Errors
/// Returns [`StackerError`] if `dir` cannot be read or contains no images
/// with a recognised extension.
pub fn collect_image_paths(dir: &Path) -> Result<Vec<std::path::PathBuf>, StackerError> {
    let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(dir)?
        .filter_map(std::result::Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            let Some(ext) = p
                .extension()
                .and_then(|e| e.to_str())
                .map(str::to_ascii_lowercase)
            else {
                return false;
            };
            let is_standard = matches!(ext.as_str(), "png" | "jpg" | "jpeg" | "tif" | "tiff");
            let is_raw = cfg!(feature = "raw") && stacker_core::io::is_raw_extension(&ext);
            p.is_file() && (is_standard || is_raw)
        })
        .collect();
    paths.sort(); // deterministic order
    if paths.is_empty() {
        return Err(StackerError::AlignmentFailed(format!(
            "no images found in '{}'",
            dir.display()
        )));
    }
    Ok(paths)
}
