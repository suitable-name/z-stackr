use image::{DynamicImage, GenericImageView};

use crate::settings;

// ── Frame preprocessing ───────────────────────────────────────────────────────

/// Parse a crop specification string into `(x, y, w, h)` clamped to image bounds.
///
/// Accepts two forms:
/// - `"w,h"` — centred crop of that size.
/// - `"x,y,w,h"` — explicit offset crop.
///
/// Returns `None` on any parse error, empty dimensions, or if the crop is
/// entirely outside the image bounds (logs a warning in all error cases).
pub fn parse_crop_spec(spec: &str, img_w: u32, img_h: u32) -> Option<(u32, u32, u32, u32)> {
    let parts: Vec<&str> = spec.split(',').collect();
    let (cx, cy, cw, ch) = match parts.len() {
        2 => {
            let w: u32 = parts[0].trim().parse().ok()?;
            let h: u32 = parts[1].trim().parse().ok()?;
            // centred crop
            let x = img_w.saturating_sub(w) / 2;
            let y = img_h.saturating_sub(h) / 2;
            (x, y, w, h)
        }
        4 => {
            let x: u32 = parts[0].trim().parse().ok()?;
            let y: u32 = parts[1].trim().parse().ok()?;
            let w: u32 = parts[2].trim().parse().ok()?;
            let h: u32 = parts[3].trim().parse().ok()?;
            (x, y, w, h)
        }
        _ => {
            tracing::warn!(spec, "pre_crop_spec: expected 'w,h' or 'x,y,w,h'");
            return None;
        }
    };

    if cw == 0 || ch == 0 {
        tracing::warn!(spec, "pre_crop_spec: zero dimension — skipping crop");
        return None;
    }
    if cx >= img_w || cy >= img_h {
        tracing::warn!(
            spec,
            cx,
            cy,
            img_w,
            img_h,
            "pre_crop_spec: origin outside image — skipping crop"
        );
        return None;
    }

    // Clamp width/height so the rectangle stays within the image.
    let cw = cw.min(img_w.saturating_sub(cx));
    let ch = ch.min(img_h.saturating_sub(cy));

    Some((cx, cy, cw, ch))
}

/// Apply preprocessing transformations to a [`DynamicImage`] in-place
/// (returning a new image).
///
/// Order: rotation → crop → resize.
///
/// # Invariants
/// - `pre_rotation = 0`, `pre_crop_enabled = false`, `pre_resize_percent = 100`
///   all produce the identity transform — the output equals the input exactly.
/// - `ignore_exif_orientation`: the `image` crate's `open` does **not**
///   auto-apply EXIF orientation, so the current pipeline already ignores it.
///   Wired conservatively: `true` (ignore) = current behaviour (no-op).
///   `false` (apply) would require `image::ImageReader` orientation support
///   which is not exposed in the version used here — stored-only, documented
///   no-op.  A comment in the caller marks this.
pub fn preprocess_frame(img: DynamicImage, pre: &settings::PreprocessingSettings) -> DynamicImage {
    // ── 1. Rotation ───────────────────────────────────────────────────────────
    // pre_rotation = 0 → identity (no call, no cost).
    let img = match pre.pre_rotation {
        90 => img.rotate90(),
        180 => img.rotate180(),
        270 => img.rotate270(),
        _ => img, // 0 or any clamped-to-0 value
    };

    // ── 2. Crop ───────────────────────────────────────────────────────────────
    // pre_crop_enabled = false → identity.
    let img = if pre.pre_crop_enabled && !pre.pre_crop_spec.is_empty() {
        let (iw, ih) = img.dimensions();
        if let Some((x, y, w, h)) = parse_crop_spec(&pre.pre_crop_spec, iw, ih) {
            img.crop_imm(x, y, w, h)
        } else {
            tracing::warn!(
                spec = pre.pre_crop_spec.as_str(),
                "crop spec invalid or out-of-bounds — skipping crop"
            );
            img
        }
    } else {
        img
    };

    // ── 3. Resize ─────────────────────────────────────────────────────────────
    // pre_resize_percent = 100 → identity (no call, no cost).
    if pre.pre_resize_percent < 100 {
        let (iw, ih) = img.dimensions();
        let pct = pre.pre_resize_percent as f32 / 100.0;
        // Round to nearest pixel; clamp to at least 1 in each dimension.
        let new_w = ((iw as f32 * pct).round() as u32).max(1);
        let new_h = ((ih as f32 * pct).round() as u32).max(1);
        img.resize(new_w, new_h, image::imageops::FilterType::Lanczos3)
    } else {
        img
    }
}
