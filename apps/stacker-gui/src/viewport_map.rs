//! Geometry of the image viewport: where an image of a given size lands inside a
//! canvas under "contain" fitting, zoom and pan, and how a pointer position maps
//! back to a raster pixel.
//!
//! The canvas shows the image scaled by `base_scale × zoom`, where `base_scale`
//! is the largest factor that fits the whole image into the canvas, centred, then
//! shifted by the pan offset. Every value is in logical canvas pixels; the Slint
//! side divides its `length`s by `1px` before calling in.

/// Smallest zoom relative to the fit.
pub const ZOOM_MIN: f32 = 0.1;
/// Largest zoom relative to the fit, unless a true 1:1 view needs more
/// ([`Frame::zoom_limits`]).
pub const ZOOM_MAX: f32 = 32.0;
/// Zoom factor of one mouse-wheel notch.
pub const WHEEL_STEP: f32 = 1.1;

/// The zoom after `notches` wheel notches (positive = zoom in); fractional
/// notches from a touchpad scale smoothly.
#[must_use]
pub fn wheel_zoom(zoom: f32, notches: f32) -> f32 {
    zoom * WHEEL_STEP.powf(notches)
}

/// The fixed geometry of one viewport: canvas size and raster size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Frame {
    /// Canvas width in logical pixels.
    pub canvas_w: f32,
    /// Canvas height in logical pixels.
    pub canvas_h: f32,
    /// Raster width in pixels.
    pub image_w: u32,
    /// Raster height in pixels.
    pub image_h: u32,
}

impl Frame {
    /// The frame for a canvas and a raster.
    #[must_use]
    pub const fn new(canvas_w: f32, canvas_h: f32, image_w: u32, image_h: u32) -> Self {
        Self {
            canvas_w,
            canvas_h,
            image_w,
            image_h,
        }
    }

    #[expect(clippy::cast_precision_loss, reason = "image sizes are far below 2^24")]
    const fn image_size(self) -> (f32, f32) {
        (self.image_w as f32, self.image_h as f32)
    }

    /// The largest scale that fits the raster into the canvas. Zero or negative
    /// sizes give `0.0`.
    #[must_use]
    pub fn fit_scale(self) -> f32 {
        if self.image_w == 0 || self.image_h == 0 || self.canvas_w <= 0.0 || self.canvas_h <= 0.0 {
            return 0.0;
        }
        let (iw, ih) = self.image_size();
        (self.canvas_w / iw).min(self.canvas_h / ih)
    }

    /// The zoom at which one raster pixel is one canvas pixel, relative to the
    /// fit. `None` when the image or canvas is empty.
    #[must_use]
    pub fn one_to_one_zoom(self) -> Option<f32> {
        let base = self.fit_scale();
        (base > 0.0).then(|| 1.0 / base)
    }

    /// The zoom at which one raster pixel covers one PHYSICAL screen pixel:
    /// canvas lengths are logical, so on a display scaled by `scale_factor` the
    /// logical scale for a true 1:1 view is `1 / scale_factor`. A scale factor
    /// that is not positive counts as 1. `None` when the image or canvas is empty.
    #[must_use]
    pub fn physical_one_to_one_zoom(self, scale_factor: f32) -> Option<f32> {
        let factor = if scale_factor > 0.0 {
            scale_factor
        } else {
            1.0
        };
        self.one_to_one_zoom().map(|zoom| zoom / factor)
    }

    /// The `(min, max)` zoom the view may take. The maximum is [`ZOOM_MAX`], raised
    /// when a true 1:1 view of a very large raster needs more, so "100 %" is
    /// always reachable.
    #[must_use]
    pub fn zoom_limits(self, scale_factor: f32) -> (f32, f32) {
        let needed = self
            .physical_one_to_one_zoom(scale_factor)
            .map_or(0.0, |one| one * 4.0);
        (ZOOM_MIN, ZOOM_MAX.max(needed))
    }

    /// The magnification the user sees, in physical pixels per raster pixel
    /// (1.0 = "100 %"), for the toolbar read-out. 1.0 when nothing is displayed.
    #[must_use]
    pub fn physical_scale(self, view: View, scale_factor: f32) -> f32 {
        let scale = self.placement(view).scale;
        if scale > 0.0 {
            scale
                * if scale_factor > 0.0 {
                    scale_factor
                } else {
                    1.0
                }
        } else {
            1.0
        }
    }

    /// Where the image lands under `view`.
    #[must_use]
    pub fn placement(self, view: View) -> Placement {
        let scale = self.fit_scale() * view.zoom.max(0.0);
        let (iw, ih) = self.image_size();
        let (width, height) = (iw * scale, ih * scale);
        Placement {
            x: (self.canvas_w - width) / 2.0 + view.pan_x,
            y: (self.canvas_h - height) / 2.0 + view.pan_y,
            scale,
            width,
            height,
        }
    }

    /// The raster pixel under canvas position `(x, y)`, or `None` when the
    /// pointer is outside the displayed image or the image is not displayable.
    #[must_use]
    pub fn to_raster(self, view: View, x: f32, y: f32) -> Option<(u32, u32)> {
        let place = self.placement(view);
        if place.scale <= 0.0 {
            return None;
        }
        let col = (x - place.x) / place.scale;
        let row = (y - place.y) / place.scale;
        let (iw, ih) = self.image_size();
        if col < 0.0 || row < 0.0 || col >= iw || row >= ih {
            return None;
        }
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "both coordinates were range-checked above"
        )]
        Some((col as u32, row as u32))
    }

    /// The canvas position of the centre of raster pixel `(col, row)`.
    #[must_use]
    pub fn to_canvas(self, view: View, col: u32, row: u32) -> (f32, f32) {
        let place = self.placement(view);
        #[expect(
            clippy::cast_precision_loss,
            reason = "raster coordinates are far below 2^24"
        )]
        let (fc, fr) = (col as f32 + 0.5, row as f32 + 0.5);
        (place.x + fc * place.scale, place.y + fr * place.scale)
    }

    /// Clamp the pan so the image can never be pushed entirely out of view.
    ///
    /// On an axis where the displayed image is smaller than the canvas it stays
    /// centred; where it is larger it may be dragged until its far edge reaches
    /// the canvas edge.
    #[must_use]
    pub fn clamp_pan(self, view: View) -> View {
        let centred = self.placement(View {
            pan_x: 0.0,
            pan_y: 0.0,
            ..view
        });
        let limit = |canvas: f32, shown: f32| ((shown - canvas) / 2.0).max(0.0);
        let lx = limit(self.canvas_w, centred.width);
        let ly = limit(self.canvas_h, centred.height);
        View {
            zoom: view.zoom,
            pan_x: view.pan_x.clamp(-lx, lx),
            pan_y: view.pan_y.clamp(-ly, ly),
        }
    }

    /// Zoom about a canvas point: the raster position under `(x, y)` stays under
    /// the pointer after the zoom changes from `view.zoom` to `new_zoom`, as far
    /// as the pan clamp allows.
    #[must_use]
    pub fn zoom_about(self, view: View, new_zoom: f32, x: f32, y: f32) -> View {
        let before = self.placement(view);
        if before.scale <= 0.0 {
            return View {
                zoom: new_zoom,
                ..view
            };
        }
        // Raster coordinate under the pointer, in raster pixels (may be outside).
        let col = (x - before.x) / before.scale;
        let row = (y - before.y) / before.scale;
        let after = self.placement(View {
            zoom: new_zoom,
            pan_x: 0.0,
            pan_y: 0.0,
        });
        // Pan so that the same raster coordinate lands back on `(x, y)`.
        self.clamp_pan(View {
            zoom: new_zoom,
            pan_x: x - (after.x + col * after.scale),
            pan_y: y - (after.y + row * after.scale),
        })
    }
}

/// Placement of the image inside the canvas: top-left corner and the scale from
/// raster pixels to canvas pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Placement {
    /// Canvas x of the image's left edge.
    pub x: f32,
    /// Canvas y of the image's top edge.
    pub y: f32,
    /// Canvas pixels per raster pixel.
    pub scale: f32,
    /// Displayed width in canvas pixels.
    pub width: f32,
    /// Displayed height in canvas pixels.
    pub height: f32,
}

/// The viewport's variable state: zoom relative to the fit, and the pan offset.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct View {
    /// 1.0 shows the whole image ("fit"); larger values magnify.
    pub zoom: f32,
    /// Pan offset in canvas pixels, added to the centred position.
    pub pan_x: f32,
    /// Pan offset in canvas pixels, added to the centred position.
    pub pan_y: f32,
}

impl Default for View {
    fn default() -> Self {
        Self {
            zoom: 1.0,
            pan_x: 0.0,
            pan_y: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Frame, View, ZOOM_MAX, ZOOM_MIN, wheel_zoom};

    const EPS: f32 = 1e-4;
    const WIDE: Frame = Frame::new(800.0, 600.0, 1600, 400);

    #[test]
    fn fit_scale_is_the_limiting_axis() {
        assert!((WIDE.fit_scale() - 0.5).abs() < EPS);
        assert!((Frame::new(800.0, 600.0, 400, 1200).fit_scale() - 0.5).abs() < EPS);
        assert_eq!(Frame::new(800.0, 600.0, 0, 10).fit_scale(), 0.0);
        assert_eq!(Frame::new(0.0, 600.0, 10, 10).fit_scale(), 0.0);
    }

    #[test]
    fn fit_view_is_centred_and_letterboxed() {
        let p = WIDE.placement(View::default());
        assert!((p.scale - 0.5).abs() < EPS);
        assert!((p.width - 800.0).abs() < EPS);
        assert!((p.height - 200.0).abs() < EPS);
        assert!(p.x.abs() < EPS);
        assert!((p.y - 200.0).abs() < EPS);
    }

    #[test]
    fn pointer_maps_to_the_raster_pixel_under_it() {
        let view = View::default();
        // Centre of the canvas is the centre of the image.
        assert_eq!(WIDE.to_raster(view, 400.0, 300.0), Some((800, 200)));
        // Letterbox area is outside.
        assert_eq!(WIDE.to_raster(view, 400.0, 100.0), None);
        // Left edge of the image.
        assert_eq!(WIDE.to_raster(view, 0.0, 250.0), Some((0, 100)));
        // Just past the right edge.
        assert_eq!(WIDE.to_raster(view, 800.0, 250.0), None);
    }

    #[test]
    fn zoom_and_pan_change_the_mapping_consistently() {
        let view = View {
            zoom: 4.0,
            pan_x: 30.0,
            pan_y: -10.0,
        };
        let (cx, cy) = WIDE.to_canvas(view, 123, 45);
        assert_eq!(WIDE.to_raster(view, cx, cy), Some((123, 45)));
    }

    #[test]
    fn one_to_one_zoom_gives_unit_scale() {
        let z = WIDE.one_to_one_zoom().unwrap();
        let p = WIDE.placement(View {
            zoom: z,
            ..View::default()
        });
        assert!((p.scale - 1.0).abs() < EPS);
        assert_eq!(Frame::new(800.0, 600.0, 0, 400).one_to_one_zoom(), None);
    }

    #[test]
    fn one_to_one_accounts_for_the_display_scale_factor() {
        // At 150 % display scaling a raster pixel is 1 physical = 1/1.5 logical px.
        let z = WIDE.physical_one_to_one_zoom(1.5).unwrap();
        let view = View {
            zoom: z,
            ..View::default()
        };
        assert!((WIDE.placement(view).scale - 1.0 / 1.5).abs() < EPS);
        assert!((WIDE.physical_scale(view, 1.5) - 1.0).abs() < EPS);
        // A nonsense scale factor counts as 1.
        assert_eq!(
            WIDE.physical_one_to_one_zoom(0.0),
            WIDE.physical_one_to_one_zoom(1.0)
        );
        assert_eq!(
            Frame::new(800.0, 600.0, 0, 0).physical_one_to_one_zoom(1.0),
            None
        );
    }

    #[test]
    fn the_readout_is_one_without_an_image() {
        let empty = Frame::new(800.0, 600.0, 0, 0);
        assert!((empty.physical_scale(View::default(), 2.0) - 1.0).abs() < EPS);
    }

    #[test]
    fn zoom_limits_always_reach_true_one_to_one() {
        let (lo, hi) = WIDE.zoom_limits(1.0);
        assert!((lo - ZOOM_MIN).abs() < EPS);
        assert!((hi - ZOOM_MAX).abs() < EPS);
        // A 40000 px wide raster in an 800 px canvas needs a zoom of 50 for 1:1.
        let huge = Frame::new(800.0, 600.0, 40_000, 100);
        let one = huge.physical_one_to_one_zoom(1.0).unwrap();
        let (_, hi) = huge.zoom_limits(1.0);
        assert!(hi >= one, "{hi} must reach {one}");
    }

    #[test]
    fn one_wheel_notch_is_a_tenth_and_notches_compose() {
        assert!((wheel_zoom(1.0, 1.0) - 1.1).abs() < EPS);
        assert!((wheel_zoom(2.0, 0.0) - 2.0).abs() < EPS);
        assert!((wheel_zoom(wheel_zoom(1.0, 1.0), -1.0) - 1.0).abs() < EPS);
        assert!((wheel_zoom(1.0, 2.0) - wheel_zoom(wheel_zoom(1.0, 1.0), 1.0)).abs() < EPS);
    }

    #[test]
    fn pan_is_clamped_to_keep_the_image_in_view() {
        // Image smaller than the canvas: no pan at all.
        let v = WIDE.clamp_pan(View {
            zoom: 1.0,
            pan_x: 500.0,
            pan_y: 500.0,
        });
        assert_eq!((v.pan_x, v.pan_y), (0.0, 0.0));
        // Image twice the canvas width: may move by half its overhang.
        let v = WIDE.clamp_pan(View {
            zoom: 2.0,
            pan_x: 9999.0,
            pan_y: 9999.0,
        });
        assert!((v.pan_x - 400.0).abs() < EPS);
        assert_eq!(v.pan_y, 0.0);
    }

    #[test]
    fn zooming_about_a_point_keeps_that_pixel_under_the_pointer() {
        // A square image on a landscape canvas: at zoom 3 both axes overflow the
        // canvas, so the pan clamp does not bind for a pointer near the centre.
        let square = Frame::new(800.0, 600.0, 1000, 1000);
        let view = View::default();
        let (x, y) = (500.0, 400.0);
        let before = square.to_raster(view, x, y).unwrap();
        let zoomed = square.zoom_about(view, 3.0, x, y);
        let after = square.to_raster(zoomed, x, y).unwrap();
        assert_eq!(before, after);
        assert!((zoomed.zoom - 3.0).abs() < EPS);
    }

    #[test]
    fn zooming_about_a_point_respects_the_clamp_on_a_fitting_axis() {
        // Height still fits at zoom 2 (400 px of 600), so the image stays centred
        // vertically and only the horizontal position is preserved.
        let view = View::default();
        let (x, y) = (600.0, 250.0);
        let before = WIDE.to_raster(view, x, y).unwrap();
        let zoomed = WIDE.zoom_about(view, 2.0, x, y);
        let after = WIDE.to_raster(zoomed, x, y).unwrap();
        assert_eq!(before.0, after.0);
        assert_eq!(zoomed.pan_y, 0.0);
    }
}
