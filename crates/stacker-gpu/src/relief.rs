//! `Relief` fusion on the device.
//!
//! A [`ReliefPlan`] is built for one frame size. Frames are folded one at a time:
//! each is uploaded, its focus measure is computed (two separable passes) and
//! folded into the best frame per pixel and a running sum, so device memory does
//! not grow with the number of frames. After the last frame the per-pixel best
//! focus (the maximum focus map) can be read back to compute the threshold on the
//! host, and [`ReliefPlan::finish`] selects (best frame above the threshold, mean
//! below), smooths the luma with the self-guided filter and reads the result
//! back.
//!
//! The focus measure, the tie rule (the first frame wins), the mask (`>=` the
//! threshold), the mean and the guided filter are those of `stacker_algo::relief`
//! and `stacker_pipeline::fuse`. The CPU evaluates the window sums in `f64`; the
//! device in `f32` on shifted values (see `relief.wgsl`), so where two frames'
//! focus values are within rounding of each other the winner can differ.

use std::sync::Arc;

use bytemuck::cast_slice;
use stacker_core::image::PlanarImage;

use crate::{
    GpuError,
    common::{self, u32_of},
    context::{Failure, GpuContext},
};

const SHADER: &str = include_str!("relief.wgsl");

/// Threads per workgroup edge; the shader declares 16 x 16 x 1.
const GROUP: u32 = 16;
/// The largest focus radius the integer window geometry is exact for (the
/// settings allow 0..=100).
pub const MAX_RADIUS: usize = 100;
/// Bytes of the shader's `Params`.
const PARAMS_BYTES: u64 = 32;

/// The WGSL source of the `Relief` kernels.
#[must_use]
pub fn shader_source() -> String {
    SHADER.to_owned()
}

#[derive(Clone, Copy)]
enum Kernel {
    FocusH,
    FocusVFold,
    Select,
    BoxHI,
    BoxHAb,
    GuidedAb,
    GuidedQ,
}

const KERNELS: [&str; 7] = [
    "focus_h",
    "focus_v_fold",
    "select_frame",
    "box_h_i",
    "box_h_ab",
    "guided_ab",
    "guided_q",
];

/// The device memory a plan for `width x height` frames needs, in bytes.
///
/// The frame, the window sums, the best focus, the best frame, the running sum
/// and the read-back buffer: 13 values per pixel and one image.
#[must_use]
pub const fn plan_bytes(width: usize, height: usize) -> u64 {
    let n = (width * height) as u64;
    (3 + 3 + 1 + 3 + 3 + 3) * 4 * n
}

/// The uniform block of `relief.wgsl`.
#[derive(Clone, Copy, Default)]
struct Params {
    radius: u32,
    n_frames: u32,
    first: bool,
    shift: f32,
    threshold: f32,
    eps: f32,
}

/// `Relief` fusion of a stack of equally sized frames on the device.
pub struct ReliefPlan {
    ctx: Arc<GpuContext>,
    pipelines: Vec<wgpu::ComputePipeline>,
    groups: Vec<wgpu::BindGroup>,
    params: wgpu::Buffer,
    frame: wgpu::Buffer,
    best_focus: wgpu::Buffer,
    sum_lab: wgpu::Buffer,
    staging: wgpu::Buffer,
    width: usize,
    height: usize,
    radius: usize,
    n_frames: u32,
    folded: u32,
    /// The last frame's submission, still running while the next is staged.
    in_flight: Option<wgpu::SubmissionIndex>,
}

impl ReliefPlan {
    /// Build a plan for `width x height` frames.
    ///
    /// # Errors
    /// [`GpuError::TooLarge`] when a buffer would exceed what the device allows,
    /// [`GpuError::Device`] when the device reported an error.
    ///
    /// # Panics
    /// Panics when a dimension exceeds `u32::MAX`.
    pub fn new(ctx: &Arc<GpuContext>, width: usize, height: usize) -> Result<Self, GpuError> {
        if ctx.injected(Failure::Plan) {
            return Err(GpuError::Injected("building a plan"));
        }
        let n = (width * height) as u64;
        let plane = 4 * n;
        let frame = common::storage(ctx, "relief frame", 3 * plane)?;
        let tmp = common::storage(ctx, "relief window sums", 3 * plane)?;
        let best_focus = common::storage(ctx, "relief best focus", plane)?;
        let best_lab = common::storage(ctx, "relief best frame", 3 * plane)?;
        let sum_lab = common::storage(ctx, "relief sum", 3 * plane)?;
        // The selected image is written into the frame buffer, which is free once
        // every frame has been folded.
        let selected = &frame;
        let staging = common::readback(ctx, "relief read back", 3 * plane);
        let params = common::uniform(ctx, "relief params", PARAMS_BYTES);
        let pipelines = common::pipelines(ctx, "relief", shader_source(), &KERNELS);
        let bind = |kernel: Kernel, buffers: &[(u32, &wgpu::Buffer)]| {
            let mut all = vec![(0, &params)];
            all.extend_from_slice(buffers);
            common::bind(ctx, &pipelines[kernel as usize], &all)
        };
        let groups = vec![
            bind(Kernel::FocusH, &[(1, &frame), (2, &tmp)]),
            bind(
                Kernel::FocusVFold,
                &[
                    (1, &frame),
                    (2, &tmp),
                    (3, &best_focus),
                    (4, &best_lab),
                    (5, &sum_lab),
                ],
            ),
            bind(
                Kernel::Select,
                &[
                    (3, &best_focus),
                    (4, &best_lab),
                    (5, &sum_lab),
                    (6, selected),
                ],
            ),
            bind(Kernel::BoxHI, &[(2, &tmp), (6, selected)]),
            bind(Kernel::BoxHAb, &[(2, &tmp), (5, &sum_lab)]),
            bind(Kernel::GuidedAb, &[(2, &tmp), (5, &sum_lab)]),
            bind(Kernel::GuidedQ, &[(2, &tmp), (5, &sum_lab), (6, selected)]),
        ];
        let _ = ctx.device.poll(wgpu::Maintain::Wait);
        common::check(ctx)?;
        Ok(Self {
            ctx: Arc::clone(ctx),
            pipelines,
            groups,
            params,
            frame,
            best_focus,
            sum_lab,
            staging,
            width,
            height,
            radius: 0,
            n_frames: 0,
            folded: 0,
            in_flight: None,
        })
    }

    /// Start a stack of `n_frames` frames with focus window half-size
    /// `estimation_radius`.
    ///
    /// # Errors
    /// [`GpuError::TooLarge`] when the radius exceeds [`MAX_RADIUS`].
    pub fn begin(&mut self, n_frames: u32, estimation_radius: usize) -> Result<(), GpuError> {
        if estimation_radius > MAX_RADIUS {
            return Err(GpuError::TooLarge {
                needed: estimation_radius as u64,
                limit: MAX_RADIUS as u64,
            });
        }
        // As on the CPU, a window larger than the image is the whole image.
        self.radius = estimation_radius.min(self.width.max(self.height));
        self.n_frames = n_frames;
        self.folded = 0;
        Ok(())
    }

    fn write_params(&self, params: Params) {
        let block: [u32; 8] = [
            u32_of(self.width),
            u32_of(self.height),
            params.radius,
            params.n_frames,
            u32::from(params.first),
            params.shift.to_bits(),
            params.threshold.to_bits(),
            params.eps.to_bits(),
        ];
        self.ctx
            .queue
            .write_buffer(&self.params, 0, cast_slice(&block));
    }

    fn grid(&self) -> [u32; 3] {
        [
            common::groups(self.width, GROUP),
            common::groups(self.height, GROUP),
            1,
        ]
    }

    fn record(&self, encoder: &mut wgpu::CommandEncoder, kernel: Kernel) {
        common::dispatch(
            encoder,
            &self.pipelines[kernel as usize],
            &self.groups[kernel as usize],
            self.grid(),
        );
    }

    /// Fold `frame` into the best frame per pixel and the running sum.
    ///
    /// # Errors
    /// [`GpuError::SizeMismatch`] when the frame is not the plan's size,
    /// [`GpuError::Device`] when the device reported an error.
    pub fn fold(&mut self, frame: &PlanarImage<f32>) -> Result<(), GpuError> {
        if frame.width != self.width || frame.height != self.height {
            return Err(GpuError::SizeMismatch {
                expected: (self.width, self.height),
                got: (frame.width, frame.height),
            });
        }
        if self.ctx.injected(Failure::Fold) {
            return Err(GpuError::Injected("folding a frame"));
        }
        let plane = (self.width * self.height) as u64 * 4;
        let queue = &self.ctx.queue;
        queue.write_buffer(&self.frame, 0, cast_slice(&frame.luma));
        queue.write_buffer(&self.frame, plane, cast_slice(&frame.chroma_a));
        queue.write_buffer(&self.frame, 2 * plane, cast_slice(&frame.chroma_b));
        self.write_params(Params {
            radius: u32_of(self.radius),
            n_frames: self.n_frames,
            first: self.folded == 0,
            shift: mean(&frame.luma),
            ..Params::default()
        });
        let mut encoder = common::encoder(&self.ctx, "relief frame");
        self.record(&mut encoder, Kernel::FocusH);
        self.record(&mut encoder, Kernel::FocusVFold);
        let index = queue.submit(Some(encoder.finish()));
        // The queue orders the next frame's upload after this frame's passes, so
        // there is nothing to wait for; waiting for the previous frame only keeps
        // one frame's staging memory in flight while the host stages the next.
        if let Some(previous) = self.in_flight.replace(index) {
            let _ = self
                .ctx
                .device
                .poll(wgpu::Maintain::WaitForSubmissionIndex(previous));
        }
        self.folded += 1;
        common::check(&self.ctx)
    }

    /// The best focus any frame reached at each pixel (the maximum focus map),
    /// `width * height` values.
    ///
    /// # Errors
    /// [`GpuError::ReadBack`] or [`GpuError::Device`] when the device failed.
    pub fn focus_max(&mut self) -> Result<Vec<f32>, GpuError> {
        if self.ctx.injected(Failure::ReadBack) {
            return Err(GpuError::Injected("reading the focus map back"));
        }
        let bytes = (self.width * self.height) as u64 * 4;
        common::read(
            &self.ctx,
            common::encoder(&self.ctx, "relief focus read back"),
            &self.best_focus,
            &self.staging,
            bytes,
        )
    }

    /// Select against `threshold` (pixels whose best focus is `>=` it take the
    /// best frame, the others the mean of the frames), smooth the luma with the
    /// self-guided filter of radius `smoothing_radius` and regularisation `eps`,
    /// and read the result back.
    ///
    /// # Errors
    /// [`GpuError::ReadBack`] or [`GpuError::Device`] when the device failed.
    ///
    /// # Panics
    /// Panics when the read-back planes do not have the frame's size (never).
    pub fn finish(
        &mut self,
        threshold: f32,
        smoothing_radius: usize,
        eps: f32,
    ) -> Result<PlanarImage<f32>, GpuError> {
        if self.ctx.injected(Failure::ReadBack) {
            return Err(GpuError::Injected("reading the result back"));
        }
        let n = self.width * self.height;
        let plane = n as u64 * 4;
        self.write_params(Params {
            radius: u32_of(smoothing_radius.min(self.width.max(self.height))),
            n_frames: self.n_frames,
            threshold,
            eps,
            ..Params::default()
        });
        let mut encoder = common::encoder(&self.ctx, "relief finish");
        for kernel in [
            Kernel::Select,
            Kernel::BoxHI,
            Kernel::GuidedAb,
            Kernel::BoxHAb,
            Kernel::GuidedQ,
        ] {
            self.record(&mut encoder, kernel);
        }
        // The smoothed luma is plane 2 of the sum buffer, the chroma planes 1 and
        // 2 of the selected image (the frame buffer); gather them into the frame
        // buffer's luma plane and read the three planes back.
        encoder.copy_buffer_to_buffer(&self.sum_lab, 2 * plane, &self.frame, 0, plane);
        let planes: Vec<f32> =
            common::read(&self.ctx, encoder, &self.frame, &self.staging, 3 * plane)?;
        Ok(PlanarImage::from_planes(
            self.width,
            self.height,
            planes[..n].to_vec(),
            planes[n..2 * n].to_vec(),
            planes[2 * n..].to_vec(),
        )
        .expect("each plane holds width * height samples"))
    }
}

/// The shift the focus passes subtract: the mean of about 64k evenly spaced
/// samples of the plane. Any constant leaves the measure unchanged; one near the
/// typical value keeps the `f32` window sums small.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "a shift needs no more than f32 precision; sample counts are far below 2^52"
)]
fn mean(plane: &[f32]) -> f32 {
    if plane.is_empty() {
        return 0.0;
    }
    let step = (plane.len() / 65_536).max(1);
    let (sum, count) = plane
        .iter()
        .step_by(step)
        .fold((0.0_f64, 0_usize), |(sum, count), &v| {
            (sum + f64::from(v), count + 1)
        });
    (sum / count as f64) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shader_parses_and_validates_without_a_gpu() {
        let module = naga::front::wgsl::parse_str(&shader_source()).expect("the WGSL parses");
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .expect("the WGSL validates");
        let _ = info;
        for name in KERNELS {
            assert!(
                module.entry_points.iter().any(|e| e.name == name),
                "entry point {name}"
            );
        }
    }

    #[test]
    fn the_window_geometry_is_exact_at_the_largest_radius() {
        // nx Σi² - (Σi)² for the full window of radius 100, as the shader forms it.
        let sum_sq_to = |k: i64| k * (k + 1) * (2 * k + 1) / 6;
        let r = i64::try_from(MAX_RADIUS).expect("a small constant");
        let nx = 2 * r + 1;
        let num = nx * (2 * sum_sq_to(r));
        assert!(num < i64::from(i32::MAX), "{num} fits an i32");
    }
}
