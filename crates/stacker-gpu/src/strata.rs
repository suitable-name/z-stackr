//! The `Strata` kernels on the device: the saliency of a luma plane and the
//! guided filter pair that refines the layer weights.
//!
//! A [`StrataPlan`] is built for one frame size and used once per frame: the plane
//! (or the guidance/source pair) is uploaded, the passes run on the device, and
//! only the result is read back.
//!
//! * [`StrataPlan::saliency`] is `stacker_algo::strata::saliency::compute_saliency`:
//!   the 4-neighbour Laplacian magnitude and five passes of the pyramid's 5-tap
//!   blur. The device runs the same taps in the same order on `f32`, so the maps
//!   agree to about `1e-6` of their range (the Laplacian uses `fma`, as the CPU
//!   does; a driver may contract the blur's multiply-adds).
//! * [`StrataPlan::guided_pair`] is
//!   `stacker_algo::relief::guided::guided_filter_pair_luma`. The CPU takes the
//!   window sums from `f64` summed-area tables; the device sums each window
//!   directly in `f32` (two separable passes, `2 r + 1` terms each) over values
//!   shifted by their mean, which keeps the sums small. Covariance and variance
//!   are unchanged by a shift, and the shift is restored in `b`, so the algebra is
//!   exactly the CPU's. The expected difference to the CPU result is below `1e-5`
//!   for masks in `0..=1` (the parity test asserts `1e-3` at most, and a mean
//!   below `1e-5`); the variance clamp at 0 and the regularisation keep the
//!   division well behaved even with the tiny epsilon of the small radius.
//!
//! Device memory: 12 planes of `f32` ([`plan_bytes`]).

use std::sync::Arc;

use bytemuck::cast_slice;

use crate::{
    GpuError,
    common::{self, plane_mean, u32_of},
    context::{Failure, GpuContext},
};

const SHADER: &str = include_str!("strata.wgsl");

/// Threads per workgroup edge; the shader declares 16 x 16 x 1.
const GROUP: u32 = 16;
/// Bytes of the shader's `Params`.
const PARAMS_BYTES: u64 = 48;
/// The saliency's blur passes, as `stacker_algo::strata::saliency::BLUR_PASSES`.
const BLUR_PASSES: usize = 5;
/// The pyramid blur's taps `[W2, W1, RF, W1, W2]`, formed exactly as the CPU's
/// `KERNEL` is (`stacker_algo::apex::pyramid`).
const KERNEL_REDUCTION_FACTOR: f32 = 0.33;
const KERNEL_WEIGHT1: f32 = 0.25;
const KERNEL_WEIGHT2: f32 = 0.25 - KERNEL_REDUCTION_FACTOR / 2.0;

/// The WGSL source of the `Strata` kernels.
#[must_use]
pub fn shader_source() -> String {
    SHADER.to_owned()
}

#[derive(Clone, Copy)]
enum Kernel {
    Laplacian,
    BlurH,
    BlurV,
    BoxHMoments,
    Coeffs,
    BoxHAb,
    Blend,
}

const KERNELS: [&str; 7] = [
    "laplacian",
    "blur_h",
    "blur_v",
    "box_h_moments",
    "coeffs",
    "box_h_ab",
    "blend",
];

/// The device memory a plan for `width x height` frames needs, in bytes: the
/// inputs (2 planes), the working planes (4), the coefficients (2), the results
/// (2) and the read-back buffer (2).
#[must_use]
pub const fn plan_bytes(width: usize, height: usize) -> u64 {
    (width * height) as u64 * 12 * 4
}

/// The uniform block of `strata.wgsl`.
#[derive(Clone, Copy, Default)]
struct Params {
    radius: u32,
    out_plane: u32,
    eps: f32,
    shift_g: f32,
    shift_p: f32,
}

/// The `Strata` saliency and guided-filter kernels for frames of one size.
pub struct StrataPlan {
    ctx: Arc<GpuContext>,
    pipelines: Vec<wgpu::ComputePipeline>,
    /// Bind groups, indexed by [`Kernel`], on the parameters of the first chain
    /// (also the saliency's).
    groups_a: Vec<wgpu::BindGroup>,
    /// The same on the parameters of the second chain.
    groups_b: Vec<wgpu::BindGroup>,
    params_a: wgpu::Buffer,
    params_b: wgpu::Buffer,
    inp: wgpu::Buffer,
    hs: wgpu::Buffer,
    outp: wgpu::Buffer,
    staging: wgpu::Buffer,
    width: usize,
    height: usize,
}

impl StrataPlan {
    /// Build a plan for `width x height` frames.
    ///
    /// # Errors
    /// [`GpuError::TooLarge`] when a buffer would exceed what the device allows,
    /// [`GpuError::Device`] when the device reported an error,
    /// [`GpuError::Injected`] when a test armed [`Failure::Plan`].
    pub fn new(ctx: &Arc<GpuContext>, width: usize, height: usize) -> Result<Self, GpuError> {
        if ctx.injected(Failure::Plan) {
            return Err(GpuError::Injected("building a plan"));
        }
        let plane = (width * height) as u64 * 4;
        let inp = common::storage(ctx, "strata inputs", 2 * plane)?;
        let hs = common::storage(ctx, "strata working planes", 4 * plane)?;
        let ab = common::storage(ctx, "strata coefficients", 2 * plane)?;
        let outp = common::storage(ctx, "strata results", 2 * plane)?;
        let staging = common::readback(ctx, "strata read back", 2 * plane);
        let params_a = common::uniform(ctx, "strata params a", PARAMS_BYTES);
        let params_b = common::uniform(ctx, "strata params b", PARAMS_BYTES);
        let pipelines = common::pipelines(ctx, "strata", shader_source(), &KERNELS);
        let groups_for = |params: &wgpu::Buffer| {
            let bind = |kernel: Kernel, buffers: &[(u32, &wgpu::Buffer)]| {
                let mut all = vec![(0, params)];
                all.extend_from_slice(buffers);
                common::bind(ctx, &pipelines[kernel as usize], &all)
            };
            vec![
                bind(Kernel::Laplacian, &[(1, &inp), (2, &hs)]),
                bind(Kernel::BlurH, &[(2, &hs)]),
                bind(Kernel::BlurV, &[(2, &hs)]),
                bind(Kernel::BoxHMoments, &[(1, &inp), (2, &hs)]),
                bind(Kernel::Coeffs, &[(2, &hs), (3, &ab)]),
                bind(Kernel::BoxHAb, &[(2, &hs), (3, &ab)]),
                bind(Kernel::Blend, &[(1, &inp), (2, &hs), (4, &outp)]),
            ]
        };
        let groups_a = groups_for(&params_a);
        let groups_b = groups_for(&params_b);
        let _ = ctx.device.poll(wgpu::Maintain::Wait);
        common::check(ctx)?;
        Ok(Self {
            ctx: Arc::clone(ctx),
            pipelines,
            groups_a,
            groups_b,
            params_a,
            params_b,
            inp,
            hs,
            outp,
            staging,
            width,
            height,
        })
    }

    /// The frame size this plan was built for, as `(width, height)`.
    #[must_use]
    pub const fn size(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    fn write_params(&self, buffer: &wgpu::Buffer, params: Params) {
        let block: [u32; 12] = [
            u32_of(self.width),
            u32_of(self.height),
            params.radius,
            params.out_plane,
            params.eps.to_bits(),
            params.shift_g.to_bits(),
            params.shift_p.to_bits(),
            KERNEL_WEIGHT2.to_bits(),
            KERNEL_WEIGHT1.to_bits(),
            KERNEL_REDUCTION_FACTOR.to_bits(),
            0,
            0,
        ];
        self.ctx.queue.write_buffer(buffer, 0, cast_slice(&block));
    }

    fn grid(&self) -> [u32; 3] {
        [
            common::groups(self.width, GROUP),
            common::groups(self.height, GROUP),
            1,
        ]
    }

    fn record(&self, encoder: &mut wgpu::CommandEncoder, second: bool, kernel: Kernel) {
        let groups = if second {
            &self.groups_b
        } else {
            &self.groups_a
        };
        common::dispatch(
            encoder,
            &self.pipelines[kernel as usize],
            &groups[kernel as usize],
            self.grid(),
        );
    }

    const fn check_len(&self, len: usize) -> Result<(), GpuError> {
        if len == self.width * self.height {
            Ok(())
        } else {
            // A slice has no shape; report its length as a one-row image.
            Err(GpuError::SizeMismatch {
                expected: (self.width, self.height),
                got: (len, 1),
            })
        }
    }

    /// The saliency of one luma plane of `width * height` samples (the output of
    /// `stacker_algo::strata::saliency::compute_saliency`).
    ///
    /// # Errors
    /// [`GpuError::SizeMismatch`] when `luma` is not the plan's size,
    /// [`GpuError::ReadBack`] or [`GpuError::Device`] when the device failed,
    /// [`GpuError::Injected`] when a test armed [`Failure::Fold`] or
    /// [`Failure::ReadBack`].
    pub fn saliency(&mut self, luma: &[f32]) -> Result<Vec<f32>, GpuError> {
        self.check_len(luma.len())?;
        if luma.is_empty() {
            return Ok(Vec::new());
        }
        if self.ctx.injected(Failure::Fold) {
            return Err(GpuError::Injected("computing the saliency"));
        }
        self.ctx.queue.write_buffer(&self.inp, 0, cast_slice(luma));
        self.write_params(&self.params_a, Params::default());
        let mut encoder = common::encoder(&self.ctx, "strata saliency");
        self.record(&mut encoder, false, Kernel::Laplacian);
        for _ in 0..BLUR_PASSES {
            self.record(&mut encoder, false, Kernel::BlurH);
            self.record(&mut encoder, false, Kernel::BlurV);
        }
        if self.ctx.injected(Failure::ReadBack) {
            return Err(GpuError::Injected("reading the saliency back"));
        }
        let bytes = (self.width * self.height) as u64 * 4;
        common::read(&self.ctx, encoder, &self.hs, &self.staging, bytes)
    }

    /// The guided filter of `src` with `guidance` at the settings
    /// `(radius_a, eps_a)` and `(radius_b, eps_b)`, returned as
    /// `(first, second)`; the twin of
    /// `stacker_algo::relief::guided::guided_filter_pair_luma`.
    ///
    /// A radius larger than the image is the whole image, as on the CPU.
    ///
    /// # Errors
    /// [`GpuError::SizeMismatch`] when a plane is not the plan's size,
    /// [`GpuError::ReadBack`] or [`GpuError::Device`] when the device failed,
    /// [`GpuError::Injected`] when a test armed [`Failure::Fold`] or
    /// [`Failure::ReadBack`].
    ///
    /// # Panics
    /// Panics when the read-back result does not hold two planes (never).
    pub fn guided_pair(
        &mut self,
        guidance: &[f32],
        src: &[f32],
        (radius_a, eps_a): (usize, f32),
        (radius_b, eps_b): (usize, f32),
    ) -> Result<(Vec<f32>, Vec<f32>), GpuError> {
        self.check_len(guidance.len())?;
        self.check_len(src.len())?;
        if guidance.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        if self.ctx.injected(Failure::Fold) {
            return Err(GpuError::Injected("running the guided filter"));
        }
        let n = self.width * self.height;
        let plane = n as u64 * 4;
        self.ctx
            .queue
            .write_buffer(&self.inp, 0, cast_slice(guidance));
        self.ctx
            .queue
            .write_buffer(&self.inp, plane, cast_slice(src));
        let (shift_g, shift_p) = (plane_mean(guidance), plane_mean(src));
        let widest = self.width.max(self.height);
        self.write_params(
            &self.params_a,
            Params {
                radius: u32_of(radius_a.min(widest)),
                out_plane: 0,
                eps: eps_a,
                shift_g,
                shift_p,
            },
        );
        self.write_params(
            &self.params_b,
            Params {
                radius: u32_of(radius_b.min(widest)),
                out_plane: 1,
                eps: eps_b,
                shift_g,
                shift_p,
            },
        );
        let mut encoder = common::encoder(&self.ctx, "strata guided filter");
        for second in [false, true] {
            for kernel in [
                Kernel::BoxHMoments,
                Kernel::Coeffs,
                Kernel::BoxHAb,
                Kernel::Blend,
            ] {
                self.record(&mut encoder, second, kernel);
            }
        }
        if self.ctx.injected(Failure::ReadBack) {
            return Err(GpuError::Injected("reading the guided filter back"));
        }
        let mut both: Vec<f32> =
            common::read(&self.ctx, encoder, &self.outp, &self.staging, 2 * plane)?;
        let second = both.split_off(n);
        Ok((both, second))
    }
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
    fn the_plan_needs_twelve_planes() {
        assert_eq!(plan_bytes(100, 50), 100 * 50 * 48);
    }
}
