//! Alignment on the device: the registration objective and the final warp.
//!
//! [`ObjectivePlan`] holds one pyramid level (reference and source luma) on the
//! device; [`ObjectivePlan::moments`] warps the reference by a candidate inverse
//! matrix and returns the moments of `source - warped` over the pixels that take
//! part, which is all the refiner in `stacker_align::refine` needs to form its
//! RMS. One evaluation is one dispatch and a read-back of one small vector per
//! workgroup, so the optimiser's hundreds of evaluations per level never move an
//! image.
//!
//! [`WarpPlan`] warps a whole frame (three planes) with the bicubic sampler and a
//! black border, or with the clamped spline, and returns the coverage mask the
//! same dispatch computed, so the mask and the black border agree on every pixel.
//!
//! The samplers and coordinate formulas are those of `stacker_align::transform`
//! (see `align.wgsl`); the results agree with the CPU within rounding, not bit
//! for bit (see the crate documentation).

use std::sync::Arc;

use bytemuck::cast_slice;
use stacker_core::image::PlanarImage;

use crate::{
    GpuError,
    common::{self, u32_of},
    context::{Failure, GpuContext},
};

const SHADER: &str = include_str!("align.wgsl");

/// Threads per workgroup of the moments kernel; the shader declares 256.
const MOMENTS_GROUP: u32 = 256;
/// The most workgroups one evaluation dispatches; each thread then folds
/// `pixels / (256 * groups)` pixels before the workgroup merges them.
const MAX_MOMENT_GROUPS: u32 = 1024;
/// Threads per workgroup edge of the warp kernel; the shader declares 16 x 16.
const WARP_GROUP: u32 = 16;
/// Bytes of the shader's `Params`: three `vec4<f32>` and four `u32`.
const PARAMS_BYTES: u64 = 64;

/// The WGSL source of the alignment kernels.
#[must_use]
pub fn shader_source() -> String {
    SHADER.to_owned()
}

/// Which sampler warps the image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sampler {
    /// Catmull-Rom bicubic. In the objective, pixels whose source lies outside
    /// the image take no part; in the warp they are black.
    Bicubic,
    /// The four-tap C0 cubic with edge-clamped borders.
    SplineClamped,
}

impl Sampler {
    const fn code(self) -> u32 {
        match self {
            Self::Bicubic => 0,
            Self::SplineClamped => 1,
        }
    }
}

/// Moments of `source - warped reference` over the pixels that took part.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Moments {
    /// Pixels that took part.
    pub count: u64,
    /// Mean of the difference.
    pub mean: f64,
    /// Sum of the squared deviations from the mean.
    pub m2: f64,
}

impl Moments {
    const EMPTY: Self = Self {
        count: 0,
        mean: 0.0,
        m2: 0.0,
    };

    /// Combine two disjoint sets (Chan et al.).
    #[expect(
        clippy::cast_precision_loss,
        reason = "pixel counts are far below 2^52"
    )]
    fn merge(self, other: Self) -> Self {
        if other.count == 0 {
            return self;
        }
        if self.count == 0 {
            return other;
        }
        let (na, nb) = (self.count as f64, other.count as f64);
        let n = na + nb;
        let delta = other.mean - self.mean;
        Self {
            count: self.count + other.count,
            mean: self.mean + delta * nb / n,
            m2: delta * delta * na * nb / n + self.m2 + other.m2,
        }
    }
}

/// The parameter block of `align.wgsl`: the inverse matrix row by row, the size,
/// the sampler and the number of workgroups.
fn params(
    inverse: &[[f32; 3]; 3],
    width: usize,
    height: usize,
    sampler: Sampler,
    groups: u32,
) -> [u32; 16] {
    let mut block = [0_u32; 16];
    for (r, row) in inverse.iter().enumerate() {
        for (c, value) in row.iter().enumerate() {
            block[r * 4 + c] = value.to_bits();
        }
    }
    block[12] = u32_of(width);
    block[13] = u32_of(height);
    block[14] = sampler.code();
    block[15] = groups;
    block
}

/// The device memory an [`ObjectivePlan`] for a `width x height` level needs, in
/// bytes (both luma planes and the per-workgroup moments).
#[must_use]
pub const fn objective_bytes(width: usize, height: usize) -> u64 {
    let n = (width * height) as u64;
    8 * n + 2 * 16 * MAX_MOMENT_GROUPS as u64
}

/// One pyramid level of the registration objective, on the device.
pub struct ObjectivePlan {
    ctx: Arc<GpuContext>,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    params: wgpu::Buffer,
    partials: wgpu::Buffer,
    staging: wgpu::Buffer,
    width: usize,
    height: usize,
    sampler: Sampler,
    groups: u32,
    // Kept alive for the bind group.
    _planes: [wgpu::Buffer; 2],
}

impl ObjectivePlan {
    /// Upload one level: `reference` is warped, `source` is compared with it.
    ///
    /// # Errors
    /// [`GpuError::SizeMismatch`] when a plane is not `width * height` long,
    /// [`GpuError::TooLarge`] when a plane exceeds what the device allows,
    /// [`GpuError::Device`] when the device reported an error.
    ///
    /// # Panics
    /// Panics when a dimension exceeds `u32::MAX`.
    pub fn new(
        ctx: &Arc<GpuContext>,
        reference: &[f32],
        source: &[f32],
        width: usize,
        height: usize,
        sampler: Sampler,
    ) -> Result<Self, GpuError> {
        if ctx.injected(Failure::Plan) {
            return Err(GpuError::Injected("building a plan"));
        }
        let n = width * height;
        for plane in [reference, source] {
            if plane.len() != n || n == 0 {
                return Err(GpuError::SizeMismatch {
                    expected: (width, height),
                    got: (plane.len(), 1),
                });
            }
        }
        let bytes = (n as u64) * 4;
        let reference_buffer = common::storage(ctx, "objective reference", bytes)?;
        let source_buffer = common::storage(ctx, "objective source", bytes)?;
        ctx.queue
            .write_buffer(&reference_buffer, 0, cast_slice(reference));
        ctx.queue
            .write_buffer(&source_buffer, 0, cast_slice(source));
        let groups = common::groups(n, MOMENTS_GROUP).clamp(1, MAX_MOMENT_GROUPS);
        let partial_bytes = 16 * u64::from(groups);
        let partials = common::storage(ctx, "objective moments", partial_bytes)?;
        let staging = common::readback(ctx, "objective read back", partial_bytes);
        let params = common::uniform(ctx, "objective params", PARAMS_BYTES);
        let pipeline = common::pipelines(ctx, "align", shader_source(), &["moments"])
            .pop()
            .expect("one entry point asked for");
        let bind_group = common::bind(
            ctx,
            &pipeline,
            &[
                (0, &params),
                (1, &reference_buffer),
                (2, &source_buffer),
                (3, &partials),
            ],
        );
        let _ = ctx.device.poll(wgpu::Maintain::Wait);
        common::check(ctx)?;
        Ok(Self {
            ctx: Arc::clone(ctx),
            pipeline,
            bind_group,
            params,
            partials,
            staging,
            width,
            height,
            sampler,
            groups,
            _planes: [reference_buffer, source_buffer],
        })
    }

    /// The moments for the warp by `inverse` (row-major: output pixel `(x, y)`
    /// samples the reference at `inverse * (x, y, 1)`).
    ///
    /// # Errors
    /// [`GpuError::ReadBack`] or [`GpuError::Device`] when the device failed.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn moments(&mut self, inverse: &[[f32; 3]; 3]) -> Result<Moments, GpuError> {
        let encoder = self.encode(inverse)?;
        let partials: Vec<[f32; 4]> = common::read(
            &self.ctx,
            encoder,
            &self.partials,
            &self.staging,
            16 * u64::from(self.groups),
        )?;
        Ok(fold_partials(&partials))
    }

    /// [`ObjectivePlan::moments`] without blocking: the same dispatch, and the
    /// read-back is awaited instead of waited for. Works inside a Web Worker.
    ///
    /// # Errors
    /// [`GpuError::ReadBack`] or [`GpuError::Device`] when the device failed.
    #[cfg_attr(
        target_arch = "wasm32",
        expect(
            clippy::future_not_send,
            reason = "WebGPU handles are not Send; the worker is single-threaded"
        )
    )]
    pub async fn moments_async(&mut self, inverse: &[[f32; 3]; 3]) -> Result<Moments, GpuError> {
        let encoder = self.encode(inverse)?;
        let partials: Vec<[f32; 4]> = common::read_async(
            &self.ctx,
            encoder,
            &self.partials,
            &self.staging,
            16 * u64::from(self.groups),
        )
        .await?;
        Ok(fold_partials(&partials))
    }

    /// Upload the parameters for `inverse` and record the dispatch; the read-back
    /// is the caller's. Shared by the blocking and the async evaluation.
    fn encode(&self, inverse: &[[f32; 3]; 3]) -> Result<wgpu::CommandEncoder, GpuError> {
        if self.ctx.injected(Failure::Fold) {
            return Err(GpuError::Injected("evaluating the objective"));
        }
        let block = params(inverse, self.width, self.height, self.sampler, self.groups);
        self.ctx
            .queue
            .write_buffer(&self.params, 0, cast_slice(&block));
        let mut encoder = common::encoder(&self.ctx, "objective");
        common::dispatch(
            &mut encoder,
            &self.pipeline,
            &self.bind_group,
            [self.groups, 1, 1],
        );
        if self.ctx.injected(Failure::ReadBack) {
            return Err(GpuError::Injected("reading the moments back"));
        }
        Ok(encoder)
    }
}

/// Merge the per-workgroup partials in workgroup order, in `f64`: the same sum
/// for the same partials every time.
fn fold_partials(partials: &[[f32; 4]]) -> Moments {
    partials.iter().fold(Moments::EMPTY, |acc, part| {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a workgroup's count is a whole number below 2^24, exact in f32"
        )]
        let count = part[0] as u64;
        acc.merge(Moments {
            count,
            mean: f64::from(part[1]),
            m2: f64::from(part[2]),
        })
    })
}

/// The device memory a [`WarpPlan`] for `width x height` frames needs, in bytes:
/// the frame, the warped frame, the coverage bits and their two read-back
/// buffers.
#[must_use]
pub const fn warp_bytes(width: usize, height: usize) -> u64 {
    let n = (width * height) as u64;
    let image = 12 * n;
    let bits = n.div_ceil(32) * 4;
    3 * image + 2 * bits
}

/// The final warp of frames of one size, on the device.
pub struct WarpPlan {
    ctx: Arc<GpuContext>,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    params: wgpu::Buffer,
    input: wgpu::Buffer,
    output: wgpu::Buffer,
    coverage: wgpu::Buffer,
    image_staging: wgpu::Buffer,
    coverage_staging: wgpu::Buffer,
    width: usize,
    height: usize,
}

impl WarpPlan {
    /// Allocate a plan for `width x height` frames.
    ///
    /// # Errors
    /// [`GpuError::TooLarge`] when a buffer exceeds what the device allows,
    /// [`GpuError::Device`] when the device reported an error.
    ///
    /// # Panics
    /// Panics when a dimension exceeds `u32::MAX`.
    pub fn new(ctx: &Arc<GpuContext>, width: usize, height: usize) -> Result<Self, GpuError> {
        if ctx.injected(Failure::Plan) {
            return Err(GpuError::Injected("building a plan"));
        }
        let n = (width * height) as u64;
        let image_bytes = 12 * n;
        let bit_bytes = n.div_ceil(32) * 4;
        let input = common::storage(ctx, "warp input", image_bytes)?;
        let output = common::storage(ctx, "warp output", image_bytes)?;
        let coverage = common::storage(ctx, "warp coverage", bit_bytes)?;
        let image_staging = common::readback(ctx, "warp read back", image_bytes);
        let coverage_staging = common::readback(ctx, "warp coverage read back", bit_bytes);
        let params = common::uniform(ctx, "warp params", PARAMS_BYTES);
        let pipeline = common::pipelines(ctx, "align", shader_source(), &["warp"])
            .pop()
            .expect("one entry point asked for");
        let bind_group = common::bind(
            ctx,
            &pipeline,
            &[(0, &params), (1, &input), (4, &output), (5, &coverage)],
        );
        let _ = ctx.device.poll(wgpu::Maintain::Wait);
        common::check(ctx)?;
        Ok(Self {
            ctx: Arc::clone(ctx),
            pipeline,
            bind_group,
            params,
            input,
            output,
            coverage,
            image_staging,
            coverage_staging,
            width,
            height,
        })
    }

    /// Warp `frame` by `inverse` (row-major: output pixel `(x, y)` samples the
    /// frame at `inverse * (x, y, 1)`, the inverse of the matrix `warp_image`
    /// takes). Returns the warped frame and its coverage mask (`true` where the
    /// source coordinate lies inside the frame).
    ///
    /// # Errors
    /// [`GpuError::SizeMismatch`] when the frame is not the plan's size,
    /// [`GpuError::ReadBack`] or [`GpuError::Device`] when the device failed.
    ///
    /// # Panics
    /// Panics when the read-back planes do not have the frame's size (never).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn warp(
        &mut self,
        frame: &PlanarImage<f32>,
        inverse: &[[f32; 3]; 3],
        sampler: Sampler,
    ) -> Result<(PlanarImage<f32>, Vec<bool>), GpuError> {
        let encoder = self.encode(frame, inverse, sampler)?;
        let n = self.width * self.height;
        let plane_bytes = (n as u64) * 4;
        let bits: Vec<u32> = common::read(
            &self.ctx,
            encoder,
            &self.coverage,
            &self.coverage_staging,
            (n as u64).div_ceil(32) * 4,
        )?;
        let planes: Vec<f32> = common::read(
            &self.ctx,
            common::encoder(&self.ctx, "warp read back"),
            &self.output,
            &self.image_staging,
            3 * plane_bytes,
        )?;
        Ok(self.assemble(&bits, &planes))
    }

    /// [`WarpPlan::warp`] without blocking: the same dispatch, and the two
    /// read-backs are awaited instead of waited for. Works inside a Web Worker.
    ///
    /// # Errors
    /// [`GpuError::SizeMismatch`] when the frame is not the plan's size,
    /// [`GpuError::ReadBack`] or [`GpuError::Device`] when the device failed.
    ///
    /// # Panics
    /// Panics when the read-back planes do not have the frame's size (never).
    #[cfg_attr(
        target_arch = "wasm32",
        expect(
            clippy::future_not_send,
            reason = "WebGPU handles are not Send; the worker is single-threaded"
        )
    )]
    pub async fn warp_async(
        &mut self,
        frame: &PlanarImage<f32>,
        inverse: &[[f32; 3]; 3],
        sampler: Sampler,
    ) -> Result<(PlanarImage<f32>, Vec<bool>), GpuError> {
        let encoder = self.encode(frame, inverse, sampler)?;
        let n = self.width * self.height;
        let plane_bytes = (n as u64) * 4;
        let bits: Vec<u32> = common::read_async(
            &self.ctx,
            encoder,
            &self.coverage,
            &self.coverage_staging,
            (n as u64).div_ceil(32) * 4,
        )
        .await?;
        let planes: Vec<f32> = common::read_async(
            &self.ctx,
            common::encoder(&self.ctx, "warp read back"),
            &self.output,
            &self.image_staging,
            3 * plane_bytes,
        )
        .await?;
        Ok(self.assemble(&bits, &planes))
    }

    /// The warped frame and the coverage mask from the two read-backs.
    fn assemble(&self, bits: &[u32], planes: &[f32]) -> (PlanarImage<f32>, Vec<bool>) {
        let n = self.width * self.height;
        let coverage = (0..n).map(|i| bits[i / 32] >> (i % 32) & 1 == 1).collect();
        let image = PlanarImage::from_planes(
            self.width,
            self.height,
            planes[..n].to_vec(),
            planes[n..2 * n].to_vec(),
            planes[2 * n..].to_vec(),
        )
        .expect("each plane holds width * height samples");
        (image, coverage)
    }

    /// Check the frame, upload it and the parameters and record the clear and the
    /// dispatch; the read-backs are the caller's. Shared by the blocking and the
    /// async warp.
    fn encode(
        &self,
        frame: &PlanarImage<f32>,
        inverse: &[[f32; 3]; 3],
        sampler: Sampler,
    ) -> Result<wgpu::CommandEncoder, GpuError> {
        if frame.width != self.width || frame.height != self.height {
            return Err(GpuError::SizeMismatch {
                expected: (self.width, self.height),
                got: (frame.width, frame.height),
            });
        }
        if self.ctx.injected(Failure::Fold) {
            return Err(GpuError::Injected("warping a frame"));
        }
        let n = self.width * self.height;
        let plane_bytes = (n as u64) * 4;
        let queue = &self.ctx.queue;
        queue.write_buffer(&self.input, 0, cast_slice(&frame.luma));
        queue.write_buffer(&self.input, plane_bytes, cast_slice(&frame.chroma_a));
        queue.write_buffer(&self.input, 2 * plane_bytes, cast_slice(&frame.chroma_b));
        let block = params(inverse, self.width, self.height, sampler, 0);
        queue.write_buffer(&self.params, 0, cast_slice(&block));

        let mut encoder = common::encoder(&self.ctx, "warp");
        encoder.clear_buffer(&self.coverage, 0, None);
        common::dispatch(
            &mut encoder,
            &self.pipeline,
            &self.bind_group,
            [
                common::groups(self.width, WARP_GROUP),
                common::groups(self.height, WARP_GROUP),
                1,
            ],
        );
        if self.ctx.injected(Failure::ReadBack) {
            return Err(GpuError::Injected("reading the warped frame back"));
        }
        Ok(encoder)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shader_parses_and_validates_without_a_gpu() {
        let module = naga::front::wgsl::parse_str(&shader_source()).expect("the WGSL parses");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .expect("the WGSL validates");
    }

    #[test]
    fn the_parameter_block_matches_the_shader_layout() {
        let inverse = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let block = params(&inverse, 640, 480, Sampler::SplineClamped, 12);
        assert_eq!(f32::from_bits(block[1]), 2.0);
        assert_eq!(f32::from_bits(block[4]), 4.0);
        assert_eq!(f32::from_bits(block[10]), 9.0);
        assert_eq!(block[3], 0, "the fourth component of a row is padding");
        assert_eq!(&block[12..], &[640, 480, 1, 12]);
        assert_eq!(std::mem::size_of_val(&block) as u64, PARAMS_BYTES);
    }

    #[test]
    fn partial_moments_merge_like_one_pass() {
        let values: Vec<f64> = (0..50)
            .map(|i| f64::from(i % 9).mul_add(0.1, -0.2))
            .collect();
        let one = |v: &[f64]| {
            #[expect(clippy::cast_precision_loss, reason = "small test counts")]
            let n = v.len() as f64;
            let mean = v.iter().sum::<f64>() / n;
            Moments {
                count: v.len() as u64,
                mean,
                m2: v.iter().map(|x| (x - mean) * (x - mean)).sum(),
            }
        };
        let merged = one(&values[..20]).merge(one(&values[20..]));
        let whole = one(&values);
        assert_eq!(merged.count, whole.count);
        assert!((merged.mean - whole.mean).abs() < 1e-12);
        assert!((merged.m2 - whole.m2).abs() < 1e-10);
    }
}
