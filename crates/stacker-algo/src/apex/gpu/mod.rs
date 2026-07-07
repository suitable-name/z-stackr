//! GPU-accelerated tiled/batch Apex (Laplacian-pyramid) fusion: a `wgpu`
//! compute-shader port of [`super::fuse::fuse_pyramids`]'s per-level blend
//! semantics.
//!
//! # Fallback contract
//!
//! [`fuse_pyramids_gpu`] returns `None` (never panics) whenever the GPU path
//! cannot run: no `wgpu` adapter/device available
//! (`stacker_core::gpu::context` returned `None`), or any `wgpu` call along
//! the way fails. [`super::fuse::fuse_pyramids`] treats `None` as "try the
//! GPU, it declined" and falls back to its own pure-CPU/rayon
//! implementation, which remains fully reachable — this is the only code
//! path exercised by a default (non-`gpu`) build.
//!
//! # Scope
//!
//! The **tiled/batch** Apex fusion (`fuse_pyramids`, called from
//! `build_and_fuse_pyramids`) is GPU-accelerated by this module directly.
//! The incremental whole-image `ApexAccumulator` (used by the GUI's in-RAM
//! path) has its own GPU-resident counterpart in [`accumulator`] — see that
//! submodule's docs for why the batch-fusion reasoning above does not carry
//! over unchanged to the online case, and for its distinct
//! GPU-fails-mid-stack fallback contract (read back and resume on CPU,
//! rather than "never engaged at all"). Strata and Relief have their own
//! GPU submodules (`strata::gpu`, `relief::gpu`) — see
//! `docs/gpu_acceleration_summary.md`'s "Implementation status" section for
//! the full picture across all four engines.
//!
//! # Tolerance
//!
//! GPU output is **tolerance-equal, not bit-equal**, to the CPU path — GPU
//! texture reads/writes and shader arithmetic don't guarantee the same
//! rounding/accumulation order as the CPU's `mul_add`-free direct
//! comparisons. The parity tests in `tests/gpu_fuse_parity.rs` assert a
//! max-abs difference under `1e-3` on synthetic pyramids (including a width
//! not divisible by 16, to exercise the row-stride padding) — that is the
//! tested epsilon referenced above.
//!
//! # Row-stride padding
//!
//! See `stacker_core::gpu::padded_bytes_per_row`'s docs — the same
//! 256-byte-alignment requirement and fix applies here as in
//! `stacker_align::transform::gpu`.
pub mod accumulator;

use crate::apex::pyramid::LaplacianPyramid;
use stacker_core::{gpu::padded_bytes_per_row, image::PlanarImage};
use wgpu::util::DeviceExt as _;

/// Per-level GPU fuse mode, selected by [`level_mode`] and passed to the
/// `fuse.wgsl` shader's `mode` uniform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LevelMode {
    /// Per-pixel mean across all source layers (the base/last level).
    Average,
    /// Per-pixel energy argmax (every level except base, and except the
    /// finest level when grit suppression is on).
    PerPixelArgmax,
    /// 3x3-neighbourhood energy argmax (finest level, grit suppression on).
    NeighborhoodArgmax,
}

impl LevelMode {
    /// The `mode` uniform value `fuse.wgsl` expects.
    const fn shader_mode(self) -> u32 {
        match self {
            Self::Average => 0,
            Self::PerPixelArgmax => 1,
            Self::NeighborhoodArgmax => 2,
        }
    }
}

/// Select the fuse mode for `level_idx` out of `levels_count` total levels,
/// mirroring `apex::fuse::fuse_pyramids`'s own level dispatch exactly:
///
/// - the LAST level (the base/low-frequency residual) is always
///   [`LevelMode::Average`], regardless of `grit_suppression`;
/// - level `0` (the finest level) is [`LevelMode::NeighborhoodArgmax`] when
///   `grit_suppression` is set, else [`LevelMode::PerPixelArgmax`];
/// - every other level is [`LevelMode::PerPixelArgmax`].
///
/// A single-level pyramid (`levels_count == 1`) is entirely the base level
/// (`Average`) — matching the CPU code's `level_idx == levels_count - 1`
/// check, which fires immediately for the only level.
#[must_use]
pub const fn level_mode(
    level_idx: usize,
    levels_count: usize,
    grit_suppression: bool,
) -> LevelMode {
    if level_idx == levels_count - 1 {
        LevelMode::Average
    } else if level_idx == 0 && grit_suppression {
        LevelMode::NeighborhoodArgmax
    } else {
        LevelMode::PerPixelArgmax
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    layer_count: u32,
    width: u32,
    height: u32,
    use_color: u32,
    mode: u32,
    _padding: [u32; 3],
}

/// Pack one Laplacian-pyramid level from every source pyramid into a
/// row-major `RGBA32Float` layer array buffer (no row/layer padding — the
/// caller pads per-row when uploading).
fn pack_level_layers(
    pyramids: &[LaplacianPyramid],
    level_idx: usize,
    width: usize,
    height: usize,
) -> Vec<f32> {
    let mut packed = vec![0.0f32; width * height * 4 * pyramids.len()];
    for (k, p) in pyramids.iter().enumerate() {
        let src = &p.levels[level_idx];
        let base = k * width * height * 4;
        for i in 0..width * height {
            packed[base + i * 4] = src.luma[i];
            packed[base + i * 4 + 1] = src.chroma_a[i];
            packed[base + i * 4 + 2] = src.chroma_b[i];
            packed[base + i * 4 + 3] = 1.0;
        }
    }
    packed
}

/// Try to fuse one pyramid level's layers on the GPU. Returns `None` on any
/// failure (no context, texture/pipeline creation failure, or a mapping
/// failure during readback) so the caller can fall back to the CPU blend
/// functions for this level.
// `too_many_lines`: one cohesive wgpu dispatch (texture/buffer setup,
// pipeline creation, dispatch, validation-scope check, readback) — splitting
// it would scatter GPU resources that must be created/used/dropped together
// across artificial function boundaries.
#[allow(clippy::too_many_lines)]
fn fuse_level_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pyramids: &[LaplacianPyramid],
    level_idx: usize,
    use_color: bool,
    mode: LevelMode,
) -> Option<PlanarImage<f32>> {
    let width = pyramids[0].levels[level_idx].width as u32;
    let height = pyramids[0].levels[level_idx].height as u32;
    let layer_count = pyramids.len() as u32;
    if width == 0 || height == 0 {
        return Some(PlanarImage::new(width as usize, height as usize));
    }

    let unpadded_bytes_per_row = width * 16;
    let padded_bytes = padded_bytes_per_row(unpadded_bytes_per_row);

    // Hold the process-wide GPU dispatch lock for this whole
    // push/pop-error-scope critical section — `stacker_core::gpu::context`
    // hands out the same process-wide `Device` to every caller (including
    // `apex::gpu::accumulator`'s concurrently-runnable dispatch calls), and
    // `push_error_scope`/`pop_error_scope` form a per-device stack that
    // must never see two callers' scopes interleaved. See
    // `stacker_core::gpu::dispatch_guard`'s docs for the full rationale.
    let _dispatch_guard = stacker_core::gpu::dispatch_guard();
    // Capture wgpu validation errors from every resource/pipeline/encoding
    // call below as a scope result instead of letting them reach the
    // process-global uncaptured handler — the fallback contract requires a
    // `None` (CPU fallback), never a panic. Popped after `queue.submit`.
    let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

    let src_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("apex fuse src array"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: layer_count,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let packed = pack_level_layers(pyramids, level_idx, width as usize, height as usize);
    for k in 0..layer_count as usize {
        let layer_bytes: &[u8] = bytemuck::cast_slice(
            &packed[k * width as usize * height as usize * 4
                ..(k + 1) * width as usize * height as usize * 4],
        );
        let padded: Vec<u8> = if padded_bytes == unpadded_bytes_per_row {
            layer_bytes.to_vec()
        } else {
            let mut out = vec![0u8; (padded_bytes * height) as usize];
            for row in 0..height as usize {
                let src_off = row * unpadded_bytes_per_row as usize;
                let dst_off = row * padded_bytes as usize;
                out[dst_off..dst_off + unpadded_bytes_per_row as usize].copy_from_slice(
                    &layer_bytes[src_off..src_off + unpadded_bytes_per_row as usize],
                );
            }
            out
        };
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &src_texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: k as u32,
                },
                aspect: wgpu::TextureAspect::All,
            },
            &padded,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }

    let dst_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("apex fuse dst"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    let uniforms = Uniforms {
        layer_count,
        width,
        height,
        use_color: u32::from(use_color),
        mode: mode.shader_mode(),
        _padding: [0, 0, 0],
    };

    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("apex fuse uniforms"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let shader = device.create_shader_module(wgpu::include_wgsl!("fuse.wgsl"));

    // Explicit bind group layout: `Rgba32Float` is a NON-filterable float
    // format, but `layout: None` auto-inference declares `texture_2d_array`
    // bindings as `Float { filterable: true }` — a guaranteed validation
    // error on every adapter. The shader only ever `textureLoad`s (no
    // sampler), so `filterable: false` is the correct declaration.
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("apex fuse bind group layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba32Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("apex fuse pipeline layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("apex fuse pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let src_view = src_texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    let dst_view = dst_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("apex fuse bind group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&src_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&dst_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(
            (width as f32 / 16.0).ceil() as u32,
            (height as f32 / 16.0).ceil() as u32,
            1,
        );
    }

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("apex fuse readback buffer"),
        size: u64::from(padded_bytes) * u64::from(height),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &dst_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &output_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    queue.submit(Some(encoder.finish()));

    // Check the validation scope BEFORE trusting the readback: with an
    // invalid bind group the dispatch silently does nothing and the
    // destination texture stays all-zero — mapping it back would "succeed"
    // and hand the caller a black tile instead of falling back to CPU.
    if let Some(err) = pollster::block_on(error_scope.pop()) {
        tracing::debug!(error = %err, "gpu: apex fuse dispatch failed validation, falling back to CPU");
        return None;
    }

    let buffer_slice = output_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
        let _ = tx.send(v);
    });
    if device.poll(wgpu::PollType::wait_indefinitely()).is_err() {
        return None;
    }
    match rx.recv() {
        Ok(Ok(())) => {}
        _ => return None,
    }

    let Ok(data) = buffer_slice.get_mapped_range() else {
        return None;
    };
    let mut dst = PlanarImage::new(width as usize, height as usize);
    for row in 0..height as usize {
        let row_start = row * padded_bytes as usize;
        let row_bytes = &data[row_start..row_start + unpadded_bytes_per_row as usize];
        let row_f32: &[f32] = bytemuck::cast_slice(row_bytes);
        for col in 0..width as usize {
            let i = row * width as usize + col;
            dst.luma[i] = row_f32[col * 4];
            dst.chroma_a[i] = row_f32[col * 4 + 1];
            dst.chroma_b[i] = row_f32[col * 4 + 2];
        }
    }
    drop(data);
    output_buffer.unmap();

    Some(dst)
}

/// Try to fuse every level of `pyramids` on the GPU, matching
/// `apex::fuse::fuse_pyramids`'s semantics (see [`level_mode`]).
///
/// # Fallback contract
/// Returns `None` — never panics — when no `wgpu` context is available, or
/// any level's GPU dispatch fails. [`super::fuse::fuse_pyramids`] falls back
/// to the pure-CPU implementation in that case (for the WHOLE pyramid, not
/// per-level, so a fusion result is never a patchwork of GPU and CPU
/// levels).
#[must_use]
pub fn fuse_pyramids_gpu(
    pyramids: &[LaplacianPyramid],
    use_color: bool,
    grit_suppression: bool,
) -> Option<LaplacianPyramid> {
    let (device, queue) = stacker_core::gpu::context()?;
    if pyramids.is_empty() {
        return None;
    }

    let levels_count = pyramids[0].levels.len();
    for p in pyramids {
        if p.levels.len() != levels_count {
            return None;
        }
    }

    let mut fused_levels = Vec::with_capacity(levels_count);
    for level_idx in 0..levels_count {
        let mode = level_mode(level_idx, levels_count, grit_suppression);
        let level = fuse_level_gpu(device, queue, pyramids, level_idx, use_color, mode)?;
        fused_levels.push(level);
    }

    Some(LaplacianPyramid {
        levels: fused_levels,
    })
}

#[cfg(test)]
mod tests {
    use super::{LevelMode, level_mode};

    #[test]
    fn base_level_is_always_average() {
        // Last level index, several level counts, grit on/off — always Average.
        assert_eq!(level_mode(3, 4, true), LevelMode::Average);
        assert_eq!(level_mode(3, 4, false), LevelMode::Average);
        assert_eq!(level_mode(0, 1, true), LevelMode::Average);
        assert_eq!(level_mode(0, 1, false), LevelMode::Average);
    }

    #[test]
    fn finest_level_depends_on_grit_suppression() {
        assert_eq!(level_mode(0, 4, true), LevelMode::NeighborhoodArgmax);
        assert_eq!(level_mode(0, 4, false), LevelMode::PerPixelArgmax);
    }

    #[test]
    fn middle_levels_are_always_per_pixel_argmax() {
        for grit in [true, false] {
            assert_eq!(level_mode(1, 5, grit), LevelMode::PerPixelArgmax);
            assert_eq!(level_mode(2, 5, grit), LevelMode::PerPixelArgmax);
            assert_eq!(level_mode(3, 5, grit), LevelMode::PerPixelArgmax);
        }
    }
}
