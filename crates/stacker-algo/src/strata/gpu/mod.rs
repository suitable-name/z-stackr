//! GPU-accelerated Strata saliency: a `wgpu` compute-shader port of the
//! per-pixel 4-neighbour Laplacian-magnitude step in
//! [`super::saliency::compute_saliency`].
//!
//! Follows the same structure as [`crate::apex::gpu`]: a small, focused
//! per-call helper with an explicit fallback contract, wired transparently
//! into the single CPU entry point it accelerates.
//!
//! # Scope
//!
//! Only the **single per-pixel Laplacian-magnitude pass** (`h = |up + down +
//! left + right - 4*center|`, clamped boundary) is GPU-accelerated — the
//! five fixed-kernel blur passes that follow it in `compute_saliency`
//! (`apex::pyramid::apply_gaussian_blur`, reused as-is) remain CPU-only. See
//! `docs/gpu_acceleration_summary.md`'s "Scope" section for the rationale.
//!
//! # Fallback contract
//!
//! [`laplacian_magnitude_gpu`] returns `None` (never panics) whenever the
//! GPU path cannot run: no `wgpu` adapter/device available (or the runtime
//! switch is off — see `stacker_core::gpu::context`), or any `wgpu` call
//! along the way fails. [`super::saliency::compute_saliency`] treats `None`
//! as "try the GPU, it declined" and falls back to its own pure-CPU/rayon
//! implementation, which remains fully reachable — this is the only code
//! path exercised by a default (non-`gpu`) build.
//!
//! # Tolerance
//!
//! GPU output is **tolerance-equal, not bit-equal**, to the CPU path, for
//! the same reasons documented in `crate::apex::gpu`'s module docs. The
//! parity test in `tests/gpu_strata_parity.rs` asserts a max-abs difference
//! under `1e-3` on a synthetic frame (including a width not divisible by
//! 16, to exercise the row-stride padding described below).
//!
//! # Row-stride padding
//!
//! See `stacker_core::gpu::padded_bytes_per_row`'s docs — the same
//! 256-byte-alignment requirement and fix applies here as in
//! `crate::apex::gpu` / `stacker_align::transform::gpu`.
use std::sync::OnceLock;

use stacker_core::gpu::padded_bytes_per_row;
use wgpu::util::DeviceExt as _;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    width: u32,
    height: u32,
}

/// Device-resident, size-independent GPU objects for the saliency
/// Laplacian-magnitude pass — same caching rationale as
/// `relief::gpu::CachedGuidedFilterPipelines`: the shader module, bind-group
/// layout, and compute pipeline never depend on the image dimensions of any
/// particular [`laplacian_magnitude_gpu`] call, only the textures/bind group
/// built per call do, so rebuilding them on every call (as this module did
/// before this cache existed) is pure per-frame CPU/driver overhead that
/// widens the sawtooth this cache exists to smooth.
struct CachedSaliencyPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl CachedSaliencyPipeline {
    fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::include_wgsl!("saliency.wgsl"));

        // Explicit bind group layout: `Rgba32Float` is non-filterable, and
        // the shader only ever `textureLoad`s (no sampler) — see the
        // identical note in `apex::gpu` / `transform::gpu`.
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("strata saliency bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
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
            label: Some("strata saliency pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("strata saliency pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
        }
    }
}

/// Process-wide cache: built once from the first call's `&'static Device`
/// (`stacker_core::gpu::context()` — see that module's docs for why the
/// context, and therefore this cache built from it, is sound to keep for
/// the whole process lifetime), reused by every subsequent
/// [`laplacian_magnitude_gpu`] call regardless of the image size it was
/// invoked with.
static SALIENCY_PIPELINE: OnceLock<CachedSaliencyPipeline> = OnceLock::new();

/// Try to compute the Laplacian-magnitude saliency pass on the GPU.
///
/// `luma` must have exactly `width * height` elements. Returns `None` on
/// any failure (no context, texture/pipeline creation failure, or a mapping
/// failure during readback) so the caller falls back to the CPU loop.
///
/// # Fallback contract
/// Returns `None` — never panics — when no `wgpu` context is available or
/// any GPU call fails; [`super::saliency::compute_saliency`] falls back to
/// the CPU kernel in that case.
#[must_use]
pub fn laplacian_magnitude_gpu(luma: &[f32], width: usize, height: usize) -> Option<Vec<f32>> {
    let (device, queue) = stacker_core::gpu::context()?;
    if width == 0 || height == 0 {
        return Some(Vec::new());
    }
    debug_assert_eq!(luma.len(), width * height);

    let width_u32 = width as u32;
    let height_u32 = height as u32;

    // Pack luma into an RGBA32Float texture (single channel used; the other
    // three are padding so the same `Rgba32Float` non-filterable format as
    // every other GPU module in this workspace can be reused verbatim).
    let mut packed = vec![0.0f32; width * height * 4];
    for i in 0..width * height {
        packed[i * 4] = luma[i];
    }

    let unpadded_bytes_per_row = width_u32 * 16;
    let padded_bytes = padded_bytes_per_row(unpadded_bytes_per_row);

    // Hold the process-wide GPU dispatch lock for this whole
    // push/pop-error-scope critical section — see
    // `stacker_core::gpu::dispatch_guard`'s docs for why concurrent callers
    // sharing the one process-wide `Device` must never interleave
    // error-scope push/pop pairs.
    let _dispatch_guard = stacker_core::gpu::dispatch_guard();
    // Capture wgpu validation errors as a scope result instead of letting
    // them reach the process-global uncaptured handler — the fallback
    // contract requires a `None` (CPU fallback), never a panic. Popped
    // after `queue.submit`, before the readback result is trusted.
    let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

    let src_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("strata saliency src"),
        size: wgpu::Extent3d {
            width: width_u32,
            height: height_u32,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let src_bytes: &[u8] = bytemuck::cast_slice(&packed);
    let padded_src: Vec<u8> = if padded_bytes == unpadded_bytes_per_row {
        src_bytes.to_vec()
    } else {
        let mut out = vec![0u8; (padded_bytes * height_u32) as usize];
        for row in 0..height {
            let src_off = row * unpadded_bytes_per_row as usize;
            let dst_off = row * padded_bytes as usize;
            out[dst_off..dst_off + unpadded_bytes_per_row as usize]
                .copy_from_slice(&src_bytes[src_off..src_off + unpadded_bytes_per_row as usize]);
        }
        out
    };
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &src_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &padded_src,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(padded_bytes),
            rows_per_image: Some(height_u32),
        },
        wgpu::Extent3d {
            width: width_u32,
            height: height_u32,
            depth_or_array_layers: 1,
        },
    );

    let dst_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("strata saliency dst"),
        size: wgpu::Extent3d {
            width: width_u32,
            height: height_u32,
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
        width: width_u32,
        height: height_u32,
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("strata saliency uniforms"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Cached shader/pipeline/bind-group-layout — see
    // `CachedSaliencyPipeline`'s doc comment; only the textures/bind group
    // (which vary per call) are built fresh.
    let cached = SALIENCY_PIPELINE.get_or_init(|| CachedSaliencyPipeline::new(device));

    let src_view = src_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let dst_view = dst_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("strata saliency bind group"),
        layout: &cached.bind_group_layout,
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
        cpass.set_pipeline(&cached.pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(
            (width_u32 as f32 / 16.0).ceil() as u32,
            (height_u32 as f32 / 16.0).ceil() as u32,
            1,
        );
    }

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("strata saliency readback buffer"),
        size: u64::from(padded_bytes) * u64::from(height_u32),
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
                rows_per_image: Some(height_u32),
            },
        },
        wgpu::Extent3d {
            width: width_u32,
            height: height_u32,
            depth_or_array_layers: 1,
        },
    );

    queue.submit(Some(encoder.finish()));

    // Check the validation scope BEFORE trusting the readback — see the
    // identical comment in `apex::gpu::fuse_level_gpu`.
    if let Some(err) = pollster::block_on(error_scope.pop()) {
        tracing::debug!(error = %err, "gpu: strata saliency dispatch failed validation, falling back to CPU");
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
    let mut out = vec![0.0f32; width * height];
    for row in 0..height {
        let row_start = row * padded_bytes as usize;
        let row_bytes = &data[row_start..row_start + unpadded_bytes_per_row as usize];
        let row_f32: &[f32] = bytemuck::cast_slice(row_bytes);
        for col in 0..width {
            out[row * width + col] = row_f32[col * 4];
        }
    }
    drop(data);
    output_buffer.unmap();

    Some(out)
}
