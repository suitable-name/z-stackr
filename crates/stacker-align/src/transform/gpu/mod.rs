//! GPU-accelerated production warp.
//!
//! A `wgpu` compute-shader port of [`super::warp::spline4x4_sample_clamped`]
//! / [`super::warp_image_clamped`].
//!
//! # Fallback contract
//!
//! [`warp_image_clamped_gpu`] returns `Ok(None)` (never panics) whenever the
//! GPU path cannot run: no `wgpu` adapter/device available
//! (`stacker_core::gpu::context` returned `None`), or any `wgpu` call along
//! the way fails. [`super::warp_image_clamped`] treats `Ok(None)` exactly
//! like "try the GPU, it declined" and falls back to
//! [`super::warp_image_clamped_cpu`] — the pure-CPU/SIMD kernel remains
//! reachable and is in fact still the code every non-`gpu` build runs. A
//! non-finite/non-invertible `matrix` is validated and rejected with
//! `Err` *before* any GPU work is attempted, matching the CPU path's own
//! `Err` contract for the same inputs (so callers see one consistent error
//! shape regardless of which backend actually executed).
//!
//! # Tolerance
//!
//! GPU output is **tolerance-equal, not bit-equal**, to the CPU/SIMD path:
//! `f32` texture filtering/accumulation order on the GPU does not exactly
//! match the CPU's `mul_add`-chained accumulation. The parity tests in
//! `tests/gpu_warp_parity.rs` assert a max-abs difference under `1e-3` on
//! synthetic images (including a width not divisible by 16, to exercise the
//! row-stride padding described below) — that is the tested epsilon
//! referenced by the module-level fallback contract above.
//!
//! # Row-stride padding
//!
//! `wgpu::TexelCopyBufferLayout::bytes_per_row` must be a multiple of
//! `wgpu::COPY_BYTES_PER_ROW_ALIGNMENT` (256). A naive `width * 16` (16
//! bytes per `Rgba32Float` texel) is only a multiple of 256 when `width` is
//! itself a multiple of 16 — every other width previously failed at
//! runtime. Both the upload ([`upload_texture`]) and the readback
//! ([`readback_texture`]) here pad the row stride to the next 256-byte
//! multiple via [`stacker_core::gpu::padded_bytes_per_row`] and de-stride
//! during readback.
use nalgebra::Matrix3;
use stacker_core::{error::StackerError, gpu::padded_bytes_per_row, image::PlanarImage};
use wgpu::util::DeviceExt as _;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    m_inv: [[f32; 4]; 3],
    width: u32,
    height: u32,
    _padding: [u32; 2],
}

/// Pack a `PlanarImage<f32>` into a row-major `RGBA32Float` buffer.
///
/// Alpha channel unused (set to `1.0`), one `[f32; 4]` per pixel, no row
/// padding (the caller pads when writing to the GPU texture).
fn pack_rgba(img: &PlanarImage<f32>) -> Vec<f32> {
    let mut packed = vec![0.0f32; img.width * img.height * 4];
    for i in 0..img.width * img.height {
        packed[i * 4] = img.luma[i];
        packed[i * 4 + 1] = img.chroma_a[i];
        packed[i * 4 + 2] = img.chroma_b[i];
        packed[i * 4 + 3] = 1.0;
    }
    packed
}

/// Upload a packed `RGBA32Float` buffer into a fresh 2-D texture.
///
/// Handles the 256-byte row-stride alignment `write_texture` requires
/// internally: the source `data` itself is tightly packed — this function
/// pads while writing, so callers never construct a padded buffer
/// themselves.
fn upload_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    width: u32,
    height: u32,
    data: &[f32],
) -> wgpu::Texture {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let unpadded_bytes_per_row = width * 16;
    let padded_bytes = padded_bytes_per_row(unpadded_bytes_per_row);
    let padded: Vec<u8> = if padded_bytes == unpadded_bytes_per_row {
        bytemuck::cast_slice(data).to_vec()
    } else {
        let src_bytes: &[u8] = bytemuck::cast_slice(data);
        let mut out = vec![0u8; (padded_bytes * height) as usize];
        for row in 0..height as usize {
            let src_off = row * unpadded_bytes_per_row as usize;
            let dst_off = row * padded_bytes as usize;
            out[dst_off..dst_off + unpadded_bytes_per_row as usize]
                .copy_from_slice(&src_bytes[src_off..src_off + unpadded_bytes_per_row as usize]);
        }
        out
    };

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
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

    texture
}

/// Read a `width x height` `Rgba32Float` texture back into a `PlanarImage`.
///
/// De-strides the padded rows `copy_texture_to_buffer` requires.
///
/// # Errors
/// Returns [`StackerError::MathError`] if the mapped-buffer channel closes
/// without a reply (should not happen with `device.poll(PollType::wait_indefinitely())`
/// immediately preceding the receive) or if `wgpu` itself reports a mapping
/// or device-poll failure.
fn readback_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
) -> Result<PlanarImage<f32>, StackerError> {
    let unpadded_bytes_per_row = width * 16;
    let padded_bytes = padded_bytes_per_row(unpadded_bytes_per_row);

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("warp_image_clamped_gpu readback buffer"),
        size: u64::from(padded_bytes) * u64::from(height),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("warp_image_clamped_gpu readback encoder"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
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

    let buffer_slice = output_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
        let _ = tx.send(v);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| StackerError::MathError(format!("gpu: device poll failed: {e}")))?;
    rx.recv()
        .map_err(|_| StackerError::MathError("gpu: readback channel closed unexpectedly".into()))?
        .map_err(|e| StackerError::MathError(format!("gpu: buffer map failed: {e}")))?;

    let data = buffer_slice
        .get_mapped_range()
        .map_err(|e| StackerError::MathError(format!("gpu: buffer map range failed: {e}")))?;
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

    Ok(dst)
}

fn create_warp_pipeline(device: &wgpu::Device) -> (wgpu::BindGroupLayout, wgpu::ComputePipeline) {
    let shader = device.create_shader_module(wgpu::include_wgsl!("warp.wgsl"));

    // Explicit bind group layout: `Rgba32Float` is a NON-filterable float
    // format, but `layout: None` auto-inference declares `texture_2d`
    // bindings as `Float { filterable: true }` — a guaranteed validation
    // error on every adapter. The shader only ever `textureLoad`s (no
    // sampler), so `filterable: false` is the correct declaration.
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("warp_image_clamped bind group layout"),
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
        label: Some("warp_image_clamped pipeline layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("warp_image_clamped pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    (bind_group_layout, pipeline)
}

fn dispatch_warp(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    width: u32,
    height: u32,
) {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("warp_image_clamped encoder"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, bind_group, &[]);
        cpass.dispatch_workgroups(
            (width as f32 / 16.0).ceil() as u32,
            (height as f32 / 16.0).ceil() as u32,
            1,
        );
    }
    queue.submit(Some(encoder.finish()));
}

/// Try to warp `src` on the GPU via the clamped 4-tap spline compute shader
/// (`warp.wgsl`).
///
/// Matches [`super::warp_image_clamped_cpu`]'s semantics within the
/// tolerance documented in the module docs.
///
/// # Fallback contract
/// Returns `Ok(None)` — never panics — when no `wgpu` context is available
/// or any GPU call fails; the caller ([`super::warp_image_clamped`]) falls
/// back to the CPU kernel in that case.
///
/// # Errors
/// Returns `Err` only for a non-finite/non-invertible `matrix` (validated
/// up front, before any GPU resource is touched) — the same input class the
/// CPU kernel itself rejects with `Err`.
// `too_many_lines`: linear wgpu resource-setup boilerplate (textures, explicit
// bind group layout, pipeline, dispatch, scoped-error handling) — splitting it
// would scatter the single validation-error scope across helpers and obscure
// the push/pop pairing the fallback contract depends on.
pub fn warp_image_clamped_gpu(
    src: &PlanarImage<f32>,
    matrix: &Matrix3<f32>,
) -> Result<Option<PlanarImage<f32>>, StackerError> {
    let Some((device, queue)) = stacker_core::gpu::context() else {
        return Ok(None);
    };

    if !matrix.iter().all(|v| v.is_finite()) {
        return Err(StackerError::MathError("non-finite matrix element".into()));
    }
    let m_inv = matrix
        .try_inverse()
        .ok_or_else(|| StackerError::MathError("non-invertible warp matrix".into()))?;

    let width = src.width as u32;
    let height = src.height as u32;
    if width == 0 || height == 0 {
        return Ok(Some(PlanarImage::new(src.width, src.height)));
    }

    // Hold the process-wide GPU dispatch lock for this whole
    // push/pop-error-scope critical section — `stacker_core::gpu::context`
    // hands out one shared `Device` to every caller across this workspace
    // (including `z-stackr-algo`'s GPU dispatch sites), and
    // `push_error_scope`/`pop_error_scope` form a per-device stack that
    // must never see two callers' scopes interleaved across threads. See
    // `stacker_core::gpu::dispatch_guard`'s docs for the full rationale.
    let _dispatch_guard = stacker_core::gpu::dispatch_guard();
    // Capture wgpu validation errors from every resource/pipeline/encoding
    // call below as a scope result instead of letting them reach the
    // process-global uncaptured handler — the fallback contract requires an
    // `Ok(None)` (CPU fallback), never a panic. Popped after the compute
    // submit, BEFORE the readback result is trusted (a failed dispatch
    // leaves the destination texture all-zero; mapping it back would
    // "succeed" and hand the caller a black frame).
    let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

    let packed_src = pack_rgba(src);
    let src_texture = upload_texture(
        device,
        queue,
        "warp_image_clamped src",
        width,
        height,
        &packed_src,
    );

    let dst_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("warp_image_clamped dst"),
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
        m_inv: [
            [m_inv[(0, 0)], m_inv[(0, 1)], m_inv[(0, 2)], 0.0],
            [m_inv[(1, 0)], m_inv[(1, 1)], m_inv[(1, 2)], 0.0],
            [m_inv[(2, 0)], m_inv[(2, 1)], m_inv[(2, 2)], 0.0],
        ],
        width,
        height,
        _padding: [0, 0],
    };

    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("warp_image_clamped uniforms"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let (bind_group_layout, pipeline) = create_warp_pipeline(device);

    let src_view = src_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let dst_view = dst_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("warp_image_clamped bind group"),
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

    dispatch_warp(device, queue, &pipeline, &bind_group, width, height);

    // The readback runs while the validation scope is still open, so its own
    // texture-to-buffer copy is covered too; the scope verdict is then
    // checked FIRST — see the comment at `push_error_scope` above for why a
    // readback result must never be trusted before the scope is clean.
    let readback = readback_texture(device, queue, &dst_texture, width, height);
    if let Some(err) = pollster::block_on(error_scope.pop()) {
        tracing::debug!(error = %err, "gpu: warp dispatch failed validation, falling back to CPU");
        return Ok(None);
    }
    match readback {
        Ok(dst) => Ok(Some(dst)),
        Err(err) => {
            tracing::debug!(error = %err, "gpu: warp readback failed, falling back to CPU");
            Ok(None)
        }
    }
}
