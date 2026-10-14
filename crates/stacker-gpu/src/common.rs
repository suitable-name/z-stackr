//! Buffers, pipelines and read-back shared by every plan.

use std::borrow::Cow;

use crate::{GpuError, context::GpuContext};

/// The largest buffer the plans may create: the smaller of the device's buffer
/// and storage-binding limits.
pub fn max_buffer(ctx: &GpuContext) -> u64 {
    ctx.limits()
        .max_buffer_size
        .min(u64::from(ctx.limits().max_storage_buffer_binding_size))
}

/// A storage buffer of `bytes` bytes that can be copied to and from, or
/// [`GpuError::TooLarge`] when the device does not allow one that big.
pub fn storage(ctx: &GpuContext, label: &str, bytes: u64) -> Result<wgpu::Buffer, GpuError> {
    let limit = max_buffer(ctx);
    if bytes > limit {
        return Err(GpuError::TooLarge {
            needed: bytes,
            limit,
        });
    }
    Ok(ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        // wgpu rejects empty bindings.
        size: bytes.max(16),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }))
}

/// A uniform buffer of `bytes` bytes the host writes before each dispatch.
pub fn uniform(ctx: &GpuContext, label: &str, bytes: u64) -> wgpu::Buffer {
    ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// A buffer the host maps to read `bytes` bytes back.
pub fn readback(ctx: &GpuContext, label: &str, bytes: u64) -> wgpu::Buffer {
    ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.max(16),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// The compute pipelines for `entries` of the WGSL `source`, each with the
/// bind-group layout derived from the bindings it uses.
pub fn pipelines(
    ctx: &GpuContext,
    label: &str,
    source: String,
    entries: &[&str],
) -> Vec<wgpu::ComputePipeline> {
    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(source)),
        });
    entries
        .iter()
        .map(|name| {
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(name),
                    layout: None,
                    module: &module,
                    entry_point: Some(name),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                })
        })
        .collect()
}

/// A bind group for `pipeline` with `buffers` at bindings `0..`, in order.
pub fn bind(
    ctx: &GpuContext,
    pipeline: &wgpu::ComputePipeline,
    buffers: &[(u32, &wgpu::Buffer)],
) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry<'_>> = buffers
        .iter()
        .map(|&(binding, buffer)| wgpu::BindGroupEntry {
            binding,
            resource: buffer.as_entire_binding(),
        })
        .collect();
    ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    })
}

/// Fail with the device's error if it reported one since the last check.
pub fn check(ctx: &GpuContext) -> Result<(), GpuError> {
    ctx.take_error()
        .map_or(Ok(()), |message| Err(GpuError::Device(message)))
}

/// Copy `bytes` bytes of `source` into `staging` (a [`readback`] buffer at least
/// that large), wait, and return them as `T`s. Checks the device's error sink
/// afterwards, so a result that comes back is one the device did not complain
/// about.
///
/// # Errors
/// [`GpuError::ReadBack`] when the mapping fails, [`GpuError::Device`] when the
/// device reported an error.
///
/// # Panics
/// Panics when the device's mapped memory is not aligned for `T`.
#[cfg(not(target_arch = "wasm32"))]
pub fn read<T: bytemuck::Pod>(
    ctx: &GpuContext,
    encoder: wgpu::CommandEncoder,
    source: &wgpu::Buffer,
    staging: &wgpu::Buffer,
    bytes: u64,
) -> Result<Vec<T>, GpuError> {
    read_at(ctx, encoder, (source, 0), staging, bytes)
}

/// [`read`] of the `bytes` bytes of `source.0` that start at byte offset
/// `source.1` (a multiple of 4).
///
/// # Errors
/// As [`read`].
///
/// # Panics
/// As [`read`].
#[cfg(not(target_arch = "wasm32"))]
pub fn read_at<T: bytemuck::Pod>(
    ctx: &GpuContext,
    encoder: wgpu::CommandEncoder,
    source: (&wgpu::Buffer, u64),
    staging: &wgpu::Buffer,
    bytes: u64,
) -> Result<Vec<T>, GpuError> {
    let mut encoder = encoder;
    encoder.copy_buffer_to_buffer(source.0, source.1, staging, 0, bytes);
    ctx.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    let _ = ctx.device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| GpuError::ReadBack(e.to_string()))?
        .map_err(|e| GpuError::ReadBack(e.to_string()))?;
    let values = {
        let mapped = slice.get_mapped_range();
        bytemuck::try_cast_slice::<u8, T>(&mapped)
            .expect("mapped memory is aligned for the element type")
            .to_vec()
    };
    staging.unmap();
    check(ctx)?;
    Ok(values)
}

/// [`read`] without blocking the caller: the copy is submitted, the mapping is
/// requested with a callback that completes a oneshot channel, and the future
/// waits on it.
///
/// On native targets the device is polled once after the request, which blocks
/// until the mapping is done (the channel is then ready); on `wasm32` the browser
/// completes it and the future simply yields until then. The buffer is unmapped
/// before returning and the device's error sink is checked, as in [`read`].
///
/// # Errors
/// [`GpuError::ReadBack`] when the mapping fails or is dropped,
/// [`GpuError::Device`] when the device reported an error.
///
/// # Panics
/// Panics when `bytes` is not a multiple of the size of `T`.
#[cfg_attr(
    target_arch = "wasm32",
    expect(
        clippy::future_not_send,
        reason = "WebGPU handles are not Send; the worker is single-threaded"
    )
)]
pub async fn read_async<T: bytemuck::Pod>(
    ctx: &GpuContext,
    encoder: wgpu::CommandEncoder,
    source: &wgpu::Buffer,
    staging: &wgpu::Buffer,
    bytes: u64,
) -> Result<Vec<T>, GpuError> {
    let mut encoder = encoder;
    encoder.copy_buffer_to_buffer(source, 0, staging, 0, bytes);
    ctx.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    let (tx, rx) = futures_channel::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    // A no-op in the browser, which drives the mapping itself.
    let _ = ctx.device.poll(wgpu::Maintain::Wait);
    rx.await
        .map_err(|e| GpuError::ReadBack(e.to_string()))?
        .map_err(|e| GpuError::ReadBack(e.to_string()))?;
    // Copied element-wise: the browser hands out a byte vector whose alignment is
    // not promised.
    let values = {
        let mapped = slice.get_mapped_range();
        bytemuck::pod_collect_to_vec::<u8, T>(&mapped)
    };
    staging.unmap();
    check(ctx)?;
    Ok(values)
}

/// A command encoder.
pub fn encoder(ctx: &GpuContext, label: &str) -> wgpu::CommandEncoder {
    ctx.device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) })
}

/// Record one dispatch of `pipeline` with `bind_group` over `grid` workgroups.
pub fn dispatch(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    grid: [u32; 3],
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(grid[0], grid[1], grid[2]);
}

/// The mean of about 64k evenly spaced samples of `plane`, used as the constant
/// the kernels subtract from values before summing windows of them: any constant
/// leaves a variance or covariance unchanged, one near the typical value keeps
/// the `f32` window sums small.
#[cfg(not(target_arch = "wasm32"))]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "a shift needs no more than f32 precision; sample counts are far below 2^52"
)]
pub fn plane_mean(plane: &[f32]) -> f32 {
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

/// `value` as a `u32`.
///
/// # Panics
/// Panics when it does not fit (an image dimension above `u32::MAX`).
pub fn u32_of(value: usize) -> u32 {
    u32::try_from(value).expect("an image dimension fits in u32")
}

/// Workgroups of `group` threads covering `extent`.
pub fn groups(extent: usize, group: u32) -> u32 {
    u32_of(extent).div_ceil(group)
}
