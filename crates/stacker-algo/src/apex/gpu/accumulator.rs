//! GPU-resident incremental `Apex` accumulator.
//!
//! A `wgpu` compute-shader counterpart of
//! [`crate::apex::fuse::ApexAccumulator`] that keeps the running per-level
//! accumulator textures resident on the GPU across
//! [`GpuApexAccumulator::accumulate`] calls, reading back to the CPU only in
//! [`GpuApexAccumulator::finish`] (or on a failure part-way through — see
//! below).
//!
//! # Why this exists
//!
//! The *batch* `apex::gpu::fuse_pyramids_gpu` shape uploads every source
//! pyramid on every call — a poor fit for the incremental whole-image
//! `ApexAccumulator` path (the GUI's in-RAM default), which folds frames in
//! one at a time. Instead this module keeps the *accumulator* resident on
//! the GPU and only uploads the one new per-frame pyramid each call — the
//! same traffic shape [`crate::apex::fuse::ApexAccumulator::blend`] already
//! has on the CPU (one accumulator, one incoming frame, no O(N) storage).
//!
//! # Fallback contract
//!
//! [`GpuApexAccumulator::new`] returns `None` when no `wgpu` context is
//! available (no adapter, or the runtime switch is off) — the caller falls
//! back to the plain CPU [`crate::apex::fuse::ApexAccumulator`] for the
//! *entire* stack in that case, decided once up front.
//!
//! Once a [`GpuApexAccumulator`] exists, [`GpuApexAccumulator::accumulate`]
//! can still fail mid-stack (a dispatch validation error, a lost device, a
//! mapping failure) — GPU compute failing after having already succeeded on
//! earlier frames is a materially different situation from never having a
//! context in the first place: reprocessing every already-accumulated frame
//! on the CPU would mean either (a) having kept every source frame in RAM
//! all along, defeating the whole memory-bounded point of this accumulator,
//! or (b) losing the frames already blended. Neither is acceptable, so on a
//! failure this module reads back the *exact accumulator state so far*
//! ([`GpuApexAccumulator::read_back`]) into a plain
//! [`crate::apex::fuse::ApexAccumulator`] (via
//! [`crate::apex::fuse::ApexAccumulator::from_gpu_state`]) and the caller
//! continues blending the remaining frames on the CPU from that exact
//! point — no frame is re-processed, none is lost, and the final result is
//! numerically the same running-accumulator computation, just finishing on
//! a different backend than it started on. This is a deliberate design
//! choice, not an accident: see [`crate::apex::fuse::fuse_pyramids_incremental_with_progress`]
//! for the call site that implements the hand-off.
//!
//! # Tolerance
//!
//! Tolerance-equal, not bit-equal, to the CPU accumulator — see
//! `crate::apex::gpu`'s module docs for why. `tests/gpu_apex_accumulator_parity.rs`
//! asserts a max-abs difference under `1e-3` after blending several
//! synthetic frames (including a width not divisible by 16, to exercise the
//! row-stride padding described in `crate::apex::gpu`'s module docs).
use crate::apex::{gpu::level_mode, pyramid::LaplacianPyramid};
use stacker_core::{gpu::padded_bytes_per_row, image::PlanarImage};
use wgpu::util::DeviceExt as _;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    width: u32,
    height: u32,
    use_color: u32,
    mode: u32,
    count: u32,
    _padding: [u32; 3],
}

/// One resident accumulator level: the GPU texture plus its dimensions
/// (kept alongside the texture so callers never need a `.size()` round
/// trip through `wgpu` to know a level's shape).
struct ResidentLevel {
    texture: wgpu::Texture,
    width: u32,
    height: u32,
}

/// GPU-resident incremental `Apex` accumulator. See the module docs for the
/// full design and fallback contract.
pub struct GpuApexAccumulator {
    levels: Vec<ResidentLevel>,
    count: usize,
    build_levels: usize,
    use_color: bool,
    grit_suppression: bool,
}

/// Pack one `PlanarImage<f32>` level into a row-major `RGBA32Float` buffer
/// (alpha unused, no row padding — the caller pads when uploading).
fn pack_level(level: &PlanarImage<f32>) -> Vec<f32> {
    let mut packed = vec![0.0f32; level.width * level.height * 4];
    for i in 0..level.width * level.height {
        packed[i * 4] = level.luma[i];
        packed[i * 4 + 1] = level.chroma_a[i];
        packed[i * 4 + 2] = level.chroma_b[i];
        packed[i * 4 + 3] = 1.0;
    }
    packed
}

/// Upload a `PlanarImage<f32>` level into a fresh `TEXTURE_BINDING |
/// COPY_DST` texture.
fn upload_level(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    level: &PlanarImage<f32>,
) -> wgpu::Texture {
    let width = level.width as u32;
    let height = level.height as u32;
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

    if width == 0 || height == 0 {
        return texture;
    }

    let packed = pack_level(level);
    let unpadded_bytes_per_row = width * 16;
    let padded_bytes = padded_bytes_per_row(unpadded_bytes_per_row);
    let src_bytes: &[u8] = bytemuck::cast_slice(&packed);
    let padded: Vec<u8> = if padded_bytes == unpadded_bytes_per_row {
        src_bytes.to_vec()
    } else {
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

/// Read a resident level texture back into a `PlanarImage<f32>`.
///
/// Returns `None` on any mapping failure (the caller treats that exactly
/// like any other mid-stack GPU failure).
fn readback_level(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    level: &ResidentLevel,
) -> Option<PlanarImage<f32>> {
    let (width, height) = (level.width, level.height);
    if width == 0 || height == 0 {
        return Some(PlanarImage::new(0, 0));
    }
    let unpadded_bytes_per_row = width * 16;
    let padded_bytes = padded_bytes_per_row(unpadded_bytes_per_row);

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("apex gpu accumulator readback buffer"),
        size: u64::from(padded_bytes) * u64::from(height),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &level.texture,
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

impl GpuApexAccumulator {
    /// Try to create a GPU-resident accumulator seeded with `first`.
    ///
    /// Builds a full-depth CPU Laplacian pyramid from `first` (pyramid
    /// construction — the reduce/expand chain — is not GPU-accelerated
    /// itself; only the per-frame accumulator UPDATE is, per this module's
    /// docs), then uploads each level into a resident texture.
    ///
    /// Returns `None` — never panics — when no `wgpu` context is available;
    /// the caller falls back to the plain CPU `ApexAccumulator` for the
    /// whole run in that case.
    #[must_use]
    pub fn new(first: &PlanarImage<f32>, use_color: bool, grit_suppression: bool) -> Option<Self> {
        let (device, queue) = stacker_core::gpu::context()?;
        let build_levels = crate::apex::fuse::FULL_DEPTH_PUB;
        let pyramid = LaplacianPyramid::build(first, build_levels);

        // Hold the process-wide GPU dispatch lock for this whole
        // push/pop-error-scope critical section — see
        // `stacker_core::gpu::dispatch_guard`'s docs for why concurrent
        // callers sharing the one process-wide `Device` must never
        // interleave error-scope push/pop pairs.
        let _dispatch_guard = stacker_core::gpu::dispatch_guard();
        let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let levels: Vec<ResidentLevel> = pyramid
            .levels
            .iter()
            .enumerate()
            .map(|(i, level)| ResidentLevel {
                texture: upload_level(
                    device,
                    queue,
                    &format!("apex gpu accumulator level {i}"),
                    level,
                ),
                width: level.width as u32,
                height: level.height as u32,
            })
            .collect();
        if pollster::block_on(error_scope.pop()).is_some() {
            return None;
        }

        Some(Self {
            levels,
            count: 1,
            build_levels,
            use_color,
            grit_suppression,
        })
    }

    /// Blend one more frame into the GPU-resident accumulator.
    ///
    /// Builds a full-depth CPU pyramid from `img` (same cost as the CPU
    /// accumulator's own `blend` — only the update step differs), uploads
    /// each level, and runs the per-level update compute shader
    /// (`accumulate.wgsl`) entirely on the GPU: the accumulator texture is
    /// read and a new texture is written, which then becomes the resident
    /// accumulator for the next call. Nothing is read back to the CPU here.
    ///
    /// Returns `true` on success. Returns `false` — never panics — on ANY
    /// GPU failure (a level's dispatch fails validation, or a resident
    /// texture's dimensions no longer match the new frame's pyramid, which
    /// should not happen in practice since all frames share one
    /// resolution). The caller must treat `false` as "read back now and
    /// continue on the CPU" — see the module docs' fallback contract.
    #[must_use]
    pub fn accumulate(&mut self, img: &PlanarImage<f32>) -> bool {
        let Some((device, queue)) = stacker_core::gpu::context() else {
            return false;
        };
        let pyramid = LaplacianPyramid::build(img, self.build_levels);
        if pyramid.levels.len() != self.levels.len() {
            return false;
        }

        let levels_count = self.levels.len();
        // Collect every level's new texture into a scratch `Vec` first and
        // only swap them into `self.levels` once EVERY level has succeeded.
        // Mutating `self.levels[idx]` as each level completes (the previous
        // behaviour) left the accumulator in a half-applied state on a
        // mid-loop failure: earlier levels would already reflect this frame
        // blended in while later levels (and `self.count`) would not, and
        // the mid-stack CPU hand-off (`ApexAccumulator::from_gpu_state`,
        // driven by `count`) would then blend this same frame into those
        // already-updated levels A SECOND TIME. Building the whole batch
        // before committing keeps `accumulate` atomic: either every level
        // (and `count`) advances together, or nothing changes and the
        // caller's read-back sees the exact pre-call state.
        let mut new_textures = Vec::with_capacity(levels_count);
        for (idx, src_level) in pyramid.levels.iter().enumerate() {
            let resident = &self.levels[idx];
            if src_level.width as u32 != resident.width
                || src_level.height as u32 != resident.height
            {
                return false;
            }
            let mode = level_mode(idx, levels_count, self.grit_suppression);
            let Some(new_texture) =
                self.update_level_gpu(device, queue, idx, src_level, mode.shader_mode())
            else {
                return false;
            };
            new_textures.push(new_texture);
        }

        for (idx, new_texture) in new_textures.into_iter().enumerate() {
            self.levels[idx].texture = new_texture;
        }
        self.count += 1;
        true
    }

    fn create_update_level_pipeline(
        device: &wgpu::Device,
    ) -> (wgpu::BindGroupLayout, wgpu::ComputePipeline) {
        let shader = device.create_shader_module(wgpu::include_wgsl!("accumulate.wgsl"));

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apex gpu accumulator bind group layout"),
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
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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
            label: Some("apex gpu accumulator pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apex gpu accumulator pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        (bind_group_layout, pipeline)
    }

    /// Run `accumulate.wgsl` for one level, returning the new resident
    /// texture (which the caller swaps into `self.levels[idx]`) or `None`
    /// on any failure.
    fn update_level_gpu(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        idx: usize,
        src_level: &PlanarImage<f32>,
        mode: u32,
    ) -> Option<wgpu::Texture> {
        let resident = &self.levels[idx];
        let (width, height) = (resident.width, resident.height);
        if width == 0 || height == 0 {
            return Some(upload_level(
                device,
                queue,
                "apex gpu accumulator empty level",
                src_level,
            ));
        }

        // See `Self::new`'s identical guard for why this whole
        // push/pop-error-scope critical section must be serialized across
        // threads sharing the one process-wide `Device`.
        let _dispatch_guard = stacker_core::gpu::dispatch_guard();
        let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

        let src_texture = upload_level(device, queue, "apex gpu accumulator src frame", src_level);

        let dst_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("apex gpu accumulator dst"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let uniforms = Uniforms {
            width,
            height,
            use_color: u32::from(self.use_color),
            mode,
            count: self.count as u32,
            _padding: [0, 0, 0],
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apex gpu accumulator uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let (bind_group_layout, pipeline) = Self::create_update_level_pipeline(device);

        let acc_view = resident
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let src_view = src_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let dst_view = dst_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apex gpu accumulator bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&acc_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&src_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&dst_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
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
        queue.submit(Some(encoder.finish()));

        if let Some(err) = pollster::block_on(error_scope.pop()) {
            tracing::debug!(error = %err, "gpu: apex accumulator dispatch failed validation, falling back to CPU");
            return None;
        }

        Some(dst_texture)
    }

    /// Read every resident level back into CPU `PlanarImage`s, without
    /// consuming `self` — used both by [`Self::finish`] and by the
    /// mid-stack failure hand-off documented in the module docs.
    ///
    /// Returns `None` — never panics — if any level's readback fails.
    #[must_use]
    pub fn read_back(&self) -> Option<Vec<PlanarImage<f32>>> {
        let (device, queue) = stacker_core::gpu::context()?;
        // No error scope here (`readback_level` doesn't push/pop one), but
        // the submit+poll sequence still shouldn't interleave with another
        // thread's push/dispatch/pop critical section on the same shared
        // `Device` — see `stacker_core::gpu::dispatch_guard`'s docs.
        let _dispatch_guard = stacker_core::gpu::dispatch_guard();
        self.levels
            .iter()
            .map(|level| readback_level(device, queue, level))
            .collect()
    }

    /// The number of frames blended so far (including the seed frame passed
    /// to [`Self::new`]).
    #[must_use]
    pub const fn count(&self) -> usize {
        self.count
    }

    /// The pyramid depth every accumulated frame was built with — needed by
    /// the CPU hand-off path to construct a CPU `ApexAccumulator` that
    /// builds subsequent frames' pyramids to the same depth.
    #[must_use]
    pub const fn build_levels(&self) -> usize {
        self.build_levels
    }

    /// Whether `use_color` was set for this accumulator.
    #[must_use]
    pub const fn use_color(&self) -> bool {
        self.use_color
    }

    /// Whether `grit_suppression` was set for this accumulator.
    #[must_use]
    pub const fn grit_suppression(&self) -> bool {
        self.grit_suppression
    }

    /// Read back every resident level and reconstruct the fused image,
    /// consuming `self`.
    ///
    /// Returns `None` — never panics — if the readback fails (should not
    /// happen immediately after a successful [`Self::accumulate`] run, but
    /// handled defensively all the same); the caller falls back to
    /// finishing on the CPU accumulator from the last successfully
    /// GPU-accumulated state in that case, exactly like a mid-stack
    /// `accumulate` failure.
    #[must_use]
    pub fn finish(self) -> Option<PlanarImage<f32>> {
        let levels = self.read_back()?;
        Some(LaplacianPyramid { levels }.reconstruct())
    }
}
