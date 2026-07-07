//! GPU-accelerated Relief primitives.
//!
//! A `wgpu` compute-shader port of [`super::guided::guided_filter`] (used by
//! both Relief's Guided-Filter engine and Strata's per-frame weight
//! refinement) and of [`super::multigrid::MultigridLayer::relax`] (the
//! Jacobi-style relaxation sweep at the core of the Multigrid V-cycle).
//!
//! # Fallback contract
//!
//! Both [`guided_filter_gpu`] and [`relax_gpu`] return `None` (never panic)
//! whenever the GPU path cannot run: no `wgpu` adapter/device available (or
//! the runtime switch is off — see `stacker_core::gpu::context`), or any
//! `wgpu` call along the way fails. Their CPU callers
//! ([`super::guided::guided_filter`], [`super::multigrid::MultigridLayer::relax`])
//! treat `None` as "try the GPU, it declined" and fall back to the pure-CPU
//! implementation, which remains fully reachable — this is the only code
//! path exercised by a default (non-`gpu`) build.
//!
//! # Scope
//!
//! [`guided_filter_gpu`] accelerates the WHOLE guided-filter pipeline (see
//! its own docs for why per-call `box_filter` GPU dispatch was removed in
//! favour of this fused version). `super::guided::box_filter` itself is
//! ALWAYS the CPU/SAT path now, including `strata::mod`'s direct call for
//! its base/detail split (see that module's docs) — only `guided_filter`'s
//! six internal box-mean steps are covered, via this module.
//!
//! [`super::multigrid::MultigridLayer::restrict_to`]/[`prolong_from`] are
//! NOT GPU-accelerated: both are already cheap (a single pass, no
//! neighbourhood loop) relative to [`relax`](super::multigrid::MultigridLayer::relax)
//! (which runs 3 times per V-cycle level, recursively, dominating the
//! solver's cost), and their recursive borrow-splitting call structure
//! (`MultigridSolver::cycle`) does not map onto a per-dispatch batch kernel
//! as directly as a single layer's relaxation sweep does. See
//! `docs/gpu_acceleration_summary.md`'s "Implementation status" section.
//!
//! # Tolerance
//!
//! GPU output is **tolerance-equal, not bit-equal**, to the CPU path, for
//! the same reasons documented in `crate::apex::gpu`'s module docs — see
//! `tests/gpu_relief_parity.rs`'s max-abs-diff assertions (`< 1e-3`) on
//! synthetic inputs (including a width not divisible by 16, to exercise the
//! row-stride padding described below), at both a tight radius (6) and a
//! Strata-like wide radius (45).
//!
//! # Row-stride padding
//!
//! [`guided_filter_gpu`] uses textures (see
//! `stacker_core::gpu::padded_bytes_per_row`'s docs for the 256-byte
//! row-stride requirement this pads for, same as every other texture-based
//! GPU module in this workspace) only at the two edges of the pipeline: the
//! initial upload of the guidance/src luma planes and the final readback of
//! `q`. Every intermediate ping-pong pass stays texture-to-texture on the
//! GPU with no row-stride concerns (those only apply to
//! `wgpu::Buffer`-backed linear readback/upload). [`relax_gpu`] instead uses
//! plain `wgpu::Buffer` storage buffers (no texture, no row-stride
//! requirement at all — `MultigridLayer`'s `a`/`b`/`c` are already flat
//! `Vec<f32>`s with no channel packing to do), so no padding is needed
//! there either.
use std::sync::OnceLock;

use stacker_core::{gpu::padded_bytes_per_row, image::PlanarImage};
use wgpu::util::DeviceExt as _;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct BoxFilterUniforms {
    width: u32,
    height: u32,
    radius: u32,
    mode: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ElementwiseUniforms {
    width: u32,
    height: u32,
    mode: u32,
    _padding: u32,
    eps: f32,
}

/// Device-resident, size-independent GPU objects for the fused guided-filter
/// pipeline: shader modules, bind-group layouts, and compute pipelines.
/// None of these depend on the image dimensions of any particular
/// [`guided_filter_gpu`] call — only the bind groups (built fresh per call,
/// since they reference that call's specific textures) and the textures
/// themselves vary by size.
///
/// # Caching — why a process-wide `OnceLock`
///
/// Building a `wgpu::ShaderModule` (which compiles WGSL to the backend's
/// native shader IR) and a `wgpu::ComputePipeline` (which links that module
/// against a bind-group layout) is measurably expensive relative to a single
/// guided-filter dispatch — before this cache existed, every
/// [`guided_filter_gpu`] call recompiled both shaders and relinked both
/// pipelines from scratch, on top of the actual compute work, which is the
/// per-frame GPU utilization sawtooth this cache exists to smooth: a burst
/// of shader-compile CPU/driver work at the start of every single call,
/// serialized with the rest of the pipeline via `dispatch_guard`.
///
/// `stacker_core::gpu::context()` hands out a `&'static (Device, Queue)` —
/// the process-wide `wgpu` context lives for the process's entire lifetime
/// (see that module's docs) — so a `OnceLock<CachedGuidedFilterPipelines>`
/// built from that same `&'static Device` is sound: the pipelines borrow
/// nothing that can be dropped or invalidated before the process exits, and
/// there is exactly one `Device` in the process to cache against (no
/// per-device keying needed).
struct CachedGuidedFilterPipelines {
    box_pipeline: wgpu::ComputePipeline,
    box_bind_group_layout: wgpu::BindGroupLayout,
    elementwise_pipeline: wgpu::ComputePipeline,
    elementwise_bind_group_layout: wgpu::BindGroupLayout,
}

impl CachedGuidedFilterPipelines {
    // `too_many_lines`: linear wgpu resource-setup boilerplate (two full
    // bind-group-layout/pipeline-layout/pipeline triples) — same rationale
    // as `box_filter_pass_gpu`'s identical allow before this rewrite.
    #[allow(clippy::too_many_lines)]
    fn new(device: &wgpu::Device) -> Self {
        let box_shader = device.create_shader_module(wgpu::include_wgsl!("box_filter.wgsl"));
        let box_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("relief guided_filter_gpu box bind group layout"),
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
        let box_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("relief guided_filter_gpu box pipeline layout"),
            bind_group_layouts: &[Some(&box_bind_group_layout)],
            immediate_size: 0,
        });
        let box_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("relief guided_filter_gpu box pipeline"),
            layout: Some(&box_pipeline_layout),
            module: &box_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let elementwise_shader =
            device.create_shader_module(wgpu::include_wgsl!("elementwise.wgsl"));
        let elementwise_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("relief guided_filter_gpu elementwise bind group layout"),
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
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
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
        let elementwise_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("relief guided_filter_gpu elementwise pipeline layout"),
                bind_group_layouts: &[Some(&elementwise_bind_group_layout)],
                immediate_size: 0,
            });
        let elementwise_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("relief guided_filter_gpu elementwise pipeline"),
                layout: Some(&elementwise_pipeline_layout),
                module: &elementwise_shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        Self {
            box_pipeline,
            box_bind_group_layout,
            elementwise_pipeline,
            elementwise_bind_group_layout,
        }
    }
}

/// Process-wide cache: built once from the first call's `&'static Device`,
/// reused by every subsequent [`guided_filter_gpu`] call regardless of the
/// image size it was invoked with. See
/// [`CachedGuidedFilterPipelines`]'s doc comment for the soundness argument.
static GUIDED_FILTER_PIPELINES: OnceLock<CachedGuidedFilterPipelines> = OnceLock::new();

/// Per-dispatch GPU "context" for the fused guided-filter pipeline: the
/// device/queue/textures are per-call (sizes vary between calls), but the
/// shader/pipeline/bind-group-layout objects are borrowed from the
/// process-wide [`GUIDED_FILTER_PIPELINES`] cache instead of being rebuilt
/// on every call.
struct GuidedFilterGpu<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    width: u32,
    height: u32,
    pipelines: &'static CachedGuidedFilterPipelines,
}

impl<'a> GuidedFilterGpu<'a> {
    fn new(device: &'a wgpu::Device, queue: &'a wgpu::Queue, width: u32, height: u32) -> Self {
        let pipelines =
            GUIDED_FILTER_PIPELINES.get_or_init(|| CachedGuidedFilterPipelines::new(device));
        Self {
            device,
            queue,
            width,
            height,
            pipelines,
        }
    }

    /// Allocate a resident `Rgba32Float` texture usable both as a compute
    /// shader's `texture_2d<f32>` read source and (via a second binding
    /// elsewhere) a `texture_storage_2d` write target across the pipeline's
    /// several passes.
    fn make_texture(&self, label: &str) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    }

    /// Upload a flat `width*height` `f32` plane into a freshly-allocated
    /// resident texture (single channel used; the other three are padding,
    /// same convention as `box_filter_gpu`/`strata::gpu` before it).
    fn upload(&self, data: &[f32], label: &str) -> wgpu::Texture {
        let texture = self.make_texture(label);
        let mut packed = vec![0.0f32; self.width as usize * self.height as usize * 4];
        for i in 0..data.len() {
            packed[i * 4] = data[i];
        }
        let unpadded_bytes_per_row = self.width * 16;
        let padded_bytes = padded_bytes_per_row(unpadded_bytes_per_row);
        let src_bytes: &[u8] = bytemuck::cast_slice(&packed);
        let padded_src: Vec<u8> = if padded_bytes == unpadded_bytes_per_row {
            src_bytes.to_vec()
        } else {
            let mut out = vec![0u8; (padded_bytes * self.height) as usize];
            for row in 0..self.height as usize {
                let src_off = row * unpadded_bytes_per_row as usize;
                let dst_off = row * padded_bytes as usize;
                out[dst_off..dst_off + unpadded_bytes_per_row as usize].copy_from_slice(
                    &src_bytes[src_off..src_off + unpadded_bytes_per_row as usize],
                );
            }
            out
        };
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &padded_src,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes),
                rows_per_image: Some(self.height),
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        texture
    }

    /// One direction (horizontal `mode=0` or vertical `mode=1`) of the
    /// separable box-filter pass, texture-to-texture — no CPU round trip.
    /// See `box_filter.wgsl`'s module doc comment for the exact clipped-
    /// window normalisation semantics this preserves.
    fn box_pass(&self, src: &wgpu::Texture, radius: u32, mode: u32, label: &str) -> wgpu::Texture {
        let dst = self.make_texture(label);
        let uniforms = BoxFilterUniforms {
            width: self.width,
            height: self.height,
            radius,
            mode,
        };
        let uniform_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("relief guided_filter_gpu box uniforms"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let src_view = src.create_view(&wgpu::TextureViewDescriptor::default());
        let dst_view = dst.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("relief guided_filter_gpu box bind group"),
            layout: &self.pipelines.box_bind_group_layout,
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
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cpass.set_pipeline(&self.pipelines.box_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(
                (self.width as f32 / 16.0).ceil() as u32,
                (self.height as f32 / 16.0).ceil() as u32,
                1,
            );
        }
        self.queue.submit(Some(encoder.finish()));
        dst
    }

    /// Full separable box-mean of `src` (horizontal pass then vertical
    /// pass), texture-to-texture.
    fn box_filter(&self, src: &wgpu::Texture, radius: u32, label: &str) -> wgpu::Texture {
        let horizontal = self.box_pass(src, radius, 0, &format!("{label} h"));
        self.box_pass(&horizontal, radius, 1, &format!("{label} v"))
    }

    /// One elementwise dispatch (see `elementwise.wgsl`'s module docs for
    /// the three modes). `inputs` are padded with a dummy read-only texture
    /// when fewer than 4 are needed (mode 0/2 only read 2/3), and likewise
    /// `outputs` with a dummy write-only texture when fewer than 2 are
    /// needed (mode 2 only writes `out0`) — see the two distinct dummy
    /// textures allocated in [`guided_filter_gpu`] and why one shared dummy
    /// does not work.
    #[allow(clippy::too_many_arguments)]
    fn elementwise(
        &self,
        inputs: [&wgpu::Texture; 4],
        outputs: [&wgpu::Texture; 2],
        mode: u32,
        eps: f32,
        label: &str,
    ) {
        let uniforms = ElementwiseUniforms {
            width: self.width,
            height: self.height,
            mode,
            _padding: 0,
            eps,
        };
        let uniform_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("relief guided_filter_gpu elementwise uniforms"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let input_views: Vec<wgpu::TextureView> = inputs
            .iter()
            .map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()))
            .collect();
        let output_views: Vec<wgpu::TextureView> = outputs
            .iter()
            .map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()))
            .collect();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &self.pipelines.elementwise_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&input_views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&input_views[1]),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&input_views[2]),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&input_views[3]),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&output_views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&output_views[1]),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cpass.set_pipeline(&self.pipelines.elementwise_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(
                (self.width as f32 / 16.0).ceil() as u32,
                (self.height as f32 / 16.0).ceil() as u32,
                1,
            );
        }
        self.queue.submit(Some(encoder.finish()));
    }

    /// Read a single-channel texture back to a flat `width*height` `Vec<f32>`.
    fn read_back(&self, texture: &wgpu::Texture) -> Option<Vec<f32>> {
        let unpadded_bytes_per_row = self.width * 16;
        let padded_bytes = padded_bytes_per_row(unpadded_bytes_per_row);
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("relief guided_filter_gpu readback buffer"),
            size: u64::from(padded_bytes) * u64::from(self.height),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
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
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = output_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            let _ = tx.send(v);
        });
        if self
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .is_err()
        {
            return None;
        }
        match rx.recv() {
            Ok(Ok(())) => {}
            _ => return None,
        }
        let Ok(data) = buffer_slice.get_mapped_range() else {
            return None;
        };
        let width = self.width as usize;
        let height = self.height as usize;
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
}

/// Try to run the ENTIRE [`super::guided::guided_filter`] pipeline on the
/// GPU in one fused dispatch sequence, returning the resulting luma plane
/// (`q` in the CPU reference).
///
/// Unlike the guided filter's old per-`box_filter`-call GPU dispatch (twelve
/// serialized upload/dispatch/readback round trips per `guided_filter`
/// call — six box-mean steps, each a separate two-pass GPU op — see
/// `super::guided::box_filter`'s doc comment for the full history), this
/// function holds [`stacker_core::gpu::dispatch_guard`] and the validation
/// error scope for the WHOLE pipeline, uploads only the two input planes
/// (`guidance.luma`, `src.luma`) once, keeps every intermediate (`mean_I`,
/// `mean_p`, `I*p`, `I*I`, `mean_Ip`, `mean_Ii`, `a`, `b`, `mean_a`,
/// `mean_b`) resident on the GPU as textures, and reads back only the final
/// `q` plane — ONE round trip total, mirroring the CPU algorithm's data
/// flow exactly:
///
/// 1. `mean_I = box(I)`, `mean_p = box(p)`
/// 2. `I*p`, `I*I` (elementwise, mode 0)
/// 3. `mean_Ip = box(I*p)`, `mean_Ii = box(I*I)`
/// 4. `a`, `b` (elementwise, mode 1) — `a = (mean_Ip - mean_I*mean_p) / (max(mean_Ii - mean_I*mean_I, 0.0) + eps)`, `b = mean_p - a*mean_I`
/// 5. `mean_a = box(a)`, `mean_b = box(b)`
/// 6. `q = mean_a*I + mean_b` (elementwise, mode 2)
///
/// # Numerical precision
///
/// Every step above runs in `f32` on the GPU, including the box-mean
/// passes: unlike the CPU reference's `f64`-accumulated summed-area table
/// (see `super::guided::box_filter`'s doc comment), the GPU box-filter pass
/// is the direct O(radius) *separable* two-pass window sum from
/// `box_filter.wgsl` — a plain `f32` accumulation over at most `2*radius+1`
/// taps per pass (not the full `(2*radius+1)²` window at once, since the
/// two 1-D passes compose). Accumulating at most ~2*45+1 = 91 taps (the
/// widest radius exercised, Strata's `R_BIG`) of values in `0..1` in `f32`
/// keeps rounding error far below the per-window signal even before the
/// second (vertical) pass renormalises again — nowhere near the
/// magnitude/tap-count where `f32` accumulation becomes numerically
/// dangerous (that failure mode needs sums reaching ~1e6-1e7, per the CPU
/// doc comment; a few dozen taps of `0..1` values never gets there). The
/// parity tests in `tests/gpu_relief_parity.rs` confirm this empirically at
/// both a tight radius (6, Relief's typical `smooth_radius`) and a wide
/// radius (45, Strata's `R_BIG`), at `eps` values `0.01` and `0.3` (Strata's
/// `EPS_BIG`), all within the workspace's standard `1e-3` max-abs-diff
/// tolerance.
///
/// # Fallback contract
/// Returns `None` — never panics — when no `wgpu` context is available or
/// any dispatch fails; [`super::guided::guided_filter`] falls back to its
/// CPU body (six `box_filter` calls, each CPU/SAT) in that case.
#[must_use]
pub fn guided_filter_gpu(
    guidance: &PlanarImage<f32>,
    src: &PlanarImage<f32>,
    radius: usize,
    eps: f32,
) -> Option<Vec<f32>> {
    // `similar_names`: the guided-filter algorithm's own variable names are
    // inherently one-letter/one-suffix apart (`I`/`p`, `I*p`/`I*I`, `mean_Ip`/
    // `mean_Ii`) — this mirrors the CPU reference's `mean_i_p`/`mean_i_i` in
    // `super::guided::guided_filter` exactly, so diverging the names here would
    // make the two implementations harder to compare side by side, not easier
    // to read.
    let (device, queue) = stacker_core::gpu::context()?;
    let width = guidance.width;
    let height = guidance.height;
    if width == 0 || height == 0 {
        return Some(Vec::new());
    }
    debug_assert_eq!(guidance.luma.len(), width * height);
    debug_assert_eq!(src.luma.len(), width * height);

    let width_u32 = width as u32;
    let height_u32 = height as u32;
    let radius_u32 = radius as u32;

    // Hold the process-wide GPU dispatch lock for the WHOLE fused pipeline
    // (every pass below) — see `stacker_core::gpu::dispatch_guard`'s docs.
    // This is the entire point of the fused rewrite: the old per-`box_filter`
    // dispatch acquired/released this lock six separate times (twelve
    // sub-dispatches) per `guided_filter` call; this version acquires it
    // once for all of them.
    let _dispatch_guard = stacker_core::gpu::dispatch_guard();
    let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

    let ctx = GuidedFilterGpu::new(device, queue, width_u32, height_u32);

    // Dummy textures bound to unused input/output slots in modes that don't
    // need all 4 inputs / 2 outputs (see `elementwise.wgsl`'s docs). TWO
    // separate dummy textures are needed, not one: within a single compute
    // dispatch, wgpu's usage-scope validation treats `STORAGE_WRITE_ONLY`
    // as an EXCLUSIVE usage that cannot share a texture with any other
    // usage (including a plain read-only `TEXTURE_BINDING`) bound
    // elsewhere in the same bind group — reusing one dummy texture for
    // both an unused read slot and an unused write slot in the mode-2
    // dispatch below ("Attempted to use Texture ... with conflicting
    // usages") fails validation even though neither slot's contents are
    // ever actually read or written by that mode.
    let dummy_read = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("relief guided_filter_gpu dummy read"),
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
    let dummy_write = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("relief guided_filter_gpu dummy write"),
        size: wgpu::Extent3d {
            width: width_u32,
            height: height_u32,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });

    let guidance_tex = ctx.upload(&guidance.luma, "relief guided_filter_gpu I");
    let src_tex = ctx.upload(&src.luma, "relief guided_filter_gpu p");

    // Step 1: mean_I, mean_p.
    let mean_guidance =
        ctx.box_filter(&guidance_tex, radius_u32, "relief guided_filter_gpu mean_I");
    let mean_src = ctx.box_filter(&src_tex, radius_u32, "relief guided_filter_gpu mean_p");

    // Step 2: I*p, I*I (mode 0).
    let prod_image_p_tex = ctx.make_texture("relief guided_filter_gpu I*p");
    let prod_image_image_tex = ctx.make_texture("relief guided_filter_gpu I*I");
    ctx.elementwise(
        [&guidance_tex, &src_tex, &dummy_read, &dummy_read],
        [&prod_image_p_tex, &prod_image_image_tex],
        0,
        eps,
        "relief guided_filter_gpu products bind group",
    );

    // Step 3: mean_Ip, mean_Ii.
    let mean_prod_image_p = ctx.box_filter(
        &prod_image_p_tex,
        radius_u32,
        "relief guided_filter_gpu mean_Ip",
    );
    let mean_prod_image_image = ctx.box_filter(
        &prod_image_image_tex,
        radius_u32,
        "relief guided_filter_gpu mean_Ii",
    );

    // Step 4: a, b (mode 1).
    let a_tex = ctx.make_texture("relief guided_filter_gpu a");
    let b_tex = ctx.make_texture("relief guided_filter_gpu b");
    ctx.elementwise(
        [
            &mean_guidance,
            &mean_src,
            &mean_prod_image_p,
            &mean_prod_image_image,
        ],
        [&a_tex, &b_tex],
        1,
        eps,
        "relief guided_filter_gpu coeffs bind group",
    );

    // Step 5: mean_a, mean_b.
    let mean_a = ctx.box_filter(&a_tex, radius_u32, "relief guided_filter_gpu mean_a");
    let mean_b = ctx.box_filter(&b_tex, radius_u32, "relief guided_filter_gpu mean_b");

    // Step 6: q = mean_a*I + mean_b (mode 2). Only out0 is meaningful.
    let q_tex = ctx.make_texture("relief guided_filter_gpu q");
    ctx.elementwise(
        [&mean_a, &mean_b, &guidance_tex, &dummy_read],
        [&q_tex, &dummy_write],
        2,
        eps,
        "relief guided_filter_gpu final bind group",
    );

    if let Some(err) = pollster::block_on(error_scope.pop()) {
        tracing::debug!(error = %err, "gpu: relief guided_filter_gpu dispatch failed validation, falling back to CPU");
        return None;
    }

    ctx.read_back(&q_tex)
}

/// Run [`super::guided::guided_filter`]'s whole pipeline TWICE against the
/// SAME `guidance`/`src` pair, at two different `(radius, eps)` combinations.
///
/// This is the shape
/// — the shape of Strata's Pass 2 (design doc §2 Step 4: `w_base` at
/// `(R_BIG, EPS_BIG)`, `w_detail` at `(R_SMALL, EPS_SMALL)`, both against the
/// same per-frame `(frame, p_image)` pair) — as ONE held dispatch, uploading
/// `guidance`/`src` only once instead of twice.
///
/// Returns `(luma_for_radius_a, luma_for_radius_b)`, `None` on any failure
/// (same fallback contract as [`guided_filter_gpu`]) — callers fall back to
/// two independent CPU `guided_filter` calls (or, per
/// [`super::guided::guided_filter_pair`]'s CPU path, a SAT-sharing pair) in
/// that case.
///
/// # Why this is worth a dedicated entry point
///
/// [`guided_filter_gpu`] already holds [`stacker_core::gpu::dispatch_guard`]
/// for one full pipeline. Calling it twice in a row (once per radius) is
/// correct but pays the guidance/src upload cost twice and releases/
/// re-acquires the dispatch guard between the two calls — exactly the
/// "dual-radius fused weights call" cost this function exists to remove:
/// one upload of `I`/`p`, both radius chains dispatched back-to-back under
/// ONE held guard, two readbacks batched at the very end (instead of one
/// readback per call interleaved with the second call's upload).
///
/// # Fallback contract
/// Returns `None` — never panics — under the same conditions as
/// [`guided_filter_gpu`] (no context, or any dispatch in either chain
/// fails validation).
#[must_use]
pub fn guided_filter_pair_gpu(
    guidance: &PlanarImage<f32>,
    src: &PlanarImage<f32>,
    radius_a: usize,
    eps_a: f32,
    radius_b: usize,
    eps_b: f32,
) -> Option<(Vec<f32>, Vec<f32>)> {
    let (device, queue) = stacker_core::gpu::context()?;
    let width = guidance.width;
    let height = guidance.height;
    if width == 0 || height == 0 {
        return Some((Vec::new(), Vec::new()));
    }
    debug_assert_eq!(guidance.luma.len(), width * height);
    debug_assert_eq!(src.luma.len(), width * height);

    let width_u32 = width as u32;
    let height_u32 = height as u32;

    // One held dispatch guard for BOTH radius chains — see this function's
    // doc comment for why that is the entire point.
    let _dispatch_guard = stacker_core::gpu::dispatch_guard();
    let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

    let ctx = GuidedFilterGpu::new(device, queue, width_u32, height_u32);

    let dummy_read = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("relief guided_filter_pair_gpu dummy read"),
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
    let dummy_write = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("relief guided_filter_pair_gpu dummy write"),
        size: wgpu::Extent3d {
            width: width_u32,
            height: height_u32,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });

    // ONE upload of I and p, shared by both radius chains below.
    let guidance_tex = ctx.upload(&guidance.luma, "relief guided_filter_pair_gpu I");
    let src_tex = ctx.upload(&src.luma, "relief guided_filter_pair_gpu p");

    // I*p, I*I do not depend on radius/eps either — computed once, reused
    // by both chains' Step 3 box-means.
    let prod_image_p_tex = ctx.make_texture("relief guided_filter_pair_gpu I*p");
    let prod_image_image_tex = ctx.make_texture("relief guided_filter_pair_gpu I*I");
    ctx.elementwise(
        [&guidance_tex, &src_tex, &dummy_read, &dummy_read],
        [&prod_image_p_tex, &prod_image_image_tex],
        0,
        eps_a,
        "relief guided_filter_pair_gpu products bind group",
    );

    // One full radius/eps chain (Steps 1, 3, 4, 5, 6 of the fused
    // pipeline — Step 2's products are shared, computed once above).
    let run_chain = |radius: usize, eps: f32, label: &str| -> wgpu::Texture {
        let radius_u32 = radius as u32;
        let mean_guidance = ctx.box_filter(&guidance_tex, radius_u32, &format!("{label} mean_I"));
        let mean_src = ctx.box_filter(&src_tex, radius_u32, &format!("{label} mean_p"));
        let mean_prod_image_p =
            ctx.box_filter(&prod_image_p_tex, radius_u32, &format!("{label} mean_Ip"));
        let mean_prod_image_image = ctx.box_filter(
            &prod_image_image_tex,
            radius_u32,
            &format!("{label} mean_Ii"),
        );

        let a_tex = ctx.make_texture(&format!("{label} a"));
        let b_tex = ctx.make_texture(&format!("{label} b"));
        ctx.elementwise(
            [
                &mean_guidance,
                &mean_src,
                &mean_prod_image_p,
                &mean_prod_image_image,
            ],
            [&a_tex, &b_tex],
            1,
            eps,
            &format!("{label} coeffs bind group"),
        );

        let mean_a = ctx.box_filter(&a_tex, radius_u32, &format!("{label} mean_a"));
        let mean_b = ctx.box_filter(&b_tex, radius_u32, &format!("{label} mean_b"));

        let q_tex = ctx.make_texture(&format!("{label} q"));
        ctx.elementwise(
            [&mean_a, &mean_b, &guidance_tex, &dummy_read],
            [&q_tex, &dummy_write],
            2,
            eps,
            &format!("{label} final bind group"),
        );
        q_tex
    };

    let q_a = run_chain(radius_a, eps_a, "relief guided_filter_pair_gpu chain a");
    let q_b = run_chain(radius_b, eps_b, "relief guided_filter_pair_gpu chain b");

    if let Some(err) = pollster::block_on(error_scope.pop()) {
        tracing::debug!(error = %err, "gpu: relief guided_filter_pair_gpu dispatch failed validation, falling back to CPU");
        return None;
    }

    // Two readbacks batched at the end, after both chains have been fully
    // dispatched — not interleaved with the second chain's dispatch.
    let out_a = ctx.read_back(&q_a)?;
    let out_b = ctx.read_back(&q_b)?;
    Some((out_a, out_b))
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct RelaxUniforms {
    width: u32,
    height: u32,
}

/// Device-resident, size-independent GPU objects for the Multigrid
/// relaxation sweep — same rationale as [`CachedGuidedFilterPipelines`]
/// above, just for [`relax_gpu`]'s single pipeline instead of the guided
/// filter's two.
struct CachedRelaxPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl CachedRelaxPipeline {
    fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::include_wgsl!("relax.wgsl"));

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("relief relax bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
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
            label: Some("relief relax pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("relief relax pipeline"),
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

/// Process-wide cache for [`relax_gpu`]'s pipeline — see
/// [`GUIDED_FILTER_PIPELINES`]'s identical rationale.
static RELAX_PIPELINE: OnceLock<CachedRelaxPipeline> = OnceLock::new();

/// Try to run one Jacobi-style relaxation sweep
/// ([`super::multigrid::MultigridLayer::relax`]'s CPU algorithm) on the GPU.
///
/// `a`/`b`/`c` must each have exactly `width * height` elements. Returns the
/// new `a` buffer (the CPU method mutates `self.a` in place via its `d`
/// scratch buffer — this returns the equivalent of that post-sweep `a`).
///
/// # Fallback contract
/// Returns `None` — never panics — when no `wgpu` context is available or
/// the dispatch fails; [`super::multigrid::MultigridLayer::relax`] falls
/// back to the CPU implementation in that case.
#[must_use]
// `too_many_lines`: one cohesive wgpu dispatch (buffer/pipeline setup,
// dispatch, validation-scope check, readback) — same rationale as
// `fuse_level_gpu`'s and `CachedGuidedFilterPipelines::new`'s identical
// allows in this crate.
#[allow(clippy::too_many_lines)]
pub fn relax_gpu(a: &[f32], b: &[f32], c: &[f32], width: usize, height: usize) -> Option<Vec<f32>> {
    let (device, queue) = stacker_core::gpu::context()?;
    if width == 0 || height == 0 {
        return Some(Vec::new());
    }
    debug_assert_eq!(a.len(), width * height);
    debug_assert_eq!(b.len(), width * height);
    debug_assert_eq!(c.len(), width * height);

    // See `guided_filter_gpu`'s identical guard for why this critical
    // section must be serialized across threads sharing the process-wide
    // `Device`.
    let _dispatch_guard = stacker_core::gpu::dispatch_guard();
    let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

    // Cached shader/pipeline/bind-group-layout — see `CachedRelaxPipeline`'s
    // doc comment; only the buffers (which vary per call) are built fresh.
    let cached = RELAX_PIPELINE.get_or_init(|| CachedRelaxPipeline::new(device));

    let make_storage_buffer = |label: &str, data: &[f32]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    };
    let a_buf = make_storage_buffer("relief relax a_in", a);
    let b_buf = make_storage_buffer("relief relax b_in", b);
    let c_buf = make_storage_buffer("relief relax c_in", c);

    let len_bytes = (width * height * std::mem::size_of::<f32>()) as u64;
    let a_out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("relief relax a_out"),
        size: len_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let uniforms = RelaxUniforms {
        width: width as u32,
        height: height as u32,
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("relief relax uniforms"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("relief relax bind group"),
        layout: &cached.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: c_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: a_out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
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
            (width as f32 / 16.0).ceil() as u32,
            (height as f32 / 16.0).ceil() as u32,
            1,
        );
    }

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("relief relax readback buffer"),
        size: len_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&a_out_buf, 0, &output_buffer, 0, len_bytes);
    queue.submit(Some(encoder.finish()));

    if let Some(err) = pollster::block_on(error_scope.pop()) {
        tracing::debug!(error = %err, "gpu: relief relax dispatch failed validation, falling back to CPU");
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
    let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    output_buffer.unmap();

    Some(out)
}
