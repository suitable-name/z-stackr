//! `Apex` fusion on the device.
//!
//! A [`ApexPlan`] is built for one frame size and one setting combination. It
//! allocates every buffer and builds every bind group once; folding a frame is
//! then one command buffer of about forty dispatches, and the only host traffic is
//! the frame going up and, at the end, the fused image coming down.
//!
//! # Per frame
//!
//! 1. Downsample the frame level by level (horizontal pass, vertical pass).
//! 2. For every level but the last: expand the next level, subtract it from this
//!    one to get the Laplacian band, and fold the band into the fused level (the
//!    pixel is taken when its selection energy is strictly greater than the best
//!    so far, so the first frame wins ties).
//! 3. Add the last (base) level to a running sum.
//!
//! # At the end
//!
//! The base sum is divided by the number of frames, then the fused pyramid is
//! reconstructed from the base up: expand, add the fused band, repeat.
//!
//! The level sizes, the band selection rule, the grit-suppression window on the
//! finest band and the averaged base are those of `stacker_algo::apex`.

use std::{
    borrow::Cow,
    collections::VecDeque,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use bytemuck::cast_slice;
use stacker_core::image::PlanarImage;
use wgpu::util::DeviceExt;

use crate::{
    GpuError,
    context::{Failure, GpuContext},
};

const SHADER_TEMPLATE: &str = include_str!("apex.wgsl");

/// Threads per workgroup edge; the shader declares 16 x 16 x 1.
const GROUP: u32 = 16;
/// Luma, `chroma_a`, `chroma_b`.
const PLANES: u32 = 3;
/// Fewest levels, as on the CPU: one band and the base.
const MIN_LEVELS: usize = 2;

// The pyramid kernel, built exactly as the CPU builds it so the f32 bits agree.
const KERNEL_REDUCTION_FACTOR: f32 = 0.33;
const KERNEL_WEIGHT1: f32 = 0.25;
const KERNEL_WEIGHT2: f32 = 0.25 - KERNEL_REDUCTION_FACTOR / 2.0;

/// The WGSL source with the kernel taps filled in.
#[must_use]
pub fn shader_source() -> String {
    SHADER_TEMPLATE
        .replace("KERNEL_OUTER_VALUE", &format!("{KERNEL_WEIGHT2:?}"))
        .replace("KERNEL_INNER_VALUE", &format!("{KERNEL_WEIGHT1:?}"))
        .replace(
            "KERNEL_CENTRE_VALUE",
            &format!("{KERNEL_REDUCTION_FACTOR:?}"),
        )
}

/// The shader's entry points, in the order of [`Plan`] pipelines.
#[derive(Clone, Copy)]
enum Kernel {
    DownH,
    DownV,
    ExpandH,
    ExpandV,
    Fold,
    Accum,
    Finalize,
}

const KERNELS: [(Kernel, &str); 7] = [
    (Kernel::DownH, "down_h"),
    (Kernel::DownV, "down_v"),
    (Kernel::ExpandH, "expand_h"),
    (Kernel::ExpandV, "expand_v"),
    (Kernel::Fold, "fold"),
    (Kernel::Accum, "accum"),
    (Kernel::Finalize, "finalize"),
];

/// The sizes of every pyramid level for a `width x height` frame, finest first;
/// the last entry is the base. Same stopping rule as the CPU pyramid.
#[must_use]
pub fn level_dims(width: usize, height: usize, levels: usize) -> Vec<(usize, usize)> {
    let levels = levels.max(MIN_LEVELS);
    let mut dims = vec![(width, height)];
    for _ in 0..levels - 1 {
        let (w, h) = dims[dims.len() - 1];
        let next = (w.div_ceil(2), h.div_ceil(2));
        dims.push(next);
        if next.0 <= 2 || next.1 <= 2 {
            break;
        }
    }
    dims
}

/// The device memory a plan for `width x height` frames needs in total, in bytes:
/// the plan's buffers, the two upload staging images and the read-back image.
///
/// Computed without touching the device, so a caller can refuse a stack that
/// will not fit before allocating anything. [`ApexPlan::device_bytes`] plus three
/// images equals this.
#[must_use]
pub fn plan_bytes(width: usize, height: usize, levels: usize) -> u64 {
    let dims = level_dims(width, height, levels);
    let last = dims.len() - 1;
    let planes = PLANES as usize;
    let words = |l: usize| dims[l].0 * dims[l].1;
    let finest = planes * words(0);
    let tmp = (0..last)
        .map(|l| {
            let (w, h) = dims[l];
            planes * (w.div_ceil(2) * h).max(w * h.div_ceil(2))
        })
        .max()
        .unwrap_or(1);
    let gaussian: usize = (1..=last).map(|l| planes * words(l)).sum();
    let fused: usize = (0..=last).map(|l| planes * words(l)).sum();
    let best: usize = (0..last).map(words).sum();
    let plan_words = finest + finest + tmp + gaussian + fused + best;
    // Two staging images and the read-back image, one image each.
    (plan_words as u64 + 3 * finest as u64) * 4
}

fn u32_of(value: usize) -> u32 {
    u32::try_from(value).expect("an image dimension fits in u32")
}

fn groups(extent: usize) -> u32 {
    u32_of(extent).div_ceil(GROUP)
}

/// One dispatch of the frame or reconstruction sequence.
struct Dispatch {
    kernel: Kernel,
    bind_group: wgpu::BindGroup,
    grid: [u32; 3],
}

/// Creates buffers and dispatches, and counts the bytes it allocated.
struct Builder<'a> {
    device: &'a wgpu::Device,
    layout: &'a wgpu::BindGroupLayout,
    globals: &'a wgpu::Buffer,
    dummy_read: &'a wgpu::Buffer,
    dummy_best: &'a wgpu::Buffer,
    max_buffer: u64,
    bytes: u64,
}

impl Builder<'_> {
    /// A storage buffer of `words` f32 values.
    fn storage(&mut self, label: &str, words: usize) -> Result<wgpu::Buffer, GpuError> {
        let size = (words as u64) * 4;
        if size > self.max_buffer {
            return Err(GpuError::TooLarge {
                needed: size,
                limit: self.max_buffer,
            });
        }
        self.bytes += size;
        Ok(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }))
    }

    /// A dispatch with its own parameter block. `a`, `b` and `best` fall back to
    /// dummies when the kernel does not read them.
    #[expect(
        clippy::too_many_arguments,
        reason = "one call site per dispatch; the buffers are the dispatch"
    )]
    fn dispatch(
        &self,
        kernel: Kernel,
        params: [u32; 8],
        a: Option<&wgpu::Buffer>,
        b: Option<&wgpu::Buffer>,
        o: &wgpu::Buffer,
        best: Option<&wgpu::Buffer>,
        grid: [u32; 3],
    ) -> Dispatch {
        let params = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("apex params"),
                contents: cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apex dispatch"),
            layout: self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a.unwrap_or(self.dummy_read).as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b.unwrap_or(self.dummy_read).as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: o.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: best.unwrap_or(self.dummy_best).as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.globals.as_entire_binding(),
                },
            ],
        });
        Dispatch {
            kernel,
            bind_group,
            grid,
        }
    }
}

/// The parameter block layout the shader's `Params` expects.
fn params(w: usize, h: usize, sw: usize, sh: usize, mode: u32, use_color: bool) -> [u32; 8] {
    [
        u32_of(w),
        u32_of(h),
        u32_of(sw),
        u32_of(sh),
        mode,
        u32::from(use_color),
        0,
        0,
    ]
}

/// Host-writable memory a frame is copied into before the device copies it to
/// the input buffer. Two of them take turns, so the host fills one while the
/// device is still reading the other; they stay mapped between uses, which avoids
/// allocating and page-faulting a fresh staging area for every frame.
struct Staging {
    buffer: wgpu::Buffer,
    /// True while the buffer is mapped for writing and the host may fill it.
    ready: Arc<AtomicBool>,
}

/// `Apex` fusion of a stack of equally sized frames on the device.
pub struct ApexPlan {
    ctx: Arc<GpuContext>,
    pipelines: Vec<wgpu::ComputePipeline>,
    width: usize,
    height: usize,
    dims: Vec<(usize, usize)>,
    input: wgpu::Buffer,
    band: wgpu::Buffer,
    globals: wgpu::Buffer,
    frame_passes: Vec<Dispatch>,
    recon_passes: Vec<Dispatch>,
    /// The reconstruction ends in `band` (true) or in `input` (false).
    result_in_band: bool,
    readback: Option<wgpu::Buffer>,
    /// The two upload staging buffers, created on the first upload.
    staging: Vec<Staging>,
    next_slot: usize,
    /// The staging slot holding a frame that has not been copied to the device.
    pending_copy: Option<usize>,
    device_bytes: u64,
    n_frames: u32,
    folded: u32,
    in_flight: VecDeque<wgpu::SubmissionIndex>,
}

impl ApexPlan {
    /// Build a plan for `width x height` frames, `levels` pyramid levels (at
    /// least 2), with the selection settings of the CPU `Apex`.
    ///
    /// # Errors
    /// [`GpuError::TooLarge`] when a buffer would exceed what the device allows.
    ///
    /// # Panics
    /// Panics when a dimension exceeds `u32::MAX`.
    #[expect(
        clippy::too_many_lines,
        reason = "one straight construction of buffers, layouts and the two dispatch lists"
    )]
    pub fn new(
        ctx: &Arc<GpuContext>,
        width: usize,
        height: usize,
        levels: usize,
        use_color: bool,
        grit_suppression: bool,
    ) -> Result<Self, GpuError> {
        if ctx.injected(Failure::Plan) {
            return Err(GpuError::Injected("building a plan"));
        }
        let device = &ctx.device;
        let dims = level_dims(width, height, levels);
        let last = dims.len() - 1;
        let max_buffer = ctx
            .limits()
            .max_buffer_size
            .min(u64::from(ctx.limits().max_storage_buffer_binding_size));

        // ── Layout, shader and pipelines ───────────────────────────────────
        let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uniform = wgpu::BufferBindingType::Uniform;
        let read = wgpu::BufferBindingType::Storage { read_only: true };
        let write = wgpu::BufferBindingType::Storage { read_only: false };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apex layout"),
            entries: &[
                entry(0, uniform),
                entry(1, read),
                entry(2, read),
                entry(3, write),
                entry(4, write),
                entry(5, uniform),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("apex pipeline layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apex"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(shader_source())),
        });
        let pipelines = KERNELS
            .iter()
            .map(|&(_, name)| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(name),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some(name),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                })
            })
            .collect();

        // ── Buffers ─────────────────────────────────────────────────────────
        let globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apex globals"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let small = |label| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: 16,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        let dummy_read = small("dummy read");
        let dummy_best = small("dummy best");
        let mut build = Builder {
            device,
            layout: &layout,
            globals: &globals,
            dummy_read: &dummy_read,
            dummy_best: &dummy_best,
            max_buffer,
            bytes: 0,
        };
        let words = |l: usize| dims[l].0 * dims[l].1;
        let planes = PLANES as usize;

        let input = build.storage("input frame", planes * words(0))?;
        let band = build.storage("band", planes * words(0))?;
        // The horizontal passes of the down- and the up-sampling are the larger
        // at the finest level.
        let tmp_words = (0..last)
            .map(|l| {
                let (w, h) = dims[l];
                planes * (w.div_ceil(2) * h).max(w * h.div_ceil(2))
            })
            .max()
            .unwrap_or(1);
        let tmp = build.storage("tmp", tmp_words)?;
        let mut cur = Vec::with_capacity(last);
        for l in 1..=last {
            cur.push(build.storage("gaussian level", planes * words(l))?);
        }
        let mut fused = Vec::with_capacity(dims.len());
        for l in 0..dims.len() {
            fused.push(build.storage("fused level", planes * words(l))?);
        }
        let mut best = Vec::with_capacity(last);
        for l in 0..last {
            best.push(build.storage("best energy", words(l))?);
        }
        // Gaussian level `l`: the input frame for 0, `cur[l - 1]` after.
        let gauss = |l: usize| if l == 0 { &input } else { &cur[l - 1] };

        // ── Per-frame dispatches ────────────────────────────────────────────
        let mut frame_passes = Vec::new();
        for l in 0..last {
            let (w, h) = dims[l];
            let (w2, h2) = dims[l + 1];
            frame_passes.push(build.dispatch(
                Kernel::DownH,
                params(w, h, 0, 0, 0, false),
                Some(gauss(l)),
                None,
                &tmp,
                None,
                [groups(w2), groups(h), PLANES],
            ));
            frame_passes.push(build.dispatch(
                Kernel::DownV,
                params(w2, h, 0, 0, 0, false),
                Some(&tmp),
                None,
                gauss(l + 1),
                None,
                [groups(w2), groups(h2), PLANES],
            ));
        }
        for l in 0..last {
            let (w, h) = dims[l];
            let (sw, sh) = dims[l + 1];
            frame_passes.push(build.dispatch(
                Kernel::ExpandH,
                params(w, h, sw, sh, 0, false),
                Some(gauss(l + 1)),
                None,
                &tmp,
                None,
                [groups(w), groups(h.div_ceil(2)), PLANES],
            ));
            frame_passes.push(build.dispatch(
                Kernel::ExpandV,
                params(w, h, sw, sh, 0, false),
                Some(gauss(l)),
                Some(&tmp),
                &band,
                None,
                [groups(w), groups(h), PLANES],
            ));
            let mode = u32::from(l == 0 && grit_suppression);
            frame_passes.push(build.dispatch(
                Kernel::Fold,
                params(w, h, 0, 0, mode, use_color),
                Some(&band),
                None,
                &fused[l],
                Some(&best[l]),
                [groups(w), groups(h), 1],
            ));
        }
        let (bw, bh) = dims[last];
        frame_passes.push(build.dispatch(
            Kernel::Accum,
            params(bw, bh, 0, 0, 0, false),
            Some(gauss(last)),
            None,
            &fused[last],
            None,
            [groups(bw), groups(bh), PLANES],
        ));

        // ── Reconstruction dispatches ───────────────────────────────────────
        let mut recon_passes = vec![build.dispatch(
            Kernel::Finalize,
            params(bw, bh, 0, 0, 0, false),
            None,
            None,
            &fused[last],
            None,
            [groups(bw), groups(bh), PLANES],
        )];
        // The level just reconstructed is the source of the next expand; the two
        // finest-size buffers that are free by now (the input frame and the band)
        // take turns as its home.
        let mut src: &wgpu::Buffer = &fused[last];
        let mut into_band = false;
        for l in (0..last).rev() {
            let (w, h) = dims[l];
            let (sw, sh) = dims[l + 1];
            let target = if into_band { &band } else { &input };
            recon_passes.push(build.dispatch(
                Kernel::ExpandH,
                params(w, h, sw, sh, 1, false),
                Some(src),
                None,
                &tmp,
                None,
                [groups(w), groups(h.div_ceil(2)), PLANES],
            ));
            recon_passes.push(build.dispatch(
                Kernel::ExpandV,
                params(w, h, sw, sh, 1, false),
                Some(&fused[l]),
                Some(&tmp),
                target,
                None,
                [groups(w), groups(h), PLANES],
            ));
            src = target;
            into_band = !into_band;
        }
        // `into_band` has flipped once more than the last write.
        let result_in_band = !into_band;
        let device_bytes = build.bytes;
        // An out-of-memory or validation error while allocating is reported
        // asynchronously through the context; refuse the plan if there was one.
        let _ = device.poll(wgpu::Maintain::Wait);
        if let Some(message) = ctx.take_error() {
            return Err(GpuError::Device(message));
        }

        Ok(Self {
            ctx: Arc::clone(ctx),
            pipelines,
            width,
            height,
            dims,
            input,
            band,
            globals,
            frame_passes,
            recon_passes,
            result_in_band,
            readback: None,
            staging: Vec::new(),
            next_slot: 0,
            pending_copy: None,
            device_bytes,
            n_frames: 0,
            folded: 0,
            in_flight: VecDeque::new(),
        })
    }

    /// Bytes of device memory the plan holds. The read-back buffer (one more
    /// image) and the two upload staging buffers (two more images), both created
    /// on first use, come on top.
    #[must_use]
    pub const fn device_bytes(&self) -> u64 {
        self.device_bytes
    }

    /// The number of pyramid levels, base included.
    #[must_use]
    pub const fn levels(&self) -> usize {
        self.dims.len()
    }

    /// Start a stack of `n_frames` frames. The averaged base divides by this.
    pub const fn begin(&mut self, n_frames: u32) {
        self.n_frames = n_frames;
        self.folded = 0;
    }

    /// Bytes of one image (three planes) on the device.
    const fn image_bytes(&self) -> u64 {
        (PLANES as u64) * (self.width as u64) * (self.height as u64) * 4
    }

    /// Create the two staging buffers, mapped and ready.
    fn create_staging(&mut self) {
        let size = self.image_bytes();
        for _ in 0..2 {
            let buffer = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("apex upload staging"),
                size,
                usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: true,
            });
            self.staging.push(Staging {
                buffer,
                ready: Arc::new(AtomicBool::new(true)),
            });
        }
    }

    /// Ask for staging slot `slot` to be mapped again; it becomes ready once the
    /// device has finished reading it.
    fn remap(&self, slot: usize) {
        let staging = &self.staging[slot];
        staging.ready.store(false, Ordering::Release);
        let flag = Arc::clone(&staging.ready);
        staging
            .buffer
            .slice(..)
            .map_async(wgpu::MapMode::Write, move |result| {
                if result.is_ok() {
                    flag.store(true, Ordering::Release);
                }
            });
    }

    /// Record the copy of the pending staged frame into the input buffer.
    fn record_copy(&self, encoder: &mut wgpu::CommandEncoder, slot: usize) {
        encoder.copy_buffer_to_buffer(
            &self.staging[slot].buffer,
            0,
            &self.input,
            0,
            self.image_bytes(),
        );
    }

    /// Submit the pending staged frame's copy on its own.
    fn submit_copy(&mut self) {
        let Some(slot) = self.pending_copy.take() else {
            return;
        };
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("apex upload"),
            });
        self.record_copy(&mut encoder, slot);
        self.ctx.queue.submit(Some(encoder.finish()));
        self.remap(slot);
    }

    /// Stage `frame` for the device: copy it into a mapped staging buffer. The
    /// device-side copy into the input buffer is recorded with the next fold (or
    /// [`Self::flush`]). Blocks only when both staging buffers are still being
    /// read by the device.
    ///
    /// # Errors
    /// [`GpuError::SizeMismatch`] when the frame is not the size of the plan.
    ///
    /// # Panics
    /// Panics when the device's mapped memory is not 4-byte aligned.
    pub fn upload(&mut self, frame: &PlanarImage<f32>) -> Result<(), GpuError> {
        if frame.width != self.width || frame.height != self.height {
            return Err(GpuError::SizeMismatch {
                expected: (self.width, self.height),
                got: (frame.width, frame.height),
            });
        }
        if self.staging.is_empty() {
            self.create_staging();
        }
        // An earlier frame that was staged but never folded goes to the device
        // first, so it is not overwritten.
        self.submit_copy();
        let slot = self.next_slot;
        self.next_slot = (slot + 1) % self.staging.len();
        let staging = &self.staging[slot];
        while !staging.ready.load(Ordering::Acquire) {
            let _ = self.ctx.device.poll(wgpu::Maintain::Wait);
        }
        let plane = self.width * self.height;
        {
            let mut view = staging.buffer.slice(..).get_mapped_range_mut();
            let floats: &mut [f32] = bytemuck::try_cast_slice_mut(&mut view)
                .expect("mapped memory is aligned to 4 bytes");
            floats[..plane].copy_from_slice(&frame.luma);
            floats[plane..2 * plane].copy_from_slice(&frame.chroma_a);
            floats[2 * plane..3 * plane].copy_from_slice(&frame.chroma_b);
        }
        staging.buffer.unmap();
        self.pending_copy = Some(slot);
        Ok(())
    }

    /// Fold the frame currently in the input buffer into the fused pyramid. A
    /// frame staged by [`Self::upload`] is copied to the input buffer first.
    pub fn fold_uploaded(&mut self) {
        let first = u32::from(self.folded == 0);
        self.ctx
            .queue
            .write_buffer(&self.globals, 0, cast_slice(&[first, self.n_frames, 0, 0]));
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("apex frame"),
            });
        let copied = self.pending_copy.take();
        if let Some(slot) = copied {
            self.record_copy(&mut encoder, slot);
        }
        self.record(&mut encoder, true);
        let index = self.ctx.queue.submit(Some(encoder.finish()));
        if let Some(slot) = copied {
            self.remap(slot);
        }
        self.folded += 1;
        // At most one frame in flight: the next upload overlaps its compute, and
        // the staging memory of older uploads is released.
        self.in_flight.push_back(index);
        while self.in_flight.len() > 1 {
            if let Some(old) = self.in_flight.pop_front() {
                let _ = self
                    .ctx
                    .device
                    .poll(wgpu::Maintain::WaitForSubmissionIndex(old));
            }
        }
    }

    /// Upload `frame` and fold it in.
    ///
    /// # Errors
    /// [`GpuError::SizeMismatch`] when the frame is not the size of the plan.
    pub fn fold(&mut self, frame: &PlanarImage<f32>) -> Result<(), GpuError> {
        if self.ctx.injected(Failure::Fold) {
            return Err(GpuError::Injected("folding a frame"));
        }
        self.upload(frame)?;
        self.fold_uploaded();
        self.ctx
            .take_error()
            .map_or(Ok(()), |message| Err(GpuError::Device(message)))
    }

    /// Record the frame (`frame = true`) or reconstruction dispatches.
    fn record(&self, encoder: &mut wgpu::CommandEncoder, frame: bool) {
        let list = if frame {
            &self.frame_passes
        } else {
            &self.recon_passes
        };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(if frame {
                "apex frame"
            } else {
                "apex reconstruct"
            }),
            timestamp_writes: None,
        });
        for d in list {
            pass.set_pipeline(&self.pipelines[d.kernel as usize]);
            pass.set_bind_group(0, &d.bind_group, &[]);
            pass.dispatch_workgroups(d.grid[0], d.grid[1], d.grid[2]);
        }
    }

    /// Copy a staged frame to the input buffer now and wait for it (otherwise it
    /// goes with the next fold).
    pub fn flush(&mut self) {
        self.submit_copy();
        self.sync();
    }

    /// Wait until everything submitted so far has finished.
    pub fn sync(&self) {
        let _ = self.ctx.device.poll(wgpu::Maintain::Wait);
    }

    /// Average the base and reconstruct the fused image on the device, then wait
    /// for it. The result stays on the device; [`Self::read_back`] fetches it.
    pub fn reconstruct(&mut self) {
        self.in_flight.clear();
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("apex reconstruct"),
            });
        self.record(&mut encoder, false);
        self.ctx.queue.submit(Some(encoder.finish()));
        self.sync();
    }

    /// Download the reconstructed image.
    ///
    /// # Errors
    /// [`GpuError::ReadBack`] when mapping the buffer fails.
    ///
    /// # Panics
    /// Panics when the device's mapped memory is not 4-byte aligned.
    pub fn read_back(&mut self) -> Result<PlanarImage<f32>, GpuError> {
        let ctx = Arc::clone(&self.ctx);
        if ctx.injected(Failure::ReadBack) {
            return Err(GpuError::Injected("reading the result back"));
        }
        let plane = self.width * self.height;
        let size = (PLANES as usize * plane * 4) as u64;
        let readback = self.readback.get_or_insert_with(|| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("apex readback"),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        let source = if self.result_in_band {
            &self.band
        } else {
            &self.input
        };
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("apex read back"),
            });
        encoder.copy_buffer_to_buffer(source, 0, readback, 0, size);
        ctx.queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..size);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = ctx.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| GpuError::ReadBack(e.to_string()))?
            .map_err(|e| GpuError::ReadBack(e.to_string()))?;
        let image = {
            let mapped = slice.get_mapped_range();
            let floats: &[f32] =
                bytemuck::try_cast_slice(&mapped).expect("mapped memory is aligned to 4 bytes");
            PlanarImage::from_planes(
                self.width,
                self.height,
                floats[..plane].to_vec(),
                floats[plane..2 * plane].to_vec(),
                floats[2 * plane..3 * plane].to_vec(),
            )
            .expect("each plane holds width * height samples")
        };
        readback.unmap();
        ctx.take_error()
            .map_or(Ok(image), |message| Err(GpuError::Device(message)))
    }

    /// Reconstruct and download: the end of a stack.
    ///
    /// # Errors
    /// [`GpuError::ReadBack`] when mapping the buffer fails.
    pub fn finish(&mut self) -> Result<PlanarImage<f32>, GpuError> {
        self.reconstruct();
        self.read_back()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shader_parses_and_validates_without_a_gpu() {
        let source = shader_source();
        assert!(!source.contains("KERNEL_"), "every placeholder is filled");
        let module = naga::front::wgsl::parse_str(&source).expect("the WGSL parses");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .expect("the WGSL validates");
    }

    #[test]
    #[expect(
        clippy::suboptimal_flops,
        reason = "a plain check of the tap sum, not a hot path"
    )]
    fn the_taps_are_the_cpu_taps_and_sum_to_one() {
        // The CPU's kernel: [w2, w1, centre, w1, w2], documented as summing to 1.
        let sum = 2.0 * KERNEL_WEIGHT2 + 2.0 * KERNEL_WEIGHT1 + KERNEL_REDUCTION_FACTOR;
        assert!((sum - 1.0).abs() < 1e-6, "sum = {sum}");
        let source = shader_source();
        assert!(source.contains(&format!("{KERNEL_WEIGHT2:?}")));
    }

    #[test]
    fn level_sizes_follow_the_cpu_rule() {
        assert_eq!(
            level_dims(100, 60, 4),
            vec![(100, 60), (50, 30), (25, 15), (13, 8)]
        );
        // Stops once a side reaches 2, as the CPU pyramid does.
        assert_eq!(level_dims(9, 9, 8), vec![(9, 9), (5, 5), (3, 3), (2, 2)]);
        // Never fewer than two levels.
        assert_eq!(level_dims(64, 64, 0).len(), 2);
    }
}
