//! The `Relief` multigrid depth solver on the device.
//!
//! A [`MultigridPlan`] is the twin of `stacker_algo::relief::multigrid`: the same
//! padding, hierarchy, restriction, prolongation and Jacobi sweeps in the same
//! order (the algorithm is described step by step in that module's
//! documentation), including the two fixes to the GitHub z-stackr algorithm (the
//! corner restriction and the full prolongation). The winning frame indices and
//! the confidence weights are uploaded **once**; the whole cycle runs on the
//! device without any traffic to the host, and the solved finest level is read
//! back **once**.
//!
//! # How the cycle is run
//!
//! The recursion `cycle(l)` visits level `l + 1` three times, so level `l` is
//! visited `3^l` times per solve: a 24 MP image has about 13 levels and the
//! coarse ones are visited hundreds of thousands of times, each visit being a
//! handful of tiny operations. Dispatching them one by one would be dominated by
//! launch overhead, so:
//!
//! * the **large levels** (more than [`TAIL_SAMPLES`] samples) are dispatched
//!   operation by operation (restrict, prolong, one dispatch per sweep), with the
//!   geometry of each operation in a prepared uniform block and bind group;
//! * once a level has at most [`TAIL_SAMPLES`] samples, one dispatch of a **single
//!   workgroup** runs the entire `cycle` of that level and all below it from a
//!   schedule of operations prepared on the host, with a barrier between
//!   operations.
//!
//! The Jacobi sweep reads `a` and writes `d`, after which the CPU swaps the two
//! planes; the device does not swap, it alternates which of the two physical
//! planes plays `a` (the host tracks the parity per level and puts it in every
//! operation). Three sweeps per visit mean the roles flip at every visit.
//!
//! `pyramid_smooth` stays on the CPU: the caller smooths the padded plane
//! [`MultigridPlan::solve_padded`] returns and crops it
//! (`stacker_pipeline::compute::Compute::multigrid_depth` does).
//!
//! # Expected agreement with the CPU
//!
//! All arithmetic is `f32` and the operations are those of the CPU in the same
//! order, so the only differences are fused multiply-adds a driver may contract
//! (`b * c + (1 - b) * value`, the sums of products in the restriction). Jacobi
//! averaging does not amplify them; the parity test asserts `1e-3` frame indices
//! at most and a mean below `1e-4`.
//!
//! # Device memory
//!
//! [`plan_bytes`]: 16 bytes per sample of the hierarchy (about `21.3` per pixel
//! with the coarse levels), 6 per pixel for the uploads and 4 for the read-back
//! buffer.

use std::{collections::HashMap, sync::Arc};

use bytemuck::cast_slice;

use crate::{
    GpuError,
    common::{self, u32_of},
    context::{Failure, GpuContext},
};

const SHADER: &str = include_str!("multigrid.wgsl");

/// Threads per workgroup edge of the dispatched operations (16 x 16 x 1).
const GROUP: u32 = 16;
/// Bytes of the shader's `Params`.
const PARAMS_BYTES: u64 = 48;

/// Levels with at most this many samples are solved by one workgroup per cycle.
///
/// This replaces one dispatch per operation. The workgroup has 256 threads, so a
/// level of 8192 samples is 32 samples per thread; larger levels have enough
/// parallelism to be dispatched. Tune by measurement.
pub const TAIL_SAMPLES: usize = 1 << 13;

/// How often a level restricts to the next one, solves it and prolongs back
/// (`stacker_algo::relief::multigrid::VISITS_PER_LEVEL`).
const VISITS_PER_LEVEL: usize = 3;
/// How many relaxation sweeps follow each prolongation
/// (`stacker_algo::relief::multigrid::SWEEPS_PER_VISIT`).
const SWEEPS_PER_VISIT: usize = 3;

/// The WGSL source of the multigrid kernels.
#[must_use]
pub fn shader_source() -> String {
    SHADER.to_owned()
}

const KERNELS: [&str; 5] = [
    "init",
    "restrict_level",
    "prolong_level",
    "relax_level",
    "tail",
];

const INIT: usize = 0;
const RESTRICT: usize = 1;
const PROLONG: usize = 2;
const RELAX: usize = 3;
const TAIL: usize = 4;

/// The hierarchy of an image: the padded size of every level, finest first
/// (`(2 * (w / 2) + 1) x (2 * (h / 2) + 1)`, then `ceil(w / 2) x ceil(h / 2)`
/// while both sides of the last level exceed 2).
fn level_dims(width: usize, height: usize) -> Vec<(usize, usize)> {
    let mut level = ((width / 2) * 2 + 1, (height / 2) * 2 + 1);
    let mut dims = vec![level];
    while level.0 > 2 && level.1 > 2 {
        level = (level.0.div_ceil(2), level.1.div_ceil(2));
        dims.push(level);
    }
    dims
}

/// The device memory a plan for a `width x height` image needs, in bytes (the
/// small schedule and level tables not counted). `0` for an empty image.
#[must_use]
pub fn plan_bytes(width: usize, height: usize) -> u64 {
    if width == 0 || height == 0 {
        return 0;
    }
    let dims = level_dims(width, height);
    let samples: u64 = dims.iter().map(|&(w, h)| (w * h) as u64).sum();
    let n = (width * height) as u64;
    let finest = (dims[0].0 * dims[0].1) as u64;
    16 * samples + 4 * n.div_ceil(2) + 4 * n + 4 * finest
}

/// One operation of the cycle. `fa` and `ca` say which of the two physical planes
/// plays `a` on the fine and the coarse level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Op {
    Restrict {
        level: usize,
        fa: u32,
        ca: u32,
    },
    Prolong {
        level: usize,
        fa: u32,
        ca: u32,
    },
    Relax {
        level: usize,
        fa: u32,
    },
    /// A whole `cycle(tail_from)` run by one workgroup, entered with every level
    /// at and below `tail_from` (but the last) at parity `entry`.
    Tail {
        entry: u32,
    },
}

/// Append the operations of `cycle(level)` to `out`, updating the plane parity of
/// every level. A cycle that reaches level `tail_from` is replaced by one
/// [`Op::Tail`]; every level from there down (but the last, which has no sweeps)
/// flips its parity once per such cycle, because each is visited an odd number of
/// times with an odd number of sweeps per visit.
fn cycle(n_levels: usize, level: usize, tail_from: usize, par: &mut [u32], out: &mut Vec<Op>) {
    if level + 1 >= n_levels {
        return;
    }
    if level == tail_from {
        out.push(Op::Tail { entry: par[level] });
        for parity in &mut par[level..n_levels - 1] {
            *parity ^= 1;
        }
        return;
    }
    for _ in 0..VISITS_PER_LEVEL {
        out.push(Op::Restrict {
            level,
            fa: par[level],
            ca: par[level + 1],
        });
        cycle(n_levels, level + 1, tail_from, par, out);
        out.push(Op::Prolong {
            level,
            fa: par[level],
            ca: par[level + 1],
        });
        for _ in 0..SWEEPS_PER_VISIT {
            out.push(Op::Relax {
                level,
                fa: par[level],
            });
            par[level] ^= 1;
        }
    }
}

/// The operations of `solve` (`cycle(0)`) and the parity of every level at the
/// end, with the levels from `tail_from` on collapsed into [`Op::Tail`]s
/// (`usize::MAX` for none).
fn schedule(n_levels: usize, tail_from: usize) -> (Vec<Op>, Vec<u32>) {
    let mut par = vec![0_u32; n_levels];
    let mut ops = Vec::new();
    cycle(n_levels, 0, tail_from, &mut par, &mut ops);
    (ops, par)
}

/// The operations one tail cycle runs when entered at parity `entry`.
fn tail_ops(n_levels: usize, tail_from: usize, entry: u32) -> Vec<Op> {
    let mut par = vec![0_u32; n_levels];
    for parity in &mut par[tail_from..n_levels - 1] {
        *parity = entry;
    }
    let mut ops = Vec::new();
    cycle(n_levels, tail_from, usize::MAX, &mut par, &mut ops);
    ops
}

/// An operation as a word of the tail schedule: `op | level << 2 | fa << 10 |
/// ca << 11`.
///
/// # Panics
/// Panics on an [`Op::Tail`] or a level above 255.
fn encode(op: Op) -> u32 {
    let (code, level, fa, ca) = match op {
        Op::Restrict { level, fa, ca } => (0, level, fa, ca),
        Op::Prolong { level, fa, ca } => (1, level, fa, ca),
        Op::Relax { level, fa } => (2, level, fa, 0),
        Op::Tail { .. } => panic!("a tail cannot contain a tail"),
    };
    assert!(level < 256, "the hierarchy has at most 256 levels");
    code | (u32_of(level) << 2) | (fa << 10) | (ca << 11)
}

/// One recorded dispatch of the solve.
struct Step {
    kernel: usize,
    group: usize,
    grid: [u32; 3],
}

/// The multigrid depth solver for one image, with its inputs resident on the
/// device.
pub struct MultigridPlan {
    ctx: Arc<GpuContext>,
    pipelines: Vec<wgpu::ComputePipeline>,
    groups: Vec<wgpu::BindGroup>,
    init: Step,
    steps: Vec<Step>,
    lv: wgpu::Buffer,
    staging: wgpu::Buffer,
    /// The padded size of the finest level.
    padded: (usize, usize),
    /// Which plane of the finest level holds the solution after a solve.
    final_plane: u32,
}

/// What a bind group is for; the key that lets equal ones be shared.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct GroupKey {
    kernel: usize,
    level: usize,
    fa: u32,
    ca: u32,
}

impl MultigridPlan {
    /// Build a plan for a `width x height` image and upload `index` (the winning
    /// frame of every pixel) and `weight` (`> 0` pins the pixel to its frame).
    ///
    /// # Errors
    /// [`GpuError::SizeMismatch`] when a slice holds fewer than `width * height`
    /// samples (or the image is empty), [`GpuError::TooLarge`] when a buffer would
    /// exceed what the device allows, [`GpuError::Device`] when the device
    /// reported an error, [`GpuError::Injected`] when a test armed
    /// [`Failure::Plan`].
    ///
    /// # Panics
    /// Panics when a dimension exceeds `u32::MAX`.
    #[expect(
        clippy::too_many_lines,
        reason = "one linear build of the level hierarchy, buffers and bind groups"
    )]
    pub fn new(
        ctx: &Arc<GpuContext>,
        width: usize,
        height: usize,
        index: &[u16],
        weight: &[f32],
    ) -> Result<Self, GpuError> {
        if ctx.injected(Failure::Plan) {
            return Err(GpuError::Injected("building a plan"));
        }
        let n = width * height;
        for len in [index.len(), weight.len()] {
            if n == 0 || len < n {
                return Err(GpuError::SizeMismatch {
                    expected: (width, height),
                    got: (len, 1),
                });
            }
        }
        let dims = level_dims(width, height);
        let n_levels = dims.len();
        let mut offsets = Vec::with_capacity(n_levels);
        let mut total = 0_u64;
        for &(w, h) in &dims {
            offsets.push(total);
            total += 4 * (w * h) as u64;
        }
        if total > u64::from(u32::MAX) {
            return Err(GpuError::TooLarge {
                needed: total * 4,
                limit: u64::from(u32::MAX) * 4,
            });
        }
        let offsets: Vec<u32> = offsets
            .iter()
            .map(|&o| u32::try_from(o).expect("checked above"))
            .collect();

        let lv = common::storage(ctx, "multigrid levels", total * 4)?;
        let idx = common::storage(ctx, "multigrid index", 4 * n.div_ceil(2) as u64)?;
        let wgt = common::storage(ctx, "multigrid weight", 4 * n as u64)?;
        let finest = dims[0].0 * dims[0].1;
        let staging = common::readback(ctx, "multigrid read back", 4 * finest as u64);

        // Upload the pixels once. The indices are packed two to a word; an odd
        // last one gets a word of its own, because writes are whole words.
        let even = n & !1;
        if even > 0 {
            ctx.queue.write_buffer(&idx, 0, cast_slice(&index[..even]));
        }
        if !n.is_multiple_of(2) {
            ctx.queue
                .write_buffer(&idx, 2 * even as u64, cast_slice(&[index[n - 1], 0_u16]));
        }
        ctx.queue.write_buffer(&wgt, 0, cast_slice(&weight[..n]));

        // The level table: width, height, first element, padding.
        let table_words: Vec<u32> = dims
            .iter()
            .zip(&offsets)
            .flat_map(|(&(w, h), &off)| [u32_of(w), u32_of(h), off, 0])
            .collect();
        let table = common::storage(ctx, "multigrid level table", 4 * table_words.len() as u64)?;
        ctx.queue.write_buffer(&table, 0, cast_slice(&table_words));

        // The levels the single workgroup takes over, and its two schedules (one
        // per parity the cycle can be entered at).
        let tail_from = dims
            .iter()
            .position(|&(w, h)| w * h <= TAIL_SAMPLES)
            .unwrap_or(n_levels);
        let has_tail = tail_from + 1 < n_levels;
        let mut sched_words: Vec<u32> = Vec::new();
        let mut sched_ranges = [(0_u32, 0_u32); 2];
        if has_tail {
            for entry in 0..2 {
                let words: Vec<u32> = tail_ops(n_levels, tail_from, entry)
                    .into_iter()
                    .map(encode)
                    .collect();
                sched_ranges[entry as usize] = (u32_of(sched_words.len()), u32_of(words.len()));
                sched_words.extend(words);
            }
        }
        let sched = common::storage(ctx, "multigrid schedule", 4 * sched_words.len() as u64)?;
        if !sched_words.is_empty() {
            ctx.queue.write_buffer(&sched, 0, cast_slice(&sched_words));
        }

        let pipelines = common::pipelines(ctx, "multigrid", shader_source(), &KERNELS);
        let mut groups: Vec<wgpu::BindGroup> = Vec::new();
        let mut cache: HashMap<GroupKey, usize> = HashMap::new();
        // The index of the bind group of `kernel` for the geometry of `level`
        // (and the sched range of a tail entered at parity `fa`).
        let mut group_for = |kernel: usize, level: usize, fa: u32, ca: u32| -> usize {
            let key = GroupKey {
                kernel,
                level,
                fa,
                ca,
            };
            if let Some(&found) = cache.get(&key) {
                return found;
            }
            let (fw, fh) = dims[level];
            let (cw, ch) = dims.get(level + 1).copied().unwrap_or((0, 0));
            let (sched_off, sched_len) = if kernel == TAIL {
                sched_ranges[fa as usize]
            } else {
                (0, 0)
            };
            let block: [u32; 12] = [
                u32_of(fw),
                u32_of(fh),
                offsets[level],
                u32_of(cw),
                u32_of(ch),
                offsets.get(level + 1).copied().unwrap_or(0),
                if kernel == TAIL { 0 } else { fa },
                ca,
                u32_of(width),
                u32_of(height),
                sched_off,
                sched_len,
            ];
            let params = common::uniform(ctx, "multigrid params", PARAMS_BYTES);
            ctx.queue.write_buffer(&params, 0, cast_slice(&block));
            let pipeline = &pipelines[kernel];
            let bound = match kernel {
                INIT => common::bind(
                    ctx,
                    pipeline,
                    &[(0, &params), (1, &lv), (2, &idx), (3, &wgt)],
                ),
                TAIL => common::bind(
                    ctx,
                    pipeline,
                    &[(0, &params), (1, &lv), (4, &sched), (5, &table)],
                ),
                _ => common::bind(ctx, pipeline, &[(0, &params), (1, &lv)]),
            };
            groups.push(bound);
            cache.insert(key, groups.len() - 1);
            groups.len() - 1
        };

        let grid_of =
            |(w, h): (usize, usize)| [common::groups(w, GROUP), common::groups(h, GROUP), 1];
        let init = Step {
            kernel: INIT,
            group: group_for(INIT, 0, 0, 0),
            grid: grid_of(dims[0]),
        };
        let (ops, parity) = schedule(n_levels, if has_tail { tail_from } else { usize::MAX });
        let mut steps = Vec::with_capacity(ops.len());
        for op in ops {
            steps.push(match op {
                Op::Restrict { level, fa, ca } => Step {
                    kernel: RESTRICT,
                    group: group_for(RESTRICT, level, fa, ca),
                    grid: grid_of(dims[level + 1]),
                },
                Op::Prolong { level, fa, ca } => Step {
                    kernel: PROLONG,
                    group: group_for(PROLONG, level, fa, ca),
                    grid: grid_of(dims[level]),
                },
                Op::Relax { level, fa } => Step {
                    kernel: RELAX,
                    group: group_for(RELAX, level, fa, 0),
                    grid: grid_of(dims[level]),
                },
                Op::Tail { entry } => Step {
                    kernel: TAIL,
                    group: group_for(TAIL, tail_from, entry, 0),
                    grid: [1, 1, 1],
                },
            });
        }
        let _ = ctx.device.poll(wgpu::Maintain::Wait);
        common::check(ctx)?;
        Ok(Self {
            ctx: Arc::clone(ctx),
            pipelines,
            groups,
            init,
            steps,
            lv,
            staging,
            padded: dims[0],
            final_plane: parity[0],
        })
    }

    /// The number of dispatches one solve records (for logs and tests).
    #[must_use]
    pub const fn dispatches(&self) -> usize {
        1 + self.steps.len()
    }

    /// Run the solve (`cycle(0)`) on the device and read the finest level's
    /// solution back: the padded plane, row by row, and its width and height. The
    /// caller smooths it (`pyramid_smooth`) and crops it to the image.
    ///
    /// The plan can solve again; it starts from the uploaded pixels every time and
    /// gives the same result.
    ///
    /// # Errors
    /// [`GpuError::ReadBack`] or [`GpuError::Device`] when the device failed,
    /// [`GpuError::Injected`] when a test armed [`Failure::Fold`] or
    /// [`Failure::ReadBack`].
    pub fn solve_padded(&mut self) -> Result<(Vec<f32>, usize, usize), GpuError> {
        if self.ctx.injected(Failure::Fold) {
            return Err(GpuError::Injected("solving the multigrid"));
        }
        let mut encoder = common::encoder(&self.ctx, "multigrid solve");
        for step in std::iter::once(&self.init).chain(&self.steps) {
            common::dispatch(
                &mut encoder,
                &self.pipelines[step.kernel],
                &self.groups[step.group],
                step.grid,
            );
        }
        if self.ctx.injected(Failure::ReadBack) {
            return Err(GpuError::Injected("reading the depth back"));
        }
        let (width, height) = self.padded;
        let plane_bytes = 4 * (width * height) as u64;
        let values = common::read_at(
            &self.ctx,
            encoder,
            (&self.lv, u64::from(self.final_plane) * plane_bytes),
            &self.staging,
            plane_bytes,
        )?;
        Ok((values, width, height))
    }
}

#[cfg(test)]
mod tests {
    use stacker_algo::relief::multigrid::{
        MultigridLayer, MultigridSolver, SWEEPS_PER_VISIT as CPU_SWEEPS,
        VISITS_PER_LEVEL as CPU_VISITS,
    };

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
    fn the_cycle_constants_are_the_cpu_ones() {
        assert_eq!(VISITS_PER_LEVEL, CPU_VISITS);
        assert_eq!(SWEEPS_PER_VISIT, CPU_SWEEPS);
    }

    #[test]
    fn the_hierarchy_is_the_cpu_hierarchy() {
        for (w, h) in [
            (1, 1),
            (1, 9),
            (2, 2),
            (7, 5),
            (64, 48),
            (101, 33),
            (640, 480),
        ] {
            let cpu = MultigridSolver::from_index(w, h, &vec![0; w * h], &vec![0.0; w * h]);
            let cpu_dims: Vec<(usize, usize)> =
                cpu.layers.iter().map(|l| (l.width, l.height)).collect();
            assert_eq!(level_dims(w, h), cpu_dims, "{w}x{h}");
        }
    }

    /// Run `ops` on CPU layers with the CPU's own operations.
    fn run_on_cpu(layers: &mut [MultigridLayer], ops: &[Op]) {
        for op in ops {
            match *op {
                Op::Restrict { level, .. } => {
                    let (fine, coarse) = layers.split_at_mut(level + 1);
                    fine[level].restrict_to(&mut coarse[0]);
                }
                Op::Prolong { level, .. } => {
                    let (fine, coarse) = layers.split_at_mut(level + 1);
                    coarse[0].prolong_from(&mut fine[level]);
                }
                Op::Relax { level, .. } => layers[level].relax(),
                Op::Tail { .. } => panic!("expanded before"),
            }
        }
    }

    /// `ops` with every tail replaced by the operations it stands for.
    fn expand(ops: &[Op], n_levels: usize, tail_from: usize) -> Vec<Op> {
        ops.iter()
            .flat_map(|&op| match op {
                Op::Tail { entry } => tail_ops(n_levels, tail_from, entry),
                other => vec![other],
            })
            .collect()
    }

    fn pins(w: usize, h: usize) -> (Vec<u16>, Vec<f32>) {
        let mut state = 0x1234_5678_u32;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let index = (0..w * h)
            .map(|_| u16::try_from(next() % 40).expect("small"))
            .collect();
        let weight = (0..w * h)
            .map(|_| if next().is_multiple_of(9) { 1.0 } else { 0.0 })
            .collect();
        (index, weight)
    }

    #[test]
    fn the_operation_order_reproduces_the_cpu_solve() {
        for (w, h) in [(5, 5), (17, 9), (40, 33), (64, 64)] {
            let (index, weight) = pins(w, h);
            let mut cpu = MultigridSolver::from_index(w, h, &index, &weight);
            let mut mine = cpu.clone();
            cpu.solve();
            let n_levels = mine.layers.len();
            let (ops, _) = schedule(n_levels, usize::MAX);
            run_on_cpu(&mut mine.layers, &ops);
            assert_eq!(
                mine.layers[0].a, cpu.layers[0].a,
                "{w}x{h}: bit-identical to solve()"
            );
        }
    }

    #[test]
    fn the_tails_expand_to_the_full_cycle_and_the_parity_is_tracked() {
        for n_levels in 2..9 {
            let (full, full_par) = schedule(n_levels, usize::MAX);
            for tail_from in 0..n_levels {
                let (collapsed, par) = schedule(n_levels, tail_from);
                assert_eq!(
                    expand(&collapsed, n_levels, tail_from),
                    full,
                    "{n_levels} levels, tail from {tail_from}"
                );
                assert_eq!(par, full_par, "{n_levels} levels, tail from {tail_from}");
            }
            // Every non-last level flips once per solve; the last never does.
            assert_eq!(full_par[0], 1);
            assert_eq!(full_par[n_levels - 1], 0);
        }
        assert_eq!(schedule(1, usize::MAX).0, Vec::new());
    }

    #[test]
    fn a_plane_is_a_where_the_operation_says_it_is() {
        // Replay the operations on two physical planes per level, swapping the
        // roles through the parity the operations carry, and check the result is
        // the CPU's, which swaps the planes themselves.
        let (w, h) = (23, 17);
        let (index, weight) = pins(w, h);
        let mut cpu = MultigridSolver::from_index(w, h, &index, &weight);
        let start = cpu.clone();
        cpu.solve();
        let n_levels = start.layers.len();
        let (ops, par) = schedule(n_levels, usize::MAX);
        // planes[l][0 or 1] = a / d storage; the CPU layer's `a` starts in 0.
        let mut planes: Vec<[Vec<f32>; 2]> = start
            .layers
            .iter()
            .map(|l| [l.a.clone(), vec![f32::NAN; l.a.len()]])
            .collect();
        let mut layers = start.layers;
        for op in &ops {
            match *op {
                Op::Restrict { level, fa, ca } => {
                    // Present the fine `a` and take the coarse result into `ca`.
                    layers[level].a.clone_from(&planes[level][fa as usize]);
                    let (fine, coarse) = layers.split_at_mut(level + 1);
                    fine[level].restrict_to(&mut coarse[0]);
                    planes[level + 1][ca as usize].clone_from(&coarse[0].a);
                }
                Op::Prolong { level, fa, ca } => {
                    layers[level].a.clone_from(&planes[level][fa as usize]);
                    layers[level + 1]
                        .a
                        .clone_from(&planes[level + 1][ca as usize]);
                    let (fine, coarse) = layers.split_at_mut(level + 1);
                    coarse[0].prolong_from(&mut fine[level]);
                    planes[level][fa as usize].clone_from(&fine[level].a);
                }
                Op::Relax { level, fa } => {
                    layers[level].a.clone_from(&planes[level][fa as usize]);
                    layers[level].relax();
                    planes[level][1 - fa as usize].clone_from(&layers[level].a);
                }
                Op::Tail { .. } => unreachable!("no tail requested"),
            }
        }
        assert_eq!(planes[0][par[0] as usize], cpu.layers[0].a);
    }

    #[test]
    fn the_plan_memory_counts_the_hierarchy() {
        assert_eq!(plan_bytes(0, 10), 0);
        let (w, h) = (1000, 800);
        let bytes = plan_bytes(w, h);
        let pad = (1001 * 801) as u64;
        assert!(bytes > 16 * pad && bytes < 16 * pad * 2 + 20 * (w * h) as u64);
    }
}
