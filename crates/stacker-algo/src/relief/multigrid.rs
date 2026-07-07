use std::cmp;

/// A single layer in the multigrid depth-solver hierarchy.
pub struct MultigridLayer {
    pub width: usize,
    pub height: usize,
    pub a: Vec<f32>, // solution
    pub b: Vec<f32>, // weight (0.0 to 1.0)
    pub c: Vec<f32>, // target value
    pub d: Vec<f32>, // temp buffer for relaxation
}

impl MultigridLayer {
    pub fn new(
        width: usize,
        height: usize,
        initial_target: Option<&[f32]>,
        initial_weight: Option<&[f32]>,
    ) -> Self {
        let len = width * height;
        let mut a = vec![0.0; len];
        let mut b = vec![0.0; len];
        let mut c = vec![0.0; len];
        let d = vec![0.0; len];

        if let (Some(target), Some(weight)) = (initial_target, initial_weight) {
            for i in 0..len {
                if weight[i] >= 0.0 {
                    if weight[i] > 0.0 {
                        c[i] = target[i];
                        b[i] = 1.0;
                        a[i] = c[i];
                    } else {
                        c[i] = 0.0;
                        b[i] = 0.0;
                        a[i] = 0.0;
                    }
                }
            }
        }

        Self {
            width,
            height,
            a,
            b,
            c,
            d,
        }
    }

    /// Restrict from `self` (fine) to `coarse`.
    pub fn restrict_to(&self, coarse: &mut MultigridLayer) {
        let cw = coarse.width;
        let ch = coarse.height;
        let fw = self.width;
        let fh = self.height;

        for cy in 0..ch {
            for cx in 0..cw {
                let c_idx = cy * cw + cx;
                let fy0 = cy * 2;
                let fx0 = cx * 2;
                let idx00 = fy0 * fw + fx0;

                if fy0 + 1 < fh && fx0 + 1 < fw {
                    // Full 2x2 block
                    let idx01 = idx00 + 1;
                    let idx10 = (fy0 + 1) * fw + fx0;
                    let idx11 = idx10 + 1;

                    coarse.a[c_idx] =
                        (self.a[idx00] + self.a[idx01] + self.a[idx10] + self.a[idx11]) / 4.0;
                    coarse.b[c_idx] =
                        (self.b[idx00] + self.b[idx01] + self.b[idx10] + self.b[idx11]) / 4.0;
                    // c is only consumed where b > 0; explicit zero for clarity.
                    coarse.c[c_idx] = 0.0;
                    if coarse.b[c_idx] > 0.0 {
                        coarse.c[c_idx] = (self.b[idx00] * self.c[idx00]
                            + self.b[idx01] * self.c[idx01]
                            + self.b[idx10] * self.c[idx10]
                            + self.b[idx11] * self.c[idx11])
                            / coarse.b[c_idx]
                            / 4.0;
                    }
                } else if fy0 + 1 < fh {
                    // Right edge (1x2 block)
                    let idx10 = (fy0 + 1) * fw + fx0;
                    coarse.a[c_idx] = f32::midpoint(self.a[idx00], self.a[idx10]);
                    coarse.b[c_idx] = f32::midpoint(self.b[idx00], self.b[idx10]);
                    coarse.c[c_idx] = 0.0;
                    if coarse.b[c_idx] > 0.0 {
                        coarse.c[c_idx] = (self.b[idx00] * self.c[idx00]
                            + self.b[idx10] * self.c[idx10])
                            / coarse.b[c_idx]
                            / 2.0;
                    }
                } else if fx0 + 1 < fw {
                    // Bottom edge (2x1 block)
                    let idx01 = idx00 + 1;
                    coarse.a[c_idx] = f32::midpoint(self.a[idx00], self.a[idx01]);
                    coarse.b[c_idx] = f32::midpoint(self.b[idx00], self.b[idx01]);
                    coarse.c[c_idx] = 0.0;
                    if coarse.b[c_idx] > 0.0 {
                        coarse.c[c_idx] = (self.b[idx00] * self.c[idx00]
                            + self.b[idx01] * self.c[idx01])
                            / coarse.b[c_idx]
                            / 2.0;
                    }
                } else {
                    // Bottom-right corner (1x1 block)
                    coarse.a[c_idx] = self.a[idx00];
                    coarse.b[c_idx] = self.b[idx00];
                    // c is only consumed where b > 0; explicit zero for clarity.
                    coarse.c[c_idx] = 0.0;
                    if coarse.b[c_idx] > 0.0 {
                        coarse.c[c_idx] = (self.b[idx00] * self.c[idx00]) / coarse.b[c_idx] / 2.0;
                    }
                }
            }
        }
    }

    /// Prolong from `self` (coarse) to `fine`.
    pub fn prolong_from(&self, fine: &mut MultigridLayer) {
        let cw = self.width;
        let ch = self.height;
        let fw = fine.width;
        let fh = fine.height;

        for cy in 0..ch {
            for cx in 0..cw {
                let c_idx = cy * cw + cx;
                let fy0 = cy * 2;
                let fx0 = cx * 2;

                if self.b[c_idx] != 1.0 {
                    let mut val = self.a[c_idx];
                    let idx00 = fy0 * fw + fx0;
                    fine.a[idx00] = fine.b[idx00] * fine.c[idx00] + (1.0 - fine.b[idx00]) * val;

                    if fx0 < fw - 1 {
                        if cx < cw - 1 {
                            val = f32::midpoint(self.a[c_idx], self.a[c_idx + 1]);
                        }
                        let idx01 = fy0 * fw + fx0 + 1;
                        fine.a[idx01] = fine.b[idx01] * fine.c[idx01] + (1.0 - fine.b[idx01]) * val;

                        if fy0 < fh - 1 {
                            if cx < cw - 1 && cy < ch - 1 {
                                val = (self.a[c_idx]
                                    + self.a[c_idx + 1]
                                    + self.a[c_idx + cw]
                                    + self.a[c_idx + cw + 1])
                                    / 4.0;
                            }
                            let idx11 = (fy0 + 1) * fw + fx0 + 1;
                            fine.a[idx11] =
                                fine.b[idx11] * fine.c[idx11] + (1.0 - fine.b[idx11]) * val;
                        }
                    } else if fy0 < fh - 1 {
                        if cy < ch - 1 {
                            val = f32::midpoint(self.a[c_idx], self.a[c_idx + cw]);
                        }
                        let idx10 = (fy0 + 1) * fw + fx0;
                        fine.a[idx10] = fine.b[idx10] * fine.c[idx10] + (1.0 - fine.b[idx10]) * val;
                    }
                }
            }
        }
    }

    /// Relax: one Jacobi-style relaxation sweep.
    ///
    /// # GPU dispatch
    /// When compiled with the `gpu` feature, tries a `wgpu` compute-shader
    /// port first ([`crate::relief::gpu::relax_gpu`]) and falls back to the
    /// CPU implementation below on any failure (no adapter, the runtime
    /// switch off, or any `wgpu` call erroring) — never a panic. See
    /// `relief::gpu`'s module docs for the tolerance-equal (not bit-equal)
    /// GPU/CPU parity guarantee.
    pub fn relax(&mut self) {
        #[cfg(feature = "gpu")]
        if let Some(new_a) =
            crate::relief::gpu::relax_gpu(&self.a, &self.b, &self.c, self.width, self.height)
        {
            self.a = new_a;
            return;
        }

        let w = self.width;
        let h = self.height;

        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;

                let neighbourhood_val = if self.b[idx] > 0.0 {
                    self.c[idx]
                } else {
                    let mut sum = 0.0;
                    let mut count = 0.0;
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            let ny = y as isize + dy;
                            let nx = x as isize + dx;
                            if ny >= 0 && ny < h as isize && nx >= 0 && nx < w as isize {
                                sum += self.a[ny as usize * w + nx as usize];
                                count += 1.0;
                            }
                        }
                    }
                    sum / count
                };

                self.d[idx] = self.b[idx] * self.c[idx] + (1.0 - self.b[idx]) * neighbourhood_val;
            }
        }

        self.a.copy_from_slice(&self.d);
    }
}

pub struct MultigridSolver {
    pub layers: Vec<MultigridLayer>,
    pub width: usize,
    pub height: usize,
}

impl MultigridSolver {
    pub fn new(width: usize, height: usize, target: &[f32], weight: &[f32]) -> Self {
        let pad_w = (width / 2) * 2 + 1;
        let pad_h = (height / 2) * 2 + 1;

        let mut padded_target = vec![0.0; pad_w * pad_h];
        let mut padded_weight = vec![0.0; pad_w * pad_h];

        for y in 0..pad_h {
            for x in 0..pad_w {
                let src_y = cmp::min(y, height - 1);
                let src_x = cmp::min(x, width - 1);
                let src_idx = src_y * width + src_x;
                let dst_idx = y * pad_w + x;
                padded_target[dst_idx] = target[src_idx];
                padded_weight[dst_idx] = weight[src_idx];
            }
        }

        let mut layers = Vec::new();
        layers.push(MultigridLayer::new(
            pad_w,
            pad_h,
            Some(&padded_target),
            Some(&padded_weight),
        ));

        let mut curr_w = pad_w;
        let mut curr_h = pad_h;

        while curr_w > 2 && curr_h > 2 {
            curr_w = (curr_w + 1) / 2;
            curr_h = (curr_h + 1) / 2;
            layers.push(MultigridLayer::new(curr_w, curr_h, None, None));
        }

        Self {
            layers,
            width,
            height,
        }
    }

    pub fn solve(&mut self) {
        self.cycle(0);
    }

    fn cycle(&mut self, level: usize) {
        // Base case: the coarsest level intentionally does nothing here (no
        // relaxation is performed at `self.layers.len() - 1`). This is a
        // deliberate, behaviour-preserving no-op — the coarsest layer is
        // constructed small enough (`new` stops subdividing once both
        // dimensions are `<= 2`) that its `a` values are already fully
        // determined by the preceding `restrict_to` averaging, and every
        // finer level still receives its full quota of `relax()` passes
        // after `prolong_from` on the way back up. Do not add relaxation
        // here without re-validating the full engine's output against the
        // synthetic-source-selection test in
        // `docs/relief-multigrid-design-notes.md`.
        if level < self.layers.len() - 1 {
            for _ in 0..3 {
                let (left, right) = self.layers.split_at_mut(level + 1);
                left[level].restrict_to(&mut right[0]);

                self.cycle(level + 1);

                let (left, right) = self.layers.split_at_mut(level + 1);
                right[0].prolong_from(&mut left[level]);

                for _ in 0..3 {
                    self.layers[level].relax();
                }
            }
        }
    }

    pub fn get_solution(&self) -> Vec<f32> {
        let finest = &self.layers[0];
        let mut out = vec![0.0; self.width * self.height];
        for y in 0..self.height {
            for x in 0..self.width {
                out[y * self.width + x] = finest.a[y * finest.width + x];
            }
        }
        out
    }

    /// Post-solve smoothing: applies `crate::relief::pyramid::pyramid_smooth`
    /// (see its docs for the algorithm) to the padded finest-level
    /// solution, then crops to `width x height`.
    ///
    /// `max_scale` is `relief_smoothing_radius`.
    pub fn get_smoothed_solution(&self, max_scale: usize) -> Vec<f32> {
        let finest = &self.layers[0];
        let smoothed_padded = crate::relief::pyramid::pyramid_smooth(
            &finest.a,
            finest.width,
            finest.height,
            max_scale,
        );

        let mut out = vec![0.0; self.width * self.height];
        for y in 0..self.height {
            for x in 0..self.width {
                out[y * self.width + x] = smoothed_padded[y * finest.width + x];
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::MultigridSolver;

    /// A constant target field with full weight (confidence 1.0) everywhere
    /// must solve back to that same constant, since every pixel is already
    /// pinned by `b == 1.0` and `relax()`/`prolong_from()` both preserve
    /// pinned values (`fine.b[idx] * fine.c[idx] + (1 - fine.b[idx]) * val`
    /// reduces to `fine.c[idx]` when `b == 1.0`).
    #[test]
    fn constant_target_with_full_weight_solves_to_constant() {
        const CONST: f32 = 0.42;
        let (w, h) = (10, 8);
        let target = vec![CONST; w * h];
        let weight = vec![1.0; w * h];

        let mut solver = MultigridSolver::new(w, h, &target, &weight);
        solver.solve();
        let solution = solver.get_solution();

        for (i, &v) in solution.iter().enumerate() {
            assert!(v.is_finite(), "solution[{i}] is not finite: {v}");
            assert!(
                (v - CONST).abs() < 1e-4,
                "solution[{i}] = {v}, expected {CONST}"
            );
        }
    }

    /// Constrained pixels (weight == 1.0) must keep their target value after
    /// `solve()`, within a small tolerance — they are pinned boundary
    /// conditions for the diffusion, not free variables.
    #[test]
    fn constrained_pixels_keep_their_target_values() {
        let (w, h) = (12, 12);
        let mut target = vec![0.0_f32; w * h];
        let mut weight = vec![0.0_f32; w * h];

        // Pin the left column to 0.0 and the right column to 1.0; leave the
        // interior unconstrained.
        for y in 0..h {
            target[y * w] = 0.0;
            weight[y * w] = 1.0;
            target[y * w + (w - 1)] = 1.0;
            weight[y * w + (w - 1)] = 1.0;
        }

        let mut solver = MultigridSolver::new(w, h, &target, &weight);
        solver.solve();
        let solution = solver.get_solution();

        for y in 0..h {
            let left = solution[y * w];
            let right = solution[y * w + (w - 1)];
            assert!(
                (left - 0.0).abs() < 1e-3,
                "left-column pixel at row {y} drifted: {left}"
            );
            assert!(
                (right - 1.0).abs() < 1e-3,
                "right-column pixel at row {y} drifted: {right}"
            );
        }
    }

    /// An unconstrained hole between two constrained regions (0.0 on the
    /// left, 1.0 on the right) must be filled with intermediate values by
    /// the diffusion — no NaN, and every solved value stays within the
    /// `[min, max]` bracket of the constraints.
    #[test]
    fn unconstrained_hole_is_filled_with_intermediate_values() {
        let (w, h) = (16, 16);
        let mut target = vec![0.0_f32; w * h];
        let mut weight = vec![0.0_f32; w * h];

        for y in 0..h {
            target[y * w] = 0.0;
            weight[y * w] = 1.0;
            target[y * w + (w - 1)] = 1.0;
            weight[y * w + (w - 1)] = 1.0;
        }

        let mut solver = MultigridSolver::new(w, h, &target, &weight);
        solver.solve();
        let solution = solver.get_solution();

        for (i, &v) in solution.iter().enumerate() {
            assert!(v.is_finite(), "solution[{i}] is not finite: {v}");
            assert!(
                (0.0..=1.0).contains(&v),
                "solution[{i}] = {v} outside constraint bracket [0, 1]"
            );
        }

        // The hole's centre column should be a genuine blend, not exactly
        // equal to either constrained endpoint.
        let mid_x = w / 2;
        let mid_y = h / 2;
        let mid_val = solution[mid_y * w + mid_x];
        assert!(
            mid_val > 0.05 && mid_val < 0.95,
            "centre pixel {mid_val} is not a genuine interior blend"
        );
    }
}
