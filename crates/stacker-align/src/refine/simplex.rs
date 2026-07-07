/// Maximum Nelder-Mead iterations before declaring convergence.
pub const NM_MAX_ITER: usize = 400;

/// Nelder-Mead convergence tolerance: stops when the simplex spread in
/// objective-function value falls below this threshold.
pub const NM_TOL: f64 = 1.0e-7;

/// Generic Nelder-Mead over an arbitrary-dimension `Vec<f64>`.
pub fn nelder_mead_generic<F: Fn(&[f64]) -> f64>(
    f: &F,
    x0: &[f64],
    step: f64,
    max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    let n = x0.len();
    let (alpha, gamma, rho, sigma) = (1.0_f64, 2.0_f64, 0.5_f64, 0.5_f64);

    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(x0.to_vec());
    for i in 0..n {
        let mut v = x0.to_vec();
        v[i] += step;
        simplex.push(v);
    }
    let mut fvals: Vec<f64> = simplex.iter().map(|v| f(v)).collect();

    for _ in 0..max_iter {
        let mut order: Vec<usize> = (0..=n).collect();
        order.sort_unstable_by(|&a, &b| {
            fvals[a]
                .partial_cmp(&fvals[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        simplex = order.iter().map(|&i| simplex[i].clone()).collect();
        fvals = order.iter().map(|&i| fvals[i]).collect();

        let f_spread = fvals[n] - fvals[0];
        if f_spread < tol && f_spread >= 0.0 {
            break;
        }

        let mut centroid = vec![0.0_f64; n];
        for v in &simplex[..n] {
            for (cj, vj) in centroid.iter_mut().zip(v.iter()) {
                *cj += vj;
            }
        }
        for cj in &mut centroid {
            *cj /= n as f64;
        }

        let xr: Vec<f64> = (0..n)
            .map(|j| centroid[j] + alpha * (centroid[j] - simplex[n][j]))
            .collect();
        let fr = f(&xr);

        if fr < fvals[0] {
            let xe: Vec<f64> = (0..n)
                .map(|j| centroid[j] + gamma * (xr[j] - centroid[j]))
                .collect();
            let fe = f(&xe);
            if fe < fr {
                simplex[n] = xe;
                fvals[n] = fe;
            } else {
                simplex[n] = xr;
                fvals[n] = fr;
            }
        } else if fr < fvals[n - 1] {
            simplex[n] = xr;
            fvals[n] = fr;
        } else {
            let (xc, fc) = if fr < fvals[n] {
                let xc: Vec<f64> = (0..n)
                    .map(|j| centroid[j] + rho * (xr[j] - centroid[j]))
                    .collect();
                let fc = f(&xc);
                (xc, fc)
            } else {
                let xc: Vec<f64> = (0..n)
                    .map(|j| centroid[j] - rho * (centroid[j] - simplex[n][j]))
                    .collect();
                let fc = f(&xc);
                (xc, fc)
            };
            if fc < fvals[n] {
                simplex[n] = xc;
                fvals[n] = fc;
            } else {
                let x0_best = simplex[0].clone();
                for i in 1..=n {
                    for j in 0..n {
                        simplex[i][j] = x0_best[j] + sigma * (simplex[i][j] - x0_best[j]);
                    }
                    fvals[i] = f(&simplex[i]);
                }
            }
        }
    }

    let mut order: Vec<usize> = (0..=n).collect();
    order.sort_unstable_by(|&a, &b| {
        fvals[a]
            .partial_cmp(&fvals[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    simplex[order[0]].clone()
}

/// Golden-section minimiser for the single-DOF bounded case.
pub fn brent_min<F: Fn(f64) -> f64>(mut a: f64, mut b: f64, tol: f64, f: &F) -> f64 {
    const R: f64 = 0.618_033_988_749_895;
    const C: f64 = 1.0 - R;
    let mut x1 = a + C * (b - a);
    let mut x2 = a + R * (b - a);
    let mut f1 = f(x1);
    let mut f2 = f(x2);
    for _ in 0..200 {
        if (b - a).abs() < tol {
            break;
        }
        if f1 < f2 {
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = a + C * (b - a);
            f1 = f(x1);
        } else {
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = a + R * (b - a);
            f2 = f(x2);
        }
    }
    if f1 < f2 { x1 } else { x2 }
}
