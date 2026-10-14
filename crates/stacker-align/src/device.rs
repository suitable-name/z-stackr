//! A hook through which the registration objective can be evaluated on another
//! device (the GPU), without this crate depending on one.
//!
//! The objective of [`crate::refine`] warps the reference by the inverse of the
//! candidate matrix and takes the mean-corrected RMS of `d = source - warped`
//! over the pixels that received a sample. An [`ObjectiveDevice`] computes the
//! moments of `d` (count, mean, sum of squared deviations); the refiner turns
//! them into the RMS and applies the coverage rule exactly as on the CPU.
//!
//! A device may decline a level (too small to be worth it, too large for the
//! device) by returning `None` from [`ObjectiveDevice::level`], and an evaluation
//! may fail by returning `None` from [`LevelDevice::moments`]. Either way the
//! refiner evaluates on the CPU instead; a run never fails because of the device.
//! A device that fails remembers it and declines from then on (that is up to the
//! implementation; the caller reads the reason from it).
//!
//! The device result is not bit for bit the CPU's (fused multiply-adds, the
//! division and the order of the sums differ), so a refinement with a device can
//! take a slightly different path through the simplex. Both end within the same
//! sub-pixel tolerance.

use std::{fmt, future::Future, sync::Arc};

use nalgebra::Matrix3;

use crate::refine::search::{AlignSearch, Ask, SearchResult};

/// Which sampler warps the reference inside the objective.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectiveSampler {
    /// Catmull-Rom bicubic; pixels whose source lies outside the reference take
    /// no part ([`crate::transform::bicubic_interpolate_channel`]).
    Bicubic,
    /// The four-tap C0 cubic with edge-clamped borders
    /// ([`crate::transform::spline4x4_sample_clamped`]).
    SplineClamped,
}

/// Moments of `d = source - warped reference` over the pixels that took part.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Moments {
    /// Pixels that took part.
    pub count: u64,
    /// Mean of `d`.
    pub mean: f64,
    /// Sum of the squared deviations of `d` from its mean.
    pub m2: f64,
}

/// Something that evaluates the registration objective of one pyramid level.
pub trait ObjectiveDevice: Send + Sync {
    /// Prepare the evaluation of one level (`width x height`, row-major luma
    /// planes of equal size). `None` when the device declines; the level is then
    /// evaluated on the CPU.
    fn level(
        &self,
        reference: &[f32],
        source: &[f32],
        width: usize,
        height: usize,
        sampler: ObjectiveSampler,
    ) -> Option<Box<dyn LevelDevice>>;
}

/// One level prepared on a device.
pub trait LevelDevice {
    /// The moments for the warp by `inverse` (the inverse of the candidate
    /// matrix: output pixel `p` samples the reference at `inverse * p`). `None`
    /// when the device failed; this evaluation then runs on the CPU.
    fn moments(&mut self, inverse: &Matrix3<f32>) -> Option<Moments>;
}

/// A shared [`ObjectiveDevice`], as carried by the refinement options.
#[derive(Clone)]
pub struct DeviceHandle(pub Arc<dyn ObjectiveDevice>);

impl fmt::Debug for DeviceHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("DeviceHandle(..)")
    }
}

// ── the asynchronous form ─────────────────────────────────────────────────────

/// A device that answers asynchronously (a browser GPU cannot block on a readback).
///
/// The counterpart of [`ObjectiveDevice`] + [`LevelDevice`] for the resumable
/// searches of [`crate::refine::search`]: one object that is told the current
/// level and then asked for moments.
///
/// There is no `Send` bound: the web worker that drives it is single-threaded.
pub trait AsyncLevelDevice {
    /// Upload one level (the arguments of [`ObjectiveDevice::level`]). `false`
    /// declines it: the search then evaluates the level on the CPU, when the
    /// caller of [`drive`] allows that.
    fn level(
        &mut self,
        reference: &[f32],
        source: &[f32],
        width: usize,
        height: usize,
        sampler: ObjectiveSampler,
    ) -> impl Future<Output = bool>;

    /// The moments for the warp by `inverse` on the level uploaded last (see
    /// [`LevelDevice::moments`]). `None` when the device failed.
    fn moments(&mut self, inverse: &Matrix3<f32>) -> impl Future<Output = Option<Moments>>;
}

/// The device declined a level or failed an evaluation while the CPU was not
/// allowed to take over.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceRefused {
    /// What the device refused, for the message to the user.
    pub what: String,
}

impl fmt::Display for DeviceRefused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "the device refused: {}", self.what)
    }
}

impl std::error::Error for DeviceRefused {}

/// Run `search` to its end, answering its questions from `device`.
///
/// With `cpu_allowed` a declined level or a failed evaluation is evaluated by
/// the search itself on the CPU (what the synchronous refiners do); without it
/// (the browser rule: alignment is GPU-only) the search ends with
/// [`DeviceRefused`].
///
/// # Errors
/// [`DeviceRefused`] as above. The inner [`SearchResult`] is the refiner's own
/// outcome, errors included.
pub async fn drive<D: AsyncLevelDevice>(
    mut search: AlignSearch<'_>,
    device: &mut D,
    cpu_allowed: bool,
) -> Result<SearchResult, DeviceRefused> {
    loop {
        match search.ask() {
            Ask::Level { level } => {
                let accepted = {
                    let planes = search.level_planes(level);
                    device
                        .level(
                            planes.reference,
                            planes.source,
                            planes.width,
                            planes.height,
                            planes.sampler,
                        )
                        .await
                };
                if !accepted && !cpu_allowed {
                    return Err(DeviceRefused {
                        what: format!("pyramid level {level} was not accepted"),
                    });
                }
                search.level_ready(accepted);
            }
            Ask::Evaluate { inverse } => {
                let moments = device.moments(&inverse).await;
                if moments.is_none() && !cpu_allowed {
                    return Err(DeviceRefused {
                        what: "an objective evaluation failed".to_owned(),
                    });
                }
                search.tell(moments);
            }
            Ask::Done(result) => return Ok(result),
        }
    }
}

impl Moments {
    /// Combine the moments of two disjoint sets of pixels (Chan et al.), the way
    /// a device's per-group partials are added up.
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "pixel counts are far below 2^52"
    )]
    pub fn merge(self, other: Self) -> Self {
        if other.count == 0 {
            return self;
        }
        if self.count == 0 {
            return other;
        }
        let (na, nb) = (self.count as f64, other.count as f64);
        let n = na + nb;
        let delta = other.mean - self.mean;
        Self {
            count: self.count + other.count,
            mean: self.mean + delta * nb / n,
            m2: delta * delta * na * nb / n + self.m2 + other.m2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Moments;

    #[expect(clippy::cast_precision_loss, reason = "small test counts")]
    fn moments_of(values: &[f64]) -> Moments {
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let m2 = values.iter().map(|v| (v - mean) * (v - mean)).sum();
        Moments {
            count: values.len() as u64,
            mean,
            m2,
        }
    }

    #[test]
    fn merging_partials_gives_the_moments_of_the_whole() {
        let values: Vec<f64> = (0..100)
            .map(|i| f64::from(i % 7).mul_add(0.3, -0.5))
            .collect();
        let whole = moments_of(&values);
        let merged = moments_of(&values[..37])
            .merge(moments_of(&values[37..80]))
            .merge(moments_of(&values[80..]));
        assert_eq!(merged.count, whole.count);
        assert!((merged.mean - whole.mean).abs() < 1e-12);
        assert!((merged.m2 - whole.m2).abs() < 1e-9);
        let empty = Moments {
            count: 0,
            mean: 0.0,
            m2: 0.0,
        };
        assert_eq!(empty.merge(whole), whole);
        assert_eq!(whole.merge(empty), whole);
    }
}
