//! The trainer's small deterministic random source and the crop sampler.

use crate::data::CropParams;

/// A tiny dependency-free `xorshift64*` generator for scene order, crops and
/// scheduled sampling. Burn's own generator (weight initialisation) is seeded
/// separately by the trainer.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    /// A generator for `seed`; any seed works (the all-zero state is avoided).
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        // Avoid the all-zero state, which is a fixed point of xorshift.
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    /// The next 64 random bits.
    pub const fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform `f64` in `[0, 1)`.
    #[expect(
        clippy::cast_precision_loss,
        reason = "the value is shifted to 53 bits, which an f64 mantissa holds exactly"
    )]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform integer in `0..n` (0 when `n == 0`).
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the remainder is below n, which is a usize"
    )]
    pub const fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() % n as u64) as usize
        }
    }

    /// Fisher-Yates shuffle.
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = self.below(i + 1);
            slice.swap(i, j);
        }
    }
}

/// A random square crop of side `min(max_size, h, w)` that fits inside an image of
/// `dims = (height, width)`.
pub fn random_crop(dims: (usize, usize), max_size: usize, rng: &mut Rng) -> CropParams {
    let (h, w) = dims;
    let size = max_size.min(w).min(h);
    let top = rng.below(h - size + 1);
    let left = rng.below(w - size + 1);
    CropParams { top, left, size }
}
