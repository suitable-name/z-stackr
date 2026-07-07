#![allow(clippy::must_use_candidate)]

use num_traits::Float;
use rayon::prelude::*;

/// Planar `Y`/`Cb`/`Cr` image storage, one contiguous `Vec<T>` per channel.
///
/// # Colour-space contract
///
/// Channels hold **gamma-encoded (sRGB) BT.601 `Y`/`Cb`/`Cr`** values with
/// no transfer function applied — this matches every loader in the
/// workspace (`stacker_pipeline::load::dynamic_to_planar` and the GUI's
/// `image_utils::load_as_planar`), both of which store raw gamma-encoded
/// samples directly. Consumers must not assume linear light here.
#[derive(Clone, Debug)]
pub struct PlanarImage<T: Float + Send + Sync> {
    pub width: usize,
    pub height: usize,
    pub luma: Vec<T>,
    pub chroma_a: Vec<T>,
    pub chroma_b: Vec<T>,
}

impl<T: Float + Send + Sync> PlanarImage<T> {
    pub fn new(w: usize, h: usize) -> Self {
        Self {
            width: w,
            height: h,
            luma: vec![T::zero(); w * h],
            chroma_a: vec![T::zero(); w * h],
            chroma_b: vec![T::zero(); w * h],
        }
    }

    pub fn luma_chunks_mut(
        &mut self,
        chunk_size: usize,
    ) -> impl rayon::iter::ParallelIterator<Item = &mut [T]> {
        self.luma.par_chunks_mut(chunk_size)
    }
}
