//! Built-in network architectures for the crate's four architecture
//! families: [`FocusMergeNet`] (pairwise fusion), [`BatchMergeNet`] (batch
//! fusion), [`BatchAlignNet`] (batch alignment), and [`FusionAlignNet`]
//! (pairwise alignment). Each implements the corresponding trait from
//! [`crate::traits`] and is configured via a companion `*Config` type built
//! from a shared [`ModelSize`] preset ([`size`]).
//!
//! ## Shared building blocks ([`blocks`])
//!
//! [`FocusMergeNet`] and [`BatchMergeNet`] both build on the same
//! fully-convolutional, size-preserving primitives in [`blocks`]:
//! `ConvNormAct` (conv → `GroupNorm` → GELU) and the dilated `ResBlock`
//! (dilations cycling `1 → 2 → 4 → 8` across the context stack, growing the
//! receptive field geometrically **without any pooling**, so tiled inference
//! stays seam-free). `GroupNorm` only, never `BatchNorm`, so statistics never
//! depend on tile or batch size. Both blocks are `pub(crate)` — internal
//! implementation detail of the built-in architectures, not part of the
//! extensibility surface (see [`crate::traits`] for that).
//!
//! [`BatchAlignNet`] (v2) is architecturally different: it needs a *strided*
//! (downsampling) encoder rather than a size-preserving one, since it
//! predicts 6 numbers per frame rather than a per-pixel output, plus an
//! explicit local-correlation comparison against the reference frame. Its
//! private stride-2 variant of `ConvNormAct` and its correlation/geometric
//! head live in [`batch_align`]; see that module's docs (which reference
//! `docs/batchalign-v2-design.md` §3.4/§3.5) for the bounded-parameterisation
//! and reference-normalisation design.
//!
//! [`FusionAlignNet`] (in [`fusion_align`]) is the streaming, two-input
//! sibling of `BatchAlignNet`: rather than duplicating the strided
//! encoder/correlation/bounded-head machinery, it **reuses** `batch_align`'s
//! private building blocks directly (`ConvNormActS`, `l2_normalize`,
//! `local_correlation`, and the shared `bounded_affine_from_raw6` helper that
//! composes the bounded geometric matrix from 6 raw head outputs) — only the
//! shape of its two encoder inputs (a `reference`/`frame` pair rather than a
//! whole `[N,S,...]` stack) and the absence of `BatchAlignNet`'s §3.5
//! reference-normalisation step differ. See `docs/fusionalign-design.md` for
//! the full rationale (in particular why no reference-normalisation is
//! applied here) and [`crate::traits::PairAlignmentModel`]'s docs for the
//! trait contract.

pub mod batch_align;
pub mod batch_merge;
pub mod blocks;
pub mod focus_merge;
pub mod fusion_align;
pub mod size;

pub use batch_align::{BatchAlignNet, BatchAlignNetConfig, local_correlation};
pub use batch_merge::{BatchMergeNet, BatchMergeNetConfig};
pub use focus_merge::{FocusMergeNet, FocusMergeNetConfig, MergeOutput};
pub use fusion_align::{FusionAlignNet, FusionAlignNetConfig};
pub use size::ModelSize;
