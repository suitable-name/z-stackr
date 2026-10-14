//! Named capacity presets shared by every architecture.

use serde::{Deserialize, Serialize};

/// Named capacity presets.
///
/// Larger presets widen the channels and deepen the dilated-context stack; the
/// architecture is otherwise identical, so a model trained at any size loads and
/// runs through the same code path. The alignment networks use their own
/// `(width, depth)` table for the same six names (see [`ModelSize::align_dims`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelSize {
    /// Extra small — 16 ch, 2 context blocks. Fastest; quick experiments.
    Xs,
    /// Small — 24 ch, 3 context blocks.
    S,
    /// Medium — 32 ch, 4 context blocks. Sensible default.
    M,
    /// Large — 48 ch, 6 context blocks.
    L,
    /// Extra large — 64 ch, 9 context blocks.
    Xl,
    /// Double extra large — 96 ch, 12 context blocks. For the A100.
    Xxl,
}

impl ModelSize {
    /// All presets, smallest to largest.
    pub const ALL: [Self; 6] = [Self::Xs, Self::S, Self::M, Self::L, Self::Xl, Self::Xxl];

    /// Lowercase tag (`"xs"`, `"s"`, …) — matches the serde representation.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Xs => "xs",
            Self::S => "s",
            Self::M => "m",
            Self::L => "l",
            Self::Xl => "xl",
            Self::Xxl => "xxl",
        }
    }

    /// Parse a preset tag, case-insensitively. Returns `None` if unrecognised.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "xs" => Some(Self::Xs),
            "s" => Some(Self::S),
            "m" => Some(Self::M),
            "l" => Some(Self::L),
            "xl" => Some(Self::Xl),
            "xxl" => Some(Self::Xxl),
            _ => None,
        }
    }

    /// `(width, depth)` of the fusion networks for this preset. Width is always a
    /// multiple of 8, so any power-of-two group count up to 8 divides every
    /// feature map.
    pub(crate) const fn dims(self) -> (usize, usize) {
        match self {
            Self::Xs => (16, 2),
            Self::S => (24, 3),
            Self::M => (32, 4),
            Self::L => (48, 6),
            Self::Xl => (64, 9),
            Self::Xxl => (96, 12),
        }
    }

    /// `(width, depth)` of the alignment networks
    /// ([`crate::model::BatchAlignNet`], [`crate::model::FusionAlignNet`]) for this
    /// preset. `depth` counts the residual blocks in *each* of the two encoder
    /// stages. Width is always even.
    pub(crate) const fn align_dims(self) -> (usize, usize) {
        match self {
            Self::Xs => (32, 1),
            Self::S => (48, 1),
            Self::M => (64, 2),
            Self::L => (96, 2),
            Self::Xl => (128, 3),
            Self::Xxl => (160, 3),
        }
    }
}
