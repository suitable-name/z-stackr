use serde::{Deserialize, Serialize};

/// Named capacity presets.
///
/// Larger presets widen the channels and deepen the dilated-context stack; the
/// architecture is otherwise identical, so a model trained at any size loads and
/// runs through the same code path.
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

    /// `(width, depth)` for this preset. Width is always a multiple of 8 so the
    /// fixed `GroupNorm` group count (8) divides every feature map.
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
}
