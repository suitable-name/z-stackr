#![feature(portable_simd)]
#![allow(
    unused_variables,
    dead_code,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::uninlined_format_args,
    clippy::use_self,
    clippy::missing_panics_doc,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::suboptimal_flops,
    clippy::suspicious_arithmetic_impl,
    clippy::float_cmp,
    clippy::cast_possible_wrap,
    clippy::manual_div_ceil
)]
pub mod apex {
    pub mod fuse;
    #[cfg(feature = "gpu")]
    pub mod gpu;
    pub mod pyramid;
}
pub mod relief {
    pub mod focus;
    pub mod fuse;
    #[cfg(feature = "gpu")]
    pub mod gpu;
    pub mod guided;
    pub mod multigrid;
    pub mod pyramid;
    pub mod threshold;
}
pub mod hybrid {
    pub mod retouch;
}
pub mod optimize;
pub mod strata;
