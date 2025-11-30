//! Compile-time Floating Point Arithmetic.
//!
//! # The Problem: `const` Floats
//!
//! In Rust (as of 2024), standard floating-point operations (`+`, `*`, etc.) are **not allowed**
//! in `const` functions. This is because floating-point math depends on hardware specifics (like rounding modes)
//! that aren't guaranteed to be identical at compile-time vs run-time.
//!
//! # The Solution: Soft Floats
//!
//! To enable `ConstDevice` tensors to perform math (like matrix multiplication) at compile time,
//! we use "Soft Floats". This is a software implementation of IEEE 754 floating-point arithmetic
//! that uses integer operations. Since integer math is allowed in `const`, we can bypass the restriction.
//!
//! This module wraps the `const_soft_float` crate to provide these operations.

use const_soft_float::soft_f32::SoftF32;

/// Adds two f32 values in a const context.
///
/// Uses software emulation to perform addition at compile time.
pub const fn const_f32_add(a: f32, b: f32) -> f32 {
    SoftF32(a).add(SoftF32(b)).to_f32()
}

/// Multiplies two f32 values in a const context.
///
/// Uses software emulation to perform multiplication at compile time.
pub const fn const_f32_mul(a: f32, b: f32) -> f32 {
    SoftF32(a).mul(SoftF32(b)).to_f32()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_const_ops() {
        assert_eq!(const_f32_add(1.0, 2.0), 3.0);
        assert_eq!(const_f32_mul(2.0, 3.0), 6.0);
    }
}
