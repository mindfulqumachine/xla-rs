use const_soft_float::soft_f32::SoftF32;

/// Adds two f32 values in a const context using soft float arithmetic.
pub const fn const_f32_add(a: f32, b: f32) -> f32 {
    SoftF32(a).add(SoftF32(b)).to_f32()
}

/// Multiplies two f32 values in a const context using soft float arithmetic.
pub const fn const_f32_mul(a: f32, b: f32) -> f32 {
    SoftF32(a).mul(SoftF32(b)).to_f32()
}
