#[test]
fn test_environment_setup() {
    println!("Welcome to xla-rs!");
    println!("If you see this, your Rust environment is correctly set up.");

    // Verify we can allocate a vector
    let v: Vec<f32> = vec![1.0, 2.0, 3.0];
    assert_eq!(v.len(), 3);

    // Verify rayon is working
    use rayon::prelude::*;
    let sum: f32 = v.par_iter().sum();
    assert_eq!(sum, 6.0);
}
