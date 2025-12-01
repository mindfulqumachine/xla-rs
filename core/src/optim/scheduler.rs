//! Learning Rate Schedulers.
//!
//! This module provides common learning rate schedules for training Transformers.

use std::f32::consts::PI;

/// A trait for learning rate schedulers.
pub trait LRScheduler {
    /// Calculates the learning rate for a given step.
    fn get_lr(&self, step: usize) -> f32;
}

/// Linear Warmup with optional Linear Decay.
///
/// Increases LR from 0 to `max_lr` over `warmup_steps`.
/// Then decays linearly to `min_lr` over `total_steps - warmup_steps`.
pub struct LinearWarmup {
    pub max_lr: f32,
    pub min_lr: f32,
    pub warmup_steps: usize,
    pub total_steps: usize,
}

impl LinearWarmup {
    pub fn new(max_lr: f32, warmup_steps: usize, total_steps: usize) -> Self {
        Self {
            max_lr,
            min_lr: 0.0,
            warmup_steps,
            total_steps,
        }
    }

    pub fn with_min_lr(mut self, min_lr: f32) -> Self {
        self.min_lr = min_lr;
        self
    }
}

impl LRScheduler for LinearWarmup {
    fn get_lr(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            self.max_lr * (step as f32 / self.warmup_steps as f32)
        } else if step < self.total_steps {
            let decay_steps = self.total_steps - self.warmup_steps;
            let current_decay_step = step - self.warmup_steps;
            let progress = current_decay_step as f32 / decay_steps as f32;
            self.max_lr - (self.max_lr - self.min_lr) * progress
        } else {
            self.min_lr
        }
    }
}

/// Cosine Decay with Warmup.
///
/// Increases LR from 0 to `max_lr` over `warmup_steps`.
/// Then decays following a cosine curve to `min_lr`.
pub struct CosineDecay {
    pub max_lr: f32,
    pub min_lr: f32,
    pub warmup_steps: usize,
    pub total_steps: usize,
}

impl CosineDecay {
    pub fn new(max_lr: f32, warmup_steps: usize, total_steps: usize) -> Self {
        Self {
            max_lr,
            min_lr: 0.0,
            warmup_steps,
            total_steps,
        }
    }

    pub fn with_min_lr(mut self, min_lr: f32) -> Self {
        self.min_lr = min_lr;
        self
    }
}

impl LRScheduler for CosineDecay {
    fn get_lr(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            self.max_lr * (step as f32 / self.warmup_steps as f32)
        } else if step < self.total_steps {
            let decay_steps = self.total_steps - self.warmup_steps;
            let current_decay_step = step - self.warmup_steps;
            let progress = current_decay_step as f32 / decay_steps as f32;
            let cosine_decay = 0.5 * (1.0 + (progress * PI).cos());
            self.min_lr + (self.max_lr - self.min_lr) * cosine_decay
        } else {
            self.min_lr
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_warmup() {
        let scheduler = LinearWarmup::new(1.0, 10, 20);

        // Warmup
        assert_eq!(scheduler.get_lr(0), 0.0);
        assert_eq!(scheduler.get_lr(5), 0.5);

        // Peak (at step 10, it starts decaying, but logic says < warmup_steps)
        // step 10 is >= 10, so decay starts.
        // progress = (10-10)/10 = 0. LR = 1.0 - 0 = 1.0
        assert_eq!(scheduler.get_lr(10), 1.0);

        // Decay
        // step 15: progress = 5/10 = 0.5. LR = 1.0 - 0.5 = 0.5
        assert_eq!(scheduler.get_lr(15), 0.5);

        // End
        assert_eq!(scheduler.get_lr(20), 0.0);
        assert_eq!(scheduler.get_lr(100), 0.0);
    }

    #[test]
    fn test_cosine_decay() {
        let scheduler = CosineDecay::new(1.0, 10, 20);

        // Warmup
        assert_eq!(scheduler.get_lr(0), 0.0);
        assert_eq!(scheduler.get_lr(5), 0.5);

        // Peak
        assert_eq!(scheduler.get_lr(10), 1.0);

        // Decay
        // step 15: progress = 0.5. cos(0.5pi) = 0. 0.5 * (1+0) = 0.5.
        // LR = 0 + 1 * 0.5 = 0.5
        assert!((scheduler.get_lr(15) - 0.5).abs() < 1e-6);

        // End
        assert_eq!(scheduler.get_lr(20), 0.0);
    }
}
