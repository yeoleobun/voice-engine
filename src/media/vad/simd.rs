#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
pub unsafe fn dot_product_avx(w: &[f32], x: &[f32]) -> f32 {
    let len = w.len();
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;

    // Process 8 elements at a time
    while i + 8 <= len {
        let w_vec = _mm256_loadu_ps(w.as_ptr().add(i));
        let x_vec = _mm256_loadu_ps(x.as_ptr().add(i));
        // If FMA is available, we could use _mm256_fmadd_ps, but let's stick to mul+add for broader compatibility or check feature
        // Actually, let's just use mul and add.
        let prod = _mm256_mul_ps(w_vec, x_vec);
        sum = _mm256_add_ps(sum, prod);
        i += 8;
    }

    // Horizontal sum
    // Extract values from AVX register
    let mut temp = [0.0f32; 8];
    _mm256_storeu_ps(temp.as_mut_ptr(), sum);
    let mut total = temp.iter().sum::<f32>();

    // Handle remaining elements
    while i < len {
        total += w[i] * x[i];
        i += 1;
    }

    total
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx", enable = "fma")]
pub unsafe fn dot_product_fma(w: &[f32], x: &[f32]) -> f32 {
    let len = w.len();
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;

    while i + 8 <= len {
        let w_vec = _mm256_loadu_ps(w.as_ptr().add(i));
        let x_vec = _mm256_loadu_ps(x.as_ptr().add(i));
        sum = _mm256_fmadd_ps(w_vec, x_vec, sum);
        i += 8;
    }

    let mut temp = [0.0f32; 8];
    _mm256_storeu_ps(temp.as_mut_ptr(), sum);
    let mut total = temp.iter().sum::<f32>();

    while i < len {
        total += w[i] * x[i];
        i += 1;
    }

    total
}

pub fn dot_product(w: &[f32], x: &[f32]) -> f32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("fma") {
            return unsafe { dot_product_fma(w, x) };
        } else if is_x86_feature_detected!("avx") {
            return unsafe { dot_product_avx(w, x) };
        }
    }

    w.iter().zip(x.iter()).map(|(a, b)| a * b).sum()
}
