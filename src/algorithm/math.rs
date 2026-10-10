//! HAHOG numerical policies and VLFeat-compatible approximations.
use nalgebra::Matrix2;
use num_traits::Float;
use std::sync::OnceLock;
/// Two Newton iterations preserve the reference's histogram normalization.
pub(crate) fn vlfeat_sqrt(x: f32) -> f32 {
    if (x as f64) < 1e-8 {
        return 0.;
    }
    let half = 0.5 * x;
    let mut r = f32::from_bits(0x5f3759df - (x.to_bits() >> 1));
    r *= (-(half * r)).mul_add(r, 1.5);
    r *= (-(half * r)).mul_add(r, 1.5);
    x * r
}

/// Approximate the gradient angle using the reference cubic polynomial.
pub(crate) fn vlfeat_atan2(y: f32, x: f32) -> f32 {
    let a = y.abs() + f32::EPSILON;
    let (r, mut angle) = if x >= 0. {
        ((x - a) / (x + a), std::f32::consts::FRAC_PI_4)
    } else {
        ((x + a) / (a - x), 3. * std::f32::consts::FRAC_PI_4)
    };
    angle = ((0.1821 * r).mul_add(r, -0.9675)).mul_add(r, angle);
    if y < 0. {
        -angle
    } else {
        angle
    }
}

/// Wrap to [0, tau], retaining tau itself as required by histogram binning.
pub(crate) fn mod_tau(mut x: f32) -> f32 {
    while x > std::f32::consts::TAU {
        x -= std::f32::consts::TAU;
    }
    while x < 0. {
        x += std::f32::consts::TAU;
    }
    x
}

/// Pivoted elimination with the reference singularity cutoff and fused updates.
pub(crate) fn solve3(a: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    if a.iter().flatten().chain(&b).any(|v| !v.is_finite()) {
        return None;
    }
    let mut m = [[0.; 4]; 3];
    for i in 0..3 {
        m[i][..3].copy_from_slice(&a[i]);
        m[i][3] = b[i];
    }
    for j in 0..3 {
        let mut pivot = j;
        for i in j + 1..3 {
            if m[i][j].abs() > m[pivot][j].abs() {
                pivot = i;
            }
        }
        let divisor = m[pivot][j];
        if divisor.abs() < 1e-10 || divisor == 0. || !divisor.is_finite() {
            return None;
        }
        m.swap(pivot, j);
        for value in &mut m[j][j..] {
            *value /= divisor;
        }
        let pivot_row = m[j];
        for row in m.iter_mut().skip(j + 1) {
            let x = row[j];
            for (value, &pivot_value) in row[j..].iter_mut().zip(&pivot_row[j..]) {
                *value = (-x).mul_add(pivot_value, *value);
            }
        }
    }
    for i in (0..3).rev() {
        for ii in 0..i {
            m[ii][3] = (-m[ii][i]).mul_add(m[i][3], m[ii][3]);
        }
    }
    let solution = [m[0][3], m[1][3], m[2][3]];
    solution.iter().all(|v| v.is_finite()).then_some(solution)
}

/// Interpolate the reference exponential lookup on [0, 25], returning zero above it.
pub(crate) fn vlfeat_exp_neg(mut x: f64) -> f64 {
    static TABLE: OnceLock<[f64; 257]> = OnceLock::new();
    let t = TABLE.get_or_init(|| std::array::from_fn(|k| (-(k as f64) * (25. / 256.)).exp()));
    if x > 25. {
        return 0.;
    }
    x *= 256. / 25.;
    let i = x.floor() as usize;
    let i = i.min(255);
    lerp_fma(t[i], t[i + 1], x - i as f64)
}

/// Normalize in row order with the reference square-root approximation.
pub(crate) fn normalize(d: &mut [f32]) {
    let squared = d.iter().map(|&value| value * value).sum();
    let norm = vlfeat_sqrt(squared) + f32::EPSILON;
    for v in d {
        *v /= norm;
    }
}

/// Build symmetric Gaussian weights with the reference mixed-precision mass sum.
pub(crate) fn gaussian_kernel(sigma: f64) -> Vec<f32> {
    let r = (3. * sigma).ceil() as usize;
    let mut k = vec![1.; 2 * r + 1];
    let mut mass = 1f32;
    for i in 1..=r {
        let x = i as f64 / sigma;
        let g = (-0.5 * x * x).exp();
        mass = (mass as f64 + (g + g)) as f32;
        k[r - i] = g as f32;
        k[r + i] = g as f32;
    }
    for v in &mut k {
        *v /= mass;
    }
    k
}

/// Multiply a row-major matrix by a planar rotation using nalgebra's two-term product.
pub(crate) fn rotate2(a: [f32; 4], angle: f64) -> [f32; 4] {
    let c = angle.cos();
    let s = angle.sin();
    let product = Matrix2::new(a[0] as f64, a[1] as f64, a[2] as f64, a[3] as f64)
        * Matrix2::new(c, -s, s, c);
    [
        product[(0, 0)] as f32,
        product[(0, 1)] as f32,
        product[(1, 0)] as f32,
        product[(1, 1)] as f32,
    ]
}

/// Wrap angles before rounding, mapping a rounded full turn back to zero.
pub(crate) fn positive_degrees(angle: f64) -> f32 {
    let degrees = angle.to_degrees().rem_euclid(360.) as f32;
    if degrees >= 360. {
        0.
    } else {
        degrees
    }
}

/// A row-major 2×2 determinant with one fused multiply/subtract.
pub(crate) fn determinant2<T: Float>(a: [T; 4]) -> T {
    a[0].mul_add(a[3], -(a[1] * a[2]))
}

/// Sum two squares with the first product fused into the second.
pub(crate) fn squared_norm2_fma<T: Float>(x: T, y: T) -> T {
    x.mul_add(x, y * y)
}

/// Interpolate with a fused difference term; the weight is not clamped.
fn lerp_fma<T: Float>(a: T, b: T, t: T) -> T {
    t.mul_add(b - a, a)
}

/// Blend endpoints with the starting contribution fused.
fn lerp_start_fma<T: Float>(a: T, b: T, t: T) -> T {
    (T::one() - t).mul_add(a, t * b)
}

/// Interpolate row-major corners, fusing each starting contribution.
pub(crate) fn bilinear_start_fma<T: Float>(values: [T; 4], x: T, y: T) -> T {
    let top = lerp_start_fma(values[0], values[1], x);
    let bottom = lerp_start_fma(values[2], values[3], x);
    lerp_start_fma(top, bottom, y)
}

/// Locate the vertex from samples at -1, 0 and 1; zero curvature is nonfinite.
pub(crate) fn quadratic_peak_offset(left: f64, center: f64, right: f64) -> f64 {
    -0.5 * (right - left) / (right + left - 2. * center)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exercise a linear system that requires row pivoting.
    #[test]
    fn pivoted_solve_recovers_a_known_solution() {
        let matrix = [[0., 2., 1.], [1., -2., -3.], [3., -1., 2.]];
        let solution = solve3(matrix, [7., -12., 7.]).unwrap();
        assert!(solution
            .into_iter()
            .zip([1., 2., 3.])
            .all(|(a, b)| (a - b).abs() < 1e-12));
    }

    /// Check the singularity boundary and reject nonfinite inputs or solutions.
    #[test]
    fn pivot_cutoff_is_strict_and_nonfinite_inputs_are_rejected() {
        let diagonal = |x| [[x, 0., 0.], [0., 1., 0.], [0., 0., 1.]];
        assert_eq!(solve3(diagonal(1e-10), [1e-10, 2., 3.]), Some([1., 2., 3.]));
        for pivot in [1e-11, 0., f64::NAN, f64::INFINITY] {
            assert!(solve3(diagonal(pivot), [1.; 3]).is_none());
        }
        assert!(solve3(diagonal(1.), [f64::INFINITY; 3]).is_none());
        assert!(solve3(diagonal(1e-10), [f64::MAX; 3]).is_none());
    }

    /// Protect fused rounding and signed zero in determinant evaluation.
    #[test]
    fn determinant_retains_fused_subtraction_and_signed_zero() {
        let epsilon = f64::EPSILON;
        assert_eq!(
            determinant2([1. + epsilon, 1., 1., 1. - epsilon]),
            -epsilon * epsilon
        );
        assert_eq!(
            determinant2([-0f64, 0., 1., 1.]).to_bits(),
            (-0f64).to_bits()
        );
        let epsilon = f32::EPSILON;
        assert_eq!(
            determinant2([1. + epsilon, 1., 1., 1. - epsilon]),
            -epsilon * epsilon
        );
    }

    /// Distinguish the two fused interpolation orders and allow extrapolation.
    #[test]
    fn interpolation_retains_reference_rounding() {
        let a = f64::from_bits(0x40148153aeaf95a0);
        let b = f64::from_bits(0xc04f75832b812dd2);
        let t = f64::from_bits(0x3fb77270f1de72b0);
        assert_eq!(lerp_fma(a, b, t).to_bits(), 0xbff1b1b5b3732741);
        assert_eq!(lerp_start_fma(a, b, t).to_bits(), 0xbff1b1b5b3732742);
        let corners = [2f32, 6., 10., 14.];
        assert_eq!(bilinear_start_fma(corners, 0., 0.), 2.);
        assert_eq!(bilinear_start_fma(corners, 1., 1.), 14.);
        assert_eq!(bilinear_start_fma(corners, 0.25, 0.5), 7.);
        assert_eq!(bilinear_start_fma(corners, -0.5, 1.), 8.);
    }

    /// Check norm values, peak offsets and undefined flat-peak curvature.
    #[test]
    fn squared_norm_and_peak_interpolation_keep_their_contracts() {
        assert_eq!(squared_norm2_fma(3f32, 4.), 25.);
        assert_eq!(squared_norm2_fma(-3f64, 4.), 25.);
        assert_eq!(quadratic_peak_offset(-1., 0., -1.), 0.);
        assert_eq!(quadratic_peak_offset(-1.5625, -0.0625, -0.5625), 0.25);
        assert!(quadratic_peak_offset(1., 1., 1.).is_nan());
    }
}
