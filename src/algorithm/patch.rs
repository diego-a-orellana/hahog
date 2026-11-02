//! Canonical bilinear patches; boundary behavior deliberately preserves the reference.
use super::{
    pyramid::{self, Pyramid},
    Frame, Plane,
};

/// Sample a square of side `2 * resolution + 1` over canonical coordinates ±`extent`.
/// Reuse the output storage and return its Gaussian sigma in canonical frame units.
/// Frames remain isotropic because affine shape adaptation is disabled.
/// Padding on either axis preserves the reference's truncated right sampling limit.
pub(super) fn sample(
    pyramid: &Pyramid,
    frame: Frame,
    resolution: usize,
    extent: f64,
    output: &mut Plane,
) -> f64 {
    // Column-major basis mapping canonical coordinates into original-image pixels.
    let basis = [
        frame.a11 as f64,
        frame.a21 as f64,
        frame.a12 as f64,
        frame.a22 as f64,
    ];
    let scale = super::math::squared_norm2_fma(basis[0], basis[1]).sqrt();
    let inverse_scale = 1. / scale;
    let log_scale = (1. / (inverse_scale * pyramid::base_sigma())).ln() * std::f64::consts::LOG2_E;
    let subdivision_for_octave = |octave: usize| {
        ((log_scale - octave as f64).floor() as i32).clamp(
            pyramid::FIRST_SUBDIVISION,
            pyramid::SUBDIVISIONS_PER_OCTAVE as i32,
        )
    };
    let last = pyramid.octaves.len() - 1;
    let mut octave = last;
    for candidate in 1..=last {
        if inverse_scale * pyramid::sigma(candidate, subdivision_for_octave(candidate)) > 1. {
            octave = candidate - 1;
            break;
        }
    }
    let subdivision = subdivision_for_octave(octave);
    let canonical_sigma = pyramid::sigma(octave, subdivision) / scale;
    let image = &pyramid.octaves[octave][(subdivision - pyramid::FIRST_SUBDIVISION) as usize];
    let octave_step = 2f64.powi(octave as i32);
    let basis = basis.map(|v| v / octave_step);
    let center_x = frame.x as f64 / octave_step;
    let center_y = frame.y as f64 / octave_step;
    let mut xmin = f64::INFINITY;
    let mut xmax = f64::NEG_INFINITY;
    let mut ymin = f64::INFINITY;
    let mut ymax = f64::NEG_INFINITY;
    for (x, y) in [
        (extent, -extent),
        (extent, extent),
        (-extent, extent),
        (-extent, -extent),
    ] {
        let xx = basis[0].mul_add(x, basis[2] * y) + center_x;
        let yy = basis[1].mul_add(x, basis[3] * y) + center_y;
        xmin = xmin.min(xx);
        xmax = xmax.max(xx);
        ymin = ymin.min(yy);
        ymax = ymax.max(yy);
    }
    let x0 = xmin.floor() as isize - 1;
    let x1 = xmax.ceil() as isize + 1;
    let y0 = ymin.floor() as isize - 1;
    let y1 = ymax.ceil() as isize + 1;
    let needs_padding =
        x0 < 0 || y0 < 0 || x1 >= image.width as isize || y1 >= image.height as isize;
    if needs_padding {
        // Reference padding repeats the penultimate column of the cropped source window.
        let right_limit = (x1 - (x1 - image.width as isize + 1).max(0) - 1)
            .max(x0.max(0))
            .min(image.width as isize - 1);
        sample_grid(
            basis,
            [center_x, center_y],
            resolution,
            extent,
            output,
            |x, y| {
                image.values[y.clamp(0, image.height as isize - 1) as usize * image.width
                    + x.clamp(0, right_limit) as usize] as f64
            },
        );
    } else {
        sample_grid(
            basis,
            [center_x, center_y],
            resolution,
            extent,
            output,
            |x, y| image.values[y as usize * image.width + x as usize] as f64,
        );
    }
    canonical_sigma
}

/// Traverse the canonical grid identically for interior and padded pixel access.
fn sample_grid(
    basis: [f64; 4],
    [center_x, center_y]: [f64; 2],
    resolution: usize,
    extent: f64,
    output: &mut Plane,
    at: impl Fn(isize, isize) -> f64,
) {
    let side = 2 * resolution + 1;
    output.width = side;
    output.height = side;
    output.values.clear();
    output.values.reserve(side * side);
    let mut canonical_y = -extent;
    let sample_step = extent / resolution as f64;
    for _ in 0..side {
        let mut canonical_x = -extent;
        let row_x = basis[2].mul_add(canonical_y, center_x);
        let row_y = basis[3].mul_add(canonical_y, center_y);
        for _ in 0..side {
            let x = basis[0].mul_add(canonical_x, row_x);
            let y = basis[1].mul_add(canonical_x, row_y);
            let xi = x.floor() as isize;
            let yi = y.floor() as isize;
            let wx = x - xi as f64;
            let wy = y - yi as f64;
            output.values.push(super::math::bilinear_start_fma(
                [
                    at(xi, yi),
                    at(xi + 1, yi),
                    at(xi, yi + 1),
                    at(xi + 1, yi + 1),
                ],
                wx,
                wy,
            ) as f32);
            canonical_x += sample_step;
        }
        canonical_y += sample_step;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compare explicit half-pixel coordinates at interior, edges and outside the image.
    #[test]
    fn sampling_preserves_bilinear_values_and_padded_right_limit() {
        let image = Plane::from_values(
            64,
            64,
            (0..64 * 64)
                .map(|i| ((i * 37 + i / 64 * 11) % 257) as f32 / 256.)
                .collect(),
        );
        let pyramid = Pyramid::build(&image);
        let source = &pyramid.octaves[0][0];
        let mut output = Plane::default();
        // Padding on either axis triggers the reference's special right limit.
        for (center_x, center_y, right_limit) in [
            (32., 32., 63),
            (0., 32., 2),
            (63., 32., 62),
            (32., 0., 34),
            (32., 63., 34),
            (63., 63., 62),
            (70., 32., 63),
            (-5., 32., 0),
        ] {
            let frame = Frame {
                x: center_x,
                y: center_y,
                a11: 1.,
                a21: 0.,
                a12: 0.,
                a22: 1.,
            };
            let sigma = sample(&pyramid, frame, 3, 1.5, &mut output);
            assert_eq!(
                sigma.to_bits(),
                pyramid::sigma(0, pyramid::FIRST_SUBDIVISION).to_bits()
            );
            assert_eq!(
                (output.width, output.height, output.values.len()),
                (7, 7, 49)
            );
            for row in 0..7 {
                for column in 0..7 {
                    let x = center_x as f64 + column as f64 * 0.5 - 1.5;
                    let y = center_y as f64 + row as f64 * 0.5 - 1.5;
                    let left = x.floor() as isize;
                    let top = y.floor() as isize;
                    let at = |x: isize, y: isize| {
                        source.values
                            [y.clamp(0, 63) as usize * 64 + x.clamp(0, right_limit) as usize]
                            as f64
                    };
                    let expected = super::super::math::bilinear_start_fma(
                        [
                            at(left, top),
                            at(left + 1, top),
                            at(left, top + 1),
                            at(left + 1, top + 1),
                        ],
                        x - left as f64,
                        y - top as f64,
                    ) as f32;
                    assert_eq!(
                        output.values[row * 7 + column].to_bits(),
                        expected.to_bits(),
                        "center ({center_x}, {center_y}), pixel ({column}, {row})"
                    );
                }
            }
        }
    }
}
