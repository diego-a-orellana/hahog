//! Scale-space Hessian extrema with quadratic localization and an edge rejection gate.
use super::{
    math,
    pyramid::{self, Pyramid},
    Frame, Plane,
};

/// Minimum absolute interpolated Hessian score, in the reference's single precision.
const PEAK_THRESHOLD: f32 = 1e-5;
/// Maximum principal-curvature ratio accepted at a localized extremum.
const EDGE_THRESHOLD: f32 = 10.;
/// Maximum spatial localization steps; the scale index remains fixed.
const LOCALIZATION_STEPS: usize = 5;

/// Localized isotropic frame and its scale-normalized Hessian score.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Detection {
    /// Frame in original-image pixels; isotropic until orientation assignment.
    pub frame: Frame,
    /// Zero-based pyramid octave.
    pub octave: i32,
    /// Signed, interpolated Hessian determinant.
    pub peak: f32,
}

/// Compute scale-normalized Hessian determinants, replicating interior border values.
fn hessian_response(image: &Plane, octave: usize, subdivision: i32, output: &mut Plane) {
    let w = image.width;
    let h = image.height;
    debug_assert!(w >= 3 && h >= 3);
    debug_assert_eq!(image.values.len(), w * h);
    output.reset(w, h);
    let values = &mut output.values;
    let factor = (pyramid::sigma(octave, subdivision) / 2f64.powi(octave as i32)).powf(4.) as f32;
    for y in 1..h - 1 {
        let above = &image.values[(y - 1) * w..y * w];
        let middle = &image.values[y * w..(y + 1) * w];
        let below = &image.values[(y + 1) * w..(y + 2) * w];
        let row = &mut values[y * w..(y + 1) * w];
        for x in 1..w - 1 {
            let center = 2. * middle[x];
            let xx = center - middle[x - 1] - middle[x + 1];
            let yy = center - above[x] - below[x];
            let xy = (above[x - 1] - below[x - 1] - above[x + 1] + below[x + 1]) / 4.;
            row[x] = math::determinant2([xx, xy, xy, yy]) * factor;
        }
        row[0] = row[1];
        row[w - 1] = row[w - 2];
    }
    values.copy_within(w..2 * w, 0);
    values.copy_within((h - 2) * w..(h - 1) * w, (h - 1) * w);
}

/// Localize an extremum in at most five steps; reject weak, singular or edge-like peaks.
fn refine_extremum(
    levels: &[Plane; 3],
    octave: usize,
    mut x: usize,
    mut y: usize,
    level_index: usize,
) -> Option<Detection> {
    let w = levels[0].width;
    let h = levels[0].height;
    let mut delta = [0.; 3];
    let mut grad = [0.; 3];
    let mut a = [[0.; 3]; 3];
    let mut value = 0.;
    for iteration in 0..LOCALIZATION_STEPS {
        let at = |dx: isize, dy: isize, dz: isize| {
            levels[(1 + dz) as usize].values
                [(y as isize + dy) as usize * w + (x as isize + dx) as usize]
        };
        value = at(0, 0, 0);
        grad = [
            0.5 * (at(1, 0, 0) - at(-1, 0, 0)) as f64,
            0.5 * (at(0, 1, 0) - at(0, -1, 0)) as f64,
            0.5 * (at(0, 0, 1) - at(0, 0, -1)) as f64,
        ];
        let xx = (at(1, 0, 0) + at(-1, 0, 0)) as f64 - 2. * value as f64;
        let yy = (at(0, 1, 0) + at(0, -1, 0)) as f64 - 2. * value as f64;
        let zz = (at(0, 0, 1) + at(0, 0, -1)) as f64 - 2. * value as f64;
        let xy = 0.25 * (at(1, 1, 0) + at(-1, -1, 0) - at(-1, 1, 0) - at(1, -1, 0)) as f64;
        let xz = 0.25 * (at(1, 0, 1) + at(-1, 0, -1) - at(-1, 0, 1) - at(1, 0, -1)) as f64;
        let yz = 0.25 * (at(0, 1, 1) + at(0, -1, -1) - at(0, -1, 1) - at(0, 1, -1)) as f64;
        a = [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]];
        delta = math::solve3(a, grad.map(|v| -v))?;
        let shift = |v: f64, p: usize, max: usize| {
            if v > 0.6 && p < max - 2 {
                1
            } else if v < -0.6 && p > 1 {
                -1
            } else {
                0
            }
        };
        let dx = shift(delta[0], x, w);
        let dy = shift(delta[1], y, h);
        if dx == 0 && dy == 0 {
            break;
        }
        // The fifth iteration's derivatives refer to its current location.
        if iteration + 1 != LOCALIZATION_STEPS {
            x = (x as isize + dx) as usize;
            y = (y as isize + dy) as usize;
        }
    }
    let peak = (0.5f64.mul_add(
        grad[2].mul_add(delta[2], grad[0].mul_add(delta[0], grad[1] * delta[1])),
        value as f64,
    )) as f32;
    let alpha =
        (a[0][0] + a[1][1]).powi(2) / math::determinant2([a[0][0], a[0][1], a[0][1], a[1][1]]);
    let edge = if alpha < 0. {
        f32::INFINITY
    } else {
        ((0.5 * alpha - 1.) + ((0.25 * alpha - 1.).max(0.) * alpha).sqrt()) as f32
    };
    if delta.iter().any(|v| v.abs() >= 1.5)
        || peak.abs() <= PEAK_THRESHOLD
        || edge >= EDGE_THRESHOLD
    {
        return None;
    }
    let xx = (x as f64 + delta[0]) as f32;
    let yy = (y as f64 + delta[1]) as f32;
    let localized_level = (level_index as f64 + delta[2]) as f32;
    if !(0. ..=w as f32 - 1.).contains(&xx)
        || !(0. ..=h as f32 - 1.).contains(&yy)
        || !(0. ..=(pyramid::LEVELS_PER_OCTAVE - 1) as f32).contains(&localized_level)
    {
        return None;
    }
    let scale = (pyramid::base_sigma()
        * 2f64.powf(
            (octave as f32
                + (localized_level + pyramid::FIRST_SUBDIVISION as f32)
                    / pyramid::SUBDIVISIONS_PER_OCTAVE as f32) as f64,
        )) as f32;
    let step = 2f64.powi(octave as i32);
    Some(Detection {
        frame: Frame {
            x: (xx as f64 * step) as f32,
            y: (yy as f64 * step) as f32,
            a11: scale,
            a12: 0.,
            a21: 0.,
            a22: scale,
        },
        octave: octave as i32,
        peak,
    })
}

/// Detect refined extrema from coarse to fine octaves, preserving scan order.
pub(crate) fn detect(pyramid: &Pyramid) -> Vec<Detection> {
    let mut detections = Vec::new();
    let mut responses = std::array::from_fn(|_| Plane::default());
    for (octave, planes) in pyramid.octaves.iter().enumerate().rev() {
        for (index, (image, response)) in planes.iter().zip(&mut responses).enumerate() {
            hessian_response(
                image,
                octave,
                index as i32 + pyramid::FIRST_SUBDIVISION,
                response,
            );
        }
        for level_index in 1..pyramid::LEVELS_PER_OCTAVE - 1 {
            if level_index > 1 {
                // Only x/y move during localization, so three adjacent scales suffice.
                responses.rotate_left(1);
                let next_index = level_index + 1;
                hessian_response(
                    &planes[next_index],
                    octave,
                    next_index as i32 + pyramid::FIRST_SUBDIVISION,
                    &mut responses[2],
                );
            }
            append_extrema(&responses, octave, level_index, &mut detections);
        }
    }
    detections
}

/// Append strict extrema from the middle response plane, in row-major scan order.
/// `level_index` is its absolute octave index, independent of the rolling window.
fn append_extrema(
    levels: &[Plane; 3],
    octave: usize,
    level_index: usize,
    out: &mut Vec<Detection>,
) {
    let w = levels[0].width;
    let h = levels[0].height;
    debug_assert!(levels
        .iter()
        .all(|plane| (plane.width, plane.height) == (w, h)));
    for y in 1..h - 1 {
        let rows: [[&[f32]; 3]; 3] = std::array::from_fn(|scale| {
            std::array::from_fn(|row| &levels[scale].values[(y + row - 1) * w..(y + row) * w])
        });
        'candidate: for x in 1..w - 1 {
            let v = rows[1][1][x];
            if (v as f64).abs() < 0.8 * (PEAK_THRESHOLD as f64) {
                continue;
            }
            for (scale, plane_rows) in rows.iter().enumerate() {
                for (row_index, row) in plane_rows.iter().enumerate() {
                    for (column, &neighbor) in row[x - 1..x + 2].iter().enumerate() {
                        if scale == 1 && row_index == 1 && column == 1 {
                            continue;
                        }
                        if (v >= 0. && v <= neighbor) || (v < 0. && v >= neighbor) {
                            continue 'candidate;
                        }
                    }
                }
            }
            if let Some(d) = refine_extremum(levels, octave, x, y, level_index) {
                out.push(d);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An analytic extremum keeps its absolute scale index as the response window moves.
    #[test]
    fn response_windows_preserve_absolute_scale_and_reject_equal_neighbors() {
        for sign in [1., -1.] {
            let mut levels = std::array::from_fn(|scale| {
                Plane::from_values(
                    9,
                    9,
                    (0..81)
                        .map(|index| {
                            let x = (index % 9) as f32 - 4.;
                            let y = (index / 9) as f32 - 4.;
                            let z = scale as f32 - 1.;
                            sign * (10. - x * x - y * y - z * z)
                        })
                        .collect(),
                )
            });
            let mut points = Vec::new();
            for level in 1..pyramid::LEVELS_PER_OCTAVE - 1 {
                append_extrema(&levels, 1, level, &mut points);
            }
            assert_eq!(points.len(), 3);
            for point in &points {
                assert_eq!((point.frame.x, point.frame.y, point.octave), (8., 8., 1));
                assert_eq!(point.peak, sign * 10.);
            }
            assert!(points
                .windows(2)
                .all(|pair| pair[0].frame.a11 < pair[1].frame.a11));

            // A neighbouring scale with an equal value disqualifies a strict extremum.
            levels[0].values[4 * 9 + 4] = sign * 10.;
            points.clear();
            append_extrema(&levels, 1, 1, &mut points);
            assert!(points.is_empty());
        }
    }
}
