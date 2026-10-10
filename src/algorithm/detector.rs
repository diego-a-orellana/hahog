//! Scale-space Hessian extrema with quadratic localization and an edge rejection gate.
use super::{
    math,
    pyramid::{self, Pyramid},
    Frame, Plane,
};

/// Localized isotropic frame and its scale-normalized Hessian score.
#[derive(Clone, Debug)]
pub(crate) struct Detection {
    /// Frame in original-image pixels; isotropic until orientation assignment.
    pub frame: Frame,
    /// Zero-based pyramid octave.
    pub octave: i32,
    /// Signed, interpolated Hessian determinant.
    pub peak: f32,
}

/// Compute scale-normalized Hessian determinants, replicating interior border values.
pub(crate) fn hessian_response(image: &Plane, o: usize, s: i32) -> Plane {
    let w = image.width;
    let h = image.height;
    let mut values = vec![0.; w * h];
    let factor = (pyramid::sigma(o, s) / 2f64.powi(o as i32)).powf(4.) as f32;
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let at = |dx: isize, dy: isize| {
                image.values[(y as isize + dy) as usize * w + (x as isize + dx) as usize]
            };
            let center = 2. * at(0, 0);
            let xx = center - at(-1, 0) - at(1, 0);
            let yy = center - at(0, -1) - at(0, 1);
            let xy = (at(-1, -1) - at(-1, 1) - at(1, -1) + at(1, 1)) / 4.;
            values[y * w + x] = math::determinant2([xx, xy, xy, yy]) * factor;
        }
    }
    for y in 1..h - 1 {
        values[y * w] = values[y * w + 1];
        values[(y + 1) * w - 1] = values[(y + 1) * w - 2];
    }
    values.copy_within(w..2 * w, 0);
    values.copy_within((h - 2) * w..(h - 1) * w, (h - 1) * w);
    Plane {
        width: w,
        height: h,
        values,
    }
}

/// Localize an extremum in at most five steps; reject weak, singular or edge-like peaks.
fn refine_extremum(
    levels: &[Plane],
    o: usize,
    mut x: usize,
    mut y: usize,
    z: usize,
) -> Option<Detection> {
    let w = levels[0].width;
    let h = levels[0].height;
    let mut delta = [0.; 3];
    let mut grad = [0.; 3];
    let mut a = [[0.; 3]; 3];
    let mut value = 0.;
    for iteration in 0..5 {
        let at = |dx: isize, dy: isize, dz: isize| {
            levels[(z as isize + dz) as usize].values
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
        if iteration != 4 {
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
    if delta.iter().any(|v| v.abs() >= 1.5) || peak.abs() <= 1e-5f32 || edge >= 10. {
        return None;
    }
    let xx = (x as f64 + delta[0]) as f32;
    let yy = (y as f64 + delta[1]) as f32;
    let zz = (z as f64 + delta[2]) as f32;
    if !(0. ..=w as f32 - 1.).contains(&xx)
        || !(0. ..=h as f32 - 1.).contains(&yy)
        || !(0. ..=4.).contains(&zz)
    {
        return None;
    }
    let scale = (pyramid::base_sigma() * 2f64.powf((o as f32 + (zz - 1.) / 3.) as f64)) as f32;
    let step = 2f64.powi(o as i32);
    Some(Detection {
        frame: Frame {
            x: (xx as f64 * step) as f32,
            y: (yy as f64 * step) as f32,
            a11: scale,
            a12: 0.,
            a21: 0.,
            a22: scale,
        },
        octave: o as i32,
        peak,
    })
}

/// Detect refined extrema from coarse to fine octaves, preserving scan order.
pub(crate) fn detect(pyramid: &Pyramid) -> Vec<Detection> {
    let mut detections = Vec::new();
    for (o, planes) in pyramid.octaves.iter().enumerate().rev() {
        let responses: Vec<_> = planes
            .iter()
            .enumerate()
            .map(|(s, p)| hessian_response(p, o, s as i32 - 1))
            .collect();
        detections.extend(find_extrema(&responses, o));
    }
    detections
}

/// Refine strict 3×3×3 extrema in the three interior response levels of an octave.
pub(crate) fn find_extrema(levels: &[Plane], o: usize) -> Vec<Detection> {
    let w = levels[0].width;
    let h = levels[0].height;
    let mut out = Vec::new();
    for z in 1..4 {
        for y in 1..h - 1 {
            'candidate: for x in 1..w - 1 {
                let v = levels[z].values[y * w + x];
                if (v as f64).abs() < 0.8 * (1e-5f32 as f64) {
                    continue;
                }
                for dz in -1isize..=1 {
                    for dy in -1isize..=1 {
                        for dx in -1isize..=1 {
                            if dx == 0 && dy == 0 && dz == 0 {
                                continue;
                            }
                            let neighbor = levels[(z as isize + dz) as usize].values
                                [(y as isize + dy) as usize * w + (x as isize + dx) as usize];
                            if (v >= 0. && v <= neighbor) || (v < 0. && v >= neighbor) {
                                continue 'candidate;
                            }
                        }
                    }
                }
                if let Some(d) = refine_extremum(levels, o, x, y, z) {
                    out.push(d);
                }
            }
        }
    }
    out
}
