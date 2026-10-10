//! Canonical bilinear patches; boundary behavior deliberately preserves the reference.
use super::{
    pyramid::{self, Pyramid},
    Frame, Plane,
};

/// Sample a square of side `2 * resolution + 1` over canonical coordinates ±`extent`.
/// Return the patch and its Gaussian sigma in canonical frame units.
/// Frames remain isotropic because affine shape adaptation is disabled.
/// Padded patches repeat the reference's penultimate right pixel, preserving its boundary quirk.
pub(super) fn sample(pyramid: &Pyramid, f: Frame, resolution: usize, extent: f64) -> (Plane, f64) {
    let a = [f.a11 as f64, f.a21 as f64, f.a12 as f64, f.a22 as f64];
    let d = super::math::squared_norm2_fma(a[0], a[1]).sqrt();
    let factor = 1. / d;
    let sub = |o: usize| {
        (((1. / (factor * pyramid::base_sigma())).ln() * std::f64::consts::LOG2_E - o as f64)
            .floor() as i32)
            .clamp(-1, 3)
    };
    let last = pyramid.octaves.len() - 1;
    let mut o = last;
    for candidate in 1..=last {
        if factor * pyramid::sigma(candidate, sub(candidate)) > 1. {
            o = candidate - 1;
            break;
        }
    }
    let s = sub(o);
    let sigma = pyramid::sigma(o, s) / d;
    let image = &pyramid.octaves[o][(s + 1) as usize];
    let step = 2f64.powi(o as i32);
    let a = a.map(|v| v / step);
    let tx = f.x as f64 / step;
    let ty = f.y as f64 / step;
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
        let xx = a[0].mul_add(x, a[2] * y) + tx;
        let yy = a[1].mul_add(x, a[3] * y) + ty;
        xmin = xmin.min(xx);
        xmax = xmax.max(xx);
        ymin = ymin.min(yy);
        ymax = ymax.max(yy);
    }
    let x0 = xmin.floor() as isize - 1;
    let x1 = xmax.ceil() as isize + 1;
    let y0 = ymin.floor() as isize - 1;
    let y1 = ymax.ceil() as isize + 1;
    let pad = x0 < 0 || y0 < 0 || x1 >= image.width as isize || y1 >= image.height as isize;
    let xmax = if pad {
        (x1 - (x1 - image.width as isize + 1).max(0) - 1)
            .max(x0.max(0))
            .min(image.width as isize - 1)
    } else {
        image.width as isize - 1
    };
    let at = |x: isize, y: isize| -> f64 {
        image.values[y.clamp(0, image.height as isize - 1) as usize * image.width
            + x.clamp(0, xmax) as usize] as f64
    };
    let side = 2 * resolution + 1;
    let mut values = Vec::with_capacity(side * side);
    let mut yhat = -extent;
    let dh = extent / resolution as f64;
    for _ in 0..side {
        let mut xhat = -extent;
        let rx = a[2].mul_add(yhat, tx);
        let ry = a[3].mul_add(yhat, ty);
        for _ in 0..side {
            let x = a[0].mul_add(xhat, rx);
            let y = a[1].mul_add(xhat, ry);
            let xi = x.floor() as isize;
            let yi = y.floor() as isize;
            let wx = x - xi as f64;
            let wy = y - yi as f64;
            values.push(super::math::bilinear_start_fma(
                [
                    at(xi, yi),
                    at(xi + 1, yi),
                    at(xi, yi + 1),
                    at(xi + 1, yi + 1),
                ],
                wx,
                wy,
            ) as f32);
            xhat += dh;
        }
        yhat += dh;
    }
    (
        Plane {
            width: side,
            height: side,
            values,
        },
        sigma,
    )
}
