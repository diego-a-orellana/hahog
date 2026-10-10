//! Preserve Darwin/BSD peak-score tie order; this affects subsequent feature IDs.
//! Adapted from Apple Libc's Bentley–McIlroy sort; see `licenses/BSD-Qsort.txt`.
//! At the recursion limit, stable sorting bounds runtime but may differ in tie order.
use super::detector::Detection;
use std::cmp::Ordering;
/// Order finite signed Hessian scores from largest to smallest.
fn compare_peaks(a: &Detection, b: &Detection) -> Ordering {
    b.peak.partial_cmp(&a.peak).expect("finite detector peaks")
}

/// Select the median of three indices with the reference tie comparisons.
fn median_index(a: &[Detection], x: usize, y: usize, z: usize) -> usize {
    if compare_peaks(&a[x], &a[y]).is_lt() {
        if compare_peaks(&a[y], &a[z]).is_lt() {
            y
        } else if compare_peaks(&a[x], &a[z]).is_lt() {
            z
        } else {
            x
        }
    } else if compare_peaks(&a[y], &a[z]).is_gt() {
        y
    } else if compare_peaks(&a[x], &a[z]).is_lt() {
        x
    } else {
        z
    }
}

/// Sort by peak; return false after exceeding a nonzero swap limit.
fn insertion_sort(a: &mut [Detection], limit: usize) -> bool {
    let mut swaps = 0;
    for i in 1..a.len() {
        let mut j = i;
        while j > 0 && compare_peaks(&a[j - 1], &a[j]).is_gt() {
            a.swap(j, j - 1);
            j -= 1;
            swaps += 1;
            if limit != 0 && swaps > limit {
                return false;
            }
        }
    }
    true
}

/// Apply BSD pivoting and equal-key swaps, with bounded recursion depth.
fn sort_by_peak(mut a: &mut [Detection], mut depth: usize) {
    loop {
        let n = a.len();
        if n < 2 {
            return;
        }
        if depth == 0 {
            a.sort_by(compare_peaks);
            return;
        }
        depth -= 1;
        if n <= 7 {
            insertion_sort(a, 0);
            return;
        }
        let mut left = 0;
        let mut mid = n / 2;
        let mut right = n - 1;
        if n > 40 {
            let d = n / 8;
            left = median_index(a, left, left + d, left + 2 * d);
            mid = median_index(a, mid - d, mid, mid + d);
            right = median_index(a, right - 2 * d, right - d, right);
        }
        mid = median_index(a, left, mid, right);
        a.swap(0, mid);
        let (mut pa, mut pb, mut pc, mut pd) = (1usize, 1usize, n as isize - 1, n as isize - 1);
        let mut swapped = false;
        loop {
            while pb as isize <= pc && !compare_peaks(&a[pb], &a[0]).is_gt() {
                if compare_peaks(&a[pb], &a[0]).is_eq() {
                    a.swap(pa, pb);
                    pa += 1;
                    swapped = true;
                }
                pb += 1;
            }
            while pb as isize <= pc && !compare_peaks(&a[pc as usize], &a[0]).is_lt() {
                if compare_peaks(&a[pc as usize], &a[0]).is_eq() {
                    a.swap(pc as usize, pd as usize);
                    pd -= 1;
                    swapped = true;
                }
                pc -= 1;
            }
            if pb as isize > pc {
                break;
            }
            a.swap(pb, pc as usize);
            swapped = true;
            pb += 1;
            pc -= 1;
        }
        let r = pa.min(pb - pa);
        for i in 0..r {
            a.swap(i, pb - r + i);
        }
        let r = ((pd - pc) as usize).min(n - pd as usize - 1);
        for i in 0..r {
            a.swap(pb + i, n - r + i);
        }
        if !swapped && insertion_sort(a, 1 + n / 4) {
            return;
        }
        let left = pb - pa;
        let right = (pd - pc) as usize;
        if left <= right {
            sort_by_peak(&mut a[..left], depth);
            a = &mut a[n - right..];
        } else {
            sort_by_peak(&mut a[n - right..], depth);
            a = &mut a[..left];
        }
    }
}

/// Apply the reference 6000-then-4000 caps before orientation expansion.
pub(super) fn select_strongest(a: &mut Vec<Detection>) {
    for limit in [6000, 4000] {
        if a.len() > limit {
            let depth = 2 * (usize::BITS - 1 - a.len().leading_zeros()) as usize;
            sort_by_peak(a, depth);
            a.truncate(limit);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Preserve uncapped scan order and select the largest signed scores above the cap.
    #[test]
    fn feature_budget_preserves_small_inputs_and_caps_larger_inputs() {
        for count in [0, 17, 4000, 6000, 7001] {
            let mut detections: Vec<_> = (0..count)
                .map(|index| Detection {
                    frame: super::super::Frame {
                        x: index as f32,
                        y: 0.,
                        a11: 1.,
                        a12: 0.,
                        a21: 0.,
                        a22: 1.,
                    },
                    octave: 0,
                    peak: (index as i32 - 4500) as f32 + 0.25,
                })
                .collect();
            select_strongest(&mut detections);
            assert_eq!(detections.len(), count.min(4000));
            for (index, detection) in detections.iter().enumerate() {
                let expected_id = if count <= 4000 {
                    index
                } else {
                    count - 1 - index
                };
                assert_eq!(detection.frame.x, expected_id as f32);
                assert_eq!(detection.peak, (expected_id as i32 - 4500) as f32 + 0.25);
            }
        }
    }

    /// Keep equal-score feature IDs in the native BSD sorting order.
    #[test]
    fn peak_ties_preserve_reference_feature_id_permutation() {
        let mut rows: Vec<_> = [2., 1., 2., 3., 2., 1., 3., 2., 1., 2., 3.]
            .into_iter()
            .enumerate()
            .map(|(id, peak)| Detection {
                frame: super::super::Frame {
                    x: id as f32,
                    y: 0.,
                    a11: 1.,
                    a12: 0.,
                    a21: 0.,
                    a22: 1.,
                },
                octave: 0,
                peak,
            })
            .collect();
        sort_by_peak(&mut rows, 6);
        let actual: Vec<_> = rows.iter().map(|row| row.frame.x as usize).collect();
        assert_eq!(actual, [3, 10, 6, 0, 2, 4, 7, 9, 8, 5, 1]);
    }
}
