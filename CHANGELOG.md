# Changelog

## Unreleased

## 1.1.0

- Speed up extraction with batched extrema filtering and gradients, blocked
  Gaussian convolution, and less repeated descriptor work, preserving keypoint
  order and descriptor bytes.

## 1.0.0

- Initial release: HAHOG extraction with an OpenCV-compatible API, masks,
  supplied keypoints, and 128-byte RootSIFT-style descriptors.
