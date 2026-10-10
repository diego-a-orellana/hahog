"""Refresh synthetic fixtures from a built native OpenSfM pyfeatures extension."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform

import numpy as np


def texture(width, height):
    """Produce the same row-major, 32-bit LCG pixels as the Rust integration test."""
    state = 1729
    pixels = []
    for _ in range(width * height):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        pixels.append(state >> 24)
    return np.array(pixels, dtype=np.uint8).reshape(height, width)


def border_blobs():
    """Draw integer-radius disks at four image corners and the center."""
    y, x = np.indices((97, 129))
    pixels = np.zeros((97, 129), dtype=np.uint8)
    for cx, cy, radius in [(0, 0, 8), (128, 0, 13), (0, 96, 11), (128, 96, 18), (64, 48, 15)]:
        pixels[(x - cx) ** 2 + (y - cy) ** 2 <= radius**2] = 255
    return pixels


def main():
    """Load the reference extension and write ordered frames and descriptor bytes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("extension", type=Path, help="path to the native pyfeatures extension")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("pyfeatures", args.extension.resolve())
    if spec is None or spec.loader is None:
        parser.error("cannot load the supplied pyfeatures extension")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    directory = Path(__file__).resolve().parent
    manifest = {
        "reference": "OpenSfM pyfeatures.hahog",
        "extension_sha256": hashlib.sha256(args.extension.read_bytes()).hexdigest(),
        "platform": f"{platform.system()} {platform.machine()}",
        "numpy": np.__version__,
        "parameters": {"peak_threshold": 1e-5, "edge_threshold": 10, "target_num_features": 4000},
        "cases": [],
    }
    for name, pixels in [
        ("texture-32x32", texture(32, 32)),
        ("texture-65x63", texture(65, 63)),
        ("border-blobs", border_blobs()),
    ]:
        frames, histograms = reference.hahog(
            pixels.astype(np.float32) / np.float32(255), 1e-5, 10.0, 4000
        )
        descriptors = np.rint(np.clip(np.float32(362) * np.sqrt(histograms), 0, 255)).astype(np.uint8)
        assert len(frames) > 0 and frames.shape == (len(frames), 4)
        assert descriptors.shape == (len(frames), 128)
        csv = "x,y,scale,angle_degrees\n" + "".join(
            ",".join(format(float(value), ".9g") for value in row) + "\n" for row in frames
        )
        (directory / f"{name}.csv").write_text(csv)
        (directory / f"{name}.bin").write_bytes(descriptors.tobytes())
        manifest["cases"].append({
            "name": name,
            "width": pixels.shape[1],
            "height": pixels.shape[0],
            "features": len(frames),
            "input_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
            "frames_sha256": hashlib.sha256(csv.encode()).hexdigest(),
            "descriptors_sha256": hashlib.sha256(descriptors.tobytes()).hexdigest(),
        })
        print(f"{name}: {len(frames)} features")
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
