"""Generate cap/tie feature IDs with Darwin qsort and OpenSfM's feature layout.

This uses only Python's standard library. Run on Darwin; regenerating on another
libc would record that platform's different equal-score ordering.
"""

import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import platform
import struct


class Frame(ct.Structure):
    _fields_ = [(name, ct.c_float) for name in ("x", "y", "a11", "a12", "a21", "a22")]


class Feature(ct.Structure):
    # opensfm/src/third_party/vlfeat/vl/covdet.h, VlCovDetFeature.
    _fields_ = [("frame", Frame), ("o", ct.c_int), ("s", ct.c_int)] + [
        (name, ct.c_float)
        for name in ("peakScore", "edgeScore", "orientationScore", "laplacianScaleScore")
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if platform.system() != "Darwin":
        parser.error("reference ordering requires Darwin's qsort")
    if ct.sizeof(Feature) != 48 or Feature.peakScore.offset != 32:
        raise RuntimeError("unexpected VlCovDetFeature ABI")
    comparator = ct.CFUNCTYPE(ct.c_int, ct.POINTER(Feature), ct.POINTER(Feature))

    @comparator
    def compare(a, b):
        fa, fb = a.contents.peakScore, b.contents.peakScore
        return int(fb > fa) - int(fb < fa)  # hahog.cc vlfeat_compare_scores.

    qsort = ct.CDLL("/usr/lib/libSystem.B.dylib").qsort
    qsort.restype = None
    qsort.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_size_t, comparator]
    manifest = {
        "reference": "Darwin qsort; OpenSfM hahog.cc select_best_features/run_features_selection",
        "upstream_revision": "c4985ced723888d2529f9b7a1902d4aa682de1d0",
        "platform": f"{platform.system()} {platform.machine()}",
        "macos": platform.mac_ver()[0],
        "feature_size": ct.sizeof(Feature),
        "peak_formula": "((input_id * 37 + 11) % 17) - 8 + 0.25 (float32)",
        "caps": [6000, 4000],
        "encoding": "ordered input IDs as little-endian u32, without header",
        "cases": [],
    }
    for count in (4001, 6000, 6001):
        features = (Feature * count)()
        for index, feature in enumerate(features):
            feature.frame.x = index
            feature.frame.a11 = feature.frame.a22 = 1.
            feature.peakScore = ((index * 37 + 11) % 17) - 8 + 0.25
        kept = count
        for cap in manifest["caps"]:
            if kept > cap:
                qsort(features, kept, ct.sizeof(Feature), compare)
                kept = cap
        ids = [int(features[index].frame.x) for index in range(kept)]
        data = struct.pack(f"<{kept}I", *ids)
        name = f"selection-ties-{count}.bin"
        (args.output / name).write_bytes(data)
        manifest["cases"].append({
            "file": name, "inputs": count, "features": kept,
            "sha256": hashlib.sha256(data).hexdigest(),
        })
        print(f"{name}: {count} inputs, {kept} retained IDs")
    (args.output / "selection-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
