"""Generate supplied-frame goldens using the original OpenSfM VLFeat exports.

Run with the Python environment used to build the extension. This does not invoke
the Rust implementation or rewrite the three detection fixtures.
"""

import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform

import numpy as np

from generate import texture


class Frame(ct.Structure):
    _fields_ = [(name, ct.c_float) for name in ("x", "y", "a11", "a12", "a21", "a22")]


class Orientation(ct.Structure):
    _fields_ = [("angle", ct.c_double), ("score", ct.c_double)]


def bind(library, name, result, *arguments):
    function = getattr(library, name)
    function.restype = result
    function.argtypes = list(arguments)
    return function


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("extension", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    extension = args.extension.resolve()
    # Import first so Python and the extension's dependent libraries are initialized.
    spec = importlib.util.spec_from_file_location("pyfeatures", extension)
    if spec is None or spec.loader is None:
        parser.error("cannot load the supplied pyfeatures extension")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    native = ct.CDLL(str(extension))
    pointer, size, double = ct.c_void_p, ct.c_size_t, ct.c_double
    floats = ct.POINTER(ct.c_float)
    new = bind(native, "vl_covdet_new", pointer, ct.c_int)
    delete = bind(native, "vl_covdet_delete", None, pointer)
    first_octave = bind(native, "vl_covdet_set_first_octave", None, pointer, ct.c_ssize_t)
    put = bind(native, "vl_covdet_put_image", ct.c_int, pointer, floats, size, size)
    orient = bind(native, "vl_covdet_extract_orientations_for_frame", ct.POINTER(Orientation),
                  pointer, ct.POINTER(size), Frame)
    patch = bind(native, "vl_covdet_extract_patch_for_frame", ct.c_int,
                 pointer, floats, size, double, double, Frame)
    gradient = bind(native, "vl_imgradient_polar_f", None,
                    floats, floats, size, size, floats, size, size, size)
    sift_new = bind(native, "vl_sift_new", pointer, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int)
    sift_delete = bind(native, "vl_sift_delete", None, pointer)
    describe = bind(native, "vl_sift_calc_raw_descriptor", None,
                    pointer, floats, floats, ct.c_int, ct.c_int, double, double, double, double)

    pixels = texture(65, 63)
    image = pixels.astype(np.float32) / np.float32(255)
    # All inputs are exact float32 values; scale is half the OpenCV diameter.
    inputs = [
        (32, 30, 4, 0),
        (18.25, 22.75, 2.25, 123),
        (0, 0, 2, 45),
        (64, 62, 5, 359.5),
        (31.5, 29.25, 12, 270),
        (32, 30, 4, -1),
        (16, 45, 1.5, -1),
    ]
    covdet = new(2)  # VL_COVDET_METHOD_HESSIAN, from covdet.h.
    sift = sift_new(16, 16, 1, 3, 0)
    if not covdet or not sift:
        raise MemoryError("VLFeat detector/SIFT allocation failed")
    rows, descriptors = [], []
    try:
        first_octave(covdet, 0)
        if put(covdet, image.ctypes.data_as(floats), 65, 63) != 0:
            raise RuntimeError("vl_covdet_put_image failed")
        for index, (x, y, scale, degrees) in enumerate(inputs):
            upright = Frame(x, y, scale, 0, 0, scale)
            if degrees == -1:
                count = size()
                orientations = orient(covdet, ct.byref(count), upright)
                angles = [orientations[i].angle for i in range(count.value)]
            else:
                angles = [math.radians(degrees)]
            for angle in angles:
                # Same circular-frame rotation as hahog.cc; native patch/SIFT code
                # performs all image sampling, smoothing and histogram operations.
                c, s = math.cos(angle), math.sin(angle)
                frame = Frame(x, y, scale * c, -scale * s, scale * s, scale * c)
                samples = np.zeros((31, 31), dtype=np.float32)
                polar = np.zeros((31, 31, 2), dtype=np.float32)
                histogram = np.zeros(128, dtype=np.float32)
                if patch(covdet, samples.ctypes.data_as(floats), 15, 7.5, 1., frame) != 0:
                    raise RuntimeError(f"vl_covdet_extract_patch_for_frame failed at input {index}")
                gradient(polar.ctypes.data_as(floats),
                         ct.cast(polar.ctypes.data + 4, floats), 2, 62,
                         samples.ctypes.data_as(floats), 31, 31, 31)
                # vl_sift_new initializes magnif=3.0 (sift.c), as in hahog.cc.
                describe(sift, polar.ctypes.data_as(floats), histogram.ctypes.data_as(floats),
                         31, 31, 15., 15., 2., math.pi / 2)
                descriptors.append(np.rint(np.clip(np.float32(362) * np.sqrt(histogram), 0, 255))
                                   .astype(np.uint8).tobytes())
                output_degrees = float(np.float32(math.degrees(angle) % 360)) if degrees == -1 else degrees
                if output_degrees >= 360:
                    output_degrees = 0.
                rows.append((index, x, y, scale, output_degrees))
    finally:
        sift_delete(sift)
        delete(covdet)

    def csv(header, values):
        return header + "\n" + "".join(
            ",".join(format(float(value), ".9g") for value in row) + "\n" for row in values
        )

    files = {
        "supplied-texture-input.csv": csv("x,y,scale,angle_degrees", inputs).encode(),
        "supplied-texture.csv": csv("input_index,x,y,scale,angle_degrees", rows).encode(),
        "supplied-texture.bin": b"".join(descriptors),
    }
    manifest = {
        "reference": "OpenSfM native VLFeat exports; patch/gradient/SIFT sequence from hahog.cc",
        "upstream_revision": "c4985ced723888d2529f9b7a1902d4aa682de1d0",
        "extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
        "platform": f"{platform.system()} {platform.machine()}",
        "numpy": np.__version__,
        "width": 65, "height": 63, "inputs": len(inputs), "features": len(rows),
        "input_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
        "sha256": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
    }
    for name, data in files.items():
        (args.output / name).write_bytes(data)
    (args.output / "supplied-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"supplied-texture: {len(inputs)} inputs, {len(rows)} output frames")


if __name__ == "__main__":
    main()
