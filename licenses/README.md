# Attribution and license scope

New contributions are licensed under [BSD-3-Clause](../LICENSE), recorded in
`Cargo.toml`. Adapted code retains the upstream copyright notices and license
terms below. Include these notices with source distributions and reproduce them
in the documentation or other materials accompanying binary distributions.

| Source | Adapted portions | Retained license |
| --- | --- | --- |
| [OpenSfM](https://github.com/mapillary/OpenSfM), revision `c4985ced723888d2529f9b7a1902d4aa682de1d0` | HAHOG stage sequencing, feature selection and descriptor conversion from `hahog.cc` and `features.py` | [BSD-2-Clause](OpenSfM.txt) |
| VLFeat bundled in that OpenSfM revision | `algorithm/`: Hessian detector, Gaussian pyramid, canonical patches, orientations, SIFT histograms and numerical approximations | [BSD-2-Clause](VLFeat.txt) |
| [Apple Libc / FreeBSD qsort](https://github.com/apple-oss-distributions/Libc/blob/main/stdlib/FreeBSD/qsort.c) | Pivoting, partitioning and equal-score ordering in `algorithm/selection.rs` | [BSD-3-Clause](BSD-Qsort.txt) |

The bundled VLFeat files also carry these individual notices, in addition to
the library-wide notice in `VLFeat.txt`:

```text
covdet.c / covdet.h:
Copyright (C) 2013-14 Andrea Vedaldi.
Copyright (C) 2012 Karel Lenc, Andrea Vedaldi and Michal Perdoch.
All rights reserved.

scalespace.c / scalespace.h, imopv.c / imopv.h, sift.c / sift.h, mathop.c / mathop.h:
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

mathop.c additionally:
Copyright (C) 2014 Andrea Vedaldi.
```

These portions remain subject to the BSD terms in `VLFeat.txt`. The Rust port
does not link to or compile VLFeat. Cargo dependencies retain their own licenses.
