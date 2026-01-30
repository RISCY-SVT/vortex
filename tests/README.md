# Vortex Tests Guide

This document describes the test layout, build workflow, and how to run the OpenCL/video compute tests in the **compute-only** Vortex fork. All instructions assume the standard **build directory** workflow (out-of-tree build in `<BUILD_DIR>`).

## Directory overview

```
<REPO_ROOT>/tests
  opencl        OpenCL apps and kernels (compute-only tests)
  regression    Regression apps (simx/rtlsim)
  kernel        Kernel-level tests (if present)
  riscv         ISA/runtime tests (if present)
  unittest      Unit tests (if present)
```

### OpenCL video/scanout apps (new)

These new compute-only apps generate framebuffers, compare against CPU reference, and optionally dump PPM images:

- `test_pattern`   RGBA pattern generator (color bars / checker / gradient)
- `yuv2rgb`        YUV420 -> RGBA conversion
- `scaler`         Scaling (nearest + bilinear, same output size)
- `convolution`    Gaussian blur + Sobel edge detect
- `alpha_blend`    Alpha compositing (foreground over background)
- `scanout_sim`    Stride/pitch, RGB565 format, double-buffer, partial update

All apps run via `blackbox.sh` and return `PASSED!/FAILED` with nonzero exit on failure.

## Build workflow (required)

This repo uses an out-of-tree build directory. **Do not build from the source tree** and do not create symlink hacks (e.g., `config.mk` in the source root).

```
export VORTEX_ROOT=<REPO_ROOT>
export VORTEX_BUILD=<BUILD_DIR>
cd "$VORTEX_BUILD"
../configure
source ./ci/toolchain_env.sh
make -s
```

Example (path only):
```
export VORTEX_ROOT=/data/vortex
export VORTEX_BUILD=/data/vortex/build
```

If you add or rename test directories or Makefiles in the source tree, **re-run `../configure`** to sync the build tree.

## Toolchain setup

- Use the repo’s toolchain installation scripts when needed (see `ci/toolchain_install.sh` and `ci/toolchain_env.sh`).
- Always source the environment before running tests:

```
cd "$VORTEX_BUILD"
source ./ci/toolchain_env.sh
```

## Blackbox test driver

`blackbox.sh` is the standard test entry point. Always run it from the **build directory**:

```
cd "$VORTEX_BUILD"
./ci/blackbox.sh --driver=simx --app=vecadd
./ci/blackbox.sh --driver=simx --app=test_pattern --args="-w 320 -h 320"
```

Notes:
- `--driver` can be `simx` or `rtlsim` (rtlsim is much slower).
- `--app` is any subfolder under `tests/opencl` or `tests/regression`.
- `--args` are passed to the app via its Makefile `OPTS` variable.

## Video suite app arguments

All new video apps share common CLI options (via `video_utils`):

```
-w, --width <n>       Input width
-h, --height <n>      Input height
-stride <bytes>       Input stride in bytes (0 = tight / aligned)
-mode <n>             App-specific mode
-seed <n>             Pattern seed
-ow, --outw <n>        Output width (scaler)
-oh, --outh <n>        Output height (scaler)
-fmt <n>              Pixel format (scanout_sim)
-rect x,y,w,h          Rectangle (scanout_sim)
--dump                Write PPM outputs on PASS
--outdir <path>        Output directory for images (default: artifacts)
--prefix <string>      Filename prefix
```

### App-specific notes

- `test_pattern`
  - `-mode 0|1|2` selects bars/checker/gradient
- `yuv2rgb`
  - YUV420 requires even width and height
- `scaler`
  - Both nearest and bilinear outputs are the same size
  - Use `-ow/-oh` to select output size
  - The runner uses non-identity input/output sizes in full mode (e.g., 191x193 -> 320x320) and fails if nearest/bilinear outputs are identical
- `convolution`
  - Outputs blur and sobel images
- `alpha_blend`
  - Composites two layers with alpha
- `scanout_sim`
  - Validates stride/pitch and format conversion
  - For stride testing, ensure `stride_bytes > width * bytes_per_pixel`
  - RGB565 format conversion is validated against CPU reference
  - Double-buffer and partial update are verified in one run

### Output artifacts (PPM)

Images are **not written by default**. Use `--dump` to enable output on PASS. On FAIL, images are written automatically (if possible). Examples:

```
./ci/blackbox.sh --driver=simx --app=test_pattern \
  --args="-w 320 -h 320 --dump --outdir $VORTEX_BUILD/artifacts/video_suite --prefix pretty_test_pattern"
```

PPM files are in P6 format and can be viewed with standard tools (e.g., `display`, `ffplay`, or image viewers that support PPM).

## Video suite runner script

A full runner is provided to execute the suite, collect results, and create a JPEG collage:

```
python3 $VORTEX_ROOT/tests/run_video_suite.py \
  --build-dir $VORTEX_BUILD \
  --driver simx \
  --mode full
```

### Runner modes

- `--mode quick`:
  - Runs each app once with small default sizes
  - Generates PPM outputs and a collage
- `--mode full`:
  - Runs **pretty** (320x320), **tail** (non-multiple-of-16), and **padding** (scanout) variants
  - Generates PPM outputs and a collage

### Runner outputs

The runner creates an output directory under:

```
<BUILD_DIR>/artifacts/video_suite/<timestamp>/
  results.md
  results.csv
  collage.jpg
  logs/
  *.ppm
```

### Collage dependencies

The runner uses Python Pillow to produce `collage.jpg`.

```
python3 -m pip install pillow
```

If Pillow is missing, the runner prints an error and exits nonzero.

## Troubleshooting

**Blackbox cannot find app**
- Ensure the app exists under `tests/opencl/<app>`.
- Re-run `../configure` from `<BUILD_DIR>`.

**Changes not visible in build tree**
- Always re-run `../configure` after adding files or directories.
- Consider `make -C <BUILD_DIR>/tests/opencl/<app> clean` to force rebuild.

**SimX failures**
- Use `OPENCL_DIAG=1` to get enhanced diagnostics from the OpenCL wrapper.
- On failure, `run_with_diag.sh` reruns with PoCL debug and cache dump.

**Performance**
- `simx` is fast enough for full matrix runs.
- `rtlsim` is much slower; use tiny sizes when testing or skip full matrix.

## Example commands

```
cd "$VORTEX_BUILD"
source ./ci/toolchain_env.sh

# single app
./ci/blackbox.sh --driver=simx --app=test_pattern \
  --args="-w 320 -h 320 --dump --outdir $VORTEX_BUILD/artifacts/video_suite --prefix pretty_test_pattern"

# full suite
python3 $VORTEX_ROOT/tests/run_video_suite.py --build-dir $VORTEX_BUILD --driver simx --mode full
```

Example (path only):
```
cd /data/vortex/build
python3 /data/vortex/tests/run_video_suite.py --build-dir /data/vortex/build --driver simx --mode full
```

## OpenCL demo: Roman dodecahedron (dodecahedron_demo)

This demo renders a stylized **Roman dodecahedron** with solid (opaque) faces, optional holes/knobs, and a subtle beveled edge look. It is **not** part of regression runs or the video suite runner.

Run via build-blackbox (SimX recommended):

```
cd "$VORTEX_BUILD"
source ./ci/toolchain_env.sh
./ci/blackbox.sh --driver=simx --app=dodecahedron_demo \\
  --args=\"--style reference_blue -w 640 -h 640 \\
         --dump --outdir $VORTEX_BUILD/artifacts/demos/dodecahedron --prefix roman_ref\"
```

Output:
```
$VORTEX_BUILD/artifacts/demos/dodecahedron/roman_ref_output.ppm
```

Performance note:
- The full 640x640 render can take tens of minutes in SimX. Use `-w 320 -h 320` for a faster preview.

Optional tweaks:
- `--style reference_blue | iso_old` selects the preset (reference_blue is solid blue; iso_old is the legacy isometric look)
- `--palette mono_blue | face_hues` overrides the palette
- `--yaw/--pitch/--roll` and `--zoom` adjust the camera
- `--alpha <0..1>` controls face transparency (useful with `iso_old`)
- `--holes 0|1` toggles pentagon holes
- `--rings 0|1` toggles decorative rings around holes
- `--knobs 0|1` toggles vertex knobs
- `--knob-diam-frac <f>` sets knob diameter as a fraction of edge length (reference_blue default is 0.20 ≈ 1/5 edge)
- `--wire 0|1` toggles subtle edge darkening
- `--wire-thickness <f>` controls edge thickness
- `--knob-radius <f>` overrides the edge-relative size with a pixel radius

Legacy isometric look (if desired):
```
./ci/blackbox.sh --driver=simx --app=dodecahedron_demo \\
  --args=\"--style iso_old --palette face_hues --alpha 0.55 -w 640 -h 640 \\
         --dump --outdir $VORTEX_BUILD/artifacts/demos/dodecahedron --prefix roman_iso\"
```

Viewing/conversion:
```
python3 - <<'PY'
from PIL import Image
img = Image.open(\"$VORTEX_BUILD/artifacts/demos/dodecahedron/roman_iso_output.ppm\")
img.save(\"$VORTEX_BUILD/artifacts/demos/dodecahedron/roman_iso_output.png\")
PY
```
