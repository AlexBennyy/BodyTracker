# MotionTracking

Dual-IMU body tracking: serial ingestion, a complementary-filter orientation
pipeline, and a pygame 3D visualizer, in Python. The numeric core of that
pipeline (bias calibration, low-pass filtering, complementary-filter feature
extraction) also has a C++ port with an optional CUDA kernel, built with
CMake and exposed back to Python through a pybind11 extension.

## Layout

- `imu_preprocess.py`, `imu_features.py` -- the original live pipeline
  (serial ingest -> bias/filter -> feature extraction), unchanged.
- `imu_dashboard.py`, `main.py`, `arm_mapper.py`, `camera.py`, `object_3d.py`,
  `projection.py`, `matrix_functions.py`, `imu_serial.py`, `read_mpu_dual.py`,
  `stream_quat.py` -- visualization, CSV replay, and other original tooling,
  unchanged.
- `imu_batch.py` -- new opt-in batch-processing entry point. Runs the same
  bias/filter/feature-extraction math as `imu_preprocess.py` +
  `imu_features.py`, but over a whole recorded array at once. Uses the
  compiled `_imu_cpp` extension when it's built, and transparently falls back
  to a pure-Python implementation otherwise -- both paths are covered by
  `tests/test_parity.py`.
- `include/`, `src/` -- the C++ core (`imu_core`): `imu_types.hpp` (POD
  types), `imu_math.{hpp,cpp}` (vector ops, `LowPass3`, `ComplementaryPitch`),
  `imu_processor.{hpp,cpp}` (`calibrate_bias`, `process_batch`), and
  `imu_cuda.{hpp,cu}` (CUDA port of the one embarrassingly-parallel piece,
  per-sample vector norm -- the rest of the pipeline is sequential in time
  per sensor and isn't a good first GPU-porting candidate).
- `src/python_bindings.cpp` -- the pybind11 module (`_imu_cpp`) that exposes
  `imu_core` to Python via numpy arrays.
- `tests/test_imu.cpp` -- GoogleTest unit tests for the C++ core.
- `tests/test_parity.py` -- checks the C++ backend and the pure-Python
  fallback in `imu_batch.py` agree, over real recorded data.
- `benchmarks/cpu_benchmark.cpp`, `benchmarks/gpu_benchmark.cu` -- Google
  Benchmark harnesses for `process_batch` and `vec3_norm_batch`, structured
  so the CPU and (when built) CUDA numbers are directly comparable.

## Building the C++ core

Requires CMake 3.18+ and a C++17 compiler. On macOS:

```sh
brew install cmake pybind11 googletest google-benchmark
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Each dependency is optional at configure time -- if `pybind11`, `googletest`,
or `google-benchmark` isn't found, that target is skipped with a warning
rather than failing the whole configure. CUDA is auto-detected
(`check_language(CUDA)`); it's off on any machine without `nvcc`, which is
every CI runner and most dev machines here, so `gpu_benchmark` won't build
unless a CUDA toolchain is present.

Useful CMake options (`-D<option>=OFF` to disable): `BUILD_TESTS`,
`BUILD_BENCHMARKS`, `BUILD_PYTHON_BINDINGS`, `BUILD_CUDA`.

The `_imu_cpp` extension is built directly into the repo root, so
`import _imu_cpp` (or `import imu_batch`, which uses it automatically) works
from Python with no `PYTHONPATH` changes once you've run `cmake --build`.

## Running tests

```sh
ctest --test-dir build --output-on-failure   # C++ unit tests
python3 tests/test_parity.py -v              # C++ vs. Python-fallback parity
```

## Running benchmarks

```sh
./build/cpu_benchmark
./build/gpu_benchmark   # only if built with CUDA support
```

## CI

`.github/workflows/ci.yml` builds and tests the C++ core plus the Python
parity check on a CPU-only GitHub-hosted macOS runner (via Homebrew, mirroring
local setup). CUDA isn't exercised in CI since hosted runners have no GPU.
