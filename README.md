# Classifying Arm Exercises from Dual-IMU Sensor Data

**Alex Benny**

### Executive summary

**Project overview and goals:** This project builds a pipeline that turns raw dual-IMU (forearm + upper-arm) accelerometer/gyroscope
recordings into a labeled, per-window feature table, then uses that table to train and compare models that classify which arm exercise
is being performed. The end-to-end system spans a live serial-ingestion + complementary-filter orientation pipeline (Python, with a
C++/CUDA numeric core for the hot path), a batch feature-extraction path for offline recordings, and a windowing/labeling notebook that
produces the modeling dataset. Planned comparisons are a classical model (random forest / gradient boosting) trained on aggregated
per-window statistics against a sequence model (LSTM) trained directly on per-sample feature windows, to see whether the extra temporal
detail the sequence model sees is worth its added complexity.

**Findings:** *In progress.* The windowing/labeling stage (notebook 1) is complete and produces a clean, labeled feature table with no
missing values. Only one exercise (`bicep_curl`) is labeled so far, from a synthetic 20s two-sensor demo recording; a second real
hardware capture (`mpu_dual_log.csv`) is present but not yet labeled. Model training and comparison (notebooks 2 and 3) have not started
-- this section will be filled in once results exist.

**Results and conclusion:** *Pending notebooks 2 and 3.*

**Future research and development:** Recording additional exercises and subjects is the main blocker to a meaningful model comparison --
with a single class and 18 windows from one subject, no classifier can be evaluated for generalization. Once more classes are recorded,
useful next steps include comparing per-window aggregated features against raw per-sample sequences (the notebook 2 vs. notebook 3
comparison this project is built around), and evaluating whether decomposition- or AR/ARMA-coefficient features add signal beyond the
current mean/std/min/max/rms aggregates.

### Rationale

Wearable IMU sensors make it possible to track and classify physical exercises without cameras or specialized lab equipment, which
matters for applications like remote physical therapy adherence tracking, home fitness coaching, and rep-counting wearables. A
dual-IMU setup (forearm + upper-arm) captures joint-level motion (e.g. elbow flexion) that a single wrist-worn sensor can't, but that
also means the feature engineering and modeling choices are less standardized than for single-sensor HAR (human activity recognition)
datasets. This project works through that pipeline end-to-end, from raw sensor streams to a compared set of classifiers, using
real captured and synthetic dual-IMU data.

### Research Question

What is the best model for classifying which arm exercise is being performed from dual-IMU forearm + upper-arm sensor data -- a
classical model trained on aggregated per-window statistics, or a sequence model trained directly on per-sample feature windows?

### Data Sources

**Dataset:** Recordings are tracked in `manifest.csv`, one row per session, each pointing at a raw CSV in one of two layouts:
`raw_dual` (native `mpu_dual_log.csv` / `read_mpu_dual.py` output) or `curl_csv` (the long-format `example_imu_data_bundle` layout, one
row per sensor per timestamp). Each row also carries the session's exercise `label` (blank rows are excluded from the dataset) and
`subject`. As more exercises and subjects are recorded, new rows are added to `manifest.csv` and the pipeline picks them up with no
code changes.

Currently one session is labeled: `curl_bicep_01` (`bicep_curl`), a synthetic 20s, 50Hz, two-sensor (forearm + bicep) recording from
the example bundle. `mpu_dual_log_01` is a real hardware capture that is present but unlabeled, so it's skipped until labeled.

**Exploratory data analysis:** The labeled session's raw accel/gyro magnitudes show a clear periodic pattern (visible in notebook 1)
consistent with the ~1-rep-per-4-second curl motion described in the example bundle's documentation. After running the pipeline's
complementary-filter feature extraction, the derived `elbow_flex` signal shows the same periodicity even more clearly, which is the
signal a window needs to capture whole and what drives the window-size choice below.

**Cleaning and preparation:** Raw accel/gyro streams are run through the existing `imu_preprocess.py` -> `imu_features.py` pipeline
(via the batch entry point `imu_batch.py`, which uses a compiled C++ backend when available and falls back to pure Python otherwise)
to produce per-sample motion features: pitch per sensor, elbow flexion, gyro/accel magnitude, and jerk. There were no missing values
or invalid ranges after this step.

**Final Dataset:** Per-sample features are sliced into fixed-size, overlapping, time-based windows (2s window, 1s hop -- 50%
overlap) and each window is aggregated into mean/std/min/max/rms per feature, producing one row per window with `session_id`,
`window_index`, `label`, `subject`, and 57 aggregated feature columns. The current labeled dataset has 18 windows, all `bicep_curl`,
all from subject `alex` -- a starting point, not yet enough to train or evaluate a classifier for generalization. This is written to
`data/exercise_windows.csv` by notebook 1 (`1. IMU_Exercise_Windowing_and_Labeling.ipynb`), which also produces the label-distribution
and feature-summary plots documented in the notebook itself.

### Methodology

*Not yet started -- planned for notebooks 2 and 3.*

**Notebook 2 (classical model):** Train and evaluate a random forest / gradient boosting classifier on the aggregated per-window
feature table (`data/exercise_windows.csv`), with holdout or cross-validation once multiple classes/subjects are recorded.

**Notebook 3 (sequence model):** Train an LSTM directly on the per-sample feature windows (before aggregation), to compare a model
that sees the full temporal shape of each rep against notebook 2's summary-statistic approach.

### Model evaluation and results

*Pending notebooks 2 and 3.*

### Outline of project

- [Notebook 1: Exercise Windowing & Labeling](<1. IMU_Exercise_Windowing_and_Labeling.ipynb>) -- raw sessions to a labeled,
  windowed feature table.
- Notebook 2: classical model training and evaluation -- not yet written.
- Notebook 3: sequence model (LSTM) training and evaluation -- not yet written.
- [`manifest.csv`](manifest.csv) -- session index (file, format, label, subject) driving the whole pipeline.
- [`imu_dataset.py`](imu_dataset.py) -- windowing/aggregation logic shared by all notebooks.
- [`data/exercise_windows.csv`](data/exercise_windows.csv) -- current output of notebook 1, the input to notebooks 2 and 3.

### Repository layout (engineering pipeline)

- `imu_preprocess.py`, `imu_features.py` -- the original live pipeline
  (serial ingest -> bias/filter -> feature extraction), unchanged.
- `imu_dashboard.py`, `main.py`, `arm_mapper.py`, `camera.py`, `object_3d.py`,
  `projection.py`, `matrix_functions.py`, `imu_serial.py`, `read_mpu_dual.py`,
  `stream_quat.py` -- visualization, CSV replay, and other original tooling,
  unchanged.
- `imu_batch.py` -- batch-processing entry point. Runs the same
  bias/filter/feature-extraction math as `imu_preprocess.py` +
  `imu_features.py`, but over a whole recorded array at once. Uses the
  compiled `_imu_cpp` extension when it's built, and transparently falls back
  to a pure-Python implementation otherwise -- both paths are covered by
  `tests/test_parity.py`.
- `imu_dataset.py` -- windowing/labeling logic for the capstone dataset
  (`manifest.csv` -> per-session feature extraction -> time-windowed
  aggregation), used by notebook 1 and shared with notebooks 2/3.
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

### Contact and Further Information

Alex Benny

Email: alexben6g@gmail.com
