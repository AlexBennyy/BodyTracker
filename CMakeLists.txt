cmake_minimum_required(VERSION 3.18)
project(motion_tracking_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---- CUDA is opt-in and auto-detected: the CPU library, tests, and
# bindings must all build fine on machines (and CI runners) with no GPU. ----
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    option(BUILD_CUDA "Build CUDA kernels and GPU benchmarks" ON)
else()
    option(BUILD_CUDA "Build CUDA kernels and GPU benchmarks" OFF)
endif()

option(BUILD_TESTS "Build GoogleTest unit tests" ON)
option(BUILD_BENCHMARKS "Build Google Benchmark benchmarks" ON)
option(BUILD_PYTHON_BINDINGS "Build pybind11 Python bindings (_imu_cpp)" ON)

if(BUILD_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# ---- core CPU library: pure C++, no Python/CUDA dependency ----
add_library(imu_core STATIC
    src/imu_math.cpp
    src/imu_processor.cpp
)
target_include_directories(imu_core PUBLIC include)
target_compile_options(imu_core PRIVATE -Wall -Wextra)

if(BUILD_CUDA)
    add_library(imu_cuda STATIC src/imu_cuda.cu)
    target_include_directories(imu_cuda PUBLIC include)
    set_target_properties(imu_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 CONFIG QUIET)
    if(pybind11_FOUND)
        pybind11_add_module(_imu_cpp src/python_bindings.cpp)
        target_link_libraries(_imu_cpp PRIVATE imu_core)
        # Land the extension next to the existing Python modules so
        # `import _imu_cpp` works with no PYTHONPATH changes.
        set_target_properties(_imu_cpp PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
        )
    else()
        message(WARNING "pybind11 not found (CONFIG mode) -- skipping _imu_cpp target. "
                         "Install with 'brew install pybind11' or 'pip install pybind11'.")
    endif()
endif()

if(BUILD_TESTS)
    find_package(GTest CONFIG QUIET)
    if(GTest_FOUND)
        enable_testing()
        add_executable(test_imu tests/test_imu.cpp)
        target_link_libraries(test_imu PRIVATE imu_core GTest::gtest GTest::gtest_main)
        include(GoogleTest)
        gtest_discover_tests(test_imu)
    else()
        message(WARNING "GTest not found (CONFIG mode) -- skipping tests target. "
                         "Install with 'brew install googletest'.")
    endif()
endif()

if(BUILD_BENCHMARKS)
    find_package(benchmark CONFIG QUIET)
    if(benchmark_FOUND)
        add_executable(cpu_benchmark benchmarks/cpu_benchmark.cpp)
        target_link_libraries(cpu_benchmark PRIVATE imu_core benchmark::benchmark)

        if(BUILD_CUDA)
            add_executable(gpu_benchmark benchmarks/gpu_benchmark.cu)
            target_link_libraries(gpu_benchmark PRIVATE imu_core imu_cuda benchmark::benchmark)
        endif()
    else()
        message(WARNING "Google Benchmark not found (CONFIG mode) -- skipping benchmarks target. "
                         "Install with 'brew install google-benchmark'.")
    endif()
endif()
