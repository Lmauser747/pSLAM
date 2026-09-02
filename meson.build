project(
  'Pool-SLAM',
  'cpp',
  version: '0.1.0',
  meson_version: '>= 0.60.0',
  default_options: [
    'cpp_std=c++17',
    'warning_level=3',
    'werror=true',
    'buildtype=debugoptimized'
  ]
)

cpp = meson.get_compiler('cpp')

extra_compile_args = []
if cpp.get_id() in ['gcc', 'clang']
  extra_compile_args += [
    '-march=native',
    '-Wno-unused-parameter'
  ]
endif
add_project_arguments(extra_compile_args, language: 'cpp')

inc_global = include_directories('include')

thread_dep = dependency('threads')
eigen_dep = dependency('eigen3', fallback: ['eigen', 'eigen_dep'])
opencv_dep = dependency('opencv4', required: true)

if get_option('build_python_bindings')
  py_mod = import('python')
  py_installation = py_mod.find_installation('python3')
  pybind11_dep = dependency('pybind11', fallback: ['pybind11', 'pybind11_dep'])
endif

subdir('src')

if get_option('enable_tests')
  gtest_dep = dependency('gtest_main', fallback: ['gtest', 'gtest_main_dep'])
  subdir('tests')
endif

if get_option('enable_benchmarks')
  benchmark_dep = dependency('benchmark', fallback: ['google-benchmark', 'google_benchmark_dep'])
  subdir('benchmarks')
endif