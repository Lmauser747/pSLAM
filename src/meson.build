core_sources = files(
)

inc_src = include_directories('.')

pool_slam_lib = library(
  'pool_slam',
  core_sources,
  include_directories: [inc_global, inc_src],
  dependencies: [eigen_dep, opencv_dep, thread_dep],
  install: true
)

pool_slam_dep = declare_dependency(
  link_with: pool_slam_lib,
  include_directories: [inc_global, inc_src],
  dependencies: [eigen_dep, opencv_dep, thread_dep]
)

executable(
  'pool_slam_app',
  files('main.cpp'),
  dependencies: [pool_slam_dep],
  install: true
)

if get_option('build_python_bindings')
  py_installation.extension_module(
    'pool_slam_py',
    files('bindings.cpp'),
    dependencies: [pool_slam_dep, pybind11_dep],
    install: true
  )
endif