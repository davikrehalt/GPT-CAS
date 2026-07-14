cmake_minimum_required(VERSION 3.20)

project(
  laughableengine
  VERSION 0.3.0
  DESCRIPTION "A small native exact-algebra engine designed to grow"
  LANGUAGES CXX
)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(GMP REQUIRED)
find_package(Threads REQUIRED)

option(LAUGHABLEENGINE_BUILD_CLI "Build the laughable command-line tool" ON)
option(LAUGHABLEENGINE_BUILD_PYTHON "Build native Python bindings" OFF)
option(
  LAUGHABLEENGINE_BUILD_BENCHMARKS
  "Build the native kernel benchmark harness"
  OFF
)

add_library(laughableengine INTERFACE)
add_library(laughableengine::laughableengine ALIAS laughableengine)

target_compile_features(laughableengine INTERFACE cxx_std_20)
target_include_directories(
  laughableengine
  INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)
target_link_libraries(
  laughableengine
  INTERFACE
    GMP::gmpxx
    Threads::Threads
)

function(laughableengine_enable_warnings target)
  if(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang|GNU")
    target_compile_options(
      ${target}
      PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
  endif()
endfunction()

if(LAUGHABLEENGINE_BUILD_CLI)
  add_executable(laughable_cli src/cli.cpp)
  set_target_properties(laughable_cli PROPERTIES OUTPUT_NAME laughable)
  target_link_libraries(
    laughable_cli
    PRIVATE
      laughableengine::laughableengine
  )
  target_compile_definitions(
    laughable_cli
    PRIVATE
      LAUGHABLEENGINE_VERSION="${PROJECT_VERSION}"
  )
  laughableengine_enable_warnings(laughable_cli)
  install(TARGETS laughable_cli RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
  install(
    PROGRAMS tools/laughable_jg_verify.py
    DESTINATION "${CMAKE_INSTALL_BINDIR}"
    RENAME laughable-jg-verify
  )
endif()

if(LAUGHABLEENGINE_BUILD_PYTHON)
  set(PYBIND11_FINDPYTHON ON)
  find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 2.13.6 CONFIG REQUIRED)

  pybind11_add_module(_laughableengine MODULE python/bindings.cpp)
  target_link_libraries(
    _laughableengine
    PRIVATE
      laughableengine::laughableengine
  )
  laughableengine_enable_warnings(_laughableengine)
  set(_laughableengine_python_package_dir
      "${CMAKE_CURRENT_BINARY_DIR}/python/laughableengine")
  file(MAKE_DIRECTORY "${_laughableengine_python_package_dir}")
  set_target_properties(
    _laughableengine
    PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${_laughableengine_python_package_dir}"
  )
  configure_file(
    python/laughableengine/__init__.py
    "${_laughableengine_python_package_dir}/__init__.py"
    COPYONLY
  )
  install(TARGETS _laughableengine LIBRARY DESTINATION laughableengine)
  install(FILES python/laughableengine/__init__.py DESTINATION laughableengine)
  install(FILES tools/laughable_jg_verify.py DESTINATION ".")
endif()

if(LAUGHABLEENGINE_BUILD_BENCHMARKS)
  add_executable(
    laughableengine_kernel_benchmark
    benchmarks/kernel_benchmark.cpp
  )
  target_link_libraries(
    laughableengine_kernel_benchmark
    PRIVATE
      laughableengine::laughableengine
  )
  laughableengine_enable_warnings(laughableengine_kernel_benchmark)
endif()

install(DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(
  TARGETS laughableengine
  EXPORT laughableengineTargets
)
install(
  EXPORT laughableengineTargets
  FILE laughableengineTargets.cmake
  NAMESPACE laughableengine::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/laughableengine"
)
configure_package_config_file(
  cmake/laughableengineConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/laughableengineConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/laughableengine"
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/laughableengineConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMajorVersion
)
install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/laughableengineConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/laughableengineConfigVersion.cmake"
    cmake/FindGMP.cmake
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/laughableengine"
)

include(CTest)
if(BUILD_TESTING)
  add_executable(laughableengine_cycle_audit_tests tests/cycle_audit.cpp)
  add_executable(laughableengine_e10_fixture_tests tests/e10_fixture.cpp)
  add_executable(
    laughableengine_e10_cotangent_h1_tests
    tests/e10_cotangent_h1.cpp
  )
  add_custom_target(
    laughableengine_check_e10_qq
    COMMAND laughableengine_e10_cotangent_h1_tests --qq
    DEPENDS laughableengine_e10_cotangent_h1_tests
    COMMENT "Running the complete exact QQ E10 regression"
    USES_TERMINAL
  )
  add_executable(laughableengine_tests tests/smoke.cpp)
  add_executable(laughableengine_groebner_tests tests/groebner.cpp)
  add_executable(laughableengine_h1_tests tests/h1.cpp)
  add_executable(
    laughableengine_cotangent_h1_tests
    tests/cotangent_h1.cpp
  )
  add_executable(
    laughableengine_finite_algebra_tests
    tests/finite_algebra.cpp
  )
  add_executable(laughableengine_ideal_tests tests/ideal.cpp)
  add_executable(laughableengine_linear_algebra_tests tests/linear_algebra.cpp)
  add_executable(laughableengine_matrix_tests tests/matrix.cpp)
  add_executable(laughableengine_matrix_space_tests tests/matrix_space.cpp)
  add_executable(laughableengine_parser_tests tests/parser.cpp)
  add_executable(laughableengine_quotient_tests tests/quotient.cpp)
  add_executable(
    laughableengine_quotient_context_tests
    tests/quotient_context.cpp
  )
  add_executable(laughableengine_sparse_matrix_tests tests/sparse_matrix.cpp)
  add_executable(
    laughableengine_incremental_sparse_row_space_tests
    tests/incremental_sparse_row_space.cpp
  )
  add_executable(laughableengine_certificate_tests tests/certificate.cpp)
  add_executable(
    laughableengine_candidate_executor_tests
    tests/candidate_executor.cpp
  )
  add_executable(laughableengine_discovery_tests tests/discovery.cpp)
  add_executable(
    laughableengine_inverse_discovery_tests
    tests/inverse_discovery.cpp
  )
  add_executable(
    laughableengine_inverse_system_tests
    tests/inverse_system.cpp
  )
  add_executable(
    laughableengine_packed_discovery_tests
    tests/packed_discovery.cpp
  )
  add_executable(
    laughableengine_reconstruction_tests
    tests/reconstruction.cpp
  )
  add_executable(
    laughableengine_truncated_monomial_space_tests
    tests/truncated_monomial_space.cpp
  )
  target_link_libraries(
    laughableengine_cycle_audit_tests
    PRIVATE
      laughableengine::laughableengine
  )
  target_link_libraries(
    laughableengine_e10_fixture_tests
    PRIVATE
      laughableengine::laughableengine
  )
  target_link_libraries(
    laughableengine_e10_cotangent_h1_tests
    PRIVATE
      laughableengine::laughableengine
  )
  target_link_libraries(
    laughableengine_tests
    PRIVATE
      laughableengine::laughableengine
  )
  target_link_libraries(
    laughableengine_groebner_tests
    PRIVATE
      laughableengine::laughableengine
  )
  target_link_libraries(
    laughableengine_h1_tests
    PRIVATE
      laughableengine::laughableengine
  )
  target_link_libraries(
    laughableengine_cotangent_h1_tests
    PRIVATE
      laughableengine::laughableengine
  )
  target_link_libraries(
    laughableengine_finite_algebra_tests
    PRIVATE
      laughableengine::laughableengine
  )
  target_link_libraries(
    laughableengine_ideal_tests
    PRIVATE
      laughableengine::laughableengine
  )
  target_link_libraries(
    laughableengine_linear_algebra_tests
    PRIVATE
      laughableengine::laughableengine
  )
  target_link_libraries(
    laughableengine_matrix_tests
    PRIVATE
      laughableengine::laughableengine
  )
  target_link_libraries(
    laughableengine_matrix_space_tests
    PRIVATE
      laughableengine::laughableengine
  )
  target_link_libraries(
    laughableengine_parser_tests
    PRIVATE
      laughableengine::laughableengine
  )
  target_link_libraries(
    laughableengine_quotient_tests
    PRIVATE
      laughableengine::laughableengine
  )
  target_link_libraries(
    laughableengine_quotient_context_tests
    PRIVATE
      laughableengine::laughableengine
  )
  target_link_libraries(
    laughableengine_sparse_matrix_tests
    PRIVATE
      laughableengine::laughableengine
  )
  target_link_libraries(
    laughableengine_incremental_sparse_row_space_tests
    PRIVATE
      laughableengine::laughableengine
  )
  target_link_libraries(
    laughableengine_certificate_tests
    PRIVATE
      laughableengine::laughableengine
  )
  target_link_libraries(
    laughableengine_candidate_executor_tests
    PRIVATE
      laughableengine::laughableengine
  )
  target_link_libraries(
    laughableengine_discovery_tests
    PRIVATE
      laughableengine::laughableengine
  )
  target_link_libraries(
    laughableengine_inverse_discovery_tests
    PRIVATE
      laughableengine::laughableengine
  )
  target_link_libraries(
    laughableengine_inverse_system_tests
    PRIVATE
      laughableengine::laughableengine
  )
  target_link_libraries(
    laughableengine_packed_discovery_tests
    PRIVATE
      laughableengine::laughableengine
  )
  target_link_libraries(
    laughableengine_reconstruction_tests
    PRIVATE
      laughableengine::laughableengine
  )
  target_link_libraries(
    laughableengine_truncated_monomial_space_tests
    PRIVATE
      laughableengine::laughableengine
  )
  laughableengine_enable_warnings(laughableengine_cycle_audit_tests)
  laughableengine_enable_warnings(laughableengine_e10_fixture_tests)
  laughableengine_enable_warnings(laughableengine_e10_cotangent_h1_tests)
  laughableengine_enable_warnings(laughableengine_tests)
  laughableengine_enable_warnings(laughableengine_groebner_tests)
  laughableengine_enable_warnings(laughableengine_h1_tests)
  laughableengine_enable_warnings(laughableengine_cotangent_h1_tests)
  laughableengine_enable_warnings(laughableengine_finite_algebra_tests)
  laughableengine_enable_warnings(laughableengine_ideal_tests)
  laughableengine_enable_warnings(laughableengine_linear_algebra_tests)
  laughableengine_enable_warnings(laughableengine_matrix_tests)
  laughableengine_enable_warnings(laughableengine_matrix_space_tests)
  laughableengine_enable_warnings(laughableengine_parser_tests)
  laughableengine_enable_warnings(laughableengine_quotient_tests)
  laughableengine_enable_warnings(laughableengine_quotient_context_tests)
  laughableengine_enable_warnings(laughableengine_sparse_matrix_tests)
  laughableengine_enable_warnings(
    laughableengine_incremental_sparse_row_space_tests
  )
  laughableengine_enable_warnings(laughableengine_certificate_tests)
  laughableengine_enable_warnings(laughableengine_candidate_executor_tests)
  laughableengine_enable_warnings(laughableengine_discovery_tests)
  laughableengine_enable_warnings(laughableengine_inverse_discovery_tests)
  laughableengine_enable_warnings(laughableengine_inverse_system_tests)
  laughableengine_enable_warnings(laughableengine_packed_discovery_tests)
  laughableengine_enable_warnings(laughableengine_reconstruction_tests)
  laughableengine_enable_warnings(
    laughableengine_truncated_monomial_space_tests
  )
  add_test(
    NAME laughableengine.cycle_audit
    COMMAND laughableengine_cycle_audit_tests
  )
  add_test(
    NAME laughableengine.e10_fixture
    COMMAND laughableengine_e10_fixture_tests
  )
  add_test(
    NAME laughableengine.e10_cotangent_h1
    COMMAND laughableengine_e10_cotangent_h1_tests
  )
  set_tests_properties(
    laughableengine.e10_cotangent_h1
    PROPERTIES LABELS "slow;e10"
  )
  add_test(NAME laughableengine.smoke COMMAND laughableengine_tests)
  add_test(NAME laughableengine.groebner COMMAND laughableengine_groebner_tests)
  add_test(NAME laughableengine.h1 COMMAND laughableengine_h1_tests)
  add_test(
    NAME laughableengine.cotangent_h1
    COMMAND laughableengine_cotangent_h1_tests
  )
  add_test(
    NAME laughableengine.finite_algebra
    COMMAND laughableengine_finite_algebra_tests
  )
  add_test(NAME laughableengine.ideal COMMAND laughableengine_ideal_tests)
  add_test(
    NAME laughableengine.linear_algebra
    COMMAND laughableengine_linear_algebra_tests
  )
  add_test(NAME laughableengine.matrix COMMAND laughableengine_matrix_tests)
  add_test(
    NAME laughableengine.matrix_space
    COMMAND laughableengine_matrix_space_tests
  )
  add_test(NAME laughableengine.parser COMMAND laughableengine_parser_tests)
  add_test(NAME laughableengine.quotient COMMAND laughableengine_quotient_tests)
  add_test(
    NAME laughableengine.quotient_context
    COMMAND laughableengine_quotient_context_tests
  )
  add_test(
    NAME laughableengine.sparse_matrix
    COMMAND laughableengine_sparse_matrix_tests
  )
  add_test(
    NAME laughableengine.incremental_sparse_row_space
    COMMAND laughableengine_incremental_sparse_row_space_tests
  )
  add_test(
    NAME laughableengine.certificate
    COMMAND laughableengine_certificate_tests
  )
  add_test(
    NAME laughableengine.candidate_executor
    COMMAND laughableengine_candidate_executor_tests
  )
  add_test(
    NAME laughableengine.discovery
    COMMAND laughableengine_discovery_tests
  )
  add_test(
    NAME laughableengine.inverse_discovery
    COMMAND laughableengine_inverse_discovery_tests
  )
  add_test(
    NAME laughableengine.inverse_system
    COMMAND laughableengine_inverse_system_tests
  )
  add_test(
    NAME laughableengine.packed_discovery
    COMMAND laughableengine_packed_discovery_tests
  )
  add_test(
    NAME laughableengine.reconstruction
    COMMAND laughableengine_reconstruction_tests
  )
  add_test(
    NAME laughableengine.truncated_monomial_space
    COMMAND laughableengine_truncated_monomial_space_tests
  )
  find_package(Python 3.9 COMPONENTS Interpreter QUIET)
  if(Python_Interpreter_FOUND)
    add_test(
      NAME laughableengine.independent_verifier
      COMMAND
        "${Python_EXECUTABLE}" -m unittest discover
        -s "${CMAKE_CURRENT_SOURCE_DIR}/tests"
        -p independent_verifier_test.py
        -v
    )
  endif()
  if(LAUGHABLEENGINE_BUILD_CLI)
    add_test(
      NAME laughableengine.cli
      COMMAND
        "${CMAKE_COMMAND}"
        "-DCLI=$<TARGET_FILE:laughable_cli>"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_test.cmake"
    )
  endif()
  if(LAUGHABLEENGINE_BUILD_PYTHON)
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -c "import pytest"
      RESULT_VARIABLE _laughableengine_has_pytest
      OUTPUT_QUIET
      ERROR_QUIET
    )
    if(_laughableengine_has_pytest EQUAL 0)
      add_test(
        NAME laughableengine.python
        COMMAND
          "${Python_EXECUTABLE}" -m pytest -q
          "${CMAKE_CURRENT_SOURCE_DIR}/tests/python"
      )
      set_tests_properties(
        laughableengine.python
        PROPERTIES
          ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
      )
    else()
      message(STATUS "pytest not found; skipping Python binding tests")
    endif()
  endif()
endif()
