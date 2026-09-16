cmake_minimum_required(VERSION 3.20)
project(RealSaSRuntime VERSION 0.4.0 LANGUAGES CXX)

option(REALSAS_BUILD_SHARED "Build shared runtime library" ON)
option(REALSAS_BUILD_EXAMPLES "Build runtime examples" ON)
option(REALSAS_BUILD_TESTS "Build runtime ABI smoke tests" ON)

find_package(ZLIB REQUIRED)
find_package(PNG REQUIRED)
find_package(LibArchive REQUIRED)

set(REALSAS_RUNTIME_SOURCES src/realsas_runtime.cpp)
if(REALSAS_BUILD_SHARED)
  add_library(realsas_runtime SHARED ${REALSAS_RUNTIME_SOURCES})
  target_compile_definitions(realsas_runtime PRIVATE REALSAS_RUNTIME_BUILD REALSAS_RUNTIME_SHARED)
else()
  add_library(realsas_runtime STATIC ${REALSAS_RUNTIME_SOURCES})
endif()
add_library(RealSaS::Runtime ALIAS realsas_runtime)
target_compile_features(realsas_runtime PUBLIC cxx_std_17)
target_include_directories(realsas_runtime PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_link_libraries(realsas_runtime PRIVATE LibArchive::LibArchive PNG::PNG ZLIB::ZLIB)
set_target_properties(realsas_runtime PROPERTIES
  OUTPUT_NAME realsas_runtime
  EXPORT_NAME Runtime
  POSITION_INDEPENDENT_CODE ON)

# Runtime-v3 is intentionally kept behind an internal reference target until
# parser/raster/clipping conformance closes. This keeps the stable v2/v3 C ABI
# surface unchanged while allowing real .rss-v3 -> native PNG execution now.
add_library(realsas_runtime_v3_reference STATIC
  src/runtime_v3_reference.cpp)
target_compile_features(realsas_runtime_v3_reference PUBLIC cxx_std_17)
target_include_directories(realsas_runtime_v3_reference PRIVATE src)
target_link_libraries(realsas_runtime_v3_reference PRIVATE LibArchive::LibArchive PNG::PNG ZLIB::ZLIB)
set_target_properties(realsas_runtime_v3_reference PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(REALSAS_BUILD_EXAMPLES)
  add_executable(realsas_runtime_demo examples/realsas_runtime_demo.cpp)
  target_compile_features(realsas_runtime_demo PRIVATE cxx_std_17)
  target_link_libraries(realsas_runtime_demo PRIVATE RealSaS::Runtime PNG::PNG)
endif()

if(REALSAS_BUILD_EXAMPLES OR REALSAS_BUILD_TESTS)
  add_executable(realsas_runtime_v3_demo examples/realsas_runtime_v3_demo.cpp)
  target_compile_features(realsas_runtime_v3_demo PRIVATE cxx_std_17)
  target_link_libraries(realsas_runtime_v3_demo PRIVATE realsas_runtime_v3_reference PNG::PNG)
endif()

if(REALSAS_BUILD_TESTS)
  enable_testing()
  find_package(Python3 COMPONENTS Interpreter REQUIRED)

  add_executable(realsas_runtime_abi_smoke tests/runtime_abi_smoke.cpp)
  target_compile_features(realsas_runtime_abi_smoke PRIVATE cxx_std_17)
  target_link_libraries(realsas_runtime_abi_smoke PRIVATE RealSaS::Runtime)
  add_test(NAME realsas_runtime_abi_smoke COMMAND realsas_runtime_abi_smoke)

  add_executable(realsas_reference_raster_v3 tests/reference_raster_v3.cpp)
  target_compile_features(realsas_reference_raster_v3 PRIVATE cxx_std_17)
  add_test(NAME realsas_reference_raster_v3 COMMAND realsas_reference_raster_v3)

  add_test(
    NAME realsas_runtime_v3_writer_native_e2e
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime_v3_writer_native_e2e.py
      --runtime-demo $<TARGET_FILE:realsas_runtime_v3_demo>)

  add_test(
    NAME realsas_runtime_v3_clipping_native_e2e
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime_v3_clipping_native_e2e.py
      --runtime-demo $<TARGET_FILE:realsas_runtime_v3_demo>)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(REALSAS_RUNTIME_CMAKE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/RealSaSRuntime")
configure_package_config_file(
  cmake/RealSaSRuntimeConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/RealSaSRuntimeConfig.cmake"
  INSTALL_DESTINATION "${REALSAS_RUNTIME_CMAKE_DIR}")
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/RealSaSRuntimeConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMajorVersion)

install(TARGETS realsas_runtime EXPORT RealSaSRuntimeTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT RealSaSRuntimeTargets
  NAMESPACE RealSaS::
  DESTINATION "${REALSAS_RUNTIME_CMAKE_DIR}")
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/RealSaSRuntimeConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/RealSaSRuntimeConfigVersion.cmake"
  DESTINATION "${REALSAS_RUNTIME_CMAKE_DIR}")
