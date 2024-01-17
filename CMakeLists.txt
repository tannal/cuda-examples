cmake_minimum_required(VERSION 3.0.0)
project(ray VERSION 0.1.0 LANGUAGES C CXX)

enable_language(CUDA)

include(CTest)
enable_testing()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

file(GLOB EXAMPLE_SOURCES "*.cu")


foreach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
    get_filename_component(EXAMPLE_NAME ${EXAMPLE_SOURCE} NAME_WE)
    add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCE})
endforeach()