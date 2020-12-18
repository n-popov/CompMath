cmake_minimum_required(VERSION 3.17)
project(CompMath)

set(CMAKE_CXX_STANDARD 20)

find_package(Boost)
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

add_executable(CompMath XI93a.cpp)

target_link_libraries(CompMath ${PYTHON_LIBRARIES})