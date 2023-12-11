cmake_minimum_required(VERSION 3.26)
project(MCTS_TicTacToe)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# configure pybind11
list(APPEND CMAKE_PREFIX_PATH "/usr/local/share/cmake/pybind11")
find_package(pybind11 REQUIRED)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")


file(GLOB_RECURSE SOURCES "lib/*.cpp")

# Define the executable and its source files
add_library(${PROJECT_NAME} MODULE ${SOURCES})
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "mcts_tic_tac_toe")
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "")


# Include directories
include_directories(include)

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE pybind11::module)