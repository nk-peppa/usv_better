cmake_minimum_required(VERSION 3.16)

project(usv_better VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc)

find_path(REALSENSE2_INCLUDE_DIR
  NAMES librealsense2/rs.hpp
)
find_library(REALSENSE2_LIBRARY
  NAMES realsense2
)

if(NOT REALSENSE2_INCLUDE_DIR OR NOT REALSENSE2_LIBRARY)
  message(FATAL_ERROR "Intel RealSense librealsense2 is required. Install librealsense2-dev, then rerun CMake.")
endif()

add_library(usv_slam_execution STATIC
  src/SlamExecutionLayer.cc
)
target_include_directories(usv_slam_execution
  PUBLIC
    "${PROJECT_SOURCE_DIR}/include"
    "${REALSENSE2_INCLUDE_DIR}"
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(usv_slam_execution
  PUBLIC
    "${REALSENSE2_LIBRARY}"
    ${OpenCV_LIBS}
)
target_compile_options(usv_slam_execution PRIVATE -Wall -Wextra -pedantic)

add_executable(CommunicationLayer
  src/CommunicationLayer.cc
)
target_compile_options(CommunicationLayer PRIVATE -Wall -Wextra -pedantic)

add_executable(MainProcessor
  src/MainProcessor.cc
)
target_link_libraries(MainProcessor PRIVATE usv_slam_execution Threads::Threads)
target_compile_options(MainProcessor PRIVATE -Wall -Wextra -pedantic)

add_executable(ThrusterActuator
  src/ThrusterActuator.cc
)
target_compile_options(ThrusterActuator PRIVATE -Wall -Wextra -pedantic)

install(TARGETS CommunicationLayer MainProcessor ThrusterActuator
  RUNTIME DESTINATION bin
)
