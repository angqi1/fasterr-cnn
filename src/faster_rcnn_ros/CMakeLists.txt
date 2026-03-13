cmake_minimum_required(VERSION 3.5)
project(faster_rcnn_ros)

# C++17 for TensorRT 8.5
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 嵌入运行时库路径，避免 LD_LIBRARY_PATH 依赖
set(CMAKE_INSTALL_RPATH "/usr/local/lib")

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

# CUDA & TensorRT paths (JetPack 5.1.2)
find_package(CUDA REQUIRED)

# Include directories
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CUDA_INCLUDE_DIRS}
  /usr/include/aarch64-linux-gnu  # For NvInfer.h
)

# Executable
add_executable(faster_rcnn_node
  src/faster_rcnn_node.cpp
  src/faster_rcnn_detector.cpp
)

# ✅ 关键：链接依赖并设置 include 路径
ament_target_dependencies(faster_rcnn_node
  rclcpp
  sensor_msgs
  cv_bridge
)

# Link libraries
target_link_libraries(faster_rcnn_node
  ${OpenCV_LIBS}
  nvinfer
  nvinfer_plugin
  cudart
  cublas
  cudnn
)

# Install
install(TARGETS faster_rcnn_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch models
  DESTINATION share/${PROJECT_NAME}
)

ament_package()