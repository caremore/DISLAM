cmake_minimum_required(VERSION 3.16)
project(FeatureVisualizer)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 寻找OpenCV
find_package(OpenCV REQUIRED)

# 包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# 创建可执行文件
add_executable(feature_visualizer
    src/main.cpp
    src/FeatureVisualizer.cpp
)

# 链接库
target_link_libraries(feature_visualizer
    ${OpenCV_LIBS}
)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
