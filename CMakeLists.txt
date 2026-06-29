cmake_minimum_required(VERSION 3.10)
project(zed_rviz)

# ── Suppress legacy CMake policy warnings from ZED/CUDA find scripts ──────────
cmake_policy(SET CMP0146 OLD)   # FindCUDA module
cmake_policy(SET CMP0153 OLD)   # exec_program
cmake_policy(SET CMP0072 NEW)   # FindOpenGL prefers GLVND
cmake_policy(SET CMP0144 NEW)   # upper-case <PKG>_ROOT variables

# ── Dependencies ──────────────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ZED 3 REQUIRED)
find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(yaml-cpp REQUIRED)

# ── Shared config ─────────────────────────────────────────────────────────────
set(COMMON_INCLUDES
  ${OpenCV_INCLUDE_DIRS}
  ${ZED_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
)

set(COMMON_LIBS
  ${OpenCV_LIBRARIES}
  ${ZED_LIBRARIES}
  ${CUDA_LIBRARIES}
  glfw
  OpenGL::GL
  Eigen3::Eigen
)

set(COMMON_ROS_DEPS
  rclcpp
  sensor_msgs
  visualization_msgs
  geometry_msgs
  std_msgs
  nav_msgs
)

# ── Target 1: original per-frame mesh viewer ──────────────────────────────────
add_executable(zed_rviz src/zed_rviz.cpp)
target_include_directories(zed_rviz PUBLIC ${COMMON_INCLUDES})
ament_target_dependencies(zed_rviz ${COMMON_ROS_DEPS})
target_link_libraries(zed_rviz ${COMMON_LIBS} yaml-cpp)
target_compile_features(zed_rviz PRIVATE cxx_std_17)

# ── Target 2: live SLAM spatial mapper ────────────────────────────────────────
add_executable(zed_slam_map src/zed_slam_map.cpp)
target_include_directories(zed_slam_map PUBLIC ${COMMON_INCLUDES})
ament_target_dependencies(zed_slam_map ${COMMON_ROS_DEPS})
target_link_libraries(zed_slam_map ${COMMON_LIBS} yaml-cpp)
target_compile_features(zed_slam_map PRIVATE cxx_std_17)

# ── Target 3: sensor viewer and positional tracking in RViz ──────────────────
add_executable(zed_sensor_rviz src/zed_sensor_rviz.cpp)
target_include_directories(zed_sensor_rviz PUBLIC ${COMMON_INCLUDES})
ament_target_dependencies(zed_sensor_rviz ${COMMON_ROS_DEPS})
target_link_libraries(zed_sensor_rviz ${COMMON_LIBS} yaml-cpp)
target_compile_features(zed_sensor_rviz PRIVATE cxx_std_17)

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS zed_rviz zed_slam_map zed_sensor_rviz
  DESTINATION lib/${PROJECT_NAME})

ament_package()

