cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(convert_pcd2ply)

find_package(PCL 1.8 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (convert_pcd2ply convert_pcd2ply.cpp)
target_link_libraries (convert_pcd2ply ${PCL_LIBRARIES})
