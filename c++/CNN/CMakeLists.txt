cmake_minimum_required(VERSION 3.10)

project(cnn)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories("${PROJECT_SOURCE_DIR}/src")

add_executable(cnn 
    src/main.cpp
    src/class.cpp
    src/numcpp.cpp
    src/utils.cpp
)

set_target_properties(cnn PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -flto -funroll-loops -ffast-math -fopenmp")

# Handle Profile Guided Optimization (PGO) -- Initial compile to generate profiling data
# set_target_properties(cnn PROPERTIES COMPILE_FLAGS "-fprofile-generate")
# set_target_properties(cnn PROPERTIES LINK_FLAGS "-fprofile-generate")

# Alternatively, to use profile data (this block should replace the above for the second compile)
# set_target_properties(cnn PROPERTIES COMPILE_FLAGS "-fprofile-use")
# set_target_properties(cnn PROPERTIES LINK_FLAGS "-fprofile-use")
