cmake_minimum_required(VERSION 3.11)

project("A2f" CXX C)

# Enable IDE Project Folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("${PROJECT_SOURCE_DIR}/src")

# Set Compiler Flags
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    add_compile_options(/std:c++latest /utf-8 /W4 /MP /wd4800)
else()
    add_compile_options("$<$<NOT:$<CONFIG:Debug>>:-O2>")
    add_compile_options(-std=c++17 -Wall -Wextra -pedantic)
    add_compile_options(-fvisibility=hidden -pthread -msse2)
    # Uncomment for profiling
    #add_compile_options(-pg)
    #add_link_options(-pg)
    add_compile_options(-Werror=return-type)
endif()

# Source Files
file(GLOB_RECURSE SOURCES "src/*.cpp")
add_executable(A2f ${SOURCES})
target_link_libraries(A2f PRIVATE ${OpenCV_LIBS})
