cmake_minimum_required(VERSION 3.11)

project("Project" CXX C)

# Enable IDE Project Folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Tesseract paths 
set(Tesseract_INCLUDE_DIRS "/usr/local/include")
set(Tesseract_LIBRARIES "/usr/local/lib/libtesseract.dylib")
set(Tesseract_LIB_DIR "/usr/local/lib")

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Tesseract_INCLUDE_DIRS})
include_directories("${PROJECT_SOURCE_DIR}/src")

# Link directories
link_directories(${Tesseract_LIB_DIR})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler Flags
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

file(GLOB_RECURSE SOURCES "${PROJECT_SOURCE_DIR}/src/*.cpp")
add_executable(main ${SOURCES})
target_link_libraries(main PRIVATE ${OpenCV_LIBS} ${Tesseract_LIBRARIES})
