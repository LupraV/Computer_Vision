# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.28.3/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.28.3/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/lvmacpro/Desktop/731_CVision/c++_projects/Kuwahara_parallel

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/lvmacpro/Desktop/731_CVision/c++_projects/Kuwahara_parallel/build

# Include any dependencies generated for this target.
include CMakeFiles/KuwaharaFilter.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/KuwaharaFilter.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/KuwaharaFilter.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/KuwaharaFilter.dir/flags.make

CMakeFiles/KuwaharaFilter.dir/src/kuwahara_filter.cpp.o: CMakeFiles/KuwaharaFilter.dir/flags.make
CMakeFiles/KuwaharaFilter.dir/src/kuwahara_filter.cpp.o: /Users/lvmacpro/Desktop/731_CVision/c++_projects/Kuwahara_parallel/src/kuwahara_filter.cpp
CMakeFiles/KuwaharaFilter.dir/src/kuwahara_filter.cpp.o: CMakeFiles/KuwaharaFilter.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/lvmacpro/Desktop/731_CVision/c++_projects/Kuwahara_parallel/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/KuwaharaFilter.dir/src/kuwahara_filter.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/KuwaharaFilter.dir/src/kuwahara_filter.cpp.o -MF CMakeFiles/KuwaharaFilter.dir/src/kuwahara_filter.cpp.o.d -o CMakeFiles/KuwaharaFilter.dir/src/kuwahara_filter.cpp.o -c /Users/lvmacpro/Desktop/731_CVision/c++_projects/Kuwahara_parallel/src/kuwahara_filter.cpp

CMakeFiles/KuwaharaFilter.dir/src/kuwahara_filter.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/KuwaharaFilter.dir/src/kuwahara_filter.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/lvmacpro/Desktop/731_CVision/c++_projects/Kuwahara_parallel/src/kuwahara_filter.cpp > CMakeFiles/KuwaharaFilter.dir/src/kuwahara_filter.cpp.i

CMakeFiles/KuwaharaFilter.dir/src/kuwahara_filter.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/KuwaharaFilter.dir/src/kuwahara_filter.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/lvmacpro/Desktop/731_CVision/c++_projects/Kuwahara_parallel/src/kuwahara_filter.cpp -o CMakeFiles/KuwaharaFilter.dir/src/kuwahara_filter.cpp.s

# Object files for target KuwaharaFilter
KuwaharaFilter_OBJECTS = \
"CMakeFiles/KuwaharaFilter.dir/src/kuwahara_filter.cpp.o"

# External object files for target KuwaharaFilter
KuwaharaFilter_EXTERNAL_OBJECTS =

KuwaharaFilter: CMakeFiles/KuwaharaFilter.dir/src/kuwahara_filter.cpp.o
KuwaharaFilter: CMakeFiles/KuwaharaFilter.dir/build.make
KuwaharaFilter: /usr/local/lib/libopencv_gapi.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_stitching.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_alphamat.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_aruco.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_bgsegm.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_bioinspired.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_ccalib.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_dnn_objdetect.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_dnn_superres.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_dpm.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_face.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_freetype.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_fuzzy.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_hfs.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_img_hash.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_intensity_transform.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_line_descriptor.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_mcc.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_quality.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_rapid.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_reg.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_rgbd.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_saliency.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_sfm.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_stereo.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_structured_light.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_superres.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_surface_matching.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_tracking.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_videostab.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_viz.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_wechat_qrcode.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_xfeatures2d.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_xobjdetect.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_xphoto.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_shape.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_highgui.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_datasets.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_plot.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_text.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_ml.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_phase_unwrapping.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_optflow.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_ximgproc.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_video.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_videoio.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_imgcodecs.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_objdetect.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_calib3d.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_dnn.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_features2d.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_flann.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_photo.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_imgproc.4.9.0.dylib
KuwaharaFilter: /usr/local/lib/libopencv_core.4.9.0.dylib
KuwaharaFilter: CMakeFiles/KuwaharaFilter.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/lvmacpro/Desktop/731_CVision/c++_projects/Kuwahara_parallel/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable KuwaharaFilter"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/KuwaharaFilter.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/KuwaharaFilter.dir/build: KuwaharaFilter
.PHONY : CMakeFiles/KuwaharaFilter.dir/build

CMakeFiles/KuwaharaFilter.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/KuwaharaFilter.dir/cmake_clean.cmake
.PHONY : CMakeFiles/KuwaharaFilter.dir/clean

CMakeFiles/KuwaharaFilter.dir/depend:
	cd /Users/lvmacpro/Desktop/731_CVision/c++_projects/Kuwahara_parallel/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/lvmacpro/Desktop/731_CVision/c++_projects/Kuwahara_parallel /Users/lvmacpro/Desktop/731_CVision/c++_projects/Kuwahara_parallel /Users/lvmacpro/Desktop/731_CVision/c++_projects/Kuwahara_parallel/build /Users/lvmacpro/Desktop/731_CVision/c++_projects/Kuwahara_parallel/build /Users/lvmacpro/Desktop/731_CVision/c++_projects/Kuwahara_parallel/build/CMakeFiles/KuwaharaFilter.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/KuwaharaFilter.dir/depend

