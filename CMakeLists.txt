project(pedestrian_detect)
cmake_minimum_required(VERSION 3.2)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(PACKAGE_DEPS "pddemo") # All dependencies for the packaging target

# Utilities to make colored output easier
if(NOT WIN32) # Windows doesn't support things.
    string(ASCII 27 Esc)
    set(ClrNone         "${Esc}[m")
    set(ClrBold         "${Esc}[1m")
    set(Red             "${Esc}[31m")
    set(Green           "${Esc}[32m")
    set(Yellow          "${Esc}[33m")
    set(Blue            "${Esc}[34m")
    set(Magenta         "${Esc}[35m")
    set(Cyan            "${Esc}[36m")
    set(White           "${Esc}[37m")
    set(BRed            "${Esc}[1;31m")
    set(BGreen          "${Esc}[1;32m")
    set(BYellow         "${Esc}[1;33m")
    set(BBlue           "${Esc}[1;34m")
    set(BMagenta        "${Esc}[1;35m")
    set(BCyan           "${Esc}[1;36m")
    set(BWhite          "${Esc}[1;37m")
endif()

# Find and configure the Altera OpenCL runtime if enabled
option(ENABLE_FPGA "Build FPGA-based algorithms" ON)
if(${ENABLE_FPGA})
    if(EXISTS $ENV{ALTERAOCLSDKROOT})
        set(AOCL_SDK_ROOT "$ENV{ALTERAOCLSDKROOT}"
            CACHE FILEPATH "Altera SDK root directory")
    endif()
    if("${AOCL_SDK_ROOT}")
        message(SEND_ERROR "Cannot find Altera OpenCL SDK")
        set(ENABLE_FPGA OFF)
    else()
        execute_process(COMMAND ${AOCL_SDK_ROOT}/bin/aocl compile-config
            OUTPUT_VARIABLE AOCL_COMPILE_CONFIG)
        execute_process(COMMAND ${AOCL_SDK_ROOT}/bin/aocl link-config
            OUTPUT_VARIABLE AOCL_LINK_CONFIG)
        string(REGEX REPLACE "\n$" "" AOCL_COMPILE_CONFIG "${AOCL_COMPILE_CONFIG}")
        string(REGEX REPLACE "\n$" "" AOCL_LINK_CONFIG "${AOCL_LINK_CONFIG}")

        string(REGEX MATCHALL "-l([^ ]*)" AOCL_LIBRARIES "${AOCL_LINK_CONFIG}")
        string(REGEX MATCHALL "-L([^ ]*)" AOCL_LINK_SEARCH "${AOCL_LINK_CONFIG}")
        string(REGEX REPLACE "-l([^;]*)" "\\1" AOCL_LIBRARIES "${AOCL_LIBRARIES}")
        string(REGEX REPLACE "-L([^;]*)" "\\1" AOCL_LINK_SEARCH "${AOCL_LINK_SEARCH}")

        foreach(LIB ${AOCL_LIBRARIES})
            unset(LIB_PATH)
            foreach(LOC ${AOCL_LINK_SEARCH})
                if(NOT "${LIB_PATH}")
                    find_library(LIB_PATH ${LIB} PATHS ${LOC} NO_DEFAULT_PATH)
                endif()
                if(NOT "${LIB_PATH}" AND EXISTS "${LOC}/lib${LIB}.so")
                    set(LIB_PATH "${LOC}/lib${LIB}.so")
                endif()
            endforeach()
            if(NOT EXISTS "${LIB_PATH}")
                message(SEND_ERROR "Cannot find Altera CL library: ${LIB}")
            endif()
            list(APPEND AOCL_LINK_LIBRARIES ${LIB_PATH})
        endforeach()

        # Make sure include files are where we expect
        message(STATUS "Sanity-checking Altera SDK installation...")
        if(NOT EXISTS ${AOCL_SDK_ROOT}/host/include/CL/cl_ext_altera.h)
            message(WARNING "${BRed}Cannot find Altera host SDK (cl_ext_altera.h)${ClrNone}"
                "\nTo enable FPGA support, fix this and reconfigure.")
            set(ENABLE_FPGA OFF)
        elseif(NOT EXISTS ${AOCL_SDK_ROOT}/host/include/CL/cl.h)
            message(WARNING "Cannot find Altera host SDK (cl.h)"
                "\nTo enable FPGA support, fix this and reconfigure.")
            set(ENABLE_FPGA OFF)
        else()
            message(STATUS "${Green}Altera SDK installation looks good${ClrNone}")
        endif()
    endif()
endif()

find_package(OpenCV 3 REQUIRED core)
set(OCV_CORE_LIBS ${OpenCV_LIBS})
find_package(OpenCV 3 REQUIRED highgui imgproc objdetect tracking)
set(OCV_APP_LIBS ${OpenCV_LIBS})

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options filesystem thread)
add_definitions(-DBOOST_THREAD_VERSION=4)

# Find GStreamer
find_package(GStreamer)
if(${GSTREAMER_FOUND})
    add_definitions(-DWITH_GSTREAMER)
    include_directories(${GSTREAMER_INCLUDE_DIRS})
    find_package(GLIB REQUIRED COMPONENTS gobject)
    include_directories(${GLIB_INCLUDE_DIRS})
    set(NETWORK_OUTPUT ON)
else()
    message(WARNING "${Yellow}GStreamer is not installed - network video streaming will not work.${ClrNone}
    To avoid silent failures, the feature has been disabled.")
    set(NETWORK_OUTPUT OFF)
endif()

if(${ENABLE_FPGA})
    separate_arguments(AOCL_COMPILER_OPTS UNIX_COMMAND
        "${AOCL_LINK_CONFIG} ${AOCL_COMPILE_CONFIG}")
    set(AOCL_UTILITY_LIB external/altera/src/AOCLUtils/opencl.cpp)
    set(AOCL_UTILITY_INC external/altera/inc)

    # FPGA-based HOG SVM
    add_library(hog-ocl-fpga MODULE ${AOCL_UTILITY_LIB}
        src/algorithms/hog_ocl_fpga.cpp)
    target_compile_options(hog-ocl-fpga PRIVATE ${AOCL_COMPILER_OPTS})
    target_link_libraries(hog-ocl-fpga ${OCV_CORE_LIBS} ${AOCL_LINK_LIBRARIES})
    target_compile_features(hog-ocl-fpga PRIVATE cxx_auto_type cxx_range_for)
    target_include_directories(hog-ocl-fpga PRIVATE ${AOCL_UTILITY_INC})
    set_target_properties(hog-ocl-fpga PROPERTIES LINK_FLAGS "-Wl,--no-as-needed")

    # Figure out the list of valid OpenCL boards. This is an awful hack, since
    # the AOCL SDK doesn't provide a way to get a machine-readable version of
    # this info.
    execute_process(COMMAND ${AOCL_SDK_ROOT}/bin/aoc --list-boards
        OUTPUT_VARIABLE AOC_BOARD_LIST)
    string(REPLACE "Board list:" "" AOC_BOARD_LIST "${AOC_BOARD_LIST}")
    string(REPLACE "\n" " " AOC_BOARD_LIST "${AOC_BOARD_LIST}")
    string(REGEX MATCHALL "([A-Za-z0-9_]+)" AOC_BOARD_LIST "${AOC_BOARD_LIST}")
    message(STATUS "Available boards: ${AOC_BOARD_LIST}")

    list(LENGTH "${AOC_BOARD_LIST}" NUM_BOARDS)
    if($NUM_BOARDS EQUAL 0)
        message(SEND_ERROR "${BRed}No available board BSPs found. FPGA support disabled.${ClrNone}")
        set(ENABLE_FPGA OFF)
    endif()

    # Set up the board options and throw an error if an invalid one is given
    list(GET "${AOC_BOARD_LIST}" 0 AOCL_DEFAULT_BOARD)
    set(AOC_BOARD "${AOCL_DEFAULT_BOARD}" CACHE STRING "Board to compile OpenCL code for")
    set_property(CACHE AOC_BOARD PROPERTY STRINGS ${AOC_BOARD_LIST})
    set(PACKAGE_DEPS "${PACKAGE_DEPS} hog-ocl-fpga")

    if(AOC_BOARD)
        add_custom_target(bitstream
            COMMAND aoc --board ${AOC_BOARD}
                ${CMAKE_CURRENT_SOURCE_DIR}/src/algorithms/pedestrian_detect.cl
                -o pedestrian_detect.aocx
            COMMENT "Generating Altera OpenCL bitstream. This can take several hours."
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            VERBATIM SOURCES
                ${CMAKE_CURRENT_SOURCE_DIR}/src/algorithms/pedestrian_detect.cl)
    else()
        add_custom_target(bitstream
            COMMAND aoc
                ${CMAKE_CURRENT_SOURCE_DIR}/src/algorithms/pedestrian_detect.cl
                -o pedestrian_detect.aocx
            COMMENT "Generating Altera OpenCL bitstream. This can take several hours."
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            VERBATIM SOURCES
                ${CMAKE_CURRENT_SOURCE_DIR}/src/algorithms/pedestrian_detect.cl)
    endif()
endif()

option(ENABLE_BRISK "Enable BRISK-based area matching algorithm")
if(${ENABLE_BRISK})
    find_package(OpenCV 3 REQUIRED core features2d calib3d)
    add_library(brisk-area-match MODULE src/algorithms/brisk_area_match.cpp)
    target_link_libraries(brisk-area-match ${OpenCV_LIBS})
    target_compile_features(brisk-area-match PRIVATE cxx_auto_type)
    set(PACKAGE_DEPS "${PACKAGE_DEPS} brisk-area-match")
endif()

# Make sure that video decoding works right
try_run(VTEST_RUN_OK VTEST_BUILD_OK
    ${CMAKE_CURRENT_BINARY_DIR}/vidtest ${CMAKE_CURRENT_SOURCE_DIR}/buildsys/video/vidtest.cpp
    CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${OpenCV_INCLUDE_DIRS}"
    LINK_LIBRARIES ${OpenCV_LIBS}
    ARGS
        ${CMAKE_CURRENT_SOURCE_DIR}/buildsys/video/bars.h265.avi
        ${CMAKE_CURRENT_SOURCE_DIR}/buildsys/video/bars.mjpeg.avi
        ${CMAKE_CURRENT_SOURCE_DIR}/buildsys/video/bars.theora.avi
        ${CMAKE_CURRENT_SOURCE_DIR}/buildsys/video/bars.vp8.avi
        ${CMAKE_CURRENT_SOURCE_DIR}/buildsys/video/bars.vp9.avi
        ${CMAKE_CURRENT_SOURCE_DIR}/buildsys/video/bars.wmv.avi
        ${CMAKE_CURRENT_SOURCE_DIR}/buildsys/video/bars.x264.avi)
if(NOT ${VTEST_BUILD_OK})
    message(SEND_ERROR "${BRed}Video format test failed to compile. OpenCV is not installed properly.${ClrNone}")
elseif(NOT ${VTEST_RUN_OK})
    message(WARNING "${Yellow}One or more test videos could not be read. This suggests
    that your OpenCV installation may not have video I/O capabilities. Camera
    input should be unaffected.

    If you want to use this binary with input from a video file, fix your OpenCV
    installation and recompile.${ClrNone}")
else()
    message(STATUS "${Green}Video codec test OK - OpenCV is good to go.${ClrNone}")
endif()

include_directories(${CMAKE_CURRENT_BINARY_DIR})
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/config.h)
add_executable(pddemo
    src/main.cpp

    src/results/metadump.cpp
    src/results/network.cpp
    src/results/http.cpp
    src/results/http_util.cpp

    src/algorithms/ocv.cpp
    src/algorithm.cpp

    src/ui/overlay.cpp
    src/ui/result_render.cpp
    src/ui/status.cpp

    src/media/capture.cpp
    src/media/sink.cpp)
target_compile_features(pddemo PRIVATE cxx_auto_type cxx_range_for)
target_link_libraries(pddemo ${OCV_APP_LIBS} ${Boost_LIBRARIES} ${CMAKE_DL_LIBS}
    Threads::Threads)
if(${GSTREAMER_FOUND})
    target_link_libraries(pddemo
        ${GSTREAMER_LIBRARIES} ${GLIB_LIBRARIES} ${GLIB_GOBJECT_LIBRARIES})
endif()

option(BUILD_PACKAGE "Generate a platform-independent output package")
if(${BUILD_PACKAGE})
    add_custom_command(pddemo.tar.gz
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/packaging/makepkg.sh
        DEPENDS ${PACKAGE_DEPS}
        COMMENT "Generating platform-independent package"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
else()
    add_custom_target(build-package
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/packaging/makepkg.sh ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDS ${PACKAGE_DEPS}
        COMMENT "Generating platform-independent package"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()
