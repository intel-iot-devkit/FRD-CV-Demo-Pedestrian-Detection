Computer Vision Demo Framework
==============================

Dependencies
------------
To build the core application, the following libraries are required:

* OpenCV version 3.1 or later (must build from source on Ubuntu 16.04)
* OpenCV extra modules 3.1 or later (must include during OpenCV build process)
* Boost version 1.58 or later (`libboost-all-dev` on Ubuntu 16.04)
* CMake version 3.2 or later

For video output support, GStreamer and associated plugins must also be
installed before compiling CMake. On Ubuntu 16.04, the following packages
should be installed:

* `ffmpeg`
* `gstreamer1.0-plugins-base`
* `gstreamer1.0-plugins-good`
* `gstreamer1.0-plugins-ugly`
* `gstreamer1.0-libav`
* `libavcodec-dev`
* `libavcodec-ffmpeg56`
* `libavdevice-ffmpeg56`
* `libavfilter-ffmpeg5`
* `libavformat-dev`
* `libavformat-ffmpeg56`
* `libavresample-dev`
* `libavresample-ffmpeg56`
* `libavutil-dev`
* `libavutil-ffmpeg56`

For FPGA-accelerated algoritms, the Altera SDK should be set up and accessible
to CMake. The build scripts will automatically test the SDK installation before
trying to use it, and display errors if it finds that any component is not set
correctly.

Installation
------------
To build the application, open a new shell in the source directory and run the
following commands:

    mkdir build && cd build
    cmake .. && make -j 8

Once the build process finishes, you should have a compiled version of the core
binary, `pddemo`, and any enabled libraries. If you're building the FPGA-based
pedestrian detection demo and didn't get a copy of the bitstream from somewhere
else, you should also build that by running `make aoc`. This does not run
automatically because bitstream generation can take 5-7 hours on a fast machine.

Running
-------
First, run `./pddemo --list-algos` to display a list of available algorithms. If
the one you want to show is not on this list, then something went wrong during
another part of the build process, or required system libraries aren't properly
installed.

Once you know the algorithm you want to run, use the `-a [algorithm]` flag to
specify it. If no `-a` flag is given, the demo will default to using OpenCV's
CPU-based pedestrian detection algorithm. To give a file or capture device as
input, use `./pddemo [flags] file.avi` or `./pddemo [flags] camera`
respectively. There is currently no supported syntax for using a video capture
device other than the default.

Once the application is running, a window should pop up showing the current
frame and the active algorithm's results. If no `DISPLAY` variable is present in
the process's environment or if the `-w` flag is given, it will not atttempt to
generate a window.
