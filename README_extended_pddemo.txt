PEDESTRIAN DETECTION DEMO
=========================

***Wherever /user_name/ is used in a command, this is your individual system user name so make sure to change that when running the lines given***

Dependencies
============

To build the core application, the following libraries are required:
----------------------------------------------------------------------------------
$ sudo apt install build-essential cmake git pkg-config libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev libv4l-dev libav-tools libavswscale-dev libxvidcore-dev libx264-dev
$ sudo apt install libgtk2.0.dev libatlas-base-dev gfortran python3.5-dev python3-pip
$ sudo -H pip3 install numpy
$ sudo apt-get install ffmpeg
$ sudo apt-get install libboost-all-dev
$ sudo apt-get install gstreamer1.0-plugins-base
$ sudo apt-get install gstreamer1.0-plugins-good
$ sudo apt-get install gstreamer1.0-plugins-ugly
$ sudo apt-get install gstreamer1.0-libav
$ sudo apt-get install gstreamer1.0
$ sudo apt-get install libgstreamer1.0-dev
$ sudo apt-get install libgstreamer-plugins-base1.0-dev
$ sudo apt-get install libavcodec-dev
$ sudo apt-get install libavcodec-ffmpeg56
$ sudo apt-get install libavdevice-ffmpeg56
$ sudo apt-get install libavfilter-ffmpeg5
$ sudo apt-get install libavformat-dev
$ sudo apt-get install libavformat-ffmpeg56
$ sudo apt-get install libavresample-dev
$ sudo apt-get install libavresample-ffmpeg2
$ sudo apt-get install libavutil-dev
$ sudo apt-get install libavutil-ffmpeg54
----------------------------------------------------------------------------------

Installing OpenCV
=================

OpenCV version 3.1 download (https://github.com/opencv/opencv/releases/tag/3.1.0) - Download the tar.gz source code 

This will send a tar.gz file to the /home/user_name/Downloads directory. Use the following command to extract the code from the file:
------------------------------
$ cd /home/user_name/Downloads
$ tar -xvf opencv-3.1.0.tar.gz
------------------------------
This will decompress the tar.gz into a directory with the same name as the download.

The OpenCV extra modules are also required (3.1 or later) so download (the green button on the webpage) the .zip file at https://github.com/opencv/opencv_contrib/tree/3.1.0 . Unzip the file and move it into the OpenCV directory that was created in the previous code steps:
----------------------------------------------------------------
$ cd /home/user_name/Downloads
$ unzip opencv_contrib-3.1.0.zip
$ mv opencv_contrib-3.1.0 /home/user_name/Downloads/opencv-3.1.0
----------------------------------------------------------------

The following steps will go through the build process. Take note that if you have previously build OpenCV for another application you will have to rebuild it with the correct 'cmake' line below. To remove and remake the build directory use:
-------------------------------------------
$ cd /home/user_name/Downloads/opencv-3.1.0
$ rm -rf build
-------------------------------------------

Next, you have to build the OpenCV application with the following commands:
----------------------------------------------------------------------------------
$ cd /home/user_name/Downloads/opencv-3.1.0
$ mkdir build && cd build
$ cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local -DWITH_V4L=ON -DWITH_FFMPEG=ON -DOPENCV_EXTRA_MODULES_PATH=/home/user_name/Downloads/opencv-3.1.0/opencv_contrib-3.1.0/modules/ ..
$ make -j8 
$ sudo make install
----------------------------------------------------------------------------------
OpenCV build is now complete.


For FPGA-accelerated algoritms, the Altera SDK should be set up and accessible to CMake. The build scripts will automatically test the SDK installation before trying to use it, and display errors if it finds that any component is not set correctly.

This requires the download and installation of Intel SDK for OpenCL, version 16.0 http://dl.altera.com/opencl/16.0/?edition=pro&download_manager=dlm3 . The download results in a .tar file. To extract the decompressed information use:
------------------------------
$ cd /home/user_name/Downloads
$ tar -xvf opencv-3.1.0.tar.gz
------------------------------
Then follow the "Download and install instructions" from the same webpage. 


**If you are not using an FPGA, it is not necessary to install the SDK file.** 


Installation
------------
To build the application, open the source directory and run the
following commands:
----------------------------------
$ cd /home/user_name/sto-il-pddemo
$ mkdir build && cd build
$ cmake ..
$ make -j8
----------------------------------
***The source directory might not be in that path given, it will depend on where you cloned the repository.

Once the build process finishes, you should have a compiled version of the core
binary, `pddemo`, and any enabled libraries. If you're building the FPGA-based
pedestrian detection demo and didn't get a copy of the bitstream from somewhere
else, you should also build that by running `make aoc`. This does not run
automatically because bitstream generation can take 5-7 hours on a fast machine.

Running
-------
First, run:
----------------------------------------
$ cd /home/user_name/sto-il-pddemo/build
$ ./pddemo --list-algos
----------------------------------------
to display a list of available algorithms. There will be two possible algorithms, one for running the demo on only the CPU and the other for using an FPGA. If the one you want to show is not on this list, then something went wrong during another part of the build process, or required system libraries aren't properly installed.

Once you know the algorithm you want to run, use the `-a [algorithm]` flag to
specify it. If no `-a` flag is given, the demo will default to using OpenCV's
CPU-based pedestrian detection algorithm. To give a file or capture device as
input, use `./pddemo [flags] file.avi` or `./pddemo [flags] camera`
respectively. 

To get the possible flags, run the help command:
----------------------------------------
$ cd /home/user_name/sto-il-pddemo/build 
$ ./pddemo --help 
---------------------------------------- 

The general command line to run this demo using a .avi file is:
----------------------------------------------
$ cd /home/user_name/sto-il-pddemo/build 
$ ./pddemo [flags] /location_of_video/file.avi
----------------------------------------------

Once the application is running, a window should pop up showing the current
frame and the active algorithm's results. If no `DISPLAY` variable is present in
the process's environment or if the `-w` flag is given, it will not atttempt to
generate a window.

To terminate the demo from running, go back to the terminal and use 'Ctrl-C'.


