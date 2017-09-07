#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

int main(int argc, char **argv) {
    for(int i = 1;i < argc;i++) {
        cv::VideoCapture cap(argv[i]);
        if(!cap.isOpened()) return -1;
        if(!cap.grab()) return -1;
    }
    return 0;
}
