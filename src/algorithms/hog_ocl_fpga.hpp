#ifndef ALGORITHM_HOG_FPGA_HPP
#define ALGORITHM_HOG_FPGA_HPP

#include "opencv2/core/core.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/tracking.hpp"
#include "AOCLUtils/aocl_utils.h"
#include "../algorithm.hpp"

#include <vector>

#define LEVELS 5

namespace ml {
namespace altera {

struct TrackingInfo {
    cv::Ptr<cv::Tracker> tracker;
    cv::Rect last_pos;
    unsigned int id;
    unsigned int confirm_frames;
};

class AlteraHOGAlgorithm : public Algorithm {
private:
    cl_context ctx;
    cl_command_queue q0, q1, q2, q3, q4;
    cl_program pgm;
    cl_kernel k_svm, k_resize, k_gradient, k_histogram, k_norm;

    // original image data buffer
    char* d_imgBuffer;

    // temporary buffers
    cl_mem d_originalData;
    cl_mem d_inData[LEVELS], d_outData[LEVELS];

    // output buffers
    int* h_results[LEVELS];

    BoundingBoxesResult* m_res;
    std::list<TrackingInfo> m_track;

    Algorithm::Info m_info = Algorithm::Info(
            "OpenCL FPGA-based HOG SVM", "hog-ocl-fpga",
            "Altera's HOG SVM classifier running on an FPGA via OpenCL",
            0, false, true);

    //! Raise an appropriate exception if the given OCL op failed
    void check_ocl_rc(cl_int stat, const char* op);

    //! Raise an appropriate exception if the given OCL op failed
    void check_ocl_rc_run(cl_int stat, const char* op);

public:
    AlteraHOGAlgorithm(const cv::Size& size);
    ~AlteraHOGAlgorithm();

    Info getInfo();
    const std::vector<AlgorithmResult*>& analyze(const cv::Mat& mat);
};

extern "C" int count();

extern "C" Algorithm* build(int idx, const cv::Size& sz);

extern "C" Algorithm::Info* describe(int idx);

extern "C" void interface_version(int* major, int* minor);
};
};

#endif

