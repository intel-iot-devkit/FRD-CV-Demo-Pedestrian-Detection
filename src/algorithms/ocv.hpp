#ifndef ALGORITHM_OCV_HPP
#define ALGORITHM_OCV_HPP

#include "opencv2/core/core.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/tracking.hpp"
#include "../algorithm.hpp"

#include <vector>
#include <list>

namespace ml {
namespace ocv {

struct TrackingInfo {
    cv::Ptr<cv::Tracker> tracker;
    cv::Rect last_pos;
    unsigned int id;
    unsigned int confirm_frames;
};

class OCVAlgorithm : public Algorithm {
public:
    OCVAlgorithm();
    ~OCVAlgorithm();

    // virtual implementations
    Info getInfo();
    const std::vector<AlgorithmResult*>& analyze(const cv::Mat& mat);

private:
    cv::HOGDescriptor m_hog;
    std::list<TrackingInfo> m_track;
    std::vector<cv::Rect> m_locs;
};

int count(void); //!< Return how many algorithms this module contains
Algorithm* build(int idx, const cv::Size& sz); //!< Build a given algorithm
Algorithm::Info* describe(int idx); //!< Describe a given algorithm

};
};

#endif
