#ifndef ALGORITHM_BRISK_REGIONS_HPP
#define ALGORITHM_BRISK_REGIONS_HPP

#include "opencv2/core/core.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "../algorithm.hpp"

#include <vector>

namespace ml {
namespace rsd {

struct ObjectData {
    std::string name;
    std::vector<cv::Point2f> keypoints;
    cv::Mat descriptors;

    ObjectData(char* nm, std::vector<cv::Point2f> pts, cv::Mat descs);

    cv::Ptr<cv::DescriptorMatcher> matcher;
    float match(std::vector<cv::Point2f>& points, cv::Mat descriptors);
};

class BRISKRegionMatchAlgorithm : public Algorithm {
private:
    Algorithm::Info m_info = Algorithm::Info(
            "BRISK-based Region Matcher", "brisk-area-match",
            "Match images to a database using the BRISK algorithm",
            0, false, false);

public:
    BRISKRegionMatchAlgorithm(const cv::Size& size);
    ~BRISKRegionMatchAlgorithm();

    Info getInfo();
    const std::vector<AlgorithmResult*>& analyze(const cv::Mat& mat);

private:
    void read_database(fs::path location);

    std::vector<ObjectData> m_database;
    std::vector<void*> m_db_ptrs;
};

extern "C" int count();

extern "C" Algorithm* build(int idx, const cv::Size& sz);

extern "C" Algorithm::Info* describe(int idx);

extern "C" void interface_version(int* major, int* minor);
};
};

#endif


