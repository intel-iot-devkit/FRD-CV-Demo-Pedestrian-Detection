#include "brisk_area_match.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <boost/endian/conversion.hpp>
#include <stdint.h>
#include <algorithm>
#include <future>
#include <iostream>

using namespace ml;
using namespace ml::rsd;

#define RATIO 0.7
#define MIN_MATCHES 40

#define BIG_TO_HOST(x) boost::endian::big_to_native_inplace((x))

ObjectData::ObjectData(char* nm, std::vector<cv::Point2f> pts, cv::Mat descs) :
        name(nm), keypoints(pts), descriptors(descs),
        matcher(cv::BFMatcher::create(cv::NORM_HAMMING)) {
}

float ObjectData::match(std::vector<cv::Point2f>& q_points, cv::Mat q_desc) {
    std::vector<std::vector<cv::DMatch> > matches;
    matcher->knnMatch(q_desc, descriptors, matches, 2);

    // filter to matches meeting requirements
    matches.erase(std::remove_if(matches.begin(), matches.end(),
            [](std::vector<cv::DMatch>& m){
                return  (m.size() == 2) &&
                        (m[0].distance < m[1].distance * RATIO); }),
            matches.end());

    if(matches.size() < MIN_MATCHES) return -1.0;

    // build homography between objects
    std::vector<cv::Point2f> pts, pts_query;
    for(auto m : matches) {
        pts.push_back(keypoints[m[0].trainIdx]);
        pts_query.push_back(q_points[m[0].queryIdx]);
    }

    cv::Mat mask;
    cv::findHomography(pts_query, pts, cv::RANSAC, 4, mask);

    // get the ratio of matched keypts to the total keypoints
    float sum = 0;
    int cols = mask.cols, rows = mask.rows;

    if(mask.isContinuous()) {
        // improve performance on contiguous matrices
        cols *= rows;
        rows = 1;
    }
    for(int i = 0;i < rows;i++) {
        const uint8_t* p = mask.ptr<uint8_t>(i);
        for(int j = 0;j < cols;j++) sum += p[j];
    }

    return sum / mask.total();
}

BRISKRegionMatchAlgorithm::BRISKRegionMatchAlgorithm(const cv::Size& size) {
    // build image database
    read_database("books.db");
}

BRISKRegionMatchAlgorithm::~BRISKRegionMatchAlgorithm() {
    for(auto p : m_db_ptrs) free(p);
}

void BRISKRegionMatchAlgorithm::read_database(fs::path loc) {
    m_database.clear();

    FILE* f = fopen(loc.c_str(), "rb");
    if(f == NULL)
        throw ml::algorithm_init_error("Cannot load BRISK database",
                "No such file or directory");

    // read entries
    while(!feof(f)) {
        // read title
        uint16_t title_len;
        if(fread(&title_len, sizeof(uint16_t), 1, f) != 1) break;
        BIG_TO_HOST(title_len);

        char title[title_len+1];
        if(fread(title, 1, title_len, f) != title_len)
            throw ml::algorithm_init_error("Cannot load BRISK database",
                    "Unexpected read failure: 1");
        title[title_len] = 0;

        // read vector of points
        std::vector<cv::Point2f> points;
        uint16_t npoints;
        if(fread(&npoints, sizeof(uint16_t), 1, f) != 1)
            throw ml::algorithm_init_error("Cannot load BRISK database",
                    "Unexpected read failure: 2");
        BIG_TO_HOST(npoints);

        for(int i = 0;i < npoints;i++) {
            float pt[2];
            if(fread(pt, sizeof(float), 2, f) != 2)
                throw ml::algorithm_init_error("Cannot load BRISK database",
                        "Unexpected read failure: 3");
            uint32_t* temp = (uint32_t*)(&pt[0]);
            BIG_TO_HOST(*temp);
            temp = (uint32_t*)(&pt[1]);
            BIG_TO_HOST(*temp);
            points.push_back(cv::Point2f(pt[0], pt[1]));
        }

        // read descriptor matrix (stored row-major)
        uint16_t ndescs;
        if(fread(&ndescs, sizeof(uint16_t), 1, f) != 1)
            throw ml::algorithm_init_error("Cannot load BRISK database",
                    "Unexpected read failure: 4");
        BIG_TO_HOST(ndescs);

        void *descbuf = malloc(((size_t)npoints)*((size_t)ndescs));
        if(fread(descbuf, npoints*ndescs, 1, f) != 1)
            throw ml::algorithm_init_error("Cannot load BRISK database",
                    "Unexpected read failure: 5");
        m_db_ptrs.push_back(descbuf);

        // translate it to an OpenCV matrix - this is zero-ish cost
        cv::Mat descriptors(npoints, ndescs, CV_8UC1, descbuf);

        // store the entry
        m_database.push_back(ObjectData(title, points, descriptors));
        printf("Read '%s'\n", title);
    }
}

Algorithm::Info BRISKRegionMatchAlgorithm::getInfo() {
    return m_info;
}

const std::vector<AlgorithmResult*>& BRISKRegionMatchAlgorithm::analyze(const cv::Mat& mat) {
    cv::Mat gray;
    cv::cvtColor(mat, gray, cv::COLOR_BGR2GRAY);
    cv::imwrite("out.png", gray);
    std::vector<cv::Point2f> keypts;
    cv::Mat descriptors;
    {
        std::vector<cv::KeyPoint> _keypts;
        auto brisk = cv::BRISK::create(30, 2, 1.0f);
        brisk->detect(gray, _keypts);
        brisk->compute(gray, _keypts, descriptors);
        cv::drawKeypoints(gray, _keypts, gray);
        cv::imwrite("updated.png", gray);

        for(auto p : _keypts) keypts.push_back(p.pt);
    }

    for(auto obj : m_database) {
        float score = obj.match(keypts, descriptors);
        printf("%s -> %.5f\n", obj.name.c_str(), score);
    }

    return m_results;
}

extern "C" int count() {
    return 1;
}

extern "C" Algorithm* build(int idx, const cv::Size& sz) {
    return new BRISKRegionMatchAlgorithm(sz);
}

extern "C" Algorithm::Info* describe(int idx) {
    return new Algorithm::Info(
            "BRISK-based Region Matcher", "brisk-area-match",
            "Match images to a database using the BRISK algorithm",
            0, false, false);
}

extern "C" void interface_version(int* major, int* minor) {
    *major = IFACE_VERSION_MAJOR;
    *minor = IFACE_VERSION_MINOR;
}

