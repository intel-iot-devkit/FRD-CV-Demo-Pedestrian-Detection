#include "ocv.hpp"
#include <vector>

#define CONF_LIMIT 20
#define INTERSECT_THRESHOLD 0.5

using namespace ml::ocv;
using namespace cv;

OCVAlgorithm::OCVAlgorithm() {
    m_hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

    m_results.push_back(new BoundingBoxesResult());
}

OCVAlgorithm::~OCVAlgorithm() {
}

ml::Algorithm::Info OCVAlgorithm::getInfo() {
    ml::Algorithm::Info info(
        "OpenCV HOG SVM", "ocv-hog-svm",
        "OpenCV's basic HOG SVM recognizer",
        0,
        true,  // tracks
        false  // fpga
    );
    return info;
}

const std::vector<ml::AlgorithmResult*>& OCVAlgorithm::analyze(const Mat& img) {
    BoundingBoxesResult& res = *dynamic_cast<BoundingBoxesResult*>(m_results[0]);
    res.boxes.clear();
    m_locs.clear();

    // update all trackers
    for(std::list<TrackingInfo>::iterator i = m_track.begin();
            i != m_track.end();) {
        Rect2d r;
        if(!i->tracker->update(img, r)) {
            i = m_track.erase(i);
            continue;
        }
        i->last_pos = r;
        i->confirm_frames++;
        i++;
    }

    // pairwise rectangle intersection/deduplication
    {
        std::list<TrackingInfo>::iterator i,j;
        for(i = m_track.begin();i != m_track.end();i++) {
            j = i;
            j++;
            for(;j != m_track.end();) {
                Rect ri = i->last_pos;
                Rect rj = j->last_pos;
                Rect join = ri & rj;
                if(join.area() == 0) {
                    j++;
                    continue;
                }

                if((join == rj) ||
                        (join.area() >= INTERSECT_THRESHOLD*rj.area())) {
                    // eliminate the second
                    j = m_track.erase(j);
                    continue;
                }
                j++;
            }
        }
    }

    m_hog.detectMultiScale(img, m_locs,
            0.5,        // hitThreshold
            Size(8,8),  // winStride
            Size(32,32),// padding
            pow(img.rows / 128, 1.0 / 24),     // scale
            2);         // finalThreshold

    // isolate bounds that already have detected people
    for(auto i : m_track) {
        for(int j = 0;j < m_locs.size();j++) {
            Rect r_t = i.last_pos;
            Rect r_d = m_locs[j];
            if(r_t.contains(r_d.tl()) && r_t.contains(r_d.br())) {
                // readjust tracking rectangle
                i.confirm_frames = 0;
                i.last_pos = r_t;
                i.tracker->init(img, r_t);
                m_locs.erase(m_locs.begin()+j);
                j--; // revisit the same index next time around
                continue;
            }
            Rect isect = r_t & r_d;
            if((isect.area() >= INTERSECT_THRESHOLD*r_t.area()) ||
                    (isect.area() >= INTERSECT_THRESHOLD*r_d.area())) {
                // they intersect - update the confirm count
                i.confirm_frames = 0;
                i.last_pos |= r_d;
                m_locs.erase(m_locs.begin()+j);
                j--; // revisit the same index next time around
            }
        }
    }

    // delete old trackers
    m_track.remove_if(
            [](TrackingInfo i) { return i.confirm_frames > CONF_LIMIT; });

    // create new tracking bounds for others
    for(auto r : m_locs) {
        TrackingInfo inf;
        inf.tracker = Tracker::create("TLD");
        inf.last_pos = r;
        inf.id = rand();
        inf.confirm_frames = 0;
        inf.tracker->init(img, r);
        m_track.push_back(inf);
    }

    // construct results
    for(auto t : m_track) {
        BoundingBox b;
        b.id = t.id;
        b.tag = 0;
        b.bounds = t.last_pos;
        res.boxes.push_back(b);
    }

    return m_results;
}

int ml::ocv::count(void) {
    return 1;
}

ml::Algorithm* ml::ocv::build(int idx, const Size& sz) {
    return new OCVAlgorithm();
}

ml::Algorithm::Info* ml::ocv::describe(int idx) {
    return new ml::Algorithm::Info(
        "OpenCV HOG SVM", "ocv-hog-svm",
        "OpenCV's basic HOG SVM recognizer",
        0,     // index
        true,  // tracks
        false  // fpga
    );
}
