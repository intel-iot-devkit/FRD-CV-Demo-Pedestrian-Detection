#include "sink.hpp"
#include <stdexcept>

using namespace vio;

VideoSink& NullSink::operator<<(const cv::Mat& strm) { return *this; }
void NullSink::close() {}

HighGUISink::HighGUISink(std::string name) : m_name(name), m_closed(false) {
    cv::namedWindow(name.c_str(), 1);
}

HighGUISink::~HighGUISink() {
    if(!m_closed) cv::destroyWindow(m_name.c_str());
}

VideoSink& HighGUISink::operator<<(const cv::Mat& strm) {
    if(!m_closed) {
        imshow(m_name.c_str(), strm);
        cv::waitKey(1);
    }

    return *this;
}

void HighGUISink::close() {
    if(m_closed) return;

    m_closed = true;
    cv::destroyWindow(m_name.c_str());
}

FileSink::FileSink(std::string fname, codec cdc, int fps) : m_pending(false),
        mc_fps(fps) {
    m_writer = new cv::VideoWriter();
    mc_fname = fname;

    switch(cdc) {
    case MJPEG: mc_fourcc = CV_FOURCC('M', 'J', 'P', 'G'); break;
    case MPEG4: mc_fourcc = CV_FOURCC('M', 'P', 'G', '4'); break;
    case NONE:  mc_fourcc = 0; break;
    default:
        throw std::invalid_argument("Unknown video codec");
    }
}

FileSink::~FileSink() {
    if(m_writer) delete m_writer;
}

VideoSink& FileSink::operator<<(const cv::Mat& strm) {
    if(!m_writer) return *this;

    if(!m_pending) {
        if(!m_writer->open(mc_fname, mc_fourcc, mc_fps,
                    cv::Size(strm.cols, strm.rows)))
            throw std::runtime_error("Failed to create video writer");
        m_pending = true;
    }

    *m_writer << strm;
    return *this;
}

void FileSink::close() {
    m_writer->release();
    delete m_writer;
    m_writer = NULL;
}

FanoutSink::FanoutSink() : m_closed(false) {
}

FanoutSink::~FanoutSink() {
    for(auto e : m_sinks) {
        delete e;
    }
}

VideoSink& FanoutSink::operator<<(const cv::Mat& strm) {
    if(m_closed) return *this;

    for(auto e : m_sinks) {
        *e << strm;
    }
}

void FanoutSink::close() {
    if(m_closed) return;

    for(auto e : m_sinks) e->close();
    m_closed = true;
}

void FanoutSink::addSink(VideoSink* sink) {
    m_sinks.push_back(sink);
}

bool FanoutSink::empty() const {
    return m_sinks.empty();
}

#ifdef WITH_GSTREAMER
GStreamerSink::GStreamerSink(std::string pl, int fps) : FileSink(pl, NONE, fps){
}

GStreamerSink::~GStreamerSink() {
}

#endif
