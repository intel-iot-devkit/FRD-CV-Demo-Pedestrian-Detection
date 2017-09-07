#include "capture.hpp"

#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>

using namespace vio;

FileCaptureBackend::FileCaptureBackend(
        const std::string& fname, bool looped) : m_loop(looped), m_end(false) {
    m_fname = fname;
    m_cap = new cv::VideoCapture(fname.c_str());

    // make sure the capture target is accessible
    if(!m_cap->isOpened()) {
        throw std::invalid_argument("Failed to open video file");
    }
    if(!m_cap->grab()) {
        throw std::invalid_argument("Failed to capture frame from file");
    }
}

FileCaptureBackend::~FileCaptureBackend() {
    delete m_cap;
}

int FileCaptureBackend::getFrame(cv::Mat& out) {
    if(m_end) return 0;

    m_cap->retrieve(out);
    if(!m_cap->grab()) {
        if(m_loop) restart();
        else m_end = true;
    }
    return 1;
}

cv::Size FileCaptureBackend::getSize() {
    cv::Mat m;
    m_cap->retrieve(m);
    return cv::Size(m.cols, m.rows);
}

void FileCaptureBackend::restart() {
    delete m_cap;
    m_cap = new cv::VideoCapture(m_fname.c_str());

    if(!m_cap->isOpened() || !m_cap->grab()) {
        fprintf(stderr, "Error: Cannot reset video stream\n");
        m_end = true;
    }
}

CameraCaptureBackend::CameraCaptureBackend(int index) : m_end(false) {
    m_cap = new cv::VideoCapture(index);
    m_index = index;

    // make sure the capture target is accessible
    if(!m_cap->isOpened()) {
        throw std::invalid_argument("Failed to open video device");
    }
    if(!m_cap->grab()) {
        throw std::invalid_argument("Failed to capture frame from device");
    }
}

CameraCaptureBackend::~CameraCaptureBackend() {
    delete m_cap;
}

int CameraCaptureBackend::getFrame(cv::Mat& out) {
    if(m_end) return 0;

    m_cap->retrieve(out);
    if(!m_cap->grab()) {
        m_end = true;
    }
    return 1;
}

cv::Size CameraCaptureBackend::getSize() {
    cv::Mat m;
    m_cap->retrieve(m);
    return cv::Size(m.cols, m.rows);
}

void CameraCaptureBackend::restart() {
    delete m_cap;
    m_cap = new cv::VideoCapture(m_index);

    if(!m_cap->isOpened() || !m_cap->grab()) {
        fprintf(stderr, "Error: Cannot reset video stream\n");
        m_end = true;
    }
}

ImageCaptureBackend::ImageCaptureBackend(const std::string& fname) {
    m_img = cv::imread(fname);
}

ImageCaptureBackend::~ImageCaptureBackend() {
}

int ImageCaptureBackend::getFrame(cv::Mat& out) {
    m_img.copyTo(out);
    return 1;
}

cv::Size ImageCaptureBackend::getSize() {
    return cv::Size(m_img.rows, m_img.cols);
}

void ImageCaptureBackend::restart() { }

CaptureBackend* vio::openBackend(std::string spec, bool infinite=false) {
    // try to parse the spec
    size_t scheme_idx = spec.find(':');
    if(scheme_idx == std::string::npos) {
        // default to file capture
        return new FileCaptureBackend(spec, infinite);
    }

    std::string scheme(spec, 0, scheme_idx);
    std::string rest(spec, scheme_idx+1);

    if(scheme.compare("cam") == 0) {
        char* end;
        unsigned int n = strtoul(rest.c_str(), &end, 10);
        if(*end != 0) throw std::invalid_argument("Invalid camera index");
        return new CameraCaptureBackend(n);
    } else if(scheme.compare("file") == 0) {
        return new FileCaptureBackend(rest, infinite);
    } else {
        throw std::invalid_argument("No such capture type");
    }
}
