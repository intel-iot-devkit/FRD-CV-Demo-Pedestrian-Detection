#ifndef CAPTURE_HPP
#define CAPTURE_HPP

#include <string>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

namespace vio {

class CaptureBackend {
public:
    /** \brief Get the next frame of input.
     *
     * \return Whether more input was available
     */
    virtual int getFrame(cv::Mat& out)=0;

    /** \brief Get the size of this backend's frames.
     *
     * \return The size of a frame
     */
    virtual cv::Size getSize()=0;

    /** \brief Restart the video capture
     */
    virtual void restart()=0;
};

class FileCaptureBackend : public CaptureBackend {
public:
    FileCaptureBackend(const std::string& fname, bool looped=false);
    ~FileCaptureBackend();

    int getFrame(cv::Mat& out);
    cv::Size getSize();
    void restart();

private:
    bool m_loop, m_end;
    cv::VideoCapture *m_cap;
    std::string m_fname;
};

class CameraCaptureBackend : public CaptureBackend {
public:
    CameraCaptureBackend(int index=0);
    ~CameraCaptureBackend();

    int getFrame(cv::Mat& out);
    cv::Size getSize();
    void restart();

private:
    cv::VideoCapture *m_cap;
    bool m_end;
    int m_index;
};

class ImageCaptureBackend : public CaptureBackend {
public:
    ImageCaptureBackend(const std::string& fname);
    ~ImageCaptureBackend();

    int getFrame(cv::Mat& out);
    cv::Size getSize();
    void restart();

private:
    cv::Mat m_img;
};

CaptureBackend* openBackend(std::string spec, bool infinite);
};

#endif
