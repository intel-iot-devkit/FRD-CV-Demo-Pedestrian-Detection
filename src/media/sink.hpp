#ifndef SINK_HPP
#define SINK_HPP

#include <string>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#ifdef WITH_GSTREAMER
#include "gst/gst.h"
#include "glib.h"
#endif

namespace vio {

class VideoSink {
public:
    /** \brief Insert a video frame
     *
     * Insert a frame into the video sink. If closed, this should either do
     * nothing or throw an exception. The input matrix should be in RGB format.
     */
    virtual VideoSink& operator<<(const cv::Mat& strm)=0;

    /** \brief Close the sink
     *
     * Closes the sink device. Any further input frames should result in either
     * an exception or no effect.
     */
    virtual void close()=0;
};

/** \brief Sink that discards all input frames
 */
class NullSink : public VideoSink {
public:
    VideoSink& operator<<(const cv::Mat& strm);
    void close();
};

/** \brief Sink that emits frames to a HighGUI window
 */
class HighGUISink : public VideoSink {
public:
    HighGUISink(std::string windowName);
    ~HighGUISink();

    VideoSink& operator<<(const cv::Mat& strm);
    void close();

private:
    std::string m_name;
    bool m_closed;
};

/** \brief Sink that emits frames to a video file
 */
class FileSink : public VideoSink {
public:
    enum codec { MJPEG, MPEG4, NONE };

public:
    FileSink(std::string fname, codec c, int fps=30);
    ~FileSink();

    VideoSink& operator<<(const cv::Mat& strm);
    void close();

private:
    bool m_pending; /**< Whether the stream has already written some frames */
    cv::VideoWriter* m_writer;

    // creation options
    std::string mc_fname;
    int mc_fourcc;
    int mc_fps;
};

/** \brief Sink that distributes input frames to zero or more subsinks
 */
class FanoutSink : public VideoSink {
public:
    FanoutSink();
    ~FanoutSink();

    VideoSink& operator<<(const cv::Mat& strm);
    void close();

    /** \brief Add a sink to the subsink list
     *
     * The FanoutSink will take ownership of the passed sink object.
     */
    void addSink(VideoSink* sink);

    //! Return whether the fanout sink is empty (i.e. no sinks are present)
    bool empty() const;

private:
    bool m_closed;
    std::vector<VideoSink*> m_sinks;
};

#ifdef WITH_GSTREAMER
/** \brief Sink that injects input frames into a GStreamer pipeline
 */
class GStreamerSink : public FileSink {
public:
    GStreamerSink(std::string pipeline, int fps=30);
    ~GStreamerSink();
};

#endif

};

#endif
