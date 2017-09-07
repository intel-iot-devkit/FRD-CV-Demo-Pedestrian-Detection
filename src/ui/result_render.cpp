#include "overlay.hpp"
#include "../algorithm.hpp"
#include <math.h>
#include <unordered_map>

using namespace ui;

/* This file implements result rendering for all algorithm result types, and
 * handles dispatching results to their appropriate renderer.
 */

/**\brief Base class for result renderers */
class ResultRenderer {
public:
    //! Notify the renderer that the system is advancing to the next frame
    virtual void nextFrame()=0;

    //! Load a result into the renderer
    virtual void accept(const ml::AlgorithmResult* result)=0;

    //! Render all accepted results onto the passed frame
    virtual void render(cv::Mat& tgt)=0;
};

//! Renderer for `RT_BOUNDING_BOX` algorithms
class BBResultRenderer : public ResultRenderer {
public:
    BBResultRenderer() {
    }

    ~BBResultRenderer() {
    }

    void nextFrame() {
        m_rects.clear();
    }

    void accept(const ml::AlgorithmResult* result) {
        auto res = static_cast<const ml::BoundingBoxesResult*>(result);

        for(auto r : res->boxes) m_rects.push_back(r.bounds);
    }

    void render(cv::Mat& tgt) {
        for(auto r : m_rects)
            rectangle(tgt, r.tl(), r.br(), cv::Scalar(0,255,0), 3);
    }

private:
    std::vector<cv::Rect> m_rects;
};

//! Internal rendering state class
struct RenderElemState {
public:
    RenderElemState() {
        renderers[ml::RT_BOUNDING_BOXES] = new BBResultRenderer();
    }

    ~RenderElemState() {
    }

public:
    std::unordered_map<int, ResultRenderer*> renderers;
};

ResultRenderElement::ResultRenderElement() {
    m_render_internal = new RenderElemState();
}

ResultRenderElement::~ResultRenderElement() {
    RenderElemState* state = static_cast<RenderElemState*>(m_render_internal);
    for(auto e : state->renderers) delete e.second;
    delete state;
}

void ResultRenderElement::render(cv::Mat& tgt) {
    RenderElemState* state = static_cast<RenderElemState*>(m_render_internal);
    for(auto e : state->renderers) {
        e.second->render(tgt);
    }
}

void ResultRenderElement::setResults(const std::vector<ml::AlgorithmResult*>& res) {
    RenderElemState* state = static_cast<RenderElemState*>(m_render_internal);
    for(auto e : state->renderers) e.second->nextFrame();
    for(auto r : res) state->renderers[(int)r->type]->accept(r);
}
