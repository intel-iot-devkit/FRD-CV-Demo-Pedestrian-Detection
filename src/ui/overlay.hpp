#ifndef OVERLAY_HPP
#define OVERLAY_HPP

#include <string>
#include <vector>
#include "status.hpp"
#include "opencv2/imgproc/imgproc.hpp"

// avoid including all of algorithm.hpp
namespace ml {
    struct AlgorithmResult;
};

namespace ui {

/**\brief Base class for UI elements capable of drawing themselves visually
 */
class UIElement {
public:
    virtual void render(cv::Mat& tgt) = 0;
};

/**\brief Base class for UI elements that are anchored at a point
 */
class AnchoredElement : public UIElement {
public:
    AnchoredElement(cv::Point2f pos);

    virtual void setPos(cv::Point2f pos);
    virtual cv::Point2f getPos();

protected:
    cv::Point2f m_anchor;
};

/**\brief UI element composed of formatted text
 */
class TextUIElement : public AnchoredElement {
public:
    // the location of the origin point relative to the center of the text
    enum alignment {
        CENTER, CENTER_SOUTH, CENTER_NORTH, CENTER_EAST, CENTER_WEST,
        NORTH_EAST, NORTH_WEST, SOUTH_EAST, SOUTH_WEST
    };

public:
    TextUIElement(std::string fmt, TUIManager* mgr, cv::Point2f anchor,
            cv::Scalar color, alignment align=alignment::CENTER_SOUTH,
            double size=1);
    ~TextUIElement();

    void render(cv::Mat& tgt);

private:
    StatusLine m_line;
    alignment m_align;
    double m_fontSize;
    cv::Scalar m_color;
};

/**\brief Stacked bar UI element
 *
 * Displays a stacked bar with configurable colors and proportions for each part
 * of the bar. Can optionally autoscale to fit a configured width.
 */
class StackedBarElement : public AnchoredElement {
public:
    struct Section {
        float size;
        cv::Scalar color;
    };

public:
    StackedBarElement(cv::Point2f anchor, float height, float scale=0.25);
    ~StackedBarElement();

    void render(cv::Mat& tgt);

    /**\brief Change the fixed bar width, or remove the limit.
     *
     * \param width The new width, or "none" on -1.
     */
    void setWidth(float width=-1);

    /**\brief Change the field scale factor
     *
     * If the element is in fixed-width mode, calling this function will
     * change it back to variable-width mode.
     *
     * \param scale The new scale factor
     */
    void setScale(float scale);

    // section access operators
    Section& operator[](int idx);
    const Section& operator[](int idx) const;
private:
    float m_scale, m_height;
    float m_width; // -1 if variable
    std::vector<Section> m_sections;
};

/**\brief Result overlay UI element
 *
 * This UI element draws information about algorithm results onto the passed
 * image.
 */
class ResultRenderElement : public UIElement {
public:
    ResultRenderElement();
    ~ResultRenderElement();

    void render(cv::Mat& tgt);

    /**\brief Load some results into the render element.
     *
     * This function clears out the last frame of results and loads the new ones
     * in for rendering. When the `render()` function is called, the most recent
     * set of values passed in here will be drawn as the current frame.
     *
     * \param res The input vector of results
     */
    void setResults(const std::vector<ml::AlgorithmResult*>& res);

private:
    /**\brief Internal rendering-management object pointer.
     *
     * Type is hidden to avoid cluttering this header file. See
     * `result_render.cpp` for full implementation details.
     */
    void* m_render_internal;
};

/**\brief Graphical overlay, composed of multiple UI elements
 */
class Overlay : public UIElement {
public:
    Overlay();
    ~Overlay();

    /**\brief Add a UI element to the overlay.
     *
     * The overlay inherits ownership of the passed UI element.
     */
    void add(UIElement* elem);

    void render(cv::Mat& tgt);

private:
    std::vector<UIElement*> m_elems;
};

 };

#endif
