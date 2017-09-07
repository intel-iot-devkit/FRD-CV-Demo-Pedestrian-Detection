#include "overlay.hpp"
#include <stdlib.h>
#include <math.h>

using namespace ui;

AnchoredElement::AnchoredElement(cv::Point2f pos) : m_anchor(pos) { }
void AnchoredElement::setPos(cv::Point2f pos) { m_anchor = pos; }
cv::Point2f AnchoredElement::getPos() { return m_anchor; }

TextUIElement::TextUIElement(std::string fmt, TUIManager* mgr,
        cv::Point2f anchor, cv::Scalar color,
        alignment a, double size) : AnchoredElement(anchor), m_line(fmt, mgr),
        m_align(a), m_fontSize(size), m_color(color) { }

TextUIElement::~TextUIElement() {
}

void TextUIElement::render(cv::Mat& tgt) {
    cv::Point2f base(m_anchor.x*tgt.cols, m_anchor.y*tgt.rows);

    // figure out what to draw
    const std::string& txt = m_line.render();

    // compute rendered size of text
    int baseline;
    cv::Size sz = cv::getTextSize(txt, cv::FONT_HERSHEY_SIMPLEX, m_fontSize, 2,
            &baseline);
    float sx = sz.width;
    float sy = sz.height;
    float hx = sx / 2;
    float hy = sy / 2;
    
    // offset base as needed
    switch(m_align) {
    case alignment::CENTER:         base.x += hx;   base.y += hy;   break;
    case alignment::CENTER_SOUTH:   base.x += hx;                   break;
    case alignment::CENTER_NORTH:   base.x += hx;   base.y += sy;   break;
    case alignment::CENTER_EAST:    base.x += sx;   base.y += hy;   break;
    case alignment::CENTER_WEST:                    base.y += hy;   break;
    case alignment::NORTH_EAST:     base.x += sx;   base.y += sy;   break;
    case alignment::NORTH_WEST:                     base.y += sy;   break;
    case alignment::SOUTH_EAST:     base.x += sx;                   break;
    case alignment::SOUTH_WEST:                                     break;
    }

    // draw
    cv::putText(tgt, txt, base, cv::FONT_HERSHEY_SIMPLEX, m_fontSize, m_color,
            2);
}

StackedBarElement::StackedBarElement(cv::Point2f anchor, float height,
        float scale) : AnchoredElement(anchor), m_height(height) {
    m_scale = scale;
    m_width = -1;
}

StackedBarElement::~StackedBarElement() {
}

void StackedBarElement::render(cv::Mat& tgt) {
    cv::Point2f base(m_anchor.x*tgt.cols, m_anchor.y*tgt.rows);

    // update scale factors if needed
    if(m_width > 0) {
        float total = 0;
        for(auto s : m_sections)
            total += s.size;

        // no point rendering if everything's zero
        if(total == 0) return;

        // update the scale factor
        m_scale = m_width / total;
    }

    // draw bars
    cv::Point2f left = base;
    for(auto s : m_sections) {
        float w = s.size * m_scale;
        cv::rectangle(tgt, left, cv::Point2f(left.x+w, left.y+m_height),
                s.color, CV_FILLED);
    }
}

void StackedBarElement::setWidth(float width) {
    if(width > 0) m_width = width;
    else m_width = -1;
}

void StackedBarElement::setScale(float scale) {
    m_width = -1;
    m_scale = scale;
}

// generate a random color
cv::Scalar randomColor() {
    // build the color in HSV space, then convert to RGB
    float h = 2*M_PI * (rand() / ((float)RAND_MAX));
    float s = 0.5 + (rand() / ((float)RAND_MAX))*0.5;
    float v = 1.0;

    float chroma = v*s;
    h /= M_PI/3;
    float x = chroma*(1-fabs(fmod(h, 2) - 1));
    float m = v-chroma;

    if((0 <= h) && (h <= 1))
        return cv::Scalar(chroma+m, x+m, m);
    else if((1 <= h) && (h <= 2))
        return cv::Scalar(x+m, chroma+m, 0);
    else if((2 <= h) && (h <= 3))
        return cv::Scalar(0, chroma+m, x+m);
    else if((3 <= h) && (h <= 4))
        return cv::Scalar(0, x+m, chroma+m);
    else if((4 <= h) && (h <= 5))
        return cv::Scalar(x+m, 0, chroma+m);
    else if((5 <= h) && (h <= 6))
        return cv::Scalar(chroma+m, 0, x+m);
}

StackedBarElement::Section& StackedBarElement::operator[](int idx) {
    // pad sections list with empties if needed
    while(m_sections.size() < idx) {
        StackedBarElement::Section s = { 0.0f, randomColor() };
        m_sections.push_back(s);
    }

    return m_sections[idx];
}

const StackedBarElement::Section& StackedBarElement::operator[](int idx) const {
    return m_sections[idx];
}

Overlay::Overlay() {
}

Overlay::~Overlay() {
    for(auto e : m_elems) delete e;
}

void Overlay::add(UIElement* elem) {
    m_elems.push_back(elem);
}

void Overlay::render(cv::Mat& tgt) {
    for(auto e : m_elems) e->render(tgt);
}
