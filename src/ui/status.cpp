#include "status.hpp"
#include <stdexcept>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

using namespace ui;

Field::Field() : m_refs(0), m_update(0) { }
void Field::ref() { m_refs++; }
void Field::unref() { m_refs--; }
int Field::refs() { return m_refs; }
unsigned long Field::getLastUpdate() const { return m_update; }
void Field::notify() { m_update++; }

std::string Field::operator()(int width) const {
    char* p = this->c_str(width);
    std::string res(p);
    free(p);
    return res;
}

TUIManager::TUIManager() {
}

TUIManager::~TUIManager() {
}

void TUIManager::registerField(std::string name, Field* field) {
    m_name_map[name] = field;
    m_dynamic_fields.push_back(field);
}

void TUIManager::registerField(std::string name, char code, Field* field) {
    m_short_map[code] = field;
    m_name_map[name] = field;
    m_dynamic_fields.push_back(field);
}

void TUIManager::update() {
    for(Field* f : m_dynamic_fields) {
        if(f->refs() > 0)
            f->update();
    }
}

const Field* TUIManager::get(const std::string& name) {
    std::map<std::string, Field*>::iterator i = m_name_map.find(name);
    if(i == m_name_map.end()) return NULL;
    i->second->ref();
    return i->second;
}

const Field* TUIManager::get(char code) {
    std::map<char, Field*>::iterator i = m_short_map.find(code);
    if(i == m_short_map.end()) return NULL;
    i->second->ref();
    return i->second;
}

const Field* TUIManager::get_text(const std::string& text) {
    std::map<std::string, Field*>::iterator i = m_static_map.find(text);
    if(i == m_static_map.end()) {
        Field* f = new StaticField(text);
        m_static_map[text] = f;
        f->ref();
        return f;
    }
    i->second->ref();
    return i->second;
}

void TUIManager::release(const Field* f) {
    Field* field = const_cast<Field*>(f);
    field->unref();
}

enum StatusLineParseState {
    PS_NORMAL, // normal parsing - reading literal chars
    PS_ESCAPED, // just after a backslash - escape next char
    PS_TEMPLATE, // reading template
    PS_TEMPL_ESC, // reading template, escaped
    PS_TEMPL_WIDTH, // reading template width spec
};

StatusLine::StatusLine(std::string templ, TUIManager* mgr) {
    // interpret template
    StatusLineParseState state = PS_NORMAL;
    std::string accum = "";
    int width = 0;
    for(char c : templ) {
        switch(state) {
        case PS_NORMAL: // normal character processing
            if(c == '\\') { // read escapes
                state = PS_ESCAPED;
            } else if(c == '{') { // push accumulator on template
                if(!accum.empty()) {
                    const Field* f = mgr->get_text(accum);
                    m_fields.push_back(std::make_pair(
                                std::make_pair(f->getLastUpdate(), width), f));
                    accum.clear();
                }
                state = PS_TEMPLATE;
            } else {
                accum.push_back(c);
            }
            break;
        case PS_ESCAPED:
            accum.push_back(c);
            state = PS_NORMAL;
            break;
        case PS_TEMPL_ESC:
            accum.push_back(c);
            state = PS_TEMPLATE;
            break;
        case PS_TEMPLATE:
            if(c == '\\') {
                state = PS_TEMPL_ESC;
            } else if(c == '}') {
                const Field* f = NULL;
                if(accum.length() == 1) {
                    f = mgr->get(accum[0]);
                } else if(!accum.empty()) {
                    f = mgr->get(accum);
                } else { // found '{}' - just emit it into the accumulator
                    accum.push_back('{');
                    accum.push_back('}');
                    state = PS_NORMAL;
                    break;
                }

                if(f == NULL)
                    throw std::invalid_argument("Cannot find given field");
                m_fields.push_back(std::make_pair(
                            std::make_pair(f->getLastUpdate(), width), f));
                state = PS_NORMAL;
                accum.clear();
            } else if(c == '/') {
                state = PS_TEMPL_WIDTH;
            } else {
                accum.push_back(c);
            }
            break;
        case PS_TEMPL_WIDTH:
            if(c >= '0' && c <= '9') {
                width *= 10;
                width += c - '0';
            } else if(c == '}') {
                if(width == 0)
                    throw std::invalid_argument("Invalid width specifier");

                const Field* f = NULL;
                if(accum.length() == 1) {
                    f = mgr->get(accum[0]);
                } else if(!accum.empty()) {
                    f = mgr->get(accum);
                }

                if(f == NULL)
                    throw std::invalid_argument("Cannot find given field");
                m_fields.push_back(std::make_pair(
                            std::make_pair(f->getLastUpdate(), width), f));
                state = PS_NORMAL;
                width = 0;
                accum.clear();
            } else {
                throw std::invalid_argument("Invalid width specifier");
            }
            break;
        }
    }
    if(!accum.empty()) {
        const Field* f = mgr->get_text(accum);
        m_fields.push_back(std::make_pair(
                    std::make_pair(f->getLastUpdate(), width), f));
    }

    m_manager = mgr;
    forceRender();
}

StatusLine::~StatusLine() {
    for(auto f : m_fields) m_manager->release(f.second);
}

const std::string& StatusLine::render() {
    bool changed = false;
    for(auto f : m_fields)
        if(f.second->getLastUpdate() > f.first.first)
            changed = true;

    if(changed) forceRender();

    return m_contents;
}

void StatusLine::forceRender() {
    std::stringstream sstrm;
    for(auto f : m_fields) {
        sstrm << (*f.second)(f.first.second);
        f.first.first = f.second->getLastUpdate();
    }
    m_contents = sstrm.str();
}

StaticField::StaticField(std::string s) : m_body(s) { }
StaticField::~StaticField() { }
int StaticField::getNativeWidth() const { return m_body.length(); }
void StaticField::update() { }

std::string StaticField::operator()(int width) const {
    if(width == 0) return m_body;

    int add = m_body.length() < width ? (width - m_body.length()) : 0;
    std::string r(add, ' '); // right padding
    r += m_body;
    return r;
}

char* StaticField::c_str(int width) const {
    if(width == 0) return strdup(m_body.c_str());

    unsigned int len = m_body.length() < width ? width : m_body.length();
    char* buf = (char*)malloc(len+1);
    memset(buf, ' ', len);
    buf[len] = 0;
    memcpy(buf+(len-m_body.length()), m_body.c_str(), m_body.length());
    return buf;
}

FloatField::FloatField(float alpha) : m_alpha(alpha) {
    m_val = 0;
}

FloatField::~FloatField() {
}

void FloatField::update(float v) {
    m_val = m_alpha*m_val + (1-m_alpha)*v;
    notify();
}

int FloatField::getNativeWidth() const {
    // compute tens digits using base-10 logarithm
    int digits;
    if(m_val != 0) {
        double lb10 = logf(fabs(m_val)) / logf(10);
        digits = abs((int)lb10);
    } else {
        digits = 1;
    }

    // return result (digits + decimal + two fractional digits + negative?)
    return digits+3+((m_val < 0) ? 1 : 0);
}

char* FloatField::c_str(int width) const {
    int native = getNativeWidth();
    if(width <= native) {
        char buf[native+1];
        snprintf(buf, native+1, "%.2f", m_val);
        return strdup(buf);
    } else {
        char buf[width+1];
        snprintf(buf, width+1, "%.*f",
                (width-(native-2)), // number of digits after the decimal
                m_val);
        return strdup(buf);
    }
}

std::string FloatField::operator()(int width) const {
    int native = getNativeWidth();
    if(width <= native) {
        char buf[native+1];
        snprintf(buf, native+1, "%.2f", m_val);
        return std::string(buf);
    } else {
        char buf[width+1];
        snprintf(buf, width+1, "%.*f",
                (width-native+2), // number of digits after the decimal
                m_val);
        return std::string(buf);
    }
}

void FloatField::setAlpha(float a) {
    m_alpha = a;
}

float FloatField::getValue() {
    return m_val;
}

CPULoad::CPULoad() : FloatField(0.9) {
    m_last_total = 0;
    m_last_me = 0;
}

void CPULoad::update() {
    FILE* f = fopen("/proc/stat", "r");
    if(f == NULL) return;

    char fname[64];
    snprintf(fname, 64, "/proc/%d/stat", getpid());
    FILE* m = fopen(fname, "r");
    if(m == NULL) return;

    // pull out fields
    long user, nice, sys, idle, iowait, irq, softirq, steal, quest, quest_nice;
    if(fscanf(f, "cpu  %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld",
            &user, &nice, &sys, &idle, &iowait, &irq, &softirq, &steal,
            &quest, &quest_nice) != 10) return;
    fclose(f);

    long total = user+nice+sys+idle+iowait+irq+softirq+steal+quest+quest_nice;

    // compute the deltas
    long d_total = total - m_last_total;

    // pull out PID-specific time stats
    char line[512];
    fgets(line, 512, m);
    fclose(m);
    if(line[0] == 0) return;

    char *ptr = line;
    for(int i = 0;i < 13;i++) {
        for(;(*ptr != ' ') && (*ptr != 0);ptr++);
        ptr++;
    }
    long unsigned int ticks;
    if(sscanf(ptr, "%lu", &ticks) != 1) return;

    long d_me = ticks - m_last_me;

    if(m_last_total != 0)
        FloatField::update(((float)d_me / (float)d_total) * 100);
    m_last_me = ticks;;
    m_last_total = total;
}

void ValueField::addSample(float val) { FloatField::update(val); }

void ValueField::update() { }
