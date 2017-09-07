#include "metadump.hpp"
#include <sstream>

using namespace mdump;

JSONWriter::JSONWriter(std::basic_ostream<char> &out, EnclosureType root) : m_out(out) {
    emit_open(root);
    m_first.push_back(true);
    m_stack.push_back(root);
}

JSONWriter::~JSONWriter() {
    while(!m_stack.empty()) {
        emit_close(m_stack.back());
        m_stack.pop_back();
        m_first.pop_back();
    }
    m_out.put('\n');
    m_out.flush();
}

void JSONWriter::emit_begin() {
    if(!m_first.back()) {
        m_out.put(',');
    } else {
        m_first.back() = false;
    }
}

void JSONWriter::emit_open(EnclosureType t) {
    switch(t) {
    case OBJECT: m_out.put('{'); break;
    case ARRAY:  m_out.put('['); break;
    }
}

void JSONWriter::emit_close(EnclosureType t) {
    switch(t) {
    case OBJECT: m_out.put('}'); break;
    case ARRAY:  m_out.put(']'); break;
    }
}

void JSONWriter::emit_str(const std::string& s) {
    m_out.put('"');
    for(auto c : s) {
        if(c == '"') m_out.write("\\\"", 2);
        else         m_out.put(c);
    }
    m_out.put('"');
}

void JSONWriter::emit_key(const std::string& s) {
    emit_begin();
    emit_str(s);
    m_out.put(':');
}

void JSONWriter::operator()(const std::string& val) {
    emit_begin();
    emit_str(val);
}
void JSONWriter::operator()(const char* val) {
    emit_begin();
    emit_str(val);
}
void JSONWriter::operator()(long val) { emit_begin(); m_out << val; }
void JSONWriter::operator()(int val) { emit_begin(); m_out << val; }
void JSONWriter::operator()(double val) { emit_begin(); m_out << val; }
void JSONWriter::operator()(bool val) { emit_begin(); m_out << val; }

void JSONWriter::array() {
    emit_begin();
    emit_open(ARRAY);
    m_stack.push_back(ARRAY);
    m_first.push_back(true);
}

void JSONWriter::object() {
    emit_begin();
    emit_open(OBJECT);
    m_stack.push_back(OBJECT);
    m_first.push_back(true);
}

void JSONWriter::operator()(const std::string &key, const char* val) {
    emit_key(key);
    emit_str(val);
}

void JSONWriter::operator()(const std::string &key, const std::string& val) {
    emit_key(key);
    emit_str(val);
}

void JSONWriter::operator()(const std::string &key, int val) {
    emit_key(key);
    m_out << val;
}

void JSONWriter::operator()(const std::string &key, long val) {
    emit_key(key);
    m_out << val;
}

void JSONWriter::operator()(const std::string &key, double val) {
    emit_key(key);
    m_out << val;
}

void JSONWriter::operator()(const std::string &key, bool val) {
    emit_key(key);
    m_out << (val ? "true" : "false");
}

void JSONWriter::array(const std::string &key) {
    emit_key(key);
    emit_open(ARRAY);
    m_stack.push_back(ARRAY);
    m_first.push_back(true);
}

void JSONWriter::object(const std::string &key) {
    emit_key(key);
    emit_open(OBJECT);
    m_stack.push_back(OBJECT);
    m_first.push_back(true);
}

void JSONWriter::end() {
    emit_close(m_stack.back());
    m_stack.pop_back();
}

Metadumper::Metadumper(std::unique_ptr<DumpTarget>&& tgt) {
    m_tgt = std::move(tgt);
}

Metadumper::~Metadumper() {
}

void Metadumper::accept(const std::vector<ml::AlgorithmResult*>& res, int fps,
        int frame, bool fpga, double cpu_use, double framerate, int fr_time) {
    // build the JSON
    std::stringstream strm;

    {
    JSONWriter json(strm, JSONWriter::OBJECT);

    json.object("frame");
        json("fps", fps);
        json("fpga", fpga);
        json.object("perf");
            json("cpu_use", cpu_use);
            json("fps", framerate);
            json("fr_time", fr_time);
        json.end();
        json.array("results");
        for(auto r : res) write_result(json, *r);
    json.end();
    }

    // transmit
    m_tgt->write(strm.str());
}

void Metadumper::write_result(JSONWriter& strm, const ml::AlgorithmResult& res) {
    switch(res.type) {
    case ml::RT_BOUNDING_BOXES:
        write_boundboxes(strm, dynamic_cast<const ml::BoundingBoxesResult&>(res));
        break;
    default:
        break;
    }
}
void Metadumper::write_boundboxes(JSONWriter& json, const ml::BoundingBoxesResult& res) {
    json.object();
    json("type", "bounding-boxes");
    json.array("boxes");

    for(auto box : res.boxes) {
        json.object();
        if(box.id != 0) json("id", (long)box.id);
        if(box.tag != 0) json("tag", (long)box.tag);

        json.object("topleft");
            json("x", box.bounds.tl().x);
            json("y", box.bounds.tl().y);
        json.end();
        json.object("btmright");
            json("x", box.bounds.br().x);
            json("y", box.bounds.br().y);
        json.end();
        json("area", box.bounds.width*box.bounds.height);
        json.end();
    }
    json.end();
    json.end();
}
