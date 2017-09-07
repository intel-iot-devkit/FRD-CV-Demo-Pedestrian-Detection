#ifndef METADUMP_HPP
#define METADUMP_HPP

#include <string>
#include <vector>
#include <ostream>
#include <memory>

#include "../algorithm.hpp"

namespace mdump {
class JSONWriter {
public:
    enum EnclosureType { OBJECT, ARRAY };

private:
    std::basic_ostream<char>& m_out;
    std::vector<EnclosureType> m_stack;
    std::vector<bool> m_first;

    void emit_begin();
    void emit_key(const std::string& s);
    void emit_str(const std::string& s);
    void emit_open(EnclosureType t);
    void emit_close(EnclosureType t);

public:
    JSONWriter(std::basic_ostream<char> &out, EnclosureType root);
    ~JSONWriter();

    void operator()(const std::string& val);
    void operator()(const char* val);
    void operator()(long val);
    void operator()(int val);
    void operator()(double val);
    void operator()(bool val);
    void array();
    void object();

    void operator()(const std::string &key, const std::string& val);
    void operator()(const std::string &key, const char* val);
    void operator()(const std::string &key, long val);
    void operator()(const std::string &key, int val);
    void operator()(const std::string &key, double val);
    void operator()(const std::string &key, bool val);
    void array(const std::string &key);
    void object(const std::string &key);
    void end();
};

class DumpTarget {
public:
    /** \brief Write a single data object to the dump target
     *
     * All implementations of this method must be asynchronous.
     */
    virtual void write(const std::string& data)=0;
};

class Metadumper {
public:
    Metadumper(std::unique_ptr<DumpTarget>&& tgt);
    ~Metadumper();

    void accept(const std::vector<ml::AlgorithmResult*>& res, int fps, int frame,
            bool fpga, double cpu_use, double framerate, int fr_time);

private:
    void write_result(JSONWriter& strm, const ml::AlgorithmResult& res);
    void write_boundboxes(JSONWriter& strm, const ml::BoundingBoxesResult& res);

    std::unique_ptr<DumpTarget> m_tgt;
};
};
#endif
