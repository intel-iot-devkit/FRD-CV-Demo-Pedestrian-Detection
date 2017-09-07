#include "http.hpp"

#include <locale>
#include <algorithm>
#include <stdexcept>

#define DEFAULT_USER_AGENT "pdfw-httptarget/1.0"

using namespace mdump::http;
using namespace boost;
using namespace std;

static string RESERVED_CHARS = "\n !*'():;@&=+$,/?#[]";
static string HEX_DIGITS = "0123456789ABCDEF";

/// Generate a URL-encoded representation of a given string
string urlencode(string s) {
    string r;
    for(auto c : s) {
        if(RESERVED_CHARS.find(c) != string::npos) {
            r.push_back('%');
            r.push_back(HEX_DIGITS[(c >> 4) & 0x0f]);
            r.push_back(HEX_DIGITS[(c >> 0) & 0x0f]);
        } else {
            r.push_back(c);
        }
    }

    return r;
}

ResponseCode::Type ResponseCode::type() const {
    switch(d[0]) {
    case 1: return INFORMATIONAL;
    case 2: return SUCCESS;
    case 3: return REDIRECTION;
    case 4: return CLIENT_ERROR;
    case 5: return SERVER_ERROR;
    default: return UNKNOWN;
    }
}

int ResponseCode::as_int() const {
    return 100*d[0] + 10*d[1] + d[2];
}

Headers::Headers() {
    m_map["user-agent"] = DEFAULT_USER_AGENT;
}

Headers::Headers(const Headers& parent) {
    m_map = parent.m_map;
}

void Headers::set(string header, string value) {
    // normalize key case
    transform(header.begin(), header.end(), header.begin(),
            [](char c) { return tolower(c); });
    m_map[header] = value;
}

optional<const string&> Headers::get(string header) const {
    // normalize key case
    transform(header.begin(), header.end(), header.begin(),
            [](char c) { return tolower(c); });

    auto itr = m_map.find(header);
    if(itr == m_map.end()) {
        return boost::none;
    } else {
        return itr->second;
    }
}

const map<string, string>& Headers::get_all() const {
    return m_map;
}

void Headers::augment(const Headers& other) {
    for(auto i : other.m_map) {
        m_map.emplace(i.first, i.second);
    }
}

enum UrlParseState {
    S_SCHEME,
    S_SLASH_1, S_SLASH_2,
    S_HOST, S_PORT, S_PATH, S_QUERY, S_FRAG
};

URL::URL(string s) {
    UrlParseState state = S_SCHEME;

    string top;
    for(auto c : s) {
        switch(state) {
        case S_SCHEME:
            if(c == ':') {
                m_scheme = top;
                top.clear();
                state = S_SLASH_1;
            } else {
                top.push_back(c);
            }
            break;
        case S_SLASH_1:
            if(c == '/') {
                state = S_SLASH_2;
            } else {
                throw runtime_error("Invalid URL given");
            }
            break;
        case S_SLASH_2:
            if(c == '/') {
                state = S_HOST;
            } else {
                throw runtime_error("Invalid URL given");
            }
            break;
        case S_HOST:
            if(c == ':' || c == '/') {
                m_host = top;
                top.clear();
                if(c == ':') state = S_PORT;
                else if(c == '/') {
                    state = S_PATH;
                    top.push_back('/');
                }
            } else {
                top.push_back(c);
            }
            break;
        case S_PORT:
            if(c == '/') {
                m_port = atoi(top.c_str());
                top.clear();
                state = S_PATH;
            } else {
                top.push_back(c);
            }
            break;
        case S_PATH:
            if(c == '?' || c == '#') {
                this->path(top);
                top.clear();
                if(c == '?') state = S_QUERY;
                else if(c == '#') state = S_FRAG;
            } else {
                top.push_back(c);
            }
            break;
        case S_QUERY:
            if(c == '?') {
                m_query = top;
                top.clear();
                state = S_FRAG;
            } else {
                top.push_back(c);
            }
            break;
        case S_FRAG:
            top.push_back(c);
            break;
        }
    }

    // finalize
    switch(state) {
    case S_HOST:
        m_host = top;
        return;
    case S_PORT:
        m_port = atoi(top.c_str());
        return;
    case S_PATH:
        this->path(top);
        return;
    case S_QUERY:
        m_query = top;
        return;
    case S_FRAG:
        m_frag = top;
        break;
    default:
        throw runtime_error("Invalid URL given");
    }
}

URL::URL(string scheme, string host, int port) {
    m_scheme = scheme;
    m_host = host;
    m_port = port;

    m_query = ""; // use an empty query string
}

URL& URL::scheme(string sch) { m_scheme = sch; return *this; }
URL& URL::host(string h) { m_host = h; return *this; }
URL& URL::port(int p) { m_port = p; return *this; }

URL& URL::path(string pth) {
    // split path into components before assignment
    list<string> parts;
    size_t last = 0;
    for(auto i = pth.find('/');i != string::npos;i = pth.find('/', last)) {
        parts.push_back(pth.substr(last, i-last));
        last = i+1;
    }
    if(last < pth.size()-1) {
        parts.push_back(pth.substr(last, string::npos));
    }

    // perform assignment
    m_path = parts;
    return *this;
}

URL& URL::path(const vector<string>& parts) {
    m_path = list<string>(parts.begin(), parts.end());
    return *this;
}

URL& URL::path(const list<string>& parts) {
    m_path = list<string>(parts.begin(), parts.end());
    return *this;
}

URL& URL::query(string q) {
    m_query = q;
    return *this;
}

URL& URL::query(const map<string, string>& kv) {
    m_query = map<string, string>(kv);
    return *this;
}

URL& URL::fragment(string f) { m_frag = f; return *this; }

const string& URL::scheme() const { return m_scheme; }
const string& URL::host() const { return m_host; }
const boost::optional<int> URL::port() const { return m_port; }

int URL::port_or_infer() const {
    if(m_port) return *m_port;

    if(m_scheme == "http") return 80;
    else if(m_scheme == "https") return 443;

    // TODO: smarter handling here
    return 80;
}

string URL::path() const {
    string r;
    if(m_path) {
        for(auto c : *m_path) {
            r.push_back('/');
            r += c;
        }
    } else {
        r.push_back('/');
    }
    return r;
}

optional<const list<string>&> URL::path_components() const {
    if(m_path) return *m_path;
    else return boost::none;
}

const optional<string>& URL::fragment() const { return m_frag; }

string URL::query() const {
    if(m_query.which() == 0) {
        return "?" + get<string>(m_query);
    } else {
        // build query string
        string s;
        bool first_elem = true;
        for(auto kv : get<map<string,string> >(m_query)) {
            if(first_elem) s.push_back('?');
            else s.push_back('&');
            first_elem = false;

            s += urlencode(kv.first);
            s.push_back('=');
            s += urlencode(kv.second);
        }
        return s;
    }
}

optional<const map<string, string>&> URL::query_kv() const {
    if(m_query.which() == 1) return get<map<string,string> >(m_query);
    return boost::none;
}

void URL::push(string part) {
    if(!m_path) m_path = list<string>();
    m_path->push_back(part);
}

void URL::pop() {
    if(!m_path) return;
    m_path->pop_back();
    if(m_path->empty()) m_path = boost::none;
}

ip::tcp::resolver::query URL::make_query() const {
    stringstream port_strm;
    port_strm << port_or_infer();
    return ip::tcp::resolver::query(m_host, port_strm.str());
}

string URL::make_request_target(bool abs) const {
    if(abs) {
        string res = m_scheme + "://" + m_host + path() + query();
        return res;
    } else {
        return path() + query();
    }
}
