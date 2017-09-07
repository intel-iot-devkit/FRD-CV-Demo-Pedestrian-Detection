#include "http.hpp"

using namespace mdump::http;
using namespace std;

/**************** Request begins here ****************/

Request::Request(Method m, URL url) : m_method(m), m_url(url) {
    m_follow_redirects = true;
}

Request Request::get(URL url) { return Request(GET, url); }
Request Request::post(URL url) { return Request(POST, url); }
Request Request::put(URL url) { return Request(PUT, url); }

Request& Request::url(URL u) { m_url = u; return *this; }
Request& Request::header(std::string hdr, std::string value) {
    m_hdrs.set(hdr, value);
    return *this;
}
Request& Request::headers(Headers hdrs) { m_hdrs = hdrs; return *this; }

Request& Request::add_headers(const Headers& hdrs) {
    m_hdrs.augment(hdrs);
    return *this;
}
Request& Request::follow_redirects(bool f) {
    m_follow_redirects = f;
    return *this;
}

const URL& Request::url() const { return m_url; }
Method Request::method() const { return m_method; }
const Headers& Request::headers() const { return m_hdrs; }
Headers& Request::headers() { return m_hdrs; }
bool Request::follow_redirects() const { return m_follow_redirects; }

/**************** Response begins here ****************/

Response::Response(Request r, ResponseCode c) : m_request(r), m_code(c) {
    m_body = unique_ptr<vector<unsigned char> >(new vector<unsigned char>());
}

const Request& Response::request() const { return m_request; }
const ResponseCode& Response::code() const { return m_code; }
Response::operator bool() const {
    return m_code.type() == ResponseCode::SUCCESS;
}
Headers& Response::headers() { return m_hdrs; }

Response& Response::header(string hdr, string value) {
    m_hdrs.set(hdr, value);
    return *this;
}

vector<unsigned char>& Response::body_mut() {
    return *m_body;
}

void Response::replace_body(unique_ptr<vector<unsigned char> > buf) {
    m_body = move(buf);
}

const vector<unsigned char>& Response::body() const {
    return *m_body;
}

unique_ptr<vector<unsigned char> > Response::take_body() {
    return move(m_body);
}

/**************** HTTPConnection begins here ****************/

HTTPConnection::HTTPConnection(shared_ptr<asio::io_service> svc) :
        m_svc(svc), m_sock(*svc) {
    m_pending = false;
}

HTTPConnection::~HTTPConnection() {
}

void HTTPConnection::async_connect(ip::tcp::endpoint tgt,
        function<void(const boost::system::error_code&)> f,
        function<void()> on_close) {
    cout << tgt << '\n';
    m_sock.async_connect(tgt, [f](const boost::system::error_code& ec) { f(ec); });
    m_onclose = on_close;
}

void HTTPConnection::async_request(request_ptr req, handler_fn handler) {
    {
        lock_guard<mutex> lck(m_queue_mut);
        m_queue.push_back(make_pair(move(req), handler));
    }

    attempt_request();
}

void HTTPConnection::attempt_request() {
    {
        lock_guard<mutex> lck(m_queue_mut);
        if(m_pending) return; // no need to fire off *another* parallel request

        if(m_queue.size() == 0) return; // done - there's no work to do

        // pull a request off and start working on it
        m_work = move(m_queue.front());
        m_queue.pop_front();
        m_pending = true;

        encode_request();
    }

    asio::async_write(m_sock, asio::buffer(m_temp_buffer),
            [this](const boost::system::error_code& ec, size_t sz) {
                {
                    if(ec) {
                        fail_request(ec);
                    } else {
                        // wait for a response
                        m_recv_buffer.consume(m_recv_buffer.size());
                        asio::async_read_until(m_sock, m_recv_buffer,
                                "\x0d\x0a\x0d\x0a",
                                boost::bind(&HTTPConnection::on_header_recvd,
                                    this, _1, _2));
                    }
                }
            });
}

void HTTPConnection::on_header_recvd(const boost::system::error_code& ec,
        size_t n) {
    if(ec) {
        // resolve with error
        fail_request(ec);
        return;
    }

    list<string> lines;
    {
        asio::streambuf::const_buffers_type bufs = m_recv_buffer.data();
        string temp(asio::buffers_begin(bufs),
                asio::buffers_begin(bufs)+m_recv_buffer.size());
        m_recv_buffer.consume(m_recv_buffer.size());
        for(auto i = m_temp_buffer.begin();i != m_temp_buffer.end();) {
            unsigned char c = *(i++);

            if(c == 0x0d && *i == 0x0a) {
                i++;
                lines.push_back(temp);
                temp.clear();
                continue;
            }
            temp.push_back(c);
        }
        if(!temp.empty()) lines.push_back(temp);
        m_temp_buffer.clear();
    }

    if(lines.empty()) {
        // parse error
        fail_request(ec);
        return;
    }

    // parse the status line
    string line0 = lines.front();
    lines.pop_front();
    int http_version = 0;
    if(line0.compare(0, 9, "HTTP/1.1 ") != 0 &&
            line0.compare(0, 9, "HTTP/1.0 ") != 0) {
        if(line0.compare(0,9,"HTTP/1.1 ") == 0) http_version = 11;
        else http_version = 10;
        // only supporting HTTP 1.0 and 1.1
        fail_request(ec);
        return;
    }
    ResponseCode code;
    code.d[0] = line0[9] - '0';
    code.d[1] = line0[10] - '0';
    code.d[2] = line0[11] - '0';
    if(code.d[0] > 9 && code.d[1] > 9 && code.d[2] > 9) {
        // status codes must be integers
        fail_request(ec);
        return;
    }

    // process headers
    Headers hdrs;
    for(auto l : lines) {
        if(l.empty()) break;
        size_t colon_idx = l.find(':');
        if(colon_idx == string::npos) {
            fail_request(ec); // bad header line
            return;
        }
        string key(l, 0, colon_idx);

        while(l[++colon_idx] == ' ');
        string val(l, colon_idx);
        hdrs.set(key, val);
    }

    // populate a response
    m_response = unique_ptr<Response>(new Response(*m_work.first, code));
    m_response->headers() = hdrs;

    if(hdrs.get("Content-Length")) { // fixed-length
        string ct_len = *hdrs.get("Content-Length");
        char* p = 0;
        size_t len = strtoul(ct_len.c_str(), &p, 10);
        if(*p != 0) {
            // invalid content-length
            fail_request(ec);
            return;
        }

        // read the response
        asio::async_read(m_sock, m_recv_buffer, asio::transfer_exactly(len),
                boost::bind(&HTTPConnection::on_body_recvd, this, http_version,
                    _1, _2));
    } else { // close-terminated
        asio::async_read(m_sock, m_recv_buffer,
                boost::bind(&HTTPConnection::on_body_recvd, this, http_version,
                    _1, _2));
    }
}

void HTTPConnection::on_body_recvd(int http_version,
        const boost::system::error_code& ec, size_t n) {
    if(ec) {
        // resolve with error
        m_recv_buffer.consume(m_recv_buffer.size());
        fail_request(ec);
        return;
    }

    // reserve space
    vector<unsigned char>& bodyv = m_response->body_mut();
    bodyv.reserve(m_recv_buffer.size());

    // populate response
    asio::buffer_copy(asio::buffer(bodyv), m_recv_buffer.data());
    m_recv_buffer.consume(m_recv_buffer.size());

    // return result
    m_work.second(ec, move(m_response));

    // continue if possible, abort otherwise
    if(http_version == 11 &&
            m_response->headers().get("connection").value_or("close")
                == "Keep-Alive") {
        finish_request();
    } else { // Connection: Close, HTTP/1.0, etc
        m_sock.close();
        for(auto i = m_queue.begin();i != m_queue.end();i++)
            i->second(asio::error::eof, nullptr);
        m_onclose();
    }
}

void HTTPConnection::fail_request(const boost::system::error_code& ec) {
    printf("Request failed: %s\n", ec.message().c_str());
    if(ec == asio::error::connection_reset ||
            ec == asio::error::broken_pipe ||
            ec == asio::error::eof) {
        // the connection is dead - don't continue
        {
            lock_guard<mutex> m(m_queue_mut);
            for(auto i = m_queue.begin();i != m_queue.end();i++)
                i->second(ec, nullptr);
        }
        m_sock.close();
        m_onclose();
        return;
    }
    m_work.second(ec, nullptr);
    finish_request();
}

void HTTPConnection::finish_request() {
    {
        lock_guard<mutex> l(m_queue_mut);
        m_pending = false;
        m_work = make_pair(nullptr, handler_fn()); // don't keep old data around
        m_temp_buffer.clear();
    }

    // try the next one if needed
    attempt_request();
}

void HTTPConnection::encode_request() {
    stringstream sstrm;

    switch(m_work.first->method()) {
    case POST: sstrm << "POST "; break;
    case PUT: sstrm << "PUT "; break;
    default:
    case GET: sstrm << "GET "; break;
    }

    sstrm << m_work.first->url().make_request_target() << " HTTP/1.1\x0d\x0a";

    for(auto kv : m_work.first->headers().get_all())
        sstrm << kv.first << ": " << kv.second << "\x0d\x0a";

    // TODO: Add support for request bodies

    sstrm << "\x0d\x0a";
    m_temp_buffer.clear();
    string s = sstrm.str();
    copy(s.begin(), s.end(), back_inserter(m_temp_buffer));
}

/**************** HTTPTarget begins here ****************/

HTTPTarget::HTTPTarget(bool keepalive, std::shared_ptr<asio::io_service> svc) :
        AsioDumpTarget(svc), m_enable_keepalive(keepalive),
        m_resolver(*m_svc) {
}

HTTPTarget::~HTTPTarget() {
}

void HTTPTarget::submit(shared_ptr<Request> req,
        function<void(unique_ptr<Response>)> handler) {
    get_conn(req->url(), [this,req,handler](const boost::system::error_code& e,
                shared_ptr<HTTPConnection> c) {
            unique_ptr<Request> p(new Request(*req));
            if(e) {
                printf("Connection error: %s\n", e.message().c_str());
                handler(nullptr);
            } else {
                c->async_request(move(p),
                    [this,handler](const boost::system::error_code& e,
                            unique_ptr<Response> p) {
                        if(e) handler(nullptr);
                        else handler(move(p));
                    });
            }
        });
}

// try to either open or create an HTTP connection
void HTTPTarget::get_conn(const URL& u,
            function<void(const boost::system::error_code&,
                shared_ptr<HTTPConnection>)> handler) {
    pair<string, int> key = make_pair(u.host(), u.port_or_infer());

    if(m_enable_keepalive) {
        lock_guard<mutex> lck(m_conns_mtx);
        auto itr = m_conns.find(key);
        if(itr != m_conns.end()) {
            handler(boost::system::error_code(), itr->second);
                printf("written\n");
            return;
        }
    }

    // construct a new HTTP connection
    shared_ptr<HTTPConnection> conn(new HTTPConnection(m_svc));

    {
        lock_guard<mutex> l(m_conns_mtx);
        if(m_enable_keepalive) {
            m_conns[key] = conn;
        } else {
            m_active_conns.push_back(conn);
        }
    }

    // no connection available for reuse - make a new one
    m_resolver.async_resolve(u.make_query(),
            [this, conn, handler, key](const boost::system::error_code& ec,
                    ip::tcp::resolver::iterator r) {
                if(ec) {
                    // TODO: handle error
                    printf("Error: %s\n", ec.message().c_str());
                    dispose_conn(key, conn);
                    return;
                }

                conn->async_connect(*r,
                        [=](const boost::system::error_code& e) {
                            if(e) {
                                handler(e, nullptr);
                                dispose_conn(key, conn);
                            } else {
                                handler(e, conn);
                            }
                        },
                        [this, key, conn]() { // closed
                            dispose_conn(key, conn); });
            });
}

void HTTPTarget::dispose_conn(pair<string, int> key,
        shared_ptr<HTTPConnection> conn) {
    lock_guard<mutex> l(m_conns_mtx);
    if(m_enable_keepalive) {
        m_conns.erase(key);
    } else {
        m_active_conns.remove(conn);
    }
}

/**************** POSTTarget begins here ****************/
POSTTarget::POSTTarget(std::string url) : HTTPTarget(false), m_url(url) {
}

POSTTarget::~POSTTarget() {
}

void POSTTarget::write(const std::string& data) {
    shared_ptr<Request> req(new Request(Request::post(m_url)));
    submit(req, [this](unique_ptr<Response> h) {
            if(h && *h) printf("OK\n"); else printf("ERR\n");
            });
}

