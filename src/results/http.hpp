#ifndef RES_HTTP_HPP
#define RES_HTTP_HPP

#include <boost/asio.hpp>
#include <boost/optional.hpp>
#include <boost/variant.hpp>
#include <boost/thread/future.hpp>

#include <memory>
#include <vector>
#include <list>
#include <utility>
#include <mutex>

#include "network.hpp"

namespace mdump {
namespace http {
namespace asio = boost::asio;
namespace ip = boost::asio::ip;

/// A type representing HTTP methods
enum Method { GET, POST, PUT };

/// A type representing an HTTP response code
struct ResponseCode {
    enum Type {
        INFORMATIONAL,
        SUCCESS,
        REDIRECTION,
        CLIENT_ERROR,
        SERVER_ERROR,
        UNKNOWN };

    unsigned char d[3]; ///< digits of the response code

    /// Get the response's class
    Type type() const;

    /// Get the code as an integer
    int as_int() const;
}; 

/** \brief Container for the HTTP headers sent along with a request
 */
class Headers {
public: // basic interface
    Headers();
    Headers(const Headers& parent);

    /** \brief Set a specific header, replacing anything that was there before.
     */
    void set(std::string header, std::string value);

    /** \brief Get the value of a specific header
     */
    boost::optional<const std::string&> get(std::string header) const;

    /** \brief Get all headers in the list */
    const std::map<std::string, std::string>& get_all() const;

    /** \brief Add all values from the given Headers into this one.
     *
     * Values which are already present in the current object will not be
     * replaced. The provided object is used as a collection of default values
     * only.
     */
    void augment(const Headers& other);

private:
    std::map<std::string, std::string> m_map;
};

/** \brief Container for arbitrary URLs
 */
class URL {
public:
    // TODO: Support user and password
    
    /** \brief Construct a URL by parsing a string.
     *
     * This may fail, raising a url_parse_exception if the string does not
     * contain a valid URL.
     */
    URL(std::string s);

    /** \brief Construct a base URL from scheme and host.
     *
     * This constructs a URL of the form "scheme://host:port/" and returns it.
     * Any further changes should be made via the associated setter methods.
     */
    URL(std::string scheme, std::string host, int port=80);

    URL& scheme(std::string sch);
    URL& host(std::string h);
    URL& port(int p);

    URL& path(std::string pth);
    URL& path(const std::vector<std::string>& parts);
    URL& path(const std::list<std::string>& parts);

    URL& query(std::string q);
    URL& query(const std::map<std::string, std::string>& kv);

    URL& fragment(std::string f);

    const std::string& scheme() const;
    const std::string& host() const;

    /// Get the port number
    const boost::optional<int> port() const;

    /** \brief Get the port number, or guess
     *
     * Get the provided port number if one exists, or guess one based on the URL
     * scheme otherwise.
     */
    int port_or_infer() const;

    /// Get the URL's path, or the empty string if it doesn't have one
    std::string path() const;

    /// Get the URL's path components, or an empty list if it doesn't have any
    boost::optional<const std::list<std::string>&> path_components() const;

    const boost::optional<std::string>& fragment() const;

    /// Get the URL's query as a string, or empty string for "no query"
    std::string query() const;

    /// Get the URL's query as a map of keys and values
    boost::optional<const std::map<std::string, std::string>&> query_kv() const;

    /** \brief Push a component on the end of the URL's path
     *
     * If the URL doesn't have a path, a new one will be created and the given
     * component will be appended to it.
     */
    void push(std::string part);

    /** \brief Remove a component from the end of the URL's path
     *
     * If the URL doesn't have a path, this operation will do nothing. If the
     * URL has a path with only one component, this removes the path from the
     * URL.
     */
    void pop();

    /// Generate a resolver query from this URL
    ip::tcp::resolver::query make_query() const;

    /** Generate an HTTP request target for this URL
     *
     * \param abs Generate an absolute form, for interacting with proxies
     */
    std::string make_request_target(bool abs=false) const;

private:
    std::string m_scheme, m_host;
    boost::optional<int> m_port;
    boost::optional<std::list<std::string> > m_path;

    // query can either be stored as a key-value map or a string
    // empty string or empty map represents "no query"
    boost::variant<std::string, std::map<std::string, std::string> > m_query;

    boost::optional<std::string> m_frag;
};

/** \brief An HTTP request before being sent */
class Request {
public:
    Request(Method m, URL url);

    /// Construct a GET request for the given URL
    static Request get(URL url);

    /// Construct a POST request for the given URL
    static Request post(URL url);

    /// Construct a PUT request for the given URL
    static Request put(URL url);

    /**** Begin construction utility methods ****/

    /// Set the request's URL
    Request& url(URL u);

    /// Set a specific header for this request
    Request& header(std::string hdr, std::string value);

    /// Replace the full set of headers for this request
    Request& headers(Headers hdrs);

    /// Augment the request's headers with an extra set
    Request& add_headers(const Headers& hdrs);

    /// Specify whether to follow redirects automatically
    Request& follow_redirects(bool f);

    /**** Begin property retrieval methods ****/
    const URL& url() const;
    Method method() const;
    const Headers& headers() const;
    Headers& headers();
    bool follow_redirects() const;

private:
    Method m_method;
    URL m_url;
    Headers m_hdrs;

    bool m_follow_redirects;
};

/** \brief An HTTP response, combined with the request that caused it. */
class Response {
public:
    Response(Request r, ResponseCode code);

    /**** Begin construction methods ****/

    /// Add a header to the response
    Response& header(std::string hdr, std::string value);

    /// Get a reference to the response headers as an object
    Headers& headers();

    /// Get a mutable reference to the request body
    std::vector<unsigned char>& body_mut();

    /// Replace the request body with the referenced vector
    void replace_body(std::unique_ptr<std::vector<unsigned char> > buf);

    /**** Begin information retrieval and querying methods ****/

    /// Get the original request
    const Request& request() const;

    /// Get the response code
    const ResponseCode& code() const;

    /** \brief Check for success when evaluated in a boolean context
     *
     * When evaluated in a boolean context, this returns whether or not the
     * request succeeded.
     */
    explicit operator bool() const;

    /** Get an immutable reference to the request body
     */
    const std::vector<unsigned char>& body() const;

    /** Transfer ownership of the request body to the caller.
     *
     * Subsequent calls to this function beyond the first will return null
     * pointers. The body() and body_mut() functions must not be called after
     * this function is called.
     */
    std::unique_ptr<std::vector<unsigned char> > take_body();

private:
    Request m_request;
    Headers m_hdrs;
    ResponseCode m_code;
    std::unique_ptr<std::vector<unsigned char> > m_body;
};

/** \brief HTTP connection handler
 *
 * Responsible for abstracting over raw asio socket operations and providing an
 * interface in terms of Request and Response objects.
 */
class HTTPConnection {
    typedef std::unique_ptr<Request> request_ptr;
    typedef std::unique_ptr<Response> response_ptr;
    typedef std::function<void(const boost::system::error_code&, response_ptr)>
        handler_fn;
public:
    HTTPConnection(std::shared_ptr<asio::io_service> svc);
    ~HTTPConnection();

    /** \brief Make a connection to the target server.
     *
     * This must only be run once, and must succeed before any other operations
     * may be performed on this connection. The connection will be closed
     * automatically when the object is destroyed.
     *
     * To perform actions upon connection close (from either the local or remote
     * end), pass a function to the on_close callback. This function will only
     * be called if the connection closes *after* async_connect finishes
     * successfully.
     */
    void async_connect(ip::tcp::endpoint tgt,
            std::function<void(const boost::system::error_code&)> f,
            std::function<void()> on_close);

    /** \brief Send a request to the target server.
     *
     * Requests will be sent in order and no more than one request will be
     * in-flight at any given time. Successive calls to async_request before
     * all prior requests have completed will add the later requests to a
     * queue.
     */
    void async_request(request_ptr req, handler_fn handler);

private:
    std::shared_ptr<asio::io_service> m_svc;
    ip::tcp::socket m_sock;
    std::function<void()> m_onclose;

    std::mutex m_queue_mut;
    std::list<std::pair<request_ptr, handler_fn> > m_queue; // under m_queue_mut
    bool m_pending; // under m_queue_mut

    std::pair<request_ptr, handler_fn> m_work; ///< Currently in-flight request
    std::vector<unsigned char> m_temp_buffer; ///< send buffer
    asio::basic_streambuf<> m_recv_buffer;
    response_ptr m_response; ///< response to the in-flight request

    /// Start working on the request at the front of the work queue
    void attempt_request();

    /// Encode the active request into the temp buffer
    void encode_request();

    /// Should be called when finished processing a request
    void finish_request();

    // Utility function to fail the current request with an error
    void fail_request(const boost::system::error_code& ec);

    /// Called when the header section of an HTTP response has been read
    void on_header_recvd(const boost::system::error_code& ec, size_t n);

    /// Called when the body section of an HTTP response has been read
    void on_body_recvd(int httpv, const boost::system::error_code& ec, size_t n);
};

/** \brief Base class for targets which encode data in HTTP requests
 *
 * This isn't a full HTTP client or anything like that, but it provides an easy
 * way to write dumps via REST APIs. It's also extensible to implement custom
 * authentication or request types.
 */
class HTTPTarget : public AsioDumpTarget {
public:
    HTTPTarget(bool keepalive=true,
            std::shared_ptr<asio::io_service> svc = nullptr);
    virtual ~HTTPTarget();

protected:
    /** \brief Send an HTTP request
     *
     * This will asynchronously send the given HTTP request, and wait for a
     * response from the server. When the server responds, the given handler
     * will be called with the response object.
     *
     * \param req The request to send
     * \param handler A function to handle the response (or error) produced
     */
    void submit(std::shared_ptr<Request> req,
            std::function<void(std::unique_ptr<Response>)> handler);

private:
    /** \brief Open a connection to the remote server specified by the given URL
     *
     * This either opens a new connection or reuses an existing one.
     */
    void get_conn(const URL& u,
            std::function<void(const boost::system::error_code&,
                std::shared_ptr<HTTPConnection>)> handler);

    /** Drop a given connection from persistent data structures. References
     * elsewhere may still keep it alive.
     */
    void dispose_conn(std::pair<std::string, int> key,
            std::shared_ptr<HTTPConnection> c);

    /// Hostname resolver
    ip::tcp::resolver m_resolver;

    bool m_enable_keepalive; ///< Whether to use HTTP keep-alive support

    /** Mapping of (host,port) pairs to persistent connections
     * 
     * If persistent connections aren't in use, then this map isn't used.
     */
    std::map<std::pair<std::string, int>,
        std::shared_ptr<HTTPConnection> > m_conns;

    /** List of active connections
     *
     * This holds active connections when persistent connections aren't in use.
     */
    std::list<std::shared_ptr<HTTPConnection> > m_active_conns;
    std::mutex m_conns_mtx;
};

/** \brief Simple target which just POSTs data to a URL */
class POSTTarget : public HTTPTarget {
public:
    POSTTarget(std::string url);
    ~POSTTarget();

    void write(const std::string& data);

private:
    URL m_url;
};
};
};
#endif
