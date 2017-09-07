#ifndef RES_NETWORK_HPP
#define RES_NETWORK_HPP

#include <boost/asio.hpp>
#include <thread>
#include <mutex>
#include <memory>
#include <list>
#include <vector>

#include "metadump.hpp"

namespace mdump {
namespace asio = boost::asio;
namespace ip = boost::asio::ip;

/** \brief Dump target base class for things that use boost::asio */
class AsioDumpTarget : public DumpTarget {
public:
    AsioDumpTarget(std::shared_ptr<asio::io_service> svc = nullptr);

    std::shared_ptr<asio::io_service>& get_service();

protected:
    std::unique_ptr<std::thread> m_svc_thread;
    std::unique_ptr<asio::io_service::work> m_work; // don't stop running
    std::shared_ptr<asio::io_service> m_svc;
};

/** \brief Abstract base class to make socket target implementation easier
 *
 * This class manages most of the asynchronous transmission and queueing logic
 * so it doesn't have to be reimplemented in every socket class.
 */
class SocketTarget : public AsioDumpTarget {
public:
    SocketTarget(std::shared_ptr<asio::io_service> svc = nullptr);

protected:
    std::mutex m_queue_mut;
    std::list<std::vector<unsigned char> > m_queue; // request queue
    std::mutex m_pending_mut;
    bool m_pending; // whether there's a request pending

    std::unique_ptr<asio::deadline_timer> m_retry_timer;

    /** \brief Enqueue a buffer for transmission
     *
     * The given buffer will be cloned, queued, and asynchronously transmitted.
     */
    void enqueue_send(const asio::const_buffer& buf);

    /** \brief Handle a successful buffer transmission */
    void on_transmit_succeed();

    /** \brief Handle a failed buffer transmission */
    void on_transmit_fail();

    /** \brief Try to transmit a buffer
     *
     * This must be implemented by the child class. It should attempt to
     * asynchronously transmit a buffer. Upon successful transmission, the
     * child should call on_transmit_succeed, and on failure it should call
     * on_transmit_fail.
     *
     * This method should return false *only* when the send is known immediately
     * to have failed.
     *
     * \return Whether the send was successfully attempted.
     */
    virtual bool attempt_send(const asio::const_buffer buf)=0;

private:
    void perform_attempt();
};

class UDPTarget : public SocketTarget {
public:
    UDPTarget(std::string host, std::string port,
            std::shared_ptr<asio::io_service> svc = nullptr);
    ~UDPTarget();

    void write(const std::string& data);

protected:
    bool attempt_send(const asio::const_buffer buf);

private:
    std::unique_ptr<ip::udp::socket> m_sock;
    ip::udp::endpoint m_tgt;
};

class TCPTarget : public SocketTarget {
public:
    TCPTarget(std::string host, std::string port,
            std::shared_ptr<asio::io_service> svc = nullptr);
    ~TCPTarget();

    void write(const std::string& data);

protected:
    bool attempt_send(const asio::const_buffer buf);

private:
    std::unique_ptr<ip::tcp::socket> m_sock;
    ip::tcp::endpoint m_tgt;

    std::mutex m_connected_mtx;
    bool m_connected;

    void handle_connect_result(const boost::system::error_code& ec);
};

};
#endif
