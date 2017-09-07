#include <boost/bind.hpp>
#include <mutex>

#include "network.hpp"

using namespace std;
using namespace mdump;

AsioDumpTarget::AsioDumpTarget(std::shared_ptr<asio::io_service> svc) {
    if(nullptr == svc) {
        m_svc = shared_ptr<asio::io_service>(new asio::io_service());
        m_work = unique_ptr<asio::io_service::work>(
                new asio::io_service::work(*m_svc));
        m_svc_thread = unique_ptr<thread>(new thread{[this]() {m_svc->run();}});
    } else {
        m_svc = svc;
        m_work = unique_ptr<asio::io_service::work>(
                new asio::io_service::work(*m_svc));
    }
}

shared_ptr<asio::io_service>& AsioDumpTarget::get_service() {
    return m_svc;
}

SocketTarget::SocketTarget(shared_ptr<asio::io_service> svc) {
    m_pending = false;
    m_retry_timer = unique_ptr<asio::deadline_timer>(
            new asio::deadline_timer(*m_svc));
    m_retry_timer->expires_from_now(boost::posix_time::seconds(0));
}

void SocketTarget::enqueue_send(const asio::const_buffer& buf) {
    // create a private copy
    vector<unsigned char> data(asio::buffer_size(buf));
    asio::buffer_copy(asio::buffer(data), buf);
    
    // enqueue the work item
    {
        lock_guard<mutex> lck(m_queue_mut);
        m_queue.push_back(std::move(data));
    }
    perform_attempt();
}

void SocketTarget::perform_attempt() {
    {
        lock_guard<mutex> plock(m_pending_mut);
        if(!m_pending) {
            {
                lock_guard<mutex> qlock(m_queue_mut);
                if(m_queue.size() == 0) return;
                if(!attempt_send(asio::buffer(m_queue.front()))) {
                    if(m_retry_timer->expires_from_now().is_negative())
                        m_retry_timer->expires_from_now(
                                boost::posix_time::seconds(1));
                    m_retry_timer->async_wait(
                            [this](const boost::system::error_code& ec) {
                                this->perform_attempt();
                            });
                } else {
                    m_pending = true;
                }
            }
        }
    }
}

void SocketTarget::on_transmit_succeed() {
    // advance to next
    {
        lock_guard<mutex> plock(m_pending_mut);
        lock_guard<mutex> qlock(m_queue_mut);
        m_pending = false;
        m_queue.pop_front();
    }
    perform_attempt();
}

void SocketTarget::on_transmit_fail() {
    // wait a short time and retry
    {
        lock_guard<mutex> lock(m_pending_mut);
        m_pending = false;
    }

    if(m_retry_timer->expires_from_now().is_negative())
        m_retry_timer->expires_from_now(boost::posix_time::milliseconds(250));
    m_retry_timer->async_wait([this](const boost::system::error_code& ec) {
                perform_attempt();
            });
}

UDPTarget::UDPTarget(string host, string port,
        shared_ptr<asio::io_service> svc) : SocketTarget(svc) {
    ip::udp::resolver resolver(*m_svc);
    m_tgt = *resolver.resolve(ip::udp::resolver::query(host, port));
    m_sock = unique_ptr<ip::udp::socket>(
            new ip::udp::socket(*m_svc, ip::udp::v4()));
}

UDPTarget::~UDPTarget() {
    m_sock->close(); // force close the socket
}

void UDPTarget::write(const string& data) {
    enqueue_send(asio::buffer(data));
}

bool UDPTarget::attempt_send(const asio::const_buffer buf) {
    m_sock->async_send_to(asio::buffer(buf), m_tgt,
            [this](const boost::system::error_code& ec, size_t n) {
                if(ec) this->on_transmit_fail();
                else this->on_transmit_succeed();
            });
    return true;
}

TCPTarget::TCPTarget(string host, string port,
        shared_ptr<asio::io_service> svc) : SocketTarget(svc) {
    ip::tcp::resolver resolver(*m_svc);
    m_tgt = *resolver.resolve(ip::tcp::resolver::query(host, port));
    m_sock = unique_ptr<ip::tcp::socket>(
            new ip::tcp::socket(*m_svc, ip::tcp::v4()));
    m_sock->async_connect(m_tgt,
            boost::bind(&TCPTarget::handle_connect_result, this, _1));
}

TCPTarget::~TCPTarget() {
    m_sock->close(); // force close the socket
}

void TCPTarget::write(const string& data) {
    enqueue_send(asio::buffer(data));
}

bool TCPTarget::attempt_send(const asio::const_buffer buf) {
    {
        lock_guard<mutex> l(m_connected_mtx);
        if(!m_connected) return false;
    }
    asio::async_write(*m_sock, asio::buffer(buf),
            [this](const boost::system::error_code& ec, size_t n) {
                if(ec) {
                    if(ec == asio::error::connection_reset ||
                            ec == asio::error::broken_pipe) {
                        printf("Connection lost - trying to reconnect...\n");
                        // try to reconnect
                        {
                            lock_guard<mutex> l(m_connected_mtx);
                            m_connected = false;
                            m_sock->async_connect(m_tgt, boost::bind(
                                        &TCPTarget::handle_connect_result,
                                        this, _1));
                        }
                    } else {
                        printf("Err: %s\n", ec.message().c_str());
                    }
                    this->on_transmit_fail();
                } else {
                    this->on_transmit_succeed();
                }
            });
    return true;
}

void TCPTarget::handle_connect_result(const boost::system::error_code& ec) {
    if(!ec) {
        lock_guard<mutex> l(m_connected_mtx);
        m_connected = true;
        return;
    }

    // connection failure. try again in a bit.
    printf("Connection failed - retrying...\n");
    if(m_retry_timer->expires_from_now().is_negative())
        m_retry_timer->expires_from_now(boost::posix_time::seconds(5));
    m_retry_timer->async_wait([this](const boost::system::error_code& ec) {
                m_sock->async_connect(m_tgt, boost::bind(
                            &TCPTarget::handle_connect_result, this, _1)); });
}
