#include <algorithm>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <deque>
#include <iomanip>
#include <iostream>
#include <netinet/in.h>
#include <optional>
#include <sstream>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

#include <arpa/inet.h>

namespace {

constexpr std::uint16_t kDefaultListenPort = 19520;
constexpr std::uint16_t kDefaultProcessorPort = 19521;
constexpr std::uint16_t kListenPorts[] = {19520, 9773, 11514, 23758, 52019};

struct ParsedMessage {
    bool valid = false;
    std::string normalized_payload;

    std::uint32_t seq = 0;
    std::uint64_t client_ts_ms = 0;
    bool has_client_ts = false;
    std::string trace_id = "gateway-local";

    std::string error;
};

struct GatewaySwitches {
    bool legacy_alias_enabled = true;
    bool sli_enabled = true;
    std::uint32_t route_timeout_ms = 50;
};

struct GatewayMetrics {
    std::uint64_t rx_total = 0;
    std::uint64_t parse_fail = 0;
    std::uint64_t route_timeout = 0;
    std::uint64_t sli_drop = 0;
    std::uint64_t ack_ok = 0;
    std::uint64_t ack_err = 0;
    std::deque<double> ack_dl_ms_samples;
};

bool sendAll(int fd, const std::string& payload);

class ProcessorClient {
public:
    ProcessorClient(std::string host, std::uint16_t port, std::uint32_t timeout_ms)
        : host_(std::move(host)), port_(port), timeout_ms_(timeout_ms) {}

    bool dispatch(const std::string& normalized_payload, std::string* ack, std::string* error) const {
        if (ack == nullptr) {
            if (error != nullptr) {
                *error = "ack_output_null";
            }
            return false;
        }

        int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) {
            if (error != nullptr) {
                *error = std::string("processor_socket_failed:") + std::strerror(errno);
            }
            return false;
        }

        timeval tv {};
        tv.tv_sec = static_cast<time_t>(timeout_ms_ / 1000u);
        tv.tv_usec = static_cast<suseconds_t>((timeout_ms_ % 1000u) * 1000u);
        (void)::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        (void)::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

        sockaddr_in addr {};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port_);
        if (::inet_pton(AF_INET, host_.c_str(), &addr.sin_addr) != 1) {
            if (error != nullptr) {
                *error = "processor_bad_host";
            }
            ::close(fd);
            return false;
        }

        if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            if (error != nullptr) {
                *error = std::string("processor_connect_failed:") + std::strerror(errno);
            }
            ::close(fd);
            return false;
        }

        const std::string outbound = normalized_payload + "\n";
        if (!sendAll(fd, outbound)) {
            if (error != nullptr) {
                *error = std::string("processor_send_failed:") + std::strerror(errno);
            }
            ::close(fd);
            return false;
        }

        std::string buffer;
        char ch = '\0';
        while (true) {
            const ssize_t received = ::recv(fd, &ch, 1, 0);
            if (received > 0) {
                if (ch == '\n') {
                    break;
                }
                if (ch != '\r') {
                    buffer.push_back(ch);
                }
                if (buffer.size() > kMaxProcessorAckBytes) {
                    if (error != nullptr) {
                        *error = "processor_ack_too_large";
                    }
                    ::close(fd);
                    return false;
                }
                continue;
            }
            if (received == 0) {
                if (error != nullptr) {
                    *error = "processor_closed_without_ack";
                }
                ::close(fd);
                return false;
            }
            if (errno == EINTR) {
                continue;
            }
            if (error != nullptr) {
                *error = std::string("processor_recv_failed:") + std::strerror(errno);
            }
            ::close(fd);
            return false;
        }

        ::close(fd);
        *ack = buffer;
        return true;
    }

    std::string endpointLabel() const {
        std::ostringstream oss;
        oss << host_ << ':' << port_;
        return oss.str();
    }

private:
    static constexpr std::size_t kMaxProcessorAckBytes = 262144;

    std::string host_;
    std::uint16_t port_ = kDefaultProcessorPort;
    std::uint32_t timeout_ms_ = 1000;
};

class CommunicationLayer {
public:
    explicit CommunicationLayer(ProcessorClient processor_client)
        : processor_client_(std::move(processor_client)) {}

    std::string onReceive(const std::string& line) {
        const auto recv_tp = std::chrono::steady_clock::now();
        const std::uint64_t recv_ms = nowMs(recv_tp);

        std::string mgmt_response;
        if (tryHandleGatewayCommand(line, recv_ms, recv_tp, &mgmt_response)) {
            return mgmt_response;
        }

        metrics_.rx_total += 1;

        const std::string trace_id = nextTraceId();
        const ParsedMessage msg = parseLine(line, switches_);
        ParsedMessage traced_msg = msg;
        traced_msg.trace_id = trace_id;

        if (!traced_msg.valid) {
            metrics_.parse_fail += 1;
            if (traced_msg.error == "sli_disabled") {
                metrics_.sli_drop += 1;
            }
            const std::string ack = buildAck(false, traced_msg.seq, 0, 0,
                                             "gw_bad_msg", traced_msg.error,
                                             trace_id, "parse_reject");
            recordAck(false, 0.0);
            return ack;
        }

        const auto route_begin = std::chrono::steady_clock::now();
        std::string processor_ack;
        std::string route_error;
        const bool dispatched = processor_client_.dispatch(traced_msg.normalized_payload, &processor_ack, &route_error);
        const double route_ms = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - route_begin).count() / 1000.0;
        if (!dispatched) {
            metrics_.route_timeout += 1;
            const std::string ack = buildAck(false, traced_msg.seq, 0, 0,
                                             "gw_route_fail", route_error.empty() ? "processor_unavailable" : route_error,
                                             trace_id, "route_fail");
            recordAck(false, 0.0);
            return ack;
        }
        if (route_ms > static_cast<double>(switches_.route_timeout_ms)) {
            metrics_.route_timeout += 1;
            const std::string ack = buildAck(false, traced_msg.seq, 0, 0,
                                             "gw_route_timeout", "timeout",
                                             trace_id, "route_timeout");
            recordAck(false, 0.0);
            return ack;
        }

        // Parse processor ACK and append gateway context
        ProcessorAckFields processor_fields = parseProcessorAck(processor_ack);
        const std::string gw_ack = buildAck(processor_fields.ok, processor_fields.seq,
                                             processor_fields.up_ms, processor_fields.down_ms,
                                             processor_fields.tag, processor_fields.detail,
                                             trace_id, "forwarded");
        recordAck(processor_fields.ok, static_cast<double>(processor_fields.down_ms));
        return gw_ack;
    }

private:
    static std::uint64_t nowMs(const std::chrono::steady_clock::time_point& tp) {
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(tp.time_since_epoch()).count());
    }

    static bool parseUInt64(const std::string& s, std::uint64_t* out) {
        if (out == nullptr || s.empty()) {
            return false;
        }
        std::istringstream iss(s);
        std::uint64_t v = 0;
        iss >> v;
        if (!iss || !iss.eof()) {
            return false;
        }
        *out = v;
        return true;
    }

    static bool parseUInt32(const std::string& s, std::uint32_t* out) {
        if (out == nullptr || s.empty()) {
            return false;
        }
        std::istringstream iss(s);
        std::uint32_t v = 0;
        iss >> v;
        if (!iss || !iss.eof()) {
            return false;
        }
        *out = v;
        return true;
    }

    static bool parseInt(const std::string& s, int* out) {
        if (out == nullptr || s.empty()) {
            return false;
        }
        std::istringstream iss(s);
        int v = 0;
        iss >> v;
        if (!iss || !iss.eof()) {
            return false;
        }
        *out = v;
        return true;
    }

    static bool parseKVToken(const std::string& token, std::string* key, std::string* value) {
        if (key == nullptr || value == nullptr) {
            return false;
        }
        const std::size_t pos = token.find('=');
        if (pos == std::string::npos || pos == 0 || pos + 1 >= token.size()) {
            return false;
        }
        *key = token.substr(0, pos);
        *value = token.substr(pos + 1);
        return true;
    }

    static ParsedMessage parseLine(const std::string& line, const GatewaySwitches& switches) {
        ParsedMessage msg;

        std::istringstream iss(line);
        std::string mode;
        if (!(iss >> mode)) {
            msg.error = "empty";
            return msg;
        }

        if (mode == "R") {
            // Realtime short frame: R <seq> <F|L|R|S> [client_ts_ms]
            std::string seq_token;
            std::string action_token;
            if (!(iss >> seq_token >> action_token)) {
                msg.error = "rt_short_fields";
                return msg;
            }
            if (!parseUInt32(seq_token, &msg.seq)) {
                msg.error = "rt_bad_seq";
                return msg;
            }
            if (!(action_token == "F" || action_token == "L" || action_token == "R" || action_token == "S")) {
                msg.error = "rt_bad_action";
                return msg;
            }

            std::string ts_token;
            if (iss >> ts_token) {
                if (!parseUInt64(ts_token, &msg.client_ts_ms)) {
                    msg.error = "rt_bad_ts";
                    return msg;
                }
                msg.has_client_ts = true;
            }

            std::ostringstream normalized;
            normalized << "R " << msg.seq << ' ' << (msg.has_client_ts ? msg.client_ts_ms : 0) << ' ' << action_token;
            msg.normalized_payload = normalized.str();
            msg.valid = true;
            return msg;
        }

        if (mode == "RT") {
            if (!switches.legacy_alias_enabled) {
                msg.error = "alias_disabled";
                return msg;
            }
            std::string seq_token;
            std::string action_token;
            if (!(iss >> seq_token >> action_token)) {
                msg.error = "rt_short_fields";
                return msg;
            }
            if (!parseUInt32(seq_token, &msg.seq)) {
                msg.error = "rt_bad_seq";
                return msg;
            }
            if (!(action_token == "F" || action_token == "L" || action_token == "R" || action_token == "S")) {
                msg.error = "rt_bad_action";
                return msg;
            }

            std::string ts_token;
            if (iss >> ts_token) {
                if (!parseUInt64(ts_token, &msg.client_ts_ms)) {
                    msg.error = "rt_bad_ts";
                    return msg;
                }
                msg.has_client_ts = true;
            }

            std::ostringstream normalized;
            normalized << "R " << msg.seq << ' ' << (msg.has_client_ts ? msg.client_ts_ms : 0) << ' ' << action_token;
            msg.normalized_payload = normalized.str();
            msg.valid = true;
            return msg;
        }

        if (mode == "SLI") {
            if (!switches.sli_enabled) {
                msg.error = "sli_disabled";
                return msg;
            }
            // SLAM image ingest frame: SLI <seq> <tx_ms> key=value...
            std::string seq_token;
            std::string tx_token;
            if (!(iss >> seq_token >> tx_token)) {
                msg.error = "sl_short_fields";
                return msg;
            }
            if (!parseUInt32(seq_token, &msg.seq) || !parseUInt64(tx_token, &msg.client_ts_ms)) {
                msg.error = "sl_bad_header";
                return msg;
            }

            bool has_frame_id = false;
            bool has_size = false;
            std::ostringstream normalized;
            normalized << "SLI " << msg.seq << ' ' << msg.client_ts_ms;
            std::string kv;
            while (iss >> kv) {
                std::string key;
                std::string value;
                if (!parseKVToken(kv, &key, &value)) {
                    msg.error = "sli_bad_kv";
                    return msg;
                }
                if (key == "frame_id") {
                    std::uint32_t frame_id = 0;
                    if (!parseUInt32(value, &frame_id) || frame_id == 0) {
                        msg.error = "sli_bad_frame_id";
                        return msg;
                    }
                    has_frame_id = true;
                }
                if (key == "width" || key == "height") {
                    int v = 0;
                    if (!parseInt(value, &v) || v <= 0) {
                        msg.error = "sli_bad_shape";
                        return msg;
                    }
                    has_size = true;
                }
                normalized << ' ' << key << '=' << value;
            }
            if (!has_frame_id || !has_size) {
                msg.error = "sli_incomplete";
                return msg;
            }
            msg.normalized_payload = normalized.str();
            msg.has_client_ts = true;
            msg.valid = true;
            return msg;
        }

        if (mode == "SL") {
            if (!switches.legacy_alias_enabled) {
                msg.error = "alias_disabled";
                return msg;
            }
            if (!switches.sli_enabled) {
                msg.error = "sli_disabled";
                return msg;
            }
            std::string seq_token;
            std::string tx_token;
            if (!(iss >> seq_token >> tx_token)) {
                msg.error = "sl_missing_header";
                return msg;
            }
            if (!parseUInt32(seq_token, &msg.seq) || !parseUInt64(tx_token, &msg.client_ts_ms)) {
                msg.error = "sl_bad_header";
                return msg;
            }

            bool has_frame_id = false;
            bool has_size = false;
            std::ostringstream normalized;
            normalized << "SLI " << msg.seq << ' ' << msg.client_ts_ms;

            std::string kv;
            while (iss >> kv) {
                std::string key;
                std::string value;
                if (!parseKVToken(kv, &key, &value)) {
                    msg.error = "sl_bad_kv";
                    return msg;
                }
                if (key == "frame_id") {
                    std::uint32_t frame_id = 0;
                    if (!parseUInt32(value, &frame_id) || frame_id == 0) {
                        msg.error = "sl_bad_frame_id";
                        return msg;
                    }
                    has_frame_id = true;
                }
                if (key == "width" || key == "height") {
                    int v = 0;
                    if (!parseInt(value, &v) || v <= 0) {
                        msg.error = "sl_bad_shape";
                        return msg;
                    }
                    has_size = true;
                }
                normalized << ' ' << key << '=' << value;
            }
            if (!has_frame_id || !has_size) {
                msg.error = "sl_incomplete";
                return msg;
            }
            msg.normalized_payload = normalized.str();
            msg.has_client_ts = true;
            msg.valid = true;
            return msg;
        }

        if (mode == "CS" || mode == "CE") {
            if (!switches.legacy_alias_enabled) {
                msg.error = "alias_disabled";
                return msg;
            }
            std::ostringstream normalized;
            normalized << (mode == "CS" ? "C START" : "C STOP");

            std::string kv;
            while (iss >> kv) {
                std::string key;
                std::string value;
                if (!parseKVToken(kv, &key, &value)) {
                    msg.error = "cfg_alias_bad_kv";
                    return msg;
                }
                if (key == "seq") {
                    if (!parseUInt32(value, &msg.seq)) {
                        msg.error = "cfg_alias_bad_seq";
                        return msg;
                    }
                } else if (key == "ts") {
                    if (!parseUInt64(value, &msg.client_ts_ms)) {
                        msg.error = "cfg_alias_bad_ts";
                        return msg;
                    }
                    msg.has_client_ts = true;
                }
                normalized << ' ' << key << '=' << value;
            }
            msg.normalized_payload = normalized.str();
            msg.valid = true;
            return msg;
        }

        if (mode == "C") {
            // Config long frame family: C START ... / C STOP ...
            std::string sub;
            if (!(iss >> sub)) {
                msg.error = "cfg_no_sub";
                return msg;
            }

            if (sub == "STOP") {
                std::ostringstream normalized;
                normalized << "C STOP";

                std::string kv;
                while (iss >> kv) {
                    std::string key;
                    std::string value;
                    if (!parseKVToken(kv, &key, &value)) {
                        msg.error = "cfg_stop_bad_kv";
                        return msg;
                    }
                    if (key == "seq") {
                        if (!parseUInt32(value, &msg.seq)) {
                            msg.error = "cfg_stop_bad_seq";
                            return msg;
                        }
                    } else if (key == "ts") {
                        if (!parseUInt64(value, &msg.client_ts_ms)) {
                            msg.error = "cfg_stop_bad_ts";
                            return msg;
                        }
                        msg.has_client_ts = true;
                    }
                    normalized << ' ' << key << '=' << value;
                }
                msg.normalized_payload = normalized.str();
                msg.valid = true;
                return msg;
            }

            if (sub == "START") {
                std::ostringstream normalized;
                normalized << "C START";
                std::string kv;
                while (iss >> kv) {
                    std::string key;
                    std::string value;
                    if (!parseKVToken(kv, &key, &value)) {
                        msg.error = "cfg_bad_kv";
                        return msg;
                    }
                    if (key == "seq") {
                        if (!parseUInt32(value, &msg.seq)) {
                            msg.error = "cfg_bad_seq";
                            return msg;
                        }
                    }
                    if (key == "ts") {
                        if (!parseUInt64(value, &msg.client_ts_ms)) {
                            msg.error = "cfg_bad_ts";
                            return msg;
                        }
                        msg.has_client_ts = true;
                    }
                    normalized << ' ' << key << '=' << value;
                }
                msg.normalized_payload = normalized.str();
                msg.valid = true;
                return msg;
            }

            msg.error = "cfg_bad_sub";
            return msg;
        }

        msg.error = "bad_mode";
        return msg;
    }

    struct ProcessorAckFields {
        bool ok = false;
        std::uint32_t seq = 0;
        std::uint64_t up_ms = 0;
        std::uint64_t down_ms = 0;
        std::string tag = "unknown";
        std::string detail = "parse_error";
    };

    static ProcessorAckFields parseProcessorAck(const std::string& ack_str) {
        ProcessorAckFields fields;
        std::istringstream iss(ack_str);

        std::string ack_kw;
        std::string status;
        if (!(iss >> ack_kw >> status) || ack_kw != "ACK" ||
            !(status == "OK" || status == "ERR")) {
            fields.tag = "gw_bad_processor_ack";
            fields.detail = "invalid_ack_header";
            return fields;
        }

        fields.ok = (status == "OK");

        std::string kv;
        while (iss >> kv) {
            std::string key, value;
            if (!parseKVToken(kv, &key, &value)) {
                continue;
            }
            if (key == "seq") {
                parseUInt32(value, &fields.seq);
            } else if (key == "up_ms") {
                parseUInt64(value, &fields.up_ms);
            } else if (key == "down_ms") {
                parseUInt64(value, &fields.down_ms);
            } else if (key == "tag") {
                fields.tag = value;
            } else if (key == "detail") {
                fields.detail = value;
            }
        }
        return fields;
    }

    static std::string sanitizeAckValue(const std::string& value) {
        std::string out;
        out.reserve(value.size());
        for (const unsigned char ch : value) {
            if (std::isspace(ch) || std::iscntrl(ch)) {
                out.push_back('_');
            } else {
                out.push_back(static_cast<char>(ch));
            }
        }
        return out.empty() ? "none" : out;
    }

    static std::string buildAck(bool ok,
                                std::uint32_t seq,
                                std::uint64_t up_ms,
                                std::uint64_t down_ms,
                                const std::string& tag,
                                const std::string& detail,
                                const std::string& gw_trace,
                                const std::string& gw_route) {
        std::ostringstream oss;
        oss << "ACK " << (ok ? "OK" : "ERR")
            << " seq=" << seq
            << " up_ms=" << up_ms
            << " down_ms=" << down_ms
            << " tag=" << sanitizeAckValue(tag)
            << " detail=" << sanitizeAckValue(detail)
            << " gw_trace=" << sanitizeAckValue(gw_trace)
            << " gw_route=" << sanitizeAckValue(gw_route);
        return oss.str();
    }

    static double computeUplinkMs(const ParsedMessage& msg, std::uint64_t recv_ms) {
        if (!msg.has_client_ts || recv_ms < msg.client_ts_ms) {
            return -1.0;
        }
        return static_cast<double>(recv_ms - msg.client_ts_ms);
    }

    static bool parseOnOff(const std::string& value, bool* out) {
        if (out == nullptr) {
            return false;
        }
        if (value == "on") {
            *out = true;
            return true;
        }
        if (value == "off") {
            *out = false;
            return true;
        }
        return false;
    }

    std::string nextTraceId() {
        std::ostringstream oss;
        oss << "gw-" << ++trace_counter_;
        return oss.str();
    }

    void recordAck(bool ok, double downlink_ms) {
        if (ok) {
            metrics_.ack_ok += 1;
        } else {
            metrics_.ack_err += 1;
        }
        // Only record downlink_ms if > 0 (gateway processing delay)
        if (downlink_ms > 0.0) {
            metrics_.ack_dl_ms_samples.push_back(downlink_ms);
            if (metrics_.ack_dl_ms_samples.size() > kMaxAckSamples) {
                metrics_.ack_dl_ms_samples.pop_front();
            }
        }
    }

    static double percentile(const std::deque<double>& samples, double p) {
        if (samples.empty()) {
            return 0.0;
        }
        std::vector<double> sorted(samples.begin(), samples.end());
        std::sort(sorted.begin(), sorted.end());
        const double rank = (p / 100.0) * static_cast<double>(sorted.size() - 1);
        const std::size_t idx = static_cast<std::size_t>(rank);
        return sorted[idx];
    }

    bool rollbackRecommended() const {
        if (metrics_.rx_total < 20) {
            return false;
        }
        const double parse_fail_rate = static_cast<double>(metrics_.parse_fail) / static_cast<double>(metrics_.rx_total);
        const double route_timeout_rate = static_cast<double>(metrics_.route_timeout) / static_cast<double>(metrics_.rx_total);
        return parse_fail_rate > 0.10 || route_timeout_rate > 0.05;
    }

    std::string buildHealthSnapshot() const {
        const double ack_p95 = percentile(metrics_.ack_dl_ms_samples, 95.0);
        const double ack_p99 = percentile(metrics_.ack_dl_ms_samples, 99.0);
        std::ostringstream oss;
        oss << "GW HEALTH"
            << " legacy_alias=" << (switches_.legacy_alias_enabled ? "on" : "off")
            << " sli_enabled=" << (switches_.sli_enabled ? "on" : "off")
            << " route_timeout_ms=" << switches_.route_timeout_ms
            << " rx_total=" << metrics_.rx_total
            << " parse_fail=" << metrics_.parse_fail
            << " route_timeout=" << metrics_.route_timeout
            << " sli_drop=" << metrics_.sli_drop
            << " ack_ok=" << metrics_.ack_ok
            << " ack_err=" << metrics_.ack_err
            << " ack_p95_ms=" << std::fixed << std::setprecision(2) << ack_p95
            << " ack_p99_ms=" << ack_p99
            << " rollback_recommended=" << (rollbackRecommended() ? 1 : 0);
        return oss.str();
    }

    bool tryHandleGatewayCommand(const std::string& line,
                                 std::uint64_t recv_ms,
                                 const std::chrono::steady_clock::time_point& recv_tp,
                                 std::string* response) {
        (void)recv_ms;
        (void)recv_tp;
        if (response == nullptr) {
            return false;
        }

        std::istringstream iss(line);
        std::string h0;
        std::string h1;
        if (!(iss >> h0)) {
            return false;
        }
        if (h0 != "GW") {
            return false;
        }
        if (!(iss >> h1)) {
            *response = buildAck(false, 0, 0, 0, "gw_bad_cmd", "cmd_missing",
                                 nextTraceId(), "mgmt");
            recordAck(false, 0.0);
            return true;
        }

        if (h1 == "HEALTH") {
            *response = buildHealthSnapshot();
            return true;
        }

        if (h1 == "ROLLBACK") {
            switches_.legacy_alias_enabled = true;
            switches_.sli_enabled = true;
            switches_.route_timeout_ms = 80;
            *response = buildAck(true, 0, 0, 0, "gw_rollback", "ok",
                                 nextTraceId(), "mgmt");
            recordAck(true, 0.0);
            return true;
        }

        if (h1 == "SWITCH") {
            std::string kv;
            bool saw_any = false;
            while (iss >> kv) {
                std::string key;
                std::string value;
                if (!parseKVToken(kv, &key, &value)) {
                    *response = buildAck(false, 0, 0, 0, "gw_bad_switch", "bad_kv",
                                         nextTraceId(), "mgmt");
                    recordAck(false, 0.0);
                    return true;
                }
                if (key == "legacy_alias") {
                    bool parsed = false;
                    if (!parseOnOff(value, &parsed)) {
                        *response = buildAck(false, 0, 0, 0, "gw_bad_switch", "bad_legacy_alias",
                                             nextTraceId(), "mgmt");
                        recordAck(false, 0.0);
                        return true;
                    }
                    switches_.legacy_alias_enabled = parsed;
                } else if (key == "sli_enabled") {
                    bool parsed = false;
                    if (!parseOnOff(value, &parsed)) {
                        *response = buildAck(false, 0, 0, 0, "gw_bad_switch", "bad_sli_enabled",
                                             nextTraceId(), "mgmt");
                        recordAck(false, 0.0);
                        return true;
                    }
                    switches_.sli_enabled = parsed;
                } else if (key == "route_timeout_ms") {
                    int timeout_ms = 0;
                    if (!parseInt(value, &timeout_ms) || timeout_ms <= 0 || timeout_ms > 5000) {
                        *response = buildAck(false, 0, 0, 0, "gw_bad_switch", "bad_route_timeout",
                                             nextTraceId(), "mgmt");
                        recordAck(false, 0.0);
                        return true;
                    }
                    switches_.route_timeout_ms = static_cast<std::uint32_t>(timeout_ms);
                } else {
                    *response = buildAck(false, 0, 0, 0, "gw_bad_switch", "unknown_key",
                                         nextTraceId(), "mgmt");
                    recordAck(false, 0.0);
                    return true;
                }
                saw_any = true;
            }

            if (!saw_any) {
                *response = buildAck(false, 0, 0, 0, "gw_bad_switch", "empty",
                                     nextTraceId(), "mgmt");
                recordAck(false, 0.0);
                return true;
            }

            *response = buildAck(true, 0, 0, 0, "gw_switch", "applied",
                                 nextTraceId(), "mgmt");
            recordAck(true, 0.0);
            return true;
        }

        *response = buildAck(false, 0, 0, 0, "gw_bad_cmd", "unknown",
                             nextTraceId(), "mgmt");
        recordAck(false, 0.0);
        return true;
    }

    ProcessorClient processor_client_;
    GatewaySwitches switches_;
    GatewayMetrics metrics_;
    std::uint64_t trace_counter_ = 0;

    static constexpr std::size_t kMaxAckSamples = 256;
};

bool sendAll(int fd, const std::string& payload) {
    std::size_t offset = 0;
    while (offset < payload.size()) {
        const ssize_t written = ::send(fd, payload.data() + offset, payload.size() - offset, 0);
        if (written > 0) {
            offset += static_cast<std::size_t>(written);
            continue;
        }
        if (written < 0 && errno == EINTR) {
            continue;
        }
        return false;
    }
    return true;
}

std::string formatPeerAddress(const sockaddr_in& addr) {
    char ip_buffer[INET_ADDRSTRLEN] = {};
    const char* ip_text = ::inet_ntop(AF_INET, &addr.sin_addr, ip_buffer, sizeof(ip_buffer));
    if (ip_text == nullptr) {
        return "unknown";
    }
    std::ostringstream oss;
    oss << ip_text << ':' << ntohs(addr.sin_port);
    return oss.str();
}

int createListenSocket(std::uint16_t port, std::string* error) {
    const int listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        if (error != nullptr) {
            *error = std::string("socket_failed: ") + std::strerror(errno);
        }
        return -1;
    }

    int reuse = 1;
    if (::setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
        if (error != nullptr) {
            *error = std::string("setsockopt_failed: ") + std::strerror(errno);
        }
        ::close(listen_fd);
        return -1;
    }

    sockaddr_in addr {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(port);

    if (::bind(listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        if (error != nullptr) {
            *error = std::string("bind_failed: ") + std::strerror(errno);
        }
        ::close(listen_fd);
        return -1;
    }

    if (::listen(listen_fd, 8) < 0) {
        if (error != nullptr) {
            *error = std::string("listen_failed: ") + std::strerror(errno);
        }
        ::close(listen_fd);
        return -1;
    }

    return listen_fd;
}

bool handleTcpClient(int client_fd, CommunicationLayer* comm, const std::string& peer_label) {
    std::cout << "TCP client connected from " << peer_label << std::endl;

    std::string buffer;
    buffer.reserve(4096);
    char chunk[4096];

    while (true) {
        const ssize_t received = ::recv(client_fd, chunk, sizeof(chunk), 0);
        if (received > 0) {
            buffer.append(chunk, static_cast<std::size_t>(received));

            std::size_t newline_pos = std::string::npos;
            while ((newline_pos = buffer.find('\n')) != std::string::npos) {
                std::string line = buffer.substr(0, newline_pos);
                buffer.erase(0, newline_pos + 1);

                if (!line.empty() && line.back() == '\r') {
                    line.pop_back();
                }

                if (line == "q" || line == "Q") {
                    return true;
                }

                const std::string ack = comm->onReceive(line);
                if (!sendAll(client_fd, ack + "\n")) {
                    std::cout << "TCP client write failed for " << peer_label << std::endl;
                    return false;
                }
            }
            continue;
        }

        if (received == 0) {
            return true;
        }

        if (errno == EINTR) {
            continue;
        }

        std::cout << "TCP client read failed for " << peer_label << ": "
                  << std::strerror(errno) << std::endl;
        return false;
    }
}

int runTcpServer(CommunicationLayer* comm, std::uint16_t port) {
    std::string error;
    int listen_fd = -1;
    for (std::uint16_t candidate_port : kListenPorts) {
        listen_fd = createListenSocket(candidate_port, &error);
        if (listen_fd >= 0) {
            port = candidate_port;
            if (candidate_port != kDefaultListenPort) {
                std::cout << "Default port " << kDefaultListenPort
                          << " unavailable; bound to fallback port " << candidate_port << std::endl;
            }
            break;
        }

        std::cerr << "Port " << candidate_port << " unavailable: " << error << std::endl;
    }

    if (listen_fd < 0) {
        std::cerr << "Failed to start TCP listener: all configured ports are occupied [";
        for (std::size_t i = 0; i < sizeof(kListenPorts) / sizeof(kListenPorts[0]); ++i) {
            std::cerr << kListenPorts[i];
            if (i + 1 < sizeof(kListenPorts) / sizeof(kListenPorts[0])) {
                std::cerr << ",";
            }
        }
        std::cerr << "]" << std::endl;
        return 1;
    }

    std::cout << "Communication layer listening on 0.0.0.0:" << port << std::endl;

    while (true) {
        sockaddr_in client_addr {};
        socklen_t client_len = sizeof(client_addr);
        const int client_fd = ::accept(listen_fd, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
        if (client_fd < 0) {
            if (errno == EINTR) {
                continue;
            }
            std::cerr << "accept_failed: " << std::strerror(errno) << std::endl;
            ::close(listen_fd);
            return 1;
        }

        const std::string peer_label = formatPeerAddress(client_addr);
        const bool ok = handleTcpClient(client_fd, comm, peer_label);
        ::close(client_fd);
        std::cout << "TCP client disconnected from " << peer_label
                  << " status=" << (ok ? "normal" : "error") << std::endl;
    }
}

void printUsage() {
    std::cout << "Communication Layer (Gateway Adapter)\n"
              << "TCP listener: 0.0.0.0:" << kDefaultListenPort << " (fallback order: 19520, 9773, 11514, 23758, 52019)\n"
              << "Processor backend: --processor-host <ip> --processor-port <port> (default 127.0.0.1:" << kDefaultProcessorPort << ")\n"
              << "Line protocol: one command per line, newline-delimited over TCP\n"
              << "Realtime short frame:\n"
              << "  R <seq> <F|L|R> [client_ts_ms]\n"
              << "  RT <seq> <F|L|R> [client_ts_ms]  # legacy alias\n"
              << "SLAM image ingest frame:\n"
              << "  SLI <seq> <tx_ms> frame_id=<n> width=<w> height=<h> pixel_fmt=<fmt> keyframe=<0|1> quality_hint=<0..100> payload_ref=<id>\n"
              << "  SL <seq> <tx_ms> frame_id=<n> width=<w> height=<h> pixel_fmt=<fmt> keyframe=<0|1> quality_hint=<0..100> payload_ref=<id>  # legacy alias\n"
              << "Config long frame:\n"
              << "  C START seq=<n> ts=<ms> soft_hz=<v> max_power=<v> left_gain=<v> right_gain=<v> left_trim=<v> right_trim=<v> slam_max_fps=<v> slam_timeout_ms=<v> slam_max_groups=<v> slam_min_quality=<v> slam_drop_policy=<reject|oldest|newest>\n"
              << "  CS seq=<n> ts=<ms> ...  # legacy alias\n"
              << "  C STOP seq=<n> ts=<ms>\n"
              << "  CE seq=<n> ts=<ms>  # legacy alias\n"
              << "ACK template:\n"
              << "  ACK <OK|ERR> seq=<n> up_ms=<n> down_ms=<n> tag=<code> detail=<code_or_payload> gw_trace=<id> gw_route=<result>\n"
              << "Gateway management:\n"
              << "  GW HEALTH\n"
              << "  GW SWITCH legacy_alias=<on|off> sli_enabled=<on|off> route_timeout_ms=<1..5000>\n"
              << "  GW ROLLBACK\n"
              << "Quit:\n"
              << "  q\n";
}

}  // namespace

bool parseIntArg(const std::string& value, int* out) {
    if (out == nullptr || value.empty()) {
        return false;
    }
    std::istringstream iss(value);
    int parsed = 0;
    iss >> parsed;
    if (!iss || !iss.eof()) {
        return false;
    }
    *out = parsed;
    return true;
}

std::optional<std::uint16_t> parsePortArg(const std::string& value) {
    int port = 0;
    if (!parseIntArg(value, &port) || port <= 0 || port > 65535) {
        return std::nullopt;
    }
    return static_cast<std::uint16_t>(port);
}

int main(int argc, char** argv) {
    std::string processor_host = "127.0.0.1";
    std::uint16_t processor_port = kDefaultProcessorPort;
    std::uint32_t processor_timeout_ms = 1000;

    if (const char* env_host = std::getenv("USV_PROCESSOR_HOST")) {
        processor_host = env_host;
    }
    if (const char* env_port = std::getenv("USV_PROCESSOR_PORT")) {
        const auto parsed = parsePortArg(env_port);
        if (parsed.has_value()) {
            processor_port = *parsed;
        }
    }

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--processor-host" && i + 1 < argc) {
            processor_host = argv[++i];
        } else if (arg == "--processor-port" && i + 1 < argc) {
            const auto parsed = parsePortArg(argv[++i]);
            if (!parsed.has_value()) {
                std::cerr << "bad --processor-port" << std::endl;
                return 2;
            }
            processor_port = *parsed;
        } else if (arg == "--processor-timeout-ms" && i + 1 < argc) {
            int timeout = 0;
            if (!parseIntArg(argv[++i], &timeout) || timeout <= 0 || timeout > 60000) {
                std::cerr << "bad --processor-timeout-ms" << std::endl;
                return 2;
            }
            processor_timeout_ms = static_cast<std::uint32_t>(timeout);
        } else if (arg == "--help" || arg == "-h") {
            printUsage();
            return 0;
        } else {
            std::cerr << "unknown argument: " << arg << std::endl;
            return 2;
        }
    }

    ProcessorClient processor_client(processor_host, processor_port, processor_timeout_ms);
    std::cout << "Gateway forwarding to processor " << processor_client.endpointLabel()
              << " timeout_ms=" << processor_timeout_ms << std::endl;
    CommunicationLayer comm(std::move(processor_client));

    printUsage();
    return runTcpServer(&comm, kDefaultListenPort);
}
