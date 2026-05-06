#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cctype>
#include <deque>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "SlamExecutionLayer.h"

namespace {

enum class ActionType {
    Forward,
    TurnLeft,
    TurnRight,
    Stop,
    Zero,
    Invalid,
};

enum class CommandType {
    Realtime,
    ConfigStart,
    ConfigStop,
    SlamImageInput,
    Health,
    Invalid,
};

enum class DropPolicy {
    Reject,
    DropOldest,
    DropNewest,
};

enum class RowChannelMode {
    R,
    G,
    B,
    Gray,
};

enum class PackMode {
    Binary,
    DebugHex,
};

enum class SlamStatus : std::uint8_t {
    Normal = 0,
    Overloaded = 1,
    ExecutorTimeout = 2,
    ExecutorError = 3,
};

struct ControlConfig {
    float soft_limit_hz = 50.0f;
    float max_power = 70.0f;
    float left_gain = 1.0f;
    float right_gain = 1.0f;
    float left_trim = 0.0f;
    float right_trim = 0.0f;
};

struct SlamConfig {
    int max_fps = 10;
    int exec_timeout_ms = 60;
    int max_groups = 8;
    int min_quality = 10;
    DropPolicy drop_policy = DropPolicy::DropNewest;
    float row_ratio = 0.333333f;
    RowChannelMode channel_mode = RowChannelMode::G;
    int sample_stride = 1;
    int max_rows = 1;
    PackMode pack_mode = PackMode::Binary;
};

struct SlamImageFrame {
    std::uint32_t seq = 0;
    std::uint64_t tx_ms = 0;
    std::uint32_t frame_id = 0;
    int width = 0;
    int height = 0;
    std::string pixel_fmt;
    bool keyframe = false;
    int quality_hint = 0;
    std::string payload_ref;
    bool is_row_feature = false;
    int row_index = -1;
    std::string channel_mode;
    int stride = 1;
    int sample_count = 0;
    int payload_len = 0;
    std::uint32_t payload_crc32 = 0;
};

struct SlamOutput {
    std::uint8_t control_state = 0;
    SlamStatus slam_status = SlamStatus::Normal;
    std::vector<std::uint32_t> groups;
    int quality_score = 0;
    std::uint32_t proc_ms = 0;
    std::uint64_t source_ts = 0;
};

struct SlamFusionState {
    std::uint64_t last_input_ts_ms = 0;
    std::uint32_t last_proc_ms = 0;
    std::uint32_t dropped_frames = 0;
    int last_quality_score = 0;
    std::uint32_t output_seq = 0;
};

struct ParsedCommand {
    CommandType type = CommandType::Invalid;
    std::uint32_t seq = 0;
    std::uint64_t tx_ms = 0;
    ActionType action = ActionType::Invalid;
    ControlConfig control_cfg;
    SlamConfig slam_cfg;
    SlamImageFrame frame;
    std::string parse_error;
};

struct SessionState {
    bool active = false;
    std::uint32_t session_id = 0;
    std::uint32_t config_version = 0;
    ControlConfig control_cfg;
    SlamConfig slam_cfg;
    std::uint64_t last_rt_rx_ms = 0;
    bool has_last_rt = false;
    std::uint64_t last_sli_rx_ms = 0;
    bool has_last_sli = false;
    SlamFusionState slam_fusion;
};

struct RuntimeMetrics {
    std::uint64_t ack_total = 0;
    std::uint64_t ack_ok = 0;
    std::uint64_t ack_err = 0;
    std::uint64_t sli_drop = 0;
    std::uint64_t executor_timeout = 0;
    std::deque<std::uint64_t> ack_down_ms_samples;
};

struct ExecutorResult {
    bool ok = false;
    bool timeout = false;
    int quality_score = 0;
    std::uint32_t proc_ms = 0;
    std::vector<std::uint32_t> groups;
};

class ISlamExecutorClient {
public:
    virtual ~ISlamExecutorClient() = default;
    virtual bool PushConfig(std::uint32_t session_id,
                            std::uint32_t config_version,
                            const ControlConfig& control_cfg,
                            const SlamConfig& slam_cfg) = 0;
    virtual ExecutorResult ProcessFrame(std::uint32_t session_id,
                                        const SlamImageFrame& frame,
                                        std::uint32_t timeout_ms) = 0;
    virtual bool StopSession(std::uint32_t session_id) = 0;
    virtual bool GetHealth() = 0;
    virtual bool InitializeD435i(std::string* error) = 0;
    virtual void ShutdownD435i() = 0;
    virtual bool CaptureD435iFrame(std::uint32_t timeout_ms,
                                   const SlamConfig& slam_cfg,
                                   slam_exec::D435iFrameSample* out,
                                   std::string* error) = 0;
    virtual bool IsD435iReady() const = 0;
};

class SlamExecutorBridgeClient : public ISlamExecutorClient {
public:
    bool PushConfig(std::uint32_t session_id,
                    std::uint32_t config_version,
                    const ControlConfig&,
                    const SlamConfig& slam_cfg) override {
        return bridge_.PushConfig(session_id, config_version, toBridgeConfig(slam_cfg));
    }

    ExecutorResult ProcessFrame(std::uint32_t session_id,
                                const SlamImageFrame& frame,
                                std::uint32_t timeout_ms) override {
        const slam_exec::ExecutorResult bridge_result =
            bridge_.ProcessFrame(session_id, toBridgeFrame(frame), timeout_ms);
        return fromBridgeResult(bridge_result);
    }

    bool StopSession(std::uint32_t session_id) override {
        return bridge_.StopSession(session_id);
    }

    bool GetHealth() override {
        return bridge_.GetHealth();
    }

    bool InitializeD435i(std::string* error) override {
        return bridge_.InitializeD435i(error);
    }

    void ShutdownD435i() override {
        bridge_.ShutdownD435i();
    }

    bool CaptureD435iFrame(std::uint32_t timeout_ms,
                           const SlamConfig& slam_cfg,
                           slam_exec::D435iFrameSample* out,
                           std::string* error) override {
        return bridge_.CaptureD435iFrame(timeout_ms, toBridgeConfig(slam_cfg), out, error);
    }

    bool IsD435iReady() const override {
        return bridge_.IsD435iReady();
    }

private:
    static slam_exec::DropPolicy toBridgeDropPolicy(DropPolicy policy) {
        switch (policy) {
            case DropPolicy::Reject:
                return slam_exec::DropPolicy::Reject;
            case DropPolicy::DropOldest:
                return slam_exec::DropPolicy::DropOldest;
            case DropPolicy::DropNewest:
                return slam_exec::DropPolicy::DropNewest;
        }
        return slam_exec::DropPolicy::DropNewest;
    }

    static slam_exec::RowChannelMode toBridgeChannelMode(RowChannelMode mode) {
        switch (mode) {
            case RowChannelMode::R:
                return slam_exec::RowChannelMode::R;
            case RowChannelMode::G:
                return slam_exec::RowChannelMode::G;
            case RowChannelMode::B:
                return slam_exec::RowChannelMode::B;
            case RowChannelMode::Gray:
                return slam_exec::RowChannelMode::Gray;
        }
        return slam_exec::RowChannelMode::G;
    }

    static slam_exec::PackMode toBridgePackMode(PackMode mode) {
        switch (mode) {
            case PackMode::Binary:
                return slam_exec::PackMode::Binary;
            case PackMode::DebugHex:
                return slam_exec::PackMode::DebugHex;
        }
        return slam_exec::PackMode::Binary;
    }

    static slam_exec::SlamConfig toBridgeConfig(const SlamConfig& cfg) {
        slam_exec::SlamConfig out;
        out.max_fps = cfg.max_fps;
        out.exec_timeout_ms = cfg.exec_timeout_ms;
        out.max_groups = cfg.max_groups;
        out.min_quality = cfg.min_quality;
        out.drop_policy = toBridgeDropPolicy(cfg.drop_policy);
        out.row_ratio = cfg.row_ratio;
        out.channel_mode = toBridgeChannelMode(cfg.channel_mode);
        out.sample_stride = cfg.sample_stride;
        out.max_rows = cfg.max_rows;
        out.pack_mode = toBridgePackMode(cfg.pack_mode);
        return out;
    }

    static slam_exec::SlamImageFrame toBridgeFrame(const SlamImageFrame& frame) {
        slam_exec::SlamImageFrame out;
        out.seq = frame.seq;
        out.tx_ms = frame.tx_ms;
        out.frame_id = frame.frame_id;
        out.width = frame.width;
        out.height = frame.height;
        out.pixel_fmt = frame.pixel_fmt;
        out.keyframe = frame.keyframe;
        out.quality_hint = frame.quality_hint;
        out.payload_ref = frame.payload_ref;
        out.is_row_feature = frame.is_row_feature;
        out.row_index = frame.row_index;
        out.channel_mode = frame.channel_mode;
        out.stride = frame.stride;
        out.sample_count = frame.sample_count;
        out.payload_len = frame.payload_len;
        out.payload_crc32 = frame.payload_crc32;
        return out;
    }

    static ExecutorResult fromBridgeResult(const slam_exec::ExecutorResult& in) {
        ExecutorResult out;
        out.ok = in.ok;
        out.timeout = in.timeout;
        out.quality_score = in.quality_score;
        out.proc_ms = in.proc_ms;
        out.groups = in.groups;
        return out;
    }

    slam_exec::SlamExecutionLayerClient bridge_;
};

class ControlDownlink {
public:
    bool sendAction(ActionType action, const ControlConfig& cfg) {
        (void)cfg;
        const auto duty_pair = actionToDutyNs(action);
        std::cerr << "DOWNLINK action=" << actionToString(action)
                  << " left_duty_ns=" << duty_pair.first
                  << " right_duty_ns=" << duty_pair.second << std::endl;
        return true;
    }

    bool sendZero() {
        std::cerr << "DOWNLINK action=ZERO" << std::endl;
        return true;
    }

    std::string buildSlamFrame(std::uint32_t seq, const SlamOutput& output) const {
        std::ostringstream oss;
        oss << "SL(seq=" << seq
            << ",ctrl=" << static_cast<int>(output.control_state)
            << ",status=" << static_cast<int>(output.slam_status)
            << ",quality=" << output.quality_score
            << ",proc_ms=" << output.proc_ms
            << ",source_ts=" << output.source_ts
            << ",groups=" << output.groups.size() << '[';
        for (std::size_t i = 0; i < output.groups.size(); ++i) {
            if (i > 0) {
                oss << '|';
            }
            oss << "0x" << std::hex << std::uppercase << output.groups[i] << std::dec;
        }
        oss << "])";
        return oss.str();
    }

private:
    static std::pair<int, int> actionToDutyNs(ActionType action) {
        constexpr int kDutyOffNs = 0;
        constexpr int kDutyOnNs = 20000000;
        switch (action) {
            case ActionType::Forward:
                return {kDutyOnNs, kDutyOnNs};
            case ActionType::TurnLeft:
                return {kDutyOffNs, kDutyOnNs};
            case ActionType::TurnRight:
                return {kDutyOnNs, kDutyOffNs};
            case ActionType::Stop:
            case ActionType::Zero:
            case ActionType::Invalid:
            default:
                return {kDutyOffNs, kDutyOffNs};
        }
    }

    static std::string actionToString(ActionType action) {
        switch (action) {
            case ActionType::Forward:
                return "FORWARD";
            case ActionType::TurnLeft:
                return "LEFT";
            case ActionType::TurnRight:
                return "RIGHT";
            case ActionType::Stop:
                return "STOP";
            case ActionType::Zero:
                return "ZERO";
            default:
                return "INVALID";
        }
    }
};

std::uint64_t nowMs() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(now).count());
}

bool parseUInt32(const std::string& token, std::uint32_t* out) {
    if (out == nullptr || token.empty()) {
        return false;
    }
    std::istringstream iss(token);
    std::uint32_t v = 0;
    iss >> v;
    if (!iss || !iss.eof()) {
        return false;
    }
    *out = v;
    return true;
}

bool parseUInt64(const std::string& token, std::uint64_t* out) {
    if (out == nullptr || token.empty()) {
        return false;
    }
    std::istringstream iss(token);
    std::uint64_t v = 0;
    iss >> v;
    if (!iss || !iss.eof()) {
        return false;
    }
    *out = v;
    return true;
}

bool parseInt(const std::string& token, int* out) {
    if (out == nullptr || token.empty()) {
        return false;
    }
    std::istringstream iss(token);
    int v = 0;
    iss >> v;
    if (!iss || !iss.eof()) {
        return false;
    }
    *out = v;
    return true;
}

bool parseFloat(const std::string& token, float* out) {
    if (out == nullptr || token.empty()) {
        return false;
    }
    std::istringstream iss(token);
    float v = 0.0f;
    iss >> v;
    if (!iss || !iss.eof()) {
        return false;
    }
    *out = v;
    return true;
}

std::optional<ActionType> parseActionToken(const std::string& token) {
    if (token == "F") {
        return ActionType::Forward;
    }
    if (token == "L") {
        return ActionType::TurnLeft;
    }
    if (token == "R") {
        return ActionType::TurnRight;
    }
    if (token == "S") {
        return ActionType::Stop;
    }
    return std::nullopt;
}

std::optional<DropPolicy> parseDropPolicy(const std::string& token) {
    if (token == "reject") {
        return DropPolicy::Reject;
    }
    if (token == "oldest") {
        return DropPolicy::DropOldest;
    }
    if (token == "newest") {
        return DropPolicy::DropNewest;
    }
    return std::nullopt;
}

std::optional<RowChannelMode> parseRowChannelMode(const std::string& token) {
    if (token == "R") {
        return RowChannelMode::R;
    }
    if (token == "G") {
        return RowChannelMode::G;
    }
    if (token == "B") {
        return RowChannelMode::B;
    }
    if (token == "GRAY") {
        return RowChannelMode::Gray;
    }
    return std::nullopt;
}

std::optional<PackMode> parsePackMode(const std::string& token) {
    if (token == "bin") {
        return PackMode::Binary;
    }
    if (token == "hex") {
        return PackMode::DebugHex;
    }
    return std::nullopt;
}

bool parseKVToken(const std::string& token, std::string* key, std::string* value) {
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

ParsedCommand parseCommand(const std::string& line) {
    ParsedCommand cmd;

    std::istringstream iss(line);
    std::string head;
    if (!(iss >> head)) {
        cmd.parse_error = "empty";
        return cmd;
    }

    if (head == "R") {
        std::string action_token;
        if (!(iss >> cmd.seq >> cmd.tx_ms >> action_token)) {
            cmd.parse_error = "rt_missing_fields";
            return cmd;
        }
        const auto action = parseActionToken(action_token);
        if (!action.has_value()) {
            cmd.parse_error = "rt_bad_action";
            return cmd;
        }
        cmd.type = CommandType::Realtime;
        cmd.action = *action;
        return cmd;
    }

    if (head == "HEALTH") {
        cmd.type = CommandType::Health;
        return cmd;
    }

    if (head == "SLI") {
        std::string seq_token;
        std::string tx_token;
        if (!(iss >> seq_token >> tx_token)) {
            cmd.parse_error = "sli_missing_header";
            return cmd;
        }
        if (!parseUInt32(seq_token, &cmd.seq) || !parseUInt64(tx_token, &cmd.tx_ms)) {
            cmd.parse_error = "sli_bad_header";
            return cmd;
        }

        SlamImageFrame frame;
        frame.seq = cmd.seq;
        frame.tx_ms = cmd.tx_ms;

        std::string kv;
        while (iss >> kv) {
            std::string key;
            std::string value;
            if (!parseKVToken(kv, &key, &value)) {
                cmd.parse_error = "sli_bad_kv";
                return cmd;
            }
            if (key == "frame_id") {
                if (!parseUInt32(value, &frame.frame_id)) {
                    cmd.parse_error = "sli_bad_frame_id";
                    return cmd;
                }
            } else if (key == "width") {
                if (!parseInt(value, &frame.width)) {
                    cmd.parse_error = "sli_bad_width";
                    return cmd;
                }
            } else if (key == "height") {
                if (!parseInt(value, &frame.height)) {
                    cmd.parse_error = "sli_bad_height";
                    return cmd;
                }
            } else if (key == "pixel_fmt") {
                frame.pixel_fmt = value;
            } else if (key == "keyframe") {
                int keyframe_i = 0;
                if (!parseInt(value, &keyframe_i)) {
                    cmd.parse_error = "sli_bad_keyframe";
                    return cmd;
                }
                frame.keyframe = (keyframe_i != 0);
            } else if (key == "quality_hint") {
                if (!parseInt(value, &frame.quality_hint)) {
                    cmd.parse_error = "sli_bad_quality";
                    return cmd;
                }
            } else if (key == "payload_ref") {
                frame.payload_ref = value;
            } else if (key == "feature") {
                frame.is_row_feature = (value == "row");
                if (!frame.is_row_feature) {
                    cmd.parse_error = "sli_bad_feature";
                    return cmd;
                }
            } else if (key == "row_index") {
                if (!parseInt(value, &frame.row_index)) {
                    cmd.parse_error = "sli_bad_row_index";
                    return cmd;
                }
            } else if (key == "channel_mode") {
                if (!parseRowChannelMode(value).has_value()) {
                    cmd.parse_error = "sli_bad_channel_mode";
                    return cmd;
                }
                frame.channel_mode = value;
            } else if (key == "stride") {
                if (!parseInt(value, &frame.stride)) {
                    cmd.parse_error = "sli_bad_stride";
                    return cmd;
                }
            } else if (key == "sample_count") {
                if (!parseInt(value, &frame.sample_count)) {
                    cmd.parse_error = "sli_bad_sample_count";
                    return cmd;
                }
            } else if (key == "payload_len") {
                if (!parseInt(value, &frame.payload_len)) {
                    cmd.parse_error = "sli_bad_payload_len";
                    return cmd;
                }
            } else if (key == "payload_crc32") {
                if (!parseUInt32(value, &frame.payload_crc32)) {
                    cmd.parse_error = "sli_bad_payload_crc32";
                    return cmd;
                }
            } else {
                cmd.parse_error = "sli_unknown_key";
                return cmd;
            }
        }

        if (frame.frame_id == 0 || frame.width <= 0 || frame.height <= 0 ||
            frame.pixel_fmt.empty() || frame.payload_ref.empty()) {
            cmd.parse_error = "sli_incomplete";
            return cmd;
        }
        if (frame.is_row_feature) {
            if (frame.row_index < 0 || frame.sample_count < 0 || frame.payload_len < 0 ||
                frame.channel_mode.empty() || frame.stride <= 0) {
                cmd.parse_error = "sli_row_incomplete";
                return cmd;
            }
        }
        cmd.frame = frame;
        cmd.type = CommandType::SlamImageInput;
        return cmd;
    }

    if (head == "C") {
        std::string sub;
        if (!(iss >> sub)) {
            cmd.parse_error = "cfg_missing_sub";
            return cmd;
        }

        if (sub == "STOP") {
            std::string kv;
            while (iss >> kv) {
                std::string key;
                std::string value;
                if (!parseKVToken(kv, &key, &value)) {
                    cmd.parse_error = "cfg_stop_bad_kv";
                    return cmd;
                }
                if (key == "seq") {
                    if (!parseUInt32(value, &cmd.seq)) {
                        cmd.parse_error = "cfg_stop_bad_seq";
                        return cmd;
                    }
                } else if (key == "ts") {
                    if (!parseUInt64(value, &cmd.tx_ms)) {
                        cmd.parse_error = "cfg_stop_bad_ts";
                        return cmd;
                    }
                } else {
                    cmd.parse_error = "cfg_stop_unknown_key";
                    return cmd;
                }
            }
            cmd.type = CommandType::ConfigStop;
            return cmd;
        }

        if (sub == "START") {
            std::string kv;
            while (iss >> kv) {
                std::string key;
                std::string value;
                if (!parseKVToken(kv, &key, &value)) {
                    cmd.parse_error = "cfg_start_bad_kv";
                    return cmd;
                }
                if (key == "seq") {
                    if (!parseUInt32(value, &cmd.seq)) {
                        cmd.parse_error = "cfg_start_bad_seq";
                        return cmd;
                    }
                } else if (key == "ts") {
                    if (!parseUInt64(value, &cmd.tx_ms)) {
                        cmd.parse_error = "cfg_start_bad_ts";
                        return cmd;
                    }
                } else if (key == "soft_hz") {
                    if (!parseFloat(value, &cmd.control_cfg.soft_limit_hz)) {
                        cmd.parse_error = "cfg_start_bad_soft_hz";
                        return cmd;
                    }
                } else if (key == "max_power") {
                    if (!parseFloat(value, &cmd.control_cfg.max_power)) {
                        cmd.parse_error = "cfg_start_bad_max_power";
                        return cmd;
                    }
                } else if (key == "left_gain") {
                    if (!parseFloat(value, &cmd.control_cfg.left_gain)) {
                        cmd.parse_error = "cfg_start_bad_left_gain";
                        return cmd;
                    }
                } else if (key == "right_gain") {
                    if (!parseFloat(value, &cmd.control_cfg.right_gain)) {
                        cmd.parse_error = "cfg_start_bad_right_gain";
                        return cmd;
                    }
                } else if (key == "left_trim") {
                    if (!parseFloat(value, &cmd.control_cfg.left_trim)) {
                        cmd.parse_error = "cfg_start_bad_left_trim";
                        return cmd;
                    }
                } else if (key == "right_trim") {
                    if (!parseFloat(value, &cmd.control_cfg.right_trim)) {
                        cmd.parse_error = "cfg_start_bad_right_trim";
                        return cmd;
                    }
                } else if (key == "slam_max_fps") {
                    if (!parseInt(value, &cmd.slam_cfg.max_fps)) {
                        cmd.parse_error = "cfg_start_bad_slam_max_fps";
                        return cmd;
                    }
                } else if (key == "slam_timeout_ms") {
                    if (!parseInt(value, &cmd.slam_cfg.exec_timeout_ms)) {
                        cmd.parse_error = "cfg_start_bad_slam_timeout_ms";
                        return cmd;
                    }
                } else if (key == "slam_max_groups") {
                    if (!parseInt(value, &cmd.slam_cfg.max_groups)) {
                        cmd.parse_error = "cfg_start_bad_slam_max_groups";
                        return cmd;
                    }
                } else if (key == "slam_min_quality") {
                    if (!parseInt(value, &cmd.slam_cfg.min_quality)) {
                        cmd.parse_error = "cfg_start_bad_slam_min_quality";
                        return cmd;
                    }
                } else if (key == "slam_drop_policy") {
                    const auto policy = parseDropPolicy(value);
                    if (!policy.has_value()) {
                        cmd.parse_error = "cfg_start_bad_slam_drop_policy";
                        return cmd;
                    }
                    cmd.slam_cfg.drop_policy = *policy;
                } else if (key == "row_ratio") {
                    if (!parseFloat(value, &cmd.slam_cfg.row_ratio)) {
                        cmd.parse_error = "cfg_start_bad_row_ratio";
                        return cmd;
                    }
                } else if (key == "channel_mode") {
                    const auto channel = parseRowChannelMode(value);
                    if (!channel.has_value()) {
                        cmd.parse_error = "cfg_start_bad_channel_mode";
                        return cmd;
                    }
                    cmd.slam_cfg.channel_mode = *channel;
                } else if (key == "sample_stride") {
                    if (!parseInt(value, &cmd.slam_cfg.sample_stride)) {
                        cmd.parse_error = "cfg_start_bad_sample_stride";
                        return cmd;
                    }
                } else if (key == "max_rows") {
                    if (!parseInt(value, &cmd.slam_cfg.max_rows)) {
                        cmd.parse_error = "cfg_start_bad_max_rows";
                        return cmd;
                    }
                } else if (key == "pack_mode") {
                    const auto mode = parsePackMode(value);
                    if (!mode.has_value()) {
                        cmd.parse_error = "cfg_start_bad_pack_mode";
                        return cmd;
                    }
                    cmd.slam_cfg.pack_mode = *mode;
                } else {
                    cmd.parse_error = "cfg_start_unknown_key";
                    return cmd;
                }
            }
            cmd.type = CommandType::ConfigStart;
            return cmd;
        }

        cmd.parse_error = "cfg_bad_sub";
        return cmd;
    }

    cmd.parse_error = "bad_head";
    return cmd;
}

class MainProcessor {
public:
    explicit MainProcessor(ControlDownlink downlink)
        : downlink_(std::move(downlink)) {}

    std::string onCommData(const std::string& payload) {
        const std::uint64_t rx_ms = nowMs();
        const ParsedCommand cmd = parseCommand(payload);
        if (cmd.type == CommandType::Invalid) {
            return buildAck(false, 0, rx_ms, 0, 0, "bad_command", cmd.parse_error);
        }

        if (cmd.type == CommandType::ConfigStart) {
            return onConfigStart(cmd, rx_ms);
        }
        if (cmd.type == CommandType::ConfigStop) {
            return onConfigStop(cmd, rx_ms);
        }
        if (cmd.type == CommandType::Realtime) {
            return onRealtime(cmd, rx_ms);
        }
        if (cmd.type == CommandType::SlamImageInput) {
            return onSlamImageInput(cmd, rx_ms);
        }
        if (cmd.type == CommandType::Health) {
            return onHealth();
        }

        return emitAck(false, cmd.seq, rx_ms, cmd.tx_ms, 0, "bad_command", "unsupported");
    }

private:
    std::string onConfigStart(const ParsedCommand& cmd, std::uint64_t rx_ms) {
        if (cmd.control_cfg.soft_limit_hz <= 0.0f || cmd.control_cfg.soft_limit_hz > kHardLimitHz) {
            return emitAck(false, cmd.seq, rx_ms, cmd.tx_ms, 0, "cfg_reject", "bad_soft_hz");
        }
        if (cmd.slam_cfg.max_fps <= 0 || cmd.slam_cfg.max_fps > kSlamHardMaxFps ||
            cmd.slam_cfg.exec_timeout_ms <= 0 || cmd.slam_cfg.exec_timeout_ms > kSlamHardMaxTimeoutMs ||
            cmd.slam_cfg.max_groups <= 0 || cmd.slam_cfg.max_groups > kSlamHardMaxGroups) {
            return emitAck(false, cmd.seq, rx_ms, cmd.tx_ms, 0, "cfg_reject", "bad_slam_cfg");
        }
        if (cmd.slam_cfg.row_ratio < 0.0f || cmd.slam_cfg.row_ratio > 1.0f ||
            cmd.slam_cfg.sample_stride <= 0 || cmd.slam_cfg.sample_stride > kSlamHardMaxStride ||
            cmd.slam_cfg.max_rows <= 0 || cmd.slam_cfg.max_rows > kSlamHardMaxRows) {
            return emitAck(false, cmd.seq, rx_ms, cmd.tx_ms, 0, "cfg_reject", "bad_row_cfg");
        }

        // Only mark session active AFTER both executor config and downlink succeed
        session_.session_id += 1;
        session_.config_version += 1;
        session_.control_cfg = cmd.control_cfg;
        session_.slam_cfg = cmd.slam_cfg;
        session_.has_last_rt = false;
        session_.has_last_sli = false;
        session_.slam_fusion = SlamFusionState{};

        const std::uint64_t t0 = nowMs();
        const bool push_ok = slam_executor_.PushConfig(session_.session_id,
                                                       session_.config_version,
                                                       session_.control_cfg,
                                                       session_.slam_cfg);
        const bool zero_ok = downlink_.sendAction(ActionType::Zero, session_.control_cfg);
        const std::uint64_t down_ms = nowMs() - t0;

        if (!push_ok) {
            // Config push failed, session remains inactive, state is recoverable
            return emitAck(false, cmd.seq, rx_ms, cmd.tx_ms, down_ms,
                           "cfg_slam_downlink_fail", "push_config_failed");
        }
        if (!zero_ok) {
            // Downlink failed, session remains inactive, state is recoverable
            return emitAck(false, cmd.seq, rx_ms, cmd.tx_ms, down_ms,
                           "cfg_downlink_fail", "zero_cmd_failed");
        }

        // Both succeeded, now mark session active
        session_.active = true;
        return emitAck(true, cmd.seq, rx_ms, cmd.tx_ms, down_ms,
                       "cfg_start", "session_active");
    }

    std::string onConfigStop(const ParsedCommand& cmd, std::uint64_t rx_ms) {
        const std::uint64_t t0 = nowMs();
        const bool slam_ok = session_.active ? slam_executor_.StopSession(session_.session_id) : true;
        const bool zero_ok = downlink_.sendZero();
        const std::uint64_t down_ms = nowMs() - t0;

        if (!slam_ok) {
            // Executor stop failed, keep session active so STOP can be retried
            return emitAck(false, cmd.seq, rx_ms, cmd.tx_ms, down_ms, "cfg_stop", "executor_stop_failed");
        }
        if (!zero_ok) {
            // Downlink failed, keep session active so STOP can be retried
            return emitAck(false, cmd.seq, rx_ms, cmd.tx_ms, down_ms, "cfg_stop", "zero_cmd_failed");
        }

        // Both succeeded, now mark session inactive
        session_.active = false;
        session_.has_last_rt = false;
        session_.has_last_sli = false;

        return emitAck(true, cmd.seq, rx_ms, cmd.tx_ms, down_ms,
                       "cfg_stop", "session_closed");
    }

    std::string onRealtime(const ParsedCommand& cmd, std::uint64_t rx_ms) {
        if (!session_.active) {
            return emitAck(false, cmd.seq, rx_ms, cmd.tx_ms, 0, "rt_reject", "no_session");
        }

        const std::string rate_err = checkRealtimeRate(rx_ms);
        if (!rate_err.empty()) {
            return emitAck(false, cmd.seq, rx_ms, cmd.tx_ms, 0, "rt_reject", rate_err);
        }

        const std::uint64_t t0 = nowMs();
        const bool ok = downlink_.sendAction(cmd.action, session_.control_cfg);
        const std::uint64_t down_ms = nowMs() - t0;
        session_.last_rt_rx_ms = rx_ms;
        session_.has_last_rt = true;

        if (ok) {
            return emitAck(true, cmd.seq, rx_ms, cmd.tx_ms, down_ms,
                       "rt_apply", "action_sent");
        } else {
            return emitAck(false, cmd.seq, rx_ms, cmd.tx_ms, down_ms,
                       "rt_fail", "downlink_error");
        }
    }

    std::string onSlamImageInput(const ParsedCommand& cmd, std::uint64_t rx_ms) {
        const bool is_row_feature = cmd.frame.is_row_feature;
        const std::string fail_tag = is_row_feature ? "slam_row_fail" : "sli_fail";

        if (!session_.active) {
            metrics_.sli_drop += 1;
            return emitAck(false, cmd.seq, rx_ms, cmd.tx_ms, 0,
                           fail_tag, "no_session");
        }
        if (!isSupportedPixelFormat(cmd.frame.pixel_fmt)) {
            metrics_.sli_drop += 1;
            return emitAck(false, cmd.seq, rx_ms, cmd.tx_ms, 0,
                           fail_tag, "bad_pixel_fmt");
        }

        const std::string gate = checkSlamIngressBudget(rx_ms);
        if (!gate.empty()) {
            session_.slam_fusion.dropped_frames += 1;
            metrics_.sli_drop += 1;
            return emitAck(false, cmd.seq, rx_ms, cmd.tx_ms, 0,
                           fail_tag, gate);
        }

        SlamOutput output;
        const std::uint64_t t0 = nowMs();
        const ExecutorResult exec_result = slam_executor_.ProcessFrame(
            session_.session_id,
            cmd.frame,
            static_cast<std::uint32_t>(session_.slam_cfg.exec_timeout_ms));
        const std::uint64_t down_ms = nowMs() - t0;

        output.control_state = session_.active ? 1 : 0;
        output.proc_ms = exec_result.proc_ms > 0 ? exec_result.proc_ms : static_cast<std::uint32_t>(down_ms);
        output.source_ts = cmd.tx_ms;
        output.quality_score = exec_result.quality_score;
        output.groups = exec_result.groups;

        if (!exec_result.ok) {
            output.slam_status = exec_result.timeout ? SlamStatus::ExecutorTimeout : SlamStatus::ExecutorError;
            session_.slam_fusion.dropped_frames += 1;
            metrics_.sli_drop += 1;
            if (exec_result.timeout) {
                metrics_.executor_timeout += 1;
            }
            updateSlamFusionState(cmd, output);
            return emitAck(false, cmd.seq, rx_ms, cmd.tx_ms, down_ms,
                           fail_tag, exec_result.timeout ? "executor_timeout" : "executor_error");
        }

        output.slam_status = SlamStatus::Normal;

        // If realtime commands are very close, keep control latency priority and only return status.
        if (session_.has_last_rt && rx_ms - session_.last_rt_rx_ms <= kRtPriorityWindowMs) {
            output.slam_status = SlamStatus::Overloaded;
            output.groups.clear();
            updateSlamFusionState(cmd, output);
            session_.last_sli_rx_ms = rx_ms;
            session_.has_last_sli = true;
            // Overload degradation: frame deferred due to RT priority
            return emitAck(false, cmd.seq, rx_ms, cmd.tx_ms, down_ms,
                           fail_tag, "overload_rt_priority");
        }

        const std::string slam_frame = downlink_.buildSlamFrame(cmd.seq, output);
        updateSlamFusionState(cmd, output);
        session_.last_sli_rx_ms = rx_ms;
        session_.has_last_sli = true;
        const std::string ok_tag = is_row_feature ? "slam_row_ok" : "sli_ok";
        return emitAck(true, cmd.seq, rx_ms, cmd.tx_ms, down_ms,
                       ok_tag, slam_frame);
    }

    std::string onHealth() const {
        const std::uint64_t p50 = percentile(metrics_.ack_down_ms_samples, 50.0);
        const std::uint64_t p95 = percentile(metrics_.ack_down_ms_samples, 95.0);
        const std::uint64_t p99 = percentile(metrics_.ack_down_ms_samples, 99.0);
        const bool rollback = shouldRollback();
        std::ostringstream oss;
        oss << "HEALTH"
            << " session_active=" << (session_.active ? 1 : 0)
            << " session_id=" << session_.session_id
            << " config_version=" << session_.config_version
            << " ack_total=" << metrics_.ack_total
            << " ack_ok=" << metrics_.ack_ok
            << " ack_err=" << metrics_.ack_err
            << " ack_p50_ms=" << p50
            << " ack_p95_ms=" << p95
            << " ack_p99_ms=" << p99
            << " sli_dropped=" << metrics_.sli_drop
            << " executor_timeout=" << metrics_.executor_timeout
            << " rollback_recommended=" << (rollback ? 1 : 0);
        return oss.str();
    }

    void updateSlamFusionState(const ParsedCommand& cmd, const SlamOutput& output) {
        session_.slam_fusion.last_input_ts_ms = cmd.tx_ms;
        session_.slam_fusion.last_proc_ms = output.proc_ms;
        session_.slam_fusion.last_quality_score = output.quality_score;
        session_.slam_fusion.output_seq = cmd.seq;
    }

    bool isSupportedPixelFormat(const std::string& pixel_fmt) const {
        return pixel_fmt == "GRAY8" || pixel_fmt == "RGB24" || pixel_fmt == "NV12" || pixel_fmt == "ROW1";
    }

    std::string checkRealtimeRate(std::uint64_t rx_ms) const {
        if (!session_.has_last_rt) {
            return "";
        }
        if (rx_ms <= session_.last_rt_rx_ms) {
            // Hard rejection: time sequence violated
            return "reject_rate_non_monotonic";
        }
        const float delta_ms = static_cast<float>(rx_ms - session_.last_rt_rx_ms);
        const float hz = 1000.0f / delta_ms;
        if (hz > kHardLimitHz + 1e-3f) {
            // Hard rejection: exceeds hard limit (100Hz max)
            return "reject_rate_hard";
        }
        if (hz > session_.control_cfg.soft_limit_hz + 1e-3f) {
            // Soft limit exceeded: user configured limit, still reject but for configuration reason
            return "reject_rate_soft";
        }
        return "";
    }

    std::string checkSlamIngressBudget(std::uint64_t rx_ms) const {
        if (!session_.has_last_sli) {
            return "";
        }
        if (rx_ms <= session_.last_sli_rx_ms) {
            // Hard rejection: time sequence violated
            return "reject_non_monotonic";
        }

        const double delta_ms = static_cast<double>(rx_ms - session_.last_sli_rx_ms);
        const double hz = 1000.0 / delta_ms;
        if (hz <= static_cast<double>(session_.slam_cfg.max_fps)) {
            return "";
        }

        // FPS budget exceeded: behavior depends on drop policy
        if (session_.slam_cfg.drop_policy == DropPolicy::Reject) {
            // Hard rejection: refuse to accept
            return "reject_fps_budget";
        }
        if (session_.slam_cfg.drop_policy == DropPolicy::DropNewest) {
            // Overload degradation: drop incoming frame
            return "overload_budget_newest";
        }
        // Overload degradation: drop oldest buffered frame
        return "overload_budget_oldest";
    }

    std::string emitAck(bool ok,
                        std::uint32_t seq,
                        std::uint64_t rx_ms,
                        std::uint64_t tx_ms,
                        std::uint64_t down_ms,
                        const std::string& tag,
                        const std::string& detail) {
        metrics_.ack_total += 1;
        if (ok) {
            metrics_.ack_ok += 1;
        } else {
            metrics_.ack_err += 1;
        }
        metrics_.ack_down_ms_samples.push_back(down_ms);
        if (metrics_.ack_down_ms_samples.size() > kAckSamplesWindow) {
            metrics_.ack_down_ms_samples.pop_front();
        }

        return buildAck(ok, seq, rx_ms, tx_ms, down_ms, tag, detail);
    }

    static std::string buildAck(bool ok,
                                std::uint32_t seq,
                                std::uint64_t rx_ms,
                                std::uint64_t tx_ms,
                                std::uint64_t down_ms,
                                const std::string& tag,
                                const std::string& detail) {
        const std::uint64_t up_ms = (tx_ms <= rx_ms) ? (rx_ms - tx_ms) : 0;
        std::ostringstream oss;
        oss << "ACK " << (ok ? "OK" : "ERR")
            << " seq=" << seq
            << " up_ms=" << up_ms
            << " down_ms=" << down_ms
            << " tag=" << sanitizeAckValue(tag)
            << " detail=" << sanitizeAckValue(detail);
        return oss.str();
    }

    static std::string sanitizeAckValue(const std::string& value) {
        if (value.empty()) {
            return "none";
        }
        std::string out;
        out.reserve(value.size());
        for (const unsigned char ch : value) {
            out.push_back(std::isspace(ch) ? '_' : static_cast<char>(ch));
        }
        return out;
    }

    static std::uint64_t percentile(const std::deque<std::uint64_t>& samples, double p) {
        if (samples.empty()) {
            return 0;
        }
        std::vector<std::uint64_t> sorted(samples.begin(), samples.end());
        std::sort(sorted.begin(), sorted.end());
        const double rank = (p / 100.0) * static_cast<double>(sorted.size() - 1);
        return sorted[static_cast<std::size_t>(rank)];
    }

    bool shouldRollback() const {
        // Require sufficient sample history before making rollback recommendation
        if (metrics_.ack_total < 100) {
            return false;
        }

        // Check error rate: > 5% indicates degradation
        const double err_rate = static_cast<double>(metrics_.ack_err) / static_cast<double>(metrics_.ack_total);
        if (err_rate > 0.05) {
            return true;
        }

        // Check p95 latency: > 100ms indicates processing bottleneck (based on executor timeout of ~60ms)
        const std::uint64_t p95 = percentile(metrics_.ack_down_ms_samples, 95.0);
        if (p95 > 100) {
            return true;
        }

        // Check timeout rate: any sustained timeout pattern is concerning
        if (metrics_.executor_timeout > 0 && 
            metrics_.executor_timeout > metrics_.ack_total / 50) {  // > 2% timeout rate
            return true;
        }

        return false;
    }

    static constexpr float kHardLimitHz = 100.0f;
    static constexpr int kSlamHardMaxFps = 30;
    static constexpr int kSlamHardMaxTimeoutMs = 200;
    static constexpr int kSlamHardMaxGroups = 64;
    static constexpr int kSlamHardMaxStride = 64;
    static constexpr int kSlamHardMaxRows = 8;
    static constexpr std::uint64_t kRtPriorityWindowMs = 20;
    static constexpr std::size_t kAckSamplesWindow = 256;

    SessionState session_;
    RuntimeMetrics metrics_;
    ControlDownlink downlink_;
    SlamExecutorBridgeClient slam_executor_;
};

void printUsage() {
    std::cout << "Main Processor Hub (Step2)\n"
              << "Self-test:\n"
              << "  --d435i-selftest\n"
              << "Config start:\n"
              << "  C START seq=<n> ts=<ms> soft_hz=<1..100> max_power=<v> left_gain=<v> right_gain=<v> left_trim=<v> right_trim=<v> slam_max_fps=<1..30> slam_timeout_ms=<1..200> slam_max_groups=<1..64> slam_min_quality=<0..100> slam_drop_policy=<reject|oldest|newest>\n"
              << "Realtime:\n"
              << "  R <seq> <tx_ms> <F|L|R>\n"
              << "SLAM image input:\n"
              << "  SLI <seq> <tx_ms> frame_id=<n> width=<w> height=<h> pixel_fmt=<GRAY8|RGB24|NV12> keyframe=<0|1> quality_hint=<0..100> payload_ref=<id>\n"
              << "Health:\n"
              << "  HEALTH\n"
              << "Config stop:\n"
              << "  C STOP seq=<n> ts=<ms>\n"
              << "Quit:\n"
              << "  q\n";
}

int runD435iSelfTest() {
    constexpr std::uint32_t kSessionId = 1u;
    constexpr std::uint32_t kConfigVersion = 1u;
    constexpr std::uint32_t kTimeoutMs = 80u;
    constexpr int kWarmupFrames = 5;
    constexpr std::uint32_t kTestDelayMs = 10000u;

    ControlConfig control_cfg;
    SlamConfig slam_cfg;
    slam_cfg.max_fps = 10;
    slam_cfg.exec_timeout_ms = static_cast<int>(kTimeoutMs);
    slam_cfg.row_ratio = 0.333333f;
    slam_cfg.sample_stride = 1;

    SlamExecutorBridgeClient client;
    std::string error;
    if (!client.InitializeD435i(&error)) {
        std::cerr << "D435i init failed: " << error << std::endl;
        return 1;
    }

    if (!client.PushConfig(kSessionId, kConfigVersion, control_cfg, slam_cfg)) {
        std::cerr << "PushConfig failed" << std::endl;
        client.ShutdownD435i();
        return 2;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(kTestDelayMs));

    for (int i = 0; i < kWarmupFrames; ++i) {
        slam_exec::D435iFrameSample warmup_sample;
        std::string warmup_error;
        client.CaptureD435iFrame(kTimeoutMs, slam_cfg, &warmup_sample, &warmup_error);
    }

    slam_exec::D435iFrameSample sample;
    error.clear();
    if (!client.CaptureD435iFrame(kTimeoutMs, slam_cfg, &sample, &error)) {
        std::cerr << "CaptureD435iFrame failed: " << error << std::endl;
        client.StopSession(kSessionId);
        client.ShutdownD435i();
        return 3;
    }

    SlamImageFrame frame;
    frame.seq = 0;
    frame.tx_ms = nowMs();
    frame.frame_id = 1;
    frame.width = static_cast<int>(sample.color_width);
    frame.height = static_cast<int>(sample.color_height);
    frame.pixel_fmt = "BGR8";
    frame.keyframe = true;
    frame.quality_hint = 60;
    frame.payload_ref = "d435i_selftest";

    const ExecutorResult result = client.ProcessFrame(kSessionId, frame, kTimeoutMs);
    std::cout << "selftest capture rgb_bytes=" << sample.rgb_row.size()
              << " depth_values=" << sample.depth_row.size()
              << " result_ok=" << (result.ok ? "true" : "false")
              << " timeout=" << (result.timeout ? "true" : "false")
              << " quality=" << result.quality_score
              << " groups=" << result.groups.size()
              << " proc_ms=" << result.proc_ms
              << std::endl;

    client.StopSession(kSessionId);
    client.ShutdownD435i();
    return result.ok ? 0 : 4;
}

}  // namespace

int main(int argc, char** argv) {
    const bool stdio_mode = argc > 1 && std::string(argv[1]) == "--stdio";
    if (argc > 1 && std::string(argv[1]) == "--d435i-selftest") {
        return runD435iSelfTest();
    }
    ControlDownlink downlink;
    MainProcessor processor(std::move(downlink));

    if (!stdio_mode) {
        printUsage();
    }
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "q" || line == "Q") {
            break;
        }
        std::cout << processor.onCommData(line) << std::endl;
    }
    return 0;
}
