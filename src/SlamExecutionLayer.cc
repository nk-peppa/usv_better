#include "usv/SlamExecutionLayer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <optional>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <librealsense2/rs.hpp>

namespace slam_exec {
namespace {

std::uint32_t fnv1a32(const std::vector<std::uint8_t>& data) {
    std::uint32_t hash = 2166136261u;
    for (const std::uint8_t byte : data) {
        hash ^= static_cast<std::uint32_t>(byte);
        hash *= 16777619u;
    }
    return hash;
}

int clampRowIndex(float row_ratio, int height) {
    if (height <= 0) {
        return 0;
    }
    const float raw = static_cast<float>(height - 1) * row_ratio;
    const int row_index = static_cast<int>(std::lround(raw));
    return std::clamp(row_index, 0, height - 1);
}


std::vector<int> selectRowIndices(const SlamConfig& slam_cfg, int height) {
    std::vector<int> rows;
    if (height <= 0) {
        rows.push_back(0);
        return rows;
    }

    const int requested_rows = std::max(1, slam_cfg.max_rows);
    const int row_count = std::clamp(requested_rows, 1, height);
    rows.reserve(static_cast<std::size_t>(row_count));
    if (row_count == 1) {
        rows.push_back(clampRowIndex(slam_cfg.row_ratio, height));
        return rows;
    }

    for (int i = 0; i < row_count; ++i) {
        const double ratio = static_cast<double>(i) / static_cast<double>(row_count - 1);
        const int row = static_cast<int>(std::lround(ratio * static_cast<double>(height - 1)));
        if (rows.empty() || rows.back() != row) {
            rows.push_back(std::clamp(row, 0, height - 1));
        }
    }
    return rows;
}

std::string channelModeToString(RowChannelMode mode) {
    switch (mode) {
        case RowChannelMode::R:
            return "R";
        case RowChannelMode::G:
            return "G";
        case RowChannelMode::B:
            return "B";
        case RowChannelMode::Gray:
            return "GRAY";
    }
    return "G";
}

void appendFloat(std::vector<std::uint8_t>* buffer, float value) {
    static_assert(sizeof(float) == 4, "float must be 4 bytes");
    const auto* raw = reinterpret_cast<const std::uint8_t*>(&value);
    buffer->insert(buffer->end(), raw, raw + sizeof(float));
}

std::uint16_t meanDepth(const std::vector<std::uint16_t>& depths) {
    if (depths.empty()) {
        return 0;
    }
    std::uint64_t sum = 0;
    for (const std::uint16_t depth : depths) {
        sum += depth;
    }
    return static_cast<std::uint16_t>(sum / depths.size());
}

class D435iCaptureBridge {
public:
    bool ensureStarted(std::string* error) {
        if (started_) {
            return true;
        }

        struct VideoProfile {
            int width;
            int height;
            int fps;
        };

        constexpr VideoProfile kProfiles[] = {
            {424, 240, 15},
            {640, 480, 15},
            {640, 480, 30},
        };

        std::string last_error;
        for (const VideoProfile& profile : kProfiles) {
            try {
                rs2::config config;
                config.enable_stream(RS2_STREAM_COLOR, profile.width, profile.height, RS2_FORMAT_BGR8, profile.fps);
                config.enable_stream(RS2_STREAM_DEPTH, profile.width, profile.height, RS2_FORMAT_Z16, profile.fps);

                pipeline_profile_ = pipeline_.start(config);
                align_to_color_ = std::make_unique<rs2::align>(RS2_STREAM_COLOR);
                started_ = true;
                const bool imu_started = startMotionSensor(pipeline_profile_.get_device());
                last_error_.clear();
                std::cout << "D435i video stream started "
                          << profile.width << "x" << profile.height
                          << "@" << profile.fps
                          << " imu=" << (imu_started ? "on" : "off")
                          << " gyro=200" << std::endl;
                return true;
            } catch (const rs2::error& e) {
                last_error = e.what();
                try {
                    pipeline_.stop();
                } catch (...) {
                }
                align_to_color_.reset();
                started_ = false;
            }
        }

        last_error_ = last_error.empty() ? "d435i_no_supported_video_profile" : last_error;
        if (error != nullptr) {
            *error = last_error_;
        }
        return false;
    }

    void shutdown() {
        stopMotionSensor();
        if (started_) {
            try {
                pipeline_.stop();
            } catch (...) {
            }
        }
        align_to_color_.reset();
        started_ = false;
    }

    bool capture(std::uint32_t timeout_ms,
                 const SlamConfig& slam_cfg,
                 D435iFrameSample* out,
                 std::string* error) {
        if (out == nullptr) {
            return false;
        }
        *out = D435iFrameSample{};
        if (error != nullptr) {
            error->clear();
        }

        if (!ensureStarted(error)) {
            return false;
        }

        try {
            rs2::frameset frames;
            if (align_to_color_ == nullptr) {
                align_to_color_ = std::make_unique<rs2::align>(RS2_STREAM_COLOR);
            }

            if (!pipeline_.poll_for_frames(&frames)) {
                frames = pipeline_.wait_for_frames(timeout_ms);
            }
            rs2::frameset newer_frames;
            while (pipeline_.poll_for_frames(&newer_frames)) {
                frames = newer_frames;
            }

            const rs2::frameset aligned = align_to_color_->process(frames);
            const rs2::frame color_frame = aligned.get_color_frame();
            const rs2::frame depth_frame = aligned.get_depth_frame();
            if (!color_frame || !depth_frame) {
                if (error != nullptr) {
                    *error = "missing_color_or_depth_frame";
                }
                return false;
            }

            const rs2::video_frame color(color_frame);
            const rs2::depth_frame depth(depth_frame);
            const int color_width = color.get_width();
            const int color_height = color.get_height();
            const int depth_width = depth.get_width();
            const int depth_height = depth.get_height();
            if (color_width <= 0 || color_height <= 0 || depth_width <= 0 || depth_height <= 0) {
                if (error != nullptr) {
                    *error = "invalid_frame_geometry";
                }
                return false;
            }

            const std::vector<int> row_indices = selectRowIndices(slam_cfg, color_height);
            const int stride = std::max(1, slam_cfg.sample_stride);
            const auto* color_bytes = reinterpret_cast<const std::uint8_t*>(color.get_data());
            const auto* depth_words = reinterpret_cast<const std::uint16_t*>(depth.get_data());
            const int color_stride_bytes = color.get_stride_in_bytes();
            const int depth_stride_words = depth.get_stride_in_bytes() / static_cast<int>(sizeof(std::uint16_t));
            const int sample_limit = std::min(color_width, depth_width);
            const std::size_t samples_per_row = static_cast<std::size_t>((sample_limit + stride - 1) / stride);

            out->valid = true;
            out->capture_ts_ms = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count());
            out->frame_id = ++capture_seq_;
            out->color_width = static_cast<std::uint32_t>(color_width);
            out->color_height = static_cast<std::uint32_t>(color_height);
            out->depth_width = static_cast<std::uint32_t>(depth_width);
            out->depth_height = static_cast<std::uint32_t>(depth_height);
            out->row_index = row_indices.empty() ? 0 : row_indices.front();
            out->row_indices = row_indices;
            out->rgb_row.clear();
            out->depth_row.clear();
            out->rgb_row.reserve(row_indices.size() * samples_per_row * 3u);
            out->depth_row.reserve(row_indices.size() * samples_per_row);
            {
                std::lock_guard<std::mutex> lock(imu_mutex_);
                out->gyro_valid = gyro_frame_count_ > 0;
                out->gyro_frame_count = gyro_frame_count_;
                out->gyro_x = last_gyro_x_;
                out->gyro_y = last_gyro_y_;
                out->gyro_z = last_gyro_z_;
            }

            for (const int row_index : row_indices) {
                for (int x = 0; x < sample_limit; x += stride) {
                    const int color_offset = row_index * color_stride_bytes + x * 3;
                    const std::uint8_t b = color_bytes[color_offset + 0];
                    const std::uint8_t g = color_bytes[color_offset + 1];
                    const std::uint8_t r = color_bytes[color_offset + 2];
                    out->rgb_row.push_back(r);
                    out->rgb_row.push_back(g);
                    out->rgb_row.push_back(b);

                    const int depth_offset = row_index * depth_stride_words + x;
                    out->depth_row.push_back(depth_words[depth_offset]);
                }
            }

            return true;
        } catch (const rs2::error& e) {
            last_error_ = e.what();
            if (error != nullptr) {
                *error = last_error_;
            }
            return false;
        }
    }

    bool started() const {
        return started_;
    }

    const std::string& lastError() const {
        return last_error_;
    }

private:
    bool startMotionSensor(const rs2::device& device) {
        if (motion_started_) {
            return true;
        }

        for (const rs2::sensor& sensor : device.query_sensors()) {
            std::vector<rs2::stream_profile> motion_profiles;
            for (const rs2::stream_profile& profile : sensor.get_stream_profiles()) {
                const rs2_stream stream = profile.stream_type();
                if (stream == RS2_STREAM_GYRO && profile.format() == RS2_FORMAT_MOTION_XYZ32F) {
                    const int fps = profile.fps();
                    if (fps == 200) {
                        motion_profiles.push_back(profile);
                    }
                }
            }

            bool has_gyro = false;
            for (const rs2::stream_profile& profile : motion_profiles) {
                has_gyro = has_gyro || profile.stream_type() == RS2_STREAM_GYRO;
            }
            if (!has_gyro) {
                continue;
            }

            try {
                motion_sensor_ = std::make_unique<rs2::sensor>(sensor);
                motion_sensor_->open(motion_profiles);
                motion_sensor_->start([this](rs2::frame frame) {
                    if (!frame || !frame.is<rs2::motion_frame>() ||
                        frame.get_profile().stream_type() != RS2_STREAM_GYRO) {
                        return;
                    }
                    const auto motion = frame.as<rs2::motion_frame>().get_motion_data();
                    std::lock_guard<std::mutex> lock(imu_mutex_);
                    last_gyro_x_ = motion.x;
                    last_gyro_y_ = motion.y;
                    last_gyro_z_ = motion.z;
                    ++gyro_frame_count_;
                });
                motion_started_ = true;
                return true;
            } catch (const rs2::error& e) {
                last_error_ = e.what();
                stopMotionSensor();
            }
        }
        return false;
    }

    void stopMotionSensor() {
        if (motion_sensor_) {
            try {
                if (motion_started_) {
                    motion_sensor_->stop();
                }
                motion_sensor_->close();
            } catch (...) {
            }
        }
        motion_sensor_.reset();
        motion_started_ = false;
    }

    rs2::pipeline pipeline_;
    rs2::pipeline_profile pipeline_profile_;
    std::unique_ptr<rs2::align> align_to_color_;
    std::unique_ptr<rs2::sensor> motion_sensor_;
    std::mutex imu_mutex_;
    bool started_ = false;
    bool motion_started_ = false;
    std::uint32_t capture_seq_ = 0;
    float last_gyro_x_ = 0.0f;
    float last_gyro_y_ = 0.0f;
    float last_gyro_z_ = 0.0f;
    std::uint32_t gyro_frame_count_ = 0;
    std::string last_error_;
};

class MockD435iCaptureBridge {
public:
    bool ensureStarted(std::string* error) {
        (void)error;
        started_ = true;
        return true;
    }

    void shutdown() {
        started_ = false;
    }

    bool capture(std::uint32_t timeout_ms,
                 const SlamConfig& slam_cfg,
                 D435iFrameSample* out,
                 std::string* error) {
        (void)timeout_ms;
        if (out == nullptr) {
            return false;
        }
        *out = D435iFrameSample{};
        if (error != nullptr) {
            error->clear();
        }
        if (!started_) {
            started_ = true;
        }
        constexpr int kWidth = 640;
        constexpr int kHeight = 480;
        const std::vector<int> row_indices = selectRowIndices(slam_cfg, kHeight);
        const int stride = std::max(1, slam_cfg.sample_stride);

        out->valid = true;
        out->capture_ts_ms = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
        out->frame_id = ++capture_seq_;
        const int phase = static_cast<int>(capture_seq_ % 64u);
        const int wave = phase <= 32 ? phase : 64 - phase;
        out->color_width = kWidth;
        out->color_height = kHeight;
        out->depth_width = kWidth;
        out->depth_height = kHeight;
        out->row_index = row_indices.empty() ? 0 : row_indices.front();
        out->row_indices = row_indices;
        out->rgb_row.clear();
        out->depth_row.clear();

        const int sample_limit = kWidth;
        const std::size_t samples_per_row = static_cast<std::size_t>((sample_limit + stride - 1) / stride);
        out->rgb_row.reserve(row_indices.size() * samples_per_row * 3u);
        out->depth_row.reserve(row_indices.size() * samples_per_row);
        for (const int row_index : row_indices) {
            for (int x = 0; x < sample_limit; x += stride) {
                const std::uint8_t r = static_cast<std::uint8_t>((x + row_index + wave * 3) & 0xFF);
                const std::uint8_t g = static_cast<std::uint8_t>((row_index + wave) & 0xFF);
                const std::uint8_t b = static_cast<std::uint8_t>((x + row_index + wave * 5) & 0xFF);
                out->rgb_row.push_back(r);
                out->rgb_row.push_back(g);
                out->rgb_row.push_back(b);

                const std::uint16_t depth = static_cast<std::uint16_t>(
                    800u + ((static_cast<std::uint32_t>(x + row_index) * 3u + static_cast<std::uint32_t>(wave) * 17u) % 2200u));
                out->depth_row.push_back(depth);
            }
        }

        out->gyro_valid = true;
        out->gyro_frame_count = capture_seq_;
        out->gyro_x = 0.01f * static_cast<float>(wave);
        out->gyro_y = 0.02f * static_cast<float>(wave - 16);
        out->gyro_z = 0.03f * static_cast<float>(32 - wave);
        return true;
    }

    bool started() const {
        return started_;
    }

private:
    bool started_ = false;
    std::uint32_t capture_seq_ = 0;
};

}  // namespace

struct SlamExecutionLayerClient::Impl {
    std::uint32_t session_id = 0;
    std::uint32_t config_version = 0;
    SlamConfig slam_cfg;
    bool active = false;
    bool transport_connected = false;
    bool d435i_ready = false;
    bool use_mock_d435i = false;
    std::string last_runtime_error;
    D435iCaptureBridge d435i_bridge;
    MockD435iCaptureBridge mock_d435i_bridge;

    bool ensureConnected() {
        transport_connected = true;
        return transport_connected;
    }

    bool pushConfig(std::uint32_t in_session_id,
                    std::uint32_t in_config_version,
                    const SlamConfig& cfg) {
        if (!ensureConnected()) {
            return false;
        }
        session_id = in_session_id;
        config_version = in_config_version;
        slam_cfg = cfg;
        active = true;
        std::string capture_error;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        do {
            d435i_ready = use_mock_d435i
                ? mock_d435i_bridge.ensureStarted(&capture_error)
                : d435i_bridge.ensureStarted(&capture_error);
            if (d435i_ready) {
                last_runtime_error.clear();
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        } while (std::chrono::steady_clock::now() < deadline);

        active = false;
        last_runtime_error = capture_error.empty() ? "d435i_start_timeout_30s" : capture_error;
        std::cerr << "PushConfig failed: " << last_runtime_error << std::endl;
        return false;
    }

    ExecutorResult processFrame(std::uint32_t in_session_id,
                                const SlamImageFrame& frame,
                                std::uint32_t timeout_ms) {
        ExecutorResult result;
        if (!active || in_session_id != session_id || !ensureConnected()) {
            result.ok = false;
            result.timeout = false;
            result.quality_score = 0;
            result.error_detail = "executor_inactive_or_session_mismatch";
            return result;
        }

        D435iFrameSample live_sample;
        std::string capture_error;
        const bool have_live_sample = d435i_ready
            && (use_mock_d435i
                ? mock_d435i_bridge.capture(timeout_ms, slam_cfg, &live_sample, &capture_error)
                : d435i_bridge.capture(timeout_ms, slam_cfg, &live_sample, &capture_error));

        int row_index = -1;
        std::vector<std::uint8_t> feature_bytes;
        if (have_live_sample) {
            row_index = live_sample.row_index;
            feature_bytes.reserve(live_sample.rgb_row.size() + (live_sample.depth_row.size() * sizeof(std::uint16_t)) + 12u);
            feature_bytes.insert(feature_bytes.end(), live_sample.rgb_row.begin(), live_sample.rgb_row.end());
            for (const std::uint16_t depth : live_sample.depth_row) {
                feature_bytes.push_back(static_cast<std::uint8_t>(depth & 0xFFu));
                feature_bytes.push_back(static_cast<std::uint8_t>((depth >> 8) & 0xFFu));
            }
            appendFloat(&feature_bytes, live_sample.gyro_x);
            appendFloat(&feature_bytes, live_sample.gyro_y);
            appendFloat(&feature_bytes, live_sample.gyro_z);
            feature_bytes.push_back(static_cast<std::uint8_t>(frame.quality_hint & 0xFF));
            feature_bytes.push_back(static_cast<std::uint8_t>((frame.quality_hint >> 8) & 0xFF));
            feature_bytes.push_back(static_cast<std::uint8_t>(frame.frame_id & 0xFFu));
            feature_bytes.push_back(static_cast<std::uint8_t>((frame.frame_id >> 8) & 0xFFu));
            feature_bytes.push_back(static_cast<std::uint8_t>((frame.frame_id >> 16) & 0xFFu));
            feature_bytes.push_back(static_cast<std::uint8_t>((frame.frame_id >> 24) & 0xFFu));
        } else {
            result.ok = false;
            result.timeout = capture_error.find("timeout") != std::string::npos ||
                             capture_error.find("didn't arrive within") != std::string::npos;
            result.quality_score = 0;
            last_runtime_error = capture_error.empty() ? "capture_failed_unknown" : capture_error;
            result.error_detail = last_runtime_error;
            std::cerr << "ProcessFrame capture failed: " << last_runtime_error << std::endl;
            return result;
        }

        const std::uint32_t hash = fnv1a32(feature_bytes);
        const std::uint32_t sample_count = have_live_sample
            ? static_cast<std::uint32_t>(live_sample.rgb_row.size() / 3u)
            : static_cast<std::uint32_t>(feature_bytes.size());
        result.row_indices = live_sample.row_indices;
        result.depth_values = live_sample.depth_row;
        result.imu_gyro_valid = live_sample.gyro_valid;
        result.imu_gyro_frames = live_sample.gyro_frame_count;
        result.imu_gyro_x = live_sample.gyro_x;
        result.imu_gyro_y = live_sample.gyro_y;
        result.imu_gyro_z = live_sample.gyro_z;
        result.r_values.clear();
        result.r_values.reserve(live_sample.rgb_row.size() / 3u);
        for (std::size_t i = 0; i + 2u < live_sample.rgb_row.size(); i += 3u) {
            result.r_values.push_back(live_sample.rgb_row[i]);
        }

        SlamImageFrame enriched = frame;
        enriched.is_row_feature = true;
        enriched.row_index = row_index;
        enriched.channel_mode = have_live_sample ? "RGB24+DEPTH" : channelModeToString(slam_cfg.channel_mode);
        enriched.stride = std::max(1, slam_cfg.sample_stride);
        enriched.sample_count = static_cast<int>(sample_count);
        enriched.payload_len = static_cast<int>(feature_bytes.size());
        enriched.payload_crc32 = hash;

        // Simulated transport + SLAM runtime RTT.
        result.proc_ms = static_cast<std::uint32_t>(8u + (enriched.frame_id % 5u) + (sample_count % 7u));
        result.timeout = result.proc_ms > timeout_ms;
        if (result.timeout) {
            result.ok = false;
            result.quality_score = 0;
            result.error_detail = "executor_timeout";
            return result;
        }

        result.ok = true;
        const int signal = static_cast<int>((hash ^ (hash >> 13)) & 0x7Fu);
        const int quality_hint = have_live_sample ? std::clamp(frame.quality_hint, 0, 100) : frame.quality_hint;
        result.quality_score = std::clamp(quality_hint + (signal / 8), 0, 100);

        if (result.quality_score >= slam_cfg.min_quality) {
            const std::uint32_t group0 = 0x01000000u
                                         | (static_cast<std::uint32_t>(row_index & 0x3FF) << 14)
                                         | (static_cast<std::uint32_t>(sample_count & 0x3FF) << 4)
                                         | static_cast<std::uint32_t>(signal & 0x0Fu);
            const std::uint32_t group1 = 0x02000000u | (hash & 0x00FFFFFFu);
            result.groups.push_back(group0);
            result.groups.push_back(group1);
            if (have_live_sample) {
                const std::uint16_t depth_mean = meanDepth(live_sample.depth_row);
                const std::uint32_t group2 = 0x03000000u
                                             | (static_cast<std::uint32_t>(depth_mean & 0x0FFFu) << 4)
                                             | static_cast<std::uint32_t>((static_cast<int>(live_sample.gyro_z * 1000.0f)) & 0x0Fu);
                if (static_cast<int>(result.groups.size()) < slam_cfg.max_groups) {
                    result.groups.push_back(group2);
                }
            }
        }
        if (static_cast<int>(result.groups.size()) > slam_cfg.max_groups) {
            result.groups.resize(static_cast<std::size_t>(slam_cfg.max_groups));
        }

        return result;
    }

    bool stopSession(std::uint32_t in_session_id) {
        if (!active) {
            return true;
        }
        if (in_session_id != session_id) {
            return false;
        }
        active = false;
        d435i_bridge.shutdown();
        mock_d435i_bridge.shutdown();
        d435i_ready = false;
        last_runtime_error.clear();
        return true;
    }

    bool getHealth() const {
        if (!transport_connected) {
            return false;
        }
        if (!active) {
            return true;
        }
        const bool bridge_started = use_mock_d435i ? mock_d435i_bridge.started() : d435i_bridge.started();
        return d435i_ready && bridge_started;
    }
};

SlamExecutionLayerClient::SlamExecutionLayerClient()
    : impl_(std::make_unique<Impl>()) {}

SlamExecutionLayerClient::~SlamExecutionLayerClient() = default;

SlamExecutionLayerClient::SlamExecutionLayerClient(SlamExecutionLayerClient&&) noexcept = default;

SlamExecutionLayerClient& SlamExecutionLayerClient::operator=(SlamExecutionLayerClient&&) noexcept = default;

bool SlamExecutionLayerClient::PushConfig(std::uint32_t session_id,
                                          std::uint32_t config_version,
                                          const SlamConfig& slam_cfg) {
    return impl_->pushConfig(session_id, config_version, slam_cfg);
}

ExecutorResult SlamExecutionLayerClient::ProcessFrame(std::uint32_t session_id,
                                                      const SlamImageFrame& frame,
                                                      std::uint32_t timeout_ms) {
    return impl_->processFrame(session_id, frame, timeout_ms);
}

bool SlamExecutionLayerClient::StopSession(std::uint32_t session_id) {
    return impl_->stopSession(session_id);
}

bool SlamExecutionLayerClient::GetHealth() const {
    return impl_->getHealth();
}

bool SlamExecutionLayerClient::InitializeD435i(std::string* error) {
    if (impl_->use_mock_d435i) {
        return impl_->mock_d435i_bridge.ensureStarted(error);
    }
    return impl_->d435i_bridge.ensureStarted(error);
}

void SlamExecutionLayerClient::EnableMockD435i(bool enabled) {
    impl_->use_mock_d435i = enabled;
}

void SlamExecutionLayerClient::ShutdownD435i() {
    impl_->d435i_bridge.shutdown();
    impl_->mock_d435i_bridge.shutdown();
    impl_->d435i_ready = false;
}

bool SlamExecutionLayerClient::CaptureD435iFrame(std::uint32_t timeout_ms,
                                                 const SlamConfig& slam_cfg,
                                                 D435iFrameSample* out,
                                                 std::string* error) {
    const bool ok = impl_->use_mock_d435i
        ? impl_->mock_d435i_bridge.capture(timeout_ms, slam_cfg, out, error)
        : impl_->d435i_bridge.capture(timeout_ms, slam_cfg, out, error);
    impl_->d435i_ready = impl_->use_mock_d435i
        ? (ok || impl_->mock_d435i_bridge.started())
        : (ok || impl_->d435i_bridge.started());
    return ok;
}

bool SlamExecutionLayerClient::IsD435iReady() const {
    return impl_->use_mock_d435i ? impl_->mock_d435i_bridge.started() : impl_->d435i_bridge.started();
}

}  // namespace slam_exec
