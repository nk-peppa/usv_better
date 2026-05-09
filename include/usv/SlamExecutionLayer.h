#ifndef SLAM_EXECUTION_LAYER_H_
#define SLAM_EXECUTION_LAYER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>


namespace slam_exec {

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

struct ExecutorResult {
    bool ok = false;
    bool timeout = false;
    int quality_score = 0;
    std::uint32_t proc_ms = 0;
    std::vector<std::uint32_t> groups;
    std::vector<int> row_indices;
    std::vector<std::uint8_t> r_values;
    std::vector<std::uint16_t> depth_values;
    bool imu_gyro_valid = false;
    std::uint32_t imu_gyro_frames = 0;
    float imu_gyro_x = 0.0f;
    float imu_gyro_y = 0.0f;
    float imu_gyro_z = 0.0f;
    std::string error_detail;
};

struct D435iFrameSample {
    bool valid = false;
    std::uint64_t capture_ts_ms = 0;
    std::uint32_t frame_id = 0;
    std::uint32_t color_width = 0;
    std::uint32_t color_height = 0;
    std::uint32_t depth_width = 0;
    std::uint32_t depth_height = 0;
    int row_index = -1;
    std::vector<int> row_indices;
    std::vector<std::uint8_t> rgb_row;
    std::vector<std::uint16_t> depth_row;
    bool gyro_valid = false;
    std::uint32_t gyro_frame_count = 0;
    float gyro_x = 0.0f;
    float gyro_y = 0.0f;
    float gyro_z = 0.0f;
};

class SlamExecutionLayerClient {
public:
    SlamExecutionLayerClient();
    ~SlamExecutionLayerClient();

    SlamExecutionLayerClient(const SlamExecutionLayerClient&) = delete;
    SlamExecutionLayerClient& operator=(const SlamExecutionLayerClient&) = delete;
    SlamExecutionLayerClient(SlamExecutionLayerClient&&) noexcept;
    SlamExecutionLayerClient& operator=(SlamExecutionLayerClient&&) noexcept;

    bool PushConfig(std::uint32_t session_id,
                    std::uint32_t config_version,
                    const SlamConfig& slam_cfg);

    ExecutorResult ProcessFrame(std::uint32_t session_id,
                                const SlamImageFrame& frame,
                                std::uint32_t timeout_ms);

    bool StopSession(std::uint32_t session_id);

    bool GetHealth() const;

    bool InitializeD435i(std::string* error = nullptr);
    void EnableMockD435i(bool enabled);
    void ShutdownD435i();

    bool CaptureD435iFrame(std::uint32_t timeout_ms,
                           const SlamConfig& slam_cfg,
                           D435iFrameSample* out,
                           std::string* error = nullptr);

    bool IsD435iReady() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace slam_exec

#endif  // SLAM_EXECUTION_LAYER_H_
