#include "usv/SlamExecutionLayer.h"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using slam_exec::SlamConfig;
using slam_exec::SlamExecutionLayerClient;

namespace {

std::uint64_t nowMs() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(now).count());
}

std::string formatSampleLine(const slam_exec::D435iFrameSample& sample) {
    std::ostringstream oss;
    oss << "sample row=" << sample.row_index
        << " rgb_bytes=" << sample.rgb_row.size()
        << " depth_values=" << sample.depth_row.size()
        << " gyro=[" << std::fixed << std::setprecision(4)
        << sample.gyro_x << "," << sample.gyro_y << "," << sample.gyro_z << "]";

    oss << " rgb_row=[";
    const std::size_t rgb_triplets = sample.rgb_row.size() / 3u;
    for (std::size_t i = 0; i < rgb_triplets; ++i) {
        const std::size_t base = i * 3u;
        oss << static_cast<int>(sample.rgb_row[base]) << ","
            << static_cast<int>(sample.rgb_row[base + 1u]) << ","
            << static_cast<int>(sample.rgb_row[base + 2u]);
        if (i + 1u != rgb_triplets) {
            oss << ";";
        }
    }
    oss << "]";

    oss << " depth_row=[";
    const std::size_t depth_count = sample.depth_row.size();
    for (std::size_t i = 0; i < depth_count; ++i) {
        oss << sample.depth_row[i];
        if (i + 1u != depth_count) {
            oss << ",";
        }
    }
    oss << "]";
    return oss.str();
}

bool parsePositiveIntArg(const std::string& value, int* out) {
    if (out == nullptr) {
        return false;
    }
    try {
        std::size_t consumed = 0;
        const int parsed = std::stoi(value, &consumed);
        if (consumed != value.size() || parsed <= 0) {
            return false;
        }
        *out = parsed;
        return true;
    } catch (...) {
        return false;
    }
}

}  // namespace

int main(int argc, char** argv) {
    constexpr std::uint32_t kSessionId = 1u;
    constexpr std::uint32_t kConfigVersion = 1u;
    int run_seconds = 10;
    int max_fps = 10;
    int exec_timeout_ms_arg = 1000;
    const float row_ratio = 0.333333f;
    const int sample_stride = 1;

    SlamConfig cfg;
    cfg.row_ratio = row_ratio;
    cfg.sample_stride = sample_stride;

    bool use_mock = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "+mock" || arg == "--mock") {
            use_mock = true;
        } else if (arg == "--timeout-ms" && i + 1 < argc) {
            if (!parsePositiveIntArg(argv[++i], &exec_timeout_ms_arg)) {
                std::cerr << "Bad --timeout-ms value" << std::endl;
                return 64;
            }
        } else if (arg == "--seconds" && i + 1 < argc) {
            if (!parsePositiveIntArg(argv[++i], &run_seconds)) {
                std::cerr << "Bad --seconds value" << std::endl;
                return 64;
            }
        } else if (arg == "--fps" && i + 1 < argc) {
            if (!parsePositiveIntArg(argv[++i], &max_fps)) {
                std::cerr << "Bad --fps value" << std::endl;
                return 64;
            }
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: SelD435iSmokeTest [--mock] [--timeout-ms N] [--seconds N] [--fps N]\n";
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return 64;
        }
    }

    const std::uint32_t exec_timeout_ms = static_cast<std::uint32_t>(exec_timeout_ms_arg);
    cfg.max_fps = max_fps;
    cfg.exec_timeout_ms = exec_timeout_ms_arg;

    SlamExecutionLayerClient client;
    client.EnableMockD435i(use_mock);
    std::string init_error;
    if (!client.InitializeD435i(&init_error)) {
        std::cerr << "D435i init failed: " << init_error << std::endl;
        return 1;
    }

    if (!client.PushConfig(kSessionId, kConfigVersion, cfg)) {
        std::cerr << "PushConfig failed" << std::endl;
        return 2;
    }

    const std::uint64_t start_ms = nowMs();
    const std::uint64_t end_ms = start_ms + static_cast<std::uint64_t>(run_seconds * 1000);
    const std::uint32_t frame_interval_ms = static_cast<std::uint32_t>(1000 / max_fps);

    std::uint32_t sent = 0;
    std::uint32_t ok = 0;
    std::uint32_t timeout = 0;
    std::uint32_t err = 0;
    std::uint32_t capture_err = 0;
    std::uint32_t ready_true = 0;
    std::uint64_t capture_ms_total = 0;
    std::uint32_t capture_ms_max = 0;
    std::string capture_error;

    while (nowMs() < end_ms) {
        const std::uint64_t t0 = nowMs();
        slam_exec::D435iFrameSample sample;
        capture_error.clear();
        const bool have_sample = client.CaptureD435iFrame(exec_timeout_ms, cfg, &sample, &capture_error);
        const std::uint32_t capture_ms = static_cast<std::uint32_t>(nowMs() - t0);
        capture_ms_total += capture_ms;
        if (capture_ms > capture_ms_max) {
            capture_ms_max = capture_ms;
        }
        if (client.IsD435iReady()) {
            ready_true++;
        }

        if (!have_sample) {
            capture_err++;
            if (capture_error.find("didn't arrive within") != std::string::npos ||
                capture_error.find("timeout") != std::string::npos) {
                timeout++;
            } else {
                err++;
            }
            std::cout << "capture_err=" << capture_error
                      << " capture_ms=" << capture_ms
                      << " ready=" << (client.IsD435iReady() ? "true" : "false")
                      << std::endl;
        } else {
            ok++;
            std::cout << formatSampleLine(sample)
                      << " capture_ms=" << capture_ms
                      << " ready=true"
                      << std::endl;
        }
        sent++;

        if (sent % 10u == 0u) {
            std::cout << "tick=" << sent
                      << " ok=" << ok
                      << " timeout=" << timeout
                      << " err=" << err
                      << " ready_rate=" << std::fixed << std::setprecision(2)
                      << (sent > 0 ? (100.0 * static_cast<double>(ready_true) / static_cast<double>(sent)) : 0.0)
                      << "% avg_capture_ms=" << (sent > 0 ? (static_cast<double>(capture_ms_total) / static_cast<double>(sent)) : 0.0)
                      << " max_capture_ms=" << capture_ms_max
                      << std::endl;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(frame_interval_ms));
    }

    const double avg_capture_ms = sent > 0 ? static_cast<double>(capture_ms_total) / static_cast<double>(sent) : 0.0;
    const double ready_rate = sent > 0 ? (100.0 * static_cast<double>(ready_true) / static_cast<double>(sent)) : 0.0;

    std::cout << "\n=== SEL D435i smoke test ===\n";
    std::cout << "sent=" << sent
              << " ok=" << ok
              << " timeout=" << timeout
              << " err=" << err
              << " capture_err=" << capture_err
              << " timeout_ms=" << exec_timeout_ms
              << " seconds=" << run_seconds
              << " ready_rate=" << ready_rate << "%"
              << " avg_capture_ms=" << avg_capture_ms
              << " max_capture_ms=" << capture_ms_max
              << " health=" << (client.GetHealth() ? "OK" : "BAD")
              << std::endl;

    client.StopSession(kSessionId);
    client.ShutdownD435i();
    return 0;
}
