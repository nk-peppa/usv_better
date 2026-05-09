#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

#include <librealsense2/rs.hpp>

namespace {

const char* streamName(rs2_stream stream) {
    switch (stream) {
        case RS2_STREAM_GYRO:
            return "GYRO";
        case RS2_STREAM_ACCEL:
            return "ACCEL";
        case RS2_STREAM_COLOR:
            return "COLOR";
        case RS2_STREAM_DEPTH:
            return "DEPTH";
        default:
            return "OTHER";
    }
}

}  // namespace

int main() {
    try {
        rs2::context ctx;
        const rs2::device_list devices = ctx.query_devices();
        std::cout << "devices=" << devices.size() << std::endl;
        if (devices.size() == 0) {
            return 2;
        }

        rs2::device device = devices.front();
        if (device.supports(RS2_CAMERA_INFO_NAME)) {
            std::cout << "device=" << device.get_info(RS2_CAMERA_INFO_NAME) << std::endl;
        }
        if (device.supports(RS2_CAMERA_INFO_SERIAL_NUMBER)) {
            std::cout << "serial=" << device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER) << std::endl;
        }
        if (device.supports(RS2_CAMERA_INFO_FIRMWARE_VERSION)) {
            std::cout << "firmware=" << device.get_info(RS2_CAMERA_INFO_FIRMWARE_VERSION) << std::endl;
        }

        rs2::sensor gyro_sensor;
        rs2::stream_profile selected_gyro;
        bool have_gyro = false;

        int sensor_index = 0;
        for (const rs2::sensor& sensor : device.query_sensors()) {
            std::string sensor_name = "unknown";
            if (sensor.supports(RS2_CAMERA_INFO_NAME)) {
                sensor_name = sensor.get_info(RS2_CAMERA_INFO_NAME);
            }
            std::cout << "sensor[" << sensor_index++ << "]=" << sensor_name << std::endl;

            for (const rs2::stream_profile& profile : sensor.get_stream_profiles()) {
                const rs2_stream stream = profile.stream_type();
                std::cout << "  profile stream=" << streamName(stream)
                          << " format=" << rs2_format_to_string(profile.format())
                          << " fps=" << profile.fps()
                          << " index=" << profile.stream_index()
                          << std::endl;

                if (stream == RS2_STREAM_GYRO && profile.format() == RS2_FORMAT_MOTION_XYZ32F &&
                    (!have_gyro || profile.fps() == 200)) {
                    gyro_sensor = sensor;
                    selected_gyro = profile;
                    have_gyro = true;
                }
            }
        }

        if (!have_gyro) {
            std::cerr << "no gyro motion profile found" << std::endl;
            return 3;
        }

        std::atomic<int> gyro_count{0};
        std::atomic<int> accel_count{0};
        std::mutex value_mutex;
        float last_x = 0.0f;
        float last_y = 0.0f;
        float last_z = 0.0f;
        float max_abs = 0.0f;

        std::vector<rs2::stream_profile> motion_profiles;
        motion_profiles.push_back(selected_gyro);

        std::cout << "opening motion sensor gyro_fps=" << selected_gyro.fps()
                  << " accel=off"
                  << std::endl;
        gyro_sensor.open(motion_profiles);
        gyro_sensor.start([&](rs2::frame frame) {
            if (!frame || !frame.is<rs2::motion_frame>()) {
                return;
            }
            if (frame.get_profile().stream_type() != RS2_STREAM_GYRO) {
                return;
            }
            const auto motion = frame.as<rs2::motion_frame>().get_motion_data();
            const float local_max = std::max({std::fabs(motion.x), std::fabs(motion.y), std::fabs(motion.z)});
            {
                std::lock_guard<std::mutex> lock(value_mutex);
                last_x = motion.x;
                last_y = motion.y;
                last_z = motion.z;
                max_abs = std::max(max_abs, local_max);
            }
            gyro_count.fetch_add(1);
        });

        for (int i = 0; i < 6; ++i) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            std::lock_guard<std::mutex> lock(value_mutex);
            std::cout << "t=" << (i + 1)
                      << " gyro_count=" << gyro_count.load()
                      << " accel_count=" << accel_count.load()
                      << " last=" << last_x << "," << last_y << "," << last_z
                      << " max_abs=" << max_abs
                      << std::endl;
        }

        gyro_sensor.stop();
        gyro_sensor.close();
        if (gyro_count.load() > 0) {
            return 0;
        }

        std::cout << "direct sensor callback got no gyro frames; trying motion-only pipeline" << std::endl;
        gyro_count = 0;
        accel_count = 0;
        max_abs = 0.0f;

        rs2::pipeline pipe;
        rs2::config config;
        config.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F, 200);
        pipe.start(config);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(6);
        while (std::chrono::steady_clock::now() < deadline) {
            rs2::frameset frames;
            if (!pipe.poll_for_frames(&frames)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            const rs2::frame gyro = frames.first_or_default(RS2_STREAM_GYRO);
            if (gyro) {
                const auto motion = gyro.as<rs2::motion_frame>().get_motion_data();
                const float local_max = std::max({std::fabs(motion.x), std::fabs(motion.y), std::fabs(motion.z)});
                std::lock_guard<std::mutex> lock(value_mutex);
                last_x = motion.x;
                last_y = motion.y;
                last_z = motion.z;
                max_abs = std::max(max_abs, local_max);
                gyro_count.fetch_add(1);
            }
            const rs2::frame accel = frames.first_or_default(RS2_STREAM_ACCEL);
            if (accel) {
                accel_count.fetch_add(1);
            }
        }
        pipe.stop();
        std::cout << "pipeline gyro_count=" << gyro_count.load()
                  << " accel_count=" << accel_count.load()
                  << " last=" << last_x << "," << last_y << "," << last_z
                  << " max_abs=" << max_abs
                  << std::endl;

        return gyro_count.load() > 0 ? 0 : 4;
    } catch (const rs2::error& e) {
        std::cerr << "rs2_error=" << e.what() << std::endl;
        return 10;
    } catch (const std::exception& e) {
        std::cerr << "error=" << e.what() << std::endl;
        return 11;
    }
}
