#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <filesystem>
#include <cstdlib>

namespace {

struct MapperConfig {
	int pwm_period_ns = 20000000;
	int duty_neutral_ns = 0;
	int duty_span_ns = 5000000;

	float max_power_percent = 70.0f;
	float left_gain = 1.0f;
	float right_gain = 1.0f;
	float left_trim = 0.0f;
	float right_trim = 0.0f;
};

struct ThrusterInput {
	float left_percent = 0.0f;
	float right_percent = 0.0f;
};

struct ThrusterOutput {
	float left_percent_final = 0.0f;
	float right_percent_final = 0.0f;
	int left_duty_ns = 0;
	int right_duty_ns = 0;
	float applied_limit_percent = 0.0f;
	int status_code = 0;
};

class ThrusterMapper {
public:
	explicit ThrusterMapper(MapperConfig cfg) : cfg_(cfg) {}

	ThrusterOutput apply(const ThrusterInput& in) const {
		ThrusterOutput out;

		float left = in.left_percent;
		float right = in.right_percent;

		const float limit = std::max(0.0f, std::min(100.0f, cfg_.max_power_percent));
		left = clamp(left, 0.0f, limit);
		right = clamp(right, 0.0f, limit);

		left = left * cfg_.left_gain + cfg_.left_trim;
		right = right * cfg_.right_gain + cfg_.right_trim;

		left = clamp(left, 0.0f, 100.0f);
		right = clamp(right, 0.0f, 100.0f);

		out.left_percent_final = left;
		out.right_percent_final = right;
		out.left_duty_ns = percentToDutyNs(left);
		out.right_duty_ns = percentToDutyNs(right);
		out.applied_limit_percent = limit;
		out.status_code = 0;
		return out;
	}

private:
	static float clamp(float v, float lo, float hi) {
		return std::max(lo, std::min(hi, v));
	}

	int percentToDutyNs(float percent) const {
		// Keep PWM shaping config in place for future hardware features, but
		// decouple it from current output behavior.
		// Current runtime behavior is binary:
		//   0 or negative -> 0ns
		//   positive      -> full period (20000000ns by default)
		if (percent <= 0.0f) {
			return 0;
		}
		const int period = std::max(0, cfg_.pwm_period_ns);
		if (period > 1) {
			return period - 1;
		}
		return period;
	}

	MapperConfig cfg_;
};

class SysfsPwmWriter {
public:
	SysfsPwmWriter(std::string left_duty_path, std::string right_duty_path)
		: left_duty_path_(std::move(left_duty_path)),
		  right_duty_path_(std::move(right_duty_path)) {}

	bool writeDuty(int left_duty_ns, int right_duty_ns) const {
		std::cout << "[DEBUG] Writing to sysfs: left=" << left_duty_ns 
				  << " (" << left_duty_path_ << "), right=" << right_duty_ns 
				  << " (" << right_duty_path_ << ")" << std::endl;
		if (!writeInt(left_duty_path_, left_duty_ns)) {
			std::cerr << "ERR: failed writing left duty path" << std::endl;
			return false;
		}
		if (!writeInt(right_duty_path_, right_duty_ns)) {
			std::cerr << "ERR: failed writing right duty path" << std::endl;
			return false;
		}
		return true;
	}

private:
	static bool writeInt(const std::string& path, int value) {
		for (int i = 0; i < 3; ++i) {
			std::ofstream file(path);
			if (!file.is_open()) {
				std::this_thread::sleep_for(std::chrono::milliseconds(10));
				continue;
			}
			file << value << '\n';
			if (file.good()) {
				return true;
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
		}
		std::cerr << "ERR: write failed: " << path << ", value=" << value << std::endl;
		return false;
	}

	std::string left_duty_path_;
	std::string right_duty_path_;
};

// PWM helper functions: init, cleanup, sysfs writes.
namespace pwm {
using namespace std::chrono_literals;

static bool writeStr(const std::string& path, const std::string& value) {
	for (int i = 0; i < 3; ++i) {
		std::ofstream f(path);
		if (!f.is_open()) {
			std::this_thread::sleep_for(10ms);
			continue;
		}
		f << value << '\n';
		if (f.good()) return true;
		std::this_thread::sleep_for(10ms);
	}
	std::cerr << "ERR: pwm write failed: " << path << " -> '" << value << "'" << std::endl;
	return false;
}

static void ensure_exported(const std::string& chip_dir, const std::string& pwm_dir, int channel) {
	if (std::filesystem::exists(pwm_dir)) return;
	for (int i = 0; i < 5; ++i) {
		writeStr(chip_dir + "/export", std::to_string(channel));
		std::this_thread::sleep_for(100ms);
		if (std::filesystem::exists(pwm_dir)) return;
	}
}

static void init_all() {
	const std::string chip = "/sys/class/pwm/pwmchip0";
	const std::string pwm1 = chip + "/pwm1";
	const std::string pwm2 = chip + "/pwm2";
	if (!std::filesystem::exists(chip)) {
		std::cerr << "WARN: pwm chip dir not present: " << chip << std::endl;
		return;
	}
	ensure_exported(chip, pwm1, 1);
	ensure_exported(chip, pwm2, 2);

	if (!std::filesystem::exists(pwm1) || !std::filesystem::exists(pwm2)) {
		std::cerr << "WARN: pwm channels missing after export" << std::endl;
		return;
	}

	// Disable -> set period -> duty=0 -> enable (same ordering as shell runner)
	writeStr(pwm1 + "/enable", "0");
	writeStr(pwm1 + "/period", std::to_string(20000000));
	writeStr(pwm1 + "/duty_cycle", "0");
	writeStr(pwm1 + "/enable", "1");

	writeStr(pwm2 + "/enable", "0");
	writeStr(pwm2 + "/period", std::to_string(20000000));
	writeStr(pwm2 + "/duty_cycle", "0");
	writeStr(pwm2 + "/enable", "1");
}

static void cleanup_all() {
    // We remove the automatic cleanup that disables/unexports PWM
    // so that the PWM state persists after the C++ process exits,
    // just like the shell script doesn't unexport on every command.
    std::cout << "[DEBUG] Keeping PWM state active for hardware output." << std::endl;
}

} // namespace pwm
void printUsage() {
	std::cout << "Thruster executor demo\n"
			  << "Input format: <left_percent> <right_percent>\n"
			  << "Example: 30 -25\n"
			  << "Type q to quit.\n";
}

}  // namespace

int main(int argc, char** argv) {
	MapperConfig cfg;
	cfg.max_power_percent = 100.0f;
	cfg.left_gain = 1.0f;
	cfg.right_gain = 0.98f;
	cfg.left_trim = 0.0f;
	cfg.right_trim = 0.0f;

	ThrusterMapper mapper(cfg);

	// Initialize PWM sysfs like thruster_test_runner.sh
	pwm::init_all();
	std::atexit(pwm::cleanup_all);

	SysfsPwmWriter writer(
		"/sys/class/pwm/pwmchip0/pwm1/duty_cycle",
		"/sys/class/pwm/pwmchip0/pwm2/duty_cycle");

	auto run_once = [&](float left, float right) {
		ThrusterInput in;
		in.left_percent = left;
		in.right_percent = right;

		const ThrusterOutput out = mapper.apply(in);
		const bool ok = writer.writeDuty(out.left_duty_ns, out.right_duty_ns);
		std::cout << std::fixed << std::setprecision(2)
				  << "left=" << out.left_percent_final
				  << "%, right=" << out.right_percent_final
				  << "%, status=" << (ok ? 0 : 1) << std::endl;
		return ok ? 0 : 1;
	};

	// Accept single-letter ACKs: F (forward), L (left), R (right), S (stop)
	if (argc == 2) {
		std::string cmd = argv[1];
		if (!cmd.empty()) {
			char c = std::toupper(cmd[0]);
			float left = 0.0f, right = 0.0f;
			switch (c) {
				case 'F': left = 100.0f; right = 100.0f; break;
				case 'L': left = 100.0f; right = 0.0f; break;
				case 'R': left = 0.0f; right = 100.0f; break;
				case 'S': left = 0.0f; right = 0.0f; break;
				default:
					std::cerr << "ERR: unknown command. Use F,L,R,S" << std::endl;
					return 2;
			}
			const int rc = run_once(left, right);
			// writer.writeDuty(0, 0); // Removed to keep output persistent
			return rc;
		}
	}

	std::cout << "Input: single letter ACK F (forward), L (left), R (right), S (stop). q to quit." << std::endl;
	std::string line;
	while (std::getline(std::cin, line)) {
		if (line.empty()) continue;
		if (line == "q" || line == "Q") break;
		char c = std::toupper(line[0]);
		float left = 0.0f, right = 0.0f;
		switch (c) {
			case 'F': left = 100.0f; right = 100.0f; break;
			case 'L': left = 100.0f; right = 0.0f; break;
			case 'R': left = 0.0f; right = 100.0f; break;
			case 'S': left = 0.0f; right = 0.0f; break;
			default:
				std::cout << "WARN: unknown command. Use F/L/R/S or q." << std::endl;
				continue;
		}
		run_once(left, right);
	}

	// Fail-safe stop on exit.
	// writer.writeDuty(0, 0); // Removed to keep output persistent
	return 0;
}
