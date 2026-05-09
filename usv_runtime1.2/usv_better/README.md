# USV Edge Demo

USV Edge Demo is a lightweight edge-validation workspace for a USV control stack.
It focuses on the contract between three runtime layers:

- `CommunicationLayer`: TCP gateway, line-protocol parsing, forwarding, and gateway management commands.
- `MainProcessor`: control dispatch, SLAM ingest gating, D435i capture orchestration, health, and rollback signals.
- `SlamExecutionLayer`: Intel RealSense D435i capture bridge with explicit mock support for local validation.

## Quick Start

Build the normal runtime targets:

```bash
bash scripts/build.sh
```

Start the stack in the default hardware mode:

```bash
./start.sh
```

Default mode uses real thruster sysfs PWM output and a real D435i camera. Use an
explicit virtual mode only when you want mock D435i data and log-only thruster
behavior:

```bash
./start.sh --virtual
```

Run the temporary test client:

```bash
python3 client_test.py --host 127.0.0.1 --port 19520 --actions F,L,R,S --print-full-detail
```

Windows interactive mode uses WASD when `msvcrt` is available:

```bash
python client_test.py --host <device-ip> --interactive --print-full-detail
```

## Repository Layout

```text
usv_better/
  CMakeLists.txt
  INSTALL.md
  README.md
  client_test.py
  start.sh
  src/
    CommunicationLayer.cc
    MainProcessor.cc
    SlamExecutionLayer.cc
    ThrusterActuator.cc
  include/
    usv/
      SlamExecutionLayer.h
  tools/
    ThrusterActuator_test_runner.sh
    thruster_test_runner.sh
  scripts/
    build.sh
    clean.sh
  docs/
    hardware.md
    protocol.md
    pwm.md
    troubleshooting.md
  tests/
    README.md
    SelD435iSmokeTest.cc
    run_d435i_smoke_test.sh
```

## Build Outputs

The normal CMake build creates these executables in `build/bin/`:

- `CommunicationLayer`
- `MainProcessor`
- `ThrusterActuator`

`SelD435iSmokeTest` is intentionally outside the normal build sequence. Run it
through the test launcher, which compiles the executable only if it is missing:

```bash
bash tests/run_d435i_smoke_test.sh --mock
```

## Runtime Entry Points

### Full stack

```bash
./start.sh
./start.sh --virtual
./start.sh --mock-d435i
./start.sh --no-thruster
```

`start.sh` prints a startup banner showing whether thruster output is real sysfs
PWM or log-only, and whether D435i mode is real or mock.

### Individual binaries

```bash
build/bin/CommunicationLayer
build/bin/MainProcessor --tcp --port 19521
USV_THRUSTER_SYSFS=0 build/bin/MainProcessor --tcp --port 19521 --mock-d435i
build/bin/ThrusterActuator
```

## Documentation

- [Installation](INSTALL.md)
- [Protocol](docs/protocol.md)
- [PWM and thruster validation](docs/pwm.md)
- [Hardware notes](docs/hardware.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Tests](tests/README.md)

## Release Artifact Guidance

Use `build/bin/` as the local executable output directory and as the source for
release packaging. Do not commit generated binaries to Git. If a distributable
bundle is needed, generate it from a clean build and package the selected
executables, scripts, and docs into a separate archive.
