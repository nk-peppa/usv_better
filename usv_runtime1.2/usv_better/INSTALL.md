# Installation

## Required Tools

Install a C++17 compiler, CMake, and Intel RealSense development files.

On Debian/Ubuntu-style systems, the build tools are typically installed with:

```bash
sudo apt update
sudo apt install -y build-essential cmake
```

Install Intel RealSense according to the target platform's RealSense setup
process. This project treats D435i support as a core requirement: CMake fails
immediately if `librealsense2` headers or the `realsense2` library are missing.

## Build

From the repository root:

```bash
bash scripts/build.sh
```

The script wraps the CMake workflow:

```bash
cmake -S . -B build
cmake --build build
```

Normal runtime executables are written to `build/bin/`:

```text
build/bin/CommunicationLayer
build/bin/MainProcessor
build/bin/ThrusterActuator
```

## Clean

```bash
bash scripts/clean.sh
```

## Start

Default hardware mode:

```bash
./start.sh
```

Explicit virtual mode:

```bash
./start.sh --virtual
```

## Test Client

```bash
python3 client_test.py --host 127.0.0.1 --port 19520 --actions F,L,R,S --print-full-detail
```
