# Troubleshooting

## PWM sysfs stops responding after a few writes

Keeping a `duty_cycle` file handle open for repeated writes can become unstable
on some kernel/driver combinations. Prefer one of these approaches:

- Reopen the sysfs file for each write. Shell redirection with `>` follows this pattern.
- If a handle must stay open, call `seekp(0)` before each write and check `good()`/`fail()` after every operation.
- Keep explicit error checks on every write path so failures are not silent.

Also verify:

- `duty_cycle` does not exceed `period`. Some drivers can also reject equality, so this project uses `period - 1 ns` for positive binary output.
- Existing exported channels are handled as a valid state instead of a fatal initialization error.
- `enable`, `period`, and `duty_cycle` writes happen in a driver-compatible order and each return status is checked.

## Build fails during CMake configuration

The project requires Intel RealSense for the normal runtime build. If CMake fails
with a RealSense error, install the RealSense development package for the target
platform and rerun:

```bash
bash scripts/build.sh
```

## Startup cannot find executables

`start.sh` expects normal runtime executables under `build/bin/`. Build first:

```bash
bash scripts/build.sh
```

If you intentionally use a different build directory, pass the same directory to
startup:

```bash
BUILD_DIR=/path/to/build ./start.sh
```
