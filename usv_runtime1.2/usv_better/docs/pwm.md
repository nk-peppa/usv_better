# PWM and Thruster Validation

## Runtime Mapping

Runtime mapping used by the current script and actuator executable:

- Neutral duty: `0 ns`
- Period: `20000000 ns` (`50 Hz`)
- Input format: left/right percentage values in the range `0` to `100`
- Negative values are rejected by the validation flow
- `0%` maps to duty `0 ns` as a hard stop
- Any positive value maps to `period - 1 ns` in the current binary validation flow

`ThrusterActuator.cc` still contains shaping configuration fields for future
hardware behavior, but current output behavior is binary: non-positive input maps
to `0 ns`; positive input maps to the active period minus one nanosecond.

## Validation Script

Dry-run mode does not write hardware sysfs files:

```bash
bash tools/thruster_test_runner.sh
```

Hardware mode writes PWM duty cycles to sysfs and requires root privileges:

```bash
sudo bash tools/thruster_test_runner.sh --hw
```

The script writes PWM duty cycles directly to sysfs and does not generate a log
file by default.

## Interactive Actuator Helper

After building the project, run:

```bash
bash tools/ThrusterActuator_test_runner.sh
```

The helper looks for `build/bin/ThrusterActuator` and sends interactive
`F`/`L`/`R`/`S` commands to it.
