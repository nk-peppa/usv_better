# Hardware Notes

## RealSense D435i

The D435i camera is a core part of this project. Targets that include the SLAM
execution bridge require Intel RealSense headers and the `realsense2` library.
The build intentionally fails during CMake configuration if RealSense is not
available.

Runtime paths that process `SLI` frames capture from the D435i bridge unless
mock mode is explicitly enabled with `--mock-d435i`, `--virtual`, or
`MOCK_D435I=1`.

## Orange Pi Zero 3 PWM Topology

The current PWM topology is:

- Processor PWM sysfs path for channel 1: `/sys/class/pwm/pwmchip0/pwm1`
- Processor PWM sysfs path for channel 2: `/sys/class/pwm/pwmchip0/pwm2`
- Channel 1 duty file: `/sys/class/pwm/pwmchip0/pwm1/duty_cycle`
- Channel 2 duty file: `/sys/class/pwm/pwmchip0/pwm2/duty_cycle`
- Channel 1 hardware pin: `PH3`
- Channel 2 hardware pin: `PH2`
- 40-pin header mapping: `PH3 -> pin8`, `PH2 -> pin10`

## Startup Modes

Default startup uses real hardware paths:

```bash
./start.sh
```

Virtual startup is explicit:

```bash
./start.sh --virtual
```

`start.sh` prints a mode banner before launching the processor and gateway so the
operator can confirm the active D435i and thruster modes.
