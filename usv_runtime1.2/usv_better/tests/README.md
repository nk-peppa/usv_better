# Tests

This directory contains test-only entry points that are not part of the normal
CMake build sequence.

## D435i Smoke Test

Run the D435i smoke test with:

```bash
bash tests/run_d435i_smoke_test.sh
```

For local validation with mock D435i data:

```bash
bash tests/run_d435i_smoke_test.sh --mock
```

The script checks for `build/bin/SelD435iSmokeTest`. If that executable does
not exist, the script compiles it from `tests/SelD435iSmokeTest.cc` and
`src/SlamExecutionLayer.cc`, then runs it with the provided arguments.
