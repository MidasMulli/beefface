# BEEFFACE

Reverse engineering the Apple Neural Engine on M5 silicon.

Direct compilation and execution of arbitrary programs on ANE hardware via private `_ANEInMemoryModel` API, bypassing CoreML entirely. Hardware execution proven via precision fingerprinting — the ANE's fixed-function approximations for transcendentals produce bit-level different results from IEEE 754 computation.

## The Midas Fingerprint

A hardware fingerprint function that runs only on Apple Neural Engine:

```
Computation: output = tanh(x) * sigmoid(x)
Pathway:     Hand-written MIL -> _ANEInMemoryModel -> ANE hardware
```

The ANE implements tanh and sigmoid via fixed-function hardware approximations. Their product compounds the deviation into a deterministic fingerprint no CPU can reproduce.

```
Nick L — March 24, 2026 — signed by Apple Neural Engine

       ord       x          ANE          CPU   ANE_hex  CPU_hex  ULP
  N     78  0.6143     0.354004     0.354980      35aa     35ae    4
  i    105  0.8267     0.469971     0.472168      3785     378e    9
  c     99  0.7793     0.445068     0.447266      371f     3728    9
  k    107  0.8423     0.478027     0.479980      37a6     37ae    8
  L     76  0.5986     0.344727     0.346191      3584     358a    6
  Mar       0.0968     0.050415     0.050598      2a74     2a7a    6
  24        0.7744     0.442627     0.444580      3715     371d    8
  26        0.8389     0.476318     0.478271      379f     37a7    8

  35aa:3785:371f:37a6:3584:2a74:3715:379f

  8/8 divergent from CPU. Deterministic 100/100.
```

**Kill test results:**
- ANE differs from CPU: **True** (33/44 values in extended sweep, up to 11 ULP)
- Deterministic 100/100: **True**
- Public CoreML API matches ANE: **False** (CoreML routes to CPU)
- CPU ground truth verified via `math.tanh`, `numpy.tanh`, `mpmath.tanh` (50-digit precision) — all agree

## What's here

### Proof

- [`MIDAS_FINGERPRINT.md`](MIDAS_FINGERPRINT.md) — Full evidence document with fingerprint table, compilation pathway, and kill test results
- [`src/midas_fingerprint.py`](src/midas_fingerprint.py) — The kill test script (requires entitled binary + SIP off)

### ANE Reverse Engineering

- [`docs/ANE_CRACK_REPORT.md`](docs/ANE_CRACK_REPORT.md) — Complete reverse engineering report: binary patching, Zin format, pipeline architecture
- [`docs/HWX_BYTE_MAP.md`](docs/HWX_BYTE_MAP.md) — Byte-level specification of the BEEFFACE Zin binary format
- [`docs/ESPRESSO_VOCABULARY.md`](docs/ESPRESSO_VOCABULARY.md) — Full map of espresso.net layer types, activation modes, elementwise operations
- [`docs/PROGMEM_OP_DIFF.md`](docs/PROGMEM_OP_DIFF.md) — Program memory analysis: 17-stage pipeline, stage names, kernel tile structure
- [`docs/ANE_5_0_EVIDENCE.md`](docs/ANE_5_0_EVIDENCE.md) — Precision fingerprinting methodology and evidence

### Tools

- [`src/zin_builder.py`](src/zin_builder.py) — Parse, modify, and rebuild BEEFFACE Zin binaries
- [`src/hwx_format.py`](src/hwx_format.py) — HWX binary format parser with opcode identification and patching
- [`src/mil_leaky_relu_kill.py`](src/mil_leaky_relu_kill.py) — MIL compilation proof: leaky relu, clip, custom ops on ANE
- [`src/mil_kill_test2.py`](src/mil_kill_test2.py) — IOSurface I/O format discovery (PlaneStride=64 fix)

## Key findings

**The ANE is not a processor. It's a configurable fixed-function pipeline with 17 hardware stages.** There is no ISA. Operations are implemented by enabling/disabling pipeline stages and configuring their parameters.

**Zin binary format (BEEFFACE):** Mach-O-like container with magic `0xBEEFFACE`, CPU type 128 (ANE), H17G subtype 9. Contains load commands, thread state descriptors, and two executable sections: `__TEXT.__text` (kernel tile descriptors) and `__TEXT.__const` (pipeline configuration).

**Execution pathway:** Hand-written MIL text is compiled via `_ANEInMemoryModelDescriptor` (private API in `AppleNeuralEngine.framework`). The ANE compiler service generates a Zin binary. The binary is loaded into the ANE hardware program table (`programHandle != 0`). I/O occurs via IOSurfaces with data at `PlaneStride`-byte offsets per channel.

**Hardware execution proof:** The ANE's tanh and sigmoid implementations produce deterministic bit-level deviations from IEEE 754 fp16 computation. The pattern is consistent with fixed-function hardware approximation: largest errors near zero (high curvature region), exact agreement at saturation. Verified against ground truth computed via three independent methods.

## Requirements

- Apple Silicon Mac (tested on M5)
- macOS with SIP and AMFI disabled (security research configuration)
- Python binary signed with `com.apple.ane.iokit-user-access` entitlement
- `coremltools`, `numpy`, `pyobjc`

## Platform

- MacBook Air M5, 16 GB, macOS 26.3.1
- ANE: H17G, 12.19 TFLOPS FP16
- Zin compiler: v9.202.0

## Legal

This project constitutes security research and interoperability analysis under DMCA Section 1201(f) and (j). All work was performed on hardware owned by the researcher for the purpose of understanding the Apple Neural Engine's computation model and achieving interoperability with the ANE hardware interface. No proprietary source code was accessed. All findings are derived from black-box observation of compiler inputs and outputs, binary format analysis, and public framework interfaces.

## License

MIT

## Author

Nick L ([@NickLo641579](https://x.com/NickLo641579))
