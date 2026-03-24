# BEEFFACE

Apple's compiled ANE binary format (Zin), reverse engineered.

Magic bytes: `0xBEEFFACE`. Apple's joke on Mach-O's `0xFEEDFACE`.

## Prior art

This work builds on:

- [**mdaiter/ane**](https://github.com/mdaiter/ane) (Jan 2026) — Reverse engineered the ANE stack: `_ANEClient`, `_ANEModel`, Espresso layer types, XPC flow to `aned`. First to identify the `0xBEEFFACE` magic and HWX binary format.
- [**maderix/ANE**](https://github.com/maderix/ANE) (Feb 2026) — Training on ANE via `_ANEInMemoryModelDescriptor` and MIL compilation. Forward + backward pass, IOSurface I/O, dynamic weight kernels, INT8 quantization. The definitive work on direct ANE compute access.
- [**eiln/ane**](https://github.com/eiln/ane) (Dec 2022) — Reverse engineered Linux kernel driver for ANE.

The private API execution path (`_ANEClient`, `_ANEInMemoryModel`, IOSurface I/O) is established prior art. This repo does not claim firsts there.

## What this adds

**Zin binary format spec** — Complete byte-level documentation of the compiled ANE program format: Mach-O header, 11 load commands, segment layout (`__PAGEZERO`, `__FVMLIB` const/data, `__TEXT`), section cross-references, thread state descriptors, symbol table. CPU type 128, H17G subtype 9. First published specification of the container internals. See [`docs/HWX_BYTE_MAP.md`](docs/HWX_BYTE_MAP.md).

**17-stage hardware pipeline** — The ANE is not a processor. It's a fixed-function pipeline with 17 named stages: `dma_conv_input`, `dequant1`, `irelu1`, `itranspose1`, `broadcast1`, `scaled_ew`, `post_process`, `postscale`, `abs_or_zero_compare`, `reduction`, `final_scale`, `post_process`, `orelu`, `ogoc`, `postogo`, `postogocrelu`, `otranspose`, `oquant`. Operations are implemented by enabling/disabling stages (0x09=active, 0x00=off, 0xFF=bypass). See [`docs/PROGMEM_OP_DIFF.md`](docs/PROGMEM_OP_DIFF.md).

**HWX byte map** — Word-by-word comparison of compiled binaries reveals the opcode encoding: Word[19] at file offset `0x404C` selects the operation (relu=`0x9361`, abs=`0x9541`). Word[12] controls stage enable flags. Word[4] encodes program size. Only 6 functional bytes differ between operations of the same shape. See [`docs/HWX_BYTE_MAP.md`](docs/HWX_BYTE_MAP.md).

**ZinBuilder** — Parse, patch, and rebuild valid BEEFFACE binaries. Verified: patching relu→abs in a cached `.hwx` changes ANE hardware output. CPU falls back to the source model while ANE executes the patched binary. The binary controls the hardware. See [`src/zin_builder.py`](src/zin_builder.py).

**Hardware LUT fingerprint** — The ANE computes tanh and sigmoid via fixed-function hardware approximations that produce bit-level different results from IEEE 754 fp16 computation. The Midas Fingerprint (`tanh(x) * sigmoid(x)`) compounds this deviation into a deterministic hardware signature. 33/44 values diverge across |x| < 1, up to 11 ULP. Exact matches at +-0.5, +-1.0 suggest LUT knot points. Deterministic 100/100 runs. CPU ground truth verified via `math.tanh`, `numpy.tanh`, and `mpmath.tanh` (50-digit precision) — all three agree. See [`MIDAS_FINGERPRINT.md`](MIDAS_FINGERPRINT.md).

**Espresso vocabulary** — 41 layer types, 50+ elementwise operations, 22+ activation modes mapped including undocumented modes (6=x+1, 13=clamp_min, 14=constant, 25=SiLU, 26=HardSwish) not surfaced by coremltools. See [`docs/ESPRESSO_VOCABULARY.md`](docs/ESPRESSO_VOCABULARY.md).

## The Midas Fingerprint

```
Computation: output = tanh(x) * sigmoid(x)
```

ANE hardware approximations for tanh and sigmoid differ from IEEE 754. Their product compounds the deviation. The result is a deterministic fingerprint of ANE hardware execution.

```
Nick L — March 24, 2026

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

## Repository structure

```
├── MIDAS_FINGERPRINT.md        # Hardware fingerprint evidence
├── src/
│   ├── midas_fingerprint.py    # Fingerprint kill test
│   ├── zin_builder.py          # BEEFFACE binary parser/patcher
│   ├── hwx_format.py           # HWX format analysis tools
│   ├── mil_leaky_relu_kill.py  # MIL compilation proofs
│   └── mil_kill_test2.py       # IOSurface I/O format discovery
└── docs/
    ├── HWX_BYTE_MAP.md         # Zin binary format specification
    ├── PROGMEM_OP_DIFF.md      # 17-stage pipeline analysis
    ├── ESPRESSO_VOCABULARY.md   # Activation/elementwise mode map
    ├── ANE_CRACK_REPORT.md     # Full reverse engineering report
    └── ANE_5_0_EVIDENCE.md     # Precision fingerprinting method
```

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
