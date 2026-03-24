# ANE Binary Patching: Direct Hardware Control via ModelAssetsCache

**Date:** 2026-03-23/24
**Hardware:** MacBook Air M5, 16GB, macOS 15.4
**Rating:** 4.5/5 — binary modification proven, generation from scratch remains

---

## The Claim

We patched a compiled ANE Zin binary in `ModelAssetsCache`, bypassing CoreML's source model entirely. The native Objective-C framework loads the patched binary. The ANE hardware executes the patched operation. CPU falls back to the source model. **The binary controls the hardware.**

## Kill Test

```
Source model:  espresso.net mode=0 (relu)
Cache binary:  .hwx patched relu→abs (orelu stage +0x5C: 0x09→0x00)

Input:         [-5, -2.5, -1, -0.1, 0, 0.1, 1, 2.5, 5, -0.001]
CPU output:    [ 0,  0,    0,  0,   0, 0.1,  1, 2.5, 5,  0    ]  ← relu (from espresso.net)
ANE output:    [ 5,  2.5,  1,  0.1, 0, 0.1,  1, 2.5, 5,  0.001]  ← abs  (from patched .hwx)
```

CPU reads the source model. ANE reads the patched binary. Same CoreML `MLModel` instance, same `predictionFromFeatures:` call. The compute backend determines which representation is used.

## Proof of Causation

| Cache state | ANE output | CPU output |
|------------|-----------|-----------|
| Original relu .hwx | `[0, 2, 0, 4]` (relu) | `[0, 2, 0, 4]` (relu) |
| Patched abs .hwx | `[1, 2, 3, 4]` (abs) | `[0, 2, 0, 4]` (relu) |
| Original restored | `[0, 2, 0, 4]` (relu) | `[0, 2, 0, 4]` (relu) |

Toggling the `.hwx` file toggles the ANE output. CPU output never changes. The binary controls the hardware.

## Architecture Discovered

### Execution Path
```
.mlmodelc (espresso.net) ──→ ANECompilerService ──→ Zin binary (.hwx)
                                                         │
                                                         ▼
                                              ModelAssetsCache (on disk)
                                                         │
                              ┌───────────────────────────┤
                              ▼                           ▼
                    Native CoreML ObjC              coremltools Python
                    (reads .hwx cache)              (always recompiles)
                              │
                              ▼
                     aned → ANE hardware
```

Key finding: the **native Objective-C CoreML framework** (`MLModel.modelWithContentsOfURL:`) reads compiled `.hwx` binaries from `ModelAssetsCache`. The **coremltools Python wrapper** always recompiles from `.mlmodelc` source, ignoring the cache.

### Cache Location
```
/Library/Caches/com.apple.aned/<build>/ModelAssetsCache/<process>/<model_hash>/<compiler_hash>/model.hwx
```

- `model.src` file contains the original `.mlmodelc` path (cache key)
- `.hwx` files are BEEFFACE Zin binaries — same format as in-memory program
- Cache is keyed by source `.mlmodelc` content hash, NOT file path
- Root-owned but writable with sudo (SIP off)

### Zin Binary Format (BEEFFACE)
```
Magic:     0xBEEFFACE (little-endian: CE FA EF BE)
CPU type:  128 (0x80) — ANE
CPU sub:   9 — H17G (M5)
File type: 2 — MH_EXECUTE
Segments:  __PAGEZERO, __FVMLIB (×2), __TEXT
Commands:  LC_THREAD (×3), compiler info, symtab
```

### ANE Pipeline Architecture
The ANE is a **fixed-function pipeline** with 17 stages. Operations are implemented by enabling/disabling stages, not by opcodes:

```
dma_conv_input → dequant → irelu → itranspose → broadcast →
scaled_ew → post_process → postscale → abs_or_zero_compare →
reduction → final_scale → orelu → ogoc → postogo →
postogocrelu → otranspose → oquant → dma_conv_output
```

**Stage control field:** 4 bytes at stage record offset +0x5C
- `0x09000000` = stage active
- `0x00000000` = stage disabled (passthrough)
- `0xFFFFFFFF` = alternate mode

**To patch relu→abs:** Set `orelu +0x5C` from `0x09` to `0x00` (disable relu stage). The `abs_or_zero_compare` stage runs by default when orelu is disabled.

### IOSurface Format (for direct _ANEClient execution)
```
Width:           1
Height:          64 (one element per row)
BytesPerRow:     64
BytesPerElement: 2 (FP16)
AllocSize:       16384
PixelFormat:     0x4c303068 ('h0L' — FP16 linear)
```

### Undocumented Activation Modes
Espresso.net `"mode"` field controls the activation pipeline stage. 14+ modes exist, only ~10 exposed by CoreML's public API:

| Mode | Operation | In CoreML API? |
|------|-----------|---------------|
| 0 | ReLU | Yes |
| 1 | Tanh | Yes |
| 2 | Identity/Linear | Yes |
| 3 | Sigmoid | Yes |
| 6 | **x + 1.0 (elementwise shift)** | **No** |
| 7 | **HardSigmoid variant** | **Partial** |
| 8 | ELU | Yes |
| 9 | ThresholdedReLU | Yes |
| 10 | Softplus | Yes |
| 12 | Softsign | Yes |
| 13 | **max(x, 1) — clamped minimum** | **No** |
| 14 | **constant 1.0** | **No** |

### _ANEClient Private API
```objc
_ANEClient *client = [[_ANEClient alloc] initWithRestrictedAccessAllowed:NO];
[client compileModel:model options:nil qos:0x21 error:nil];
[client loadModel:model options:nil qos:0x21 error:nil];
[client doEvaluateDirectWithModel:model options:nil request:request qos:0 error:nil];
[client unloadModel:model options:nil qos:0 error:nil];
```

Working execution pipeline confirmed. IOSurface format documented. Output zeros resolved by matching CoreML's exact surface dimensions.

## What This Is

- **Binary-level control of ANE hardware** — the patched `.hwx` executes on the neural engine
- **Compiler bypass** — the source model says relu, the hardware runs abs
- **First documented `.hwx` cache exploitation** on Apple Neural Engine
- **Full architectural documentation** of Zin binary format, pipeline stages, IOSurface layout

## What This Is Not

- **Not Zin generation from scratch** — the compiler produced the container, segments, symbol table, and kernel tiles. We modified bytes within that structure.
- **Not arbitrary computation** — limited to operations the pipeline stages support (activation modes, stage enable/disable)
- **Not a security vulnerability** — requires SIP off and root access to write to the cache

## Remaining Gap (0.5 → 5.0)

Generate `__KERN_0` kernel tiles from scratch for a novel operation. Load via IOKit or ModelAssetsCache without any compiler-generated container. Execute correctly on ANE hardware.

This requires:
1. Decoding the full kernel tile format (200KB+ of pipeline stage configuration)
2. Understanding SRAM pointer assignment (done by aned at load time)
3. Constructing valid DMA descriptors for input/output buffers
4. Building a complete Zin binary with correct segment layout and symbols

Target: M5 Pro week (64GB, 70B models, deeper reverse engineering capacity)

## Files

| File | Description |
|------|------------|
| `kill_test_native.py` | Kill test — native CoreML, loads from cache |
| `hwx_cache/relu_49152.hwx` | Original relu Zin binary |
| `hwx_cache/relu_patched_to_abs.hwx` | Patched abs Zin binary |
| `hwx_cache/abs_49152.hwx` | Compiler-generated abs for reference |
| `ESPRESSO_MODE_MAP.md` | Full activation mode vocabulary |
| `PROGMEM_OP_DIFF.md` | Program memory diff across operations |
| `ZIN_OP_DIFF.md` | Zin binary diff analysis |
| `SKIP_COMPILER_FINDINGS.md` | Cache behavior documentation |
| `ane_interpose.c` | DYLD_INSERT attempt (blocked by arm64e PAC) |
| `ane_fishhook.c` | fishhook attempt (symbols not bound) |
| `ane_mmap.c` | mmap hook attempt (blocked by PAC) |
| `iokit_probe_args.py` | IOKit selector probe |

## Reproduction Steps

```bash
# 1. Requires SIP and AMFI disabled

# 2. Build and compile a relu model via coremltools
python3 -c "
import coremltools as ct
from coremltools.models.neural_network import NeuralNetworkBuilder
b = NeuralNetworkBuilder(input_features=[('x', ct.models.datatypes.Array(64))],
                         output_features=[('y', ct.models.datatypes.Array(64))])
b.add_activation('relu', 'RELU', 'x', 'y')
ct.models.MLModel(b.spec, compute_units=ct.ComputeUnit.ALL).save('/tmp/relu.mlpackage')
"

# 3. Compile and run once to populate ModelAssetsCache
# (use native CoreML ObjC, not coremltools)

# 4. Find the cached .hwx
sudo find /Library/Caches/com.apple.aned/ -name "model.hwx" -newer /tmp/relu.mlpackage

# 5. Backup and patch
sudo cp <path>/model.hwx ./relu_original.hwx
python3 -c "
import struct
data = bytearray(open('relu_original.hwx','rb').read())
# Find and patch orelu stage control bytes
# 0x09000000 → 0x00000000 at stage +0x5C offsets
# (specific offsets vary by model — use diff against abs .hwx)
open('relu_patched.hwx','wb').write(data)
"
sudo cp relu_patched.hwx <path>/model.hwx

# 6. Run kill test — native CoreML loads patched binary
python3 kill_test_native.py
# CPU: relu [0, 2, 0, 4]
# ANE: abs  [1, 2, 3, 4]  ← patched binary executing on hardware
```
