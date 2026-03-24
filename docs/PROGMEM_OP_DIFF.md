# ANE Program Memory Operation Diff Analysis

Byte-level reverse engineering of the 262KB ANE program memory format to identify
which bytes control operation type (relu vs add vs matmul).

**Date:** 2026-03-23
**Platform:** M5 Air, ANE H13+
**Source struct:** `ane_progmem_struct.h`

---

## 1. Dump-to-Operation Mapping

Identified via non-zero byte count matching the kill test results in the struct header:

| File | Operation | Non-zero bytes | Addr |
|------|-----------|---------------|------|
| `ane_progmem_0x90d824880_0005_262144B.bin` | **add_64** | 55,604 | 0x90d824880 |
| `ane_progmem_0x90d824e80_0010_262144B.bin` | **relu_64** | 61,075 | 0x90d824e80 |
| `ane_progmem_0x90d824260_0015_262144B.bin` | **matmul_64x128** | 61,331 | 0x90d824260 |
| `ane_progmem_0x90d824d40_0020_262144B.bin` | **matmul_256x512** | 61,843 | 0x90d824d40 |
| `ane_progmem_0x90d8242a0_0025_262144B.bin` | **matmul_512x1024** | 62,243 | 0x90d8242a0 |

**Confirmation:** ASCII op names found in header records at 0x0FC9 node type:
- add_64: contains "relu_0", "x", "add_0" (note: "relu_0" present because add model includes a relu subgraph)
- matmul_64x128: contains "matmul_0", "main", "x"
- relu_64: no 0x0FC9 records (different header structure)

---

## 2. Structural Architecture: The Kernel Pipeline Region (0x7180-0xADBF)

### Critical Finding: Operation Type Determines Pipeline Structure

The single most important finding: **the operation type is NOT encoded as a single byte flag**.
Instead, the ANE compiler generates fundamentally different kernel pipeline configurations:

| Operation | Pipeline passes | Has stage names | Kernel data size | Structure |
|-----------|----------------|-----------------|-----------------|-----------|
| **add_64** | 0 | NO | 1,002 bytes | Address table + hash refs only |
| **relu_64** | 4 | YES | 6,373 bytes | 4x full pipeline pass |
| **matmul_64x128** | 4 | YES | 6,386 bytes | 4x full pipeline pass |
| **matmul_256x512** | 2 | YES | 2 pipeline passes | 2x full pipeline pass |
| **matmul_512x1024** | 2 | YES | 2 pipeline passes | addr block + 2x pipeline pass |

**add_64 is structurally unique:** it has zero pipeline stage names (no dma_conv_input, no irelu,
no orelu, nothing). Its kernel pipeline region contains only memory addresses (XX XX XX 0D 09 00 00 00
and XX XX 05 18 09 00 00 00 patterns) and hash-like data. This means the add operation is
implemented entirely through DMA + register configuration, not through the named pipeline stages.

---

## 3. Pipeline Pass Internal Structure

Each pipeline pass follows a fixed stage ordering with consistent byte spacing:

```
Stage                  Relative offset    Size (to next stage)
---------------------------------------------------------------
weight.bin / _input1   +0x0055            19 bytes (name only, no descriptor)
_input1                +0x0068            88 bytes
dequant1               +0x00C0            96 bytes (0x60)
irelu1                 +0x0120            96 bytes
itranspose1            +0x0180            96 bytes
broadcast1             +0x01E0            96 bytes
epsilon                +0x01F0            16 bytes (embedded in broadcast)
scaled_ew              +0x0240            104 bytes
post_process           +0x02A8            88 bytes
postscale              +0x0300            96 bytes
abs_or_zero_compare    +0x0360            96 bytes
reduction              +0x03C0            96 bytes
final_scale            +0x0420            96 bytes
epsilon (2nd)          +0x0480            96 bytes
post_process (2nd)     +0x04E0            96 bytes
orelu                  +0x0540            192 bytes (0xC0, double-sized!)
ogoc                   +0x0600            16 bytes
postogo                +0x0610            80 bytes
postogocrelu           +0x0660            96 bytes
otranspose             +0x06C0            96 bytes
oquant                 +0x0720            96 bytes
dma_conv_output        +0x0780            (end of pass)
```

Offsets are relative to the start of non-zero data for each model's first pass.
Subsequent passes start with `dma_conv_input1` at approximately +0x0860 from the previous pass start.

---

## 4. The 85-Byte Kernel Pipeline Preamble

Every model's kernel pipeline data begins with an identical 85-byte preamble structure
(at model-specific absolute offsets within 0x7180-0xADBF):

| Model | Preamble starts at | First stage name at |
|-------|--------------------|---------------------|
| add_64 | 0x07780 | N/A (no stages) |
| relu_64 | 0x07180 | 0x071D5 (weight.bin) |
| matmul_64x128 | 0x07DA0 | 0x07DF5 (weight.bin) |
| matmul_256x512 | 0x072C0 | 0x07315 (weight.bin) |
| matmul_512x1024 | 0x07D60 | 0x085C0 (dma_conv_input1) |

### Preamble byte-level diff (4 models aligned):

```
Offset  add_64   relu_64  mm_64x128  mm_256x512  Classification
------  ------   -------  ---------  ----------  --------------
+0x00   0f 2f 05 ac 4f 29 32 b9     (identical: HASH_REF tag, ID=0x2F0F)
+0x08   00       00       00         00          (identical)
+0x09   0x00     0x00     0xEC       0xAC        MATMUL WEIGHT SIZE
+0x0A   0x00     0x00     0x01       0x02        MATMUL WEIGHT SIZE (high byte)
+0x10   20 00 01 00 04 2e            (identical: version "." encoding)
+0x20   b4 b9 33                     (identical: hash value)
+0x30   20 00 02 00 04 2e 2e         (identical: version ".." encoding)
+0x40   0x00     0x71     0x95       0xB9        PER-MODEL UNIQUE (low byte)
+0x41   0x00     0x0B     0x0B       0x0B        *** ADD-SPECIFIC: 0x00 vs 0x0B ***
+0x42   0x00     0x34     0x34       0x34        *** ADD-SPECIFIC: 0x00 vs 0x34 ***
+0x50   0xA0     0x28     0x28       0x28        *** ADD-SPECIFIC: 0xA0 vs 0x28 ***
+0x51   0x48     0x00     0x00       0x00        *** ADD-SPECIFIC: 0x48 vs 0x00 ***
+0x52   0x05     0x0A     0x0A       0x0A        *** ADD-SPECIFIC: 0x05 vs 0x0A ***
+0x53   0x18     0x00     0x00       0x00        *** ADD-SPECIFIC: 0x18 vs 0x00 ***
+0x54   0x09     0x08     0x08       0x08        *** ADD-SPECIFIC: 0x09 vs 0x08 ***
```

### Candidate Operation-Type Control Bytes

**Confidence: HIGH** -- These 7 bytes distinguish add from all pipeline-based operations:

| Preamble offset | add_64 | relu_64/matmul | Hypothesis |
|-----------------|--------|----------------|------------|
| **+0x41** | 0x00 | 0x0B | Pipeline enable flag (0x00=disabled, 0x0B=pipeline present) |
| **+0x42** | 0x00 | 0x34 | Pipeline type selector (0x00=DMA-only, 0x34=full pipeline) |
| **+0x50** | 0xA0 | 0x28 | Dispatch mode (0xA0=register-based add, 0x28=pipeline-based) |
| **+0x51** | 0x48 | 0x00 | Secondary dispatch flag |
| **+0x52** | 0x05 | 0x0A | Execution path selector (0x05=elementwise, 0x0A=staged pipeline) |
| **+0x53** | 0x18 | 0x00 | DMA config (0x18=direct memory, 0x00=pipeline-routed) |
| **+0x54** | 0x09 | 0x08 | Address space selector |

**Confidence: MEDIUM** -- Weight size encoding (matmul-specific):

| Preamble offset | add_64 | relu_64 | mm_64x128 | mm_256x512 | Hypothesis |
|-----------------|--------|---------|-----------|------------|------------|
| **+0x09** | 0x00 | 0x00 | 0xEC | 0xAC | Weight tensor size (low byte) |
| **+0x0A** | 0x00 | 0x00 | 0x01 | 0x02 | Weight tensor size (high byte) |

Values: mm_64x128 = 0x01EC (492), mm_256x512 = 0x02AC (684). These encode
weight buffer metadata. For operations without weights (add, relu), these are zero.

---

## 5. Stage Descriptor Internal Structure (96 bytes per stage)

Each named stage (irelu1, orelu, dequant1, etc.) occupies a 96-byte (0x60) descriptor.
Format relative to the ASCII stage name:

```
+0x00: ASCII stage name (8 bytes, null-padded)
+0x08: Config field A (8 bytes) -- includes scale factor (0x3F800000 = 1.0f at +0x08-0x0B)
+0x10: Primary reference (8 bytes) -- tag + flags (e.g., 08 90 34 6b 01 00 00 05)
+0x18: Hash dispatch (8 bytes) -- 71 0C 02 pattern (pipeline hash)
+0x20: Base reference (8 bytes) -- d0 a1 34 6b 01 fixed prefix
+0x28: Secondary address (8 bytes) -- ** KEY VARIABLE FIELD **
+0x30: Pipeline state pointer (8 bytes) -- 58 XX 82 0d 09 pattern (address)
+0x38: Reserved (8 bytes) -- always zero
+0x40: Counter/type field (8 bytes) -- 02 00 00 00 00 00 00 00
+0x48: Buffer address (8 bytes) -- XX XX XX 0d 09 00 00 00 (per-model address)
+0x50: Instance counter (8 bytes) -- 01 00 00 00 00 00 00 00
+0x58: Tail config (8 bytes) -- 00 00 80 3f XX XX XX XX (float 1.0 + flags)
```

### Cross-Model Stage Comparison

When aligned by stage name, the stage descriptors are **remarkably stable** across
relu_64, matmul_64x128, and matmul_256x512. The only differences are:

**Per-model address bytes (confidence: HIGH -- these are buffer addresses, NOT op control):**
- +0x49, +0x4A: High bytes of buffer address. Values cluster by model memory allocation:
  - relu_64 + matmul_256x512: 0xXX 0xA5 (shared address space)
  - matmul_64x128: 0xXX 0x8A (different allocation)
  - matmul_512x1024: 0xXX 0xA5 (same space as relu)

**Per-model counter byte:**
- +0x31: Increments per stage within a pass (dequant=C0/C1, irelu=C1, scaled_ew=C2, etc.)
  - matmul_512x1024 adds +0x08 to each counter (0xC8, 0xC9, 0xCA...)
  - This is a pipeline stage index, not an operation selector

**CRITICAL: No stage descriptor byte changes between relu and matmul that could be an "operation type" selector.**
The pipeline stages are operation-agnostic execution units.

---

## 6. The orelu Stage: Activation Function Analysis

The orelu (output relu) stage is 192 bytes -- double the standard 96 bytes.
Key field analysis across all instances:

### orelu +0x28 (Secondary Address Field, 8 bytes)

| Model | Instance | +0x28 value | +0x5C value | Interpretation |
|-------|----------|-------------|-------------|----------------|
| relu_64 | #1 | 88 e0 04 19 09 00 00 00 | 09 00 00 00 | Active: valid output buffer addr |
| relu_64 | #2 | 68 64 04 19 09 00 00 00 | 09 00 00 00 | Active: valid output buffer addr |
| relu_64 | #3 | 00 f0 ff ff ff ff ff ff | 09 00 00 00 | Sentinel: disabled/passthrough |
| relu_64 | #4 | 00 00 00 00 00 00 00 00 | 00 00 00 00 | Terminal: last pass cleanup |
| mm_64x128 | #1 | 00 f0 ff ff ff ff ff ff | ff ff ff ff | **Disabled: 0xFFF0... sentinel** |
| mm_64x128 | #2 | 78 c3 06 19 09 00 00 00 | 09 00 00 00 | Active |
| mm_64x128 | #3 | d8 35 05 19 09 00 00 00 | ff ff ff ff | Mixed |
| mm_64x128 | #4 | 00 00 00 00 00 00 00 00 | 00 00 00 00 | Terminal |
| mm_256x512 | #1 | 88 e0 04 19 09 00 00 00 | 09 00 00 00 | Active (identical to relu #1) |
| mm_256x512 | #2 | 68 64 04 19 09 00 00 00 | 09 00 00 00 | Active (identical to relu #2) |
| mm_512x1024| #1 | 68 64 04 19 09 00 00 00 | 09 00 00 00 | Active |

### orelu +0x5C (Tail Config Flags, 4 bytes)

Three distinct values observed:
- **0x09000000**: Active relu -- output goes to a valid buffer
- **0xFFFFFFFF**: Disabled/passthrough -- relu is bypassed
- **0x00000000**: Terminal pass -- cleanup, no output routing

**Confidence: MEDIUM-HIGH** -- The +0x5C field appears to be a relu enable/disable flag:
- `09 00 00 00` = relu active (clamp negative values to zero)
- `FF FF FF FF` = relu bypassed (pass through unchanged)
- `00 00 00 00` = terminal state (no output)

This is seen in matmul_64x128's first orelu instance: +0x5C = `FF FF FF FF` while the
corresponding relu_64 instance has +0x5C = `09 00 00 00`. The matmul needs the pipeline
stages but does NOT want relu activation applied to its intermediate results.

---

## 7. Summary of All Identified Control Bytes

### Operation-Level Control (in the 85-byte preamble)

These bytes determine the fundamental execution mode:

| Absolute offset formula | Byte | add_64 | relu/matmul | Confidence | Role |
|------------------------|------|--------|-------------|------------|------|
| preamble_start + 0x41 | 1 | 0x00 | 0x0B | HIGH | Pipeline enable (0=off, 0x0B=on) |
| preamble_start + 0x42 | 1 | 0x00 | 0x34 | HIGH | Pipeline type (0=DMA-only, 0x34=staged) |
| preamble_start + 0x50 | 1 | 0xA0 | 0x28 | HIGH | Dispatch mode |
| preamble_start + 0x51 | 1 | 0x48 | 0x00 | MEDIUM | Secondary dispatch |
| preamble_start + 0x52 | 1 | 0x05 | 0x0A | HIGH | Exec path (0x05=ew, 0x0A=pipeline) |
| preamble_start + 0x53 | 1 | 0x18 | 0x00 | MEDIUM | DMA routing |
| preamble_start + 0x54 | 1 | 0x09 | 0x08 | MEDIUM | Address space |

**To convert add to relu:** Patch these 7 preamble bytes AND insert pipeline stage
descriptors. This is NOT a simple byte-patch -- it requires adding ~5KB of pipeline
configuration data.

### Activation Function Control (in orelu stage descriptor)

| Stage-relative offset | Byte count | Values | Confidence | Role |
|----------------------|-----------|--------|------------|------|
| orelu_name + 0x5C | 4 | 0x09000000/0xFFFFFFFF/0x00000000 | MEDIUM-HIGH | Relu enable/disable |
| orelu_name + 0x28 | 8 | address or 0x00F0FFFFFFFFFFFF | MEDIUM | Output buffer (sentinel=disabled) |

**To disable relu in a relu model:** Patch orelu +0x5C from `09 00 00 00` to `FF FF FF FF`
and orelu +0x28 from a valid address to `00 F0 FF FF FF FF FF FF`.

### Dimension/Weight Parameters (in preamble)

| Preamble offset | Byte count | add/relu | mm_64x128 | mm_256x512 | Confidence | Role |
|-----------------|-----------|----------|-----------|------------|------------|------|
| +0x09, +0x0A | 2 | 0x0000 | 0x01EC | 0x02AC | MEDIUM | Weight buffer size metadata |

---

## 8. Key Architectural Insights

1. **add is fundamentally different from relu/matmul.** The add operation does not use
   the ANE pipeline stages at all. It operates through direct DMA register configuration
   in the kernel pipeline region. relu and matmul both use the full 17-stage pipeline.

2. **relu and matmul share identical pipeline topology.** The stage descriptors between
   relu_64 and matmul models differ only in per-model buffer addresses. The pipeline
   stages are generic execution units, not operation-specific.

3. **The operation is defined by the COMBINATION of:** (a) preamble dispatch bytes that
   select pipeline vs DMA-only mode, (b) the number and configuration of pipeline passes,
   (c) the activation enable/disable flags in orelu/irelu stages, and (d) the weight
   references and dimension parameters.

4. **Pipeline pass count correlates with tensor dimensions:**
   - add_64: 0 passes (elementwise, no pipeline)
   - relu_64: 4 passes (64-element tensor, 4 ANE tiles)
   - matmul_64x128: 4 passes (small matmul, 4 tiles)
   - matmul_256x512: 2 passes (medium matmul, tiled differently)
   - matmul_512x1024: 2 passes (large matmul, 2 macro-tiles)

5. **orelu +0x5C = 0xFFFFFFFF is the "relu bypass" sentinel.** This allows the pipeline
   to run without applying relu activation. Seen in matmul first-pass orelu where
   intermediate results must not be clamped.

---

## 9. Patching Feasibility Assessment

### Easy patch: Disable relu activation in a relu model
- Patch orelu +0x5C: `09 00 00 00` -> `FF FF FF FF` (in each pass)
- Patch orelu +0x28: valid address -> `00 F0 FF FF FF FF FF FF`
- **Risk: LOW** -- well-attested pattern, matmul models already use this sentinel

### Medium patch: Change matmul dimensions
- Modify preamble +0x09/+0x0A (weight size)
- Adjust pipeline pass count (add/remove dma_conv_input blocks)
- Update buffer addresses throughout
- **Risk: MEDIUM** -- requires consistent address updates

### Hard patch: Convert add to relu (or vice versa)
- Requires switching from DMA-only (0 passes) to full pipeline (4 passes)
- Must generate ~5KB of pipeline stage descriptors
- Must patch 7 preamble bytes
- **Risk: HIGH** -- essentially recompiling the kernel

### Hardest: Create a novel operation (e.g., GELU, SiLU)
- The pipeline stages are FIXED (irelu = integer relu, orelu = output relu)
- No evidence of a "function selector" byte that switches relu to a different activation
- The activation function appears HARDWIRED into the irelu/orelu hardware units
- Creating GELU would require either: (a) finding hidden activation selector bits in the
  stage descriptors, or (b) composing multiple ANE programs (relu + scaled_ew + add)
- **Risk: VERY HIGH** -- may be architecturally impossible without multi-program composition
