# ZinIrOpLayerOpCodeType: the layer opcode enum

Compiler: `ZinAneCompiler-10.26.6`, extracted from the macOS 27.0 (26A5421a) dyld shared cache.
131 opcodes, `0x00` through `0x82`, terminating at `INVALID`.

Recovered from the name table that `ZinIrEnumToStringUtil::OpCodeToString` indexes, rather than
inferred from class names.

## Method

Reproducible on any build in a few minutes. The addresses are for this build and will move.

1. Extract `ANECompiler` from the shared cache, via `dyld_shared_cache_extract_dylibs_progress`
   in Apple's `/usr/lib/dsc_extractor.bundle`.
2. `nm -n ANECompiler | grep OpCodeToString` locates
   `__ZN21ZinIrEnumToStringUtil14OpCodeToStringE22ZinIrOpLayerOpCodeType` at `0x20bb57ef8`.
3. Disassemble two instructions. The function is a table index with no bounds check:

```
adrp x9, #0x270363000
add  x9, x9, #0xe10          ; table base 0x270363E10
ldr  x1, [x9, w0, uxtw #3]   ; table[opcode], 8-byte entries
mov  x0, x8
b    std::string ctor
```

4. The table is in `(__DATA_CONST,__const)`, addr `0x270360c90`, offset `33089836`. Convert the
   table VA to a file offset and read 8-byte entries.
5. Entries are **chained fixups, not raw pointers**. Take the low 36 bits and add the arm64e
   shared-cache base `0x180000000` to get a `__cstring` VA (addr `0x20c35f0f0`, offset `30454000`).

Reading past `0x82` walks into an adjacent, unrelated string table (`placeholder`, `static`,
`runtime`), which is how the end of the enum is identified. The sibling `NonLinearModeToString`
in the same class does bounds-check, at `#0x2f`, giving 48 non-linear modes.

## Cross-check

Two of these values were independently derived from a decompile of `ClassifyConcatGroup`, which
accepts a conv only when its input producer is op-kind `7` and it joins a connected group of
op-kind `0x60`. This table gives `7 = CONCAT` and `0x60 = NEFUSED_CONV`, so the condition reads as
a group of convs fed by a concat, matching the function name. The two derivations were independent.

## Build sensitivity

The ordering is not stable across builds. maderix ("Inside the M4 ANE", part 4b) reports `0x5d`
for the conv layer and `0x64` for the bypass layer on M4. This build has `NEFUSED_CONV` at `0x60`
and `NEFUSED_BYPASS` at `0x68`, shifts of +3 and +4. Because the two deltas differ, ops were
inserted at more than one point: three before conv, and one more between conv and bypass. An
opcode value from one build must be re-read on another, not carried across.

## Table

| opcode | name | | opcode | name |
|---|---|---|---|---|
| `0x00` | `CONV` | | `0x42` | `PLANE_WRITER` |
| `0x01` | `POOL` | | `0x43` | `SORT` |
| `0x02` | `SCALE_BIAS` | | `0x44` | `TOP_K` |
| `0x03` | `TERNARY_DYNAMIC_GOC` | | `0x45` | `RCAS` |
| `0x04` | `ACTIVATION` | | `0x46` | `INDEX` |
| `0x05` | `EW` | | `0x47` | `NMS` |
| `0x06` | `SCALED_EW` | | `0x48` | `DROPOUT` |
| `0x07` | `CONCAT` | | `0x49` | `TYPE_CAST` |
| `0x08` | `SPLIT` | | `0x4a` | `STOCHASTIC_ROUND` |
| `0x09` | `COPY` | | `0x4b` | `RANDOM_GENERATOR` |
| `0x0a` | `FLATTEN` | | `0x4c` | `LINEAR` |
| `0x0b` | `UNFLATTEN` | | `0x4d` | `RINGBUFFER_WRITER` |
| `0x0c` | `CROSS_CORRELATION` | | `0x4e` | `RINGBUFFER_READER` |
| `0x0d` | `CROSS_PRODUCT` | | `0x4f` | `CONDITION` |
| `0x0e` | `KERNEL_RASTERIZER` | | `0x50` | `PHI` |
| `0x0f` | `ARG_MIN_MAX` | | `0x51` | `BASICBLOCK_IN` |
| `0x10` | `GLOBAL_ARG_MIN_MAX` | | `0x52` | `BASICBLOCK_OUT` |
| `0x11` | `MATRIX_MULT` | | `0x53` | `BATCHNORM` |
| `0x12` | `BROADCAST` | | `0x54` | `WAIT_FOR_EVENT` |
| `0x13` | `FLATTEN_COMPOSITE` | | `0x55` | `SIGNAL_EVENT` |
| `0x14` | `UNFLATTEN_COMPOSITE` | | `0x56` | `ALL_SLICE` |
| `0x15` | `FPS_WITH_RADIUS_COMPOSITE` | | `0x57` | `ALL_GATHER` |
| `0x16` | `PIXEL_SHUFFLE_COMPOSITE` | | `0x58` | `SCALED_DOT_PRODUCT_ATTENTION` |
| `0x17` | `PIXEL_UNSHUFFLE_COMPOSITE` | | `0x59` | `ALL_REDUCE` |
| `0x18` | `CONV_COMPOSITE` | | `0x5a` | `REDUCE_SCATTER` |
| `0x19` | `MATDECOMP_MATMULT_COMPOSITE` | | `0x5b` | `ATOMIC_READ_MODIFY_WRITE` |
| `0x1a` | `CHANNEL_TO_SPACE_LARGE_FACTOR_COMPOSITE` | | `0x5c` | `PEFUSED_ELEMENTWISE` |
| `0x1b` | `LIVE_IN` | | `0x5d` | `PEFUSED_SECUREFLUSH` |
| `0x1c` | `LIVEIN_PARAM` | | `0x5e` | `PEFUSED_POOL` |
| `0x1d` | `CONST_IN` | | `0x5f` | `PEFUSED_GOC` |
| `0x1e` | `LIVE_STATE` | | `0x60` | `NEFUSED_CONV` |
| `0x1f` | `LIVE_OUT` | | `0x61` | `NEFUSED_KERNEL_RASTERIZER` |
| `0x20` | `REDUCTION` | | `0x62` | `NEFUSED_CROSS_CORRELATION` |
| `0x21` | `ALIAS` | | `0x63` | `NEFUSED_MATMUL` |
| `0x22` | `REINTERPRET_INNERMOST_DIMENSION` | | `0x64` | `NEFUSED_POOL` |
| `0x23` | `REINTERPRET_CAST` | | `0x65` | `NEFUSED_EW` |
| `0x24` | `RESHAPE` | | `0x66` | `NEFUSED_DUAL_SOURCE_EW` |
| `0x25` | `VIEW` | | `0x67` | `NEFUSED_UNARY_EW` |
| `0x26` | `TRANSPOSE` | | `0x68` | `NEFUSED_BYPASS` |
| `0x27` | `SPACE_TO_BATCH` | | `0x69` | `NEFUSED_RCAS` |
| `0x28` | `BATCH_TO_SPACE` | | `0x6a` | `TRANSPOSE_ENGINE_OP` |
| `0x29` | `SPACE_TO_CHANNEL` | | `0x6b` | `TE_RESAMPLE` |
| `0x2a` | `CHANNEL_TO_SPACE` | | `0x6c` | `TE_AFFINE_TRANSFORM` |
| `0x2b` | `SOFTMAX` | | `0x6d` | `TE_PAD` |
| `0x2c` | `INSTANCE_NORM` | | `0x6e` | `TE_CROP_RESIZE` |
| `0x2d` | `L2_NORM` | | `0x6f` | `TE_SLICE` |
| `0x2e` | `RMS_NORM` | | `0x70` | `TE_GATHER` |
| `0x2f` | `MINMAX_NORM` | | `0x71` | `TE_RESIZE` |
| `0x30` | `LAYER_NORM` | | `0x72` | `TM_WAIT_FOR_EVENT` |
| `0x31` | `LRN` | | `0x73` | `TM_SIGNAL_EVENT` |
| `0x32` | `COST_VOLUME` | | `0x74` | `TM_BRANCH` |
| `0x33` | `PIXEL_SHUFFLE` | | `0x75` | `TM_FETCH` |
| `0x34` | `PIXEL_UNSHUFFLE` | | `0x76` | `TM_STORE` |
| `0x35` | `MATRIX_DECOMPOSITION` | | `0x77` | `TM_OPERATE` |
| `0x36` | `FPS` | | `0x78` | `TM_USER_SLOT_LOAD` |
| `0x37` | `RS` | | `0x79` | `DMA_CONVERT` |
| `0x38` | `RESAMPLE` | | `0x7a` | `QUANT` |
| `0x39` | `GATHER` | | `0x7b` | `DEQUANT` |
| `0x3a` | `TILE` | | `0x7c` | `SNE_COND` |
| `0x3b` | `SLICE` | | `0x7d` | `SNE_GOC` |
| `0x3c` | `PAD` | | `0x7e` | `CCDMA_CONST` |
| `0x3d` | `RESIZE` | | `0x7f` | `CCDMA_MEMORY` |
| `0x3e` | `RESIZEAS` | | `0x80` | `SPILL_FILL_DUMMY` |
| `0x3f` | `CROP_RESIZE` | | `0x81` | `CLEARING_TASK` |
| `0x40` | `AFFINE_TRANFORM` | | `0x82` | `INVALID` |
| `0x41` | `PLANE_READER` | | | |

## Groupings

- `0x5c`-`0x5f` are `PEFUSED_` forms (elementwise, secure flush, pool, GOC); `0x60`-`0x69` are
  `NEFUSED_` forms (conv, kernel rasterizer, cross correlation, matmul, pool, EW, dual-source EW,
  unary EW, bypass, RCAS). The fused ops are where the NE and PE engine split shows up in the enum.
- `0x6a`-`0x71` are transpose-engine ops (`TRANSPOSE_ENGINE_OP`, `TE_` forms).
- `0x72`-`0x78` are `TM_` ops including `TM_BRANCH`; `0x7c` and `0x7d` are `SNE_COND` and `SNE_GOC`.
- `0x1b`-`0x1f` are graph boundaries: `LIVE_IN`, `LIVEIN_PARAM`, `CONST_IN`, `LIVE_STATE`,
  `LIVE_OUT`.
