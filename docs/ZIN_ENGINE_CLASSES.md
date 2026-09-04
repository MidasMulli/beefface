# Zin Engine Classes and the Layer Opcode Table

Compiler: `ZinAneCompiler-9.509.0` (version string read from `(__TEXT,__const)`,
`_compilerVersionString`). Method: symbol and disassembly analysis of
`ANECompiler.framework` extracted from the dyld shared cache, arm64e.

This documents the compiler's own class taxonomy and the function that maps a layer opcode to a
name. It is static analysis of the compiler, not of the sealed ANE firmware.

---

## 1. Three engine-prefixed layer families

`ANECompiler` names its layer classes with an engine prefix. All three families are present in the
same binary:

| prefix | classes |
|---|---|
| `ZinNE*` | `ZinNEConvLayer`, `ZinNEMatMulLayer`, `ZinNEPoolLayer`, `ZinNEElementWiseLayer`, `ZinNEDualSourceElementWiseLayer`, `ZinNEBypassLayer`, `ZinNECrossCorrelationLayer`, `ZinNEKernelRasterizerLayer`, `ZinNERCASLayer`, `ZinNELayer` |
| `ZinPE*` | `ZinPEElementWiseLayer`, `ZinPEPoolLayer`, `ZinPEGOCLayer`, `ZinPESecureFlushLayer`, `ZinPELayer` |
| `ZinSNE*` | `ZinSNEConditionLayer`, `ZinSNEConditionOperation`, `ZinSNEGOCLayer`, `ZinSNEAtoms` |

The split of work implied by the class names is that convolution and matmul are `NE`, while
elementwise and pooling appear under both `NE` and `PE`.

Credit and lineage: maderix ("Inside the M4 ANE", part 4b) describes the same three-way split as
NE / PE (Planar Engine) / SNE, with PE handling elementwise and pooling work. The class families
above are the compiler-side symbols corresponding to that description. We did not originate the
naming; this is the symbol evidence for it.

Note on the word "planar": it also appears in this binary as a tensor memory-layout term, in
symbols such as `CreateRuntimeMapPlanarTiledCompressed`, `CreateRuntimeMapPlanarSinglePlaneLinear`
and `alias_planar_src`. Those are unrelated to the `ZinPE*` engine classes. Searching for the
string "planar" alone will find the layout symbols and miss the engine classes entirely.

### GOC appears in all three families

`GOC` layers exist per engine: `ZinGOCLayer`, `ZinPEGOCLayer`, `ZinSNEGOCLayer`, plus
`ZinTernaryDynamicGOCLayer`. There are matching unit types (`ZinIrGOCUnit`, `ZinIrPEGOCUnit`,
`ZinIrDynamicGOCUnit`, `ZinMirPEGOCUnit`) and constructors (`ZinCreateGOCUnit`,
`ZinCreatePEGOCUnit`, `ZinCreateDynamicGOCUnit`). A pass named `ZinMirGOCEngineReassignment`
moves GOC work between engines.

`ogoc` and `postogo` are also stage names in the 17-stage pipeline documented in
`ANE_CRACK_REPORT.md`.

### Task-descriptor branching

`TdBranching` is a distinct family: `TdBranchingPredicateOp`, `TdBranchingOperand`,
`TdBranchingPredicateImmediateOperand`, `TdBranchingInfo`. The condition object it operates on is
`ZinSNEConditionOperation`, carried by `ZinSNEConditionLayer`. Both are `SNE` classes, so a search
restricted to `NE` or `PE` layer constructors will not surface them.

---

## 2. Layer opcode to name: `OpCodeToString`

The enum `ZinIrOpLayerOpCodeType` is converted to a name by:

```
__ZN21ZinIrEnumToStringUtil14OpCodeToStringE22ZinIrOpLayerOpCodeType
```

Disassembly (addresses are from one extracted image and are not stable across builds; the
structure is):

```
adrp x9, <page>            ; page base of the table
add  x9, x9, #0x848        ; table base
ldr  x1, [x9, w0, uxtw #3] ; table[opcode], 8-byte entries
mov  x0, x8
b    std::string::string(const char*)
```

Three properties worth noting:

- It is a **flat pointer table indexed directly by the opcode value**, 8 bytes per entry, each
  entry a `const char*`. There is no switch and no computed dispatch.
- There is **no bounds check**. The sibling function `NonLinearModeToString` in the same utility
  class does bounds-check (`cmp w9, #0x2f`, so 48 non-linear modes, table at `+0xc38`) and falls
  back to a default string. `OpCodeToString` does not.
- The two tables are `0x848` and `0xc38` on the same page, a gap of `0x3F0` = **126 entries** of
  8 bytes. If the tables are adjacent, that bounds the opcode enum at roughly 126 values.

Reading `table[n]` and dereferencing gives the name for opcode `n`. Externally reported anchor
values that can be used to validate such a read are `0x5d` for the conv layer and `0x64` for the
bypass layer (maderix, part 4b).

We have not published the table contents. Extracting them requires reading the data page from a
shared cache image rather than from a `__text` disassembly.

---

## 3. `mac_cfg` field layout

`mac_cfg` is one 32-bit word at task-descriptor object offset `[0x4b8]`, hardware bank `0x4900`.
`HWX_BYTE_MAP.md` documents the enclosing block as `LC[6] LC_THREAD flavor=4 (TD config)`.

Five setters write this word. Fields were located by setter symbol, not by pattern matching:

| field | bits | setter | notes |
|---|---|---|---|
| `op_mode` | `[2:0]` | `ZinAneTd<19>::SetOpMode` | remaps 0->0, 2->3, 3->1, 4->2, 5->4, 6->5; values 1 and 7 assert |
| `kernel_mode` | `[3]` | `ZinAneTd<19>::SetKernelMode` | 0 or 1; value 2 asserts |
| `passthrough` | `[5]` | `ZinAneTd<19>::SetPassthroughEnable` | mask 0x20 |
| `binary_point` | `[13:8]` | `ZinAneTd<19>::SetNEBinaryPoint` | 6-bit field, `bfi w1,#8,#6` |
| `non_linear_mode` | `[17:16]` | `ZinAneTd<19>::SetNENonLinearMode` | relu sets bit 16, sigmoid sets bit 17 |

`<19>` is the `ZinAneTd` template instantiation for H17G. Instantiations present in the binary are
N in {1,4,5,6,7,8,10,11,17,19,20}.

Two observations about the field set:

- `non_linear_mode[17:16]` matches the empirically-found relu/sigmoid toggle at MACCfg bits 16 and
  17 reported by allbilly for H13, arrived at by a different method.
- There is **no accumulator-width setter** among the writable fields. `binary_point` is a
  radix-point position for the fixed-point path, which is scaling rather than width.

Verification status: candidate. This is the compiler's model of the descriptor, meaning how it
emits and would parse it, not a firmware dump.

---

## What this does not establish

Static compiler analysis says what the compiler emits and how it names things. It does not
establish hardware behaviour. Where a hardware claim is wanted, see `ACCUMULATOR_PRECISION.md` for
a measured example, and `ANE_CRACK_REPORT.md` for the patch-and-toggle method.

---

## 4. Machine patterns: `ZinNEPatterns`

Compiler `ZinAneCompiler-10.26.6` (macOS 27.0, 26A5421a). Extracted from the shared cache; note
this is a newer build than the 9.509.0 used for sections 1 to 3 above.

`ZinNEPatterns` holds the post-lowering fusion patterns. Present: `Conv`, `NEConv`, `UConv`,
`Pool`, `UPool`, `Bypass`, `MatMul`, `ElementWise`, `UnaryElementWise`, `DualSourceElementWise`,
`CrossCorrelation`, `KernelRasterizer`, `RCAS`, `MxQuant`, `MxDeQuant`, `UMaxWithConst`,
`TdBranching`.

Two entry points carry the logic:

| function | address |
|---|---|
| `ZinNEPatterns::Conv::AnalyzeConv(ZinIrOpLayerGraph*, ZinIrParameters&, ZinPattern*)` | `0x20bd51e78` |
| `ZinNEPatterns::Conv::Fuse(...)` | `0x20bd52688` |
| `ZinNEPatterns::Conv::ToActivation(ZinElementWiseLayer*)` | `0x20bd52f10` |
| `ZinNEPatterns::NEConv::Fuse(...)` | `0x20bd635c4` |

### What the NEConv pattern references

Resolving the adrp/add string references inside `NEConv::Fuse` gives its vocabulary directly:

```
_encap
dma_conv_input
dma_conv_output
ane_layer
condition_layer
Mult has more than 2 inputs
Unsupported operand type in td branching fusion.
```

Two things follow.

`dma_conv_input` and `dma_conv_output` are the **first and last stages of the 17-stage pipeline**
documented in `ANE_CRACK_REPORT.md`. The pattern's DMA-symbol condition is a check on those symbol
names, so the pattern legality is tied to the pipeline's own input and output stages rather than to
an abstract layout predicate.

`ane_layer`, `condition_layer` and the `td branching fusion` error string all appear in this same
function, so **task-descriptor branching fusion is performed inside `NEConv::Fuse`**, not in a
separate pass. That connects the `TdBranching` pattern to the conv path, and the condition object
side of it is `ZinSNEConditionOperation` / `ZinSNEConditionLayer` (section 1), with `SNE_COND`
(`0x7c`) and `TM_BRANCH` (`0x74`) in the opcode enum (`LAYER_OPCODES.md`).

---

## 5. The kernel-size splitter cost function

`ZinMirKernelSizeSplitterEngine` scores candidate splits. The scoring functions:

| function | address |
|---|---|
| `Run()` | `0x20bab7490` |
| `GetUtilizationFor(ZinIrLayerSplitInfo::Part const&, ...)` | `0x20bab77e8` |
| `GetUtilizationFor(ZinIrLayerSplitInfo const&, ...)` | `0x20bab78ac` |
| `GetTotalPassCount(ZinIrLayerSplitInfo const&, ...)` | `0x20bab7940` |
| `GetOCGFactor(ZinNEConvLayer const*)` | `0x20bab79d8` |
| `PartitionLinearly(...)` | `0x20bab7ca4` |
| `DeterminePartition(ZinNEConvLayer const*, ..., Analysis const&, ZinIrLayerSplitInfo&)` | `0x20bab83e8` |
| `DetermineSplittingUsingCompression(...)` | `0x20bab9820` |
| `DetermineSplittingForPerChannelPaletteLUT(...)` | `0x20bab992c` |

`GetUtilizationFor` is a work-weighted mean utilization, and contains no constants:

```
ucvtf s0, x22          ; part work
ucvtf s1, x0
fdiv  s1, s0, s1       ; part / capacity
ldr   x8, [x20]        ; total
ucvtf s2, x8
fdiv  s0, s0, s2       ; part / total
fmadd s8, s0, s1, s8   ; acc += (part/total) * (part/capacity)
```

`GetTotalPassCount` walks the parts array at stride `0x480`, gated on a byte flag at part offset
`+0x470` (the same flag `GetUtilizationFor` reads), accumulating a per-part pass count.

So the two cost terms are **utilization and total pass count**, both computed from shapes. Neither
function carries a threshold immediate; the selection among legal splits happens in
`DeterminePartition`, with `PartitionLinearly` generating candidates under an `OCGSizeRange` and a
`ZinIrLayerSplitInfo::Part::Constraints`.

### Hardware parameters

Per-target parameters come from a `ZinIrHal*::GetParams()` family returning `ZinIrHalParameters`,
consumed by `L2Request(WorkUnitShape const&, ...)`. 28 targets are present:

```
H11 H12 H13 H13g H14 H14c H14g H15 H15c H15g H16 H16c H16g H16s
H17 H17a H17c H17g H17s H18  M9 M11  T0 T1  U1 U2 U3 U4
```

maderix reports pulling a 2360-byte hardware abstraction table from `ZinIrHalH16g::GetParams()` on
M4. `H17g` is the M5 Pro analogue in the same family, and `H18` is present in this build.
