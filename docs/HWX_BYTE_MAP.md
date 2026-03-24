# HWX Binary Format Specification (BEEFFACE / Zin ANE)

Complete byte-level analysis of the ANE Zin binary format, derived from comparing
three 49,152-byte HWX files:
- `relu_49152.hwx` — compiler-generated ReLU operation
- `abs_49152.hwx` — compiler-generated absolute value operation
- `relu_patched_to_abs.hwx` — hand-patched relu binary modified to perform abs

All three files: 49,152 bytes (0xC000), BEEFFACE magic, CPU type 0x80 (ANE), H17G subtype 9.

---

## 1. File Layout Overview

```
Offset      Size      Content
─────────────────────────────────────────────────────────────
0x0000      0x0020    Mach-O Header (32 bytes)
0x0020      0x0048    LC[0]  LC_SEGMENT_64 __PAGEZERO
0x0068      0x0098    LC[1]  LC_SEGMENT_64 __FVMLIB (const)
0x0100      0x0098    LC[2]  LC_SEGMENT_64 __FVMLIB (data)
0x0198      0x00E8    LC[3]  LC_SEGMENT_64 __TEXT (2 sections)
0x0280      0x0020    LC[4]  LC_CMD_0x40 (FVMLIB const ref)
0x02A0      0x0020    LC[5]  LC_CMD_0x40 (FVMLIB data ref)
0x02C0      0x08A0    LC[6]  LC_THREAD flavor=4 (TD config)
0x0B60      0x0D40    LC[7]  LC_THREAD flavor=3 (kernel section 0)
0x18A0      0x0D48    LC[8]  LC_THREAD flavor=3 (kernel section 1)
0x25E8      0x0930    LC[9]  LC_CMD_0x08 (compiler metadata string)
0x2F18      0x0018    LC[10] LC_CMD_0x02 (symbol table offsets)
0x2F30      0x10D0    Zero padding to 0x4000
0x4000      0x0104*   __TEXT.__text section (kernel program)
0x4104*     0x003C*   Zero padding to __const alignment
0x4140*     0x4000    __TEXT.__const section (weight/config data)
0x8140*     0x3EC0*   Zero padding to file end (0xC000)
─────────────────────────────────────────────────────────────
* __text size varies: relu=0x104 (260 bytes), abs=0xFC (252 bytes).
  __const position shifts accordingly: relu @ 0x4140, abs @ 0x4100.
```

---

## 2. Mach-O Header (0x0000 - 0x001F)

```
Offset  Size  Field            Value (both files)
──────────────────────────────────────────────────
0x0000  4     magic            0xBEEFFACE
0x0004  4     cputype          0x00000080 (ANE)
0x0008  4     cpusubtype       0x00000009 (H17G)
0x000C  4     filetype         0x00000002 (MH_EXECUTE)
0x0010  4     ncmds            11 (0x0B)
0x0014  4     sizeofcmds       12048 (0x2F10)
0x0018  4     flags            0x00200000
0x001C  4     reserved         0x00000000
```

All little-endian. Identical between relu and abs.

---

## 3. Load Commands

### LC[0] LC_SEGMENT_64 __PAGEZERO (0x0020)

```
cmd=0x19, cmdsize=72 (0x48)
segname:   __PAGEZERO
vmaddr:    0x0000000000000000
vmsize:    0x0000000000004000  (16 KB)
fileoff:   0x0000000000000000
filesize:  0x0000000000000000
maxprot=0, initprot=0, nsects=0, flags=0x4
```

Guard page, no file backing. Identical in all files.

### LC[1] LC_SEGMENT_64 __FVMLIB const (0x0068)

```
cmd=0x19, cmdsize=152 (0x98) — includes 1 section header
segname:   __FVMLIB
vmaddr:    0x0000000030000000
vmsize:    0x0000000000004000
fileoff:   0x0000000000000000
filesize:  0x0000000000000000
maxprot=1(R), initprot=1(R), nsects=1, flags=0x6

  Section: __const in __FVMLIB
    addr=0x30000000, size=0x1000, fileoff=0, align=14
    flags=0x21, reserved1=0, reserved2=0
```

Read-only FVMLIB segment for constants. No file data (BSS-like, DRAM-allocated).

### LC[2] LC_SEGMENT_64 __FVMLIB data (0x0100)

```
cmd=0x19, cmdsize=152 (0x98) — includes 1 section header
segname:   __FVMLIB
vmaddr:    0x0000000030004000
vmsize:    0x0000000000004000
fileoff:   0x0000000000000000
filesize:  0x0000000000000000
maxprot=2(W), initprot=2(W), nsects=1, flags=0x6

  Section: __data in __FVMLIB
    addr=0x30004000, size=0x1000, fileoff=0, align=14
    flags=0x23, reserved1=0, reserved2=0
```

Write-only FVMLIB segment for mutable data (output buffer). No file data.

### LC[3] LC_SEGMENT_64 __TEXT (0x0198)

```
cmd=0x19, cmdsize=232 (0xE8) — includes 2 section headers
segname:   __TEXT
vmaddr:    0x0000000030008000
vmsize:    0x0000000000008000  (32 KB)
fileoff:   0x0000000000004000
filesize:  0x0000000000008000
maxprot=5(RX), initprot=5(RX), nsects=2, flags=0x4
```

Contains the executable kernel program and configuration constants.

**Section: __text in __TEXT**
```
  addr:    0x30008000
  size:    relu=0x0104 (260), abs=0x00FC (252)  *** DIFFERS ***
  fileoff: 0x4000
  align:   14, flags=0x28
```

**Section: __const in __TEXT**
```
  addr:    relu=0x30008140, abs=0x30008100  *** DIFFERS ***
  size:    0x4000
  fileoff: relu=0x4140, abs=0x4100          *** DIFFERS ***
  align:   6, flags=0x26
```

### LC[4] LC_CMD_0x40 (0x0280) — FVMLIB Const Reference

```
cmd=0x40, cmdsize=32
Payload (24 bytes):
  +0x00: 0x00000018  (size of this descriptor)
  +0x04: 0x00000000
  +0x08: 0x30000000  (vmaddr of __FVMLIB const)
  +0x0C: 0x00000000
  +0x10: 0x00000078  (descriptor ID: 0x78 = 120)
  +0x14: 0x00000000
```

### LC[5] LC_CMD_0x40 (0x02A0) — FVMLIB Data Reference

```
cmd=0x40, cmdsize=32
Payload (24 bytes):
  +0x00: 0x00000018
  +0x04: 0x00000000
  +0x08: 0x30004000  (vmaddr of __FVMLIB data)
  +0x0C: 0x00000000
  +0x10: 0x00000079  (descriptor ID: 0x79 = 121)
  +0x14: 0x00000000
```

### LC[6] LC_THREAD flavor=4 (0x02C0) — Task Descriptor Config

```
cmd=0x04, cmdsize=2208 (0x8A0)
flavor=4, count=546 (0x222) -> 2184 bytes of state data
```

This is the master task descriptor. Internal structure:

```
Payload offset  File offset  Content
──────────────────────────────────────────────────
+0x0000         0x02C8       flavor=4, count=0x222
+0x0008         0x02D0       __text vmaddr: 0x30008000
+0x0018         0x02E0       __const vmaddr: relu=0x30008140, abs=0x30008100 *** DIFFERS ***
+0x0068         0x0330       __FVMLIB const vmaddr: 0x30000000
+0x0078         0x0340       __FVMLIB data vmaddr: 0x30004000
+0x0080-0x07FF  0x0348-0xAC7 Zero padding (sparse TD state)
+0x0808         0x0AD0       __text vmaddr: 0x30008000 (repeated)
+0x0818         0x0AE0       Config: 0x04, 0x00, 0x00, 0x00
+0x081C         0x0AE4       Instruction count: relu=0x41(65), abs=0x3F(63) *** DIFFERS ***
+0x0828         0x0AF0       0x01
+0x0830         0x0AF8       -1 mask (0xFFFFFFFFFFFFFFFF)
+0x083C         0x0B04       0xFFFF
+0x0840         0x0B08       0x0898 (offset to string table within payload)
+0x0858         0x0B20       0x10
+0x0860         0x0B28       0x04
+0x0868         0x0B30       0x02
+0x0870         0x0B38       Tag 7, size 8, value=0x461C4000 (signature hash?)
+0x0880         0x0B48       Tag 9, size 8, value=0 (padding)
+0x0890         0x0B58       "net\0" (network name string)
```

### LC[7] LC_THREAD flavor=3 (0x0B60) — Kernel Section 0

```
cmd=0x04, cmdsize=3392 (0x0D40)
flavor=3, count=842 (0x34A) -> 3368 bytes of state data
```

Internal structure:
```
+0x0000  flavor=3, count=0x34A
+0x0008  inner_flavor=3, kernel_index=1
+0x0010  TD offset: 0x0D38
+0x0018  TD offset: 0x0D3C, flags: 0x05
+0x0020  batch=1, stride=0x40, dims: 1,1,1
+0x0048  buffer_size: 0x1000
+0x0050  stride: 0x40, 0x40
+0x0060  num_buffers=2, buffer_size=0x1000
+0x0070  1, offset=0x0D3E, size=0x1000
+0x0080-0x0D2F  Zero (sparse kernel config)
+0x0D30  "net\0" + "x\0x\0" (tensor names: input "x", output "x")
```

Identical between relu and abs. This defines I/O buffer descriptors.

### LC[8] LC_THREAD flavor=3 (0x18A0) — Kernel Section 1

```
cmd=0x04, cmdsize=3400 (0x0D48)
flavor=3, count=842 (0x34A) -> 3368 bytes of state data
```

Same structure as LC[7] but kernel_index=2, offset=0x0D45.
Contains "net\0" + "y@output\0y\0" (tensor name "y" with output annotation).

Identical between relu and abs. Defines the second I/O descriptor.

### LC[9] LC_CMD_0x08 (0x25E8) — Compiler Metadata

```
cmd=0x08, cmdsize=2352 (0x0930)
```

Contains a null-terminated ASCII string with compiler metadata:
- ANEC version (v1)
- Zin compiler version (v9.202.0)
- Module info (EspressoFramework v3515.2.4.14.1)
- All compiler flags (-t h17g, allocator settings, etc.)
- Input/output file paths (contain hash values unique to each compilation)

This section differs between relu and abs **only in file path hashes**.
The hand-patched file retains relu's metadata (not updated).

### LC[10] LC_CMD_0x02 (0x2F18) — Symbol Table Offsets

```
cmd=0x02, cmdsize=24 (0x18)
+0x08: 0x00002F30  (string table offset)
+0x0C: 0x0000001B  (string table size: 27 bytes)
+0x10: 0x000030E0  (secondary offset)
+0x14: 0x0000028B  (secondary size: 651 bytes)
```

Identical between relu and abs.

---

## 4. __TEXT.__text Section — Kernel Program

This is the core ANE program: a sequence of 4-byte instruction words.

- relu: 65 words (260 bytes, 0x104)
- abs:  63 words (252 bytes, 0x0FC) — 2 words shorter

### 4.1 Instruction Word Layout

Each 4-byte little-endian word encodes a pipeline instruction. The program is
structured as a header block followed by pipeline stage configurations.

### 4.2 Word-by-Word Comparison

```
Word  Offset  relu          abs           Notes
──────────────────────────────────────────────────────────────────
 0    0x000   0x00000001    0x00000001    Program version/magic
 1    0x004   0x00000000    0x00000000    (reserved)
 2    0x008   0x00000000    0x00000000    (reserved)
 3    0x00C   0x00000000    0x00000000    (reserved)
 4    0x010   0x003D0000    0x003B0000    *** Bitfield size/count (61 vs 59)
 5    0x014   0x00000000    0x00000000
 6    0x018   0x04000068    0x04000068    Buffer config (size=0x400, offset=0x68)
 7    0x01C   0x00000000    0x00000000
 8    0x020   0x00FFF868    0x00FFF868    Negative offset (complement addressing)
 9    0x024   0x00000000    0x00000000
10    0x028   0x00000000    0x00000000
11    0x02C   0x00000000    0x00000000
12    0x030   0x00050009    0x00000009    *** relu flag=0x0005, abs flag=0x0000
13    0x034   0x00001540    0x00001540    Pipeline config
14    0x038   0x00000080    0x00000080    Tile size (128)
15    0x03C   0x00010001    0x00010001    Dimensions (1x1)
16    0x040   0x00000001    0x00000001    Batch count
17    0x044   0x00000001    0x00000001
18    0x048   0x00000040    0x00000040    Stride (64)
19    0x04C   0x93618005    0x95418005    *** OPCODE: relu=0x9361, abs=0x9541
20    0x050   0x00000001    0x00000001
21    0x054   0x00000001    0x00000001
22    0x058   0x00000040    0x00000040    Stride (64)
23    0x05C   0x00014000    0x00000001    *** relu: extra config 0x14000
24    0x060   0x00000001    0x00000020    *** shift begins here
25    0x064   0x00200000    0x00000004
26    0x068   0x00000000    0x00000000    (matches)
27    0x06C   0x00200000    0x80081342    relu has extra word here
28    0x070   0x80081342    0x000000CE    (shifted -4 bytes in abs)
...   ...     (all subsequent words shifted by 8 bytes = 2 words)
63    0x0FC   0x22001440    (end of abs)
64    0x100   0x01000021    (end of relu, abs has no word here)
```

### 4.3 Key Semantic Differences

1. **Word[4] (0x010)**: Program size indicator. relu=0x3D (61), abs=0x3B (59).
   Encodes the number of pipeline operations or config entries.

2. **Word[12] (0x030)**: Stage enable flags. relu=0x00050009, abs=0x00000009.
   The 0x0005 in the high half-word enables the relu-specific pipeline stage.

3. **Word[19] (0x04C)**: **Operation opcode**. This is the critical NE operation selector:
   - relu: 0x93618005 (opcode 0x9361)
   - abs:  0x95418005 (opcode 0x9541)
   The low 16 bits (0x8005) are common flags. The high 16 bits select the operation.

4. **Word[23] (0x05C)**: relu has 0x00014000 (extra NE configuration),
   abs has 0x00000001. relu enables an additional pipeline processing step.

5. **Words 24-64**: relu program is 2 words longer. After word 26, the abs
   program content is identical but shifted back by 8 bytes. This means relu
   inserts 2 extra configuration words (at positions 27 and 40-area) for its
   additional pipeline stage.

### 4.4 Hand-Patch vs Compiler Observations

The hand-patched file (`relu_patched_to_abs.hwx`) matches the abs instruction
stream exactly (including the 2-word removal), but retains relu's compiler
metadata. This confirms the __TEXT.__text section is the only functionally
relevant code — the metadata is informational only.

---

## 5. Complete Byte-Level Diff Tables

### 5.1 relu vs abs (240 differing bytes)

#### LC[3] __TEXT Segment Header (4 bytes)

| Offset | relu | abs  | Field |
|--------|------|------|-------|
| 0x0208 | 0x04 | 0xFC | __text section size low byte (0x104 vs 0xFC) |
| 0x0209 | 0x01 | 0x00 | __text section size high byte |
| 0x0250 | 0x40 | 0x00 | __const vmaddr low byte (0x8140 vs 0x8100) |
| 0x0260 | 0x40 | 0x00 | __const fileoff low byte (0x4140 vs 0x4100) |

#### LC[6] LC_THREAD[0] TD Config (2 bytes)

| Offset | relu | abs  | Field |
|--------|------|------|-------|
| 0x02E0 | 0x40 | 0x00 | __const vmaddr ref low byte (0x8140 vs 0x8100) |
| 0x0AE4 | 0x41 | 0x3F | Instruction word count (65 vs 63) |

#### LC[9] Metadata String (123 bytes)

All 123 byte differences are in compiler input/output path hashes
(hex hash strings differ between independent compilations). Not functionally relevant.

#### __TEXT.__text Kernel Program (111 bytes)

111 bytes differ across the 260/252 byte instruction stream.
See Section 4.2 above for word-by-word breakdown.

### 5.2 relu vs hand-patched (117 differing bytes)

**Zero diffs in metadata** (hand-patch preserved relu metadata).

| Region | Bytes Changed | Description |
|--------|---------------|-------------|
| LC[3] header | 4 | __text size + __const addr/offset |
| LC[6] TD config | 2 | __const vmaddr ref + instruction count |
| __TEXT.__text | 111 | Kernel program (identical to abs content) |

### 5.3 abs vs hand-patched (123 differing bytes)

**All 123 diffs are in LC[9] metadata** (different path hashes).
Zero diffs in headers, TD config, and kernel program.

This proves the hand-patch is functionally identical to the compiler-generated abs.

---

## 6. Segment Purpose Summary

| Segment | Purpose | File Data |
|---------|---------|-----------|
| __PAGEZERO | Guard page (0x0000-0x3FFF virtual) | None |
| __FVMLIB (const) | Read-only DRAM input buffer (0x30000000) | None (BSS) |
| __FVMLIB (data) | Write-only DRAM output buffer (0x30004000) | None (BSS) |
| __TEXT | Executable kernel code + constants (0x30008000) | 0x4000-0xBFFF |

The FVMLIB segments define the I/O buffer layout in ANE virtual memory (VM base 0x30000000).
The __TEXT segment contains the actual kernel program (__text) and static configuration (__const).

---

## 7. LC_THREAD Roles

| LC_THREAD | Flavor | Role | Key Content |
|-----------|--------|------|-------------|
| [0] flavor=4 | TD Master | Task descriptor master config | VM address map, instruction count, network name "net" |
| [1] flavor=3, idx=1 | Kernel I/O 0 | Input buffer descriptor | Buffer sizes, strides, tensor name "x" |
| [2] flavor=3, idx=2 | Kernel I/O 1 | Output buffer descriptor | Buffer sizes, strides, tensor name "y@output" |

LC_THREAD[1] and [2] are **identical** between relu and abs — only the operation changes,
not the I/O layout.

---

## 8. Critical Fields for Zin Generation

To generate a new HWX binary from scratch, the minimum fields to modify:

1. **Word[19] in __text (file offset 0x404C)**: Operation opcode (0x9361=relu, 0x9541=abs)
2. **Word[12] in __text (file offset 0x4030)**: Stage enable flags (0x00050009 vs 0x00000009)
3. **Word[4] in __text (file offset 0x4010)**: Program size (0x3D vs 0x3B)
4. **Word[23] in __text (file offset 0x405C)**: Extra pipeline config
5. **LC[3] section headers**: Adjust sizes/offsets if instruction count changes
6. **LC[6] TD at 0x0AE4**: Update instruction word count
7. **LC[6] TD at 0x02E0**: Update __const vmaddr if __text size changes

For same-size operations (where instruction count doesn't change), only fields 1 and 2
need modification. For operations that change pipeline stage count, all 7 fields must
be updated consistently.
