# Reduction Accumulator Precision, Measured

Hardware: M5 Pro (H17). Date: 2026-04-14. Toolchain: `coremltools 9.0` MIL builder plus
`xcrun coremlcompiler`.

A measurement of the accumulator behind the ANE's reduction path, using a single-op `reduce_sum`.
Two axes are probed separately, because they exclude different formats.

---

## Setup

Single-op `reduce_sum` over shape `[1, 4096, 1, 1]`, FP16 input, FP16 `compute_precision`.

Execution target was established by a compute-unit sweep rather than by an energy counter:

| compute unit | time |
|---|---|
| `CPU_ONLY` | 16.99 ms |
| `CPU_AND_NE` | 0.18 ms |
| `ALL` | 0.20 ms |

GPU is excluded in `CPU_AND_NE`, and the roughly 100x gap against `CPU_ONLY` rules out a silent
CPU fallback. This is timing discrimination, not a placement counter, and is stated as such.

---

## Range

Input: 4096 values of `np.float16(65000.0)`. That rounds to 64992, since the FP16 mantissa step at
exponent 15 is 32.

```
true sum      = 4096 * 64992 = 266,207,232
FP32 reference =              266,207,232.0
ANE output     =              266,207,232.0     (bit-exact, absolute difference 0.0)
```

FP16 maximum is 65,504, so the accumulator carries a range far wider than FP16.

## Mantissa

Range alone does not exclude a format with a wide exponent and a narrow mantissa. A second input
probes the mantissa:

```
input          = 4095 * 1.0 + 1 * 0.001
FP32 reference = 4095.0009765625
ANE output     = 4095.0009765625              (bit-exact)
```

Two supplementary cases at the same shape are non-discriminating and are recorded for
completeness: `4095 * 1.0 + 1 * 0.0001` gives 4095.0 (the addend falls below the FP32 mantissa
step at that magnitude, so the reference is itself 4095.0), and `4095 * 100.0 + 1 * 0.01` gives
409500.0 for the same reason at larger scale.

### Why the mantissa case is the informative one

`4095.0009765625` is `4095 + 2^-10`. In binary it spans from `2^11` down to `2^-10`, so
representing it requires **22 significand bits**:

| format | significand bits | can represent 4095.0009765625 |
|---|---|---|
| bf16 | 8 | no |
| FP16 | 11 | no |
| 19-bit float, 1 sign + 8 exponent + 10 mantissa | 11 | no |
| FP32 | 24 | yes |

This is a property of the value that came back, not of the order in which the sum was accumulated.
A tree reduction, a sequential reduction and any other ordering all have to hold the final value,
so no summation order lets a narrower format produce it.

---

## Scope

This measures the **reduction** path. It is not a measurement of the conv or matmul accumulator,
which is a different unit.

The conv accumulator is not measurable this way. Per the 17-stage pipeline in
`ANE_CRACK_REPORT.md`, a conv output narrows at `oquant` into `dma_conv_output` at the end of the
pass. A wide accumulator and a narrow one therefore produce results differing only below what the
output can represent. Measured directly: an input of `4095 * 1.0 + 0.001` (true 4095.0009765625)
and an input of `4096 * 1.0` (true 4096.0) both return **4096.0** from a 1x1 conv, and requesting
an FP32 output does not change it, because the cast is applied after the narrowing store.

Order-based workarounds do not recover it either. Summing 4096 ones and reaching 4096 exactly
proves a tree reduction rather than a wide accumulator, since a pairwise tree in FP16 sums 4096
ones exactly. In our own attempts the saturation point moved with the output channel count, an
parameter that cannot affect any accumulator's precision, which is what identifies those results
as algorithm selection rather than a width measurement.

Single machine, single session, one op form.
