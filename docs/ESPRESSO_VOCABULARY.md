# espresso.net Layer Type Vocabulary

Complete map of the Apple Neural Engine instruction set as exposed through the `model.espresso.net` JSON format inside compiled `.mlmodelc` bundles.

**Platform:** macOS 15 / Apple M5 / CoreML 3500.32.1 / coremltools 9.0
**Date:** 2026-03-23
**Method:** Systematic model generation via coremltools MIL -> NeuralNetwork -> `xcrun coremlcompiler compile`, plus direct JSON patching and probing via native CoreML ObjC API.

---

## Format Overview

Every `model.espresso.net` file has this structure:

```json
{
  "storage": "model.espresso.weights",
  "format_version": 200,
  "layers": [ ... ]
}
```

Each layer has at minimum:
- `type` (string) -- the layer type from the vocabulary below
- `name` (string) -- unique layer name
- `bottom` (string) -- comma-separated input blob names
- `top` (string) -- comma-separated output blob names
- `weights` (object) -- usually empty `{}`
- `attributes` (object) -- `{"is_output": 1}` on final layers

The companion `model.espresso.shape` file defines tensor shapes as `{n, k, h, w}` (batch, channels, height, width).

---

## Layer Types (31 confirmed + 6 hidden)

### 1. `activation`

Pointwise activation functions. The `mode` field selects which function.

| Mode | Function | Parameters | Formula |
|------|----------|-----------|---------|
| 0 | ReLU | -- | max(0, x) |
| 1 | Tanh | -- | tanh(x) |
| 2 | Leaky ReLU / PReLU | alpha, weights | x if x>0 else alpha*x |
| 3 | Sigmoid | -- | 1/(1+exp(-x)) |
| 5 | Scaled Tanh | alpha, beta | alpha * tanh(beta * x) |
| 6 | Linear | alpha, beta | alpha * x + beta |
| 7 | Hard Sigmoid | alpha, beta | clamp(alpha*x + beta, 0, 1) |
| 8 | ELU | alpha | x if x>0 else alpha*(exp(x)-1) |
| 9 | Thresholded ReLU | alpha | x if x>alpha else 0 |
| 10 | Softplus | -- | log(1+exp(x)) |
| 12 | Softsign | -- | x/(1+abs(x)) |
| 13 | Clamp Min / Max(x, alpha) | alpha | max(x, alpha) |
| 14 | Constant | alpha | alpha (ignores input) |
| 16 | SELU | -- | lambda*(x if x>0 else alpha*(exp(x)-1)), lambda=1.0507, alpha=1.6733 |
| 19 | GELU (exact) | -- | x/2 * (1 + erf(x/sqrt(2))) |
| 20 | Erf | -- | erf(x) |
| 21 | GELU (tanh approx) | -- | 0.5x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))) |
| 22 | GELU (sigmoid approx) | -- | x * sigmoid(1.702*x) |
| 23 | Heaviside / Step | -- | x >= 0 ? 1 : 0 |
| 24 | Clamp Max / Min(x, alpha) | alpha | min(x, alpha) |
| 25 | SiLU / Swish | -- | x * sigmoid(x) |
| 26 | HardSwish | -- | x * clamp(x/6 + 0.5, 0, 1) |

**Fields:** `mode`, `alpha`, `beta`, `weights` (for PReLU per-channel alpha)

**JSON example (ReLU):**
```json
{
  "type": "activation",
  "mode": 0,
  "bottom": "x",
  "top": "y",
  "name": "relu_0",
  "weights": {},
  "attributes": {"is_output": 1}
}
```

---

### 2. `elementwise`

The workhorse operation type. Handles binary ops, unary ops, comparisons, and transcendentals through the `operation` field.

**Fields:** `operation` (int), `alpha`, `beta`, `fused_relu`, `nd_mode` (bool, enables broadcastable N-D mode), `eps`

#### Binary Operations (two inputs via comma-separated `bottom`)

| Op | Function | Notes |
|----|----------|-------|
| 0 | Add | x + y |
| 1 | Multiply | x * y |
| 13 | Power | x^y (nd_mode=true) |
| 22 | Minimum | min(x, y) |
| 23 | Maximum | max(x, y) |
| 33 | Subtract | x - y (nd_mode=true) |
| 36 | Modulo | x % y (nd_mode=true) |
| 100 | Equal | x == y |
| 101 | Select/Where | cond ? a : b (3 inputs) |
| 102 | Divide | x / y (nd_mode=true) |
| 103 | Less Than | x < y |
| 104 | Less Equal | x <= y |
| 105 | Greater Than | x > y |
| 106 | Greater Equal | x >= y |
| 107 | Not Equal | x != y |
| 1021 | Floor Divide | floor(x / y) (nd_mode=true) |

#### Unary Operations (single input)

| Op | Function | Notes |
|----|----------|-------|
| 2 | Add Constant | x + alpha |
| 3 | Copy / Identity | x (passthrough) |
| 4 | Add Constant | x + alpha (variant) |
| 5 | Scale | alpha * x |
| 6 | Scale | alpha * x (variant) |
| 7 | Identity | x |
| 8 | Square | x^2 |
| 9 | Square | x^2 (variant) |
| 10 | Inverse | 1/x |
| 11 | Sqrt | sqrt(x) |
| 12 | Rsqrt | 1/sqrt(x) |
| 24 | Abs | abs(x) |
| 25 | Clamp Min | max(x, alpha) |
| 26 | Log | ln(x) |
| 27 | Exp | exp(x) |
| 28 | Reduce Sum | sum(x) (scalar output) |
| 29 | Add Constant | x + alpha (variant) |
| 34 | Scale | alpha * x (variant) |
| 35 | Exp2 | 2^x |
| 37 | Round | round(x) |
| 38 | Sign | sign(x) |

#### Trigonometric / Transcendental

| Op | Function |
|----|----------|
| 114 | Cos |
| 115 | Sin |
| 117 | Floor |
| 118 | Ceil |
| 119 | Clip (uses alpha, beta) |
| 121 | Tan |
| 122 | Cosh |
| 123 | Sinh |
| 124 | Tanh |
| 125 | Acos |
| 126 | Asin |
| 127 | Atan |
| 128 | Acosh |
| 129 | Asinh |
| 130 | Atanh |

#### Constant Comparisons (compare x against alpha)

| Op | Function |
|----|----------|
| 112 | Is Zero (x == 0) |
| 131 | Less Than Const (x < alpha) |
| 132 | Less Equal Const (x <= alpha) |
| 133 | Greater Than Const (x > alpha) |
| 134 | Greater Equal Const (x >= alpha) |
| 135 | Not Equal Const (x != alpha) |
| 136 | Equal Const (x == alpha) |

**JSON example (multiply):**
```json
{
  "type": "elementwise",
  "operation": 1,
  "alpha": 1, "beta": 0, "fused_relu": 0,
  "bottom": "x,y",
  "top": "z",
  "name": "mul_0"
}
```

**Note:** Operations 14-21, 39-99, 108-111, 113, 116, 120, and 137+ load/run but produce zeros on single-input test. Many of these are likely binary ops requiring two inputs, or ops requiring specific weight blobs. Operations 200+ also accept but output zeros -- likely reserved or require special configuration.

---

### 3. `inner_product`

Dense / fully-connected / linear layer.

**Fields:**
- `nC` (int) -- output channels
- `nB` (int) -- input channels
- `has_biases` (0/1)
- `has_relu`, `has_tanh`, `has_prelu` (0/1) -- fused activation
- `blob_weights`, `blob_biases` (int) -- weight blob indices

**JSON example:**
```json
{
  "type": "inner_product",
  "nC": 64, "nB": 64,
  "has_biases": 1, "has_relu": 0, "has_tanh": 0, "has_prelu": 0,
  "blob_weights": 3, "blob_biases": 1,
  "bottom": "x", "top": "y",
  "name": "linear_0"
}
```

---

### 4. `convolution`

2D convolution (also used for 1D via expand/squeeze wrapper).

**Fields:**
- `C` (int) -- output channels
- `K` (int) -- input channels
- `Nx`, `Ny` (int) -- kernel width, height
- `n_groups` (int) -- groups (=C for depthwise)
- `dilation_x`, `dilation_y` (int) -- dilation
- `pad_t`, `pad_b`, `pad_l`, `pad_r` (int) -- padding
- `pad_mode` (int) -- 0=zeros, 1=same, 2=valid
- `has_biases` (0/1)
- `fused_relu`, `fused_tanh` (0/1) -- fused activation
- `has_batch_norm` (0/1)
- `blob_weights`, `blob_biases` (int) -- weight blob indices
- `n_parallel` (int) -- parallelism hint

---

### 5. `deconvolution`

Transposed convolution. Same fields as `convolution` plus:
- `deconv_out_height`, `deconv_out_width` (int) -- output spatial dims

---

### 6. `pool`

Pooling layer. The `avg_or_max` field selects the type.

| avg_or_max | Type |
|-----------|------|
| 0 | Average pooling |
| 1 | Max pooling |
| 2 | L2 pooling |

**Fields:**
- `size_x`, `size_y` (int) -- kernel size
- `stride_x`, `stride_y` (int) -- stride
- `pad_t`, `pad_b`, `pad_l`, `pad_r` (int)
- `is_global` (0/1) -- global pooling
- `average_count_exclude_padding` (0/1)
- `top_shape_style` (int)

---

### 7. `softmax`

Softmax normalization.

**Fields:** `C` (int) -- number of classes / channel dim

---

### 8. `batchnorm`

Batch normalization and instance normalization (same type, differentiated by flags).

**Fields:**
- `C` (int) -- channels
- `blob_batchnorm_params` (int) -- weight blob index (gamma, beta, mean, variance)
- `training` (0/1) -- if 1, computes running stats
- `training_instancenorm` (0/1) -- if 1, instance norm mode
- `training_eps` (float)
- `training_momentum` (float)

---

### 9. `lrn`

Local Response Normalization.

**Fields:** `alpha`, `beta`, `local_size` (int)

---

### 10. `reduce`

Reduction operations. The `mode` field selects which reduction.

| Mode | Function |
|------|----------|
| 0 | Sum |
| 1 | Mean |
| 2 | Product |
| 3 | Log Sum (exp first, sum, log) |
| 4 | Sum of Squares (L2 squared) |
| 5 | L2 Norm |
| 8 | Sum of Abs (L1 Norm) |
| 9 | Max |
| 10 | Min |
| 11 | Argmax |
| 12 | Argmin |

**Fields:**
- `axis_mode` (int) -- which axes to reduce
- `nd_axis` (int) -- axis index for nd_mode
- `nd_mode` (bool) -- N-D mode
- `use_version` (int)

Modes 13-19 also load and return max-like results (possibly aliases or reserved).

---

### 11. `reshape`

Reshape tensor.

**Fields:**
- `mode` (int) -- 0 = standard
- `dst_n`, `dst_k`, `dst_h`, `dst_w`, `dst_seq` (int) -- target shape
- `dst_nd_rank` (int) -- target rank
- `dynamic_shape` (bool)
- `version` (int)

---

### 12. `transpose`

Permute / transpose tensor axes.

**Fields:** `axis_n`, `axis_k`, `axis_h`, `axis_w`, `axis_seq` (int) -- permutation mapping

---

### 13. `concat`

Concatenate tensors along channel axis.

**Fields:** `bottom` is comma-separated list of inputs.

---

### 14. `split_nd`

Split tensor along an axis into multiple outputs.

**Fields:**
- `nd_axis` (int) -- axis to split
- `begin_N` (int) -- start index for each output (N = 0, 1, 2, ...)
- `top` is comma-separated output names

---

### 15. `stack_nd`

Stack tensors along a new axis.

**Fields:** `nd_axis` (int) -- axis to stack along

---

### 16. `slice`

Slice tensor along an axis.

**Fields:** `axis` (int), `start` (int), `end` (int)

---

### 17. `expand_dims`

Add dimension(s) to tensor.

**Fields:** `axes_0` (int), `size_of_axes` (int), `nd_axis` (int)

---

### 18. `squeeze`

Remove dimension(s) from tensor.

**Fields:** `axes_0` (int), `size_of_axes` (int), `nd_axis` (int), `version` (int)

---

### 19. `tile`

Tile / repeat tensor.

**Fields:** `dst_n`, `dst_k`, `dst_h`, `dst_w`, `dst_seq` (int) -- repeat counts per axis

---

### 20. `general_padding`

Padding with configurable mode.

| pad_pad_mode | Type |
|-------------|------|
| 0 | Constant (value = pad_value) |
| 1 | Reflect |
| 2 | Replicate / Edge |

**Fields:** `pad_t`, `pad_b`, `pad_l`, `pad_r`, `pad_value`, `pad_fill_mode`, `pad_pad_mode`

---

### 21. `crop`

Crop spatial dimensions.

**Fields:** `t`, `b`, `l`, `r` (int) -- pixels to crop from each side

---

### 22. `upsample`

Spatial upsampling.

| mode | Type |
|------|------|
| 0 | Nearest Neighbor |
| 1 | Bilinear |

**Fields:**
- `scaling_factor_x`, `scaling_factor_y` (int)
- `fractional_scaling_factor_x`, `fractional_scaling_factor_y` (float)
- `use_fractional_scale_factors` (bool)
- `align_corners` (0/1)
- `is_legacy_mode` (0/1)

---

### 23. `crop_and_resize`

Combined crop and resize (used for resize_bilinear).

**Fields:**
- `target_size_h`, `target_size_w` (int)
- `spatial_scale` (float)
- `grid_sampling_mode` (int) -- 2 = bilinear
- `crop_index_mode` (int)
- `mode` (int) -- 1 = bilinear

---

### 24. `space_to_depth` / `depth_to_space`

Pixel shuffle operations. Both use type `space_to_depth`.

**Fields:**
- `block_size` (int)
- `reverse` (0/1) -- 0 = space_to_depth, 1 = depth_to_space
- `pixel_shuffle` (0/1)

---

### 25. `batch_matmul`

Batched matrix multiplication.

**Fields:** `channel_mode` (bool), `bottom` = comma-separated "x,y"

---

### 26. `topk`

Top-K selection. Also used for argmax/argmin (k=1).

**Fields:**
- `num_k` (int) -- K value
- `axis` (int) -- axis to select along
- `single_axis_topk` (bool)
- `do_bottomk` (bool) -- if true, select bottom-K (for argmin)
- `top` is "values_name,indices_name"

---

### 27. `argsort`

Sort indices.

**Fields:** `axis` (int), `ascending` (bool)

---

### 28. `gather_nd`

Gather elements by index.

| mode | Type |
|------|------|
| 0 | Gather (select rows/slices) |
| 2 | Gather Along Axis |

**Fields:**
- `axis` (int)
- `batch_dims` (int)
- `allow_negative_indices` (bool)
- `validate_indices` (bool)

---

### 29. `scatter_nd`

Scatter updates by index.

| mode | Type |
|------|------|
| 2 | Scatter Along Axis |

**Fields:**
- `axis` (int)
- `accumulation_mode` (int) -- 1 = replace
- `allow_negative_indices` (bool)
- `validate_indices` (bool)

---

### 30. `load_constant`

Load a constant tensor from weights.

**Fields:**
- `constant_blob` (int) -- blob index
- `n`, `k`, `h`, `w` (int) -- shape
- `nd_rank` (int)

---

### 31. `cumsum`

Cumulative sum along an axis.

**Fields:** Takes axis as second input blob (from load_constant).

---

## Hidden / Undocumented Layer Types (probed via JSON patching)

These types are not generated by coremltools but are recognized by the CoreML runtime:

| Type | Behavior | Notes |
|------|----------|-------|
| `copy` | Identity / passthrough | x -> x |
| `cast` | Type cast (identity in float) | x -> x |
| `flatten` | Flatten tensor | x -> x (for 1D) |
| `l2_normalize` | L2 normalization | x / norm(x) |
| `sequence_repeat` | Repeat sequence | Passthrough for 1 repeat |
| `get_shape` | Return tensor shape | Returns [n, k, h, w, ...] |
| `nonzero` | Non-zero element indices | Returns index positions |
| `one_hot` | One-hot encoding | Requires configuration |
| `range` | Range generation | Loads but needs dynamic params |
| `fill` | Fill with constant | Loads but needs params |

---

## Summary Statistics

| Category | Count |
|----------|-------|
| Layer types (confirmed via coremltools) | 31 |
| Hidden layer types (probed) | 10 |
| **Total layer types** | **41** |
| Activation modes | 22 (0-26, with gaps) |
| Elementwise operations | 50+ confirmed functional |
| Reduce modes | 12+ (0-12) |
| Pooling modes | 3 (avg, max, L2) |

---

## Cross-Reference: coremltools MIL op -> espresso.net type

| MIL Operation | espresso.net type | mode/op |
|--------------|-------------------|---------|
| `mb.relu` | activation | mode=0 |
| `mb.sigmoid` | activation | mode=3 |
| `mb.tanh` | activation | mode=1 |
| `mb.elu` | activation | mode=8 |
| `mb.leaky_relu` | activation | mode=2 |
| `mb.prelu` | activation | mode=2 |
| `mb.thresholded_relu` | activation | mode=9 |
| `mb.softplus` | activation | mode=10 |
| `mb.softsign` | activation | mode=12 |
| `mb.gelu(EXACT)` | activation | mode=19 |
| `mb.gelu(TANH_APPROXIMATION)` | activation | mode=21 |
| `mb.gelu(SIGMOID_APPROXIMATION)` | activation | mode=22 |
| `mb.erf` | activation | mode=20 |
| `mb.silu` | activation(3) + elementwise(1) | sigmoid then multiply |
| `mb.linear_activation` | activation | mode=6 |
| `mb.scaled_tanh` | activation | mode=5 |
| `mb.linear` | inner_product | -- |
| `mb.conv` | convolution | -- |
| `mb.conv_transpose` | deconvolution | -- |
| `mb.max_pool` | pool | avg_or_max=1 |
| `mb.avg_pool` | pool | avg_or_max=0 |
| `mb.l2_pool` | pool | avg_or_max=2 |
| `mb.softmax` | softmax | -- |
| `mb.batch_norm` | batchnorm | -- |
| `mb.instance_norm` | batchnorm | training_instancenorm=1 |
| `mb.layer_norm` | (decomposed) | reduce + elementwise chain |
| `mb.local_response_norm` | lrn | -- |
| `mb.add` | elementwise | op=0 |
| `mb.sub` | activation(6) + elementwise(0) | negate then add |
| `mb.mul` | elementwise | op=1 |
| `mb.real_div` | elementwise | op=102 (nd_mode) |
| `mb.floor_div` | elementwise | op=1021 (nd_mode) |
| `mb.mod` | elementwise | op=36 (nd_mode) |
| `mb.pow` | elementwise | op=13 (nd_mode) |
| `mb.maximum` | elementwise | op=23 |
| `mb.minimum` | elementwise | op=22 |
| `mb.sqrt` | elementwise | op=11 |
| `mb.rsqrt` | elementwise | op=12 |
| `mb.exp` | elementwise | op=27 |
| `mb.exp2` | elementwise | op=35 |
| `mb.log` | elementwise | op=26 |
| `mb.abs` | elementwise | op=24 |
| `mb.sign` | elementwise | op=38 |
| `mb.ceil` | elementwise | op=118 |
| `mb.floor` | elementwise | op=117 |
| `mb.round` | elementwise | op=37 |
| `mb.sin` | elementwise | op=115 |
| `mb.cos` | elementwise | op=114 |
| `mb.tan` | elementwise | op=121 |
| `mb.asin` | elementwise | op=126 |
| `mb.acos` | elementwise | op=125 |
| `mb.atan` | elementwise | op=127 |
| `mb.sinh` | elementwise | op=123 |
| `mb.cosh` | elementwise | op=122 |
| `mb.atanh` | elementwise | op=130 |
| `mb.inverse` | elementwise | op=10 |
| `mb.clip` | elementwise | op=119 |
| `mb.less` | elementwise | op=103 |
| `mb.less_equal` | elementwise | op=104 |
| `mb.greater` | elementwise | op=105 |
| `mb.greater_equal` | elementwise | op=106 |
| `mb.equal` | elementwise | op=100 |
| `mb.not_equal` | elementwise | op=107 |
| `mb.select` | elementwise | op=101 (nd_mode) |
| `mb.reduce_sum` | reduce | mode=0 |
| `mb.reduce_mean` | pool | avg_or_max=0, is_global=1 |
| `mb.reduce_max` | pool | avg_or_max=1, is_global=1 |
| `mb.reduce_min` | reduce | mode=10 |
| `mb.reduce_prod` | reduce | mode=2 |
| `mb.reduce_l2_norm` | reduce | mode=5 |
| `mb.reduce_argmax` | topk | do_bottomk=false, num_k=1 |
| `mb.reduce_argmin` | topk | do_bottomk=true, num_k=1 |
| `mb.reshape` | reshape | mode=0 |
| `mb.transpose` | transpose | -- |
| `mb.concat` | concat | -- |
| `mb.split` | split_nd | -- |
| `mb.slice_by_index` | slice | -- |
| `mb.expand_dims` | expand_dims | -- |
| `mb.squeeze` | squeeze | -- |
| `mb.tile` | tile | -- |
| `mb.stack` | stack_nd | -- |
| `mb.pad` | general_padding | -- |
| `mb.crop` | crop | -- |
| `mb.upsample_nearest_neighbor` | upsample | mode=0 |
| `mb.upsample_bilinear` | upsample | mode=1 |
| `mb.resize_bilinear` | crop_and_resize | mode=1 |
| `mb.space_to_depth` | space_to_depth | reverse=0 |
| `mb.depth_to_space` | space_to_depth | reverse=1 |
| `mb.matmul` | batch_matmul | -- |
| `mb.gather` | gather_nd | mode=0 |
| `mb.gather_along_axis` | gather_nd | mode=2 |
| `mb.scatter_along_axis` | scatter_nd | mode=2 |
| `mb.topk` | topk | -- |
| `mb.argsort` | argsort | -- |
| `mb.cumsum` | cumsum | -- |

---

## Fused Operations

Several layer types support fused activations, eliminating a separate activation layer:

| Layer | Fused Fields |
|-------|-------------|
| `convolution` | `fused_relu`, `fused_tanh` |
| `deconvolution` | `fused_relu`, `fused_tanh` |
| `inner_product` | `has_relu`, `has_tanh`, `has_prelu` |
| `elementwise` | `fused_relu` |
| `convolution` | `has_batch_norm` |

---

## Key Observations for ANE Optimization

1. **Layer normalization is decomposed** into 14 espresso layers (reduce_mean, elementwise chain). This is a significant overhead versus a native layernorm instruction. A fused `batchnorm` with `training_instancenorm=1` is far more efficient for per-instance normalization.

2. **SiLU/Swish is decomposed** into sigmoid(mode=3) + elementwise multiply(op=1). Activation mode 25 provides it natively but is not used by coremltools.

3. **Subtract is decomposed** into linear activation(mode=6, alpha=-1) + add(op=0). Direct elementwise subtract(op=33, nd_mode=true) exists but coremltools doesn't use it.

4. **Divide uses nd_mode** (op=102) while simpler binary ops like add/multiply don't.

5. **Reduce mean/max map to pool** (with is_global=1) rather than reduce. This is likely because the ANE pool hardware is more efficient than the reduce path.

6. **Argmax/argmin map to topk** (k=1) rather than a dedicated argmax layer.

7. **Constant comparisons** (elementwise ops 131-136) compare against the `alpha` parameter directly -- no need for a load_constant + binary comparison. These are not generated by coremltools.

8. **SELU (mode=16)**, **Hard Sigmoid (mode=7)**, **HardSwish (mode=26)**, **Heaviside (mode=23)**, **Clamp variants (modes=13,24)** all exist natively but are not accessible via standard coremltools. Direct espresso.net patching required.

9. **Reduce modes 4 (sum_squares), 8 (L1 norm/sum_abs), 9 (max), 11 (argmax), 12 (argmin)** exist natively in the reduce layer but are only partially used by coremltools.

---

## Data Files

- `espresso_type_catalog.json` -- Full JSON schemas for all 31 confirmed layer types with sample layers
- `espresso_vocab_raw.json` -- Raw espresso.net layers extracted from 106 reference models
- `probe_results.json` -- Results from brute-force probing of hidden modes and types
- `map_espresso_vocab.py` -- Phase 1: build reference models with coremltools MIL
- `probe_espresso_types.py` -- Phase 2: probe hidden operations via JSON patching
