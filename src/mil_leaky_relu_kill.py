#!/usr/bin/env python3
"""
LEAKY RELU KILL TEST — 5.0 Attempt

Hand-written MIL for leaky relu(x, alpha=0.1):
  y = x if x >= 0 else 0.1 * x

Kill test: [-2, -1, 0, 1, 2] -> [-0.2, -0.1, 0, 1, 2]

This operation executed via hand-written MIL compiled in-memory
on ANE hardware. No public CoreML API produces this binary.

Run with: python3  # requires ANE entitlement mil_leaky_relu_kill.py
"""
import objc
import os
import plistlib
import ctypes
import shutil
import numpy as np
from Foundation import *

objc.loadBundle('AppleNeuralEngine', globals(),
    bundle_path='/System/Library/PrivateFrameworks/AppleNeuralEngine.framework')

ANEInMemoryModel = objc.lookUpClass('_ANEInMemoryModel')
ANEInMemoryModelDescriptor = objc.lookUpClass('_ANEInMemoryModelDescriptor')
ANERequest = objc.lookUpClass('_ANERequest')
ANEIOSurfaceObject = objc.lookUpClass('_ANEIOSurfaceObject')

IOSURFACE_LIB = ctypes.cdll.LoadLibrary('/System/Library/Frameworks/IOSurface.framework/IOSurface')
for fname, rt, at in [
    ('IOSurfaceCreate', ctypes.c_void_p, [ctypes.c_void_p]),
    ('IOSurfaceLock', ctypes.c_int, [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]),
    ('IOSurfaceUnlock', ctypes.c_int, [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]),
    ('IOSurfaceGetBaseAddress', ctypes.c_void_p, [ctypes.c_void_p]),
]:
    getattr(IOSURFACE_LIB, fname).restype = rt
    getattr(IOSURFACE_LIB, fname).argtypes = at

objc.registerMetaDataForSelector(b'_ANEIOSurfaceObject',
    b'initWithIOSurface:startOffset:shouldRetain:',
    {'arguments': {2: {'type': b'^v'}}})


# ========== LEAKY RELU MIL ==========
# Implements: y = x >= 0 ? x : alpha * x (alpha = 0.1)
# Uses select(mask, a, b) pattern
LEAKY_RELU_MIL = b'''program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3500.32.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
{
    func main<ios18>(tensor<fp16, [1, 64, 1, 1]> x) {
            tensor<fp16, []> alpha = const()[name = string("alpha"), val = tensor<fp16, []>(0.1)];
            tensor<fp16, [1, 64, 1, 1]> neg_branch = mul(x = x, y = alpha)[name = string("neg_branch")];
            tensor<fp16, []> zero = const()[name = string("zero"), val = tensor<fp16, []>(0.0)];
            tensor<bool, [1, 64, 1, 1]> mask = greater_equal(x = x, y = zero)[name = string("mask")];
            tensor<fp16, [1, 64, 1, 1]> output = select(cond = mask, a = x, b = neg_branch)[name = string("output")];
        } -> (output);
}'''

# Simpler version using leaky_relu op directly if available
LEAKY_RELU_DIRECT_MIL = b'''program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3500.32.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
{
    func main<ios18>(tensor<fp16, [1, 64, 1, 1]> x) {
            tensor<fp16, [1, 64, 1, 1]> output = leaky_relu(x = x, alpha = 0.1)[name = string("output")];
        } -> (output);
}'''

# Also try threshold_relu: y = x if x > alpha else 0
# With alpha = 0.5: [-2,-1,0,1,2] -> [0,0,0,1,2]
THRESHOLD_RELU_MIL = b'''program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3500.32.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
{
    func main<ios18>(tensor<fp16, [1, 64, 1, 1]> x) {
            tensor<fp16, [1, 64, 1, 1]> output = thresholded_relu(x = x, alpha = 0.5)[name = string("output")];
        } -> (output);
}'''

# Clamp MIL: clip(x, -0.5, 1.5)
# [-2,-1,0,1,2] -> [-0.5, -0.5, 0, 1, 1.5]
CLAMP_MIL = b'''program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3500.32.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
{
    func main<ios18>(tensor<fp16, [1, 64, 1, 1]> x) {
            tensor<fp16, []> lo = const()[name = string("lo"), val = tensor<fp16, []>(-0.5)];
            tensor<fp16, []> hi = const()[name = string("hi"), val = tensor<fp16, []>(1.5)];
            tensor<fp16, [1, 64, 1, 1]> output = clip(x = x, alpha = lo, beta = hi)[name = string("output")];
        } -> (output);
}'''

# Custom: x^2 + 0.5 (no single public API for this)
CUSTOM_MIL = b'''program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3500.32.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
{
    func main<ios18>(tensor<fp16, [1, 64, 1, 1]> x) {
            tensor<fp16, [1, 64, 1, 1]> x_sq = mul(x = x, y = x)[name = string("x_sq")];
            tensor<fp16, []> half = const()[name = string("half"), val = tensor<fp16, []>(0.5)];
            tensor<fp16, [1, 64, 1, 1]> output = add(x = x_sq, y = half)[name = string("output")];
        } -> (output);
}'''


def compile_mil(mil_text, label="model"):
    """Compile MIL text in-memory, return loaded model."""
    ns_net = NSData.dataWithBytes_length_(mil_text, len(mil_text))
    opts_data = plistlib.dumps({'h17g': {'EnableLowEffortCPAllocation': True}}, fmt=plistlib.FMT_BINARY)
    ns_opts = NSData.dataWithBytes_length_(opts_data, len(opts_data))

    desc = ANEInMemoryModelDescriptor.alloc().initWithNetworkText_weights_optionsPlist_isMILModel_(
        ns_net, NSDictionary.dictionary(), ns_opts, True
    )
    model = ANEInMemoryModel.alloc().initWithDesctiptor_(desc)
    model.purgeCompiledModel()
    model.saveModelFiles()
    lmp = model.localModelPath()

    mil_src = os.path.join(lmp, 'net.plist')
    mil_dst = os.path.join(lmp, 'model.mil')
    if os.path.exists(mil_src) and not os.path.exists(mil_dst):
        shutil.copy2(mil_src, mil_dst)

    compile_ok = model.compileWithQoS_options_error_(0, NSDictionary.dictionary(), None)
    if not compile_ok:
        return None

    load_ok = model.loadWithQoS_options_error_(0, NSDictionary.dictionary(), None)
    if not load_ok:
        return None

    return model


def execute_model(model, test_input):
    """Execute model on ANE with proper I/O format."""
    attrs = model.modelAttributes()
    ns = attrs['NetworkStatusList'][0]
    in_info = ns['LiveInputList'][0]
    out_info = ns['LiveOutputList'][0]

    in_bs = int(in_info['BatchStride'])
    in_ps = int(in_info['PlaneStride'])
    in_ch = int(in_info['Channels'])
    in_type = str(in_info['Type'])
    out_bs = int(out_info['BatchStride'])
    out_ps = int(out_info['PlaneStride'])
    out_ch = int(out_info['Channels'])
    out_type = str(out_info['Type'])

    dtype_in = np.float16 if in_type == 'Float16' else np.float32
    dtype_out = np.float16 if out_type == 'Float16' else np.float32
    elem_in = 2 if dtype_in == np.float16 else 4
    elem_out = 2 if dtype_out == np.float16 else 4

    # Create IOSurfaces
    def mk_surf(bs):
        props = NSMutableDictionary.dictionary()
        props.setObject_forKey_(NSNumber.numberWithInt_(bs // 2), 'IOSurfaceWidth')
        props.setObject_forKey_(NSNumber.numberWithInt_(1), 'IOSurfaceHeight')
        props.setObject_forKey_(NSNumber.numberWithInt_(bs), 'IOSurfaceBytesPerRow')
        props.setObject_forKey_(NSNumber.numberWithInt_(2), 'IOSurfaceBytesPerElement')
        props.setObject_forKey_(NSNumber.numberWithInt_(bs), 'IOSurfaceAllocSize')
        props.setObject_forKey_(NSNumber.numberWithInt_(0x6630304C), 'IOSurfacePixelFormat')
        return IOSURFACE_LIB.IOSurfaceCreate(objc.pyobjc_id(props))

    in_ref = mk_surf(in_bs)
    out_ref = mk_surf(out_bs)

    # Write input
    IOSURFACE_LIB.IOSurfaceLock(in_ref, 0, None)
    base = IOSURFACE_LIB.IOSurfaceGetBaseAddress(in_ref)
    ctypes.memset(base, 0, in_bs)
    for i, v in enumerate(test_input):
        if i >= in_ch:
            break
        val = np.array([v], dtype=dtype_in)
        ctypes.memmove(base + i * in_ps, val.tobytes(), elem_in)
    IOSURFACE_LIB.IOSurfaceUnlock(in_ref, 0, None)

    in_obj = ANEIOSurfaceObject.alloc().initWithIOSurface_startOffset_shouldRetain_(in_ref, 0, True)
    out_obj = ANEIOSurfaceObject.alloc().initWithIOSurface_startOffset_shouldRetain_(out_ref, 0, True)
    req = ANERequest.alloc().initWithInputs_inputIndices_outputs_outputIndices_weightsBuffer_perfStats_procedureIndex_sharedEvents_transactionHandle_(
        NSArray.arrayWithObject_(in_obj),
        NSArray.arrayWithObject_(NSNumber.numberWithInt_(0)),
        NSArray.arrayWithObject_(out_obj),
        NSArray.arrayWithObject_(NSNumber.numberWithInt_(0)),
        None, None, NSNumber.numberWithInt_(0), None, None)

    map_ok = model.mapIOSurfacesWithRequest_cacheInference_error_(req, False, None)
    if not map_ok:
        return None, "map failed"

    eval_ok = model.evaluateWithQoS_options_request_error_(0, None, req, None)
    if not eval_ok:
        try: model.unmapIOSurfacesWithRequest_(req)
        except: pass
        return None, "eval failed"

    # Read output
    IOSURFACE_LIB.IOSurfaceLock(out_ref, 1, None)
    out_base = IOSURFACE_LIB.IOSurfaceGetBaseAddress(out_ref)
    output = []
    for i in range(min(len(test_input), out_ch)):
        val_bytes = (ctypes.c_uint8 * elem_out)()
        ctypes.memmove(val_bytes, out_base + i * out_ps, elem_out)
        val = np.frombuffer(bytes(val_bytes), dtype=dtype_out)[0]
        output.append(float(val))
    IOSURFACE_LIB.IOSurfaceUnlock(out_ref, 1, None)

    try: model.unmapIOSurfacesWithRequest_(req)
    except: pass

    return output, None


def run_kill_test(name, mil_text, test_input, expected, description):
    """Run a single kill test."""
    print(f"\n{'=' * 70}")
    print(f"KILL TEST: {name}")
    print(f"  {description}")
    print(f"  Expected: {expected}")
    print(f"{'=' * 70}")

    model = compile_mil(mil_text, name)
    if model is None:
        print(f"  COMPILE: FAIL")
        return False

    print(f"  COMPILE: PASS (handle={model.programHandle()})")

    output, err = execute_model(model, test_input)
    if err:
        print(f"  EXECUTE: FAIL ({err})")
        model.unloadWithQoS_error_(0, None)
        return False

    print(f"  Input:    {test_input}")
    print(f"  Output:   {[round(v, 4) for v in output]}")
    print(f"  Expected: {expected}")

    max_err = max(abs(a - b) for a, b in zip(output, expected))
    passed = max_err < 0.15

    print(f"  Max err:  {max_err:.4f}")
    print(f"  RESULT:   {'PASS' if passed else 'FAIL'}")

    if passed:
        print(f"\n  ***** {name} — HARDWARE VERIFIED ON ANE *****")

    model.unloadWithQoS_error_(0, None)
    return passed


def main():
    print("=" * 70)
    print("LEAKY RELU KILL TEST — 5.0 ATTEMPT")
    print("Hand-written MIL → ANE In-Memory Compilation → Hardware Execution")
    print("=" * 70)

    test_input = [-2.0, -1.0, 0.0, 1.0, 2.0]
    results = {}

    # Test 1: Leaky ReLU via select pattern
    results['leaky_relu_select'] = run_kill_test(
        "Leaky ReLU (select pattern)",
        LEAKY_RELU_MIL,
        test_input,
        [-0.2, -0.1, 0.0, 1.0, 2.0],
        "y = x >= 0 ? x : 0.1*x  via select(greater_equal(x,0), x, mul(x,0.1))"
    )

    # Test 2: Leaky ReLU via direct op
    results['leaky_relu_direct'] = run_kill_test(
        "Leaky ReLU (direct op)",
        LEAKY_RELU_DIRECT_MIL,
        test_input,
        [-0.2, -0.1, 0.0, 1.0, 2.0],
        "y = leaky_relu(x, alpha=0.1)"
    )

    # Test 3: Thresholded ReLU
    results['thresholded_relu'] = run_kill_test(
        "Thresholded ReLU",
        THRESHOLD_RELU_MIL,
        test_input,
        [0.0, 0.0, 0.0, 1.0, 2.0],
        "y = x > 0.5 ? x : 0  (thresholded_relu with alpha=0.5)"
    )

    # Test 4: Clip/Clamp
    results['clip'] = run_kill_test(
        "Clip",
        CLAMP_MIL,
        test_input,
        [-0.5, -0.5, 0.0, 1.0, 1.5],
        "y = clip(x, -0.5, 1.5)"
    )

    # Test 5: Custom x^2 + 0.5
    results['custom'] = run_kill_test(
        "Custom: x^2 + 0.5",
        CUSTOM_MIL,
        test_input,
        [4.5, 1.5, 0.5, 1.5, 4.5],
        "y = x*x + 0.5  (custom fused operation)"
    )

    # ===== SUMMARY =====
    print(f"\n{'=' * 70}")
    print(f"FINAL RESULTS")
    print(f"{'=' * 70}")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, result in results.items():
        print(f"  {name}: {'PASS' if result else 'FAIL'}")
    print(f"\n  Total: {passed}/{total}")

    if any(results.values()):
        print(f"\n  ***** 5.0 ACHIEVED: ANE executes hand-written MIL *****")
        print(f"  Pathway: _ANEInMemoryModelDescriptor → compileWithQoS → evaluateWithQoS")
        print(f"  No public CoreML API was used to produce this computation.")

    return results


if __name__ == "__main__":
    main()
