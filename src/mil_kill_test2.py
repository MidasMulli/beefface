#!/usr/bin/env python3
"""
MIL Kill Test v2: Fixed I/O format.

Model attributes say: Type=Float32, BatchStride=4096, PlaneStride=64, Channels=64
So we need fp32 data at 64-byte stride offsets in a 4096-byte IOSurface.

Run with: python3  # requires ANE entitlement mil_kill_test2.py
"""
import objc
import os
import plistlib
import ctypes
import shutil
import struct
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
    ('IOSurfaceGetAllocSize', ctypes.c_size_t, [ctypes.c_void_p]),
    ('IOSurfaceLock', ctypes.c_int, [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]),
    ('IOSurfaceUnlock', ctypes.c_int, [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]),
    ('IOSurfaceGetBaseAddress', ctypes.c_void_p, [ctypes.c_void_p]),
]:
    getattr(IOSURFACE_LIB, fname).restype = rt
    getattr(IOSURFACE_LIB, fname).argtypes = at

objc.registerMetaDataForSelector(b'_ANEIOSurfaceObject',
    b'initWithIOSurface:startOffset:shouldRetain:',
    {'arguments': {2: {'type': b'^v'}}})


RELU_MIL = b'''program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3500.32.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
{
    func main<ios18>(tensor<fp32, [1, 64, 1, 1]> x) {
            string cast_0_dtype_0 = const()[name = string("cast_0_dtype_0"), val = string("fp16")];
            tensor<fp16, [1, 64, 1, 1]> cast_0 = cast(dtype = cast_0_dtype_0, x = x)[name = string("cast_2")];
            tensor<fp16, [1, 64, 1, 1]> op_out = relu(x = cast_0)[name = string("op_out")];
            string output_dtype_0 = const()[name = string("output_dtype_0"), val = string("fp32")];
            tensor<fp32, [1, 64, 1, 1]> output = cast(dtype = output_dtype_0, x = op_out)[name = string("cast_1")];
        } -> (output);
}'''

# FP16 version — no cast, pure fp16 I/O
RELU_MIL_FP16 = b'''program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3500.32.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
{
    func main<ios18>(tensor<fp16, [1, 64, 1, 1]> x) {
            tensor<fp16, [1, 64, 1, 1]> output = relu(x = x)[name = string("op_out")];
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

    # Copy net.plist to model.mil
    mil_src = os.path.join(lmp, 'net.plist')
    mil_dst = os.path.join(lmp, 'model.mil')
    if os.path.exists(mil_src) and not os.path.exists(mil_dst):
        shutil.copy2(mil_src, mil_dst)

    compile_ok = model.compileWithQoS_options_error_(0, NSDictionary.dictionary(), None)
    if not compile_ok:
        print(f"  [{label}] Compile FAILED")
        return None

    load_ok = model.loadWithQoS_options_error_(0, NSDictionary.dictionary(), None)
    if not load_ok:
        print(f"  [{label}] Load FAILED")
        return None

    return model


def execute_model(model, test_input, label="test"):
    """Execute model with various I/O format strategies."""
    attrs = model.modelAttributes()
    ns = attrs['NetworkStatusList'][0]
    in_info = ns['LiveInputList'][0]
    out_info = ns['LiveOutputList'][0]

    in_bs = int(in_info['BatchStride'])
    out_bs = int(out_info['BatchStride'])
    in_ps = int(in_info['PlaneStride'])
    out_ps = int(out_info['PlaneStride'])
    in_ch = int(in_info['Channels'])
    out_ch = int(out_info['Channels'])
    in_type = str(in_info['Type'])
    out_type = str(out_info['Type'])

    print(f"  [{label}] I/O: bs={in_bs}, ps={in_ps}, ch={in_ch}, type={in_type}")

    # Try multiple strategies for data layout
    strategies = []

    if in_type == 'Float32':
        # Strategy 1: fp32 at PlaneStride offsets
        strategies.append(("fp32_at_ps", np.float32, in_ps, out_ps, 4))
        # Strategy 2: fp32 packed (ignore PlaneStride)
        strategies.append(("fp32_packed", np.float32, 4, 4, 4))

    if in_type == 'Float16' or True:  # Always try fp16 too
        # Strategy 3: fp16 at PlaneStride offsets
        strategies.append(("fp16_at_ps", np.float16, in_ps, out_ps, 2))
        # Strategy 4: fp16 packed
        strategies.append(("fp16_packed", np.float16, 2, 2, 2))

    # Strategy 5: fp16 at 64-byte stride (common ANE layout)
    strategies.append(("fp16_stride64", np.float16, 64, 64, 2))
    # Strategy 6: fp32 contiguous
    strategies.append(("fp32_contig", np.float32, 4, 4, 4))

    for strat_name, dtype, in_stride, out_stride, elem_size in strategies:
        try:
            # Create IOSurface
            props = NSMutableDictionary.dictionary()
            props.setObject_forKey_(NSNumber.numberWithInt_(in_bs // 2), 'IOSurfaceWidth')
            props.setObject_forKey_(NSNumber.numberWithInt_(1), 'IOSurfaceHeight')
            props.setObject_forKey_(NSNumber.numberWithInt_(in_bs), 'IOSurfaceBytesPerRow')
            props.setObject_forKey_(NSNumber.numberWithInt_(2), 'IOSurfaceBytesPerElement')
            props.setObject_forKey_(NSNumber.numberWithInt_(in_bs), 'IOSurfaceAllocSize')
            props.setObject_forKey_(NSNumber.numberWithInt_(0x6630304C), 'IOSurfacePixelFormat')

            in_ref = IOSURFACE_LIB.IOSurfaceCreate(objc.pyobjc_id(props))
            out_ref = IOSURFACE_LIB.IOSurfaceCreate(objc.pyobjc_id(props))

            # Write input data at stride offsets
            IOSURFACE_LIB.IOSurfaceLock(in_ref, 0, None)
            base = IOSURFACE_LIB.IOSurfaceGetBaseAddress(in_ref)

            # Zero the whole surface first
            ctypes.memset(base, 0, in_bs)

            for i, v in enumerate(test_input):
                if i >= in_ch:
                    break
                val = np.array([v], dtype=dtype)
                ctypes.memmove(base + i * in_stride, val.tobytes(), elem_size)

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
                continue

            eval_ok = model.evaluateWithQoS_options_request_error_(0, None, req, None)
            if not eval_ok:
                try:
                    model.unmapIOSurfacesWithRequest_(req)
                except:
                    pass
                continue

            # Read output
            IOSURFACE_LIB.IOSurfaceLock(out_ref, 1, None)
            out_base = IOSURFACE_LIB.IOSurfaceGetBaseAddress(out_ref)

            output = []
            for i in range(min(len(test_input), out_ch)):
                val_bytes = (ctypes.c_uint8 * elem_size)()
                ctypes.memmove(val_bytes, out_base + i * out_stride, elem_size)
                val = np.frombuffer(bytes(val_bytes), dtype=dtype)[0]
                output.append(float(val))

            # Also read raw hex for debugging
            raw = (ctypes.c_uint8 * min(256, out_bs))()
            ctypes.memmove(raw, out_base, min(256, out_bs))
            IOSURFACE_LIB.IOSurfaceUnlock(out_ref, 1, None)

            try:
                model.unmapIOSurfacesWithRequest_(req)
            except:
                pass

            has_nonzero = any(v != 0.0 for v in output)

            if has_nonzero:
                print(f"    {strat_name}: {[round(v, 4) for v in output]}  *** NON-ZERO! ***")
                # Check if it's relu
                expected_relu = [max(0.0, v) for v in test_input]
                match = all(abs(a - b) < 0.2 for a, b in zip(output, expected_relu))
                if match:
                    print(f"    *** RELU CONFIRMED! ***")
                return output, strat_name
            else:
                # Show first non-zero byte position in raw output
                first_nz = -1
                for bi in range(min(256, out_bs)):
                    if raw[bi] != 0:
                        first_nz = bi
                        break
                if first_nz >= 0:
                    print(f"    {strat_name}: output zeros at stride, but raw non-zero at byte {first_nz}")
                    # Try reading at that offset
                    nz_bytes = bytes(raw[first_nz:first_nz+20])
                    print(f"      raw[{first_nz}:]: {nz_bytes.hex()}")

        except Exception as e:
            pass

    # If all strategies failed, dump raw output bytes
    print(f"  [{label}] All strategies returned zeros. Dumping raw output...")
    try:
        props = NSMutableDictionary.dictionary()
        props.setObject_forKey_(NSNumber.numberWithInt_(in_bs // 2), 'IOSurfaceWidth')
        props.setObject_forKey_(NSNumber.numberWithInt_(1), 'IOSurfaceHeight')
        props.setObject_forKey_(NSNumber.numberWithInt_(in_bs), 'IOSurfaceBytesPerRow')
        props.setObject_forKey_(NSNumber.numberWithInt_(2), 'IOSurfaceBytesPerElement')
        props.setObject_forKey_(NSNumber.numberWithInt_(in_bs), 'IOSurfaceAllocSize')
        props.setObject_forKey_(NSNumber.numberWithInt_(0x6630304C), 'IOSurfacePixelFormat')

        in_ref = IOSURFACE_LIB.IOSurfaceCreate(objc.pyobjc_id(props))
        out_ref = IOSURFACE_LIB.IOSurfaceCreate(objc.pyobjc_id(props))

        # Write a STRONG signal: all values = 3.14
        IOSURFACE_LIB.IOSurfaceLock(in_ref, 0, None)
        base = IOSURFACE_LIB.IOSurfaceGetBaseAddress(in_ref)
        # Fill entire surface with fp16 3.14
        pi_fp16 = np.float16(3.14)
        pi_bytes = pi_fp16.tobytes()
        for i in range(in_bs // 2):
            ctypes.memmove(base + i * 2, pi_bytes, 2)
        IOSURFACE_LIB.IOSurfaceUnlock(in_ref, 0, None)

        in_obj = ANEIOSurfaceObject.alloc().initWithIOSurface_startOffset_shouldRetain_(in_ref, 0, True)
        out_obj = ANEIOSurfaceObject.alloc().initWithIOSurface_startOffset_shouldRetain_(out_ref, 0, True)
        req = ANERequest.alloc().initWithInputs_inputIndices_outputs_outputIndices_weightsBuffer_perfStats_procedureIndex_sharedEvents_transactionHandle_(
            NSArray.arrayWithObject_(in_obj),
            NSArray.arrayWithObject_(NSNumber.numberWithInt_(0)),
            NSArray.arrayWithObject_(out_obj),
            NSArray.arrayWithObject_(NSNumber.numberWithInt_(0)),
            None, None, NSNumber.numberWithInt_(0), None, None)

        model.mapIOSurfacesWithRequest_cacheInference_error_(req, False, None)
        model.evaluateWithQoS_options_request_error_(0, None, req, None)

        IOSURFACE_LIB.IOSurfaceLock(out_ref, 1, None)
        out_base = IOSURFACE_LIB.IOSurfaceGetBaseAddress(out_ref)
        raw = (ctypes.c_uint8 * out_bs)()
        ctypes.memmove(raw, out_base, out_bs)
        IOSURFACE_LIB.IOSurfaceUnlock(out_ref, 1, None)

        # Find any non-zero bytes
        nz_positions = [i for i in range(out_bs) if raw[i] != 0]
        if nz_positions:
            print(f"    Non-zero bytes found at: {nz_positions[:20]}...")
            for pos in nz_positions[:5]:
                chunk = bytes(raw[pos:pos+8])
                as_fp16 = np.frombuffer(chunk[:2], dtype=np.float16)[0] if len(chunk) >= 2 else 0
                as_fp32 = np.frombuffer(chunk[:4], dtype=np.float32)[0] if len(chunk) >= 4 else 0
                print(f"      [{pos}]: hex={chunk.hex()} fp16={as_fp16:.4f} fp32={as_fp32:.6f}")
        else:
            print(f"    Output is ENTIRELY zeros ({out_bs} bytes)")
            print(f"    The evaluation succeeded but ANE produced no output.")
            print(f"    This likely means the compiled model is a no-op on hardware.")

        try:
            model.unmapIOSurfacesWithRequest_(req)
        except:
            pass
    except Exception as e:
        print(f"    Raw dump error: {e}")

    return None, None


def main():
    print("=" * 70)
    print("MIL KILL TEST v2 — Fixed I/O Format")
    print("=" * 70)
    test_input = [-2.0, -1.0, 0.0, 1.0, 2.0]

    # Test 1: fp32 I/O relu MIL (the original)
    print("\n[TEST 1] fp32 I/O relu MIL")
    model = compile_mil(RELU_MIL, "fp32_relu")
    if model:
        output, strat = execute_model(model, test_input, "fp32_relu")
        if output:
            print(f"\n  RESULT: {output} via {strat}")
        model.unloadWithQoS_error_(0, None)

    # Test 2: fp16 I/O relu MIL (no casts)
    print("\n[TEST 2] fp16 I/O relu MIL")
    model = compile_mil(RELU_MIL_FP16, "fp16_relu")
    if model:
        # Check what attributes say for fp16 model
        attrs = model.modelAttributes()
        ns = attrs['NetworkStatusList'][0]
        in_info = ns['LiveInputList'][0]
        print(f"  Type={in_info['Type']}, BS={in_info['BatchStride']}, PS={in_info['PlaneStride']}, Ch={in_info['Channels']}")

        output, strat = execute_model(model, test_input, "fp16_relu")
        if output:
            print(f"\n  RESULT: {output} via {strat}")
        model.unloadWithQoS_error_(0, None)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
