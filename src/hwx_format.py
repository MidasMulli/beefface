"""
HWX Binary Format Specification for ANE Zin Binaries (BEEFFACE)

Complete format definition derived from byte-level analysis of compiler-generated
relu and abs HWX files (49,152 bytes each, H17G subtype 9).

This module provides data structures for parsing and generating HWX binaries.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import struct

# =============================================================================
# Constants
# =============================================================================

BEEFFACE_MAGIC = 0xBEEFFACE
CPU_TYPE_ANE = 0x80
CPU_SUBTYPE_H17G = 0x09
FILETYPE_EXECUTE = 0x02
FLAGS_DEFAULT = 0x00200000

# Load command types
LC_CMD_0x02 = 0x02      # Symbol table offsets
LC_THREAD = 0x04         # Thread state (TD config / kernel I/O)
LC_CMD_0x08 = 0x08       # Compiler metadata string
LC_SEGMENT_64 = 0x19     # Segment (Mach-O style)
LC_CMD_0x40 = 0x40       # FVMLIB reference descriptor

# VM address base for ANE
ANE_VM_BASE = 0x30000000

# Standard file size for simple single-op kernels
STANDARD_FILE_SIZE = 49152  # 0xC000

# =============================================================================
# Mach-O Header
# =============================================================================

@dataclass
class MachOHeader:
    """32-byte Mach-O header at file offset 0x0000."""
    magic: int = BEEFFACE_MAGIC
    cputype: int = CPU_TYPE_ANE
    cpusubtype: int = CPU_SUBTYPE_H17G
    filetype: int = FILETYPE_EXECUTE
    ncmds: int = 11
    sizeofcmds: int = 0x2F10  # 12048
    flags: int = FLAGS_DEFAULT
    reserved: int = 0

    SIZE = 32
    FORMAT = '<IIIIIIII'

    def pack(self) -> bytes:
        return struct.pack(self.FORMAT,
            self.magic, self.cputype, self.cpusubtype, self.filetype,
            self.ncmds, self.sizeofcmds, self.flags, self.reserved)

    @classmethod
    def unpack(cls, data: bytes, offset: int = 0) -> 'MachOHeader':
        vals = struct.unpack_from(cls.FORMAT, data, offset)
        return cls(*vals)


# =============================================================================
# Section Header (within LC_SEGMENT_64)
# =============================================================================

@dataclass
class SectionHeader:
    """80-byte section header within an LC_SEGMENT_64 command."""
    sectname: str
    segname: str
    addr: int
    size: int
    fileoff: int
    align: int
    reloff: int = 0
    nreloc: int = 0
    flags: int = 0
    reserved1: int = 0
    reserved2: int = 0

    SIZE = 80

    def pack(self) -> bytes:
        return (
            self.sectname.encode('ascii').ljust(16, b'\x00')[:16] +
            self.segname.encode('ascii').ljust(16, b'\x00')[:16] +
            struct.pack('<QQIIIII II',
                self.addr, self.size,
                self.fileoff, self.align, self.reloff, self.nreloc, self.flags,
                self.reserved1, self.reserved2)
        )


# =============================================================================
# Load Command Definitions
# =============================================================================

@dataclass
class SegmentCommand:
    """LC_SEGMENT_64 (cmd=0x19) load command."""
    segname: str
    vmaddr: int
    vmsize: int
    fileoff: int
    filesize: int
    maxprot: int
    initprot: int
    nsects: int
    flags: int
    sections: List[SectionHeader] = field(default_factory=list)

    @property
    def cmdsize(self) -> int:
        return 72 + 80 * self.nsects

    def pack(self) -> bytes:
        data = struct.pack('<II', LC_SEGMENT_64, self.cmdsize)
        data += self.segname.encode('ascii').ljust(16, b'\x00')[:16]
        data += struct.pack('<QQQQIIII',
            self.vmaddr, self.vmsize, self.fileoff, self.filesize,
            self.maxprot, self.initprot, self.nsects, self.flags)
        for sect in self.sections:
            data += sect.pack()
        return data


@dataclass
class FVMLibRef:
    """LC_CMD_0x40 — FVMLIB reference descriptor (32 bytes)."""
    vmaddr: int         # VM address of referenced FVMLIB segment
    descriptor_id: int  # 0x78 for const, 0x79 for data

    CMDSIZE = 32

    def pack(self) -> bytes:
        return struct.pack('<II', LC_CMD_0x40, self.CMDSIZE) + struct.pack(
            '<IIQI',
            0x18,           # inner size
            0,              # reserved
            self.vmaddr,    # 64-bit VM address
            self.descriptor_id
        ) + b'\x00' * 4


# =============================================================================
# LC_THREAD Payloads
# =============================================================================

@dataclass
class TDConfig:
    """
    LC_THREAD flavor=4 — Task Descriptor master config.

    This is the largest LC_THREAD (2208 bytes including cmd header).
    Contains VM address mapping, instruction count, and network name.
    """
    text_vmaddr: int = 0x30008000        # __TEXT.__text vmaddr
    const_vmaddr: int = 0x30008140       # __TEXT.__const vmaddr (varies with __text size)
    fvmlib_const_vmaddr: int = 0x30000000
    fvmlib_data_vmaddr: int = 0x30004000
    instruction_word_count: int = 65     # Number of 4-byte words in __text
    network_name: str = "net"
    signature_hash: int = 0x461C4000     # 4-byte value at +0x0878 in payload

    FLAVOR = 4
    COUNT = 0x222       # 546 uint32s = 2184 bytes of state
    CMDSIZE = 2208      # 8 (cmd+size) + 8 (flavor+count) + 2184 + padding
    PAYLOAD_SIZE = 2200  # cmdsize - 8

    # Key offsets within the payload (after flavor+count):
    OFFSETS = {
        0x0008: 'text_vmaddr',          # Q: __TEXT.__text VM address
        0x0018: 'const_vmaddr',         # Q: __TEXT.__const VM address
        0x0068: 'fvmlib_const_vmaddr',  # Q: __FVMLIB const VM address
        0x0078: 'fvmlib_data_vmaddr',   # Q: __FVMLIB data VM address
        0x0808: 'text_vmaddr_repeat',   # Q: __TEXT.__text VM address (repeated)
        0x0818: 'config_type',          # I: always 4
        0x081C: 'instruction_count',    # I: word count in __text (0x41=65, 0x3F=63)
        0x0828: 'flag_1',              # I: always 1
        0x0830: 'neg_mask',            # Q: 0xFFFFFFFFFFFFFFFF
        0x083C: 'mask_16',            # H: 0xFFFF
        0x0840: 'string_table_offset', # I: offset to name strings (0x0898)
        0x0858: 'param_16',           # I: 0x10
        0x0860: 'param_4',            # I: 0x04
        0x0868: 'param_2',            # I: 0x02
        0x0870: 'tag_7',              # I: 7 (tag type)
        0x0874: 'tag_7_size',         # I: 8 (tag size)
        0x0878: 'signature_hash',     # I: signature value (e.g., 0x461C4000)
        0x0880: 'tag_9',              # I: 9 (tag type)
        0x0884: 'tag_9_size',         # I: 8 (tag size)
        0x0890: 'network_name',       # str: "net\0"
    }


@dataclass
class KernelIODescriptor:
    """
    LC_THREAD flavor=3 — Kernel I/O buffer descriptor.

    Two of these exist per HWX: one for input (index=1), one for output (index=2).
    Contains buffer sizes, strides, and tensor names.
    """
    kernel_index: int           # 1 for input, 2 for output
    buffer_size: int = 0x1000   # 4096 bytes
    stride: int = 0x40          # 64 bytes
    num_buffers: int = 2
    network_name: str = "net"
    tensor_names: List[str] = field(default_factory=list)  # ["x"] or ["y@output", "y"]

    FLAVOR = 3
    COUNT = 0x34A  # 842 uint32s = 3368 bytes of state
    # cmdsize varies slightly: 3392 for index=1, 3400 for index=2
    # (due to longer tensor name strings)

    OFFSETS = {
        0x0008: 'inner_flavor',     # I: always 3
        0x000C: 'kernel_index',     # I: 1 or 2
        0x0010: 'td_offset_1',      # I: 0x0D38
        0x0018: 'td_offset_2',      # I: 0x0D3C
        0x001C: 'flags',            # I: 5
        0x0020: 'batch_size',       # I: 1
        0x0024: 'stride_param',     # I: 0x40
        0x0028: 'dim_x',           # I: 1
        0x002C: 'dim_y',           # I: 1
        0x0030: 'dim_z',           # I: 1
        0x0048: 'buffer_size',     # Q: 0x1000
        0x0050: 'stride_1',       # Q: 0x40
        0x0058: 'stride_2',       # Q: 0x40
        0x0060: 'num_buffers',    # Q: 2
        0x0068: 'buf_size_2',     # Q: 0x1000
        0x0070: 'index_flag',     # I: 1
        0x0074: 'td_offset_3',    # I: varies (0x0D3E for idx=1, 0x0D45 for idx=2)
        0x0078: 'buf_size_3',     # Q: 0x1000
        0x0D30: 'string_table',   # str: "net\0" + tensor names
    }


# =============================================================================
# Compiler Metadata (LC_CMD_0x08)
# =============================================================================

@dataclass
class CompilerMetadata:
    """
    LC_CMD_0x08 — Compiler metadata string.

    Contains ANEC version, compiler version, module info, compilation flags,
    and input/output file paths. The path hashes change per compilation but
    are not functionally relevant.
    """
    CMDSIZE = 2352  # 0x0930

    anec_version: str = "ANEC v1"
    compiler_version: str = "zin_ane_compiler v9.202.0"
    module_version: str = "3515.2.4.14.1"
    target: str = "h17g"

    # Standard compiler flags (all observed values)
    FLAGS = {
        'fno-fold-scale': 'true',
        'fdram-allocator': 'ffreuse',
        'fdram-tensor-priority': 'sizebyliverange',
        'fl2-allocator': 'ffreuse',
        'fl3-allocator': 'ffreuse',
        'fl2-cache-mode': 'resident',
        'fsignature': 'ident',
        'fdisable-bonded-networks': 'true',
        'memcache-size': '4194304',
        'fspatial-split': 'disabled',
        'fenable-circular-buffer-in-spatial-split': '-1',
        'fkernel-rewind': 'enabled',
        'foptimize-ne-utilization': 'true',
        'disable-cache-prefetch-mask': '1',
        'optimize-mutable-kernel-section': 'true',
        'split-kernel-section': 'true',
        'max-kernel-section-size': '134217728',
        'enable-function-inlining': 'true',
        'enable-afm-mlir-features': 'true',
        'enable-l2-batch-splitting': 'true',
        'enable-global-cw-optimization': 'true',
        'enable-l2-cached-buffer': 'true',
        'Wl-undefined': 'fvmlib',
    }


# =============================================================================
# Symbol Table (LC_CMD_0x02)
# =============================================================================

@dataclass
class SymbolTableCommand:
    """LC_CMD_0x02 — Symbol table / string table offsets."""
    string_table_offset: int = 0x2F30
    string_table_size: int = 27  # 0x1B
    secondary_offset: int = 0x30E0
    secondary_size: int = 651  # 0x28B

    CMDSIZE = 24  # 0x18

    def pack(self) -> bytes:
        return struct.pack('<IIIIII',
            LC_CMD_0x02, self.CMDSIZE,
            self.string_table_offset, self.string_table_size,
            self.secondary_offset, self.secondary_size)


# =============================================================================
# __TEXT.__text Instruction Format
# =============================================================================

# Known operation opcodes (high 16 bits of Word[19])
ANE_OPCODES = {
    0x9361: 'relu',
    0x9541: 'abs',
    # More to be discovered by compiling additional operations
}

# Reverse map
ANE_OPCODE_BY_NAME = {v: k for k, v in ANE_OPCODES.items()}

@dataclass
class KernelProgram:
    """
    The __TEXT.__text section: a sequence of 4-byte little-endian instruction words.

    Header layout (words 0-18 are common to all observed programs):
    """
    # Fixed header words
    WORD_0_VERSION = 0           # Word[0]: always 1
    WORD_4_PROGRAM_SIZE = 4      # Word[4]: program size indicator (0x3D for relu, 0x3B for abs)
    WORD_6_BUFFER_CONFIG = 6     # Word[6]: buffer config (0x04000068)
    WORD_8_NEG_OFFSET = 8        # Word[8]: negative offset (0x00FFF868)
    WORD_12_STAGE_FLAGS = 12     # Word[12]: stage enable flags
    WORD_13_PIPELINE = 13        # Word[13]: pipeline config (0x1540)
    WORD_14_TILE_SIZE = 14       # Word[14]: tile size (0x80 = 128)
    WORD_15_DIMS = 15            # Word[15]: dimensions (0x10001 = 1x1)
    WORD_19_OPCODE = 19          # Word[19]: OPERATION OPCODE (critical)
    WORD_23_EXTRA_CONFIG = 23    # Word[23]: extra pipeline config (varies)

    # Field definitions for header region
    HEADER_FIELDS = {
        0: ('version', 'Program version, always 1'),
        4: ('program_size', 'Number of pipeline entries (e.g., 0x3D=61, 0x3B=59)'),
        6: ('buffer_config', 'Buffer configuration word'),
        8: ('neg_offset', 'Complement addressing offset'),
        12: ('stage_flags', 'Stage enable flags (high half-word = operation-specific)'),
        13: ('pipeline_config', 'Pipeline configuration'),
        14: ('tile_size', 'Processing tile size (0x80=128)'),
        15: ('dimensions', 'Packed dimensions (high16=H, low16=W)'),
        16: ('batch_count', 'Batch count'),
        18: ('stride', 'Processing stride (0x40=64)'),
        19: ('opcode', 'NE operation opcode (high 16 bits) + flags (low 16 bits)'),
        22: ('stride_2', 'Secondary stride'),
        23: ('extra_config', 'Additional pipeline config (operation-dependent)'),
    }


# =============================================================================
# Complete File Layout
# =============================================================================

# Standard segment definitions for a simple single-op kernel
STANDARD_SEGMENTS = [
    SegmentCommand(
        segname='__PAGEZERO',
        vmaddr=0x0000000000000000,
        vmsize=0x4000,
        fileoff=0, filesize=0,
        maxprot=0, initprot=0, nsects=0, flags=0x4
    ),
    SegmentCommand(
        segname='__FVMLIB',
        vmaddr=0x30000000,
        vmsize=0x4000,
        fileoff=0, filesize=0,
        maxprot=1, initprot=1, nsects=1, flags=0x6,
        sections=[SectionHeader(
            sectname='__const', segname='__FVMLIB',
            addr=0x30000000, size=0x1000, fileoff=0,
            align=14, flags=0x21
        )]
    ),
    SegmentCommand(
        segname='__FVMLIB',
        vmaddr=0x30004000,
        vmsize=0x4000,
        fileoff=0, filesize=0,
        maxprot=2, initprot=2, nsects=1, flags=0x6,
        sections=[SectionHeader(
            sectname='__data', segname='__FVMLIB',
            addr=0x30004000, size=0x1000, fileoff=0,
            align=14, flags=0x23
        )]
    ),
]

# File layout offsets for standard 49152-byte HWX
FILE_LAYOUT = {
    'header':           (0x0000, 0x0020),
    'lc0_pagezero':     (0x0020, 0x0068),
    'lc1_fvmlib_const': (0x0068, 0x0100),
    'lc2_fvmlib_data':  (0x0100, 0x0198),
    'lc3_text_segment': (0x0198, 0x0280),
    'lc4_fvmlib_ref_0': (0x0280, 0x02A0),
    'lc5_fvmlib_ref_1': (0x02A0, 0x02C0),
    'lc6_td_config':    (0x02C0, 0x0B60),
    'lc7_kernel_io_0':  (0x0B60, 0x18A0),
    'lc8_kernel_io_1':  (0x18A0, 0x25E8),
    'lc9_metadata':     (0x25E8, 0x2F18),
    'lc10_symtab':      (0x2F18, 0x2F30),
    'zero_padding':     (0x2F30, 0x4000),
    'text_section':     (0x4000, None),    # size varies
    'const_section':    (None, None),       # position depends on __text size
}

# Critical patch points for changing operation type
# (file offsets assuming relu-sized __text at 0x4000)
PATCH_POINTS = {
    'opcode_word':      0x404C,  # Word[19]: operation opcode (4 bytes)
    'stage_flags':      0x4030,  # Word[12]: stage enable flags (4 bytes)
    'program_size':     0x4010,  # Word[4]: program size indicator (4 bytes)
    'extra_config':     0x405C,  # Word[23]: extra pipeline config (4 bytes)
    'text_sect_size':   0x0208,  # __text section size in LC[3] (4 bytes, actually 8-byte field)
    'const_vmaddr':     0x0250,  # __const vmaddr in LC[3] section header
    'const_fileoff':    0x0260,  # __const fileoff in LC[3] section header
    'td_const_vmaddr':  0x02E0,  # __const vmaddr in LC_THREAD[0] payload
    'td_instr_count':   0x0AE4,  # Instruction word count in LC_THREAD[0]
}

# Known operation configurations
# Each entry: (opcode_word19, stage_flags_word12, program_size_word4, extra_config_word23, num_words)
OPERATION_CONFIGS = {
    'relu': {
        'opcode': 0x93618005,
        'stage_flags': 0x00050009,
        'program_size': 0x003D0000,
        'extra_config': 0x00014000,
        'num_words': 65,
    },
    'abs': {
        'opcode': 0x95418005,
        'stage_flags': 0x00000009,
        'program_size': 0x003B0000,
        'extra_config': 0x00000001,
        'num_words': 63,
    },
}


# =============================================================================
# Utility Functions
# =============================================================================

def parse_hwx_header(data: bytes) -> dict:
    """Parse the Mach-O header from HWX binary data."""
    hdr = MachOHeader.unpack(data)
    assert hdr.magic == BEEFFACE_MAGIC, f"Bad magic: 0x{hdr.magic:08X}"
    assert hdr.cputype == CPU_TYPE_ANE, f"Not ANE: cputype={hdr.cputype}"
    return {
        'magic': hdr.magic,
        'cputype': hdr.cputype,
        'cpusubtype': hdr.cpusubtype,
        'filetype': hdr.filetype,
        'ncmds': hdr.ncmds,
        'sizeofcmds': hdr.sizeofcmds,
        'flags': hdr.flags,
    }


def parse_load_commands(data: bytes) -> list:
    """Parse all load commands from HWX binary data."""
    hdr = MachOHeader.unpack(data)
    commands = []
    offset = MachOHeader.SIZE

    for i in range(hdr.ncmds):
        cmd, cmdsize = struct.unpack_from('<II', data, offset)
        lc = {
            'index': i,
            'cmd': cmd,
            'cmdsize': cmdsize,
            'file_offset': offset,
        }

        if cmd == LC_SEGMENT_64:
            segname = data[offset+8:offset+24].split(b'\x00')[0].decode('ascii')
            vmaddr, vmsize, fileoff, filesize = struct.unpack_from('<QQQQ', data, offset+24)
            maxprot, initprot, nsects, flags = struct.unpack_from('<IIII', data, offset+56)
            lc['type'] = 'LC_SEGMENT_64'
            lc['segname'] = segname
            lc['vmaddr'] = vmaddr
            lc['vmsize'] = vmsize
            lc['fileoff'] = fileoff
            lc['filesize'] = filesize
            lc['nsects'] = nsects

        elif cmd == LC_THREAD:
            flavor, count = struct.unpack_from('<II', data, offset+8)
            lc['type'] = 'LC_THREAD'
            lc['flavor'] = flavor
            lc['count'] = count
            if flavor == 3:
                inner_flavor, kernel_index = struct.unpack_from('<II', data, offset+16)
                lc['kernel_index'] = kernel_index

        elif cmd == LC_CMD_0x08:
            payload = data[offset+8:offset+cmdsize]
            null_pos = payload.find(b'\x00')
            lc['type'] = 'LC_CMD_0x08_METADATA'
            lc['text'] = payload[:null_pos].decode('ascii', errors='replace') if null_pos > 0 else ''

        elif cmd == LC_CMD_0x40:
            lc['type'] = 'LC_CMD_0x40_FVMLIB_REF'
            vmaddr = struct.unpack_from('<Q', data, offset+16)[0]
            desc_id = struct.unpack_from('<I', data, offset+24)[0]
            lc['ref_vmaddr'] = vmaddr
            lc['descriptor_id'] = desc_id

        elif cmd == LC_CMD_0x02:
            lc['type'] = 'LC_CMD_0x02_SYMTAB'

        commands.append(lc)
        offset += cmdsize

    return commands


def get_text_section_info(data: bytes) -> dict:
    """Extract __TEXT.__text section info from the binary."""
    # __text section header starts at 0x01E0 (first section in LC[3])
    # Section size field is at +40 bytes into the section header = 0x01E0 + 40 = 0x0208
    text_size = struct.unpack_from('<Q', data, 0x0208)[0]
    # __text starts at file offset 0x4000
    return {
        'file_offset': 0x4000,
        'size': text_size,
        'num_words': text_size // 4,
    }


def read_instruction_words(data: bytes) -> list:
    """Read all instruction words from __TEXT.__text section."""
    info = get_text_section_info(data)
    words = []
    for i in range(info['num_words']):
        w = struct.unpack_from('<I', data, info['file_offset'] + i * 4)[0]
        words.append(w)
    return words


def identify_operation(data: bytes) -> str:
    """Identify the ANE operation from the opcode word."""
    words = read_instruction_words(data)
    if len(words) > 19:
        opcode_word = words[19]
        opcode = opcode_word >> 16
        for name, code in ANE_OPCODE_BY_NAME.items():
            if code == opcode:
                return name
    return 'unknown'


def patch_operation(data: bytearray, target_op: str) -> bytearray:
    """
    Patch an HWX binary to change its operation type.

    For now, only supports patching between operations with the same
    instruction count (e.g., if a future pair has equal num_words).

    For operations with different instruction counts (like relu->abs),
    the full instruction stream must be replaced and headers updated.
    """
    if target_op not in OPERATION_CONFIGS:
        raise ValueError(f"Unknown operation: {target_op}. Known: {list(OPERATION_CONFIGS.keys())}")

    config = OPERATION_CONFIGS[target_op]

    # Patch opcode word
    struct.pack_into('<I', data, PATCH_POINTS['opcode_word'], config['opcode'])
    # Patch stage flags
    struct.pack_into('<I', data, PATCH_POINTS['stage_flags'], config['stage_flags'])
    # Patch program size
    struct.pack_into('<I', data, PATCH_POINTS['program_size'], config['program_size'])
    # Patch extra config
    struct.pack_into('<I', data, PATCH_POINTS['extra_config'], config['extra_config'])

    # Update __text section size (at section header offset 0x0208)
    struct.pack_into('<Q', data, 0x0208, config['num_words'] * 4)

    # Update __const vmaddr and fileoff based on new __text size
    text_end = 0x4000 + config['num_words'] * 4
    # Align to 0x40 boundary (align=6 means 2^6 = 64 byte alignment)
    const_file_offset = (text_end + 0x3F) & ~0x3F
    const_vmaddr = 0x30008000 + (const_file_offset - 0x4000)

    # __const section vmaddr in LC[3] (at 0x0250)
    struct.pack_into('<Q', data, 0x0250, const_vmaddr)
    # __const section fileoff in LC[3] (at 0x0260)
    struct.pack_into('<I', data, 0x0260, const_file_offset)

    # TD config: __const vmaddr reference
    struct.pack_into('<Q', data, 0x02E0, const_vmaddr)
    # TD config: also at +0x0018 in payload
    # (0x02C8 + 0x18 = 0x02E0, already handled above)

    # TD config: instruction word count
    struct.pack_into('<I', data, PATCH_POINTS['td_instr_count'], config['num_words'])

    # Also update the __const vmaddr in LC_THREAD[0] second reference
    struct.pack_into('<Q', data, 0x02C8 + 0x18, const_vmaddr)

    return data


def diff_hwx_files(data_a: bytes, data_b: bytes) -> list:
    """Byte-level diff between two HWX files. Returns list of (offset, val_a, val_b)."""
    assert len(data_a) == len(data_b), "Files must be same size"
    return [(i, data_a[i], data_b[i]) for i in range(len(data_a)) if data_a[i] != data_b[i]]


# =============================================================================
# Main: Self-test
# =============================================================================

if __name__ == '__main__':
    import sys
    import os

    hwx_dir = os.path.join(os.path.dirname(__file__), 'hwx_cache')

    for name in ['relu_49152.hwx', 'abs_49152.hwx']:
        path = os.path.join(hwx_dir, name)
        if not os.path.exists(path):
            print(f"[!] {path} not found, skipping self-test")
            continue

        with open(path, 'rb') as f:
            data = f.read()

        print(f"\n=== {name} ===")
        hdr = parse_hwx_header(data)
        print(f"  Magic: 0x{hdr['magic']:08X}, CPU: 0x{hdr['cputype']:02X}, "
              f"Sub: {hdr['cpusubtype']}, Cmds: {hdr['ncmds']}")

        info = get_text_section_info(data)
        print(f"  __text: {info['size']} bytes, {info['num_words']} words")

        op = identify_operation(data)
        print(f"  Operation: {op}")

        words = read_instruction_words(data)
        print(f"  Opcode word[19]: 0x{words[19]:08X}")
        print(f"  Stage flags[12]: 0x{words[12]:08X}")
        print(f"  Program size[4]: 0x{words[4]:08X}")

    # Cross-diff
    relu_path = os.path.join(hwx_dir, 'relu_49152.hwx')
    abs_path = os.path.join(hwx_dir, 'abs_49152.hwx')
    if os.path.exists(relu_path) and os.path.exists(abs_path):
        with open(relu_path, 'rb') as f:
            relu = f.read()
        with open(abs_path, 'rb') as f:
            abso = f.read()

        diffs = diff_hwx_files(relu, abso)
        print(f"\n=== relu vs abs: {len(diffs)} byte differences ===")
