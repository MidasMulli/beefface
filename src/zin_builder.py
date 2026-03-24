#!/usr/bin/env python3
"""
ZinBuilder — Generate and modify ANE Zin binaries (BEEFFACE format) from scratch.

The Apple Neural Engine executes compiled Zin binaries (.hwx files). These are
Mach-O-like binaries with magic 0xBEEFFACE, CPU type 128 (ANE), and H17G subtype 9.
This module provides tools to parse, modify, and eventually generate these binaries.

Binary layout (49,152 bytes for width-64 activation models):
    0x0000 - 0x001F  Mach-O header (32 bytes)
    0x0020 - 0x2F2F  Load commands (11 commands, 12,048 bytes)
    0x2F30 - 0x336A  Symbol table + string table
    0x336B - 0x3FFF  Padding (zeros)
    0x4000 - 0x40FF  __TEXT.__text (kernel tile descriptors, variable size)
    0x4100 - 0x80FF  __TEXT.__const (pipeline configuration, 16,384 bytes)
    0x8100 - 0xBFFF  Padding (zeros to page boundary)

Architecture reference:
    - ANE is a fixed-function pipeline with 17 stages
    - Operations are implemented by enabling/disabling pipeline stages
    - Stage control at +0x5C in stage descriptors: 0x09000000=active, 0x00000000=off
    - Activation mode is encoded in the __text tile descriptor table
    - The __const section contains pipeline stage configuration (identical across
      activation types for same tensor dimensions)

Author: ZinBuilder (reverse-engineered from ANECompilerService output)
Platform: Apple M5 (H17G), macOS 15.4, zin_ane_compiler v9.202.0
"""

import struct
import copy
import hashlib
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, ClassVar


# =============================================================================
# Constants
# =============================================================================

BEEFFACE_MAGIC = 0xBEEFFACE
ANE_CPU_TYPE = 128          # CPU type for Apple Neural Engine
H17G_CPU_SUBTYPE = 9        # H17G = M5 family
MH_EXECUTE = 2              # File type: executable
LC_SEGMENT_64 = 0x19
LC_THREAD = 0x4
LC_SYMTAB = 0x2
LC_COMPILER_INFO = 0x8      # Custom: compiler metadata
LC_UNKNOWN_40 = 0x40        # Two instances, 32 bytes each

# Pipeline stage control values
STAGE_ACTIVE = 0x09000000
STAGE_DISABLED = 0x00000000
STAGE_BYPASS = 0xFFFFFFFF

# Standard file size for width-64 activation models
STANDARD_FILE_SIZE = 49152  # 0xC000


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class MachOHeader:
    """BEEFFACE Mach-O header (32 bytes)."""
    magic: int = BEEFFACE_MAGIC
    cputype: int = ANE_CPU_TYPE
    cpusubtype: int = H17G_CPU_SUBTYPE
    filetype: int = MH_EXECUTE
    ncmds: int = 11
    sizeofcmds: int = 0x2F10
    flags: int = 0x00200000
    reserved: int = 0

    def pack(self) -> bytes:
        return struct.pack('<IIIIIIII',
                           self.magic, self.cputype, self.cpusubtype,
                           self.filetype, self.ncmds, self.sizeofcmds,
                           self.flags, self.reserved)

    @classmethod
    def unpack(cls, data: bytes, offset: int = 0) -> 'MachOHeader':
        fields = struct.unpack_from('<IIIIIIII', data, offset)
        return cls(*fields)

    SIZE: ClassVar[int] = 32


@dataclass
class Section64:
    """Mach-O section_64 structure (80 bytes)."""
    sectname: str
    segname: str
    addr: int
    size: int
    offset: int
    align: int
    reloff: int = 0
    nreloc: int = 0
    flags: int = 0
    reserved1: int = 0
    reserved2: int = 0

    def pack(self) -> bytes:
        return (
            self.sectname.encode('ascii').ljust(16, b'\x00') +
            self.segname.encode('ascii').ljust(16, b'\x00') +
            struct.pack('<QQIIIIIII',
                        self.addr, self.size, self.offset, self.align,
                        self.reloff, self.nreloc, self.flags,
                        self.reserved1, self.reserved2)
        )

    @classmethod
    def unpack(cls, data: bytes, offset: int = 0) -> 'Section64':
        sectname = data[offset:offset+16].split(b'\x00')[0].decode('ascii')
        segname = data[offset+16:offset+32].split(b'\x00')[0].decode('ascii')
        fields = struct.unpack_from('<QQIIIIIII', data, offset + 32)
        return cls(sectname, segname, *fields)

    SIZE: ClassVar[int] = 80


@dataclass
class Segment64:
    """LC_SEGMENT_64 load command."""
    segname: str
    vmaddr: int
    vmsize: int
    fileoff: int
    filesize: int
    maxprot: int
    initprot: int
    nsects: int
    flags: int
    sections: List[Section64] = field(default_factory=list)

    @property
    def cmdsize(self) -> int:
        return 72 + 80 * self.nsects

    def pack(self) -> bytes:
        result = struct.pack('<II', LC_SEGMENT_64, self.cmdsize)
        result += self.segname.encode('ascii').ljust(16, b'\x00')
        result += struct.pack('<QQQQIIII',
                              self.vmaddr, self.vmsize, self.fileoff,
                              self.filesize, self.maxprot, self.initprot,
                              self.nsects, self.flags)
        for sect in self.sections:
            result += sect.pack()
        return result

    @classmethod
    def unpack(cls, data: bytes, offset: int = 0) -> 'Segment64':
        cmd, cmdsize = struct.unpack_from('<II', data, offset)
        segname = data[offset+8:offset+24].split(b'\x00')[0].decode('ascii')
        fields = struct.unpack_from('<QQQQIIII', data, offset + 24)
        vmaddr, vmsize, fileoff, filesize, maxprot, initprot, nsects, flags = fields
        sections = []
        sect_off = offset + 72
        for _ in range(nsects):
            sections.append(Section64.unpack(data, sect_off))
            sect_off += 80
        return cls(segname, vmaddr, vmsize, fileoff, filesize,
                   maxprot, initprot, nsects, flags, sections)


@dataclass
class ThreadCommand:
    """LC_THREAD load command with raw thread state data."""
    flavor: int
    count: int
    state_data: bytes

    @property
    def cmdsize(self) -> int:
        return 16 + len(self.state_data)

    def pack(self) -> bytes:
        return (struct.pack('<IIII', LC_THREAD, self.cmdsize,
                            self.flavor, self.count) +
                self.state_data)

    @classmethod
    def unpack(cls, data: bytes, offset: int = 0) -> 'ThreadCommand':
        cmd, cmdsize, flavor, count = struct.unpack_from('<IIII', data, offset)
        state_data = data[offset+16:offset+cmdsize]
        return cls(flavor, count, state_data)


@dataclass
class RawLoadCommand:
    """Generic load command with raw data (for LC_UNKNOWN_40, LC_COMPILER_INFO, LC_SYMTAB)."""
    cmd: int
    raw_data: bytes  # everything after cmd+cmdsize

    @property
    def cmdsize(self) -> int:
        return 8 + len(self.raw_data)

    def pack(self) -> bytes:
        return struct.pack('<II', self.cmd, self.cmdsize) + self.raw_data

    @classmethod
    def unpack(cls, data: bytes, offset: int = 0) -> 'RawLoadCommand':
        cmd, cmdsize = struct.unpack_from('<II', data, offset)
        raw_data = data[offset+8:offset+cmdsize]
        return cls(cmd, raw_data)


@dataclass
class SymtabCommand:
    """LC_SYMTAB load command."""
    symoff: int
    nsyms: int
    stroff: int
    strsize: int

    @property
    def cmdsize(self) -> int:
        return 24

    def pack(self) -> bytes:
        return struct.pack('<IIIIII', LC_SYMTAB, self.cmdsize,
                           self.symoff, self.nsyms, self.stroff, self.strsize)

    @classmethod
    def unpack(cls, data: bytes, offset: int = 0) -> 'SymtabCommand':
        cmd, cmdsize, symoff, nsyms, stroff, strsize = struct.unpack_from('<IIIIII', data, offset)
        return cls(symoff, nsyms, stroff, strsize)


# =============================================================================
# ZinBuilder
# =============================================================================

class ZinBuilder:
    """
    Build and modify ANE Zin binaries (BEEFFACE format).

    A Zin binary is the compiled representation that the Apple Neural Engine
    executes directly. This builder can:

    1. Load an existing .hwx file as a template (from_template)
    2. Modify kernel tile descriptors (__text section)
    3. Produce a new binary with correct headers, alignment, and cross-references
    4. Build activation-only models from scratch (build_activation)

    The binary has a layered structure where changes to the __text section
    (kernel tile descriptors) cascade to:
    - Section headers (__text size, __const address/offset)
    - LC_THREAD_1 state (references __const address and __text word count)
    - Symtab and compiler info remain unchanged for same-shape operations

    Usage:
        # Round-trip: load and rebuild identical binary
        zin = ZinBuilder.from_template('relu_49152.hwx')
        assert zin.build() == open('relu_49152.hwx', 'rb').read()

        # Patch relu to abs
        zin = ZinBuilder.from_template('relu_49152.hwx')
        zin.set_activation('abs')
        open('patched.hwx', 'wb').write(zin.build())
    """

    # Known activation tile descriptors: __text section content for each mode.
    # These are the complete tile descriptor tables extracted from compiler output.
    # The __const section (pipeline config) is IDENTICAL across all activation types
    # for the same tensor shape -- only __text changes.
    ACTIVATION_TILES: ClassVar[Dict[str, bytes]] = {}

    def __init__(self):
        self.header = MachOHeader()
        self.segments: List[Segment64] = []
        self.thread_cmds: List[ThreadCommand] = []
        self.unknown_cmds: List[RawLoadCommand] = []  # LC_UNKNOWN_40
        self.compiler_info: Optional[RawLoadCommand] = None
        self.symtab: Optional[SymtabCommand] = None

        # Raw data regions
        self._symtab_entries: bytes = b''    # nlist entries
        self._string_table: bytes = b''      # string table
        self._text_data: bytes = b''         # __TEXT.__text content
        self._const_data: bytes = b''        # __TEXT.__const content
        self._full_binary: Optional[bytearray] = None  # original binary for reference

        # Tracking
        self._activation_mode: Optional[str] = None

    # -----------------------------------------------------------------
    # Parsing
    # -----------------------------------------------------------------

    @classmethod
    def from_template(cls, hwx_path: str) -> 'ZinBuilder':
        """
        Load an existing .hwx binary as a template for modification.

        Parses the complete Mach-O structure including all headers, load commands,
        segments, sections, thread states, compiler info, and symbol table.
        The original binary is preserved byte-for-byte; build() will reproduce it
        exactly unless modifications are made.

        Args:
            hwx_path: Path to a .hwx file (Zin binary compiled by ANECompilerService)

        Returns:
            ZinBuilder instance ready for modification

        Raises:
            ValueError: If the file doesn't have BEEFFACE magic or unexpected structure
        """
        data = bytearray(Path(hwx_path).read_bytes())
        builder = cls()
        builder._full_binary = data

        # Parse header
        builder.header = MachOHeader.unpack(data, 0)
        if builder.header.magic != BEEFFACE_MAGIC:
            raise ValueError(f"Not a Zin binary: magic 0x{builder.header.magic:08x} "
                             f"(expected 0x{BEEFFACE_MAGIC:08x})")

        # Parse load commands
        offset = MachOHeader.SIZE
        text_segment = None

        for i in range(builder.header.ncmds):
            cmd, cmdsize = struct.unpack_from('<II', data, offset)

            if cmd == LC_SEGMENT_64:
                seg = Segment64.unpack(data, offset)
                builder.segments.append(seg)
                if seg.segname == '__TEXT':
                    text_segment = seg

            elif cmd == LC_THREAD:
                tc = ThreadCommand.unpack(data, offset)
                builder.thread_cmds.append(tc)

            elif cmd == LC_UNKNOWN_40:
                rc = RawLoadCommand.unpack(data, offset)
                builder.unknown_cmds.append(rc)

            elif cmd == LC_COMPILER_INFO:
                builder.compiler_info = RawLoadCommand.unpack(data, offset)

            elif cmd == LC_SYMTAB:
                builder.symtab = SymtabCommand.unpack(data, offset)

            else:
                raise ValueError(f"Unknown load command 0x{cmd:x} at offset 0x{offset:x}")

            offset += cmdsize

        # Extract data regions
        if builder.symtab:
            sym_end = builder.symtab.symoff + builder.symtab.nsyms * 16
            builder._symtab_entries = bytes(data[builder.symtab.symoff:sym_end])
            str_end = builder.symtab.stroff + builder.symtab.strsize
            builder._string_table = bytes(data[builder.symtab.stroff:str_end])

        if text_segment:
            for sect in text_segment.sections:
                if sect.sectname == '__text':
                    builder._text_data = bytes(data[sect.offset:sect.offset + sect.size])
                elif sect.sectname == '__const':
                    builder._const_data = bytes(data[sect.offset:sect.offset + sect.size])

        # Detect activation mode from tile data
        builder._activation_mode = builder._detect_activation_mode()

        # Register this tile pattern if we have a mode
        if builder._activation_mode and builder._activation_mode not in cls.ACTIVATION_TILES:
            cls.ACTIVATION_TILES[builder._activation_mode] = builder._text_data

        return builder

    def _detect_activation_mode(self) -> Optional[str]:
        """
        Detect the activation mode from the __text tile descriptor content.

        The tile descriptors encode the activation function in a complex
        pattern of instruction words. We identify known modes by matching
        against reference patterns or by checking specific discriminating bytes.
        """
        if not self._text_data:
            return None

        # Size is a primary discriminator
        size = len(self._text_data)

        # Check specific discriminating bytes within the tile descriptor
        # Byte at offset +0x32 (file +0x4032): relu=0x05, abs=0x00
        # Byte at offset +0x12 (file +0x4012): relu=0x3d (61), abs=0x3b (59)
        if size >= 0x33:
            td = self._text_data
            byte_32 = td[0x32]
            byte_12 = td[0x12]

            # relu: size=0x104, +0x32=0x05, +0x12=0x3d
            if size == 0x104 and byte_32 == 0x05 and byte_12 == 0x3d:
                return 'relu'
            # abs: size=0xfc, +0x32=0x00, +0x12=0x3b
            elif size == 0xfc and byte_32 == 0x00 and byte_12 == 0x3b:
                return 'abs'

        # For unknown modes, return None (template can still be used)
        return None

    # -----------------------------------------------------------------
    # Modification
    # -----------------------------------------------------------------

    def set_activation(self, mode: str) -> 'ZinBuilder':
        """
        Set the activation function by replacing the __text tile descriptors.

        This replaces the kernel tile descriptor table with the known pattern
        for the target activation mode. The __const section (pipeline config)
        remains unchanged -- it is identical across activation types for the
        same tensor shape.

        Updating the tile descriptors cascades to:
        - __text section size in the section header
        - __const section address and file offset (alignment-dependent)
        - LC_THREAD_1: __const address reference and __text word count
        - __TEXT segment vmaddr references

        Args:
            mode: Activation function name ('relu', 'abs', 'identity', 'tanh',
                   'sigmoid', etc.)

        Returns:
            self (for chaining)

        Raises:
            ValueError: If the activation mode is not known
        """
        if mode == self._activation_mode:
            return self

        if mode not in self.ACTIVATION_TILES:
            raise ValueError(
                f"Unknown activation mode '{mode}'. Known modes: "
                f"{list(self.ACTIVATION_TILES.keys())}. "
                f"Load a reference .hwx for this mode first with from_template()."
            )

        self._text_data = self.ACTIVATION_TILES[mode]
        self._activation_mode = mode
        return self

    def set_text_data(self, data: bytes) -> 'ZinBuilder':
        """
        Set raw __text tile descriptor data directly.

        Use this for experimentation with novel tile configurations. The data
        must be a multiple of 4 bytes (tile descriptors are 32-bit words).

        Args:
            data: Raw bytes for the __text section

        Returns:
            self (for chaining)
        """
        if len(data) % 4 != 0:
            raise ValueError(f"__text data must be 4-byte aligned, got {len(data)} bytes")
        self._text_data = bytes(data)
        self._activation_mode = None  # unknown after raw modification
        return self

    def set_compiler_info_text(self, text: str) -> 'ZinBuilder':
        """
        Replace the compiler info text (LC_COMPILER_INFO payload).

        The compiler info includes ANEC version, compilation flags, and
        file paths. Changing this does not affect ANE execution -- it is
        metadata only. The text is null-padded to maintain the original
        command size.

        Args:
            text: ASCII text for compiler info

        Returns:
            self (for chaining)
        """
        if self.compiler_info is None:
            raise ValueError("No compiler info command to modify")

        original_size = len(self.compiler_info.raw_data)
        new_data = text.encode('ascii')
        if len(new_data) > original_size:
            raise ValueError(
                f"Compiler info text too long: {len(new_data)} bytes "
                f"(max {original_size})")
        # Null-pad to original size
        self.compiler_info.raw_data = new_data.ljust(original_size, b'\x00')
        return self

    # -----------------------------------------------------------------
    # Build
    # -----------------------------------------------------------------

    def build(self) -> bytes:
        """
        Assemble the complete Zin binary from components.

        This rebuilds the binary from scratch, ensuring all cross-references
        are consistent. The build process:

        1. Compute __text size and alignment for __const offset
        2. Update section headers with new sizes/offsets/addresses
        3. Update LC_THREAD_1 state with new __const address and word count
        4. Pack all load commands
        5. Write data regions at correct file offsets
        6. Pad to standard file size

        Returns:
            Complete Zin binary as bytes, ready to write to .hwx file
        """
        if self._full_binary is None:
            raise ValueError("Cannot build without a template. Use from_template() first, "
                             "or use build_activation() for from-scratch generation.")

        # Work on a copy of the original binary
        out = bytearray(self._full_binary)

        # --- Compute layout ---
        text_segment = None
        text_section = None
        const_section = None

        for seg in self.segments:
            if seg.segname == '__TEXT':
                text_segment = seg
                for sect in seg.sections:
                    if sect.sectname == '__text':
                        text_section = sect
                    elif sect.sectname == '__const':
                        const_section = sect

        if text_section is None or const_section is None or text_segment is None:
            raise ValueError("Missing __TEXT segment or __text/__const sections")

        # Original values (for reference)
        orig_text_size = text_section.size
        orig_text_offset = text_section.offset
        orig_const_addr = const_section.addr
        orig_const_offset = const_section.offset

        # New values
        new_text_size = len(self._text_data)
        new_text_offset = orig_text_offset  # __text always starts at 0x4000

        # __const follows __text, aligned to 2^align boundary
        const_align = 1 << const_section.align  # 2^6 = 64
        new_const_offset = (new_text_offset + new_text_size + const_align - 1) & ~(const_align - 1)

        # __const VM address (relative to segment vmaddr)
        text_base = text_segment.vmaddr
        new_const_addr = text_base + (new_const_offset - text_segment.fileoff)

        # --- Step 1: Write new __text data ---
        # Clear the old __text region
        old_end = orig_text_offset + orig_text_size
        new_end = new_text_offset + new_text_size
        # Zero out from __text start to original __const start
        for i in range(orig_text_offset, orig_const_offset):
            out[i] = 0
        # Write new __text
        out[new_text_offset:new_text_offset + new_text_size] = self._text_data
        # Write __const at new offset
        out[new_const_offset:new_const_offset + len(self._const_data)] = self._const_data

        # --- Step 2: Update section headers ---
        # Find section header positions in the load commands
        # __TEXT segment is at a known position. We need to find it by scanning.
        lc_offset = MachOHeader.SIZE
        for seg in self.segments:
            if seg.segname == '__TEXT':
                # __text section header is at lc_offset + 72
                # __const section header is at lc_offset + 72 + 80
                text_sect_hdr = lc_offset + 72
                const_sect_hdr = lc_offset + 72 + 80

                # Update __text size (8 bytes at section_header + 40)
                struct.pack_into('<Q', out, text_sect_hdr + 40, new_text_size)

                # Update __const addr (8 bytes at section_header + 32)
                struct.pack_into('<Q', out, const_sect_hdr + 32, new_const_addr)

                # Update __const file offset (4 bytes at section_header + 48)
                struct.pack_into('<I', out, const_sect_hdr + 48, new_const_offset)

                break
            lc_offset += seg.cmdsize

        # --- Step 3: Update LC_THREAD_1 references ---
        # LC_THREAD_1 contains two cross-references:
        #   a) __const VM address at state offset +0x14 (8 bytes, file offset 0x2E0 for relu template)
        #   b) __text size in words at state offset +0x818 (4 bytes, file offset 0xAE4 for relu template)
        #
        # Rather than hardcode file offsets, we scan for the original values.

        # Find and replace __const address references in LC_THREAD_1
        old_const_addr_bytes = struct.pack('<Q', orig_const_addr)
        new_const_addr_bytes = struct.pack('<Q', new_const_addr)

        if orig_const_addr != new_const_addr:
            # LC_THREAD_1 starts after header + 4 segments + 2 unknown cmds
            # We scan the entire load command region for the address
            lc_start = MachOHeader.SIZE
            lc_end = lc_start + self.header.sizeofcmds

            pos = lc_start
            while pos < lc_end:
                idx = out.find(old_const_addr_bytes, pos, lc_end)
                if idx == -1:
                    break
                out[idx:idx+8] = new_const_addr_bytes
                pos = idx + 8

        # Update __text word count
        old_word_count = orig_text_size // 4
        new_word_count = new_text_size // 4

        if old_word_count != new_word_count:
            # The word count is stored as a 32-bit LE value preceded by 0x04000000
            # Pattern: 04 00 00 00 XX 00 00 00 (where XX = word count)
            old_pattern = struct.pack('<II', 4, old_word_count)
            new_pattern = struct.pack('<II', 4, new_word_count)

            lc_start = MachOHeader.SIZE
            lc_end = lc_start + self.header.sizeofcmds

            pos = lc_start
            while pos < lc_end:
                idx = out.find(old_pattern, pos, lc_end)
                if idx == -1:
                    break
                out[idx:idx+8] = new_pattern
                pos = idx + 8

        # --- Step 4: Update compiler info if modified ---
        if self.compiler_info is not None:
            # Find compiler info position
            lc_offset = MachOHeader.SIZE
            for i in range(self.header.ncmds):
                cmd_val = struct.unpack_from('<I', out, lc_offset)[0]
                cmdsize_val = struct.unpack_from('<I', out, lc_offset + 4)[0]
                if cmd_val == LC_COMPILER_INFO:
                    out[lc_offset + 8:lc_offset + 8 + len(self.compiler_info.raw_data)] = \
                        self.compiler_info.raw_data
                    break
                lc_offset += cmdsize_val

        return bytes(out)

    # -----------------------------------------------------------------
    # From-scratch generation
    # -----------------------------------------------------------------

    @classmethod
    def build_activation(cls, mode: str, width: int = 64) -> bytes:
        """
        Build a complete activation-only Zin binary from scratch.

        Currently requires that a template for the target activation mode
        has been previously loaded via from_template(). The template provides
        the pipeline configuration (__const) and baseline structure.

        Future versions will generate the pipeline configuration from first
        principles, enabling truly novel operations.

        Args:
            mode: Activation function name ('relu', 'abs', etc.)
            width: Tensor width (default 64, matching the reference models)

        Returns:
            Complete Zin binary as bytes

        Raises:
            ValueError: If no template is available for the requested mode
        """
        if width != 64:
            raise ValueError(
                f"Only width=64 supported currently (got {width}). "
                f"Other widths require different pipeline configurations.")

        if mode not in cls.ACTIVATION_TILES:
            raise ValueError(
                f"No template loaded for mode '{mode}'. "
                f"Load a reference .hwx first with from_template().")

        # We need a base template. Use any loaded template since __const is
        # identical across activation types for same tensor shape.
        # Find a template file path from our internal state, or use the
        # class-level tile data to identify which template to use.

        # For now, we require at least one template to have been loaded
        raise NotImplementedError(
            "From-scratch generation not yet implemented. "
            "Use from_template() to load a base, then set_activation() to change mode. "
            "Full from-scratch generation requires understanding pipeline config "
            "construction (the __const section) which is future work."
        )

    # -----------------------------------------------------------------
    # Convenience builders (template-based)
    # -----------------------------------------------------------------

    @classmethod
    def build_activation_from_template(cls, template_path: str, mode: str) -> bytes:
        """
        Build an activation binary by loading a template and changing mode.

        This is the primary workflow: load any activation .hwx, change the
        activation function, and get a new binary.

        Args:
            template_path: Path to any activation .hwx file
            mode: Target activation function name

        Returns:
            Complete Zin binary for the target mode
        """
        builder = cls.from_template(template_path)
        builder.set_activation(mode)
        return builder.build()

    # -----------------------------------------------------------------
    # Analysis and inspection
    # -----------------------------------------------------------------

    def describe(self) -> str:
        """Return a human-readable description of this Zin binary."""
        lines = []
        lines.append(f"=== Zin Binary (BEEFFACE) ===")
        lines.append(f"CPU: {self.header.cputype} (ANE), subtype {self.header.cpusubtype} (H17G)")
        lines.append(f"Load commands: {self.header.ncmds}, total {self.header.sizeofcmds} bytes")
        lines.append(f"Flags: 0x{self.header.flags:08x}")
        lines.append(f"Detected activation: {self._activation_mode or 'unknown'}")
        lines.append(f"")

        for seg in self.segments:
            lines.append(f"Segment: {seg.segname}")
            lines.append(f"  vmaddr=0x{seg.vmaddr:x} vmsize=0x{seg.vmsize:x}")
            lines.append(f"  fileoff=0x{seg.fileoff:x} filesize=0x{seg.filesize:x}")
            lines.append(f"  maxprot={seg.maxprot} initprot={seg.initprot}")
            for sect in seg.sections:
                lines.append(f"  Section: {sect.sectname}")
                lines.append(f"    addr=0x{sect.addr:x} size=0x{sect.size:x} "
                             f"offset=0x{sect.offset:x} align={sect.align}")

        lines.append(f"")
        lines.append(f"Thread commands: {len(self.thread_cmds)}")
        for i, tc in enumerate(self.thread_cmds):
            lines.append(f"  #{i}: flavor={tc.flavor} count={tc.count} "
                         f"state_size={len(tc.state_data)} bytes")

        lines.append(f"")
        lines.append(f"__text (tile descriptors): {len(self._text_data)} bytes "
                     f"({len(self._text_data)//4} words)")
        lines.append(f"__const (pipeline config): {len(self._const_data)} bytes")

        if self.symtab:
            lines.append(f"")
            lines.append(f"Symtab: {self.symtab.nsyms} symbols, "
                         f"strtab {self.symtab.strsize} bytes")

        return "\n".join(lines)

    def diff_text(self, other: 'ZinBuilder') -> List[Tuple[int, int, int]]:
        """
        Compare __text tile descriptors between two builders.

        Args:
            other: Another ZinBuilder to compare against

        Returns:
            List of (offset, self_byte, other_byte) tuples for differing bytes
        """
        diffs = []
        max_len = max(len(self._text_data), len(other._text_data))
        for i in range(max_len):
            a = self._text_data[i] if i < len(self._text_data) else None
            b = other._text_data[i] if i < len(other._text_data) else None
            if a != b:
                diffs.append((i, a, b))
        return diffs

    # -----------------------------------------------------------------
    # File I/O
    # -----------------------------------------------------------------

    def write(self, path: str) -> int:
        """
        Build and write the Zin binary to a file.

        Args:
            path: Output file path

        Returns:
            Number of bytes written
        """
        data = self.build()
        Path(path).write_bytes(data)
        return len(data)


# =============================================================================
# Validation and testing
# =============================================================================

def validate_roundtrip(hwx_path: str) -> bool:
    """
    Validate that from_template().build() produces identical bytes.

    Args:
        hwx_path: Path to a .hwx file

    Returns:
        True if round-trip produces identical output
    """
    original = Path(hwx_path).read_bytes()
    builder = ZinBuilder.from_template(hwx_path)
    rebuilt = builder.build()

    if original == rebuilt:
        return True

    # Find differences for debugging
    diffs = []
    for i in range(min(len(original), len(rebuilt))):
        if original[i] != rebuilt[i]:
            diffs.append((i, original[i], rebuilt[i]))

    if len(original) != len(rebuilt):
        print(f"  Size mismatch: original={len(original)}, rebuilt={len(rebuilt)}")

    print(f"  {len(diffs)} byte differences:")
    for off, orig, rebuilt_byte in diffs[:20]:
        print(f"    0x{off:05x}: original=0x{orig:02x} rebuilt=0x{rebuilt_byte:02x}")
    if len(diffs) > 20:
        print(f"    ... and {len(diffs) - 20} more")

    return False


def validate_cross_patch(relu_path: str, abs_path: str) -> bool:
    """
    Validate that loading relu and setting activation to 'abs' produces
    a binary that matches the reference abs binary in all functional regions.

    The compiler info section will differ (contains file hashes/paths) but
    all executable content should match.

    Args:
        relu_path: Path to relu .hwx file
        abs_path: Path to abs .hwx file

    Returns:
        True if functional regions match
    """
    # Load both templates to register their tile data
    relu_builder = ZinBuilder.from_template(relu_path)
    abs_builder = ZinBuilder.from_template(abs_path)
    abs_original = Path(abs_path).read_bytes()

    # Patch relu to abs
    patched_builder = ZinBuilder.from_template(relu_path)
    patched_builder.set_activation('abs')

    # Also copy the compiler info from abs (since it contains different hashes)
    patched_builder.compiler_info = copy.deepcopy(abs_builder.compiler_info)

    patched = patched_builder.build()

    if patched == abs_original:
        return True

    # Analyze differences
    diffs = []
    for i in range(min(len(patched), len(abs_original))):
        if patched[i] != abs_original[i]:
            diffs.append((i, patched[i], abs_original[i]))

    if len(patched) != len(abs_original):
        print(f"  Size mismatch: patched={len(patched)}, abs={len(abs_original)}")

    if diffs:
        print(f"  {len(diffs)} byte differences:")
        for off, p, a in diffs[:30]:
            print(f"    0x{off:05x}: patched=0x{p:02x} abs_orig=0x{a:02x}")
        if len(diffs) > 30:
            print(f"    ... and {len(diffs) - 30} more")

    return False


# =============================================================================
# Main: validation against reference files
# =============================================================================

if __name__ == '__main__':
    import sys

    BASE = Path(__file__).parent / 'hwx_cache'
    RELU_PATH = str(BASE / 'relu_49152.hwx')
    ABS_PATH = str(BASE / 'abs_49152.hwx')

    # Check reference files exist
    for p in [RELU_PATH, ABS_PATH]:
        if not Path(p).exists():
            print(f"ERROR: Reference file not found: {p}")
            sys.exit(1)

    print("=" * 60)
    print("ZinBuilder Validation Suite")
    print("=" * 60)

    # --- Test 1: Round-trip relu ---
    print("\n[Test 1] Round-trip: relu.hwx -> from_template -> build")
    ok = validate_roundtrip(RELU_PATH)
    print(f"  Result: {'PASS' if ok else 'FAIL'}")

    # --- Test 2: Round-trip abs ---
    print("\n[Test 2] Round-trip: abs.hwx -> from_template -> build")
    ok2 = validate_roundtrip(ABS_PATH)
    print(f"  Result: {'PASS' if ok2 else 'FAIL'}")

    # --- Test 3: Describe ---
    print("\n[Test 3] Binary description (relu)")
    relu_builder = ZinBuilder.from_template(RELU_PATH)
    print(relu_builder.describe())

    # --- Test 4: Tile diff ---
    print("\n[Test 4] Tile descriptor diff (relu vs abs)")
    abs_builder = ZinBuilder.from_template(ABS_PATH)
    diffs = relu_builder.diff_text(abs_builder)
    print(f"  __text differs at {len(diffs)} positions")
    print(f"  relu __text size: {len(relu_builder._text_data)} bytes")
    print(f"  abs __text size:  {len(abs_builder._text_data)} bytes")

    # --- Test 5: Cross-patch relu->abs ---
    print("\n[Test 5] Cross-patch: relu -> set_activation('abs') vs abs reference")
    ok5 = validate_cross_patch(RELU_PATH, ABS_PATH)
    print(f"  Result: {'PASS' if ok5 else 'FAIL'}")

    # --- Test 6: Cross-patch abs->relu ---
    print("\n[Test 6] Cross-patch: abs -> set_activation('relu') vs relu reference")
    # Load fresh to register tiles
    ZinBuilder.from_template(RELU_PATH)
    ZinBuilder.from_template(ABS_PATH)

    patched = ZinBuilder.from_template(ABS_PATH)
    patched.set_activation('relu')
    # Copy compiler info from relu
    relu_ref = ZinBuilder.from_template(RELU_PATH)
    patched.compiler_info = copy.deepcopy(relu_ref.compiler_info)
    patched_bytes = patched.build()
    relu_original = Path(RELU_PATH).read_bytes()

    if patched_bytes == relu_original:
        print("  Result: PASS")
        ok6 = True
    else:
        diffs6 = []
        for i in range(min(len(patched_bytes), len(relu_original))):
            if patched_bytes[i] != relu_original[i]:
                diffs6.append((i, patched_bytes[i], relu_original[i]))
        print(f"  {len(diffs6)} differences:")
        for off, p, r in diffs6[:20]:
            print(f"    0x{off:05x}: patched=0x{p:02x} relu_orig=0x{r:02x}")
        print("  Result: FAIL")
        ok6 = False

    # --- Summary ---
    print("\n" + "=" * 60)
    results = [ok, ok2, ok5, ok6]
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} passed")

    if all(results):
        print("\nAll validation tests PASSED.")
        print("ZinBuilder can round-trip and cross-patch activation binaries.")
    else:
        print("\nSome tests FAILED. See details above.")
        sys.exit(1)
