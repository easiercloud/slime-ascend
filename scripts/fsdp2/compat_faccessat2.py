#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run a command with a narrow legacy-container libc compatibility fallback.

Old Docker filters can return EPERM for faccessat2 instead of ENOSYS. Deny
that syscall with ENOSYS so libc uses its existing permitted older interface.
This adds a process-local restriction; it grants no privileges or syscalls.
"""

import ctypes
import errno
import os
import platform
import sys


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: compat_faccessat2.py COMMAND [ARG ...]")
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise SystemExit("This compatibility wrapper is for Linux aarch64 containers.")
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.syscall(439, -100, b".", 4, 0x200)
    if result == -1 and ctypes.get_errno() == errno.EPERM:
        class Filter(ctypes.Structure):
            _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte),
                        ("jf", ctypes.c_ubyte), ("k", ctypes.c_uint32)]

        class Program(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ushort), ("filter", ctypes.POINTER(Filter))]

        instructions = (Filter * 4)(Filter(0x20, 0, 0, 0), Filter(0x15, 0, 1, 439),
                                    Filter(0x06, 0, 0, 0x00050000 | errno.ENOSYS),
                                    Filter(0x06, 0, 0, 0x7FFF0000))
        program = Program(4, instructions)
        if libc.prctl(38, 1, 0, 0, 0) or libc.prctl(22, 2, ctypes.byref(program), 0, 0):
            raise OSError(ctypes.get_errno(), os.strerror(ctypes.get_errno()))
        print("Applied process-local faccessat2 ENOSYS fallback for legacy Docker.", flush=True)
    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    main()
