"""Offline target checks. This cannot repair missing system components."""
from __future__ import annotations
import json
import platform
import ssl
import struct
import sys


def platform_issues(python_version, bits, windows_version):
    issues = []
    if tuple(python_version[:2]) != (3, 8):
        issues.append('PYTHON38_REQUIRED')
    if bits != 64:
        issues.append('X64_RUNTIME_REQUIRED')
    major, minor, sp = windows_version
    if (major, minor) < (6, 1):
        issues.append('WINDOWS7_OR_NEWER_REQUIRED')
    elif (major, minor) == (6, 1) and sp < 1:
        issues.append('WINDOWS7_SP1_REQUIRED')
    return issues


def main():
    report = {'python': sys.version.split()[0], 'bits': struct.calcsize('P') * 8,
              'system': platform.platform(), 'openssl': ssl.OPENSSL_VERSION,
              'network_used': False, 'native_win7': False, 'issues': []}
    if sys.platform == 'win32':
        import ctypes
        version = sys.getwindowsversion()
        report['native_win7'] = (version.major, version.minor) == (6, 1)
        report['issues'] = platform_issues(sys.version_info, report['bits'],
                                          (version.major, version.minor, version.service_pack_major))
        try:
            kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
            getattr(kernel32, 'AddDllDirectory')
        except (OSError, AttributeError):
            report['issues'].append('KB2533623_OR_SUPERSEDING_UPDATE_REQUIRED')
        try:
            ctypes.WinDLL('api-ms-win-crt-runtime-l1-1-0.dll')
        except OSError:
            report['issues'].append('UNIVERSAL_C_RUNTIME_REQUIRED')
    else:
        report['issues'] = ['NOT_WINDOWS_TARGET']
    report['status'] = 'PASS' if not report['issues'] else 'FAILED'
    print(json.dumps(report, ensure_ascii=True, indent=2))
    print('Legacy OS/runtime: use trusted materials and protected networks. Model software is not bundled.')
    return 1 if report['issues'] else 0

if __name__ == '__main__':
    raise SystemExit(main())
