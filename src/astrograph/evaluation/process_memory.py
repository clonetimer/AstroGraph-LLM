"""OS process memory measurements; cumulative peak is not an isolated case peak."""
import ctypes,sys

def observe_process_memory():
    if sys.platform!='win32':
        import resource
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {'peak_working_set_bytes':int(peak*(1 if sys.platform=='darwin' else 1024)),
          'measurement':'OS cumulative process resident high-water mark; includes parsing and prior cases'}
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in
          ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    counters=Counters();counters.cb=ctypes.sizeof(counters)
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.GetCurrentProcess.restype=wintypes.HANDLE
    api=ctypes.WinDLL('psapi',use_last_error=True);api.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    if not api.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):raise OSError(ctypes.get_last_error(),'GetProcessMemoryInfo')
    return {'peak_working_set_bytes':int(counters.PeakWorkingSetSize),'current_working_set_bytes':int(counters.WorkingSetSize),
      'measurement':'Windows OS cumulative process working-set high-water mark; includes parsing and prior cases, not isolated case peak'}
