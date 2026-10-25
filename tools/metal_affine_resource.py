"""Windows system commit headroom guard; never changes numerical tolerances."""
import ctypes
import os


def commit_headroom_bytes():
    if os.name!='nt':return None
    class Performance(ctypes.Structure):
        _fields_=[('cb',ctypes.c_uint32)]+[(name,ctypes.c_size_t) for name in
            ('commit_total','commit_limit','commit_peak','physical_total','physical_available','system_cache',
             'kernel_total','kernel_paged','kernel_nonpaged','page_size')]+[(name,ctypes.c_uint32) for name in
            ('handle_count','process_count','thread_count')]
    value=Performance();value.cb=ctypes.sizeof(value)
    if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(value),ctypes.sizeof(value)):
        raise OSError('Cannot inspect system CommitLimit minus CommitTotal')
    return int(value.commit_limit-value.commit_total)*int(value.page_size)


def require_commit_headroom(minimum_bytes=1536*1024**2):
    available=commit_headroom_bytes()
    if available is not None and available<minimum_bytes:
        raise MemoryError(f'Resource guard: system commit headroom {available} B below required {minimum_bytes} B; no numerical fallback')
    return available
