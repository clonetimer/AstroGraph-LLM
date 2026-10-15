"""OS-owned nonblocking task lease; released by the OS on crash (no stale PID guessing)."""
from contextlib import contextmanager
import os
from pathlib import Path

@contextmanager
def task_lease(output_root,task_id):
    directory=Path(output_root).resolve()/'.task-locks';directory.mkdir(parents=True,exist_ok=True)
    path=directory/(task_id+'.lock')
    flags=os.O_RDWR|os.O_CREAT
    if hasattr(os,'O_NOFOLLOW'):flags|=os.O_NOFOLLOW
    fd=os.open(path,flags,0o600)
    f=os.fdopen(fd,'r+b')
    try:
        if os.name=='posix':
            import fcntl
            try:fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as exc:raise RuntimeError('TASK_ALREADY_RUNNING') from exc
        elif os.name=='nt':
            import msvcrt
            if path.stat().st_size==0:f.write(b'0');f.flush()
            f.seek(0)
            try:msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
            except OSError as exc:raise RuntimeError('TASK_ALREADY_RUNNING') from exc
        else:raise RuntimeError('TASK_LOCK_PLATFORM_NOT_SUPPORTED')
        yield
    finally:
        f.close()
