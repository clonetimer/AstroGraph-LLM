from __future__ import annotations

import errno
import json
import os
import tempfile
from pathlib import Path
from collections.abc import Callable
from typing import Any


class ArtifactPathError(ValueError):
    """Raised when an artifact path can escape its governed root."""


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


class ArtifactRoot:
    """Governed root for atomic experiment/runtime artifact writes.

    The root itself and every parent of a target are resolved before use.  Absolute
    child paths, ``..`` traversal, source-tree writes, and symlink escapes are
    rejected.  Writes use fsync + atomic replace.
    """

    def __init__(self, root: str | Path, *, source_root: str | Path | None = None):
        raw = Path(root).expanduser()
        resolved = raw.resolve(strict=False)
        if source_root is not None:
            source = Path(source_root).expanduser().resolve(strict=False)
            if _is_within(resolved, source):
                raise ArtifactPathError("artifact root must be outside the AstroGraph source tree")
        self.root = resolved
        self.source_root = (
            Path(source_root).expanduser().resolve(strict=False) if source_root is not None else None
        )

    def ensure(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        resolved = self.root.resolve(strict=True)
        if self.source_root is not None and _is_within(resolved, self.source_root):
            raise ArtifactPathError("resolved artifact root is inside the AstroGraph source tree")
        return resolved

    def resolve(self, relative_path: str | Path) -> Path:
        child = Path(relative_path)
        if child.is_absolute():
            raise ArtifactPathError("artifact child path must be relative")
        if not child.parts:
            raise ArtifactPathError("artifact child path must name a file")
        if any(part in {"", ".", ".."} for part in child.parts):
            raise ArtifactPathError("artifact child path contains traversal or empty components")
        root = self.ensure()
        current = root
        for part in child.parts[:-1]:
            current = current / part
            if current.exists() and current.is_symlink():
                target = current.resolve(strict=True)
                if not _is_within(target, root):
                    raise ArtifactPathError("artifact parent symlink escapes governed root")
            current.mkdir(exist_ok=True)
            resolved_current = current.resolve(strict=True)
            if not _is_within(resolved_current, root):
                raise ArtifactPathError("artifact parent escapes governed root")
        target = current / child.name
        if target.is_symlink():
            resolved_target = target.resolve(strict=True)
            if not _is_within(resolved_target, root):
                raise ArtifactPathError("artifact target symlink escapes governed root")
            raise ArtifactPathError("writing through artifact target symlinks is forbidden")
        prospective = target.resolve(strict=False)
        if not _is_within(prospective, root):
            raise ArtifactPathError("artifact target escapes governed root")
        return target

    @staticmethod
    def _sync_directory(path: Path) -> None:
        """Best-effort parent-directory durability sync.

        POSIX permits ``fsync`` on an opened directory and AstroGraph uses that
        to make the atomic replace durable.  Windows exposes ``os.fsync`` via
        the MSVCRT ``_commit`` API, whose CRT file descriptors are not a
        portable representation of directory handles.  Depending on the
        launcher/runtime, attempting to commit such a descriptor can surface as
        ``EBADF`` (``[Errno 9] Bad file descriptor``) even though the target file
        itself was written correctly.

        Directory syncing is therefore best-effort: unsupported-descriptor
        errors are ignored, while real I/O failures still propagate.
        """
        try:
            directory_fd = os.open(path, os.O_RDONLY)
        except OSError:
            return
        try:
            try:
                os.fsync(directory_fd)
            except OSError as exc:
                unsupported = {errno.EBADF, errno.EINVAL}
                if hasattr(errno, "ENOTSUP"):
                    unsupported.add(errno.ENOTSUP)
                if hasattr(errno, "EOPNOTSUPP"):
                    unsupported.add(errno.EOPNOTSUPP)
                if exc.errno not in unsupported:
                    raise
        finally:
            try:
                os.close(directory_fd)
            except OSError as exc:
                if exc.errno != errno.EBADF:
                    raise


    @staticmethod
    def _sync_regular_file(stream: Any, path: Path) -> bool:
        """Synchronize a regular artifact without misclassifying CRT EBADF as data failure.

        Returns ``True`` when an fsync barrier completed and ``False`` when the
        runtime rejected only the durability barrier with a known descriptor
        compatibility error.  The fallback is deliberately narrow: the open
        descriptor must still be valid and point to a regular file, and the
        on-disk temporary path must still exist.  EIO/ENOSPC and other genuine
        storage errors always propagate.
        """
        stream.flush()
        fd = stream.fileno()
        try:
            os.fsync(fd)
            return True
        except OSError as exc:
            unsupported = {errno.EBADF, errno.EINVAL}
            if hasattr(errno, "ENOTSUP"):
                unsupported.add(errno.ENOTSUP)
            if hasattr(errno, "EOPNOTSUPP"):
                unsupported.add(errno.EOPNOTSUPP)
            if exc.errno not in unsupported:
                raise

            # Do not swallow a genuinely closed/corrupt descriptor.
            try:
                info = os.fstat(fd)
            except OSError:
                raise exc
            import stat
            if not stat.S_ISREG(info.st_mode) or not path.is_file():
                raise exc

            # A second descriptor occasionally succeeds even when the file
            # object's CRT descriptor does not.  Prefer that stronger barrier.
            flags = os.O_RDWR | getattr(os, "O_BINARY", 0)
            retry_fd: int | None = None
            try:
                retry_fd = os.open(path, flags)
                try:
                    os.fsync(retry_fd)
                    return True
                except OSError as retry_exc:
                    if retry_exc.errno not in unsupported:
                        raise
            except OSError as open_exc:
                if open_exc.errno not in unsupported:
                    raise
            finally:
                if retry_fd is not None:
                    try:
                        os.close(retry_fd)
                    except OSError as close_exc:
                        if close_exc.errno != errno.EBADF:
                            raise

            # Bytes were written and the temp file is a valid regular file;
            # continue with atomic publication but without a durability claim.
            return False

    def write_generated(
        self,
        relative_path: str | Path,
        writer: Callable[[Path], None],
    ) -> Path:
        """Atomically publish a file produced by ``writer``.

        The writer receives a unique sibling path inside the governed artifact
        root.  The temporary file is synchronized before an atomic replacement,
        so a failed or interrupted serializer cannot expose a partial target.
        """

        target = self.resolve(relative_path)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            writer(temporary)
            if not temporary.is_file():
                raise ArtifactPathError("artifact writer did not produce a regular file")
            # Use a write-capable descriptor for the durability commit.  On
            # Windows ``os.fsync`` maps to MSVCRT ``_commit``.  Some desktop /
            # GUI-hosted Python runtimes can still surface EBADF/EINVAL for a
            # perfectly live regular-file descriptor.  This must not turn a
            # completed diagnosis into a false CORE_DIAGNOSIS failure.
            #
            # We therefore keep durability strict for real I/O failures, but
            # treat descriptor-compatibility errors as best-effort *only after*
            # confirming that the descriptor still names the just-written
            # regular file.  Atomic replace remains mandatory.
            with temporary.open("rb+") as stream:
                self._sync_regular_file(stream, temporary)
            os.replace(temporary, target)
            self._sync_directory(target.parent)
        finally:
            if temporary.exists():
                temporary.unlink()
        return target

    def write_bytes(self, relative_path: str | Path, payload: bytes) -> Path:
        def _writer(temporary: Path) -> None:
            with temporary.open("wb") as stream:
                stream.write(payload)
                stream.flush()

        return self.write_generated(relative_path, _writer)

    def write_text(self, relative_path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
        return self.write_bytes(relative_path, text.encode(encoding))

    def write_json(self, relative_path: str | Path, payload: Any) -> Path:
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        return self.write_text(relative_path, text)


__all__ = ["ArtifactPathError", "ArtifactRoot"]
