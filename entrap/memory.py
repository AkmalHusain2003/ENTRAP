"""Manage temporary arrays and optional memory-mapped storage."""

import gc
import os
import tempfile
import numpy as np


class MemoryManager:
    """Manage reusable arrays and optional memory-mapped scratch storage."""

    def __init__(self, use_memmap=False, folder=None):
        """Initialize the memory manager and its temporary-storage settings."""
        self.use_memmap = use_memmap
        self.folder = folder or tempfile.mkdtemp(prefix="entrap_")
        self._paths = []

    def array(self, shape, dtype=np.float64, fill=None):
        """Allocate a NumPy array or memory-mapped scratch array."""
        if self.use_memmap:
            p = os.path.join(self.folder, f"arr_{len(self._paths)}.dat")
            self._paths.append(p)
            a = np.memmap(p, dtype=dtype, mode="w+", shape=shape)
        else:
            a = np.empty(shape, dtype=dtype)
        if fill is not None:
            a[:] = fill
        return a

    def collect(self):
        """Trigger garbage collection for released working arrays."""
        gc.collect()

    def cleanup(self):
        """Remove temporary memory-mapped files managed by this object."""
        gc.collect()
        for p in self._paths:
            try:
                os.remove(p)
            except OSError:
                pass
        self._paths.clear()
