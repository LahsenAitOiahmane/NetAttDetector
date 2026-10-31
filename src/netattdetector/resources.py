"""Lightweight sampled process accounting, without importing model-training libraries."""

import threading
import time

import psutil


class ResourceMonitor:
    """Sample RSS every 50 ms. Child RSS sums can count shared pages more than once."""

    def __init__(self, include_children=False):
        self.include_children = include_children

    def _sample(self):
        parent_rss = self.process.memory_info().rss
        self.peak = max(self.peak, parent_rss)
        total = parent_rss
        if self.include_children:
            for child in self.process.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        self.peak_tree = max(self.peak_tree, total)

    def __enter__(self):
        self.process = psutil.Process()
        self.peak = self.peak_tree = 0
        self._sample()
        self.start = time.perf_counter()
        self.done = threading.Event()

        def sample():
            while not self.done.wait(0.05):
                self._sample()

        self.worker = threading.Thread(target=sample, daemon=True)
        self.worker.start()
        return self

    def __exit__(self, *args):
        self.done.set()
        self.worker.join()
        self._sample()
        self.elapsed = time.perf_counter() - self.start
