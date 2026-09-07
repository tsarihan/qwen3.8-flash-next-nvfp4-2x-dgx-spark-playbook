#!/usr/bin/env python3
"""Bound page cache growth while writing/reading very large model files.

On GB10 the GPU and the page cache draw on the SAME unified memory pool, so a large
sequential write (a 185 GB checkpoint download, or loading 41 GB shards) silently
consumes the memory the engine needs. The node does not OOM-kill cleanly -- it wedges:
still pings, still accepts TCP, but sshd stops completing logins.

This walks a directory and calls posix_fadvise(POSIX_FADV_DONTNEED) on every file whose
size has stopped changing, telling the kernel it can drop those pages. Needs no root and
no engine patches. Run it alongside any bulk download or model load.

  usage: cache-warden.py <dir> [interval_s] [--once]
"""
import os, sys, time

def drop(path):
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return 0
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return os.fstat(fd).st_size
    except OSError:
        return 0
    finally:
        os.close(fd)

def mem_avail_gb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    except Exception:
        pass
    return -1.0

def walk(root):
    for dirpath, _, names in os.walk(root):
        for n in names:
            yield os.path.join(dirpath, n)

def main():
    root = sys.argv[1]
    interval = float(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else 20.0
    once = "--once" in sys.argv
    sizes = {}
    while True:
        dropped = 0
        for p in walk(root):
            try:
                sz = os.path.getsize(p)
            except OSError:
                continue
            # only drop pages for files that have stopped growing (download finished)
            if sizes.get(p) == sz and sz > 0:
                dropped += drop(p)
            sizes[p] = sz
        print("[cache-warden] dropped %6.1f GB of page cache | MemAvailable %5.1f GB | %s"
              % (dropped / 1e9, mem_avail_gb(), time.strftime("%H:%M:%S")), flush=True)
        if once:
            return
        time.sleep(interval)

if __name__ == "__main__":
    main()
