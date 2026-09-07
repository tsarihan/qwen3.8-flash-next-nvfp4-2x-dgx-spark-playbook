#!/usr/bin/env python3
"""Verify local NVFP4 snapshot against HuggingFace LFS sha256 checksums."""
import hashlib, os, sys
from huggingface_hub import HfApi

REPO = "RadixArk/Qwen3.8-Flash-Next-NVFP4"
LOCAL = "/data/models/qwen3.8-flash-next-nvfp4"

api = HfApi()
tree = list(api.list_repo_tree(REPO, recursive=True, expand=True))
lfs = {}
plain = []
for e in tree:
    if getattr(e, "size", None) is None:
        continue
    if getattr(e, "lfs", None) and getattr(e.lfs, "sha256", None):
        lfs[e.path] = (e.lfs.sha256, e.size)
    else:
        plain.append((e.path, e.size))

print("repo files: %d  lfs-with-sha: %d  plain: %d" % (len(tree), len(lfs), len(plain)), flush=True)

bad, missing, ok = [], [], 0
for path, (sha, size) in sorted(lfs.items()):
    fp = os.path.join(LOCAL, path)
    if not os.path.exists(fp):
        missing.append(path); continue
    actual_size = os.path.getsize(fp)
    if actual_size != size:
        bad.append((path, "size %d != %d" % (actual_size, size))); continue
    h = hashlib.sha256()
    with open(fp, "rb") as f:
        while True:
            b = f.read(16 << 20)
            if not b: break
            h.update(b)
    if h.hexdigest() != sha:
        bad.append((path, "sha256 mismatch"))
    else:
        ok += 1
        print("  ok %s" % path, flush=True)

for path, size in plain:
    fp = os.path.join(LOCAL, path)
    if not os.path.exists(fp):
        missing.append(path)

print()
print("LFS verified ok : %d" % ok)
print("missing         : %d %s" % (len(missing), missing[:5]))
print("corrupt         : %d %s" % (len(bad), bad[:5]))
print("RESULT: %s" % ("VERIFY_OK" if (ok == len(lfs) and not bad and not missing and ok > 0) else "VERIFY_FAIL"))
