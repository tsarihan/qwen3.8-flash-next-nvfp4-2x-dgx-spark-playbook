#!/usr/bin/env python3
"""Morning report: read the matrix summary and rank configs."""
import json, os, re, sys
R = os.path.expanduser("~/matrix-results")
p = os.path.join(R, "summary.txt")
rows, failed = [], []
if os.path.exists(p):
    for line in open(p):
        parts = line.strip().split("|", 2)
        if len(parts) < 2: continue
        if parts[1] == "OK":
            try: rows.append(json.loads(parts[2]))
            except Exception: pass
        else:
            failed.append((parts[0], parts[1], parts[2] if len(parts) > 2 else ""))

def kvtok(r):
    kv = r.get("kv") or ""
    import re
    m = re.search(r"GPU KV cache size: ([0-9,]+) tokens", kv)
    return int(m.group(1).replace(",", "")) if m else 0

print("=" * 108)
print("QWEN3.8-FLASH-NEXT-NVFP4 (NVIDIA) - OVERNIGHT CONFIG MATRIX")
print("2x DGX Spark, TP=2+EP, GPU_UTIL 0.85 throughout")
print("=" * 108)
if not rows:
    print("no completed configs yet")
else:
    hdr = "%-16s %10s %9s %9s %9s %9s %9s %14s  %-12s" % (
        "config", "KV tokens", "c1 agg", "c4 agg", "c16 agg", "c1 str", "c1 ttft", "weights/load", "NIAH 4k/131k")
    print(hdr); print("-" * 108)
    for r in sorted(rows, key=lambda x: -(x.get("c1_agg") or 0)):
        _l = r.get("load") or ""
        _m = re.search(r"took ([0-9.]+) GiB memory and ([0-9.]+) seconds", _l)
        load = ("%s GiB/%ss" % (_m.group(1), int(float(_m.group(2))))) if _m else _l[:12]
        niah = "%s / %s" % (r.get("niah_4096", "-"), r.get("niah_131072", "-"))
        print("%-16s %10s %9s %9s %9s %9s %9s %14s  %-12s" % (
            r.get("config", "?"), format(kvtok(r), ",") if kvtok(r) else "-",
            r.get("c1_agg", "-"), r.get("c4_agg", "-"), r.get("c16_agg", "-"),
            r.get("c1_str", "-"), r.get("c1_ttft", "-"), load, niah))
    print()
    best_c1 = max(rows, key=lambda x: x.get("c1_agg") or 0)
    best_c16 = max(rows, key=lambda x: x.get("c16_agg") or 0)
    best_kv = max(rows, key=kvtok)
    print("SWEET SPOTS")
    print("  best single stream (c1 agg) : %s  (%s tok/s)" % (best_c1.get("config"), best_c1.get("c1_agg")))
    print("  best throughput  (c16 agg)  : %s  (%s tok/s)" % (best_c16.get("config"), best_c16.get("c16_agg")))
    print("  largest KV pool             : %s  (%s tokens)" % (best_kv.get("config"), format(kvtok(best_kv), ",")))
    bad = [r for r in rows if str(r.get("niah_4096")) not in ("5/5",) or str(r.get("niah_131072")) not in ("5/5",)]
    if bad:
        print("\n  CORRECTNESS WARNING - these did not return 5/5 needles, do not use:")
        for r in bad: print("    %s -> 4k=%s 131k=%s" % (r.get("config"), r.get("niah_4096"), r.get("niah_131072")))
    else:
        print("\n  all completed configs returned 5/5 needles at 4k and 131k")
if failed:
    print("\nFAILED / ABORTED")
    for n, why, det in failed:
        note = ""
        if n.startswith("fp8kv"):
            note = "  <- FlashAttention has no fp8_e4m3 kv-cache path on sm_121 (kernel limit, not config)"
        print("  %-16s %-14s %s%s" % (n, why, det[:70], note))


# ---- full concurrency ladder on the winner ----
lp = os.path.join(R, "sweep-winner-fullladder.json")
if os.path.exists(lp):
    try:
        d = json.load(open(lp))
        print()
        print("FULL LADDER on the winner (max_num_seqs raised to 64 so the ladder can reach c=64)")
        print("  %4s %11s %12s %9s" % ("conc", "agg tok/s", "per-stream", "ttft s"))
        best = None
        for r in d["rows"]:
            print("  %4d %11.2f %12.2f %9.3f" % (r["concurrency"], r["aggregate_tps"],
                  r["per_stream_decode_tps"], r["ttft_mean_s"]))
            if best is None or r["aggregate_tps"] > best["aggregate_tps"]: best = r
        print("  peak aggregate at c=%d: %.2f tok/s; knee at c=16" % (best["concurrency"], best["aggregate_tps"]))
    except Exception as e:
        print("\nladder unreadable: %s" % e)

# ---- SWE-bench Pro 40 on the winner, through litellm ----
import glob
print()
print("SWE-BENCH PRO 40 on the winner, driven from OMEN through the litellm proxy")
ev = sorted(glob.glob(os.path.expanduser("~/swe-enterprise40/eval-nvidia-*/eval_results.json")))
outs = sorted(glob.glob(os.path.expanduser("~/swe-enterprise40/out-nvidia-*")))
if ev:
    for f in ev:
        try:
            d = json.load(open(f))
            if isinstance(d, dict) and "resolved" in d:
                res, tot = len(d["resolved"]), len(d.get("resolved", [])) + len(d.get("unresolved", []))
            else:
                items = d if isinstance(d, list) else d.get("results", [])
                tot = len(items)
                res = sum(1 for i in items if i.get("resolved") or i.get("is_resolved"))
            name = os.path.basename(os.path.dirname(f)).replace("eval-nvidia-", "")
            pct = (100.0 * res / tot) if tot else 0.0
            print("  %-16s %d/%d resolved  (%.1f%%)   baseline was 36/40 (90.0%%) on the prior build"
                  % (name, res, tot, pct))
        except Exception as e:
            print("  %s unreadable: %s" % (f, e))
elif outs:
    n = len(glob.glob(os.path.join(outs[-1], "*", "*.traj.json")))
    print("  still running or not yet graded: %d/40 trajectories in %s" % (n, os.path.basename(outs[-1])))
else:
    print("  no SWE output found")

print("""
NOTES FROM THE RUN
  MTP  vLLM PR 55513 was ported to this older image (patches/, MTP_FIX=1) and works.
       num_speculative_tokens=1 beats 3 everywhere, which is what NVIDIA's card specifies.
       MTP costs about 395k KV tokens and ~4% prefill, and returns +44% single-stream
       decode and +56% decode at 131k context. Worth it for long single-stream agent work.
  fp8 KV  Not available. The QSA gate was widened successfully (patches/qsa.py, QSA_FP8=1)
       but FlashAttention itself refuses fp8_e4m3 KV on this device. Needs kernel work,
       not configuration. Upstream PRs 55557 and 54846 are still in review.
  Safety  GPU_UTIL was pinned at 0.85 for every config. 0.90 wedged a node earlier and
       needed a power cycle; the engine reports the true ceiling as --kv-cache-memory.""")
print("=" * 108)
