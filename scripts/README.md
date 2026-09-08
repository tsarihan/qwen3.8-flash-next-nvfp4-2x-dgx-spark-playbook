# Harness

The matrix harness runs from a **separate host**, not from the Spark nodes. Driving the
client from one of the nodes competes with the engine for the same unified memory and
skews every number you collect. Put the load generator somewhere else on the LAN.

| file | what it does |
|---|---|
| `serve-qwen38fn-nvfp4-vllm.sh` | starts one node of the TP=2 pair; all knobs are env vars (`SPEC`, `MTP_K`, `MTP_FIX`, `YARN`, `MAX_NUM_SEQS`, `KV_DTYPE`, `GPU_UTIL`, …) |
| `matrix-lib.sh` | `run_cfg` / `teardown` / `mem_gib`; pins `GPU_UTIL` at 0.85 and refuses to start a config if free memory is below `MEM_FLOOR_GIB` |
| `phase1.sh` `phase2.sh` `phase3.sh` | MTP configs, then stream-count and KV-dtype configs, then the winner: full ladder + SWE-bench Pro 40 through litellm |
| `run-overnight.sh` | chains the three phases, appending to `~/overnight.log` |
| `sweep.py` | concurrency ladder; reports aggregate tok/s, per-stream decode, TTFT |
| `needles.py` | needle-in-a-haystack at a given depth, 5 needles |
| `report.py` | reads `matrix-results/summary.txt` and prints the ranked table, the ladder, and the SWE score |

`teardown` between configs is not optional. A vLLM process that is still unwinding holds
tens of GiB, and the next config will either start with a truncated KV pool or wedge the
node. `matrix-lib.sh` waits for memory to actually come back before it launches anything.
