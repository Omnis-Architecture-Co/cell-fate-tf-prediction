#!/usr/bin/env python3
"""
Progress monitor for knockout simulation shards.
Run:  python3 validation/knockout/check_progress.py
"""

import json
import os
import glob
import time

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def check():
    progress_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "progress_*.json")))
    shard_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "shard_*.jsonl")))

    if not progress_files and not shard_files:
        print("No results found yet. Start shards with:")
        print("  python3 validation/knockout/knockout_simulation.py --shard 0 --total-shards 4")
        return

    print("=" * 72)
    print("  KNOCKOUT SIMULATION — PROGRESS MONITOR")
    print("=" * 72)

    total_done = 0
    total_genes = 0
    all_complete = True

    for pf in progress_files:
        with open(pf) as f:
            p = json.load(f)
        shard = p["shard"]
        done = p["done"]
        total = p["total"]
        pct = p["pct"]
        status = p.get("status", "RUNNING")
        elapsed = p.get("elapsed_s", 0)
        ts = p.get("timestamp", "?")

        total_done += done
        total_genes += total

        bar_len = 30
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)

        status_str = "DONE" if status == "COMPLETE" else f"{pct:.1f}%"
        print(f"  Shard {shard:2d}: [{bar}] {done:5d}/{total:5d} ({status_str})  {elapsed:.0f}s  {ts}")

        if status != "COMPLETE":
            all_complete = False

    for sf in shard_files:
        shard_id = os.path.basename(sf).replace("shard_", "").replace(".jsonl", "")
        n_lines = sum(1 for _ in open(sf))
        pf_match = os.path.join(RESULTS_DIR, f"progress_{shard_id}.json")
        if not os.path.exists(pf_match):
            print(f"  Shard {shard_id}: {n_lines} results (no progress file)")

    if total_genes > 0:
        print(f"\n  TOTAL: {total_done}/{total_genes} genes ({total_done/total_genes*100:.1f}%)")

    if all_complete and progress_files:
        print("\n  ALL SHARDS COMPLETE — run merge_results.py to combine")
    elif progress_files:
        rates = []
        for pf in progress_files:
            with open(pf) as f:
                p = json.load(f)
            if p.get("status") != "COMPLETE" and p["elapsed_s"] > 0 and p["done"] > 0:
                rate = p["done"] / p["elapsed_s"]
                remaining = p["total"] - p["done"]
                eta = remaining / rate
                rates.append(eta)
        if rates:
            max_eta = max(rates)
            print(f"\n  Estimated time remaining: {max_eta/60:.0f} minutes")


if __name__ == "__main__":
    check()
