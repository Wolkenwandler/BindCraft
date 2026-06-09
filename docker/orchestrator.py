#!/usr/bin/env python
"""orchestrator.py — BindCraft batch orchestrator (lightweight, no jax / no pyrosetta).

Responsibilities:
  1. Enqueue: copy every target settings JSON from ${BATCH_IN} into ${QUEUE}/pending/
     so the compute workers can claim them. This is the "batch input" entry point —
     drop N target JSONs in batch_inputs/ and they all get processed.
  2. Monitor: poll the queue until every enqueued job lands in done/ or failed/.
  3. Aggregate: read each target's <design_path>/final_design_stats.csv and concatenate
     them into ${WORKSPACE}/combined_final_stats.csv, with a per-target summary.

Each batch input file uses the ordinary BindCraft target schema (see settings_target/*.json):
its "design_path" should point inside the shared workspace volume (e.g. /workspace/designs/<name>/)
so outputs persist and can be aggregated here.

Environment:
    BATCH_IN     dir with input target JSONs   (default /batch_inputs)
    QUEUE        shared queue root             (default /workspace/queue)
    WORKSPACE    shared workspace root         (default /workspace)
    POLL         seconds between status polls  (default 20)
    ENQUEUE      "1" copy BATCH_IN -> pending  (default 1)
    WAIT         "1" block until all done      (default 1)
"""
import os
import sys
import csv
import json
import time
import glob
import shutil

BATCH_IN = os.environ.get("BATCH_IN", "/batch_inputs")
QUEUE = os.environ.get("QUEUE", "/workspace/queue")
WORKSPACE = os.environ.get("WORKSPACE", "/workspace")
POLL = int(os.environ.get("POLL", "20"))
ENQUEUE = os.environ.get("ENQUEUE", "1") == "1"
WAIT = os.environ.get("WAIT", "1") == "1"

PENDING = os.path.join(QUEUE, "pending")
DONE = os.path.join(QUEUE, "done")
FAILED = os.path.join(QUEUE, "failed")
COMBINED = os.path.join(WORKSPACE, "combined_final_stats.csv")


def ensure_dirs():
    for d in (PENDING, DONE, FAILED, WORKSPACE):
        os.makedirs(d, exist_ok=True)


def enqueue():
    """Copy each batch input target JSON into the pending queue. Returns list of job names."""
    inputs = sorted(glob.glob(os.path.join(BATCH_IN, "*.json")))
    if not inputs:
        print(f"[orchestrator] no *.json found in {BATCH_IN}", flush=True)
    jobs = []
    for path in inputs:
        name = os.path.basename(path)
        # validate it parses and grab design_path for later aggregation
        try:
            with open(path) as fh:
                json.load(fh)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[orchestrator] skipping invalid input {name}: {e}", flush=True)
            continue
        shutil.copy(path, os.path.join(PENDING, name))
        jobs.append(name)
        print(f"[orchestrator] enqueued {name}", flush=True)
    return jobs


def settled(job_names):
    """Return (done, failed) sets among job_names that have settled."""
    done = {n for n in job_names if os.path.exists(os.path.join(DONE, n))}
    failed = {n for n in job_names if os.path.exists(os.path.join(FAILED, n))}
    return done, failed


def wait_for(job_names):
    remaining = set(job_names)
    while remaining:
        done, failed = settled(job_names)
        finished = done | failed
        remaining = set(job_names) - finished
        print(f"[orchestrator] progress: {len(finished)}/{len(job_names)} settled "
              f"({len(done)} done, {len(failed)} failed), {len(remaining)} remaining", flush=True)
        if not remaining:
            break
        time.sleep(POLL)
    done, failed = settled(job_names)
    return done, failed


def design_path_of(job_name):
    """Read the queued/settled job JSON to find its design_path."""
    for base in (DONE, FAILED, PENDING):
        p = os.path.join(base, job_name)
        if os.path.exists(p):
            try:
                with open(p) as fh:
                    return json.load(fh).get("design_path")
            except (json.JSONDecodeError, OSError):
                return None
    return None


def aggregate(job_names):
    """Concatenate each target's final_design_stats.csv into one combined CSV."""
    rows = []
    header = None
    summary = []
    for name in job_names:
        dp = design_path_of(name)
        if not dp:
            summary.append((name, "no design_path", 0))
            continue
        csv_path = os.path.join(dp, "final_design_stats.csv")
        if not os.path.exists(csv_path):
            summary.append((name, "no final_design_stats.csv", 0))
            continue
        with open(csv_path, newline="") as fh:
            reader = csv.reader(fh)
            file_header = next(reader, None)
            if file_header and header is None:
                header = ["target"] + file_header
            count = 0
            for row in reader:
                rows.append([name] + row)
                count += 1
        summary.append((name, "ok", count))

    if header is not None:
        with open(COMBINED, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"[orchestrator] wrote {COMBINED} ({len(rows)} rows from {len(job_names)} targets)", flush=True)
    else:
        print("[orchestrator] no final_design_stats.csv found for any target; nothing aggregated", flush=True)

    print("[orchestrator] per-target summary:", flush=True)
    for name, status, count in summary:
        print(f"    {name}: {status} ({count} final designs)", flush=True)


def main():
    ensure_dirs()
    if ENQUEUE:
        jobs = enqueue()
    else:
        # operate over whatever is already queued/settled
        jobs = sorted(set(os.listdir(PENDING)) | set(os.listdir(DONE)) | set(os.listdir(FAILED)))
        jobs = [j for j in jobs if j.endswith(".json")]

    if not jobs:
        print("[orchestrator] no jobs to process", flush=True)
        return 0

    if WAIT:
        done, failed = wait_for(jobs)
        if failed:
            print(f"[orchestrator] WARNING: {len(failed)} job(s) failed: {sorted(failed)}", flush=True)
        aggregate(jobs)
    else:
        print(f"[orchestrator] enqueued {len(jobs)} job(s); not waiting (WAIT=0)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
