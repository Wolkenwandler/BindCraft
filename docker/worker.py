#!/usr/bin/env python
"""worker.py — BindCraft compute worker.

Pulls one job at a time from a shared file-system queue and runs the *unmodified*
bindcraft.py CLI on it. Multiple worker replicas can run concurrently; claiming a job
is atomic via os.rename (a single rename either succeeds for exactly one worker or fails).

A "job" is simply a target settings JSON (same schema as settings_target/*.json) placed
in ${QUEUE}/pending/. The worker runs:

    python -u /app/bindcraft.py --settings <job> --filters <DEFAULT_FILTERS> --advanced <DEFAULT_ADVANCED>

Filters / advanced settings come from environment defaults so the queued files stay
identical to ordinary BindCraft target JSONs.

Queue layout (all under ${QUEUE}):
    pending/    jobs waiting to be claimed
    processing/ jobs currently running (named <job>.<worker_id>)
    done/       jobs that finished successfully
    failed/     jobs whose bindcraft.py run exited non-zero
    logs/       <job>.log stdout+stderr for each run

Environment:
    QUEUE              queue root dir            (default /workspace/queue)
    DEFAULT_FILTERS    --filters path            (default /app/settings_filters/default_filters.json)
    DEFAULT_ADVANCED   --advanced path           (default /app/settings_advanced/default_4stage_multimer.json)
    WORKER_ID          label for logs/claims     (default hostname)
    IDLE_POLL          seconds between polls when queue empty (default 15)
    EXIT_WHEN_EMPTY    "1" exit once queue drained, "0" keep polling (default 1)
    BINDCRAFT          path to bindcraft.py      (default /app/bindcraft.py)
"""
import os
import sys
import time
import socket
import subprocess

QUEUE = os.environ.get("QUEUE", "/workspace/queue")
DEFAULT_FILTERS = os.environ.get("DEFAULT_FILTERS", "/app/settings_filters/default_filters.json")
DEFAULT_ADVANCED = os.environ.get("DEFAULT_ADVANCED", "/app/settings_advanced/default_4stage_multimer.json")
WORKER_ID = os.environ.get("WORKER_ID") or socket.gethostname()
IDLE_POLL = int(os.environ.get("IDLE_POLL", "15"))
EXIT_WHEN_EMPTY = os.environ.get("EXIT_WHEN_EMPTY", "1") == "1"
BINDCRAFT = os.environ.get("BINDCRAFT", "/app/bindcraft.py")

PENDING = os.path.join(QUEUE, "pending")
PROCESSING = os.path.join(QUEUE, "processing")
DONE = os.path.join(QUEUE, "done")
FAILED = os.path.join(QUEUE, "failed")
LOGS = os.path.join(QUEUE, "logs")


def ensure_dirs():
    for d in (PENDING, PROCESSING, DONE, FAILED, LOGS):
        os.makedirs(d, exist_ok=True)


def claim_one():
    """Atomically claim a single pending job. Returns (job_name, processing_path) or None."""
    try:
        candidates = sorted(f for f in os.listdir(PENDING) if f.endswith(".json"))
    except FileNotFoundError:
        return None
    for name in candidates:
        src = os.path.join(PENDING, name)
        dst = os.path.join(PROCESSING, f"{name}.{WORKER_ID}")
        try:
            os.rename(src, dst)  # atomic on the same filesystem; only one worker wins
            return name, dst
        except (FileNotFoundError, OSError):
            continue  # another worker grabbed it first; try the next
    return None


def run_job(job_name, job_path):
    log_path = os.path.join(LOGS, f"{job_name}.log")
    cmd = [
        sys.executable, "-u", BINDCRAFT,
        "--settings", job_path,
        "--filters", DEFAULT_FILTERS,
        "--advanced", DEFAULT_ADVANCED,
    ]
    print(f"[{WORKER_ID}] running {job_name}: {' '.join(cmd)}", flush=True)
    with open(log_path, "w") as log:
        log.write(f"# worker={WORKER_ID}\n# cmd={' '.join(cmd)}\n\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    return proc.returncode


def main():
    ensure_dirs()
    print(f"[{WORKER_ID}] worker started; queue={QUEUE} exit_when_empty={EXIT_WHEN_EMPTY}", flush=True)
    while True:
        claim = claim_one()
        if claim is None:
            if EXIT_WHEN_EMPTY:
                print(f"[{WORKER_ID}] queue empty -> exiting", flush=True)
                return 0
            time.sleep(IDLE_POLL)
            continue

        job_name, job_path = claim
        rc = run_job(job_name, job_path)
        dest_dir = DONE if rc == 0 else FAILED
        final = os.path.join(dest_dir, job_name)
        try:
            os.replace(job_path, final)
        except OSError as e:
            print(f"[{WORKER_ID}] WARN: could not move {job_path} -> {final}: {e}", flush=True)
        status = "done" if rc == 0 else f"FAILED(rc={rc})"
        print(f"[{WORKER_ID}] {job_name} {status}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
