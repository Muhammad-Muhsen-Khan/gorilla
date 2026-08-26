#!/usr/bin/env python3
"""Unattended driver for the rest of the BFCL v4 suite.

Replaces schedule.py, which had a real bug: it launched the next model the
instant the previous bfcl process exited, but that process's vLLM server holds
~36 GiB (gpu_memory_utilization=0.9) for up to a minute afterwards. The new
server then OOMed at startup, and because each failure takes only ~1 min the
whole queue drained in about 13 minutes without running anything.

Fixes here:
  * wait_for_gpu() blocks until the GPU actually reports free memory
  * failed models are retried instead of being consumed from the queue
  * the two no-stop-token models (base, instruct) are sharded by category
    across all 8 GPUs instead of running one-per-GPU for 20 h / 5 h
"""
import os, subprocess, sys, time, pathlib

ROOT = pathlib.Path("/home/ubuntu/gorilla/berkeley-function-call-leaderboard")
LOGS = ROOT / "runlogs"
BIN  = f"{ROOT}/.venv/bin/bfcl"
CATS = ("simple_python,simple_java,simple_javascript,multiple,parallel,parallel_multiple,"
        "irrelevance,live_simple,live_multiple,live_parallel,live_parallel_multiple,"
        "live_irrelevance,live_relevance,multi_turn_base,multi_turn_miss_func,"
        "multi_turn_miss_param,multi_turn_long_context,memory_kv,memory_vector,memory_rec_sum")
EXPECTED, NGPU, BASE_PORT = 5017, 8, 1053
FREE_MIB, GPU_WAIT_MAX, MAX_TRIES = 2000, 900, 3

SHARDS = ["live_multiple",
          "live_irrelevance",
          "simple_python,simple_java,simple_javascript",
          "multiple,parallel,parallel_multiple",
          "irrelevance,live_simple,live_parallel,live_parallel_multiple,live_relevance",
          "multi_turn_base,multi_turn_miss_func",
          "multi_turn_miss_param,multi_turn_long_context",
          "memory_kv,memory_vector,memory_rec_sum"]

def say(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def running_models():
    """Models with a live `bfcl generate` process, so we never start a second
    writer for the same result files (that produced duplicate rows once)."""
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    found = set()
    for line in out.splitlines():
        if "bfcl" in line and "generate" in line and "--model" in line:
            parts = line.split()
            found.add(parts[parts.index("--model") + 1])
    return found

def dedupe_results():
    """Keep the last row per test id. A duplicate writer briefly existed for
    reasoning-258; this makes the evaluation input clean regardless."""
    import json
    total = 0
    for f in (ROOT / "result").rglob("*_result.json"):
        seen = {}
        for line in open(f):
            line = line.strip()
            if not line: continue
            try: seen[json.loads(line)["id"]] = line
            except Exception: continue
        rows_in = sum(1 for l in open(f) if l.strip())
        if len(seen) != rows_in:
            f.write_text("\n".join(seen.values()) + "\n")
            total += rows_in - len(seen)
            say(f"  deduped {f.parent.name}/{f.name}: -{rows_in - len(seen)}")
    say(f"dedupe removed {total} duplicate rows")

def rows(model):
    d = ROOT / "result" / model
    return sum(sum(1 for _ in open(f)) for f in d.rglob("*_result.json")) if d.is_dir() else 0

def gpu_used(i):
    out = subprocess.run(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits",
                          "-i",str(i)], capture_output=True, text=True).stdout.strip()
    try: return int(out)
    except ValueError: return 10**9

def wait_for_gpu(i):
    """Block until GPU i is actually free. This is the guard that was missing."""
    t0 = time.time()
    while gpu_used(i) > FREE_MIB:
        if time.time() - t0 > GPU_WAIT_MAX:
            say(f"  WARN gpu{i} still at {gpu_used(i)} MiB after {GPU_WAIT_MAX}s; proceeding")
            return
        time.sleep(15)

def spawn(model, gpu, cats, tag, threads):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["LOCAL_SERVER_PORT"] = str(BASE_PORT + gpu)
    env["PATH"] = f"{ROOT}/.venv/bin:" + env["PATH"]
    log = open(LOGS / f"gen_{tag}.log", "w")
    p = subprocess.Popen([BIN, "generate", "--model", model, "--test-category", cats,
                          "--backend", "vllm", "--num-gpus", "1",
                          "--gpu-memory-utilization", "0.9",
                          "--num-threads", str(threads)],
                         cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    return p, log

def run_pool(models, threads=16):
    """One model per GPU, retried on failure, never launching onto a busy GPU."""
    busy = running_models()
    for m in models:
        if m in busy: say(f"  not queueing {m}: already running elsewhere")
    pending = [[m, 0] for m in models if rows(m) < EXPECTED and m not in busy]
    slots = [None] * NGPU
    while pending or any(slots):
        for i, s in enumerate(slots):
            if s is not None or not pending:
                continue
            # Non-blocking: skip a GPU that is still busy rather than waiting on
            # it, otherwise one occupied GPU stalls every free one behind it.
            if gpu_used(i) > FREE_MIB:
                continue
            m, tries = pending.pop(0)
            p, log = spawn(m, i, CATS, m, threads)
            slots[i] = (m, tries, p, log, time.time())
            say(f"gpu{i} START {m} (try {tries+1}, {len(pending)} queued)")
        time.sleep(20)
        for i, s in enumerate(slots):
            if s is None: continue
            m, tries, p, log, t0 = s
            if p.poll() is None: continue
            log.close(); slots[i] = None
            n, mins = rows(m), (time.time()-t0)/60
            if n >= EXPECTED:
                say(f"gpu{i} DONE  {m}  {mins:.1f} min  OK ({n})")
            elif tries + 1 < MAX_TRIES:
                say(f"gpu{i} FAIL  {m} rc={p.returncode} {n}/{EXPECTED} -> requeue")
                pending.append([m, tries+1])
            else:
                say(f"gpu{i} GIVEUP {m} after {tries+1} tries ({n}/{EXPECTED})")

def run_sharded(model, threads=32):
    """Split one model's categories across all 8 GPUs; shards write distinct files."""
    say(f"=== {model}: sharding across {NGPU} GPUs ===")
    procs = []
    for i, cats in enumerate(SHARDS):
        wait_for_gpu(i)
        p, log = spawn(model, i, cats, f"{model}_shard{i}", threads)
        procs.append((i, p, log)); say(f"  gpu{i} <- {cats}")
    for i, p, log in procs:
        p.wait(); log.close(); say(f"  gpu{i} shard exit rc={p.returncode}")
    say(f"=== {model}: {rows(model)}/{EXPECTED} rows ===")

def main():
    old = int(sys.argv[1]) if len(sys.argv) > 1 else None
    if old:
        say(f"waiting for old scheduler pid {old} to drain its 5 running jobs")
        while True:
            try: os.kill(old, 0)
            except OSError: break
            time.sleep(30)
        say("old scheduler exited")

    sys.path.insert(0, str(LOGS)); import schedule
    slow = {"qwen3-4b-base-FC", "qwen3-4b-instruct-FC"}
    # Repeat until nothing is left incomplete. Models that were already running
    # under an earlier driver get skipped on the first pass, so a second sweep
    # (after they finish on their own) is what actually closes them out.
    for sweep in (1, 2, 3):
        todo = [m for m in schedule.queue() if m not in slow and rows(m) < EXPECTED]
        if not todo:
            say(f"PHASE B sweep {sweep}: all SFT checkpoints complete")
            break
        say(f"PHASE B sweep {sweep}: {len(todo)} incomplete -> {todo}")
        run_pool(todo)
        # let any externally-running job finish before the next sweep re-checks
        while running_models():
            say(f"  waiting on externally-running: {sorted(running_models())}")
            time.sleep(60)

    say("PHASE C: the two no-stop-token models, sharded")
    for m in ("qwen3-4b-base-FC", "qwen3-4b-instruct-FC"):
        if rows(m) < EXPECTED: run_sharded(m)

    say("PHASE D: dedupe then evaluate")
    dedupe_results()
    models = ",".join(m for m in schedule.queue() if rows(m) > 0)
    say(f"evaluating {len(models.split(','))} models")
    env = dict(os.environ); env["PATH"] = f"{ROOT}/.venv/bin:" + env["PATH"]
    with open(LOGS / "evaluate.log", "w") as lg:
        subprocess.run([BIN, "evaluate", "--model", models, "--test-category", CATS],
                       cwd=ROOT, env=env, stdout=lg, stderr=subprocess.STDOUT)
    say("EVALUATION COMPLETE")

if __name__ == "__main__":
    main()
