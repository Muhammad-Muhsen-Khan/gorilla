#!/usr/bin/env python3
"""Round-robin BFCL v4 generation across all 8 GPUs.

Order: epoch 1 no-reason, epoch 1 reasoning, Qwen3-4B instruct, Qwen3-4B base,
then epoch 2 no-reason, epoch 2 reasoning, epoch 3 ... through epoch 10.
Eight slots, one GPU each; as a slot frees, the next model starts.

Each slot gets its OWN vLLM port. `.env` must not define LOCAL_SERVER_PORT --
bfcl calls load_dotenv(override=True), so a value there would clobber these
and route every run at one server.
"""
import json, os, subprocess, time, pathlib, sys

ROOT = pathlib.Path("/home/ubuntu/gorilla/berkeley-function-call-leaderboard")
LOGS = ROOT / "runlogs"
CATS = ("simple_python,simple_java,simple_javascript,multiple,parallel,parallel_multiple,"
        "irrelevance,live_simple,live_multiple,live_parallel,live_parallel_multiple,"
        "live_irrelevance,live_relevance,multi_turn_base,multi_turn_miss_func,"
        "multi_turn_miss_param,multi_turn_long_context,memory_kv,memory_vector,memory_rec_sum")
EXPECTED = 5017          # total generated rows incl. memory prereqs
NGPU, BASE_PORT = 8, 1053

def queue():
    q = ["qwen3-4b-sft-noreason-64-FC", "qwen3-4b-sft-reasoning-86-FC",
         "qwen3-4b-instruct-FC", "qwen3-4b-base-FC"]
    for epoch in range(2, 11):
        q.append(f"qwen3-4b-sft-noreason-{64*epoch}-FC")
        q.append(f"qwen3-4b-sft-reasoning-{86*epoch}-FC")
    return q

def rows(model):
    d = ROOT / "result" / model
    return sum(sum(1 for _ in open(f)) for f in d.rglob("*_result.json")) if d.is_dir() else 0

def launch(model, gpu):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["LOCAL_SERVER_PORT"] = str(BASE_PORT + gpu)
    env["PATH"] = f"{ROOT}/.venv/bin:" + env["PATH"]
    log = open(LOGS / f"gen_{model}.log", "w")
    p = subprocess.Popen(
        [f"{ROOT}/.venv/bin/bfcl", "generate", "--model", model,
         "--test-category", CATS, "--backend", "vllm", "--num-gpus", "1",
         "--gpu-memory-utilization", "0.9", "--num-threads", "16"],
        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    return p, log

def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def main():
    pending = [m for m in queue() if rows(m) < EXPECTED]
    for m in queue():
        if m not in pending:
            say(f"SKIP {m} (already has {rows(m)} rows)")
    say(f"{len(pending)} models to generate on {NGPU} GPUs")

    slots = [None] * NGPU          # (model, proc, logfile, t0)
    started = {}
    while pending or any(slots):
        for i, s in enumerate(slots):
            if s is None and pending:
                m = pending.pop(0)
                p, log = launch(m, i)
                slots[i] = (m, p, log, time.time())
                started[m] = time.time()
                say(f"GPU{i} START {m}   ({len(pending)} left in queue)")
        time.sleep(20)
        for i, s in enumerate(slots):
            if s is None: continue
            m, p, log, t0 = s
            if p.poll() is None: continue
            log.close()
            mins, n = (time.time() - t0) / 60, rows(m)
            ok = "OK" if n >= EXPECTED else f"INCOMPLETE {n}/{EXPECTED}"
            say(f"GPU{i} DONE  {m}  rc={p.returncode}  {mins:.1f} min  {ok}")
            slots[i] = None
    say("ALL GENERATIONS FINISHED")

if __name__ == "__main__":
    main()
