"""Find the fastest self-play configuration for *this* machine.

    python scripts/tune_selfplay.py --config configs/small.yaml \
        --network networks/latest.mylc0

It detects the hardware, runs a short benchmark per candidate configuration,
and prints the best one together with the measured bottleneck. The search
knobs of the algorithm are never touched: visits, ``search.minibatch_size``,
the network, the input planes and the heads are exactly what the config says.
Only implementation parameters are swept:

    workers          how many self-play processes
    parallel_games   games in flight inside one process
    nn_batch         upper bound on one network call (NOT minibatch_size)
    torch_threads    intra-op threads per worker
    affinity         pin each worker to a block of CPUs

Stages are run in that order, each keeping the winner of the previous one, so
the cost is linear rather than a full grid.
"""

import argparse
import json
import multiprocessing as mp
import os
import shutil
import statistics
import tempfile
import time

import _bootstrap  # noqa: F401

from mylc0.net.config import load_config
from mylc0.perf import (detect_cpu, detect_gpu, diagnose, limit_thread_pools,
                        runnable_threads, sample_cpu, sample_gpu, usable_cpus)
from mylc0.selfplay.worker import run_worker


def _sampler(stop, gpu_samples, cpu_samples, thread_samples, pids,
             interval=0.5):
    import threading
    def loop():
        while not stop.is_set():
            g = sample_gpu()
            if g:
                gpu_samples.append(g)
            total, _cores = sample_cpu(per_core=False)
            if total is not None:
                cpu_samples.append(total)
            thread_samples.append(runnable_threads(pids))
            stop.wait(interval)
    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return thread


def run_trial(args, workers, parallel_games, nn_batch, torch_threads,
              affinity, seconds, warmup):
    """One measured configuration. Returns a dict of metrics."""
    import threading

    workdir = tempfile.mkdtemp(prefix="mylc0-tune-")
    perf_paths = [os.path.join(workdir, f"perf_{i}.json")
                  for i in range(workers)]
    stats_paths = [os.path.join(workdir, f"stats_{i}.json")
                   for i in range(workers)]

    # The knobs live in a config copy so nothing global is mutated.
    config = load_config(args.config)
    config.selfplay.parallel_games = parallel_games
    config.selfplay.batch_size = nn_batch
    config_path = os.path.join(workdir, "config.yaml")
    _dump_config(args.config, config_path, parallel_games, nn_batch)

    common = dict(config_path=config_path, network_path=args.network,
                  output_dir=os.path.join(workdir, "data"), num_games=0,
                  device=args.device, log_every=0,
                  max_seconds=seconds + warmup,
                  heartbeat_seconds=0, watchdog_seconds=0,
                  perf_debug=True, perf_warmup=warmup,
                  torch_threads=torch_threads,
                  affinity=affinity, workers_total=workers)

    ctx = mp.get_context("spawn")
    procs = []
    t0 = time.perf_counter()
    for i in range(workers):
        proc = ctx.Process(target=run_worker, kwargs=dict(
            worker_id=i, seed=1000 + i * 7919, stats_path=stats_paths[i],
            perf_path=perf_paths[i], **common))
        proc.start()
        procs.append(proc)

    # Sample the measurement window only, never the ramp-up.
    pids = [p.pid for p in procs]
    gpu_samples, cpu_samples, thread_samples = [], [], []
    stop = threading.Event()
    deadline = t0 + warmup
    while (time.perf_counter() < deadline
           and any(p.is_alive() for p in procs)):
        time.sleep(0.2)
    _sampler(stop, gpu_samples, cpu_samples, thread_samples, pids)

    for proc in procs:
        proc.join()
    elapsed = time.perf_counter() - t0
    stop.set()

    per_worker = []
    for path in perf_paths:
        try:
            with open(path, encoding="utf-8") as f:
                per_worker.append(json.load(f))
        except (OSError, ValueError):
            pass
    failed = [p.exitcode for p in procs if p.exitcode]
    shutil.rmtree(workdir, ignore_errors=True)

    if not per_worker:
        return {"ok": False, "reason": f"no telemetry (exit codes {failed})",
                "workers": workers, "parallel_games": parallel_games,
                "nn_batch": nn_batch}

    def total(key):
        return sum(w.get(key, 0.0) for w in per_worker)

    def mean(key):
        values = [w.get(key, 0.0) for w in per_worker]
        return statistics.fmean(values) if values else 0.0

    wall = max(w.get("wall_s", elapsed) for w in per_worker)
    gpu_util = (statistics.fmean(s["utilization.gpu"] for s in gpu_samples)
                if gpu_samples else float("nan"))
    # With one process the CUDA events measure GPU busy time directly.
    # With several, each process's events also span the time its kernels
    # waited behind another process, so they would double count; the
    # driver's own utilisation figure is the honest one there.
    if workers == 1:
        forward = sum(w.get("gpu_forward_pct", 0.0) * w.get("wall_s", 0.0)
                      for w in per_worker) / 100.0
        gpu_busy_pct = min(100.0, 100.0 * forward / max(wall, 1e-9))
    else:
        gpu_busy_pct = gpu_util
    vram = (max(s["memory.used"] for s in gpu_samples) / 1024
            if gpu_samples else float("nan"))
    power = (statistics.fmean(s["power.draw"] for s in gpu_samples)
             if gpu_samples else float("nan"))

    return {
        "ok": True,
        "workers": workers, "parallel_games": parallel_games,
        "nn_batch": nn_batch, "torch_threads": torch_threads,
        "affinity": affinity,
        "positions_per_min": total("plies") / max(wall, 1e-9) * 60,
        "nodes_per_s": total("nodes_per_s"),
        "evals_per_s": total("evals_per_s"),
        "avg_batch": mean("avg_batch"), "p50_batch": mean("p50_batch"),
        "p95_batch": mean("p95_batch"), "max_batch": max(
            (w.get("max_batch", 0) for w in per_worker), default=0),
        "batches_per_s": total("batches_per_s"),
        "gpu_busy_pct": gpu_busy_pct,
        "gpu_starvation_pct": max(0.0, 100.0 - gpu_busy_pct),
        "cpu_wait_gpu_pct": mean("cpu_wait_gpu_pct"),
        "cpu_busy_pct": mean("cpu_busy_pct"),
        "gather_pct": mean("gather_pct"), "encode_pct": mean("encode_pct"),
        "terminal_pct": mean("terminal_pct"), "apply_pct": mean("apply_pct"),
        "gpu_util": gpu_util, "vram_gb": vram, "power_w": power,
        "cpu_total": statistics.fmean(cpu_samples) if cpu_samples else float("nan"),
        "threads": max(thread_samples) if thread_samples else 0,
        "games": total("games"), "positions": total("positions"),
        "plies": total("plies"), "wall_s": wall,
        "per_worker": per_worker,
    }


def _dump_config(source, target, parallel_games, nn_batch):
    """Copy the config with only the two implementation knobs overridden."""
    with open(source, encoding="utf-8") as f:
        text = f.read()
    out = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("parallel_games:"):
            line = f"  parallel_games: {parallel_games}"
        elif stripped.startswith("batch_size:") and line.startswith("  batch_size:"):
            line = f"  batch_size: {nn_batch}"
        out.append(line)
    with open(target, "w", encoding="utf-8") as f:
        f.write("\n".join(out))


def config_key(result):
    return (result["workers"], result["parallel_games"], result["nn_batch"],
            bool(result["affinity"]))


def pick_finalists(results, count):
    """Best distinct configurations from the screening sweep."""
    best_by_key = {}
    for r in results:
        if not r.get("ok"):
            continue
        key = config_key(r)
        if (key not in best_by_key
                or r["positions_per_min"] > best_by_key[key]["positions_per_min"]):
            best_by_key[key] = r
    ordered = sorted(best_by_key.values(),
                     key=lambda r: -r["positions_per_min"])
    return ordered[:count]


def complexity(result):
    """Lower is simpler.

    Only the process count counts as complexity: every extra worker is another
    OS process, another copy of the weights in VRAM and another thing that can
    die mid-run. ``parallel_games`` and ``nn_batch`` live inside one process and
    cost nothing structurally, so they are not a tie-break -- among equally
    simple configurations the faster one wins.
    """
    return (result["workers"], -result["positions_per_min"])


def choose_winner(results, tolerance=0.03):
    """Fastest configuration, but prefer a simpler one within ``tolerance``.

    A few percent is inside run-to-run noise, and every extra process costs
    memory, a network copy and one more thing that can go wrong -- so a tie is
    resolved towards the simpler setup rather than the nominally fastest one.
    """
    usable = [r for r in results if r.get("ok")]
    if not usable:
        return None, []
    fastest = max(usable, key=lambda r: r["positions_per_min"])
    threshold = fastest["positions_per_min"] * (1.0 - tolerance)
    contenders = [r for r in usable if r["positions_per_min"] >= threshold]
    winner = min(contenders, key=complexity)
    return winner, contenders


def final_table(results, winner):
    lines = []
    head = (f"  {'configuration':<26} {'pos/min':>8} {'nodes/s':>8} "
            f"{'evals/s':>8} {'GPU':>5} {'CPU':>5} {'avg':>5} {'p50':>5} "
            f"{'p95':>5} {'starv':>6} {'waitGPU':>8}")
    lines.append(head)
    lines.append("  " + "-" * (len(head) - 2))
    for r in results:
        if not r.get("ok"):
            lines.append(f"  {'(failed)':<26} {r.get('reason', '')}")
            continue
        label = (f"w{r['workers']} g{r['parallel_games']} b{r['nn_batch']}"
                 + (" aff" if r["affinity"] else ""))
        mark = " <-- winner" if r is winner else ""
        lines.append(
            f"  {label:<26} {r['positions_per_min']:>8.0f} "
            f"{r['nodes_per_s']:>8.0f} {r['evals_per_s']:>8.0f} "
            f"{r['gpu_util']:>4.0f}% {r['cpu_total']:>4.0f}% "
            f"{r['avg_batch']:>5.0f} {r['p50_batch']:>5.0f} "
            f"{r['p95_batch']:>5.0f} {r['gpu_starvation_pct']:>5.0f}% "
            f"{r['cpu_wait_gpu_pct']:>7.0f}%{mark}")
    return "\n".join(lines)


def show(result, label=""):
    if not result.get("ok"):
        print(f"  {label:<28s} FAILED: {result.get('reason')}")
        return
    print(f"  {label:<28s} {result['positions_per_min']:>7.0f} pos/min "
          f"| {result['evals_per_s']:>7.0f} evals/s "
          f"| batch {result['avg_batch']:>5.0f} "
          f"| GPU {result['gpu_util']:>3.0f}% busy {result['gpu_busy_pct']:>3.0f}% "
          f"| CPU {result['cpu_total']:>3.0f}% "
          f"| waitGPU {result['cpu_wait_gpu_pct']:>3.0f}%")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/small.yaml")
    parser.add_argument("--network", default="networks/latest.mylc0")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seconds", type=float, default=90.0,
                        help="measured window per trial, after warm-up")
    parser.add_argument("--warmup", type=float, default=25.0,
                        help="seconds discarded before measuring")
    parser.add_argument("--workers", type=int, nargs="*", default=None)
    parser.add_argument("--parallel-games", type=int, nargs="*", default=None)
    parser.add_argument("--nn-batch", type=int, nargs="*", default=None)
    parser.add_argument("--quick", action="store_true",
                        help="30s trials, fewer candidates")
    parser.add_argument("--json", default=None, help="write all results")
    parser.add_argument("--final-top", type=int, default=3,
                        help="configurations to confirm on a long run")
    parser.add_argument("--final-seconds", type=float, default=300.0,
                        help="measured window of each final run")
    parser.add_argument("--final-warmup", type=float, default=30.0)
    parser.add_argument("--final-from", default=None,
                        help="take candidates from a previous --json "
                             "instead of sweeping again")
    parser.add_argument("--tolerance", type=float, default=0.03,
                        help="within this margin, prefer the simpler "
                             "configuration")
    args = parser.parse_args()

    limit_thread_pools(1)
    if args.quick:
        args.seconds = min(args.seconds, 30.0)
        args.warmup = min(args.warmup, 15.0)

    gpu = detect_gpu()
    cpu = detect_cpu()
    cpus = usable_cpus()
    print("Detected hardware")
    print(f"  GPU              {gpu.get('name')} "
          f"{gpu.get('vram_gb')} GB  (CUDA {gpu.get('cuda')}, "
          f"torch {gpu.get('torch')})")
    print(f"  CPU              {cpu.get('model')}")
    print(f"  usable CPUs      {cpus} (logical {cpu.get('logical')}, "
          f"physical {cpu.get('physical')})")
    print(f"  RAM              {cpu.get('ram_gb')} GB")
    config = load_config(args.config)
    print(f"  config           visits={config.selfplay.visits} "
          f"minibatch={config.selfplay.search.minibatch_size} "
          f"(both fixed, never tuned)")
    if not os.path.isfile(args.network):
        print(f"\nnetwork not found: {args.network}")
        return 1
    print(f"  network          {args.network}")
    print(f"  trial length     {args.seconds:.0f}s\n")

    results = []
    vram_gb = gpu.get("vram_gb") or 8

    if args.final_from:
        with open(args.final_from, encoding="utf-8") as handle:
            results = json.load(handle).get("results", [])
        print(f"screening skipped: {len(results)} earlier trials read from "
              f"{args.final_from}")
        return run_finals(args, results, gpu, cpu, cpus)

    # -- stage 1: workers -------------------------------------------------
    candidates = args.workers or sorted({1, 2, max(2, cpus // 4),
                                         max(3, cpus // 2), max(4, cpus - 2)})
    candidates = [w for w in candidates if 1 <= w <= max(1, cpus)]
    base_games = config.selfplay.parallel_games
    base_batch = config.selfplay.batch_size
    print(f"stage 1/4  workers (parallel_games={base_games}, "
          f"nn_batch={base_batch})")
    best = None
    for workers in candidates:
        r = run_trial(args, workers, base_games, base_batch, 1, False,
                      args.seconds, args.warmup)
        results.append(r)
        show(r, f"workers={workers}")
        if r.get("ok") and (best is None
                            or r["positions_per_min"] > best["positions_per_min"]):
            best = r
    if best is None:
        print("every trial failed; see the messages above")
        return 1
    workers = best["workers"]

    # -- stage 2: parallel games -----------------------------------------
    games_candidates = args.parallel_games or [4, 8, 16, 24]
    print(f"\nstage 2/4  parallel_games (workers={workers})")
    for games in games_candidates:
        if games == base_games:
            show(best, f"parallel_games={games}")
            continue
        r = run_trial(args, workers, games, base_batch, 1, False,
                      args.seconds, args.warmup)
        results.append(r)
        show(r, f"parallel_games={games}")
        if r.get("ok") and r["positions_per_min"] > best["positions_per_min"]:
            best = r
    games = best["parallel_games"]

    # -- stage 3: NN batch ------------------------------------------------
    batch_candidates = args.nn_batch or [b for b in (128, 256, 512, 1024)
                                         if b <= 256 * max(1, int(vram_gb // 4))]
    print(f"\nstage 3/4  nn_batch (workers={workers}, parallel_games={games})")
    for batch in batch_candidates:
        if batch == best["nn_batch"]:
            show(best, f"nn_batch={batch}")
            continue
        r = run_trial(args, workers, games, batch, 1, False, args.seconds, args.warmup)
        results.append(r)
        show(r, f"nn_batch={batch}")
        if r.get("ok") and r["positions_per_min"] > best["positions_per_min"]:
            best = r

    # -- stage 4: affinity ------------------------------------------------
    print("\nstage 4/4  CPU affinity")
    r = run_trial(args, best["workers"], best["parallel_games"],
                  best["nn_batch"], 1, True, args.seconds, args.warmup)
    results.append(r)
    show(r, "affinity=on")
    if r.get("ok") and r["positions_per_min"] > best["positions_per_min"] * 1.02:
        best = r        # only keep it if it clearly helps

    return run_finals(args, results, gpu, cpu, cpus)


def run_finals(args, results, gpu, cpu, cpus):
    """Confirm the best screened configurations on a long, quiet window."""
    finalists = pick_finalists(results, args.final_top)
    if not finalists:
        print("nothing to confirm")
        return 1

    print("\n" + "=" * 78)
    print(f"final benchmark: top {len(finalists)} configuration(s), "
          f"{args.final_warmup:.0f}s warm-up + {args.final_seconds:.0f}s "
          f"measured each")
    print("=" * 78)
    for r in finalists:
        print(f"  candidate  w{r['workers']} g{r['parallel_games']} "
              f"b{r['nn_batch']}{' aff' if r['affinity'] else ''}"
              f"   (screened at {r['positions_per_min']:.0f} pos/min)")

    finals = []
    for candidate in finalists:
        label = (f"w{candidate['workers']} g{candidate['parallel_games']} "
                 f"b{candidate['nn_batch']}"
                 + (" aff" if candidate["affinity"] else ""))
        print(f"\n  running {label} ...", flush=True)
        r = run_trial(args, candidate["workers"], candidate["parallel_games"],
                      candidate["nn_batch"], 1, candidate["affinity"],
                      args.final_seconds, args.final_warmup)
        finals.append(r)
        show(r, label)

    winner, contenders = choose_winner(finals, args.tolerance)
    if winner is None:
        print("every final run failed")
        return 1

    print("\n" + "=" * 78)
    print("FINAL BENCHMARK")
    print("=" * 78)
    print(final_table(finals, winner))

    fastest = max((r for r in finals if r.get("ok")),
                  key=lambda r: r["positions_per_min"])
    if winner is not fastest:
        gap = 100 * (1 - winner["positions_per_min"]
                     / fastest["positions_per_min"])
        print(f"\n  The fastest run was w{fastest['workers']} "
              f"g{fastest['parallel_games']} b{fastest['nn_batch']} at "
              f"{fastest['positions_per_min']:.0f} pos/min, but the winner is "
              f"within {gap:.1f}% ({args.tolerance * 100:.0f}% tolerance) and "
              f"uses fewer processes.")

    print("\nWinner")
    print(f"  workers          {winner['workers']}")
    print(f"  parallel_games   {winner['parallel_games']}")
    print(f"  nn_batch         {winner['nn_batch']}")
    print("  precision        fp16")
    print(f"  CPU affinity     {'yes' if winner['affinity'] else 'no'}")
    print(f"  torch threads    {winner['torch_threads']}")
    print("\nMeasured over "
          f"{winner['wall_s']:.0f}s ({winner['plies']:.0f} positions, "
          f"{winner['games']:.0f} games finished)")
    print(f"  positions/min    {winner['positions_per_min']:.0f}")
    print(f"  nodes/s          {winner['nodes_per_s']:.0f}")
    print(f"  evals/s          {winner['evals_per_s']:.0f}")
    print(f"  avg/p50/p95 batch{winner['avg_batch']:>5.0f} /"
          f"{winner['p50_batch']:>4.0f} /{winner['p95_batch']:>4.0f}")
    print(f"  GPU util         {winner['gpu_util']:.0f}%   "
          f"VRAM {winner['vram_gb']:.1f} GB   {winner['power_w']:.0f} W")
    print(f"  CPU util         {winner['cpu_total']:.0f}%")
    print(f"  GPU starvation   {winner['gpu_starvation_pct']:.0f}%")
    print(f"  CPU wait for GPU {winner['cpu_wait_gpu_pct']:.0f}%")
    print("\nWhere a worker's time goes")
    for text, key in (("MCTS gather", "gather_pct"),
                      ("terminal / legal moves", "terminal_pct"),
                      ("input encoding", "encode_pct"),
                      ("apply + backup", "apply_pct"),
                      ("blocked on GPU", "cpu_wait_gpu_pct")):
        print(f"  {text:<24s} {winner.get(key, 0):>5.1f}%")

    alerts = diagnose(winner, {"utilization.gpu": winner["gpu_util"]},
                      winner["cpu_total"], winner["threads"], cpus)
    if alerts:
        print()
        for alert in alerts:
            print(alert)

    print("\nPut this in your config:")
    print(f"  selfplay:\n    parallel_games: {winner['parallel_games']}\n"
          f"    batch_size: {winner['nn_batch']}")
    print(f"and run: python scripts/loop.py --workers {winner['workers']}"
          + ("  (with CPU affinity)" if winner["affinity"] else ""))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"gpu": gpu, "cpu": cpu, "usable_cpus": cpus,
                       "results": results, "finals": finals,
                       "winner": winner}, f, indent=1)
        print(f"\nall trials written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
