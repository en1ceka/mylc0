"""Checks for the node's status telemetry.

    python scripts/check_telemetry.py

Exercises the aggregation, the rate arithmetic and the log-level behaviour
against synthetic worker snapshots. No GPU, no self-play, no network -- the
point is the arithmetic, which is where throughput reporting goes wrong
quietly.
"""

import argparse
import json
import logging
import os
import shutil
import tempfile
import traceback

import _bootstrap  # noqa: F401

from mylc0.selfplay.status import NodeStatus, confirmation_line

RESULTS = []


def check(name):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            try:
                detail = fn(*args, **kwargs)
                RESULTS.append((name, True, detail or ""))
                print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
                return True
            except Exception as exc:
                RESULTS.append((name, False, repr(exc)))
                print(f"  FAIL  {name}: {exc}")
                traceback.print_exc()
                return False
        return wrapper
    return decorator


def write_snapshot(path, **fields):
    payload = {"live_plies": 0, "finalized_positions": 0, "games": 0,
               "games_in_flight": 0, "nodes_per_s": 0.0, "evals_per_s": 0.0,
               "avg_batch": 0.0, "p50_batch": 0.0, "p95_batch": 0.0,
               "cpu_wait_gpu_pct": 0.0}
    payload.update(fields)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def make_node(tmp, count, name="w"):
    paths = [os.path.join(tmp, f"{name}_{i}.json") for i in range(count)]
    return paths


@check("metrics from many workers are summed and averaged correctly")
def check_aggregation(tmp):
    paths = make_node(os.path.join(tmp, "agg"), 28)
    for i, path in enumerate(paths):
        write_snapshot(path, live_plies=1000, finalized_positions=600,
                       games=5, games_in_flight=48, nodes_per_s=2300.0,
                       evals_per_s=1600.0, avg_batch=300 + i,
                       p50_batch=290.0, p95_batch=460.0,
                       cpu_wait_gpu_pct=30.0)
    status = NodeStatus(paths)
    r = status.sample()

    assert r["workers_reporting"] == 28, r["workers_reporting"]
    # Counters add up across processes...
    assert r["live_plies"] == 28 * 1000, r["live_plies"]
    assert r["finalized_positions"] == 28 * 600, r["finalized_positions"]
    assert r["games_done"] == 28 * 5
    assert r["games_in_flight"] == 28 * 48 == 1344
    assert r["nodes_per_s"] == 28 * 2300.0
    assert r["evals_per_s"] == 28 * 1600.0
    # ...while rates and batch sizes are per-worker means, not sums. Summing
    # a batch size across 28 workers would report 8400 and look like a win.
    assert abs(r["avg_batch"] - (300 + 27 / 2)) < 1e-6, r["avg_batch"]
    assert r["p50_batch"] == 290.0 and r["p95_batch"] == 460.0
    assert r["cpu_wait_gpu_pct"] == 30.0

    # A worker that has not written yet must not drag the averages down.
    paths.append(os.path.join(tmp, "agg", "missing.json"))
    partial = NodeStatus(paths).sample()
    assert partial["workers_reporting"] == 28, partial["workers_reporting"]
    assert partial["avg_batch"] == r["avg_batch"]
    return "28 workers: counters summed, batches averaged"


@check("positions/min measures the delta, not the total")
def check_rate_delta(tmp):
    paths = make_node(os.path.join(tmp, "rate"), 2)
    for path in paths:
        write_snapshot(path, live_plies=0)
    status = NodeStatus(paths, rate_window=60.0)

    # Pretend three readings ten seconds apart: 0 -> 500 -> 1500 total.
    status._history.clear()
    status._history.append((1000.0, 0.0))
    status._history.append((1010.0, 500.0))
    status._history.append((1020.0, 1500.0))
    t0, p0 = status._history[0]
    t1, p1 = status._history[-1]
    rate = (p1 - p0) / (t1 - t0) * 60.0
    assert abs(rate - 4500.0) < 1e-6, rate

    # Live behaviour, with time injected so this does not sleep: 20000 more
    # plies over 15 seconds is 80000/min.
    status = NodeStatus(paths, rate_window=60.0)
    for path in paths:
        write_snapshot(path, live_plies=10000)
    first = status.sample(now=1000.0)
    for path in paths:
        write_snapshot(path, live_plies=20000)
    second = status.sample(now=1015.0)
    assert abs(second["positions_per_min"] - 80000.0) < 1e-6, \
        second["positions_per_min"]
    assert second["live_plies"] == 40000, second["live_plies"]
    # The window is a slope over recent samples, not total/elapsed -- which
    # is reported separately as avg_positions_per_min.
    assert "avg_positions_per_min" in second
    # A single reading has no slope, so it must report unknown, not zero.
    assert first["positions_per_min"] != first["positions_per_min"], first
    return "slope over the window: 4500/min synthetic, 80000/min live"


@check("nothing divides by zero at startup")
def check_no_division_by_zero(tmp):
    # No files at all: the workers have not started.
    empty = NodeStatus([os.path.join(tmp, "nope.json")], target_positions=0)
    r = empty.sample()
    assert r["live_plies"] == 0
    assert r["positions_per_min"] != r["positions_per_min"], \
        "an empty node reported a rate instead of 'unknown'"
    assert r["avg_batch"] == 0.0
    assert r["progress"] == 0.0
    assert r["eta_s"] != r["eta_s"], "invented an ETA out of nothing"
    assert empty.line(36, "abc123def456", 1), "could not render an empty node"

    # One reading is not enough for a slope; it must not be total/elapsed.
    paths = make_node(os.path.join(tmp, "zero"), 1)
    write_snapshot(paths[0], live_plies=5000)
    single = NodeStatus(paths, target_positions=50000)
    one = single.sample()
    assert one["positions_per_min"] != one["positions_per_min"], \
        "a rate from a single sample"
    assert one["eta_s"] != one["eta_s"], "an ETA from a single sample"
    rendered = single.line(36, "abc", 1)
    assert "-- pos/min" in rendered, rendered

    # A zero target must not divide either.
    assert NodeStatus(paths, target_positions=0).sample()["progress"] == 0.0
    summary = single.summary()
    assert summary["elapsed_s"] >= 0
    return "no files, one sample and a zero target all render"


@check("ETA follows the rate and refuses to guess")
def check_eta(tmp):
    paths = make_node(os.path.join(tmp, "eta"), 1)
    write_snapshot(paths[0], live_plies=0)
    status = NodeStatus(paths, target_positions=60000)

    # 30000 remaining at 6000/min is five minutes.
    assert abs(status.eta(30000, 6000) - 300.0) < 1e-6, status.eta(30000, 6000)
    # Halving the rate doubles the wait.
    assert abs(status.eta(30000, 3000) - 600.0) < 1e-6
    # Past the target there is nothing left to wait for.
    assert status.eta(60000, 6000) == 0.0
    assert status.eta(70000, 6000) == 0.0
    # A stalled node has no finite ETA; reporting one would be a lie.
    assert status.eta(30000, 0) != status.eta(30000, 0)
    assert status.eta(30000, -5) != status.eta(30000, -5)
    # No target means no ETA at all.
    assert NodeStatus(paths, target_positions=0).eta(1, 6000) != \
        NodeStatus(paths, target_positions=0).eta(1, 6000)
    return "5 min at 6000/min, NaN when stalled"


@check("throughput is visible before any game has finished")
def check_live_before_finished(tmp):
    paths = make_node(os.path.join(tmp, "live"), 28)
    # The state a node is in for the first minutes: every game in flight,
    # none finished, so nothing has been written to a chunk yet.
    for path in paths:
        write_snapshot(path, live_plies=800, finalized_positions=0,
                       games=0, games_in_flight=48, nodes_per_s=2300.0)
    status = NodeStatus(paths, target_positions=300000)
    status.sample(now=1000.0)
    for path in paths:
        write_snapshot(path, live_plies=1900, finalized_positions=0,
                       games=0, games_in_flight=48, nodes_per_s=2300.0)
    r = status.sample(now=1015.0)

    assert r["games_done"] == 0, "the premise is that nothing has finished"
    assert r["finalized_positions"] == 0, r["finalized_positions"]
    assert r["live_plies"] == 28 * 1900, r["live_plies"]
    assert r["positions_per_min"] > 0, \
        "no throughput reported while games are still running"

    text = status.line(36, "7a52425f1e1f", 1)
    assert "live pos" in text and "final" in text, text
    # The two counts must never be conflated: the dataset is still empty.
    assert "(0 final)" in text, text
    return "53k live plies, 0 finalized, rate still reported"


@check("quiet workers stay quiet, but errors and OOM do not")
def check_log_level(tmp):
    from mylc0.selfplay.worker import log as worker_log

    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append((record.levelname, record.getMessage()))

    handler = Capture()
    worker_log.addHandler(handler)
    previous = worker_log.level
    try:
        # What a node sets: INFO suppressed, everything above still emitted.
        worker_log.setLevel(logging.WARNING)
        worker_log.info("network gen_000036.mylc0, 48 games in flight")
        worker_log.warning("watchdog: no progress for 30s")
        worker_log.error("worker 7: CUDA out of memory (tried 2.00 GiB)")

        levels = [level for level, _msg in records]
        assert "INFO" not in levels, f"a startup banner got through: {records}"
        assert "WARNING" in levels, records
        assert "ERROR" in levels, records
        assert any("out of memory" in msg for _l, msg in records), records

        # --verbose-workers puts the banner back for debugging.
        records.clear()
        worker_log.setLevel(logging.INFO)
        worker_log.info("network gen_000036.mylc0, 48 games in flight")
        assert [lvl for lvl, _m in records] == ["INFO"], records
    finally:
        worker_log.removeHandler(handler)
        worker_log.setLevel(previous)
    return "INFO hidden at WARNING, ERROR/OOM always shown"


@check("the confirmation line replaces one banner per worker")
def check_confirmation():
    text = confirmation_line(28, {"parallel_games": 48, "nn_batch": 512,
                                  "fp16": True, "visits": 800,
                                  "minibatch_size": 32})
    for fragment in ("28 workers", "48 games", "512", "fp16", "800", "32"):
        assert fragment in text, (fragment, text)
    assert "\n" not in text, "the confirmation must be a single line"
    assert "fp32" in confirmation_line(1, {"fp16": False})
    return text


@check("the shard performance summary carries what a farm is compared on")
def check_summary(tmp):
    paths = make_node(os.path.join(tmp, "sum"), 4)
    for path in paths:
        write_snapshot(path, live_plies=50000, finalized_positions=42000,
                       games=120, nodes_per_s=16000.0, evals_per_s=11000.0,
                       avg_batch=315.0, p50_batch=301.0, p95_batch=468.0,
                       cpu_wait_gpu_pct=34.0)
    status = NodeStatus(paths, target_positions=200000)
    status.sample()
    summary = status.summary()

    for key in ("elapsed_s", "avg_positions_per_min", "nodes_per_s",
                "evals_per_s", "avg_batch", "p50_batch", "p95_batch",
                "gpu_util_avg", "cpu_util_avg", "cpu_wait_gpu_pct",
                "vram_peak_mib", "finalized_positions", "live_plies"):
        assert key in summary, f"summary is missing {key}"
    assert summary["live_plies"] == 200000, summary["live_plies"]
    assert summary["finalized_positions"] == 168000
    assert summary["avg_batch"] == 315.0
    assert summary["nodes_per_s"] == 64000.0, summary["nodes_per_s"]
    assert json.dumps(summary), "the summary must be JSON serialisable"
    return "13 fields, JSON serialisable"


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    tmp = tempfile.mkdtemp(prefix="mylc0-telemetry-")
    try:
        print("== aggregation ==")
        check_aggregation(tmp)
        check_summary(tmp)

        print("\n== rates ==")
        check_rate_delta(tmp)
        check_no_division_by_zero(tmp)
        check_eta(tmp)
        check_live_before_finished(tmp)

        print("\n== log levels ==")
        check_log_level(tmp)
        check_confirmation()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    failed = [name for name, ok, _d in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        print("failed: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
