"""Regression checks for how self-play knobs reach a worker.

    python scripts/check_selfplay_config.py

``--workers``, ``--parallel-games`` and ``--nn-batch`` travel from the command
line through a rewritten YAML file, across a process boundary and into the
driver constructor. A break anywhere along that chain still produces correct
games -- just far fewer of them -- so it does not show up as a failure, only
as a throughput number nobody is watching.

This walks the same path ``scripts/selfplay_node.py`` uses and asserts the
values that actually arrive. No GPU, no self-play, no network.

Written after 28 workers x 48 games silently ran at 1 game each.
"""

import argparse
import importlib.util
import json
import os
import shutil
import tempfile
import traceback

import _bootstrap  # noqa: F401

from mylc0.net.config import load_config
from mylc0.selfplay.worker import (effective_parallel_games,
                                   write_runtime_config)

RESULTS = []
CONFIG = "configs/small.yaml"


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


def load_script(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(os.path.dirname(__file__), f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@check("CLI overrides survive the config rewrite the node uses")
def check_config_rewrite(tmp):
    node = load_script("selfplay_node")
    target = os.path.join(tmp, "run_config.yaml")
    node._dump_config(CONFIG, target, 48, 512)

    effective = load_config(target)
    assert effective.selfplay.parallel_games == 48, \
        f"parallel_games arrived as {effective.selfplay.parallel_games}"
    assert effective.selfplay.batch_size == 512, \
        f"selfplay batch_size arrived as {effective.selfplay.batch_size}"

    # Nothing else may move: these are algorithm parameters, not knobs.
    baseline = load_config(CONFIG)
    assert effective.selfplay.visits == baseline.selfplay.visits == 800
    assert (effective.selfplay.search.minibatch_size
            == baseline.selfplay.search.minibatch_size == 32)
    assert effective.selfplay.fp16 == baseline.selfplay.fp16
    assert effective.selfplay.max_game_ply == baseline.selfplay.max_game_ply
    return "g48 b512, visits/minibatch/fp16 untouched"


@check("--nn-batch does not reach into the training section")
def check_batch_size_section(tmp):
    node = load_script("selfplay_node")
    tuner = load_script("tune_selfplay")
    baseline = load_config(CONFIG)
    # batch_size exists under both training: and selfplay: at the same
    # indentation, so a key-only match rewrites the trainer's too.
    assert baseline.training.batch_size == 256, baseline.training.batch_size

    for label, dump in (("selfplay_node", node._dump_config),
                        ("tune_selfplay", tuner._dump_config)):
        target = os.path.join(tmp, f"cfg_{label}.yaml")
        dump(CONFIG, target, 48, 512)
        result = load_config(target)
        assert result.selfplay.batch_size == 512, (label, "selfplay")
        assert result.training.batch_size == 256, (
            f"{label} rewrote training.batch_size to "
            f"{result.training.batch_size}")
    return "selfplay 512, training still 256, in both scripts"


@check("a node's shard target never scales parallel_games down")
def check_no_clamp_for_nodes():
    cfg = load_config(CONFIG).selfplay
    cfg.parallel_games = 48
    # The exact case that broke: 2000 positions over 28 workers is 72 each,
    # far less than one game, which used to collapse 48 games to 1.
    for target in (72, 1786, 0, 1, 10 ** 9):
        got = effective_parallel_games(cfg, target, scale_to_target=False)
        assert got == 48, f"target={target} gave {got} games in flight"
    return "48 games kept at every shard target"


@check("loop.py's exact quota still scales parallel_games down")
def check_clamp_preserved_for_quota():
    cfg = load_config(CONFIG).selfplay
    cfg.parallel_games = 8
    typical = cfg.max_game_ply // 2      # 225

    # A tiny quota must not start 8 games it cannot absorb.
    assert effective_parallel_games(cfg, 300, True) == 1, "clamp is gone"
    assert effective_parallel_games(cfg, typical * 4, True) == 4
    # A quota with room keeps everything.
    assert effective_parallel_games(cfg, typical * 8, True) == 8
    assert effective_parallel_games(cfg, 10 ** 6, True) == 8
    # No target at all means no scaling.
    assert effective_parallel_games(cfg, 0, True) == 8
    return "unchanged for loop.py"


@check("the reported floor matches what a shard can actually be")
def check_overshoot_floor():
    node = load_script("selfplay_node")
    cfg = load_config(CONFIG).selfplay
    floor = node.overshoot_floor(28, 48, cfg.max_game_ply)
    assert floor == 28 * 48 * 225, floor
    # The command that produced the bug report asked for 2000.
    assert 2000 < floor, "the floor should have flagged --shard-positions 2000"
    # A configuration meant for a short test is under it.
    assert node.overshoot_floor(2, 4, cfg.max_game_ply) == 1800
    return f"w28 g48 -> {floor} positions minimum"


@check("a worker publishes the configuration it really started with")
def check_runtime_config_published(tmp):
    cfg = load_config(CONFIG).selfplay
    cfg.parallel_games = 48
    cfg.batch_size = 512
    path = os.path.join(tmp, "runtime.json")
    write_runtime_config(path, cfg, parallel=48, device="cuda", fp16=True,
                         generation=36)
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["parallel_games"] == 48, payload
    assert payload["nn_batch"] == 512, payload
    assert payload["fp16"] is True, payload
    assert payload["visits"] == 800, payload
    assert payload["minibatch_size"] == 32, payload
    return "48 / 512 / fp16 / 800 visits reported"


@check("the node aborts when a worker reports the wrong configuration")
def check_fail_fast(tmp):
    node = load_script("selfplay_node")
    cfg = load_config(CONFIG).selfplay
    cfg.parallel_games = 48
    cfg.batch_size = 512
    expected = {"parallel_games": 48, "nn_batch": 512, "visits": 800,
                "minibatch_size": 32, "fp16": True}

    good = os.path.join(tmp, "ok_0.json")
    write_runtime_config(good, cfg, 48, "cuda", True, 36)
    seen = node.verify_runtime_config([good], expected, timeout=2.0)
    assert len(seen) == 1, seen

    # The exact regression: the driver got 1 game instead of 48.
    bad = os.path.join(tmp, "bad_0.json")
    write_runtime_config(bad, cfg, 1, "cuda", True, 36)
    try:
        node.verify_runtime_config([bad], expected, timeout=2.0)
    except node.ConfigMismatch as exc:
        assert "parallel_games is 1" in str(exc), str(exc)
    else:
        raise AssertionError("accepted 1 game in flight where 48 was asked")

    # And the analogous failures for the other knobs.
    for field, wrong in (("nn_batch", 256), ("fp16", False), ("visits", 400)):
        broken = dict(expected)
        broken[field] = wrong
        try:
            node.verify_runtime_config([good], broken, timeout=2.0)
        except node.ConfigMismatch:
            continue
        raise AssertionError(f"a wrong {field} was accepted")

    # A worker that never reports at all is also a failure, not a hang.
    try:
        node.verify_runtime_config([os.path.join(tmp, "nope.json")],
                                   expected, timeout=1.0)
    except node.ConfigMismatch as exc:
        assert "did not start" in str(exc), str(exc)
    else:
        raise AssertionError("a silent worker was accepted")
    return "wrong games/batch/fp16/visits and silence all rejected"


@check("worker keyword arguments match what the node passes")
def check_worker_signature():
    import inspect

    from mylc0.selfplay.worker import run_worker
    params = inspect.signature(run_worker).parameters
    for name in ("scale_parallel_to_target", "runtime_config_path",
                 "target_positions", "stats_path", "workers_total"):
        assert name in params, f"run_worker lost {name}"
    # The default must stay True so loop.py and the tuner are unaffected.
    assert params["scale_parallel_to_target"].default is True
    return "signature intact, default preserved"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG)
    args = parser.parse_args()
    globals()["CONFIG"] = args.config

    tmp = tempfile.mkdtemp(prefix="mylc0-cfg-check-")
    try:
        print("== config rewrite ==")
        check_config_rewrite(tmp)
        check_batch_size_section(tmp)

        print("\n== games in flight ==")
        check_no_clamp_for_nodes()
        check_clamp_preserved_for_quota()
        check_overshoot_floor()

        print("\n== fail-fast validation ==")
        check_runtime_config_published(tmp)
        check_fail_fast(tmp)
        check_worker_signature()
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
