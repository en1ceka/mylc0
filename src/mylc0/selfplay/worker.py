"""Self-play workers.

A worker owns one network instance and plays games in a loop, writing one
gzipped V6 chunk per game. Workers are independent processes writing into
separate directories, which is what makes it possible to scale self-play out
later (more processes here, or several machines writing into the same data
root) without touching the trainer.

The trainer never talks to the workers: they communicate only through the
files on disk, exactly like a real Lc0 training run.
"""

from __future__ import annotations

import faulthandler
import json
import logging
import multiprocessing
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from ..chessrules.position import BLACK_WON, DRAW, UNDECIDED, WHITE_WON
from ..net.backend import Backend
from ..net.config import Config, SelfPlayConfig, load_config
from ..net.netfile import load_network
from ..perf import (PerfCounters, affinity_slice, gpu_memory, set_affinity,
                    write_json)
from .batched import BatchedSelfPlay

log = logging.getLogger("mylc0.selfplay")


@dataclass
class WorkerStats:
    games: int = 0
    positions: int = 0
    white_wins: int = 0
    black_wins: int = 0
    draws: int = 0
    adjudicated: int = 0
    plies: int = 0
    nodes: int = 0
    nn_evals: int = 0
    cache_hits: int = 0
    seconds: float = 0.0

    def add_game(self, stats) -> None:
        self.games += 1
        self.positions += stats.frames
        self.plies += stats.plies
        self.nodes += stats.nodes
        self.nn_evals += stats.nn_evals
        self.cache_hits += stats.cache_hits
        self.seconds += stats.seconds
        if stats.adjudicated:
            self.adjudicated += 1
        if stats.result == WHITE_WON:
            self.white_wins += 1
        elif stats.result == BLACK_WON:
            self.black_wins += 1
        else:
            self.draws += 1

    def as_dict(self) -> Dict[str, float]:
        d = dict(self.__dict__)
        d["avg_game_length"] = self.plies / max(1, self.games)
        d["games_per_hour"] = self.games / max(1e-9, self.seconds) * 3600
        d["positions_per_second"] = self.positions / max(1e-9, self.seconds)
        d["nn_evals_per_second"] = self.nn_evals / max(1e-9, self.seconds)
        d["nodes_per_move"] = self.nodes / max(1, self.plies)
        d["nodes_per_second"] = self.nodes / max(1e-9, self.seconds)
        return d


def make_backend(network_path: str, cfg: SelfPlayConfig, device: str,
                 fp16: bool) -> Backend:
    model, model_config, metadata = load_network(network_path, device="cpu")
    backend = Backend(
        model, model_config,
        device=device,
        fp16=fp16,
        max_batch_size=cfg.batch_size,
        policy_softmax_temp=cfg.search.policy_softmax_temp,
        cache_size=cfg.search.nncache_size,
        # tournament.cc sets CacheHistoryLength to 7 for self-play.
        cache_history_length=7,
        history_fill=cfg.search.history_fill)
    backend.network_metadata = metadata
    return backend


OOM_EXIT_CODE = 42
"""Exit code a worker uses to say "I ran out of VRAM", as opposed to crashing.

The tuner has to tell the two apart: an OOM means the configuration is too
large for the card and the next one should still be tried, while a crash means
the benchmark itself is broken and hiding it would be wrong.
"""


def _release_cuda() -> None:
    try:
        torch.cuda.synchronize()
    except Exception:
        pass                     # a dead context cannot be synchronised
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


# Wordings that mean "the allocation failed" and nothing else.
_OOM_CERTAIN = (
    "out of memory",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
)
# Wordings a genuinely exhausted card also produces. Measured on a 3060 Ti
# with 32 workers: exactly one process raised torch.OutOfMemoryError while the
# rest reported these instead -- cuBLAS cannot allocate its workspace and calls
# that a failed execution, and a context that cannot be created at all is
# reported as a busy device. Treating them as crashes would abort a benchmark
# that is merely too ambitious, so they are resolved by asking the driver how
# much VRAM is actually left.
_OOM_LIKELY = (
    "cublas_status_execution_failed",
    "cublas_status_not_initialized",
    "cudnn_status_not_initialized",
    "device(s) is/are busy or unavailable",
    "no kernel image is available",
)


def _vram_exhausted(floor_mib: float = 512.0) -> bool:
    """Ask nvidia-smi, not torch: after a CUDA error the context is unusable.

    Racy by nature -- sibling workers may already be dying and handing memory
    back -- so it only ever promotes an already-suspicious CUDA error, and is
    never the sole reason to call something an OOM.
    """
    mem = gpu_memory()
    if mem is None:
        return False
    return mem["free_mib"] <= floor_mib


def _is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    if not isinstance(exc, RuntimeError):
        return False
    text = str(exc).lower()
    if any(hint in text for hint in _OOM_CERTAIN):
        return True
    if any(hint in text for hint in _OOM_LIKELY):
        return _vram_exhausted()
    return False


def _write_failure(path: Optional[str], kind: str, detail: str) -> None:
    """Leave a breadcrumb the parent can read after the process is gone."""
    if not path:
        return
    try:
        with open(path + ".fail", "w", encoding="utf-8") as handle:
            json.dump({"error": kind, "detail": detail[:4000],
                       "pid": os.getpid()}, handle)
    except OSError:
        pass


def effective_parallel_games(cfg, target_positions: int,
                             scale_to_target: bool = True) -> int:
    """How many games this worker actually keeps in flight.

    Every game in flight is played to its result, so a worker asked for a
    small, exact number of positions should not start more games than that
    number can absorb: 8 games against a 300-position target overshoots by an
    order of magnitude. ``loop.py`` wants that behaviour -- it asks for a
    precise quota per generation.

    A self-play node does not. Its target is a *shard boundary*, not a quota:
    whatever the games in flight produce past it simply goes into the shard.
    Scaling down there is actively harmful -- 28 workers x 48 games against a
    2000-position shard target collapses to one game each, which drops the NN
    batch from ~318 to ~10 and costs roughly 6x throughput. Such callers pass
    ``scale_to_target=False`` and stop starting new games instead.
    """
    parallel = max(1, cfg.parallel_games)
    if not (scale_to_target and target_positions):
        return parallel
    # Half of max_game_ply is a conservative guess at a game's length.
    typical = max(1, cfg.max_game_ply // 2)
    return max(1, min(parallel, target_positions // typical))


def write_runtime_config(path, cfg, parallel: int, device: str, fp16: bool,
                         generation) -> None:
    """Publish what this worker is *actually* running with.

    The knobs travel through a config file and a process boundary, and a
    mismatch is invisible in the output -- it just produces correct games
    slowly. A supervisor reads this back and refuses to keep going if what
    arrived is not what it asked for.
    """
    if not path:
        return
    payload = {"parallel_games": parallel,
               "requested_parallel_games": cfg.parallel_games,
               "nn_batch": cfg.batch_size,
               "fp16": bool(fp16),
               "visits": cfg.visits,
               "minibatch_size": cfg.search.minibatch_size,
               "device": str(device),
               "generation": generation,
               "pid": os.getpid()}
    tmp = path + f".{os.getpid()}.tmp"
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1)
        os.replace(tmp, path)
    except OSError:
        pass                    # telemetry must never take a worker down


def run_worker(config_path: str, network_path: str, output_dir: str,
               worker_id: int = 0, num_games: int = 0, seed: int = 0,
               device: Optional[str] = None, fp16: Optional[bool] = None,
               log_every: int = 1, stats_path: Optional[str] = None,
               max_seconds: float = 0.0,
               target_positions: int = 0,
               watchdog_seconds: float = 30.0,
               heartbeat_seconds: float = 10.0,
               perf_debug: bool = False, perf_path: Optional[str] = None,
               perf_warmup: float = 0.0,
               torch_threads: int = 1, affinity: bool = False,
               workers_total: int = 1,
               scale_parallel_to_target: bool = True,
               runtime_config_path: Optional[str] = None,
               log_level: str = "INFO",
               freeze_perf_on_drain: bool = True,
               stop_file: Optional[str] = None,
               chunk_dir: Optional[str] = None) -> Dict[str, float]:
    # A node runs 28 of these. At INFO each one announces itself at startup
    # and the useful output scrolls away, so a supervisor that prints its own
    # aggregate status turns them down to WARNING. Errors and OOM are logged
    # above that level and stay visible either way.
    logging.basicConfig(
        level=getattr(logging, str(log_level).upper(), logging.INFO),
        format=f"%(asctime)s [selfplay-{worker_id}] %(message)s")
    config: Config = load_config(config_path)
    cfg = config.selfplay
    device = device or cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    fp16 = cfg.fp16 if fp16 is None else fp16

    # Restored at the end: when a worker runs in-process (loop.py with a
    # single worker) the trainer still needs autograd afterwards.
    if affinity:
        cpus = affinity_slice(worker_id, workers_total)
        if set_affinity(cpus):
            log.info("pinned to CPUs %s", cpus)
        else:
            log.info("CPU affinity not supported here; running unpinned")

    prev_grad = torch.is_grad_enabled()
    prev_threads = torch.get_num_threads()
    torch.set_grad_enabled(False)
    # One intra-op thread: the search is single-threaded Python and the GPU
    # does the maths, so a BLAS pool per worker only fights for cores.
    torch.set_num_threads(max(1, torch_threads))
    try:
        return _play_games(config, cfg, network_path, output_dir, worker_id,
                           num_games, seed, device, fp16, log_every,
                           stats_path, max_seconds, target_positions,
                           watchdog_seconds, heartbeat_seconds, perf_debug,
                           perf_path, perf_warmup, scale_parallel_to_target,
                           runtime_config_path, freeze_perf_on_drain,
                           stop_file, chunk_dir)
    except BaseException as exc:
        if _is_cuda_oom(exc):
            # Not a crash: the card is simply too small for this many workers.
            # Report it as its own exit code so the tuner can mark the trial
            # OOM and move on instead of aborting the whole benchmark.
            log.error("worker %d: CUDA out of memory (%s)", worker_id,
                      str(exc).splitlines()[0])
            _write_failure(perf_path or stats_path, "cuda_oom", str(exc))
            _release_cuda()
            if multiprocessing.parent_process() is not None:
                sys.exit(OOM_EXIT_CODE)
            raise
        # Never swallow a worker failure: log it here (the parent only sees an
        # exit code) and re-raise so the exit code is non-zero.
        log.exception("worker %d died", worker_id)
        _write_failure(perf_path or stats_path, "crash", repr(exc))
        raise
    finally:
        torch.set_grad_enabled(prev_grad)
        torch.set_num_threads(prev_threads)
        if multiprocessing.parent_process() is not None:
            # A child is about to exit; drop the allocator's cache now so the
            # parent sees VRAM return to baseline promptly rather than waiting
            # on the driver to reap the context.
            _release_cuda()


def _play_games(config, cfg, network_path, output_dir, worker_id, num_games,
                seed, device, fp16, log_every, stats_path, max_seconds,
                target_positions, watchdog_seconds=30.0,
                heartbeat_seconds=10.0, perf_debug=False,
                perf_path=None, perf_warmup=0.0,
                scale_parallel_to_target=True,
                runtime_config_path=None,
                freeze_perf_on_drain=True, stop_file=None,
                chunk_dir=None) -> Dict[str, float]:
    backend = make_backend(network_path, cfg, device, fp16)
    perf = None
    if perf_debug or perf_path:
        from ..net.backend import BackendTiming
        from ..search.search import SearchTiming
        import mylc0.search.search as search_module
        perf = PerfCounters()
        backend.timing = BackendTiming()
        shared_timing = SearchTiming()
        original_init = search_module.Search.__init__

        def patched_init(self, *a, **kw):
            original_init(self, *a, **kw)
            self.timing = shared_timing
        search_module.Search.__init__ = patched_init
        perf.search_timing = shared_timing
    generation = backend.network_metadata.get("generation", 0)
    # A collector may be watching this directory, so every worker can write
    # into one spool: the file names already carry the worker id, and each
    # chunk is renamed into place atomically once it is complete.
    out_dir = chunk_dir or os.path.join(output_dir, f"worker_{worker_id:02d}")
    os.makedirs(out_dir, exist_ok=True)

    parallel = effective_parallel_games(cfg, target_positions,
                                        scale_parallel_to_target)
    driver = BatchedSelfPlay(backend, cfg, parallel, seed=seed)
    stats = WorkerStats()
    t_start = time.perf_counter()
    state = {"index": 0, "last_flush": t_start, "last_progress": t_start,
             "last_heartbeat": t_start, "last_perf": t_start,
             "signature": None, "stalled": False,
             "warmed": not bool(perf_warmup), "perf_frozen": False,
             "drain_logged": False}
    write_runtime_config(runtime_config_path, cfg, parallel, device, fp16,
                         generation)
    scaled = ("" if parallel == cfg.parallel_games else
              f" (scaled down from {cfg.parallel_games} to fit a "
              f"{target_positions}-position target)")
    log.info("network %s (generation %s), device=%s fp16=%s, visits=%d, "
             "%d games in flight%s, max NN batch %d",
             os.path.basename(network_path), generation, device, fp16,
             cfg.visits, parallel, scaled, cfg.batch_size)

    def on_game(game) -> None:
        index = state["index"]
        state["index"] = index + 1
        # The generation and the position count are in the name so a
        # collector can group and size chunks without decompressing them.
        # Nothing else parses this; the loader and the shard packer only
        # ever glob for "*.gz".
        stem = (f"g{generation:06d}_w{worker_id:02d}_"
                f"{int(time.time() * 1000):013d}_{index:06d}")
        frames = game.write(os.path.join(out_dir, stem + ".gz"))
        if chunk_dir and frames:
            # Rename rather than write under the final name: the collector
            # must never see a size in the name it cannot trust.
            final = os.path.join(out_dir, f"{stem}_n{frames:05d}.gz")
            try:
                os.replace(os.path.join(out_dir, stem + ".gz"), final)
            except OSError:
                pass
        stats.add_game(game.stats)
        if log_every and (index + 1) % log_every == 0:
            result = {WHITE_WON: "1-0", BLACK_WON: "0-1",
                      DRAW: "1/2-1/2", UNDECIDED: "*"}[game.stats.result]
            log.info("game %d: %s in %d plies, %d frames, %.1fs "
                     "| avg NN batch %.0f | W%d D%d B%d",
                     index + 1, result, game.stats.plies, frames,
                     game.stats.seconds, driver.avg_batch, stats.white_wins,
                     stats.draws, stats.black_wins)
        if stats_path:
            _write_stats(stats_path, stats, generation,
                         current_plies=driver.plies_in_flight(),
                         active=driver.active_games())

    def refresh_perf(force=False):
        if perf is None or (state.get("perf_frozen") and not force):
            return None
        perf.absorb_backend(backend.timing)
        perf.absorb_search(perf.search_timing)
        perf.cache_hits = backend.cache.hits
        perf.games = stats.games
        perf.positions = stats.positions
        perf.plies = stats.plies + driver.plies_in_flight()
        perf.games_in_flight = driver.active_games()
        perf.phase = "draining" if driver.draining else "running"
        perf.nodes = stats.nodes + driver.nodes_in_flight()
        perf.batch_sizes = driver.batch_history
        snapshot = perf.snapshot()
        snapshot["worker_id"] = worker_id
        snapshot["pid"] = os.getpid()
        if perf_path:
            write_json(perf_path, snapshot)
        return snapshot

    def should_stop() -> bool:
        """Stop *admitting new games*. Games in flight play on to a result.

        The driver treats this as the start of its drain, not as a stop: a
        game abandoned mid-way has no result, so no value target, and could
        not be written even if we wanted it.
        """
        stop = _should_stop()
        if stop and perf is not None and not state.get("perf_frozen"):
            if freeze_perf_on_drain:
                # A benchmark wants the steady-state rate. The drain has games
                # finishing one by one and throughput falling off a cliff, so
                # measuring through it would understate the machine.
                refresh_perf(force=True)
                state["perf_frozen"] = True
            elif not state.get("drain_logged"):
                # A node wants the opposite: the drain is real work that
                # produces most of a shard's finished games, and freezing the
                # counters here makes a busy node look hung -- live positions
                # stuck, rate zero, games in flight never falling.
                state["drain_logged"] = True
                log.info("target reached at %d finished positions; draining "
                         "%d game(s) still in flight", stats.positions,
                         driver.active_games())
        return stop

    def _should_stop() -> bool:
        if num_games > 0 and stats.games >= num_games:
            return True
        if stop_file and os.path.exists(stop_file):
            # A supervisor retiring this worker -- to move it onto a new
            # network, say. Same meaning as any other stop: admit no more
            # games, finish the ones in flight. Checked by stat() on a tick
            # that already runs between search batches, so it costs nothing
            # measurable next to a forward pass.
            return True
        # Count the plies already played in the games still running: they will
        # become training positions once those games finish.
        if (target_positions
                and stats.positions + driver.plies_in_flight() >= target_positions):
            return True
        if max_seconds and (time.perf_counter() - t_start) > max_seconds:
            return True
        return False

    def hard_stop() -> bool:
        # Only a wall-clock limit abandons games that are already running.
        return bool(max_seconds
                    and (time.perf_counter() - t_start) > max_seconds * 1.5)

    def progress_signature():
        """Everything that must move if self-play is alive."""
        return (stats.games, stats.positions, driver.plies_in_flight(),
                backend.evaluations, driver.stats.batches)

    def on_tick() -> None:
        now = time.perf_counter()
        if stats_path and now - state["last_flush"] >= 1.0:
            state["last_flush"] = now
            _write_stats(stats_path, stats, generation,
                         current_plies=driver.plies_in_flight(),
                         active=driver.active_games())

        signature = progress_signature()
        if signature != state["signature"]:
            state["signature"] = signature
            state["last_progress"] = now
            state["stalled"] = False
        elif (watchdog_seconds
              and now - state["last_progress"] >= watchdog_seconds
              and not state["stalled"]):
            state["stalled"] = True
            _dump_diagnostics(worker_id, driver, backend, stats,
                              now - state["last_progress"])

        if perf is not None and not state["warmed"] and perf_warmup:
            if now - t_start >= perf_warmup:
                # Drop the ramp-up: games start one at a time, CUDA is
                # cold and the NN cache is empty, all of which skew the
                # averages downwards.
                state["warmed"] = True
                refresh_perf()
                perf.rebaseline()
                log.info("perf: measurement window starts now "
                         "(%.0fs warm-up dropped)", perf_warmup)
        if perf is not None and now - state["last_perf"] >= 2.0:
            state["last_perf"] = now
            refresh_perf()

        if heartbeat_seconds and now - state["last_heartbeat"] >= heartbeat_seconds:
            elapsed = max(1e-9, now - t_start)
            log.info("heartbeat: %d games done, %d positions, %d in flight "
                     "(%d plies) | %.0f nodes/s %.0f evals/s | avg batch %.0f "
                     "last batch %d | last progress %.1fs ago",
                     stats.games, stats.positions, driver.active_games(),
                     driver.plies_in_flight(),
                     (stats.nodes + driver.nodes_in_flight()) / elapsed,
                     backend.evaluations / elapsed, driver.avg_batch,
                     driver.stats.last_batch_size, now - state["last_progress"])
            state["last_heartbeat"] = now

    driver.run(on_game=on_game, should_stop=should_stop, on_tick=on_tick,
               hard_stop=hard_stop)

    abandoned = driver.active_games()
    if abandoned:
        log.warning("abandoned %d unfinished game(s) with %d plies: a game "
                    "without a result has no value target and cannot be "
                    "written. Prefer --target-positions or --games over a "
                    "wall-clock limit.", abandoned, driver.plies_in_flight())

    refresh_perf()
    if stats_path:
        _write_stats(stats_path, stats, generation, active=0)
    log.info("done: %d games, %d positions, %.1f s wall, avg NN batch %.1f "
             "(min %d, max %d over %d batches)",
             stats.games, stats.positions, time.perf_counter() - t_start,
             driver.avg_batch, driver.stats.requests_per_batch_min,
             driver.stats.requests_per_batch_max, driver.stats.batches)
    return stats.as_dict()


_WARNED = set()


def _warn_once(message: str, *args) -> None:
    if message not in _WARNED:
        _WARNED.add(message)
        log.warning(message, *args)


def _dump_diagnostics(worker_id, driver, backend, stats, stalled_for) -> None:
    """Everything needed to tell where self-play got stuck."""
    now = time.monotonic()
    since_batch = (now - driver.stats.last_batch_at
                   if driver.stats.last_batch_at else float("nan"))
    lines = [
        "",
        "=" * 70,
        f"WATCHDOG: worker {worker_id} (pid {os.getpid()}) made no progress "
        f"for {stalled_for:.0f}s",
        "=" * 70,
        f"  games done            {stats.games}",
        f"  positions written     {stats.positions}",
        f"  games in flight       {driver.active_games()} "
        f"({driver.plies_in_flight()} plies)",
        f"  NN evaluations        {backend.evaluations}",
        f"  NN batches            {driver.stats.batches}",
        f"  last NN batch size    {driver.stats.last_batch_size}",
        f"  last NN batch         {since_batch:.1f}s ago",
        f"  NN cache hits         {backend.cache.hits}",
        f"  draining              {driver.draining}",
        "  per-game state:",
    ]
    lines.extend("    " + line for line in driver.describe())
    lines.append("=" * 70)
    log.error("%s", "\n".join(lines))
    # Where every thread of this process actually is.
    try:
        faulthandler.dump_traceback(file=sys.stderr)
    except Exception:
        pass
    sys.stderr.flush()


def _write_stats(path: str, stats: WorkerStats, generation,
                 current_plies: int = 0, active: int = 0) -> None:
    payload = stats.as_dict()
    payload["generation"] = generation
    # Live fields, so a supervisor can show progress inside the current game.
    payload["current_plies"] = current_plies
    payload["active"] = active
    payload["updated"] = time.time()
    tmp = path + f".{os.getpid()}.tmp"
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        # On Windows os.replace fails with "access denied" while another
        # process has the destination open for reading -- and the supervisor
        # polls this file twice a second. Retry briefly, then give up: this is
        # telemetry and must never take a worker down with it.
        for attempt in range(6):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                time.sleep(0.02 * (attempt + 1))
        _warn_once("could not update %s (file busy); statistics may lag",
                   path)
    except OSError as exc:
        _warn_once("could not write %s: %s", path, exc)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def read_live_stats(paths: List[str]) -> Dict[str, float]:
    """Snapshot of the workers' progress, including the games in flight.

    ``positions`` counts finished games only, so ``current_plies`` (the plies
    played so far in the games currently running) is added to give a smooth
    number to show while a long game is still being played.
    """
    total = aggregate_stats(paths)
    total["live_positions"] = (total.get("positions", 0)
                               + total.get("current_plies", 0))
    return total


def monitor_selfplay(stats_paths: List[str], target_positions: int,
                     progress, stop_event, workers: int = 1,
                     processes=None, stall_seconds: float = 60.0) -> None:
    """Render a live self-play progress line until ``stop_event`` is set.

    Also watches the workers: a process that dies is reported immediately
    (instead of leaving a progress bar spinning on a dead run), and a global
    lack of progress is flagged.
    """
    from ..progress import bar, format_duration, format_eta
    t0 = time.perf_counter()
    last_total = -1
    last_change = t0
    warned = False
    reported_dead = set()
    while True:
        live = read_live_stats(stats_paths)
        elapsed = time.perf_counter() - t0
        done = live.get("live_positions", 0)
        games = int(live.get("games", 0))
        active = int(live.get("active", 0))
        rate = done / elapsed if elapsed > 0 else 0.0
        parts = ["self-play"]
        if target_positions > 0:
            parts.append(f"[{bar(done / target_positions)}] "
                         f"{done}/{target_positions} pos")
            if done >= target_positions:
                # The target is only checked between games, so the games
                # already in flight are played out -- they cannot be cut off
                # without losing their result.
                parts.append("finishing games in flight")
            else:
                parts.append(
                    f"ETA {format_eta(done, target_positions, elapsed)}")
        else:
            parts.append(f"{done} pos")
        parts.append(f"{games} done")
        # "active" counts games in flight, which is workers x parallel_games
        # (clamped inside each worker), so it is reported as a count.
        parts.append(f"{active} in flight")
        parts.append(f"{rate * 60:.0f} pos/min")
        parts.append(format_duration(elapsed))
        progress.set("  ".join(parts))

        if processes:
            for proc in processes:
                if (not proc.is_alive() and proc.exitcode
                        and proc.pid not in reported_dead):
                    reported_dead.add(proc.pid)
                    log.error("self-play worker pid %s exited with code %s "
                              "-- see its traceback above", proc.pid,
                              proc.exitcode)
        now = time.perf_counter()
        if done != last_total:
            last_total = done
            last_change = now
            warned = False
        elif not warned and now - last_change >= stall_seconds:
            warned = True
            log.warning("no self-play progress for %.0fs: %d positions, "
                        "%d games in flight. Each worker dumps its state "
                        "after its own watchdog interval.",
                        now - last_change, done, active)

        if stop_event.wait(0.5):
            break


def aggregate_stats(paths: List[str]) -> Dict[str, float]:
    total: Dict[str, float] = {}
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        for key, value in data.items():
            if isinstance(value, (int, float)) and key not in (
                    "avg_game_length", "games_per_hour",
                    "positions_per_second", "nn_evals_per_second",
                    "nodes_per_move", "nodes_per_second", "generation",
                    "updated"):
                total[key] = total.get(key, 0) + value
    games = max(1, total.get("games", 0))
    seconds = max(1e-9, total.get("seconds", 0.0))
    total["avg_game_length"] = total.get("plies", 0) / games
    total["games_per_hour"] = total.get("games", 0) / seconds * 3600
    total["positions_per_second"] = total.get("positions", 0) / seconds
    total["nodes_per_move"] = total.get("nodes", 0) / max(1, total.get("plies", 0))
    total["nodes_per_second"] = total.get("nodes", 0) / seconds
    return total
