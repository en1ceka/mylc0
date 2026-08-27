"""Train on the data produced by self-play.

    python scripts/train.py --config configs/small.yaml --data data --generations 1

Resumes from the newest checkpoint unless ``--fresh`` is given. Every
``steps_per_network`` steps a checkpoint is written and an inference network is
exported as ``networks/gen_NNNNNN.mylc0``.
"""

import argparse
import logging
import os
import re
import time

import _bootstrap  # noqa: F401

from mylc0.net.config import load_config
from mylc0.progress import Progress, attach_logging, format_duration
from mylc0.training.dataset import TrainingDataLoader
from mylc0.training.trainer import Trainer


def replay_paths(roots, generations):
    """The newest ``generations`` ``gen_NNNNNN`` directories under ``roots``.

    Self-play never pauses, so data for a generation keeps arriving after the
    next one is published. Those games are not stale -- they came from a
    network one step behind -- but they must not outweigh the newest data, so
    the trainer reads a rolling window rather than everything ever produced.

    Nothing is deleted. A generation that falls out of the window stays on
    disk and comes back if the window is widened.
    """
    found = {}
    for root in roots:
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            match = re.fullmatch(r"gen_(\d{6})", name)
            path = os.path.join(root, name)
            if match and os.path.isdir(path):
                found.setdefault(int(match.group(1)), []).append(path)
    if not found:
        return []
    window = sorted(found)[-max(1, generations):]
    paths = [p for gen in window for p in found[gen]]
    print(f"replay window: generations {window} "
          f"({len(paths)} director{'y' if len(paths) == 1 else 'ies'})")
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/small.yaml")
    parser.add_argument("--data", nargs="+", default=None)
    parser.add_argument("--generations", type=int, default=1,
                        help="how many networks to produce (0 = forever)")
    parser.add_argument("--steps", type=int, default=None,
                        help="override steps_per_network")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fresh", action="store_true",
                        help="ignore existing checkpoints")
    parser.add_argument("--checkpoints", default=None)
    parser.add_argument("--networks", default=None)
    parser.add_argument("--tensorboard", default=None)
    parser.add_argument("--progress", default="auto",
                        choices=["auto", "on", "off"])
    parser.add_argument("--replay-generations", type=int, default=0,
                        help="train on the newest N generation directories "
                             "under --data (0 = every directory, the old "
                             "behaviour). Older generations stay on disk.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [train] %(message)s")
    progress = Progress(enabled={"on": True, "off": False}.get(args.progress))
    attach_logging(progress)
    config = load_config(args.config)
    data_paths = args.data or [config.selfplay.output_path]
    if args.replay_generations:
        data_paths = replay_paths(data_paths, args.replay_generations)
        if not data_paths:
            print("no gen_NNNNNN/ directories found; run "
                  "scripts/sync_selfplay.py first, or drop "
                  "--replay-generations to train on a flat data directory")
            return 1

    trainer = Trainer(config, device=args.device,
                      checkpoint_dir=args.checkpoints,
                      networks_dir=args.networks,
                      tensorboard_dir=args.tensorboard)
    trainer.report_device()
    if not args.fresh:
        trainer.load_checkpoint()

    loader = TrainingDataLoader(
        data_paths, batch_size=config.training.batch_size,
        chunk_pool_size=config.training.chunk_pool_size,
        position_sampling_rate=config.training.position_sampling_rate,
        shuffle_buffer_size=config.training.shuffle_buffer_size,
        workers=config.training.loader_workers,
        seed=trainer.step + 1)
    loader.maybe_rescan(force=True)
    if len(loader.pool) == 0:
        print(f"no training chunks found under {data_paths}; run self-play first")
        return 1
    print(f"chunk pool: {len(loader.pool)} chunks")
    loader.start()

    generations = args.generations
    produced = 0
    try:
        while generations <= 0 or produced < generations:
            t0 = time.perf_counter()
            trainer.train_generation(loader, steps=args.steps,
                                     progress=progress)
            progress.close()
            produced += 1
            print(f"generation {trainer.generation} done in "
                  f"{format_duration(time.perf_counter() - t0)} "
                  f"(step {trainer.step})")
    except KeyboardInterrupt:
        print("interrupted; saving checkpoint")
        trainer.save_checkpoint()
    finally:
        progress.close()
        loader.stop()
        trainer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
