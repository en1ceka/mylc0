"""Create generation 0: a randomly initialised network.

    python scripts/init_network.py --config configs/small.yaml

Writes ``networks/gen_000000.mylc0`` (plus ``networks/latest.mylc0``) and an
initial checkpoint so the training loop can resume from it. The weights are
random -- the network knows nothing beyond the shape of the problem.
"""

import argparse
import logging
import os

import _bootstrap  # noqa: F401
import torch

from mylc0.net.config import load_config
from mylc0.net.model import build_model
from mylc0.net.netfile import copy_as_latest, generation_filename, save_network
from mylc0.training.trainer import Trainer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/small.yaml")
    parser.add_argument("--networks", default=None)
    parser.add_argument("--checkpoints", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing generation 0")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    config = load_config(args.config)
    networks_dir = args.networks or config.training.networks_path
    path = generation_filename(networks_dir, 0)
    if os.path.exists(path) and not args.force:
        print(f"{path} already exists (use --force to overwrite)")
        return 1

    torch.manual_seed(args.seed)
    model = build_model(config.model)
    save_network(path, model, config.model, metadata={
        "generation": 0, "training_step": 0, "positions_seen": 0,
        "name": config.name, "seed": args.seed})
    copy_as_latest(path)

    trainer = Trainer(config, device="cpu",
                      checkpoint_dir=args.checkpoints,
                      networks_dir=networks_dir)
    trainer.model.load_state_dict(model.state_dict())
    trainer.save_checkpoint()

    print(f"generation 0 network: {path}")
    print(f"parameters: {model.num_parameters():,}")
    print(f"input format: {config.model.input_format}, policy size: 1858")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
