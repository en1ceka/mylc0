"""Export an inference network from a training checkpoint.

    python scripts/export_network.py --checkpoint checkpoints/generation_000012/checkpoint.pt
    python scripts/export_network.py --config configs/small.yaml --all
    python scripts/export_network.py --network networks/gen_000012.mylc0 --onnx out.onnx

The exported file contains only the model configuration and the weights: it is
what the UCI engine loads, and it never needs the training environment.
"""

import argparse
import json
import os

import _bootstrap  # noqa: F401
import torch

from mylc0.net.config import load_config, model_config_from_dict
from mylc0.net.model import build_model
from mylc0.net.netfile import (copy_as_latest, export_onnx,
                               generation_filename, load_network, read_metadata,
                               save_network)


def export_from_checkpoint(checkpoint_path: str, networks_dir: str,
                           generation: int = None) -> str:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = model_config_from_dict(json.loads(payload["model_config"]))
    model = build_model(model_config)
    model.load_state_dict(payload["model"])
    generation = payload["generation"] if generation is None else generation
    path = generation_filename(networks_dir, generation)
    save_network(path, model, model_config, metadata={
        "generation": generation,
        "training_step": payload["step"],
        "positions_seen": payload.get("positions_seen", 0),
        "source_checkpoint": os.path.abspath(checkpoint_path),
    })
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default="configs/small.yaml")
    parser.add_argument("--networks", default=None)
    parser.add_argument("--generation", type=int, default=None)
    parser.add_argument("--all", action="store_true",
                        help="export every checkpoint that is still on disk")
    parser.add_argument("--latest", action="store_true",
                        help="also refresh networks/latest.mylc0")
    parser.add_argument("--network", default=None,
                        help="an existing .mylc0 file, for --onnx or --info")
    parser.add_argument("--onnx", default=None, help="also write an ONNX model")
    parser.add_argument("--info", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    networks_dir = args.networks or config.training.networks_path
    checkpoint_dir = config.training.checkpoint_path

    if args.info and args.network:
        print(json.dumps(read_metadata(args.network), indent=2))
        return 0

    if args.onnx and args.network:
        model, model_config, meta = load_network(args.network)
        export_onnx(args.onnx, model, model_config)
        print(f"ONNX written: {args.onnx}")
        return 0

    exported = []
    if args.all:
        for name in sorted(os.listdir(checkpoint_dir)):
            path = os.path.join(checkpoint_dir, name, "checkpoint.pt")
            if os.path.isfile(path):
                exported.append(export_from_checkpoint(path, networks_dir))
    else:
        checkpoint = args.checkpoint
        if checkpoint is None:
            entries = sorted(d for d in os.listdir(checkpoint_dir)
                             if d.startswith("generation_"))
            if not entries:
                print("no checkpoints found")
                return 1
            checkpoint = os.path.join(checkpoint_dir, entries[-1],
                                      "checkpoint.pt")
        exported.append(export_from_checkpoint(checkpoint, networks_dir,
                                               args.generation))

    for path in exported:
        print(f"exported {path}")
    if args.latest and exported:
        print(f"latest -> {copy_as_latest(exported[-1])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
