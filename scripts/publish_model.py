"""Publish a trained network to R2 and make it the one nodes will pick up.

    python scripts/publish_model.py --network networks/gen_000037.mylc0
    python scripts/publish_model.py --latest          # newest local export

The order is fixed and matters: the weights go up first under an immutable
key, then their metadata, then the upload is read back, and only after all of
that does ``models/latest.json`` start pointing at it. A failure anywhere
earlier leaves the previous generation current, which is a working state.

Models are never overwritten. Publishing the same generation twice is refused
unless the bytes are identical, in which case it just re-points the pointer.
"""

import argparse
import glob
import logging
import os
import re

import _bootstrap  # noqa: F401

from mylc0.cloud.models import fetch_latest, publish_model
from mylc0.cloud.storage import (StorageError, describe_env, sha256_file,
                                 store_from_env)

log = logging.getLogger("publish")


def generation_of(path: str):
    match = re.search(r"gen_(\d+)", os.path.basename(path))
    return int(match.group(1)) if match else None


def newest_export(networks_dir: str):
    paths = sorted(glob.glob(os.path.join(networks_dir, "gen_*.mylc0")))
    return paths[-1] if paths else None


def read_metadata(path: str) -> dict:
    """Whatever the network file already records about itself."""
    try:
        from mylc0.net.netfile import load_network
        _model, _config, metadata = load_network(path, device="cpu")
        return dict(metadata or {})
    except Exception as exc:              # noqa: BLE001
        log.warning("could not read metadata from %s: %s", path, exc)
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", default=None,
                        help="the .mylc0 file to publish")
    parser.add_argument("--networks-dir", default="networks")
    parser.add_argument("--latest", action="store_true",
                        help="publish the newest gen_*.mylc0 export")
    parser.add_argument("--generation", type=int, default=None,
                        help="override the generation number")
    parser.add_argument("--retry-attempts", type=int, default=6)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing generation (do not use "
                             "unless you know why)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [publish] %(message)s")

    path = args.network
    if args.latest or not path:
        path = newest_export(args.networks_dir)
    if not path or not os.path.isfile(path):
        print(f"no network to publish (looked in {args.networks_dir})")
        return 1

    generation = args.generation
    if generation is None:
        metadata = read_metadata(path)
        generation = metadata.get("generation")
        if generation is None:
            generation = generation_of(path)
    else:
        metadata = read_metadata(path)
    if generation is None:
        print(f"cannot tell which generation {path} is. Pass --generation.")
        return 1

    digest = sha256_file(path)
    print(f"network      {path}")
    print(f"generation   {generation}")
    print(f"size         {os.path.getsize(path) / 1e6:.1f} MB")
    print(f"sha256       {digest}")
    print("R2 configuration")
    print(describe_env())

    try:
        store = store_from_env()
    except StorageError as exc:
        print(f"\n{exc}")
        return 2

    current = fetch_latest(store)
    if current is not None:
        print(f"\ncurrently published: generation {current.generation} "
              f"({current.sha256[:12]}...)")
        if current.generation > int(generation) and not args.overwrite:
            print(f"\nrefusing to point latest.json back at generation "
                  f"{generation} when {current.generation} is already "
                  f"published.\nPass --overwrite if that is really what you "
                  f"want.")
            return 3
    else:
        print("\nnothing published yet; this will be the first model")

    if args.dry_run:
        print("\ndry run: nothing uploaded")
        return 0

    try:
        pointer = publish_model(
            store, path, int(generation),
            metadata={k: v for k, v in (metadata or {}).items()
                      if k != "state_dict"},
            attempts=args.retry_attempts, base_delay=args.retry_backoff,
            overwrite=args.overwrite)
    except StorageError as exc:
        print(f"\npublish failed: {exc}")
        print("latest.json was not changed; nodes keep using the previous "
              "generation.")
        return 4

    print(f"\npublished generation {pointer.generation}")
    print(f"  {pointer.key}")
    print("  latest.json now points here; nodes will switch after their "
          "current shard.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
