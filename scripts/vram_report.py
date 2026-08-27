"""Where does a self-play worker's VRAM actually go, and how many fit?

    python scripts/vram_report.py --config configs/small.yaml \
        --network networks/latest.mylc0 --parallel-games 48 --nn-batch 512

Each worker is a separate process with its own CUDA context and its own copy of
the weights, so the card's capacity -- not the CPU -- is what caps ``--workers``
once the count gets large. This measures the four pieces of one worker's
footprint and turns them into a worker budget for this GPU.

Everything is measured by nvidia-smi deltas rather than ``torch.cuda``: the
CUDA context is itself one of the things being measured, and torch's own
accounting cannot see memory the driver allocated before torch existed.

Nothing here changes the search or the network; it only allocates and measures.
"""

import argparse
import os
import subprocess

import _bootstrap  # noqa: F401


def smi_used_mib():
    """Total VRAM in use on the GPU, in MiB, or None."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        used, total = out.stdout.strip().splitlines()[0].split(",")
        return float(used), float(total)
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/small.yaml")
    parser.add_argument("--network", default="networks/latest.mylc0")
    parser.add_argument("--nn-batch", type=int, default=512,
                        help="selfplay.batch_size the workers would use")
    parser.add_argument("--parallel-games", type=int, default=48,
                        help="selfplay.parallel_games the workers would use")
    parser.add_argument("--headroom", type=float, default=0.90,
                        help="fraction of the card a run may occupy")
    args = parser.parse_args()

    if not os.path.isfile(args.network):
        print(f"network not found: {args.network}")
        return 1

    sample = smi_used_mib()
    if sample is None:
        print("nvidia-smi is unavailable; cannot measure VRAM")
        return 1
    baseline, total = sample
    print(f"GPU total {total / 1024:.1f} GB, "
          f"{baseline:.0f} MiB already in use by other processes\n")

    # -- 1. the CUDA context, before anything of ours exists -----------------
    import torch
    if not torch.cuda.is_available():
        print("no CUDA device")
        return 1
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    ctx = smi_used_mib()[0] - baseline

    # -- 2. the weights ------------------------------------------------------
    from mylc0.net.config import load_config
    from mylc0.net.encoder import TOTAL_PLANES
    from mylc0.selfplay.worker import make_backend

    config = load_config(args.config)
    cfg = config.selfplay
    cfg.batch_size = args.nn_batch
    cfg.parallel_games = args.parallel_games
    backend = make_backend(args.network, cfg, "cuda", cfg.fp16)
    weights = smi_used_mib()[0] - baseline - ctx
    params = sum(p.numel() for p in backend.model.parameters())
    param_mib = sum(p.numel() * p.element_size()
                    for p in backend.model.parameters()) / 2 ** 20

    # -- 3. activations, at the largest batch this config can actually reach --
    minibatch = cfg.search.minibatch_size
    reachable = min(args.nn_batch, minibatch * args.parallel_games)
    peak_per_batch = {}
    for batch in sorted({64, 128, 256, 512, reachable}):
        if batch > args.nn_batch:
            continue
        planes = torch.zeros((batch, TOTAL_PLANES, 8, 8),
                             dtype=torch.float16 if cfg.fp16 else torch.float32,
                             device="cuda")
        with torch.no_grad():
            backend.model(planes)
        torch.cuda.synchronize()
        del planes
        peak_per_batch[batch] = smi_used_mib()[0] - baseline - ctx - weights

    activations = peak_per_batch[reachable]
    per_worker = ctx + weights + activations

    # -- report --------------------------------------------------------------
    print("One worker, measured")
    print(f"  CUDA context + cuBLAS      {ctx:>7.0f} MiB")
    print(f"  weights ({params / 1e6:.1f}M params, "
          f"{'fp16' if cfg.fp16 else 'fp32'}){'':<3}{weights:>7.0f} MiB"
          f"   (tensors alone: {param_mib:.0f} MiB)")
    print(f"  allocator + activations    {activations:>7.0f} MiB"
          f"   at batch {reachable}")
    print(f"  {'':<26} {'-' * 7}")
    print(f"  total per worker           {per_worker:>7.0f} MiB")
    print(f"\n  torch reserved {torch.cuda.memory_reserved() / 2 ** 20:.0f} MiB "
          f"vs allocated {torch.cuda.memory_allocated() / 2 ** 20:.0f} MiB -- "
          f"the gap is\n  the caching allocator holding freed blocks, and it "
          f"counts against the card.")
    print(f"\n  host-side scratch buffer   "
          f"{backend._scratch.nbytes / 2 ** 20:>7.1f} MiB   (RAM, not VRAM; "
          f"scales with nn_batch)")

    print("\nActivations vs batch size")
    for batch, mib in sorted(peak_per_batch.items()):
        note = "  <- largest this config can reach" if batch == reachable else ""
        print(f"  batch {batch:>5}   {mib:>7.0f} MiB{note}")

    if args.nn_batch > minibatch * args.parallel_games:
        print(f"\n  Note: nn_batch={args.nn_batch} cannot be filled. A worker "
              f"gathers at most\n  minibatch_size x parallel_games = "
              f"{minibatch} x {args.parallel_games} = "
              f"{minibatch * args.parallel_games} leaves per step, so the "
              f"extra capacity only\n  enlarges the host scratch buffer.")

    # -- worker budget -------------------------------------------------------
    usable = total * args.headroom - baseline
    fits = int(usable // per_worker) if per_worker > 0 else 0
    print("\nWorker budget on this card")
    print(f"  usable at {args.headroom * 100:.0f}% of "
          f"{total / 1024:.0f} GB      {usable:>7.0f} MiB")
    print(f"  workers that fit           {fits:>7d}")
    for count in (fits, fits + 2, fits + 4):
        need = count * per_worker + baseline
        verdict = "fits" if need <= total * args.headroom else (
            "OOM likely" if need <= total else "OOM")
        print(f"    {count:>3d} workers -> {need:>6.0f} MiB "
              f"({100 * need / total:>3.0f}% of the card)   {verdict}")
    print("\n  Measured on an idle card with one process. Under load the peak "
          "is higher:\n  the allocator grows with fragmentation, and every "
          "worker peaks at a different\n  moment. Treat the number above as an "
          "upper bound on the worker count.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
