"""Per-generation training report, reconstructed from the TensorBoard logs.

    python scripts/report_generations.py
    python scripts/report_generations.py --csv generations.csv --from 1

Each row is one generation: the self-play phase that produced its data, and
the training block that turned that data into the exported network. Nothing
here is recomputed -- it is exactly what the run wrote to ``runs/``.
"""

import argparse
import collections
import csv
import glob
import os

import _bootstrap  # noqa: F401

COLUMNS = [
    ("gen", "generation", "{:>4}"),
    ("step", "training step at export", "{:>5}"),
    ("loss", "total loss", "{:>7.3f}"),
    ("pol_ce", "policy cross-entropy", "{:>7.3f}"),
    ("pol_kl", "policy KL", "{:>7.3f}"),
    ("value", "value (WDL) loss", "{:>6.3f}"),
    ("mlh", "moves-left loss", "{:>6.3f}"),
    ("acc", "policy accuracy", "{:>6.3f}"),
    ("lr", "learning rate", "{:>9.2e}"),
    ("gnorm", "gradient norm", "{:>7.2f}"),
    ("games", "self-play games", "{:>6.0f}"),
    ("pos", "self-play positions", "{:>7.0f}"),
    ("len", "average game length (plies)", "{:>6.1f}"),
    ("W", "white wins", "{:>4.0f}"),
    ("D", "draws", "{:>4.0f}"),
    ("B", "black wins", "{:>4.0f}"),
    ("adj", "adjudicated at max length", "{:>4.0f}"),
    ("nps", "self-play nodes/s", "{:>6.0f}"),
    ("sp_min", "self-play minutes", "{:>7.1f}"),
    ("seen", "positions seen by the trainer", "{:>9.0f}"),
]


def load_scalars(run_dir):
    from tensorboard.backend.event_processing import event_accumulator
    scalars = collections.defaultdict(dict)
    files = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")))
    if not files:
        raise SystemExit(f"no TensorBoard event files in {run_dir}")
    for path in files:
        acc = event_accumulator.EventAccumulator(
            path, size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags()["scalars"]:
            for event in acc.Scalars(tag):
                scalars[tag][event.step] = event.value
    return scalars, len(files)


def build_rows(scalars):
    """One row per generation.

    ``loop.py`` logs a self-play phase at step S, then trains up to S + steps
    while ``train/generation`` still holds the *previous* number, and only then
    increments and exports. So the block tagged generation g produced the
    network ``gen_{g+1}``.
    """
    blocks = collections.defaultdict(list)
    for step, value in scalars.get("train/generation", {}).items():
        blocks[int(value)].append(step)

    def at(tag, step, before=False):
        series = scalars.get(tag, {})
        candidates = [s for s in series if (s < step if before else s <= step)]
        return series[max(candidates)] if candidates else float("nan")

    rows = []
    for source_gen in sorted(blocks):
        steps = sorted(blocks[source_gen])
        last = steps[-1]
        first = steps[0]
        # The self-play phase feeding this block was logged at or just before
        # its first training step.
        sp_step = max([s for s in scalars.get("selfplay/games", {})
                       if s <= first] or [first])

        def sp(tag):
            return scalars.get(tag, {}).get(sp_step, float("nan"))

        rows.append({
            "gen": source_gen + 1,
            "step": last,
            "loss": at("loss/total_loss", last),
            "pol_ce": at("loss/policy/main_ce", last),
            "pol_kl": at("loss/policy/main_kl", last),
            "value": at("loss/value/winner", last),
            "mlh": at("loss/movesleft/main", last),
            "acc": at("loss/policy/accuracy", last),
            "lr": at("train/lr", last),
            "gnorm": at("train/grad_norm", last),
            "games": sp("selfplay/games"),
            "pos": sp("selfplay/positions"),
            "len": sp("selfplay/avg_game_length"),
            "W": sp("selfplay/white_wins"),
            "D": sp("selfplay/draws"),
            "B": sp("selfplay/black_wins"),
            "adj": sp("selfplay/adjudicated"),
            "nps": sp("selfplay/nodes_per_second"),
            "sp_min": sp("selfplay/wall_seconds") / 60.0,
            "seen": at("train/positions_seen", last),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--csv", default=None, help="also write a CSV")
    parser.add_argument("--from", dest="first", type=int, default=1)
    parser.add_argument("--to", dest="last", type=int, default=None)
    args = parser.parse_args()

    scalars, files = load_scalars(args.runs)
    rows = [r for r in build_rows(scalars)
            if r["gen"] >= args.first and (args.last is None
                                           or r["gen"] <= args.last)]
    if not rows:
        print("no generations in that range")
        return 1

    print(f"# {len(rows)} generations, from {files} TensorBoard run file(s)\n")
    print("  ".join(f"{key:>7}" for key, _d, _f in COLUMNS))
    print("-" * (9 * len(COLUMNS)))
    for row in rows:
        cells = []
        for key, _desc, fmt in COLUMNS:
            try:
                cells.append(f"{fmt.format(row[key]):>7}")
            except (ValueError, TypeError):
                cells.append(f"{'?':>7}")
        print("  ".join(cells))

    first, last = rows[0], rows[-1]
    print("\nchange from generation "
          f"{first['gen']} to {last['gen']}:")
    for key in ("loss", "pol_ce", "value", "mlh", "acc", "len"):
        print(f"  {key:<8} {first[key]:>8.3f} -> {last[key]:>8.3f}"
              f"   ({last[key] - first[key]:+.3f})")
    print(f"  total self-play games      "
          f"{sum(r['games'] for r in rows):.0f}")
    print(f"  total self-play positions  "
          f"{sum(r['pos'] for r in rows):.0f}")
    print(f"  total self-play time       "
          f"{sum(r['sp_min'] for r in rows) / 60:.1f} h")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle,
                                    fieldnames=[k for k, _d, _f in COLUMNS])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV written: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
