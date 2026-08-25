# mylc0 — chess reinforcement learning, built to follow Leela Chess Zero

A network that starts from random weights and learns chess from nothing but the
rules, its own games and its own results:

```
random network -> self-play (MCTS) -> training data -> training -> better network -> ...
```

Every generation is exported as a stand-alone network file, and a UCI engine
loads any of them, so generation 10 can be played against generation 500 in any
engine-testing GUI.

The design follows **Lc0**, not a generic AlphaZero description. Read
[ARCHITECTURE.md](ARCHITECTURE.md) first: it names the upstream files each part
was transcribed from, and lists — explicitly — everything that differs and why.

**What the network is allowed to know:** the rules of chess, which moves are
legal, and who won. No Stockfish, no Lc0 weights, no game databases, no opening
book, no tablebases, no piece values, no handcrafted evaluation.

---

## Which network format this implements

`NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT` with `POLICY_ATTENTION`,
`VALUE_WDL` and `MOVES_LEFT_V1` — the current mainline Lc0 architecture:

* **input**: 112 planes, `INPUT_112_WITH_CANONICALIZATION_V2`, 8 positions of
  history, castling as rook files, en-passant plane, rule-50 plane,
  canonicalization by flip/mirror/diagonal reflection;
* **body**: per-square embedding with positional preprocessing and ma-gating,
  then N transformer encoder blocks with **smolgen** and DeepNorm scaling,
  mish activations;
* **heads**: attention policy over Lc0's **1858** move slots, **WDL** value
  head, **moves-left** head;
* **data**: Lc0's **V6** training records (8356 bytes/frame), gzipped one chunk
  per game — the same format lc0's own self-play writes.

---

## A note on network size, and your GPU

Network size in Lc0 is a configuration field, not part of the architecture
(upstream ships everything from small nets to 1024x15). Three configs are
included and they differ **only in the layer sizes** — same planes, same heads,
same losses, same search:

| config | blocks x d_model | parameters | what it is for |
|---|---|---|---|
| `configs/reference_lc0.yaml` | 15 x 1024 | 185 M | transcription of upstream's own example config |
| `configs/small.yaml` | 10 x 256 | 23 M | a from-scratch run on one consumer GPU |
| `configs/tiny.yaml` | 4 x 64 | 2.3 M | smoke-testing the pipeline in minutes |

Measured on an RTX 3060 Ti (8 GB), fp16:

| config | self-play (1 worker) | self-play (4 workers) | training | training VRAM |
|---|---|---|---|---|
| `small` (23 M) | 210 positions/min | **547 positions/min**, GPU 91 % | 2760 pos/s @ batch 256 | 2.3 GB |
| `reference_lc0` (185 M) | ~1.1 s/move | scales the same way | 240 pos/s @ batch 128 | 7.2 GB |

Self-play plays `parallel_games` (8, Lc0's default) games at once per worker and
merges their network requests into one batch — without that the GPU sits at a
fraction of its throughput, because a small batch costs almost as much as a
large one. Scaling on an RTX 3060 Ti with `configs/small.yaml`:

| workers x games | positions/min | GPU |
|---|---|---|
| 1 x 8 | 210 | 46 % |
| 2 x 8 | 367 | 64 % |
| 3 x 8 | 475 | 82 % |
| 4 x 8 | 547 | 91 % |

Use `--workers 3` or `4`; beyond that the GPU is saturated. Games from a random
network run long (~250 plies), so games *per hour* looks low relative to moves
per second; they get shorter as the network learns.

So the reference config does run on a single consumer GPU. It is roughly 11x
slower per training position, and at batch 128 it sits close to the 8 GB limit
— `activation_checkpointing: true` takes that down to 2.1 GB (see below). It is
included because it is what Lc0 actually trains. But an 800-visit self-play
game is a few hundred thousand network evaluations, a generation is thousands
of games, and a real Lc0 run is millions of games — so if the goal is to *watch
generations improve* within days rather than months, start with
`configs/small.yaml`. Nothing about the architecture changes between them.

---

## Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118   # or your CUDA build
pip install chess pyyaml tensorboard numpy
```

CUDA is used automatically when available (`--device cpu` to force CPU).
Inference runs in fp16 by default; training uses mixed precision with a
gradient scaler and uploads batches through pinned memory.

If VRAM is tight, three levers change memory without touching the network:

* `training.gradient_accumulation: N` with `batch_size` divided by N — same
  effective batch, same maths;
* `training.activation_checkpointing: true` — recomputes encoder activations in
  the backward pass. Measured on the reference config at batch 128: 7.2 GB and
  425 ms/step becomes 2.1 GB and 569 ms/step;
* `selfplay.batch_size` / `selfplay.parallel_games` for the self-play side
  (`search.minibatch_size` is **not** a memory knob — it is part of the search
  algorithm, see ARCHITECTURE.md §9).

---

## Run the loop

```bash
# 0. check that everything agrees with itself (no unit tests in this project;
#    this script is the check)
python scripts/sanity_check.py

# 1. generation 0: a randomly initialised network
python scripts/init_network.py --config configs/small.yaml

# 2. self-play -> training -> next generation, forever
python scripts/loop.py --config configs/small.yaml --workers 2
```

`loop.py` alternates phases and is restartable: it resumes from the newest
checkpoint, keeps every exported generation, and can be stopped with Ctrl-C at
any point (it writes a checkpoint on the way out).

The phases can also be run separately:

```bash
python scripts/selfplay.py --config configs/small.yaml \
    --network networks/latest.mylc0 --output data --workers 4 --games 50

python scripts/train.py --config configs/small.yaml --data data --generations 5

python scripts/export_network.py --checkpoint checkpoints/generation_000012/checkpoint.pt
```

Self-play workers are independent processes that only share the network file
and the output directory, so `--workers N` scales to as many as the GPU can
feed (and to several machines writing into the same `data/`). Each worker
additionally keeps `selfplay.parallel_games` games in flight and batches their
network requests together.

`scripts/profile_selfplay.py` measures where self-play time actually goes
(batch sizes, GPU vs CPU split, nodes/s, evals/s, GPU utilisation);
`--workers N` turns it into a scaling test.

---

## Watch it

Every long phase prints a live line that refreshes in place:

```
2026-08-25 15:16:36 [loop] === iteration 1 | generation 0 | step 0 ===
2026-08-25 15:16:36 [loop] phase 1/2 self-play: 20000 positions with 3 worker(s), network gen_000000.mylc0
self-play  [######------------] 6234/20000 pos  ETA 12m30s  28 games  3/3 busy  1180 pos/min  5m12s
...
2026-08-25 15:29:44 [loop] phase 2/2 training: 250 steps, batch 256 x 1, chunk pool 428
training  [##########--------] 137/250  loss 3.214  p 2.58 v 0.31 m 0.42  acc 0.41  4.2 it/s  ETA 27s
```

The self-play counter includes the plies of the games still being played, so
it moves smoothly even though a single game takes minutes. When the target is
reached the line says `finishing games in flight`: a game in progress is always
played to its result, because a game without a result has no value target.

`--progress auto|on|off` controls it. On a terminal the bar is on by default;
when the output is redirected the bar turns itself off, the same information is
logged every 30 seconds instead, and the per-game result lines come back — so a
log file stays readable.

```bash
tensorboard --logdir runs
```

Logged every generation and every few steps: policy / value / moves-left /
total loss, policy accuracy, learning rate, gradient norm, positions per
second, GPU memory, chunk pool size, and from the self-play side games,
positions, average game length, white/black/draw counts, adjudications, nodes
per second and network evaluations. The console logs the same numbers.

---

## Play a generation

The engine is separate from the weights, as in Lc0:

```
engine/mylc0.py          the engine (search + inference only)
engine/mylc0.bat         Windows launcher
networks/gen_000500.mylc0 the weights
```

```bash
python engine/mylc0.py --weights networks/gen_000500.mylc0
engine\mylc0.bat --weights networks\gen_000500.mylc0     # Windows
```

With no `--weights` it loads `networks/latest.mylc0`. Point any UCI GUI or
tournament runner at the launcher and pass the network you want; to play
generation against generation, register the same launcher twice with different
`--weights`.

Supported: `uci`, `isready`, `ucinewgame`, `setoption`, `position startpos
[moves ...]`, `position fen ... [moves ...]`, `go` with `nodes`, `movetime`,
`wtime/btime/winc/binc/movestogo`, `depth`, `infinite`, plus `stop` and `quit`.
It replies with `info depth/seldepth/time/nodes/score cp/wdl/nps/pv` and
`bestmove ... [ponder ...]`.

Useful options: `WeightsFile`, `Backend` (cuda/cpu), `Fp16`, `MinibatchSize`,
`NNCacheSize`, `CPuct`, `CPuctBase`, `CPuctFactor`, `FpuValue`, `FpuStrategy`,
`PolicyTemperature`, `HistoryFill`, `Temperature`, `MoveOverheadMs`,
`TwoFoldDraws`, `StickyEndgames`, `ReuseTree`, `UCI_ShowWDL`,
`VerboseMoveStats` — the names and defaults are Lc0's.

There is deliberately no `play.py` and no internal arena: a generation is
exercised as a UCI engine, nothing else.

---

## Layout

```
configs/                model + training + self-play configuration (YAML)
engine/                 the UCI engine entry point and launchers
scripts/                init_network, selfplay, train, loop, export_network, sanity_check
src/mylc0/
  chessrules/           position/history in Lc0's frame; the 1858-move policy map
  net/                  input encoder, model config, the network, backend + NN cache,
                        network file format and export
  search/               tree nodes and the PUCT search
  selfplay/             one self-play game, V6 training data, workers
  training/             data loader, losses, optimizer/LR schedule, trainer
checkpoints/            full training state, one directory per generation
networks/               exported inference networks: gen_NNNNNN.mylc0, latest.mylc0
data/                   self-play chunks (one gzipped V6 chunk per game)
runs/                   TensorBoard
```

---

## Checkpoints and resuming

Each generation writes `checkpoints/generation_NNNNNN/checkpoint.pt` with the
weights, optimizer state, step and generation counters, positions seen, the
model config and the RNG state. `train.py` and `loop.py` resume from the newest
one automatically (`--fresh` to ignore them). `checkpoint_max_to_keep` prunes
old checkpoints; **exported networks are never pruned**, so every generation
stays playable.

To re-export a network from any checkpoint (for instance after pulling old
checkpoints from a backup):

```bash
python scripts/export_network.py --all                    # every checkpoint on disk
python scripts/export_network.py --network networks/gen_000100.mylc0 --info
python scripts/export_network.py --network networks/gen_000100.mylc0 --onnx gen100.onnx
```

---

## What to expect

A random network plays random-looking chess and games run to the 450-ply
adjudication limit. The first things to improve are usually the value head and
game length; policy accuracy against the search's own visit distribution rises
early. Real strength takes a very large number of generations — Lc0's own runs
consume millions of games. This project is built so that the *process* is
faithful and observable, not so that it is fast.
