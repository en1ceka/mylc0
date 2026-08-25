# ARCHITECTURE

What this project is: a from-scratch reinforcement-learning chess system that
follows **Leela Chess Zero** as closely as I could make it, rather than a
generic AlphaZero re-implementation. Everything below is written against the
current upstream sources, which were read while building this (not from
memory or from articles):

| repository | what was used |
|---|---|
| [`LeelaChessZero/lc0`](https://github.com/LeelaChessZero/lc0) | `src/neural/encoder.cc` (input planes, canonicalization, the 1858-move list), `src/chess/board.cc`, `src/chess/types.h`, `src/utils/bititer.h`, `src/search/classic/{search.cc,node.cc,params.cc}`, `src/search/classic/stoppers/{legacy.cc,simple.cc,factory.cc}`, `src/selfplay/{game.cc,tournament.cc}`, `src/trainingdata/{trainingdata.cc,trainingdata_v6.h,trainingdata_v7.h}`, `src/neural/shared_params.cc` |
| [`LeelaChessZero/lczero-training`](https://github.com/LeelaChessZero/lczero-training) | `src/lczero_training/model/{model,embedding,encoder,policy_head,value_head,movesleft_head,shared,loss_function}.py`, `src/lczero_training/training/{optimizer,lr_schedule}.py`, `proto/{model_config,training_config}.proto`, `docs/{architecture,heads,training_tuple,example.textproto}`, `csrc/loader/stages/tensor_generator.cc`, `tf/chunkparser.py` |
| [`LeelaChessZero/lczero-common`](https://github.com/LeelaChessZero/lczero-common) | `proto/net.proto` |

---

## 1. How current Lc0 works

Lc0 is split in two halves that communicate only through files:

* **lc0** (C++) — the engine. It does a batched PUCT tree search and evaluates
  leaves with a neural network loaded from a `.pb.gz` weights file. The same
  binary, run with `selfplay --training`, plays games against itself and writes
  *training data chunks*.
* **lczero-training** (Python) — the trainer. It reads chunks, shuffles them,
  builds batches, trains the network and exports a new weights file. The
  network then goes back to self-play, and the cycle repeats.

Upstream has recently moved the trainer from TensorFlow to **JAX/Flax**: the
model now lives in `src/lczero_training/model/*.py`, the data loader is a
multi-threaded C++ pipeline exposed through pybind11, and the configuration is
a set of protobufs (`model_config.proto`, `training_config.proto`). The legacy
TensorFlow trainer is still in `tf/` and its `chunkparser.py` is still the
clearest description of the on-disk data format. Both were used here.

Network *formats* are described by `NetworkFormat` in `net.proto`. The modern
ones are:

```
NETWORK_ATTENTIONBODY_WITH_HEADFORMAT       = 6
NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT  = 7   <- current
POLICY_ATTENTION, VALUE_WDL, MOVES_LEFT_V1
```

i.e. a transformer over the 64 squares ("attention body") with smolgen, an
attention policy head, a WDL value head and a moves-left head, plus optional
extra heads (`policy_heads {vanilla, optimistic_st, soft, opponent}` and
`value_heads {winner, q, st}`).

---

## 2. Which network format this project implements, and why

**`NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT` with `POLICY_ATTENTION`,
`VALUE_WDL`, `MOVES_LEFT_V1`, `DEFAULT_ACTIVATION_MISH` and smolgen** — the
current mainline Lc0 architecture, ported to PyTorch from the JAX model in
`lczero-training/src/lczero_training/model/`.

Why this one:

* It is what upstream trains today; the residual/SE tower (`NETWORK_SE`) is the
  previous generation.
* The multi-head format is the one that supports several policy/value heads.
  This project configures one of each (`vanilla`, `winner`, `main`), which is
  what a from-zero run needs — the extra heads (`optimistic_st`, `soft`,
  `opponent`, `q`, `st`, error and categorical outputs) are trained against
  targets that only become meaningful once the data contains a real search
  signal, and two of them (`q_st`, `d_st`) are filled in by upstream's
  *rescorer*, which needs tablebases. The head classes here already accept
  `has_error_output` and `num_categorical_buckets` and the model holds head
  *dictionaries*, so adding them later is a config change, not a rewrite.

Network **size** is a configuration field, exactly as upstream
(`ModelConfig.embedding/encoder/...`). Three configs ship with the project and
they differ **only** in those numbers — never in the architecture:

| config | blocks | d_model | embedding | heads | params |
|---|---|---|---|---|---|
| `configs/reference_lc0.yaml` | 15 | 1024 | 1024 | 32 | 185 M |
| `configs/small.yaml` | 10 | 256 | 256 | 8 | 23 M |
| `configs/tiny.yaml` | 4 | 64 | 64 | 4 | 2.3 M |

`reference_lc0.yaml` is a transcription of upstream's own example
(`lczero-training/docs/example.textproto`, run name "little-teapot").

---

## 3. Input representation

`src/mylc0/net/encoder.py` is a transcription of `EncodePositionForNN`
(`lc0/src/neural/encoder.cc`). The tensor is **112 planes of 8x8**, all from
the point of view of the side to move — Lc0's board is mirrored vertically and
colour-swapped when black is to move, so "our pieces" are always at the bottom
(`src/mylc0/chessrules/position.py`).

```
plane   0..103   8 history positions x 13 planes:
                 our P N B R Q K, their P N B R Q K, "this position repeated"
plane 104        rooks with a-side castling rights   (ours rank 1, theirs rank 8)
plane 105        rooks with h-side castling rights
plane 106,107    zero (they hold the third/fourth castling flag in format 1)
plane 108        en-passant file, marked on rank 8
plane 109        rule-50 ply count / 100
plane 110        zero (was the move counter)
plane 111        all ones - lets the network find the edge of the board
```

Selected input format: **`INPUT_112_WITH_CANONICALIZATION_V2` (5)**. All of
formats 1..5 (and the two armageddon variants) are implemented in
`encode_position`; 5 is the default because it is the most modern one and
because it is the family where upstream is self-consistent about the rule-50
scaling (see §12).

What "canonicalization" does, following `ChooseTransform`:

* If the side to move has **no castling rights**, the board is flipped
  left/right so that our king is on files e..h.
* If additionally there are **no pawns**, it may also be mirrored top/bottom
  and reflected across the a8-h1 diagonal, so that our king always ends in the
  bottom-right octant. Ties on the diagonal are broken by comparing the
  bitboards of all/ours/kings/queens/rooks/knights/bishops in that order.
* The chosen transform is applied to every board-shaped plane **and** to the
  policy indices, so the network only ever sees one representative of each
  symmetry class.

V2 also prunes history: `skip_non_repeats` writes an older position into the
history planes only if it is a repetition, and the walk stops at a capture, a
pawn move, a castling-rights change or an en-passant (none of those can repeat).
History is *not* invented: `FillEmptyHistory` is `no` during self-play and
`fen_only` in the engine, matching `tournament.cc` and `shared_params.cc`.

En passant follows Lc0's odd but deliberate encoding: a "phantom pawn" bit on
rank 8 at the file of the pawn that just double-pushed, set only when an enemy
pawn actually attacks the square (`board.cc`).

---

## 4. Policy

The policy space is Lc0's **1858 moves**, not from-square x to-square. The list
is copied verbatim from `kMoveStrs` in `encoder.cc` into
`src/mylc0/chessrules/policy_map.py` — its order *is* the index. It contains
every geometrically possible queen/knight move from every square, plus
under-promotions; **knight promotions share the plain from/to slot**, exactly
as `MoveAsPackedInt` does (its promotion bit is only set for queen/rook/bishop).

Moves are indexed in "our" orientation: for a black-to-move position the move
is vertically flipped first, then the canonicalization transform is applied
(`MoveToNNIndex(move, transform)`).

The head is `POLICY_ATTENTION` (`policy_head.py`): the 64 square embeddings are
projected to Q and K, `QK^T` gives a 64x64 logit matrix, and a separate
`promotion_dense` produces per-file promotion offsets which are added to the
rank-7 x rank-8 block. The resulting 64*64 + 8*24 = **4288** logits are gathered
into the 1858 policy vector through a fixed index map. That map is rebuilt here
from the move list and was verified element-by-element against upstream's
`_policy_map` array.

Illegal moves are masked twice, as upstream does:

* in the search, only the legal moves' logits are softmaxed (with Lc0's policy
  softmax temperature — 1.359 in the engine, 1.0 in self-play);
* in training, illegal moves are stored as `-1` in the target and their logits
  are set to `-inf` before the loss (`illegal_moves: MASK`).

---

## 5. Value / WDL

The value head is `VALUE_WDL`: three logits, softmaxed into (win, draw, loss)
**from the side-to-move's point of view**. The search uses

```
Q = W - L          (and D separately, for the draw score / contempt machinery)
```

which is exactly how Lc0 collapses WDL to a scalar. Perspective is handled by
one rule applied everywhere: a node stores its value in *its own* frame, so a
child seen from its parent is `-child.Q`, and the backup flips sign once per
ply (§7).

Training target (`heads.md`, "Winner"): from `(result_q, result_d)`

```
W = (1 + q - d) / 2      D = d      L = (1 - q - d) / 2
```

and the loss is softmax cross-entropy against that distribution.

---

## 6. Moves-left head

Implemented (`MOVES_LEFT_V1`, `movesleft_head.py`): per-square embedding ->
flatten -> dense(128) -> dense(1) -> **ReLU**, so the output is a non-negative
number of plies.

Two things use it:

* **Training**: Huber loss with `delta = 10/20` on target and prediction both
  divided by 20 (`loss_function.py: MovesLeftLoss`). The target `plies_left` is
  filled in backwards over a finished game starting from `best_m` of the final
  position (`V6TrainingDataArray::Write`).
* **Search**: `MEvaluator` (`search.cc`) adds a small utility term that prefers
  shorter wins and longer losses, gated on the parent evaluation being decisive
  (`|Q| > MovesLeftThreshold`), with Lc0's default slope/cap/polynomial
  (0.0027 / 0.0345 / `0 + 1.6521|q| - 0.6521 q^2`).

---

## 7. Search

`src/mylc0/search/{node,search}.py`, transcribed from
`lc0/src/search/classic/{node,search}.cc`. No minimax, no alpha-beta, no
rollouts: every leaf is evaluated by the network.

**Selection** (`PickNodesToExtend`):

```
score(child) = P * cpuct * sqrt(max(N_children, 1)) / (1 + N_started) + U
cpuct        = CPuct + CPuctFactor * log((N + CPuctBase) / CPuctBase)
U            = -child.Q + M_utility        (visited child)
             = FPU                         (unvisited child)
FPU          = Q_parent - FpuValue * sqrt(sum of visited children's P)   ["reduction"]
             = FpuValue                                                  ["absolute"]
```

`N_started = N + N_in_flight`: in-flight visits enter the *denominator* only,
they never perturb Q. Defaults are Lc0's (`params.cc`): CPuct 1.745,
CPuctBase 38739, CPuctFactor 3.894, FpuValue 0.330 — with
`selfplay/tournament.cc`'s training overrides (CPuct 1.2, CPuctFactor 0,
FpuValue 0, policy softmax temp 1.0, minibatch 32, collisions 1).

**Terminal nodes** (`ExtendNode`): checkmate / stalemate always; and away from
the root also insufficient mating material, rule 50 >= 100, threefold
(`repetitions >= 2`) and — when `TwoFoldDraws` is on — a twofold repetition
deep enough in the tree, scored as a draw with `M = cycle length`.

**Backup** (`DoBackupUpdateSingleNode`): running averages
`wl += k*(v - wl)/(n + k)` up the path, `v` negated and `m` incremented once per
ply, terminal nodes overriding the value on the way up. Proven-terminal
propagation (`MaybeSetBounds`, "sticky endgames") is implemented including the
`AdjustForTerminal` correction that fixes ancestors' averages retroactively;
it is on in the engine and off in self-play, matching Lc0's defaults.

**Batching**: leaves are collected into a minibatch of `MinibatchSize`, then
evaluated in one network call. A path that lands on a node already queued in
the batch is a *collision*: its in-flight visits stay on the path (so the next
descent is pushed elsewhere) and are cancelled after the batch, bounded by
`MaxCollisionEvents`/`MaxCollisionVisits`. This is the same mathematics as Lc0
with one search thread.

**Move choice**: `GetBestChildrenNoTemperature` — prefer proven wins (shortest),
then most visits, then eval, then prior; avoid proven losses (longest). With
temperature, `GetBestRootChildWithTemperature` samples
`((N + offset)/N_max)^(1/T)` over the moves that pass `TemperatureWinpctCutoff`.

**Time management** in the engine is `LegacyTimeManager`
(`stoppers/legacy.cc`), Lc0's default: a log-logistic estimate of the moves
remaining, an opening bonus, and a "piggy bank" of time saved on earlier moves.

---

## 8. Self-play

`src/mylc0/selfplay/{game,batched,worker}.py`, following `SelfPlayGame::Play`.
Both sides are the same network with the same parameters. Per move:

1. search `visits` nodes (800 by default) with Dirichlet noise at the root;
2. take `GetBestEval` for the training record and the resignation check;
3. pick the move actually played by sampling the visit counts with temperature;
4. write one training frame;
5. play the move; the tree is trimmed (Lc0's self-play default is no reuse).

Exploration is Lc0's, with no invented heuristics:

* **Root noise** (`ApplyDirichletNoise`): `P <- (1-eps)P + eps*Dir(alpha)` with
  `eps = 0.25`, `alpha = 0.3`, drawn per edge as `Gamma(alpha, 1)` and
  normalised.
* **Temperature**: `Temperature = 1.0` for training games, with the full
  `TempDecayMoves` / `TempDecayDelayMoves` / `TemperatureCutoffMove` /
  `TemperatureEndgame` / `TemperatureVisitOffset` / `TemperatureWinpctCutoff`
  schedule from `EnsureBestMoveKnown`.
* Games are adjudicated at game ply 450; resignation is available and, like
  upstream, disabled by default (`ResignPercentage = 0`).

**Games in flight.** Lc0 plays several games at once and merges their network
requests into one batch -- `selfplay/tournament.cc` defaults to
`parallelism = 8` with the `multiplexing` backend. The same arrangement is
implemented here (`selfplay/batched.py`): a worker drives `parallel_games`
runners, and one step collects the leaves of all of them into a single network
call:

    game 1 --.
    game 2 --+--> one batch --> network --> results split back per game
    ...    --'

Each game keeps its own tree, history, RNG and Dirichlet noise, and each
search's gather/apply sequence is identical to what it would be running alone
-- only the evaluation, a pure function of the position, is shared. The search
is split into `Search.gather()` (pick leaves, pure CPU) and `Search.apply()`
(expand and back up) so a driver can interleave them; `Search.run()` still
drives a single search for the engine. A sanity check runs the same search both
ways and asserts the visit distribution is identical.

Workers are separate processes that only share the network file and the output
directory, so scaling out further is a matter of starting more of them.

---

## 9. Training targets

One frame per position, in Lc0's **V6** format
(`src/mylc0/selfplay/trainingdata.py`), byte-compatible with
`V6TrainingData` (8356 bytes, packed, little-endian). A chunk is the gzipped
concatenation of one game's frames — the same file a real Lc0 self-play run
writes.

* **Policy target**: `child_visits / total_child_visits` for every legal move,
  `-1` for illegal ones. It is the MCTS visit distribution, never a one-hot of
  the played move.
* **Value target**: `result_q` / `result_d` — the actual game result seen from
  the side to move, filled in for the whole game once it ends. `best_q/d`,
  `played_q/d`, `root_q/d` and the raw network evaluation `orig_q/d/m` are
  recorded too, so other value targets can be trained later without regenerating
  data.
* **Moves-left target**: `plies_left`, counting down one per ply from `best_m`
  of the final position.
* Also stored, as upstream does: `policy_kld` (KL between visits and priors),
  `visits`, `best_idx`/`played_idx`, the canonicalization transform and the
  side to move in `invariance_info`, castling rook files, rule-50 count.

---

## 10. Training pipeline

`src/mylc0/training/`, mirroring the stages of upstream's data loader:

```
chunk files -> shuffling chunk pool -> position sampling -> frame shuffle
            -> batches (planes 112x8x8, probabilities 1858, values 6x3)
```

The `values` tensor is indexed exactly like `tensor_generator.cc`:
`[result, best, played, orig, root, st] x [q, d, m]`.

**Losses** (`loss_function.py`): policy cross-entropy/KL with masked illegal
moves and optional target temperature, value softmax cross-entropy against the
WDL target, moves-left Huber. Weights come from the config; the reference
config reproduces upstream's example (see §12).

**Optimizer**: NAdamW written out explicitly to match `optax.nadamw`
(`b1 = 0.9`, `b2 = 0.98`, `eps = 1e-7`, decoupled weight decay 1e-4 applied
through the same first-match-wins selector upstream uses — no decay on biases,
layernorms, gates or the embedding). Global gradient-norm clipping at 10.

**LR schedule**: the piecewise schedule from `training/lr_schedule.py`
(intervals with CONSTANT / LINEAR / COSINE transitions, open-ended tail,
optional looping). The default is a linear warm-up to 5e-4 held constant.

**Generations**: every `steps_per_network` steps the trainer writes a full
checkpoint (weights, optimizer state, step, generation, positions seen, config,
RNG state) and exports an inference network `networks/gen_NNNNNN.mylc0`. Old
generations are never overwritten.

---

## 11. Implemented 1:1

* The 1858-move policy space and its index order (copied from `kMoveStrs`).
* The attention-policy gather map (4288 -> 1858), verified against upstream's.
* All 112 input planes for input formats 1, 2, 3, 4, 5, 132, 133, including
  canonicalization, the V2 history pruning, the en-passant phantom pawn, the
  castling-as-rook-files encoding and the repetition planes.
* Board mirroring for black to move, and move flipping to match.
* `ChooseTransform` including the diagonal tie-breaking order.
* The network: embedding with positional preprocessing, ma-gating, DeepNorm
  alpha/beta scaling and initialisation, smolgen with a shared generator
  matrix, mish/swish activations, LayerNorm eps 1e-3, attention policy head
  with promotion offsets, WDL value head, moves-left head.
* PUCT, cpuct schedule, both FPU strategies, MEvaluator, terminal detection,
  backup with proven-terminal propagation, best-child ordering, temperature
  sampling, Dirichlet noise.
* Self-play with several games in flight sharing one network batch
  (`parallelism` + `multiplexing` in lc0's self-play defaults).
* V6 training data: every field, the bit-reversed plane packing, the drift
  correction, the policy KLD, and the backwards `plies_left` fill.
* Loss functions, NAdamW, LR schedule, weight-decay selector semantics.
* The legacy time manager.

## 12. Deliberate differences (and why)

1. **Language: Python, not C++.** Lc0's engine is C++ and its trainer is
   JAX/Python. Here both are Python/PyTorch. The search is single-threaded and
   batched, which is mathematically the same as Lc0 with one search thread; it
   is perhaps 50-100x slower per node than the C++ engine. This is the largest
   practical difference in the project, and it costs throughput, not
   correctness.
2. **Rules library.** Move generation, FEN parsing and Chess960 castling rights
   come from `python-chess` rather than Lc0's own bitboard code. The *semantics*
   that matter for the network (repetitions, rule-50, en-passant legality,
   `HasMatingMaterial`, `ComputeGameResult`) are re-implemented to match Lc0's
   definitions exactly.
3. **The transpose transform.** Lc0 master's `Transform(Square, int)` in
   `encoder.cc` flips the rank and flops the file independently, which is a
   180-degree rotation, while `TransposeBitsInBytes` (which transforms the
   *planes*) reflects across the a8-h1 diagonal. The two disagree, so planes and
   policy indices would describe different geometries. Lc0 v0.31's
   `Transform(BoardSquare, int)` (`chess/bitboard.cc`) applies the three
   transforms in sequence with `set(7 - col, 7 - row)` for the transpose, which
   *is* consistent with the bitboard operation. This project uses the
   consistent (v0.31) definition; the sanity check verifies square and bitboard
   transforms agree for all 8 transforms and all 64 squares.
4. **Rule-50 plane scaling.** For input formats 1-3 lc0's encoder fills the
   plane with the raw ply count while lczero-training divides by 99
   (`chunkparser.py`, `tensor_generator.cc`); for the hectoplies formats (4, 5,
   132, 133) both divide by 100. Self-play and training share one encoder here,
   so there is no mismatch either way, and the default format (5) is one where
   upstream agrees with itself.
5. **Extra heads not trained.** `optimistic_st`, `soft`, `opponent`, `q`, `st`,
   the value-error and categorical outputs exist in `net.proto` and in the head
   classes here, but are not configured. Their targets need either a rescorer
   (tablebases — forbidden in this project) or search statistics that a random
   network does not produce. Adding one is a config change.
6. **No rescorer.** Upstream's `chunk_rescorer` stage rewrites targets using
   Syzygy tablebases and applies deblundering. Tablebases are excluded by the
   project's rules, so the stage is absent. `dist_temp`/`dist_offset`-style
   policy sharpening is likewise not applied.
7. **No contempt / WDL rescaling.** `WDLRescale`, `Contempt`,
   `WDLCalibrationElo` and friends in `search.cc` shift evaluations for
   opponent modelling in engine matches. They do not affect training, so they
   are not implemented. `DrawScore` *is* implemented and defaults to 0.
8. **Time management.** Lc0's default (`legacy`) manager is implemented;
   `smooth`, `alphazero` and `simple`, and smart pruning, are not. `go nodes`
   and `go movetime` are exact.
9. **Search parallelism.** Within one game the search is single-threaded with
   batched leaf collection; across games, a worker interleaves
   `parallel_games` searches into shared network batches (Lc0's `parallelism`
   and `multiplexing`, see section 8). Lc0 additionally runs many *threads*
   inside one search; that extra machinery (`MaxConcurrentSearchers`,
   out-of-order evaluation, solid trees, task workspaces) is about scaling, not
   about what the search computes.

   Note the two "batch sizes" this involves, which are not the same thing:
   `search.minibatch_size` (32, Lc0's `MinibatchSize`) is part of the
   *algorithm* -- it bounds how many leaves one search reserves before
   evaluating, and through collisions it affects which nodes get visited.
   `selfplay.batch_size` (256) is purely an implementation limit on one network
   call and has no effect on the search.
10. **Policy loss weights in the reference config.** Upstream's example lists a
    cross-entropy *and* a KL policy loss on the same head, each with weight 1.0.
    They differ by the (constant) target entropy, so the gradients are the same
    and the effective policy weight is 2.0 against value 1.0 and moves-left 1.0.
    The reference config keeps it as upstream has it; setting one weight to 0
    gives a 1:1:1 balance.
11. **Network file format.** Lc0 stores weights as a protobuf (`net.proto`,
    optionally wrapping an ONNX model). This project uses its own versioned
    container (`.mylc0`: magic, format version, model config as JSON, weights)
    because a byte-compatible `.pb.gz` would only be useful for running these
    networks *inside real lc0*, which would also require the layer layout to
    match `Weights.*` field by field. An ONNX export is provided
    (`scripts/export_network.py --onnx`), which is the same interchange path
    lc0 itself offers through `leela2onnx`.
12. **Two-fold-draw reversion on tree reuse.** Lc0 reverts stale twofold
    terminals lazily during search; the engine here does the same reversion
    eagerly when the tree is advanced (`purge_twofold_terminals`), using the
    same `RevertTerminalVisits` / `MakeNotTerminal` arithmetic.
13. **`MinimumAllowedVisits` game discarding.** The retry loop is implemented;
    the "discarded games" callback (which upstream uses to report rejected
    games to a training server) is not.
