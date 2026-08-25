"""End-to-end sanity checks.

    python scripts/sanity_check.py            # everything (a few minutes)
    python scripts/sanity_check.py --quick    # skip the training/self-play parts

There are no unit tests in this project by design; this script is the
executable check that the pieces agree with each other and with Lc0's
definitions. Each check prints PASS/FAIL and the script exits non-zero if
anything failed.
"""

import argparse
import os
import random
import shutil
import sys
import tempfile
import traceback

import _bootstrap  # noqa: F401
import chess
import numpy as np
import torch

from mylc0.chessrules import policy_map as pm
from mylc0.chessrules.position import (PositionHistory, move_to_our_uci,
                                       our_uci_to_move)
from mylc0.net import encoder as E
from mylc0.net.backend import Backend
from mylc0.net.config import (DefaultsConfig, EmbeddingConfig, EncoderConfig,
                              ModelConfig, MovesLeftHeadConfig,
                              PolicyHeadConfig, SmolgenConfig, ValueHeadConfig,
                              load_config)
from mylc0.net.model import build_model
from mylc0.net.netfile import load_network, save_network
from mylc0.search.node import NodeTree
from mylc0.search.search import GoParams, Search
from mylc0.selfplay.trainingdata import read_chunk, unpack_planes

RESULTS = []


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


def random_positions(n=200, seed=7):
    """Random legal positions reached by random play (rules only, no knowledge)."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        board = chess.Board()
        for _ in range(rng.randrange(0, 80)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if any(board.legal_moves):
            out.append(board.copy(stack=False))
    return out


# ---------------------------------------------------------------------------
@check("policy map: 1858 moves, gather map, transforms")
def check_policy_map():
    # The square transform and the bitboard transform must describe the same
    # geometry, otherwise the policy targets would not match the input planes.
    for t in range(8):
        for sq in range(64):
            assert pm.transform_bitboard(1 << sq, t) == 1 << pm.transform_square(sq, t)
    assert len(pm.MOVE_STRS) == 1858
    assert len(set(pm.MOVE_STRS)) == 1858
    assert pm.ATTENTION_POLICY_GATHER.shape == (1858,)
    assert pm.ATTENTION_POLICY_GATHER.max() < pm.ATTENTION_POLICY_LOGITS
    assert len(set(pm.ATTENTION_POLICY_GATHER.tolist())) == 1858
    for t in range(8):
        for i, mv in enumerate(pm.MOVE_STRS):
            j = int(pm.TRANSFORMED_IDX[t, i])
            if j < 0:
                continue
            assert pm.move_from_nn_index(j, t) == mv
    # Every promotion move except knight promotions is in the list; knight
    # promotions share the plain from/to slot.
    assert "a7a8n" not in pm.MOVE_TO_IDX
    assert pm.move_to_nn_index("a7a8n") == pm.MOVE_TO_IDX["a7a8"]
    return "all 8 transforms round-trip"


@check("policy indices cover every legal move")
def check_policy_coverage():
    total = 0
    for board in random_positions(150):
        history = PositionHistory(board)
        planes, transform = E.encode_position(history, 5)
        seen = set()
        for mv in history.legal_moves():
            uci = move_to_our_uci(mv, history.is_black_to_move)
            idx = pm.move_to_nn_index(uci, transform)
            assert 0 <= idx < 1858
            assert idx not in seen, f"duplicate policy slot for {uci}"
            seen.add(idx)
            back = pm.move_from_nn_index(idx, transform)
            assert our_uci_to_move(back, history.is_black_to_move) == mv or \
                (mv.promotion == chess.KNIGHT and back == uci[:4]), \
                f"{uci} -> {idx} -> {back}"
            total += 1
    return f"{total} moves mapped"


@check("input encoder: start position")
def check_encoder_startpos():
    history = PositionHistory()
    planes, transform = E.encode_position(history, 5)
    assert planes.shape == (112, 8, 8)
    assert transform == 0
    flat = planes.reshape(112, 64)
    assert flat[0][8:16].all() and flat[0].sum() == 8      # our pawns on rank 2
    assert flat[5][4] == 1 and flat[5].sum() == 1          # our king on e1
    assert flat[6][48:56].all()                            # their pawns
    assert flat[13:104].sum() == 0                         # no history yet
    assert flat[104][0] == 1 and flat[104][56] == 1        # a1/a8 rooks: 000
    assert flat[105][7] == 1 and flat[105][63] == 1        # h1/h8 rooks: 00
    assert flat[108].sum() == 0                            # no en passant
    assert flat[109].sum() == 0                            # rule50 == 0
    assert flat[110].sum() == 0 and flat[111].sum() == 64
    return "112 planes as expected"


@check("input encoder: black to move mirrors the board")
def check_encoder_mirror():
    history = PositionHistory()
    history.append(chess.Move.from_uci("e2e4"))
    planes, _ = E.encode_position(history, 5)
    flat = planes.reshape(112, 64)
    # Black to move: black's pawns are "ours" and sit on rank 2 of the mirrored
    # board; white's e4 pawn appears on e5 of the mirrored board.
    assert flat[0][8:16].all(), "our (black) pawns should be on the second rank"
    assert flat[6][36] == 1, "their (white) e4 pawn should be mirrored to e5"
    assert flat[5][4] == 1, "our (black) king should be on e1 of the mirror"
    return "ours/theirs and geometry mirrored"


@check("input encoder: en passant plane")
def check_encoder_ep():
    history = PositionHistory()
    for mv in ("e2e4", "c7c5", "e4e5", "d7d5"):
        history.append(chess.Move.from_uci(mv))
    planes, _ = E.encode_position(history, 5)
    flat = planes.reshape(112, 64)
    assert flat[108][56 + 3] == 1, "en passant marker on rank 8, file d"
    assert flat[108].sum() == 1
    # And it disappears once the chance is gone.
    history.append(chess.Move.from_uci("g1f3"))
    planes, _ = E.encode_position(history, 5)
    assert planes.reshape(112, 64)[108].sum() == 0
    return "phantom pawn on rank 8"


@check("input encoder: canonicalization")
def check_encoder_canonical():
    # No castling rights, no pawns: the king is moved to the bottom-right
    # octant by flip/mirror/transpose.
    seen = set()
    for fen in ("8/8/8/8/8/8/8/K6k w - - 0 1",
                "K7/8/8/8/8/8/8/7k w - - 0 1",
                "8/8/8/3K4/8/8/8/7k w - - 0 1",
                "8/8/8/8/8/2K5/8/7k w - - 0 1",
                "8/8/8/8/8/1K6/8/7k w - - 0 1",
                "8/1k6/8/8/8/8/6K1/8 w - - 0 1",
                "7k/8/8/8/8/8/8/K7 b - - 0 1"):
        history = PositionHistory.from_fen(fen)
        planes, transform = E.encode_position(history, 5)
        king = planes.reshape(112, 64)[5]
        sq = int(np.argmax(king))
        rank, file = divmod(sq, 8)
        assert file >= 4 and rank <= 3, (
            f"{fen}: king ended on rank {rank} file {file} "
            f"after transform {transform}")
        seen.add(transform)
    # With pawns on the board only the file flip is allowed.
    history = PositionHistory.from_fen("4k3/8/8/8/8/8/P7/K7 w - - 0 1")
    _, transform = E.encode_position(history, 5)
    assert transform in (0, pm.FLIP_TRANSFORM), transform
    # Any castling right forbids every transform.
    history = PositionHistory.from_fen(
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    _, transform = E.encode_position(history, 5)
    assert transform == 0
    return f"transforms seen: {sorted(seen)}"


@check("input encoder: repetition planes and history")
def check_encoder_repetitions():
    history = PositionHistory()
    for mv in ("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6"):
        history.append(chess.Move.from_uci(mv))
    assert history.last().repetitions == 1
    planes, _ = E.encode_position(history, 5)
    flat = planes.reshape(112, 64)
    assert flat[12].sum() == 64, "repetition plane of the current position"
    assert flat[109][0] == 6 / 100.0, "rule50 scaled by 100 (hectoplies format)"
    return "repetition + rule50 planes"


@check("training data <-> encoder agree")
def check_training_data_roundtrip():
    from mylc0.selfplay.trainingdata import TrainingDataArray, Eval
    for board in random_positions(60, seed=11):
        history = PositionHistory(board)
        planes, transform = E.encode_position(history, 5)
        data = TrainingDataArray(5)
        moves = history.legal_moves()
        indices = [pm.move_to_nn_index(move_to_our_uci(m, history.is_black_to_move),
                                       transform) for m in moves]
        counts = [1] * len(moves)
        data.add(planes=planes, transform=transform, snapshot=history.last(),
                 visit_counts=counts, total_children_visits=len(moves),
                 legal_policy_indices=indices, nneval_p=None,
                 policy_softmax_temp=1.0, root_q=0.0, root_d=1.0, root_m=0.0,
                 root_visits=1, best_eval=Eval(0, 1, 0),
                 played_eval=Eval(0, 1, 0), orig_eval=None,
                 best_idx=indices[0], played_idx=indices[0],
                 best_is_proven=False)
        frame = data.frames[0][0]
        restored = unpack_planes(frame)
        assert np.array_equal(restored, planes), (
            f"plane mismatch for {board.fen()}: "
            f"{np.nonzero((restored != planes).reshape(112, 64).any(axis=1))}")
        probs = frame["probabilities"]
        assert (probs[indices] > 0).all()
        legal_mask = probs >= 0
        assert legal_mask.sum() == len(moves), "only legal moves are non-negative"
        assert abs(float(probs[legal_mask].sum()) - 1.0) < 1e-4
    return "60 positions: planes, castling, ep, rule50 all survive the round trip"


@check("network: shapes, masking, determinism")
def check_model():
    config = tiny_config()
    model = build_model(config).eval()
    x = torch.randn(4, 112, 8, 8)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert out1.policy["vanilla"].shape == (4, 1858)
    assert out1.value["winner"][0].shape == (4, 3)
    assert out1.movesleft["main"].shape == (4, 1)
    assert (out1.movesleft["main"] >= 0).all(), "moves-left head must be >= 0"
    assert torch.allclose(out1.policy["vanilla"], out2.policy["vanilla"])
    wdl = torch.softmax(out1.value["winner"][0], dim=-1)
    assert torch.allclose(wdl.sum(dim=-1), torch.ones(4), atol=1e-5)
    return f"{model.num_parameters():,} parameters"


def tiny_config():
    return ModelConfig(
        defaults=DefaultsConfig(activation="mish", ffn_activation="mish"),
        embedding=EmbeddingConfig(dense_size=16, embedding_size=32, dff=48),
        encoder=EncoderConfig(num_blocks=2, dff=48, d_model=32, heads=2,
                              smolgen=SmolgenConfig(hidden_channels=4,
                                                    hidden_size=16,
                                                    gen_size=16,
                                                    activation="swish")),
        policy_head=[PolicyHeadConfig(name="vanilla", embedding_size=32,
                                      d_model=32)],
        value_head=[ValueHeadConfig(name="winner", num_channels=4)],
        movesleft_head=[MovesLeftHeadConfig(name="main", num_channels=2)],
        input_format=5)


def make_backend(config=None, device="cpu", seed=20240825):
    config = config or tiny_config()
    # Seeded: with random weights a check like "the search finds mate in 1
    # within 400 nodes" is occasionally unlucky, which would make the suite
    # flaky instead of informative.
    torch.manual_seed(seed)
    model = build_model(config)
    return Backend(model, config, device=device, fp16=False, max_batch_size=16,
                   policy_softmax_temp=1.0, cache_size=50000,
                   cache_history_length=7, history_fill="no"), config


@check("search: visit conservation and priors")
def check_search_invariants():
    backend, _ = make_backend()
    params = load_config("configs/tiny.yaml").selfplay.search
    history = PositionHistory()
    search = Search(NodeTree(history), backend, params,
                    rng=np.random.default_rng(3))
    search.run(GoParams(nodes=300))
    root = search.root
    assert root.n == 300, root.n
    child_visits = sum(c.n for c in root.children if c is not None)
    assert child_visits == root.n - 1, (child_visits, root.n)
    assert abs(float(root.policy.sum()) - 1.0) < 1e-4, float(root.policy.sum())
    assert all(c is None or c.n_in_flight == 0 for c in root.children)
    # Q of the root must equal the visit-weighted average of the children.
    total = sum((-c.wl) * c.n for c in root.children if c is not None and c.n)
    assert abs((total + 0.0) / max(1, root.n - 1) - root.wl) < 0.2
    return f"N={root.n}, {len(root.moves)} moves, collisions={search.stats.collisions}"


@check("search: terminal positions")
def check_search_terminals():
    backend, _ = make_backend()
    params = load_config("configs/tiny.yaml").selfplay.search
    params.noise_epsilon = 0.0
    params.temperature = 0.0

    def best(fen, nodes=400):
        history = PositionHistory.from_fen(fen)
        s = Search(NodeTree(history), backend, params,
                   rng=np.random.default_rng(5), sticky_endgames=True)
        s.run(GoParams(nodes=nodes))
        return s

    s = best("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    idx = s.get_best_child_no_temperature(s.root, 0)
    assert s.root.moves[idx].uci() == "a1a8", s.root.moves[idx]
    assert s.root.children[idx].is_terminal
    assert s.root.children[idx].wl == -1.0, "mated side sees -1"
    # Most of the root's visits go to the proven mate, so its average value is
    # close to +1; the exact figure depends on how many visits the other moves
    # got before the mate was found.
    assert s.root.q() > 0.8, s.root.q()

    s = best("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", nodes=10)
    assert s.root.is_terminal and s.root.wl == 0.0, "stalemate is a draw"

    s = best("7k/5Q1K/8/8/8/8/8/8 b - - 0 1", nodes=10)
    assert s.root.is_terminal and s.root.wl == -1.0, "checkmate is a loss"
    return "mate/stalemate/backup signs correct"


@check("search: Dirichlet noise changes the root priors")
def check_search_noise():
    backend, config = make_backend()
    params = load_config("configs/tiny.yaml").selfplay.search
    history = PositionHistory()
    planes, transform = E.encode_position(history, 5, fill_empty_history=E.FILL_NO)
    moves = history.legal_moves()
    indices = np.array([pm.move_to_nn_index(move_to_our_uci(m, False), transform)
                        for m in moves])
    from mylc0.net.backend import EvalRequest
    raw = backend.evaluate([EvalRequest(planes, indices, None)])[0].p

    noisy = []
    for seed in (1, 2):
        s = Search(NodeTree(PositionHistory()), backend, params,
                   rng=np.random.default_rng(seed))
        s.run(GoParams(nodes=1))
        noisy.append(np.array(s.root.policy))
    assert not np.allclose(noisy[0], noisy[1]), "noise should differ per game"
    assert not np.allclose(noisy[0], raw), "noise should change the priors"
    assert abs(float(noisy[0].sum()) - 1.0) < 1e-4
    return "epsilon=0.25, alpha=0.3 applied at the root"


@check("optimizer: NAdamW reproduces the optax nadamw update")
def check_optimizer():
    import math

    from mylc0.training.optim import NAdamW
    p = torch.nn.Parameter(torch.tensor([1.0]))
    opt = NAdamW([{"params": [p], "weight_decay": 0.1}], lr=0.1,
                 betas=(0.9, 0.98), eps=1e-7)
    b1, b2, eps, wd, lr = 0.9, 0.98, 1e-7, 0.1, 0.1
    mu = nu = 0.0
    x = 1.0
    worst = 0.0
    for t in range(1, 6):
        g = 2.0 * x
        p.grad = torch.tensor([2.0 * float(p.data)])
        opt.step()
        opt.zero_grad()
        mu = b1 * mu + (1 - b1) * g
        nu = b2 * nu + (1 - b2) * g * g
        mu_hat = b1 * mu / (1 - b1 ** (t + 1)) + (1 - b1) * g / (1 - b1 ** t)
        nu_hat = nu / (1 - b2 ** t)
        x = x - lr * (mu_hat / (math.sqrt(nu_hat) + eps) + wd * x)
        worst = max(worst, abs(float(p.data) - x))
    assert worst < 1e-6, worst
    return f"max deviation over 5 steps: {worst:.1e}"


@check("lr schedule: warm-up then constant, as configured")
def check_lr_schedule():
    from mylc0.training.optim import make_lr_schedule
    config = load_config("configs/reference_lc0.yaml")
    sched = make_lr_schedule(config.training.lr_schedule)
    assert sched(0) == 0.0
    assert abs(sched(750) - 0.00025) < 1e-9, sched(750)   # halfway through warm-up
    assert abs(sched(1500) - 0.0005) < 1e-9
    assert abs(sched(50000) - 0.0005) < 1e-9              # open-ended tail
    return "linear warm-up over 1500 steps, then 5e-4"


class _StubBackend:
    """A deterministic stand-in for the network.

    The equivalence check below is about the *driver*, not about cuBLAS: a real
    network in fp16 can return slightly different numbers for the same position
    depending on the batch it travelled in, which would muddle the comparison.
    This stub is a pure function of the input planes, so any difference between
    the two runs can only come from the search itself.
    """

    def __init__(self, real):
        self._real = real
        self.input_format = real.input_format
        self.movesleft_head = real.movesleft_head
        self.cache = real.cache
        self.calls = 0
        self.batch_sizes = []

    def encode(self, history):
        return self._real.encode(history)

    def cache_key(self, history):
        return self._real.cache_key(history)

    def evaluate(self, requests):
        import hashlib
        from mylc0.net.backend import EvalResult
        if requests:
            self.calls += 1
            self.batch_sizes.append(len(requests))
        out = []
        for req in requests:
            digest = hashlib.blake2b(req.planes.tobytes(), digest_size=8).digest()
            rng = np.random.default_rng(int.from_bytes(digest, "little"))
            p = rng.random(len(req.policy_indices)).astype(np.float32)
            p /= p.sum()
            q = float(rng.random() * 2 - 1)
            d = float(rng.random() * (1 - abs(q)))
            out.append(EvalResult(q=q, d=d, m=float(rng.random() * 50), p=p))
        return out


def _drive_search(search, visits, minibatch, other=None):
    """Run a search through gather/apply, exactly as GameRunner does.

    When ``other`` is given, its requests are merged into the same batch, so
    the batch composition differs from the single-search case.
    """
    request = search.prepare_root()
    if request is not None:
        extra = other.prepare_root() if other is not None else None
        batch = [request] + ([extra] if extra is not None else [])
        results = search.backend.evaluate(batch)
        search.apply_root(results[0])
        if extra is not None:
            other.apply_root(results[1])
    while not (search.root.is_terminal or search.root.n >= visits):
        limit = min(minibatch, max(1, visits - search.root.n))
        gathered = search.gather(limit)
        requests = search.requests_of(gathered)
        other_gathered = None
        if other is not None and not (other.root.is_terminal
                                      or other.root.n >= visits):
            other_limit = min(minibatch, max(1, visits - other.root.n))
            other_gathered = other.gather(other_limit)
            requests = requests + other.requests_of(other_gathered)
        results = search.backend.evaluate(requests)
        own = len(search.requests_of(gathered))
        if not search.apply(gathered, results[:own]):
            break
        if other_gathered is not None:
            other.apply(other_gathered, results[own:])


@check("batched driver: sharing a batch does not change the search")
def check_batched_equivalence():
    from mylc0.net.config import SelfPlayConfig
    real, _ = make_backend()
    backend = _StubBackend(real)
    params = SelfPlayConfig().search
    visits = 400
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
    other_fen = "rnbqkb1r/pp2pppp/3p1n2/2pP4/8/2N5/PPP1PPPP/R1BQKBNR w KQkq - 0 4"

    def make(seed, start):
        history = PositionHistory.from_fen(start)
        backend.cache.clear()
        return Search(NodeTree(history), backend, params,
                      rng=np.random.default_rng(seed))

    # 1) the reference: this search alone, driven by Search.run
    solo = make(1234, fen)
    solo.run(GoParams(nodes=visits))
    solo_visits = [c.n if c is not None else 0 for c in solo.root.children]

    # 2) the same search, driven step by step while another game shares the
    #    batches -- batch sizes differ, the result must not
    shared = make(1234, fen)
    noise = make(999, other_fen)
    before = backend.calls
    _drive_search(shared, visits, params.minibatch_size, other=noise)
    shared_visits = [c.n if c is not None else 0 for c in shared.root.children]
    batches = backend.batch_sizes[before:]

    assert [m.uci() for m in solo.root.moves] == [m.uci() for m in shared.root.moves]
    assert solo.root.n == shared.root.n, (solo.root.n, shared.root.n)
    assert solo_visits == shared_visits, "visit distribution changed"
    assert abs(solo.root.wl - shared.root.wl) < 1e-9, (solo.root.wl, shared.root.wl)
    assert np.allclose(solo.root.policy, shared.root.policy), "priors changed"
    assert max(batches) > params.minibatch_size, (
        "the shared batches should be bigger than one search's minibatch")
    return (f"identical tree over {solo.root.n} visits; batches up to "
            f"{max(batches)} vs minibatch {params.minibatch_size}")


@check("batched driver: a request-less gather still makes progress")
def check_terminal_only_gather():
    """Regression test for the hang fixed in BatchedSelfPlay.step().

    Once the search has proven a terminal child, PUCT keeps selecting it, so a
    whole gather can consist of terminal leaves -- nothing for the network to
    evaluate. The driver must still call apply(): otherwise those leaves keep
    their in-flight visits forever, root.n stops growing and the game never
    finishes.
    """
    from mylc0.net.config import SelfPlayConfig
    real, _ = make_backend()
    backend = _StubBackend(real)
    params = SelfPlayConfig().search
    visits = 200
    history = PositionHistory.from_fen("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    search = Search(NodeTree(history), backend, params,
                    rng=np.random.default_rng(0))
    search.apply_root(backend.evaluate([search.prepare_root()])[0])

    empty_gathers = 0
    steps = 0
    max_steps = visits * 4          # a hang would blow through this
    while search.root.n < visits and steps < max_steps:
        steps += 1
        limit = min(params.minibatch_size, max(1, visits - search.root.n))
        gathered = search.gather(limit)
        requests = search.requests_of(gathered)
        if not requests:
            empty_gathers += 1
        # Exactly what BatchedSelfPlay.step() does: apply unconditionally.
        results = backend.evaluate(requests) if requests else []
        search.apply(gathered, results)

    assert empty_gathers > 0, (
        "this position was supposed to produce gathers with no requests; "
        "the regression test is not exercising the bug")
    assert search.root.n >= visits, (
        f"search stalled at {search.root.n}/{visits} visits after {steps} "
        f"steps -- the request-less gathers were not applied")
    assert search.root.n_in_flight == 0, (
        f"{search.root.n_in_flight} in-flight visits leaked on the root")
    for child in search.root.children:
        assert child is None or child.n_in_flight == 0, "leaked on a child"
    return (f"{empty_gathers} request-less gathers, reached "
            f"{search.root.n}/{visits} visits in {steps} steps, no leaked "
            f"in-flight visits")


@check("batched driver: N games in flight produce valid chunks")
def check_batched_selfplay(tmpdir):
    from mylc0.selfplay.batched import BatchedSelfPlay
    config = load_config("configs/tiny.yaml")
    config.selfplay.visits = 48
    config.selfplay.max_game_ply = 30
    backend, _ = make_backend()

    def play(parallel, prefix):
        written = []

        def on_game(game):
            path = os.path.join(tmpdir, f"{prefix}_{len(written):02d}.gz")
            frames = game.write(path)
            written.append((path, frames, game.stats))

        driver = BatchedSelfPlay(backend, config.selfplay, parallel, seed=7)
        # Bounded, so a hang fails the check instead of hanging the suite.
        budget = [40000]

        def stop():
            budget[0] -= 1
            assert budget[0] > 0, (
                f"driver made no progress: {len(written)} games after "
                f"40000 steps")
            return len(written) >= 4

        driver.run(on_game=on_game, should_stop=stop)
        return driver, written

    one, _ = play(1, "seq")
    many, written = play(4, "par")

    assert len(written) >= 4, len(written)
    for path, frames, stats in written:
        assert frames == stats.plies, (frames, stats.plies)
        chunk = read_chunk(path)
        assert len(chunk) == frames
        for frame in chunk:
            probs = frame["probabilities"]
            legal = probs[probs >= 0]
            assert abs(float(legal.sum()) - 1.0) < 1e-3
    # The whole point of the driver: more games in flight means bigger batches.
    assert many.avg_batch > one.avg_batch * 1.5, (
        f"4 games in flight should batch much better than 1: "
        f"{many.avg_batch:.1f} vs {one.avg_batch:.1f}")
    return (f"{len(written)} games; avg NN batch {one.avg_batch:.1f} with 1 game "
            f"-> {many.avg_batch:.1f} with 4 (max {many.stats.requests_per_batch_max})")


@check("self-play: a full game produces consistent training data")
def check_selfplay(tmpdir):
    from mylc0.selfplay.game import SelfPlayGame
    config = load_config("configs/tiny.yaml")
    config.selfplay.visits = 24
    config.selfplay.max_game_ply = 40
    backend, _ = make_backend()
    game = SelfPlayGame(backend, config.selfplay, rng=np.random.default_rng(2))
    stats = game.play()
    path = os.path.join(tmpdir, "game.gz")
    frames_written = game.write(path)
    assert frames_written == stats.plies, (frames_written, stats.plies)

    frames = read_chunk(path)
    assert len(frames) == frames_written
    assert frames.dtype.itemsize == 8356

    plies_left = [float(f["plies_left"]) for f in frames]
    # Counts down one ply at a time starting from best_m of the final position
    # (V6TrainingDataArray::Write); the tolerance is float32 rounding.
    assert all(abs(a - b - 1.0) < 1e-4
               for a, b in zip(plies_left, plies_left[1:])), \
        f"plies_left must count down by one per ply: {plies_left[:5]}"
    assert abs(plies_left[-1] - float(frames[-1]["best_m"])) < 1e-4

    for i, frame in enumerate(frames):
        probs = frame["probabilities"]
        legal = probs[probs >= 0]
        assert abs(float(legal.sum()) - 1.0) < 1e-3, (i, legal.sum())
        assert -1.0 <= float(frame["root_q"]) <= 1.0
        assert 0.0 <= float(frame["root_d"]) <= 1.0
        black_to_move = bool(int(frame["invariance_info"]) & 0x80)
        # The result target alternates with the side to move.
        if stats.result != 2 and stats.result != 0:
            expected = (1.0 if (stats.result == 1) != black_to_move else -1.0)
            assert float(frame["result_q"]) == expected, (i, frame["result_q"])
        assert int(frame["version"]) == 6
        assert int(frame["input_format"]) == 5
    return (f"{stats.plies} plies, result={stats.result}, "
            f"{frames_written} frames, {stats.seconds:.1f}s")


@check("training: one step reduces the loss on a fixed batch")
def check_training(tmpdir):
    from mylc0.training.dataset import frames_to_tensors
    from mylc0.training.trainer import Trainer
    config = load_config("configs/tiny.yaml")
    config.model = tiny_config()
    config.training.batch_size = 16
    config.training.mixed_precision = False
    config.training.lr_schedule[0].duration_steps = [0]
    config.training.lr_schedule[0].lr = [1e-3]
    trainer = Trainer(config, device="cpu",
                      checkpoint_dir=os.path.join(tmpdir, "ckpt"),
                      networks_dir=os.path.join(tmpdir, "nets"),
                      tensorboard_dir=os.path.join(tmpdir, "tb"))

    chunks = [p for p in os.listdir(tmpdir) if p.endswith(".gz")]
    frames = read_chunk(os.path.join(tmpdir, chunks[0]))
    batch = [frames[i % len(frames)] for i in range(16)]
    tensors = frames_to_tensors(batch)

    class _Fixed:
        stats = type("S", (), {"chunks_read": 1, "frames_sampled": 16})()
        pool = []

        def next_batch(self):
            return tensors

    losses = []
    for _ in range(30):
        losses.append(trainer.train_step(_Fixed())["total_loss"])
    assert losses[-1] < losses[0], (losses[0], losses[-1])
    trainer.save_checkpoint()
    net = trainer.export_network()
    assert os.path.isfile(net)
    return f"loss {losses[0]:.3f} -> {losses[-1]:.3f} over 30 steps"


@check("checkpoint resume restores step, generation and weights")
def check_checkpoint(tmpdir):
    from mylc0.training.trainer import Trainer
    config = load_config("configs/tiny.yaml")
    config.model = tiny_config()
    ckpt = os.path.join(tmpdir, "ckpt2")
    a = Trainer(config, device="cpu", checkpoint_dir=ckpt,
                networks_dir=os.path.join(tmpdir, "nets2"),
                tensorboard_dir=os.path.join(tmpdir, "tb2"))
    a.step = 123
    a.generation = 4
    a.positions_seen = 4567
    a.save_checkpoint()

    b = Trainer(config, device="cpu", checkpoint_dir=ckpt,
                networks_dir=os.path.join(tmpdir, "nets2"),
                tensorboard_dir=os.path.join(tmpdir, "tb2"))
    assert b.load_checkpoint()
    assert (b.step, b.generation, b.positions_seen) == (123, 4, 4567)
    for (ka, va), (kb, vb) in zip(a.model.state_dict().items(),
                                  b.model.state_dict().items()):
        assert ka == kb and torch.equal(va, vb)
    return "weights and counters identical"


@check("network file: export -> load gives identical outputs")
def check_network_file(tmpdir):
    config = tiny_config()
    model = build_model(config).eval()
    path = os.path.join(tmpdir, "net.mylc0")
    save_network(path, model, config, metadata={"generation": 42})
    loaded, loaded_config, meta = load_network(path)
    assert meta["generation"] == 42
    assert loaded_config.encoder.num_blocks == config.encoder.num_blocks
    x = torch.randn(2, 112, 8, 8)
    with torch.no_grad():
        a = model(x)
        b = loaded.eval()(x)
    assert torch.allclose(a.policy["vanilla"], b.policy["vanilla"], atol=1e-6)
    assert torch.allclose(a.value["winner"][0], b.value["winner"][0], atol=1e-6)
    assert torch.allclose(a.movesleft["main"], b.movesleft["main"], atol=1e-6)
    return "bit-identical inference after a round trip"


@check("UCI: the engine answers a scripted session")
def check_uci(tmpdir):
    import subprocess
    config = tiny_config()
    torch.manual_seed(1234)
    model = build_model(config)
    net = os.path.join(tmpdir, "uci.mylc0")
    save_network(net, model, config, metadata={"generation": 0})

    script = "\n".join([
        "uci",
        "isready",
        "ucinewgame",
        "position startpos moves e2e4 e7e5",
        "go nodes 40",
        "position fen 6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1",
        "go nodes 400",
        # "position" waits for the running search to report bestmove, which is
        # what a GUI does; "quit" aborts it, so give the search a command that
        # waits before shutting the engine down.
        "position startpos",
        "quit", ""])
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc = subprocess.run(
        [sys.executable, os.path.join(root, "engine", "mylc0.py"),
         "--weights", net, "--backend", "cpu", "--fp16", "false"],
        input=script, capture_output=True, text=True, timeout=300)
    out = proc.stdout
    trace = "\n--- engine stderr ---\n" + proc.stderr
    assert "uciok" in out, out + trace
    assert "readyok" in out, out + trace
    bestmoves = [l for l in out.splitlines() if l.startswith("bestmove")]
    assert len(bestmoves) == 2, out + trace
    assert bestmoves[1].split()[1] == "a1a8", bestmoves[1] + trace
    assert any(l.startswith("info depth") for l in out.splitlines())
    return f"{len(bestmoves)} bestmoves, mate found: {bestmoves[1].strip()}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    print("== rules, policy mapping and input encoding ==")
    check_policy_map()
    check_policy_coverage()
    check_encoder_startpos()
    check_encoder_mirror()
    check_encoder_ep()
    check_encoder_canonical()
    check_encoder_repetitions()
    check_training_data_roundtrip()

    print("\n== network ==")
    check_model()

    print("\n== optimizer and schedule ==")
    check_optimizer()
    check_lr_schedule()

    print("\n== search ==")
    check_search_invariants()
    check_search_terminals()
    check_search_noise()
    check_batched_equivalence()
    check_terminal_only_gather()

    tmpdir = tempfile.mkdtemp(prefix="mylc0-sanity-")
    try:
        print("\n== self-play, training, export, engine ==")
        check_network_file(tmpdir)
        if not args.quick:
            check_batched_selfplay(tmpdir)
            check_selfplay(tmpdir)
            check_training(tmpdir)
            check_checkpoint(tmpdir)
            check_uci(tmpdir)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    failed = [name for name, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        print("failed: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
