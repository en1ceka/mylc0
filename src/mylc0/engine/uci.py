"""UCI engine.

Speaks the subset of UCI that engine-vs-engine tools need:

    uci / isready / ucinewgame / setoption / position / go / stop / quit

``position startpos [moves ...]``, ``position fen <fen> [moves ...]`` and
``go`` with ``nodes``, ``movetime``, ``wtime/btime/winc/binc/movestogo``,
``depth`` (treated as a node budget guard), ``infinite`` are all understood.

The engine is one half of the Lc0 split: it contains only search + inference
and loads its strength from a network file (``--weights``). Any generation
exported by the training loop can be loaded by the same binary.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from typing import List, Optional

import chess
import numpy as np
import torch

from ..chessrules.position import PositionHistory
from ..net.backend import Backend
from ..net.config import SearchConfig
from ..net.netfile import load_network
from ..search.node import Node, NodeTree, TERMINAL_TWOFOLD
from ..search.search import GoParams, LegacyTimeManager, Search

ENGINE_NAME = "mylc0"
ENGINE_VERSION = "1.0"
ENGINE_AUTHOR = "trained from zero with self-play"


def purge_twofold_terminals(root: Node) -> None:
    """Undo two-fold-draw terminals kept from a previous search.

    A position that was a two-fold repetition relative to the *old* root may
    not be one relative to the new root, so lc0 reverts those visits and makes
    the node non-terminal again (``search.cc``, around ``MakeNotTerminal``).
    """
    stack = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        if node.children:
            for child in node.children:
                if child is not None and child.n > 0:
                    stack.append((child, depth + 1))
        if node.terminal != TERMINAL_TWOFOLD:
            continue
        wl, d, m, visits = node.wl, node.d, node.m, node.n
        counter = 0
        walker: Optional[Node] = node
        while walker is not None:
            walker.revert_terminal_visits(wl, d, m + counter, visits)
            counter += 1
            if counter > depth:
                break
            walker = walker.parent
        node.make_not_terminal()


class UciEngine:
    def __init__(self, weights: Optional[str], device: str = "cuda",
                 fp16: bool = True):
        self.weights_path = weights
        self.device = device
        self.fp16 = fp16
        self.params = SearchConfig()   # lc0 engine defaults
        self.backend: Optional[Backend] = None
        self.model_config = None
        self.metadata = {}
        self.history = PositionHistory()
        self.tree = NodeTree(self.history)
        self.tree_moves: List[chess.Move] = []
        self.time_manager = LegacyTimeManager()
        self.move_overhead = 200
        self.nncache_size = 200000
        self.cache_history_length = 0
        self.show_wdl = True
        self.verbose_stats = False
        self.reuse_tree = True
        self.sticky_endgames = True
        self._stop = threading.Event()
        self._search_thread: Optional[threading.Thread] = None
        self._rng = np.random.default_rng()
        self._start_fen = chess.Board().fen()

    # -- io ----------------------------------------------------------------
    def send(self, line: str) -> None:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    def log(self, line: str) -> None:
        sys.stderr.write(line + "\n")
        sys.stderr.flush()

    # -- setup -------------------------------------------------------------
    def load_weights(self, path: str) -> None:
        model, config, metadata = load_network(path, device="cpu")
        self.model_config = config
        self.metadata = metadata
        self.backend = Backend(
            model, config,
            device=self.device if torch.cuda.is_available() or self.device == "cpu"
            else "cpu",
            fp16=self.fp16,
            max_batch_size=self.params.minibatch_size,
            policy_softmax_temp=self.params.policy_softmax_temp,
            cache_size=self.nncache_size,
            cache_history_length=self.cache_history_length,
            history_fill=self.params.history_fill)
        self.weights_path = path
        self.log(f"info string loaded {path} "
                 f"(generation {metadata.get('generation', '?')}, "
                 f"{metadata.get('parameters', '?')} params)")

    def ensure_backend(self) -> bool:
        if self.backend is not None:
            return True
        if self.weights_path and os.path.isfile(self.weights_path):
            self.load_weights(self.weights_path)
            return True
        self.log("info string no weights loaded; set the WeightsFile option "
                 "or pass --weights")
        return False

    # -- uci commands ------------------------------------------------------
    def cmd_uci(self) -> None:
        # Load the network first so the name can carry the generation: a GUI
        # registry keys engines by name, and "mylc0 1.0" for every generation
        # would make them indistinguishable when playing one against another.
        try:
            self.ensure_backend()
        except Exception as exc:
            self.log(f"info string could not load weights: {exc!r}")
        generation = self.metadata.get("generation")
        name = (f"{ENGINE_NAME} gen {generation:06d}"
                if isinstance(generation, int)
                else f"{ENGINE_NAME} {ENGINE_VERSION}")
        self.send(f"id name {name}")
        self.send(f"id author {ENGINE_AUTHOR}")
        self.send("option name WeightsFile type string default "
                  + (self.weights_path or "<none>"))
        self.send("option name Backend type combo default cuda var cuda var cpu")
        self.send("option name Fp16 type check default "
                  + ("true" if self.fp16 else "false"))
        self.send("option name Threads type spin default 1 min 1 max 1")
        self.send(f"option name MinibatchSize type spin default "
                  f"{self.params.minibatch_size} min 1 max 1024")
        self.send(f"option name NNCacheSize type spin default "
                  f"{self.nncache_size} min 0 max 100000000")
        self.send(f"option name CPuct type string default {self.params.cpuct}")
        self.send(f"option name CPuctBase type string default {self.params.cpuct_base}")
        self.send(f"option name CPuctFactor type string default "
                  f"{self.params.cpuct_factor}")
        self.send(f"option name FpuValue type string default {self.params.fpu_value}")
        self.send("option name FpuStrategy type combo default reduction "
                  "var reduction var absolute")
        self.send(f"option name PolicyTemperature type string default "
                  f"{self.params.policy_softmax_temp}")
        self.send("option name HistoryFill type combo default fen_only "
                  "var no var fen_only var always")
        self.send(f"option name Temperature type string default {self.params.temperature}")
        self.send(f"option name MoveOverheadMs type spin default "
                  f"{self.move_overhead} min 0 max 100000000")
        self.send("option name TwoFoldDraws type check default true")
        self.send("option name StickyEndgames type check default true")
        self.send("option name ReuseTree type check default true")
        self.send("option name UCI_ShowWDL type check default true")
        self.send("option name VerboseMoveStats type check default false")
        self.send("uciok")

    def cmd_setoption(self, tokens: List[str]) -> None:
        if "name" not in tokens:
            return
        name_idx = tokens.index("name")
        value_idx = tokens.index("value") if "value" in tokens else len(tokens)
        name = " ".join(tokens[name_idx + 1:value_idx]).lower()
        value = " ".join(tokens[value_idx + 1:]) if value_idx < len(tokens) else ""

        def as_bool(v: str) -> bool:
            return v.strip().lower() in ("true", "1", "yes", "on")

        if name == "weightsfile":
            self.load_weights(value)
        elif name == "backend":
            self.device = value.strip() or "cuda"
            if self.weights_path:
                self.load_weights(self.weights_path)
        elif name == "fp16":
            self.fp16 = as_bool(value)
            if self.weights_path:
                self.load_weights(self.weights_path)
        elif name == "minibatchsize":
            self.params.minibatch_size = int(value)
            if self.backend:
                self.backend.max_batch_size = int(value)
                self.backend._scratch = np.zeros((int(value), 112, 8, 8),
                                                 dtype=np.float32)
        elif name == "nncachesize":
            self.nncache_size = int(value)
            if self.backend:
                self.backend.cache.capacity = int(value)
        elif name == "cpuct":
            self.params.cpuct = float(value)
            self.params.cpuct_at_root = float(value)
        elif name == "cpuctbase":
            self.params.cpuct_base = float(value)
            self.params.cpuct_base_at_root = float(value)
        elif name == "cpuctfactor":
            self.params.cpuct_factor = float(value)
            self.params.cpuct_factor_at_root = float(value)
        elif name == "fpuvalue":
            self.params.fpu_value = float(value)
        elif name == "fpustrategy":
            self.params.fpu_strategy = value.strip()
        elif name == "policytemperature":
            self.params.policy_softmax_temp = float(value)
            if self.backend:
                self.backend.policy_softmax_temp = float(value)
                self.backend.cache.clear()
        elif name == "historyfill":
            self.params.history_fill = value.strip()
            if self.weights_path:
                self.load_weights(self.weights_path)
        elif name == "temperature":
            self.params.temperature = float(value)
        elif name == "moveoverheadms":
            self.move_overhead = int(value)
            self.time_manager.move_overhead = self.move_overhead
        elif name == "twofolddraws":
            self.params.two_fold_draws = as_bool(value)
        elif name == "stickyendgames":
            self.sticky_endgames = as_bool(value)
        elif name == "reusetree":
            self.reuse_tree = as_bool(value)
        elif name == "uci_showwdl":
            self.show_wdl = as_bool(value)
        elif name == "verbosemovestats":
            self.verbose_stats = as_bool(value)

    def cmd_ucinewgame(self) -> None:
        self.history = PositionHistory()
        self.tree = NodeTree(self.history)
        self.tree_moves = []
        self.time_manager.reset()
        if self.backend is not None:
            self.backend.cache.clear()

    def cmd_position(self, tokens: List[str]) -> None:
        if not tokens:
            return
        moves: List[str] = []
        if tokens[0] == "startpos":
            board = chess.Board()
            rest = tokens[1:]
        elif tokens[0] == "fen":
            fen_parts = []
            rest = tokens[1:]
            while rest and rest[0] != "moves":
                fen_parts.append(rest.pop(0))
            board = chess.Board(" ".join(fen_parts))
        else:
            return
        if rest and rest[0] == "moves":
            moves = rest[1:]

        new_moves = [chess.Move.from_uci(m) for m in moves]
        start_fen = board.fen()
        same_start = self._start_fen == start_fen
        prefix_ok = (same_start and self.reuse_tree
                     and len(new_moves) >= len(self.tree_moves)
                     and new_moves[:len(self.tree_moves)] == self.tree_moves)
        if prefix_ok:
            for mv in new_moves[len(self.tree_moves):]:
                self.tree.make_move(mv, reuse=True)
                self.tree_moves.append(mv)
            if self.params.two_fold_draws:
                purge_twofold_terminals(self.tree.root)
        else:
            self.history = PositionHistory(board)
            for mv in new_moves:
                self.history.append(mv)
            self.tree = NodeTree(self.history)
            self.tree_moves = list(new_moves)
        self._start_fen = start_fen

    def _parse_go(self, tokens: List[str]) -> GoParams:
        params = GoParams()
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token in ("wtime", "btime", "winc", "binc", "movestogo",
                         "movetime", "nodes", "depth") and i + 1 < len(tokens):
                setattr(params, token, int(float(tokens[i + 1])))
                i += 2
                continue
            if token == "infinite":
                params.infinite = True
            elif token == "ponder":
                params.ponder = True
            i += 1
        return params

    def cmd_go(self, tokens: List[str]) -> None:
        if not self.ensure_backend():
            self.send("bestmove 0000")
            return
        self.wait_for_search()
        params = self._parse_go(tokens)
        self._stop.clear()
        self._search_thread = threading.Thread(target=self._run_search,
                                               args=(params,), daemon=True)
        self._search_thread.start()

    def _run_search(self, params: GoParams) -> None:
        try:
            start = time.perf_counter()
            search = Search(self.tree, self.backend, self.params,
                            rng=self._rng, sticky_endgames=self.sticky_endgames,
                            info_callback=self._send_info)
            deadline = self.time_manager.deadline_ms(
                params, self.history.last().ply, self.history.is_black_to_move)
            search.run(params, deadline_ms=deadline,
                       stop_check=self._stop.is_set)
            elapsed = (time.perf_counter() - start) * 1000.0
            self.time_manager.on_search_done(deadline, elapsed)

            if self.verbose_stats:
                self._send_verbose_stats(search)
            self._send_info(search.build_info(elapsed))

            if not search.root.is_expanded or search.root.n == 0:
                legal = self.history.legal_moves()
                self.send(f"bestmove {legal[0].uci()}" if legal else "bestmove 0000")
                return
            idx = search.pick_move(self.history.last().ply)
            if idx < 0:
                legal = self.history.legal_moves()
                self.send(f"bestmove {legal[0].uci()}" if legal else "bestmove 0000")
                return
            move = search.root.moves[idx]
            ponder = None
            child = search.root.children[idx]
            if child is not None and child.is_expanded and child.n > 0:
                pidx = search.get_best_child_no_temperature(child, 1)
                if pidx >= 0:
                    ponder = child.moves[pidx]
            if ponder is not None:
                self.send(f"bestmove {move.uci()} ponder {ponder.uci()}")
            else:
                self.send(f"bestmove {move.uci()}")
        except Exception as exc:  # never leave the GUI hanging
            self.log(f"info string search error: {exc!r}")
            import traceback
            traceback.print_exc(file=sys.stderr)
            legal = self.history.legal_moves()
            self.send(f"bestmove {legal[0].uci()}" if legal else "bestmove 0000")

    def _send_info(self, info: dict) -> None:
        parts = [f"info depth {info['depth']}", f"seldepth {info['seldepth']}",
                 f"time {info['time']}", f"nodes {info['nodes']}",
                 f"score cp {info['score_cp']}"]
        if self.show_wdl:
            w, d, l = info["wdl"]
            parts.append(f"wdl {w} {d} {l}")
        parts.append(f"nps {info['nps']}")
        parts.append("tbhits 0")
        if info["pv"]:
            parts.append("pv " + " ".join(info["pv"]))
        self.send(" ".join(parts))

    def _send_verbose_stats(self, search: Search) -> None:
        root = search.root
        if not root.is_expanded:
            return
        order = search.get_best_children_no_temperature(root, len(root.moves), 0)
        for i in order:
            child = root.children[i]
            n = child.n if child is not None else 0
            q = -child.wl if (child is not None and n) else float("nan")
            d = child.d if (child is not None and n) else float("nan")
            m = child.m if (child is not None and n) else float("nan")
            self.send(f"info string {root.moves[i].uci():>6} "
                      f"N: {n:7d} (P: {100 * float(root.policy[i]):6.2f}%) "
                      f"(Q: {q:+.5f}) (D: {d:.3f}) (M: {m:5.1f})")

    def wait_for_search(self) -> None:
        """Block until the current search reports its bestmove.

        A GUI is only allowed to send ``position``/``go`` after ``bestmove``,
        so waiting (rather than aborting) is the correct behaviour here -- and
        it is what makes a fully piped, scripted session work.
        """
        if self._search_thread is not None:
            self._search_thread.join()
        self._search_thread = None

    def stop_search(self) -> None:
        """Abort the current search (``stop`` / ``quit``)."""
        self._stop.set()
        self.wait_for_search()

    # -- main loop ---------------------------------------------------------
    def loop(self) -> None:
        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue
            tokens = line.split()
            command, args = tokens[0], tokens[1:]
            if command == "uci":
                self.cmd_uci()
            elif command == "isready":
                self.ensure_backend()
                self.send("readyok")
            elif command == "ucinewgame":
                self.wait_for_search()
                self.cmd_ucinewgame()
            elif command == "setoption":
                self.cmd_setoption(args)
            elif command == "position":
                self.wait_for_search()
                self.cmd_position(args)
            elif command == "go":
                self.cmd_go(args)
            elif command == "stop":
                self.stop_search()
            elif command == "ponderhit":
                pass
            elif command in ("quit", "exit"):
                self.stop_search()
                return
            elif command == "d":
                self.send(str(self.history.board))
                self.send(f"fen: {self.history.fen()}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="mylc0 UCI engine")
    parser.add_argument("--weights", "-w", default=os.environ.get("MYLC0_WEIGHTS"),
                        help="network file (.mylc0) to load")
    parser.add_argument("--backend", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--fp16", default=True, type=lambda v: str(v).lower()
                        not in ("0", "false", "no"))
    args = parser.parse_args(argv)

    if args.weights is None:
        # Fall back to networks/latest.mylc0 next to the project, like lc0's
        # auto-discovery of a weights file.
        for candidate in ("networks/latest.mylc0", "latest.mylc0"):
            if os.path.isfile(candidate):
                args.weights = candidate
                break

    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    engine = UciEngine(args.weights, device=args.backend, fp16=args.fp16)
    engine.loop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
